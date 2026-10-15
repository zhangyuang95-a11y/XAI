"""Development-only checks for the isolated structured Warehouse NN candidate."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.training.warehouse_structured_v5 import (
    FrozenStructuredDualNN, STRUCTURED_FEATURE_DIM, STRUCTURED_FEATURE_NAMES,
    StructuredDualNN, TREE_DEPTH, TREE_LEAVES, export_structured_actor,
    TREE_MAX_FEATURES_PER_NODE, structured_features_np,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks, observations


def _public_state():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=1000)
    return env


def test_structured_encoder_uses_public_pre_action_values_only():
    env = _public_state()
    obs, masks = observations(env.state), action_masks(env.state)
    encoded = structured_features_np(obs, masks)
    assert encoded.shape == (2, STRUCTURED_FEATURE_DIM)
    assert np.isfinite(encoded).all()
    assert "order_0_pickup_row" in STRUCTURED_FEATURE_NAMES
    assert "legal_wait" in STRUCTURED_FEATURE_NAMES
    alternate = masks.copy()
    role, action = next((role, action) for role in range(2)
                        for action in range(len(ACTIONS) - 1)
                        if alternate[role, action] > 0.5)
    alternate[role, action] = 0.0
    changed = structured_features_np(obs, alternate)
    legal_index = STRUCTURED_FEATURE_NAMES.index("legal_" + ACTIONS[action].lower())
    assert encoded[role, legal_index] == 1.0
    assert changed[role, legal_index] == 0.0
    assert np.array_equal(observations(env.state), obs)


def test_two_trainable_role_actors_and_bounded_tree_branch():
    torch.manual_seed(11)
    model = StructuredDualNN()
    assert model.actors["robot_1"] is not model.actors["robot_2"]
    assert TREE_DEPTH <= 5 and TREE_LEAVES <= 16
    assert model.actors["robot_1"].tree.leaf_count == TREE_LEAVES
    assert model.actors["robot_1"].residual[0].out_features == 256
    assert model.actors["robot_1"].residual[2].out_features == 256
    assert (model.actors["robot_1"].tree.leaf_logits.data_ptr() !=
            model.actors["robot_2"].tree.leaf_logits.data_ptr())
    env = _public_state()
    obs, masks = observations(env.state), action_masks(env.state)
    t_obs, t_masks = torch.from_numpy(obs), torch.from_numpy(masks)
    with torch.no_grad():
        before = model.outputs(t_obs, t_masks).clone()
        legal_action = int(np.flatnonzero(masks[0] > 0.5)[0])
        model.actors["robot_1"].tree.leaf_logits[:, legal_action].add_(0.25)
        after = model.outputs(t_obs, t_masks)
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    loss = model.outputs(t_obs, t_masks)[0, legal_action]
    loss.backward()
    actor = model.actors["robot_1"]
    assert actor.tree.leaf_logits.grad is not None
    assert actor.tree.split.weight.grad is not None
    assert actor.residual[0].weight.grad is not None
    effective = actor.tree.effective_split_weights().detach().numpy()
    assert np.all(np.count_nonzero(effective, axis=1) <= TREE_MAX_FEATURES_PER_NODE)


def test_batched_ppo_forward_updates_both_roles():
    env = _public_state()
    obs = np.stack([observations(env.state)] * 3)
    masks = np.stack([action_masks(env.state)] * 3)
    model = StructuredDualNN()
    logits = model.outputs(torch.from_numpy(obs), torch.from_numpy(masks))
    assert logits.shape == (3, 2, len(ACTIONS))
    logits[logits > -1.0e8].sum().backward()
    for role in AGENTS:
        assert model.actors[role].tree.leaf_logits.grad.abs().sum() > 0
        assert model.actors[role].residual[0].weight.grad.abs().sum() > 0
    frozen = FrozenStructuredDualNN.from_model(model)
    actual = np.where(masks > 0.5, frozen.logits(obs, masks), -1.0e9)
    assert np.allclose(logits.detach().numpy(), actual, atol=1e-4, rtol=1e-5)


def test_frozen_export_matches_both_masked_neural_logits_and_rejects_tamper(tmp_path: Path):
    torch.manual_seed(13)
    model = StructuredDualNN()
    path = tmp_path / "structured_actor.npz"
    export = export_structured_actor(model, path, pair_seed=6001, variant="regularized")
    assert export["maximum_logit_difference"] < 1e-4
    assert export["parity_pre_action_rows"] >= 3
    assert export["parity_mask_patterns"] >= 2
    assert export["role_sha256"]["robot_1"] != export["role_sha256"]["robot_2"]
    frozen = FrozenStructuredDualNN.load(
        path, expected_sha256=export["sha256"],
        expected_role_sha256=export["role_sha256"])
    env = _public_state()
    obs, masks = observations(env.state), action_masks(env.state)
    with torch.no_grad():
        expected = model.outputs(torch.from_numpy(obs), torch.from_numpy(masks)).numpy()
    actual = np.where(masks > 0.5, frozen.logits(obs, masks), -1.0e9)
    assert np.allclose(expected, actual, atol=1e-4, rtol=1e-5)
    actions, distributions = frozen.act(env.state, base_seed=1000)
    for index, role in enumerate(AGENTS):
        assert masks[index, ACTIONS.index(actions[role])] > 0.5
        assert distributions[role].proposed_action == actions[role]
        assert all(probability == 0.0 for probability, legal in
                   zip(distributions[role].probabilities, masks[index], strict=True)
                   if legal <= 0.5)
    _, _, _, _, info = env.step(actions)
    assert info["proposed_actions"] == actions
    assert not info["action_override_agents"]
    wrong_roles = dict(export["role_sha256"])
    wrong_roles["robot_2"] = "0" * 64
    with pytest.raises(RuntimeError, match="role hash"):
        FrozenStructuredDualNN.load(
            path, expected_sha256=export["sha256"],
            expected_role_sha256=wrong_roles)
    path.write_bytes(path.read_bytes() + b"tamper")
    assert sha256(path.read_bytes()).hexdigest() != export["sha256"]
    with pytest.raises(RuntimeError, match="artifact hash"):
        FrozenStructuredDualNN.load(path, expected_sha256=export["sha256"])


def test_frozen_structured_metadata_role_hash_is_checked_even_with_expected_hash(tmp_path):
    model = StructuredDualNN()
    path = tmp_path / "structured_actor.npz"
    artifact = export_structured_actor(model, path, pair_seed=6099, variant="raw")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["role_sha256"]["robot_1"] = "0" * 64
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)
    with pytest.raises(RuntimeError, match="metadata role hash"):
        FrozenStructuredDualNN.load(
            path, expected_sha256=sha256(path.read_bytes()).hexdigest(),
            expected_role_sha256=artifact["role_sha256"])
