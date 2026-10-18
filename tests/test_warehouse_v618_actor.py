"""Additive Actor guarantees, using constructed public states and no resets."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    MODEL_VERSION, V617_MODEL_VERSION, V617_CORE_BOUND_MODE,
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, WarehouseV6Actor,
    _role_arrays, export_actor, role_sha256,
)
from backend.training.warehouse_v6_actor_training import RoleActor
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.public_action_features_v617 import (
    TREE_MASK_START, encode_actor_tree_features,
)
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks


def _rows():
    states = [WarehouseState(
        episode_id=618, frame=frame,
        agents=[AgentState("robot_1", position, 70.0),
                AgentState("robot_2", STUDY_MAP_LAYOUT.robot_start_positions[1], 70.0)],
        tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
               DeliveryTask("task_1", (1, 0), (3, 1))], next_task_index=2)
        for frame, position in ((1, (1, 0)), (2, (1, 1)), (3, (5, 2)))]
    obs = np.stack([encode_public_observation(state) for state in states])
    mask = np.stack([action_masks(state) for state in states])
    return obs, mask, encode_actor_tree_features(obs, mask)


def test_zero_gate_is_exact_primary_nn_and_nonzero_gate_is_additive():
    actor = RoleActor()
    rows = torch.from_numpy(_rows()[2][:, 0])
    actor.set_tree_branch_enabled(False)
    assert all(not parameter.requires_grad for parameter in actor.gate.parameters())
    with torch.no_grad():
        _, relation = actor._relation_features(rows)
        primary = actor.action_residual(relation).squeeze(-1)
        assert torch.equal(actor(rows), primary)
        branch, full_primary, bounds = actor.components(rows)
        assert torch.count_nonzero(branch) == 0
        assert torch.equal(bounds, torch.ones(3))
        assert torch.equal(full_primary, primary)
        actor.gate[-1].bias.fill_(0.75)
        tree, _ = actor.tree(rows)
        expected = primary + np.tanh(.75) * (tree-tree.mean(-1, keepdim=True))
        torch.testing.assert_close(actor(rows), expected)
    actor.set_tree_branch_enabled(True)
    assert all(parameter.requires_grad for parameter in actor.gate.parameters())
    with pytest.raises(ValueError, match="nonzero learned"):
        actor.set_tree_branch_enabled(False)


def test_qualification_and_obsolete_bounds_never_change_forward_or_hash():
    actor = RoleActor()
    rows = torch.from_numpy(_rows()[2][:, 0])
    with torch.no_grad():
        actor.gate[-1].bias.fill_(.5)
        before = actor(rows).clone()
        digest = role_sha256(_role_arrays(actor))
        actor.protected_leaf_mask.fill_(1)
        actor.leaf_bound_logits.fill_(-100)
    actor.install_certified_core_mask(np.ones(16, np.float32))
    assert actor.prune_unproven_core_leaves() == 0
    assert torch.equal(actor(rows), before)
    assert role_sha256(_role_arrays(actor)) == digest
    assert not {"core_leaf_mask", "protected_leaf_mask", "leaf_bound_logits"}.intersection(
        _role_arrays(actor))


def test_topology_freeze_keeps_leaf_outputs_trainable_and_restores_state():
    actor = RoleActor()
    actor.refresh_sparse_support()
    actor.set_topology_trainable(False)
    assert not actor.tree.split.weight.requires_grad
    assert not actor.tree.split.bias.requires_grad
    report = actor.set_certified_core_logits_trainable(False)
    assert report["frozen_leaf_count"] == 0
    assert actor.tree.leaf_logits.requires_grad
    restored = RoleActor()
    restored.load_state_dict(deepcopy(actor.state_dict()))
    restored.set_topology_trainable(False)
    rows = torch.from_numpy(_rows()[2][:, 0])
    assert torch.equal(actor(rows), restored(rows))
    assert torch.equal(actor.tree.support_mask, restored.tree.support_mask)
    assert role_sha256(_role_arrays(actor)) == role_sha256(_role_arrays(restored))


def test_frozen_numpy_parity_export_and_independent_roles(tmp_path):
    model = WarehouseV6Actor()
    obs, mask, encoded = _rows()
    with torch.no_grad():
        for index, role in enumerate(AGENTS):
            model.actors[role].gate[-1].bias.fill_(.4+index*.2)
            model.actors[role].action_residual[-1].weight.normal_(0, .7)
            report = model.actors[role].topology_certificate_report(encoded[:, index])
            assert report["routing_mismatches"] == 0
            assert report["maximum_logit_abs_difference"] < 1e-4
    assert (model.actors[AGENTS[0]].action_residual[0].weight.data_ptr() !=
            model.actors[AGENTS[1]].action_residual[0].weight.data_ptr())
    path = tmp_path / "actor.npz"
    result = export_actor(model, path, pair_seed=1, variant="raw", candidate_id="test")
    frozen = FrozenWarehouseV6Actor.load(path, expected_sha256=result["sha256"])
    assert frozen.model_version == MODEL_VERSION
    expected = np.stack([model.actors[role](torch.from_numpy(encoded[:, index])).detach().numpy()
                         for index, role in enumerate(AGENTS)], axis=1)
    np.testing.assert_allclose(frozen.logits(obs, mask), expected, atol=1e-4, rtol=0)
    assert len(set(frozen.role_sha256.values())) == 2
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FrozenWarehouseV6Actor.load(path, expected_sha256="0"*64)


def test_v617_explicit_dispatch_preserves_bounded_residual_and_hash():
    actor = RoleActor(model_version=V617_MODEL_VERSION)
    rows = _rows()[2][:, 0]
    with torch.no_grad():
        actor.tree.split.weight.zero_()
        actor.tree.split.bias.zero_()
        actor.tree.left_refs[0] = -1
        actor.tree.right_refs[0] = -2
        actor.tree.split.weight[0, TREE_MASK_START+3] = 1
        actor.tree.split.bias[0] = -.5
        actor.tree.node_count[0] = 1
        actor.tree.leaf_count[0] = 2
        actor.tree.leaf_logits.zero_()
        actor.tree.leaf_logits[0, 4] = 3
        actor.tree.leaf_logits[1, 3] = 3
        actor.protected_leaf_mask[:2] = 1
        actor.gate[-1].bias.fill_(10)
        actor.action_residual[-1].bias.fill_(10)
    actor.refresh_sparse_support()
    frozen = FrozenWarehouseV6Role(_role_arrays(actor), core_bound_mode=V617_CORE_BOUND_MODE)
    tree, residual, _ = actor.components(torch.from_numpy(rows))
    assert residual.abs().max().item() <= .080001
    np.testing.assert_allclose(frozen.logits_encoded(rows), (tree+residual).detach().numpy(),
                               atol=1e-4, rtol=0)
    assert frozen.role_sha256 == role_sha256(_role_arrays(actor), model_version=V617_MODEL_VERSION)


def test_primary_can_correct_tree_without_certificate_clamp_and_leaves_receive_gradients():
    actor = RoleActor()
    rows = torch.from_numpy(_rows()[2][:, 0])
    with torch.no_grad():
        actor.gate[-1].bias.fill_(.5)
        actor.action_residual[-1].bias.fill_(20.)
        actor.protected_leaf_mask.fill_(1)
        actor.leaf_bound_logits.fill_(-100.)
    branch, primary, _ = actor.components(rows)
    assert primary.min() > 19.  # No historical tanh or 0.08 residual cap.
    actor.set_topology_trainable(False)
    actor.set_certified_core_logits_trainable(False)
    torch.nn.functional.cross_entropy(actor(rows), torch.tensor([0, 1, 2])).backward()
    assert actor.tree.leaf_logits.grad is not None
    assert actor.tree.leaf_logits.grad.abs().sum() > 0
    assert actor.action_residual[0].weight.grad.abs().sum() > 0
    assert actor.gate[-1].bias.grad.abs().sum() > 0
    assert actor.tree.split.weight.grad is None


def test_sparse_support_surrogate_preserves_exact_forward_coefficients_and_legacy_rounding():
    current = RoleActor()
    legacy = RoleActor(model_version=V617_MODEL_VERSION)
    coefficients = torch.tensor([-9.996800422668457, -9.996600151062012,
                                  -9.996399879455566], dtype=torch.float32)
    for actor in (current, legacy):
        with torch.no_grad():
            actor.tree.split.weight.zero_()
            actor.tree.split.weight[0, :3] = coefficients
        actor.refresh_sparse_support()
    effective = current.tree.effective_weights()
    expected = current.tree.split.weight * current.tree.support_mask
    assert torch.equal(effective, expected)
    # The historical dispatch keeps the original trainer's finite-precision
    # arithmetic, while frozen historical artifacts still load their bytes.
    legacy_expected = (legacy.tree.split.weight * legacy.tree.support_mask
                       + .1*legacy.tree.split.weight - (.1*legacy.tree.split.weight).detach())
    assert torch.equal(legacy.tree.effective_weights(), legacy_expected)
    assert not torch.equal(legacy_expected, expected)
    effective[0].sum().backward()
    torch.testing.assert_close(current.tree.split.weight.grad[0, :3], torch.full((3,), 1.1))
    torch.testing.assert_close(current.tree.split.weight.grad[0, 3:],
                               torch.full_like(current.tree.split.weight.grad[0, 3:], .1))
