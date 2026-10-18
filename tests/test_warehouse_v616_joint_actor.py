"""Focused v6.16 Actor contracts; no training scenario or held-out reset."""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    CORE_BOUND_MODE, CORE_MARGIN_FLOOR, FEATURE_DIM, LEGACY_CORE_BOUND_MODE,
    LEGACY_MODEL_VERSION, MODEL_VERSION, TREE_LEAVES,
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, WarehouseV6Actor,
    _TENSOR_NAMES, _role_arrays, export_actor, role_sha256,
)
from backend.training.warehouse_v6_tree_routing import explicit_mask_branches
from backend.training.warehouse_v6_runner import (
    _clear_core_leaf_adam_momentum, _core_leaf_logits_hashes,
)
from backend.training import warehouse_v6_runner as runner
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v614 import (
    TREE_MASK_START, encode_actor_tree_features,
)
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks


def _encoded_rows() -> np.ndarray:
    states = [WarehouseState(
        episode_id=1616, frame=frame,
        agents=[AgentState("robot_1", position, 70.0),
                AgentState("robot_2", STUDY_MAP_LAYOUT.robot_start_positions[1], 70.0)],
        tasks=[DeliveryTask("task_0", (0, 3), (3, 2)),
               DeliveryTask("task_1", (1, 0), (3, 1))],
        next_task_index=2)
        for frame, position in ((1, (1, 0)), (2, (1, 1)), (3, (5, 2)))]
    return encode_actor_tree_features(
        np.stack([encode_public_observation(state)[0] for state in states]),
        np.stack([action_masks(state)[0] for state in states]))


def _guarded_role():
    actor = WarehouseV6Actor().actors[AGENTS[0]]
    right = ACTIONS.index("RIGHT")
    wait = ACTIONS.index("WAIT")
    with torch.no_grad():
        actor.tree.split.weight.zero_()
        actor.tree.split.bias.zero_()
        actor.tree.split.weight[0, TREE_MASK_START + right] = 1.0
        actor.tree.split.bias[0] = -0.5
        actor.tree.left_refs[0] = -1
        actor.tree.right_refs[0] = -2
        actor.tree.node_count[0] = 1
        actor.tree.leaf_count[0] = 2
        actor.tree.leaf_logits.zero_()
        actor.tree.leaf_logits[0, wait] = 2.0
        actor.tree.leaf_logits[1, right] = 2.0
        actor.tree.guard_node_mask.zero_()
        actor.tree.guard_node_mask[0] = True
    actor.refresh_sparse_support()
    actor.install_certified_core_mask(np.asarray([1, 1] + [0] * (
        TREE_LEAVES - 2), dtype=np.float32))
    return actor


def _guarded_model() -> WarehouseV6Actor:
    model = WarehouseV6Actor()
    for role in AGENTS:
        model.actors[role] = _guarded_role()
        with torch.no_grad():
            model.actors[role].tree.leaf_logits[2].fill_(1.0)
    return model


def test_dedicated_mask_path_and_margin_cap_match_frozen_numpy():
    rows = _encoded_rows()
    actor = _guarded_role()
    certified = actor.topology_certificate_report(rows)
    assert certified["node_count"] == 1
    assert certified["leaf_count"] == 2
    assert certified["maximum_features_per_node"] <= 3
    assert certified["explicit_guard_node_count"] == 1
    assert certified["routing_mismatches"] == 0
    assert certified["maximum_logit_abs_difference"] < 1e-4
    with torch.no_grad():
        tree, residual, learned_bound = actor.components(torch.from_numpy(rows))
        full = tree + residual
    physical = rows[:, TREE_MASK_START:TREE_MASK_START + len(ACTIONS)]
    masked_tree = np.where(physical > 0.5, tree.numpy(), -1e9)
    masked_full = np.where(physical > 0.5, full.numpy(), -1e9)
    np.testing.assert_array_equal(masked_tree.argmax(-1), masked_full.argmax(-1))
    margin = np.sort(masked_tree, axis=-1)[:, -1] - np.sort(masked_tree, axis=-1)[:, -2]
    cap = np.maximum(0.0, (margin - CORE_MARGIN_FLOOR) / 2.0)
    assert np.all(np.abs(residual.numpy()) <= cap[:, None] + 1e-6)
    assert np.all(np.abs(residual.numpy()) <= learned_bound.numpy()[:, None] + 1e-6)
    frozen = FrozenWarehouseV6Role(_role_arrays(actor))
    np.testing.assert_allclose(frozen.logits_encoded(rows), full.numpy(), atol=1e-4, rtol=0)


def test_public_core_initializer_keeps_guard_fixed_and_other_nodes_trainable():
    actor = _guarded_role()
    source = _role_arrays(actor)
    result = SimpleNamespace(
        split_weights=source["tree.split.weight"],
        split_bias=source["tree.split.bias"],
        left_refs=source["tree.left_refs"].astype(np.int64),
        right_refs=source["tree.right_refs"].astype(np.int64),
        leaf_logits=source["tree.leaf_logits"],
        core_mask=source["core_leaf_mask"],
        node_count=1, leaf_count=2, report={"test": True})
    fresh = WarehouseV6Actor().actors[AGENTS[1]]
    report = fresh.initialize_tree_from_core_result(result)
    assert report["split_frozen_after_public_fit"] is False
    assert report["explicit_guard_node_count"] == 1
    assert fresh.tree.split.weight.requires_grad
    assert fresh.tree.split.bias.requires_grad
    assert fresh.tree.guard_node_mask[0]
    assert fresh.topology_certificate_report(_encoded_rows())[
        "routing_mismatches"] == 0


def test_core_install_rejects_move_without_explicit_mask_and_refresh_invalidates():
    actor = _guarded_role()
    with torch.no_grad():
        actor.tree.guard_node_mask.zero_()
        actor.tree.split.weight.zero_()
        actor.tree.split.weight[0, 0] = 1.0
    actor.refresh_sparse_support()
    assert actor.core_leaf_mask.sum().item() == 0
    with pytest.raises(ValueError, match="path legality proof"):
        actor.install_certified_core_mask(np.asarray([0, 1] + [0] * (
            TREE_LEAVES - 2), dtype=np.float32))
    actor.install_certified_core_mask(np.zeros(TREE_LEAVES, dtype=np.float32))


def test_post_update_prunes_only_core_leaf_whose_new_raw_action_lacks_proof():
    actor = _guarded_role()
    left = ACTIONS.index("LEFT")
    with torch.no_grad():
        actor.tree.leaf_logits[1].zero_()
        actor.tree.leaf_logits[1, left] = 3.0
    with pytest.raises(ValueError, match="path legality proof"):
        FrozenWarehouseV6Role(_role_arrays(actor))
    result = actor.prune_unproven_core_leaves()
    assert result == 1
    assert actor.core_leaf_mask.sum().item() == 1
    assert actor.core_leaf_mask[:2].tolist() == [1.0, 0.0]
    assert actor.topology_certificate_report(_encoded_rows())[
        "routing_mismatches"] == 0


def test_topology_freeze_locks_only_certified_leaf_logits():
    actor = _guarded_role()
    with torch.no_grad():
        actor.tree.leaf_logits[2].fill_(1.0)
    logits = actor.tree.leaf_logits
    optimizer = torch.optim.Adam([logits], lr=0.01)
    optimizer.zero_grad()
    logits.square().sum().backward()
    optimizer.step()  # Establish moments before the 75% freeze.

    result = actor.set_certified_core_logits_trainable(False)
    assert result["frozen_leaf_count"] == 2
    core = actor.core_leaf_mask.bool()
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = optimizer.state[logits].get(name)
        if value is not None:
            value[core] = 0
    before = logits.detach().clone()
    optimizer.zero_grad()
    logits.square().sum().backward()
    assert torch.count_nonzero(logits.grad[core]) == 0
    assert torch.count_nonzero(logits.grad[2]) > 0
    optimizer.step()
    torch.testing.assert_close(logits[core], before[core])
    assert not torch.equal(logits[2], before[2])
    assert actor.prune_unproven_core_leaves() == 0

    actor.set_certified_core_logits_trainable(True)
    optimizer.zero_grad()
    logits.square().sum().backward()
    assert torch.count_nonzero(logits.grad[core]) > 0


def test_runner_core_hash_and_momentum_clear_survive_optimizer_step():
    model = _guarded_model()
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=0.01)
    optimizer.zero_grad()
    sum(actor.tree.leaf_logits.square().sum()
        for actor in model.actors.values()).backward()
    optimizer.step()  # Create nonzero Adam moments in certified rows.

    for role in AGENTS:
        actor = model.actors[role]
        actor.set_topology_trainable(False)
        actor.set_certified_core_logits_trainable(False)
    _clear_core_leaf_adam_momentum(optimizer, model)
    frozen_hash = _core_leaf_logits_hashes(model)
    before = {role: model.actors[role].tree.leaf_logits.detach().clone()
              for role in AGENTS}
    for role in AGENTS:
        actor = model.actors[role]
        core = actor.core_leaf_mask.bool()
        state = optimizer.state[actor.tree.leaf_logits]
        assert torch.count_nonzero(state["exp_avg"][core]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][core]) == 0

    optimizer.zero_grad()
    rows = torch.from_numpy(_encoded_rows())
    loss = sum(actor(rows).square().mean() +
               0.01 * actor.tree.leaf_logits.square().sum()
               for actor in model.actors.values())
    loss.backward()
    optimizer.step()
    assert _core_leaf_logits_hashes(model) == frozen_hash
    for role in AGENTS:
        actor = model.actors[role]
        core = actor.core_leaf_mask.bool()
        torch.testing.assert_close(actor.tree.leaf_logits[core], before[role][core])
        assert not torch.equal(actor.tree.leaf_logits[2], before[role][2])
        assert actor.prune_unproven_core_leaves() == 0


def test_checkpoint_load_reapplies_certified_core_freeze():
    model = _guarded_model()
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=0.01)
    optimizer.zero_grad()
    sum(actor.tree.leaf_logits.square().sum()
        for actor in model.actors.values()).backward()
    optimizer.step()
    for actor in model.actors.values():
        actor.set_topology_trainable(False)
        actor.set_certified_core_logits_trainable(False)
    _clear_core_leaf_adam_momentum(optimizer, model)
    checkpoint_model = deepcopy(model.state_dict())
    checkpoint_optimizer = deepcopy(optimizer.state_dict())
    frozen_hash = _core_leaf_logits_hashes(model)

    restored = WarehouseV6Actor()
    restored.load_state_dict(checkpoint_model, strict=True)
    restored_optimizer = torch.optim.Adam(restored.actors.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(checkpoint_optimizer)
    for actor in restored.actors.values():
        actor.set_topology_trainable(False)
        actor.set_certified_core_logits_trainable(False)
    assert _core_leaf_logits_hashes(restored) == frozen_hash
    restored_optimizer.zero_grad()
    sum(actor.tree.leaf_logits.square().sum()
        for actor in restored.actors.values()).backward()
    restored_optimizer.step()
    assert _core_leaf_logits_hashes(restored) == frozen_hash
    for actor in restored.actors.values():
        assert actor.prune_unproven_core_leaves() == 0


def test_train_variant_resume_after_75_percent_matches_continuous_run(
        monkeypatch, tmp_path):
    """Exercise the real runner checkpoint state machine without environment resets."""
    class InterruptedTraining(RuntimeError):
        pass

    monkeypatch.setattr(runner, "_require_locked_execution", lambda: None)
    monkeypatch.setattr(runner, "_assert_source_closure", lambda _expected: None)
    monkeypatch.setattr(runner, "stage1_reset_phase",
                        lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(runner, "_environment", lambda **_kwargs:
                        (_ for _ in ()).throw(AssertionError("unexpected environment")))

    def fake_refresh(model, _buffers, *, stage, refresh_support=True):
        core = np.asarray([1, 1] + [0] * (TREE_LEAVES - 2), dtype=np.float32)
        for role in AGENTS:
            actor = model.actors[role]
            if refresh_support:
                actor.refresh_sparse_support()
            actor.install_certified_core_mask(core)
        return {"stage": stage, "roles": {role: {
                    "qualification_bound_to_installed_actor": True,
                    "qualification": {"validation_passed": True}}
                    for role in AGENTS},
                "role_tree_sha256": runner._role_topology_hashes(model),
                "route_parity_passed": True, "path_legality_passed": True,
                "core_qualification_recomputed": True}

    observation = np.zeros((len(AGENTS), runner.OBSERVATION_DIM), dtype=np.float32)
    masks = np.ones((len(AGENTS), len(ACTIONS)), dtype=np.float32)
    wait = ACTIONS.index("WAIT")
    right = ACTIONS.index("RIGHT")
    active = {"interrupt": False}
    phases = []

    def fake_episode(_model, *, seed, generator, max_steps, episode_index):
        records = [{"obs": observation, "mask": masks,
                    "action": np.full(len(AGENTS), wait, dtype=np.int64),
                    "seed": seed, "episode_id": episode_index,
                    "episode_key": f"{seed}:{episode_index}", "frame": frame}
                   for frame in range(max_steps)]
        complete = max_steps == runner.SCENARIO_BLOCK_STEPS
        summary = {"seed": seed, "episode_index": episode_index,
                   "episode_key": f"{seed}:{episode_index}",
                   "joint_steps": len(records), "complete": complete,
                   "shutdown_episode": False}
        return ({"records": records, "complete": complete,
                 "shutdown_episode": False}, summary)

    def fake_ppo_update(model, optimizer, _batch, _programs, *,
                        topology_frozen, **_kwargs):
        phases.append(topology_frozen)
        for actor in model.actors.values():
            assert actor.tree.split.weight.requires_grad is not topology_frozen
            assert actor._certified_core_logits_trainable is not topology_frozen
        loss = -sum(actor.tree.leaf_logits[0, wait] +
                    actor.tree.leaf_logits[1, right] +
                    actor.tree.leaf_logits[2, wait]
                    for actor in model.actors.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for actor in model.actors.values():
            assert actor.prune_unproven_core_leaves() == 0
        zeros = {role: 0 for role in AGENTS}
        return {"feedback_rows": zeros, "feedback_gradient_norm": zeros,
                "physical_no_legal_action_rows": zeros,
                "feedback_disabled_epochs": zeros,
                "feedback_window_actor_rows": zeros,
                "zero_action_coverage_rows": zeros}

    def fake_evaluate(model, _seeds):
        quality = float(model.actors[AGENTS[0]].tree.leaf_logits[2, wait].detach())
        return {"summary": {"quality": quality,
                            "formal_quality_passed": True,
                            "development_quality_passed": True}}

    monkeypatch.setattr(runner, "_refresh_qualified_topology", fake_refresh)
    monkeypatch.setattr(runner, "collect_episode", fake_episode)
    monkeypatch.setattr(runner, "ppo_update", fake_ppo_update)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    original_save = runner._save_checkpoint

    def maybe_interrupt(path, payload):
        original_save(path, payload)
        if (active["interrupt"] and not payload["complete"] and
                payload["used_steps"] >= 750):
            active["interrupt"] = False
            raise InterruptedTraining("checkpoint saved after topology freeze")

    monkeypatch.setattr(runner, "_save_checkpoint", maybe_interrupt)
    initial = deepcopy(_guarded_model().state_dict())

    def run(directory):
        directory.mkdir(parents=True, exist_ok=True)
        spec = {"candidate_id": "freeze-resume-smoke", "pair_seed": 1616,
                "joint_steps": 1000, "output_dir": str(directory),
                "scenario_namespace": "freeze-resume-smoke",
                "checkpoint_seeds": [1617], "smoke": True,
                "source_closure_sha256": runner.source_closure_sha256(),
                "topology_refresh_steps": [240, 480, 720],
                "topology_frozen_at_step": 750,
                "online_program_extraction_steps": []}
        return runner._train_variant(initial, variant="raw", spec=spec,
                                     directory=directory)

    continuous, _, continuous_report = run(tmp_path / "continuous")
    assert any(not frozen for frozen in phases) and any(phases)
    phases.clear()
    active["interrupt"] = True
    with pytest.raises(InterruptedTraining):
        run(tmp_path / "resumed")
    checkpoint = tmp_path / "resumed" / "raw.checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["topology_frozen"] is True
    assert saved["used_steps"] >= 750
    phases.clear()
    resumed, _, resumed_report = run(tmp_path / "resumed")
    assert phases and all(phases)
    assert resumed_report["selected_role_sha256"] == \
        continuous_report["selected_role_sha256"]
    assert resumed_report["final_topology_role_sha256"] == \
        continuous_report["final_topology_role_sha256"]
    assert resumed_report["final_core_leaf_logits_sha256"] == \
        continuous_report["final_core_leaf_logits_sha256"]
    for role in AGENTS:
        assert not continuous.actors[role].tree.split.weight.requires_grad
        assert not resumed.actors[role].tree.split.weight.requires_grad
        assert not resumed.actors[role]._certified_core_logits_trainable


def test_guard_node_immutable_while_inactive_support_receives_gradient():
    actor = _guarded_role()
    tree = actor.tree
    assert tree.split.weight.requires_grad and tree.split.bias.requires_grad
    rows = torch.from_numpy(_encoded_rows())
    before_weight = tree.split.weight.detach().clone()
    before_bias = tree.split.bias.detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss = actor(rows).square().sum()
    optimizer.zero_grad()
    loss.backward()
    assert torch.count_nonzero(tree.split.weight.grad[0]) == 0
    assert tree.split.bias.grad[0] == 0
    optimizer.step()
    torch.testing.assert_close(tree.split.weight[0], before_weight[0])
    torch.testing.assert_close(tree.split.bias[0], before_bias[0])

    # A non-guard split is hard sparse in the forward pass but unselected
    # features remain trainable through the support surrogate.
    with torch.no_grad():
        tree.guard_node_mask[0] = False
        tree.split.weight[0].zero_()
        tree.split.weight[0, 0] = 1.0
        tree.split.bias[0] = -0.5
        tree.leaf_logits[0].zero_()
        tree.leaf_logits[1].fill_(1.0)
    actor.refresh_sparse_support()
    row = rows.clone()
    row[:, 10] = 1.0
    tree.split.weight.grad = None
    tree(row)[0].sum().backward()
    assert tree.split.weight.grad[0, 10].abs() > 0
    assert np.count_nonzero(_role_arrays(actor)["tree.split.weight"][0]) <= 3


def test_legacy_bound_mode_is_dispatched_from_metadata(tmp_path):
    current = tmp_path / "v616.npz"
    export_actor(WarehouseV6Actor(), current, pair_seed=616,
                 variant="raw", candidate_id="joint-certified-test")
    with np.load(current, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["core_bound_mode"] == CORE_BOUND_MODE
    metadata["model_version"] = LEGACY_MODEL_VERSION
    metadata["core_bound_mode"] = LEGACY_CORE_BOUND_MODE
    metadata["role_sha256"] = {role: role_sha256({
        name: arrays[f"{role}.{name}"] for name in _TENSOR_NAMES},
        model_version=LEGACY_MODEL_VERSION) for role in AGENTS}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    legacy_path = tmp_path / "v614.npz"
    np.savez_compressed(legacy_path, **arrays)
    loaded = FrozenWarehouseV6Actor.load(
        legacy_path, expected_sha256=sha256(legacy_path.read_bytes()).hexdigest())
    assert loaded.model_version == LEGACY_MODEL_VERSION
    assert all(role.core_bound_mode == LEGACY_CORE_BOUND_MODE
               for role in loaded.roles.values())
    current_loaded = FrozenWarehouseV6Actor.load(
        current, expected_sha256=sha256(current.read_bytes()).hexdigest())
    assert current_loaded.role_sha256 != loaded.role_sha256
    metadata["core_bound_mode"] = CORE_BOUND_MODE
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    forged = tmp_path / "forged.npz"
    np.savez_compressed(forged, **arrays)
    with pytest.raises(RuntimeError, match="artifact contract mismatch"):
        FrozenWarehouseV6Actor.load(
            forged, expected_sha256=sha256(forged.read_bytes()).hexdigest())


def test_training_state_loader_accepts_pre_support_checkpoint():
    source = WarehouseV6Actor()
    state = {key: value for key, value in source.state_dict().items()
             if not key.endswith(("support_mask", "support_fixed",
                                  "guard_node_mask"))}
    restored = WarehouseV6Actor()
    restored.load_state_dict(state, strict=True)
    for role in AGENTS:
        tree = restored.actors[role].tree
        assert not bool(tree.support_fixed[0])
        assert not bool(tree.guard_node_mask.any())


def test_explicit_guard_rejects_oblique_public_split():
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    row[TREE_MASK_START + ACTIONS.index("LEFT")] = 1.0
    assert ACTIONS.index("LEFT") in explicit_mask_branches(
        row, -0.5, tuple(range(TREE_MASK_START, TREE_MASK_START + len(ACTIONS))))[1]
    row[0] = 0.1
    assert explicit_mask_branches(
        row, -0.5, tuple(range(TREE_MASK_START, TREE_MASK_START + len(ACTIONS)))) == (
            frozenset(), frozenset())
