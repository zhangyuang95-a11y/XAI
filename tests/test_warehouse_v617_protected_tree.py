"""Structural guarantees that specifically prevent the zero-core deadlock."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Role, WarehouseV6Actor, _role_arrays, role_sha256,
)
from backend.training.warehouse_v617_risk_tree import (
    fit_risk_tree, fit_boxed_logits, PROTECTED_RESIDUAL_CAP,
)
from backend.training.warehouse_v6_core_tree import _masked_probabilities
from env.warehouse.public_action_features_v617 import (
    append_context, CONTEXT_INDEX, NEAR_CONTEXT_INDEX,
)
from tests.test_warehouse_v616_joint_actor import _guarded_role, _encoded_rows


def test_qualification_sidecar_never_changes_frozen_actor_identity_or_logits():
    actor = _guarded_role()
    rows = _encoded_rows()
    before = FrozenWarehouseV6Role(_role_arrays(actor))
    actor.install_certified_core_mask(np.zeros(16, np.float32))
    after = FrozenWarehouseV6Role(_role_arrays(actor))
    assert before.role_sha256 == after.role_sha256
    np.testing.assert_array_equal(before.logits_encoded(rows), after.logits_encoded(rows))
    _, residual, _ = actor.components(torch.from_numpy(rows))
    assert residual.abs().max().item() <= PROTECTED_RESIDUAL_CAP + 1e-7
    assert actor.core_leaf_mask.sum() == 0
    assert actor.protected_leaf_mask.sum() == 2


def test_unprotected_leaf_cannot_gain_qualification_by_installing_a_certificate():
    actor = _guarded_role()
    with torch.no_grad():
        actor.protected_leaf_mask[1] = 0
    with pytest.raises(ValueError, match="protected subset"):
        actor.install_certified_core_mask(np.array([1, 1]+[0]*14, np.float32))


def test_context_bit_uses_exact_auditor_boundaries_and_detects_tampering():
    from env.warehouse.public_action_features_v614 import TREE_FEATURE_NAMES
    from env.warehouse.public_action_features_v617 import action_features_from_tree_encoded
    base = _encoded_rows()[:, :171].copy()
    base[:, TREE_FEATURE_NAMES.index("same_target_collision_streak_capped")] = [0, 1, 0]
    base[:, TREE_FEATURE_NAMES.index("self_charger_return_margin")] = [4/50, 1, 5/50]
    base[:, TREE_FEATURE_NAMES.index("robots_shortest_distance")] = [2/42, 3/42, 1/42]
    result = append_context(base)
    np.testing.assert_array_equal(result[:, CONTEXT_INDEX], [1, 1, 0])
    np.testing.assert_array_equal(result[:, NEAR_CONTEXT_INDEX], [1, 0, 1])
    real = _encoded_rows().copy()
    real[:, CONTEXT_INDEX] = 1-real[:, CONTEXT_INDEX]
    with pytest.raises(ValueError, match="altered"):
        action_features_from_tree_encoded(real)


def test_fit_only_tree_has_legal_protected_paths_and_no_independent_certificate():
    rows = np.repeat(_encoded_rows(), 100, axis=0)
    masks = rows[:, 141:146]
    logits = np.zeros_like(masks)
    logits[:, 4] = 1
    logits[:, 3] = 3  # Full NN target distribution, never an executed rule.
    target = _masked_probabilities(logits, masks)
    episodes = tuple(f"nn-visit:{900000+i//3}" for i in range(len(rows)))
    fitted = fit_risk_tree(rows, masks, target, episodes)
    assert fitted.leaf_count <= 16
    assert fitted.node_count <= 15
    assert fitted.report["fit_only_not_independent_certificate"]
    actor = WarehouseV6Actor().actors["robot_1"]
    actor.initialize_tree_from_core_result(fitted)
    assert actor.core_leaf_mask.sum() == 0
    assert actor.protected_leaf_mask.sum() > 0
    proof = actor.topology_certificate_report(rows)
    assert proof["routing_mismatches"] == 0
    assert proof["maximum_features_per_node"] <= 3
    frozen = FrozenWarehouseV6Role(_role_arrays(actor))
    leaf = frozen.leaf(rows)
    for i in range(len(rows)):
        if frozen.weights["protected_leaf_mask"][leaf[i]]:
            action = fitted.leaf_logits[leaf[i]].argmax()
            assert masks[i, action] == 1


def test_boxed_distribution_refit_has_masked_kl_bound_for_all_physical_masks():
    actor = _guarded_role()
    rows = _encoded_rows()
    frozen = FrozenWarehouseV6Role(_role_arrays(actor))
    center = frozen.weights["tree.leaf_logits"][1]
    for bits in range(16):
        mask = np.array([bool(bits & (1 << i)) for i in range(4)]+[True], np.float32)
        # Physical WAIT always remains available. This verifies the bound
        # independently of the observed training-mask distribution.
        delta = np.array([-.08, .08, -.06, .07, .01], np.float32)
        p = _masked_probabilities((center+delta)[None], mask[None])
        z = fit_boxed_logits(p, mask[None], center)
        q = _masked_probabilities(z[None], mask[None])
        kl = np.sum(p * (np.log(np.maximum(p, 1e-9))-np.log(np.maximum(q, 1e-9))))
        assert kl <= .24 + 1e-6


def test_protection_buffer_is_checkpointed_and_hashed():
    actor = _guarded_role()
    restored = _guarded_role()
    restored.load_state_dict(actor.state_dict())
    assert role_sha256(_role_arrays(actor)) == role_sha256(_role_arrays(restored))
    with torch.no_grad():
        restored.protected_leaf_mask[1] = 0
    assert role_sha256(_role_arrays(actor)) != role_sha256(_role_arrays(restored))


def test_actual_topology_refresh_rebuilds_from_fit_episodes_and_binds_sidecar():
    from backend.training.warehouse_v6_runner import _refresh_qualified_topology
    rows = _encoded_rows()
    buffers = {role: [
        {"obs": row[:141], "mask": row[141:146], "base_seed": 910000+episode,
         "teacher_action": 4, "actual_collision": 0.0}
        for episode in range(48) for row in rows]
        for role in ("robot_1", "robot_2")}
    torch.set_num_threads(2)
    model = WarehouseV6Actor()
    receipt = _refresh_qualified_topology(model, buffers, stage="test_fit_only")
    assert receipt["route_parity_passed"] and receipt["path_legality_passed"]
    for role in buffers:
        entry = receipt["roles"][role]
        assert entry["support"]["fit_only_alignment_epochs"] == 4
        assert entry["qualification_bound_to_installed_actor"]
        assert entry["protection_is_not_certificate"]
        assert entry["qualification"]["frozen_role_sha256"] == role_sha256(
            _role_arrays(model.actors[role]))


def test_bc_quality_screen_enters_real_reset_audit_context(tmp_path):
    from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from env.warehouse.domain import participant_study_config
    with capture_stage1_resets(tmp_path / "resets.jsonl") as rows:
        with stage1_reset_phase("bc_quality_screen"):
            WarehouseMultiAgentEnv(participant_study_config()).reset(seed=950617)
    assert len(rows) == 1
    assert rows[0]["phase"] == "bc_quality_screen"
    assert rows[0]["seed"] == 950617


def test_independent_bc_collection_retains_repeated_collision_frames(monkeypatch):
    from types import SimpleNamespace
    from backend.training import warehouse_v6_runner as runner
    class Environment:
        def reset(self, seed):
            self.state = SimpleNamespace(terminated=False, truncated=False,
                robot_collision_events=0, total_deliveries=0, shutdown_count=0,
                by_id=lambda _: SimpleNamespace(position=(1, 1)))
        def step(self, actions):
            self.state.robot_collision_events += 1
            self.state.truncated = self.state.robot_collision_events == 120
            return None, None, None, None, {"proposed_actions": actions,
                                           "action_override_agents": []}
    roles = ("robot_1", "robot_2")
    actor = SimpleNamespace(act=lambda *a, **k: (
        {r: "UP" for r in roles},
        {r: SimpleNamespace(probabilities=(1, 0, 0, 0, 0)) for r in roles}))
    monkeypatch.setattr(runner.FrozenWarehouseV6Actor, "from_model", lambda _: actor)
    monkeypatch.setattr(runner, "_environment", lambda **_: Environment())
    monkeypatch.setattr(runner, "_set_dual_nn_mode", lambda _: None)
    monkeypatch.setattr(runner, "encode_public_observation", lambda _: np.zeros((2, 141), np.float32))
    monkeypatch.setattr(runner, "action_masks", lambda _: np.ones((2, 5), np.float32))
    monkeypatch.setattr(runner, "public_lookahead_teacher", lambda *_: np.zeros(2, np.int64))
    rows, report = runner._collect_bc_visits(None, [950618], deterministic=True,
                                           deduplicate_collisions=False)
    assert len(rows[0]) == report["physical_frames"] == 120
    assert report["excluded_frames"] == 0
    training_rows, training_report = runner._collect_bc_visits(
        None, [950618], deterministic=True, deduplicate_collisions=True)
    assert len(training_rows[0]) < 120
    assert training_report["excluded_frames"] == 120-len(training_rows[0])


def test_admission_rejects_a_bc_precheck_with_filtered_frames():
    from experiments.warehouse import protocol as p
    from tests.warehouse_v6_contract_fixture import bc_receipt
    import json
    seed = p.DEVELOPMENT_PAIR_SEEDS[0]
    receipt = bc_receipt(seed, "f"*64)
    assert p.bc_report_valid(receipt, seed, "f"*64)
    visits = receipt["final_bc_internal_precheck"]["final_bc_visits"]
    next(iter(visits.values()))["excluded_frames"] = 1
    receipt["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        (json.dumps(receipt["final_bc_internal_precheck"],sort_keys=True,allow_nan=False)+"\n").encode()).hexdigest()
    assert not p.bc_report_valid(receipt, seed, "f"*64)
