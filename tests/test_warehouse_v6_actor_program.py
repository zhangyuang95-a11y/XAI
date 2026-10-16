"""Warehouse v6 parity, program fidelity gates, and paired runner smoke."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor,
)
from backend.training.warehouse_v6_program import (
    FEATURE_DIM, WarehouseV6Program, _metrics, fit_program, audit_frozen_program,
    current_actor_fidelity, actor_sparse_topology_matches,
    ACTOR_SPARSE_TOPOLOGY_METHOD,
)
from backend.training.warehouse_v6_runner import (
    _program_feedback, audit_programs, collect_program_audit_rows,
    _resample_program_rows, collect_episode, ppo_update,
    source_closure_sha256, train_pair,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import action_masks
from env.warehouse.public_observation_v6 import (
    encode_policy_features, encode_public_observation,
)

SMOKE_PAIR_SEED = 986_000
SMOKE_EVALUATION_SEEDS = (987_000, 987_001)


def _state(seed: int):
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=seed)
    marked = env.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    env.set_state(marked)
    return env


def _split_rows(actor, role_index=0):
    groups = []
    for start in (100, 102, 104, 106):
        rows = []
        for seed in (start, start + 1):
            env = _state(seed)
            for _ in range(4):
                obs, masks = encode_public_observation(env.state), action_masks(env.state)
                commands, _ = actor.act(env.state, base_seed=seed)
                role = ("robot_1", "robot_2")[role_index]
                rows.append({"obs": obs[role_index].copy(),
                             "mask": masks[role_index].copy(),
                             "episode": str(seed), "base_seed": seed,
                             "sampled_action": commands[role]})
                env.step(commands)
        groups.append(rows)
    return groups


def test_actor_torch_numpy_export_parity_on_public_states(tmp_path):
    torch.manual_seed(41)
    model = WarehouseV6Actor()
    path = tmp_path / "v6.npz"
    artifact = export_actor(model, path, pair_seed=SMOKE_PAIR_SEED,
                            variant="regularized", candidate_id="test")
    frozen = FrozenWarehouseV6Actor.load(path, expected_sha256=artifact["sha256"],
                                          expected_role_sha256=artifact["role_sha256"])
    for seed in (101, 102, 103):
        env = _state(seed)
        for _ in range(3):
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            with torch.no_grad():
                expected = model.outputs(torch.from_numpy(obs)[None],
                                         torch.from_numpy(masks)[None])[0].numpy()
            actual = np.where(masks > 0.5, frozen.logits(obs, masks), -1e9)
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
            commands, distributions = frozen.act(env.state, base_seed=seed)
            for role, row in (("robot_1", 0), ("robot_2", 1)):
                assert masks[row, ACTIONS.index(commands[role])] == 1
                assert distributions[role].proposed_action == commands[role]
            env.step(commands)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FrozenWarehouseV6Actor.load(path, expected_sha256="0" * 64)


def test_program_uses_full_masked_nn_and_rejects_raw_illegal_actions():
    torch.manual_seed(42)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    program = fit_program("robot_1", actor, fit, calibration, eligibility, validation)
    assert program.report["split_episodes"] == {
        "fit": 2, "calibration": 2, "eligibility": 2, "validation": 2}
    sample = validation[0]
    leaf, _ = program._leaf(sample)
    assert 0 <= leaf < program.report["leaf_count"] <= 16
    assert len(program.lookup(sample)["path"]) <= 5
    assert program.lookup(sample)["margin_certified"] is False
    with pytest.raises(ValueError, match="reused fitting episodes"):
        audit_frozen_program(program, actor, validation)
    repeated_scenario = [{**row, "episode": f"audit-{row['episode']}",
                          "base_seed": fit[0]["base_seed"]}
                         for row in validation]
    with pytest.raises(ValueError, match="reused fitting scenario seeds"):
        audit_frozen_program(program, actor, repeated_scenario)
    illegal = int(np.flatnonzero(sample["mask"] <= 0.5)[0])
    changed = program.leaf_logits.copy()
    changed[leaf] = -20
    changed[leaf, illegal] = 20
    invalid = replace(program, leaf_logits=changed)
    metric = _metrics(invalid, actor.roles["robot_1"], [sample], {leaf})
    assert metric["illegal_count"] == 1
    assert metric["masked_illegal_count"] == 0
    assert invalid.lookup(sample)["raw_action_legal"] is False


def test_program_validation_cannot_reselect_eligibility_leaves():
    torch.manual_seed(45)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    original = fit_program("robot_1", actor, fit, calibration, eligibility, validation)
    changed_validation = [{**row, "sampled_action": "WAIT"}
                          for row in validation]
    changed = fit_program("robot_1", actor, fit, calibration, eligibility,
                          changed_validation)
    assert original.feedback_leaves == changed.feedback_leaves
    assert original.explanation_leaves == changed.explanation_leaves
    np.testing.assert_array_equal(original.split_weights, changed.split_weights)
    np.testing.assert_array_equal(original.leaf_logits, changed.leaf_logits)


def test_program_splits_keep_repeated_scenario_resets_together():
    torch.manual_seed(451)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    # Different episode IDs can be resets of the same scenario seed. They
    # cannot cross fit/calibration/eligibility/validation boundaries.
    repeated = [{**row, "base_seed": fit[0]["base_seed"]}
                for row in validation]
    with pytest.raises(ValueError, match="scenario seed splits overlap"):
        fit_program("robot_1", actor, fit, calibration, eligibility, repeated)


def test_program_fit_targets_masked_nn_distribution_not_sampled_labels():
    torch.manual_seed(452)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    original = fit_program("robot_1", actor, fit, calibration, eligibility, validation)
    changed_fit = [{**row, "sampled_action": "WAIT"} for row in fit]
    changed = fit_program("robot_1", actor, changed_fit, calibration,
                          eligibility, validation)
    np.testing.assert_array_equal(original.split_weights, changed.split_weights)
    np.testing.assert_array_equal(original.split_bias, changed.split_bias)
    np.testing.assert_array_equal(original.leaf_logits, changed.leaf_logits)


def test_program_can_extract_sparse_actor_topology_without_copying_leaf_actions():
    torch.manual_seed(453)
    model = WarehouseV6Actor()
    actor = FrozenWarehouseV6Actor.from_model(model)
    fit, calibration, eligibility, validation = _split_rows(actor)
    program = fit_program(
        "robot_1", actor, fit, calibration, eligibility, validation,
        method=ACTOR_SPARSE_TOPOLOGY_METHOD)
    assert actor_sparse_topology_matches(program, actor.roles["robot_1"])
    assert program.report["fit_method"] == ACTOR_SPARSE_TOPOLOGY_METHOD
    assert len(program.leaf_logits) <= 16
    assert program.report["max_depth"] <= 5
    assert np.count_nonzero(program.split_weights, axis=1).max(initial=0) <= 3
    assert sum(program.report["fit_leaf_row_counts"]) == len(fit)
    supported = {i for i, count in enumerate(
        program.report["fit_leaf_row_counts"]) if count}
    assert program.feedback_leaves <= supported
    assert program.explanation_leaves <= supported

    # The sampled label cannot influence the full-distribution fit. The
    # program topology is copied, while leaf logits come from complete NN
    # probabilities on fit rows rather than the Actor tree's stored actions.
    changed_fit = [{**row, "sampled_action": "WAIT"} for row in fit]
    changed = fit_program(
        "robot_1", actor, changed_fit, calibration, eligibility, validation,
        method=ACTOR_SPARSE_TOPOLOGY_METHOD)
    np.testing.assert_array_equal(program.leaf_logits, changed.leaf_logits)
    np.testing.assert_array_equal(program.split_weights, changed.split_weights)
    tampered = program.split_weights.copy()
    tampered[0, int(np.flatnonzero(tampered[0])[0])] += 0.25
    assert not actor_sparse_topology_matches(
        replace(program, split_weights=tampered), actor.roles["robot_1"])
    with pytest.raises(RuntimeError, match="topology differs"):
        audit_frozen_program(
            replace(program, split_weights=tampered), actor, validation)

    # The exact sparse route is shared by the extracted program, frozen NumPy
    # Actor, and Torch training Actor away from floating-point ambiguity.
    x = encode_policy_features(
        np.stack([row["obs"] for row in fit]),
        np.stack([row["mask"] for row in fit]))
    root_weights = program.split_weights[0]
    first = int(np.flatnonzero(root_weights)[0])
    boundary = np.zeros(FEATURE_DIM, dtype=np.float32)
    boundary[first] = -program.split_bias[0] / root_weights[first]
    near = np.stack([boundary.copy(), boundary.copy()])
    near[0, first] -= 0.01
    near[1, first] += 0.01
    x = np.concatenate((x, near), axis=0)
    expected = actor.roles["robot_1"].leaf(x)
    np.testing.assert_array_equal(program.leaf_indices(x), expected)
    with torch.no_grad():
        trained = model.actors["robot_1"].tree.path(
            torch.from_numpy(x)).argmax(-1).numpy()
    np.testing.assert_array_equal(trained, expected)


def test_unfitted_actor_leaf_cannot_become_eligible_from_later_splits(monkeypatch):
    from backend.training import warehouse_v6_program as module
    torch.manual_seed(455)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    # Deliberately make the two later-stage checks permissive: a leaf with no
    # fit data must remain ineligible regardless of subsequent observations.
    monkeypatch.setattr(module, "_eligible", lambda _metric, _threshold: True)
    program = fit_program(
        "robot_1", actor, fit, calibration, eligibility, validation,
        method=ACTOR_SPARSE_TOPOLOGY_METHOD)
    counts = program.report["fit_leaf_row_counts"]
    assert 0 in counts
    supported = {index for index, count in enumerate(counts) if count}
    assert program.feedback_leaves == supported
    assert program.explanation_leaves == supported


def test_audit_pass_requires_explanation_for_robot_1(monkeypatch):
    from backend.training import warehouse_v6_program as module
    torch.manual_seed(454)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    program = fit_program("robot_1", actor, *_split_rows(actor))
    report = {**program.report, "validation": {
        **program.report["validation"],
        "feedback_passed": True, "explanation_passed": True}}
    program = replace(program, report=report)
    fresh = [{**row, "episode": f"fresh-{row['episode']}",
              "base_seed": int(row["base_seed"]) + 1000}
             for row in _split_rows(actor)[-1]]

    def metric(_program, _actor, _rows, _leaves, purpose):
        return {"coverage": 0.4 if purpose == "feedback" else 0.0,
                "actionable_coverage": 0.35 if purpose == "feedback" else 0.0,
                "agreement": 0.98, "kl": 0.1, "illegal_count": 0,
                "rows": 20, "actionable_rows": 18, "sample_count": len(_rows)}

    monkeypatch.setattr(module, "_metrics", metric)
    receipt = audit_frozen_program(program, actor, fresh)
    assert receipt["feedback_eligible"] is True
    assert receipt["explanation_eligible"] is False
    assert receipt["passed"] is False


def test_actionable_coverage_uses_actual_nn_sample_and_raw_legality():
    torch.manual_seed(46)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    program = fit_program("robot_1", actor, fit, calibration, eligibility, validation)
    sample = validation[0]
    leaf, _ = program._leaf(sample)
    lookup = program.lookup(sample)
    matching = {**sample, "sampled_action": lookup["action"]}
    metric = _metrics(program, actor.roles["robot_1"], [matching], {leaf},
                      "explanation")
    assert metric["actionable_rows"] == int(lookup["raw_action_legal"])
    other = next(action for action in ACTIONS if action != lookup["action"]
                 and sample["mask"][ACTIONS.index(action)] > 0.5)
    disagreement = _metrics(program, actor.roles["robot_1"],
                            [{**sample, "sampled_action": other}], {leaf},
                            "explanation")
    assert disagreement["rows"] == metric["rows"] == 1
    assert disagreement["actionable_rows"] == 0


def test_explanation_requires_the_current_frozen_actor_role():
    torch.manual_seed(460)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    rows = _split_rows(actor, role_index=1)
    program = fit_program("robot_2", actor, *rows)
    # Exercise the runtime contract with a real program object. Eligibility
    # and validation are prepared here because this test checks binding, not
    # the independent scientific admission threshold.
    report = {**program.report,
              "validation": {**program.report["validation"],
                             "explanation_passed": True}}
    bound = replace(program, report=report,
                    explanation_leaves=frozenset(range(len(program.leaf_logits))))
    sample = {"obs": rows[-1][0]["obs"],
              "mask": np.ones(len(ACTIONS), dtype=np.float32)}
    selected = bound.lookup(sample)["action"]
    evidence = bound.covered(sample, actor_action=selected,
                             purpose="explanation",
                             actor_role=actor.roles["robot_2"])
    assert evidence is not None and evidence["actor_role_sha256"] == \
        actor.role_sha256["robot_2"]
    assert bound.covered(sample, actor_action=selected,
                         purpose="explanation") is None
    assert bound.covered(sample, actor_action=selected,
                         purpose="explanation",
                         actor_role=actor.roles["robot_1"]) is None
    other = next(action for action in ACTIONS if action != selected)
    assert bound.covered(sample, actor_action=other,
                         purpose="explanation",
                         actor_role=actor.roles["robot_2"]) is None


def test_program_audit_trace_rebuilds_public_state_and_is_immutable(tmp_path):
    torch.manual_seed(47)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    samples = collect_program_audit_rows(actor, (100,))
    from env.warehouse.decision_protocol import canonical_sha256
    from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state
    for role in ("robot_1", "robot_2"):
        assert len(samples[role]) > 0
        first = samples[role][0]
        state = rebuild_public_pre_action_state(first["pre_state_public"])
        index = ("robot_1", "robot_2").index(role)
        np.testing.assert_array_equal(first["obs"], encode_public_observation(state)[index])
        np.testing.assert_array_equal(first["mask"], action_masks(state)[index])
        assert canonical_sha256(first["pre_state_public"]) == first[
            "pre_state_public_sha256"]
        assert first["nn_sampled_action"] == first["sampled_action"]
    programs = {role: fit_program(role, actor, *_split_rows(actor, index))
                for index, role in enumerate(("robot_1", "robot_2"))}
    trace_dir = tmp_path / "traces"
    receipts = audit_programs(actor, programs, (108,), trace_dir=trace_dir)
    for role in ("robot_1", "robot_2"):
        receipt = receipts[role]
        raw = (trace_dir / receipt["trace_sha256"] / f"{role}.jsonl").read_bytes()
        assert sha256(raw).hexdigest() == receipt["trace_sha256"]
        assert receipt["trace_row_count"] == receipt["sample_count"]
        row = json.loads(raw.splitlines()[0])
        assert row["role"] == role
        assert row["nn_sampled_action"] == row["sampled_action"]
        assert row["pre_state_public_sha256"] == canonical_sha256(row[
            "pre_state_public"])
        assert row["post_state_public_sha256"] == canonical_sha256(row[
            "post_state_public"])
    assert audit_programs(actor, programs, (108,), trace_dir=trace_dir) == receipts
    (trace_dir / "robot_1.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError, match="immutable audit trace"):
        audit_programs(actor, programs, (108,), trace_dir=trace_dir)


def test_zero_split_program_serializes_as_one_leaf():
    torch.manual_seed(48)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    program = fit_program("robot_1", actor, *_split_rows(actor))
    single = replace(program, root_ref=-1,
                     split_weights=np.zeros((0, FEATURE_DIM), dtype=np.float32),
                     split_bias=np.zeros(0, dtype=np.float32),
                     left_refs=np.zeros(0, dtype=np.int64),
                     right_refs=np.zeros(0, dtype=np.int64),
                     leaf_logits=program.leaf_logits[:1].copy(),
                     feedback_leaves=frozenset(), explanation_leaves=frozenset(),
                     report={**program.report, "leaf_count": 1, "max_depth": 0})
    restored = WarehouseV6Program.from_dict(
        single.to_dict(), actor_role_sha256=actor.role_sha256["robot_1"])
    assert restored.split_weights.shape == (0, FEATURE_DIM)
    assert restored.report["max_depth"] == 0


def test_each_ppo_epoch_rechecks_current_actor_before_program_gradient():
    torch.manual_seed(49)
    model = WarehouseV6Actor()
    wait = ACTIONS.index("WAIT")
    with torch.no_grad():
        model.actors["robot_1"].tree.leaf_logits.zero_()
        model.actors["robot_1"].tree.leaf_logits[:, wait] = 30.0
    original = FrozenWarehouseV6Actor.from_model(model)
    env = _state(700)
    obs, mask = encode_public_observation(env.state), action_masks(env.state)
    full_logits = original.logits(obs, mask)
    report = {"validation": {"feedback_passed": True,
                             "feedback_online_passed": True}}
    program = WarehouseV6Program(
        role="robot_1", actor_role_sha256=original.role_sha256["robot_1"],
        root_ref=-1,
        split_weights=np.zeros((0, FEATURE_DIM), dtype=np.float32),
        split_bias=np.zeros(0, dtype=np.float32),
        left_refs=np.zeros(0, dtype=np.int64),
        right_refs=np.zeros(0, dtype=np.int64),
        leaf_logits=full_logits[:1].copy(),
        feedback_leaves=frozenset({0}), explanation_leaves=frozenset(),
        report=report)
    fixed = [{"obs": obs[0].copy(), "mask": mask[0].copy(),
              "episode": str(700 + i), "base_seed": 700 + i,
              "episode_id": int(env.state.episode_id), "frame": int(env.state.frame)}
             for i in range(12)]
    current = _resample_program_rows(original, "robot_1", fixed)
    assert all(row["sampled_action"] == "WAIT" for row in current)
    assert current_actor_fidelity(program, original, current)["passed"]
    episode, _ = collect_episode(model, seed=700, generator=torch.Generator().manual_seed(1),
                                 max_steps=1)
    other = next(index for index, action in enumerate(ACTIONS)
                 if action != "WAIT" and mask[0, index] > 0.5)
    with torch.no_grad():
        model.actors["robot_1"].tree.leaf_logits[:, other] = 100.0
    changed = FrozenWarehouseV6Actor.from_model(model)
    changed_rows = _resample_program_rows(changed, "robot_1", fixed)
    assert not current_actor_fidelity(program, changed, changed_rows)["passed"]
    update = ppo_update(
        model, torch.optim.Adam(model.parameters(), lr=1e-5), [episode],
        {"robot_1": program, "robot_2": None}, progress=0.9,
        cost_multiplier=0.0, epochs=1,
        program_validation_rows={"robot_1": fixed})
    assert update["feedback_disabled_epochs"]["robot_1"] == 1
    assert update["feedback_rows"]["robot_1"] == 0


def test_program_feedback_has_a_real_actor_gradient():
    torch.manual_seed(43)
    model = WarehouseV6Actor()
    actor = FrozenWarehouseV6Actor.from_model(model)
    program = fit_program("robot_1", actor, *_split_rows(actor))
    env = _state(100)
    obs, masks = encode_public_observation(env.state), action_masks(env.state)
    role_obs = torch.from_numpy(obs[:1])
    role_mask = torch.from_numpy(masks[:1])
    logits = model.role_logits("robot_1", role_obs, role_mask)
    action = int(logits.argmax(-1)[0])
    leaf, _ = program._leaf({"obs": obs[0], "mask": masks[0]})
    changed = program.leaf_logits.copy()
    changed[leaf, action] += 0.2
    report = dict(program.report)
    report["validation"] = {**report["validation"],
                            "feedback_online_passed": True,
                            "feedback_passed": True}
    program = replace(program, leaf_logits=changed,
                      feedback_leaves=frozenset({leaf}), report=report)
    penalty, count, _ = _program_feedback(
        logits, role_obs, role_mask, torch.tensor([action]),
        program, coefficient=0.01)
    assert count == 1
    gradients = torch.autograd.grad(penalty, (parameter for parameter in
                                    model.actors["robot_1"].parameters()
                                    if parameter.requires_grad),
                                    allow_unused=True)
    assert sum(float(grad.square().sum()) for grad in gradients if grad is not None) > 0


def test_paired_smoke_keeps_bc_start_and_zero_raw_feedback(tmp_path):
    spec = {
        "candidate_id": "smoke", "pair_seed": SMOKE_PAIR_SEED, "joint_steps": 120,
        "output_dir": str(tmp_path), "scenario_namespace": "smoke-v6",
        "checkpoint_seeds": list(SMOKE_EVALUATION_SEEDS), "bc_episodes": 2,
        "bc_epochs": 1, "dagger_rounds": 0, "smoke": True,
    }
    report = train_pair(spec)
    raw, regularized = (report["variants"][variant]
                        for variant in ("raw", "regularized"))
    assert raw["initial_role_sha256"] == regularized["initial_role_sha256"]
    assert raw["scenario_schedule_sha256"] == regularized["scenario_schedule_sha256"]
    assert raw["actor_decisions"] == regularized["actor_decisions"] == 240
    assert raw["feedback_rows"] == {"robot_1": 0, "robot_2": 0}
    assert raw["feedback_gradient_norm"] == {"robot_1": 0.0, "robot_2": 0.0}
    assert all(value > 0 for value in raw["role_weight_change_norm"].values())
    closure = source_closure_sha256()
    assert report["source_closure_sha256"] == closure
    assert raw["source_closure_sha256"] == regularized["source_closure_sha256"] == closure
    checkpoint = (tmp_path / "smoke" /
                  f"pair_{SMOKE_PAIR_SEED}_steps_120" /
                  "raw.checkpoint.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["source_closure_sha256"] = "0" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        train_pair(spec)
