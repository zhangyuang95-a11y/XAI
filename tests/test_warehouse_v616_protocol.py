"""No-rollout checks for the registered joint-tree training contract."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.warehouse import protocol as p
from experiments.warehouse.admission import NativeBackend
from tests.warehouse_v6_contract_fixture import bc_receipt, topology_freeze_evidence


def _history(steps: int) -> list[dict]:
    return [{"joint_step": step,
             "role_tree_sha256": {"robot_1": str(index + 1) * 64,
                                  "robot_2": str(index + 4) * 64},
             "route_parity_passed": True,
             "path_legality_passed": True,
             "core_qualification_recomputed": True}
            for index, step in enumerate(p.topology_refresh_steps(steps))]


@pytest.mark.parametrize("budget, expected", [
    (200_000, (48_000, 96_000, 144_000)),
    (300_000, (72_000, 144_000, 216_000)),
    (400_000, (96_000, 192_000, 288_000)),
])
def test_topology_refresh_is_block_aligned_and_frozen_before_extraction(
        budget: int, expected: tuple[int, ...]) -> None:
    assert p.topology_refresh_steps(budget) == expected
    assert all(step % p.SCENARIO_BLOCK_STEPS == 0 for step in expected)
    assert expected[-1] < p.topology_frozen_at_step(budget)
    assert p.topology_frozen_at_step(budget) == \
        p.online_program_extraction_steps(budget)[0]


def test_run_spec_seals_joint_tree_schedule_and_method(tmp_path) -> None:
    args = (p.REGISTERED_CANDIDATE_ID, p.DEVELOPMENT_PAIR_SEEDS[0],
            p.BUDGETS[0], str(tmp_path), p.REGISTERED_PROGRAM_FIT_METHOD)
    spec = p.PairRunSpec(*args, checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    mapping = spec.to_mapping()
    assert mapping["topology_refresh_steps"] == \
        p.topology_refresh_steps(p.BUDGETS[0])
    assert mapping["topology_frozen_at_step"] == p.topology_frozen_at_step(
        p.BUDGETS[0])
    moved = p.PairRunSpec(args[0], args[1], args[2],
                          str(tmp_path / "other"), args[4],
                          checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    assert spec.sha256() == moved.sha256()
    assert spec.training_config_sha256("a" * 64) == \
        moved.training_config_sha256("a" * 64)
    with pytest.raises(ValueError, match="topology refresh schedule"):
        p.PairRunSpec(*args, checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS,
                      topology_refresh_steps=(48_120, 96_000, 144_000))
    with pytest.raises(ValueError, match="topology freeze step"):
        p.PairRunSpec(*args, checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS,
                      topology_frozen_at_step=149_880)
    with pytest.raises(ValueError, match="program"):
        p.PairRunSpec(*args[:-1], "actor_sparse_topology_v1",
                      checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)


def test_formal_bc_reset_order_matches_topology_before_risk_at_index_180() -> None:
    from backend.training import warehouse_v6_runner as runner

    seed = p.DEVELOPMENT_PAIR_SEEDS[0]
    schedule = p.bc_data_seed_schedule(seed)
    spec = {
        "bc_data_seed_schedule": schedule,
        "bc_critic_reset_schedule": p.bc_critic_reset_schedule(
            seed, p.INITIAL_RELEASE_ID),
    }
    expected = runner._formal_bc_reset_sequence(spec)
    # The old mapping iteration put risk first at 180. The real BC visits
    # both topology stages, then performs critic resets, before risk data.
    assert expected[180] == {"phase": "formal_bc",
                             "seed": schedule["topology_fit_sampled"][0]}
    assert expected[205] == {"phase": "formal_bc",
                             "seed": schedule["topology_fit_argmax"][0]}
    assert expected[230] == {"phase": "formal_bc",
                             "seed": spec["bc_critic_reset_schedule"]["seed"]}
    assert expected[232] == {"phase": "formal_bc",
                             "seed": schedule["risk_public_teacher"][0]}
    assert len(expected) == 1_396
    reordered = {**spec, "bc_data_seed_schedule": dict(sorted(schedule.items()))}
    assert runner._formal_bc_reset_sequence(reordered) == expected
    wrong_prefix = expected[:180] + [{"phase": "formal_bc",
                                      "seed": schedule["risk_public_teacher"][0]}]
    assert runner._formal_partial_reset_prefix(spec, expected[:233])
    assert not runner._formal_partial_reset_prefix(spec, wrong_prefix)
    incomplete = {**spec, "bc_data_seed_schedule": {
        name: seeds for name, seeds in schedule.items()
        if name != "topology_fit_sampled"}}
    with pytest.raises(RuntimeError, match="purpose schedule changed"):
        runner._formal_bc_reset_sequence(incomplete)


def test_refresh_history_requires_both_role_certificates_at_every_step() -> None:
    valid = _history(200_000)
    assert p.topology_refresh_history_valid(valid, joint_steps=200_000)
    for field, bad_value in (("joint_step", 48_120),
                             ("route_parity_passed", False),
                             ("path_legality_passed", False),
                             ("core_qualification_recomputed", False)):
        tampered = deepcopy(valid)
        tampered[0][field] = bad_value
        assert not p.topology_refresh_history_valid(
            tampered, joint_steps=200_000)
    tampered = deepcopy(valid)
    tampered[1]["role_tree_sha256"]["robot_2"] = "not-a-sha"
    assert not p.topology_refresh_history_valid(tampered, joint_steps=200_000)
    assert not p.topology_refresh_history_valid(valid[:2], joint_steps=200_000)


def test_frozen_topology_requires_unchanged_role_hashes_and_zero_updates() -> None:
    valid = topology_freeze_evidence(200_000)
    assert p.frozen_topology_valid(valid)
    robot_core = valid["topology_freeze_qualification"]["roles"]["robot_1"]
    assert robot_core["qualification"]["core_mask_sha256"] != \
        robot_core["installed_core"]["core_mask_sha256"]
    for key, value in (("topology_updates_after_freeze", 1),
                       ("topology_updates_after_freeze", False),
                       ("final_topology_role_sha256",
                        {"robot_1": "a" * 64, "robot_2": "c" * 64}),
                       ("final_core_leaf_logits_sha256",
                        {"robot_1": "e" * 64, "robot_2": "c" * 64}),
                       ("frozen_topology_role_sha256",
                        {"robot_1": "bad", "robot_2": "b" * 64})):
        tampered = deepcopy(valid)
        tampered[key] = value
        assert not p.frozen_topology_valid(tampered)
    for path, value in (
            (("topology_freeze_qualification", "route_parity_passed"), False),
            (("topology_freeze_qualification", "path_legality_passed"), False),
            (("topology_freeze_qualification", "roles", "robot_2",
              "qualification_bound_to_installed_actor"), False),
            (("topology_freeze_qualification", "roles", "robot_1",
              "qualification", "frozen_role_sha256"), "0" * 64),
            (("topology_freeze_qualification", "roles", "robot_1",
              "qualification", "core_mask_sha256"), "0" * 64),
            (("topology_freeze_qualification", "roles", "robot_1",
              "installed_core", "core_mask_sha256"), "0" * 64),
            (("topology_freeze_qualification", "roles", "robot_1",
              "qualification", "leaf_path_proof", "0", "proven_actions"), [])):
        tampered = deepcopy(valid)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not p.frozen_topology_valid(tampered)


def test_native_backend_creates_sha_bound_authorization_without_direct_training(
        tmp_path, monkeypatch) -> None:
    from backend.training import warehouse_v6_runner as runner

    spec = p.PairRunSpec(p.REGISTERED_CANDIDATE_ID,
                         p.DEVELOPMENT_PAIR_SEEDS[0], p.BUDGETS[0],
                         str(tmp_path), p.REGISTERED_PROGRAM_FIT_METHOD,
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)

    def forbidden(_spec):
        raise AssertionError("ordinary train_pair must remain closed")

    def fake_run(argv, **kwargs):
        assert Path(argv[1]).name == "run_warehouse_v617_locked_pair.py"
        spec_path = Path(argv[argv.index("--spec") + 1])
        authorization_path = Path(argv[argv.index("--authorization") + 1])
        run_spec = json.loads(spec_path.read_bytes())
        authorization = json.loads(authorization_path.read_bytes())
        assert authorization["run_spec_file_sha256"] == sha256(
            spec_path.read_bytes()).hexdigest()
        assert authorization["source_closure_sha256"] == \
            runner.source_closure_sha256()
        assert authorization["training_config_sha256"] == \
            runner._training_config_sha256(run_spec)
        pair_path = (tmp_path / spec.candidate_id /
                     f"pair_{spec.pair_seed}_steps_{spec.joint_steps}" /
                     "pair.training.json")
        pair_path.parent.mkdir(parents=True)
        pair_path.write_text(json.dumps({"candidate_id": spec.candidate_id,
                                         "pair_seed": spec.pair_seed}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "train_pair", forbidden)
    monkeypatch.setattr("experiments.warehouse.admission.subprocess.run", fake_run)
    report = NativeBackend().train_pair(spec.to_mapping())
    assert report == {"candidate_id": spec.candidate_id,
                      "pair_seed": spec.pair_seed}
    with pytest.raises(RuntimeError, match="already has an output directory"):
        NativeBackend().train_pair(spec.to_mapping())


def test_native_backend_refuses_formal_pair_without_verified_budget_lock(
        tmp_path) -> None:
    spec = p.PairRunSpec(p.REGISTERED_CANDIDATE_ID,
                         p.FORMAL_PAIR_SEEDS[0], p.BUDGETS[0],
                         str(tmp_path), p.REGISTERED_PROGRAM_FIT_METHOD,
                         checkpoint_seeds=p.FORMAL_SELECTION_SEEDS)
    with pytest.raises(RuntimeError, match="prerequisite receipt missing"):
        NativeBackend().train_pair(spec.to_mapping())


def test_plain_train_pair_is_closed_before_spec_validation_or_output(
        tmp_path, monkeypatch) -> None:
    from backend.training import warehouse_v6_runner as runner

    def forbidden(_spec):
        raise AssertionError("unlocked train_pair reached spec validation")

    monkeypatch.setattr(runner, "_checked_spec", forbidden)
    with pytest.raises(RuntimeError, match="review-only"):
        runner.train_pair({"output_dir": str(tmp_path)})
    assert not list(tmp_path.iterdir())


def test_failed_bc_precheck_never_starts_raw_or_regularized_ppo(
        tmp_path, monkeypatch) -> None:
    from backend.training import warehouse_v6_runner as runner
    from env.warehouse.reset_audit_v614 import capture_stage1_resets

    spec = p.PairRunSpec(p.REGISTERED_CANDIDATE_ID,
                         p.DEVELOPMENT_PAIR_SEEDS[0], p.BUDGETS[0],
                         str(tmp_path), p.REGISTERED_PROGRAM_FIT_METHOD,
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    pair_dir = (tmp_path / spec.candidate_id /
                f"pair_{spec.pair_seed}_steps_{spec.joint_steps}")
    run_spec = {**spec.to_mapping(),
                "warmup_actor_path": str(pair_dir / "warmup.actor.npz"),
                "final_bc_actor_path": str(pair_dir / "final-bc.actor.npz"),
                "final_bc_precheck_receipt_path": str(
                    pair_dir / "final-bc-internal-precheck.json")}
    reached = []

    def failed_precheck(_model, *, pair_seed, spec):
        reached.append((pair_seed, spec["joint_steps"]))
        raise RuntimeError("synthetic BC precheck failed")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("BC failure proceeded to PPO or environment")

    monkeypatch.setattr(runner, "_pretrain_pair_production", failed_precheck)
    monkeypatch.setattr(runner, "_train_variant", forbidden)
    monkeypatch.setattr(runner, "_environment", forbidden)
    token = runner._ACTIVE_LOCKED_EXECUTION.set(runner._LOCKED_FORMAL_MARKER)
    try:
        with capture_stage1_resets(tmp_path / "resets.jsonl") as resets:
            with pytest.raises(RuntimeError, match="synthetic BC precheck failed"):
                runner.train_pair(run_spec)
    finally:
        runner._ACTIVE_LOCKED_EXECUTION.reset(token)
    assert resets == []
    assert reached == [(spec.pair_seed, spec.joint_steps)]
    assert pair_dir.is_dir() and not list(pair_dir.iterdir())


def test_current_bc_contract_rejects_missing_core_evidence_and_legacy_is_read_only():
    seed, source = p.DEVELOPMENT_PAIR_SEEDS[0], "f" * 64
    current = bc_receipt(seed, source)
    assert p.bc_report_valid(current, seed, source)
    tampered = deepcopy(current)
    tampered["bc_topology_refresh_history"][2]["path_legality_passed"] = False
    assert not p.bc_report_valid(tampered, seed, source)
    tampered = deepcopy(current)
    tampered["final_bc_internal_precheck"]["roles"]["robot_1"][
        "installed_core_validation_mass"] = 0.34
    assert not p.bc_report_valid(tampered, seed, source)

    old = deepcopy(current)
    old["bc_recipe_version"] = p.LEGACY_V614_BC_RECIPE_VERSION
    old["bc_recipe_sha256"] = p.LEGACY_V614_BC_SPEC_SHA256
    old["executable_recipe_file_sha256"] = p.LEGACY_V614_RECIPE_FILE_SHA256
    old.pop("bc_topology_refresh_history")
    for role in p.ROLES:
        row = old["final_bc_internal_precheck"]["roles"][role]
        row.pop("actor_core_qualification")
        row.pop("installed_core_validation_mass")
        row.pop("installed_core_expected_actionable_mass")
    old["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        (json.dumps(old["final_bc_internal_precheck"], sort_keys=True,
                    allow_nan=False) + "\n").encode()).hexdigest()
    assert p.legacy_v614_bc_report_valid(old, seed, source)
    assert not p.bc_report_valid(old, seed, source)
