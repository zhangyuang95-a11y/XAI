"""A genuine locked BC failure closes development before any PPO budget."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor_training import (
    WarehouseV6Actor, export_actor,
)
from experiments.warehouse import admission as a
from experiments.warehouse import protocol as p
from tests.test_warehouse_v6_admission import FakeBackend


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _failed_locked_bc_child(argv, *, quality_failure=False, **kwargs) -> SimpleNamespace:
    """Write the same pre-PPO artifacts as the locked runner failure path."""
    run_spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_bytes())
    authorization_path = Path(argv[argv.index("--authorization") + 1])
    authorization = json.loads(authorization_path.read_bytes())
    pair_dir = Path(run_spec["final_bc_actor_path"]).parent
    pair_dir.mkdir()
    actor_receipt = export_actor(
        WarehouseV6Actor(), run_spec["final_bc_actor_path"],
        pair_seed=run_spec["pair_seed"], variant="raw",
        candidate_id=f'{run_spec["candidate_id"]}-final-bc')
    schedule = p.bc_data_seed_schedule(run_spec["pair_seed"])
    final_schedule = {
        name.removeprefix("final_bc_precheck_"): seeds
        for name, seeds in schedule.items()
        if name.startswith("final_bc_precheck_")
    }
    roles = {}
    for index, role in enumerate(p.ROLES):
        passing = quality_failure or index == 1
        roles[role] = {
            "actor_role_sha256": actor_receipt["role_sha256"][role],
            "refit_report": {"method": "fit_episode_separated_core", "validation_passed": passing},
            "actor_core_qualification": {
                "method": p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD,
                "validation_passed": passing,
            },
            "installed_core_is_newly_qualified": passing,
            "installed_core_validation_mass": .40 if passing else .20,
            "installed_core_expected_actionable_mass": .35 if passing else .10,
        }
    precheck = {
        "purpose": "development_internal_precheck_after_complete_bc",
        "passed": False,
        "final_bc_actor": actor_receipt,
        "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
            "action_physics_config_sha256"],
        "final_bc_seed_schedule_sha256": a._sha(final_schedule),
        "final_bc_stage_counts": runner.BC_FINAL_CORE_STAGE_COUNTS,
        "final_bc_visits": {
            name: {"scenario_seeds": seeds, "rows": 10, "physical_frames": 10, "excluded_frames": 0}
            for name, seeds in final_schedule.items()},
        "roles": roles,
    }
    if quality_failure:
        episodes = [{"seed": seed, "deliveries": 0, "collisions": 0,
                     "shutdown_episode": False,
                     "quality_trace": {
                         "schema": "warehouse-quality-trace.v1", "scenario_seed": seed,
                         "action_seed": seed, "episode_id": seed, "frame_count": 1,
                         "action_encoding": "base64-u8-a1-times-5-plus-a2",
                         "actions_b64": "GA==", "actor_sha256": actor_receipt["sha256"],
                         "actor_role_sha256": actor_receipt["role_sha256"],
                         "initial_pre_state_sha256": "1"*64,
                         "final_post_state_sha256": "2"*64, "chain_sha256": "3"*64}}
                    for seed in run_spec["checkpoint_seeds"]]
        precheck["quality_screen_seeds"] = run_spec["checkpoint_seeds"]
        precheck["quality_screen"] = {"episodes": episodes, "summary": {
            "mean_deliveries": 0.0, "mean_collisions": 0.0, "shutdown_episode_rate": 0.0}}
    _write_json(Path(run_spec["final_bc_precheck_receipt_path"]), precheck)

    # The runner's registered final program and selection scenes are kept
    # outside every BC purpose; the reservation is derived, not caller set.
    checked = runner._checked_spec(run_spec)
    reserved = set(checked["reserved_external_scenario_seeds"])
    assert reserved == ({seed for block in run_spec[
        "final_program_fit_seed_schedule"].values() for seed in block}
        | set(run_spec["checkpoint_seeds"]))
    assert not reserved & {seed for block in schedule.values() for seed in block}

    expected = runner._formal_bc_reset_sequence(checked, include_quality_screen=quality_failure)
    physics = runner.ACTION_PHYSICS_CONFIG
    actual_rows = [
        {**row, "episode_id": index,
         "map_layout_id": physics["map_layout_id"],
         "move_battery_cost": float(physics["move_battery_cost"]),
         "charge_per_wait": float(physics["charge_per_wait"])}
        for index, row in enumerate(expected)
    ]
    journal_path = Path(authorization["reset_audit_journal_path"])
    journal_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"),
                   allow_nan=False) + "\n" for row in actual_rows))
    audit = {
        "status": "failed",
        "error": "RuntimeError: Warehouse final BC four-purpose internal precheck failed",
        "candidate_id": run_spec["candidate_id"],
        "pair_seed": run_spec["pair_seed"],
        "joint_steps": run_spec["joint_steps"],
        "authorization_receipt_sha256": _file_sha(authorization_path),
        "run_spec_sha256": authorization["run_spec_sha256"],
        "training_config_sha256": authorization["training_config_sha256"],
        "source_closure_sha256": authorization["source_closure_sha256"],
        "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
        "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
            "action_physics_config_sha256"],
        "pair_report_sha256": None,
        "selected_actor_role_sha256": None,
        "actual_reset_rows": actual_rows,
        "actual_reset_count": len(actual_rows),
        "actual_reset_sequence_sha256": a._sha(expected),
        "declared_prefix_matches": True,
        "reset_journal_path": str(journal_path),
        "reset_journal_sha256": _file_sha(journal_path),
    }
    _write_json(Path(authorization["reset_audit_path"]), audit)
    kwargs["stdout"].write(
        b"RuntimeError: Warehouse final BC four-purpose internal precheck failed\n")
    return SimpleNamespace(returncode=1)


def test_failed_development_bc_writes_immutable_closure_and_skips_ppo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a.subprocess, "run", _failed_locked_bc_child)
    admission = a.Admission(
        tmp_path, p.REGISTERED_CANDIDATE_ID,
        backend=a.NativeBackend(), claim_root=tmp_path / "claims",
        program_fit_method=p.REGISTERED_PROGRAM_FIT_METHOD)
    failed = admission.develop()
    assert failed["failure_kind"] == "final_bc_internal_precheck"
    assert failed["failed_pair_seed"] == p.DEVELOPMENT_PAIR_SEEDS[0]
    assert failed["failed_budget"] == p.BUDGETS[0]
    assert failed["remaining_budgets_skipped"] == list(p.BUDGETS[1:])
    assert failed["passed"] is False
    row_path = admission._path(
        f'development.{p.BUDGETS[0]}.{p.DEVELOPMENT_PAIR_SEEDS[0]}.json')
    row = json.loads(row_path.read_bytes())
    evidence = row["bc_precheck_failure"]
    assert row["training"] == {} and row["evaluation"] == {}
    assert row["stress_episodes"] == 0
    assert failed["failed_pair_receipt_sha256"] == _file_sha(row_path)
    assert failed["bc_precheck_receipt_sha256"] == evidence[
        "bc_precheck_receipt_sha256"]
    for name in ("authorization_receipt", "bc_precheck_receipt",
                 "final_bc_actor", "reset_audit", "reset_journal",
                 "training_log"):
        assert evidence[f"{name}_sha256"] == _file_sha(
            Path(evidence[f"{name}_path"]))
    closed = json.loads(admission._path("candidate.closed.json").read_bytes())
    assert closed["failed_pair_receipt_sha256"] == _file_sha(row_path)
    assert closed["source_closure_sha256"] == evidence[
        "source_closure_sha256"]
    assert not admission._path("budget.lock.json").exists()
    pair_dir = admission.directory / f'pair_{p.DEVELOPMENT_PAIR_SEEDS[0]}_steps_{p.BUDGETS[0]}'
    assert not (pair_dir / "paired_bc.checkpoint.pt").exists()
    assert not list(pair_dir.glob("raw.*"))
    assert not list(pair_dir.glob("regularized.*"))
    assert not (pair_dir / "pair.training.json").exists()
    assert len(list(admission.directory.glob("pair_*_steps_*"))) == 1
    with pytest.raises(RuntimeError, match="candidate is closed"):
        admission.develop()
    with pytest.raises(RuntimeError, match="candidate is closed"):
        admission.formal_train()


def test_unverified_subprocess_failure_remains_loud_and_does_not_close(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crashed_child(_argv, **kwargs):
        kwargs["stdout"].write(b"unrelated subprocess crash\n")
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(a.subprocess, "run", crashed_child)
    admission = a.Admission(
        tmp_path, p.REGISTERED_CANDIDATE_ID,
        backend=a.NativeBackend(), claim_root=tmp_path / "claims",
        program_fit_method=p.REGISTERED_PROGRAM_FIT_METHOD)
    with pytest.raises(RuntimeError, match=r"locked pair failed \(17\)"):
        admission.develop()
    assert not admission._path("development.failed.json").exists()
    assert not admission._path("candidate.closed.json").exists()
    assert admission._path(
        f'development.{p.BUDGETS[0]}.{p.DEVELOPMENT_PAIR_SEEDS[0]}.failure.json'
    ).is_file()


def test_interrupted_claimed_formal_pair_closes_candidate_with_immutable_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p, "stress_schedule", lambda: ((1, 2_600_001, 123),))

    class InterruptedAfterClaim(FakeBackend):
        def train_pair(self, spec):
            if spec["pair_seed"] in p.FORMAL_PAIR_SEEDS:
                pair_dir = (Path(spec["output_dir"]) / spec["candidate_id"] /
                            f'pair_{spec["pair_seed"]}_steps_{spec["joint_steps"]}')
                pair_dir.mkdir()
                (pair_dir / "partial.bc").write_text("interrupted BC")
                raise KeyboardInterrupt("synthetic formal interruption")
            return super().train_pair(spec)

    admission = a.Admission(
        tmp_path, p.REGISTERED_CANDIDATE_ID,
        backend=InterruptedAfterClaim(tmp_path),
        claim_root=tmp_path / "claims",
        program_fit_method=p.REGISTERED_PROGRAM_FIT_METHOD)
    assert admission.develop()["locked_budget"] == p.BUDGETS[0]
    with pytest.raises(KeyboardInterrupt, match="synthetic formal interruption"):
        admission.formal_train()
    phase = f"formal.training.{p.FORMAL_PAIR_SEEDS[0]}"
    failure_path = admission._path(f"{phase}.failure.json")
    closed = json.loads(admission._path("candidate.closed.json").read_bytes())
    assert json.loads(failure_path.read_bytes())["error_type"] == "KeyboardInterrupt"
    assert closed["phase"] == phase
    assert closed["failure_receipt_sha256"] == _file_sha(failure_path)
    assert closed["budget_lock_sha256"] == _file_sha(
        admission._path("budget.lock.json"))
    assert closed["source_closure_sha256"] == admission.backend.source
    assert closed["claimed_pair_output_exists"] is True
    assert not admission._path("formal.selection.json").exists()
    assert not (admission.directory /
                f'pair_{p.FORMAL_PAIR_SEEDS[1]}_steps_{p.BUDGETS[0]}').exists()
    with pytest.raises(RuntimeError, match="candidate is closed"):
        admission.formal_train()


def test_passing_program_but_failed_bc_quality_never_starts_ppo(tmp_path, monkeypatch):
    def child(argv, **kwargs):
        return _failed_locked_bc_child(argv, quality_failure=True, **kwargs)
    monkeypatch.setattr(a.subprocess, "run", child)
    admission = a.Admission(tmp_path, p.REGISTERED_CANDIDATE_ID,
        backend=a.NativeBackend(), claim_root=tmp_path / "claims",
        program_fit_method=p.REGISTERED_PROGRAM_FIT_METHOD)
    failed = admission.develop()
    assert failed["passed"] is False
    row = a._read(admission._path(f"development.{p.BUDGETS[0]}.{p.DEVELOPMENT_PAIR_SEEDS[0]}.json"))
    assert row["bc_precheck_failure"]["failed_roles"] == list(p.ROLES)
    assert row["bc_precheck_failure"]["quality_screen_failed"] is True
    assert row["training"] == {} and row["evaluation"] == {}
    assert not admission._path("budget.lock.json").exists()
