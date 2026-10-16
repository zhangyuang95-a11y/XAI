"""Exploratory pressure and program evidence cannot be rebound after sealing."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor)
from backend.training.warehouse_v6_runner import (
    _SOURCE_FILES, _write_final_fit_trace, collect_program_audit_rows)
from env.warehouse.decision_protocol import canonical_sha256
from experiments.warehouse.stage2_evidence import (
    replay_program_split_trace, verify_pressure_receipt_identity)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


@pytest.fixture
def program_trace(tmp_path):
    torch.manual_seed(87712)
    artifact = export_actor(WarehouseV6Actor(), tmp_path / "actor.npz",
                            pair_seed=87712, variant="regularized",
                            candidate_id="stage2-evidence-test")
    actor = FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    seed = 700019
    rows = collect_program_audit_rows(actor, [seed])
    trace = _write_final_fit_trace(
        rows, split="fit", seeds=[seed], trace_root=tmp_path / "traces",
        actor_sha256=actor.artifact_sha256)
    return actor, Path(trace["path"]), seed, trace["sha256"]


def test_program_trace_physically_replays_both_actors_and_public_inputs(program_trace):
    actor, path, seed, digest = program_trace
    receipt = replay_program_split_trace(
        actor, path, split="fit", seeds=[seed], expected_sha256=digest)
    assert receipt["independent_physical_replay_equal"] is True
    assert receipt["episode_count"] == 1
    assert receipt["frame_count"] > 0
    assert "experiments/warehouse/stage2_evidence.py" in _SOURCE_FILES


@pytest.mark.parametrize("mutate, expected", [
    (lambda row: row["roles"]["robot_2"].update(
        sampled_action="WAIT" if row["roles"]["robot_2"]["sampled_action"] != "WAIT"
        else "UP"), "NN action or input differs"),
    (lambda row: row["roles"]["robot_1"]["obs"].__setitem__(
        0, row["roles"]["robot_1"]["obs"][0] + 0.25),
        "NN action or input differs"),
    (lambda row: row["roles"]["robot_1"]["mask"].__setitem__(
        0, 1.0 - row["roles"]["robot_1"]["mask"][0]),
        "NN action or input differs"),
    (lambda row: row["pre_state_public"]["agents"][0].update(battery=0.0),
        "public pre-state differs"),
])
def test_program_trace_rejects_forged_frame_even_with_new_file_hash(
        program_trace, tmp_path, mutate, expected):
    actor, path, seed, _ = program_trace
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[0])
    forged = tmp_path / "fit.jsonl"
    forged.write_text("".join(json.dumps(row, sort_keys=True,
                                       separators=(",", ":")) + "\n"
                              for row in rows))
    with pytest.raises(ValueError, match=expected):
        replay_program_split_trace(
            actor, forged, split="fit", seeds=[seed],
            expected_sha256=_sha(forged))


def test_program_trace_rejects_changed_bytes_before_replay(program_trace):
    actor, path, seed, digest = program_trace
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="trace bytes differ"):
        replay_program_split_trace(
            actor, path, split="fit", seeds=[seed], expected_sha256=digest)


def test_program_trace_rejects_rehashed_forged_physical_transition(
        program_trace, tmp_path):
    actor, path, seed, _ = program_trace
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    forged_state = rows[0]["post_state_public"]
    forged_state["agents"][0]["battery"] += 1.0
    rows[0]["post_state_public_sha256"] = canonical_sha256(forged_state)
    forged = tmp_path / "fit.jsonl"
    forged.write_text("".join(json.dumps(row, sort_keys=True,
                                       separators=(",", ":")) + "\n"
                              for row in rows))
    with pytest.raises(ValueError, match="public post-state differs"):
        replay_program_split_trace(
            actor, forged, split="fit", seeds=[seed],
            expected_sha256=_sha(forged))


@pytest.fixture
def pressure_bundle(tmp_path):
    schedule = [{"task": 1, "scenario_seed": 12002001,
                 "action_seed": 12002001}]
    expected = {"actor": "a" * 64,
                "roles": {"robot_1": "b" * 64, "robot_2": "c" * 64},
                "source": "d" * 64, "schedule": _digest(schedule)}
    files = {name: tmp_path / name for name in
             ("pressure12069.py", "stage2_common.py", "prepare_stage2.py",
              "verify_quality12069.py", "stage2-seed-plan.json")}
    for name, path in files.items():
        path.write_text(name)
    config = {
        "schema": "warehouse-v611-pressure-predeclared.v1",
        "status": "sealed_unregistered_not_run",
        "schedule": schedule, "schedule_sha256": expected["schedule"],
        "stage2_seed_plan_sha256": _sha(files["stage2-seed-plan.json"]),
        "stage2_common_sha256": _sha(files["stage2_common.py"]),
        "stage2_prepare_script_sha256": _sha(files["prepare_stage2.py"]),
        "quality_integrity_script_sha256": _sha(files["verify_quality12069.py"]),
        "actor_sha256": expected["actor"],
        "actor_role_sha256": expected["roles"],
        "source_closure_sha256": expected["source"],
        "required_cwd": "/tmp/warehouse-stage2-neutral",
    }
    pre = tmp_path / "pressure-predeclared.json"
    _write(pre, config)
    lock = {
        "schema": "warehouse-v611-pressure-script-lock.v1",
        "stage": "pressure",
        "script_sha256": _sha(files["pressure12069.py"]),
        "common_sha256": _sha(files["stage2_common.py"]),
        "prepare_script_sha256": _sha(files["prepare_stage2.py"]),
        "quality_integrity_script_sha256": _sha(files["verify_quality12069.py"]),
        "predeclared_sha256": _sha(pre),
        "seed_plan_sha256": _sha(files["stage2-seed-plan.json"]),
        "schedule_sha256": expected["schedule"],
        "source_closure_sha256": expected["source"],
        "actor_sha256": expected["actor"],
        "actor_role_sha256": expected["roles"],
        "python_command": f"python3 -I {files['pressure12069.py'].resolve()}",
        "required_cwd": config["required_cwd"],
    }
    script_lock = tmp_path / "pressure-script-lock.json"
    _write(script_lock, lock)
    raw, replay = (tmp_path / name for name in
                   ("pressure-episodes.jsonl", "pressure-replay.jsonl"))
    raw.write_text("{}\n")
    replay.write_text("{}\n")
    result = {
        "schema": "warehouse-v611-pressure-result.v1",
        "status": "unregistered_frozen_bc_pressure_complete",
        "predeclared_sha256": _sha(pre),
        "script_lock_sha256": _sha(script_lock),
        "script_sha256": _sha(files["pressure12069.py"]),
        "actor_sha256": expected["actor"],
        "actor_role_sha256": expected["roles"],
        "source_closure_sha256": expected["source"],
        "pressure_schedule_sha256": expected["schedule"],
        "raw_jsonl_sha256": _sha(raw),
        "replay_jsonl_sha256": _sha(replay),
        "pressure_passed": True, "pressure_gates": {"quality": True},
        "registered_or_sealed_seeds_used": False,
        "program_or_ppo_used": False,
    }
    _write(tmp_path / "pressure-result.json", result)
    kwargs = {
        "runner_script": files["pressure12069.py"],
        "common_script": files["stage2_common.py"],
        "prepare_script": files["prepare_stage2.py"],
        "quality_script": files["verify_quality12069.py"],
        "seed_plan_path": files["stage2-seed-plan.json"],
        "expected_actor_sha256": expected["actor"],
        "expected_role_sha256": expected["roles"],
        "expected_source_closure_sha256": expected["source"],
        "expected_schedule_sha256": expected["schedule"],
    }
    return tmp_path, kwargs


def test_pressure_receipt_rebinds_predeclared_lock_and_current_script_bytes(
        pressure_bundle):
    root, kwargs = pressure_bundle
    receipt = verify_pressure_receipt_identity(root, **kwargs)
    assert receipt["pressure_runner_script_sha256"] == _sha(kwargs["runner_script"])
    assert receipt["pressure_result_sha256"] == _sha(root / "pressure-result.json")


@pytest.mark.parametrize("target", [
    "pressure12069.py", "stage2_common.py", "prepare_stage2.py",
    "verify_quality12069.py", "stage2-seed-plan.json",
    "pressure-predeclared.json", "pressure-script-lock.json",
    "pressure-episodes.jsonl", "pressure-replay.jsonl",
])
def test_pressure_receipt_rejects_any_changed_bound_input(pressure_bundle, target):
    root, kwargs = pressure_bundle
    path = root / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding differs"):
        verify_pressure_receipt_identity(root, **kwargs)


def test_pressure_receipt_rejects_self_consistent_but_unbound_result(
        pressure_bundle):
    root, kwargs = pressure_bundle
    result_path = root / "pressure-result.json"
    result = json.loads(result_path.read_text())
    result["script_sha256"] = "e" * 64
    _write(result_path, result)
    with pytest.raises(ValueError, match="binding differs"):
        verify_pressure_receipt_identity(root, **kwargs)
