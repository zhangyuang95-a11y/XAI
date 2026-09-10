from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from backend.training import warehouse_r4_training_ledger as ledger
from env.warehouse_native.policy import NativeActorCritic


def h(value) -> str:
    return sha256(str(value).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def protocol(version: str, stage=None):
    value = {
        "version": version,
        "source_actor_sha256": ledger.R3_ACTOR_SHA256,
        "source_checkpoint_sha256": ledger.R3_CHECKPOINT_SHA256,
        "source_state_sha256": ledger.R3_STATE_SHA256,
        "source_cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS,
        "training": {"checkpoint_interval": 50_000, "environments": 16,
            "rollout_steps": 128, "epochs": 4, "gae_lambda": .95,
            "gamma": .99, "clip": .2},
        "partners": {"selfplay": .2, "skilled": .2, "assertive": .5, "noisy": .1},
        "shaping": {"productive_progress": .02, "avoidable_wait": -.03,
                    "task_distance_regression": -.02},
        "feedback": {"direction": "KL(nn||program)", "role_scope": "robot_2",
                     "runtime_action_override": False},
        "maximum_additional_joint_steps": 1_000_000,
    }
    if stage is not None:
        value["stage_source"] = stage
    return value


def export_actor(path: Path, version: str, step: int, protocol_value, suffix: str):
    model = NativeActorCritic(3, 4, hidden=4)
    parameter_sha = h("parameters-" + suffix)
    model.export_npz(path, {
        "experiment_version": version, "joint_steps": step,
        "cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS + step,
        "source_actor_sha256": ledger.R3_ACTOR_SHA256,
        "source_checkpoint_sha256": ledger.R3_CHECKPOINT_SHA256,
        "actor_parameters_sha256": parameter_sha,
        "protocol_sha256": ledger.digest(protocol_value),
        "action_controller": "neural_actor_only",
    })
    return parameter_sha, ledger.file_hash(path)


def training_rows(start, end):
    rows = []
    step = start
    while step < end:
        next_step = min(end, step + 2048)
        rows.append({"joint_steps": next_step,
                     "cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS + next_step,
                     "time_steps": (next_step - step + 15) // 16})
        step = next_step
    return rows


def make_attempt(root: Path, name: str, version: str, start: int, end: int,
                 scenario_sha: str, stage=None):
    directory = root / name
    directory.mkdir()
    proto = protocol(version, stage)
    receipt = {"version": version,
        "source_actor_sha256": ledger.R3_ACTOR_SHA256,
        "source_checkpoint_sha256": ledger.R3_CHECKPOINT_SHA256,
        "source_state_sha256": ledger.R3_STATE_SHA256,
        "source_cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS,
        "scenario_manifest_sha256": scenario_sha,
        "runtime_action_override": False}
    if stage is not None:
        receipt["stage_source"] = stage
    write_json(directory / "protocol.json", proto)
    write_json(directory / "source_receipt.json", receipt)
    write_json(directory / "run.json", {"version": version,
        "status": "training_target_completed", "pid": 1,
        "target_additional_joint_steps": end,
        "current_additional_joint_steps": end,
        "cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS + end,
        "started_unix": float(1000 + int(version.rsplit("v", 1)[-1]) * 100),
        "runtime_action_override": False, "elapsed_seconds": 20.})
    with (directory / "training.jsonl").open("w", encoding="utf-8") as stream:
        for row in training_rows(start, end):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    boundaries = []
    for step in range(start + 50_000, end + 1, 50_000):
        boundary = directory / "boundaries" / f"step_{step:07d}"
        checkpoint = directory / "checkpoints" / f"step_{step:07d}.pt"
        boundary.mkdir(parents=True)
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{name}-{step}".encode())
        actor = boundary / "actor.npz"
        parameters, actor_sha = export_actor(actor, version, step, proto, f"{name}-{step}")
        summary = {"version": version, "step": step,
            "cumulative_joint_steps": ledger.R3_CUMULATIVE_JOINT_STEPS + step,
            "actor": {"path": actor.relative_to(directory).as_posix(),
                      "sha256": actor_sha, "parameters_sha256": parameters},
            "checkpoint": {"path": checkpoint.relative_to(directory).as_posix(),
                           "sha256": ledger.file_hash(checkpoint)},
            "action_authority": {"trainable": step, "equal": step, "overrides": 0},
            "elapsed_seconds": float(step / 1000), "evaluation_pending": True,
            "candidate_selected": False}
        write_json(boundary / "summary.json", summary)
        boundaries.append(summary)
    return directory, boundaries


def fixture(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    monkeypatch.setattr(ledger, "ROOT", root)
    marker = root / "ledger.py"
    marker.write_text("frozen ledger source\n")
    monkeypatch.setattr(ledger, "_producer_sources",
        lambda: {"ledger.py": ledger.file_hash(marker)})
    source_model = NativeActorCritic(3, 4, hidden=4)
    r3_actor = root / "r3_actor.npz"
    source_model.export_npz(r3_actor, {})
    r3_checkpoint = root / "r3.pt"
    r3_checkpoint.write_bytes(b"opaque-r3-checkpoint")
    monkeypatch.setattr(ledger, "R3_ACTOR_SHA256", ledger.file_hash(r3_actor))
    monkeypatch.setattr(ledger, "R3_CHECKPOINT_SHA256", ledger.file_hash(r3_checkpoint))
    monkeypatch.setattr(ledger, "R3_STATE_SHA256", h("r3-state"))
    scenarios = root / "scenarios.json"
    write_json(scenarios, {"version": "scenes", "splits": {"validation": []}})
    inventory = root / "inventory"
    inventory.mkdir()
    base, base_boundaries = make_attempt(inventory, "r4_active_base", "warehouse-r4-active-trainer.v1",
        0, 100_000, ledger.digest(json.loads(scenarios.read_text())))
    parent = base_boundaries[-1]
    stage = {"trainer_version": "warehouse-r4-active-trainer.v1",
        "actor_sha256": parent["actor"]["sha256"],
        "checkpoint_sha256": parent["checkpoint"]["sha256"],
        "additional_ppo_joint_steps": 100_000,
        "parent_actor_parameters_sha256": parent["actor"]["parameters_sha256"],
        "program_content_sha256": h("program"), "parent_optimizer_sha256": h("optimizer"),
        "parent_rng_sha256": h("rng"), "parent_owned_rng_sha256": h("owned-rng"),
        "feedback_lambda": .01,
        "actor_and_optimizer_preserved": True, "pre_update_actor_parity": True,
        "pre_update_optimizer_parity": True,
        "rng_restored_before_curriculum_resets": True,
        "inflight_episodes_restarted_for_versioned_curriculum": True,
        "runtime_action_override": False}
    child, child_boundaries = make_attempt(inventory, "r4_active_child", "warehouse-r4-active-trainer.v2",
        100_000, 150_000, ledger.digest(json.loads(scenarios.read_text())), stage)
    audit = base / "evaluations" / "step_0100000" / "paired_report.json"
    write_json(audit, {"version": ledger.ACTIVE_VERSION, "status": "passed", "selected": True,
        "absolute_checks": {"all": True}, "relative_checks_passed": 4,
        "baseline": {"actor": {"artifact_sha256": ledger.R3_ACTOR_SHA256}},
        "candidate": {"actor": {"artifact_sha256": parent["actor"]["sha256"]}},
        "scenario_manifest": {"path": str(scenarios),
                              "sha256": ledger.file_hash(scenarios)}})
    zero = root / "zero.json"
    write_json(zero, {"version": "zero.v1", "status": "completed",
                     "execution": {"ppo_joint_steps": 0, "optimizer_updates": 0}})
    return {"inventory": inventory, "attempts": [base, child], "r3_actor": r3_actor,
        "r3_checkpoint": r3_checkpoint, "scenarios": scenarios, "audit": audit,
        "zero": zero, "output": root / "ledger.json", "child": child,
        "selected_actor": parent["actor"]["sha256"]}


def test_dag_delta_ledger_reopens_every_immutable_source(monkeypatch, tmp_path):
    values = fixture(monkeypatch, tmp_path)
    report = ledger.build(inventory_root=values["inventory"], attempts=values["attempts"],
        r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
        scenario_manifest=values["scenarios"], active_policy_report=values["audit"],
        zero_step_records=[values["zero"]], output=values["output"])
    assert report["total_actual_additional_joint_steps"] == 150_000
    assert report["unique_segment_count"] == 3
    assert [row["fresh_ppo_joint_steps"] for row in report["attempts"]] == [100_000, 50_000]
    assert sum(row["selected"] for row in report["attempts"]) == 1
    assert report["selection"]["training_actor_sha256"] == values["selected_actor"]
    assert ledger.read_saved_ledger(values["output"],
        expected_report_sha256=ledger.file_hash(values["output"]),
        expected_selected_actor_sha256=values["selected_actor"],
        expected_active_report_sha256=ledger.file_hash(values["audit"]),
        expected_scenario_manifest_sha256=ledger.file_hash(values["scenarios"])) == report

    # A hand-edited terminal counter cannot survive semantic reopening.
    run_path = values["child"] / "run.json"
    run = json.loads(run_path.read_text()); run["current_additional_joint_steps"] = 100_000
    write_json(run_path, run)
    with pytest.raises(ValueError):
        ledger.read_saved_ledger(values["output"],
            expected_report_sha256=ledger.file_hash(values["output"]),
            expected_selected_actor_sha256=values["selected_actor"],
            expected_active_report_sha256=ledger.file_hash(values["audit"]),
            expected_scenario_manifest_sha256=ledger.file_hash(values["scenarios"]))


def test_attempt_omission_and_nonpassing_selection_fail_closed(monkeypatch, tmp_path):
    values = fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exactly match"):
        ledger.build(inventory_root=values["inventory"], attempts=values["attempts"][:1],
            r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
            scenario_manifest=values["scenarios"], active_policy_report=values["audit"],
            zero_step_records=[values["zero"]], output=values["output"])
    audit = json.loads(values["audit"].read_text())
    audit.update(status="failed", selected=False)
    write_json(values["audit"], audit)
    with pytest.raises(ValueError, match="lacks a passing"):
        ledger.build(inventory_root=values["inventory"], attempts=values["attempts"],
            r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
            scenario_manifest=values["scenarios"], active_policy_report=values["audit"],
            zero_step_records=[values["zero"]], output=values["output"])
    assert not values["output"].exists()

    # A separate, explicitly requested closeout artifact may preserve the
    # authenticated budget and failed selection.  It can be semantically
    # replayed for diagnosis, but the ordinary admission reader rejects it.
    report = ledger.build(
        inventory_root=values["inventory"], attempts=values["attempts"],
        r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
        scenario_manifest=values["scenarios"], active_policy_report=values["audit"],
        zero_step_records=[values["zero"]], output=values["output"],
        diagnostic_failed_ledger=True,
    )
    assert report["status"] == "failed_no_eligible_actor"
    assert report["admission_eligible"] is False
    assert report["selection"] is None
    assert report["failure_reason"] == "no_passing_full_paired_audit"
    assert sum(row["selected"] for row in report["attempts"]) == 0
    assert ledger.read_saved_ledger(
        values["output"], expected_report_sha256=ledger.file_hash(values["output"]),
        expected_selected_actor_sha256=None,
        expected_active_report_sha256=ledger.file_hash(values["audit"]),
        expected_scenario_manifest_sha256=ledger.file_hash(values["scenarios"]),
        require_admission_eligible=False,
    ) == report
    with pytest.raises(ValueError, match="semantic replay or release binding"):
        ledger.read_saved_ledger(
            values["output"], expected_report_sha256=ledger.file_hash(values["output"]),
            expected_selected_actor_sha256=None,
            expected_active_report_sha256=ledger.file_hash(values["audit"]),
            expected_scenario_manifest_sha256=ledger.file_hash(values["scenarios"]),
        )


def test_zero_step_record_must_explicitly_be_zero(monkeypatch, tmp_path):
    values = fixture(monkeypatch, tmp_path)
    write_json(values["zero"], {"version": "zero.v1", "status": "completed",
                               "ppo_joint_steps": 1})
    with pytest.raises(ValueError, match="explicitly report zero"):
        ledger.build(inventory_root=values["inventory"], attempts=values["attempts"],
            r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
            scenario_manifest=values["scenarios"], active_policy_report=values["audit"],
            zero_step_records=[values["zero"]], output=values["output"])


def test_diagnostic_mode_cannot_wrap_a_passing_selection(monkeypatch, tmp_path):
    values = fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="only valid when no Actor passed"):
        ledger.build(
            inventory_root=values["inventory"], attempts=values["attempts"],
            r3_actor=values["r3_actor"], r3_checkpoint=values["r3_checkpoint"],
            scenario_manifest=values["scenarios"],
            active_policy_report=values["audit"],
            zero_step_records=[values["zero"]], output=values["output"],
            diagnostic_failed_ledger=True,
        )
    assert not values["output"].exists()
