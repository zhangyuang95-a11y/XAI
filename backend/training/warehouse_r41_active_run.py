"""Run/resume the bounded r4.1 continuation with 50k fail-closed gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import warehouse_r41_active_evaluation as evaluation
from .warehouse_native import atomic_torch_save
from .warehouse_native_common import atomic_json, digest, file_hash
from .warehouse_r41_active_trainer import (
    BOUNDARY_INTERVAL, MAXIMUM_ADDITIONAL_JOINT_STEPS,
    R41ActiveTrainer, SOURCE_ACTOR_PATH, SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT_SHA256, SOURCE_CUMULATIVE_STEPS, VERSION,
    _actor_parameter_sha, load_source,
)


RUN_VERSION = "warehouse-r41-active-run.v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _original_binding(path):
    path = Path(path).resolve()
    value = _read(path)
    validation = value.get("splits", {}).get("validation", [])
    if len(validation) != 50:
        raise ValueError("R4.1 requires all 50 original validation scenes")
    return {"path": str(path), "file_sha256": file_hash(path),
            "validation_entries_sha256": digest(validation), "count": 50}


def _source_receipt(trainer, conflict_path, conflict, original_path):
    """Build the complete immutable r3 + conflict-runtime provenance record."""
    sampling = trainer.protocol["scenario_sampling"]
    return {
        "version": RUN_VERSION,
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_state_sha256": trainer.source_state_sha256,
        "source_cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
        "failed_r4_parent_used": False,
        "conflict_manifest": {
            "path": str(Path(conflict_path).resolve()),
            "file_sha256": file_hash(conflict_path),
            "semantic_sha256": digest(conflict),
            "version": sampling["manifest_version"],
            "contract_sha256": sampling["conflict_contract_sha256"],
            "contract_version": sampling["conflict_contract_version"],
            "graph_sha256": sampling["conflict_graph_sha256"],
        },
        "original_validation": _original_binding(original_path),
        "runtime_action_override": False,
    }


def _checkpoint_payload(trainer):
    return {
        "run_version": RUN_VERSION,
        "trainer": trainer.state_dict_r41(),
        "joint_steps": trainer.joint_steps,
        "actor_parameters_sha256": _actor_parameter_sha(trainer.model),
        "protocol_sha256": digest(trainer.protocol),
        "runtime_action_override": False,
    }


def _load_checkpoint(path):
    path = Path(path).resolve()
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (value.get("run_version") != RUN_VERSION
            or value.get("runtime_action_override") is not False
            or value.get("trainer", {}).get("version") != VERSION
            or value.get("joint_steps") != value.get("trainer", {}).get("joint_steps")):
        raise ValueError("R4.1 resume checkpoint envelope differs")
    return value


def _boundary_dir(output, step):
    return output / "boundaries" / f"step_{step:07d}"


def _resume_head(output):
    """Return the sole contiguous committed checkpoint allowed for resume."""
    boundaries = output / "boundaries"
    committed = []
    if boundaries.is_dir():
        for path in boundaries.glob("step_*"):
            if path.is_dir() and path.name[5:].isdigit():
                committed.append((int(path.name[5:]), path))
    committed.sort()
    prior = 0
    for step, _ in committed:
        if step != prior + BOUNDARY_INTERVAL:
            raise ValueError("Committed r4.1 boundaries are not contiguous")
        prior = step
    if committed:
        step, path = committed[-1]
        summary = _read(path / "summary.json")
        checkpoint = path / summary["checkpoint"]["path"]
        if (summary.get("step") != step or not checkpoint.is_file()
                or file_hash(checkpoint) != summary["checkpoint"].get("sha256")):
            raise ValueError("Committed r4.1 resume head differs")
        return step, checkpoint.resolve()
    initial = output / "initial" / "checkpoint.pt"
    receipt = _read(output / "initial" / "receipt.json")
    if (receipt.get("step") != 0 or not initial.is_file()
            or file_hash(initial) != receipt.get("checkpoint_sha256")):
        raise ValueError("Initial r4.1 resume checkpoint differs")
    return 0, initial.resolve()


def _commit_boundary(output, trainer, *, updates, baseline_report_path,
                     original_scenarios_path, conflict_manifest_path):
    step = trainer.joint_steps
    if step <= 0 or step % BOUNDARY_INTERVAL:
        raise ValueError("R4.1 boundary must be a positive 50k step")
    destination = _boundary_dir(output, step)
    if destination.exists():
        raise FileExistsError("R4.1 boundary already exists")
    pending_root = output / "pending"
    pending_root.mkdir(parents=True, exist_ok=True)
    pending = pending_root / f"step_{step:07d}.{os.getpid()}"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    try:
        actor_path = pending / "actor.npz"
        trainer.export(actor_path)
        actor_sha = file_hash(actor_path)
        fit_started = time.monotonic()
        fit = trainer.fit_program(actor_sha)
        atomic_json(pending / "tree_fit.json", fit)
        if trainer.feedback_manager.program is None:
            raise RuntimeError("Every r4.1 boundary must produce an RCPD candidate")
        atomic_json(pending / "program.json", trainer.feedback_manager.program.to_dict())
        checkpoint_path = pending / "checkpoint.pt"
        atomic_torch_save(checkpoint_path, _checkpoint_payload(trainer))
        with (pending / "training_segment.jsonl").open("x", encoding="utf-8") as stream:
            for item in updates:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
        audit = evaluation.paired_dual_audit(
            SOURCE_ACTOR_PATH.resolve(), actor_path,
            original_scenarios_path, conflict_manifest_path,
            pending / "evaluation", baseline_report_path=baseline_report_path,
        )
        authority = {**dict(trainer.action_authority), "overrides":
                     trainer.action_authority["trainable"] - trainer.action_authority["equal"]}
        selected = bool(
            audit["selected"]
            and trainer.feedback_manager.reliable
            and authority["overrides"] == 0
        )
        summary = {
            "version": RUN_VERSION, "step": step,
            "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + step,
            "actor": {"path": "actor.npz", "sha256": actor_sha,
                      "parameters_sha256": _actor_parameter_sha(trainer.model)},
            "checkpoint": {"path": "checkpoint.pt", "sha256": file_hash(checkpoint_path)},
            "training_segment": {"path": "training_segment.jsonl",
                                 "sha256": file_hash(pending / "training_segment.jsonl"),
                                 "start_step": step - BOUNDARY_INTERVAL,
                                 "end_step": step},
            "RCPD": {
                "report_path": "tree_fit.json", "report_sha256": file_hash(pending / "tree_fit.json"),
                "program_path": "program.json", "program_sha256": file_hash(pending / "program.json"),
                "reliable_for_training": trainer.feedback_manager.reliable,
                "lambda_next_cycle": trainer.current_lambda,
                "fidelity": fit["selected"]["fidelity"],
                "mean_kl": fit["selected"]["mean_kl"],
                "fit_seconds": time.monotonic() - fit_started,
            },
            "dual_evaluation": {
                "path": "evaluation/paired_dual_report.json",
                "sha256": file_hash(pending / "evaluation/paired_dual_report.json"),
                "selected": audit["selected"],
            },
            "partner_mix": deepcopy(trainer.protocol["partners"]),
            "shaping_counts": dict(trainer.shaping_counts),
            "sampling_counts": dict(trainer.sampling_counts),
            "action_authority": authority,
            "candidate_selected": selected,
            "selection_requires_both_validation_suites": True,
            "runtime_action_override": False,
        }
        atomic_json(pending / "summary.json", summary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, destination)
        return _read(destination / "summary.json")
    except BaseException:
        # An incomplete directory is deliberately outside the immutable
        # boundary namespace and cannot be mistaken for a committed checkpoint.
        raise


def execute(output, *, target, conflict_manifest_path, original_scenarios_path,
            initial_feedback=None, initial_lambda=.001, resume=None):
    output = Path(output).resolve()
    conflict_path = Path(conflict_manifest_path).resolve()
    original_path = Path(original_scenarios_path).resolve()
    if (target <= 0 or target > MAXIMUM_ADDITIONAL_JOINT_STEPS
            or target % BOUNDARY_INTERVAL):
        raise ValueError("Target must be a positive 50k boundary no larger than 2M")
    if resume is None and output.exists():
        raise FileExistsError("Fresh r4.1 run requires a new output directory")
    if resume is not None and not output.is_dir():
        raise ValueError("Resume output directory does not exist")
    conflict = _read(conflict_path)
    source, _, _ = load_source()
    trainer = R41ActiveTrainer(
        source, conflict,
        conflict_manifest_file_sha256=file_hash(conflict_path),
        original_validation_binding=_original_binding(original_path),
    )
    if resume is None:
        if initial_feedback is None:
            raise ValueError("Fresh r4.1 training requires conflict-bound r3 RCPD")
        feedback_path = Path(initial_feedback).resolve()
        artifact = _read(feedback_path)
        trainer.bind_initial_feedback(
            artifact, artifact_sha256=file_hash(feedback_path),
            lambda_value=initial_lambda,
        )
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", trainer.protocol)
        atomic_json(output / "source_receipt.json", _source_receipt(
            trainer, conflict_path, conflict, original_path))
        initial_dir = output / "initial"
        initial_dir.mkdir()
        initial_checkpoint = initial_dir / "checkpoint.pt"
        atomic_torch_save(initial_checkpoint, _checkpoint_payload(trainer))
        atomic_json(initial_dir / "receipt.json", {
            "version": RUN_VERSION, "step": 0,
            "checkpoint_sha256": file_hash(initial_checkpoint),
            "initial_feedback_sha256": file_hash(feedback_path),
            "protocol_sha256": digest(trainer.protocol),
        })
    else:
        if initial_feedback is not None:
            raise ValueError("Resume uses checkpoint-bound RCPD state")
        saved_protocol = _read(output / "protocol.json")
        trainer.protocol["initial_feedback"] = deepcopy(saved_protocol.get("initial_feedback"))
        if trainer.protocol != saved_protocol:
            raise ValueError("R4.1 run protocol or manifest changed")
        expected_receipt = _source_receipt(
            trainer, conflict_path, conflict, original_path)
        if _read(output / "source_receipt.json") != expected_receipt:
            raise ValueError("R4.1 source receipt changed before resume")
        resume_step, resume_head = _resume_head(output)
        if Path(resume).resolve() != resume_head:
            raise ValueError("Resume must use the latest committed r4.1 checkpoint")
        checkpoint = _load_checkpoint(resume_head)
        trainer.load_state_dict_r41(checkpoint["trainer"])
        if (trainer.joint_steps != resume_step
                or _actor_parameter_sha(trainer.model) != checkpoint["actor_parameters_sha256"]
                or trainer.joint_steps >= target):
            raise ValueError("Resume checkpoint is stale or already at target")
    baseline_dir = output / "baseline"
    baseline_report_path = baseline_dir / "baseline_report.json"
    if baseline_report_path.exists():
        evaluation.read_dual_baseline(
            baseline_report_path, baseline_actor=SOURCE_ACTOR_PATH.resolve(),
            original_scenarios_path=original_path,
            conflict_manifest_path=conflict_path,
        )
    else:
        evaluation.prepare_dual_baseline(
            SOURCE_ACTOR_PATH.resolve(), original_path, conflict_path, baseline_dir)
    run = {
        "version": RUN_VERSION, "status": "running", "pid": os.getpid(),
        "target_additional_joint_steps": target,
        "current_additional_joint_steps": trainer.joint_steps,
        "maximum_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
        "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
        "started_unix": time.time(), "runtime_action_override": False,
    }
    atomic_json(output / "run.json", run)
    started = time.monotonic()
    selected_summary = None
    while trainer.joint_steps < target and selected_summary is None:
        next_boundary = min(
            target,
            ((trainer.joint_steps // BOUNDARY_INTERVAL) + 1) * BOUNDARY_INTERVAL,
        )
        updates = []
        while trainer.joint_steps < next_boundary:
            remaining = next_boundary - trainer.joint_steps
            time_steps = min(trainer.cfg["rollout_steps"], remaining // len(trainer.envs))
            if time_steps <= 0:
                raise RuntimeError("50k boundary is not divisible by environment batch")
            before = trainer.joint_steps
            batch = trainer.collect(time_steps)
            update = trainer.update(batch)
            expected = before + time_steps * len(trainer.envs)
            if trainer.joint_steps != expected:
                raise RuntimeError("Counted PPO steps differ from sampled batch")
            if trainer.action_authority["trainable"] != trainer.action_authority["equal"]:
                raise RuntimeError("A neural action authority audit failed")
            updates.append({
                "start_joint_step": before, "end_joint_step": trainer.joint_steps,
                "time_steps": time_steps,
                "feedback_lambda": trainer.current_lambda,
                "update_path": update.get("stable_update", {}).get("path"),
                "actor_loss": update.get("actor_loss"),
                "critic_loss": update.get("critic_loss"),
                "feedback_loss": update.get("feedback_loss", 0.),
            })
        summary = _commit_boundary(
            output, trainer, updates=updates,
            baseline_report_path=baseline_report_path,
            original_scenarios_path=original_path,
            conflict_manifest_path=conflict_path,
        )
        if summary["candidate_selected"]:
            selected_summary = summary
            atomic_json(output / "selection.json", {
                "version": RUN_VERSION, "status": "selected",
                "earliest_passing_step": trainer.joint_steps,
                "boundary_summary_path": str(
                    (_boundary_dir(output, trainer.joint_steps) / "summary.json").relative_to(output)),
                "boundary_summary_sha256": file_hash(
                    _boundary_dir(output, trainer.joint_steps) / "summary.json"),
                "actor_sha256": summary["actor"]["sha256"],
                "runtime_action_override": False,
            })
    if selected_summary is not None:
        status = "selected_early"
    elif trainer.joint_steps == MAXIMUM_ADDITIONAL_JOINT_STEPS:
        status = "budget_exhausted_no_selection"
    else:
        status = "target_completed_no_selection"
    run.update(
        status=status,
        current_additional_joint_steps=trainer.joint_steps,
        cumulative_joint_steps=SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
        elapsed_seconds=time.monotonic() - started,
        selected=selected_summary is not None,
    )
    atomic_json(output / "run.json", run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--conflict-manifest", required=True)
    parser.add_argument("--original-scenarios", required=True)
    parser.add_argument("--initial-feedback")
    parser.add_argument("--initial-lambda", type=float, default=.001)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    result = execute(
        args.output, target=args.target,
        conflict_manifest_path=args.conflict_manifest,
        original_scenarios_path=args.original_scenarios,
        initial_feedback=args.initial_feedback, initial_lambda=args.initial_lambda,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
