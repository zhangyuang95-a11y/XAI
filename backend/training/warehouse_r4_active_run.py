"""Run or resume the bounded r4 active-neural continuation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import torch

from .warehouse_native import atomic_torch_save
from .warehouse_native_common import atomic_json, digest, file_hash
from .warehouse_r4_active_trainer import (
    ActiveTrainer, SOURCE_CUMULATIVE_STEPS, VERSION, _actor_parameter_sha, load_source,
)


def _write_json(path, value):
    atomic_json(Path(path), value)


def _checkpoint_path(output, step):
    return output / "checkpoints" / f"step_{step:07d}.pt"


def execute(output, *, target, resume=None, initial_feedback=None, initial_lambda=.005,
            stage_source_checkpoint=None, stage_source_actor=None):
    output = Path(output).resolve()
    if target <= 0 or target > 1_000_000 or target % 50_000:
        raise ValueError("Target must be a positive 50k boundary no larger than 1M")
    if resume is None:
        if stage_source_checkpoint is None and initial_feedback is None:
            raise ValueError("Fresh r4 v7 training requires r3-bound initial feedback")
        if stage_source_checkpoint is not None and (initial_feedback is not None
                or stage_source_actor is None):
            raise ValueError("A stage continuation needs its Actor and no initial artifact")
        if output.exists():
            raise FileExistsError("Fresh r4 run requires a new output directory")
        output.mkdir(parents=True)
        source, scenes, _ = load_source()
        trainer = ActiveTrainer(source)
        if stage_source_checkpoint is not None:
            stage_path = Path(stage_source_checkpoint).resolve()
            stage_payload = torch.load(stage_path, map_location="cpu", weights_only=False)
            trainer.bind_r4_stage_source(
                stage_payload, checkpoint_sha256=file_hash(stage_path),
                actor_path=stage_source_actor,
            )
        else:
            initial_path = Path(initial_feedback).resolve()
            initial_payload = json.loads(initial_path.read_text())
            trainer.bind_initial_feedback(
                initial_payload, artifact_sha256=file_hash(initial_path),
                lambda_value=initial_lambda,
            )
        _write_json(output / "protocol.json", trainer.protocol)
        _write_json(output / "source_receipt.json", {
            "version": VERSION,
            "source_actor_sha256": trainer.source_actor_sha256,
            "source_checkpoint_sha256": trainer.protocol["source_checkpoint_sha256"],
            "source_state_sha256": trainer.source_state_sha256,
            "source_cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
            "scenario_manifest_sha256": sha256(json.dumps(scenes, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "runtime_action_override": False,
            "stage_source": deepcopy(trainer.protocol.get("stage_source")),
        })
    else:
        if initial_feedback is not None:
            raise ValueError("Resume uses the checkpoint-bound feedback program")
        source, _, _ = load_source()
        trainer = ActiveTrainer(source)
        saved_protocol = json.loads((output / "protocol.json").read_text())
        trainer.protocol["initial_feedback"] = saved_protocol.get("initial_feedback")
        if "stage_source" in saved_protocol:
            trainer.protocol["stage_source"] = saved_protocol["stage_source"]
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        trainer.load_state_dict_r4(payload)
        if trainer.joint_steps >= target:
            raise ValueError("Resume checkpoint already reached the requested target")
        if json.loads((output / "protocol.json").read_text()) != trainer.protocol:
            raise ValueError("Run protocol changed")
    run = {
        "version": VERSION, "status": "running", "pid": os.getpid(),
        "target_additional_joint_steps": target,
        "current_additional_joint_steps": trainer.joint_steps,
        "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
        "started_unix": time.time(), "runtime_action_override": False,
    }
    _write_json(output / "run.json", run)
    started = time.monotonic()
    while trainer.joint_steps < target:
        remaining = target - trainer.joint_steps
        time_steps = min(trainer.cfg["rollout_steps"], remaining // len(trainer.envs))
        if time_steps <= 0:
            raise RuntimeError("Boundary is not divisible by the environment batch")
        before = trainer.joint_steps
        batch = trainer.collect(time_steps)
        update = trainer.update(batch)
        if trainer.joint_steps != before + time_steps * len(trainer.envs):
            raise RuntimeError("Counted PPO steps differ from the sampled batch")
        if trainer.action_authority["trainable"] != trainer.action_authority["equal"]:
            raise RuntimeError("A neural action authority audit failed")
        record = {
            "joint_steps": trainer.joint_steps,
            "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
            "time_steps": time_steps,
            "feedback_lambda": trainer.current_lambda,
            "update_path": update.get("stable_update", {}).get("path"),
            "actor_loss": update.get("actor_loss"),
            "critic_loss": update.get("critic_loss"),
            "feedback_loss": update.get("feedback_loss", 0.0),
            "feedback_program_content_sha256": (
                digest(trainer.feedback_manager.program.to_dict())
                if trainer.current_lambda > 0 and trainer.feedback_manager.program is not None
                else None
            ),
            "feedback_program_source_actor_sha256": (
                trainer.feedback_manager.program.metadata.get("native_source_actor_sha256")
                if trainer.current_lambda > 0 and trainer.feedback_manager.program is not None
                else None
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if trainer.joint_steps % 50_000 == 0:
            boundary = output / "boundaries" / f"step_{trainer.joint_steps:07d}"
            boundary.mkdir(parents=True, exist_ok=False)
            actor = boundary / "actor.npz"
            trainer.export(actor)
            actor_sha = file_hash(actor)
            fit_started = time.monotonic()
            fit = trainer.fit_program(actor_sha)
            _write_json(boundary / "tree_fit.json", fit)
            if trainer.feedback_manager.program is not None:
                _write_json(boundary / "program.json", trainer.feedback_manager.program.to_dict())
            checkpoint = _checkpoint_path(output, trainer.joint_steps)
            atomic_torch_save(checkpoint, trainer.state_dict_r4())
            summary = {
                "version": VERSION,
                "step": trainer.joint_steps,
                "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
                "actor": {"path": str(actor.relative_to(output)), "sha256": actor_sha,
                          "parameters_sha256": _actor_parameter_sha(trainer.model)},
                "checkpoint": {"path": str(checkpoint.relative_to(output)),
                               "sha256": file_hash(checkpoint)},
                "RCPD": {"reliable": trainer.feedback_manager.reliable,
                         "lambda_next_cycle": trainer.current_lambda,
                         "fidelity": fit["selected"]["fidelity"],
                         "mean_kl": fit["selected"]["mean_kl"],
                         "fit_seconds": time.monotonic() - fit_started},
                "partner_mix": trainer.protocol["partners"],
                "shaping_counts": dict(trainer.shaping_counts),
                "sampling_counts": dict(trainer.sampling_counts),
                "action_authority": {**dict(trainer.action_authority), "overrides": 0},
                "elapsed_seconds": time.monotonic() - started,
                "evaluation_pending": True,
                "candidate_selected": False,
            }
            _write_json(boundary / "summary.json", summary)
            print(json.dumps({"event": "boundary", **summary}, sort_keys=True), flush=True)
    run.update(status="training_target_completed", current_additional_joint_steps=trainer.joint_steps,
        cumulative_joint_steps=SOURCE_CUMULATIVE_STEPS + trainer.joint_steps,
        elapsed_seconds=time.monotonic() - started)
    _write_json(output / "run.json", run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--initial-feedback")
    parser.add_argument("--initial-lambda", type=float, default=.005)
    parser.add_argument("--stage-source-checkpoint")
    parser.add_argument("--stage-source-actor")
    args = parser.parse_args(argv)
    execute(args.output, target=args.target, resume=args.resume,
            initial_feedback=args.initial_feedback, initial_lambda=args.initial_lambda,
            stage_source_checkpoint=args.stage_source_checkpoint,
            stage_source_actor=args.stage_source_actor)


if __name__ == "__main__":
    main()
