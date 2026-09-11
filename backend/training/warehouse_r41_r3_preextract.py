"""Extract the initial r4.1 RCPD from frozen-r3 conflict trajectories.

Evidence collection uses the same successor-table environment as online play.
It performs no optimizer update and is excluded from the 2M PPO budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .warehouse_native_common import atomic_json, digest, file_hash
from .warehouse_r41_active_trainer import (
    R41ActiveTrainer, SOURCE_ACTOR_SHA256, SOURCE_CUMULATIVE_STEPS,
    _actor_parameter_sha, load_source,
)


VERSION = "warehouse-r41-r3-preextraction.v1"


def _binding(original_scenarios_path):
    path = Path(original_scenarios_path).resolve()
    value = json.loads(path.read_text())
    validation = value.get("splits", {}).get("validation", [])
    if len(validation) != 50:
        raise ValueError("R4.1 requires the frozen 50-scene original validation split")
    return {"path": str(path), "file_sha256": file_hash(path),
            "validation_entries_sha256": digest(validation), "count": len(validation)}


def execute(output, *, conflict_manifest_path, original_scenarios_path,
            environment_steps=50_000):
    output = Path(output).resolve()
    conflict_path = Path(conflict_manifest_path).resolve()
    if output.exists():
        raise FileExistsError("Preextraction requires a new output directory")
    if environment_steps <= 0 or environment_steps % 16:
        raise ValueError("Evidence steps must be positive and divisible by 16")
    conflict = json.loads(conflict_path.read_text())
    output.mkdir(parents=True)
    source, _, _ = load_source()
    source_parameter_sha = _actor_parameter_sha(source.model)
    probe = R41ActiveTrainer(
        source, conflict,
        conflict_manifest_file_sha256=file_hash(conflict_path),
        original_validation_binding=_binding(original_scenarios_path),
    )
    while probe.joint_steps < environment_steps:
        remaining = environment_steps - probe.joint_steps
        time_steps = min(probe.cfg["rollout_steps"], remaining // len(probe.envs))
        if time_steps <= 0:
            raise RuntimeError("Evidence boundary is not divisible by environment batch")
        probe.collect(time_steps)
    if (_actor_parameter_sha(probe.model) != source_parameter_sha
            or probe.optimizer_updates != 0):
        raise RuntimeError("Preextraction changed the frozen source learner")
    if probe.action_authority["trainable"] != probe.action_authority["equal"]:
        raise RuntimeError("Preextraction observed an overwritten neural command")
    fit = probe.fit_program(SOURCE_ACTOR_SHA256)
    manager = probe.feedback_manager
    if manager.program is None or not manager.reliable:
        raise RuntimeError("Frozen-r3 conflict RCPD did not pass training admission")
    fit["evidence_collection_clock"] = int(fit["step"])
    fit["step"] = SOURCE_CUMULATIVE_STEPS
    fit["step_semantics"] = "unchanged_source_actor_ppo_clock"
    fit["extraction_environment_steps"] = environment_steps
    fit["r41_ppo_joint_steps"] = 0
    manager.last_fit_report = fit
    manager.last_step = SOURCE_CUMULATIVE_STEPS
    manager.last_fit_step = SOURCE_CUMULATIVE_STEPS
    conflict_api = probe.protocol["scenario_sampling"]
    payload = {
        "version": VERSION, "test_fixture": False,
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_actor_parameters_sha256": source_parameter_sha,
        "source_cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
        "conflict_contract_sha256": conflict_api["conflict_contract_sha256"],
        "conflict_contract_version": conflict_api["conflict_contract_version"],
        "conflict_graph_sha256": conflict_api["conflict_graph_sha256"],
        "conflict_manifest_version": conflict_api["manifest_version"],
        "conflict_manifest_file_sha256": file_hash(conflict_path),
        "conflict_manifest_semantic_sha256": conflict_api["manifest_semantic_sha256"],
        "conflict_train_entries_sha256": conflict_api["train_entries_sha256"],
        "extraction_environment_steps": environment_steps,
        "ppo_joint_steps": 0, "optimizer_updates": 0,
        "actor_changed": False, "runtime_action_override": False,
        "action_authority": {
            "trainable": probe.action_authority["trainable"],
            "equal": probe.action_authority["equal"], "overrides": 0,
        },
        "program_content_sha256": digest(manager.program.to_dict()),
        "feedback_manager": manager.state_dict(), "fit_report": fit,
    }
    atomic_json(output / "r3_preextraction.json", payload)
    atomic_json(output / "program.json", manager.program.to_dict())
    atomic_json(output / "report.json", {
        "version": VERSION, "status": "completed",
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_actor_parameters_sha256": source_parameter_sha,
        "conflict_contract_sha256": payload["conflict_contract_sha256"],
        "conflict_contract_version": payload["conflict_contract_version"],
        "conflict_graph_sha256": payload["conflict_graph_sha256"],
        "conflict_manifest_version": payload["conflict_manifest_version"],
        "conflict_manifest_semantic_sha256": payload["conflict_manifest_semantic_sha256"],
        "extraction_environment_steps": environment_steps,
        "ppo_joint_steps": 0,
        "program_content_sha256": payload["program_content_sha256"],
        "selected": fit["selected"], "runtime_action_override": False,
    })
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conflict-manifest", required=True)
    parser.add_argument("--original-scenarios", required=True)
    parser.add_argument("--environment-steps", type=int, default=50_000)
    args = parser.parse_args(argv)
    payload = execute(
        args.output, conflict_manifest_path=args.conflict_manifest,
        original_scenarios_path=args.original_scenarios,
        environment_steps=args.environment_steps,
    )
    print(json.dumps({
        "status": "completed", "source_actor_sha256": payload["source_actor_sha256"],
        "program_content_sha256": payload["program_content_sha256"],
        "fidelity": payload["fit_report"]["selected"]["fidelity"],
        "ppo_joint_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
