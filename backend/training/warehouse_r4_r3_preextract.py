"""Extract an observed197 RCPD program from frozen-r3 neural trajectories.

This command performs no optimizer update.  Its environment steps are evidence
collection and are explicitly excluded from the r4 PPO budget ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .warehouse_native_common import atomic_json, digest
from .warehouse_r4_active_trainer import (
    ActiveTrainer, SOURCE_ACTOR_SHA256, SOURCE_CUMULATIVE_STEPS,
    _actor_parameter_sha, load_source,
)


VERSION = "warehouse-r4-r3-preextraction.v1"


def execute(output, *, environment_steps=50_000):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preextraction requires a new output directory")
    if environment_steps <= 0 or environment_steps % 16:
        raise ValueError("Evidence steps must be positive and divisible by 16")
    output.mkdir(parents=True)
    source, _, _ = load_source()
    source_parameter_sha = _actor_parameter_sha(source.model)
    probe = ActiveTrainer(source)
    # Collect actual stochastic neural trajectories, but deliberately perform
    # no PPO/critic/Adam update.  The frozen Actor is re-evaluated by fit_program.
    while probe.joint_steps < environment_steps:
        remaining = environment_steps - probe.joint_steps
        time_steps = min(probe.cfg["rollout_steps"], remaining // len(probe.envs))
        if time_steps <= 0:
            raise RuntimeError("Evidence boundary is not divisible by environment batch")
        probe.collect(time_steps)
    if _actor_parameter_sha(probe.model) != source_parameter_sha:
        raise RuntimeError("Preextraction changed the frozen source Actor")
    fit = probe.fit_program(SOURCE_ACTOR_SHA256)
    manager = probe.feedback_manager
    if manager.program is None or not manager.reliable:
        raise RuntimeError("Frozen-r3 observed197 program did not pass training admission")
    # ``fit_program`` normally uses a learner's cumulative clock.  These steps
    # are read-only evidence collection, so bind the fit to the
    # unchanged 3.95M Actor clock and preserve the probe clock explicitly.
    fit["evidence_collection_clock"] = int(fit["step"])
    fit["step"] = SOURCE_CUMULATIVE_STEPS
    fit["step_semantics"] = "unchanged_source_actor_ppo_clock"
    fit["extraction_environment_steps"] = environment_steps
    fit["r4_ppo_joint_steps"] = 0
    manager.last_fit_report = fit
    manager.last_step = SOURCE_CUMULATIVE_STEPS
    manager.last_fit_step = SOURCE_CUMULATIVE_STEPS
    payload = {
        "version": VERSION,
        "test_fixture": False,
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_actor_parameters_sha256": source_parameter_sha,
        "source_cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
        "extraction_environment_steps": environment_steps,
        "ppo_joint_steps": 0,
        "optimizer_updates": 0,
        "actor_changed": False,
        "runtime_action_override": False,
        "program_content_sha256": digest(manager.program.to_dict()),
        "feedback_manager": manager.state_dict(),
        "fit_report": fit,
    }
    atomic_json(output / "r3_preextraction.json", payload)
    atomic_json(output / "program.json", manager.program.to_dict())
    atomic_json(output / "report.json", {
        "version": VERSION,
        "status": "completed",
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_actor_parameters_sha256": source_parameter_sha,
        "extraction_environment_steps": environment_steps,
        "ppo_joint_steps": 0,
        "program_content_sha256": payload["program_content_sha256"],
        "selected": fit["selected"],
        "runtime_action_override": False,
    })
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--environment-steps", type=int, default=50_000)
    args = parser.parse_args(argv)
    payload = execute(args.output, environment_steps=args.environment_steps)
    print(json.dumps({
        "status": "completed",
        "source_actor_sha256": payload["source_actor_sha256"],
        "program_content_sha256": payload["program_content_sha256"],
        "fidelity": payload["fit_report"]["selected"]["fidelity"],
        "critical": payload["fit_report"]["selected"]["critical"],
        "ppo_joint_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
