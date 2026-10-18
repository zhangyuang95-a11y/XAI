#!/usr/bin/env python3
"""Run the real bounded v6.18 pipeline; outputs cannot authorize admission."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_program import fit_program
from experiments.warehouse import protocol as p
from experiments.warehouse.admission import NativeBackend, _exclusive_json
from experiments.warehouse.quality_replay import replay_series


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--joint-steps", type=int, default=600, choices=range(120, 1201, 120))
    args = parser.parse_args(argv)
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("engineering output directory must be empty and isolated")
    output.mkdir(parents=True, exist_ok=True)
    source = runner.source_closure_sha256()
    spec = runner._checked_spec({
        "candidate_id": "warehouse-v618-engineering-only",
        "scenario_namespace": "warehouse-v618-engineering-only",
        "pair_seed": 6149, "joint_steps": args.joint_steps,
        "output_dir": str(output), "checkpoint_seeds": [264900, 264901],
        "smoke": True, "bc_episodes": 4, "bc_epochs": 2,
        "dagger_rounds": 0, "dagger_episodes": 0, "dagger_epochs": 0,
        "program_fit_method": p.REGISTERED_PROGRAM_FIT_METHOD,
        "source_closure_sha256": source,
    })
    directory = output / spec["candidate_id"] / f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}"
    split_seeds = {name: list(range(264920 + index*4, 264924 + index*4))
                   for index, name in enumerate(("fit", "calibration", "eligibility", "validation"))}
    plan = {"schema": "warehouse-v618-engineering-plan.v1", "publishable": False,
            "purpose": "engineering_execution_only_not_model_acceptance",
            "source_closure_sha256": source, "run_spec": spec,
            "program_split_seeds": split_seeds, "program_audit_seeds": [264940, 264941],
            "scientific_program_gates_expected": False,
            "note": "Tiny splits exercise the real fitter; no admission threshold is relaxed."}
    _exclusive_json(output / "plan.json", plan)
    authorization = {"authorization_kind": "warehouse-v618-engineering-only-v1",
        "publishable": False, "source_closure_sha256": source,
        "run_spec_sha256": runner._scientific_run_spec_sha256(spec),
        "training_config_sha256": runner._training_config_sha256(spec),
        "lock_script_path": str(Path(__file__).resolve()),
        "lock_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_directory": str(directory)}
    authorization_path = output / "authorization.json"
    _exclusive_json(authorization_path, authorization)
    try:
        training = runner.run_locked_engineering_pair(
            spec, authorization_receipt_path=authorization_path)
        _exclusive_json(output / "training-and-restore.json", training)
        backend = NativeBackend()
        results = {}
        for variant in p.VARIANTS:
            report = training["pair"]["variants"][variant]
            artifact = report["actor_artifact"]
            actor = backend.load_actor(artifact)
            evaluation = backend.evaluate(actor, spec["checkpoint_seeds"], trace=True)
            replay = replay_series(
                actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                role_sha256=actor.role_sha256, rows=evaluation["episodes"],
                seeds=[(seed, seed) for seed in spec["checkpoint_seeds"]])
            seed_list = [seed for values in split_seeds.values() for seed in values]
            rows, collection_evidence = runner.collect_authorized_program_rows(actor, seed_list)
            programs, program_artifacts, program_reports = {}, {}, {}
            before = dict(actor.role_sha256)
            for role in p.ROLES:
                parts = {name: [row for row in rows[role] if row["base_seed"] in seeds]
                         for name, seeds in split_seeds.items()}
                program = fit_program(role, actor, *(parts[name] for name in
                    ("fit", "calibration", "eligibility", "validation")),
                    method=p.REGISTERED_PROGRAM_FIT_METHOD)
                path = output / "programs" / f"{variant}.{role}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                program_sha = program.save(path)
                programs[role] = backend.load_program(
                    {"path": str(path), "sha256": program_sha}, actor.role_sha256[role])
                program_artifacts[role] = {"path": str(path), "sha256": program_sha,
                                           "actor_role_sha256": actor.role_sha256[role]}
                program_reports[role] = program.report
            if before != actor.role_sha256:
                raise RuntimeError("engineering program fitting changed frozen Actor identity")
            audits = backend.audit_programs(
                actor, programs, plan["program_audit_seeds"],
                trace_dir=output / "audit-traces" / variant)
            trace_bytes = {role: Path(audits[role]["trace_path"]).read_bytes() for role in p.ROLES}
            trace_pass = backend.verify_program_trace(
                actor=actor, programs=programs, trace_bytes_by_role=trace_bytes,
                audits=audits, seeds=plan["program_audit_seeds"])
            if trace_pass is not True:
                raise RuntimeError("engineering frozen Actor/program trace verification failed")
            results[variant] = {"actor_artifact": artifact, "quality_evaluation": evaluation,
                "quality_replay": replay, "program_collection_authorization": collection_evidence,
                "program_artifacts": program_artifacts, "program_fit_reports": program_reports,
                "program_audits": audits, "program_trace_verification": trace_pass,
                "feedback_rows": report["feedback_rows"],
                "feedback_gradient_norm": report["feedback_gradient_norm"],
                "scientific_program_gates_evaluated": False}
            _exclusive_json(output / f"{variant}.execution.json", results[variant])
            print(json.dumps({"completed": variant,
                              "engineering_quality": evaluation["summary"],
                              "program_trace_verified": trace_pass,
                              "publishable": False}), flush=True)
        if runner.source_closure_sha256() != source:
            raise RuntimeError("source changed during engineering execution")
        if any(results["raw"][key][role] != 0 for key in
               ("feedback_rows", "feedback_gradient_norm") for role in p.ROLES):
            raise RuntimeError("raw engineering run received program feedback")
        result = {"schema": "warehouse-v618-engineering-result.v1",
                  "purpose": plan["purpose"], "publishable": False,
                  "engineering_execution_passed": True, "model_acceptance_passed": False,
                  "source_closure_sha256": source,
                  "authorization_sha256": sha256(authorization_path.read_bytes()).hexdigest(),
                  "paired_start_and_scenario_equal": bool(
                      training["pair"]["variants"]["raw"]["initial_role_sha256"] ==
                      training["pair"]["variants"]["regularized"]["initial_role_sha256"] and
                      training["pair"]["variants"]["raw"]["scenario_schedule_sha256"] ==
                      training["pair"]["variants"]["regularized"]["scenario_schedule_sha256"]),
                  "restore_equivalence": training["restore_equivalence"],
                  "training_reset_audit": training["reset_audit"], "variants": results}
        if not result["paired_start_and_scenario_equal"]:
            raise RuntimeError("engineering paired start or scenario differs")
        _exclusive_json(output / "result.json", result)
        print(json.dumps({"engineering_execution_passed": True,
                          "model_acceptance_passed": False,
                          "result_path": str(output / "result.json")}), flush=True)
        return 0
    except Exception as exc:
        _exclusive_json(output / "failure.json", {
            "schema": "warehouse-v618-engineering-failure.v1", "publishable": False,
            "source_closure_sha256": source, "model_acceptance_passed": False,
            "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
