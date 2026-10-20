#!/usr/bin/env python3
"""Single-use, hash-bound A32 continuation on spent development episodes only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v623_protocol as v
from experiments.warehouse import v622_recovery_protocol as old
from scripts.run_warehouse_v622_capacity import arrays
from scripts.run_warehouse_v621_diagnostics import file_sha, pointer, read_pinned, runtime_identity, write_new

ZERO = {"environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "pilot_opened": False, "publishable": False, "sealed_audit_used": False,
        "heldout_used": False}


def _prior(stage, source):
    if stage == "S1":
        return None
    prior = v.folder("S1")
    summary = json.loads((prior / "summary.json").read_bytes())
    audit = json.loads((prior / "audit.json").read_bytes())
    if (summary.get("all_six_pass") is not False or summary.get("next_stage") != "S2"
            or audit.get("passed") is not True
            or audit.get("source_sha256") != source
            or audit.get("summary_sha256") != file_sha(prior / "summary.json")):
        raise RuntimeError("S2 requires independently audited complete S1 failure")
    return {"summary": pointer(prior / "summary.json"), "audit": pointer(prior / "audit.json")}


def prepare(stage):
    source = v.source_sha()
    actor, actor_path = old.checked_actor(ROOT)
    episodes, manifest = old.checked_inputs(ROOT)
    prior = {job["job_id"]: v.checked_prior(job) for job in v.jobs()}
    plan = {"schema": "warehouse-v623-refinement-plan.v1", "candidate_id": v.CANDIDATE,
        "stage": stage, "source_sha256": source, "source_files": v.source_files(),
        "recipe_sha256": v.recipe(), "runtime": runtime_identity(),
        "actor": pointer(actor_path), "actor_role_sha256": old.ROLE_SHA256,
        "inputs": manifest, "targets": old.target_identity(actor, episodes),
        "folds": old.grouped_folds(), "jobs": v.jobs(), "prior_A32": prior,
        "stage_dependency": _prior(stage, source), "scientific_retry": False, **ZERO}
    if source != v.source_sha():
        raise RuntimeError("source changed while preparing")
    folder = v.folder(stage)
    folder.mkdir(parents=True, exist_ok=False)
    source_root = folder / "source" / source
    for relative, expected in plan["source_files"].items():
        raw = read_pinned(ROOT / relative, expected)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o444)
    write_new(folder / "plan.json", plan)
    write_new(folder / "plan.claim.json", {"sha256": file_sha(folder / "plan.json"),
        "single_use": True})
    return folder / "plan.json"


def checked_plan(stage):
    path = v.folder(stage) / "plan.json"
    plan = json.loads(path.read_bytes())
    claim = json.loads((path.parent / "plan.claim.json").read_bytes())
    if (plan.get("schema") != "warehouse-v623-refinement-plan.v1"
            or plan.get("candidate_id") != v.CANDIDATE or plan.get("stage") != stage
            or claim != {"sha256": file_sha(path), "single_use": True}
            or plan.get("source_sha256") != v.source_sha()
            or plan.get("source_files") != v.source_files()
            or plan.get("recipe_sha256") != v.recipe()
            or plan.get("runtime") != runtime_identity()
            or plan.get("jobs") != v.jobs() or plan.get("folds") != old.grouped_folds()
            or plan.get("actor", {}).get("sha256") != old.ACTOR_SHA256
            or plan.get("actor_role_sha256") != old.ROLE_SHA256
            or plan.get("scientific_retry") is not False
            or any(plan.get(k) != value for k, value in ZERO.items())):
        raise RuntimeError("v6.23 source, plan, actor or contract mismatch")
    read_pinned(plan["actor"]["path"], plan["actor"]["sha256"])
    for path, expected in plan["inputs"]["files_sha256"].items():
        read_pinned(path, expected)
    if plan["prior_A32"] != {job["job_id"]: v.checked_prior(job) for job in v.jobs()}:
        raise RuntimeError("historical A32 input changed")
    if plan["stage_dependency"] != _prior(stage, plan["source_sha256"]):
        raise RuntimeError("stage authority changed")
    return path, plan


def job(stage, job_id):
    from backend.training.warehouse_v623_search import fit_structure, ContinuationBudget
    from backend.training.warehouse_v622_program import (fit_program, program_metrics,
        program_structure_sha256, gate_metrics, ProgramBudgetExhausted)
    from backend.training.warehouse_v6_actor import role_sha256
    plan_path, plan = checked_plan(stage)
    spec = next((j for j in plan["jobs"] if j["job_id"] == job_id), None)
    if spec is None:
        raise ValueError("unknown job")
    folder = plan_path.parent / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "started.json", {"job": spec, "stage": stage,
        "plan_sha256": file_sha(plan_path), **ZERO})
    start = time.monotonic()
    try:
        actor, actor_path = old.checked_actor(ROOT)
        episodes, manifest = old.checked_inputs(ROOT)
        if manifest != plan["inputs"] or old.target_identity(actor, episodes) != plan["targets"]:
            raise RuntimeError("frozen target cohort changed")
        fold = plan["folds"][spec["fold"]]
        parts = {split: [row for case in fold["splits"][split]
                         for row in episodes[case["base"]][spec["role"]]]
                 for split in v.SPLITS}
        role_actor = actor.roles[spec["role"]]
        prior_ref = plan["prior_A32"][job_id]["structure"]
        initial = json.loads(read_pinned(prior_ref["path"], prior_ref["sha256"]))
        initial_sha = program_structure_sha256(initial)
        budget = ContinuationBudget()
        fit_begin = time.monotonic()
        structure, search = fit_structure(stage, *arrays(actor, spec["role"], parts["fit"]),
            feature_names=role_actor.feature_names, actor_role_sha256=role_actor.role_sha256,
            initial_structure=initial, initial_structure_sha256=initial_sha,
            budget=budget, checkpoint_path=folder / "checkpoint.json")
        write_new(folder / "structure.json", structure)
        write_new(folder / "search-report.json", search)
        print(json.dumps({"event": "structure_ready", "stage": stage, "job": job_id,
            "leaf_count": structure["leaf_count"], "search_calls": search["new_leaf_calls"],
            "search_stop": search["stop_reason"]}), flush=True)
        program = fit_program(spec["role"], actor, *(parts[s] for s in v.SPLITS),
            structure=structure, structure_sha256=program_structure_sha256(structure),
            fit_method="v620_m2_expanded_distribution_refit_v622",
            fit_budget={"deadline_monotonic": fit_begin + budget.wall_seconds,
                "maximum_leaf_api_calls": budget.final_leaf_reserve,
                "prior_leaf_optimization_calls": search["new_leaf_calls"],
                "max_total_leaf_optimization_calls": budget.maximum_new_leaf_calls,
                "provenance_sha256": file_sha(folder / "search-report.json")})
        metrics = {s: program_metrics(program, actor, rows) for s, rows in parts.items()}
        if time.monotonic() - fit_begin >= budget.wall_seconds:
            raise ProgramBudgetExhausted("metrics exceeded 90-minute job budget",
                report=dict(program.report, status="budget_exhausted", scientific_failure=True))
        program.save(folder / "program.json")
        search_calls, final_calls = search["new_leaf_calls"], program.report["leaf_optimization_calls_outside_search"]
        checks = {"search_calls_with_reserved_final": search_calls <= budget.maximum_new_leaf_calls - budget.final_leaf_reserve,
            "final_calls_at_most_32": final_calls <= 32,
            "total_new_calls_at_most_20000": search_calls + final_calls <= 20000,
            "all_final_leaves_refitted": final_calls == structure["leaf_count"],
            "wall_at_most_90_minutes": time.monotonic() - fit_begin < 5400,
            "no_actor_change": file_sha(actor_path) == old.ACTOR_SHA256 and all(
                role_sha256(actor.roles[r].weights, model_version=actor.roles[r].model_version)
                    == old.ROLE_SHA256[r] for r in v.ROLES)}
        if not all(checks.values()):
            raise RuntimeError("scientific contract check failed: " + repr(checks))
        result = {"schema": "warehouse-v623-refinement-result.v1", "job": spec, "stage": stage,
            "status": "completed", "plan_sha256": file_sha(plan_path),
            "actor": plan["actor"], "initial_structure": prior_ref,
            "initial_structure_sha256": initial_sha,
            "structure": pointer(folder / "structure.json"),
            "search_report": pointer(folder / "search-report.json"),
            "program": pointer(folder / "program.json"), "metrics": metrics,
            "gate": gate_metrics(metrics["validation"]), "budget_checks": checks,
            "search_calls": search_calls, "final_calls": final_calls,
            "elapsed_seconds": time.monotonic() - start, **ZERO}
    except ProgramBudgetExhausted as exc:
        result = {"schema": "warehouse-v623-refinement-result.v1", "job": spec, "stage": stage,
            "status": "scientific_budget_exhausted", "plan_sha256": file_sha(plan_path),
            "failure": exc.report, "gate": {"passed": False},
            "elapsed_seconds": time.monotonic() - start, **ZERO}
    except BaseException as exc:
        write_new(folder / "failed.json", {"error_type": type(exc).__name__,
            "error": str(exc), "scientific_retry": False, **ZERO})
        raise
    checked_plan(stage)
    write_new(folder / "result.json", result)
    write_new(folder / "completed.json", {"plan_sha256": file_sha(plan_path),
        "result_sha256": file_sha(folder / "result.json"),
        "passed": result["gate"]["passed"]})
    print(json.dumps({"event": "job_complete", "stage": stage, "job": job_id,
        "status": result["status"], "passed": result["gate"]["passed"],
        "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def finalize(stage):
    path, plan = checked_plan(stage)
    rows = []
    for spec in plan["jobs"]:
        folder = path.parent / "jobs" / spec["job_id"]
        result = json.loads((folder / "result.json").read_bytes())
        completed = json.loads((folder / "completed.json").read_bytes())
        if (result["job"] != spec or result["plan_sha256"] != file_sha(path)
                or completed != {"plan_sha256": file_sha(path),
                    "result_sha256": file_sha(folder / "result.json"),
                    "passed": result["gate"]["passed"]}):
            raise RuntimeError("incomplete or unbound job")
        rows.append(result)
    passed = all(r["status"] == "completed" and r["gate"]["passed"] for r in rows)
    summary = {"schema": "warehouse-v623-refinement-summary.v1", "candidate_id": v.CANDIDATE,
        "stage": stage, "plan_sha256": file_sha(path), "source_sha256": plan["source_sha256"],
        "all_six_pass": passed, "next_stage": "fresh_confirmation" if passed else "S2" if stage == "S1" else "closed_failed_program_precheck",
        "role_folds": [{"job_id": r["job"]["job_id"], "status": r["status"],
            "passed": r["gate"]["passed"],
            "explanation_coverage": r.get("metrics", {}).get("validation", {}).get("explanation", {}).get("coverage"),
            "expected_feedback_coverage": r.get("metrics", {}).get("validation", {}).get("feedback", {}).get("expected_actionable_coverage"),
            "result": pointer(path.parent / "jobs" / r["job"]["job_id"] / "result.json")}
            for r in rows], **ZERO}
    write_new(path.parent / "summary.json", summary)
    return summary


def audit(stage):
    """Recompute routes, proofs, qualifications and gates without the fitter."""
    import numpy as np
    from scripts import audit_warehouse_v621_diagnostics as math
    from scripts.audit_warehouse_v622_capacity import context, encoded_data, section, independent_gate
    path, plan = checked_plan(stage)
    summary = json.loads((path.parent / "summary.json").read_bytes())
    actor, _ = old.checked_actor(ROOT)
    episodes, _ = old.checked_inputs(ROOT)
    checks = {}
    for spec in plan["jobs"]:
        folder = path.parent / "jobs" / spec["job_id"]
        result = json.loads((folder / "result.json").read_bytes())
        if result["status"] != "completed":
            checks[spec["job_id"]] = result["status"] == "scientific_budget_exhausted" and not result["gate"]["passed"]
            continue
        raw = json.loads(read_pinned(result["program"]["path"], result["program"]["sha256"]))
        structure = json.loads(read_pinned(result["structure"]["path"], result["structure"]["sha256"]))
        search = json.loads(read_pinned(result["search_report"]["path"], result["search_report"]["sha256"]))
        fold = plan["folds"][spec["fold"]]
        names = tuple(actor.roles[spec["role"]].feature_names)
        contexts = {}
        for split in v.SPLITS:
            rows = [row for case in fold["splits"][split]
                    for row in episodes[case["base"]][spec["role"]]]
            contexts[split] = context(raw, encoded_data(actor, spec["role"], rows), names)
            for purpose in ("feedback", "explanation"):
                actual = section(contexts[split], raw[purpose + "_leaves"], purpose)
                math.assert_equal(actual, result["metrics"][split][purpose],
                                  spec["job_id"] + "." + split + "." + purpose)
        tree, cap = math.structure(structure, names, enforce_limits=False)
        embedded, _ = math.structure(raw["structure"], names, enforce_limits=False)
        math.assert_equal(tree, embedded, spec["job_id"] + ".structure")
        for purpose, threshold in (("feedback", .93), ("explanation", .95)):
            expected = [i for i in range(cap["leaf_count"])
                if (contexts["fit"]["leaves"] == i).any()
                and math.eligible(section(contexts["eligibility"], [i], purpose), threshold)]
            math.assert_equal(raw[purpose + "_leaves"], expected,
                              spec["job_id"] + ".qualified." + purpose)
        if (raw["actor_role_sha256"] != plan["actor_role_sha256"][spec["role"]]
                or raw["report"]["fit_target_sha256"] != sha256(
                    np.ascontiguousarray(contexts["fit"]["p"], dtype="<f4").tobytes()).hexdigest()
                or raw["report"]["fit_budget"]["prior_leaf_optimization_calls"] != search["new_leaf_calls"]
                or raw["report"]["fit_budget"]["actual_total_leaf_optimization_calls"] > 20000
                or search["new_leaf_calls"] > 19968
                or raw["report"]["leaf_optimization_calls_outside_search"] != cap["leaf_count"]
                or not all(int(np.asarray(z, np.float32).argmax()) in contexts["fit"]["proof"][i]
                           for i, z in enumerate(raw["leaf_logits"]))):
            raise RuntimeError("independent v6.23 target, budget or path certificate failed")
        actual_validation = {purpose: section(contexts["validation"],
            raw[purpose + "_leaves"], purpose) for purpose in ("feedback", "explanation")}
        checks[spec["job_id"]] = independent_gate(actual_validation) is result["gate"]["passed"]
    expected = all(r["passed"] for r in summary["role_folds"])
    passed = (all(checks.values()) and summary["all_six_pass"] == expected
              and summary["plan_sha256"] == file_sha(path))
    receipt = {"schema": "warehouse-v623-refinement-audit.v1", "stage": stage,
        "source_sha256": plan["source_sha256"], "summary_sha256": file_sha(path.parent / "summary.json"),
        "checks": checks, "passed": passed, **ZERO}
    write_new(path.parent / "audit.json", receipt)
    if not passed:
        raise RuntimeError("independent recomputation failed")
    return receipt


def execute(max_workers):
    control = v.OUTPUT / "execution"
    control.mkdir(parents=True, exist_ok=False)
    journal = (control / "events.jsonl").open("x", buffering=1)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    def event(kind, **fields):
        row = {"event": kind, "at_utc": datetime.now(timezone.utc).isoformat(), **fields}
        journal.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    def run_stage(stage):
        path = prepare(stage)
        event("stage_started", stage=stage, source_sha256=v.source_sha())
        pending = [j["job_id"] for j in v.jobs()]
        active = {}
        last = time.monotonic()
        while pending or active:
            while pending and len(active) < max_workers:
                name = pending.pop(0)
                logfile = (path.parent / (name + ".log")).open("x")
                argv = [sys.executable, str(Path(__file__).resolve()), "--job", name, "--stage", stage]
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=logfile,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = (process, logfile, time.monotonic())
                event("job_started", stage=stage, job=name, pid=process.pid)
            for name, (process, logfile, begin) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    if time.monotonic() - begin > 6300:
                        process.kill(); process.wait(); logfile.close()
                        raise RuntimeError("hard timeout without scientific receipt: " + name)
                    continue
                logfile.close(); del active[name]
                event("job_finished", stage=stage, job=name, returncode=rc,
                      elapsed_seconds=time.monotonic() - begin)
                if rc or not (path.parent / "jobs" / name / "completed.json").is_file():
                    raise RuntimeError("job failed without completion: " + name)
            if time.monotonic() - last > 45:
                event("heartbeat", stage=stage, completed=6-len(pending)-len(active),
                      pending=len(pending), active=list(active))
                last = time.monotonic()
            if active:
                time.sleep(5)
        result = finalize(stage)
        audit(stage)
        event("stage_complete", stage=stage, all_six_pass=result["all_six_pass"],
              next_stage=result["next_stage"])
        return result
    try:
        first = run_stage("S1")
        last = first if first["all_six_pass"] else run_stage("S2")
        outcome = {"status": "complete", "candidate_id": v.CANDIDATE,
            "next_stage": last["next_stage"], "selected_method": last["stage"] if last["all_six_pass"] else None,
            **ZERO}
        write_new(control / "completed.json", outcome)
        event("execution_complete", **outcome)
    except BaseException as exc:
        write_new(control / "failed.json", {"error_type": type(exc).__name__,
            "error": str(exc), "automatic_retry": False, **ZERO})
        event("execution_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--stage", choices=("S1", "S2"))
    parser.add_argument("--max-workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.execute:
        execute(args.max_workers)
    elif args.stage and args.prepare:
        print(prepare(args.stage))
    elif args.stage and args.job:
        job(args.stage, args.job)
    elif args.stage and args.finalize:
        print(json.dumps(finalize(args.stage)))
    elif args.stage and args.audit:
        print(json.dumps(audit(args.stage)))
    else:
        parser.error("choose exactly one operation and a stage when required")


if __name__ == "__main__":
    main()
