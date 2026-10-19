#!/usr/bin/env python3
"""Single-use capacity comparison on spent episodes; never collects or trains.

Every job is bound to source bytes, the original complete dual NN, all physical
frames and the exact old whole-episode folds. Fallback requires the complete,
independently audited failure of the primary batch.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from time import perf_counter, monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_protocol as v
from scripts.run_warehouse_v621_diagnostics import (
    file_sha, runtime_identity, process_peak_rss_bytes, write_new, pointer, read_pinned,
)

LOGGER = logging.getLogger("warehouse.v622.capacity")
PLAN_SCHEMA = "warehouse-v622-capacity-plan.v1"
ZERO = {"environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "pilot_opened": False, "fresh_validation_evidence": False}


def output_directory(stage):
    return ROOT/v.OUTPUT_ROOT/(v.BATCH_ID if stage == "primary" else "fallback-01")


def read_json_bound(ref):
    return json.loads(read_pinned(ref["path"], ref["sha256"]))


def baseline_binding():
    summary_path, audit_path = ROOT/v.BASELINE_SUMMARY, ROOT/v.BASELINE_AUDIT
    summary = json.loads(read_pinned(summary_path, v.BASELINE_SUMMARY_SHA256))
    audit = json.loads(read_pinned(audit_path, v.BASELINE_AUDIT_SHA256))
    if summary.get("selected_arm") is not None or audit.get("passed") is not True:
        raise RuntimeError("A16 reference must remain the independently audited failed batch")
    return {"summary": pointer(summary_path), "independent_audit": pointer(audit_path),
            "A16": summary["arms"]["A"], "refitted": False}


def prepare(stage, expected_source):
    if v.source_closure_sha256(ROOT) != expected_source:
        raise RuntimeError("candidate source differs from requested freeze")
    parent = None
    if stage == "fallback":
        primary = output_directory("primary")
        summary = json.loads((primary/"summary.json").read_bytes())
        audit = json.loads((primary/"independent-audit.json").read_bytes())
        if (summary.get("selected_arm") is not None or summary.get("next_stage") != "fallback"
                or audit.get("passed") is not True
                or audit.get("summary_sha256") != file_sha(primary/"summary.json")
                or audit.get("source_closure_sha256") != expected_source):
            raise RuntimeError("fallback requires a complete independently audited primary failure")
        parent = {"summary": pointer(primary/"summary.json"),
                  "independent_audit": pointer(primary/"independent-audit.json")}
    actor, actor_path = v.checked_actor(ROOT)
    episodes, manifest = v.checked_inputs(ROOT)
    plan = {"schema": PLAN_SCHEMA, "candidate_id": v.CANDIDATE_ID, "stage": stage,
        "identity": {"source_closure_sha256": expected_source,
            "source_files_sha256": v.source_closure_files(ROOT), "runtime": runtime_identity(),
            "recipe_sha256": v.validate_recipe(ROOT), "actor": pointer(actor_path),
            "actor_role_sha256": v.ROLE_SHA256, "inputs": manifest,
            "full_nn_targets": v.target_identity(actor, episodes)},
        "folds": v.grouped_folds(), "jobs": v.jobs(stage), "gates": v.GATES,
        "limits": v.LIMITS, "budgets": v.round_budget_contract(),
        "baseline_A16": baseline_binding(), "fallback_parent": parent,
        "scientific_retry_permitted": False, **ZERO}
    if v.source_closure_sha256(ROOT) != expected_source:
        raise RuntimeError("candidate source changed during prepare")
    folder = output_directory(stage)
    folder.mkdir(parents=True, exist_ok=False)
    source_root = folder/"source"/expected_source
    for relative, expected in plan["identity"]["source_files_sha256"].items():
        destination = source_root/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = read_pinned(ROOT/relative, expected)
        with destination.open("xb") as handle:
            handle.write(raw)
        destination.chmod(0o444)
    write_new(folder/"source-manifest.json", {
        "source_closure_sha256": expected_source,
        "files_sha256": plan["identity"]["source_files_sha256"],
        "path": str(source_root.resolve())})
    write_new(folder/"plan.json", plan)
    write_new(folder/"plan.claim.json", {"sha256": file_sha(folder/"plan.json"), "single_use": True})
    return folder/"plan.json"


def load_plan(path):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    stage = plan.get("stage")
    if stage not in ("primary", "fallback"):
        raise RuntimeError("unregistered capacity stage")
    claim = json.loads((path.parent/"plan.claim.json").read_bytes())
    if (path.parent != output_directory(stage).resolve() or plan.get("schema") != PLAN_SCHEMA
            or plan.get("candidate_id") != v.CANDIDATE_ID or plan.get("jobs") != v.jobs(stage)
            or plan.get("folds") != v.grouped_folds() or plan.get("gates") != v.GATES
            or plan.get("limits") != v.LIMITS or plan.get("budgets") != v.round_budget_contract()
            or claim != {"sha256": file_sha(path), "single_use": True}
            or any(type(plan.get(k)) is not type(value) or plan[k] != value for k, value in ZERO.items())
            or plan.get("scientific_retry_permitted") is not False):
        raise RuntimeError("immutable capacity plan changed")
    identity = plan["identity"]
    if (identity["source_closure_sha256"] != v.source_closure_sha256(ROOT)
            or identity["source_files_sha256"] != v.source_closure_files(ROOT)
            or identity["runtime"] != runtime_identity()
            or identity["recipe_sha256"] != v.validate_recipe(ROOT)
            or identity["actor"]["sha256"] != v.ACTOR_SHA256
            or identity["actor_role_sha256"] != v.ROLE_SHA256
            or file_sha(identity["actor"]["path"]) != v.ACTOR_SHA256
            or plan["baseline_A16"] != baseline_binding()):
        raise RuntimeError("source, recipe, runtime, Actor or baseline changed")
    for file, expected in identity["inputs"]["files_sha256"].items():
        read_pinned(file, expected)
    if stage == "fallback":
        parent = plan["fallback_parent"]
        summary, audit = read_json_bound(parent["summary"]), read_json_bound(parent["independent_audit"])
        if (summary.get("selected_arm") is not None or audit.get("passed") is not True
                or audit.get("summary_sha256") != parent["summary"]["sha256"]):
            raise RuntimeError("fallback parent authority changed")
    return path, plan


def arrays(actor, role, rows):
    from backend.training.warehouse_v6_program import _sample_arrays, _masked_probabilities
    frozen = actor.roles[role]
    x, masks = _sample_arrays(rows, "v622 capacity", frozen.feature_dim)
    return x, masks, _masked_probabilities(frozen.logits_encoded(x), masks), [str(r["episode"]) for r in rows]


def budget_audit(search, report, arm):
    calls = search.get("leaf_optimization_calls", 0)
    outside = report.get("leaf_optimization_calls_outside_search")
    if any(type(n) is not int or n < 0 for n in (calls, outside)):
        raise RuntimeError("missing actual leaf API counters")
    if arm == "B32":
        checks = {"fixed_structure_no_search": calls == 0,
                  "leaf_fits_at_most_32": outside <= 32}
    elif arm == "A32":
        checks = {"search_and_final_at_most_20000": calls+outside <= 20000,
                  "reserved_final_at_most_32": outside <= 32,
                  "search_at_most_19968": calls <= 19968,
                  "global_rounds_at_most_60": search.get("global_search_rounds", 61) <= 60}
    elif arm == "D32":
        table = search.get("frozen_table_leaf_optimization_calls", 0)
        generation = search.get("template_generation_leaf_optimization_calls", 0)
        checks = {"generation_budget": type(generation) is int and generation <= 20000,
                  "table_and_final_budget": type(table) is int and table+outside <= 540,
                  "scoped_certificate": search.get("certificate", {}).get("global_function_class_optimality_gap", "missing") is None,
                  "final_at_most_32": outside <= 32}
    else:
        raise RuntimeError("unknown arm")
    checks["all_final_leaves_fitted_or_frozen_table_preserved"] = (
        outside == report.get("leaf_count") if arm != "D32" else outside == 0)
    checks["whole_complexity_checked"] = report.get("leaf_count", 33) <= 32
    return {"passed": all(checks.values()), "checks": checks,
            "search_leaf_api_calls": calls, "final_refit_leaf_api_calls": outside,
            "end_to_end_leaf_api_calls": calls+outside}


def execute_job(plan_path, job_id):
    from backend.training.warehouse_v622_program import (
        compile_actor_structure, fit_program, program_metrics, program_structure_sha256,
        ProgramBudgetExhausted,
    )
    from backend.training.warehouse_v622_search import fit_structure, SearchConfig, SearchIncompleteError
    path, plan = load_plan(plan_path)
    job = next((j for j in plan["jobs"] if j["job_id"] == job_id), None)
    if job is None:
        raise ValueError("job absent from registered plan")
    folder = path.parent/"jobs"/job_id
    folder.mkdir(parents=True, exist_ok=False)
    started = {"schema": "warehouse-v622-capacity-start.v1", "job": job,
               "plan_sha256": file_sha(path), **ZERO}
    write_new(folder/"started.json", started)
    begin = perf_counter()
    LOGGER.info("job_started %s", job_id)
    try:
        actor, actor_path = v.checked_actor(ROOT)
        episodes, manifest = v.checked_inputs(ROOT)
        if (manifest != plan["identity"]["inputs"]
                or v.target_identity(actor, episodes) != plan["identity"]["full_nn_targets"]):
            raise RuntimeError("complete NN targets or source frames changed")
        fold = plan["folds"][job["fold"]]
        parts = {split: [row for case in fold["splits"][split]
                        for row in episodes[case["base"]][job["role"]]] for split in v.SPLITS}
        role_actor = actor.roles[job["role"]]
        compiler = None
        search = {"leaf_optimization_calls": 0, "global_search_rounds": 0}
        fitting_started = monotonic()
        limits = v.round_budget_contract()[job["arm"]]
        fitting_deadline = fitting_started+limits["wall_seconds"]
        if job["arm"] == "B32":
            structure, compiler = compile_actor_structure(role_actor)
            if structure is None:
                raise RuntimeError("registered v620 B32 topology unexpectedly infeasible")
            search["report_sha256"] = v.digest(search)
        else:
            structure, search = fit_structure(job["arm"], *arrays(actor, job["role"], parts["fit"]),
                feature_names=role_actor.feature_names, actor_role_sha256=role_actor.role_sha256,
                config=SearchConfig(wall_time_seconds=limits["wall_seconds"]-limits["reserved_final_wall_seconds"]))
        write_new(folder/"structure.json", structure)
        write_new(folder/"search-report.json", search)
        if compiler is not None:
            write_new(folder/"compiler-report.json", compiler)
        LOGGER.info("structure_ready %s leaves=%s", job_id, structure["leaf_count"])
        program = fit_program(job["role"], actor, *(parts[s] for s in v.SPLITS),
            structure=structure, structure_sha256=program_structure_sha256(structure),
            fit_method=v.METHODS[job["arm"]],
            fit_budget={"deadline_monotonic": fitting_deadline,
                "maximum_leaf_api_calls": 0 if job["arm"] == "D32" else 32,
                "prior_leaf_optimization_calls": search["leaf_optimization_calls"],
                "max_total_leaf_optimization_calls": 20540 if job["arm"] == "D32" else 20000 if job["arm"] == "A32" else 32,
                "provenance_sha256": search["report_sha256"]})
        metrics = {s: program_metrics(program, actor, rows) for s, rows in parts.items()}
        fitting_elapsed = monotonic()-fitting_started
        if monotonic() >= fitting_deadline:
            report = dict(program.report, status="budget_exhausted", scientific_failure=True,
                          failure_stage="after_native_metrics")
            raise ProgramBudgetExhausted("native metrics exceeded registered role/fold wall budget", report=report)
        budget = budget_audit(search, program.report, job["arm"])
        if not budget["passed"]:
            raise RuntimeError("registered capacity fitting budget exceeded")
        program.save(folder/"program.json")
        load_plan(path)
        from backend.training.warehouse_v6_actor import role_sha256
        actual_roles = {role: role_sha256(actor.roles[role].weights,
            model_version=actor.roles[role].model_version) for role in v.ROLES}
        if file_sha(actor_path) != v.ACTOR_SHA256 or actual_roles != v.ROLE_SHA256:
            raise RuntimeError("frozen Actor changed")
        result = {**started, "schema": "warehouse-v622-capacity-result.v1", "status": "completed",
            "elapsed_seconds": perf_counter()-begin, "program": pointer(folder/"program.json"),
            "fitting_wall_seconds": fitting_elapsed,
            "fitting_wall_limit_seconds": limits["wall_seconds"],
            "structure": pointer(folder/"structure.json"), "search_report": pointer(folder/"search-report.json"),
            "compiler_report": pointer(folder/"compiler-report.json") if compiler else None,
            "metrics": metrics, "development_gate": v.gate_metrics(metrics["validation"]),
            "budget_audit": budget, "leaf_count": len(program.leaf_logits),
            "physical_frames_per_split": {s: len(rows) for s, rows in parts.items()},
            "excluded_frames": 0, "actor_weights_unchanged": True,
            "process_peak_rss_bytes": process_peak_rss_bytes()}
        write_new(folder/"result.json", result)
        write_new(folder/"completed.json", {"result_sha256": file_sha(folder/"result.json"),
            "plan_sha256": file_sha(path), "passed": result["development_gate"]["passed"]})
        LOGGER.info("job_completed %s passed=%s seconds=%.2f", job_id,
                    result["development_gate"]["passed"], result["elapsed_seconds"])
        return result
    except (ProgramBudgetExhausted, SearchIncompleteError) as error:
        # An exhausted finite scientific budget is a failed role/fold, not an
        # excuse to change limits, reuse validation, or claim a half-fit tree.
        receipt = {**started, "schema": "warehouse-v622-capacity-result.v1",
            "status": "scientific_budget_exhausted", "error_type": type(error).__name__,
            "error": str(error), "budget_exhaustion": v.plain(getattr(error, "report", {})),
            "elapsed_seconds": perf_counter()-begin, "program": None, "metrics": None,
            "fitting_wall_seconds": monotonic()-fitting_started,
            "fitting_wall_limit_seconds": limits["wall_seconds"],
            "fitting_deadline_monotonic": fitting_deadline,
            "structure": pointer(folder/"structure.json") if (folder/"structure.json").is_file() else None,
            "search_report": pointer(folder/"search-report.json") if (folder/"search-report.json").is_file() else None,
            "leaf_count": 0, "development_gate": {"passed": False, "reason": "no_complete_certified_program_within_budget"},
            "budget_audit": {"passed": True, "checks": {"exhaustion_is_not_admission": True}},
            "excluded_frames": 0, "actor_weights_unchanged": True}
        load_plan(path)
        write_new(folder/"result.json", receipt)
        write_new(folder/"completed.json", {"result_sha256": file_sha(folder/"result.json"),
            "plan_sha256": file_sha(path), "passed": False})
        LOGGER.info("job_budget_exhausted %s", job_id)
        return receipt
    except BaseException as error:
        write_new(folder/"failed.json", {**started, "error": str(error),
            "error_type": type(error).__name__, "elapsed_seconds": perf_counter()-begin,
            "attempted_fit_work": v.plain(getattr(error, "report", None))})
        raise


def summarize(plan_path):
    path, plan = load_plan(plan_path)
    results, refs = [], {}
    for job in plan["jobs"]:
        folder = path.parent/"jobs"/job["job_id"]
        if (folder/"failed.json").exists():
            raise RuntimeError("failed job cannot enter method selection")
        completion = json.loads((folder/"completed.json").read_bytes())
        ref = {"path": str(folder/"result.json"), "sha256": completion["result_sha256"]}
        result = read_json_bound(ref)
        if (result["job"] != job or result["plan_sha256"] != file_sha(path)
                or any(type(result.get(k)) is not type(value) or result[k] != value for k, value in ZERO.items())
                or result.get("actor_weights_unchanged") is not True or result.get("excluded_frames") != 0):
            raise RuntimeError("capacity result binding or zero-update declarations changed")
        results.append(result)
        refs[job["job_id"]] = ref
    summary = {"schema": "warehouse-v622-capacity-summary.v1", "candidate_id": v.CANDIDATE_ID,
        "stage": plan["stage"], "plan_sha256": file_sha(path), "jobs": refs,
        "baseline_A16": plan["baseline_A16"],
        **v.summarize_units(results, stage=plan["stage"])}
    write_new(path.parent/"summary.json", summary)
    return summary


def main(argv=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--job-id")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--stage", choices=("primary", "fallback"), default="primary")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.plan or not args.expected_source_sha256:
            parser.error("prepare requires expected-source-sha256 and no plan")
        result = {"plan": str(prepare(args.stage, args.expected_source_sha256))}
    else:
        if not args.plan or args.expected_source_sha256:
            parser.error("job/summary requires the immutable plan")
        result = execute_job(args.plan, args.job_id) if args.job_id else summarize(args.plan)
    print(json.dumps(v.plain({k: result.get(k) for k in (
        "plan", "job", "status", "development_gate", "selected_arm", "next_stage")}), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
