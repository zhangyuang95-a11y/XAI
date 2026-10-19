#!/usr/bin/env python3
"""Single-use, source-bound diagnostics on already spent v6.20 episodes.

This entry cannot collect trajectories, update Actors, authorize training, or
publish. Infeasible compilation is a scientific result, not a crashed fit.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v621_protocol as v

LOGGER = logging.getLogger("warehouse.v621.diagnostics")
PLAN_SCHEMA = "warehouse-v621-diagnostics-plan.v1"
RESULT_SCHEMA = "warehouse-v621-diagnostics-job-result.v1"
ZERO_FIELDS = ("environment_resets", "actor_updates", "PPO_joint_steps")
FALSE_FIELDS = ("publishable", "admission_evidence", "fresh_validation_evidence")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    return {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "scikit-learn")}}


def process_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def read_pinned(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"bound diagnostic input changed: {path}")
    return raw


def write_new(path, value):
    def jsonable(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {key: jsonable(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [jsonable(child) for child in item]
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(jsonable(value), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def pointer(path):
    return {"path": str(Path(path).resolve(strict=True)), "sha256": file_sha(path)}


def development_gate(metrics):
    return v.gate_metrics(metrics)


def budget_audit(search, program_report, arm):
    """Count histories and real API calls, including final fixed-tree refits."""
    outside = program_report.get("leaf_optimization_calls_outside_search", 0)
    calls = search.get("leaf_optimization_calls", 0)
    runs = search.get("search_runs", [])
    if (type(calls) is not int or calls < 0 or type(outside) is not int or outside < 0
            or not isinstance(runs, list)
            or any(not isinstance(row, dict) or not isinstance(row.get("rounds"), list) for row in runs)):
        raise RuntimeError("missing or malformed actual search counters")
    rounds = [len(row["rounds"]) for row in runs]
    checks = {"total_leaf_api_at_most_20000": calls + outside <= 20000,
        "reserved_final_refits_at_most_16": outside <= 16,
        "search_leaf_api_at_most_19984": calls <= 19984,
        "global_rounds_at_most_60": sum(rounds) <= 60}
    if program_report:
        count = program_report.get("leaf_count")
        fits = program_report.get("constrained_leaf_fits", {})
        solver_calls = [item.get("enumerated_action_solves") for item in fits.values()]
        checks["all_final_leaf_refits_accounted"] = (type(count) is int and count >= 1
            and outside == count and set(fits) == {str(i) for i in range(count)})
        checks["actual_final_solver_count_matches_leaves"] = (
            all(type(value) is int and value >= 0 for value in solver_calls)
            and type(program_report.get("convex_solver_calls_outside_search")) is int
            and sum(solver_calls) == program_report.get("convex_solver_calls_outside_search"))
    else:
        checks["empty_final_report_only_for_infeasible_compiler"] = arm == "B"
    if arm in ("A", "C"):
        checks.update({"all_eight_initializations_accounted": len(runs) == 8,
            "global_counter_matches_histories": type(search.get("global_search_rounds")) is int
                and search["global_search_rounds"] == sum(rounds)})
    elif arm == "B":
        checks["fixed_structure_has_no_search"] = not runs and calls == 0
    else:
        raise ValueError("unknown diagnostics arm")
    return {"passed": all(checks.values()), "checks": checks,
        "rounds_per_initialization": rounds, "actual_total_rounds": sum(rounds),
        "search_leaf_api_calls": calls, "final_refit_leaf_api_calls": outside,
        "end_to_end_leaf_api_calls": calls + outside,
        "actual_search_convex_solver_calls": search.get("actual_convex_solver_calls", 0),
        "actual_final_convex_solver_calls": program_report.get("convex_solver_calls_outside_search", 0),
        "scope": "global_per_role_fit_including_final_fixed_structure_refits"}


def _output_directory():
    return (ROOT / v.OUTPUT_ROOT / v.BATCH_ID).resolve()


def prepare(output, expected_source):
    if v.source_closure_sha256(ROOT) != expected_source:
        raise RuntimeError("diagnostic source differs from requested freeze")
    output = Path(output).resolve()
    if output != _output_directory():
        raise RuntimeError("diagnostics require the configured exclusive output directory")
    actor, actor_path = v.checked_actor(ROOT)
    episodes, manifest = v.checked_inputs(ROOT)
    targets = v.target_identity(actor, episodes)
    recipe_sha = v.validate_recipe(ROOT)
    folds = v.grouped_folds()
    plan = {"schema": PLAN_SCHEMA, "candidate_id": v.CANDIDATE_ID,
        "batch_id": v.BATCH_ID, "output_directory": str(output),
        "identity": {"source_closure_sha256": expected_source,
            "runtime": runtime_identity(),
            "source_files_sha256": v.source_closure_files(ROOT),
            "script_sha256": file_sha(__file__), "recipe_sha256": recipe_sha,
            "actor_path": str(actor_path.resolve()), "actor_sha256": actor.artifact_sha256,
            "actor_role_sha256": v.ROLE_SHA256, "inputs": manifest,
            "full_nn_targets": targets},
        "folds": folds, "fold_assignment_sha256": v.digest(folds),
        "jobs": v.jobs(), "round_budget_contract": v.round_budget_contract(),
        "gates": v.GATES, **dict.fromkeys(ZERO_FIELDS, 0),
        **dict.fromkeys(FALSE_FIELDS, False), "scientific_retry_permitted": False,
        "purpose": "already_spent_v620_data_method_development_only"}
    if v.source_closure_sha256(ROOT) != expected_source:
        raise RuntimeError("diagnostic source changed during prepare")
    for path, expected in manifest["files_sha256"].items():
        read_pinned(path, expected)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "input-manifest.json", manifest)
    write_new(output / "plan.json", plan)
    write_new(output / "plan.claim.json", {"plan_sha256": file_sha(output / "plan.json"),
        "single_use": True, "publishable": False})
    return output / "plan.json"


def load_plan(plan_path):
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    claim = json.loads((path.parent / "plan.claim.json").read_bytes())
    if (plan.get("schema") != PLAN_SCHEMA or path.parent != _output_directory()
            or plan.get("output_directory") != str(path.parent)
            or plan.get("candidate_id") != v.CANDIDATE_ID or plan.get("batch_id") != v.BATCH_ID
            or plan.get("jobs") != v.jobs() or plan.get("folds") != v.grouped_folds()
            or plan.get("fold_assignment_sha256") != v.digest(plan.get("folds"))
            or plan.get("round_budget_contract") != v.round_budget_contract()
            or plan.get("gates") != v.GATES or claim.get("single_use") is not True
            or claim.get("plan_sha256") != file_sha(path)
            or plan.get("scientific_retry_permitted") is not False
            or any(type(plan.get(k)) is not int or plan[k] != 0 for k in ZERO_FIELDS)
            or any(plan.get(k) is not False for k in FALSE_FIELDS)):
        raise RuntimeError("diagnostic immutable parent plan changed")
    identity = plan["identity"]
    if (identity["source_closure_sha256"] != v.source_closure_sha256(ROOT)
            or identity["runtime"] != runtime_identity()
            or identity["source_files_sha256"] != v.source_closure_files(ROOT)
            or identity["script_sha256"] != file_sha(__file__)
            or identity["recipe_sha256"] != v.validate_recipe(ROOT)
            or identity["actor_sha256"] != v.ACTOR_SHA256
            or identity["actor_role_sha256"] != v.ROLE_SHA256
            or file_sha(identity["actor_path"]) != v.ACTOR_SHA256):
        raise RuntimeError("diagnostic source, recipe or Actor changed")
    for input_path, expected in identity["inputs"]["files_sha256"].items():
        read_pinned(input_path, expected)
    return path, plan


def _arrays(actor, role, rows):
    from backend.training.warehouse_v6_program import _sample_arrays, _masked_probabilities
    role_actor = actor.roles[role]
    x, masks = _sample_arrays(rows, "diagnostic", role_actor.feature_dim)
    probabilities = _masked_probabilities(role_actor.logits_encoded(x), masks)
    return x, masks, probabilities, [str(row["episode"]) for row in rows]


def execute_job(plan_path, job_id):
    from backend.training.warehouse_v6_program import (
        PATH_CONSTRAINED_METHOD, WarehouseV6Program, fit_program, program_structure_sha256)
    from backend.training.warehouse_v621_compile import compile_actor_structure, fixed_topology_action_oracle
    from backend.training.warehouse_v621_search import fit_structure
    from scripts.run_warehouse_v619_cached_comparison import actor_identity, program_metrics
    path, plan = load_plan(plan_path)
    job = next((row for row in plan["jobs"] if row["job_id"] == job_id), None)
    if job is None:
        raise ValueError("job is not in the frozen diagnostics plan")
    directory = path.parent / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=False)
    started = {"schema": "warehouse-v621-diagnostics-job-start.v1", "job": job,
        "plan_sha256": file_sha(path), **dict.fromkeys(ZERO_FIELDS, 0),
        **dict.fromkeys(FALSE_FIELDS, False)}
    write_new(directory / "started.json", started)
    begin = perf_counter()
    LOGGER.info("job_started job=%s", job_id)
    try:
        actor, actor_path = v.checked_actor(ROOT)
        episodes, manifest = v.checked_inputs(ROOT)
        if (manifest != plan["identity"]["inputs"]
                or v.target_identity(actor, episodes) != plan["identity"]["full_nn_targets"]):
            raise RuntimeError("spent inputs or complete NN targets changed")
        fold = plan["folds"][job["fold"]]
        parts = {name: [row for case in fold["splits"][name]
                        for row in episodes[case["base"]][job["role"]]] for name in v.SPLITS}
        role_actor = actor.roles[job["role"]]
        arrays = {name: _arrays(actor, job["role"], rows) for name, rows in parts.items()}
        load_plan(path)
        compiler = None
        search = {"search_runs": [], "leaf_optimization_calls": 0, "global_search_rounds": 0}
        if job["arm"] == "B":
            structure, compiler = compile_actor_structure(role_actor)
            original = compiler["original_structure"]
            oracle = {name: fixed_topology_action_oracle(original, *data,
                feature_names=role_actor.feature_names) for name, data in arrays.items()}
            write_new(directory / "original-topology-oracle.json", oracle)
        else:
            LOGGER.info("search_started job=%s fit_frames=%s", job_id, len(parts["fit"]))
            structure, search = fit_structure(job["arm"], *arrays["fit"],
                feature_names=role_actor.feature_names, actor_role_sha256=role_actor.role_sha256)
        if structure is None:
            if job["arm"] != "B" or compiler.get("feasible") is not False:
                raise RuntimeError("missing structure without typed compiler infeasibility")
            metrics, program_pointer, program_report = None, None, {}
            gate = {"passed": False, "checks": {"program_structure_feasible": False}}
            status = "structurally_infeasible"
        else:
            write_new(directory / "structure.json", structure)
            program = fit_program(job["role"], actor,
                *(parts[name] for name in v.SPLITS), method=PATH_CONSTRAINED_METHOD,
                fixed_structure=structure, fixed_structure_sha256=program_structure_sha256(structure))
            program_path = directory / "program.json"
            program.save(program_path)
            program_pointer = pointer(program_path)
            program = WarehouseV6Program.load(program_path, expected_sha256=program_pointer["sha256"],
                actor_role_sha256=v.ROLE_SHA256[job["role"]])
            program_report = program.report
            metrics = {name: program_metrics(program, actor, rows) for name, rows in parts.items()}
            gate = development_gate(metrics["validation"])
            status = "completed"
            write_new(directory / "fixed-topology-oracle.json", {
                name: fixed_topology_action_oracle(structure, *data,
                    feature_names=role_actor.feature_names) for name, data in arrays.items()})
        budget = budget_audit(search, program_report, job["arm"])
        write_new(directory / "search-report.json", search)
        write_new(directory / "budget-audit.json", budget)
        if compiler is not None:
            write_new(directory / "compiler-report.json", compiler)
        if not budget["passed"]:
            raise RuntimeError("actual end-to-end fit exceeded the registered diagnostics budget")
        _, after = load_plan(path)
        if after != plan or actor_identity(actor) != v.ROLE_SHA256 or file_sha(actor_path) != v.ACTOR_SHA256:
            raise RuntimeError("source, plan or frozen Actor changed during diagnostics")
        result = {**started, "schema": RESULT_SCHEMA, "status": status,
            "elapsed_seconds": perf_counter() - begin, "program": program_pointer,
            "structure": pointer(directory / "structure.json") if structure is not None else None,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "memory_scope": "whole_job_process_including_input_loading_and_full_nn_target_recomputation",
            "metrics": metrics, "development_gate": gate, "budget_audit": budget,
            "search_report": pointer(directory / "search-report.json"),
            "compiler_report": pointer(directory / "compiler-report.json") if compiler is not None else None,
            "physical_frames_per_split": {name: len(rows) for name, rows in parts.items()},
            "fold_assignment_sha256": fold["assignment_sha256"], "excluded_frames": 0,
            "actor_weights_unchanged": True,
            "leaf_count": structure["leaf_count"] if structure is not None else None,
            "scientific_retry_permitted": False}
        write_new(directory / "result.json", result)
        write_new(directory / "completed.json", {"result_sha256": file_sha(directory / "result.json"),
            "plan_sha256": file_sha(path), "status": status, "passed": gate["passed"]})
        LOGGER.info("job_completed job=%s status=%s gate=%s seconds=%.3f", job_id, status,
            gate["passed"], perf_counter() - begin)
        return result
    except BaseException as error:
        LOGGER.exception("job_failed job=%s", job_id)
        write_new(directory / "failed.json", {**started, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": perf_counter() - begin,
            "scientific_retry_permitted": False})
        raise


def summarize_results(results):
    expected = {row["job_id"]: row for row in v.jobs()}
    if (len(results) != 18 or {r["job"]["job_id"] for r in results} != set(expected)
            or any(r["job"] != expected[r["job"]["job_id"]] for r in results)
            or any(r.get("status") not in ("completed", "structurally_infeasible")
                   or (r["status"] == "structurally_infeasible" and r["job"]["arm"] != "B")
                   for r in results)
            or any(r.get("budget_audit", {}).get("passed") is not True for r in results)):
        raise RuntimeError("all eighteen valid diagnostics units must complete before route selection")
    arms = {}
    for arm in ("A", "B", "C"):
        rows = [row for row in results if row["job"]["arm"] == arm]
        feasible = [row for row in rows if row["status"] == "completed"]
        e = [row["metrics"]["validation"]["explanation"] for row in feasible]
        f = [row["metrics"]["validation"]["feedback"] for row in feasible]
        max_kl = max((r["kl"] for r in e + f if r["kl"] is not None), default=None)
        score = [min((r["coverage"] for r in e), default=0.),
            min((r["expected_actionable_coverage"] for r in f), default=0.),
            -max_kl if max_kl is not None else None,
            -sum(row["leaf_count"] for row in feasible)]
        arms[arm] = {"all_role_fold_gates_passed": len(feasible) == 6
                and all(row["development_gate"]["passed"] for row in rows),
            "structurally_infeasible_units": len(rows) - len(feasible),
            "minimum_explanation_coverage": score[0],
            "mean_explanation_coverage": float(np.mean([r["coverage"] for r in e])) if e else None,
            "minimum_expected_feedback_coverage": score[1], "maximum_qualified_kl": max_kl,
            "selection_score": score}
    eligible = [arm for arm in ("B", "C", "A") if arms[arm]["all_role_fold_gates_passed"]]
    selected = max(eligible, key=lambda arm: tuple(arms[arm]["selection_score"])) if eligible else None
    return {"arms": arms, "selected_arm": selected,
        "next_stage": "three_initializer_bc_development" if selected else "shared_structure_actor_implementation",
        "new_data_or_training_authorized": False, **dict.fromkeys(ZERO_FIELDS, 0),
        **dict.fromkeys(FALSE_FIELDS, False),
        "inference": "bounded_development_search_not_global_capacity_or_optimality_proof"}


def checked_program_result(result, job, fold):
    from backend.training.warehouse_v6_program import (
        PATH_CONSTRAINED_METHOD, WarehouseV6Program, program_structure_sha256)
    ref = result["program"]
    program = WarehouseV6Program.load(ref["path"], expected_sha256=ref["sha256"],
        actor_role_sha256=v.ROLE_SHA256[job["role"]])
    report = program.report
    structure = json.loads(read_pinned(result["structure"]["path"], result["structure"]["sha256"]))
    stored_structure = {"root_ref": program.root_ref, "split_weights": program.split_weights,
        "split_bias": program.split_bias, "left_refs": program.left_refs, "right_refs": program.right_refs,
        "leaf_count": len(program.leaf_logits)}
    expected_sha = program_structure_sha256(structure)
    if (program.role != job["role"] or len(program.leaf_logits) != result["leaf_count"]
            or report.get("fit_method") != PATH_CONSTRAINED_METHOD
            or report.get("fixed_structure_sha256") != expected_sha
            or report.get("program_structure_sha256") != expected_sha
            or program_structure_sha256(stored_structure) != expected_sha
            or report.get("split_episode_ids") != {name: sorted(
                f'spent-v620-base-{case["base"]}' for case in fold["splits"][name]) for name in v.SPLITS}
            or report.get("split_scenario_seed_ids") != {name: sorted(
                str(case["scenario_seed"]) for case in fold["splits"][name]) for name in v.SPLITS}
            or any(report["validation"][purpose] != result["metrics"]["validation"][purpose]
                   for purpose in ("feedback", "explanation"))):
        raise RuntimeError("diagnostic program, topology, split or validation metric binding changed")
    return report


def summarize(plan_path):
    path, plan = load_plan(plan_path)
    results, refs = [], {}
    for job in plan["jobs"]:
        directory = path.parent / "jobs" / job["job_id"]
        if (directory / "failed.json").exists():
            raise RuntimeError("a failed diagnostics unit cannot enter route selection")
        started = json.loads((directory / "started.json").read_bytes())
        completed = json.loads((directory / "completed.json").read_bytes())
        result = json.loads(read_pinned(directory / "result.json", completed["result_sha256"]))
        expected_started = {"schema": "warehouse-v621-diagnostics-job-start.v1", "job": job,
            "plan_sha256": file_sha(path), **dict.fromkeys(ZERO_FIELDS, 0),
            **dict.fromkeys(FALSE_FIELDS, False)}
        if (started != expected_started or result.get("schema") != RESULT_SCHEMA or result["job"] != job
                or result["plan_sha256"] != file_sha(path) or completed["plan_sha256"] != file_sha(path)
                or completed.get("status") != result.get("status")
                or completed.get("passed") is not result["development_gate"]["passed"]
                or result.get("actor_weights_unchanged") is not True
                or type(result.get("excluded_frames")) is not int or result["excluded_frames"] != 0
                or result.get("scientific_retry_permitted") is not False
                or any(result.get(k) != 0 or type(result.get(k)) is not int for k in ZERO_FIELDS)
                or any(result.get(k) is not False for k in FALSE_FIELDS)
                or result["status"] not in ("completed", "structurally_infeasible")
                or result["fold_assignment_sha256"] != plan["folds"][job["fold"]]["assignment_sha256"]):
            raise RuntimeError("diagnostic result differs from immutable parent/job")
        search = json.loads(read_pinned(result["search_report"]["path"], result["search_report"]["sha256"]))
        if result["status"] == "completed":
            report = checked_program_result(result, job, plan["folds"][job["fold"]])
            if result["development_gate"] != development_gate(result["metrics"]["validation"]):
                raise RuntimeError("diagnostic gate changed")
        else:
            if job["arm"] != "B" or result["program"] is not None or result["metrics"] is not None:
                raise RuntimeError("only compiler infeasibility can omit a program")
            compiler = json.loads(read_pinned(result["compiler_report"]["path"], result["compiler_report"]["sha256"]))
            if (compiler.get("feasible") is not False
                    or result["development_gate"] != {"passed": False,
                        "checks": {"program_structure_feasible": False}}):
                raise RuntimeError("compiler infeasibility receipt is inconsistent")
            report = {}
        if budget_audit(search, report, job["arm"]) != result["budget_audit"]:
            raise RuntimeError("diagnostic budget counters changed")
        results.append(result)
        refs[job["job_id"]] = pointer(directory / "result.json")
    summary = {"schema": "warehouse-v621-diagnostics-summary.v1", "candidate_id": v.CANDIDATE_ID,
        "plan_sha256": file_sha(path), "jobs": refs, **summarize_results(results)}
    write_new(path.parent / "summary.json", summary)
    return summary


def main(argv=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--job-id")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.plan or not args.output_dir or not args.expected_source_sha256:
            parser.error("prepare requires output-dir and expected-source-sha256")
        print(json.dumps({"plan": str(prepare(args.output_dir, args.expected_source_sha256)), "environment_resets": 0}))
    else:
        if not args.plan or args.output_dir or args.expected_source_sha256:
            parser.error("job/summary requires only immutable plan")
        result = execute_job(args.plan, args.job_id) if args.job_id else summarize(args.plan)
        print(json.dumps({"job": result.get("job"), "status": result.get("status"),
            "development_gate": result.get("development_gate"), "selected_arm": result.get("selected_arm"),
            "environment_resets": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
