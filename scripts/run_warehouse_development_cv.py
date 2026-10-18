#!/usr/bin/env python3
"""Source-locked, zero-rollout grouped CV on already spent Warehouse data."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v620_protocol as v
from scripts.run_warehouse_v619_cached_comparison import (
    ROLE_SHA256, array_digest, actor_identity, program_metrics, solver_counts,
)

PINNED_INPUTS = {
    "plan.json": "11f2b6519ae8bd437784f7197ac4d1d51d54272c32e55f54ff1998d980968037",
    "result.json": "3adb4f20dd757ff2776003ea620b3dc453764f032bc29704ac7f969883974d6a",
    "collection/fit.json": "4936463aa88c516ce7c7da64f812ec55898f09f7f1bebc40de1d51f7482bf8eb",
    "collection/calibration.json": "ff4c2f2d73a43dfba522e397a9f67f01369f275c0e57e08eedc308c5f8950298",
    "collection/eligibility.json": "95160c478a7f2ff0493184495dcf79d219a9f6d95bc19a54e3a7a21cfaeb9360",
    "collection/validation.json": "130909111a7fd0089c9a6df960d07a6bb1a1811921c6ef2880632e83463e5c67",
}
TRACE_SHA256 = {
    "fit": "9c04670333bf1399420cda8540205b4187f20d57bff36cfc22e7ab9c5d17b490",
    "calibration": "445495560732ca904450c7cdc176ee5c2d6df68390e5b4a6ee52e8e2787dcec4",
    "eligibility": "4d9577a265a5896cd9f95d1dff3b40b826eeb1a38a7a9490262e2744fb3297f3",
    "validation": "6cf1323d0b4c09da6471383fd821a592e3e2b0ceea723f1604f9622bb742c87c",
}
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
LOGGER = logging.getLogger("warehouse.development_cv")


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_pinned(path, expected):
    """Hash the exact bytes returned to the caller, avoiding a hash/read race."""
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"pinned development input changed: {path}")
    return raw


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def restore_episodes(traces, cases, *, actor_sha256=v.ACTOR_SHA256, role_shas=ROLE_SHA256):
    """Keep all physical frames and both roles together under their scene base."""
    by_scene = {row["scenario_seed"]: row for row in cases}
    if len(by_scene) != len(cases) or len({r["base"] for r in cases}) != len(cases):
        raise RuntimeError("duplicate scene/base group in development inputs")
    episodes, prior_states = {}, {}
    for row in traces:
        scene, frame = row["base_seed"], row["frame"]
        if scene not in by_scene or type(frame) is not int or frame < 0:
            raise RuntimeError("unregistered scene or invalid frame in development trace")
        case = by_scene[scene]
        base = case["base"]
        if row["actor_sha256"] != actor_sha256 or set(row["roles"]) != set(v.ROLES):
            raise RuntimeError("trace frozen Actor/role identity differs")
        for prefix in ("pre", "post"):
            if v.digest(row[f"{prefix}_state_public"]) != row[f"{prefix}_state_public_sha256"]:
                raise RuntimeError("trace public state hash differs")
        if base not in episodes:
            episodes[base] = {role: [] for role in v.ROLES}
        expected_frame = len(episodes[base][v.ROLES[0]])
        if frame != expected_frame or (frame and prior_states[base] != row["pre_state_public_sha256"]):
            raise RuntimeError("omitted, duplicated, reordered or disconnected physical frame")
        prior_states[base] = row["post_state_public_sha256"]
        for role in v.ROLES:
            recorded = row["roles"][role]
            obs = np.asarray(recorded["obs"], dtype=np.float32)
            mask = np.asarray(recorded["mask"], dtype=np.float32)
            target = np.asarray(recorded["nn_probabilities"], dtype=np.float32)
            action = recorded["sampled_action"]
            if (obs.ndim != 1 or not np.isfinite(obs).all()
                    or mask.shape != (5,) or np.any((mask != 0) & (mask != 1))
                    or not mask.any() or target.shape != (5,) or not np.isfinite(target).all()
                    or np.any(target < 0) or abs(float(target.sum())-1) > 1e-5
                    or np.any(target[mask == 0] != 0) or action not in ACTIONS
                    or mask[ACTIONS.index(action)] != 1
                    or recorded["actor_role_sha256"] != role_shas[role]
                    or type(recorded["rng_seed"]) is not int
                    or (case["sampling_mode"] == "argmax" and ACTIONS.index(action) != target.argmax())):
                raise RuntimeError("invalid physical mask, NN target, action or role identity")
            episodes[base][role].append({
                "obs": obs, "mask": mask, "episode": f"spent-v619-base-{base}",
                "base_seed": scene, "frame": frame, "episode_id": row["episode_id"],
                "sampled_action": action, "nn_probabilities": target,
                "rng_seed": recorded["rng_seed"], "actor_sha256": actor_sha256,
                "actor_role_sha256": role_shas[role],
                "pre_state_public": row["pre_state_public"],
                "pre_state_public_sha256": row["pre_state_public_sha256"],
                "post_state_public": row["post_state_public"],
                "post_state_public_sha256": row["post_state_public_sha256"],
            })
    if set(episodes) != {row["base"] for row in cases}:
        raise RuntimeError("development trace omitted an entire episode")
    return episodes


def checked_inputs():
    data_root = ROOT / v.DATA_ROOT
    pinned = {str((data_root/name).resolve()): expected for name, expected in PINNED_INPUTS.items()}
    blobs = {name: read_pinned(data_root/name, expected) for name, expected in PINNED_INPUTS.items()}
    receipts = {split: json.loads(blobs[f"collection/{split}.json"]) for split in v.SPLITS}
    case_parts = v.seed_plan()["splits"]
    all_cases, all_traces, episodes_manifest = [], [], []
    for split in v.SPLITS:
        receipt, cases = receipts[split], case_parts[split]
        trace = receipt["trace"]
        trace_path = (data_root / "collection/traces" / TRACE_SHA256[split] / f"{split}.jsonl").resolve()
        if (trace["sha256"] != TRACE_SHA256[split] or Path(trace["path"]).resolve() != trace_path
                or receipt["sampling_modes"] != cases or receipt.get("excluded_frames") != 0):
            raise RuntimeError("spent-data collection identity changed")
        raw = read_pinned(trace_path, TRACE_SHA256[split])
        pinned[str(trace_path)] = TRACE_SHA256[split]
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != trace["row_count"]:
            raise RuntimeError("spent-data physical frame count changed")
        authorization_counts = {item["episode_outcomes"][0]["seed"]:
                                item["episode_outcomes"][0]["frame_count"]
                                for item in receipt["authorizations"]}
        if (set(authorization_counts) != {r["scenario_seed"] for r in cases}
                or any(sum(row["base_seed"] == scene for row in rows) != count
                       for scene, count in authorization_counts.items())):
            raise RuntimeError("trace does not contain all authorized physical frames")
        all_cases += [dict(case, original_split=split) for case in cases]
        all_traces += rows
        episodes_manifest.append({"original_split": split, "episodes": len(cases),
            "physical_frames_per_role": len(rows), "excluded_frames": 0,
            "trace_sha256": TRACE_SHA256[split]})
    episodes = restore_episodes(all_traces, all_cases)
    return episodes, {"files_sha256": pinned, "source_episodes": episodes_manifest,
        "episode_counts": {str(base): len(rows[v.ROLES[0]]) for base, rows in sorted(episodes.items())},
        "physical_frames_per_role": len(all_traces), "excluded_frames": 0,
        "purpose": "already_spent_development_data_not_independent_admission"}


def checked_actor():
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    path = (ROOT / v.BASELINE_RUN / "final-bc.actor.npz").resolve()
    actor = FrozenWarehouseV6Actor.load(path, expected_sha256=v.ACTOR_SHA256,
                                       expected_role_sha256=ROLE_SHA256)
    if actor_identity(actor) != ROLE_SHA256:
        raise RuntimeError("frozen Actor roles changed")
    return actor, path


def target_identity(actor, episodes):
    from backend.training.warehouse_v6_program import _masked_probabilities
    ordered = [episodes[base] for base in sorted(episodes)]
    obs = np.asarray([[rows[role][i]["obs"] for role in v.ROLES]
                      for rows in ordered for i in range(len(rows[v.ROLES[0]]))], dtype=np.float32)
    mask = np.asarray([[rows[role][i]["mask"] for role in v.ROLES]
                       for rows in ordered for i in range(len(rows[v.ROLES[0]]))], dtype=np.float32)
    stored = np.asarray([[rows[role][i]["nn_probabilities"] for role in v.ROLES]
                        for rows in ordered for i in range(len(rows[v.ROLES[0]]))], dtype=np.float32)
    target = _masked_probabilities(actor.logits(obs, mask), mask)
    if not np.allclose(target, stored, atol=2e-6, rtol=2e-5):
        raise RuntimeError("recorded full NN distributions differ from frozen Actor")
    return {"recomputed_full_masked_nn_target_sha256": array_digest(target),
        "recorded_full_masked_nn_target_sha256": array_digest(stored),
        "full_observation_sha256": array_digest(obs), "physical_mask_sha256": array_digest(mask),
        "ordering": "ascending_base_then_physical_frame_then_role"}


def prepare(output_dir, expected_source):
    from backend.training import warehouse_v6_runner as runner
    if runner.source_closure_sha256() != expected_source:
        raise RuntimeError("CV prepare source differs from requested frozen closure")
    output = Path(output_dir).resolve()
    if output != (ROOT/v.OUTPUT_ROOT/v.CV_BATCH_ID).resolve():
        raise RuntimeError("corrected recipe requires new cv-02; cv-01 is closed read-only engineering evidence")
    actor, actor_path = checked_actor()
    episodes, manifest = checked_inputs()
    targets = target_identity(actor, episodes)
    recipe_sha = v.validate_recipe(ROOT)
    identity = {"source_closure_sha256": expected_source, "script_sha256": file_sha(__file__),
        "recipe_sha256": recipe_sha, "actor_path": str(actor_path), "actor_sha256": v.ACTOR_SHA256,
        "actor_role_sha256": ROLE_SHA256, "inputs": manifest, "full_nn_targets": targets}
    plan = {"schema": v.CV_PLAN_SCHEMA, "candidate_id": v.CANDIDATE_ID,
        "batch_id": v.CV_BATCH_ID, "round_budget_contract": v.round_budget_contract(),
        "output_directory": str(output), "identity": identity, "folds": v.grouped_folds(),
        "fold_assignment_sha256": v.digest(v.grouped_folds()), "jobs": v.jobs(),
        "gates": v.GATES, "selection_arms": ["M1", "M2"],
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False}
    # No directory exists until every source/input/target precondition succeeds.
    if runner.source_closure_sha256() != expected_source:
        raise RuntimeError("CV source changed during prepare")
    for path, expected in manifest["files_sha256"].items():
        read_pinned(path, expected)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"input-manifest.json", manifest)
    write_new(output/"fold-contract.json", {"folds": plan["folds"],
        "sha256": plan["fold_assignment_sha256"], "frame_selection": "all_physical_frames"})
    write_new(output/"plan.json", plan)
    return output/"plan.json"


def load_plan(plan_path):
    from backend.training import warehouse_v6_runner as runner
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(path.read_bytes())
    directory = path.parent
    if (plan.get("schema") != v.CV_PLAN_SCHEMA
            or directory != (ROOT/v.OUTPUT_ROOT/v.CV_BATCH_ID).resolve()
            or plan.get("batch_id") != v.CV_BATCH_ID
            or plan.get("round_budget_contract") != v.round_budget_contract()
            or plan.get("output_directory") != str(directory)
            or plan.get("candidate_id") != v.CANDIDATE_ID
            or plan.get("folds") != v.grouped_folds()
            or plan.get("fold_assignment_sha256") != v.digest(v.grouped_folds())
            or plan.get("jobs") != v.jobs() or plan.get("gates") != v.GATES
            or plan.get("selection_arms") != ["M1", "M2"]
            or any(plan.get(k) is not False for k in ("publishable", "admission_evidence", "fresh_validation_evidence"))
            or any(type(plan.get(k)) is not int or plan[k] != 0 for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))):
        raise RuntimeError("CV immutable parent plan changed")
    identity = plan["identity"]
    if (runner.source_closure_sha256() != identity["source_closure_sha256"]
            or file_sha(__file__) != identity["script_sha256"]
            or v.validate_recipe(ROOT) != identity["recipe_sha256"]
            or identity["actor_sha256"] != v.ACTOR_SHA256
            or identity["actor_role_sha256"] != ROLE_SHA256
            or file_sha(identity["actor_path"]) != v.ACTOR_SHA256):
        raise RuntimeError("CV source, recipe or frozen Actor changed")
    for input_path, expected in identity["inputs"]["files_sha256"].items():
        read_pinned(input_path, expected)
    return path, plan


def development_gate(metrics):
    feedback, explanation = metrics["feedback"], metrics["explanation"]
    checks = {"explanation_coverage": explanation["coverage"] >= .40,
        "expected_feedback_coverage": feedback["expected_actionable_coverage"] >= .35,
        "explanation_agreement": explanation.get("agreement") is not None and explanation["agreement"] >= .97,
        "feedback_kl": feedback.get("kl") is not None and feedback["kl"] <= .20,
        "explanation_kl": explanation.get("kl") is not None and explanation["kl"] <= .20,
        "whole_tree_raw_legal": feedback["whole_tree_raw_illegal_count"] == 0,
        "whole_tree_path_certified": feedback["whole_tree_path_uncertified_count"] == 0}
    return {"passed": all(checks.values()), "checks": checks}


def search_budget_audit(report, arm):
    """Independently count actual rounds; a config name cannot prove a budget."""
    if arm not in v.METHODS:
        raise ValueError("unknown comparison arm")
    search = report.get("fit_tree", {})
    runs = search.get("search_runs", [])
    if not isinstance(runs, list) or any(not isinstance(run, dict) or not isinstance(run.get("rounds"), list) for run in runs):
        raise RuntimeError("search round accounting is missing or malformed")
    contract = v.round_budget_contract()[arm]
    per_start = [len(run["rounds"]) for run in runs]
    calls = search.get("leaf_optimization_calls")
    outside = report.get("leaf_optimization_calls_outside_search", 0)
    if type(calls) is not int or type(outside) is not int or calls < 0 or outside < 0:
        raise RuntimeError("search leaf API accounting is missing or malformed")
    checks = {"all_eight_initializations_accounted": len(runs) == contract["maximum_initializations"],
        "total_round_limit": sum(per_start) <= contract["maximum_total_rounds"],
        "total_leaf_api_limit": calls+outside <= contract["maximum_leaf_api_calls_per_role_fit"]}
    if arm == "M0":
        checks["historical_per_start_round_limit"] = all(value <= 30 for value in per_start)
    else:
        checks["declared_global_round_scope"] = search.get("round_budget_scope") == "global_per_role_fit_across_all_initializations"
        checks["global_round_counter_matches_histories"] = (type(search.get("global_search_rounds")) is int
            and search["global_search_rounds"] == sum(per_start))
        checks["global_remaining_counter_matches_histories"] = (type(search.get("global_search_rounds_remaining")) is int
            and search["global_search_rounds_remaining"] == 30-sum(per_start))
    return {"passed": all(checks.values()), "checks": checks,
        "round_budget_scope": contract["scope"], "rounds_per_initialization": per_start,
        "actual_total_rounds": sum(per_start), "end_to_end_leaf_api_calls": calls+outside,
        "contract": contract}


def execute_job(plan_path, job_id):
    from backend.training.warehouse_v6_program import fit_program, WarehouseV6Program
    path, plan = load_plan(plan_path)
    job = next((row for row in plan["jobs"] if row["job_id"] == job_id), None)
    if job is None:
        raise ValueError("job is not registered in the frozen CV parent")
    directory = path.parent / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=False)
    started = {"schema": "warehouse-development-cv-job-start.v1", "job": job,
        "plan_sha256": file_sha(path), "identity_sha256": v.digest(plan["identity"]),
        "publishable": False, "environment_resets": 0, "PPO_joint_steps": 0}
    write_new(directory/"started.json", started)
    LOGGER.info("job_started job=%s fold=%s role=%s method=%s", job_id, job["fold"], job["role"], job["method"])
    start = perf_counter()
    try:
        actor, actor_path = checked_actor()
        episodes, manifest = checked_inputs()
        targets = target_identity(actor, episodes)
        if manifest != plan["identity"]["inputs"] or targets != plan["identity"]["full_nn_targets"]:
            raise RuntimeError("CV input/full-target identity changed")
        fold = plan["folds"][job["fold"]]
        parts = {name: [row for case in fold["splits"][name]
                        for row in episodes[case["base"]][job["role"]]] for name in v.SPLITS}
        load_plan(path)
        LOGGER.info("fit_started job=%s frames=%s round_scope=%s", job_id,
                    {name: len(rows) for name, rows in parts.items()},
                    v.round_budget_contract()[job["arm"]]["scope"])
        # Every arm starts from fit data only, including M0. No historical tree
        # or earlier arm's output is supplied as a warm start.
        program = fit_program(job["role"], actor, *(parts[name] for name in v.SPLITS), method=job["method"])
        program_path = directory/"program.json"
        program.save(program_path)
        program_sha = file_sha(program_path)
        program = WarehouseV6Program.load(program_path, expected_sha256=program_sha,
                                         actor_role_sha256=ROLE_SHA256[job["role"]])
        budget = search_budget_audit(program.report, job["arm"])
        write_new(directory/"search-budget-audit.json", budget)
        LOGGER.info("fit_completed job=%s rounds=%s leaf_api=%s budget_passed=%s elapsed_seconds=%.3f",
                    job_id, budget["actual_total_rounds"], budget["end_to_end_leaf_api_calls"],
                    budget["passed"], perf_counter()-start)
        if not budget["passed"]:
            raise RuntimeError("fit exceeded the frozen role-level search budget; see search-budget-audit.json")
        metrics = {name: program_metrics(program, actor, rows) for name, rows in parts.items()}
        gate = development_gate(metrics["validation"])
        solver = solver_counts(program.report)
        search = program.report.get("fit_tree", {})
        solver.update({
            "end_to_end_leaf_api_calls": int(search.get("leaf_optimization_calls", 0))
                + int(program.report.get("leaf_optimization_calls_outside_search", 0)),
            "end_to_end_convex_solver_calls": int(search.get("actual_convex_solver_calls", 0))
                + int(program.report.get("convex_solver_calls_outside_search", 0)),
            "final_leaf_solves_are_search_subset": True,
            "solver_count_scope": "search_plus_outside_search_final_leaf_counts_are_subset",
        })
        _, after = load_plan(path)
        if actor_identity(actor) != ROLE_SHA256 or file_sha(actor_path) != v.ACTOR_SHA256 or after != plan:
            raise RuntimeError("CV source, Actor or plan changed during fitting")
        result = {**started, "schema": "warehouse-development-cv-job-result.v1",
            "program_path": str(program_path), "program_sha256": program_sha,
            "fold_assignment_sha256": fold["assignment_sha256"],
            "physical_frames_per_split": {name: len(rows) for name, rows in parts.items()},
            "excluded_frames": 0, "metrics": metrics, "development_gate": gate,
            "solver": solver, "search_budget_audit": budget, "leaf_count": len(program.leaf_logits),
            "elapsed_seconds": perf_counter()-start, "actor_weights_unchanged": True,
            "admission_evidence": False, "fresh_validation_evidence": False}
        write_new(directory/"result.json", result)
        write_new(directory/"completed.json", {"result_sha256": file_sha(directory/"result.json"),
            "program_sha256": program_sha, "plan_sha256": file_sha(path), "passed": gate["passed"]})
        LOGGER.info("job_completed job=%s development_gate=%s elapsed_seconds=%.3f", job_id,
                    gate["passed"], perf_counter()-start)
        return result
    except BaseException as error:
        LOGGER.exception("job_failed job=%s elapsed_seconds=%.3f", job_id, perf_counter()-start)
        write_new(directory/"failed.json", {**started, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": perf_counter()-start,
            "scientific_retry_permitted": False})
        raise


def summarize_results(results):
    if len(results) != 18 or {r["job"]["job_id"] for r in results} != {r["job_id"] for r in v.jobs()}:
        raise RuntimeError("all eighteen paired CV jobs must complete before method selection")
    if any(row.get("search_budget_audit", {}).get("passed") is not True for row in results):
        raise RuntimeError("all eighteen corrected-batch jobs must pass independent search budget accounting")
    arms = {}
    for arm in v.METHODS:
        rows = [r for r in results if r["job"]["arm"] == arm]
        explanation = [r["metrics"]["validation"]["explanation"] for r in rows]
        feedback = [r["metrics"]["validation"]["feedback"] for r in rows]
        score = [min(r["coverage"] for r in explanation),
                 min(r["expected_actionable_coverage"] for r in feedback),
                 -max(r["kl"] if r["kl"] is not None else float("inf") for r in explanation+feedback),
                 -sum(r["leaf_count"] for r in rows)]
        arms[arm] = {"all_role_fold_gates_passed": all(r["development_gate"]["passed"] for r in rows),
            "minimum_explanation_coverage": score[0],
            "mean_explanation_coverage": float(np.mean([r["coverage"] for r in explanation])),
            "mean_45_percent_target_reached": bool(np.mean([r["coverage"] for r in explanation]) >= .45),
            "minimum_expected_feedback_coverage": score[1],
            "maximum_qualified_kl": -score[2] if np.isfinite(score[2]) else None,
            "total_leaf_count": -score[3],
            "selection_score": [x if np.isfinite(x) else None for x in score]}
    eligible = [arm for arm in ("M1", "M2") if arms[arm]["all_role_fold_gates_passed"]]
    chosen = max(eligible, key=lambda arm: tuple(arms[arm]["selection_score"])) if eligible else None
    return {"arms": arms, "chosen_method_arm": chosen,
        "chosen_method": v.METHODS[chosen] if chosen else None,
        "next_stage": "lock_method_before_new_seed_clearance" if chosen else "common_bc_structure_learning_before_fresh_validation",
        "publishable": False, "admission_evidence": False, "fresh_validation_evidence": False,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0}


def summarize(plan_path):
    path, plan = load_plan(plan_path)
    results, references = [], {}
    for job in plan["jobs"]:
        directory = path.parent/"jobs"/job["job_id"]
        completed = json.loads((directory/"completed.json").read_bytes())
        result_path = directory/"result.json"
        raw = read_pinned(result_path, completed["result_sha256"])
        result = json.loads(raw)
        if (result["job"] != job or result["plan_sha256"] != file_sha(path)
                or completed["plan_sha256"] != file_sha(path)
                or result["identity_sha256"] != v.digest(plan["identity"])
                or file_sha(result["program_path"]) != completed["program_sha256"]
                or result["program_sha256"] != completed["program_sha256"]
                or result["development_gate"] != development_gate(result["metrics"]["validation"])):
            raise RuntimeError("CV result does not match its frozen parent/job/program")
        results.append(result)
        references[job["job_id"]] = {"path": str(result_path), "sha256": completed["result_sha256"]}
    result = {"schema": "warehouse-development-cv-summary.v1", "candidate_id": v.CANDIDATE_ID,
        "plan_sha256": file_sha(path), "job_results": references, **summarize_results(results)}
    write_new(path.parent/"summary.json", result)
    return result


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout, force=True)
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
        if not args.output_dir or not args.expected_source_sha256 or args.plan:
            parser.error("--prepare requires --output-dir and --expected-source-sha256")
        print(prepare(args.output_dir, args.expected_source_sha256), flush=True)
        return 0
    if not args.plan or args.output_dir or args.expected_source_sha256:
        parser.error("job/summary requires only --plan with its selected mode")
    if args.job_id:
        result = execute_job(args.plan, args.job_id)
        print(json.dumps({"job": result["job"], "development_gate": result["development_gate"]}), flush=True)
        return 0
    result = summarize(args.plan)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
