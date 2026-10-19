#!/usr/bin/env python3
"""Independent arithmetic audit of a complete v6.22 development batch.

No production program loader, metrics, qualification, fitter, environment or
optimizer is called. Reuses the prior independent auditor's scalar tree math,
the public feature encoder and the frozen NN forward pass. This is not a sealed
program audit and cannot turn spent development episodes into admission data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_protocol as v
from scripts import audit_warehouse_v621_diagnostics as math
from scripts.run_warehouse_v621_diagnostics import file_sha, write_new, read_pinned

READ_FILES = {}


def pinned(path, expected):
    path = str(Path(path).resolve(strict=True))
    if path in READ_FILES and READ_FILES[path] != expected:
        raise RuntimeError("independent audit: conflicting evidence bindings")
    raw = read_pinned(path, expected)
    READ_FILES[path] = expected
    return raw


def bound(ref):
    return json.loads(pinned(ref["path"], ref["sha256"]))


def context(raw, data, names):
    tree, capacity = math.structure(dict(raw["structure"], leaf_count=len(raw["leaf_logits"])), names,
                                    enforce_limits=False)
    if (capacity["depth"] > 6 or capacity["leaf_count"] > 32
            or capacity["maximum_source_features_per_node"] > 3):
        raise RuntimeError("independent audit: program exceeds registered v622 capacity")
    x, masks, p, sampled, episodes, scenes = data
    leaves = math.route(tree, x)
    logits = np.asarray(raw["leaf_logits"], np.float32)
    if logits.shape != (tree["leaf_count"], 5) or not np.isfinite(logits).all():
        raise RuntimeError("independent audit: invalid full five-action leaf distribution")
    proof = math.certificates(tree, names, explicit_crossing=True)
    raw_logits = logits[leaves]
    q = math.softmax(raw_logits, masks)
    raw_action, top, nn_top = raw_logits.argmax(-1), q.argmax(-1), p.argmax(-1)
    legal = masks[np.arange(len(x)), raw_action] > .5
    proven = np.asarray([int(a) in proof[int(leaf)] for a, leaf in zip(raw_action, leaves, strict=True)])
    kl = np.sum(p*(np.log(np.maximum(p, 1.e-8))-np.log(np.maximum(q, 1.e-8))), axis=-1)
    return {"tree": tree, "capacity": capacity, "proof": proof, "x": x, "m": masks,
        "p": p, "sampled": sampled, "episode": episodes, "scenario": scenes,
        "leaves": leaves, "q": q, "raw_action": raw_action, "top": top,
        "nn_top": nn_top, "raw_legal": legal, "proven": proven, "row_kl": kl, "names": tuple(names)}


def encoded_data(actor, role, rows):
    # Encoder and NN are common public/model infrastructure, not program math.
    from backend.training.warehouse_v6_program import _sample_arrays
    x, masks = _sample_arrays(rows, "independent v622 audit", actor.roles[role].feature_dim)
    p = math.softmax(actor.roles[role].logits_encoded(x), masks)
    sampled = np.asarray([math.ACTIONS.index(r["sampled_action"]) for r in rows])
    return x, masks, p, sampled, np.asarray([str(r["episode"]) for r in rows]), np.asarray([str(r["base_seed"]) for r in rows])


def check_action_streams(episodes):
    checks = 0
    for roles in episodes.values():
        for rows in roles.values():
            for row in rows:
                p = np.asarray(row["nn_probabilities"], np.float32)
                action = (int(p.argmax()) if row["sampling_mode"] == "argmax" else
                          int(np.random.default_rng(row["rng_seed"]).choice(5, p=p)))
                if math.ACTIONS[action] != row["sampled_action"]:
                    raise RuntimeError("independent audit: stored execution differs from bound NN draw")
                checks += 1
    return checks


def independent_gate(value):
    return math.gate(value)


def section(c, leaves, purpose):
    result = math.section(c, leaves, purpose)
    chosen = np.isin(c["leaves"], leaves)
    result["certified_rows"] = int((chosen & c["raw_legal"] & c["proven"]).sum())
    result["whole_tree_agreement"] = float((c["top"] == c["nn_top"]).mean())
    result["whole_tree_kl"] = float(c["row_kl"].mean())
    if not chosen.any():
        result["per_nn_action"] = {
            label: {"physical_frames": int((c["nn_top"] == i).sum()), "rows": 0,
                "qualified_coverage": 0. if (c["nn_top"] == i).any() else None,
                "agreement": None, "actual_actionable_rows": 0, "expected_actionable_rows": 0.}
            for i, label in enumerate(math.ACTIONS)}
        names, x = c["names"], c["x"]
        conflict = x[:, names.index("same_target_collision_streak_capped")] > 0
        near = x[:, names.index("robots_shortest_distance")]*42 <= 2
        low = x[:, names.index("self_charger_return_margin")]*50 <= 4
        result["public_strata"] = {
            label: {"rows": 0, "physical_frames": int(take.sum()),
                "qualified_coverage": 0. if take.any() else None, "agreement": None,
                "kl": None, "illegal_count": 0, "actual_actionable_rows": 0,
                "expected_actionable_rows": 0.}
            for label, take in {"ordinary": ~(conflict | near | low), "conflict": conflict,
                "near_robots": near, "low_return_margin": low}.items()}
    return result


def audit(plan_path):
    READ_FILES.clear()
    path = Path(plan_path).resolve(strict=True)
    plan_sha = file_sha(path)
    plan = json.loads(pinned(path, plan_sha))
    stage = plan.get("stage")
    if (plan.get("candidate_id") != v.CANDIDATE_ID or stage not in ("primary", "fallback")
            or plan.get("jobs") != v.jobs(stage) or plan.get("folds") != v.grouped_folds()
            or plan.get("gates") != v.GATES or plan.get("limits") != v.LIMITS
            or json.loads(pinned(path.parent/"plan.claim.json", file_sha(path.parent/"plan.claim.json"))) != {"sha256": plan_sha, "single_use": True}):
        raise RuntimeError("independent audit: parent binding differs")
    identity = plan["identity"]
    if (identity["source_closure_sha256"] != v.source_closure_sha256(ROOT)
            or identity["source_files_sha256"] != v.source_closure_files(ROOT)
            or identity["recipe_sha256"] != v.validate_recipe(ROOT)):
        raise RuntimeError("independent audit: source or recipe changed")
    summary_sha = file_sha(path.parent/"summary.json")
    summary = json.loads(pinned(path.parent/"summary.json", summary_sha))
    if summary.get("plan_sha256") != file_sha(path) or set(summary["jobs"]) != {j["job_id"] for j in plan["jobs"]}:
        raise RuntimeError("independent audit: incomplete batch summary")
    baseline = plan["baseline_A16"]
    if (baseline["summary"]["sha256"] != v.BASELINE_SUMMARY_SHA256
            or baseline["independent_audit"]["sha256"] != v.BASELINE_AUDIT_SHA256):
        raise RuntimeError("independent audit: old A16 source receipt changed")
    bound(baseline["summary"])
    if bound(baseline["independent_audit"]).get("passed") is not True:
        raise RuntimeError("independent audit: old A16 reference is not audited")
    if stage == "fallback":
        parent = plan["fallback_parent"]
        parent_summary, parent_audit = bound(parent["summary"]), bound(parent["independent_audit"])
        if (parent_summary.get("schema") != "warehouse-v622-capacity-summary.v1"
                or parent_summary.get("stage") != "primary" or parent_summary.get("candidate_id") != v.CANDIDATE_ID
                or parent_summary.get("selected_arm") is not None or parent_summary.get("next_stage") != "fallback"
                or set(parent_summary["jobs"]) != {j["job_id"] for j in v.jobs("primary")}
                or parent_audit.get("passed") is not True or parent_audit.get("stage") != "primary"
                or parent_audit.get("candidate_id") != v.CANDIDATE_ID or parent_audit.get("job_count") != 12
                or parent_audit.get("summary_sha256") != parent["summary"]["sha256"]
                or parent_audit.get("source_closure_sha256") != identity["source_closure_sha256"]
                or any(r.get("all_role_fold_gates_passed") is not False for r in parent_summary["arms"].values())):
            raise RuntimeError("independent audit: fallback primary parent is not a complete failed comparison")
        for ref in parent_summary["jobs"].values():
            bound(ref)
    actor, _ = v.checked_actor(ROOT)
    episodes, manifest = v.checked_inputs(ROOT)
    if manifest != identity["inputs"] or v.target_identity(actor, episodes) != identity["full_nn_targets"]:
        raise RuntimeError("independent audit: all-frame input/NN target mismatch")
    for file, expected in manifest["files_sha256"].items():
        pinned(file, expected)
    action_checks = check_action_streams(episodes)
    job_reports, recomputed_results = [], []
    for job in plan["jobs"]:
        folder = path.parent/"jobs"/job["job_id"]
        if (folder/"failed.json").exists():
            raise RuntimeError("independent audit: failed unit cannot enter selection")
        completion = json.loads(pinned(folder/"completed.json", file_sha(folder/"completed.json")))
        result = bound(summary["jobs"][job["job_id"]])
        started = json.loads(pinned(folder/"started.json", file_sha(folder/"started.json")))
        expected_started = {"schema": "warehouse-v622-capacity-start.v1", "job": job,
            "plan_sha256": plan_sha, "environment_resets": 0, "actor_updates": 0,
            "PPO_joint_steps": 0, "publishable": False, "pilot_opened": False,
            "fresh_validation_evidence": False}
        if started != expected_started or completion.get("plan_sha256") != plan_sha:
            raise RuntimeError("independent audit: start/completion plan binding differs")
        if (result["job"] != job or result["plan_sha256"] != file_sha(path)
                or completion["result_sha256"] != summary["jobs"][job["job_id"]]["sha256"]
                or result.get("status") not in ("completed", "scientific_budget_exhausted") or result.get("actor_weights_unchanged") is not True
                or result.get("excluded_frames") != 0
                or any(type(result.get(k)) is not int or result[k] != 0
                       for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))
                or result.get("publishable") is not False or result.get("pilot_opened") is not False):
            raise RuntimeError("independent audit: job identity/authority mismatch")
        if result["status"] == "scientific_budget_exhausted":
            if result.get("program") is not None or result.get("metrics") is not None or result["development_gate"]["passed"] is not False:
                raise RuntimeError("independent audit: exhausted role/fold claims admission")
            if completion.get("passed") is not False:
                raise RuntimeError("independent audit: exhausted completion claims a pass")
            failure = result.get("budget_exhaustion", {})
            wall = result.get("fitting_wall_seconds")
            limit = v.round_budget_contract()[job["arm"]]["wall_seconds"]
            if (type(wall) not in (int, float) or not np.isfinite(wall) or wall < 0
                    or result.get("fitting_wall_limit_seconds") != limit):
                raise RuntimeError("independent audit: incomplete job lacks actual wall accounting")
            if result.get("error_type") == "ProgramBudgetExhausted":
                if (wall < limit or failure.get("status") != "budget_exhausted"
                        or failure.get("scientific_failure") is not True
                        or not isinstance(failure.get("failure_stage"), str)
                        or failure.get("actor_role_sha256") != v.ROLE_SHA256[job["role"]]):
                    raise RuntimeError("independent audit: final-fit timeout not evidenced")
                calls, solves = failure.get("leaf_optimization_calls"), failure.get("convex_solver_calls")
                if (type(calls) is not int or not 0 <= calls <= 32 or type(solves) is not int
                        or not 0 <= solves <= 5*calls):
                    raise RuntimeError("independent audit: timeout actual fit counters invalid")
                fb = failure.get("fit_budget", {})
                if (fb.get("actual_final_leaf_optimization_calls") != calls
                        or fb.get("actual_total_leaf_optimization_calls") != fb.get("prior_leaf_optimization_calls", -1)+calls
                        or fb.get("deadline_monotonic") != result.get("fitting_deadline_monotonic")):
                    raise RuntimeError("independent audit: timeout fit ledger differs")
                if result.get("search_report") is None:
                    raise RuntimeError("independent audit: final timeout lost its completed search evidence")
                bound(result["search_report"])
            elif result.get("error_type") == "SearchIncompleteError":
                from experiments.warehouse.v622_capacity_audit_math import _attempts
                if failure.get("complete_incumbent") is not False:
                    raise RuntimeError("independent audit: search failure claimed a complete incumbent")
                if job["arm"] == "A32":
                    calls, _ = _attempts(failure, "incomplete_A32")
                    if not 1 <= calls <= 19968:
                        raise RuntimeError("independent audit: incomplete A32 has no bounded work")
                elif job["arm"] == "D32":
                    gc, _ = _attempts(failure.get("generation", {}), "incomplete_D32_generation")
                    tc, _ = _attempts(failure.get("terminal_table", {}), "incomplete_D32_table")
                    if not 1 <= gc <= 20000 or tc > 508:
                        raise RuntimeError("independent audit: incomplete D32 work exceeds budget")
                else:
                    raise RuntimeError("independent audit: fixed B32 cannot have a search failure")
            else:
                raise RuntimeError("independent audit: unregistered incomplete scientific error")
            job_reports.append({"job": job, "passed": True, "development_gate_passed": False,
                "status": "scientific_budget_exhausted"})
            recomputed_results.append(result)
            continue
        raw, search = bound(result["program"]), bound(result["search_report"])
        structure = bound(result["structure"])
        from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
        if (raw.get("version") != "warehouse-role-program.v6.22-depth6-leaves32"
                or raw.get("role") != job["role"] or raw.get("method") != v.METHODS[job["arm"]]
                or raw.get("actor_role_sha256") != v.ROLE_SHA256[job["role"]]
                or raw.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or raw.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256
                or raw.get("limits") != v.LIMITS):
            raise RuntimeError("independent audit: program identity/schema/limits differ")
        names = tuple(actor.roles[job["role"]].feature_names)
        fold = plan["folds"][job["fold"]]
        contexts = {}
        fit_arrays = None
        for split in v.SPLITS:
            rows = [r for case in fold["splits"][split] for r in episodes[case["base"]][job["role"]]]
            data = encoded_data(actor, job["role"], rows)
            if split == "fit":
                fit_arrays = (*data[:3], data[4], names, v.ROLE_SHA256[job["role"]])
            c = context(raw, data, names)
            contexts[split] = c
            for purpose in ("feedback", "explanation"):
                actual = section(c, raw[f"{purpose}_leaves"], purpose)
                math.assert_equal(actual, result["metrics"][split][purpose], f"{job['job_id']}.{split}.{purpose}")
            if c["capacity"]["leaf_count"] != result["leaf_count"]:
                raise RuntimeError("independent audit: leaf count differs")
        # Qualification must be rebuilt from eligibility episodes only.
        eligibility = contexts["eligibility"]
        for purpose, threshold in (("feedback", .93), ("explanation", .95)):
            expected = [leaf for leaf in range(len(raw["leaf_logits"]))
                        if (contexts["fit"]["leaves"] == leaf).any()
                        and math.eligible(section(eligibility, [leaf], purpose), threshold)]
            if sorted(raw[f"{purpose}_leaves"]) != expected:
                raise RuntimeError("independent audit: eligibility membership differs")
        tree, _ = math.structure(structure, names, enforce_limits=False)
        canonical, _ = math.structure(dict(raw["structure"], leaf_count=len(raw["leaf_logits"])), names, enforce_limits=False)
        math.assert_equal(tree, canonical, f"{job['job_id']}.structure")
        f = result["program"]
        if raw.get("actor_role_sha256") != v.ROLE_SHA256[job["role"]]:
            raise RuntimeError("independent audit: wrong bound role")
        proof = contexts["fit"]["proof"]
        for leaf, logits in enumerate(raw["leaf_logits"]):
            if int(np.argmax(np.asarray(logits, np.float32))) not in proof[leaf]:
                raise RuntimeError("independent audit: refitted raw action lacks path proof")
        passed = independent_gate(result["metrics"]["validation"])
        if result["development_gate"]["passed"] is not passed or result["budget_audit"]["passed"] is not True:
            raise RuntimeError("independent audit: gate/budget receipt differs")
        report = raw["report"]
        if (report.get("actor_role_sha256") != raw["actor_role_sha256"]
                or report.get("fit_method") != raw["method"]
                or report.get("feature_schema_sha256") != raw["feature_schema_sha256"]
                or report.get("action_physics_config_sha256") != raw["action_physics_config_sha256"]
                or report.get("limits") != v.LIMITS or report.get("full_NN_distribution_targets") is not True
                or report.get("external_program_selects_execution_action") is not False):
            raise RuntimeError("independent audit: program report lost full-policy binding")
        actual_fit_counts = np.bincount(contexts["fit"]["leaves"], minlength=len(raw["leaf_logits"])).tolist()
        math.assert_equal(report["fit_leaf_row_counts"], actual_fit_counts, f"{job['job_id']}.fit_counts")
        from hashlib import sha256
        if report.get("fit_target_sha256") != sha256(np.ascontiguousarray(contexts["fit"]["p"], dtype="<f4").tobytes()).hexdigest():
            raise RuntimeError("independent audit: complete NN target hash differs")
        for leaf in range(len(raw["leaf_logits"])):
            for purpose in ("feedback", "explanation"):
                math.assert_equal(section(contexts["eligibility"], [leaf], purpose),
                    report["leaf_eligibility"][str(leaf)][purpose], f"{job['job_id']}.eligibility.{leaf}.{purpose}")
        for purpose in ("feedback", "explanation"):
            math.assert_equal(result["metrics"]["validation"][purpose], report["validation"][purpose],
                              f"{job['job_id']}.report.validation.{purpose}")
        for key, expected in (
            ("split_episode_ids", {s: sorted(f"spent-v620-base-{case['base']}" for case in fold["splits"][s]) for s in v.SPLITS}),
            ("split_scenario_seed_ids", {s: sorted(str(case["scenario_seed"]) for case in fold["splits"][s]) for s in v.SPLITS}),
        ):
            math.assert_equal(report[key], expected, f"{job['job_id']}.{key}")
        final_calls = report["leaf_optimization_calls_outside_search"]
        calls = search.get("leaf_optimization_calls", 0)
        if (final_calls != (0 if job["arm"] == "D32" else len(raw["leaf_logits"])) or final_calls > 32):
            raise RuntimeError("independent audit: unaccounted final leaf fits")
        if job["arm"] == "A32" and (calls > 19968 or calls+final_calls > 20000
                or search.get("global_search_rounds", 61) > 60):
            raise RuntimeError("independent audit: A32 budget exceeded")
        if job["arm"] == "B32" and calls != 0:
            raise RuntimeError("independent audit: B32 searched unexpectedly")
        if job["arm"] == "D32" and (search.get("template_generation_leaf_optimization_calls", 20001) > 20000
                or search.get("frozen_table_leaf_optimization_calls", 541)+final_calls > 540
                or search.get("certificate", {}).get("global_function_class_optimality_gap", "missing") is not None):
            raise RuntimeError("independent audit: invalid D32 budget or optimality claim")
        final_budget = report["fit_budget"]
        if (final_budget.get("prior_leaf_optimization_calls") != calls
                or final_budget.get("actual_total_leaf_optimization_calls") != calls+final_calls
                or final_budget.get("provenance_sha256") != search.get("report_sha256")
                or final_budget.get("max_total_leaf_optimization_calls") != (20540 if job["arm"] == "D32" else 20000 if job["arm"] == "A32" else 32)):
            raise RuntimeError("independent audit: final program lost end-to-end search provenance")
        if (type(result.get("fitting_wall_seconds")) not in (int, float)
                or not np.isfinite(result["fitting_wall_seconds"])
                or not 0 <= result["fitting_wall_seconds"] <= v.round_budget_contract()[job["arm"]]["wall_seconds"]):
            raise RuntimeError("independent audit: registered fitting wall exceeded")
        if completion.get("passed") is not passed:
            raise RuntimeError("independent audit: completion gate changed")
        from experiments.warehouse.v622_capacity_audit_math import (
            audit_search_budget, audit_compiler, audit_frozen_table,
        )
        budget_review = audit_search_budget(search, job["arm"], report, fit_arrays=fit_arrays)
        if job["arm"] == "B32":
            compiler = bound(result["compiler_report"])
            audit_compiler(compiler, actor.roles[job["role"]], structure)
        if job["arm"] == "D32":
            audit_frozen_table(search, raw, structure, fit_arrays=fit_arrays)
        job_reports.append({"job": job, "passed": True, "development_gate_passed": passed,
            "program_sha256": f["sha256"], "physical_frames": {s: len(c["x"]) for s, c in contexts.items()},
            "leaf_count": len(raw["leaf_logits"]), "all_leaf_preferred_actions_reproved": True,
            "independent_budget_review": budget_review})
        recomputed_results.append(result)
    # Reconstruct route selection independently of the native summarizer.
    arm_summaries = {}
    for arm in sorted({j["arm"] for j in plan["jobs"]}):
        rows = [r for r in recomputed_results if r["job"]["arm"] == arm]
        complete = [r for r in rows if r["status"] == "completed"]
        passed = len(complete) == 6 and all(independent_gate(r["metrics"]["validation"]) for r in complete)
        min_e = min((r["metrics"]["validation"]["explanation"]["coverage"] for r in complete), default=0.)
        min_f = min((r["metrics"]["validation"]["feedback"]["expected_actionable_coverage"] for r in complete), default=0.)
        kls = [r["metrics"]["validation"][p]["kl"] for r in complete for p in ("feedback", "explanation")]
        max_kl = max(kls) if kls and all(k is not None and np.isfinite(k) for k in kls) else None
        count = sum(r["leaf_count"] for r in rows)
        arm_summaries[arm] = {"all_role_fold_gates_passed": passed,
            "completed_role_folds": len(complete), "budget_exhausted_role_folds": 6-len(complete),
            "minimum_explanation_coverage": min_e, "minimum_expected_feedback_coverage": min_f,
            "maximum_qualified_kl": max_kl, "total_leaf_count": count,
            "selection_score": [min_e, min_f, -max_kl if max_kl is not None else None, -count]}
    eligible = [a for a, row in arm_summaries.items() if row["all_role_fold_gates_passed"]]
    winner = sorted(eligible, key=lambda a: (*(-s for s in arm_summaries[a]["selection_score"]), a))[0] if eligible else None
    math.assert_equal(arm_summaries, summary["arms"], "summary.arms")
    expected_next = "fresh_confirmation" if winner else "fallback" if stage == "primary" else "closed_failed_program_precheck"
    if summary["selected_arm"] != winner or summary["next_stage"] != expected_next:
        raise RuntimeError("independent audit: route selection differs")
    if (v.source_closure_files(ROOT) != identity["source_files_sha256"]
            or v.source_closure_sha256(ROOT) != identity["source_closure_sha256"]):
        raise RuntimeError("independent audit: source changed during audit")
    for file, expected in READ_FILES.items():
        read_pinned(file, expected)
    result = {"schema": "warehouse-v622-independent-capacity-audit.v1", "passed": True,
        "candidate_id": v.CANDIDATE_ID, "stage": stage, "plan_sha256": plan_sha,
        "summary_sha256": summary_sha,
        "source_closure_sha256": identity["source_closure_sha256"],
        "job_count": len(job_reports), "jobs": job_reports,
        "bound_NN_sampled_action_checks": action_checks,
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "pilot_opened": False,
        "scope": "spent_development_evidence_arithmetic_not_sealed_admission_or_model_success"}
    write_new(path.parent/"independent-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan)
    print(json.dumps({k: result[k] for k in ("passed", "job_count", "bound_NN_sampled_action_checks")}))


if __name__ == "__main__":
    main()
