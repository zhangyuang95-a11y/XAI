#!/usr/bin/env python3
"""Read-only independent arithmetic and provenance audit of v6.24 stage A.

The production search, program loader, fitter, metrics and gate are never
called. Only frozen NN inference, public encoding and immutable input loading
are shared. Returning ``passed`` certifies the recorded arithmetic; scientific
admission still requires all six units of one arm to meet the original gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_recovery_protocol as old
from scripts import audit_warehouse_v621_diagnostics as math
from scripts.audit_warehouse_v622_capacity import (
    check_action_streams, context, encoded_data, section,
)
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from backend.training.warehouse_v6_actor import (
    ACTION_PHYSICS_CONFIG_SHA256, FEATURE_SCHEMA_SHA256,
)

CANDIDATE = "warehouse-v624-s1-upgrade-01"
ARMS = ("P0", "P1")
ZERO = {
    "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
    "pilot_opened": False, "publishable": False,
    "sealed_audit_used": False, "heldout_used": False,
}
LIMITS = {
    "guards_and_empty_branches_count": True, "maximum_depth": 6,
    "maximum_leaves": 32, "maximum_source_features_per_node": 3,
}
WALL_OVERSHOOT_SECONDS = 5.
METHODS = {
    "P0": "v624_s1_cached_continuation_v1",
    "P1": "v624_downstream_cost_atomic_refinement_v1",
}
SEARCH_BUDGET = {
    "maximum_new_leaf_calls": 20000, "final_leaf_reserve": 32,
    "wall_seconds": 5400, "final_wall_reserve": 600,
    "beam_width": 4, "stagnant_recipe_cycles": 3, "maximum_rounds": 512,
    "pause_after_candidates": 0,
}
SEARCH_CONFIGURATION = {
    "maximum_depth": 6, "maximum_leaves": 32, "maximum_source_feature_cost": 3,
    "guard_nodes_fixed_in_cost_optimization": True,
    "node_feature_menu": 8, "node_candidates_per_node": 8, "nodes_per_round": 8,
    "node_ridge": 1e-5, "node_threshold_quantiles": [.1, .25, .5, .75, .9],
    "node_min_child_rows": 20, "bad_leaves_per_round": 3,
    "deep_bad_leaf_repair_anchor": "nearest_ancestor_with_two_remaining_levels",
    "repair_parent_predicates": 4, "repair_child_predicates": 2,
    "repair_min_child_rows": 40, "donors_per_round": 6,
    "donor_screening": "reserve_best_disjoint_donor_per_repair_region_then_global_rank",
    "move_family_page_size": 8, "s1_candidate_menu_retained": True,
    "full_leaf_cache_entries": 8192, "tree_state_cache_entries": 128,
    "node_candidate_cache_entries": 128, "cache_eviction": "FIFO",
    "checkpoint_candidate_interval": 64, "checkpoint_seconds_interval": 30.,
    "checkpoint_small_commit_charge_seconds": 1.,
    "checkpoint_finish_reserve_seconds": 10.,
    "checkpoint_scope": "committed_candidate_boundary_only",
    "maximum_nonimproving_steps": 2,
    "acceptance": "lexicographic_fit_robust_coverage_expected_feedback_negative_KL_negative_leaves",
    "cost_objective": "full_five_action_physical_masked_cross_entropy",
    "cost_support_menu": "top8_weighted_covariance_all_1_2_3_source_cost_supports",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return sha256(json.dumps(math.plain(value), sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer(value, minimum=0, maximum=None):
    return (type(value) is int and value >= minimum
            and (maximum is None or value <= maximum))


def finite(value):
    return type(value) in (int, float) and bool(np.isfinite(value))


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def zero(value):
    for key, expected in ZERO.items():
        require(type(value.get(key)) is type(expected) and value[key] == expected,
                "closed execution authority differs: " + key)


class Evidence:
    """Bind every byte read, then recheck all evidence before returning."""

    def __init__(self):
        self.files = {}

    def raw(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(),
                "missing or symlink evidence: " + str(path))
        path = path.resolve(strict=True)
        data = path.read_bytes()
        actual = sha256(data).hexdigest()
        require(expected is None or actual == expected,
                "evidence SHA mismatch: " + str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual,
                "evidence changed within audit: " + str(path))
        self.files[str(path)] = actual
        return data

    def obj(self, path, expected=None):
        return json.loads(self.raw(path, expected))

    def ref(self, path):
        path = Path(path)
        self.raw(path)
        resolved = str(path.resolve())
        return {"path": resolved, "sha256": self.files[resolved]}

    def bound(self, pointer, expected_path=None):
        require(isinstance(pointer, dict) and set(pointer) == {"path", "sha256"}
                and isinstance(pointer["path"], str) and valid_sha(pointer["sha256"]),
                "malformed evidence pointer")
        if expected_path is not None:
            require(pointer["path"] == str(Path(expected_path).resolve()),
                    "evidence path mismatch: " + str(expected_path))
        return self.obj(pointer["path"], pointer["sha256"])

    def bind_graph(self, value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                require(valid_sha(value["sha256"]), "malformed graph SHA")
                self.raw(value["path"], value["sha256"])
            else:
                for item in value.values():
                    self.bind_graph(item)
        elif isinstance(value, list):
            for item in value:
                self.bind_graph(item)

    def recheck(self):
        for path, expected in list(self.files.items()):
            self.raw(path, expected)


def expected_jobs():
    return [
        {"job_id": f"fold-{fold}-{role}-{arm}", "arm": arm,
         "fold": fold, "role": role}
        for arm in ARMS for fold in range(3) for role in old.ROLES
    ]


def validate_job_set(specs):
    expected = expected_jobs()
    require(isinstance(specs, list) and len(specs) == 12
            and sorted(specs, key=lambda s: s.get("job_id", ""))
            == sorted(expected, key=lambda s: s["job_id"]),
            "stage A must include all twelve unique registered role/fold/arm jobs")


def development_gate(metrics):
    """Reconstruct every gate from independent all-physical-frame metrics."""
    f, e = metrics.get("feedback", {}), metrics.get("explanation", {})

    def probability(row, key, minimum):
        value = row.get(key)
        return finite(value) and 0 <= value <= 1 and value >= minimum

    checks = {
        "feedback_coverage": probability(f, "coverage", .45),
        "explanation_coverage": probability(e, "coverage", .45),
        "expected_feedback_coverage": probability(f, "expected_actionable_coverage", .40),
        "feedback_agreement": probability(f, "agreement", .95),
        "explanation_agreement": probability(e, "agreement", .97),
        "feedback_kl": finite(f.get("kl")) and 0 <= f["kl"] <= .20,
        "explanation_kl": finite(e.get("kl")) and 0 <= e["kl"] <= .20,
    }
    for purpose, m in (("feedback", f), ("explanation", e)):
        for key in ("illegal_count", "path_uncertified_count",
                    "whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count"):
            checks[purpose + "_" + key] = type(m.get(key)) is int and m[key] == 0
    return {"checks": checks, "passed": all(checks.values())}


def reconstruct_summary(plan, plan_sha, audited_jobs):
    """Use independently computed metrics, never a saved pass/selection flag."""
    validate_job_set(plan["jobs"])
    require(isinstance(audited_jobs, list) and len(audited_jobs) == 12
            and [r.get("job") for r in audited_jobs] == plan["jobs"],
            "summary requires all twelve independently audited jobs in plan order")
    arms = {}
    for arm in ARMS:
        records = [r for r in audited_jobs if r["job"]["arm"] == arm]
        rows = []
        for r in records:
            require(r.get("status") in ("completed", "scientific_budget_exhausted")
                    and type(r.get("gate_passed")) is bool,
                    "engineering interruptions cannot be selected or scientifically closed")
            complete = r["status"] == "completed"
            require(complete or r["gate_passed"] is False,
                    "incomplete unit claims a scientific pass")
            m = r["validation"] if complete else None
            if complete:
                require(development_gate(m)["passed"] == r["gate_passed"],
                        "audited gate differs from independent metrics")
            rows.append({
                "job_id": r["job"]["job_id"], "status": r["status"],
                "passed": r["gate_passed"],
                "explanation_coverage": m["explanation"]["coverage"] if complete else None,
                "expected_feedback_coverage": m["feedback"]["expected_actionable_coverage"]
                if complete else None,
                "result": r["result"],
            })
        complete = all(r["status"] == "completed" for r in records)
        score = None
        rank_valid = complete and all(
            finite(r["validation"][p].get("kl")) and r["validation"][p]["kl"] >= 0
            for r in records for p in ("feedback", "explanation"))
        rank_valid = rank_valid and all(
            finite(r["validation"][p].get(k)) and 0 <= r["validation"][p][k] <= 1
            for r in records for p, k in (
                ("explanation", "coverage"), ("feedback", "expected_actionable_coverage")))
        if rank_valid:
            score = [
                min(r["validation"]["explanation"]["coverage"] for r in records),
                min(r["validation"]["feedback"]["expected_actionable_coverage"]
                    for r in records),
                -max(r["validation"][purpose]["kl"] for r in records
                     for purpose in ("feedback", "explanation")),
                -sum(r["leaf_count"] for r in records),
            ]
        arms[arm] = {
            "all_six_pass": all(r["status"] == "completed" and r["gate_passed"]
                               for r in records),
            "role_folds": rows, "selection_score": score,
        }
    passing = [a for a in ARMS if arms[a]["all_six_pass"]]
    selected = max(passing, key=lambda a: (arms[a]["selection_score"], a == "P0")) \
        if passing else None
    return dict(
        ZERO, schema="warehouse-v624-upgrade-summary.v1",
        candidate_id=CANDIDATE, stage="A", plan_sha256=plan_sha,
        source_sha256=plan["source_sha256"],
        jobs={r["job"]["job_id"]: r["result"] for r in audited_jobs},
        arms=arms, selected_arm=selected,
        next_stage="fresh_confirmation" if selected else "closed_failed_program_precheck",
    )


def check_leaf_ledger(search, *, incomplete=False):
    """Verify actual cache-miss API attempts and all attempted action solves."""
    count = search.get("new_leaf_calls")
    attempts = search.get("leaf_api_attempts")
    require(integer(count, 0 if incomplete else 1, 19968) and isinstance(attempts, list)
            and len(attempts) == count, "search leaf API ledger length or budget")
    solves = failed = 0
    for index, event in enumerate(attempts, 1):
        legal = event.get("path_legal_actions")
        actual = event.get("actual_convex_solver_calls")
        require(event.get("call") == index and event.get("cache_miss") is True,
                "search API sequence/cache declaration")
        require(isinstance(legal, list) and legal == sorted(set(legal)) and 4 in legal
                and all(integer(a, 0, 4) for a in legal), "invalid path legal-action set")
        require(integer(actual, 0, len(legal)) and type(event.get("success")) is bool,
                "invalid actual convex-solver count")
        require(integer(event.get("rows")) and valid_sha(event.get("rows_sha256")),
                "missing leaf fit-row identity")
        if event["success"]:
            require(actual == len(legal), "successful fit skipped a legal-action solve")
        solves += actual
        failed += not event["success"]
    require(search.get("actual_convex_solver_calls") == solves
            and search.get("failed_leaf_api_attempts") == failed,
            "search aggregate solve/failure ledger differs")
    require(search.get("reserved_final_leaf_calls") == 32
            and search.get("maximum_new_leaf_calls_including_final_reserve") == 20000
            and (incomplete or search.get("new_leaf_calls_remaining") == 19968 - count),
            "search/final reserve accounting differs")
    return {"new_leaf_calls": count, "convex_solver_calls": solves,
            "failed_leaf_calls": failed}


def check_phase_ledger(search):
    """Reconcile phase counters to observed cache misses and solver attempts."""
    phases = search.get("cache_statistics_by_phase")
    allowed = {"initial", "proposal", "candidate"}
    counters = {"full_statistic_hits", "full_statistic_misses", "full_statistic_evictions",
                "actual_leaf_API_calls", "actual_convex_solver_calls"}
    require(isinstance(phases, dict) and set(phases) <= allowed,
            "invalid cache phase statistics")
    for event in search["leaf_api_attempts"]:
        require(event.get("phase") in phases, "API ledger phase missing")
    for phase, counts in phases.items():
        require(isinstance(counts, dict) and set(counts) <= counters
                and all(integer(v) for v in counts.values()), "invalid phase counter")
        events = [e for e in search["leaf_api_attempts"] if e["phase"] == phase]
        require(counts.get("actual_leaf_API_calls", 0) == len(events)
                and counts.get("actual_convex_solver_calls", 0)
                == sum(e["actual_convex_solver_calls"] for e in events)
                and counts.get("full_statistic_misses", 0) >= len(events),
                "phase API/solver/cache-miss accounting differs")
    for phase_key, total_key in (
        ("full_statistic_hits", "full_leaf_statistic_cache_hits"),
        ("full_statistic_evictions", "full_leaf_statistic_cache_evictions"),
    ):
        require(sum(c.get(phase_key, 0) for c in phases.values()) == search.get(total_key),
                "phase cache aggregate differs")


def check_fit_identity(report, role, arm, initial_sha, feature_names, fit_data):
    algorithm_budget = dict(SEARCH_BUDGET)
    algorithm_budget.pop("pause_after_candidates")
    identity = {
        "mode": arm,
        "source": {name: sha256((ROOT / "backend/training" / name).read_bytes()).hexdigest()
                   for name in ("warehouse_v624_search.py", "warehouse_v623_search.py",
                                "warehouse_v622_search.py")},
        "budget": algorithm_budget, "configuration": SEARCH_CONFIGURATION,
        "actor_role_sha256": old.ROLE_SHA256[role], "initial_structure_sha256": initial_sha,
        "feature_names": list(feature_names), "episode_ids": [str(e) for e in fit_data[4]],
        **{key: sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()
           for key, a in zip(("encoded", "masks", "targets"), fit_data[:3], strict=True)},
    }
    require(report.get("fit_input_sha256") == digest(identity),
            "search full fit-input identity differs")


def check_node_ledger(search, arm, fit_rows, names):
    events = search.get("node_fit_events")
    require(isinstance(events, list) and search.get("node_fit_calls") == len(events),
            "node-fit event count differs")
    require(arm != "P0" or not events, "control arm performed an upgraded node fit")
    calls = 0
    seconds = 0.
    for number, event in enumerate(events, 1):
        require(event.get("call") == number and event.get("fit_only") is True
                and valid_sha(event.get("state_sha256"))
                and valid_sha(event.get("fit_rows_sha256"))
                and integer(event.get("fit_rows"), 0, fit_rows)
                and isinstance(event.get("path"), list) and len(event["path"]) <= 5
                and all(type(v) is int and v in (0, 1) for v in event["path"]),
                "node-fit row/state/path identity differs")
        require(type(event.get("success")) is bool
                and finite(event.get("duration_seconds")) and event["duration_seconds"] >= 0,
                "node-fit timing or status missing")
        supports = event.get("support_fits")
        require(isinstance(supports, list) and len(supports) <= 92,
                "node sparse support menu exceeds eight-feature combinations")
        unique = set()
        linear_seconds = 0.
        for attempt in supports:
            support = attempt.get("support")
            require(isinstance(support, list) and 1 <= len(support) <= 3
                    and len(set(support)) == len(support)
                    and all(integer(i, 0, len(names) - 1) for i in support),
                    "invalid node-fit support indices")
            key = tuple(sorted(support))
            require(key not in unique, "duplicate node-fit support solve")
            unique.add(key)
            source = [names[i] for i in support]
            cost = sum(2 if name == "public_hard_risk_context" else 1 for name in source)
            require(attempt.get("source_features") == source
                    and attempt.get("source_feature_cost") == cost <= 3
                    and not any(name.startswith("legal_") for name in source),
                    "node support changes a guard or exceeds public-source cost")
            require(type(attempt.get("success")) is bool
                    and finite(attempt.get("duration_seconds"))
                    and attempt["duration_seconds"] >= 0, "node linear-solve timing/status")
            linear_seconds += attempt["duration_seconds"]
        require(linear_seconds <= event["duration_seconds"] + 1e-6,
                "node support solves exceed recorded node duration")
        if event["success"]:
            candidates = event.get("candidates")
            require(isinstance(candidates, list)
                    and event.get("candidate_count") == len(candidates) <= 8,
                    "node candidate count/menu differs")
            for candidate in candidates:
                support, weights = candidate.get("support"), candidate.get("weights")
                require(isinstance(support, list) and 1 <= len(support) <= 3
                        and support == sorted(set(support))
                        and all(integer(i, 0, len(names) - 1) for i in support)
                        and isinstance(weights, list) and len(weights) == len(support)
                        and all(finite(w) and w != 0 for w in weights)
                        and finite(candidate.get("bias")) and valid_sha(candidate.get("tree_sha256")),
                        "node candidate parameter identity")
                source = [names[i] for i in support]
                require(not any(n.startswith("legal_") for n in source)
                        and sum(2 if n == "public_hard_risk_context" else 1 for n in source) <= 3,
                        "node candidate source support cost/guard")
        calls += len(supports)
        seconds += event["duration_seconds"]
    require(search.get("node_linear_solves") == calls
            and finite(search.get("node_fit_seconds"))
            and np.isclose(search["node_fit_seconds"], seconds, atol=1e-6, rtol=1e-10),
            "node aggregate solve count/duration differs")
    require(seconds <= search["elapsed_seconds"] + WALL_OVERSHOOT_SECONDS,
            "node fit duration exceeds active search budget")
    return {"node_fit_calls": len(events), "node_linear_solves": calls,
            "node_fit_seconds": seconds}


def search_ledger(search, *, role, arm, initial_sha, target_sha, fit_rows,
                  feature_names=(), fit_data=None):
    """Validate search identity, wall limits and the new node-fit accounting."""
    require(search.get("schema") == "warehouse-v624-fit-search.v1",
            "v624 search schema differs")
    require(search.get("mode") == arm and search.get("method") == METHODS[arm]
            and search.get("actor_role_sha256") == old.ROLE_SHA256[role]
            and search.get("initial_structure_sha256") == initial_sha
            and search.get("fit_target_sha256") == target_sha, "search data/model binding")
    require(search.get("search_uses_fit_only") is True
            and search.get("calibration_eligibility_validation_used") is False,
            "search leaks non-fit partitions")
    require(search.get("complete_incumbent") is True
            and search.get("execution_action_override") is False
            and search.get("zero_actor_updates") is True
            and search.get("zero_environment_resets") is True,
            "search action/update/completion authority")
    require(search.get("budget") == SEARCH_BUDGET
            and search.get("production_configuration") is True
            and search.get("configuration") == SEARCH_CONFIGURATION,
            "search scientific budget/configuration differs")
    ledger = check_leaf_ledger(search)
    check_phase_ledger(search)
    require(finite(search.get("elapsed_seconds"))
            and 0 <= search["elapsed_seconds"] <= 4800 + WALL_OVERSHOOT_SECONDS,
            "search wall exceeds registered budget")
    ledger["search_elapsed_seconds"] = search["elapsed_seconds"]
    ledger["stop_reason"] = search.get("stop_reason")
    require(search.get("stop_reason") in (
        "wall_budget", "leaf_call_budget", "three_complete_recipe_cycles_without_improvement",
        "maximum_rounds"), "paused or unrecognized search cannot be finalized")
    require(search.get("wall_seconds") == 5400 and search.get("search_wall_seconds") == 4800,
            "search active budget identity differs")
    rounds = search.get("rounds")
    require(isinstance(rounds, list) and len(rounds) <= 512, "search round ceiling")
    previous_calls = 0
    for number, row in enumerate(rounds, 1):
        require(row.get("round") == number and row.get("recipe") == (number - 1) % 8
                and integer(row.get("new_leaf_calls"), previous_calls, ledger["new_leaf_calls"])
                and integer(row.get("evaluated")) and integer(row.get("failed"))
                and type(row.get("improved")) is bool, "search round sequence or actual work")
        previous_calls = row["new_leaf_calls"]
    for key in ("leaf_cache_hits", "full_leaf_statistic_cache_hits",
                "full_leaf_statistic_cache_evictions", "incremental_routing_rows",
                "incremental_routing_reused_regions", "node_candidate_cache_hits",
                "node_candidate_cache_evictions", "checkpoint_count", "resume_count"):
        require(integer(search.get(key)), "missing actual cache/routing/checkpoint count: " + key)
    require(integer(search.get("full_leaf_statistic_cache_entries"), 0, 8192)
            and search["full_leaf_statistic_cache_hits"] <= search["leaf_cache_hits"]
            and search.get("cache_limits") == {
                "full_leaf_statistics": 8192, "tree_states": 128, "node_candidates": 128},
            "cache accounting or memory limits differ")
    ledger.update(check_node_ledger(search, arm, fit_rows, tuple(feature_names)))
    if fit_data is not None:
        check_fit_identity(search, role, arm, initial_sha, feature_names, fit_data)
    ledger["fit_rows"] = fit_rows
    return ledger


def validate_incomplete_search(result, report, *, role, arm, initial_sha, target_sha,
                               fit_rows, feature_names):
    """Typed solver infeasibility and real exhausted budgets are observable failures.

    Arbitrary RuntimeError, missing jobs and process interruption remain
    engineering failures and cannot acquire scientific closure through a flag.
    """
    require(result.get("status") == "scientific_budget_exhausted"
            and result.get("error_type") == "SearchIncompleteError"
            and result.get("gate") == {"passed": False}
            and result.get("failure") == report
            and "metrics" not in result and "program" not in result,
            "incomplete search result claims usable/unbound evidence")
    require(report.get("schema") == "warehouse-v624-search-incomplete.v1"
            and report.get("status") == "search_no_complete_incumbent"
            and report.get("scientific_failure") is True
            and report.get("mode") == arm and report.get("method") == METHODS[arm]
            and report.get("actor_role_sha256") == old.ROLE_SHA256[role]
            and report.get("initial_structure_sha256") == initial_sha
            and report.get("fit_target_sha256") == target_sha,
            "incomplete search identity differs")
    require(report.get("budget") == SEARCH_BUDGET
            and report.get("production_configuration") is True
            and report.get("configuration") == SEARCH_CONFIGURATION
            and report.get("zero_actor_updates") is True
            and report.get("zero_environment_resets") is True
            and report.get("execution_action_override") is False,
            "incomplete search configuration/authority differs")
    elapsed = report.get("elapsed_seconds")
    require(finite(elapsed) and 0 <= elapsed <= 4800 + WALL_OVERSHOOT_SECONDS
            and result.get("fitting_wall_seconds") == elapsed,
            "incomplete search time evidence differs")
    ledger = check_leaf_ledger(report, incomplete=True)
    ledger.update(check_node_ledger(report, arm, fit_rows, tuple(feature_names)))
    kind, error = report.get("failure_kind"), report.get("error_type")
    if kind == "initial_leaf_solver_infeasible":
        require(error == "ConstrainedLeafFitError" and ledger["failed_leaf_calls"] >= 1
                and ledger["convex_solver_calls"] >= 1,
                "infeasible leaf has no actual typed solver failure")
    elif kind == "initial_fit_budget_exhausted":
        require(error in ("_BudgetExhausted", "_DeadlineExhausted")
                and (ledger["new_leaf_calls"] >= 19968 or elapsed >= 4800 - 10.),
                "search budget failure without exhausted actual budget")
    else:
        raise RuntimeError("unregistered incomplete search failure kind")
    return dict(ledger, search_elapsed_seconds=elapsed, stop_reason=kind,
                final_leaf_calls=0, final_convex_solver_calls=0,
                total_new_leaf_calls=ledger["new_leaf_calls"])


def validate_timeout_failure(result, search_ledger_value, role):
    """A missing/partial engineering result cannot masquerade as a budget fail."""
    failure = result.get("failure", {})
    require(result.get("status") == "scientific_budget_exhausted"
            and result.get("gate") == {"passed": False}
            and failure.get("status") == "budget_exhausted"
            and failure.get("scientific_failure") is True
            and failure.get("actor_role_sha256") == old.ROLE_SHA256[role],
            "unproven scientific budget failure")
    wall = result.get("fitting_wall_seconds")
    require(finite(wall) and 5400 <= wall <= 5400 + WALL_OVERSHOOT_SECONDS
            and isinstance(failure.get("failure_stage"), str)
            and bool(failure["failure_stage"]), "timeout lacks actual fit wall/stage")
    final = failure.get("leaf_optimization_calls")
    solves = failure.get("convex_solver_calls")
    budget = failure.get("fit_budget", {})
    prior = search_ledger_value["new_leaf_calls"]
    require(integer(final, 0, 32) and integer(solves, 0, 5 * final)
            and budget.get("prior_leaf_optimization_calls") == prior
            and budget.get("actual_final_leaf_optimization_calls") == final
            and budget.get("actual_total_leaf_optimization_calls") == prior + final <= 20000,
            "timeout API/solver budget not evidenced")
    require("program" not in result and "metrics" not in result,
            "timeout falsely carries a usable program")
    return dict(search_ledger_value, final_leaf_calls=final,
                final_convex_solver_calls=solves, total_new_leaf_calls=prior + final)


def verify_final_fit(raw_program, contexts, search, search_ref, result):
    """Reprove every refitted leaf and reconcile actual solver work."""
    report = raw_program["report"]
    count = len(raw_program["leaf_logits"])
    fits = report.get("leaf_fits")
    require(isinstance(fits, dict) and set(fits) == {str(i) for i in range(count)},
            "final fit does not cover all leaves including empty branches")
    solves = 0
    for leaf in range(count):
        fit = fits[str(leaf)]
        proof = contexts["fit"]["proof"][leaf]
        chosen = int(np.asarray(raw_program["leaf_logits"][leaf], np.float32).argmax())
        require(chosen in proof, "new refitted leaf action is not path-certified")
        candidates = fit.get("candidate_reports")
        require(isinstance(candidates, list)
                and len(candidates) == fit.get("enumerated_action_solves") == len(proof)
                and sorted(z.get("action") for z in candidates) == sorted(proof),
                "final leaf did not enumerate its legal action set")
        require(fit.get("success") is True and fit.get("float32_feasible") is True,
                "final leaf optimizer has no feasible completion")
        rows = int((contexts["fit"]["leaves"] == leaf).sum())
        require(fit.get("row_count") == rows and fit.get("physical_fit_frame_count") == rows
                and report["fit_leaf_row_counts"][leaf] == rows,
                "final leaf row accounting differs")
        solves += len(candidates)
    calls = search["new_leaf_calls"]
    budget = report.get("fit_budget", {})
    require(report.get("leaf_optimization_calls")
            == report.get("leaf_optimization_calls_outside_search") == count
            and report.get("convex_solver_calls") == solves, "final API aggregate differs")
    require(budget.get("prior_leaf_optimization_calls") == calls
            and budget.get("actual_final_leaf_optimization_calls") == count
            and budget.get("actual_total_leaf_optimization_calls") == calls + count <= 20000
            and budget.get("max_leaf_optimization_calls") == 32
            and budget.get("max_total_leaf_optimization_calls") == 20000
            and budget.get("provenance_sha256") == search_ref["sha256"],
            "combined search/final API provenance differs")
    require(result.get("search_calls") == calls and result.get("final_calls") == count,
            "result actual call totals differ")
    return dict(search, final_leaf_calls=count, final_convex_solver_calls=solves,
                total_new_leaf_calls=calls + count)


def audit(plan_path):
    """Recompute the complete registered batch. This function writes nothing."""
    from experiments.warehouse import v624_protocol as v

    begin = time.monotonic()
    evidence = Evidence()
    own = evidence.ref(__file__)
    path = Path(plan_path).resolve(strict=True)
    folder = path.parent
    plan = evidence.obj(path)
    plan_sha = evidence.files[str(path)]
    require(plan.get("schema") == "warehouse-v624-upgrade-plan.v1"
            and plan.get("candidate_id") == CANDIDATE and plan.get("stage") == "A",
            "unregistered stage-A plan")
    validate_job_set(plan.get("jobs"))
    require(plan["jobs"] == v.jobs(), "registered paired job scheduling differs")
    zero(plan)
    require(evidence.obj(folder / "plan.claim.json")
            == {"sha256": plan_sha, "single_use": True}, "single-use plan claim differs")
    require(plan["source_files"] == v.source_files()
            and plan["source_sha256"] == v.source_sha()
            and plan["recipe_sha256"] == v.recipe()
            and plan["runtime"] == runtime_identity()
            and plan["folds"] == old.grouped_folds()
            and plan.get("budget") == v.BUDGET and plan.get("gates") == v.GATES,
            "source/recipe/runtime/fold/budget drift")
    for key in ("fresh_confirmation_authorized", "automatic_retry", "scientific_retry"):
        require(plan.get(key) is False, "stage A grants unregistered authority: " + key)
    for relative, expected in plan["source_files"].items():
        evidence.raw(ROOT / relative, expected)
        evidence.raw(folder / "source" / plan["source_sha256"] / relative, expected)
    actor, actor_path = old.checked_actor(ROOT)
    require(plan["actor"] == evidence.ref(actor_path)
            and plan["actor"]["sha256"] == old.ACTOR_SHA256
            and plan["actor_role_sha256"] == old.ROLE_SHA256, "frozen Actor binding differs")
    pinned_baseline, pinned_initial = v.checked_baseline(ROOT)
    require(plan["baseline"] == pinned_baseline and plan["initial_S1"] == pinned_initial,
            "prior S1 baseline not the pinned complete audited batch")
    evidence.bind_graph(plan["baseline"])
    evidence.bind_graph(plan["initial_S1"])
    episodes, manifest = old.checked_inputs(ROOT)
    require(manifest == plan["inputs"] and old.target_identity(actor, episodes) == plan["targets"],
            "spent input cohort or full-NN targets changed")
    for file, expected in manifest["files_sha256"].items():
        evidence.raw(file, expected)
    action_checks = check_action_streams(episodes)
    cached_arrays = {}
    audited = []
    for spec in plan["jobs"]:
        name, role = spec["job_id"], spec["role"]
        work = folder / "jobs" / name
        require(not (work / "failed.json").exists(), "engineering failure: " + name)
        result = evidence.obj(work / "result.json")
        started = evidence.obj(work / "started.json")
        completion = evidence.obj(work / "completed.json")
        zero(started)
        zero(result)
        require(started == dict(ZERO, job=spec, stage="A", plan_sha256=plan_sha),
                "job start binding differs: " + name)
        require(result.get("schema") == "warehouse-v624-upgrade-result.v1"
                and result.get("job") == spec and result.get("stage") == "A"
                and result.get("plan_sha256") == plan_sha, "job result binding: " + name)
        result_ref = evidence.ref(work / "result.json")
        require(completion == {"plan_sha256": plan_sha,
                               "result_sha256": result_ref["sha256"],
                               "passed": result["gate"]["passed"]},
                "job completion binding differs: " + name)
        require(finite(result.get("elapsed_seconds")) and result["elapsed_seconds"] >= 0,
                "invalid elapsed wall time")
        # Only fitting/search/final-metrics wall is subject to the 90-minute
        # budget; immutable cohort/hash initialization is reported separately.
        require(finite(result.get("fitting_wall_seconds"))
                and 0 <= result["fitting_wall_seconds"] <= 5400 + WALL_OVERSHOOT_SECONDS
                and result["elapsed_seconds"] >= result["fitting_wall_seconds"],
                "whole fitting wall exceeds budget")
        initial_ref = plan["initial_S1"][f"fold-{spec['fold']}-{role}"]["structure"]
        initial = evidence.bound(initial_ref)
        names = tuple(actor.roles[role].feature_names)
        initial_tree, _ = math.structure(initial, names, enforce_limits=False)
        initial_sha = digest(initial_tree)
        key = (spec["fold"], role)
        if key not in cached_arrays:
            fold = plan["folds"][spec["fold"]]
            cached_arrays[key] = {}
            for split in old.SPLITS:
                rows = [r for case in fold["splits"][split] for r in episodes[case["base"]][role]]
                cached_arrays[key][split] = encoded_data(actor, role, rows)
        data = cached_arrays[key]
        target_sha = sha256(np.ascontiguousarray(data["fit"][2], dtype="<f4").tobytes()).hexdigest()
        if result.get("error_type") == "SearchIncompleteError":
            incomplete = evidence.bound(result["search_report"], work / "incomplete-search-report.json")
            check_fit_identity(incomplete, role, spec["arm"], initial_sha, names, data["fit"])
            ledger = validate_incomplete_search(
                result, incomplete, role=role, arm=spec["arm"], initial_sha=initial_sha,
                target_sha=target_sha, fit_rows=len(data["fit"][0]), feature_names=names)
            audited.append({"job": spec, "status": result["status"], "gate_passed": False,
                            "ledger": ledger, "validation": None, "leaf_count": None,
                            "result": result_ref})
            continue
        search = evidence.obj(work / "search-report.json")
        search_ref = evidence.ref(work / "search-report.json")
        ledger = search_ledger(search, role=role, arm=spec["arm"], initial_sha=initial_sha,
                               target_sha=target_sha, fit_rows=len(data["fit"][0]),
                               feature_names=names, fit_data=data["fit"])
        if result.get("status") == "scientific_budget_exhausted":
            ledger = validate_timeout_failure(result, ledger, role)
            audited.append({"job": spec, "status": result["status"], "gate_passed": False,
                            "ledger": ledger, "validation": None, "leaf_count": None,
                            "result": result_ref})
            continue
        require(result.get("status") == "completed" and result.get("actor") == plan["actor"],
                "completed result/Actor binding differs: " + name)
        require(result.get("initial_structure") == initial_ref
                and result.get("initial_structure_sha256") == initial_sha,
                "job did not start from the registered S1 topology")
        require(result.get("search_report") == search_ref, "search-report pointer differs")
        raw_program = evidence.bound(result["program"], work / "program.json")
        structure = evidence.bound(result["structure"], work / "structure.json")
        tree, capacity = math.structure(structure, names, enforce_limits=False)
        require(capacity["depth"] <= 6 and capacity["leaf_count"] <= 32
                and capacity["maximum_source_features_per_node"] <= 3, "program complexity")
        require(result.get("leaf_count") == search.get("leaf_count") == capacity["leaf_count"]
                and search.get("depth") == capacity["depth"], "result/search complexity metadata")
        require(raw_program.get("version") == "warehouse-role-program.v6.22-depth6-leaves32"
                and raw_program.get("method") == "v620_m2_expanded_distribution_refit_v622"
                and raw_program.get("role") == role
                and raw_program.get("actor_role_sha256") == old.ROLE_SHA256[role]
                and raw_program.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
                and raw_program.get("action_physics_config_sha256") == ACTION_PHYSICS_CONFIG_SHA256
                and raw_program.get("limits") == LIMITS, "final program identity differs")
        embedded, _ = math.structure(raw_program["structure"], names, enforce_limits=False)
        math.assert_equal(tree, embedded, name + ".embedded_structure")
        contexts = {split: context(raw_program, arrays, names) for split, arrays in data.items()}
        report = raw_program["report"]
        require(report.get("fit_target_sha256") == target_sha
                and report.get("full_NN_distribution_targets") is True
                and report.get("external_program_selects_execution_action") is False
                and report.get("excluded_frames") == 0, "full distribution fit evidence")
        for key, index in (("split_episode_ids", 4), ("split_scenario_seed_ids", 5)):
            expected_ids = {split: sorted(set(str(x) for x in arrays[index]))
                            for split, arrays in data.items()}
            math.assert_equal(expected_ids, report[key], name + "." + key)
            for left in old.SPLITS:
                for right in old.SPLITS:
                    require(left == right or not set(expected_ids[left]) & set(expected_ids[right]),
                            "program fit/calibration/eligibility/validation partitions overlap")
        math.assert_equal({s: len(a[0]) for s, a in data.items()},
                          report["physical_frames_per_split"], name + ".physical_frames")
        for split, c in contexts.items():
            for purpose in ("feedback", "explanation"):
                actual = section(c, raw_program[purpose + "_leaves"], purpose)
                math.assert_equal(actual, result["metrics"][split][purpose],
                                  name + "." + split + "." + purpose)
                for leaf in range(capacity["leaf_count"]):
                    one = section(c, [leaf], purpose)
                    math.assert_equal(one, result["metrics"][split]["per_leaf"][str(leaf)][purpose],
                                      name + "." + split + ".leaf." + str(leaf) + "." + purpose)
                    if split == "eligibility":
                        math.assert_equal(one, report["leaf_eligibility"][str(leaf)][purpose],
                                          name + ".recorded_eligibility." + str(leaf) + "." + purpose)
        for purpose, threshold in (("feedback", .93), ("explanation", .95)):
            selected = [
                leaf for leaf in range(capacity["leaf_count"])
                if (contexts["fit"]["leaves"] == leaf).any()
                and math.eligible(section(contexts["eligibility"], [leaf], purpose), threshold)
            ]
            math.assert_equal(selected, raw_program[purpose + "_leaves"],
                              name + ".qualified." + purpose)
        ledger = verify_final_fit(raw_program, contexts, ledger, search_ref, result)
        require(result.get("budget_checks") == dict.fromkeys((
            "search_calls_with_reserved_final", "final_calls_at_most_32",
            "total_new_calls_at_most_20000", "all_final_leaves_refitted",
            "wall_at_most_90_minutes", "no_actor_change"), True),
            "production final budget ledger differs")
        validation = {purpose: section(contexts["validation"], raw_program[purpose + "_leaves"], purpose)
                      for purpose in ("feedback", "explanation")}
        gate = development_gate(validation)
        math.assert_equal(gate, result["gate"], name + ".independent_gate")
        audited.append({
            "job": spec, "status": "completed", "gate_passed": gate["passed"],
            "ledger": ledger, "validation": validation, "leaf_count": capacity["leaf_count"],
            "result": result_ref,
        })
    expected = reconstruct_summary(plan, plan_sha, audited)
    summary_ref = evidence.ref(folder / "summary.json")
    math.assert_equal(expected, evidence.obj(folder / "summary.json"), "independent_summary")
    require(v.source_sha() == plan["source_sha256"], "source changed during independent audit")
    evidence.recheck()
    return dict(
        ZERO, schema="warehouse-v624-independent-audit.v1",
        candidate_id=CANDIDATE, stage="A", passed=True, summary_verified=True,
        audit_script=own, plan_sha256=plan_sha, source_sha256=plan["source_sha256"],
        summary_sha256=summary_ref["sha256"], jobs=audited,
        independently_reconstructed_summary=expected,
        independent_action_stream_checks=action_checks,
        evidence_sha256=dict(sorted(evidence.files.items())),
        at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - begin,
        scope="already-spent stage-A development arithmetic; no fitting or new environment",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    receipt = audit(args.plan)
    # The runner owns the single-use receipt write. CLI is read-only too.
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
