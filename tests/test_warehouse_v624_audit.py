"""Independent arithmetic/receipt guards; no experiment data or fitting."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import audit_warehouse_v624_upgrade as audit


def test_registered_independent_configuration_matches_frozen_search():
    from backend.training.warehouse_v624_search import search_configuration, UpgradeBudget
    from dataclasses import asdict
    assert audit.SEARCH_CONFIGURATION == search_configuration()
    assert audit.SEARCH_BUDGET == asdict(UpgradeBudget())


def metric(coverage=.5):
    return {
        "coverage": coverage, "expected_actionable_coverage": .45,
        "agreement": .99, "kl": .1, "illegal_count": 0,
        "path_uncertified_count": 0, "whole_tree_raw_illegal_count": 0,
        "whole_tree_path_uncertified_count": 0,
    }


def audited_jobs(coverage=.5):
    result = []
    for spec in audit.expected_jobs():
        metrics = {purpose: metric(coverage) for purpose in ("feedback", "explanation")}
        result.append({
            "job": spec, "status": "completed",
            "gate_passed": audit.development_gate(metrics)["passed"],
            "validation": metrics, "leaf_count": 32,
            "result": {"path": "/evidence/" + spec["job_id"], "sha256": "c" * 64},
        })
    return result


def plan(jobs):
    return {"jobs": [r["job"] for r in jobs], "source_sha256": "d" * 64}


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1., 1.01, True])
@pytest.mark.parametrize("purpose,key", [
    ("feedback", "coverage"), ("explanation", "coverage"),
    ("feedback", "expected_actionable_coverage"), ("explanation", "agreement"),
])
def test_invalid_probability_cannot_satisfy_independent_gate(value, purpose, key):
    metrics = {p: metric() for p in ("feedback", "explanation")}
    metrics[purpose][key] = value
    assert audit.development_gate(metrics)["passed"] is False


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -.01, True])
def test_invalid_kl_cannot_pass_or_be_selected(value):
    jobs = audited_jobs()
    jobs[-1]["validation"]["explanation"]["kl"] = value
    jobs[-1]["gate_passed"] = False
    summary = audit.reconstruct_summary(plan(jobs), "e" * 64, jobs)
    assert summary["arms"]["P1"]["all_six_pass"] is False
    assert summary["arms"]["P1"]["selection_score"] is None
    assert summary["selected_arm"] == "P0"


def test_all_twelve_required_and_pass_flags_recomputed_before_selection():
    jobs = audited_jobs()
    p = plan(jobs)
    summary = audit.reconstruct_summary(p, "e" * 64, jobs)
    assert summary["selected_arm"] == "P0"  # deterministic equal-score tie
    for bad in (jobs[:-1], jobs + [jobs[0]], [jobs[0], jobs[0], *jobs[2:]]):
        with pytest.raises(RuntimeError):
            audit.reconstruct_summary(p, "e" * 64, bad)
    bad = deepcopy(jobs)
    bad[-1]["validation"]["explanation"]["coverage"] = .4
    with pytest.raises(RuntimeError, match="independent metrics"):
        audit.reconstruct_summary(p, "e" * 64, bad)
    bad[-1]["gate_passed"] = False
    assert audit.reconstruct_summary(p, "e" * 64, bad)["selected_arm"] == "P0"
    for row in bad:
        row["validation"]["explanation"]["coverage"] = .4
        row["gate_passed"] = False
    closed = audit.reconstruct_summary(p, "e" * 64, bad)
    assert closed["selected_arm"] is None
    assert closed["next_stage"] == "closed_failed_program_precheck"
    tampered = deepcopy(closed)
    tampered["selected_arm"] = "P1"
    with pytest.raises(RuntimeError):
        audit.math.assert_equal(closed, tampered)


def leaf_ledger():
    return {
        "new_leaf_calls": 1, "leaf_api_attempts": [{
            "call": 1, "rows": 20, "rows_sha256": "a" * 64,
            "path_legal_actions": [1, 4], "cache_miss": True,
            "success": True, "actual_convex_solver_calls": 2,
        }],
        "actual_convex_solver_calls": 2, "failed_leaf_api_attempts": 0,
        "reserved_final_leaf_calls": 32,
        "maximum_new_leaf_calls_including_final_reserve": 20000,
        "new_leaf_calls_remaining": 19967,
    }


@pytest.mark.parametrize("field", ["actual_leaf_API_calls", "actual_convex_solver_calls",
                                 "full_statistic_hits", "full_statistic_misses",
                                 "full_statistic_evictions"])
def test_phase_cache_counts_must_reconcile_with_actual_calls(field):
    report = leaf_ledger()
    report["leaf_api_attempts"][0]["phase"] = "initial"
    report.update(full_leaf_statistic_cache_hits=2, full_leaf_statistic_cache_evictions=0,
                  cache_statistics_by_phase={"initial": {
                      "actual_leaf_API_calls": 1, "actual_convex_solver_calls": 2,
                      "full_statistic_hits": 2, "full_statistic_misses": 1,
                      "full_statistic_evictions": 0}})
    audit.check_phase_ledger(report)
    report["cache_statistics_by_phase"]["initial"][field] -= 1
    with pytest.raises(RuntimeError):
        audit.check_phase_ledger(report)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(new_leaf_calls=2),
    lambda s: s.update(actual_convex_solver_calls=1),
    lambda s: s.update(failed_leaf_api_attempts=1),
    lambda s: s.update(new_leaf_calls_remaining=19968),
    lambda s: s["leaf_api_attempts"][0].update(call=2),
    lambda s: s["leaf_api_attempts"][0].update(rows_sha256="bad"),
    lambda s: s["leaf_api_attempts"][0].update(actual_convex_solver_calls=1),
    lambda s: s["leaf_api_attempts"][0].update(path_legal_actions=[1, 1, 4]),
    lambda s: s["leaf_api_attempts"][0].update(cache_miss=False),
])
def test_actual_leaf_and_solver_work_cannot_be_relabelled(mutation):
    good = leaf_ledger()
    assert audit.check_leaf_ledger(good)["convex_solver_calls"] == 2
    mutation(good)
    with pytest.raises(RuntimeError):
        audit.check_leaf_ledger(good)


def node_ledger():
    return {
        "elapsed_seconds": 1.,
        "node_fit_calls": 1, "node_linear_solves": 1, "node_fit_seconds": .2,
        "node_fit_events": [{
            "call": 1, "fit_only": True, "state_sha256": "a" * 64,
            "path": [0], "fit_rows": 40, "fit_rows_sha256": "b" * 64,
            "success": True, "duration_seconds": .2,
            "support_fits": [{"support": [0, 1], "source_features": ["x", "public_hard_risk_context"],
                             "source_feature_cost": 3, "success": True,
                             "duration_seconds": .1}],
            "candidate_count": 1, "candidates": [{
                "support": [0, 1], "weights": [.8, .6], "bias": -.1,
                "tree_sha256": "c" * 64,
            }],
        }],
    }


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(node_fit_calls=2),
    lambda s: s.update(node_linear_solves=2),
    lambda s: s.update(node_fit_seconds=.3),
    lambda s: s["node_fit_events"][0].update(fit_rows=101),
    lambda s: s["node_fit_events"][0].update(fit_only=False),
    lambda s: s["node_fit_events"][0]["support_fits"][0].update(source_feature_cost=2),
    lambda s: s["node_fit_events"][0]["support_fits"][0].update(source_features=["hidden", "x"]),
    lambda s: s["node_fit_events"][0]["support_fits"][0].update(duration_seconds=.3),
    lambda s: s["node_fit_events"][0]["candidates"][0].update(support=[0, 1, 2], weights=[.2, .3, .4]),
])
def test_node_work_support_cost_and_timing_are_independently_reconciled(mutation):
    good = node_ledger()
    names = ("x", "public_hard_risk_context", "legal_up")
    assert audit.check_node_ledger(good, "P1", 100, names)["node_linear_solves"] == 1
    mutation(good)
    with pytest.raises(RuntimeError):
        audit.check_node_ledger(good, "P1", 100, names)


def test_control_arm_cannot_claim_upgraded_node_fit():
    with pytest.raises(RuntimeError, match="control arm"):
        audit.check_node_ledger(node_ledger(), "P0", 100, ("x", "public_hard_risk_context"))


def incomplete():
    r = dict(
        leaf_ledger(), schema="warehouse-v624-search-incomplete.v1",
        status="search_no_complete_incumbent", scientific_failure=True,
        method=audit.METHODS["P1"], mode="P1",
        actor_role_sha256=audit.old.ROLE_SHA256["robot_1"],
        initial_structure_sha256="a" * 64, fit_target_sha256="b" * 64,
        configuration=dict(audit.SEARCH_CONFIGURATION), budget=dict(audit.SEARCH_BUDGET),
        production_configuration=True, zero_actor_updates=True,
        zero_environment_resets=True, execution_action_override=False,
        failure_kind="initial_leaf_solver_infeasible", error_type="ConstrainedLeafFitError",
        elapsed_seconds=.2, node_fit_calls=0, node_linear_solves=0,
        node_fit_seconds=0., node_fit_events=[],
    )
    r["leaf_api_attempts"][0].update(success=False)
    r["failed_leaf_api_attempts"] = 1
    result = {
        "status": "scientific_budget_exhausted", "error_type": "SearchIncompleteError",
        "gate": {"passed": False}, "failure": r, "fitting_wall_seconds": .2,
    }
    return result, r


def verify_incomplete(result, report):
    return audit.validate_incomplete_search(
        result, report, role="robot_1", arm="P1", initial_sha="a" * 64,
        target_sha="b" * 64, fit_rows=100, feature_names=("x",))


def test_only_observed_typed_initial_failures_count_as_scientific_failure():
    result, report = incomplete()
    assert verify_incomplete(result, report)["final_leaf_calls"] == 0
    report["error_type"] = "RuntimeError"
    with pytest.raises(RuntimeError):
        verify_incomplete(result, report)
    report["failure_kind"] = "initial_fit_budget_exhausted"
    report["error_type"] = "_DeadlineExhausted"
    with pytest.raises(RuntimeError, match="exhausted actual budget"):
        verify_incomplete(result, report)
    report["elapsed_seconds"] = result["fitting_wall_seconds"] = 4790.
    assert verify_incomplete(result, report)["stop_reason"] == "initial_fit_budget_exhausted"
    result["metrics"] = {}
    with pytest.raises(RuntimeError):
        verify_incomplete(result, report)


def test_final_timeout_requires_real_wall_and_actual_final_work():
    ledger = {"new_leaf_calls": 4, "convex_solver_calls": 10}
    result = {
        "status": "scientific_budget_exhausted", "gate": {"passed": False},
        "fitting_wall_seconds": 5400.1,
        "failure": {
            "status": "budget_exhausted", "scientific_failure": True,
            "actor_role_sha256": audit.old.ROLE_SHA256["robot_1"],
            "failure_stage": "final_metrics", "leaf_optimization_calls": 2,
            "convex_solver_calls": 4,
            "fit_budget": {"prior_leaf_optimization_calls": 4,
                           "actual_final_leaf_optimization_calls": 2,
                           "actual_total_leaf_optimization_calls": 6},
        },
    }
    assert audit.validate_timeout_failure(result, ledger, "robot_1")["total_new_leaf_calls"] == 6
    for change in (
        lambda r: r.update(fitting_wall_seconds=5399.),
        lambda r: r["failure"]["fit_budget"].update(actual_total_leaf_optimization_calls=5),
        lambda r: r["failure"].update(failure_stage=""),
        lambda r: r.update(program={}),
    ):
        bad = deepcopy(result)
        change(bad)
        with pytest.raises(RuntimeError):
            audit.validate_timeout_failure(bad, ledger, "robot_1")


def test_refit_must_reprove_new_action_even_when_observed_mask_allows_it():
    report = {
        "fit_leaf_row_counts": [3],
        "leaf_fits": {"0": {
            "candidate_reports": [{"action": 4}], "enumerated_action_solves": 1,
            "success": True, "float32_feasible": True, "row_count": 3,
            "physical_fit_frame_count": 3,
        }},
        "leaf_optimization_calls": 1, "leaf_optimization_calls_outside_search": 1,
        "convex_solver_calls": 1,
        "fit_budget": {"prior_leaf_optimization_calls": 1, "actual_final_leaf_optimization_calls": 1,
                       "actual_total_leaf_optimization_calls": 2, "max_leaf_optimization_calls": 32,
                       "max_total_leaf_optimization_calls": 20000, "provenance_sha256": "a" * 64},
    }
    raw = {"leaf_logits": [[0, 0, 0, 0, 1]], "report": report}
    contexts = {"fit": {"proof": {0: {4}}, "leaves": np.zeros(3, dtype=int)}}
    result = {"search_calls": 1, "final_calls": 1}
    audit.verify_final_fit(raw, contexts, {"new_leaf_calls": 1}, {"sha256": "a" * 64}, result)
    raw["leaf_logits"] = [[2, 0, 0, 0, 1]]
    with pytest.raises(RuntimeError, match="path-certified"):
        audit.verify_final_fit(raw, contexts, {"new_leaf_calls": 1}, {"sha256": "a" * 64}, result)


def test_evidence_rejects_byte_swap_symlinks_and_mid_audit_mutation(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"passed":false}')
    reader = audit.Evidence()
    ref = reader.ref(path)
    assert reader.bound(ref) == {"passed": False}
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(RuntimeError):
        reader.ref(link)
    with pytest.raises(RuntimeError):
        reader.bound(dict(ref, sha256="0" * 64))
    path.write_text('{"passed":true}')
    with pytest.raises(RuntimeError):
        reader.recheck()
