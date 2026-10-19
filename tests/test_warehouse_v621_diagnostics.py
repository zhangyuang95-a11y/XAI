"""Admission-control tests on synthetic receipts; no environments or fitting."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import run_warehouse_v621_diagnostics as run


def metrics():
    row = {"coverage": .5, "agreement": .99, "kl": .1,
           "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0,
           "expected_actionable_coverage": .45}
    return {"feedback": dict(row), "explanation": dict(row)}


def units():
    return [{"job": job, "status": "completed", "budget_audit": {"passed": True},
             "metrics": {"validation": metrics()}, "leaf_count": 12,
             "development_gate": run.development_gate(metrics())} for job in run.v.jobs()]


def search_report():
    return {"leaf_optimization_calls": 19984,
            "search_runs": [{"rounds": list(range(8 if i < 4 else 7))} for i in range(8)],
            "global_search_rounds": 60, "actual_convex_solver_calls": 10000}


def final_report(count=16):
    return {"leaf_count": count, "leaf_optimization_calls_outside_search": count,
        "convex_solver_calls_outside_search": count,
        "constrained_leaf_fits": {str(i): {"enumerated_action_solves": 1} for i in range(count)}}


def test_json_preserves_actual_numpy_structure_and_excludes_nonfinite(tmp_path):
    path = tmp_path/"structure.json"
    run.write_new(path, {"weights": np.array([[1., 0.]], np.float32),
                         "count": np.int64(1)})
    assert json.loads(path.read_text()) == {"weights": [[1., 0.]], "count": 1}
    with pytest.raises(FileExistsError):
        run.write_new(path, {})
    with pytest.raises(ValueError):
        run.write_new(tmp_path/"bad.json", {"bad": np.float32(np.nan)})


def test_gate_matches_frozen_recipe_and_rejects_missing_or_nonfinite_values():
    assert run.development_gate(metrics()) == run.v.gate_metrics(metrics())
    assert run.development_gate(metrics())["passed"]
    for key, bad in (("coverage", float("inf")), ("agreement", True),
                     ("kl", float("nan")), ("kl", None),
                     ("whole_tree_raw_illegal_count", False)):
        invalid = metrics()
        invalid["feedback"][key] = bad
        assert not run.development_gate(invalid)["passed"]


def test_budget_counts_final_refit_and_all_initializations_globally():
    report = search_report()
    audit = run.budget_audit(report, final_report(), "C")
    assert audit["passed"] and audit["end_to_end_leaf_api_calls"] == 20000
    report["leaf_optimization_calls"] += 1
    assert not run.budget_audit(report, final_report(), "A")["passed"]
    report = search_report()
    report["search_runs"][0]["rounds"].append(9)
    assert not run.budget_audit(report, {}, "C")["passed"]
    with pytest.raises(RuntimeError):
        run.budget_audit({"leaf_optimization_calls": True}, {}, "A")
    assert run.budget_audit({"leaf_optimization_calls": 0, "search_runs": []}, {}, "B")["passed"]
    assert not run.budget_audit(search_report(), {}, "B")["passed"]
    invalid = final_report()
    del invalid["leaf_optimization_calls_outside_search"]
    assert not run.budget_audit(search_report(), invalid, "C")["passed"]
    invalid = final_report()
    invalid["constrained_leaf_fits"]["0"]["enumerated_action_solves"] = 2
    assert not run.budget_audit(search_report(), invalid, "C")["passed"]


def test_no_route_selection_before_all_eighteen_exact_jobs_complete():
    rows = units()
    for invalid in (rows[:-1], rows[:-1]+[deepcopy(rows[0])]):
        with pytest.raises(RuntimeError):
            run.summarize_results(invalid)
    rows[0]["job"]["role"] = "other_role"
    with pytest.raises(RuntimeError):
        run.summarize_results(rows)
    for status, arm in (("failed", "B"), ("structurally_infeasible", "A")):
        rows = units()
        row = next(r for r in rows if r["job"]["arm"] == arm)
        row["status"] = status
        with pytest.raises(RuntimeError):
            run.summarize_results(rows)


def test_scientific_infeasibility_chooses_fallback_without_opening_any_stage():
    rows = units()
    for row in rows:
        row["development_gate"]["passed"] = False
        if row["job"]["arm"] == "B":
            row.update(status="structurally_infeasible", metrics=None, leaf_count=None)
    summary = run.summarize_results(rows)
    assert summary["selected_arm"] is None
    assert summary["next_stage"] == "shared_structure_actor_implementation"
    assert summary["arms"]["B"]["structurally_infeasible_units"] == 6
    assert summary["new_data_or_training_authorized"] is False
    assert all(summary[key] == 0 for key in run.ZERO_FIELDS)
    assert all(summary[key] is False for key in run.FALSE_FIELDS)


def test_all_six_role_fold_gates_required_and_selection_uses_worst_case():
    rows = units()
    for row in rows:
        if row["job"]["arm"] == "C":
            row["metrics"]["validation"]["explanation"]["coverage"] = .6
    assert run.summarize_results(rows)["selected_arm"] == "C"
    next(row for row in rows if row["job"]["arm"] == "C")["development_gate"]["passed"] = False
    assert run.summarize_results(rows)["selected_arm"] == "B"
    next(row for row in rows if row["job"]["arm"] == "A")["budget_audit"]["passed"] = False
    with pytest.raises(RuntimeError):
        run.summarize_results(rows)


def test_input_mutation_rejected_before_use(tmp_path):
    path = tmp_path/"bound.json"
    path.write_text('{"a":1}')
    expected = run.file_sha(path)
    assert run.read_pinned(path, expected) == b'{"a":1}'
    path.write_text('{"a":2}')
    with pytest.raises(RuntimeError, match="changed"):
        run.read_pinned(path, expected)


def test_started_job_directory_forbids_scientific_retry(tmp_path, monkeypatch):
    job = run.v.jobs()[0]
    plan_path = tmp_path/"plan.json"
    plan_path.write_text('{}')
    directory = tmp_path/"jobs"/job["job_id"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(run, "load_plan", lambda _: (plan_path, {"jobs": [job]}))
    with pytest.raises(FileExistsError):
        run.execute_job(plan_path, job["job_id"])


@pytest.fixture
def completed_batch(tmp_path, monkeypatch):
    plan_path = tmp_path/"plan.json"
    plan_path.write_text('{}')
    plan = {"jobs": run.v.jobs(), "folds": [{"assignment_sha256": str(i)} for i in range(3)]}
    monkeypatch.setattr(run, "load_plan", lambda _: (plan_path, plan))
    monkeypatch.setattr(run, "checked_program_result", lambda *args: final_report(12))
    for row in units():
        job = row["job"]
        directory = tmp_path/"jobs"/job["job_id"]
        started = {"schema": "warehouse-v621-diagnostics-job-start.v1", "job": job,
            "plan_sha256": run.file_sha(plan_path), **dict.fromkeys(run.ZERO_FIELDS, 0),
            **dict.fromkeys(run.FALSE_FIELDS, False)}
        run.write_new(directory/"started.json", started)
        search = {"leaf_optimization_calls": 0, "global_search_rounds": 0,
            "search_runs": [] if job["arm"] == "B" else [{"rounds": []} for _ in range(8)]}
        run.write_new(directory/"search.json", search)
        run.write_new(directory/"program.json", {"report": final_report(12)})
        row.update({**started, "schema": run.RESULT_SCHEMA, "actor_weights_unchanged": True,
            "excluded_frames": 0, "scientific_retry_permitted": False,
            "fold_assignment_sha256": str(job["fold"]),
            "search_report": run.pointer(directory/"search.json"),
            "program": run.pointer(directory/"program.json"),
            "budget_audit": run.budget_audit(search, final_report(12), job["arm"])})
        run.write_new(directory/"result.json", row)
        run.write_new(directory/"completed.json", {"result_sha256": run.file_sha(directory/"result.json"),
            "plan_sha256": run.file_sha(plan_path), "status": "completed", "passed": True})
    return plan_path, tmp_path/"jobs"/run.v.jobs()[0]["job_id"]


def test_completed_batch_can_select_but_grants_no_new_authority(completed_batch):
    path, _ = completed_batch
    summary = run.summarize(path)
    assert summary["selected_arm"] == "B"
    assert summary["new_data_or_training_authorized"] is False


@pytest.mark.parametrize("corruption", ["start", "failed", "status", "passed", "result_hash"])
def test_summary_rejects_incomplete_or_contradictory_receipts(completed_batch, corruption):
    path, directory = completed_batch
    if corruption == "start":
        (directory/"started.json").unlink()
    elif corruption == "failed":
        run.write_new(directory/"failed.json", {"error": "interrupted"})
    elif corruption == "result_hash":
        (directory/"result.json").write_text('{}')
    else:
        receipt = json.loads((directory/"completed.json").read_text())
        receipt[corruption] = False if corruption == "passed" else "failed"
        (directory/"completed.json").write_text(json.dumps(receipt))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run.summarize(path)


def test_summary_cannot_accept_a_program_that_native_loader_rejects(tmp_path):
    path = tmp_path/"fake-program.json"
    path.write_text('{"report":{}}')
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        run.checked_program_result({"program": run.pointer(path)}, run.v.jobs()[0], {})
