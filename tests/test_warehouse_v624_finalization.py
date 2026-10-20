"""Terminal receipt tests with synthetic records; no fitting or environment."""
from copy import deepcopy
import csv
from hashlib import sha256
from io import StringIO
import json

import pytest

from scripts import audit_warehouse_v624_upgrade as audit
from scripts import finalize_warehouse_v624_upgrade as final


def records(coverage=.5):
    rows = []
    for spec in audit.expected_jobs():
        metric = dict(coverage=coverage, expected_actionable_coverage=.45,
                      agreement=.99, kl=.1, illegal_count=0, path_uncertified_count=0,
                      whole_tree_raw_illegal_count=0, whole_tree_path_uncertified_count=0)
        metrics = {p: dict(metric) for p in ("feedback", "explanation")}
        rows.append(dict(job=spec, status="completed",
                         gate_passed=audit.development_gate(metrics)["passed"],
                         validation=metrics, leaf_count=32,
                         result=dict(path="/evidence/"+spec["job_id"], sha256="a"*64),
                         ledger=dict(new_leaf_calls=10, final_leaf_calls=32,
                                     total_new_leaf_calls=42, convex_solver_calls=20,
                                     final_convex_solver_calls=64)))
    return rows


def summary(rows):
    return audit.reconstruct_summary(dict(jobs=[r["job"] for r in rows],
                                         source_sha256="b"*64), "c"*64, rows)


def test_terminal_decision_requires_all_six_each_and_never_authorizes_future_stages():
    passing = summary(records())
    assert final.closed_or_pending(passing) == dict(
        status="pending_fresh_confirmation", candidate_closed=False, selected_method="P0")
    failing = summary(records(.4))
    assert final.closed_or_pending(failing)["candidate_closed"] is True
    for mutate in (
        lambda s: s["arms"]["P1"]["role_folds"].pop(),
        lambda s: s["arms"]["P1"]["role_folds"][0].update(status="failed"),
        lambda s: s.update(selected_arm="P1"),
        lambda s: s["arms"]["P1"].update(all_six_pass=False),
    ):
        bad = deepcopy(passing)
        mutate(bad)
        with pytest.raises(RuntimeError):
            final.closed_or_pending(bad)
    assert final.AUTHORITY["formal_training_authorized"] is False
    assert final.AUTHORITY["pilot_opened"] is False
    assert final.AUTHORITY["fresh_development_episodes"] == 0


def synthetic_folder(tmp_path, monkeypatch, coverage=.5):
    from experiments.warehouse import v624_protocol as protocol
    rows = records(coverage)
    p = dict(candidate_id=audit.CANDIDATE, stage="A", source_sha256="b"*64,
             source_files={}, actor={"sha256": "d"*64}, actor_role_sha256={},
             baseline={}, initial_S1={})
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(final.canonical(p))
    s = summary(rows)
    s["plan_sha256"] = final.pointer(plan_path)["sha256"]
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(final.canonical(s))
    refs = [final.pointer(plan_path), final.pointer(summary_path)]
    live = dict(audit.ZERO, schema="warehouse-v624-independent-audit.v1",
                candidate_id=audit.CANDIDATE, stage="A", passed=True,
                summary_verified=True, source_sha256=p["source_sha256"],
                audit_script=final.pointer(audit.__file__),
                plan_sha256=refs[0]["sha256"], summary_sha256=refs[1]["sha256"],
                jobs=rows, independently_reconstructed_summary=s,
                independent_action_stream_checks=0,
                evidence_sha256={r["path"]: r["sha256"] for r in refs})
    (tmp_path / "independent-audit.json").write_bytes(final.canonical(live))
    monkeypatch.setattr(protocol, "source_sha", lambda: p["source_sha256"])
    monkeypatch.setattr(audit, "audit", lambda _: deepcopy(live))
    return plan_path, live


@pytest.mark.parametrize("coverage,status,method", [
    (.4, "closed_failed_program_precheck", None),
    (.5, "pending_fresh_confirmation", "P0"),
])
def test_finalization_writes_content_addressed_manifest_and_single_use_receipt(
        tmp_path, monkeypatch, coverage, status, method):
    path, live = synthetic_folder(tmp_path, monkeypatch, coverage)
    receipt = final.finalize(path)
    assert receipt["status"] == status
    assert receipt["selected_method"] == method
    assert all(receipt[k] == v for k, v in final.AUTHORITY.items())
    manifest_ref = receipt["evidence_manifest"]
    manifest_bytes = open(manifest_ref["path"], "rb").read()
    assert sha256(manifest_bytes).hexdigest() == manifest_ref["sha256"]
    assert manifest_ref["sha256"] in manifest_ref["path"]
    manifest = final.bound(manifest_ref)
    assert manifest["actual_compute_totals_by_arm"]["P0"]["total_new_leaf_api_calls"] == 252
    report = (path.parent / "finalization" / "report.zh.md").read_text()
    assert f"{coverage:.4%}" in report
    csv_rows = list(csv.DictReader(StringIO(
        (path.parent / "finalization" / "metrics.csv").read_text())))
    assert len(csv_rows) == 12
    assert {r["job_id"] for r in csv_rows} == {r["job"]["job_id"] for r in live["jobs"]}
    with pytest.raises(RuntimeError, match="single-use"):
        final.finalize(path)


@pytest.mark.parametrize("tamper", ["jobs", "evidence_sha256", "source_sha256"])
def test_saved_pass_boolean_does_not_override_independent_recomputation(
        tmp_path, monkeypatch, tamper):
    path, live = synthetic_folder(tmp_path, monkeypatch)
    bad = deepcopy(live)
    if tamper == "jobs":
        bad["jobs"].pop()
    elif tamper == "evidence_sha256":
        bad[tamper][str(path)] = "f"*64
    else:
        bad[tamper] = "f"*64
    (tmp_path / "independent-audit.json").write_bytes(final.canonical(bad))
    with pytest.raises(RuntimeError):
        final.finalize(path)
    assert not (tmp_path / "finalization").exists()


def test_input_mutation_after_audit_prevents_terminal_receipt(tmp_path, monkeypatch):
    path, live = synthetic_folder(tmp_path, monkeypatch)
    old_report = final.report_markdown
    def mutate(*args):
        path.write_text("{}")
        return old_report(*args)
    monkeypatch.setattr(final, "report_markdown", mutate)
    with pytest.raises(RuntimeError, match="SHA"):
        final.finalize(path)
    assert not (tmp_path / "finalization" / "receipt.json").exists()


def test_reports_use_per_unit_mean_and_minimum_and_reject_budget_overflow():
    rows = records()
    rows[0]["validation"]["explanation"]["coverage"] = .6
    table = final.report_markdown(summary(rows), rows, final.closed_or_pending(summary(rows)),
                                  final.compute_totals(rows)).decode()
    assert "51.6667%" in table and "50.0000%" in table
    rows[0]["ledger"]["total_new_leaf_calls"] = 20001
    with pytest.raises(RuntimeError, match="budget"):
        final.compute_totals(rows)


def test_no_qualified_region_is_a_scientific_failure_not_a_reporting_error():
    rows = records(.0)
    for row in rows:
        for metric in row["validation"].values():
            metric.update(coverage=0., expected_actionable_coverage=0., agreement=0., kl=None)
        row["gate_passed"] = False
    s = summary(rows)
    assert s["selected_arm"] is None
    assert all(s["arms"][arm]["selection_score"] is None for arm in audit.ARMS)
    report = final.report_markdown(s, rows, final.closed_or_pending(s),
                                   final.compute_totals(rows)).decode()
    assert "无合格区域" in report
