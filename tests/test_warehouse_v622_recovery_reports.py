"""Pure synthetic reporting dispatch and import-scope restoration checks."""
import ast
import json
from pathlib import Path

import pytest

from scripts import report_warehouse_v622_recovery as r


def identities():
    return (r.warehouse_package.v622_protocol, r.metrics.v, r.metrics.load_plan,
            r.runner.base.v, r.runner.base.output_directory)


@pytest.mark.parametrize("fail", [False, True])
def test_scope_handles_posthoc_local_imports_and_restores_all_aliases(fail):
    before = identities()
    def local_imports():
        from experiments.warehouse import v622_protocol as v
        from scripts.run_warehouse_v622_capacity import load_plan
        return v, load_plan
    try:
        with r.recovery_scope():
            local_v, local_loader = local_imports()
            assert local_v is r.recovery
            assert local_loader is r.runner.base.load_plan
            assert r.metrics.v is r.runner.base.v is r.recovery
            assert r.metrics.load_plan is local_loader
            assert r.runner.base.output_directory is r.recovery.output_directory
            if fail:
                raise RuntimeError("synthetic reporting failure")
    except RuntimeError as error:
        assert fail and str(error) == "synthetic reporting failure"
    assert identities() == before


def fake_completed(tmp_path, count=1):
    plans = []
    for i in range(count):
        path = tmp_path / ("capacity-02" if i == 0 else "fallback-02") / "plan.json"
        path.parent.mkdir()
        path.write_text("{}")
        plan = {"stage": "primary" if i == 0 else "fallback",
                "identity": {"source_closure_sha256": "a" * 64}}
        plans.append((path, plan, []))
    return plans


def test_uncompleted_guard_fails_before_any_reporter_or_cached_array(monkeypatch, tmp_path):
    before = identities()
    def guard(paths, evidence):
        assert r.runner.base.v is r.recovery
        raise RuntimeError("synthetic incomplete independent audit")
    monkeypatch.setattr(r.posthoc, "completed_inputs", guard)
    monkeypatch.setattr(r.metrics, "snapshot", lambda *a: pytest.fail("report before completion gate"))
    monkeypatch.setattr(r.posthoc, "prepare_report", lambda *a: pytest.fail("cached frames before completion gate"))
    with pytest.raises(RuntimeError, match="incomplete"):
        r.prepare_report("posthoc", [tmp_path / "plan.json"])
    assert identities() == before


def test_metrics_uses_only_audited_single_plan_and_marks_carried_quality(monkeypatch, tmp_path):
    plans = fake_completed(tmp_path)
    events = []
    def guard(paths, evidence):
        events.append("completion-gate")
        evidence.hashes[str(plans[0][0])] = r.sha256(plans[0][0].read_bytes()).hexdigest()
        return plans
    def snapshot(path):
        assert path == plans[0][0] and r.metrics.v is r.recovery
        events.append("metrics")
        return {"independent_batch_arithmetic_audit_passed": True,
                "quality": {"source": "carried original receipt"}}
    monkeypatch.setattr(r.posthoc, "completed_inputs", guard)
    monkeypatch.setattr(r.metrics, "snapshot", snapshot)
    monkeypatch.setattr(r.posthoc, "prepare_report", lambda *a: pytest.fail("metrics inferred posthoc frames"))
    report = r.prepare_report("metrics", [plans[0][0]])
    binding = report["recovery_reporting"]
    assert events == ["completion-gate", "metrics"]
    assert binding["complete_independent_audit_required"] and not binding["used_for_selection"]
    assert not binding["actor_quality_change_claimed"]
    assert binding["old_quality_is_carried_provenance_not_new_evaluation"]
    assert binding["new_NN_forward_calls"] == binding["fit_calls"] == binding["actor_updates"] == 0
    assert len(binding["reporting_sources_sha256"]) == 3


def test_posthoc_accepts_audited_primary_and_fallback_and_restores_scope(monkeypatch, tmp_path):
    plans = fake_completed(tmp_path, 2)
    before = identities()
    monkeypatch.setattr(r.posthoc, "completed_inputs", lambda paths, evidence: plans)
    def cached(paths):
        from experiments.warehouse import v622_protocol as v
        assert v is r.recovery and len(paths) == 2
        return {"used_for_selection": False, "strata": []}
    monkeypatch.setattr(r.posthoc, "prepare_report", cached)
    monkeypatch.setattr(r.metrics, "snapshot", lambda *a: pytest.fail("posthoc dispatched metrics"))
    report = r.prepare_report("posthoc", [row[0] for row in plans])
    assert [p["stage"] for p in report["recovery_reporting"]["plans"]] == ["primary", "fallback"]
    assert identities() == before


def test_single_fallback_metrics_also_checks_complete_primary_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(r.recovery, "ROOT", tmp_path)
    plans = []
    for stage in ("primary", "fallback"):
        path = r.recovery.output_directory(stage) / "plan.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        plans.append((path, {"stage": stage, "identity": {"source_closure_sha256": "a" * 64}}, []))
    def guard(paths, evidence):
        assert paths == [plans[0][0], plans[1][0]]
        return plans
    def snapshot(path):
        assert path == plans[1][0]
        return {"independent_batch_arithmetic_audit_passed": True}
    monkeypatch.setattr(r.posthoc, "completed_inputs", guard)
    monkeypatch.setattr(r.metrics, "snapshot", snapshot)
    report = r.prepare_report("metrics", [plans[1][0]])
    assert [p["stage"] for p in report["recovery_reporting"]["plans"]] == ["primary", "fallback"]


@pytest.mark.parametrize("mode,paths", [("unknown", ["x"]), ("metrics", []),
                                        ("metrics", ["x", "y"]), ("posthoc", [])])
def test_invalid_read_mode_does_not_open_evidence(mode, paths, monkeypatch):
    monkeypatch.setattr(r.posthoc, "completed_inputs", lambda *a: pytest.fail("invalid request opened evidence"))
    with pytest.raises(ValueError):
        r.prepare_report(mode, paths)


def test_cli_exclusive_output_and_safe_identifier(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(r, "prepare_report", lambda mode, paths: calls.append((mode, paths)) or {"synthetic": True})
    monkeypatch.setattr(r.metrics, "markdown", lambda report: "synthetic read-only report\n")
    args = ["--mode", "metrics", "--plan", "synthetic-plan.json", "--report-id", "synthetic-01"]
    assert r.main(args) == 0
    folder = tmp_path / r.recovery.OUTPUT_ROOT / "reports" / "synthetic-01"
    assert json.loads((folder / "metrics.json").read_bytes()) == {"synthetic": True}
    assert (folder / "metrics.zh.md").read_text() == "synthetic read-only report\n"
    assert json.loads(capsys.readouterr().out)["used_for_selection"] is False
    with pytest.raises(FileExistsError):
        r.main(args)
    with pytest.raises(SystemExit):
        r.main(["--mode", "metrics", "--plan", "x", "--report-id", "../unsafe"])
    assert len(calls) == 2


def test_adapter_does_not_add_actor_or_environment_operations():
    tree = ast.parse(Path(r.__file__).read_text())
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not calls & {"checked_actor", "checked_inputs", "target_identity", "logits_encoded",
                        "reset", "step", "fit_program", "fit_structure", "minimize",
                        "summarize", "summarize_units", "backward", "actor_step"}
