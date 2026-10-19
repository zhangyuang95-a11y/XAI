"""Synthetic receipt chains only: no model, fit, RNG, environment or rollout."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import finalize_warehouse_v622_development as f
from tests.test_warehouse_v622_completeness import complete as complete_d32


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n"
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha256(raw).hexdigest()}


def metrics(coverage=.35):
    row = {"coverage": coverage, "expected_actionable_coverage": max(coverage-.05, 0),
        "actionable_coverage": max(coverage-.10, 0), "agreement": .99, "kl": .1,
        "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0}
    return {"feedback": deepcopy(row), "explanation": deepcopy(row)}


class Fixture:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.output = root/f.v.OUTPUT_ROOT
        reporting_source=root/"scripts"/"finalize_warehouse_v622_development.py"
        reporting_source.parent.mkdir(parents=True,exist_ok=True)
        reporting_source.write_bytes(Path(f.__file__).read_bytes())
        helper=root/"experiments"/"warehouse"/"v622_completeness.py"
        helper.parent.mkdir(parents=True,exist_ok=True)
        helper.write_bytes((Path(f.__file__).resolve().parents[1]/"experiments/warehouse/v622_completeness.py").read_bytes())
        source = root/"frozen.py"; source.write_text("frozen scientific source\n")
        self.sources = {"frozen.py": sha256(source.read_bytes()).hexdigest()}
        self.baseline = self.old_baseline()
        monkeypatch.setattr(f, "BASELINE_SUMMARY_SHA256", self.baseline["summary"]["sha256"])
        monkeypatch.setattr(f, "BASELINE_AUDIT_SHA256", self.baseline["independent_audit"]["sha256"])
        self.history()

    def old_baseline(self):
        jobs = {}
        for fold in range(3):
            for role in f.v.ROLES:
                name = f"fold-{fold}-{role}-A"
                result = {"job": {"job_id": name, "fold": fold, "role": role, "arm": "A"},
                    "status": "completed", "leaf_count": 16, "metrics": {"validation": metrics(.30)},
                    "development_gate": {"passed": False}}
                jobs[name] = put(self.root/"old-baseline"/name/"result.json", result)
        summary = put(self.root/"old-baseline"/"summary.json", {"jobs": jobs, "selected_arm": None})
        audit = put(self.root/"old-baseline"/"audit.json", {"passed": True, "summary_sha256": summary["sha256"]})
        return {"summary": summary, "independent_audit": audit}

    def history(self):
        folder = self.output/"capacity-01"
        plan = put(folder/"plan.json", {"candidate_id": f.v.CANDIDATE_ID, "stage": "primary"})
        jobs = []
        for job in f.v.jobs("primary"):
            name = job["job_id"];d = folder/"jobs"/name
            started = put(d/"started.json", {"job": job})
            partial = job["arm"] == "A32" and (job["fold"], job["role"]) in (
                (1, "robot_2"), (2, "robot_1"), (2, "robot_2"))
            log = folder/"logs"/(name+".log");log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("global_rounds=43/60 api_calls=100 convex_calls=200\n" if partial else "complete\n")
            log_ref = {"path": str(log), "sha256": sha256(log.read_bytes()).hexdigest()}
            if partial:
                row = {"job": job, "status": "interrupted_without_search_checkpoint", "started": started,
                    "result": None, "completion": None, "log": log_ref,
                    "last_logged_progress": {"global_rounds": 43, "leaf_api_calls": 100, "convex_solver_calls": 200}}
            else:
                result = put(d/"result.json", {"job": job, "plan_sha256": plan["sha256"],
                    "budget_audit": {"end_to_end_leaf_api_calls": 10}})
                completion = put(d/"completed.json", {"result_sha256": result["sha256"]})
                row = {"job": job, "status": "completed_before_interruption", "started": started,
                    "result": result, "completion": completion, "log": log_ref}
            jobs.append(row)
        put(folder/"interrupted.json", {"schema": "warehouse-v622-engineering-interruption.v1",
            "candidate_id": f.v.CANDIDATE_ID, "complete_batch": False,
            "scientific_method_failure_claimed": False, "scientific_timeout_claimed": False,
            "plan": plan, "completed_jobs": 9, "interrupted_jobs": 3, "jobs": jobs})

    def batch(self, stage="primary", passing=None, parent=None, exhausted=False, partial_templates=None):
        folder = self.output/("capacity-02" if stage == "primary" else "fallback-02")
        identity = {"source_files_sha256": self.sources, "source_closure_sha256": f.digest(self.sources)}
        plan = {"schema": "warehouse-v622-capacity-plan.v1", "candidate_id": f.v.CANDIDATE_ID,
            "stage": stage, "identity": identity, "jobs": f.v.jobs(stage), "folds": f.v.grouped_folds(),
            "gates": f.v.GATES, "limits": f.v.LIMITS, "budgets": f.v.round_budget_contract(),
            "baseline_A16": self.baseline, "fallback_parent": parent, "scientific_retry_permitted": False,
            "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
            "publishable": False, "pilot_opened": False}
        pref = put(folder/"plan.json", plan)
        put(folder/"plan.claim.json", {"sha256": pref["sha256"], "single_use": True})
        results, refs, reviewed = [], {}, []
        for i,job in enumerate(plan["jobs"]):
            d = folder/"jobs"/job["job_id"]
            put(d/"started.json", {"job": job, "plan_sha256": pref["sha256"]})
            search_calls = 10 if job["arm"] == "A32" else 20 if job["arm"] == "D32" else 0
            outside = 0 if job["arm"] == "D32" else 2
            search={"leaf_optimization_calls": search_calls}
            if job["arm"]=="D32":
                search.update(complete_d32())
                if partial_templates is not None:
                    for key in ("template_generation_reports","frozen_terminal_tables"):search[key]=search[key][:partial_templates]
                    search["complete_templates"]=search["complete_frozen_tables"]=partial_templates
                    search["certificate"]["all_requested_templates_completed_and_enumerated"]=False
            sref = put(d/"search-report.json", search)
            zref = put(d/"program.json", {"report": {"leaf_optimization_calls_outside_search": outside}})
            tref = put(d/"structure.json", {"leaf_count": 2})
            m = metrics(.55 if job["arm"] == passing else .35)
            gate = f.v.gate_metrics(m)["passed"]
            result = {"schema": "warehouse-v622-capacity-result.v1", "job": job,
                "plan_sha256": pref["sha256"], "status": "completed", "metrics": {"validation": m},
                "development_gate": {"passed": gate}, "leaf_count": 2, "search_report": sref,
                "program": zref, "structure": tref, "compiler_report": None,
                "actor_weights_unchanged": True, "excluded_frames": 0,
                "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
                "publishable": False, "pilot_opened": False,
                "budget_audit": {"passed": True, "end_to_end_leaf_api_calls": search_calls+outside}}
            if exhausted and i == 0:
                result.update(status="scientific_budget_exhausted", metrics=None, program=None, leaf_count=0,
                    development_gate={"passed": False}, error_type="ProgramBudgetExhausted",
                    budget_exhaustion={"leaf_optimization_calls": 1})
                gate = False
            refs[job["job_id"]] = put(d/"result.json", result)
            put(d/"completed.json", {"result_sha256": refs[job["job_id"]]["sha256"],
                "plan_sha256": pref["sha256"], "passed": gate})
            reviewed.append({"job": job, "passed": True, "development_gate_passed": gate})
            results.append(result)
        summary = put(folder/"summary.json", {"schema": "warehouse-v622-capacity-summary.v1",
            "candidate_id": f.v.CANDIDATE_ID, "stage": stage, "plan_sha256": pref["sha256"], "jobs": refs,
            **f.v.summarize_units(results, stage=stage)})
        audit = put(folder/"independent-audit.json", {"schema": "warehouse-v622-independent-capacity-audit.v1",
            "passed": True, "candidate_id": f.v.CANDIDATE_ID, "stage": stage,
            "plan_sha256": pref["sha256"], "summary_sha256": summary["sha256"],
            "source_closure_sha256": identity["source_closure_sha256"], "job_count": len(results), "jobs": reviewed,
            "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0, "publishable": False, "pilot_opened": False})
        return Path(pref["path"]), {"summary": summary, "independent_audit": audit}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def read(path):return json.loads(Path(path).read_bytes())


def test_both_audited_failures_close_without_permissions_or_copy(fixture):
    p,parent = fixture.batch();b,_ = fixture.batch("fallback",parent=parent)
    old_mode = (fixture.root/"frozen.py").stat().st_mode
    refs = f.finalize(p,b,report_id="final-01",root=fixture.root)
    report = read(refs["report"]["path"])
    assert report["reason"] == "program_precheck_failed" and refs["candidate_closed"]
    assert len(report["rows"]) == 24 and len(report["paired_differences_from_A16"]) == 18
    assert report["compute"]["interrupted_01"]["completed_actual_leaf_api_calls"] == 90
    assert report["compute"]["interrupted_01"]["interrupted_logged_leaf_api_calls_lower_bound"] == 300
    assert report["compute"]["current_02"]["actual_leaf_api_calls"] == 204
    assert report["compute"]["all_v622_leaf_api_calls_lower_bound"] == 594
    assert not report["compute"]["all_v622_total_is_exact"]
    assert all(report[k] == value for k,value in f.ZERO.items())
    assert report["reporting_source_dependencies"]["D32_completeness"]["sha256"] == sha256(
        (fixture.root/"experiments/warehouse/v622_completeness.py").read_bytes()).hexdigest()
    assert len(report["D32_protocol_completeness"]) == 6
    assert report["stage_B_pressure_started"] is False
    assert "stage_C_pressure_started" not in report
    manifest = read(refs["archive_manifest"]["path"])
    assert manifest["copy_performed"] is manifest["permissions_changed"] is False
    assert manifest["payload_file_count"] == len(manifest["payload"])
    assert not any(r["relative_path"].endswith(("archive-manifest.json","candidate.closed.json","completed.json"))
        for r in manifest["payload"] if "/finalizations/" in r["relative_path"])
    assert (fixture.root/"frozen.py").stat().st_mode == old_mode
    assert "14.93" in Path(refs["markdown"]["path"]).read_text()


@pytest.mark.parametrize("stage", ["primary","fallback"])
def test_winner_opens_only_next_fresh_planning_gate(fixture,stage):
    p,parent = fixture.batch(passing="A32" if stage=="primary" else None)
    b = fixture.batch("fallback",passing="D32",parent=parent)[0] if stage=="fallback" else None
    refs = f.finalize(p,b,report_id="winner",root=fixture.root)
    report = read(refs["report"]["path"])
    assert refs["candidate_closed"] is None and report["selected_arm"] == ("A32" if stage=="primary" else "D32")
    assert report["next_stage"] == "fresh_confirmation_requires_new_authorization"
    assert not report["fresh_precheck_authorized"] and not report["PPO_started"] and not report["pilot_opened"]


def test_failed_primary_cannot_close_before_complete_fallback(fixture):
    p,_ = fixture.batch()
    with pytest.raises(RuntimeError,match="requires complete audited fallback"):
        f.finalize(p,report_id="too-early",root=fixture.root)
    assert not (fixture.output/"candidate.closed.json").exists()


def test_primary_winner_rejects_optional_competing_fallback(fixture):
    p,parent = fixture.batch(passing="A32");b,_ = fixture.batch("fallback",passing="D32",parent=parent)
    with pytest.raises(RuntimeError,match="must not be compared"):
        f.build_report(p,b,report_id="no-comparison",root=fixture.root)


@pytest.mark.parametrize("damage", ["missing_audit","audit_failed","summary_sha","audit_family","completion"])
def test_uncompleted_or_stale_evidence_is_rejected(fixture,damage):
    p,_ = fixture.batch(passing="A32");folder=p.parent
    audit_path=folder/"independent-audit.json"
    if damage=="missing_audit":audit_path.unlink()
    elif damage=="completion":(folder/"jobs"/f.v.jobs()[0]["job_id"]/"completed.json").unlink()
    else:
        audit=read(audit_path)
        if damage=="audit_failed":audit["passed"]=False
        if damage=="summary_sha":audit["summary_sha256"]="0"*64
        if damage=="audit_family":audit["jobs"].pop()
        put(audit_path,audit)
    with pytest.raises(RuntimeError):f.build_report(p,report_id="bad",root=fixture.root)


def test_source_change_and_input_symlink_are_rejected(fixture,tmp_path):
    p,_=fixture.batch(passing="A32")
    (fixture.root/"frozen.py").write_text("changed")
    with pytest.raises(RuntimeError,match="SHA differs"):f.build_report(p,report_id="bad",root=fixture.root)
    alias=fixture.root/"linked-plan.json";alias.symlink_to(p)
    with pytest.raises(RuntimeError,match="symlink"):f.build_report(alias,report_id="bad",root=fixture.root)


def test_cross_batch_fallback_parent_rejected(fixture):
    p,parent=fixture.batch();parent=deepcopy(parent);parent["summary"]["sha256"]="f"*64
    b,_=fixture.batch("fallback",parent=parent)
    with pytest.raises(RuntimeError,match="not bound"):f.build_report(p,b,report_id="bad-parent",root=fixture.root)


def test_interruption_lower_bound_recomputed_from_pinned_log(fixture):
    p,_=fixture.batch(passing="A32")
    path=fixture.root/f.INTERRUPTED_RELATIVE;history=read(path)
    next(r for r in history["jobs"] if r["result"] is None)["last_logged_progress"]["leaf_api_calls"]+=1
    put(path,history)
    with pytest.raises(RuntimeError,match="lower bound differs"):f.build_report(p,report_id="bad-log",root=fixture.root)


def test_no_overwrite_and_no_output_symlink(fixture,tmp_path):
    p,_=fixture.batch(passing="A32")
    outside=tmp_path/"outside";outside.mkdir()
    parent=fixture.output/"finalizations";parent.symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match="symlink"):f.finalize(p,report_id="unsafe",root=fixture.root)
    assert not list(outside.iterdir())
    parent.unlink()
    f.finalize(p,report_id="once",root=fixture.root)
    with pytest.raises(RuntimeError,match="already exists"):f.finalize(p,report_id="once",root=fixture.root)


def test_aggregate_missing_metrics_are_not_invented_zero():
    row={"arm":"D32","role":"robot_1","fold":0,"status":"scientific_budget_exhausted",
        **dict.fromkeys(f.FIELDS)}
    group=f.aggregate([row])["D32"]
    assert group["metrics"]["E_KL"] == {"available_role_folds":0,"minimum":None,"mean":None,"maximum":None}


def test_exhausted_actual_calls_are_kept_separate_from_missing_metrics(fixture):
    p,parent=fixture.batch(exhausted=True);b,_=fixture.batch("fallback",parent=parent)
    report,_,_=f.build_report(p,b,report_id="exhausted",root=fixture.root)
    first=next(r for r in report["rows"] if r["arm"]=="A32" and r["fold"]==0 and r["role"]=="robot_1")
    assert first["F_leaf_coverage"] is None
    assert report["compute"]["current_02"]["actual_leaf_api_calls"]==203
    assert report["aggregates"]["A32"]["metrics"]["E_KL"]["available_role_folds"]==5


@pytest.mark.parametrize("count", [1,3])
@pytest.mark.parametrize("native_winner", [None,"D32"])
def test_partial_D32_arithmetic_audit_cannot_close_or_succeed_finalization(fixture,count,native_winner):
    p,parent=fixture.batch()
    b,_=fixture.batch("fallback",parent=parent,passing=native_winner,partial_templates=count)
    with pytest.raises(RuntimeError,match="D32 protocol incomplete"):
        f.finalize(p,b,report_id="partial-family",root=fixture.root)
    assert not (fixture.output/"candidate.closed.json").exists()
    assert not (fixture.output/"finalizations"/"partial-family").exists()


@pytest.mark.parametrize("report_id", ["../escape","UPPER","", "x"*65])
def test_report_id_is_path_safe(fixture,report_id):
    with pytest.raises(RuntimeError,match="unsafe report"):
        f.build_report(fixture.output/"not-read.json",report_id=report_id,root=fixture.root)
