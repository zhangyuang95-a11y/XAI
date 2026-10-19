#!/usr/bin/env python3
"""Finalize completed, independently audited capacity batches without experiments.

Produces a report and a finite archive payload manifest. It never copies assets,
changes permissions, runs an Actor, fits a program, or authorizes a later stage.
Stage B includes fresh confirmation and pressure; stage C is paired PPO.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_protocol as v

BASELINE_SUMMARY_SHA256 = v.BASELINE_SUMMARY_SHA256
BASELINE_AUDIT_SHA256 = v.BASELINE_AUDIT_SHA256
INTERRUPTED_RELATIVE = Path(v.OUTPUT_ROOT)/"capacity-01"/"interrupted.json"
FINAL_SCHEMA = "warehouse-v622-development-finalization.v1"
ZERO = {"environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
    "stage_B_fresh_confirmation_started": False, "stage_B_pressure_started": False,
    "PPO_started": False, "final_sealed_audit_started": False,
    "fresh_precheck_authorized": False, "publishable": False, "pilot_opened": False,
    "admission_evidence": False, "human_study_evidence": False}
FIELDS = ("F_leaf_coverage", "E_leaf_coverage", "F_expected_coverage",
    "F_actual_coverage", "E_actual_coverage", "F_agreement", "E_agreement",
    "F_KL", "E_KL", "raw_illegal_count", "path_uncertified_count")


def require(ok, message):
    if not ok:
        raise RuntimeError("v622 finalization: "+message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer(value, label):
    require(type(value) is int and value >= 0, "invalid counter "+label)
    return value


class Evidence:
    """One byte read for JSON identity and parsing, plus final streamed recheck."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.files = {}

    def location(self, value):
        p = Path(value)
        p = p if p.is_absolute() else self.root/p
        require(".." not in p.parts, "parent traversal is forbidden")
        try:
            relative = p.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("v622 finalization: dependency outside checkout") from error
        cursor = self.root
        for component in relative.parts:
            cursor = cursor/component
            require(not cursor.is_symlink(), "symlink dependency is forbidden")
        return p

    def path(self, value):
        p = self.location(value)
        require(p.is_file(), "missing evidence: "+str(p))
        return p

    @staticmethod
    def hash_file(path):
        h = sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                h.update(block)
        return h.hexdigest()

    def add(self, path, expected=None):
        p = self.path(path)
        actual = self.hash_file(p)
        require(expected is None or actual == expected, "SHA differs: "+str(p))
        key = str(p)
        require(key not in self.files or self.files[key] == actual, "evidence changed during finalization")
        self.files[key] = actual
        return {"path": key, "sha256": actual}

    def read(self, path, expected=None):
        p = self.path(path)
        raw = p.read_bytes()
        actual = sha256(raw).hexdigest()
        require(expected is None or actual == expected, "SHA differs: "+str(p))
        key = str(p)
        require(key not in self.files or self.files[key] == actual, "evidence changed during finalization")
        self.files[key] = actual
        return json.loads(raw), {"path": key, "sha256": actual}

    def bound(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid evidence pointer")
        return self.read(ref["path"], ref["sha256"])[0]

    def verify(self):
        for path, expected in self.files.items():
            require(self.hash_file(self.path(path)) == expected, "bound evidence changed: "+path)


def metric_row(result, arm):
    validation = (result.get("metrics") or {}).get("validation")
    row = {"arm": arm, "role": result["job"]["role"], "fold": result["job"]["fold"],
        "status": result["status"], "leaf_count": result.get("leaf_count"),
        "development_gate_passed": result["development_gate"]["passed"]}
    if validation is None:
        row.update(dict.fromkeys(FIELDS))
        return row
    f, e = validation["feedback"], validation["explanation"]
    row.update({"F_leaf_coverage": f.get("coverage"), "E_leaf_coverage": e.get("coverage"),
        "F_expected_coverage": f.get("expected_actionable_coverage"),
        "F_actual_coverage": f.get("actionable_coverage"), "E_actual_coverage": e.get("actionable_coverage"),
        "F_agreement": f.get("agreement"), "E_agreement": e.get("agreement"),
        "F_KL": f.get("kl"), "E_KL": e.get("kl"),
        "raw_illegal_count": e.get("whole_tree_raw_illegal_count"),
        "path_uncertified_count": e.get("whole_tree_path_uncertified_count")})
    for key in FIELDS:
        value = row[key]
        require(value is None or type(value) in (int, float) and math.isfinite(value) and value >= -1e-6,
            "invalid metric "+key)
    return row


def aggregate(rows):
    result = {}
    for arm in ("A16", "A32", "B32", "D32"):
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        result[arm] = {"role_folds": len(group), "completed_role_folds": sum(r["status"] == "completed" for r in group),
            "mean_scope": "unweighted_mean_of_available_role_folds_not_pooled_frames",
            "metrics": {}}
        for key in FIELDS:
            values = [r[key] for r in group if r[key] is not None]
            result[arm]["metrics"][key] = {"available_role_folds": len(values),
                "minimum": min(values) if values else None,
                "mean": math.fsum(values)/len(values) if values else None,
                "maximum": max(values) if values else None}
    return result


def compare(rows):
    baseline = {(r["role"], r["fold"]): r for r in rows if r["arm"] == "A16"}
    result = []
    for row in rows:
        if row["arm"] == "A16":
            continue
        old = baseline[(row["role"], row["fold"])]
        result.append({"arm": row["arm"], "role": row["role"], "fold": row["fold"],
            "difference_current_minus_A16": {key: row[key]-old[key]
                if row[key] is not None and old[key] is not None else None for key in FIELDS}})
    return result


def check_source_identity(evidence, plan):
    identity = plan["identity"]
    sources = identity["source_files_sha256"]
    require(isinstance(sources, dict) and sources and digest(sources) == identity["source_closure_sha256"],
        "source closure mapping differs")
    for path, expected in sources.items():
        evidence.add(path, expected)
    return identity["source_closure_sha256"]


def completed_batch(evidence, plan_path, stage):
    plan, plan_ref = evidence.read(plan_path)
    folder = Path(plan_ref["path"]).parent
    require(plan.get("schema") == "warehouse-v622-capacity-plan.v1" and plan.get("stage") == stage
        and plan.get("candidate_id") == v.CANDIDATE_ID and plan.get("jobs") == v.jobs(stage)
        and plan.get("folds") == v.grouped_folds() and plan.get("gates") == v.GATES
        and plan.get("limits") == v.LIMITS and plan.get("budgets") == v.round_budget_contract(),
        "unregistered batch plan")
    require(all(type(plan.get(k)) is int and plan[k] == 0 for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))
        and plan.get("publishable") is False and plan.get("pilot_opened") is False
        and plan.get("scientific_retry_permitted") is False, "plan grants unrelated experiment authority")
    require(evidence.read(folder/"plan.claim.json")[0] == {"sha256": plan_ref["sha256"], "single_use": True},
        "plan claim differs")
    require(not (folder/"interrupted.json").exists(), "interrupted batch cannot supply selection")
    source = check_source_identity(evidence, plan)
    summary, summary_ref = evidence.read(folder/"summary.json")
    audit, audit_ref = evidence.read(folder/"independent-audit.json")
    require(summary.get("schema") == "warehouse-v622-capacity-summary.v1"
        and summary.get("candidate_id") == v.CANDIDATE_ID and summary.get("stage") == stage
        and summary.get("plan_sha256") == plan_ref["sha256"], "summary plan/stage differs")
    require(audit.get("schema") == "warehouse-v622-independent-capacity-audit.v1" and audit.get("passed") is True
        and audit.get("candidate_id") == v.CANDIDATE_ID and audit.get("stage") == stage
        and audit.get("plan_sha256") == plan_ref["sha256"] and audit.get("summary_sha256") == summary_ref["sha256"]
        and audit.get("source_closure_sha256") == source and audit.get("job_count") == len(plan["jobs"]),
        "complete independent audit is absent or mismatched")
    require(all(type(audit.get(k)) is int and audit[k] == 0 for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))
        and audit.get("publishable") is False and audit.get("pilot_opened") is False, "audit grants unrelated experiment authority")
    require(set(summary["jobs"]) == {j["job_id"] for j in plan["jobs"]}, "summary job family is incomplete")
    audit_jobs = audit.get("jobs", [])
    require(len(audit_jobs) == len(plan["jobs"]) and {r["job"]["job_id"] for r in audit_jobs} == set(summary["jobs"])
        and {r["job"]["job_id"]: r["job"] for r in audit_jobs} == {j["job_id"]: j for j in plan["jobs"]}
        and all(r.get("passed") is True for r in audit_jobs), "independent audit job family is incomplete")
    results, rows, calls = [], [], []
    for job in plan["jobs"]:
        jobdir = folder/"jobs"/job["job_id"]
        require(not (jobdir/"failed.json").exists(), "failed job cannot enter finalization")
        result = evidence.bound(summary["jobs"][job["job_id"]])
        require(Path(summary["jobs"][job["job_id"]]["path"]) == jobdir/"result.json", "cross-batch result pointer")
        completion = evidence.read(jobdir/"completed.json")[0]
        started = evidence.read(jobdir/"started.json")[0]
        require(result.get("schema") == "warehouse-v622-capacity-result.v1" and result.get("job") == job
            and result.get("plan_sha256") == plan_ref["sha256"] and started.get("job") == job
            and started.get("plan_sha256") == plan_ref["sha256"]
            and completion.get("result_sha256") == summary["jobs"][job["job_id"]]["sha256"]
            and completion.get("plan_sha256") == plan_ref["sha256"]
            and result.get("status") in ("completed", "scientific_budget_exhausted")
            and result.get("actor_weights_unchanged") is True and result.get("excluded_frames") == 0
            and all(result.get(k) == 0 and type(result.get(k)) is int for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))
            and result.get("publishable") is False and result.get("pilot_opened") is False,
            "job/completion/authority binding differs")
        if result["status"] == "completed":
            gate = v.gate_metrics(result["metrics"]["validation"])["passed"]
            require(result["development_gate"]["passed"] is gate and completion.get("passed") is gate,
                "native completed gate differs")
            search = evidence.bound(result["search_report"])
            program = evidence.bound(result["program"])
            evidence.bound(result["structure"])
            if result.get("compiler_report"):
                evidence.bound(result["compiler_report"])
            n = integer(result["budget_audit"]["end_to_end_leaf_api_calls"], "completed actual calls")
            require(n == integer(search["leaf_optimization_calls"], "search calls")+
                integer(program["report"]["leaf_optimization_calls_outside_search"], "final calls"), "actual total calls differ")
            calls.append({"job_id": job["job_id"], "actual_leaf_api_calls": n, "lower_bound_only": False})
        else:
            require(result.get("program") is None and result.get("metrics") is None
                and result["development_gate"]["passed"] is False and completion.get("passed") is False,
                "incomplete scientific job claims program success")
            failure = result["budget_exhaustion"]
            search = evidence.bound(result["search_report"]) if result.get("search_report") else None
            if result["error_type"] == "ProgramBudgetExhausted":
                require(search is not None, "final timeout lost search report")
                n = integer(search["leaf_optimization_calls"], "completed search calls")+integer(failure["leaf_optimization_calls"], "attempted final calls")
            elif result["error_type"] == "SearchIncompleteError":
                n = integer(failure["leaf_optimization_calls"], "attempted search calls") if job["arm"] == "A32" else sum(
                    integer(failure[part]["leaf_optimization_calls"], "attempted "+part) for part in ("generation", "terminal_table"))
            else:
                raise RuntimeError("v622 finalization: unregistered incomplete job")
            calls.append({"job_id": job["job_id"], "actual_leaf_api_calls": n, "lower_bound_only": False,
                "status": result["status"]})
        require(result["budget_audit"]["passed"] is True, "native budget gate failed")
        audited_job = next(r for r in audit_jobs if r["job"]["job_id"] == job["job_id"])
        require(audited_job.get("development_gate_passed") is result["development_gate"]["passed"], "audit and native job gates differ")
        results.append(result)
        rows.append(metric_row(result, job["arm"]))
    recomputed = v.summarize_units(results, stage=stage)
    require(all(summary.get(k) == value for k, value in recomputed.items()), "summary selection arithmetic differs")
    return {"plan": plan, "summary": summary, "plan_ref": plan_ref, "summary_ref": summary_ref,
        "audit_ref": audit_ref, "folder": folder, "source": source, "rows": rows, "calls": calls}


def baseline(evidence, primary):
    refs = primary["plan"]["baseline_A16"]
    require(refs["summary"]["sha256"] == BASELINE_SUMMARY_SHA256
        and refs["independent_audit"]["sha256"] == BASELINE_AUDIT_SHA256, "A16 baseline differs from registered receipts")
    summary = evidence.bound(refs["summary"])
    audit = evidence.bound(refs["independent_audit"])
    require(audit.get("passed") is True and audit.get("summary_sha256") == refs["summary"]["sha256"]
        and summary.get("selected_arm") is None, "A16 failed baseline is not independently audited")
    rows = []
    for fold in range(3):
        for role in v.ROLES:
            name = f"fold-{fold}-{role}-A"
            result = evidence.bound(summary["jobs"][name])
            require(result["job"]["role"] == role and result["job"]["fold"] == fold
                and result["job"]["arm"] == "A" and result["status"] == "completed", "A16 role/fold differs")
            rows.append(metric_row(result, "A16"))
    return rows, refs


def historical_cost(evidence):
    interrupted, ref = evidence.read(INTERRUPTED_RELATIVE)
    require(interrupted.get("schema") == "warehouse-v622-engineering-interruption.v1"
        and interrupted.get("candidate_id") == v.CANDIDATE_ID and interrupted.get("complete_batch") is False
        and interrupted.get("scientific_method_failure_claimed") is False
        and interrupted.get("scientific_timeout_claimed") is False, "old 01 interruption receipt differs")
    plan = evidence.bound(interrupted["plan"])
    require(plan.get("candidate_id") == v.CANDIDATE_ID and plan.get("stage") == "primary", "interrupted parent differs")
    completed, partial = [], []
    for item in interrupted["jobs"]:
        if item.get("started"):
            evidence.bound(item["started"])
        if item.get("log"):
            evidence.add(item["log"]["path"], item["log"]["sha256"])
        if item["status"] == "completed_before_interruption":
            result = evidence.bound(item["result"])
            completion = evidence.bound(item["completion"])
            require(completion["result_sha256"] == item["result"]["sha256"] and result["job"] == item["job"]
                and result["plan_sha256"] == interrupted["plan"]["sha256"], "old completed cost receipt differs")
            n = integer(result["budget_audit"]["end_to_end_leaf_api_calls"], "old completed calls")
            completed.append({"job_id": item["job"]["job_id"], "actual_leaf_api_calls": n})
        elif item["status"] == "interrupted_without_search_checkpoint":
            require(item.get("result") is None and item.get("completion") is None, "old partial job is mislabeled")
            progress = item["last_logged_progress"]
            text = evidence.path(item["log"]["path"]).read_text()
            logged = re.findall(r"global_rounds=(\d+)/\d+.*?api_calls=(\d+)\s+convex_calls=(\d+)", text)
            require(logged and max(int(row[1]) for row in logged) == progress["leaf_api_calls"]
                and max(int(row[2]) for row in logged) == progress["convex_solver_calls"]
                and max(int(row[0]) for row in logged) == progress["global_rounds"], "old logged lower bound differs from pinned log")
            partial.append({"job_id": item["job"]["job_id"],
                "logged_leaf_api_calls_lower_bound": integer(progress["leaf_api_calls"], "old logged lower bound"),
                "logged_convex_solver_calls_lower_bound": integer(progress["convex_solver_calls"], "old logged solver lower bound")})
        else:
            raise RuntimeError("v622 finalization: unknown interrupted job status")
    require(len(completed) == interrupted.get("completed_jobs") == 9
        and len(partial) == interrupted.get("interrupted_jobs") == 3
        and {r["job"]["job_id"] for r in interrupted["jobs"]} == {j["job_id"] for j in v.jobs("primary")},
        "old interrupted job family differs")
    exact = sum(r["actual_leaf_api_calls"] for r in completed)
    lower = sum(r["logged_leaf_api_calls_lower_bound"] for r in partial)
    return {"interrupted_receipt": ref, "completed_job_count": 9, "interrupted_job_count": 3,
        "completed_actual_leaf_api_calls": exact, "interrupted_logged_leaf_api_calls_lower_bound": lower,
        "total_leaf_api_calls_lower_bound": exact+lower, "total_is_exact": False,
        "unlogged_in_flight_work": "unknown; no upper bound inferred", "completed_jobs": completed,
        "interrupted_jobs": partial, "used_for_current_selection": False}, Path(ref["path"]).parent


def build_report(primary_plan, fallback_plan=None, *, report_id, root=ROOT):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", report_id) is not None, "unsafe report ID")
    evidence = Evidence(root)
    reporting_source = evidence.add("scripts/finalize_warehouse_v622_development.py")
    primary = completed_batch(evidence, primary_plan, "primary")
    require(primary["folder"] == evidence.root/v.OUTPUT_ROOT/"capacity-02", "only the explicit complete current 02 primary supplies selection")
    fallback = None
    if primary["summary"]["selected_arm"] is None:
        require(fallback_plan is not None, "failed primary requires complete audited fallback before finalization")
        fallback = completed_batch(evidence, fallback_plan, "fallback")
        parent = fallback["plan"].get("fallback_parent", {})
        require(parent.get("summary") == primary["summary_ref"] and parent.get("independent_audit") == primary["audit_ref"]
            and fallback["source"] == primary["source"] and fallback["plan"]["identity"] == primary["plan"]["identity"]
            and fallback["folder"] == evidence.root/v.OUTPUT_ROOT/"fallback-02", "fallback is not bound to the same failed primary batch")
    else:
        require(fallback_plan is None, "passing primary must not be compared with fallback")
    old_rows, baseline_refs = baseline(evidence, primary)
    old_cost, historical_folder = historical_cost(evidence)
    selected = (fallback or primary)["summary"]["selected_arm"]
    closed = selected is None
    rows = sorted(old_rows+primary["rows"]+(fallback["rows"] if fallback else []),
        key=lambda r: (r["arm"], r["role"], r["fold"]))
    calls = primary["calls"]+(fallback["calls"] if fallback else [])
    current_total = sum(row["actual_leaf_api_calls"] for row in calls)
    report = {"schema": FINAL_SCHEMA, "candidate_id": v.CANDIDATE_ID, "report_id": report_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed_failed_program_precheck" if closed else "capacity_precheck_passed_next_fresh_gate_only",
        "reason": "program_precheck_failed" if closed else "complete_capacity_batch_passed",
        "selected_arm": selected, "selection_uses_only_explicit_complete_current_batches": True,
        "reporting_source": reporting_source, "reporting_source_is_outside_frozen_scientific_closure": True,
        "cross_batch_stitching": False, "current_primary": {k: primary[k] for k in ("plan_ref", "summary_ref", "audit_ref")},
        "current_fallback": {k: fallback[k] for k in ("plan_ref", "summary_ref", "audit_ref")} if fallback else None,
        "baseline_A16": baseline_refs, "source_closure_sha256": primary["source"],
        "rows": rows, "aggregates": aggregate(rows), "paired_differences_from_A16": compare(rows),
        "method_gates": {arm: {"complete_role_folds": sum(r["status"] == "completed" for r in rows if r["arm"] == arm),
            "all_role_fold_gates_passed": all(r["status"] == "completed" and r["development_gate_passed"] for r in rows if r["arm"] == arm)}
            for arm in dict.fromkeys(r["arm"] for r in rows)},
        "stage_gates": {"capacity_primary": primary["summary"]["selected_arm"] is not None,
            "capacity_fallback": fallback["summary"]["selected_arm"] is not None if fallback else None,
            "fresh_confirmation_completed": False, "pressure_completed": False,
            "paired_PPO_completed": False, "sealed_program_audit_completed": False, "heldout_completed": False},
        "next_stage": "closed" if closed else "fresh_confirmation_requires_new_authorization",
        "eligible_to_request_fresh_authorization": not closed,
        "old_quality_not_new_evaluation": {"source": "carried_original_75_episode_v620_development_receipt_not_new_rollout",
            "episodes": 75, "total_deliveries": 1120, "mean_deliveries": 1120/75,
            "total_collisions": 213, "mean_collisions": 213/75, "shutdown_episodes": 1,
            "shutdown_episode_rate": 1/75},
        "compute": {"interrupted_01": old_cost, "current_02": {"jobs": calls,
            "actual_leaf_api_calls": current_total, "total_is_exact": True},
            "all_v622_leaf_api_calls_lower_bound": old_cost["total_leaf_api_calls_lower_bound"]+current_total,
            "all_v622_total_is_exact": False},
        "evidence_scope": "spent development episode arithmetic; not fresh, sealed, heldout, or human-study evidence",
        **ZERO}
    folders = [primary["folder"], historical_folder]+([fallback["folder"]] if fallback else [])
    for folder in folders:
        for file in sorted(folder.rglob("*")):
            require(not file.is_symlink(), "symlink in archive payload directory")
            if file.is_file():
                evidence.add(file)
    evidence.verify()
    return report, evidence, folders


def markdown(report):
    def percent(x): return "—" if x is None else f"{100*x:.2f}%"
    def number(x): return "—" if x is None else f"{x:.4f}"
    lines = ["# Warehouse v6.22 开发阶段最终报告", "", f"候选：`{report['candidate_id']}`。", "",
        "结论："+("程序预检未通过，候选关闭。" if report["selected_arm"] is None else
        f"`{report['selected_arm']}`通过本阶段；后续 fresh 确认仍需独立授权。"), "",
        "只使用显式指定且完整独立审计通过的当前批次；旧01中断批次不参与选择，也不与02拼接。", "",
        "配送14.93、碰撞2.84、停机1/75是沿用的旧75局开发质量，不是本轮新评估。", "",
        "## 两角色、三折实测指标", "",
        "| 方法 | 角色 | 折 | F叶覆盖 | E叶覆盖 | 预期F | 实际F | 实际E | F一致率 | E一致率 | F KL | E KL | 非法/未证路径 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        lines.append("| "+" | ".join([row["arm"], row["role"], str(row["fold"]),
            *(percent(row[k]) for k in FIELDS[:7]), number(row["F_KL"]), number(row["E_KL"]),
            f"{row['raw_illegal_count']}/{row['path_uncertified_count']}"])+" |")
    lines += ["", "## 各方法最小值／平均值（可用角色折，非合并帧）", "",
        "| 方法 | F叶覆盖 | E叶覆盖 | 预期F | 实际F | E一致率 | E KL |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, group in report["aggregates"].items():
        values = []
        for key in ("F_leaf_coverage", "E_leaf_coverage", "F_expected_coverage", "F_actual_coverage", "E_agreement", "E_KL"):
            m = group["metrics"][key];fmt = number if key == "E_KL" else percent
            values.append(fmt(m["minimum"])+" / "+fmt(m["mean"]))
        lines.append("| "+" | ".join([arm]+values)+" |")
    cost = report["compute"]
    old = cost["interrupted_01"]
    lines += ["", "科学预算失败没有可计算模型的指标记为缺失，不回填0；均值只描述可用折，准入仍要求全部6个角色折通过。",
        "相同角色、相同折与A16的逐项差值保存在JSON；覆盖差值为比例差，乘100后为百分点。", "",
        "## 计算开销", "",
        f"旧01已完成9作业实际叶API：{old['completed_actual_leaf_api_calls']}；另3作业日志下界：{old['interrupted_logged_leaf_api_calls_lower_bound']}。",
        f"当前02完整批次实际叶API：{cost['current_02']['actual_leaf_api_calls']}。",
        f"本轮累计叶API至少 {cost['all_v622_leaf_api_calls_lower_bound']}；旧中断作业未落盘的在途工作未知，累计值不是精确总数。", "",
        "BC后续阶段、fresh确认、压力评估、PPO、封存审计与留出测试均未执行；pilot及发布保持关闭。",
        "归档仅生成路径与SHA清单，未复制资产或修改权限。目录权限不构成不可变保证。"]
    return "\n".join(lines)+"\n"


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False);f.write("\n")


def finalize(primary_plan, fallback_plan=None, *, report_id, root=ROOT):
    report, evidence, folders = build_report(primary_plan, fallback_plan, report_id=report_id, root=root)
    output = evidence.root/v.OUTPUT_ROOT
    directory = output/"finalizations"/report_id
    closed_path = output/"candidate.closed.json"
    evidence.location(directory)
    evidence.location(closed_path)
    require(not directory.exists() and not directory.is_symlink(), "report output already exists")
    require(not closed_path.exists() and not closed_path.is_symlink(), "candidate already has a closure receipt")
    directory.parent.mkdir(parents=True, exist_ok=True)
    require(not directory.parent.is_symlink(), "report parent is a symlink")
    directory.mkdir(exist_ok=False)
    write_new(directory/"final.json", report)
    write_new(directory/"final.zh.md", markdown(report))
    final_ref = evidence.add(directory/"final.json")
    markdown_ref = evidence.add(directory/"final.zh.md")
    payload = [{"source_path": path, "relative_path": str(Path(path).relative_to(evidence.root)),
        "sha256": expected, "bytes": Path(path).stat().st_size} for path, expected in sorted(evidence.files.items())]
    manifest = {"schema": "warehouse-v622-development-archive-payload-manifest.v1",
        "candidate_id": v.CANDIDATE_ID, "report_id": report_id, "payload": payload,
        "payload_file_count": len(payload), "payload_bytes": sum(r["bytes"] for r in payload),
        "selected_completed_output_directories": [str(p) for p in folders],
        "manifest_is_not_a_copy_or_publication": True, "copy_performed": False, "permissions_changed": False,
        "self_contained": False, "old_v620_model_and_collection_dependencies_retained_in_place": True,
        "control_receipts_excluded_from_self_hash": ["archive-manifest.json", "completed.json", "candidate.closed.json"], **ZERO}
    write_new(directory/"archive-manifest.json", manifest)
    manifest_ref = evidence.add(directory/"archive-manifest.json")
    evidence.verify()
    closure_ref = None
    if report["selected_arm"] is None:
        write_new(closed_path, {"schema": "warehouse-v622-candidate-closed.v1", "candidate_id": v.CANDIDATE_ID,
            "reason": "program_precheck_failed", "selected_arm": None, "final_report": final_ref,
            "archive_manifest": manifest_ref, "scientific_method_or_threshold_changed": False, **ZERO})
        closure_ref = evidence.add(closed_path)
    write_new(directory/"completed.json", {"schema": "warehouse-v622-finalization-completed.v1",
        "final_report": final_ref, "markdown": markdown_ref, "archive_manifest": manifest_ref,
        "candidate_closed": closure_ref, **ZERO})
    return {"report": final_ref, "markdown": markdown_ref, "archive_manifest": manifest_ref,
        "candidate_closed": closure_ref, "selected_arm": report["selected_arm"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-plan", type=Path, required=True)
    parser.add_argument("--fallback-plan", type=Path)
    parser.add_argument("--report-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.primary_plan, args.fallback_plan, report_id=args.report_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
