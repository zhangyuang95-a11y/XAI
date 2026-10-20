#!/usr/bin/env python3
"""Seal all twelve independently audited v6.24 stage-A results exactly once.

This module may write only development reports and a closed or pending receipt.
It cannot generate new episodes, authorize PPO, consume sealed data or publish.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_warehouse_v624_upgrade as independent

CANDIDATE = independent.CANDIDATE
AUTHORITY = dict(
    independent.ZERO, fresh_development_episodes=0,
    fresh_pressure_trajectories=0, formal_training_authorized=False,
    release_generated=False,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def pointer(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(),
            "missing or symlink finalization evidence: " + str(p))
    return {"path": str(p.resolve()), "sha256": sha256(p.read_bytes()).hexdigest()}


def bound(ref, expected_path=None):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
            "malformed finalization evidence pointer")
    require(expected_path is None or ref["path"] == str(Path(expected_path).resolve()),
            "finalization evidence path mismatch")
    require(pointer(ref["path"]) == ref, "finalization evidence SHA mismatch")
    return json.loads(Path(ref["path"]).read_bytes())


def closed_or_pending(summary):
    """A selected arm only permits review for the separate confirmation gate."""
    require(summary.get("candidate_id") == CANDIDATE and summary.get("stage") == "A",
            "foreign stage/candidate summary")
    arms = summary.get("arms", {})
    require(set(arms) == set(independent.ARMS), "final decision needs both arms")
    for arm in independent.ARMS:
        rows = arms[arm].get("role_folds")
        require(isinstance(rows, list) and len(rows) == 6
                and {r.get("job_id") for r in rows}
                == {j["job_id"] for j in independent.expected_jobs() if j["arm"] == arm},
                "partial or duplicate arm cannot be finalized")
        require(all(type(r.get("passed")) is bool
                    and r.get("status") in ("completed", "scientific_budget_exhausted")
                    and (r["status"] == "completed" or r["passed"] is False)
                    for r in rows), "engineering interruption cannot be scientifically finalized")
        passed = all(r["status"] == "completed" and r["passed"] for r in rows)
        require(type(arms[arm].get("all_six_pass")) is bool
                and arms[arm]["all_six_pass"] == passed, "arm gate is not six individual passes")
    passing = [a for a in independent.ARMS if arms[a]["all_six_pass"]]
    if passing:
        require(all(isinstance(arms[a].get("selection_score"), list)
                    and len(arms[a]["selection_score"]) == 4
                    and all(independent.finite(v) for v in arms[a]["selection_score"])
                    for a in passing), "qualifying arm lacks registered ranking")
        selected = max(passing, key=lambda a: (arms[a]["selection_score"], a == "P0"))
    else:
        selected = None
    require(summary.get("selected_arm") == selected
            and summary.get("next_stage") == (
                "fresh_confirmation" if selected else "closed_failed_program_precheck"),
            "selected method/next-stage does not match all twelve results")
    return {
        "status": "pending_fresh_confirmation" if selected else "closed_failed_program_precheck",
        "candidate_closed": selected is None, "selected_method": selected,
    }


def verify_saved_audit(saved, live, plan_ref, summary_ref, auditor_ref):
    """A saved ``passed`` flag alone never provides finalization authority."""
    require(saved.get("schema") == "warehouse-v624-independent-audit.v1"
            and saved.get("candidate_id") == CANDIDATE and saved.get("stage") == "A"
            and saved.get("passed") is True and saved.get("summary_verified") is True,
            "missing complete independent development audit")
    independent.zero(saved)
    independent.zero(live)
    require(saved.get("audit_script") == auditor_ref
            and saved.get("plan_sha256") == plan_ref["sha256"]
            and saved.get("summary_sha256") == summary_ref["sha256"],
            "saved auditor/plan/summary identity differs")
    for key in ("source_sha256", "jobs", "independently_reconstructed_summary",
                "independent_action_stream_checks", "evidence_sha256"):
        require(saved.get(key) == live.get(key), "saved/live independent audit differs: " + key)
    require(saved["evidence_sha256"].get(plan_ref["path"]) == plan_ref["sha256"]
            and saved["evidence_sha256"].get(summary_ref["path"]) == summary_ref["sha256"],
            "audit does not bind the final plan and summary")


def compute_totals(jobs):
    """Sum observed actual calls, including failed scientific-budget units."""
    totals = {arm: {
        "search_leaf_api_calls": 0, "final_leaf_api_calls": 0,
        "total_new_leaf_api_calls": 0, "search_convex_solver_calls": 0,
        "final_convex_solver_calls": 0,
    } for arm in independent.ARMS}
    for row in jobs:
        arm, ledger = row["job"]["arm"], row["ledger"]
        mapping = {
            "search_leaf_api_calls": "new_leaf_calls",
            "final_leaf_api_calls": "final_leaf_calls",
            "total_new_leaf_api_calls": "total_new_leaf_calls",
            "search_convex_solver_calls": "convex_solver_calls",
            "final_convex_solver_calls": "final_convex_solver_calls",
        }
        for target, source in mapping.items():
            require(independent.integer(ledger.get(source)), "incomplete observed call ledger")
            totals[arm][target] += ledger[source]
        require(ledger["total_new_leaf_calls"]
                == ledger["new_leaf_calls"] + ledger["final_leaf_calls"] <= 20000,
                "observed call ledger exceeds per-job budget")
    return totals


def metrics_csv(jobs):
    output = StringIO()
    fields = ["arm", "job_id", "fold", "role", "status", "scientific_gate_passed",
              "leaf_count", "feedback_coverage", "explanation_coverage",
              "expected_feedback_coverage", "feedback_agreement", "explanation_agreement",
              "feedback_kl", "explanation_kl", "whole_tree_raw_illegal_count",
              "whole_tree_path_uncertified_count", "search_leaf_api_calls",
              "final_leaf_api_calls", "total_new_leaf_api_calls",
              "result_path", "result_sha256"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in jobs:
        row = dict(record["job"], status=record["status"],
                   scientific_gate_passed=record["gate_passed"], leaf_count=record["leaf_count"],
                   search_leaf_api_calls=record["ledger"]["new_leaf_calls"],
                   final_leaf_api_calls=record["ledger"]["final_leaf_calls"],
                   total_new_leaf_api_calls=record["ledger"]["total_new_leaf_calls"],
                   result_path=record["result"]["path"], result_sha256=record["result"]["sha256"])
        if record["status"] == "completed":
            f, e = (record["validation"][p] for p in ("feedback", "explanation"))
            row.update(feedback_coverage=f["coverage"], explanation_coverage=e["coverage"],
                       expected_feedback_coverage=f["expected_actionable_coverage"],
                       feedback_agreement=f["agreement"], explanation_agreement=e["agreement"],
                       feedback_kl=f["kl"], explanation_kl=e["kl"],
                       whole_tree_raw_illegal_count=e["whole_tree_raw_illegal_count"],
                       whole_tree_path_uncertified_count=e["whole_tree_path_uncertified_count"])
        writer.writerow(row)
    return output.getvalue().encode()


def report_markdown(summary, jobs, outcome, totals):
    lines = [
        "# Warehouse v6.24 阶段 A 结果", "",
        f"候选：`{CANDIDATE}`；状态：`{outcome['status']}`。",
        f"选中方法：`{outcome['selected_method']}`。",
        "",
        "全部结果来自已花费的 275 个开发 episode 及原三折分区。"
        "v6.20 双 Actor 保持冻结；本次没有新的配送、碰撞或停机评估。"
        "程序审计通过表示算术和证据一致，不等于科学门槛通过。",
        "",
        "| 方法 | 完整预检通过 | 平均解释覆盖 | 最差解释覆盖 | 最差预期反馈覆盖 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in independent.ARMS:
        records = [r for r in jobs if r["job"]["arm"] == arm]
        complete = all(r["status"] == "completed" for r in records)
        if complete:
            coverage = [r["validation"]["explanation"]["coverage"] for r in records]
            feedback = [r["validation"]["feedback"]["expected_actionable_coverage"] for r in records]
            values = f"{statistics.mean(coverage):.4%} | {min(coverage):.4%} | {min(feedback):.4%}"
        else:
            values = "未完成完整科学拟合 | — | —"
        lines.append(f"| {arm} | {sum(r['gate_passed'] for r in records)}/6 | {values} |")
    lines += [
        "",
        "均值为六个角色／折的非加权算术均值，仅供描述。"
        "每个角色／折独立要求反馈与解释叶覆盖≥45%、预期反馈覆盖≥40%、"
        "反馈／解释一致率≥95%／97%、KL≤0.20、原始非法动作及未证明路径为零。",
        "",
        "| 作业 | 解释叶覆盖 | 预期反馈覆盖 | 解释一致率 | 解释 KL | 门槛 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in jobs:
        if record["status"] == "completed":
            f, e = (record["validation"][p] for p in ("feedback", "explanation"))
            kl = f"{e['kl']:.6f}" if independent.finite(e["kl"]) else "无合格区域"
            values = (f"{e['coverage']:.4%} | {f['expected_actionable_coverage']:.4%} | "
                      f"{e['agreement']:.4%} | {kl}")
        else:
            values = "科学预算耗尽 | — | — | —"
        lines.append(f"| {record['job']['job_id']} | {values} | "
                     f"{'通过' if record['gate_passed'] else '未通过'} |")
    lines += [
        "", "实际计算量（搜索与最终叶拟合均计入）：", "",
        "| 方法 | 搜索叶 API | 最终叶 API | 合计 API | 搜索凸求解 | 最终凸求解 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, counts in totals.items():
        lines.append(f"| {arm} | " + " | ".join(str(counts[k]) for k in (
            "search_leaf_api_calls", "final_leaf_api_calls", "total_new_leaf_api_calls",
            "search_convex_solver_calls", "final_convex_solver_calls")) + " |")
    lines += [
        "", "新开发 episode=0、新压力轨迹=0、Actor 更新=0、PPO 联合步=0；"
        "封存审计、留出测试与 pilot 未启动，未生成发布包。",
        "若有方法通过，本收据只锁定方法供下一阶段独立确认；"
        "它本身不授权生成新数据、PPO 或发布。",
    ]
    return ("\n".join(lines) + "\n").encode()


def finalize(plan_path):
    from experiments.warehouse import v624_protocol as v

    path = Path(plan_path).resolve(strict=True)
    destination = path.parent / "finalization"
    require(not destination.exists(), "single-use stage-A finalization already exists")
    plan_ref = pointer(path)
    plan = bound(plan_ref)
    require(plan.get("candidate_id") == CANDIDATE and plan.get("stage") == "A"
            and plan.get("source_sha256") == v.source_sha(), "finalization source/candidate differs")
    own = pointer(__file__)
    auditor_ref = pointer(ROOT / "scripts/audit_warehouse_v624_upgrade.py")
    summary_ref = pointer(path.parent / "summary.json")
    summary = bound(summary_ref)
    saved_ref = pointer(path.parent / "independent-audit.json")
    saved = bound(saved_ref)
    # Recompute all frozen data statistics and ledger guards; no fitting, writes
    # or environment transitions occur in this call.
    live = independent.audit(path)
    verify_saved_audit(saved, live, plan_ref, summary_ref, auditor_ref)
    independent.math.assert_equal(summary, live["independently_reconstructed_summary"],
                                  "finalization.independent_summary")
    jobs = live["jobs"]
    independent.validate_job_set([r["job"] for r in jobs])
    outcome = closed_or_pending(summary)
    totals = compute_totals(jobs)
    csv_bytes = metrics_csv(jobs)
    md_bytes = report_markdown(summary, jobs, outcome, totals)
    source_evidence = independent.Evidence()
    for file, expected in live["evidence_sha256"].items():
        source_evidence.raw(file, expected)
    source_evidence.raw(saved_ref["path"], saved_ref["sha256"])
    require(v.source_sha() == plan["source_sha256"] and pointer(__file__) == own
            and pointer(auditor_ref["path"]) == auditor_ref, "source changed before finalization")
    manifest = dict(
        AUTHORITY, schema="warehouse-v624-development-evidence-manifest.v1",
        candidate_id=CANDIDATE, stage="A", source_sha256=plan["source_sha256"],
        source_files_sha256=plan["source_files"],
        source_snapshot_root=str(path.parent / "source" / plan["source_sha256"]),
        actor=plan["actor"], actor_role_sha256=plan["actor_role_sha256"],
        baseline=plan["baseline"], initial_S1=plan["initial_S1"],
        plan=plan_ref, summary=summary_ref, independent_audit=saved_ref,
        auditor=auditor_ref, finalizer=own,
        evidence_sha256=live["evidence_sha256"],
        independently_verified_jobs=jobs, actual_compute_totals_by_arm=totals,
        scope="already spent development only; no new quality or human collaboration evidence",
    )
    # All checks finish before making the single-use output directory. An I/O
    # interruption thereafter remains visibly partial, never silently retried.
    destination.mkdir(parents=False, exist_ok=False)
    csv_path, md_path = destination / "metrics.csv", destination / "report.zh.md"
    with csv_path.open("xb") as stream:
        stream.write(csv_bytes)
    with md_path.open("xb") as stream:
        stream.write(md_bytes)
    manifest["reports"] = {"csv": pointer(csv_path), "markdown": pointer(md_path)}
    manifest_bytes = canonical(manifest)
    manifest_sha = sha256(manifest_bytes).hexdigest()
    manifest_path = destination / f"evidence-manifest.{manifest_sha}.json"
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    receipt = dict(
        AUTHORITY, schema="warehouse-v624-development-finalization.v1",
        candidate_id=CANDIDATE, stage="A", source_sha256=plan["source_sha256"],
        release_id="policylens-three-domain-20261016.v6.0", **outcome,
        single_use=True, finalizer=own, evidence_manifest=pointer(manifest_path),
        all_arm_gates=[
            {"arm": arm, "all_six_pass": summary["arms"][arm]["all_six_pass"],
             "passed_role_folds": sum(r["passed"] for r in summary["arms"][arm]["role_folds"]),
             "required_role_folds": 6}
            for arm in independent.ARMS
        ],
        at_utc=datetime.now(timezone.utc).isoformat(),
    )
    # Verify inputs again immediately before the terminal receipt is committed.
    # If bytes changed while reports were written, leave a partial directory
    # without a success/closed terminal receipt.
    source_evidence.recheck()
    with (destination / "receipt.json").open("xb") as stream:
        stream.write(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    result = finalize(args.plan)
    print(json.dumps({k: result[k] for k in (
        "status", "candidate_closed", "selected_method", "evidence_manifest")}, sort_keys=True))


if __name__ == "__main__":
    main()
