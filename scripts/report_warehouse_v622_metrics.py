#!/usr/bin/env python3
"""Read-only metrics snapshot; no fitting, NN inference, rollouts or authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_protocol as v
from scripts.run_warehouse_v622_capacity import load_plan, read_json_bound


def row(result, label):
    m = (result.get("metrics") or {}).get("validation", {})
    f, e = m.get("feedback", {}), m.get("explanation", {})
    return {"arm": label, "role": result["job"]["role"], "fold": result["job"]["fold"],
        "status": result["status"], "leaf_count": result.get("leaf_count"),
        "F_leaf_coverage": f.get("coverage"), "E_leaf_coverage": e.get("coverage"),
        "F_actual_coverage": f.get("actionable_coverage"), "E_actual_coverage": e.get("actionable_coverage"),
        "F_expected_coverage": f.get("expected_actionable_coverage"),
        "F_agreement": f.get("agreement"), "E_agreement": e.get("agreement"),
        "F_KL": f.get("kl"), "E_KL": e.get("kl"),
        "raw_illegal_count": e.get("whole_tree_raw_illegal_count"),
        "path_uncertified_count": e.get("whole_tree_path_uncertified_count"),
        "development_gate_passed": result["development_gate"]["passed"],
        "per_nn_action": e.get("per_nn_action", {}), "public_strata": e.get("public_strata", {})}


def snapshot(plan_path):
    path, plan = load_plan(plan_path)
    rows, states, references = [], [], {}
    old = read_json_bound(plan["baseline_A16"]["summary"])
    for name, ref in old["jobs"].items():
        if name.endswith("-A"):
            rows.append(row(read_json_bound(ref), "A16"))
            references["A16:"+name] = ref
    for job in plan["jobs"]:
        folder = path.parent/"jobs"/job["job_id"]
        if (folder/"failed.json").is_file():
            states.append({"job": job, "status": "engineering_failed"})
        elif (folder/"completed.json").is_file():
            completion = json.loads((folder/"completed.json").read_bytes())
            ref = {"path": str(folder/"result.json"), "sha256": completion["result_sha256"]}
            result = read_json_bound(ref)
            if result["job"] != job or result["plan_sha256"] != sha256(path.read_bytes()).hexdigest():
                raise RuntimeError("metrics snapshot job/plan binding differs")
            rows.append(row(result, job["arm"]))
            references[job["job_id"]] = ref
            states.append({"job": job, "status": result["status"]})
        else:
            states.append({"job": job, "status": "running" if (folder/"started.json").exists() else "not_started"})
    independent = path.parent/"independent-audit.json"
    audited = False
    if independent.exists():
        audit = json.loads(independent.read_bytes())
        audited = (audit.get("passed") is True and (path.parent/"summary.json").is_file()
            and audit.get("summary_sha256") == sha256((path.parent/"summary.json").read_bytes()).hexdigest())
    return {"schema": "warehouse-v622-development-metrics-snapshot.v1", "candidate_id": v.CANDIDATE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": plan["stage"],
        "plan_sha256": sha256(path.read_bytes()).hexdigest(), "actor_sha256": v.ACTOR_SHA256,
        "actor_role_sha256": v.ROLE_SHA256, "independent_batch_arithmetic_audit_passed": audited,
        "quality": {"source": "carried_original_75_episode_v620_development_receipt_not_new_rollout",
            "mean_deliveries": 1120/75, "mean_collisions": 213/75, "shutdown_episodes": 1, "episodes": 75},
        "rows": sorted(rows, key=lambda r: (r["arm"], r["role"], r["fold"])),
        "jobs": states, "result_references": references, "new_environment_resets": 0,
        "actor_updates": 0, "PPO_joint_steps": 0, "pilot_opened": False,
        "admission_or_human_study_evidence": False}


def markdown(report):
    def pct(value): return "—" if value is None else f"{100*value:.2f}%"
    def number(value): return "—" if value is None else f"{value:.4f}"
    lines = ["# Warehouse v6.22 开发指标", "", f"候选：`{report['candidate_id']}`。阶段：`{report['stage']}`。", "",
        "完整 NN 权重不变。本阶段不新增环境 episode、不更新 Actor、不运行 PPO；pilot 关闭。", "",
        "配送14.93、碰撞2.84、停机1/75仅为沿用的旧开发质量收据，不是新评估。", "",
        "A16是同一275 episode三折上的旧M2抽取基线；不能与原单批程序约21%的覆盖混为同一对照。", "",
        f"本批独立算术审计完成：{'是' if report['independent_batch_arithmetic_audit_passed'] else '否；当前仅为 native 描述'}。", "",
        "| 方法 | 角色 | 折 | 叶 | F叶覆盖 | E叶覆盖 | 实际F | 预期F | E一致率 | E KL | 预检 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['arm']} | {r['role']} | {r['fold']} | {r['leaf_count']} | {pct(r['F_leaf_coverage'])} | {pct(r['E_leaf_coverage'])} | {pct(r['F_actual_coverage'])} | {pct(r['F_expected_coverage'])} | {pct(r['E_agreement'])} | {number(r['E_KL'])} | {'通过' if r['development_gate_passed'] else '未通过'} |")
    lines += ["", "门槛：F/E叶覆盖各45%，预期F40%，F/E一致率95%/97%，KL≤.20；全树原始非法动作及未证明路径均须为0。", "",
        "叶覆盖、实际反馈覆盖、预期反馈覆盖分别列出；合格区域的一致率不能代表未覆盖区域。", "",
        "## 作业状态", ""]
    lines += [f"- `{r['job']['job_id']}`：`{r['status']}`" for r in report["jobs"]]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.report_id):
        parser.error("report-id must be a safe lowercase identifier")
    report = snapshot(args.plan)
    folder = ROOT/v.OUTPUT_ROOT/"reports"/args.report_id
    folder.mkdir(parents=True, exist_ok=False)
    with (folder/"metrics.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    with (folder/"metrics.zh.md").open("x") as f:
        f.write(markdown(report))
    print(json.dumps({"report": str(folder/"metrics.zh.md"), "complete_rows": len(report["rows"]),
        "independent_audit_passed": report["independent_batch_arithmetic_audit_passed"]}))


if __name__ == "__main__":
    main()
