#!/usr/bin/env python3
"""Audited batch02 reporting adapter; never changes scientific selection.

Both modes require every registered job and the independent batch arithmetic
audit before reading their results for reporting.  Posthoc calculations reuse
cached physical frames; no Actor inference, sampling, fitting or update occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.warehouse as warehouse_package
from experiments.warehouse import v622_recovery_protocol as recovery
from experiments.warehouse import v622_completeness as completeness
from scripts import run_warehouse_v622_capacity_recovery as runner
from scripts import report_warehouse_v622_metrics as metrics
from scripts import report_warehouse_v622_posthoc as posthoc


@contextmanager
def recovery_scope():
    """Scope all actual import styles used by the two unchanged reporters.

    Posthoc imports its protocol and loader inside its functions, so the package
    attribute alias is necessary in addition to the metrics module's globals.
    Existing references to the original module are not modified.  Every alias
    is restored on success or failure; this adapter runs in its own process.
    """
    old_package_v = warehouse_package.v622_protocol
    old_metrics_v, old_metrics_loader = metrics.v, metrics.load_plan
    with runner.recovery_scope():
        warehouse_package.v622_protocol = recovery
        metrics.v, metrics.load_plan = recovery, runner.base.load_plan
        try:
            yield
        finally:
            warehouse_package.v622_protocol = old_package_v
            metrics.v, metrics.load_plan = old_metrics_v, old_metrics_loader


def _source_bindings():
    paths = (Path(__file__).resolve(), Path(metrics.__file__).resolve(), Path(posthoc.__file__).resolve(),
             Path(completeness.__file__).resolve())
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare_report(mode, plan_paths):
    if mode not in ("metrics", "posthoc"):
        raise ValueError("reporting mode must be metrics or posthoc")
    plan_paths = list(plan_paths)
    if not plan_paths or (mode == "metrics" and len(plan_paths) != 1):
        raise ValueError("metrics requires one plan; posthoc accepts primary plus optional fallback")
    sources = _source_bindings()
    with recovery_scope():
        evidence = posthoc.Evidence()
        requested_path = Path(plan_paths[0]).resolve()
        completion_paths = plan_paths
        # A fallback snapshot still depends on the complete audited primary.
        # The paired posthoc mode requires both plans explicitly, whereas a
        # single fallback metrics request adds its canonical primary parent.
        if mode == "metrics" and requested_path.parent == recovery.output_directory("fallback"):
            completion_paths = [recovery.output_directory("primary") / "plan.json", requested_path]
        plans = posthoc.completed_inputs(completion_paths, evidence)
        family_receipts = {}
        for path, plan, results in plans:
            for result in results:
                if result["job"]["arm"] != "D32":
                    continue
                ref = result.get("search_report")
                if ref is not None:
                    expected = path.parent / "jobs" / result["job"]["job_id"] / "search-report.json"
                    if Path(ref["path"]).resolve() != expected.resolve():
                        raise RuntimeError("D32 reporting search belongs to another job")
                    search = evidence.pointer(ref)
                else:
                    search = result.get("budget_exhaustion")
                family_receipts[result["job"]["job_id"]] = completeness.require_complete_d32_search(search)
        if mode == "metrics":
            report = metrics.snapshot(requested_path)
            if report.get("independent_batch_arithmetic_audit_passed") is not True:
                raise RuntimeError("recovery metrics require the completed independent batch audit")
        else:
            report = posthoc.prepare_report(plan_paths)
        evidence.verify()
    if sources != _source_bindings():
        raise RuntimeError("reporting adapter source changed while preparing report")
    report["recovery_reporting"] = {
        "schema": "warehouse-v622-recovery-report-binding.v1", "mode": mode,
        "primary_batch_id": recovery.BATCH_ID, "fallback_batch_id": recovery.FALLBACK_BATCH_ID,
        "plans": [{"path": str(path), "stage": plan["stage"],
                   "sha256": sha256(Path(path).read_bytes()).hexdigest(),
                   "source_closure_sha256": plan["identity"]["source_closure_sha256"]}
                  for path, plan, _ in plans],
        "complete_independent_audit_required": True,
        "D32_four_complete_templates_required": True,
        "D32_protocol_completeness": family_receipts,
        "used_for_selection": False, "actor_quality_change_claimed": False,
        "old_quality_is_carried_provenance_not_new_evaluation": True,
        "new_environment_resets": 0, "new_NN_forward_calls": 0,
        "fit_calls": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "reporting_sources_sha256": sources,
        "completion_gate_evidence_sha256": dict(sorted(evidence.hashes.items())),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metrics", "posthoc"), required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--report-id", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.report_id):
        parser.error("report-id must be a safe lowercase identifier")
    if args.mode == "metrics" and len(args.plan) != 1:
        parser.error("metrics accepts exactly one plan; posthoc accepts primary and optional fallback")
    report = prepare_report(args.mode, args.plan)
    folder = ROOT / recovery.OUTPUT_ROOT / "reports" / args.report_id
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / (args.mode + ".json")).open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    renderer = metrics.markdown if args.mode == "metrics" else posthoc.markdown
    with (folder / (args.mode + ".zh.md")).open("x") as handle:
        handle.write(renderer(report))
    print(json.dumps({"report": str(folder / (args.mode + ".zh.md")),
        "mode": args.mode, "used_for_selection": False, "new_environment_resets": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
