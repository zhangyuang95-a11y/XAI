#!/usr/bin/env python3
"""Posthoc refit on frozen diagnostic caches; zero rollouts and no admission."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.warehouse_v6_program import INDEPENDENT_SELECTIVE_METHOD, fit_program
from backend.training.warehouse_v618_selective_tree import SEARCH_BUDGET, SEARCH_VARIANTS
from experiments.warehouse import protocol as p
from experiments.warehouse.admission import NativeBackend, _exclusive_json
from scripts.diagnose_warehouse_v618_program_feasibility import (
    SPLITS, _actor_identity, _restore_split, metric_gates,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                        help="Completed fixed-NN feasibility diagnostic with four saved traces")
    parser.add_argument("--output-dir", required=True,
                        help="New isolated directory; existing nonempty directories are refused")
    args = parser.parse_args(argv)
    source_dir, output = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if (output == source_dir or output.is_relative_to(source_dir)
            or source_dir.is_relative_to(output)):
        raise RuntimeError("cached refit requires an isolated output outside the original diagnostic")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("cached refit output must be empty; old programs cannot be overwritten")
    old_plan_raw = (source_dir / "plan.json").read_bytes()
    old_result_raw = (source_dir / "result.json").read_bytes()
    old_plan, old_result = json.loads(old_plan_raw), json.loads(old_result_raw)
    if (old_plan.get("publishable") is not False
            or old_plan.get("sealed_evidence") is not False
            or old_result.get("publishable") is not False
            or old_result.get("actor") != old_plan.get("actor")
            or old_result.get("plan_sha256") != sha256(old_plan_raw).hexdigest()):
        raise RuntimeError("cached refit requires an intact completed unsealed diagnostic")
    backend = NativeBackend()
    artifact = old_result["actor"]
    actor = backend.load_actor(artifact)
    before = _actor_identity(actor)
    source = backend.source_closure_sha256()
    script_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    parts, receipts = {}, {}
    for split in SPLITS:
        path = source_dir / "collection" / f"{split}.json"
        raw = path.read_bytes()
        receipt = json.loads(raw)
        trace = receipt["trace"]
        if (receipt["actor_sha256"] != artifact["sha256"]
                or trace["seeds"] != old_plan["seed_plan"]["splits"][split]
                or receipt.get("excluded_frames") != 0):
            raise RuntimeError("cached refit Actor, scene schedule or full-frame receipt differs")
        parts[split] = _restore_split(trace)
        receipts[split] = {"receipt_path": str(path),
                           "receipt_sha256": sha256(raw).hexdigest(),
                           "trace": trace,
                           "per_role_rows": {role: len(parts[split][role]) for role in p.ROLES}}
    plan = {"schema": "warehouse-v618-posthoc-cache-refit-plan.v1",
            "purpose": "search_implementation_diagnostic_not_fresh_method_validation",
            "publishable": False, "admission_evidence": False, "sealed_evidence": False,
            "actor": artifact, "actor_role_sha256": before,
            "source_closure_sha256": source, "script_sha256": script_sha,
            "original_plan_path": str(source_dir / "plan.json"),
            "original_plan_sha256": sha256(old_plan_raw).hexdigest(),
            "original_result_sha256": sha256(old_result_raw).hexdigest(),
            "method": INDEPENDENT_SELECTIVE_METHOD,
            "search_budget": dict(SEARCH_BUDGET), "search_variants": SEARCH_VARIANTS,
            "seed_plan": old_plan["seed_plan"], "collection_receipts": receipts,
            "previously_inspected_validation_is_not_new_acceptance_evidence": True,
            "original_fixed_diagnostic_audit_reused": False,
            "environment_resets": 0, "training_steps": 0}
    _exclusive_json(output / "plan.json", plan)

    def guard():
        if (backend.source_closure_sha256() != source
                or sha256(Path(__file__).read_bytes()).hexdigest() != script_sha
                or _actor_identity(actor) != before
                or sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]):
            raise RuntimeError("source or frozen Actor changed during cached refit")

    roles = {}
    try:
        for role in p.ROLES:
            guard()
            program = fit_program(role, actor, *(parts[name][role] for name in SPLITS),
                                  method=INDEPENDENT_SELECTIVE_METHOD)
            path = output / f"{role}.program.json"
            digest = program.save(path)
            restored = backend.load_program({"path": str(path), "sha256": digest}, before[role])
            if restored.sha256() != program.sha256():
                raise RuntimeError("cached refit program save/load changed contents")
            guard()
            validation = program.report["validation"]
            roles[role] = {"program_artifact": {"path": str(path), "sha256": digest,
                "actor_role_sha256": before[role]}, "report": program.report,
                "diagnostic_gates": metric_gates(validation["feedback"], validation["explanation"])}
            _exclusive_json(output / f"{role}.result.json", roles[role])
            print(json.dumps({"role": role, "coverage": validation["feedback"]["coverage"],
                              "expected_actionable": validation["feedback"].get("expected_actionable_coverage"),
                              "publishable": False}), flush=True)
        result = {"schema": "warehouse-v618-posthoc-cache-refit-result.v1",
                  "purpose": plan["purpose"], "publishable": False,
                  "admission_evidence": False, "sealed_evidence": False,
                  "source_closure_sha256": source,
                  "plan_sha256": sha256((output / "plan.json").read_bytes()).hexdigest(),
                  "environment_resets": 0, "training_steps": 0,
                  "original_fixed_diagnostic_audit_reused": False,
                  "actor_role_sha256_before": before,
                  "actor_role_sha256_after": _actor_identity(actor), "roles": roles}
        _exclusive_json(output / "result.json", result)
    except Exception as error:
        _exclusive_json(output / "failure.json", {
            "publishable": False, "admission_evidence": False,
            "source_closure_sha256": source, "environment_resets": 0,
            "error": f"{type(error).__name__}: {error}"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
