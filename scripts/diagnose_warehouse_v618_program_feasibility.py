#!/usr/bin/env python3
"""Fixed-NN, unsealed program feasibility diagnostics; never admission evidence."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import role_sha256
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from backend.training.warehouse_v6_program import (
    INDEPENDENT_SELECTIVE_METHOD, _passes, FEEDBACK_AGREEMENT,
    EXPLANATION_AGREEMENT, fit_program,
)
from experiments.warehouse import protocol as p
from experiments.warehouse.admission import NativeBackend, _exclusive_json

SPLITS = ("fit", "calibration", "eligibility", "validation")
SPLIT_COUNTS = (50, 25, 25, 24)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def fixed_seed_plan(fit_base=263600, audit_base=263800):
    """Check every full scene list before an environment is created."""
    if any(type(value) is not int or value < 0 for value in (fit_base, audit_base)):
        raise ValueError("diagnostic bases must be nonnegative integers")
    cursor, splits = fit_base, {}
    for split, count in zip(SPLITS, SPLIT_COUNTS, strict=True):
        splits[split] = list(range(cursor, cursor+count))
        cursor += count
    audit = list(range(audit_base, audit_base+24))
    all_seeds = [seed for values in splits.values() for seed in values] + audit
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("diagnostic fit/calibration/eligibility/validation/audit overlap")
    if any(is_sealed_scenario_seed(seed) for seed in all_seeds):
        raise ValueError("program feasibility diagnostic overlaps sealed scenarios")
    return {"splits": splits, "audit": audit,
            "split_seed_sha256": {name: digest(seeds) for name, seeds in splits.items()},
            "split_episode_ids": {name: list(map(str, seeds)) for name, seeds in splits.items()},
            "split_episode_ids_sha256": {
                name: digest(list(map(str, seeds))) for name, seeds in splits.items()},
            "audit_seed_sha256": digest(audit),
            "all_seeds_sha256": digest(all_seeds),
            "sealed_seed_overlap_count": 0, "purpose_overlap_count": 0}


def metric_gates(feedback, explanation):
    """Report the registered BC and development definitions separately."""
    qualified = min(float(feedback["coverage"]), float(explanation["coverage"]))
    expected = float(feedback["expected_actionable_coverage"])
    fidelity = all(
        metric["rows"] > 0 and metric["agreement"] >= threshold
        and metric["kl"] is not None and metric["kl"] <= .25
        and metric["illegal_count"] == 0 and metric.get("path_uncertified_count", 0) == 0
        and type(metric.get("whole_tree_raw_illegal_count")) is int
        and metric["whole_tree_raw_illegal_count"] == 0
        and type(metric.get("whole_tree_path_uncertified_count")) is int
        and metric["whole_tree_path_uncertified_count"] >= 0
        for metric, threshold in ((feedback, .93), (explanation, .95)))
    return {"bc_precheck_35_30": {
                "qualified_leaf_coverage": qualified,
                "expected_feedback_actionable_coverage": expected,
                "passed": bool(qualified >= .35 and expected >= .30 and fidelity)},
            "development_30_realized": {
                "feedback_actionable_coverage": feedback["actionable_coverage"],
                "explanation_actionable_coverage": explanation["actionable_coverage"],
                "passed": bool(_passes(feedback, FEEDBACK_AGREEMENT, development=True, require_whole_tree_zero=True)
                               and _passes(explanation, EXPLANATION_AGREEMENT, development=True, require_whole_tree_zero=True))},
            "feedback": feedback, "explanation": explanation}


def _write_or_check(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise RuntimeError(f"immutable diagnostic receipt changed: {path}")
    else:
        _exclusive_json(path, value)


def _actor_identity(actor):
    return {role: role_sha256(actor.roles[role].weights,
                              model_version=actor.roles[role].model_version)
            for role in p.ROLES}


def _restore_split(trace):
    path = Path(trace["path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != trace["sha256"] or not raw.endswith(b"\n"):
        raise RuntimeError("saved feasibility trajectory bytes changed")
    lines = [json.loads(line) for line in raw.splitlines()]
    if len(lines) != trace["row_count"]:
        raise RuntimeError("saved feasibility trajectory frame count changed")
    rows = {role: [] for role in p.ROLES}
    seen = []
    for line in lines:
        seed = int(line["base_seed"])
        if not seen or seen[-1] != seed:
            seen.append(seed)
        for role in p.ROLES:
            claim = line["roles"][role]
            rows[role].append({**claim,
                "obs": np.asarray(claim["obs"], np.float32),
                "mask": np.asarray(claim["mask"], np.float32),
                "episode": str(seed), "base_seed": seed,
                "episode_id": int(line["episode_id"]), "frame": int(line["frame"])})
    if seen != trace["seeds"]:
        raise RuntimeError("saved feasibility scene order changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-json", required=True,
                        help="One frozen artifact {path,sha256,role_sha256}; legacy or current")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-base-seed", type=int, default=263600)
    parser.add_argument("--audit-base-seed", type=int, default=263800)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    seeds = fixed_seed_plan(args.fit_base_seed, args.audit_base_seed)
    artifact = json.loads(Path(args.actor_json).read_bytes())
    if not isinstance(artifact, dict):
        parser.error("actor-json must contain a single frozen artifact object")
    artifact = {**artifact, "path": str(Path(artifact["path"]).resolve())}
    backend = NativeBackend()
    actor = backend.load_actor(artifact)
    before = _actor_identity(actor)
    source = backend.source_closure_sha256()
    script_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output_dir).resolve()
    plan = {"schema": "warehouse-v618-program-feasibility-plan.v1",
            "purpose": "unsealed_fixed_nn_program_diagnostic_not_admission",
            "publishable": False, "sealed_evidence": False,
            "source_closure_sha256": source, "script_sha256": script_sha,
            "actor": artifact, "actor_model_version": actor.model_version,
            "actor_role_sha256": before,
            "actor_feature_dim": {role: actor.roles[role].feature_dim for role in p.ROLES},
            "method": INDEPENDENT_SELECTIVE_METHOD, "seed_plan": seeds,
            "collection": "natural_stochastic_nn_nn_all_physical_frames_no_deduplication",
            "program_eligibility_locked_before_validation": True,
            "programs_locked_before_diagnostic_audit": True,
            "audit_cannot_change_program_or_select_a_different_actor": True}
    _write_or_check(output / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"plan_path": str(output / "plan.json"),
                          "publishable": False, "rollouts_started": False}), flush=True)
        return 0

    def guard():
        if (backend.source_closure_sha256() != source
                or sha256(Path(__file__).read_bytes()).hexdigest() != script_sha
                or sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]
                or _actor_identity(actor) != before or dict(actor.role_sha256) != before):
            raise RuntimeError("source or frozen Actor changed during feasibility diagnostic")

    parts, collection_receipts = {}, {}
    for split in SPLITS:
        guard()
        receipt_path = output / "collection" / f"{split}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if (receipt["source_closure_sha256"] != source
                    or receipt["actor_sha256"] != actor.artifact_sha256
                    or receipt["trace"]["seeds"] != seeds["splits"][split]):
                raise RuntimeError("saved feasibility collection binding changed")
            rows = _restore_split(receipt["trace"])
        else:
            rows, authorization = runner.collect_authorized_program_rows(
                actor, seeds["splits"][split])
            trace = runner._write_final_fit_trace(rows, split=split,
                seeds=seeds["splits"][split], trace_root=output / "collection" / "traces",
                actor_sha256=actor.artifact_sha256)
            receipt = {"source_closure_sha256": source,
                       "actor_sha256": actor.artifact_sha256,
                       "trace": trace, "authorization": authorization,
                       "per_role_physical_frame_count": {role: len(rows[role]) for role in p.ROLES},
                       "excluded_frames": 0}
            guard()
            _write_or_check(receipt_path, receipt)
        parts[split] = rows
        collection_receipts[split] = receipt
        print(json.dumps({"collected": split,
                          "episodes": len(seeds["splits"][split]),
                          "physical_frames": len(rows[p.ROLES[0]])}), flush=True)

    programs, program_artifacts, program_reports = {}, {}, {}
    for role in p.ROLES:
        guard()
        item_path = output / "programs" / f"{role}.artifact.json"
        if item_path.exists():
            item = json.loads(item_path.read_bytes())
            program = backend.load_program(item, before[role])
        else:
            program = fit_program(role, actor, *(parts[name][role] for name in SPLITS),
                                  method=INDEPENDENT_SELECTIVE_METHOD)
            target = output / "programs" / f"{role}.program.json"
            if target.exists():
                existing = backend.load_program({"path": str(target),
                    "sha256": sha256(target.read_bytes()).hexdigest()}, before[role])
                if existing.sha256() != program.sha256():
                    raise RuntimeError("interrupted fit left a different frozen program")
            else:
                program.save(target)
            item = {"path": str(target), "sha256": sha256(target.read_bytes()).hexdigest(),
                    "actor_role_sha256": before[role]}
            _write_or_check(item_path, item)
        programs[role] = backend.load_program(item, before[role])
        if programs[role].sha256() != program.sha256():
            raise RuntimeError("program save/load changed contents")
        program_artifacts[role] = item
        validation = program.report["validation"]
        program_reports[role] = {"validation": metric_gates(
            validation["feedback"], validation["explanation"]), "report": program.report}
        guard()
        print(json.dumps({"fitted": role,
            "validation_gates": {key: value for key, value in
                program_reports[role]["validation"].items() if key not in ("feedback", "explanation")}}), flush=True)

    guard()
    audit_binding = {"source_closure_sha256": source,
                     "actor_sha256": actor.artifact_sha256,
                     "program_sha256": {role: programs[role].sha256() for role in p.ROLES},
                     "scenario_seeds": seeds["audit"],
                     "publishable": False, "sealed_evidence": False}
    audit_path = output / "fixed-audit.json"
    started_path = output / "fixed-audit.started.json"
    if audit_path.exists():
        audit_result = json.loads(audit_path.read_bytes())
        if audit_result["binding"] != audit_binding:
            raise RuntimeError("saved fixed audit or frozen program binding changed")
        audits = audit_result["audits"]
    else:
        if started_path.exists():
            raise RuntimeError("diagnostic audit was interrupted; keep evidence and use a new plan")
        _write_or_check(started_path, audit_binding)
        audits = backend.audit_programs(actor, programs, seeds["audit"],
                                        trace_dir=output / "audit-traces")
        guard()
        _write_or_check(audit_path, {"binding": audit_binding, "audits": audits})
    trace_bytes = {role: Path(audits[role]["trace_path"]).read_bytes() for role in p.ROLES}
    trace_verified = backend.verify_program_trace(actor=actor, programs=programs,
        trace_bytes_by_role=trace_bytes, audits=audits, seeds=seeds["audit"])
    if trace_verified is not True:
        raise RuntimeError("diagnostic frozen NN and physical audit trace disagree")
    guard()
    result = {"schema": "warehouse-v618-program-feasibility-result.v1",
              "purpose": plan["purpose"], "publishable": False, "sealed_evidence": False,
              "model_acceptance_passed": False,
              "plan_sha256": sha256((output / "plan.json").read_bytes()).hexdigest(),
              "source_closure_sha256": source, "actor": artifact,
              "actor_role_sha256_before": before, "actor_role_sha256_after": _actor_identity(actor),
              "actor_weights_unchanged": True, "program_artifacts": program_artifacts,
              "collection_receipts": collection_receipts,
              "roles": {role: {**program_reports[role], "fixed_diagnostic_audit":
                  metric_gates(audits[role]["feedback"], audits[role]["explanation"])} for role in p.ROLES},
              "audit_receipt_sha256": sha256(audit_path.read_bytes()).hexdigest(),
              "fixed_audit_trace_verified": trace_verified,
              "scope": "single_fixed_policy_feasibility_not_training_method_effect_or_release_acceptance"}
    _write_or_check(output / "result.json", result)
    print(json.dumps({"completed": True, "publishable": False,
                      "result_path": str(output / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
