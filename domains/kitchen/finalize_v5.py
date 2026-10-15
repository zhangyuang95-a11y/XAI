"""Finish an already trained Kitchen v5 pair without retraining or reusing tests.

The in-loop RCPD programs remain immutable training evidence.  Separate,
bounded programs extracted from a frozen Actor on training-only scenes may be
used for explanation after one independent final-program validation.  The
held-out receipt is created exclusively before any held-out episode runs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from core.program import ExecutableProgram
from . import neural_policy as p
from .v5_training import (DEVELOPMENT_START, FINAL_PROGRAM_VALIDATION_SEEDS,
                          FINAL_PROGRAM_VALIDATION_START, HELDOUT_START,
                          JOINT_STEPS, PAIR_SEEDS, RELEASE_ID, ROOT,
                          SCENARIO_BLOCK_STEPS, _heldout_already_used,
                          _passes, _quality, _write_json,
                          final_role_program_development_audit,
                          scenario_schedule_digest)

OUTPUT = ROOT / "output/kitchen_v5_matched_candidates"
BASE_REPORT = OUTPUT / "kitchen_dual_nn_v5_pair_report.json"
FINAL_REPORT = OUTPUT / "kitchen_dual_nn_v5_reextracted_pair_report.json"
VALIDATION_RECEIPT = OUTPUT / "kitchen_dual_nn_v5_final_program_validation_receipt.json"
HELDOUT_RECEIPT = OUTPUT / "kitchen_dual_nn_v5_heldout_receipt.json"


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _verify_training(base):
    if (tuple(base.get("pair_seeds", ())) != PAIR_SEEDS
            or len(base.get("pairs", ())) != 3
            or base.get("heldout") or base.get("heldout_evaluated")):
        raise RuntimeError("Kitchen matched training report is incomplete or held-out was already evaluated")
    pairs = {pair.get("seed"): pair for pair in base["pairs"]}
    if set(pairs) != set(PAIR_SEEDS):
        raise RuntimeError("Kitchen paired seeds do not match the protocol")
    for seed, pair in pairs.items():
        digest = scenario_schedule_digest(seed)
        if (pair.get("scenario_seed_schedule_sha256") != digest
                or pair.get("scenario_block_steps") != SCENARIO_BLOCK_STEPS
                or not pair.get("development_passed")):
            raise RuntimeError(f"Pair {seed} fails the common schedule or development gate")
        for variant in ("raw", "regularized"):
            run = pair["variants"][variant]
            frozen = p.FrozenActors(ROOT / run["actor_path"], run["actor_sha256"])
            feedback = run.get("feedback_rows") or {}
            if (run.get("joint_steps") != JOINT_STEPS
                    or run.get("expanded_scenario_seed_sha256") != digest
                    or run.get("role_sha256") != frozen.role_sha256
                    or frozen.role_sha256["human"] == frozen.role_sha256["ai"]
                    or run.get("post_policy_override_count") != 0
                    or any(float(run.get("actor_l2_change", {}).get(role, 0)) <= 0
                           for role in ("human", "ai"))
                    or (variant == "raw" and feedback != {"human": 0, "ai": 0})
                    or (variant == "regularized" and any(feedback.get(role, 0) <= 0
                                                         for role in ("human", "ai")))
                    or not _passes(run["development"], 20)):
                raise RuntimeError(f"Pair {seed} {variant} fails training evidence")
        if (pair["variants"]["raw"]["actor_sha256"]
                == pair["variants"]["regularized"]["actor_sha256"]):
            raise RuntimeError(f"Pair {seed} variants share Actor weights")
    return pairs


def _program_source(actor, run, role, path):
    source = Path(path)
    program = ExecutableProgram.load_json(source)
    metadata = program.metadata
    feedback = run["programs"][role]
    feedback_path = ROOT / feedback["path"]
    if (_sha(feedback_path) != feedback["sha256"]
            or not feedback.get("feedback_eligible")
            or metadata.get("domain") != "kitchen"
            or metadata.get("role") != role
            or metadata.get("training_source") != "final_frozen_actor_nn_nn_training_only_reextraction"
            or metadata.get("training_seed_start") != 60_000
            or metadata.get("training_seed_count") != 20
            or metadata.get("actor_sha256") != actor.sha256
            or metadata.get("role_sha256") != actor.role_sha256[role]
            or metadata.get("runtime_control_allowed") is not False
            or not (metadata.get("metrics") or {}).get("feedback_eligible")
            or program.root.depth() > 5 or program.root.leaf_count() > 16):
        raise RuntimeError(f"Invalid {role} frozen-Actor program source")
    destination = OUTPUT / f"kitchen_dual_nn_v5_pair{run['pair_seed']}_regularized_{role}_final_frozen_actor_program.json"
    provenance = {**metadata,
                  "program_phase": "final_frozen_actor_training_only_reextraction",
                  "reextraction_source_sha256": _sha(source),
                  "training_feedback_program_sha256": feedback["sha256"],
                  "training_feedback_program_path": feedback["path"]}
    final = replace(program, metadata=provenance)
    final.save_json(destination)
    return final, {"path": str(destination.relative_to(ROOT)), "sha256": _sha(destination),
                   "feedback_eligible": True, "explanation_eligible": True,
                   "provenance": "final_frozen_actor_training_only_reextraction",
                   "source_sha256": _sha(source),
                   "training_feedback_program_sha256": feedback["sha256"]}


def validate_final_programs(human_program, ai_program):
    base_bytes = BASE_REPORT.read_bytes()
    base = json.loads(base_bytes)
    pairs = _verify_training(base)
    if _heldout_already_used(OUTPUT) or HELDOUT_RECEIPT.exists():
        raise RuntimeError("Kitchen held-out set has already been used")
    # Model quality alone selects the best pair.  The tie-break is deterministic.
    best = max(pairs.values(), key=lambda pair: (pair["worst_development_successes"], -pair["seed"]))
    if best["seed"] != 7:
        raise RuntimeError("The supplied final programs do not belong to the best development pair")
    selected = best["variants"]["regularized"]
    actor = p.FrozenActors(ROOT / selected["actor_path"], selected["actor_sha256"])
    final = {}
    descriptions = {}
    for role, source in (("human", human_program), ("ai", ai_program)):
        final[role], descriptions[role] = _program_source(actor, selected, role, source)
    programs = {role: (final[role], descriptions[role]["sha256"])
                for role in ("human", "ai")}
    # Training-only check is free to tune.  The untouched admission block below
    # is consumed just once, after this candidate is fixed and hashed.
    internal = final_role_program_development_audit(actor, programs,
                                                     seed_start=60_100, seed_count=20)
    if not all(internal[role]["passed"] for role in ("human", "ai")):
        raise RuntimeError("Final programs fail training-only internal validation")
    started = {"release_id": RELEASE_ID, "status": "started",
               "selected_pair_seed": best["seed"], "actor_sha256": actor.sha256,
               "role_sha256": actor.role_sha256,
               "program_sha256": {role: descriptions[role]["sha256"] for role in programs},
               "development_seed_start": FINAL_PROGRAM_VALIDATION_START,
               "development_seed_count": FINAL_PROGRAM_VALIDATION_SEEDS,
               "original_pair_report_sha256": sha256(base_bytes).hexdigest()}
    # A started receipt deliberately prevents repeating a partly run audit.
    with VALIDATION_RECEIPT.open("x") as handle:
        json.dump(started, handle, indent=2, sort_keys=True)
        handle.write("\n")
    independent = final_role_program_development_audit(
        actor, programs, seed_start=FINAL_PROGRAM_VALIDATION_START,
        seed_count=FINAL_PROGRAM_VALIDATION_SEEDS)
    passed = all(independent[role]["passed"] for role in ("human", "ai"))
    validation = {**started, "status": "completed", "passed": passed,
                  "training_only_internal": internal,
                  "final_role_program_development_audit": independent,
                  "explanation_programs": descriptions,
                  "training_feedback_program_sha256": {
                      role: selected["programs"][role]["sha256"] for role in programs}}
    receipt_sha = _write_json(VALIDATION_RECEIPT, validation)
    if not passed:
        return validation
    report = deepcopy(base)
    report["release_id"] = RELEASE_ID
    report["original_pair_report_path"] = str(BASE_REPORT.relative_to(ROOT))
    report["original_pair_report_sha256"] = sha256(base_bytes).hexdigest()
    report["final_program_validation_receipt_path"] = str(VALIDATION_RECEIPT.relative_to(ROOT))
    report["final_program_validation_receipt_sha256"] = receipt_sha
    chosen = next(pair for pair in report["pairs"] if pair["seed"] == best["seed"])
    chosen_run = chosen["variants"]["regularized"]
    chosen_run["training_feedback_program_development_audit"] = chosen_run[
        "final_role_program_development_audit"]
    chosen_run["explanation_programs"] = descriptions
    chosen_run["final_role_program_development_audit"] = independent
    chosen["programs_eligible"] = True
    report["selected_pair_seed"] = chosen["seed"]
    report["heldout"] = {}
    report["passed"] = False
    _write_json(FINAL_REPORT, report)
    return validation


def _pinned_report(source, entry, variant):
    target = ROOT / entry["actor_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / source["actor_path"]).read_bytes())
    entry["actor_sha256"] = _sha(target)
    entry["role_sha256"] = source["role_sha256"]
    if variant == "regularized":
        for role, destination in zip(("human", "ai"), entry["program_paths"]):
            program = source["explanation_programs"][role]
            program_target = ROOT / destination
            program_target.parent.mkdir(parents=True, exist_ok=True)
            program_target.write_bytes((ROOT / program["path"]).read_bytes())
            entry["program_sha256"][role] = _sha(program_target)
    report = {key: value for key, value in source.items() if key != "program_state"}
    entry["training_report_sha256"] = _write_json(ROOT / entry["training_report_path"], report)


def evaluate_heldout_and_pin():
    validation_bytes = VALIDATION_RECEIPT.read_bytes()
    validation = json.loads(validation_bytes)
    report = json.loads(FINAL_REPORT.read_text())
    manifest_path = ROOT / "configs/study_v3_kitchen_v5.json"
    manifest = json.loads(manifest_path.read_text())
    acceptance_path = ROOT / manifest["policies"]["raw"]["acceptance_path"]
    if acceptance_path.exists() and json.loads(acceptance_path.read_text()).get("passed"):
        raise RuntimeError("Refusing to modify an already accepted Kitchen v5 release")
    selected_seed = report.get("selected_pair_seed")
    chosen = next((pair for pair in report["pairs"] if pair["seed"] == selected_seed), None)
    if (validation.get("status") != "completed" or validation.get("passed") is not True
            or chosen is None or selected_seed not in PAIR_SEEDS
            or report.get("final_program_validation_receipt_sha256") != sha256(validation_bytes).hexdigest()
            or chosen["variants"]["regularized"].get("final_role_program_development_audit")
            != validation["final_role_program_development_audit"]
            or not all(validation["final_role_program_development_audit"][role]["passed"]
                       for role in ("human", "ai"))):
        raise RuntimeError("Both final frozen-Actor programs must pass untouched validation first")
    selected = chosen["variants"]
    for variant in ("raw", "regularized"):
        p.FrozenActors(ROOT / selected[variant]["actor_path"], selected[variant]["actor_sha256"])
    for role in ("human", "ai"):
        program = selected["regularized"]["explanation_programs"][role]
        if _sha(ROOT / program["path"]) != program["sha256"]:
            raise RuntimeError("Final program changed after validation")
    if HELDOUT_RECEIPT.exists():
        receipt = json.loads(HELDOUT_RECEIPT.read_text())
        if (receipt.get("status") != "completed"
                or receipt.get("selected_pair_seed") != selected_seed
                or receipt.get("actor_sha256") != {
                    variant: selected[variant]["actor_sha256"]
                    for variant in ("raw", "regularized")}):
            raise RuntimeError("Kitchen held-out receipt is incomplete or bound to different Actors")
        heldout = receipt["heldout"]
    else:
        if _heldout_already_used(OUTPUT) or report.get("heldout"):
            raise RuntimeError("Kitchen held-out set has already been used")
        started = {"release_id": RELEASE_ID, "status": "started",
                   "selected_pair_seed": selected_seed,
                   "seed_start": HELDOUT_START, "seed_count": 100,
                   "actor_sha256": {variant: selected[variant]["actor_sha256"]
                                    for variant in ("raw", "regularized")},
                   "final_program_validation_receipt_sha256": sha256(validation_bytes).hexdigest()}
        # The exclusive receipt is written before the first held-out rollout.
        with HELDOUT_RECEIPT.open("x") as handle:
            json.dump(started, handle, indent=2, sort_keys=True)
            handle.write("\n")
        heldout = {variant: _quality(p.FrozenActors(ROOT / selected[variant]["actor_path"]),
                                     HELDOUT_START, 100)
                   for variant in ("raw", "regularized")}
        _write_json(HELDOUT_RECEIPT, {**started, "status": "completed", "heldout": heldout})
    receipt_sha = _sha(HELDOUT_RECEIPT)
    passed = all(_passes(heldout[variant], 100) for variant in ("raw", "regularized"))
    report["heldout"] = heldout
    report["heldout_receipt_sha256"] = receipt_sha
    report["passed"] = passed
    _write_json(FINAL_REPORT, report)
    if not passed:
        return {"passed": False, "heldout_evaluated": True, "heldout": heldout}
    for variant in ("raw", "regularized"):
        _pinned_report(selected[variant], manifest["policies"][variant], variant)
    entries = manifest["policies"]
    acceptance = {"release_id": RELEASE_ID, "passed": False,
                  "pre_audit_quality_passed": True, "heldout_evaluated": True,
                  "selected_pair_seed": selected_seed, "pair_seeds": list(PAIR_SEEDS),
                  "scenario_block_steps": SCENARIO_BLOCK_STEPS,
                  "expanded_scenario_seed_sha256": chosen["scenario_seed_schedule_sha256"],
                  "joint_steps": {variant: selected[variant]["joint_steps"]
                                  for variant in ("raw", "regularized")},
                  "development": {variant: selected[variant]["development"]
                                  for variant in ("raw", "regularized")},
                  "heldout": heldout,
                  "artifact_sha256": {variant: entries[variant]["actor_sha256"]
                                      for variant in ("raw", "regularized")},
                  "training_report_sha256": {variant: entries[variant]["training_report_sha256"]
                                             for variant in ("raw", "regularized")},
                  "program_sha256": {"regularized": entries["regularized"]["program_sha256"]},
                  "training_feedback_program_sha256": validation["training_feedback_program_sha256"],
                  "final_program_source": "final_frozen_actor_training_only_reextraction",
                  "original_pair_report_sha256": report["original_pair_report_sha256"],
                  "final_program_validation_receipt_sha256": sha256(validation_bytes).hexdigest(),
                  "heldout_receipt_sha256": receipt_sha,
                  "final_role_program_development_audit": validation["final_role_program_development_audit"],
                  "final_ai_program_development_audit": validation["final_role_program_development_audit"]["ai"],
                  "raw_program_feedback_rows": selected["raw"]["feedback_rows"],
                  "regularized_program_feedback_rows": selected["regularized"]["feedback_rows"],
                  "post_policy_override_count": 0,
                  "verification_errors": ["independent_audit_pending"]}
    _write_json(acceptance_path, acceptance)
    _write_json(manifest_path, manifest)
    # Complete the hash and six-run evidence audit in the same command.  The
    # audit reads the saved held-out receipt and never reruns held-out scenes.
    from .audit_v5 import audit
    verified = audit(FINAL_REPORT)
    if not verified["passed"]:
        raise RuntimeError(f"Kitchen v5 independent audit failed: {verified['verification_errors']}")
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "heldout"))
    parser.add_argument("--human-program", default=str(
        OUTPUT / "probe_pair7_human_geometric_all_features.json"))
    parser.add_argument("--ai-program", default=str(
        OUTPUT / "probe_pair7_ai_geometric_all_features.json"))
    args = parser.parse_args()
    result = (validate_final_programs(args.human_program, args.ai_program)
              if args.stage == "validate" else evaluate_heldout_and_pin())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
