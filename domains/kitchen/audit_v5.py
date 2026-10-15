"""Finalize Kitchen v5 acceptance from completed, immutable training evidence.

This audit does not re-use or re-run the held-out set.  It verifies the one
recorded evaluation and every pinned artifact hash before opening the pilot.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import numpy as np
import torch

from core.program import ExecutableProgram
from .neural_policy import FrozenActors
from .training import DualKitchenNetwork, bc_update, teacher_data
from .v5_training import (FINAL_PROGRAM_VALIDATION_SEEDS, FINAL_PROGRAM_VALIDATION_START,
                          HELDOUT_START, JOINT_STEPS, PAIR_SEEDS, RELEASE_ID, _write_json,
                          SCENARIO_BLOCK_STEPS,
                          initial_actor_digest, scenario_schedule_digest)

ROOT = Path(__file__).resolve().parents[2]


def _candidate_training_evidence(pair_report, errors):
    """Verify six saved runs and reconstruct the common BC fork per seed."""
    candidates = []
    teacher = None
    teacher_completed = None
    torch.set_num_threads(4)
    for pair in pair_report.get("pairs", ()):
        seed = pair["seed"]
        variants = pair.get("variants", {})
        if teacher is None:
            teacher, teacher_completed = teacher_data(seed_count=300)
        if pair.get("bc_rows") != len(teacher[0]) or pair.get("bc_teacher_completed") != teacher_completed:
            errors.append(f"{seed}_bc_data_mismatch")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        initial = DualKitchenNetwork(hidden=256)
        bc_update(initial, teacher, epochs=15, seed=seed)
        initial_digest = initial_actor_digest(initial)
        schedule_digest = scenario_schedule_digest(seed)
        if pair.get("initial_bc_actor_sha256") != initial_digest:
            errors.append(f"{seed}_initial_actor_hash_mismatch")
        if pair.get("scenario_seed_schedule_sha256") != schedule_digest:
            errors.append(f"{seed}_scenario_schedule_hash_mismatch")
        if pair.get("scenario_block_steps") != SCENARIO_BLOCK_STEPS:
            errors.append(f"{seed}_scenario_block_horizon_mismatch")
        candidate = {"pair_seed": seed, "joint_steps": {},
                     "shared_initial_bc_actor_sha256": initial_digest,
                     "initial_hash_provenance": "deterministic_BC_reconstruction_from_fixed_seed_and_saved_training_protocol",
                     "shared_scenario_seed_schedule_sha256": schedule_digest,
                     "schedule_contract": "global_joint_step//360 selects seed=17000+pair_seed*1000000+block and task=1+block%3; terminal scenes restart inside the same block",
                     "expanded_scenario_seed_sha256": {},
                     "actor_l2_change": {}, "program_feedback_rows": {},
                     "post_policy_override_count": 0, "actor_sha256": {},
                     "final_role_program_development_audit": variants.get("regularized", {}).get(
                         "final_role_program_development_audit")}
        for variant in ("raw", "regularized"):
            run = variants.get(variant, {})
            candidate["joint_steps"][variant] = run.get("joint_steps")
            candidate["expanded_scenario_seed_sha256"][variant] = run.get("expanded_scenario_seed_sha256")
            candidate["actor_l2_change"][variant] = run.get("actor_l2_change")
            candidate["program_feedback_rows"][variant] = run.get("feedback_rows")
            candidate["actor_sha256"][variant] = run.get("actor_sha256")
            if (run.get("pair_seed") != seed or run.get("joint_steps") != JOINT_STEPS
                    or run.get("action_overrides") != 0):
                errors.append(f"{seed}_{variant}_run_contract_failed")
            if (run.get("scenario_block_steps") != SCENARIO_BLOCK_STEPS
                    or run.get("expanded_scenario_seed_sha256") != schedule_digest):
                errors.append(f"{seed}_{variant}_expanded_step_schedule_mismatch")
            changes = run.get("actor_l2_change") or {}
            if any(float(changes.get(role, 0)) <= 0 for role in ("human", "ai")):
                errors.append(f"{seed}_{variant}_actor_not_updated")
            feedback = run.get("feedback_rows") or {}
            if (variant == "raw" and any(feedback.get(role) != 0 for role in ("human", "ai"))
                    or variant == "regularized" and any(feedback.get(role, 0) <= 0 for role in ("human", "ai"))):
                errors.append(f"{seed}_{variant}_feedback_contract_failed")
            try:
                path = ROOT / run["actor_path"]
                frozen = FrozenActors(path, run["actor_sha256"])
                if frozen.role_sha256 != run.get("role_sha256"):
                    errors.append(f"{seed}_{variant}_role_hash_mismatch")
            except (OSError, KeyError, ValueError, RuntimeError, TypeError):
                errors.append(f"{seed}_{variant}_candidate_artifact_invalid")
        if candidate["actor_sha256"].get("raw") == candidate["actor_sha256"].get("regularized"):
            errors.append(f"{seed}_variants_identical")
        candidates.append(candidate)
    return candidates


def audit(pair_report_path):
    manifest = json.loads((ROOT / "configs/study_v3_kitchen_v5.json").read_text())
    entries = manifest["policies"]
    acceptance_path = ROOT / entries["raw"]["acceptance_path"]
    acceptance = json.loads(acceptance_path.read_text())
    pair_report_bytes = Path(pair_report_path).read_bytes()
    pair_report = json.loads(pair_report_bytes)
    errors = []
    if (manifest.get("release_id") != RELEASE_ID
            or pair_report.get("release_id") not in (None, RELEASE_ID)
            or acceptance.get("release_id") not in (None, RELEASE_ID)):
        errors.append("release_id_mismatch")
    acceptance["release_id"] = RELEASE_ID
    selected = acceptance.get("selected_pair_seed")
    if tuple(acceptance.get("pair_seeds", ())) != PAIR_SEEDS:
        errors.append("pair_seeds_mismatch")
    if tuple(pair_report.get("pair_seeds", ())) != PAIR_SEEDS:
        errors.append("pair_report_seeds_mismatch")
    if selected not in PAIR_SEEDS or selected != pair_report.get("selected_pair_seed"):
        errors.append("selected_pair_seed_mismatch")
    if len(pair_report.get("pairs", ())) != 3:
        errors.append("missing_pair_runs")
    if not pair_report.get("passed") or not acceptance.get("pre_audit_quality_passed"):
        errors.append("paired_quality_not_passed")
    if pair_report.get("scenario_block_steps") != SCENARIO_BLOCK_STEPS:
        errors.append("pair_report_scenario_horizon_mismatch")
    if selected in PAIR_SEEDS and (
            acceptance.get("scenario_block_steps") != SCENARIO_BLOCK_STEPS
            or acceptance.get("expanded_scenario_seed_sha256") != scenario_schedule_digest(selected)):
        errors.append("acceptance_expanded_step_schedule_mismatch")
    acceptance["pair_report_sha256"] = sha256(pair_report_bytes).hexdigest()
    acceptance["candidate_training"] = _candidate_training_evidence(pair_report, errors)
    validation_receipt = None
    receipt_path = ROOT / pair_report.get("final_program_validation_receipt_path", "")
    try:
        receipt_bytes = receipt_path.read_bytes()
        validation_receipt = json.loads(receipt_bytes)
        if (sha256(receipt_bytes).hexdigest() != pair_report.get("final_program_validation_receipt_sha256")
                or sha256(receipt_bytes).hexdigest() != acceptance.get("final_program_validation_receipt_sha256")
                or validation_receipt.get("release_id") != RELEASE_ID
                or validation_receipt.get("status") != "completed"
                or validation_receipt.get("passed") is not True
                or validation_receipt.get("selected_pair_seed") != selected
                or validation_receipt.get("program_sha256") != acceptance.get("program_sha256", {}).get("regularized")
                or validation_receipt.get("training_feedback_program_sha256") != acceptance.get("training_feedback_program_sha256")
                or validation_receipt.get("development_seed_start") != FINAL_PROGRAM_VALIDATION_START
                or validation_receipt.get("development_seed_count") != FINAL_PROGRAM_VALIDATION_SEEDS):
            errors.append("final_program_validation_receipt_mismatch")
    except (OSError, ValueError, TypeError):
        errors.append("final_program_validation_receipt_unavailable")
    try:
        original_path = ROOT / pair_report["original_pair_report_path"]
        original_sha = sha256(original_path.read_bytes()).hexdigest()
        if (original_sha != pair_report.get("original_pair_report_sha256")
                or original_sha != acceptance.get("original_pair_report_sha256")):
            errors.append("original_training_report_hash_mismatch")
    except (KeyError, OSError, TypeError):
        errors.append("original_training_report_unavailable")
    heldout_receipt_path = Path(pair_report_path).parent / "kitchen_dual_nn_v5_heldout_receipt.json"
    try:
        heldout_receipt_bytes = heldout_receipt_path.read_bytes()
        heldout_receipt = json.loads(heldout_receipt_bytes)
        if (sha256(heldout_receipt_bytes).hexdigest() != pair_report.get("heldout_receipt_sha256")
                or sha256(heldout_receipt_bytes).hexdigest() != acceptance.get("heldout_receipt_sha256")
                or heldout_receipt.get("status") != "completed"
                or heldout_receipt.get("selected_pair_seed") != selected
                or heldout_receipt.get("actor_sha256") != acceptance.get("artifact_sha256")
                or heldout_receipt.get("final_program_validation_receipt_sha256")
                    != acceptance.get("final_program_validation_receipt_sha256")
                or heldout_receipt.get("seed_start") != HELDOUT_START
                or heldout_receipt.get("seed_count") != 100
                or heldout_receipt.get("heldout") != acceptance.get("heldout")):
            errors.append("heldout_receipt_mismatch")
    except (OSError, ValueError, TypeError):
        errors.append("heldout_receipt_unavailable")
    for pair in pair_report.get("pairs", ()):
        for variant in ("raw", "regularized"):
            if pair["variants"][variant].get("joint_steps") != JOINT_STEPS:
                errors.append(f"{pair['seed']}_{variant}_training_steps_mismatch")
    raw_feedback, regularized_feedback = {}, {}
    final_actor = None
    final_programs = {}
    selected_training_report = None
    for variant in ("raw", "regularized"):
        entry = entries[variant]
        report_path = ROOT / entry["training_report_path"]
        actor_path = ROOT / entry["actor_path"]
        try:
            report_bytes = report_path.read_bytes()
            actor_bytes = actor_path.read_bytes()
            report = json.loads(report_bytes)
            actor = FrozenActors(actor_path, entry["actor_sha256"])
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            errors.append(f"{variant}_artifact_invalid:{type(exc).__name__}")
            continue
        if sha256(report_bytes).hexdigest() != entry.get("training_report_sha256"):
            errors.append(f"{variant}_training_report_hash_mismatch")
        if sha256(actor_bytes).hexdigest() != acceptance.get("artifact_sha256", {}).get(variant):
            errors.append(f"{variant}_acceptance_actor_hash_mismatch")
        if sha256(report_bytes).hexdigest() != acceptance.get("training_report_sha256", {}).get(variant):
            errors.append(f"{variant}_acceptance_report_hash_mismatch")
        if actor.role_sha256 != entry.get("role_sha256") or actor.role_sha256 != report.get("role_sha256"):
            errors.append(f"{variant}_role_hash_mismatch")
        if actor.sha256 != report.get("actor_sha256"):
            errors.append(f"{variant}_report_actor_hash_mismatch")
        if actor.role_sha256["human"] == actor.role_sha256["ai"]:
            errors.append(f"{variant}_roles_share_weights")
        if report.get("pair_seed") != selected or report.get("joint_steps") != JOINT_STEPS:
            errors.append(f"{variant}_selected_run_mismatch")
        if report.get("expanded_scenario_seed_sha256") != scenario_schedule_digest(selected):
            errors.append(f"{variant}_selected_step_schedule_mismatch")
        if report.get("action_overrides") != 0:
            errors.append(f"{variant}_post_policy_override")
        feedback = report.get("feedback_rows") or {}
        if variant == "raw":
            raw_feedback = feedback
            if sum(feedback.values()) != 0:
                errors.append("raw_received_program_feedback")
        else:
            final_actor = actor
            selected_training_report = report
            regularized_feedback = feedback
            if any(feedback.get(role, 0) <= 0 for role in ("human", "ai")):
                errors.append("regularized_role_missing_program_feedback")
            for role, program_path in zip(("human", "ai"), entry["program_paths"]):
                program = report.get("explanation_programs", {}).get(role, {})
                feedback_program = report.get("programs", {}).get(role, {})
                expected = entry.get("program_sha256", {}).get(role)
                try:
                    actual = sha256((ROOT / program_path).read_bytes()).hexdigest()
                except OSError:
                    actual = None
                if (actual != expected or actual != program.get("sha256")
                        or actual != acceptance.get("program_sha256", {}).get("regularized", {}).get(role)):
                    errors.append(f"{role}_program_hash_mismatch")
                if actual is not None:
                    try:
                        extracted = ExecutableProgram.load_json(ROOT / program_path)
                        final_programs[role] = (extracted, actual)
                        metrics = extracted.metadata.get("metrics", {})
                        if (extracted.metadata.get("role") != role
                                or extracted.metadata.get("program_phase") !=
                                    "final_frozen_actor_training_only_reextraction"
                                or extracted.metadata.get("training_seed_start") != 60_000
                                or extracted.metadata.get("training_seed_count") != 20
                                or extracted.metadata.get("actor_sha256") != actor.sha256
                                or extracted.metadata.get("role_sha256") != actor.role_sha256[role]
                                or extracted.metadata.get("training_feedback_program_sha256")
                                    != feedback_program.get("sha256")
                                or extracted.metadata.get("runtime_control_allowed") is not False
                                or not metrics.get("feedback_eligible")
                                or extracted.root.depth() > 5 or extracted.root.leaf_count() > 16):
                            errors.append(f"{role}_program_integrity_failed")
                    except (OSError, ValueError, TypeError, KeyError):
                        errors.append(f"{role}_program_unreadable")
                try:
                    feedback_actual = sha256((ROOT / feedback_program["path"]).read_bytes()).hexdigest()
                except (KeyError, OSError, TypeError):
                    feedback_actual = None
                if (feedback_actual != feedback_program.get("sha256")
                        or feedback_actual != acceptance.get("training_feedback_program_sha256", {}).get(role)
                        or program.get("training_feedback_program_sha256") != feedback_actual):
                    errors.append(f"{role}_training_feedback_program_hash_mismatch")
                if not program.get("feedback_eligible") or not feedback_program.get("feedback_eligible"):
                    errors.append(f"{role}_program_eligibility_failed")
    if (entries["raw"].get("actor_sha256") == entries["regularized"].get("actor_sha256")
            or not entries["raw"].get("actor_sha256")):
        errors.append("variants_share_actor_artifact")
    if acceptance.get("joint_steps") != {"raw": JOINT_STEPS, "regularized": JOINT_STEPS}:
        errors.append("acceptance_steps_mismatch")
    if not acceptance.get("heldout_evaluated"):
        errors.append("heldout_not_evaluated")
    for stage, count, start in (("development", 20, 40_000), ("heldout", 100, HELDOUT_START)):
        for variant in ("raw", "regularized"):
            for partner in ("nn", "proxy"):
                result = acceptance.get(stage, {}).get(variant, {}).get(partner, {})
                if (result.get("seeds") != list(range(start, start + count))
                        or result.get("episodes") != 3 * count
                        or result.get("successes", -1) < int(.8 * 3 * count)):
                    errors.append(f"{stage}_{variant}_{partner}_quality_failed")
    chosen_pair = next((pair for pair in pair_report.get("pairs", ()) if pair.get("seed") == selected), None)
    if chosen_pair is None or not chosen_pair.get("development_passed") or not chosen_pair.get("programs_eligible"):
        errors.append("selected_pair_development_failed")
    if final_actor is not None and set(final_programs) == {"human", "ai"} and validation_receipt:
        # The untouched admission block was evaluated once before held-out.
        # Verify that immutable receipt rather than running those scenes again.
        final_program_metrics = validation_receipt.get("final_role_program_development_audit") or {}
        acceptance["final_role_program_development_audit"] = final_program_metrics
        acceptance["final_ai_program_development_audit"] = final_program_metrics.get("ai")
        if (chosen_pair is None
                or validation_receipt.get("actor_sha256") != final_actor.sha256
                or validation_receipt.get("role_sha256") != final_actor.role_sha256
                or set(final_program_metrics) != {"human", "ai"}
                or not all(final_program_metrics[role].get("passed") is True
                           and final_program_metrics[role].get("development_seeds") == list(range(
                               FINAL_PROGRAM_VALIDATION_START,
                               FINAL_PROGRAM_VALIDATION_START + FINAL_PROGRAM_VALIDATION_SEEDS))
                           and final_program_metrics[role].get("actor_sha256") == final_actor.sha256
                           and final_program_metrics[role].get("role_sha256") == final_actor.role_sha256[role]
                           and final_program_metrics[role].get("program_sha256") == final_programs[role][1]
                           and final_program_metrics[role].get("action_fidelity", 0) >= .80
                           and final_program_metrics[role].get("mean_kl_divergence", float("inf")) <= .35
                           for role in ("human", "ai"))
                or final_program_metrics != selected_training_report.get("final_role_program_development_audit")
                or final_program_metrics != chosen_pair["variants"]["regularized"].get(
                    "final_role_program_development_audit")):
            errors.append("final_role_program_fidelity_failed")
    else:
        errors.append("final_role_program_missing")
    acceptance["raw_program_feedback_rows"] = raw_feedback
    acceptance["regularized_program_feedback_rows"] = regularized_feedback
    acceptance["post_policy_override_count"] = 0 if not any("override" in error for error in errors) else None
    acceptance["passed"] = not errors
    acceptance["verification_errors"] = errors
    _write_json(acceptance_path, acceptance)
    return acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-report", default="output/kitchen_v5_matched_candidates/kitchen_dual_nn_v5_reextracted_pair_report.json")
    args = parser.parse_args()
    try:
        result = audit(args.pair_report)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"passed": False, "verification_errors": [f"evidence_unavailable:{type(exc).__name__}"]}))
        raise SystemExit(1) from exc
    print(json.dumps({"passed": result["passed"], "verification_errors": result["verification_errors"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
