"""Immutable v6.22 development protocol; no implicit rollout or pilot authority.

The first stage consumes only the already spent v6.20 cohort.  Actor and
environment source files from the closed candidates are never rewritten.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from experiments.warehouse import v621_protocol as previous

CANDIDATE_ID = "warehouse-v622-v620-preserving-01"
OUTPUT_ROOT = "output/warehouse_v622"
BATCH_ID = "capacity-01"
RECIPE_PATH = "configs/warehouse_v622_recipe.json"
ACTOR_SHA256 = previous.ACTOR_SHA256
ACTOR_PATH = previous.ACTOR_PATH
ROLE_SHA256 = dict(previous.ROLE_SHA256)
ROLES = previous.ROLES
SPLITS = previous.SPLITS
GATES = dict(previous.GATES)
LIMITS = {"maximum_depth": 6, "maximum_leaves": 32,
          "maximum_source_features_per_node": 3,
          "guards_and_empty_branches_count": True}
METHODS = {"A32": "v620_m2_expanded_distribution_refit_v622",
           "B32": "actor_topology_guarded_distribution_refit_v622",
           "D32": "fixed_template_leaf_budget_dp_v622"}
BASELINE_SUMMARY = "output/warehouse_v621/diagnostics-01/summary.json"
BASELINE_SUMMARY_SHA256 = "38b462305234e8ff2694370043079a34d234966234d214c8295a9e7438034e2c"
BASELINE_AUDIT = "output/warehouse_v621/independent-01/audit.json"
BASELINE_AUDIT_SHA256 = "581d33516f4840ab92cee6586af829cf804ba182e1b21182d9057d71e376380d"
NEW_SOURCE_FILES = (
    "experiments/warehouse/v622_protocol.py",
    "experiments/warehouse/v622_capacity_audit_math.py",
    "backend/training/warehouse_v622_program.py",
    "backend/training/warehouse_v622_search.py",
    "backend/training/warehouse_v622_actor_training.py",
    "scripts/run_warehouse_v622_capacity.py",
    "scripts/audit_warehouse_v622_capacity.py",
    "scripts/audit_warehouse_v621_diagnostics.py",
    "configs/warehouse_v622_recipe.json",
)


def plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def grouped_folds():
    # Exactly the old grouping: capacity is the experimental variable.
    return previous.grouped_folds()


checked_actor = previous.checked_actor
checked_inputs = previous.checked_inputs
target_identity = previous.target_identity
gate_metrics = previous.gate_metrics


def jobs(stage="primary"):
    if stage not in ("primary", "fallback"):
        raise ValueError("unknown v6.22 capacity stage")
    arms = ("A32", "B32") if stage == "primary" else ("D32",)
    return [{"job_id": f"fold-{fold}-{role}-{arm}", "fold": fold,
             "role": role, "arm": arm, "method": METHODS[arm]}
            for fold in range(3) for role in ROLES for arm in arms]


def round_budget_contract():
    return {
        "A32": {"initializations": 8, "global_rounds": 60,
                 "leaf_api_calls": 20000, "reserved_final_leaf_calls": 32,
                 "wall_seconds": 5400, "reserved_final_wall_seconds": 600},
        "B32": {"initializations": 0, "global_rounds": 0,
                 "leaf_api_calls": 32, "reserved_final_leaf_calls": 32,
                 "wall_seconds": 600},
        "D32": {"maximum_templates": 4, "template_depth": 6,
                 "template_leaves": 64, "template_generation_leaf_api_calls": 20000,
                 "node_table_leaf_api_calls": 540, "wall_seconds": 5400,
                 "reserved_final_wall_seconds": 600,
                 "optimality_scope": "declared_templates_and_frozen_leaf_output_tables_only"},
    }


def quality_retained(candidate, reference):
    """Evaluate one stream, never pool failed streams into passing ones."""
    names = ("mean_deliveries", "mean_collisions", "shutdown_episode_rate")
    for row in (candidate, reference):
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
               or row[k] < 0 for k in names) or row["shutdown_episode_rate"] > 1:
            return {"passed": False, "checks": {"finite_metrics": False}}
    checks = {
        "deliveries_original_development_gate": candidate["mean_deliveries"] >= 5.5,
        "deliveries_at_least_95_percent_reference": candidate["mean_deliveries"] >= .95*reference["mean_deliveries"],
        "collisions_at_most_six_and_reference_plus_one": candidate["mean_collisions"] <= min(6., reference["mean_collisions"]+1.),
        "shutdown_at_most_five_percent": candidate["shutdown_episode_rate"] <= .05,
    }
    return {"passed": all(checks.values()), "checks": checks}


def summarize_units(results, *, stage="primary"):
    expected = {j["job_id"]: j for j in jobs(stage)}
    if (len(results) != len(expected)
            or {r["job"]["job_id"] for r in results} != set(expected)
            or any(r["job"] != expected[r["job"]["job_id"]] for r in results)
            or any(r.get("status") not in ("completed", "scientific_budget_exhausted") or not r.get("budget_audit", {}).get("passed")
                   for r in results)):
        raise RuntimeError("every registered role/fold must complete before route selection")
    arms = {}
    for arm in dict.fromkeys(j["arm"] for j in expected.values()):
        rows = [r for r in results if r["job"]["arm"] == arm]
        completed = [r for r in rows if r["status"] == "completed"]
        passed = len(completed) == len(rows) and all(gate_metrics(r["metrics"]["validation"])["passed"] for r in completed)
        e = min((r["metrics"]["validation"]["explanation"]["coverage"] for r in completed), default=0.)
        f = min((r["metrics"]["validation"]["feedback"]["expected_actionable_coverage"] for r in completed), default=0.)
        kls = [r["metrics"]["validation"][p]["kl"] for r in completed for p in ("feedback", "explanation")]
        max_kl = max(kls) if kls and all(k is not None and math.isfinite(k) for k in kls) else None
        leaves = sum(r["leaf_count"] for r in rows)
        arms[arm] = {"all_role_fold_gates_passed": passed, "completed_role_folds": len(completed),
            "budget_exhausted_role_folds": len(rows)-len(completed),
            "minimum_explanation_coverage": e, "minimum_expected_feedback_coverage": f,
            "maximum_qualified_kl": max_kl, "total_leaf_count": leaves,
            "selection_score": [e, f, -max_kl if max_kl is not None else None, -leaves]}
    eligible = [arm for arm, row in arms.items() if row["all_role_fold_gates_passed"]]
    selected = sorted(eligible, key=lambda a: (*(-x for x in arms[a]["selection_score"]), a))[0] if eligible else None
    return {"arms": arms, "selected_arm": selected,
        "next_stage": "fresh_confirmation" if selected else "fallback" if stage == "primary" else "closed_failed_program_precheck",
        "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "publishable": False, "pilot_opened": False,
        "spent_cohort_is_independent_admission_evidence": False}


def executable_recipe():
    return {"schema": "warehouse-v622-preserving-recipe.v1", "candidate_id": CANDIDATE_ID,
        "frozen_actor_sha256": ACTOR_SHA256, "frozen_actor_role_sha256": ROLE_SHA256,
        "whole_tree_limits": LIMITS, "methods": METHODS, "gates": GATES,
        "cv": {"source_episodes": 275, "physical_frames_per_role": 32782,
            "folds": 3, "fold_assignments_sha256": digest(grouped_folds()),
            "counts": {"fit": 125, "calibration": 25, "eligibility": 50, "validation": 75},
            "argmax_episodes_fit_only": 50, "excluded_frames": 0},
        "budgets": round_budget_contract(),
        "baseline_A16": {"summary_path": BASELINE_SUMMARY, "summary_sha256": BASELINE_SUMMARY_SHA256,
            "audit_path": BASELINE_AUDIT, "audit_sha256": BASELINE_AUDIT_SHA256,
            "refit": False},
        "fresh_confirmation": {"requires_successful_independent_capacity_audit": True,
            "counts": {"fit": 100, "calibration": 25, "eligibility": 75, "validation": 75},
            "fit_argmax": 50, "pressure_bases": 72, "pressure_trajectories_per_base": 3,
            "no_validation_or_pressure_rows_in_any_gradient": True},
        "training": {"source": "same_v620_BC_snapshot_continuation_not_independent_BC_reproductions",
            "calibration_pair_joint_steps_per_variant": 20000, "development_streams": 3,
            "budgets": [200000, 300000, 400000], "restart_each_budget_from_same_BC": True,
            "critic_only_joint_steps_in_budget": 4000, "reference_KL_direction": "reference_to_current",
            "reference_target_nats_per_role": .01, "reference_beta_initial": 1.,
            "reference_beta_bounds": [.001, 100.], "beta_update_only_after_actor_update": True,
            "reference_buffer_fractions": {"frozen_initial_training": .5, "current_training": .5},
            "reference_loss_shared_between_raw_and_regularized": True,
            "program_loss_denominator": "all_physical_role_rows",
            "program_extraction_progress": [.75, .84, .93],
            "program_lambda": .01, "program_ramp_progress": [.80, .85],
            "raw_program_gradients": 0, "runtime_action_controller": "frozen_NN_with_physical_mask_only"},
        "quality_retention": {"minimum_delivery_reference_ratio": .95,
            "maximum_collision_reference_increment": 1., "maximum_collisions": 6.,
            "maximum_shutdown_rate": .05, "minimum_deliveries": 5.5,
            "applies_separately_to_every_stream_and_variant": True},
        "formal": {"paired_streams": 3, "pair_selection": "quality_only_before_program_preaudit",
            "minimum_deliveries": 5., "maximum_collisions": 8., "maximum_shutdown_rate": .10,
            "minimum_program_coverage": .20, "minimum_feedback_agreement": .85,
            "minimum_explanation_agreement": .90, "maximum_KL": .35,
            "raw_illegal_actions": 0, "sealed_audit_attempts": 1, "heldout_seeds_per_task": 100},
        "release_id": "policylens-three-domain-20261016.v6.0",
        "pilot_opened": False}


def validate_recipe(root):
    raw = (Path(root)/RECIPE_PATH).read_bytes()
    if json.loads(raw) != executable_recipe():
        raise RuntimeError("v6.22 candidate recipe changed")
    return sha256(raw).hexdigest()


def source_closure_files(root):
    root = Path(root).resolve()
    files = previous.source_closure_files(root)
    files.update({p: sha256((root/p).read_bytes()).hexdigest() for p in NEW_SOURCE_FILES})
    return dict(sorted(files.items()))


def source_closure_sha256(root):
    return digest(source_closure_files(root))
