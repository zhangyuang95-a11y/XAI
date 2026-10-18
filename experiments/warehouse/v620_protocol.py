"""Development-only grouped comparison on spent v6.19 trajectories.

This protocol grants no rollout, training, validation or publication authority.
The same frozen Actor is used for all eighteen method/role/fold jobs.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.warehouse.v619_protocol import ACTOR_SHA256, BASELINE_RUN, ROLES, seed_plan

CANDIDATE_ID = "warehouse-v620-01"
DATA_ROOT = "output/warehouse_v619/fixed-validation-01"
OUTPUT_ROOT = "output/warehouse_v620"
RECIPE_PATH = "configs/warehouse_v620_recipe.json"
METHODS = {
    "M0": "path_constrained_distribution_tree_v1",
    "M1": "queued_path_constrained_tree_v1",
    "M2": "episode_robust_explanation_tree_v1",
}
FOLD_SEED = "warehouse-v620-spent-data-grouped-cv.v1"
SPLITS = ("fit", "calibration", "eligibility", "validation")
GATES = {"minimum_explanation_coverage": .40,
    "mean_explanation_coverage_target": .45,
    "minimum_expected_feedback_coverage": .35,
    "minimum_explanation_agreement": .97, "maximum_kl": .20,
    "whole_tree_raw_illegal_count": 0, "whole_tree_path_uncertified_count": 0}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _ordered(cases, purpose):
    return sorted(cases, key=lambda row: digest([FOLD_SEED, purpose, row["base"]]))


def grouped_folds(cases=None):
    """Partition whole scene bases, never individual role rows or physical frames."""
    if cases is None:
        cases = [dict(case, original_split=split)
                 for split, rows in seed_plan()["splits"].items() for case in rows]
    cases = [dict(row) for row in cases]
    if (len(cases) != 124 or len({row["base"] for row in cases}) != 124
            or len({row["scenario_seed"] for row in cases}) != 124
            or any(row["task"] not in (1, 2, 3)
                   or row["sampling_mode"] not in ("sampled", "argmax") for row in cases)):
        # A future sampled/argmax pair on the same base must receive a new
        # whole-group protocol, rather than silently leaking the shared scene.
        raise RuntimeError("CV requires the pinned 124 distinct whole-base episodes")
    sampled = [row for row in cases if row["sampling_mode"] == "sampled"]
    argmax = [row for row in cases if row["sampling_mode"] == "argmax"]
    if len(sampled) != 99 or len(argmax) != 25:
        raise RuntimeError("CV requires 99 natural and 25 fit-only argmax episodes")
    held = [[], [], []]
    # Rotating the stratum offset gives exactly 33 held groups for the pinned
    # task counts (34,32,33), with per-task differences at most one.
    offset = 0
    for task in (1, 2, 3):
        group = _ordered([r for r in sampled if r["task"] == task], f"held-task-{task}")
        for index, row in enumerate(group):
            held[(offset+index) % 3].append(row)
        offset = (offset+len(group)) % 3
    result = []
    for fold in range(3):
        validation = sorted(held[fold], key=lambda r: r["base"])
        excluded = {r["base"] for r in validation}
        train = [r for r in sampled if r["base"] not in excluded]
        assignments = {"fit": [], "calibration": [], "eligibility": []}
        capacities = {"fit": 30, "calibration": 18, "eligibility": 18}
        # A hash shuffles each task stratum; relative task representation and
        # remaining total capacity determine placement without model outputs.
        for task in (1, 2, 3):
            group = _ordered([r for r in train if r["task"] == task], f"fold-{fold}-task-{task}")
            for row in group:
                options = [name for name in capacities if len(assignments[name]) < capacities[name]]
                name = min(options, key=lambda name: (
                    sum(r["task"] == task for r in assignments[name])/capacities[name],
                    len(assignments[name])/capacities[name], name))
                assignments[name].append(row)
        assignments["fit"] += argmax
        assignments["validation"] = validation
        assignments = {name: sorted(rows, key=lambda r: r["base"])
                       for name, rows in assignments.items()}
        if {k: len(v) for k, v in assignments.items()} != {
                "fit": 55, "calibration": 18, "eligibility": 18, "validation": 33}:
            raise RuntimeError("whole-base CV allocation does not match locked counts")
        groups = [{r["base"] for r in assignments[name]} for name in SPLITS]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise RuntimeError("whole-base CV split leakage")
        result.append({"fold": fold, "splits": assignments,
                       "assignment_sha256": digest(assignments)})
    if sorted(r["base"] for item in result for r in item["splits"]["validation"]) != sorted(r["base"] for r in sampled):
        raise RuntimeError("natural-sampling episodes are not held out exactly once")
    return result


def jobs():
    return [{"job_id": f"fold-{fold}-{role}-{arm}", "fold": fold,
             "role": role, "arm": arm, "method": method}
            for fold in range(3) for role in ROLES for arm, method in METHODS.items()]


def executable_recipe():
    from backend.training.warehouse_program_search import PRODUCTION_SEARCH_CONFIG
    return {"schema": "warehouse-v620-development-recipe.v1", "candidate_id": CANDIDATE_ID,
        "methods": METHODS, "frozen_actor_sha256": ACTOR_SHA256,
        "search": asdict(PRODUCTION_SEARCH_CONFIG),
        "leaf_fit": {"epsilon": .001, "regularization": .0001, "maxiter": 200, "tolerance": .000001},
        "calibration_temperatures": [.8, .9, 1., 1.1, 1.2],
        "cv": {"fold_seed": FOLD_SEED, "fold_seed_sha256": digest(FOLD_SEED),
               "episodes": 124, "natural_episodes": 99, "fit_only_argmax_episodes": 25,
               "split_episode_counts": {"fit": 55, "calibration": 18, "eligibility": 18, "validation": 33},
               "whole_base_grouping": True, "fold_assignments_sha256": digest(grouped_folds()),
               "m0_refitted_without_historical_topology": True},
        "gates": GATES, "selection": {
            "eligible_arms": ["M1", "M2"],
            "order": ["minimum_explanation_coverage", "minimum_expected_feedback_coverage",
                      "negative_maximum_qualified_kl", "negative_total_leaf_count"],
            "failure_next_stage": "common_bc_structure_learning_before_fresh_validation"},
        "future_common_bc": {"residual_soft_penalty_candidates": [.01, .03],
            "external_unqualified_program_gradients": False,
            "raw_regularized_same_architecture_and_bc_snapshot": True,
            "raw_ppo_external_program_feedback": 0,
            "initialization_seeds_proposed_not_authorized": {"pilot": 6160, "development": [6161, 6162, 6163]},
            "formal_pair_seeds_require_ledger_clearance": [6101, 6102, 6103]},
        "future_fresh": {"requires_method_lock_and_new_seed_clearance": True,
            "fit": {"episodes": 100, "sampled": 50, "argmax": 50},
            "calibration": {"episodes": 25, "sampled": 25},
            "eligibility": {"episodes": 75, "sampled": 75},
            "validation": {"episodes": 75, "sampled": 75},
            "bc_gates": {"feedback_and_explanation_coverage": .35,
                "expected_feedback_coverage": .30, "feedback_agreement": .93,
                "explanation_agreement": .95, "maximum_kl": .25,
                "whole_tree_illegal_and_uncertified": 0},
            "independent_trajectories": 216, "budget_candidates": [200000, 300000, 400000],
            "budget_restarts_from_same_bc": True,
            "formal_gates": {"deliveries": 5., "collisions": 8., "shutdown_rate": .10,
                "program_coverage": .20, "feedback_agreement": .85,
                "explanation_agreement": .90, "maximum_kl": .35, "illegal_count": 0}},
        "all_physical_frames": True, "publishable": False, "admission_evidence": False,
        "fresh_validation_evidence": False, "environment_resets": 0,
        "actor_updates": 0, "PPO_joint_steps": 0}


def validate_recipe(root: Path):
    raw = (root / RECIPE_PATH).read_bytes()
    if json.loads(raw) != json.loads(json.dumps(executable_recipe())):
        raise RuntimeError("v6.20 executable recipe drifted")
    return sha256(raw).hexdigest()
