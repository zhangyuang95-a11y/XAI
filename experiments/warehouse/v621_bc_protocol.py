"""Finite development recipes. Definitions never authorize a reset or update."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.warehouse import v621_protocol as diagnostics

INITIALIZATIONS = (6171, 6172, 6173)
BATCH = "bc-development-01"
RECIPES = ("additive_control", "shared_b035", "shared_b050")
STAGES = (
    ("initial_public_teacher", 100),
    ("initial_nn_dagger_1", 40), ("initial_nn_dagger_2", 40),
    ("joint_augmentation_sampled", 25), ("joint_augmentation_argmax", 25),
    ("risk_public_teacher", 40), ("risk_nn_visitation", 80),
    ("large_public_teacher", 200), ("large_nn_visitation", 120),
    ("deep_public_teacher", 200),
    *((f"deep_{index}_{kind}", 50) for index in range(1, 5) for kind in ("sampled", "argmax")),
)
TEACHER_STAGES = tuple(name for name, _ in STAGES if "public_teacher" in name)
NEW_SOURCES = (
    "experiments/warehouse/v621_bc_protocol.py",
    "backend/training/warehouse_v621_actor.py",
    "backend/training/warehouse_v621_actor_training.py",
    "backend/training/warehouse_v621_program.py",
    "backend/training/warehouse_v621_bc_pipeline.py",
    "backend/training/warehouse_v621_bc_capability.py",
    "scripts/run_warehouse_v621_bc_development.py",
    "configs/warehouse_v621_bc_recipe.json",
    "configs/warehouse_v621_bc_seed_plan.json",
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_plan():
    """Exact repeated arguments are intentionally paired, never independent."""
    return {"schema": "warehouse-v621-bc-development-seeds.v1",
        "candidate_id": diagnostics.CANDIDATE_ID, "batch_id": BATCH,
        "initializations": {str(seed): {
            "actor_initialization": seed,
            "stages": {name: [{"scenario_seed": seed*100000+(10+index)*1000+i,
                "action_seed": 1850000000+(seed-INITIALIZATIONS[0])*100000+index*1000+i}
                for i in range(count)] for index, (name, count) in enumerate(STAGES)},
            "critic_seed": 1860000000+(seed-INITIALIZATIONS[0]),
            "critic_environment_reset_seeds": [1860000000+(seed-INITIALIZATIONS[0])]*2,
            "rng": {"initial_bc_and_dagger_torch_shuffle": seed+31,
                "risk_torch_shuffle": seed+201, "large_torch_shuffle": seed+301,
                "deep_torch_shuffle": [seed+400+i for i in range(1, 5)],
                "final_torch_shuffle": seed+499,
                "public_classifier_random_states": [19, 20],
                "critic_cache_load_torch": 1860000000+(seed-INITIALIZATIONS[0]),
                "critic_torch_initialization": 1860000000+(seed-INITIALIZATIONS[0]),
                "critic_torch_shuffle": 1860000001+(seed-INITIALIZATIONS[0]),
                "critic_unordered_pair_hash_argument": 1860000000+(seed-INITIALIZATIONS[0])}}
            for seed in INITIALIZATIONS},
        "development_evaluation": {"cases": diagnostics.source_cases(),
            "purpose": "spent_v620_scene_and_rng_arguments_reused_for_development_only",
            "freshness_claimed": False},
        "shared_teacher_cache_per_initialization": True,
        "fresh_275_precheck_authorized": False, "PPO_joint_steps_authorized": 0,
        "pilot_opened": False}


def recipe():
    return {"schema": "warehouse-v621-shared-actor-bc-development.v1",
        "candidate_id": diagnostics.CANDIDATE_ID, "batch_id": BATCH,
        "initializations": list(INITIALIZATIONS), "recipes": list(RECIPES),
        "activation": "all_eighteen_diagnostics_and_independent_audit_complete_with_no_passing_method",
        "backup_activation": "all_three_main_runs_complete_and_at_least_one_main_quality_failure",
        "all_three_main_must_pass": True, "control_is_diagnostic_not_candidate_admission": True,
        "backup_starts_from_same_initialization_not_main_checkpoint": True,
        "training_episode_exposure_per_model": 1270, "stage_counts": [list(row) for row in STAGES],
        "teacher_cache_episodes_per_initialization": 540, "nn_visitation_episodes_per_model": 730,
        "critic_environment_resets_per_initialization": 2,
        "teacher_only_pre_action_public_information": True,
        "shared_initial_primary_nn_tensors_between_architectures": True,
        "identical_full_bc_start_required_between_raw_and_regularized": True,
        "initial_bc": {"epochs": 20, "batch_size": 256, "learning_rate": .0003},
        "initial_dagger": {"epochs": 10, "rounds": 2},
        "augmentation": {"epochs": 8, "batch_size": 256, "learning_rate": .0001},
        "later_bc": {"epochs": 8, "batch_size": 128, "learning_rate": .0001},
        "deep_bc": {"epochs": 6, "rounds": 4, "final_epochs": 6},
        "gradient_clip": .5,
        "shared_losses": {"full_task_ce": 1., "tree_task_ce": .2,
            "tree_to_full_kl": .2, "full_to_tree_kl": .2,
            "training_core_occupancy_weight": .05, "training_core_occupancy_target": .50,
            "condition_on_current_teacher_correctness": False},
        "control_losses": {"recipe": "warehouse-public-shared-bc-alignment.v6.20.r1", "gamma": .03},
        "initial_shared_topology": {"target": "initial_public_teacher_one_hot_only_before_any_bc",
            "algorithm": "public_supervised_tree_then_frozen_action_legal_guard_compilation",
            "criterion": "log_loss", "maximum_classifier_depth": 4,
            "maximum_classifier_leaves": 8, "minimum_classifier_leaf_rows": 80,
            "static_core_minimum_teacher_agreement": .75, "minimum_rows": 80,
            "minimum_episodes": 8, "maximum_episode_fraction": .20,
            "reject_no_core_instead_of_silently_changing_recipe": True},
        "core_bounds": {"main": .35, "backup": .50, "program_centered_box": .05,
            "minimum_tree_gap": "2*(bound+0.05)+0.26",
            "no_sample_dependent_escape": True, "no_confidence_guarantee_claimed": True,
            "noncore_residual": "unbounded_256_nn"},
        "offline_collision_risk_weight": 2., "offline_margin_weight": 0.,
        "risk_near_increment": 0., "later_near_increment": 2., "low_return_increment": 8.,
        "sampled_collision_increment": 3., "argmax_collision_increment": 5.,
        "all_physical_training_frames_retained": True,
        "critic": {"epochs": 80, "batch_size": 256, "learning_rate": .001, "positive_weight": 3.},
        "development_cases": "exact_spent_v620_275_cases_original_purpose_split",
        "quality_cases": "75_natural_original_validation_episodes_not_independent_evidence",
        "quality_gates": {"mean_deliveries": 5.5, "maximum_mean_collisions": 6.,
            "maximum_shutdown_rate": .05}, "program_gates": diagnostics.GATES,
        "feedback_gradient_probe_required_both_roles": True,
        "feedback_gradient_probe_coefficient": .01,
        "external_program_actor_updates_during_bc": 0,
        "fresh_precheck_authorized": False, "PPO_joint_steps": 0,
        "publishable": False, "pilot_opened": False}


def source_files(root):
    root = Path(root)
    files = diagnostics.source_closure_files(root)
    files.update({name: sha256((root/name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return dict(sorted(files.items()))


def source_sha256(root):
    return digest(source_files(root))


def checked_configuration(root):
    for name, expected in (("configs/warehouse_v621_bc_recipe.json", recipe()),
                           ("configs/warehouse_v621_bc_seed_plan.json", seed_plan())):
        if json.loads((Path(root)/name).read_bytes()) != expected:
            raise RuntimeError("v621 development recipe or exact seed plan changed")
