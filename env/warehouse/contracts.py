"""Single source of truth for the collaborative warehouse contracts.

These identifiers are persisted in checkpoints, programs, seed libraries and
study logs.  Keeping them together prevents a runtime component from silently
accepting an artifact produced against a different environment contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WarehouseContractVersions:
    environment: str
    observation: str
    reward: str
    model: str
    training_checkpoint: str
    rcpd_program: str
    seed_library: str
    reference_trajectory: str
    study_log: str
    action_execution: str
    runtime_controller: str
    artifact_namespace: str


CURRENT_VERSIONS = WarehouseContractVersions(
    environment="warehouse_collaborative_delivery_v22_individual_credit",
    observation="collaborative_observation_v23_avoidable_wait_memory",
    reward="warehouse_safe_mission_reward_v19_individual_credit",
    model="warehouse_mappo_v35_individual_credit",
    training_checkpoint="warehouse_mappo_training_v27_individual_credit",
    rcpd_program="warehouse_rcpd_v30_individual_credit_posthoc",
    seed_library="warehouse_parallel_seed_pairs_v31_individual_credit",
    reference_trajectory="warehouse_reference_trajectory_v30_individual_credit",
    study_log="human-study-log.v22",
    action_execution="autoregressive_direct_mappo_actor_action_v9_neural_mission",
    runtime_controller="mappo_autoregressive_actor_direct_execution",
    artifact_namespace="safe_mission_v28_individual_credit",
)


ENVIRONMENT_VERSION = CURRENT_VERSIONS.environment
OBSERVATION_CONTRACT_VERSION = CURRENT_VERSIONS.observation
REWARD_VERSION = CURRENT_VERSIONS.reward
MODEL_VERSION = CURRENT_VERSIONS.model
TRAINING_CHECKPOINT_VERSION = CURRENT_VERSIONS.training_checkpoint
RCPD_PROGRAM_VERSION = CURRENT_VERSIONS.rcpd_program
SEED_LIBRARY_VERSION = CURRENT_VERSIONS.seed_library
REFERENCE_TRAJECTORY_FORMAT = CURRENT_VERSIONS.reference_trajectory
STUDY_LOG_VERSION = CURRENT_VERSIONS.study_log
ACTION_EXECUTION_VERSION = CURRENT_VERSIONS.action_execution
RUNTIME_CONTROLLER = CURRENT_VERSIONS.runtime_controller
ARTIFACT_NAMESPACE = CURRENT_VERSIONS.artifact_namespace

FORMAL_ACCEPTANCE_CHECKS = frozenset(
    {
        "episodes_per_condition_ge_200",
        "formal_seed_ranges_disjoint_from_training",
        "shutdown_episode_rate_le_0_05",
        "charger_utilization_positive",
        "mean_minimum_battery_positive",
        "collision_episode_rate_le_0_05",
        "maximum_collision_events_per_episode_le_1",
        "repeated_collision_episode_rate_eq_0",
        "deadlock_episode_rate_le_0_01",
        "avoidable_wait_rate_le_0_005",
        "head_on_yield_success_ge_0_90",
        "delivery_bootstrap_lower_positive",
        "score_bootstrap_lower_positive",
        "noisy_delivery_episode_rate_ge_0_80",
        "charger_departure_return_cycle_rate_le_0_01",
        "task_starvation_episode_rate_le_0_05",
        "seed_42027_detour_regressions_pass",
        "avoidable_loaded_delivery_detours_eq_0",
        "ai_ai_post_policy_action_interventions_eq_0",
        "noisy_post_policy_action_interventions_eq_0",
        "posthoc_rcpd_artifact_contract_valid",
        "pure_neural_reference_artifact_contract_valid",
        "parallel_seed_artifact_contract_valid",
        "explanation_eligible",
    }
)

# The SQLite storage schema and the exported event-log schema evolve
# independently.  The explicit name prevents the old bare ``schema_version``
# integer from being confused with ``human-study-log.v10``.
SQLITE_SCHEMA_VERSION = 5
