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
    environment="warehouse_collaborative_delivery_v41_commitment_aligned",
    observation="collaborative_observation_v37_causal_queue_commitment",
    reward="warehouse_safe_mission_reward_v28_committed_mission_regret",
    model="warehouse_mappo_v65_commitment_aligned_actor",
    training_checkpoint="warehouse_mappo_training_v57_commitment_aligned",
    rcpd_program="warehouse_rcpd_v58_compact8_posthoc",
    seed_library="warehouse_parallel_seed_pairs_v59_compact8",
    reference_trajectory="warehouse_reference_trajectory_v58_compact8",
    study_log="human-study-log.v28",
    action_execution="batched_independent_simultaneous_actor_v13",
    runtime_controller="mappo_batched_actor_atomic_joint_execution",
    artifact_namespace="simultaneous_compact8_v58_commitment_aligned",
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
        "episodes_per_condition_ge_1000",
        "multi_partner_episodes_ge_1000",
        "formal_seed_ranges_disjoint_from_training",
        "compact_map_topology_contract_valid",
        "shutdown_episode_rate_le_0_01",
        "charger_utilization_positive",
        "mean_minimum_battery_positive",
        "collision_episode_rate_le_0_01",
        "maximum_collision_events_per_episode_le_1",
        "repeated_collision_episode_rate_eq_0",
        "deadlock_episode_rate_le_0_01",
        "avoidable_wait_rate_le_0_005",
        "head_on_yield_success_ge_0_90",
        "empty_delivery_clearance_success_eq_1",
        "dual_charger_approach_success_eq_1",
        "outer_exit_charger_approach_success_eq_1",
        "occupied_charger_handoff_success_eq_1",
        "delivery_bootstrap_lower_positive",
        "score_bootstrap_lower_positive",
        "noisy_delivery_episode_rate_ge_0_80",
        "multi_partner_collision_episode_rate_le_0_01",
        "multi_partner_maximum_collision_events_per_episode_le_1",
        "multi_partner_repeated_collision_episode_rate_eq_0",
        "multi_partner_shutdown_episode_rate_le_0_01",
        "multi_partner_deadlock_episode_rate_le_0_01",
        "multi_partner_avoidable_wait_rate_le_0_005",
        "multi_partner_charger_departure_return_cycle_rate_le_0_01",
        "multi_partner_task_starvation_episode_rate_le_0_05",
        "multi_partner_path_efficiency_le_1_10",
        "multi_partner_avoidable_loaded_delivery_detours_eq_0",
        "charger_departure_return_cycle_rate_le_0_01",
        "task_starvation_episode_rate_le_0_05",
        "path_efficiency_le_1_10",
        "seed_42027_detour_regressions_pass",
        "avoidable_loaded_delivery_detours_eq_0",
        "avoidable_mission_detours_eq_0",
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
