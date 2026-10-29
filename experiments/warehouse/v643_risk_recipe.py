"""Next-candidate recipe registration only; no training or seed capability.

v6.42's failed candidate and its claim remain closed. The planned complete-episode
critic buffer and explicit RiskDualConfig must be implemented in a separately
source-bound owner before this declaration can be used for a new 20k run.
"""
from dataclasses import asdict, dataclass
from .v643_shutdown_diagnostic import VERSION as DIAGNOSTIC_VERSION, read_bound, check

VERSION = 'warehouse-v643-risk-calibration-recipe.v1'
CANDIDATE = 'warehouse-v643-risk-calibration-01'
REFERENCE_SHA256 = '1b6310f6d93309fb17e64516519a71680ad3c30e73f222be27df4dff10da81ec'


@dataclass(frozen=True)
class ProposedRiskSettings:
    target: float = .02
    window_episodes: int = 256
    minimum_episodes: int = 64
    confidence: float = .90
    release_upper: float = .01
    ascent_gain: float = 2.
    descent_gain: float = 1.
    max_ascent_per_update: float = .10
    max_descent_per_update: float = .05
    max_multiplier: float = 20.
    initial_multiplier: float = 2.


def prepare_recipe(diagnostic_ref):
    inputs = {}
    diagnostic = read_bound(diagnostic_ref, inputs)
    check(diagnostic['version'] == DIAGNOSTIC_VERSION
          and diagnostic['scope'] == 'spent_development_evidence_diagnostic_only'
          and diagnostic['training_authority'] is False
          and diagnostic['model_acceptance_passed'] is False,
          'recipe requires a bound diagnostic of the original failed development run')
    check(diagnostic['quality']['raw']['shutdown_rate'] > .05,
          'this risk-only recipe must identify the original shutdown failure')
    check(all(diagnostic[k] == 0 for k in ('new_environment_resets','new_Actor_updates','new_seed_allocations')),
          'diagnostic must not consume new science')
    return dict(version=VERSION,candidate_id=CANDIDATE,stage='registered_development_recipe_not_executable',
        basis=diagnostic_ref,previous_candidate='warehouse-v642-depth9-depth12-01',
        previous_failed_candidate_and_claim_remain_closed=True,
        reference_actor_sha256=REFERENCE_SHA256,
        start='same_complete_v620_BC_dual_Actor_with_new_shared_critic_and_optimizer_state',
        starting_from_v642_raw_or_regularized_final_weights_forbidden=True,
        paired_variants=['raw','regularized'],joint_step_budget_per_variant=20000,
        actor_warmup_joint_steps=4000,
        common_recipe_changes=dict(risk_dual=asdict(ProposedRiskSettings()),
            cost_critic_complete_episode_replay=dict(maximum_complete_episodes=256,
                sampling='uniform_complete_episode',weighting='one_mean_BCE_per_complete_episode',
                labels='actual_complete_episode_shutdown_only',
                retain_warmup_positive_examples=True,old_Actor_policy_gradient_replay=False,
                calibration_metrics=['Brier_score','binary_cross_entropy','low_battery_public_state_strata'],
                metrics_scope='training_only_not_independent_quality_evidence')),
        cost_Actor_surrogate='current_on_policy_complete_episodes_only',
        common_unchanged=['full_v620_Actor_forward','physical_legal_action_mask','reward_PPO',
            'reference_NN_KL','native_reward_and_physics','critic_learning_rate','Actor_learning_rate'],
        external_program=dict(maximum_depth=12,maximum_leaves=256,maximum_public_sources=3,
            extraction_fractions=[.75,.84,.93],lambda_ramp_fractions=[.80,.85],lambda_max=.01,
            raw_actual_external_gradient=0,regularized_actual_positive_gradient_required_per_role=True,
            never_selects_environment_actions=True),
        development_quality=dict(deliveries_min=5.5,reference_delivery_fraction_min=.95,
            collisions_max=6,reference_collision_increase_max=1,shutdown_rate_max=.05),
        development_program=dict(feedback_and_explanation_coverage_min=.45,
            expected_feedback_coverage_min=.40,feedback_agreement_min=.95,
            explanation_agreement_min=.97,KL_max=.20,illegal_actions=0,unproved_paths=0),
        still_required_before_any_training=['new_configurable_cost_core_implementation_and_gradient_tests',
            'new_finite_single_use_owner_candidate_binding_and_source_snapshot',
            'reference_Actor_and_fixed_program_startup_compatibility_audit',
            'explicit_finite_training_program_and_quality_episode_inventories',
            'full_history_exact_seed_audit_and_single_use_clearance',
            'paired_initialization_and_scenario_schedule_equality_audit',
            'new_output_and_claim_namespace_not_old_v642_claim'],
        fresh_seed_values_allocated=False,new_environment_resets=0,new_Actor_updates=0,
        execution_authority=False,training_authority=False,sealed_authority=False,
        model_acceptance_passed=False,pilot_opened=False,executor_implemented=False)


def execution_authority(*_args, **_kwargs):
    raise RuntimeError('v643 recipe is not an executable training admission; new finite owner and exact seed clearance are required')
