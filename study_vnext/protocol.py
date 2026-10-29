"""Immutable study design declarations; this module never grants pilot authority.

The scientific catalog and independent operational evidence readers remain the
only admission authorities. Validating a protocol, sample-plan declaration or
receipt inventory is not verification of their scientific/institutional claims.
No environment, Actor, optimizer, participant database or network is accessed.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType

from runtime_v642.contracts import PolicyBinding, is_sha

PROTOCOL_ID = 'study-protocol-20261029.v1'
CONSENT_VERSION = 'study-consent-20261029.v1'
DOMAINS = ('warehouse', 'pong', 'kitchen')
GROUPS = ('A', 'B', 'C')
GROUP_VARIANTS = MappingProxyType({'A': 'regularized', 'B': 'regularized', 'C': 'raw'})
FORMAL_ENROLLMENT_AUTHORITY = False
_EVIDENCE_KINDS = ('sample_power_registration', 'model_admission', 'explanation_audit',
                   'execution_consistency', 'storage_restore', 'institution_scope', 'engineering')
_METRICS = {
    'warehouse': dict(name='native_team_score', field='score.task_score', unit='native_team_points'),
    'pong': dict(name='percent_of_contact_capacity', field='derived.percent_of_contact_capacity',
                 unit='percentage_points', numerator='final_state.raw_score',
                 denominator='fixed_initial_schedule_contact_capacity', multiplier=100,
                 definition='scripts.calibrate_rolling_pong.contact_capacity',
                 native_score_is_not_primary=True),
    'kitchen': dict(name='completed_orders', field='score.metrics.completed_orders', unit='orders',
                    minimum=0, maximum=5),
}


class ProtocolError(ValueError):
    pass


def _check(condition, reason):
    if not condition:
        raise ProtocolError(reason)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _design():
    return dict(
        version=PROTOCOL_ID, consent_version=CONSENT_VERSION,
        status='registered_design_declaration_not_experiment_result',
        formal_enrollment_authority=False,
        domains=list(DOMAINS), groups={
            'A': dict(variant='regularized', questions_in_tasks=[2]),
            'B': dict(variant='regularized', questions_in_tasks=[]),
            'C': dict(variant='raw', questions_in_tasks=[]),
        },
        model=dict(independent_role_actors=True, frozen_during_human_play=True,
                   human_replaces_only_one_role=True, actor_receives_group=False,
                   actor_receives_questions=False, execution='bound_NN_then_physical_mask',
                   post_policy_controller_overrides=0, raw_external_program_feedback='zero_entire_ancestry',
                   common_BC_and_reference_KL_must_be_disclosed=True,
                   domains_may_retain_their_registered_training_versions=True),
        participants=dict(domains_per_participant=1, groups_per_participant=1,
                          assignment='server_randomized_balanced_within_domain_and_scenario_triplet',
                          participant_may_choose_group=False, participant_may_choose_model=False,
                          participant_may_choose_scene=False,
                          researcher_override_modes=['preview', 'test']),
        phases=[dict(name='consent', measured=False),
                dict(name='neutral_controls_practice', measured=False),
                dict(name='bound_two_NN_demonstration', measured=False),
                dict(name='task1', task=1, purpose='baseline', questions=False),
                dict(name='task2', task=2, purpose='intervention', questions_groups=['A']),
                dict(name='task3', task=3, purpose='unaided_transfer_primary', questions=False),
                dict(name='questionnaire_and_understanding', after_performance=True)],
        explanation=dict(authority='server', factual_history='committed_human_AI_transitions_only',
                         target_tasks=[1, 2], decision_turn_must_precede_committed_turn=True,
                         frame_zero_has_executed_action=False,
                         next_unexecuted_NN_action_exposure=False,
                         eligible_program_must_match_saved_NN_action=True,
                         insufficient_evidence='cannot_infer_motive',
                         unique_causal_motive_claim=False,
                         counterfactual=dict(environment='isolated_copy', policy='same_frozen_binding',
                                             maximum_steps=12, window='already_committed_history_only',
                                             advances_real_game=False, enters_actor_input=False),
                         revoke_answers_on_task3=True, recheck_before_async_answer_delivery=True),
        replay=dict(available_to_all_groups=True, neutral=True,
                    explanation_content_available_only_to_A_in_task2=True),
        timing=dict(equal_physical_action_budgets=True, equal_native_rules=True,
                    equal_post_task2_review_window=True, review_window_seconds=None,
                    review_window_must_lock_before_formal_enrollment=True,
                    log_separately=['active', 'reading', 'replay', 'provider_wait'],
                    estimand='access_to_explanations_under_registered_usage_opportunities'),
        analysis=dict(primary_task=3, unit='participant', domains_analyzed_separately=True,
                      primary_metrics=_copy(_METRICS),
                      contrasts={
                          'A-B': dict(primary=True, estimand='explanation_access_same_NN',
                                      task1_baseline_adjustment=True,
                                      adjust_for=['task1_primary_metric', 'scenario_block'],
                                      exclude_from_adjustment=['question_count', 'reading_time']),
                          'A-C': dict(primary=False, estimand='full_package_vs_raw_NN',
                                      task1_baseline_adjustment=False,
                                      adjust_for=['scenario_block', 'registered_pretreatment_covariates']),
                          'B-C': dict(primary=False, estimand='model_program_regularization',
                                      task1_baseline_adjustment=False,
                                      adjust_for=['scenario_block', 'registered_pretreatment_covariates']),
                      }, primary_family='three_domain_A-B', primary_multiplicity='Holm_FWER_0.05',
                      randomized_A_non_questioners_remain_A=True,
                      minimum_effects=None, formal_participant_counts=None,
                      formal_sample_size_requires_power_registration=True,
                      target_power_minimum=0.80,
                      frames_orders_balls_are_not_independent_participants=True,
                      disclosure='effects_conditional_on_selected_frozen_model_pairs',
                      task2_is_secondary=True, do_not_select_primary_task_after_results=True),
        stages=dict(technical_pilot=dict(participants_per_domain_group=3, total_participants=27,
                                         evidence='usability_and_data_integrity_not_effectiveness'),
                    small_independent_pilot=dict(formal_results_pooling=False),
                    formal=dict(participant_counts=None, power=None,
                                enroll_only_after_independent_evidence_verification=True,
                                required_evidence=list(_EVIDENCE_KINDS))),
        preservation=dict(old_models_and_receipts='retain_original_identity_and_verdict',
                          old_participant_data='read_only', reused_heldout_evaluations=False,
                          offline_human_proxy_is_human_evidence=False),
    )


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """A closed, immutable v1 design. Detached documents cannot mutate it."""
    _canonical_json: str

    def __post_init__(self):
        _check(type(self._canonical_json) is str and self._canonical_json == _json(_design()),
               'unregistered protocol design')

    @property
    def protocol_id(self):
        return PROTOCOL_ID

    @property
    def sha256(self):
        return sha256(self._canonical_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self._canonical_json)

    @classmethod
    def from_dict(cls, value):
        _check(type(value) is dict, 'protocol must be an object')
        return cls(_json(value))


SPEC = ProtocolSpec(_json(_design()))
PROTOCOL_SHA256 = SPEC.sha256


def protocol_document():
    return SPEC.to_dict()


def protocol_sha256():
    return SPEC.sha256


def policy_variant(group):
    _check(type(group) is str and group in GROUP_VARIANTS, 'unknown study group')
    return GROUP_VARIANTS[group]


def question_permission(group, active_task, target_task, decision_turn, committed_turns):
    """Metadata-only condition. The Store still authenticates and verifies frames."""
    return (group == 'A' and type(active_task) is int and active_task == 2
            and type(target_task) is int and target_task in (1, 2)
            and type(decision_turn) is int and type(committed_turns) is int
            and 0 <= decision_turn < committed_turns)


def analysis_contract(contrast):
    _check(type(contrast) is str and contrast in ('A-B', 'A-C', 'B-C'), 'unknown study contrast')
    return _copy(protocol_document()['analysis']['contrasts'][contrast])


def validate_policy_set(policies):
    """Assert exact group/model bindings, without authenticating model evidence."""
    _check(type(policies) is dict and set(policies) == set(DOMAINS), 'all three domain policy sets required')
    release_ids = set()
    identities = {}
    for domain in DOMAINS:
        row = policies[domain]
        _check(type(row) is dict and set(row) == set(GROUPS), 'exact A/B/C binding set required')
        for group in GROUPS:
            binding = row[group]
            _check(type(binding) is PolicyBinding and binding.domain == domain
                   and binding.variant == policy_variant(group), 'group policy domain/variant differs')
            release_ids.add(binding.release_id)
        a, b, c = (row[group] for group in GROUPS)
        _check(a.to_dict() == b.to_dict(), 'A and B must use the exact same binding')
        _check(a.actor_sha256 != c.actor_sha256 and len({a.human_actor_sha256, a.ai_actor_sha256,
                                                       c.human_actor_sha256, c.ai_actor_sha256}) == 4,
               'raw needs its independent paired Actor artifact and role weights')
        _check(all(getattr(a, field) == getattr(c, field) for field in
                   ('model_schema', 'method_schema', 'feature_schema_sha256',
                    'action_physics_config_sha256', 'accepted_receipt_sha256')),
               'paired variants differ in registered architecture/observation/physics/method/acceptance')
        identities[domain] = {group: row[group].identity_sha256 for group in GROUPS}
    _check(len(release_ids) == 1, 'policy set belongs to multiple deployment catalogs')
    return dict(protocol_sha256=protocol_sha256(), policy_identity_sha256=identities,
                declaration_validation_only=True, formal_enrollment_allowed=False)


def validate_raw_ancestry(binding, declaration):
    """Check a hash-bound zero-feedback lineage *declaration*, not its truth.

    The release auditor must additionally verify every source receipt and parent
    checkpoint. This return value cannot grant scientific or study admission.
    """
    _check(type(binding) is PolicyBinding and binding.variant == 'raw', 'raw binding required')
    _check(type(declaration) is dict and set(declaration) == {
        'version', 'protocol_sha256', 'domain', 'final_actor_sha256', 'stages'}, 'invalid raw ancestry fields')
    _check(declaration['version'] == 'study-vnext-raw-ancestry.v1'
           and declaration['protocol_sha256'] == protocol_sha256()
           and declaration['domain'] == binding.domain
           and declaration['final_actor_sha256'] == binding.actor_sha256, 'raw ancestry binding differs')
    rows = declaration['stages']
    _check(type(rows) is list and rows, 'entire raw ancestry including initialization is required')
    previous, stage_ids = None, set()
    receipts = []
    for row in rows:
        _check(type(row) is dict and set(row) == {'stage_id', 'checkpoint_sha256',
            'parent_checkpoint_sha256', 'source_receipt_sha256', 'external_program_feedback_updates'},
            'invalid raw ancestry stage')
        _check(type(row['stage_id']) is str and row['stage_id'] and row['stage_id'] not in stage_ids,
               'duplicate or missing raw ancestry stage ID')
        stage_ids.add(row['stage_id'])
        _check(is_sha(row['checkpoint_sha256']) and is_sha(row['source_receipt_sha256'])
               and row['parent_checkpoint_sha256'] == previous, 'raw ancestry is disconnected or lacks receipt identity')
        counts = row['external_program_feedback_updates']
        _check(type(counts) is dict and set(counts) == {'human', 'ai'}
               and all(type(counts[role]) is int and counts[role] == 0 for role in counts),
               'raw ancestry contains external program feedback or unmeasured counts')
        previous = row['checkpoint_sha256']; receipts.append(row['source_receipt_sha256'])
    _check(previous == binding.actor_sha256, 'raw ancestry does not reach deployed Actor')
    return dict(protocol_sha256=protocol_sha256(), actor_sha256=binding.actor_sha256,
                source_receipt_sha256=receipts, declaration_validation_only=True,
                requires_source_receipt_verification=True, formal_enrollment_allowed=False)


def validate_sample_plan(value):
    """Validate a registered analysis/sample declaration; never invent power."""
    _check(type(value) is dict and set(value) == {'version', 'protocol_sha256', 'participant_counts',
        'minimum_effects', 'target_power', 'familywise_alpha', 'multiple_comparison', 'power_report_sha256'},
        'sample plan must have the closed registered fields')
    _check(value['version'] == 'study-vnext-formal-sample-plan.v1'
           and value['protocol_sha256'] == protocol_sha256()
           and is_sha(value['power_report_sha256']), 'sample plan is not bound to protocol/power evidence')
    power = value['target_power']
    _check(type(power) in (float, int) and math.isfinite(power) and 0.8 <= power < 1,
           'formal power target must be measured against at least 80 percent')
    _check(type(value['familywise_alpha']) in (float, int) and value['familywise_alpha'] == 0.05
           and value['multiple_comparison'] == 'Holm_FWER_0.05', 'registered primary multiplicity differs')
    counts, effects = value['participant_counts'], value['minimum_effects']
    _check(type(counts) is dict and set(counts) == set(DOMAINS)
           and type(effects) is dict and set(effects) == set(DOMAINS), 'each domain needs sample/effect registration')
    for domain in DOMAINS:
        row, effect = counts[domain], effects[domain]
        _check(type(row) is dict and set(row) == set(GROUPS)
               and all(type(n) is int and n > 0 for n in row.values())
               and len(set(row.values())) == 1, 'formal participant counts must balance A/B/C in each domain')
        _check(type(effect) is dict and set(effect) == {'value', 'unit'}
               and type(effect['value']) in (float, int) and math.isfinite(effect['value']) and effect['value'] > 0
               and effect['unit'] == _METRICS[domain]['unit'], 'minimum meaningful effect or native metric unit differs')
    return dict(protocol_sha256=protocol_sha256(), total_participants=sum(sum(r.values()) for r in counts.values()),
                power_report_sha256=value['power_report_sha256'], declaration_validation_only=True,
                computed_power_verified=False, formal_enrollment_allowed=False)


def formal_evidence_requirements():
    return _EVIDENCE_KINDS


def validate_evidence_references(references):
    """Closed inventory only. Dicts containing ``passed: true`` are not permits."""
    _check(type(references) is dict and set(references) == set(_EVIDENCE_KINDS),
           'all independently verified enrollment evidence kinds are required')
    for kind, reference in references.items():
        _check(type(reference) is dict and set(reference) == {'protocol_sha256', 'sha256'}
               and reference['protocol_sha256'] == protocol_sha256() and is_sha(reference['sha256']),
               'evidence reference lacks exact protocol/content identity')
    return dict(protocol_sha256=protocol_sha256(), evidence_references=_copy(references),
                inventory_complete=True, semantic_evidence_verified=False,
                formal_enrollment_allowed=False)
