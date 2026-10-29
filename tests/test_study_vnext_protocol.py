"""The study declaration must not turn missing scientific evidence into authority."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json

import pytest

from runtime_v642.contracts import PolicyBinding, ProgramBinding
from study_vnext import protocol as p


def h(label):
    return sha256(label.encode()).hexdigest()


def binding(domain, variant):
    programs = () if variant == 'raw' else tuple(ProgramBinding(
        role=role, native_role=role, reader_schema='synthetic-test-program',
        archive_sha256=h(domain+role+'archive'), source_program_sha256=h(domain+role+'source'),
        independent_audit_sha256=h(domain+role+'audit')) for role in ('human', 'ai'))
    return PolicyBinding(release_id='synthetic-test-release', policy_id=f'{domain}-{variant}',
        domain=domain, variant=variant, model_schema='same-pair-architecture', method_schema='same-pair-method',
        actor_sha256=h(domain+variant+'pair'), human_actor_sha256=h(domain+variant+'human'),
        ai_actor_sha256=h(domain+variant+'ai'), feature_schema_sha256=h(domain+'features'),
        action_physics_config_sha256=h(domain+'physics'), programs=programs,
        accepted_receipt_sha256=h(domain+'receipt'))


def policies():
    return {domain: {'A': binding(domain, 'regularized'), 'B': binding(domain, 'regularized'),
                     'C': binding(domain, 'raw')} for domain in p.DOMAINS}


def lineage(b):
    return dict(version='study-vnext-raw-ancestry.v1', protocol_sha256=p.protocol_sha256(),
        domain=b.domain, final_actor_sha256=b.actor_sha256, stages=[
            dict(stage_id='BC', checkpoint_sha256=h('initial'), parent_checkpoint_sha256=None,
                 source_receipt_sha256=h('BC-receipt'), external_program_feedback_updates=dict(human=0, ai=0)),
            dict(stage_id='PPO', checkpoint_sha256=b.actor_sha256, parent_checkpoint_sha256=h('initial'),
                 source_receipt_sha256=h('PPO-receipt'), external_program_feedback_updates=dict(human=0, ai=0)),
        ])


def sample_plan():
    return dict(version='study-vnext-formal-sample-plan.v1', protocol_sha256=p.protocol_sha256(),
        participant_counts={d: dict(A=45, B=45, C=45) for d in p.DOMAINS},
        minimum_effects=dict(warehouse=dict(value=10.0, unit='native_team_points'),
                             pong=dict(value=8.0, unit='percentage_points'),
                             kitchen=dict(value=0.5, unit='orders')),
        target_power=0.8, familywise_alpha=0.05, multiple_comparison='Holm_FWER_0.05',
        power_report_sha256=h('synthetic-unverified-power-report'))


def test_spec_is_detached_closed_and_hash_stable():
    doc = p.protocol_document()
    assert p.PROTOCOL_ID == 'study-protocol-20261029.v1'
    assert p.CONSENT_VERSION == 'study-consent-20261029.v1'
    assert p.PROTOCOL_SHA256 == h(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    assert p.ProtocolSpec.from_dict(doc) == p.SPEC
    doc['analysis']['primary_task'] = 2
    assert p.protocol_document()['analysis']['primary_task'] == 3
    with pytest.raises(p.ProtocolError, match='unregistered'):
        p.ProtocolSpec.from_dict(doc)
    with pytest.raises(FrozenInstanceError):
        p.SPEC._canonical_json = '{}'
    assert p.protocol_document()['analysis']['formal_participant_counts'] is None
    assert p.protocol_document()['stages']['technical_pilot']['total_participants'] == 27
    assert p.FORMAL_ENROLLMENT_AUTHORITY is False


def test_estimands_do_not_adjust_away_model_treatment_or_use_post_treatment_questions():
    ab, ac, bc = (p.analysis_contract(c) for c in ('A-B', 'A-C', 'B-C'))
    assert ab['primary'] is True and ab['task1_baseline_adjustment'] is True
    assert 'task1_primary_metric' in ab['adjust_for']
    for contrast in (ac, bc):
        assert contrast['task1_baseline_adjustment'] is False
        assert 'task1_primary_metric' not in contrast['adjust_for']
    assert 'question_count' in ab['exclude_from_adjustment']
    assert ac['estimand'] == 'full_package_vs_raw_NN'
    assert bc['estimand'] == 'model_program_regularization'
    metrics = p.protocol_document()['analysis']['primary_metrics']
    assert metrics['pong']['name'] == 'percent_of_contact_capacity'
    assert metrics['pong']['native_score_is_not_primary'] is True
    assert metrics['kitchen']['field'] == 'score.metrics.completed_orders'
    assert p.protocol_document()['analysis']['unit'] == 'participant'


@pytest.mark.parametrize('group,active,target,decision,committed,allowed', [
    ('A', 2, 1, 0, 1, True), ('A', 2, 2, 6, 7, True),
    ('B', 2, 2, 0, 1, False), ('C', 2, 2, 0, 1, False),
    ('A', 1, 1, 0, 1, False), ('A', 3, 2, 0, 1, False),
    ('A', 2, 3, 0, 1, False), ('A', 2, 2, 0, 0, False),
    ('A', 2, 2, 7, 7, False), ('A', 2, 2, -1, 1, False),
    ('A', True, 1, 0, 1, False), ('A', 2, 2, False, 1, False),
])
def test_only_A_task2_committed_history_may_be_asked(group,active,target,decision,committed,allowed):
    assert p.question_permission(group,active,target,decision,committed) is allowed


def test_bindings_match_A_B_and_independent_raw_but_do_not_grant_acceptance():
    values = policies()
    result = p.validate_policy_set(values)
    assert result['policy_identity_sha256']['pong']['A'] == result['policy_identity_sha256']['pong']['B']
    assert result['policy_identity_sha256']['pong']['A'] != result['policy_identity_sha256']['pong']['C']
    assert result['formal_enrollment_allowed'] is False
    changed = policies(); changed['warehouse']['B'] = replace(changed['warehouse']['B'], actor_sha256=h('another'))
    with pytest.raises(p.ProtocolError, match='exact same binding'):
        p.validate_policy_set(changed)
    changed = policies(); changed['warehouse']['C'] = replace(changed['warehouse']['C'], ai_actor_sha256=changed['warehouse']['A'].ai_actor_sha256)
    with pytest.raises(p.ProtocolError, match='independent'):
        p.validate_policy_set(changed)
    changed = policies(); changed['pong']['C'] = replace(changed['pong']['C'], action_physics_config_sha256=h('other-physics'))
    with pytest.raises(p.ProtocolError, match='architecture/observation/physics'):
        p.validate_policy_set(changed)
    with pytest.raises(p.ProtocolError):
        p.validate_policy_set({'warehouse': values['warehouse']})


def test_entire_raw_ancestry_must_have_measured_zero_external_feedback():
    b = binding('warehouse', 'raw'); declaration = lineage(b)
    result = p.validate_raw_ancestry(b, declaration)
    assert result['requires_source_receipt_verification'] is True
    assert result['formal_enrollment_allowed'] is False
    for bad in (1, None, False):
        declaration = lineage(b); declaration['stages'][0]['external_program_feedback_updates']['human'] = bad
        with pytest.raises(p.ProtocolError, match='feedback or unmeasured'):
            p.validate_raw_ancestry(b, declaration)
    declaration = lineage(b); declaration['stages'][1]['parent_checkpoint_sha256'] = h('missing-parent')
    with pytest.raises(p.ProtocolError, match='disconnected'):
        p.validate_raw_ancestry(b, declaration)
    declaration = lineage(b); declaration['stages'][-1]['checkpoint_sha256'] = h('wrong-final')
    with pytest.raises(p.ProtocolError, match='deployed Actor'):
        p.validate_raw_ancestry(b, declaration)


def test_sample_registration_never_fabricates_computed_power():
    result = p.validate_sample_plan(sample_plan())
    assert result['total_participants'] == 405
    assert result['computed_power_verified'] is False
    assert result['formal_enrollment_allowed'] is False
    for mutate in (
        lambda v: v['participant_counts']['pong'].update(A=44),
        lambda v: v['participant_counts']['pong'].update(A=True, B=True, C=True),
        lambda v: v['minimum_effects']['kitchen'].update(unit='percentage_points'),
        lambda v: v.update(target_power=0.5),
        lambda v: v.update(multiple_comparison='none'),
        lambda v: v.update(power_report_sha256='not-a-hash'),
        lambda v: v.update(passed=True),
    ):
        value = sample_plan(); mutate(value)
        with pytest.raises(p.ProtocolError):
            p.validate_sample_plan(value)


def test_receipt_inventories_are_not_enrollment_permissions():
    references = {kind: dict(protocol_sha256=p.protocol_sha256(), sha256=h(kind))
                  for kind in p.formal_evidence_requirements()}
    result = p.validate_evidence_references(references)
    assert result['inventory_complete'] is True
    assert result['semantic_evidence_verified'] is False
    assert result['formal_enrollment_allowed'] is False
    references['model_admission']['passed'] = True
    with pytest.raises(p.ProtocolError, match='exact protocol/content identity'):
        p.validate_evidence_references(references)
    with pytest.raises(p.ProtocolError, match='all independently'):
        p.validate_evidence_references({'passed': True})
    with pytest.raises(p.ProtocolError):
        p.policy_variant(True)
