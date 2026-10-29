"""All exports and outcomes here are synthetic; no participant data are read."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from runtime_v642.contracts import PolicyBinding, ProgramBinding, digest
from study_vnext import analysis as a
from study_vnext.protocol import DOMAINS, GROUPS, protocol_sha256, protocol_document


def h(text): return sha256(text.encode()).hexdigest()


def bound(domain, variant):
    programs = () if variant=='raw' else tuple(ProgramBinding(role=r,native_role=r,reader_schema='synthetic',
        archive_sha256=h(domain+r+'archive'),source_program_sha256=h(domain+r+'source'),independent_audit_sha256=h(domain+r+'audit')) for r in ('human','ai'))
    return PolicyBinding(release_id='synthetic-release',policy_id=domain+'-'+variant,domain=domain,variant=variant,
        model_schema='synthetic',method_schema='synthetic',actor_sha256=h(domain+variant+'pair'),
        human_actor_sha256=h(domain+variant+'human'),ai_actor_sha256=h(domain+variant+'ai'),
        feature_schema_sha256=h(domain+'feature'),action_physics_config_sha256=h(domain+'physics'),
        programs=programs,accepted_receipt_sha256=h(domain+'receipt'))


BINDINGS = {d:{g:bound(d,'raw' if g=='C' else 'regularized') for g in GROUPS} for d in DOMAINS}
LOCKED = {d:{g:digest(b.to_dict()) for g,b in values.items()} for d,values in BINDINGS.items()}


def exported(domain='warehouse',group='A',person='one',mode='pilot',missing_task3=False):
    b=BINDINGS[domain][group];iid=h(domain+group+person)[:32]
    instance=dict(id=iid,release_id='synthetic-release',domain=domain,mode=mode,group=group,scenario_seed=1,
                  stage='task_3' if missing_task3 else 'complete',binding=b.to_dict(),binding_sha256=digest(b.to_dict()))
    runs,frames,capacity=[],[],{}
    for task in (1,2,3):
        if task==3 and missing_task3:continue
        runid=iid+'-'+str(task)
        native=dict(lanes=9,_schedule=[dict(arrival_turn=1,ball=dict(kind='ordinary',contacts=[1])),
                                      dict(arrival_turn=2,ball=dict(kind='cooperative',contacts=[2,3]))])
        initial=dict(domain=domain,task=task,seed=1,turn=0,terminal=False,snapshot=native)
        final={**deepcopy(initial),'turn':1,'terminal':True}
        run=dict(id=runid,instance_id=iid,task=task,status='complete',initial_state_json=json.dumps(initial),
                 initial_state_sha256=digest(initial),state_json=json.dumps(final))
        runs.append(run);previous=None
        for turn,state in ((0,initial),(1,final)):
            score=dict(task_score=-10+task if turn else 0,raw_score=1 if turn else 0,
                       metrics=dict(completed_orders=task if turn else 0))
            if domain=='pong':score['task_score']=99 if turn else 0
            frame=dict(version='synthetic-frame-not-a-real-neural-evaluation',turn=turn,binding=b.to_dict(),binding_sha256=digest(b.to_dict()),
                       state=deepcopy(state),public=dict(score=score),decision={},transition=None,
                       previous_frame_sha256=previous)
            previous=digest(frame);frames.append(dict(run_id=runid,**frame,frame_sha256=previous))
        capacity[runid]=dict(capacity=4,initial_state_sha256=run['initial_state_sha256'],
            definition_sha256=a.CONTACT_CAPACITY_DEFINITION_SHA256,verification_receipt_sha256=h('synthetic-unverified-capacity-report'))
    result=dict(instance=instance,binding=b.to_dict(),protocol_sha256=protocol_sha256(),study_protocol=protocol_document(),all_AI_frames_recomputed=True,
                records=dict(instances=[dict(**instance,participant_key=h(person))],runs=runs,frames=frames))
    return result,capacity


def extract(exports,capacity=None):
    return a.participant_rows(exports,expected_release_id='synthetic-release',expected_binding_sha256=LOCKED,capacity_evidence=capacity)


def test_primary_quantities_use_exact_registered_native_units_and_one_row_per_person():
    w,_=exported(person='w');p,capacity=exported('pong',person='p');k,_=exported('kitchen',person='k')
    result=extract([w,p,k],capacity)
    assert len(result['rows'])==3
    assert [r['task3'] for r in result['rows']]==[-7.0,25.0,3.0]
    assert [r['unit'] for r in result['rows']]==['native_team_points','percentage_points','orders']
    assert result['scientific_acceptance_claimed'] is False
    assert result['formal_enrollment_allowed'] is False
    assert result['missing_primary_outcomes']==[]


def test_pong_requires_independent_denominator_identity_and_checks_schedule_not_native_score():
    p,capacity=exported('pong')
    with pytest.raises(a.AnalysisError,match='independent Pong'):extract([p])
    for mutate in (
        lambda x:x.update(capacity=8),
        lambda x:x.update(definition_sha256=h('wrong-definition')),
        lambda x:x.update(initial_state_sha256=h('wrong-origin')),
        lambda x:x.update(verification_receipt_sha256=''),
    ):
        wrong=deepcopy(capacity);mutate(next(iter(wrong.values())))
        with pytest.raises(a.AnalysisError):extract([p],wrong)
    assert extract([p],capacity)['rows'][0]['task3']!=99


def test_preview_excluded_missing_outcome_retained_and_no_silent_zero_or_complete_case_analysis():
    preview,_=exported(mode='preview');missing,_=exported(person='missing',missing_task3=True)
    result=extract([preview,missing])
    assert len(result['excluded'])==1 and result['excluded'][0]['reason']=='non_scientific_mode'
    assert len(result['rows'])==1 and result['rows'][0]['task3'] is None
    assert result['missing_primary_outcomes']==[h('missing')]
    rows=design_rows();rows[0]['task3']=None
    with pytest.raises(a.AnalysisError,match='missing outcomes'):a.contrast_design(rows,domain='warehouse',contrast='A-B')


def test_duplicate_person_mixed_release_binding_and_corrupt_frame_are_rejected():
    w,_=exported();other,_=exported('kitchen',person='same')
    with pytest.raises(a.AnalysisError,match='duplicate'):extract([w,w])
    other['records']['instances'][0]['participant_key']=h('one')
    with pytest.raises(a.AnalysisError,match='duplicate'):extract([w,other])
    for mutate in (
        lambda x:x['instance'].update(release_id='older-release'),
        lambda x:x['instance'].update(binding_sha256=h('wrong-binding')),
        lambda x:x['records']['frames'][-1]['public']['score'].update(task_score=100000),
        lambda x:x.update(all_AI_frames_recomputed=False),
        lambda x:x.update(protocol_sha256=h('different-protocol')),
    ):
        bad=deepcopy(w);mutate(bad)
        with pytest.raises(a.AnalysisError):extract([bad])


def design_rows():
    rows=[]
    for group in GROUPS:
        for index in range(8):
            baseline=float(index-3)
            rows.append(dict(participant_key=h(group+str(index)),domain='warehouse',group=group,
                release_id='synthetic-release',scenario_seed=index%2,binding_sha256=LOCKED['warehouse'][group],
                task1=baseline,task2=0,task3=5+(2 if group=='A' else 0)+baseline*.3+(index%3-1)*.4,
                metric='native_team_score',unit='native_team_points'))
    return rows


def test_contrast_design_uses_task1_only_for_explanation_contrast_and_keeps_scene_block():
    rows=design_rows()
    for contrast in ('A-B','A-C','B-C'):
        design=a.contrast_design(rows,domain='warehouse',contrast=contrast)
        assert ('task1_primary_metric' in design['columns'])==(contrast=='A-B')
        assert 'scenario_1' in design['columns']
        assert len(design['y'])==16
    with pytest.raises(a.AnalysisError,match='pseudoreplication'):
        a.contrast_design(rows+[rows[0]],domain='warehouse',contrast='A-B')
    rows[0]['unit']='percentage_points'
    with pytest.raises(a.AnalysisError,match='metric definitions'):a.contrast_design(rows,domain='warehouse',contrast='A-B')


def test_hc3_reports_absolute_units_and_uncertainty_without_effect_claim_or_negative_baseline_ratio():
    result=a.hc3_contrast(design_rows(),domain='warehouse',contrast='A-B')
    assert np.isclose(result['estimate_first_minus_second'],2)
    assert result['CI95'][0]<2<result['CI95'][1]
    assert result['unit']=='native_team_points'
    assert result['relative_improvement_not_computed'] is True
    assert result['effect_claim']=='not_automatically_determined'
    assert 0<=result['p_value_asymptotic_normal']<=1
    assert result['formal_enrollment_allowed'] is False


def test_holm_uses_all_three_domain_primary_results():
    result=a.holm_three_domains(dict(warehouse=.01,pong=.03,kitchen=.04))
    assert result['warehouse']['holm_p']==.03 and result['warehouse']['rejected_at_FWER_0_05']
    assert np.isclose(result['pong']['holm_p'],.06) and not result['pong']['rejected_at_FWER_0_05']
    assert np.isclose(result['kitchen']['holm_p'],.06)
    with pytest.raises(a.AnalysisError):a.holm_three_domains(dict(warehouse=.01))


def test_power_uses_explicit_assumptions_is_reproducible_and_does_not_select_formal_N():
    metrics=protocol_document()['analysis']['primary_metrics']
    assumptions={d:dict(meaningful_effect=.8,outcome_sd=1.,task1_correlation=.4,unit=metrics[d]['unit']) for d in DOMAINS}
    result=a.power_candidates(assumptions,[8,40],simulations=2000,seed=4)
    assert result==a.power_candidates(assumptions,[8,40],simulations=2000,seed=4)
    assert all(result['candidates'][1]['estimated_primary_power'][d]>result['candidates'][0]['estimated_primary_power'][d] for d in DOMAINS)
    assert result['candidates'][1]['total_participants']==360
    assert result['selected_formal_N'] is None
    assert result['computed_trial_power_verified'] is False and result['formal_enrollment_allowed'] is False
    for mutate in (lambda x:x['warehouse'].update(outcome_sd=0),lambda x:x['pong'].update(task1_correlation=1),lambda x:x['kitchen'].update(unit='points')):
        bad=deepcopy(assumptions);mutate(bad)
        with pytest.raises(a.AnalysisError):a.power_candidates(bad,[8],simulations=100)
