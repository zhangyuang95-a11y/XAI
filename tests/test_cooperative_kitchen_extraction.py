"""Posthoc labels, split isolation, counterfactual evidence and frozen questions."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from core.program import ExecutableProgram, ProgramNode
from env.cooperative_kitchen import CooperativeKitchen,ACTIONS,OBSERVATION_FEATURES
from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
from backend.cooperative_kitchen.calibration import unique_scenarios,make_pairs,make_questionnaire
from backend.cooperative_kitchen.splits import scenario_fingerprint
from backend.training.cooperative_kitchen_extract import collect,intervention_pairs,audit,_actor_row,CRITICAL_GROUPS

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def actor():
    path=ROOT/'output/cooperative_kitchen/smoke/actor_000000256.npz'
    if not path.exists():pytest.skip('Genuine 256-step training smoke artifact unavailable')
    return NumpyKitchenPolicy(path)


def constant_program():
    return ExecutableProgram(tuple(ACTIONS),tuple(OBSERVATION_FEATURES),ProgramNode(probabilities=(1.,0.,0.,0.,0.,0.)))


def test_disjoint_scene_allocator_rejects_actual_fingerprint():
    excluded={scenario_fingerprint(CooperativeKitchen(seed=2000000,scenario_id='generated').public_view())}
    fit,rejected=unique_scenarios('extraction_fit',3,excluded)
    assert 2000000 in rejected
    assert all(d['fingerprint'] not in excluded for d in fit)
    selection,_=unique_scenarios('extraction_selection',3,excluded|{d['fingerprint'] for d in fit})
    assert not {d['fingerprint'] for d in fit}&{d['fingerprint'] for d in selection}


def test_actual_frozen_actor_supplies_all_extraction_labels(actor):
    descriptors,_=unique_scenarios('extraction_fit',1)
    rows=collect(actor,descriptors,sample_stride=20,intervention_stride=60)
    assert rows
    for row in rows:
        features=np.asarray([row['features'][name] for name in OBSERVATION_FEATURES],dtype=np.float32)
        action,distribution=actor.act({'ai':features})
        assert action['ai']==row['neural_action']
        np.testing.assert_array_equal([row['probabilities'][a] for a in ACTIONS],distribution['ai']['probabilities'])
        assert 'program' not in row['source']


def test_counterfactual_evidence_isolation_and_explicit_hypothesis(actor):
    env=CooperativeKitchen(scenario_id='base_congestion')
    before=env.snapshot();scene={'seed':0,'fingerprint':scenario_fingerprint(env.public_view())}
    pairs=intervention_pairs(actor,env,scene)
    assert env.snapshot()==before
    kinds={b['intervention_kind'] for _,b in pairs}
    assert 'counter_clear_hypothetical' in kinds
    assert 'player_actions_three_steps' in kinds
    for a,b in pairs:
        assert a['pair']==b['pair'] and a['intervention_role']=='baseline' and b['intervention_role']=='altered'
        assert a['assumption']==b['assumption']
        if b['intervention_kind']=='counter_clear_hypothetical':
            assert 'not a one-step' in b['assumption']['scope']
            assert a['frame']==b['frame']==0
        else:assert a['frame']==b['frame']==3


def test_missing_critical_and_uninformative_pairs_do_not_pass():
    row={'features':{},'probabilities':dict(zip(ACTIONS,[1.,0.,0.,0.,0.,0.])),
         'groups':['ordinary'],'scene':'one','seed':1,'frame':0,'intervention':False}
    result,evidence=audit(constant_program(),[row])
    assert result['overall_fidelity']==1
    assert not result['passes']
    assert all(result['critical_states'][g]['n']==0 for g in CRITICAL_GROUPS)
    a={**row,'intervention':True,'pair':'same','intervention_kind':'counter_clear_hypothetical','intervention_role':'baseline'}
    b={**a,'intervention_role':'altered'}
    result,evidence=audit(constant_program(),[row,a,b])
    assert result['counterfactual_by_mechanism']['counter_clear_hypothetical']['nontrivial_pairs']==0
    assert evidence[0]['direction_correct'] is None and not result['passes']


def test_nontrivial_direction_uses_true_probability_difference():
    row={'features':{},'probabilities':dict(zip(ACTIONS,[.6,.4,0,0,0,0])),
         'groups':['ordinary',*CRITICAL_GROUPS],'scene':'one','seed':1,'frame':0,'intervention':False}
    a={**row,'intervention':True,'pair':'changed','intervention_kind':'counter_clear_hypothetical','intervention_role':'baseline'}
    b={**a,'intervention_role':'altered','probabilities':dict(zip(ACTIONS,[.4,.6,0,0,0,0]))}
    metrics,evidence=audit(constant_program(),[row,a,b])
    assert evidence[0]['nontrivial'] and evidence[0]['actor_action_changed']
    assert evidence[0]['direction_correct'] is False
    assert evidence[0]['program_probability_change']==0
    assert not metrics['passes']


def test_incomplete_counterfactual_pair_rejected():
    row={'features':{},'probabilities':dict(zip(ACTIONS,[1.,0.,0.,0.,0.,0.])),
         'groups':['ordinary'],'scene':'one','seed':1,'frame':0,'intervention':True,'pair':'missing',
         'intervention_role':'baseline','intervention_kind':'counter_clear_hypothetical'}
    with pytest.raises(ValueError,match='Incomplete'):audit(constant_program(),[row])


def test_calibration_pairs_are_program_matched_and_cover_three_states():
    descriptors,_=unique_scenarios('calibration',30)
    pairs,evaluated=make_pairs(descriptors)
    assert len(pairs)==3 and {p['initial_kind'] for p in pairs}=={'empty','inprogress','congestion'}
    assert all(max(p['baseline_steps'])<=160 and p['step_difference']<=15 for p in pairs)
    assert len({d['fingerprint'] for p in pairs for d in p['scenarios']})==6
    assert all('correct_answer' not in d for p in pairs for d in p['scenarios'])


def test_frozen_eight_questions_have_truth_labels_and_public_option_values(actor):
    descriptors,_=unique_scenarios('questionnaire',3)
    q=make_questionnaire(actor,descriptors)
    assert len(q['items'])==8
    assert sum(i['type']=='prediction' for i in q['items'])==4
    assert sum(i['type']=='counterfactual' for i in q['items'])==4
    assert all(i['correct_answer'] in {o['value'] for o in i['options']} for i in q['items'])
    assert all(i['actor_sha256']==actor.artifact_sha256 for i in q['items'])
    assert all(set(i['prompt'])=={'zh','en'} for i in q['items'])
    for i in q['items']:
        if i['type']=='counterfactual':assert len(i['private_trajectory'])==3 and i['correct_answer']==i['private_trajectory'][-1]


def test_validation_final_scenes_exclude_optimization_and_each_other():
    from backend.training.cooperative_kitchen_validation import evaluation_scenes,optimization_fingerprints
    validation,_=evaluation_scenes('validation',60)
    final,rejected=evaluation_scenes('final_test',60)
    train=optimization_fingerprints()
    validation_ids={r['fingerprint'] for r in validation}
    final_ids={r['fingerprint'] for r in final}
    assert not validation_ids&train and not final_ids&train and not final_ids&validation_ids
    assert len(validation_ids)==60 and len(final_ids)==60 and rejected


def test_posthoc_metadata_cannot_claim_training_feedback_or_failed_explanation():
    from backend.training.cooperative_kitchen_extract import posthoc_program_payload
    program=constant_program()
    before=program.to_dict()
    payload=posthoc_program_payload(program,{'passes':False},selection_passes=True)
    assert payload['root']==before['root']
    metadata=payload['metadata']
    assert metadata['training_role']=='posthoc_extraction_only'
    assert metadata['program_roles']==['local_explanation_audit']
    assert metadata['feedback_eligible'] is False and metadata['metrics']['feedback_eligible'] is False
    assert metadata['explanation_eligible'] is False and metadata['metrics']['explanation_eligible'] is False
    assert metadata['metrics']['explanation_ineligibility_reasons']==['kitchen_held_out_extraction_gate_failed']
