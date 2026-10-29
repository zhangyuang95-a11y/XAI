"""Read-only next-candidate diagnostics must never mint training authority."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.warehouse import v643_shutdown_diagnostic as d
from experiments.warehouse import v643_risk_recipe as recipe


def bind(tmp_path,name,value,binary=False):
    path=tmp_path/name
    path.write_bytes(value if binary else json.dumps(value).encode())
    return d.pointer(path)


def update(start,end,*,active=False,positive=0):
    return {'start_joint_step':start,'end_joint_step':end,'complete_episodes':2,
        'shutdown_episodes':positive,'partial_episodes_labeled_negative':0,
        'accepted_positive_program_gradient_updates':{'robot_1':0,'robot_2':0},
        'risk':{'previous_multiplier':1.,'next_multiplier':1.1,'phase':'increase'},
        'epochs':[{'updates':[{'actor_updated':active,'actor':{'gradient_norm':2.},
                              'common_gradient_clip_factor':.25}]}]}


def test_receipt_recount_separates_actor_frozen_failure_labels(tmp_path):
    refs=[bind(tmp_path,'warmup.json',update(0,4000,positive=1)),
          bind(tmp_path,'active.json',update(4000,20000,active=True,positive=1))]
    got=d.summarize_updates(refs,{})
    assert got['complete_training_episodes']==4 and got['shutdown_training_episodes']==2
    assert got['Actor_update_complete_episodes']==2 and got['Actor_update_shutdown_episodes']==1
    assert got['accepted_external_program_updates']=={'robot_1':0,'robot_2':0}
    assert got['individual_cost_gradient_or_critic_calibration_not_identifiable_from_receipts']


def test_corrupt_pointer_partial_labels_and_discontinuous_history_rejected(tmp_path):
    row=update(0,20000,active=True);ref=bind(tmp_path,'receipt.json',row)
    changed=dict(ref,sha256='0'*64)
    with pytest.raises(ValueError,match='hash mismatch'):d.summarize_updates([changed],{})
    row['partial_episodes_labeled_negative']=1
    with pytest.raises(ValueError,match='partial'):d.summarize_updates([bind(tmp_path,'partial.json',row)],{})
    refs=[bind(tmp_path,'one.json',update(0,4000,active=True)),bind(tmp_path,'gap.json',update(4001,20000,active=True))]
    with pytest.raises(ValueError,match='gap'):d.summarize_updates(refs,{})


def test_exact_paired_failure_comparison_does_not_promote_improvement():
    raw={str(i):i<5 for i in range(75)};reg={str(i):5<=i<7 for i in range(75)}
    value=d.paired_failure_difference(raw,reg)
    assert value['paired_discordant_scenes']==7
    assert value['two_sided_exact_discordance_p']==.453125
    assert not value['causal_program_effect_established']
    with pytest.raises(ValueError,match='paired scenes'):d.paired_failure_difference(raw,{'different':False})
    low,high=d.wilson_interval(5,75)
    assert 0<low<5/75<high<1


def failure_fixture(tmp_path):
    before={'frame':0,'agents':[{'agent_id':r,'active':True,'battery':2.,'position':[2,i],
        'carrying_task_id':None,'recent_positions':[[2,i]]} for i,r in enumerate(('robot_1','robot_2'))]}
    after=deepcopy(before);after['frame']=1;after['agents'][0].update(active=False,battery=0.,position=[3,0])
    case={'case_id':'spent-case','task':1}
    trace={'frame':0,'case':case,'pre_state_public':before,'post_state_public':after,
           'proposed_actions':{'robot_1':'DOWN','robot_2':'WAIT'},'action_override_agents':[],'actor_sha256':'old-actor'}
    trace_ref=bind(tmp_path,'trace.gz',gzip.compress((json.dumps(trace)+'\n').encode()),binary=True)
    x=np.zeros((1,2,173),np.float32);x[:,:,47]=.02;x[:,:,99]=-.04
    mask=np.ones((1,2,5),np.float32);p=np.full((1,2,5),.2,np.float32);sampled=np.array([[1,4]],np.int64)
    stream=io.BytesIO();np.savez(stream,x=x,m=mask,p=p,sampled=sampled)
    arrays=bind(tmp_path,'arrays.npz',stream.getvalue(),binary=True)
    outcome={'case':case,'variant':'raw','actor':{'sha256':'old-actor'},'physical_frames':1,'deliveries':3,'robot_collisions':1}
    entry={'case':case,'variant':'raw','trace':trace_ref,'arrays':arrays}
    return outcome,entry


def test_failure_details_are_actual_public_events_not_teacher_motives(tmp_path):
    outcome,entry=failure_fixture(tmp_path);got=d.failure_detail(outcome,entry,{})
    row=got['shutdown_agents'][0]
    assert row['role']=='robot_1' and row['pre_battery']==2 and row['post_battery']==0
    assert row['first_negative_charger_return_margin_decision_turn']==0
    assert row['selected_action']=='DOWN' and not got['new_NN_forward_or_environment_replay']


def test_sample_mismatch_or_unbound_trace_is_rejected(tmp_path):
    outcome,entry=failure_fixture(tmp_path);outcome['actor']['sha256']='different'
    with pytest.raises(ValueError,match='binding'):d.failure_detail(outcome,entry,{})
    outcome['actor']['sha256']='old-actor'
    payload=gzip.decompress(Path(entry['trace']['path']).read_bytes()).decode()
    payload=payload.replace('"DOWN"','"UP"')
    entry['trace']=bind(tmp_path,'different.gz',gzip.compress(payload.encode()),binary=True)
    with pytest.raises(ValueError,match='NN sample'):d.failure_detail(outcome,entry,{})


def test_proposal_retains_thresholds_and_never_authorizes_work(tmp_path):
    diag={'version':d.VERSION,'scope':'spent_development_evidence_diagnostic_only',
          'training_authority':False,'model_acceptance_passed':False,
          'new_environment_resets':0,'new_Actor_updates':0,'new_seed_allocations':0,
          'quality':{'raw':{'shutdown_rate':5/75}}}
    proposed=recipe.prepare_recipe(bind(tmp_path,'diagnostic.json',diag))
    assert proposed['development_quality']['shutdown_rate_max']==.05
    assert proposed['external_program']['raw_actual_external_gradient']==0
    assert proposed['common_recipe_changes']['risk_dual']['target']==.02
    assert not proposed['training_authority'] and not proposed['executor_implemented']
    assert proposed['starting_from_v642_raw_or_regularized_final_weights_forbidden']
    with pytest.raises(RuntimeError,match='not an executable'):recipe.execution_authority(proposed)
    diag['new_environment_resets']=1
    with pytest.raises(ValueError,match='new science'):recipe.prepare_recipe(bind(tmp_path,'dirty.json',diag))
