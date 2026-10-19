"""Cached/synthetic posthoc arithmetic; no NN, environment, fitting or rollout."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_warehouse_v622_posthoc as r


def cases():
    return [{'base':10*fold+task,'fold':fold,'task':task,'sampling_mode':'sampled'}
            for fold in range(3) for task in (1,2,3)]


def test_cluster_draws_preserve_whole_base_and_fold_task_counts():
    cs=cases()+[{**cases()[0],'base':999}]
    before=np.random.get_state();weights=r.bootstrap_weights(cs,draws=100)
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert np.array_equal(weights,r.bootstrap_weights(cs,draws=100))
    for fold in range(3):
        for task in (1,2,3):
            take=[i for i,c in enumerate(cs) if (c['fold'],c['task'])==(fold,task)]
            assert np.all(weights[:,take].sum(-1)==len(take))
    with pytest.raises(ValueError):r.bootstrap_weights(cs+[cs[0]],draws=10)
    with pytest.raises(ValueError):r.bootstrap_weights([{**cs[0],'sampling_mode':'argmax'}],draws=10)


def test_intervals_use_scenario_not_frame_iid_and_paired_same_draws():
    cs=[{'base':i,'fold':0,'task':1,'sampling_mode':'sampled'} for i in range(2)]
    weights=r.bootstrap_weights(cs,draws=2000)
    # 1000 identical physical rows in one episode remain a *single* cluster.
    out=r.ratio_estimate([0.,1.],[1000.,1.],weights)
    assert out['point']==pytest.approx(1/1001)
    assert out['ci95']==[0.,1.]
    doubled=r.ratio_estimate([0.,2.],[1000.,1.],weights)
    delta=r.paired_difference(doubled,out)
    assert delta['point']==pytest.approx(1/1001)
    assert delta['ci95']==[0.,1.] and delta['paired_same_scenario_draws']


def test_zero_denominator_and_one_scenario_are_not_claimed_precise():
    weights=np.asarray([[0,2],[1,1],[2,0]],np.int32)
    empty=r.ratio_estimate([0.,0.],[0.,0.],weights)
    assert empty['point'] is None and empty['ci95'] is None
    assert empty['valid_bootstrap_replicates']==0 and empty['zero_denominator_replicates']==3
    sparse=r.ratio_estimate([1.,0.],[1.,0.],weights)
    assert sparse['point']==1 and sparse['ci95'] is None
    assert sparse['zero_denominator_replicates']==1


def fixture_rows():
    from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
    from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
    from env.warehouse.navigation import ACTIONS,MOVE_DELTAS
    names={n:i for i,n in enumerate(OBSERVATION_FEATURE_NAMES)};points=list(layout.passable_positions)
    own=layout.charger_position;partner=points[-1]
    mask=np.asarray([float(a=='WAIT' or (own[0]+MOVE_DELTAS[a][0],own[1]+MOVE_DELTAS[a][1]) in points) for a in ACTIONS],np.float32)
    move=next(i for i in range(4) if mask[i])
    rows=[]
    for i,battery in enumerate((.1,.5)):
        obs=np.zeros(len(names),np.float32)
        for endpoint,pos in [('self',own),('partner',partner),('charger',layout.charger_position),
            ('order_0_pickup',points[0]),('order_0_delivery',points[-1]),('order_1_pickup',points[1]),('order_1_delivery',points[-2])]:
            obs[names[endpoint+'_row']]=pos[0]/(layout.rows-1);obs[names[endpoint+'_column']]=pos[1]/(layout.cols-1)
        for rr,line in enumerate(layout.tiles):
            for cc,tile in enumerate(line):obs[names[f'wall_row_{rr}_column_{cc}']]=tile=='#'
        obs[names['self_active']]=1.;obs[names['self_battery_fraction']]=battery
        p=mask*.1/(mask.sum()-1);p[4]=.9
        rows.append({'obs':obs,'mask':mask,'nn_probabilities':p,'sampled_action':ACTIONS[4 if i==0 else move]})
    tree={'root_ref':-1,'split_weights':[],'split_bias':[],'left_refs':[],'right_refs':[],'leaf_count':1}
    return {'structure':tree,'leaf_logits':[[0,0,0,0,2]],'feedback_leaves':[0],'explanation_leaves':[0]},rows


def test_cached_top_and_sampled_nn_actions_have_distinct_denominators():
    raw,rows=fixture_rows();values,strata=r.frame_statistics(raw,rows,2)
    totals=dict(zip(r.COLS,values.sum(0)))
    assert totals['frames']==2 and totals['F_qualified']==totals['E_qualified']==2
    assert totals['F_actual']==totals['E_actual']==1
    assert totals['F_expected']==pytest.approx(1.8)
    assert strata['NN_top_action:WAIT'].tolist()==[True,True]
    assert strata['sampled_NN_proposal:WAIT'].tolist()==[True,False]
    assert strata['battery_fraction:at_most_10_percent'].tolist()==[True,False]
    assert strata['task:2'].all()
    raw['feedback_leaves']=[];raw['explanation_leaves']=[]
    values,_=r.frame_statistics(raw,rows,2)
    assert values[:,r.COLS.index('frames')].sum()==2  # never drop uncovered frames
    assert values[:,r.COLS.index('F_qualified')].sum()==0


def test_completion_guard_before_program_or_cached_data_access(monkeypatch,tmp_path):
    from scripts import run_warehouse_v622_capacity as native
    path=tmp_path/'plan.json';path.write_text('{}')
    plan={'jobs':[{'job_id':'job-0'}],'identity':{'source_closure_sha256':'a'*64},'folds':[],'stage':'primary'}
    monkeypatch.setattr(native,'load_plan',lambda value:(path,plan))
    (tmp_path/'summary.json').write_text(json.dumps({'jobs':{},'plan_sha256':sha256(path.read_bytes()).hexdigest()}))
    (tmp_path/'independent-audit.json').write_text(json.dumps({'passed':False}))
    evidence=r.Evidence()
    with pytest.raises(RuntimeError,match='every registered job'):
        r.completed_inputs([path],evidence)
    assert not any('program' in name or '.jsonl' in name for name in evidence.hashes)


def test_report_is_separate_from_selection_and_never_calls_nn_or_fit():
    tree=ast.parse(Path(r.__file__).read_text())
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name):calls.append(node.func.id)
            elif isinstance(node.func,ast.Attribute):calls.append(node.func.attr)
    assert not set(calls)&{'checked_actor','target_identity','logits_encoded','act','step','reset','fit_program','fit_structure','minimize','summarize','summarize_units'}
    assert set(r.PAIRED_METRICS).issubset({k for k in r.METRICS if k.endswith('_coverage')})
