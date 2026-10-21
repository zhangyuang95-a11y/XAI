"""Synthetic target distributions only; no scientific candidate experiment."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from backend.training import warehouse_v630_partition_search as c
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v629_program import protected_indices
from scripts.audit_warehouse_v626 import basic,same_metrics


class Budget:
    def __init__(self):self.calls=[];self.checkpoints=0
    def check(self):pass
    def solve(self,kind,metadata,callback):self.calls.append((kind,metadata));return callback()
    def checkpoint(self,*args,**kwargs):self.checkpoints+=1


@pytest.fixture(scope='module')
def data():
    spec=importlib.util.spec_from_file_location('search_program_fixture',Path(__file__).with_name('test_warehouse_v630_program.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before,_,_,x,m,*_=module.refined.__wrapped__()
    route,_,q=before.distributions(x,m);domain=domain_table()['features'];chosen=None
    for i in np.flatnonzero((q.max(1)>.85)&(m.sum(1)>1)):
        leaf=int(route[i]);path=c.region(before.structure,leaf,before.maximum_leaves)['path']
        if leaf not in protected_indices(before) or len(path)>=9:continue
        used=set(before.leaves[leaf]['support'])|{j for p in path for j,w in enumerate(p['weight']) if w}
        feature=next((j for j,d in enumerate(domain) if j not in used and not d['constant']
            and d['lower']<.25<.75<d['upper'] and not d['name'].startswith('legal_')),None)
        if feature is not None:chosen=(i,leaf,feature);break
    assert chosen is not None
    index,leaf,feature=chosen;xx=np.repeat(x[index:index+1],600,axis=0);mm=np.repeat(m[index:index+1],600,axis=0)
    cases=[];rows=[]
    for task in (1,2,3):
        for n in range(200):
            episode=f'synthetic-{task}-{n:03}';cases.append(dict(case_id=episode,task=task,sampling_mode='sampled'))
            rows.append(dict(episode=episode,task=task,base_seed=episode))
    xx[:,feature]=np.tile([.25,.75],300)
    # Surrogate output is constant w.r.t this unused feature. A synthetic
    # 'NN' target agrees on one side and swaps two legal probabilities on the
    # other. This is NOT the real frozen NN and cannot be acceptance evidence.
    rr,zz,qq=before.distributions(xx,mm);assert np.all(rr==leaf)
    target=qq.copy();top=int(qq[0].argmax());other=next(j for j in np.flatnonzero(mm[0]) if j!=top)
    bad=xx[:,feature]<.5;target[bad,top]=qq[bad,other];target[bad,other]=qq[bad,top]
    sampled=target.argmax(1);a=dict(x=xx,m=mm,p=target,sampled=sampled,rows=rows)
    return before,a,cases,leaf,feature


def test_lanes_are_whole_scene_disjoint_balanced_and_exclude_calibration(data):
    _,a,cases,*_=data;parts=c.lane_parts(a,cases)
    ids={key:{r['episode'] for r in value['rows']} for key,value in parts.items()}
    assert {k:len(v) for k,v in ids.items()}==dict(fit=300,eligibility=150,validation=150,calibration=0)
    assert all(not x&y for i,x in enumerate(ids.values()) for y in list(ids.values())[:i])
    for key in ('fit','eligibility','validation'):
        assert len({r['task'] for r in parts[key]['rows']})==3


def test_risk_repeats_use_ninety_whole_task_balanced_scenes(data):
    _,a,cases,*_=data;parts=c.lane_parts(a,cases);risk=parts['eligibility'];masks=c.risk_subsets(risk['rows'])
    assert len(masks)==30 and all(int(m.sum())==90 for m in masks)
    assert all(sum(keep and row['task']==task for keep,row in zip(mask,risk['rows']))==30
        for mask in masks for task in (1,2,3))
    assert all(np.array_equal(x,y) for x,y in zip(masks,c.risk_subsets(risk['rows']),strict=True))


def test_cached_frame_metrics_equal_independent_whole_frame_arithmetic(data):
    source,a,cases,leaf,_=data;pred=source.distributions(a['x'],a['m']);signals=c.frame_signals(source,a,pred)
    selected=pred[0]==leaf;m=c.row_metrics(signals,selected)
    same_metrics(m,basic(source,a,pred,{leaf},'explanation'))


@pytest.fixture(scope='module')
def searched(data):
    source,a,cases,leaf,feature=data;parts=c.lane_parts(a,cases);budget=Budget()
    after,report=c.run(source,parts,budget,scope='protected')
    return source,a,after,report,budget


def test_learned_condition_and_global_budget_complete_without_changing_outputs(searched):
    source,a,after,report,budget=searched
    old=source.distributions(a['x'],a['m']);new=after.distributions(a['x'],a['m'])
    assert np.array_equal(old[1],new[1]) and np.array_equal(old[2],new[2])
    assert report['mandatory_complete'] and report['reliable_selection_passed']
    assert len(budget.calls)==1 and budget.calls[0][0]=='node' and budget.checkpoints>=len(source.leaves)
    assert c.structure_info(after.structure,256)[1]['depth']<=9 and len(after.leaves)<=256
    assert len(report['candidates'])==4 and report['coefficient_updates']==0
    metrics=report['candidates'][report['best_candidate']]['metrics']
    assert metrics['explanation']['coverage']>=.45 and metrics['feedback']['coverage']>=.45
    assert report['optimizer_scope'].startswith('one_CART')


def test_control_has_no_fit_and_no_new_condition(data):
    source,a,cases,*_=data;b=Budget();after,report=c.run(source,c.lane_parts(a,cases),b,scope='control')
    assert report['mandatory_complete'] and not b.calls and not b.checkpoints and len(after.leaves)==len(source.leaves)


def test_risk_lane_cannot_alias_fit_or_selection(data):
    source,a,cases,*_=data;parts=c.lane_parts(a,cases);parts['eligibility']['rows'][0]=parts['fit']['rows'][0]
    with pytest.raises((RuntimeError,ValueError)):c.run(source,parts,Budget(),scope='protected')
