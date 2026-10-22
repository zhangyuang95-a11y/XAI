"""Direct coverage objective, hard routing, source budget and no action output."""
import numpy as np
import pytest
from backend.training import warehouse_v634_partitions as p
from backend.training.warehouse_v6_tree_routing import split_scores,HARD_SPLIT_TIE_EPSILON
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v630_program import Program
from experiments.warehouse import v630_protocol as v


def test_precision_cut_exposes_reliable_tail_that_entropy_does_not_choose():
    scores=np.arange(1000,dtype=np.float32);labels=np.ones(1000)
    labels[:300:2]=0.;labels[300:800:10]=0.
    _,entropy=p.old._threshold(scores,labels)
    old_mass=-p.score_split(labels,scores>entropy,.99)[0]
    cuts=p.thresholds(scores,labels,.99)
    new_mass=max(-p.score_split(labels,scores>cut,.99)[0] for cut in cuts)
    assert new_mass>=200 and new_mass>old_mass
    for cut in cuts:assert min((scores>cut).sum(),(scores<=cut).sum())>=40


def test_cached_float64_reduction_matches_shared_hard_router_at_nearby_boundaries():
    rng=np.random.default_rng(634);x=rng.normal(size=(100,173)).astype(np.float32)
    w=np.zeros(173,np.float32);w[[0,5,40]]=[.125,-17.,.03]
    linear=split_scores(x,w[None],np.zeros(1,np.float32))[:,0]
    for bias in [0.,1.,-100.,*linear[:10]]:
        b=np.float32(bias)
        cached=linear+float(b)
        actual=split_scores(x,w[None],np.array([b],np.float32))[:,0]
        assert np.array_equal(cached,actual)
        assert np.array_equal(cached>HARD_SPLIT_TIE_EPSILON,actual>HARD_SPLIT_TIE_EPSILON)


def test_real_proposals_respect_source_depth_and_charge_only_classifier_calls():
    _,plan=v.checked_plan();root=v.ROOT/'output/warehouse_v633/warehouse-v633-support-refresh-01'
    result=v.bound(v.pointer(root/'fold-0-robot_1/result.json'))
    source=Program.from_dict(v.bound(result['program']),plan['actor_role_sha256']['robot_1'])
    base,_=p.old.collapse_protected(source);ref=v.bound(plan['collection'])['cases']['dev-0300']['arrays']
    v.require(v.pointer(ref['path'])==ref,'fixture changed')
    with np.load(ref['path']) as z:x=z['x'][:,0].copy();m=z['m'][:,0].copy()
    route,_,q=base.distributions(x,m)
    leaf=next(int(i) for i in route if len(p.old.region(base.structure,int(i),256)['path'])<=7)
    row=np.flatnonzero(route==leaf)[0];xx=np.repeat(x[row:row+1],200,axis=0);mm=np.repeat(m[row:row+1],200,axis=0)
    path=p.old.region(base.structure,leaf,256)['path'];used={j for a in path for j,w in enumerate(a['weight']) if w}|set(base.leaves[leaf]['support'])
    d=domain_table()['features'];feature=next(j for j,a in enumerate(d) if j not in used and a['lower']==0 and a['upper']==1 and not a['name'].startswith('legal_'))
    xx[:,feature]=np.linspace(0,1,200,dtype=np.float32);_,_,qq=base.distributions(xx,mm)
    top=int(qq[0].argmax());other=next(a for a in np.flatnonzero(mm[0]) if a!=top)
    target=qq.copy();bad=np.arange(200)%10==0
    target[bad,top],target[bad,other]=qq[bad,other],qq[bad,top]
    class Budget:
        def __init__(self):self.calls=[]
        def check(self):pass
        def solve(self,kind,metadata,fn):self.calls.append((kind,metadata));return fn()
    b=Budget();before=source.to_dict();out=p.propose(base,dict(x=xx,m=mm,p=target),leaf,b,.99)
    assert source.to_dict()==before and not out['execution_actions_produced'] and out['classifier_outputs_discarded']
    assert len(b.calls)<=49 and all(kind=='node' and p._support_cost(meta['source_indices'])<=3 for kind,meta in b.calls)
    for t in out['proposals']:
        _,info=p.old.structure_info(t,64);assert info['depth']<=min(3,9-len(path)) and info['maximum_source_features_per_node']<=3
    with pytest.raises((ValueError,RuntimeError)):p.propose(base,dict(x=xx,m=mm,p=target),leaf,b,.9)
