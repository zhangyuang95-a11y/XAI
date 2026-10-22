"""Numerical distribution proposal and source-budget boundaries, no rollout."""
import numpy as np
import pytest
from backend.training import warehouse_v637_leaf_refresh as s

class Budget:
    def __init__(self):self.counts=dict(node=0,leaf=0,proof=0)
    def check(self):pass
    def solve(self,kind,metadata,fn):self.counts[kind]+=1;return fn()


def test_dense_complete_masked_distribution_gradient():
    rng=np.random.default_rng(0);u=rng.normal(size=(9,4));m=np.ones((9,5));m[:4,1]=0
    p=rng.uniform(size=m.shape)*m;p/=p.sum(1)[:,None];theta=rng.normal(size=25)*.1
    value,grad=s.dense_objective(theta,u,m,p);eps=1e-6
    for j in range(len(theta)):
        plus=theta.copy();minus=theta.copy();plus[j]+=eps;minus[j]-=eps
        finite=(s.dense_objective(plus,u,m,p)[0]-s.dense_objective(minus,u,m,p)[0])/(2*eps)
        assert grad[j]==pytest.approx(finite,abs=2e-7)
    assert np.isfinite(value)


def test_group_distribution_proposal_finds_public_signal_and_never_becomes_leaf():
    d=s.domain_table()['features'];ids=[i for i,v in enumerate(d) if not v['constant'] and v['lower']==0 and v['upper']==1 and s._support_cost([i])==1][:2]
    x=np.repeat(np.array([v['center'] for v in d],np.float32)[None],240,axis=0)
    x[:,ids[0]]=np.linspace(0,1,240,dtype=np.float32);x[:,ids[1]]=np.repeat([.2,.8],120)
    z=np.zeros((240,5));z[:,0]=5*(x[:,ids[0]]-.5);z[:,4]=-z[:,0]
    from scipy.special import softmax
    p=softmax(z,axis=1);m=np.ones_like(p);budget=Budget()
    ranked,report=s.dense_sources(x,m,p,budget,dict(leaf=0))
    assert ranked[0]==ids[0] and report['dense_output_never_stored_as_program']
    assert budget.counts==dict(node=1,leaf=0,proof=0)
    assert 'weights' not in report and 'bias' not in report and s.TRIALS==24


def test_sparse_selection_and_final_calibration_have_real_path_certificates():
    from scipy.special import softmax
    from backend.training.warehouse_v626_search import encode
    from backend.training.warehouse_v622_program import MASK_INDICES
    from scripts.audit_warehouse_v626 import independent_dual
    d=s.domain_table()['features'];feature=next(i for i,v in enumerate(d) if not v['constant'] and v['lower']==0 and v['upper']==1 and i not in MASK_INDICES and s._support_cost([i])==1)
    x=np.repeat(np.array([v['center'] for v in d],np.float32)[None],240,axis=0);x[:,MASK_INDICES]=1
    x[:,feature]=np.linspace(0,1,240,dtype=np.float32);m=np.ones((240,5),np.float32)
    z=np.zeros_like(m);z[:,4]=2+3*x[:,feature];p=softmax(z,axis=1).astype(np.float32)
    structure,_=encode(None);path=s.region(s.plain(structure),0,256);budget=Budget()
    incumbent=s._trial(x,m,p,(),4,path,{4},True,budget)
    assert incumbent is not None
    selected,report=s.select_leaf(incumbent,(x[::2],m[::2],p[::2]),(x[1::2],m[1::2],p[1::2]),path,{4},budget,0)
    assert feature in selected['support'] and not report['selected_original'] and len(report['trials'])<=24
    assert s._support_cost(selected['support'])<=3 and len(selected['weights'])*len(selected['weights'][0])+len(selected['bias'])<=20
    fitted=s.fit_leaf(x,m,p,arm='C3',path=path,legal={4},budget=budget,initial=selected,final=True)
    fitted=s.calibrate(fitted,x,m,p,arm='C3',path=path,legal={4},budget=budget)
    independent_dual(fitted,path,{4});_,q=s.evaluate_leaf(fitted,x,m)
    assert np.max(np.abs(p-q))<.04 and np.all(q.argmax(1)==p.argmax(1))
    assert budget.counts['leaf']>0 and fitted['certificate']['passed']
    # A stump with this one-source witness has an exact interval proof; no
    # optimization LP is required for that legitimate certificate.
