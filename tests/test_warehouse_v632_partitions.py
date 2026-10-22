"""Numeric/certificate and synthetic classifier checks, not model admission."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from backend.training import warehouse_v632_partitions as s
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v626_search import decode, encode
from backend.training.warehouse_v622_program import _route
from backend.training.warehouse_v630_program import Program
from experiments.warehouse import v630_protocol as v
from scripts.audit_warehouse_v628 import forward


class Budget:
    def __init__(self): self.calls=[]
    def check(self): pass
    def solve(self, kind, metadata, fn): self.calls.append((kind,metadata)); return fn()


@pytest.fixture(scope='module')
def asset():
    folder=v.ROOT/'output/warehouse_v631/diagnostics/full-fixed-refit-matrix-20261022-03'
    result=v.bound(v.pointer(folder/'fold-0-robot_1/result.json'))
    _,plan=v.checked_plan()
    source=Program.from_dict(v.bound(result['program']), plan['actor_role_sha256']['robot_1'])
    collection=v.bound(plan['collection']); x=[];m=[]
    for k in range(300,312):
        ref=collection['cases'][f'dev-{k:04}']['arrays'];v.require(v.pointer(ref['path'])==ref,'fixture input changed')
        with np.load(ref['path']) as z:x.append(z['x'][:,0]);m.append(z['m'][:,0])
    return source,np.concatenate(x),np.concatenate(m)


def test_collapsed_protected_prefixes_preserve_exact_full_outputs_and_clear_qualification(asset):
    source,x,m=asset; out,options=s.collapse_protected(source)
    a=source.distributions(x,m);b=out.distributions(x,m)
    assert all(np.array_equal(i,j) for i,j in zip(a[1:],b[1:]))
    assert len(out.leaves)<=len(source.leaves) and len(options)==len(out.leaves)
    assert out.report is None and not out.feedback_leaves and not out.explanation_leaves
    s.semantic.verify(out)
    assert all(np.array_equal(i,j) for i,j in zip(b,forward(out,x,m)))


def test_oblique_fit_uses_no_more_than_three_public_sources_and_replays_float32(asset):
    source,x,m=asset; base,_=s.collapse_protected(source)
    route,z,q=base.distributions(x,m)
    candidate=next(i for i in range(len(base.leaves)) if len(s.region(base.structure,i,256)['path'])<=6
                   and any((route==i)&(q.max(1)>.85)&(m.sum(1)>1)))
    index=np.flatnonzero((route==candidate)&(q.max(1)>.85)&(m.sum(1)>1))[0]
    path=s.region(base.structure,candidate,256)['path']
    used={j for item in path for j,w in enumerate(item['weight']) if w}|set(base.leaves[candidate]['support'])
    d=domain_table()['features']
    features=[j for j,a in enumerate(d) if j not in used and not a['constant'] and not a['name'].startswith('legal_')
              and a['lower']==0 and a['upper']==1][:2]
    assert len(features)==2
    rng=np.random.default_rng(932) # synthetic fixture only, no scientific scene allocation
    xx=np.repeat(x[index:index+1],1200,axis=0);mm=np.repeat(m[index:index+1],1200,axis=0)
    xx[:,features]=rng.random((1200,2)).astype(np.float32)
    rr,_,qq=base.distributions(xx,mm);assert np.all(rr==candidate)
    target=qq.copy(); top=qq.argmax(1);alt=next(a for a in range(5) if mm[0,a] and a!=top[0])
    bad=xx[:,features].sum(1)<1.
    target[bad,top[bad]],target[bad,alt]=qq[bad,alt],qq[bad,top[bad]]
    a=dict(x=xx,m=mm,p=target); budget=Budget()
    result=s.propose(base,a,candidate,budget,oblique=True)
    assert result['classifier_outputs_discarded'] and not result['execution_actions_produced']
    assert any(r['origin']=='sparse_logistic' for r in result['records'])
    assert 1<=len(budget.calls)<=49 # at most seven support fits at seven internal nodes
    for tree in result['proposals']:
        t,info=s.structure_info(tree,64)
        assert info['maximum_source_features_per_node']<=3 and info['depth']<=3
    # The full sparse tree separates a diagonal synthetic region more cleanly
    # than a same-depth axis tree; this is not a real NN quality measurement.
    axis=s.propose(base,a,candidate,Budget(),oblique=False)
    def impurity(result):
        answer=[]
        for tree in result['proposals']:
            t,info=s.structure_info(tree,64);r=_route(t,xx,info['depth'])
            label=~bad;answer.append(sum(min(int(label[r==i].sum()),int((~label[r==i]).sum())) for i in range(t['leaf_count'])))
        return min(answer)
    assert impurity(result)<impurity(axis)


def test_failed_optimizer_is_propagated_and_bad_target_is_rejected(asset):
    source,x,m=asset;base,_=s.collapse_protected(source)
    class Reject(Budget):
        def solve(self,*args):raise RuntimeError('predebit denied')
    with pytest.raises(RuntimeError,match='predebit denied'):
        s._fit(np.ones((100,1)),np.tile([0.,1.],50),[0],Reject(),{})
    leaf=int(base.leaf_indices(x)[0]);p=np.ones((len(x),5),np.float32)
    with pytest.raises((ValueError,RuntimeError)):
        s.propose(base,dict(x=x,m=m,p=p),leaf,Budget())
