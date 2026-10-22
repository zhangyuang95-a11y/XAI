"""Critical numerical proof, data isolation and immutable-core checks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from backend.training import warehouse_v633_support_refresh as s
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v626_certificate import verify
from scripts.audit_warehouse_v626 import independent_dual
from experiments.warehouse import v630_protocol as v


class Budget:
    def __init__(self):self.counts=dict(leaf=0,proof=0,node=0)
    def check(self):pass
    def solve(self,kind,metadata,fn):self.counts[kind]+=1;return fn()


@pytest.fixture(scope='module')
def fixture():
    run=v.ROOT/'output/warehouse_v632/warehouse-v632-oblique-confidence-01/outer-evaluation-01'
    r=v.bound(v.pointer(run/'oblique-fold-0-robot_1/result.json'));_,plan=v.checked_plan()
    p=Program.from_dict(v.bound(r['program']),plan['actor_role_sha256']['robot_1'])
    receipt=v.bound(plan['collection']);blocks=[]
    for k in range(300,315):
        ref=receipt['cases'][f'dev-{k:04}']['arrays'];v.require(v.pointer(ref['path'])==ref,'fixture source changed')
        with np.load(ref['path']) as z:blocks.append(tuple(z[key][:,0].copy() for key in ('x','m','p')))
    x,m,target=(np.concatenate([b[i] for b in blocks]) for i in range(3))
    route=p.leaf_indices(x);protected=s.protected_indices(p)
    choices=[i for i in range(len(p.leaves)) if i not in protected and int((route==i).sum())>=30]
    leaf=max(choices,key=lambda i:int((route==i).sum()));take=route==leaf
    return p,leaf,x[take],m[take],target[take]


def test_real_full_distribution_support_trials_are_certified_budgeted_and_nonmutating(fixture):
    p,i,x,m,target=fixture;old=deepcopy(p.leaves[i]);budget=Budget()
    take=np.arange(len(x))%5!=0;path=s.region(p.structure,i,256);legal=p.certificates()[i]
    leaf,record=s.select_leaf(old,(x[take],m[take],target[take]),(x[~take],m[~take],target[~take]),path,legal,budget)
    assert p.leaves[i]==old and 1<=len(record['trials'])<=8
    assert len({(tuple(r['support']),r['witness']) for r in record['trials']})==len(record['trials'])
    assert budget.counts['node']==0 and 1<=budget.counts['leaf']<=64 and budget.counts['proof']<=256
    assert s._support_cost(leaf['support'])<=3 and leaf['witness'] in legal
    verify(leaf,path,legal,leaf['certificate']);independent_dual(leaf,path,legal)
    assert tuple(record['selected_key'])>=tuple(record['incumbent_key'])


def test_refresh_never_uses_other_partitions_for_support_selection_and_keeps_core(fixture,monkeypatch):
    p,i,x,m,target=fixture;parts={};fit_target=target[0];captured=[]
    for split in ('fit','calibration','eligibility','validation'):
        xx=np.repeat(x[:1],6,axis=0);mm=np.repeat(m[:1],6,axis=0)
        pp=np.repeat(target[:1],6,axis=0)
        if split!='fit':pp=mm/mm.sum(1,keepdims=True)
        rows=[dict(episode=f'{split}-{j}',base_seed=f'{split}-{j}',task=j//2+1) for j in range(6)]
        parts[split]=dict(x=xx,m=mm,p=pp,sampled=pp.argmax(1),rows=rows)
    def spy(incumbent,train,selection,path,legal,budget):
        assert np.all(train[2]==fit_target) and np.all(selection[2]==fit_target)
        assert len(train[0])+len(selection[0])==6
        captured.append(path['leaf'])
        return deepcopy(incumbent),dict(selected_original=True)
    monkeypatch.setattr(s,'select_leaf',spy);budget=Budget();before=p.to_dict()
    out,report=s.refresh(p,parts,budget)
    assert captured==[i] and p.to_dict()==before and budget.counts==dict(leaf=0,proof=0,node=0)
    assert not set(report['train_scenes'])&set(report['selection_scenes'])
    assert s.digest(out.structure)==s.digest(p.structure)
    assert out.report is None and not out.feedback_leaves and not out.explanation_leaves
    for leaf in report['protected_descendants']:assert s.payload(out.leaves[leaf])==s.payload(p.leaves[leaf])
    assert all(s.payload(a)==s.payload(b) for a,b in zip(out.leaves,p.leaves,strict=True))


def test_optimization_denial_is_propagated_not_counted_as_a_fit_success(fixture):
    p,i,x,m,target=fixture
    class Denied(Budget):
        def solve(self,*args):raise RuntimeError('call budget denied')
    with pytest.raises(RuntimeError,match='call budget denied'):
        s.select_leaf(p.leaves[i],(x,m,target),(x,m,target),s.region(p.structure,i,256),p.certificates()[i],Denied())
