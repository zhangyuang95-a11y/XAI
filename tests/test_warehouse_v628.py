"""Synthetic contracts only. No environment reset or scientific fitting."""
from copy import deepcopy
import json
import numpy as np
import pytest
from backend.training.warehouse_v626_search import Node,encode,decode,replace
from backend.training.warehouse_v626_program import Program as Parent,masked_probabilities
from backend.training.warehouse_v626_certificate import certify as old_certify,structure_info as old_structure
from backend.training.warehouse_v628_certificate import structure_info,region,rebind
from backend.training.warehouse_v628_program import from_parent,Program,payload,protected_indices,complete_program
from backend.training.warehouse_v628_search import Budget,Context,optimize
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v626_domain import domain_table,domain_sha
from experiments.warehouse import v628_data as data,v628_protocol as v


def axis(name='self_row'):
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index(name)]=1
    return tuple(map(float,w))


def fixture(tmp_path,maximum=64):
    t=encode(Node(axis(),-.5))[0];legal=({4},{4});leaves=[]
    for i in range(2):
        leaf=dict(version='warehouse-role-program.v6.26-domain-complete',domain_sha256=domain_sha(),normalized=True,
            support=[],bias=[-3.,-3.,-3.,-3.,3.],weights=[[],[],[],[],[]],witness=4)
        leaf['certificate']=old_certify(leaf,region(t,i),legal[i],use_path=True)[0];leaves.append(leaf)
    parent=Parent('robot_1','a'*64,t,leaves,'C3',64,frozenset({0}),frozenset({0}))
    p=tmp_path/'parent.json';p.write_text(json.dumps(parent.to_dict()));ref=v.pointer(p)
    model=from_parent(parent,ref,maximum)
    d=domain_table()['features'];parts={}
    for s in ('fit','calibration','eligibility','validation'):
        x=np.tile(np.array([r['center'] for r in d],np.float32),(120,1))
        for j,n in enumerate(FEATURE_NAMES):
            if n.startswith('legal_'):x[:,j]=1
        x[:,FEATURE_NAMES.index('self_row')]=np.linspace(0,1,120,dtype=np.float32)
        m=np.ones((120,5),np.float32);p=masked_probabilities(np.tile(leaves[0]['bias'],(120,1)).astype(np.float32),m)
        parts[s]=dict(x=x,m=m,p=p,sampled=np.full(120,4),
            rows=[dict(episode=f'{s}-{i//10}',base_seed=f'{s}-{i//10}',task=i//10%3+1) for i in range(120)])
    return parent,model,parts


@pytest.mark.parametrize('capacity',[64,128,256])
def test_new_capacity_counts_every_leaf_and_does_not_change_old_reader(capacity):
    def balanced(depth):return None if not depth else Node(axis(),-.5,balanced(depth-1),balanced(depth-1))
    t=encode(balanced(int(np.log2(capacity))))[0]
    assert structure_info(t,capacity)[1]['leaf_count']==capacity
    with pytest.raises(ValueError):structure_info(encode(balanced(9))[0],capacity)
    with pytest.raises(ValueError):old_structure(t,256)
    chain=None
    for _ in range(10):chain=Node(axis(),-.5,None,chain)
    with pytest.raises(ValueError):structure_info(encode(chain)[0],capacity)


def test_all_scenario_partitions_and_capacities_share_one_immutable_pool():
    cases=data.cases();folds=data.folds();assert len(cases)==1650
    assert len([c for c in cases if c['sampling_mode']=='argmax'])==300
    byid={c['case_id']:c for c in cases}
    for f in folds:
        for kind in ('outer','inner'):
            for split,ids in f[kind].items():
                assert len({byid[i]['base'] for i in ids})==len(ids)
                assert len(ids)%3==0
                assert all(sum(byid[i]['task']==task for i in ids)==len(ids)//3 for task in (1,2,3))
                if split!='fit':assert all(byid[i]['sampling_mode']=='sampled' for i in ids)
        assert set(sum(f['inner'].values(),[]))==set(f['outer']['fit'])
    assert len(v.jobs())==18 and len({j['job_id'] for j in v.jobs()})==18


def test_protected_coefficients_boundaries_and_exact_forward_are_immutable(tmp_path):
    parent,p,parts=fixture(tmp_path);x,m=parts['fit']['x'],parts['fit']['m']
    assert not p.feedback_leaves and not p.explanation_leaves
    original=parent.distributions(x,m);actual=p.distributions(x,m)
    assert all(np.array_equal(a,b) for a,b in zip(original,actual))
    p.leaves[0]['bias'][4]=2.
    with pytest.raises(ValueError):p.validate()
    p=from_parent(parent,v.pointer(tmp_path/'parent.json'),64)
    p.structure=encode(Node(axis(),-.4))[0]
    with pytest.raises(ValueError):p.validate()


def test_proof_rebinding_replays_exact_arithmetic_and_rejects_changed_path(tmp_path):
    parent,p,parts=fixture(tmp_path)
    tree=replace(decode(p.structure),(1,),Node(axis(),-.7))
    t=encode(tree)[0]
    rebound=rebind(p.leaves[0],region(p.structure,0),region(t,0),{4},{4})
    assert payload(rebound)==payload(p.leaves[0])
    with pytest.raises(ValueError):rebind(p.leaves[0],region(p.structure,0),region(t,1),{4},{4})
    rebound['certificate']['checks'][0]['upper'][0]+=1
    from backend.training.warehouse_v628_certificate import verify
    with pytest.raises(ValueError):verify(rebound,region(t,0),{4},rebound['certificate'])


def test_final_refit_and_temperature_skip_protected_and_requalify(tmp_path,monkeypatch):
    import backend.training.warehouse_v628_program as kernel
    parent,p,parts=fixture(tmp_path);calls=[];before=deepcopy(p.leaves[0])
    def fit(x,m,targets,**kwargs):calls.append(('fit',kwargs['path']['leaf']));return deepcopy(kwargs['initial'])
    def cal(leaf,x,m,targets,**kwargs):calls.append(('cal',kwargs['path']['leaf']));return leaf
    monkeypatch.setattr(kernel,'fit_leaf',fit);monkeypatch.setattr(kernel,'calibrate',cal)
    class B:
        def check(self):pass
    # Deliberately insufficient NEW eligibility: inherited old leaf0 cannot qualify.
    for k in ('x','m','p','sampled'):parts['eligibility'][k]=parts['eligibility'][k][:1]
    parts['eligibility']['rows']=parts['eligibility']['rows'][:1]
    p,results=complete_program(p,parts,B())
    assert calls==[('fit',1),('cal',1)] and p.leaves[0]==before
    assert not p.feedback_leaves and not p.explanation_leaves


def test_common_reserve_and_new_constructor_does_not_relax_old_limits(tmp_path):
    from backend.training.warehouse_v625_checkpoint import BudgetExhausted,CheckpointError
    from backend.training.warehouse_v626_checkpoint import CheckpointSession as Old
    b=Budget(tmp_path/'b.json',dict(synthetic=True))
    assert b.session.budget==dict(maximum_calls=484096,wall_seconds=6000.)
    b.counts['leaf']=77952
    with pytest.raises(BudgetExhausted):b.consume('leaf')
    b.counts['proof']=386688
    with pytest.raises(BudgetExhausted):b.consume('proof')
    b.session.close()
    with pytest.raises(ValueError):Old(tmp_path/'old.json',{},max_calls=484096,wall_seconds=6000)
    with pytest.raises(CheckpointError):Budget(tmp_path/'b.json',dict(synthetic=True),resume=True)


def test_editable_regions_never_include_protected_ancestors(tmp_path):
    _,p,parts=fixture(tmp_path);b=Budget(tmp_path/'b.json',dict(synthetic=True));ctx=Context(parts,p,b)
    assert [path for _,path,_ in ctx.editable(decode(p.structure))]==[(1,)]
    b.session.close()


def test_search_resume_restores_queue_frontier_cache_model_and_accounting(tmp_path,monkeypatch):
    import backend.training.warehouse_v628_search as search
    _,p,parts=fixture(tmp_path)
    def proposals(self,program,path):
        if len(program.leaves)>2:return [],'fully_checked'
        t=replace(decode(program.structure),(1,),Node(axis(),-.7))
        return [(search.plain(encode(t)[0]),'synthetic_split')],'enumerated'
    monkeypatch.setattr(search.Context,'proposals',proposals)
    b1=Budget(tmp_path/'whole.json',dict(synthetic='resume'));c1=Context(parts,p,b1)
    expected,er=optimize(c1)
    b2=Budget(tmp_path/'pause.json',dict(synthetic='resume'));c2=Context(parts,p,b2)
    _,paused=optimize(c2,pause_after=2);assert paused['paused']
    resumed=Budget(tmp_path/'pause.json',dict(synthetic='resume'),resume=True)
    actual,ar=optimize(Context(parts,p,resumed),resume=True)
    assert actual.to_dict()==expected.to_dict() and ar==er and b1.counts==resumed.counts
    b1.session.close();resumed.session.close()
