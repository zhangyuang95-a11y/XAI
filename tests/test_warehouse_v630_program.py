"""Real fixed surrogate assets, no fit/LP/new environment or NN training."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.warehouse import v629_protocol as v
from backend.training import warehouse_v630_program as p
from backend.training import warehouse_v630_certificate as c
from backend.training import warehouse_v626_certificate as base_cert
from backend.training.warehouse_v629_program import Program as OldProgram,protected_indices
from backend.training.warehouse_v629_guards import predicate
from backend.training.warehouse_v626_domain import domain_table
from scripts.audit_warehouse_v626 import independent_dual


@pytest.fixture(scope='module')
def refined():
    _,plan=v.checked_plan();result=v.bound(v.pointer(v.OUTPUT/'jobs/G128-fold-0-robot_1/result.json'))
    assert result['status']=='completed'
    before=OldProgram.from_dict(v.bound(result['program']),plan['actor_role_sha256']['robot_1'])
    receipt=v.bound(plan['collection']);xs=[];ms=[]
    for i in range(300,312):
        ref=receipt['cases'][f'dev-{i:04}']['arrays'];assert v.pointer(ref['path'])==ref
        with np.load(ref['path'],allow_pickle=False) as z:xs.append(z['x'][:,0]);ms.append(z['m'][:,0])
    x=np.concatenate(xs);m=np.concatenate(ms);route=before.leaf_indices(x);counts=np.bincount(route,minlength=len(before.leaves))
    choices=[i for i in protected_indices(before) if len(p.region(before.structure,i,before.maximum_leaves)['path'])<9]
    leaf=max(choices,key=lambda i:(counts[i],-i));assert counts[leaf]>0
    path=p.region(before.structure,leaf,before.maximum_leaves)
    used={i for item in path['path'] for i,w in enumerate(item['weight']) if w}
    domain=domain_table()['features']
    feature=next(i for i,row in enumerate(domain) if i not in used and not row['constant']
        and row['lower']<row['center']<row['upper'] and 'legal_' not in row['name'])
    threshold=np.float32(domain[feature]['center']);w=np.zeros(x.shape[1],np.float32);w[feature]=1.
    with pytest.MonkeyPatch.context() as mp:
        def forbidden(*args,**kwargs):raise AssertionError('exact proof extension cannot solve an LP')
        mp.setattr(base_cert,'linprog',forbidden)
        after,derivation=p.split_identical(before,leaf,w,-float(threshold))
        after.validate()
    return before,after,derivation,x,m,leaf,feature,threshold


def test_complete_real_surrogate_distribution_is_bitwise_preserved(refined):
    before,after,d,x,m,*_=refined
    _,z,q=before.distributions(x,m);_,zz,qq=after.distributions(x,m)
    assert np.array_equal(z,zz) and np.array_equal(q,qq)
    assert d['every_old_leaf_output_preserved'] and d['solver_calls']==d['NN_updates']==d['environment_resets']==0
    assert not after.feedback_leaves and not after.explanation_leaves and after.report is None
    assert not d['model_acceptance_passed'] and not d['pilot_opened']


def test_new_split_tie_and_both_sides_preserve_old_prefix_and_leaf_arithmetic(refined):
    before,after,d,x,m,leaf,feature,threshold=refined
    row=x[np.flatnonzero(before.leaf_indices(x)==leaf)[0]]
    boundary=np.stack([row.copy() for _ in range(3)])
    boundary[:,feature]=[np.nextafter(threshold,np.float32(-np.inf)),threshold,np.nextafter(threshold,np.float32(np.inf))]
    masks=np.stack([m[np.flatnonzero(before.leaf_indices(x)==leaf)[0]]]*3)
    assert np.all(before.leaf_indices(boundary)==leaf)
    child=after.leaf_indices(boundary);assert child[0]==child[1] and child[1]!=child[2]
    for old,new in zip(before.distributions(boundary,masks)[1:],after.distributions(boundary,masks)[1:],strict=True):
        assert np.array_equal(old,new)


def test_each_new_child_has_independent_exact_rational_numeric_proof(refined):
    before,after,d,*_=refined
    changed=[entry for entry in d['derivations'] if len(entry['proof']['new_path']['path'])>len(entry['proof']['old_path']['path'])]
    assert len(changed)==2
    for entry in changed:
        i=entry['child_leaf'];independent_dual(after.leaves[i],p.region(after.structure,i,after.maximum_leaves),after.certificates()[i])
        assert entry['proof']['exact_upper_bounds_unchanged'] and entry['proof']['solver_calls']==0


def test_stored_new_asset_roundtrip_clears_old_qualification_and_is_rejected_by_old_reader(refined):
    _,after,*_=refined;encoded=after.to_dict();restored=p.Program.from_dict(encoded,after.actor_role_sha256)
    assert restored.to_dict()==encoded and not restored.feedback_leaves and not restored.explanation_leaves
    assert encoded['external_program_selects_execution_action'] is False
    with pytest.raises(ValueError,match='unknown v629'):OldProgram.from_dict(encoded,after.actor_role_sha256)


@pytest.mark.parametrize('change',['coefficient','numeric_proof','semantic_proof','qualification','role','version'])
def test_tampered_refined_assets_are_refused(refined,change):
    _,after,*_=refined;encoded=deepcopy(after.to_dict())
    if change=='coefficient':encoded['leaves'][0]['bias'][0]=float(np.float32(encoded['leaves'][0]['bias'][0]+.001))
    if change=='numeric_proof':encoded['leaves'][0]['certificate']['binding_sha256']='0'*64
    if change=='semantic_proof':encoded['semantic_protection']['entries'][0]['descendant_leaves']=[]
    if change=='qualification':encoded['explanation_leaves']=[0]
    if change=='role':encoded['role']='robot_2'
    if change=='version':encoded['version']='warehouse-role-program.fake'
    with pytest.raises((RuntimeError,ValueError)):p.Program.from_dict(encoded,after.actor_role_sha256)


def test_depth_limit_precedes_any_proof_extension(refined,monkeypatch):
    before,*_=refined;leaf=next(i for i in range(len(before.leaves)) if len(p.region(before.structure,i,before.maximum_leaves)['path'])==9)
    monkeypatch.setattr(p,'extend',lambda *args:pytest.fail('over-depth condition cannot enter proof work'))
    weights=np.zeros(len(before.structure['split_weights'][0]),np.float32);weights[47]=1.
    with pytest.raises(ValueError,match='limit exceeded'):p.split_identical(before,leaf,weights,-.5)


def test_original_prefix_and_old_certificate_are_mandatory(refined):
    before,*_=refined;leaf=next(i for i,l in enumerate(before.leaves) if l['certificate']['checks'])
    path=p.region(before.structure,leaf,before.maximum_leaves);legal=before.certificates()[leaf];new=deepcopy(path)
    new['path'][0]['bias']+=.01
    with pytest.raises(RuntimeError,match='original route prefix'):c.extend(before.leaves[leaf],path,new,legal,legal)
    bad=deepcopy(before.leaves[leaf]);bad['certificate']['checks'][0]['upper'][0]+=1
    with pytest.raises(ValueError,match='exact rational'):c.extend(bad,path,path,legal,legal)


def test_new_explicit_legal_fact_can_remove_only_its_own_redundant_numeric_check(refined):
    before,*_=refined;leaf=next(i for i,a in enumerate(before.certificates()) if len(a)<5)
    path=p.region(before.structure,leaf,before.maximum_leaves);legal=before.certificates()[leaf];action=next(a for a in range(4) if a not in legal)
    w,b=predicate((action,));new=deepcopy(path);new['path'].append(dict(weight=list(w),bias=b,branch='right'))
    new['tree_sha256']='a'*64;new['leaf']=0;new_legal=c.path_actions(new)
    output,proof=c.extend(before.leaves[leaf],path,new,legal,new_legal)
    assert set(new_legal)==set(legal)|{action}
    assert [q['action'] for q in output['certificate']['checks']]==sorted(set(range(5))-set(new_legal))
    independent_dual(output,new,new_legal);assert proof['solver_calls']==0
