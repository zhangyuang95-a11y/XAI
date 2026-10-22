"""Declared capacities, historic rejection, exact proof and matched budgets."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from backend.training import warehouse_v642_certificate as certificate,warehouse_v642_program as language
from backend.training import warehouse_v642_guards as guards,warehouse_v642_contexts as contexts
from backend.training.warehouse_v628_certificate import structure_info as historical_structure
from backend.training.warehouse_v626_search import Node,encode
from backend.training.warehouse_v622_program import plain,MASK_INDICES
from backend.training.warehouse_v626_domain import domain_table,domain_sha
from scripts.audit_warehouse_v642_program import forward


def chain(depth):
    w=np.zeros(173,np.float32);w[MASK_INDICES[0]]=1
    tree=None
    for _ in range(depth):tree=Node(tuple(w),-.5,None,tree)
    return plain(encode(tree)[0])


def test_depth12_boundary_and_historical_reader_remains_depth9():
    s=chain(12);_,info=certificate.structure_info(s,256,12)
    assert info['depth']==12 and info['leaf_count']==13
    for call in (lambda:certificate.structure_info(s,256,9),lambda:historical_structure(s,256),
                 lambda:certificate.structure_info(chain(13),256,12),lambda:certificate.structure_info(s,256,True)):
        with pytest.raises(ValueError):call()
    tree=None;w=np.zeros(173,np.float32);w[MASK_INDICES[0]]=1
    for _ in range(8):tree=Node(tuple(w),-.5,deepcopy(tree),deepcopy(tree))
    assert certificate.structure_info(plain(encode(tree)[0]),256,12)[1]['leaf_count']==256
    tree=Node(tuple(w),-.5,tree,None)
    with pytest.raises(ValueError):certificate.structure_info(plain(encode(tree)[0]),256,12)


def test_depth12_independent_forward_and_per_leaf_proofs():
    s=chain(12);legal=guards.leaf_actions(s,256,12);leaves=[]
    for i,known in enumerate(legal):
        leaf=dict(version='warehouse-role-program.v6.26-domain-complete',support=[],bias=[-5.,-5.,-5.,-5.,5.],
                  weights=[[],[],[],[],[]],witness=4,normalized=True,domain_sha256=domain_sha())
        p=certificate.region(s,i,256,12);leaf['certificate'],counter=certificate.certify(leaf,p,known,use_path=True)
        assert not counter and certificate.verify(leaf,p,known,leaf['certificate']);leaves.append(leaf)
    d=domain_table()['features'];x=np.array([[f['center'] for f in d]]*3,np.float32);x[:,MASK_INDICES]=1;x[0,MASK_INDICES[0]]=0
    program=language.Program('robot_1','a'*64,s,leaves,'C3',256,maximum_depth=12)
    m=x[:,MASK_INDICES];direct=program.distributions(x,m);independent=forward(program,x,m)
    assert all(np.array_equal(a,b) for a,b in zip(direct,independent,strict=True))
    assert direct[0].tolist()==[0,12,12]
    tampered=deepcopy(leaves[-1]['certificate']);tampered['binding_sha256']='0'*64
    with pytest.raises(ValueError):certificate.verify(leaves[-1],certificate.region(s,12,256,12),legal[-1],tampered)


def test_real_asset_import_roundtrip_depth_binding_and_frozen_outputs():
    from experiments.warehouse import v630_protocol as v
    from backend.training import warehouse_v635_program as previous
    _,old=v.checked_plan();r=v.bound(v.pointer(v.ROOT/'output/warehouse_v640/warehouse-v640-alternating-regions-01/fold-0-robot_2/result.json'))
    source=previous.Program.from_dict(v.bound(r['program']),old['actor_role_sha256']['robot_2']);before=v.digest(source.to_dict())
    d=domain_table()['features'];x=np.array([[f['center'] for f in d]]*8,np.float32);x[:,MASK_INDICES]=1
    original=source.distributions(x,x[:,MASK_INDICES])
    for depth in (9,12):
        out=language.import_previous(source,depth);serialized=out.to_dict();copy=language.Program.from_dict(serialized,out.target_actor_role_sha256)
        assert copy.maximum_depth==depth and len(copy.leaves)==len(source.leaves)
        assert all(np.array_equal(a,b) for a,b in zip(original,copy.distributions(x,x[:,MASK_INDICES]),strict=True))
        bad=deepcopy(serialized);bad['maximum_depth']=12 if depth==9 else 9
        with pytest.raises((ValueError,RuntimeError)):language.Program.from_dict(bad,out.target_actor_role_sha256)
        with pytest.raises((ValueError,RuntimeError)):previous.Program.from_dict(serialized,out.target_actor_role_sha256)
    assert v.digest(source.to_dict())==before


def test_real_depth10_refit_assembly_and_independent_router():
    from experiments.warehouse import v630_protocol as v,v628_data as data
    from backend.training import warehouse_v635_program as previous,warehouse_v642_subtrees as build
    from backend.training.warehouse_v626_search import decode
    class Budget:
        def check(self):pass
        def solve(self,kind,metadata,callback):return callback()
    _,old=v.checked_plan();prior=v.bound(old['previous_plan']);dp=v.bound(prior['previous_plan'])
    r=v.bound(v.pointer(v.ROOT/'output/warehouse_v640/warehouse-v640-alternating-regions-01/fold-0-robot_2/result.json'))
    source=language.import_previous(previous.Program.from_dict(v.bound(r['program']),old['actor_role_sha256']['robot_2']),12)
    core=set(language.protection(source)['protected_descendants']);_,paths=encode(decode(source.structure))
    leaf=next(i for i,p in enumerate(paths) if len(p)==9 and i not in core);root=paths[leaf]
    w,b=guards.old.predicate([0]);tree=Node(w,b,None,None);parts=data.load_parts(dp,0,'robot_2')
    option=build.fit_option(source,root,tree,parts,Budget());candidate=build.assemble(source,{root:option})
    assert language.structure_info(candidate.structure,256,12)[1]['depth']==10
    assert len(candidate.leaves)==len(source.leaves)+1
    a=parts['validation'];before=source.distributions(a['x'],a['m']);direct=candidate.distributions(a['x'],a['m']);independent=forward(candidate,a['x'],a['m'])
    assert all(np.array_equal(a,b) for a,b in zip(direct,independent,strict=True))
    frozen=before[0]!=leaf
    assert all(np.array_equal(a[frozen],b[frozen]) for a,b in zip(before[1:],direct[1:],strict=True))
