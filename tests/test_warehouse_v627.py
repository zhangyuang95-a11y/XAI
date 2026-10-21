"""Synthetic-only v627 science/safety contracts, no historical trajectories."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from backend.training.warehouse_v627_program import (Semantic,semantic_dict,semantic_from_dict,compile_tree,
    Context,Budget,CertifiedSemanticProgram,inner_parts,qualification)
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v622_program import masked_probabilities,_eligible


def fixture(moving=False):
    n=720;d=domain_table()['features']
    x=np.tile(np.array([v['center'] for v in d],np.float32),(n,1))
    for i,name in enumerate(FEATURE_NAMES):
        if name.startswith('legal_'):x[:,i]=1
    x[:,FEATURE_NAMES.index('self_row')]=np.linspace(0,1,n,dtype=np.float32)
    m=np.ones((n,5),np.float32);z=np.full((n,5),-4.,np.float32)
    chosen=np.where(x[:,FEATURE_NAMES.index('self_row')]<.5,0,3) if moving else np.full(n,4)
    z[np.arange(n),chosen]=4.
    a=dict(x=x,m=m,p=masked_probabilities(z,m),sampled=chosen,
        rows=[dict(episode=f'e-{i//12}',base_seed=str(i//12),task=(i//12)%3+1) for i in range(n)])
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_row')]=1.
    return a,Semantic(tuple(map(float,w)),-.5,Semantic(action=0 if moving else 4),Semantic(action=3 if moving else 4))


def budget(path,maximum=32):
    return Budget(path,{'fixture':'v627-synthetic'},dict(leaf=4096,node=1024,proof=10000),120,maximum_leaves=maximum)


def full(depth):
    if depth==0:return Semantic(action=0)
    _,node=fixture()
    return Semantic(node.weights,node.bias,full(depth-1),full(depth-1))


def test_compiler_counts_every_guard_and_empty_fallback_and_enforces_capacities():
    from backend.training.warehouse_v626_certificate import structure_info
    t,mapping=compile_tree(full(4),32)
    _,info=structure_info(t,32)
    assert t['leaf_count']==32 and len(mapping)==32 and info['depth']==5
    with pytest.raises(ValueError):compile_tree(full(5),32)
    t,mapping=compile_tree(full(5),64)
    assert t['leaf_count']==64
    with pytest.raises(ValueError):compile_tree(full(6),64)


def test_semantic_roundtrip_and_unknown_action_are_rejected():
    _,tree=fixture(True)
    assert semantic_dict(semantic_from_dict(semantic_dict(tree)))==semantic_dict(tree)
    bad=semantic_dict(tree);bad['actions'][0]=5
    with pytest.raises(ValueError):semantic_from_dict(bad)


def test_inner_four_partitions_are_task_stratified_whole_scenes_and_disjoint():
    a,_=fixture();parts=inner_parts(a);sets=[]
    for s,part in parts.items():
        scenes={r['base_seed'] for r in part['rows']};sets.append(scenes)
        assert {r['task'] for r in part['rows']}=={1,2,3}
    assert sum(len(part['rows']) for part in parts.values())==len(a['rows'])
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))


def test_compiled_program_uses_real_eligibility_and_exact_certificate_replay(tmp_path):
    from scripts.audit_warehouse_v626 import independent_forward,basic,eligible,independent_dual
    from backend.training.warehouse_v626_certificate import region
    a,tree=fixture(True);b=budget(tmp_path/'fit.json');ctx=Context(a,'robot_1','a'*64,32,b)
    p,score,m=ctx.evaluate(tree)
    assert len(p.leaves)==4
    assert p.kernel.report is not None and score[0]>.8
    pred=independent_forward(p,ctx.parts['eligibility']['x'],ctx.parts['eligibility']['m'])
    q=qualification(p,ctx.parts['eligibility'],pred)
    for i in range(len(p.leaves)):
        real=basic(p,ctx.parts['eligibility'],pred,{i},'explanation')
        assert _eligible(q[str(i)]['explanation'],.95)==eligible(real,.95)
        independent_dual(p.leaves[i],region(p.structure,i),p.certificates()[i])
    encoded=p.to_dict();assert CertifiedSemanticProgram.from_dict(encoded,'a'*64).to_dict()==encoded
    encoded['semantic']['actions'][0]=4
    with pytest.raises(ValueError):CertifiedSemanticProgram.from_dict(encoded,'a'*64)
    b.session.close()


def test_zero_illegal_over_full_bound_domain_when_physical_legal_bit_changes(tmp_path):
    a,tree=fixture(True);b=budget(tmp_path/'guards.json');ctx=Context(a,'robot_1','a'*64,32,b)
    program,_,_=ctx.evaluate(tree)
    for action in range(4):
        x=a['x'].copy();m=a['m'].copy()
        from backend.training.warehouse_v622_program import MASK_INDICES
        x[:,MASK_INDICES[action]]=0;m[:,action]=0
        _,z,q=program.distributions(x,m)
        assert np.all(m[np.arange(len(x)),z.argmax(1)]==1)
    b.session.close()


def test_no_fake_qualification_for_high_agreement_with_too_few_scenes(tmp_path):
    a,_=fixture();b=budget(tmp_path/'counts.json');ctx=Context(a,'robot_1','a'*64,32,b)
    p,_,_=ctx.evaluate(Semantic(action=4))
    part=deepcopy(ctx.parts['eligibility'])
    for r in part['rows']:r['base_seed']='one';r['episode']='one'
    m=qualification(p,part,p.distributions(part['x'],part['m']))['0']['explanation']
    assert m['agreement']==1 and not _eligible(m,.95)
    b.session.close()


def test_surrogate_gradient_reservation_is_prepaid_and_never_refunded(tmp_path):
    b=budget(tmp_path/'gradient.json')
    def fail():
        assert b.gradient_steps==200 and b.session.events()[-1]['event']=='call_started'
        raise RuntimeError('injected')
    with pytest.raises(RuntimeError):b.solve('node',{'reserved_gradient_steps':200},fail)
    assert b.gradient_steps==200 and b.counts['node']==1
    b.gradient_steps=20000
    from backend.training.warehouse_v625_checkpoint import BudgetExhausted
    with pytest.raises(BudgetExhausted):b.consume('node',{'reserved_gradient_steps':1})
    b.session.close()


def test_whole_tree_gradient_is_surrogate_only_and_three_sources_with_real_hard_router(tmp_path):
    from backend.training.warehouse_v627_search import torch_tree_proposal,sparse_weights
    a,tree=fixture(True);b=budget(tmp_path/'torch.json');ctx=Context(a,'robot_1','a'*64,32,b)
    p,_,_=ctx.evaluate(tree)
    proposal,loss,steps=torch_tree_proposal(ctx,p,123,steps=3)
    assert steps==3 and np.isfinite(loss)
    compile_tree(proposal,32)
    original=semantic_dict(tree);assert original==semantic_dict(p.semantic)
    ids=[i for i,v in enumerate(domain_table()['features']) if not v['constant']]
    actual=sparse_weights(np.ones((5,len(ids))),ids)
    for row in actual:
        assert sum(2 if FEATURE_NAMES[ids[j]]=='public_hard_risk_context' else 1 for j in np.flatnonzero(row))<=3
    b.session.close()


def test_subtree_rebuild_can_reorganize_depth_saturated_regions(tmp_path):
    from backend.training.warehouse_v627_search import rebuild_proposals
    a,tree=fixture(True);b=budget(tmp_path/'rebuild.json');ctx=Context(a,'robot_1','a'*64,32,b)
    p,_,_=ctx.evaluate(tree);proposals=rebuild_proposals(ctx,p,0)
    assert proposals and proposals[0][1]=='rebuild:()'
    for tree,label in proposals:compile_tree(tree,32)
    b.session.close()


def test_safe_resume_restores_full_beam_queue_cache_best_and_call_counts(monkeypatch,tmp_path):
    import backend.training.warehouse_v627_search as s
    a,tree=fixture()
    monkeypatch.setattr(s,'seed_tree',lambda *args:tree)
    monkeypatch.setattr(s,'gradient_proposal',lambda ctx,p,*args:p.semantic)
    monkeypatch.setattr(s,'rebuild_proposals',lambda ctx,p,*args:[(p.semantic,'r0'),(p.semantic,'r1')])
    first=budget(tmp_path/'all.json');ctx=Context(a,'robot_1','a'*64,32,first)
    p,report=s.optimize(ctx,{},rounds=1)
    paused=budget(tmp_path/'pause.json');ctx2=Context(a,'robot_1','a'*64,32,paused)
    with pytest.raises(s.SearchPaused):s.optimize(ctx2,{},rounds=1,pause_after_evaluations=13)
    resumed=Budget(tmp_path/'pause.json',{'fixture':'v627-synthetic'},dict(leaf=4096,node=1024,proof=10000),120,maximum_leaves=32,resume=True)
    assert len(resumed.session.state['machine']['queue'])>1 and resumed.session.state['machine']['cursor']==1
    ctx3=Context(a,'robot_1','a'*64,32,resumed)
    actual,actual_report=s.optimize(ctx3,{},rounds=1,resume=True)
    assert actual.to_dict()==p.to_dict() and actual_report==report and resumed.counts==first.counts
    assert actual_report['mandatory_completed']==[0,1,2,3]
    first.session.close();resumed.session.close()


@pytest.mark.parametrize('stage',['S32','S64'])
def test_entire_worker_outputs_bound_model_and_zero_actor_training(monkeypatch,tmp_path,stage):
    from scripts import run_warehouse_v627 as runner
    from backend.training.warehouse_v626_search import encode
    from backend.training.warehouse_v627_search import optimize
    a,_=fixture();parts={}
    for split in ('fit','calibration','eligibility','validation'):
        value=deepcopy(a)
        for r in value['rows']:r['base_seed']=split+'-'+r['base_seed'];r['episode']=split+'-'+r['episode']
        parts[split]=value
    spec=dict(job_id='fold-0-robot_1',fold=0,role='robot_1',stage=stage,arm='C3')
    import json
    from pathlib import Path
    from scripts.run_warehouse_v626 import load_parts
    parent=tmp_path/'parent.json';npz=tmp_path/'arrays.npz';metadata=tmp_path/'metadata.json'
    np.savez_compressed(npz,**{s+'_'+key:a[key] for s,a in parts.items() for key in ('x','m','p','sampled')})
    metadata.write_text(json.dumps({s:a['rows'] for s,a in parts.items()}))
    parent.write_text(json.dumps({'datasets':{spec['job_id']:dict(arrays=runner.v.pointer(npz),metadata=runner.v.pointer(metadata))}}))
    path=tmp_path/'plan.json';actor=dict(path='fake.npz',sha256='f'*64)
    plan=dict(jobs=[spec],maximum_leaves=32 if stage=='S32' else 64,actor=actor,initial_S1={spec['job_id']:dict(structure={})},
              parent_plan=runner.v.pointer(parent),source_sha256='b'*64)
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner,'checked_plan',lambda *a:(path,plan))
    monkeypatch.setattr(runner,'checked_parent',lambda:(None,{}))
    monkeypatch.setattr(runner,'load_parts',lambda *a:parts)
    original_bound=runner.v.bound
    monkeypatch.setattr(runner.v,'bound',lambda *a:encode(None)[0])
    original=runner.v.pointer
    monkeypatch.setattr(runner.v,'pointer',lambda p:actor if str(p)=='fake.npz' else original(p))
    monkeypatch.setattr(runner,'optimize',lambda ctx,t:optimize(ctx,t,rounds=0))
    r=runner.job(stage,spec['job_id'])
    assert r['status']=='completed' and r['gate']['passed']
    assert r['actor_updates']==r['PPO_joint_steps']==r['environment_resets']==0
    assert r['solver']['surrogate_gradient_steps_reserved']==800
    monkeypatch.setattr(runner.v,'bound',original_bound)
    from scripts.audit_warehouse_v627 import audit
    assert audit(path,[r])['integrity_passed']
