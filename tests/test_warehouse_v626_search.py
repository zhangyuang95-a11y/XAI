"""Synthetic search/resume contracts; never reads scientific trajectories."""
from copy import deepcopy
import numpy as np
import pytest
from backend.training.warehouse_v626_search import Context,SearchBudget,SearchPaused,optimize,Node,decode,encode,regions,node_candidates
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v626_domain import domain_table
from backend.training.warehouse_v622_program import masked_probabilities


def fixture():
    d=domain_table()['features'];x=np.tile(np.array([v['center'] for v in d],np.float32),(180,1))
    for i,n in enumerate(FEATURE_NAMES):
        if n.startswith('legal_'):x[:,i]=1
    x[:,FEATURE_NAMES.index('self_row')]=np.linspace(0,1,len(x),dtype=np.float32)
    m=np.ones((len(x),5),np.float32);z=np.tile(np.array([-3.,-3.,-3.,-3.,3.],np.float32),(len(x),1))
    a=dict(x=x,m=m,p=masked_probabilities(z,m),sampled=np.full(len(x),4),
        rows=[dict(episode=f'fit-{i//3}',base_seed=str(i//3),task=(i//3)%3+1) for i in range(len(x))])
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_row')]=1
    t,_=encode(Node(tuple(map(float,w)),-.5))
    return {'fit':a},t


def budget(path):return SearchBudget(path,{'fixture':'synthetic_search'},dict(leaf=4096,node=1024,proof=10000),120,maximum_leaves=32)


def test_tree_codec_and_guard_nodes_are_immutable():
    parts,t=fixture();assert np.array_equal(encode(decode(t))[0]['split_weights'],t['split_weights'])
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('legal_up')]=1
    node=Node(tuple(map(float,w)),-.5)
    assert sum(item[0] is None for item in regions(node,parts['fit']['x']))==2
    class Clock:
        def solve(self,*a,**kw):pytest.fail('legal guard must never be optimized')
    class C:pass
    c=C();c.a=parts['fit'];c.budget=Clock()
    assert node_candidates(c,None,node,())==[]


def test_safe_resume_restores_entire_remaining_queue_cache_and_numerics(monkeypatch,tmp_path):
    import backend.training.warehouse_v626_search as s
    parts,t=fixture();spec=dict(role='robot_1',actor_role_sha256='a'*64)
    def proposals(ctx,current,tree,path):
        node=s.at(tree,path)
        return [Node(node.weight,-.4,node.left,node.right),Node(node.weight,-.6,node.left,node.right)]
    monkeypatch.setattr(s,'node_candidates',proposals)
    first=budget(tmp_path/'uninterrupted.json');c=Context(parts,spec,first,32)
    baseline,report=optimize(c,t,rounds=2)
    second=budget(tmp_path/'paused.json');paused=Context(parts,spec,second,32)
    with pytest.raises(SearchPaused):optimize(paused,t,rounds=2,pause_after_evaluations=2)
    restored=SearchBudget(tmp_path/'paused.json',{'fixture':'synthetic_search'},dict(leaf=4096,node=1024,proof=10000),120,
        maximum_leaves=32,resume=True)
    saved=restored.session.state
    assert len(saved['machine']['queue'])==2 and saved['machine']['cursor']==1
    resumed=Context(parts,spec,restored,32);actual,actual_report=optimize(resumed,t,rounds=2,resume=True)
    assert actual.to_dict()==baseline.to_dict()
    assert actual_report==report and restored.counts==first.counts
    first.session.close();restored.session.close()


def test_search_budget_retains_final_leaf_and_lp_reserves(tmp_path):
    from backend.training.warehouse_v625_checkpoint import BudgetExhausted
    b=budget(tmp_path/'reserve.json')
    b.counts['leaf']=4096-256
    with pytest.raises(BudgetExhausted):b.consume('leaf')
    b.counts['proof']=10000-52*32
    with pytest.raises(BudgetExhausted):b.consume('proof')
    b.session.close()


def test_new_constructor_caps_do_not_reinterpret_legacy_admission(tmp_path):
    from backend.training.warehouse_v626_checkpoint import CheckpointSession as New
    from backend.training.warehouse_v625_checkpoint import CheckpointSession as Old
    with pytest.raises(ValueError):Old(tmp_path/'old.json',{},max_calls=121024,wall_seconds=5400)
    with pytest.raises(ValueError):New(tmp_path/'too-many.json',{},max_calls=121025,wall_seconds=5400)
    with pytest.raises(ValueError):New(tmp_path/'too-long.json',{},max_calls=121024,wall_seconds=5401)
    c=New(tmp_path/'new.json',{},max_calls=121024,wall_seconds=5400)
    assert c.budget==dict(maximum_calls=121024,wall_seconds=5400.)
    c.close()


def test_64_leaf_growth_adds_real_legal_guard_and_improves_fit_proxy(tmp_path):
    from backend.training.warehouse_v626_search import growth_candidates
    parts,t=fixture();x=parts['fit']['x'];z=np.full((len(x),5),-3.,np.float32)
    z[np.arange(len(x)),np.where(x[:,FEATURE_NAMES.index('self_row')]<.5,0,3)]=3.
    parts['fit']['p']=masked_probabilities(z,parts['fit']['m'])
    b=SearchBudget(tmp_path/'growth.json',{'fixture':'toy_growth'},dict(leaf=4096,node=1024,proof=10000),120,maximum_leaves=64)
    ctx=Context(parts,dict(role='robot_1',actor_role_sha256='a'*64),b,64)
    current,score=ctx.evaluate(decode(t));queue=growth_candidates(ctx,current,decode(t))
    assert queue and queue[0][1].startswith('grow-legal-')
    candidate,new=ctx.evaluate(queue[0][0])
    assert len(candidate.leaves)==len(current.leaves)+1 and new[0]>score[0]
    assert max(len(p) for p in encode(queue[0][0])[1])<=7
    b.session.close()


@pytest.mark.parametrize('stage',['TAO32','TAO64'])
def test_entire_search_worker_emits_valid_metrics_and_search_artifact(monkeypatch,tmp_path,stage):
    from scripts import run_warehouse_v626_search as runner
    from backend.training.warehouse_v626_program import Program,VERSION
    from backend.training.warehouse_v626_certificate import certify,region
    from backend.training.warehouse_v626_domain import domain_sha
    parts0,_=fixture();parts={}
    for s in ('fit','calibration','eligibility','validation'):
        a=deepcopy(parts0['fit'])
        for r in a['rows']:r['base_seed']=s+'-'+r['base_seed'];r['episode']=s+'-'+r['episode']
        parts[s]=a
    w=np.zeros((0,len(FEATURE_NAMES)),np.float32)
    t=dict(root_ref=-1,split_weights=w,split_bias=[],left_refs=[],right_refs=[],leaf_count=1,feature_dim=len(FEATURE_NAMES))
    leaf=dict(version=VERSION,normalized=True,domain_sha256=domain_sha(),support=[],bias=[-3.,-3.,-3.,-3.,3.],weights=[[] for _ in range(5)],witness=4)
    leaf['certificate']=certify(leaf,region(t,0),{4},use_path=True)[0]
    spec=dict(job_id='fold-0-robot_1',fold=0,role='robot_1',arm='C3',stage=stage)
    role=runner.v.old.ROLE_SHA256['robot_1'];initial=Program('robot_1',role,t,[leaf],'C3').to_dict() if stage=='TAO64' else t
    path=tmp_path/'plan.json';path.write_text('{}')
    plan=dict(jobs=[spec],initial={spec['job_id']:{}},maximum_leaves=64 if stage=='TAO64' else 32,actor={'path':'fake-actor.npz'})
    monkeypatch.setattr(runner,'checked_stage',lambda *a:(path,plan))
    monkeypatch.setattr(runner,'checked_A0',lambda:(None,{}))
    monkeypatch.setattr(runner,'load_parts',lambda *a:parts)
    monkeypatch.setattr(runner.v,'bound',lambda *a:initial)
    original=runner.v.file_sha
    monkeypatch.setattr(runner.v,'file_sha',lambda p:runner.v.old.ACTOR_SHA256 if str(p)=='fake-actor.npz' else original(p))
    result=runner.job(stage,spec['job_id'])
    assert result['status']=='completed' and result['gate']['passed']
    assert (tmp_path/'jobs'/spec['job_id']/'search-best.json').is_file()
    assert result['solver']['counts']['leaf']>0 and result['solver']['counts']['node']==0
