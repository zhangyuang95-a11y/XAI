"""Synthetic contracts; these tests do not use scientific fit episodes."""
from copy import deepcopy
import json
import numpy as np
import pytest
from backend.training.warehouse_v626_domain import domain_table,domain_sha,transform,validate_rows
from backend.training.warehouse_v626_certificate import certify,verify,region,structure_info,dual_upper,F
from backend.training.warehouse_v626_program import fit_leaf,evaluate_leaf,kl,VERSION
from backend.training.warehouse_v626_budget import Budget
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v625_program import public_bounds
from backend.training.warehouse_v625_checkpoint import CheckpointError,BudgetExhausted


def rows(n):
    d=domain_table()['features'];x=np.tile(np.array([v['center'] for v in d],np.float32),(n,1))
    for i,name in enumerate(FEATURE_NAMES):
        if name.startswith('legal_'):x[:,i]=1.
    return x


def tree():
    w=np.zeros((1,len(FEATURE_NAMES)),np.float32);w[0,FEATURE_NAMES.index('self_row')]=1
    return dict(root_ref=0,split_weights=w,split_bias=[-.25],left_refs=[-1],right_refs=[-2],leaf_count=2,feature_dim=len(FEATURE_NAMES))


def model():
    return dict(version=VERSION,domain_sha256=domain_sha(),normalized=True,
        support=[FEATURE_NAMES.index('self_row')],bias=[0.,-3.,-3.,-3.,.25],
        weights=[[1.],[0.],[0.],[0.],[0.]],witness=4)


def test_complete_public_schema_and_mission_information_is_not_silently_filtered():
    table=domain_table();assert len(table['features'])==len(FEATURE_NAMES)==173
    old=public_bounds(FEATURE_NAMES)
    for name in ('self_carrying_task_0','self_charger_shortest_distance','progress_to_order_0_delivery_if_down'):
        i=FEATURE_NAMES.index(name);assert i not in old
        assert not table['features'][i]['constant']
    assert table['map_diameter']==9


def test_geodesic_small_features_are_scaled_using_physics_not_observed_extrema():
    x=rows(3);i=FEATURE_NAMES.index('progress_to_order_0_delivery_if_down')
    x[:,i]=np.array([-1/42,0.,1/42],np.float32)
    assert np.allclose(transform(x,[i])[:,0],[-1.,0.,1.],atol=1e-6)
    assert domain_table()['features'][i]['reason']=='geodesic_one_edge_triangle_inequality'


def test_out_of_domain_constants_and_extreme_values_are_rejected():
    x=rows(2);validate_rows(x)
    x[0,FEATURE_NAMES.index('charger_row')]+=.1
    with pytest.raises(ValueError,match='charger_row'):validate_rows(x)
    x=rows(1);x[0,FEATURE_NAMES.index('partner_battery_fraction')]=1.01
    with pytest.raises(ValueError,match='partner_battery_fraction'):validate_rows(x)


def test_path_certificate_is_stronger_than_global_box_without_changing_leaf():
    leaf=model();path=region(tree(),0)
    assert not certify(leaf,path,{4},use_path=False)[0]['passed']
    proof,_=certify(leaf,path,{4},use_path=True)
    assert proof['passed'];assert verify(leaf,path,{4},proof)
    assert any(check['kind']=='path' for check in proof['checks'])
    with pytest.raises(ValueError):verify(leaf,region(tree(),1),{4},proof)


@pytest.mark.parametrize('what',['dual','coefficient','domain','upper','missing'])
def test_corrupt_certificates_fail_without_calling_any_lp(what,monkeypatch):
    leaf=model();path=region(tree(),0);proof,_=certify(leaf,path,{4},use_path=True)
    leaf,proof=deepcopy(leaf),deepcopy(proof)
    if what=='dual':next(c for c in proof['checks'] if c['kind']=='path')['dual'][0]=-1
    elif what=='coefficient':leaf['bias'][0]=1.
    elif what=='domain':leaf['domain_sha256']='a'*64
    elif what=='upper':proof['checks'][0]['upper']=[-999,1]
    else:proof['checks'].pop()
    import backend.training.warehouse_v626_certificate as c
    monkeypatch.setattr(c,'linprog',lambda *a,**kw:pytest.fail('auditing must not call an LP'))
    with pytest.raises(ValueError):verify(leaf,path,{4},proof)


def test_farkas_residual_correction_does_not_trust_approximate_dual_equality():
    # x <= .25, x in [0,1], approximate dual .99: upper=.2475+.01=.2575.
    bound=dual_upper([F(1)],F(0),np.array([[1.]]),np.array([.25]),[(0.,1.)],[.99])
    assert float(bound)==pytest.approx(.2575)


def test_limits_are_asset_specific_not_a_change_to_historical_32_leaf_parser():
    structure_info(tree(),32);structure_info(tree(),64)
    with pytest.raises(ValueError):structure_info(tree(),63)


def test_durable_debit_precedes_each_solver_and_safe_resume_preserves_categories(tmp_path):
    path=tmp_path/'checkpoint.json';limits=dict(leaf=1,node=0,proof=1)
    b=Budget(path,{'fixture':'toy'},limits,60)
    def callback():
        events=b.session.events();assert events[-1]['event']=='call_started'
        from types import SimpleNamespace
        return SimpleNamespace(success=True,status=0,nit=1,fun=0.)
    b.solve('leaf',{},callback)
    with pytest.raises(BudgetExhausted):b.solve('leaf',{},callback)
    b.checkpoint({'cursor':1,'queue':['all','items']},status='paused')
    restored=Budget(path,{'fixture':'toy'},limits,60,resume=True)
    assert restored.counts==dict(leaf=1,node=0,proof=0)
    assert restored.session.state['queue']==['all','items']
    from types import SimpleNamespace
    restored.solve('proof',{},lambda:SimpleNamespace(success=True,status=0,fun=0.))
    assert restored.counts==dict(leaf=1,node=0,proof=1)
    restored.session.close()
    # Closing an unfinished resumed run cannot be treated as a safe pause.
    with pytest.raises(CheckpointError):Budget(path,{'fixture':'toy'},limits,60,resume=True)


def test_complete_domain_logistic_leaf_can_fit_public_carrying_state(tmp_path):
    x=rows(80);i=FEATURE_NAMES.index('self_carrying_task_0');x[:,i]=np.arange(len(x))%2
    masks=np.ones((len(x),5),np.float32)
    z=np.zeros_like(masks);z[:,0]=4*(2*x[:,i]-1);z[:,1]=-z[:,0]
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    b=Budget(tmp_path/'cp.json',{'fixture':'toy'},dict(leaf=128,proof=128,node=0),60)
    leaf=fit_leaf(x,masks,p,arm='C2',path={'path':[],'leaf':0},legal=set(range(5)),budget=b)
    _,q=evaluate_leaf(leaf,x,masks)
    assert i in leaf['support'] and float(kl(p,q).mean())<.01
    assert leaf['selection_report']['support_trials']<=8
    assert not b.pending
    b.session.close()


def test_full_program_qualification_and_independent_frame_replay(tmp_path):
    from backend.training.warehouse_v626_program import Program,complete_program
    from scripts.audit_warehouse_v626 import independent_forward,basic,same_metrics,independent_dual
    from backend.training.warehouse_v622_program import masked_probabilities
    t=dict(root_ref=-1,split_weights=np.zeros((0,len(FEATURE_NAMES)),np.float32),split_bias=[],
        left_refs=[],right_refs=[],leaf_count=1,feature_dim=len(FEATURE_NAMES))
    parts={}
    for split in ('fit','calibration','eligibility','validation'):
        x=rows(80);m=np.ones((80,5),np.float32);z=np.tile(np.array([-3.,-3.,-3.,-3.,3.],np.float32),(80,1))
        parts[split]=dict(x=x,m=m,p=masked_probabilities(z,m),sampled=np.full(80,4),
            rows=[dict(episode=f'{split}-ep-{i//10}',base_seed=f'{split}-scene-{i//10}',task=i%3+1) for i in range(80)])
    b=Budget(tmp_path/'cp.json',{'fixture':'toy'},dict(leaf=64,proof=128,node=0),60)
    a=parts['fit'];leaf=fit_leaf(a['x'],a['m'],a['p'],arm='C3',path=region(t,0),legal={4},budget=b)
    p=Program('robot_1','a'*64,t,[leaf],'C3')
    p,metrics=complete_program(p,parts,b)
    assert metrics['validation']['gate']['passed']
    restored=Program.from_dict(p.to_dict(),'a'*64)
    a=parts['validation'];pred=independent_forward(restored,a['x'],a['m'])
    for purpose in ('feedback','explanation'):
        same_metrics(basic(restored,a,pred,{0},purpose),metrics['validation'][purpose])
    independent_dual(p.leaves[0],region(t,0),{4})
    broken=p.to_dict();broken['leaves'][0]['bias'][0]=5.
    with pytest.raises(ValueError):Program.from_dict(broken,'a'*64)
    b.session.close()


def test_inner_selection_cannot_split_trajectories_from_the_same_scenario():
    from scripts.run_warehouse_v626 import internal_split
    a=[dict(base_seed=str(scene),task=scene%3+1,episode=f'{scene}-trajectory-{trajectory}')
        for scene in range(30) for trajectory in range(3)]
    train,selection=internal_split(a)
    assert set(np.array([r['base_seed'] for r in a])[train]).isdisjoint(set(np.array([r['base_seed'] for r in a])[selection]))
    assert sum(selection)==18


def test_complete_worker_writes_science_result_without_optional_performance_field(monkeypatch,tmp_path):
    from scripts import run_warehouse_v626 as runner
    from backend.training.warehouse_v622_program import masked_probabilities
    structure=dict(root_ref=-1,split_weights=np.zeros((0,len(FEATURE_NAMES)),np.float32),split_bias=[],
        left_refs=[],right_refs=[],leaf_count=1,feature_dim=len(FEATURE_NAMES))
    parts={}
    for split in ('fit','calibration','eligibility','validation'):
        x=rows(120);m=np.ones((120,5),np.float32);z=np.tile(np.array([-3.,-3.,-3.,-3.,3.],np.float32),(120,1))
        parts[split]=dict(x=x,m=m,p=masked_probabilities(z,m),sampled=np.full(120,4),
            rows=[dict(episode=f'{split}-ep-{i//10}',base_seed=f'{split}-scene-{i//10}',task=(i//10)%3+1) for i in range(120)])
    path=tmp_path/'plan.json';path.write_text('{}');spec=dict(job_id='toy-C0',fold=0,role='robot_1',arm='C0')
    plan=dict(jobs=[spec],initial_S1={'fold-0-robot_1':{'structure':{}}},actor={'path':'fake-frozen-nn.npz'})
    monkeypatch.setattr(runner,'checked_plan',lambda:(path,plan))
    monkeypatch.setattr(runner,'load_parts',lambda *a:parts)
    monkeypatch.setattr(runner.v,'bound',lambda *a:structure)
    original=runner.v.file_sha
    monkeypatch.setattr(runner.v,'file_sha',lambda p:runner.v.old.ACTOR_SHA256 if str(p)=='fake-frozen-nn.npz' else original(p))
    result=runner.job('toy-C0')
    assert result['status']=='completed' and result['gate']['passed']
    assert result['performance_pass'] is False and result['scientific_gate_ignores_performance'] is True
    assert (tmp_path/'jobs/toy-C0/result.json').exists() and (tmp_path/'jobs/toy-C0/completed.json').exists()
