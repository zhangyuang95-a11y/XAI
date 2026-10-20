"""Synthetic independent proof, metric, ledger and engineering safety tests."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_warehouse_v625 as a
from tests.test_warehouse_v624_audit import audited_jobs, plan

NAMES=('self_battery_fraction','legal_up','legal_right','legal_down','legal_left','legal_wait',
       'public_hard_risk_context','self_row','unknown_unbounded')


def leaf(support=(0,),bias=None,weights=None,witness=4):
    bias=np.asarray([-2,-3,-4,-5,1] if bias is None else bias,np.float32)
    w=np.zeros((5,len(support)),np.float32) if weights is None else np.asarray(weights,np.float32)
    return {'version':a.LEAF_VERSION,'arithmetic':a.ARITHMETIC,'feature_names':list(NAMES),
            'support':list(support),'support_names':[NAMES[i] for i in support],
            'bias':bias.tolist(),'weights':w.tolist(),'witness':witness,'mode':'P1'}


def test_independent_exact_certificate_matches_production_without_shared_certifier():
    from backend.training.warehouse_v625_program import certify_leaf
    model=leaf(weights=np.asarray([[.5],[.1],[-.2],[.7],[1]],np.float32))
    independent=a.certificate(model,NAMES,'P1',{4})
    assert independent==certify_leaf(model,NAMES,[4])
    assert independent['corner_count']==2
    assert all(c['minimum_exact_gap'][1]>0 for c in independent['illegal_action_checks'])


def test_sample_safe_model_fails_public_box_not_empirical_extrema():
    model=leaf(weights=[[10],[0],[0],[0],[0]])
    # On x=0 the raw action is WAIT; at x=1 illegal UP overtakes it.
    assert a.sequential_logits(np.zeros((1,len(NAMES)),np.float32),[0],np.asarray(model['weights'],np.float32),np.asarray(model['bias'],np.float32)).argmax()==4
    with pytest.raises(RuntimeError,match='global float32'):a.certificate(model,NAMES,'P1',{4})


def test_legal_actions_can_change_within_leaf_without_illegal_raw_argmax():
    model=leaf(bias=[0,-3,-4,-5,1],weights=[[2],[0],[0],[0],[0]])
    proof=a.certificate(model,NAMES,'P1',{0,4})
    x=np.zeros((2,len(NAMES)),np.float32);x[1,0]=1
    raw=a.sequential_logits(x,[0],np.asarray(model['weights'],np.float32),np.asarray(model['bias'],np.float32))
    assert raw.argmax(1).tolist()==[4,0]
    assert proof['path_legal_actions']==[0,4]


@pytest.mark.parametrize('change',[
    lambda l:l.update(support=[8],support_names=[NAMES[8]]),
    lambda l:l.update(support=[0,6,7],support_names=[NAMES[i] for i in (0,6,7)],weights=np.zeros((5,3)).tolist()),
    lambda l:l['bias'].__setitem__(0,.1),
    lambda l:l['bias'].__setitem__(0,33.),
    lambda l:l.update(arithmetic='ordinary_matmul'),
    lambda l:l.update(witness=0),
])
def test_bad_domain_source_budget_numeric_contract_and_witness_rejected(change):
    model=leaf();change(model)
    with pytest.raises(RuntimeError):a.certificate(model,NAMES,'P1',{4})


def test_strict_margin_and_p0_language_are_enforced():
    with pytest.raises(RuntimeError):a.certificate(leaf(),NAMES,'P0',{4})
    model=leaf(support=(),bias=[0,-3,-4,-5,.0009])
    with pytest.raises(RuntimeError,match='global float32'):a.certificate(model,NAMES,'P0',{4})
    good=leaf(support=());assert a.certificate(good,NAMES,'P0',{4})['float32_error_per_action']==[0.]*5


def test_all12_and_only_p1_can_advance():
    jobs=audited_jobs();p=plan(jobs)
    assert a.reconstruct_summary(p,'e'*64,jobs)['selected_arm']=='P1'
    jobs[-1]['validation']['explanation']['coverage']=.44;jobs[-1]['gate_passed']=False
    summary=a.reconstruct_summary(p,'e'*64,jobs)
    assert summary['arms']['P0']['all_six_pass'] and summary['selected_arm'] is None
    assert summary['next_stage']=='closed_failed_program_precheck'
    with pytest.raises(RuntimeError):a.reconstruct_summary(p,'e'*64,jobs[:-1])
    jobs[-1]['gate_passed']=True
    with pytest.raises(RuntimeError):a.reconstruct_summary(p,'e'*64,jobs)


def event_chain():
    rows=[]
    def add(kind,data):
        body={'sequence':len(rows)+1,'previous_sha256':rows[-1]['record_sha256'] if rows else '0'*64,
              'event':kind,'data':data,'elapsed_seconds':float(len(rows))}
        rows.append(dict(body,record_sha256=a.digest(body)))
    add('created',{'identity_sha256':'a'*64,'budget':{'maximum_calls':19968,'wall_seconds':4800.}})
    add('call_started',{'call':1,'metadata':{'leaf_key':'b'*64,'rows':100,'phase':'candidate','path':[1],
                                          'path_legal_actions':[4]}})
    detail={'support':[0],'witness':4,'solver':'SLSQP','iterations':5,'success':True,'solver_success':True,
            'certificate_passed':True,'convex_fixed_support_witness_problem':True}
    add('call_completed',{'call':1,'success':True,'detail':detail})
    return rows


def test_durable_predebit_hash_chain_and_actual_solver_reconciliation():
    rows=event_chain();value=a.audit_ledger(rows,identity_sha='a'*64,expected_calls=1,names=NAMES)
    assert value['solver_completions']==1 and value['pending_calls']==0
    tampered=deepcopy(rows);tampered[1]['data']['call']=2
    with pytest.raises(RuntimeError):a.audit_ledger(tampered,names=NAMES)
    with pytest.raises(RuntimeError):a.audit_ledger(rows[:-1],names=NAMES)
    assert a.audit_ledger(rows[:-1],names=NAMES,incomplete=True)['pending_calls']==1
    with pytest.raises(RuntimeError):a.audit_ledger(rows,expected_calls=2,names=NAMES)


def test_independent_context_rejects_stale_certificate_and_checks_full_distribution():
    from backend.training.warehouse_v6_actor import FEATURE_NAMES
    names=tuple(FEATURE_NAMES);model=leaf(support=())
    model['feature_names']=list(names);model['certificate']=a.certificate(model,names,'P0',{4})
    tree={'split_weights':[],'split_bias':[],'left_refs':[],'right_refs':[],'root_ref':-1,'leaf_count':1}
    raw={'mode':'P0','feature_names':list(names),'leaves':[model],'structure':tree}
    x=np.zeros((4,len(names)),np.float32);m=np.ones((4,5),np.float32)
    for i,name in enumerate(names):
        if name.startswith('legal_'):x[:,i]=1
    p=np.full((4,5),.2,np.float32);data=(x,m,p,np.zeros(4,dtype=int),np.array(['e']*4),np.array(['s']*4))
    context=a.context(raw,data,names)
    assert context['q'].shape==(4,5) and np.allclose(context['q'].sum(1),1)
    assert (context['raw_action']==4).all()
    raw['leaves'][0]['certificate']['binding_sha256']='a'*64
    with pytest.raises(RuntimeError):a.context(raw,data,names)


def test_engineering_fault_crash_deadline_corruption_suite(tmp_path):
    from scripts.verify_warehouse_v625_engineering import fault_suite
    result=fault_suite(tmp_path)
    assert result['passed'] and len(result['checks'])>=18


def test_calibration_all_five_temperatures_independently_recomputed():
    from backend.training.warehouse_v625_program import calibrate_leaf,certify_leaf
    model=leaf(weights=[[.5],[.1],[-.2],[.7],[1.]])
    model['certificate']=certify_leaf(model,NAMES,[4])
    x=np.zeros((12,len(NAMES)),np.float32);x[:,0]=np.linspace(0,1,12,dtype=np.float32)
    m=np.ones((12,5),np.float32)
    for index in range(1,6):x[:,index]=1.
    p=np.tile(np.asarray([.03,.02,.01,.01,.93],np.float32),(12,1))
    calibrated,report=calibrate_leaf(model,x,m,p,NAMES,[4])
    c={'names':NAMES,'proof':{0:{4}},'leaves':np.zeros(12,int),'x':x,'m':m,'p':p}
    a.check_calibration(calibrated,report,c,0,'P1')
    bad=deepcopy(report);bad['candidate_reports'][0]['full_distribution_forward_kl']+=.01
    with pytest.raises(RuntimeError):a.check_calibration(calibrated,bad,c,0,'P1')


def test_cas_graph_rejects_corrupted_reachable_object(tmp_path):
    from backend.training.warehouse_v625_checkpoint import ContentStore
    store=ContentStore(tmp_path/'checkpoint.json.objects')
    model=store.put_json({'model':leaf()})
    best=store.put_json({'kind':'fitted_tree_state','leaves':[model],'score':[0,0,0,-1],'tree_sha256':'e'*64})
    root=store.put_json({'schema':'warehouse-v625-search-state.v1','best':best,'states':[], 'leaf_cache':[],'transitions':[]})
    manifest={'state':root,'object_pack_registry':store.flush()}
    evidence=a.Evidence();result=a.checkpoint_graph(manifest,tmp_path,evidence)
    assert result['verified_objects']==3
    store.physical_path(model).write_text('{}')
    with pytest.raises(RuntimeError):a.checkpoint_graph(manifest,tmp_path,a.Evidence())


def test_production_complete_program_metrics_equal_independent_every_partition(monkeypatch,tmp_path):
    from backend.training import warehouse_v625_program as production
    from tests.test_warehouse_v625_program import complete_fixture
    actor,parts,tree=complete_fixture(monkeypatch)
    program=production.fit_program('robot_1',actor,*parts,structure=tree,mode='P0',
        fit_budget={'maximum_leaf_api_calls':32},checkpoint_path=tmp_path/'final.json')
    raw=program.to_dict()
    for split,rows in zip(production.SPLITS,parts,strict=True):
        x,m,sampled=production._arrays(rows)
        data=(x,m,a.math.softmax(actor.roles['robot_1'].logits_encoded(x),m),sampled,
              np.asarray([r['episode'] for r in rows]),np.asarray([r['base_seed'] for r in rows]))
        c=a.context(raw,data,production.FEATURE_NAMES)
        actual=production.program_metrics(program,actor,rows)
        for purpose in ('feedback','explanation'):
            a.math.assert_equal(a.section(c,raw[purpose+'_leaves'],purpose),actual[purpose],split+'.'+purpose)
            a.math.assert_equal(a.section(c,[0],purpose),actual['per_leaf']['0'][purpose],split+'.leaf')
        if split=='calibration':a.check_calibration(raw['leaves'][0],raw['report']['leaf_calibrations']['0'],c,0,'P0')


def test_real_checkpoint_session_events_pass_independent_ledger_contract(tmp_path):
    from backend.training.warehouse_v625_checkpoint import CheckpointSession
    identity={'actual_checkpoint_API':True}
    session=CheckpointSession(tmp_path/'checkpoint.json',identity)
    token=session.consume({'leaf_key':'b'*64,'rows':100,'phase':'candidate','path':[1],
                           'path_legal_actions':[4]})
    detail={'support':[0],'witness':4,'solver':'SLSQP','iterations':5,'success':True,
            'solver_success':True,'certificate_passed':True,'convex_fixed_support_witness_problem':True}
    session.finish(token,success=True,detail=detail)
    session.commit({'engineering_only':True},status='completed')
    result=a.audit_ledger(session.events(),identity_sha=a.digest(identity),expected_calls=1,names=NAMES)
    assert result['search_calls']==result['solver_completions']==1
    assert result['pending_calls']==0


@pytest.mark.parametrize('resumed',[False,True])
def test_checkpoint_fraction_uses_explicit_actual_or_conservative_scope(resumed):
    timing={'checkpoint_segment_io_seconds':2.,'checkpoint_total_io_lower_bound_seconds':5. if resumed else 2.,
            'checkpoint_total_io_upper_bound_seconds':6. if resumed else 2.,
            'checkpoint_total_io_seconds':6. if resumed else 2.,
            'actual_elapsed_lower_bound_seconds':100.,'actual_elapsed_upper_bound_seconds':101. if resumed else 100.}
    numerator=6. if resumed else 2.;denominator=100. if resumed else 101.
    report={'checkpoint_timing':timing,'checkpoint_io_seconds':numerator,
            'checkpoint_fraction_numerator_seconds':numerator,'checkpoint_fraction_denominator_seconds':denominator,
            'checkpoint_fraction_scope':'resumed_cumulative_io_upper_over_actual_lower_bound' if resumed else 'fresh_measured_segment',
            'runner_segment_elapsed_seconds':101.,'resume_count':1 if resumed else 0,
            'checkpoint_fraction':numerator/denominator,'checkpoint_overhead_at_most_ten_percent':True}
    assert a.checkpoint_timing_checks(report)['checkpoint_fraction']==numerator/denominator
    tampered=deepcopy(report);tampered['checkpoint_fraction_denominator_seconds']+=1
    with pytest.raises(RuntimeError):a.checkpoint_timing_checks(tampered)
    tampered=deepcopy(report);tampered['checkpoint_fraction_scope']='actual_wall_even_after_resume'
    with pytest.raises(RuntimeError):a.checkpoint_timing_checks(tampered)


def test_packed_cas_checks_logical_hash_even_when_carrier_hash_is_valid(tmp_path):
    import io,json
    from hashlib import sha256
    directory=tmp_path/'checkpoint.json.objects';directory.mkdir()
    def physical(raw,kind):
        identity=sha256(raw).hexdigest();(directory/(identity+'.'+kind)).write_bytes(raw)
        return {'sha256':identity,'kind':kind,'size':len(raw)}
    def doc(value):return physical(a.canonical(value),'json')
    buffer=io.BytesIO();np.save(buffer,np.frombuffer(b'{}',dtype=np.uint8),allow_pickle=False)
    data=physical(buffer.getvalue(),'npy')
    index=doc({'schema':'warehouse-v625-CAS-packed-index.v1','data':data,'entries':[['a'*64,'json',2,0]]})
    registry=doc({'schema':'warehouse-v625-CAS-pack-registry.v1','packs':[index]})
    with pytest.raises(RuntimeError,match='logical SHA'):
        a.checkpoint_graph({'object_pack_registry':registry,'state':{'sha256':'a'*64,'kind':'json','size':2}},tmp_path,a.Evidence())


def test_preflight_source_snapshot_is_created_before_work_and_cannot_be_rewritten(tmp_path):
    from hashlib import sha256
    from scripts.verify_warehouse_v625_engineering import snapshot_sources
    root=tmp_path/'source';root.mkdir();(root/'entry.py').write_bytes(b'print(1)\n')
    files={'entry.py':sha256(b'print(1)\n').hexdigest()};identity=a.digest(files)
    receipt=snapshot_sources(files,identity,tmp_path/'snapshots',root=root)
    target=Path(receipt['path'])/'entry.py'
    assert target.read_bytes()==b'print(1)\n' and not target.stat().st_mode & 0o222
    assert receipt['verified_before_measurement'] and receipt['file_count']==1
    assert snapshot_sources(files,identity,tmp_path/'snapshots',root=root)==receipt
    (root/'entry.py').write_bytes(b'print(2)\n')
    with pytest.raises(RuntimeError,match='source changed'):
        snapshot_sources(files,identity,tmp_path/'snapshots',root=root)
