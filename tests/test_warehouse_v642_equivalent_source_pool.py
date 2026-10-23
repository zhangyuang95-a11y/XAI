"""Clearly MOCK source/NN fixtures; no real physics or research NN evaluation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,pytest
from backend.training import warehouse_v642_equivalent_source_pool as e


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));p.chmod(0o444);return e.pointer(p)


@pytest.fixture
def data(tmp_path,monkeypatch):
    cases=[{'episode':s} for s in e.native.online.SPLITS]
    inventory=put(tmp_path/'inventory.json',dict(cases=cases));sources={}
    for variant in ('raw','regularized'):
        actor=put(tmp_path/variant/'actor.json',{'MOCK_actor':'same'})
        owner=put(tmp_path/variant/'owner.json',dict(program_evidence_inventory=inventory));items={}
        for case in cases:
            initial=put(tmp_path/variant/(case['episode']+'-initial.json'),{'MOCK_private_initial':0})
            final=put(tmp_path/variant/(case['episode']+'-final.json'),{'MOCK_private_final':1})
            trace=put(tmp_path/variant/(case['episode']+'-trace.json'),dict(schema=e.native.EPISODE_VERSION,
                case=case,behavior_actor=actor,events=[{'MOCK_frame':0,'value':1.}]))
            items[case['episode']]=dict(behavior_actor=actor,trace=trace,initial_environment=initial,final_environment=final)
        sources[variant]=put(tmp_path/variant/'source.json',dict(schema=e.native.VERSION,scope=e.native.SCOPE,
            owner=owner,inventory=inventory,cases=items))
    monkeypatch.setattr(e.native.FrozenWarehouseV6Actor,'load',lambda *a,**kw:SimpleNamespace(model_version='MOCK_same_mode',role_sha256={'robot_1':'a','robot_2':'b'}))
    return sources


def test_complete_equivalence_and_explicit_no_final_target_reuse(data):
    v=e.equivalent_inputs(data['raw'],data['regularized'])
    assert v['complete_episodes']==4 and v['joint_physical_frames']==4
    assert v['all_source_episodes_compared'] and v['no_final_NN_target_reuse']
    assert v['new_model_forwards']==v['new_environment_resets']==0


@pytest.mark.parametrize('change',['trace','float_type','endpoint','weights','model_mode'])
def test_any_input_or_NN_semantic_difference_prevents_reuse(data,tmp_path,monkeypatch,change):
    source=e.bound(data['regularized']);item=next(iter(source['cases'].values()))
    if change in ('trace','float_type'):
        t=e.bound(item['trace']);t['events'][0]['value']=2. if change=='trace' else 1
        item['trace']=put(tmp_path/'different-trace.json',t)
    elif change=='endpoint':item['final_environment']=put(tmp_path/'different-final.json',{'MOCK_private_final':2})
    else:
        original=e.native.FrozenWarehouseV6Actor.load
        def load(path,**kw):
            x=original(path,**kw)
            if 'regularized' in str(path):
                if change=='weights':x.role_sha256['robot_2']='different'
                else:x.model_version='MOCK_different_mode'
            return x
        monkeypatch.setattr(e.native.FrozenWarehouseV6Actor,'load',load)
    altered=put(tmp_path/'altered-source.json',source)
    assert e.equivalent_inputs(data['raw'],altered) is None


def test_actual_bytes_checked_not_just_declared_sha(data):
    item=next(iter(e.bound(data['regularized'])['cases'].values()))['final_environment'];p=Path(item['path'])
    p.chmod(0o644);p.write_text('{"MOCK_private_final":123}');p.chmod(0o444)
    with pytest.raises(RuntimeError,match='changed'):e.equivalent_inputs(data['raw'],data['regularized'])


@pytest.mark.parametrize('reuse',[True,False])
def test_audit_pair_calls_real_boundary_once_and_never_mints_target_token(data,tmp_path,monkeypatch,reuse):
    if not reuse:
        source=e.bound(data['regularized']);item=next(iter(source['cases'].values()))
        trace=e.bound(item['trace']);trace['events'][0]['value']=2.
        item['trace']=put(tmp_path/'different-source-trace.json',trace)
        data['regularized']=put(tmp_path/'different-source.json',source)
    calls=[]
    monkeypatch.setattr(e.native,'source_files',lambda:{})
    def MOCK_native_full_physics_NN_audit(ref,**kw):
        calls.append(ref);s=e.bound(ref);parts={}
        for split in e.native.online.SPLITS:
            a={key:np.zeros((1,2,5),np.float32) for key in ('x','m','behavior_p')}
            a['executed']=np.zeros((1,2),np.int64);a['rows']=[{'MOCK':split}]
            for v in a.values():
                if isinstance(v,np.ndarray):v.flags.writeable=False
            parts[split]=a
        refs=(ref,s['owner'],s['inventory']);sha=e.native.data_sha(parts)
        audit=dict(source=ref,owner=s['owner'],inventory=s['inventory'],source_sha256=e.digest({}),
            observed_data_sha256=sha,asset_manifest_sha256=e.digest(refs),complete_source_episodes=4,
            joint_physical_frames=4,outcomes=[dict(case=c,behavior_actor=s['cases'][c['episode']]['behavior_actor']) for c in e.bound(s['inventory'])['cases']],
            current_policy_task_quality_estimate=None,source_policy_outcomes_only=True)
        return e.native.ObservedPool(ref,s['owner'],parts,refs,{},SimpleNamespace(verify=lambda:None),sha,audit,e.digest(audit)).verify()
    monkeypatch.setattr(e.native,'audit_observed_pool',MOCK_native_full_physics_NN_audit)
    result=e.audit_pair(data,reference_directory='MOCK',output=tmp_path/'pair')
    assert calls==([data['raw']] if reuse else [data['raw'],data['regularized']])
    a,b=result['pools']['raw'],result['pools']['regularized']
    assert type(a) is type(b) is e.native.ObservedPool
    assert (a.parts is b.parts) is reuse
    assert a._replayed_target_cache is b._replayed_target_cache is None
    assert b.audit.get('identical_input_source_simulation_reused',False) is reuse
    assert b.audit['source']==data['regularized']
    b.verify();assert len({r['path'] for r in b.refs})==len(b.refs)
    checked=e.verify_report(result['proof_refs'][0],data)
    assert checked['audit_sha256']['regularized']==b.audit_sha256
    assert checked['mode']==('native_first_audit_then_complete_input_equivalence' if reuse else 'separate_native_full_audits')
