"""Confirmation authority tests; no environment, scene RNG, fitting or LP."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import pytest
from experiments.warehouse import v642_confirmation as c
from scripts.run_warehouse_v625 import write_new


def schedule():
    rows=[]
    for split,count,draws,offset in (('validation',300,1,0),('pressure',72,3,300)):
        for i in range(count):
            for draw in range(draws):
                base=40000000+offset+i
                rows.append(dict(case_id=f'{split}-{i}-{draw}',split=split,base=base,scenario_seed=10*base+i%3+1,
                    task=i%3+1,action_seed=6000000000+len(rows),sampling_mode='sampled',draw=draw))
    return rows


def test_no_winner_blocks_allocation_before_directory_or_seed_use(tmp_path,monkeypatch):
    def denied():raise RuntimeError('no locked audited winner')
    monkeypatch.setattr(c.metrics,'load_locked_bundle',denied)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='no locked'):c.allocate('must-not-exist',40000000,6000000000)
    assert not list(tmp_path.iterdir())


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    """Source-bound planning only; fake winner is never a model evaluation."""
    from types import SimpleNamespace
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c.v,'OUTPUT',tmp_path/'candidate')
    source=tmp_path/'source.py';source.write_text('# synthetic planning fixture\n')
    refs=[]
    for name in ('actor','parent','program','reference','all_jobs_audit'):
        path=tmp_path/(name+'.json');write_new(path,dict(synthetic=name));refs.append(c.v.pointer(path))
    actor,parent,program,reference,audit=refs
    unit=SimpleNamespace(program_ref=program,reference_ref=reference,
            program=SimpleNamespace(protection=dict(parent=parent)),
            reference=SimpleNamespace(protection=dict(parent=parent)))
    winner=SimpleNamespace(actor=actor,provenance=dict(audit=audit),asset_refs=(audit,),units=(unit,),
                           actor_role_sha256={r:r+'-synthetic' for r in ('robot_1','robot_2')})
    monkeypatch.setattr(c.metrics,'load_locked_bundle',lambda:winner)
    monkeypatch.setattr(c,'source_files',lambda:{'source.py':c.v.file_sha(source)})
    allocation=c.allocate('synthetic-plan',40000000,6000000000)
    path=c.prepare(allocation)
    return path,winner


def test_prepare_and_recheck_bind_full_audit_and_distinct_baseline_assets(prepared):
    path,winner=prepared;_,p,_=c.checked_plan(path)
    assert p['actor_role_sha256']==winner.actor_role_sha256
    assert p['prerequisites']==c.prerequisites(winner)
    assert len(p['prerequisites'])==5 and not p['training_authority']
    assert len(p['cases'])==516


@pytest.mark.parametrize('fault',['omitted_prerequisite','authority','allocation'])
def test_resealed_plan_cannot_omit_prerequisites_or_expand_authority(prepared,fault):
    path,winner=prepared;p=json.loads(path.read_bytes())
    if fault=='omitted_prerequisite':p['prerequisites'].pop(winner.provenance['audit']['path'])
    elif fault=='authority':p['training_authority']=True
    else:
        other=path.parent/'other-allocation.json';write_new(other,c.v.bound(p['allocation']))
        p['allocation']=c.v.pointer(other)
    path.write_text(json.dumps(p));path.with_name('plan.claim.json').write_text(json.dumps(dict(sha256=c.v.file_sha(path),single_use=True)))
    with pytest.raises(RuntimeError):c.checked_plan(path)


def test_exact_516_schedule_explicit_pressure_reuse_and_no_authority(tmp_path):
    cases=schedule();p=c.exact_plan(cases)
    assert len(p['uses'])==516*4 and not p['authorization_granted']
    path=tmp_path/'exact.json';write_new(path,p)
    assert c.exact.read_exact_plan(path)[0]==p
    families=c.exact.compile_families(p)
    assert sum(len(f['values']) for f in families if f['kind']=='scenario')==372
    assert sum(len(f['values']) for f in families if f['kind']=='action_rng')==516
    assert len({row['value'] for row in p['uses'] if row['argument']=='reset(seed)'})==372


@pytest.fixture
def cleared(tmp_path,monkeypatch):
    root=tmp_path/'ICLR'/'XAI-v6-integration';root.mkdir(parents=True)
    monkeypatch.setattr(c,'ROOT',root);monkeypatch.setattr(c.v,'OUTPUT',root/'output/candidate')
    for name in c.HISTORY_NAMES:(root.parent/name).mkdir(exist_ok=True)
    plan=dict(run_id='synthetic',cases=schedule(),source_files={'synthetic':'bound'},prerequisites={})
    folder=c.output_for(plan['run_id']);ledger=folder/'seed-ledger';ledger.mkdir(parents=True)
    path=folder/'plan.json';write_new(path,plan)
    exact_path=ledger/'exact-plan.json';write_new(exact_path,c.exact_plan(plan['cases']))
    traces=[];registry=tmp_path/'registry.json';write_new(registry,dict(registered_trace_paths=traces))
    monkeypatch.setattr(c,'INVENTORY_REF',c.v.pointer(registry))
    history=root/'history.json';write_new(history,{'count':0});stat=history.stat()
    row=dict(path=str(history),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=c.v.file_sha(history),
        classification='declaration_or_reservation_not_execution_proof',resolved=True,detail={})
    template=root/'backend/training/seed_ledger.py';template.parent.mkdir(parents=True)
    template.write_text('# 500_004 10_500_204\n')
    template_row=dict(path=str(template),bytes=template.stat().st_size,sha256=c.v.file_sha(template),resolved=True,
        classification='source_template_environment_reservation',detail={})
    families=c.exact.compile_families(c.exact_plan(plan['cases']))
    inv=dict(inventory=[row,template_row],families=families,missing_roots=[],selection_stats={},
        scan_roots=[str(root.parent/name) for name in c.HISTORY_NAMES],registered_trace_paths=traces,
        selection_scope=c.metadata.SELECTION_SCOPE,read_model_weight_files=False,
        scope_exclusions=list(map(str,[ledger/'proposal.json',ledger/'bounded',*c.exclusions(plan)])))
    inv_path=ledger/'inventory.json';write_new(inv_path,inv)
    evidence=c.exact.project_inventory(inv,families)
    evidence.update(exact_plan=c.v.pointer(exact_path),bounded_or_triaged_inventory=c.v.pointer(inv_path),
        exact_auditor_sha256=c.v.file_sha(c.exact.__file__))
    evidence_path=ledger/'exact/result.json';evidence_path.parent.mkdir();write_new(evidence_path,evidence)
    now=datetime.now(timezone.utc)
    clearance=dict(schema='warehouse-v642-confirmation-clearance.v1',plan=c.v.pointer(path),cases_sha256=c.v.digest(plan['cases']),
        exact=c.v.pointer(evidence_path),registered_traces=traces,
        selection=[dict(path=str(history),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)],weight_paths=[],
        issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),approved_finite_resets=516)
    clear_path=ledger/'clearance.json';write_new(clear_path,clearance)
    monkeypatch.setattr(c,'history_selection',lambda p,t:(deepcopy(clearance['selection']),[]))
    return path,plan,clearance,evidence,clear_path,evidence_path,template


def overwrite(path,value):path.write_text(json.dumps(value))


def test_clearance_replays_exact_inventory_and_fixed_scope(cleared):
    path,plan,value,*_=cleared
    assert c.checked_clearance(path,plan)==value


def test_clearance_selection_cannot_omit_audited_file_even_if_current_scan_matches(cleared):
    path,plan,clearance,_,file,*_=cleared
    clearance['selection']=[];overwrite(file,clearance)
    with pytest.raises(RuntimeError,match='audited inventory'):c.checked_clearance(path,plan)


@pytest.mark.parametrize('field,value',[
    ('registered_traces',['unregistered']),('expires_utc','2000-01-02T00:00:00+00:00')])
def test_changed_trace_registry_or_expiry_cannot_authorize(cleared,field,value):
    path,plan,clearance,_,file,*_=cleared
    clearance[field]=value;overwrite(file,clearance)
    with pytest.raises(RuntimeError):c.checked_clearance(path,plan)


def test_template_change_is_not_hidden_by_metadata_stat_selection(cleared):
    path,plan,_,_,_,_,template=cleared;template.write_text('# changed 500_004 10_500_204\n')
    with pytest.raises(RuntimeError,match='source reservation changed'):c.checked_clearance(path,plan)


@pytest.mark.parametrize('key,value',[
    ('scope_exclusions',[]),('selection_scope','partial'),('registered_trace_paths',['other']),('read_model_weight_files',True)])
def test_scope_substitution_is_rejected_even_when_evidence_replay_matches(cleared,key,value):
    path,plan,clearance,evidence,clear_path,evidence_path,_=cleared
    inv_path=Path(evidence['bounded_or_triaged_inventory']['path']);inv=json.loads(inv_path.read_bytes());inv[key]=value;overwrite(inv_path,inv)
    new=c.exact.project_inventory(inv,c.exact.compile_families(c.exact_plan(plan['cases'])))
    new.update(exact_plan=evidence['exact_plan'],bounded_or_triaged_inventory=c.v.pointer(inv_path),exact_auditor_sha256=evidence['exact_auditor_sha256'])
    overwrite(evidence_path,new);clearance['exact']=c.v.pointer(evidence_path);overwrite(clear_path,clearance)
    with pytest.raises(RuntimeError,match='scope/exclusions'):c.checked_clearance(path,plan)


def cap_for(cleared,monkeypatch):
    path,plan,clearance,_,clear_path,*_=cleared
    monkeypatch.setattr(c,'source_files',lambda:plan['source_files'])
    return c.Capability(path,plan,c.v.pointer(clear_path),clearance['exact'],clearance['expires_utc'])


def test_preauthorization_is_ordered_bound_and_fsynced_before_constructor(cleared,monkeypatch,tmp_path):
    cap=cap_for(cleared,monkeypatch);token=c.ACTIVE.set(cap)
    try:
        with (tmp_path/'auth').open('x') as journal:
            cap.before_reset(cap.plan['cases'][0],journal)
            row=json.loads((tmp_path/'auth').read_text())
            assert row['index']==0 and row['case']==cap.plan['cases'][0]
            assert row['clearance_sha256']==cap.clearance['sha256'] and cap.cursor==1
            with pytest.raises(RuntimeError,match='schedule'):cap.before_reset(cap.plan['cases'][0],journal)
    finally:c.ACTIVE.reset(token)


@pytest.mark.parametrize('mode',['inactive','expired','source','clearance'])
def test_no_reset_after_authority_expiry_source_or_evidence_change(cleared,monkeypatch,tmp_path,mode):
    cap=cap_for(cleared,monkeypatch)
    token=c.ACTIVE.set(None if mode=='inactive' else cap)
    if mode=='expired':cap.expires_utc='2000-01-01T00:00:00+00:00'
    if mode=='source':monkeypatch.setattr(c,'source_files',lambda:{'changed':'yes'})
    if mode=='clearance':Path(cap.clearance['path']).write_text('{}')
    try:
        with (tmp_path/'auth').open('x') as journal:
            with pytest.raises(RuntimeError):cap.before_reset(cap.plan['cases'][0],journal)
        assert cap.cursor==0 and (tmp_path/'auth').read_text()==''
    finally:c.ACTIVE.reset(token)


def test_partial_failure_keeps_bound_single_use_claim_and_denies_retry(cleared,monkeypatch):
    path,plan,clearance,_,clear_path,*_=cleared
    monkeypatch.setattr(c,'checked_plan',lambda path:(path,plan,None))
    monkeypatch.setattr(c,'source_files',lambda:plan['source_files'])
    c.claim_path().parent.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        with c.capability(path) as cap:
            assert c.claim_path().exists()
            claim=json.loads(c.claim_path().read_bytes())
            assert claim['clearance']==c.v.pointer(clear_path) and claim['exact_evidence']==clearance['exact']
            raise RuntimeError('synthetic interruption')
    assert json.loads((path.parent/'collection/failed.json').read_bytes())['preauthorized_cases']==0
    with pytest.raises(RuntimeError,match='already consumed'):
        with c.capability(path):pytest.fail('repeated attempt entered')
    assert c.ACTIVE.get() is None


@pytest.mark.parametrize('terminal',['passed','claimed_foreign_plan'])
def test_repeated_pipeline_cannot_close_a_previous_success_or_foreign_attempt(tmp_path,monkeypatch,terminal):
    from scripts import run_warehouse_v642_confirmation as runner
    monkeypatch.setattr(c.v,'OUTPUT',tmp_path)
    write_new(c.claim_path(),dict(plan={'path':'other','sha256':'other'},owner_pid=0))
    if terminal=='passed':write_new(tmp_path/'confirmation.passed.json',dict(confirmation_passed=True))
    with pytest.raises(RuntimeError,match='already consumed or terminal'):
        runner.main(['--pipeline','--plan',str(tmp_path/'unused.json')])
    assert not (tmp_path/'confirmation.failed.json').exists()
    if terminal=='passed':assert json.loads((tmp_path/'confirmation.passed.json').read_bytes())['confirmation_passed']


@pytest.mark.parametrize('owned',[True,False])
def test_pipeline_only_closes_a_new_claim_owned_by_this_call(tmp_path,monkeypatch,owned):
    import os
    from scripts import run_warehouse_v642_confirmation as runner
    monkeypatch.setattr(c.v,'OUTPUT',tmp_path)
    plan=tmp_path/'plan.json';write_new(plan,{'synthetic':True})
    monkeypatch.setattr(c,'audit_seeds',lambda path:None)
    def interrupted(path):
        write_new(c.claim_path(),dict(plan=c.v.pointer(path),owner_pid=os.getpid() if owned else -1))
        raise RuntimeError('synthetic collection interruption')
    monkeypatch.setattr(c,'collect',interrupted)
    with pytest.raises(RuntimeError,match='synthetic collection interruption'):
        runner.main(['--pipeline','--plan',str(plan)])
    assert (tmp_path/'confirmation.failed.json').exists() is owned


@pytest.mark.parametrize('changed',['claim','clearance','exact_evidence'])
def test_scoring_rejects_changed_post_collection_authority(tmp_path,changed):
    from scripts import audit_warehouse_v642_confirmation as audit
    plan=tmp_path/'plan.json';write_new(plan,{'synthetic':True})
    value=dict(actor={'synthetic':'actor'},source_sha256='source',cases=[])
    refs={}
    for key in ('claim','clearance','exact_evidence'):
        target=tmp_path/(key+'.json');write_new(target,{'immutable':key});refs[key]=c.v.pointer(target)
    receipt=tmp_path/'receipt.json';write_new(receipt,dict(**refs,cases={},outcomes=[]))
    record=dict(schema='warehouse-v642-confirmation-collection-audit.v1',integrity_passed=True,
        plan=c.v.pointer(plan),source_sha256=value['source_sha256'],actor=value['actor'],
        actual_reset_count=516,public_complete_episodes=516,collection=c.v.pointer(receipt),**refs)
    write_new(tmp_path/'collection-audit.json',record)
    Path(refs[changed]['path']).write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='audited collection authority changed or missing'):
        audit.load_datasets(plan,value)
