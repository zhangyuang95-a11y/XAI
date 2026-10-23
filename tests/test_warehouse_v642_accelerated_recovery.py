"""Exact numerical continuation guards; no real scenes or training."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import json
import pytest
from experiments.warehouse import v642_accelerated_recovery as r


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444)
    return r.pointer(path)


def test_no_authorization_reads_no_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'checked_sources',lambda *a:pytest.fail('unauthorized contract read'))
    with pytest.raises(RuntimeError,match='explicit acceleration'):
        r.prepare(parent_plan_ref=None,cutover_ref=None,source_contract_ref=None,recovery_id='x',authorization='implicit')
    assert not list(tmp_path.iterdir())


def test_source_snapshot_detects_live_change_writable_and_byte_tamper(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();source=root/'example.py';source.write_text('a=1\n')
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'source_files',lambda:{'example.py':r.file_sha(source)})
    ref=r.prepare_sources(tmp_path/'contract');assert r.checked_sources(ref)['training_authority'] is False
    snap=tmp_path/'contract/source/example.py';snap.chmod(0o644)
    with pytest.raises(RuntimeError,match='snapshot'):r.checked_sources(ref)
    snap.chmod(0o444);source.write_text('a=2\n')
    with pytest.raises(RuntimeError,match='contract'):r.checked_sources(ref)


def test_live_rejects_original_expiry_without_renewal(monkeypatch):
    active=r.AcceleratedOwner.__new__(r.AcceleratedOwner);active.failed=active.closed=False
    active.recovery_claim_ref={'path':'claim','sha256':'x'}
    active.recovery_plan={'expires_utc':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}
    monkeypatch.setattr(r,'pointer',lambda p:active.recovery_claim_ref)
    token=r.previous.old.ACTIVE.set(active)
    try:
        with pytest.raises(RuntimeError,match='does not renew'):active.live()
    finally:r.previous.old.ACTIVE.reset(token)


@pytest.mark.parametrize('changed',[False,True])
def test_committed_update_requires_bit_and_type_exact_equality(tmp_path,monkeypatch,changed):
    result={'epochs':[{'gradient':0.125}],'end_joint_step':17824};prior=put(tmp_path/'original.json',result)
    recomputed=deepcopy(result)
    if changed:recomputed['epochs'][0]['gradient']+=1e-12
    active=r.AcceleratedOwner.__new__(r.AcceleratedOwner);active.live=lambda **kw:None
    active.states={'raw':SimpleNamespace(joint_steps=17824)}
    active.pending={'raw':dict(state_sha256='same',trace={'path':'trace','sha256':'x'},
        batch={'start_joint_step':16800,'rows':[{}]*1024},original_update=prior)}
    active.providers={};active.cores={'raw':SimpleNamespace(update_batch=lambda *a,**kw:recomputed)}
    active.event=lambda *a,**kw:None;active.folder=tmp_path/'new';active.updates={'raw':[prior]}
    active.replay_proofs=[];active.state_fingerprints={};monkeypatch.setattr(r,'training_fingerprint',lambda s:'same')
    if changed:
        with pytest.raises(RuntimeError,match='differs from the original'):active.update('raw')
        assert active.replay_proofs==[] and 'raw' in active.pending
    else:
        active.update('raw');assert active.updates['raw']==[prior] and not active.pending
        assert active.replay_proofs[0]['added_scientific_joint_steps']==0
        assert active.replay_proofs[0]['replayed_physical_frames']==1024


def test_restore_raw_uses_actual_full_checkpoint_without_observed_pool(tmp_path,monkeypatch):
    snapshot=put(tmp_path/'snapshot.json',dict(training_state={'path':'state','sha256':'state-sha'},core={},reader={},
        environments=[None]*8,active_cases=[None]*8,reset_cursors=[2]*8,program_source_cursor=1650,
        reserved_joint_steps=18600,refits=[15000,16800,18600]))
    state=SimpleNamespace(reference_sha256='reference',joint_steps=18600)
    active=SimpleNamespace(recovery_plan={'checkpoints':{'raw':snapshot}},
        plan=SimpleNamespace(proposal=SimpleNamespace(readiness=SimpleNamespace(value={
            'reference':{'buffer':{'path':'buffer','sha256':'buffer-sha'}}}),value={'train_case_inventory':'inventory'})),
        owners={'raw':'owner'},config='config',event=lambda *a,**kw:None)
    for name in ('states','cores','readers','environments','active_cases','cursors','evidence_cursor','reserved','refits','state_fingerprints','pools','providers'):
        setattr(active,name,{})
    monkeypatch.setattr(r,'load_checkpoint',lambda p,s:state)
    monkeypatch.setattr(r.ReferenceTrainingBuffer,'load',lambda *a:'buffer')
    monkeypatch.setattr(r.previous.old.seeds.partitions,'check_reference_buffer',lambda *a:None)
    monkeypatch.setattr(r.PPOUpdateCore,'restore',lambda *a,**kw:'core')
    monkeypatch.setattr(r.EnvironmentRolloutReplay,'restore',lambda *a,**kw:'reader')
    monkeypatch.setattr(r,'training_fingerprint',lambda s:'state-fingerprint')
    monkeypatch.setattr(r.previous.old.observed,'audit_observed_pool',lambda *a,**kw:pytest.fail('raw has no remaining program use'))
    r.restore_variant(active,'raw')
    assert active.states['raw'] is state and active.reserved['raw']==18600
    assert active.refits['raw']==[15000,16800,18600] and active.providers==active.pools=={}


def test_actual_supervisor_observes_failure_without_success_binding(tmp_path,monkeypatch):
    from scripts import resume_warehouse_v642_accelerated as cli
    plan=put(tmp_path/'plan.json',{})
    monkeypatch.setattr(cli.recovery,'checked_plan',lambda ref:dict(
        source_files={'scripts/resume_warehouse_v642_accelerated.py':r.file_sha(cli.__file__)},original_plan=plan))
    monkeypatch.setattr(cli.recovery,'claim_path',lambda ref:tmp_path/'no.claim')
    class Child:
        pid=123
        def __init__(self,*a,**kw):pass
        def wait(self):return -15
    monkeypatch.setattr(cli.subprocess,'Popen',Child)
    with pytest.raises(SystemExit) as failure:cli.supervise(plan)
    assert failure.value.code==143
    value=r.bound(r.pointer(tmp_path/'supervisor-exit.json'))
    assert value['exit_code']==-15 and 'result' not in value and 'completed' not in value
