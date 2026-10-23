"""Named synthetic provider/NN doubles; no real candidate or environment."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from backend.training import warehouse_v642_long_evidence as c


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);return c._put(path,value)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    p=c.native.CurrentNNEvidenceProvider.__new__(c.native.CurrentNNEvidenceProvider)
    p.output=tmp_path/'native-provider';p.output.mkdir();p.owner_ref=put(tmp_path/'owner.json',dict(synthetic_only=True))
    p.sources={'fake-native':'fake'};p.started_ref=put(p.output/'started.json',dict(synthetic_only=True))
    p.config=c.native.PPOConfig(joint_step_budget=200000);p.variant='regularized';p.pair_seed=17
    p.events=[];p.installed=[];p.used_epochs=set();p.call_count=0;p.latch=False;p.reason=None;p.failed=False
    p.observed=SimpleNamespace(refs=(),source_ref=put(tmp_path/'observed.json',dict(synthetic_only=True)))
    p.programs={role:put(tmp_path/(role+'-program.json'),dict(synthetic_only=True,role=role)) for role in c.native.ROLES}
    p.verify=lambda:None;p.check_state=lambda state:None
    binding=put(tmp_path/'binding.json',dict(synthetic_only=True))
    event=p.append(dict(kind='install',latch_before=False,latch_after=False,asset_refs=[],step=150000))
    p.installed=[dict(step=150000,binding=binding,event=event)]
    active=SimpleNamespace(providers={'regularized':p},owners={'regularized':p.owner_ref},sources=c.source_files())
    monkeypatch.setattr(c.targets,'_active_owner',lambda a,v,pool:c.require(a is active and pool is p.observed,'fake closed owner'))
    state=SimpleNamespace(joint_steps=162000,model='fake-model',revision=0)
    monkeypatch.setattr(c.native,'training_fingerprint',lambda s:str((s.joint_steps,s.revision)))
    monkeypatch.setattr(c.native,'role_hashes',lambda m:{r:'fake-'+r for r in c.native.ROLES})
    def export(state,path,**kw):return put(path,dict(synthetic_only=True,revision=state.revision))
    monkeypatch.setattr(c.native,'export_current_actor',export)
    calls=[]
    def target(a,variant,pool,ref,*,stage_id):
        calls.append((variant,stage_id,ref));folder=tmp_path/'targets'/stage_id
        provenance=put(folder/'materialization.json',dict(synthetic_only=True))
        return put(folder/'result.json',dict(long_materialization_provenance=provenance))
    monkeypatch.setattr(c.targets,'relabel_current_NN',target)
    monkeypatch.setattr(c.targets,'read_materialization',lambda ref,**kw:dict(inputs=[ref],provenance_ref=c.bound(ref)['long_materialization_provenance']))
    gates={r:True for r in c.native.ROLES}
    class Evidence:
        def __init__(self,**kw):self.__dict__.update(kw)
        def payload(self):return dict(report=self.report,epoch_id=self.epoch_id,
            feedback_enabled=getattr(self,'feedback_enabled',None),provider_receipt_ref=getattr(self,'provider_receipt_ref',None))
        def verify(self,**kw):
            assert kw['epoch_id']==self.epoch_id;return self
    monkeypatch.setattr(c.native,'ProviderEvidence',Evidence)
    def evidence(program,pool,target,*,epoch_id):
        role=c.bound(program)['role'];return Evidence(report=dict(gate=dict(passed=gates[role])),
            epoch_id=epoch_id,asset_refs=(program,target),program_ref=program)
    monkeypatch.setattr(c.native.pools,'build_observed_evidence',evidence)
    return SimpleNamespace(p=p,active=active,state=state,calls=calls,gates=gates,callback=c.LongEvidenceCallback(active,p))


def test_composed_callback_keeps_actual_native_instance_and_records_exact_native_epoch_and_storage_chain(setup):
    s=setup;out=s.callback(s.state,'synthetic-epoch-1')
    assert type(s.p) is c.native.CurrentNNEvidenceProvider and s.callback.native is s.p
    assert s.p.call_count==1 and s.p.used_epochs=={'synthetic-epoch-1'}
    assert s.calls[0][:2]==('regularized','call-000000')
    assert all(e.feedback_enabled for e in out.values()) and not s.p.latch
    event=c.bound(s.p.events[-1]);ledger=c.bound(s.callback.events[-1])
    assert event['long_storage_callback_started']==s.callback.started
    assert event['long_materialization_provenance']==ledger['provenance']
    assert event['optimizer_steps']==0 and event['accepted_gradient_updates']=='recorded_only_by_actual_PPO_update_core'
    snapshot=s.callback.snapshot(s.state)
    assert snapshot['payload_sha256']==c.digest(snapshot['payload']) and snapshot['payload']['call_count']==1
    assert snapshot['payload']['safe_snapshot_not_resume_authority']


def test_native_fidelity_failure_latches_and_cannot_reopen_without_scheduled_install(setup):
    s=setup;s.gates['robot_1']=False
    out=s.callback(s.state,'fail-one');reason=s.p.reason
    assert s.p.latch and reason and all(not e.feedback_enabled for e in out.values())
    s.gates['robot_1']=True;s.state.revision+=1
    out=s.callback(s.state,'all-later-agree')
    assert s.p.reason==reason and s.p.latch and all(not e.feedback_enabled for e in out.values())
    assert s.p.call_count==2 and c.bound(s.p.events[-1])['prior_failure']==reason


@pytest.mark.parametrize('change',['duplicate_epoch','before_window','missed_refit','bypassed_callback','source_changed'])
def test_callback_rejects_control_or_source_violation_before_further_materialization(setup,monkeypatch,change):
    s=setup
    if change=='duplicate_epoch':s.callback(s.state,'once');epoch='once'
    else:epoch='new'
    before=len(s.calls)
    if change=='before_window':s.state.joint_steps=159999
    elif change=='missed_refit':s.state.joint_steps=168001
    elif change=='bypassed_callback':s.p.call_count+=1
    elif change=='source_changed':monkeypatch.setattr(c,'source_files',lambda:{'changed':'fake'})
    with pytest.raises(RuntimeError):s.callback(s.state,epoch)
    assert len(s.calls)==before and s.p.failed and s.callback.failed


def test_bare_native_substitute_and_old_owner_cannot_construct_callback(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='real native'):c.LongEvidenceCallback(None,object())
    p=c.native.CurrentNNEvidenceProvider.__new__(c.native.CurrentNNEvidenceProvider);p.observed=None
    def closed(*a):raise RuntimeError('synthetic old calibration owner denied')
    monkeypatch.setattr(c.targets,'_active_owner',closed)
    with pytest.raises(RuntimeError,match='old calibration'):c.LongEvidenceCallback(None,p)
    assert not list(tmp_path.iterdir())


def test_evidence_materialization_cannot_mutate_NN_or_optimizer_state_before_gradient(setup,monkeypatch):
    original=c.targets.relabel_current_NN;s=setup
    def mutate(*args,**kw):
        out=original(*args,**kw);s.state.revision+=1;return out
    monkeypatch.setattr(c.targets,'relabel_current_NN',mutate)
    with pytest.raises(RuntimeError,match='changed NN/critic/optimizer/RNG'):
        s.callback(s.state,'mutated-state')
    assert s.p.failed and s.callback.failed and not s.callback.events
    assert s.p.call_count==1 and s.p.used_epochs=={'mutated-state'}
