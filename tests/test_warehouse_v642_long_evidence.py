"""Named synthetic provider/NN doubles; no real candidate or environment."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import importlib.util
import json
import numpy as np
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
    monkeypatch.setattr(c.PreparedObservedEvidenceBuilder,'build',lambda self,*args,**kw:evidence(*args,**kw))
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


def test_raw_cannot_construct_any_prepared_builder_or_materialize_targets(setup,monkeypatch):
    s=setup;s.p.variant='raw'
    def forbidden(*a,**kw):pytest.fail('raw constructed program preparation')
    monkeypatch.setattr(c,'PreparedObservedEvidenceBuilder',forbidden)
    with pytest.raises(RuntimeError,match='regularized native provider'):c.LongEvidenceCallback(s.active,s.p)
    assert not s.calls


@pytest.fixture
def named_low_level_doubles(tmp_path,monkeypatch):
    """Explicit MOCK provenance, never an audited ObservedPool/private token.

    Only the real native pool *type names* and checked_targets are patched in
    this test.  No original ObservedPool or ReplayedTargets is manufactured.
    Scientific end-to-end authority is deliberately outside this fixture.
    """
    spec=importlib.util.spec_from_file_location('long_preparation_software_fixture',
        Path(__file__).with_name('test_warehouse_v642_prepared_program.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    native_program,parts,_,file=helper.prepared.__wrapped__(tmp_path)
    role=native_program.role;target_actor=put(tmp_path/'MOCK-not-a-NN.json',dict(mock_only=True))
    manifest_parts={}
    for split,a in parts.items():
        arrays=tmp_path/(split+'.npz')
        with arrays.open('xb') as handle:np.savez_compressed(handle,**{k:a[k] for k in ('x','m','p','sampled')})
        manifest_parts[split]=dict(arrays=c.pointer(arrays),rows=put(tmp_path/(split+'-rows.json'),a['rows']))
    pool=put(tmp_path/'MOCK-pool.json',dict(parts=manifest_parts))
    target_ref=put(tmp_path/'MOCK-target-package.json',dict(mock_only=True))
    target=dict(target_actor=target_actor,pools={role:pool});refs=[target_ref]
    class MockCheckedTargets:
        def __init__(self):self.parts={role:deepcopy(parts)};self.role_hashes={role:'c'*64};self.checks=0
        def verify(self,observed):self.checks+=1;return self
    class MockObservedProvenance:
        def __init__(self):
            self._replayed_target_cache=MockCheckedTargets()
            self.source_ref=put(tmp_path/'MOCK-source.json',dict(mock_only=True))
            self.audit_sha256='d'*64;self.checks=0
        def verify(self):self.checks+=1;return self
    observed=MockObservedProvenance();checks=[]
    def mock_checked(pool,ref):
        assert pool is observed and ref==target_ref;checks.append(ref);return deepcopy(target),deepcopy(refs)
    monkeypatch.setattr(c.native.pools,'ObservedPool',MockObservedProvenance)
    monkeypatch.setattr(c.native.pools.target_cache,'ReplayedTargets',MockCheckedTargets)
    monkeypatch.setattr(c.native.pools,'checked_targets',mock_checked)
    return SimpleNamespace(program=native_program,program_ref=c.pointer(file),parts=parts,observed=observed,
        target_ref=target_ref,target=target,checks=checks,helper=helper)


def test_mock_provenance_builder_preserves_original_native_types_payload_refs_and_pool_targets(named_low_level_doubles):
    s=named_low_level_doubles;builder=c.PreparedObservedEvidenceBuilder()
    old=c.native.pools.build_observed_evidence(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-epoch')
    actual=builder.build(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-epoch')
    assert type(actual) is type(old) is c.native.pools.ObservedOnlineEvidence
    s.helper.strict_equal(actual.payload(),old.payload());assert actual.asset_refs==old.asset_refs
    assert actual.envelope_digest==old.envelope_digest
    for a in s.parts.values():
        args=dict(current_role_sha256=actual.target_role_sha256,epoch_id=actual.epoch_id)
        left=old.loss_targets(a['x'],a['m'],**args);right=actual.loss_targets(a['x'],a['m'],**args)
        assert set(left)==set(right)
        for key in left:
            assert left[key].dtype==right[key].dtype and left[key].tobytes()==right[key].tobytes()
    assert len(s.checks)==2 and s.observed._replayed_target_cache.checks>=2


def test_mock_builder_reuses_only_program_and_recomputes_both_qualifications_every_epoch(named_low_level_doubles,monkeypatch):
    s=named_low_level_doubles;builder=c.PreparedObservedEvidenceBuilder();prepared_calls=[];report_calls=[]
    original=c.prepared.PreparedProgram.from_reference;report=c.native.pools.online.current_report
    def prepare(ref,**kw):prepared_calls.append(ref);return original(ref,**kw)
    def recompute(program,parts):report_calls.append(parts);return report(program,parts)
    monkeypatch.setattr(c.prepared.PreparedProgram,'from_reference',prepare)
    monkeypatch.setattr(c.native.pools.online,'current_report',recompute)
    first=builder.build(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-first')
    for a in s.observed._replayed_target_cache.parts[s.program.role].values():
        a['p'][:]=np.float32(.2);a['sampled'][:]=0
    second=builder.build(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-second')
    assert len(prepared_calls)==1 and len(report_calls)==2 and first.program is second.program
    assert first.report!=second.report and first.report['feedback_leaves']!=second.report['feedback_leaves']
    old=c.native.pools.build_observed_evidence(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-second')
    s.helper.strict_equal(second.payload(),old.payload())
    assert builder.bindings()[0]['program']==s.program_ref


@pytest.mark.parametrize('fault',['not_native_pool','not_native_cache','program_mutation','source_changed','epoch'])
def test_builder_rejects_provenance_or_cached_program_fault_before_feedback(named_low_level_doubles,monkeypatch,fault):
    s=named_low_level_doubles;builder=c.PreparedObservedEvidenceBuilder()
    if fault=='program_mutation':
        builder.build(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-first')
        builder.program(s.program_ref).leaves[0]['bias'][0]+=.1
    elif fault=='not_native_cache':s.observed._replayed_target_cache=object()
    elif fault=='source_changed':monkeypatch.setattr(c,'source_files',lambda:{'MOCK-changed-source':'0'*64})
    observed=object() if fault=='not_native_pool' else s.observed
    with pytest.raises(RuntimeError):
        builder.build(s.program_ref,observed,s.target_ref,epoch_id='' if fault=='epoch' else 'MOCK-next')


def test_complete_long_source_closure_explicitly_binds_prepared_native_and_gradient_implementations():
    files=c.source_files()
    for name in ('backend/training/warehouse_v642_prepared_program.py',
        'backend/training/warehouse_v642_program.py','backend/training/warehouse_v642_observed_pool.py',
        'backend/training/warehouse_v630_target_cache.py','backend/training/warehouse_v629_training_feedback.py',
        'backend/training/warehouse_v642_long_evidence.py'):
        assert files[name]==c.file_sha(c.ROOT/name)


def test_native_target_rejection_occurs_before_any_program_preparation(named_low_level_doubles,monkeypatch):
    s=named_low_level_doubles;builder=c.PreparedObservedEvidenceBuilder();calls=[]
    def rejected(*a,**kw):calls.append('native-checked-targets');raise RuntimeError('MOCK target replay failed')
    def forbidden(*a,**kw):pytest.fail('program prepared before native target admission')
    monkeypatch.setattr(c.native.pools,'checked_targets',rejected)
    monkeypatch.setattr(c.prepared.PreparedProgram,'from_reference',forbidden)
    with pytest.raises(RuntimeError,match='target replay failed'):
        builder.build(s.program_ref,s.observed,s.target_ref,epoch_id='MOCK-rejected')
    assert calls==['native-checked-targets'] and not builder._programs


def test_original_native_reader_class_and_global_builder_are_never_replaced():
    from backend.training.warehouse_v642_program import Program
    from backend.training.warehouse_v642_online_evidence import OnlineEvidence
    assert c.native.pools.evidence.Program is Program
    assert c.native.pools.evidence.OnlineEvidence is OnlineEvidence
    assert c.native.pools.build_observed_evidence.__module__=='backend.training.warehouse_v642_observed_pool'
