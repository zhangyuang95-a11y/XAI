"""Software-only checks; no new audited pool/private target certificate."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import importlib.util
import json
import builtins
import symtable
import pytest
from backend.training import warehouse_v642_accelerated_recovery_evidence as a


def test_repeated_references_hash_actual_file_once(tmp_path,monkeypatch):
    path=tmp_path/'file.json';path.write_text('{}');ref=a.pointer(path);calls=[]
    original=a.original._immutable
    def checked(value):calls.append(value);return original(value)
    monkeypatch.setattr(a.original,'_immutable',checked)
    assert a.check_unique_assets([ref,ref,ref])==(ref,)
    assert calls==[ref]


def test_conflicting_path_sha_rejected_before_any_hash(tmp_path,monkeypatch):
    path=tmp_path/'file.json';path.write_text('{}');ref=a.pointer(path)
    monkeypatch.setattr(a.original,'_immutable',lambda value:pytest.fail('conflict read a payload'))
    with pytest.raises(RuntimeError,match='conflicting SHA'):
        a.check_unique_assets([ref,dict(path=ref['path'],sha256='0'*64)])


def test_distinct_files_with_identical_bytes_are_both_checked(tmp_path,monkeypatch):
    refs=[]
    for name in ('first','second'):
        path=tmp_path/(name+'.json');path.write_text('{}');refs.append(a.pointer(path))
    calls=[];original=a.original._immutable
    monkeypatch.setattr(a.original,'_immutable',lambda ref:(calls.append(ref),original(ref)))
    assert len(a.check_unique_assets(refs*3))==2 and len(calls)==2


def test_missing_changed_and_symlink_assets_refused(tmp_path):
    path=tmp_path/'file.json';path.write_text('{}');ref=a.pointer(path)
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError):a.check_unique_assets([ref])
    path.unlink()
    with pytest.raises(RuntimeError):a.check_unique_assets([ref])
    target=tmp_path/'target.json';target.write_text('{}');path.symlink_to(target)
    with pytest.raises(RuntimeError):a.check_unique_assets([ref])


@pytest.fixture
def mock_control(tmp_path,monkeypatch):
    """Named low-level doubles, not genuine NN/ObservedPool authority.

    The actual original restore_prefix and event/latch logic run; only the
    original test suite's explicit tiny NN/pool doubles replace numerics.
    No original private target certificate is constructed by this fixture.
    """
    spec=importlib.util.spec_from_file_location('accelerated_recovery_MOCK_control',
        Path(__file__).with_name('test_warehouse_v642_recovery_evidence.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    c=helper.case.__wrapped__(tmp_path,monkeypatch)
    manifests=[]
    for kind in ('original407','recovery-v2'):
        manifests.append(helper.put(tmp_path/(kind+'-MOCK-manifest.json'),dict(MOCK_only=True,kind=kind)))
    files=tuple(a.pointer(p) for p in sorted(tmp_path.rglob('*')) if p.is_file())
    roots=tuple(r['path'] for r in files)
    for r in files:Path(r['path']).chmod(0o444)
    monkeypatch.setattr(a,'source_files',lambda:{'MOCK-new-source':'a'*64})
    monkeypatch.setattr(a.pools,'ObservedPool',SimpleNamespace)
    def prefixes(refs):
        assert tuple(refs)==tuple(manifests)
        return tuple(manifests),roots,files,()
    monkeypatch.setattr(a,'checked_prefix_manifests',prefixes)
    monkeypatch.setattr(a.AcceleratedRecoveryEvidenceProvider,'_completed_native_evidence',
        a.original.RecoveryEvidenceProvider._completed_native_evidence)
    monkeypatch.setattr(a.AcceleratedRecoveryEvidenceProvider,'_build_observed_evidence',
        lambda self,program,target,*,epoch_id:a.pools.build_observed_evidence(program,self.observed,target,epoch_id=epoch_id))
    c.manifests=manifests;c.files=files
    c.restore_accelerated=lambda:a.AcceleratedRecoveryEvidenceProvider.restore_original_prefix(c.snapshot,c.state,
        reference_directory=tmp_path/'MOCK-reference',owner_ref=c.owner,observed=c.observed,
        output=c.root/'recovery'/'accelerated',recovery_claim_ref=c.claim,
        prefix_manifest_refs=manifests,config=c.config)
    return c


def test_real_original_factory_and_new_object_preserve_cached_prefix_and_partial_latch(mock_control):
    c=mock_control;before={r['path']:Path(r['path']).read_bytes() for r in c.files}
    p=c.restore_accelerated()
    assert type(p) is a.AcceleratedRecoveryEvidenceProvider
    assert isinstance(p,a.original.RecoveryEvidenceProvider) and p.observed is c.observed
    assert p.cached_call_range==(0,1) and p.partial['sequence_call']==2
    first=p(c.state,'epoch-0');assert all(e.feedback_enabled for e in first.values())
    second=p(c.state,'epoch-1');assert p.latch and all(not e.feedback_enabled for e in second.values())
    third=p(c.state,'epoch-2');assert p.latch and all(not e.feedback_enabled for e in third.values())
    assert p.call_count==3 and len(p.audit_refs)==3
    assert a.bound(p.events[-1])['recovery_target_mode']=='interrupted_native_targets_replayed'
    assert all(Path(name).read_bytes()==value for name,value in before.items())
    snapshot=p.snapshot(c.state);payload=snapshot['payload']
    assert payload['version']==a.original.VERSION and payload['accelerated_wrapper']['version']==a.VERSION
    assert snapshot['payload_sha256']==a.digest(payload) and payload['latch']


@pytest.mark.parametrize('fault',['epoch','NN','source','old_bytes','latch','extra_event'])
def test_new_adapter_rejects_prefix_state_and_source_changes(mock_control,monkeypatch,fault):
    c=mock_control;p=c.restore_accelerated()
    if fault=='NN':c.state.model.hashes['robot_2']='f'*64
    elif fault=='source':monkeypatch.setattr(a,'source_files',lambda:{'MOCK-changed':'b'*64})
    elif fault=='old_bytes':
        path=Path(c.files[0]['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
    elif fault=='latch':p.latch=True
    elif fault=='extra_event':
        path=p.output/'events'/'999999.json';path.parent.mkdir(exist_ok=True);path.write_text('{}')
    with pytest.raises((RuntimeError,ValueError)):
        p(c.state,'wrong' if fault=='epoch' else 'epoch-0')
    assert p.failed and p.call_count==0


def test_deduplicated_control_preserves_every_native_control_field_and_asset_read(mock_control,monkeypatch):
    c=mock_control;sources=a.native.source_files();events=c.events
    old=a.original._control(events,owner_ref=c.owner,sources=sources,config=c.config,check_assets=True)
    new=a.checked_event_control(events,owner_ref=c.owner,sources=sources,config=c.config)
    assert a.prepared._typed_sha(old)==a.prepared._typed_sha(new)
    path=Path(a.bound(events[-1])['asset_refs'][0]['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(RuntimeError):a.checked_event_control(events,owner_ref=c.owner,sources=sources,config=c.config)


def test_cached_payload_type_or_value_change_refuses_original_update(mock_control,monkeypatch):
    c=mock_control;p=c.restore_accelerated();previous=p._completed_native_evidence
    def changed(*args,**kw):
        evidence,accounting=previous(*args,**kw)
        evidence['robot_1'].report['changed']=True
        return evidence,accounting
    monkeypatch.setattr(p,'_completed_native_evidence',changed)
    with pytest.raises(RuntimeError,match='report differs'):p(c.state,'epoch-0')
    assert p.failed and p.call_count==0


def test_unknown_prefix_manifest_schema_rejected_before_native_restore(tmp_path):
    refs=[]
    for kind in ('original407','recovery-v2'):
        path=tmp_path/(kind+'.json');path.write_text(json.dumps(dict(version='unknown',kind=kind)))
        path.chmod(0o444);refs.append(a.pointer(path))
    with pytest.raises(RuntimeError,match='unknown or incomplete'):a.checked_prefix_manifests(refs)


def test_no_original_class_or_global_builder_is_replaced():
    assert a.original.RecoveryEvidenceProvider is not a.AcceleratedRecoveryEvidenceProvider
    assert a.pools.build_observed_evidence.__module__=='backend.training.warehouse_v642_observed_pool'


def test_all_accelerated_method_global_names_resolve_before_any_real_work():
    table=symtable.symtable(Path(a.__file__).read_text(),a.__file__,'exec')
    missing=[]
    def visit(node):
        for symbol in node.get_symbols():
            name=symbol.get_name()
            if symbol.is_referenced() and symbol.is_global() and name not in vars(a) and not hasattr(builtins,name):
                missing.append((node.get_name(),name))
        for child in node.get_children():visit(child)
    # Imports local to a function are local symbols, not global false positives.
    for child in table.get_children():visit(child)
    assert not missing


def test_explicit_canonical_writer_keeps_live_owner_guard_and_provenance_without_private_certificate(tmp_path,monkeypatch):
    from backend.training import warehouse_v642_canonical_relabel as canonical
    p=object.__new__(a.AcceleratedRecoveryEvidenceProvider)
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));path.chmod(0o444);return a.pointer(path)
    p.owner_ref=put('MOCK-owner.json',dict(MOCK_only=True));p.recovery_claim_ref=put('MOCK-claim.json',dict(MOCK_only=True))
    p.observed=object();p.acceleration_sources={'MOCK-source':'a'*64};p._pending_target_provenance={}
    guard=dict(owner=p.owner_ref,claim=p.recovery_claim_ref,source_files=p.acceleration_sources,executor='MOCK-only')
    p._source_guard=lambda:deepcopy(guard)
    monkeypatch.setattr(a,'source_files',lambda:dict(p.acceleration_sources))
    provenance=put('MOCK-provenance.json',dict(MOCK_only=True));result=put('MOCK-result.json',dict(canonical_producer_provenance=provenance))
    calls=[]
    def writer(observed,actor,output,*,epoch_id,sequence,source_guard):
        assert observed is p.observed and epoch_id=='MOCK-epoch' and sequence==59
        assert source_guard()==guard;calls.append('writer');return result
    def audit(observed,ref,*,epoch_id,sequence,source_guard):
        assert ref==result and source_guard()==guard;calls.append('provenance')
        return dict(provenance_ref=provenance,native_scalar_audit_still_required=True)
    monkeypatch.setattr(canonical,'relabel_current_NN',writer);monkeypatch.setattr(canonical,'read_provenance',audit)
    assert p._target_package(put('MOCK-actor.json',{}),epoch_id='MOCK-epoch',sequence=59)==result
    assert calls==['writer','provenance'] and p._pending_target_provenance[59]==provenance


def test_full_source_closure_pins_original_six_and_new_canonical_prepared_metrics_files():
    sources=a.source_files()
    for name in (*a.FROZEN_RECOVERY_FILES,*a.FILES,'backend/training/warehouse_v642_prepared_program.py',
        'backend/training/warehouse_v642_static_metrics.py','backend/training/warehouse_v642_canonical_relabel.py'):
        assert sources[name]==a.native.file_sha(a.ROOT/name)
