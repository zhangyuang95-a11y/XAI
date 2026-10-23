"""Recovery control/file tests: no environment, NN optimizer or new seeds."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

import pytest

from backend.training import warehouse_v642_recovery_evidence as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    r.write_new(path, value)
    return r.pointer(path)


class FakeBase:
    def __init__(self, **values):
        self.__dict__.update(values)

    def payload(self):
        return dict(version=r.native.EVIDENCE_VERSION, epoch_id=self.epoch_id,
                    anchor_program=self.program_ref, current_target_actor=self.target_actor_ref,
                    report=self.report, asset_manifest_sha256=r.digest(self.asset_refs),
                    provider_receipt=self.provider_receipt_ref,
                    provider_feedback_enabled=self.feedback_enabled)

    def verify(self, *, current_role_sha256, epoch_id):
        assert self.target_role_sha256 == current_role_sha256
        assert self.epoch_id == epoch_id
        event=r.bound(self.provider_receipt_ref)
        assert event['reports'][self.role] == r.digest(self.report)
        assert self.feedback_enabled is (not event['latch_after'])
        return self


@pytest.fixture
def case(tmp_path, monkeypatch):
    native_root=tmp_path/'provider'; target_root=tmp_path/'targets'
    original=native_root/'original'; original.mkdir(parents=True)
    owner=put(tmp_path/'owner.json', dict(owner='synthetic-no-training'))
    claim=put(tmp_path/'claim.json', dict(explicit_interruption_recovery=True))
    sources={'native-source':'native-source-hash'}
    recovery_sources={'new-source':'new-source-hash'}
    state=SimpleNamespace(joint_steps=16500, resumed_training_state=True,
                          initialization_seed=123, model=SimpleNamespace(hashes={role:'a'*64 for role in r.ROLES}))
    observed=SimpleNamespace(source_ref=put(tmp_path/'observed.json',dict(synthetic=True)),
                             audit_sha256='b'*64, refs=(), verify=lambda:None)
    ready=SimpleNamespace(payload_sha256='c'*64, verify=lambda:None)
    config=r.native.PPOConfig()
    monkeypatch.setattr(r.native,'PROVIDER_ROOT',native_root)
    monkeypatch.setattr(r.pools,'TARGET_ROOT',target_root)
    monkeypatch.setattr(r.native,'source_files',lambda:sources)
    monkeypatch.setattr(r,'source_files',lambda:recovery_sources)
    monkeypatch.setattr(r.native.admission,'load_readiness',lambda _:ready)
    monkeypatch.setattr(r.native,'training_fingerprint',lambda s:r.digest(dict(step=s.joint_steps,hashes=s.model.hashes)))
    monkeypatch.setattr(r.native,'role_hashes',lambda model:dict(model.hashes))
    monkeypatch.setattr(r.native.CurrentNNEvidenceProvider,'check_state',
                        lambda self,s:r.require(s.initialization_seed==self.pair_seed,'wrong initialization seed'))
    monkeypatch.setattr(r.native,'ProviderEvidence',FakeBase)
    def load(path, *, expected_sha256):
        assert r.pointer(path)['sha256']==expected_sha256
        return SimpleNamespace(model_version=r.MODEL_VERSION, role_sha256=r.bound(r.pointer(path))['role_sha256'])
    monkeypatch.setattr(r,'FrozenWarehouseV6Actor',SimpleNamespace(load=load))
    programs={role:put(tmp_path/(role+'-program.json'),dict(role=role)) for role in r.ROLES}
    binding=put(tmp_path/'binding.json',dict(synthetic=True))
    def base(role, actor, targets, epoch):
        target=r.bound(targets)
        return FakeBase(role=role, report=dict(gate=dict(passed=target['passed'][role])),
                        asset_refs=(programs[role],actor,targets), program_ref=programs[role],
                        target_actor_ref=actor, epoch_id=epoch,
                        target_role_sha256=r.bound(actor)['role_sha256'][role], envelope_digest='')
    build_calls=[]
    def build(program, observed, targets, *, epoch_id):
        role=r.bound(program)['role']; build_calls.append((epoch_id,role))
        return base(role,r.bound(targets)['target_actor'],targets,epoch_id)
    monkeypatch.setattr(r.pools,'build_observed_evidence',build)
    cached_calls=[]
    def rebuild(self,state,event_ref,epoch_id):
        event=r.bound(event_ref); cached_calls.append(event['sequence_call'])
        return ({role:build(self.programs[role],self.observed,event['target_package'],epoch_id=epoch_id)
                 for role in r.ROLES},dict(completed_native_target_facts_reused=True,
                                          original_memory_certificate_issued=False))
    monkeypatch.setattr(r.RecoveryEvidenceProvider,'_completed_native_evidence',rebuild)
    header=dict(version=r.native.VERSION, source_files=sources, owner=owner, pair_seed=123,
                variant='regularized', config=asdict(config), readiness_sha256=ready.payload_sha256,
                training_authority=False)
    started=put(original/'started.json',header)
    install=dict(version=r.native.VERSION,owner=owner,sequence=0,previous=None,
                 source_sha256=r.digest(sources),training_authority=False,
                 kind='install',step=15000,binding=binding,observed_source=observed.source_ref,
                 observed_audit_sha256=observed.audit_sha256,programs=programs,
                 asset_refs=[binding],passed_by_role={role:True for role in r.ROLES},
                 latch_before=False,latch_after=False)
    first=put(original/'events/000000.json',install)
    control=dict(installed=[dict(step=15000,binding=binding,event=first)], used_epochs=[],
                 call_count=0,latch=False,reason=None,programs=programs)
    payload=dict(version=r.native.VERSION,source_files=sources,owner=owner,started=started,
                 pair_seed=123,variant='regularized',config=asdict(config),output=str(original),
                 full_training_state_sha256=r.native.training_fingerprint(state),events=[first],
                 observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
                 training_authority=False,**control)
    snapshot=dict(payload=payload,payload_sha256=r.digest(payload))
    previous=first; latch=False; reason=None
    events=[first]
    for sequence in range(3):
        actor=put(original/'calls'/f'{sequence:06d}'/'current-NN.npz',dict(role_sha256=state.model.hashes))
        passed={role:(sequence!=1 or role!='robot_2') for role in r.ROLES}
        targets=put(target_root/r.digest(owner)/f'call-{sequence:06d}'/'result.json',
                    dict(target_actor=actor,passed=passed))
        put(Path(targets['path']).with_name('completed.json'),dict(result_sha256=targets['sha256']))
        if sequence==2:
            break  # Native call 2 interrupted after its complete target package.
        evidence={role:base(role,actor,targets,f'epoch-{sequence}') for role in r.ROLES}
        new_latch=latch or not all(passed.values())
        event=dict(version=r.native.VERSION,owner=owner,sequence=sequence+1,previous=previous,
                   source_sha256=r.digest(sources),training_authority=False,kind='validation',
                   epoch_id=f'epoch-{sequence}',step=16500,sequence_call=sequence,target_actor=actor,
                   target_package=targets,programs=programs,reports={role:r.digest(e.report) for role,e in evidence.items()},
                   passed_by_role=passed,asset_refs=[binding],latch_before=latch,latch_after=new_latch,
                   prior_failure=reason)
        ref=put(original/'events'/f'{sequence+1:06d}.json',event)
        if not latch and new_latch:reason=ref
        latch=new_latch
        for role,e in evidence.items():
            args={**e.__dict__,'asset_refs':tuple([*e.asset_refs,ref,binding,owner]),
                  'provider_receipt_ref':ref,'feedback_enabled':not latch,'envelope_digest':''}
            obj=FakeBase(**args)
            put(original/'calls'/f'{sequence:06d}'/(role+'-evidence.json'),obj.payload())
        previous=ref; events.append(ref)
    def restore(value=snapshot):
        return r.RecoveryEvidenceProvider.restore_prefix(value,state,reference_directory=tmp_path/'unused',
            owner_ref=owner,observed=observed,output=native_root/'recovery'/'one',recovery_claim_ref=claim)
    return SimpleNamespace(state=state,observed=observed,snapshot=snapshot,events=events,
                           original=original,restore=restore,build_calls=build_calls,root=native_root,
                           claim=claim,programs=programs,config=config,owner=owner)


def test_completed_calls_rebuild_exact_envelopes_without_writing_native_files(case):
    before={str(p):p.read_bytes() for p in case.original.rglob('*') if p.is_file()}
    provider=case.restore()
    first=provider(case.state,'epoch-0')
    assert all(e.feedback_enabled for e in first.values())
    assert provider.events==case.events[:2] and len(case.build_calls)==2
    failed=provider(case.state,'epoch-1')
    assert all(not e.feedback_enabled for e in failed.values())
    assert provider.call_count==2 and provider.latch and len(case.build_calls)==4
    assert {str(p):p.read_bytes() for p in case.original.rglob('*') if p.is_file()}==before
    assert not (provider.output/'calls').exists()
    assert len(provider.audit_refs)==2
    assert all(r.bound(ref)['replay_is_not_new_or_independent_data'] for ref in provider.audit_refs)


def test_partial_complete_targets_are_rebuilt_in_isolated_tail_and_do_not_reopen_latch(case,monkeypatch):
    provider=case.restore(); provider(case.state,'epoch-0'); provider(case.state,'epoch-1')
    def forbidden(*args,**kwargs):pytest.fail('partial targets must use the original full-NN replay path')
    monkeypatch.setattr(provider,'_completed_native_evidence',forbidden)
    recovered=provider(case.state,'epoch-2')
    assert all(e.report['gate']['passed'] and not e.feedback_enabled for e in recovered.values())
    assert provider.latch and provider.call_count==3
    event=r.bound(provider.events[-1])
    assert event['recovery_target_mode']=='interrupted_native_targets_replayed'
    assert Path(event['target_actor']['path']).is_relative_to(case.original)
    assert not (case.original/'calls/000002/robot_1-evidence.json').exists()
    assert (provider.output/'calls/000002/robot_1-evidence.json').exists()
    provider.verify()
    snap=provider.snapshot(case.state)
    assert snap['payload']['latch'] and snap['payload']['call_count']==3


def test_wrong_epoch_refused_before_target_replay(case):
    provider=case.restore()
    with pytest.raises(RuntimeError,match='epoch, step or failure state'):
        provider(case.state,'wrong-epoch')
    assert provider.failed and not case.build_calls


def test_wrong_current_nn_refused_before_report_rebuild(case):
    provider=case.restore();case.state.model.hashes['robot_2']='d'*64
    with pytest.raises(RuntimeError,match='current NN differs'):
        provider(case.state,'epoch-0')
    assert provider.failed and not case.build_calls


def test_changed_checkpoint_latch_does_not_restore(case):
    bad=deepcopy(case.snapshot);bad['payload']['latch']=True
    bad['payload_sha256']=r.digest(bad['payload'])
    with pytest.raises(RuntimeError,match='event-derived latch'):
        case.restore(bad)
    assert not (case.root/'recovery').exists()


def test_original_evidence_mutation_is_rejected(case):
    provider=case.restore()
    path=case.original/'calls/000001/robot_2-evidence.json'
    value=json.loads(path.read_text());value['provider_feedback_enabled']=True
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='original provider directory changed'):
        provider.verify()


def test_same_epoch_cannot_be_consumed_twice(case):
    provider=case.restore();provider(case.state,'epoch-0')
    with pytest.raises(RuntimeError,match='already consumed'):
        provider(case.state,'epoch-0')
    assert provider.call_count==1


def test_report_changed_on_recomputation_is_rejected(case,monkeypatch):
    provider=case.restore();original=provider._completed_native_evidence
    def changed(*args,**kwargs):
        evidence,accounting=original(*args,**kwargs)
        evidence['robot_2'].report['changed']=True
        return evidence,accounting
    monkeypatch.setattr(provider,'_completed_native_evidence',changed)
    with pytest.raises(RuntimeError,match='recomputed recovery report differs'):
        provider(case.state,'epoch-0')
    assert provider.failed and provider.call_count==0


def test_non_complete_partial_target_package_cannot_restore(case):
    partial=r.pools.TARGET_ROOT/r.digest(case.owner)/'call-000002'/'completed.json'
    partial.unlink()
    with pytest.raises((RuntimeError,FileNotFoundError)):
        case.restore()
    assert not (case.root/'recovery').exists()


def test_scheduled_native_install_is_the_only_latch_reset_then_new_calls_remain_isolated(case,monkeypatch,tmp_path):
    provider=case.restore()
    for index in range(3):
        provider(case.state,f'epoch-{index}')
    def forbidden(*args,**kwargs):pytest.fail('new validation cannot reuse a completed-native-call fact')
    monkeypatch.setattr(provider,'_completed_native_evidence',forbidden)
    assert provider.latch
    # Use the original installation implementation, with tiny synthetic
    # fixtures replacing only NN/data numerics, not its schedule/latch checks.
    assert r.RecoveryEvidenceProvider.install_extraction is r.native.CurrentNNEvidenceProvider.install_extraction
    case.state.joint_steps=16800
    actor=put(tmp_path/'next-current-NN.npz',dict(role_sha256=case.state.model.hashes))
    target=put(tmp_path/'next-targets.json',dict(target_actor=actor,passed={role:True for role in r.ROLES}))
    pool_sources={'observed':'same-native-observed-version'}
    monkeypatch.setattr(r.pools,'source_files',lambda:pool_sources)
    monkeypatch.setattr(r.pools,'checked_targets',lambda observed,ref:(r.bound(ref),[]))
    identity=dict(step=16800,owner=case.owner,variant='regularized',config=asdict(case.config),current_target_actor=actor)
    result=put(tmp_path/'extraction/result.json',dict(identity=identity,
        full_training_state_sha256=r.native.training_fingerprint(case.state),
        roles={role:dict(program=case.programs[role],current=dict(gate=dict(passed=True))) for role in r.ROLES}))
    binding=put(tmp_path/'extraction/observed-source-binding.json',dict(
        schema='warehouse-v642-observed-refit-source-binding.v1',source_files=pool_sources,owner=case.owner,
        observed_source=case.observed.source_ref,observed_audit_sha256=case.observed.audit_sha256,
        target_package=target,extraction=result))
    put(tmp_path/'extraction/observed-source-binding.completed.json',dict(binding_sha256=binding['sha256']))
    provider.install_extraction(binding,case.observed,case.state)
    assert not provider.latch and provider.reason is None
    assert [i['step'] for i in provider.installed]==[15000,16800]
    def export(state,path,**kwargs):
        return put(path,dict(role_sha256=state.model.hashes))
    def relabel(observed,actor,output):
        result=put(output/'result.json',dict(target_actor=actor,passed={role:True for role in r.ROLES}))
        put(output/'completed.json',dict(result_sha256=result['sha256']))
        return result
    monkeypatch.setattr(r.native,'export_current_actor',export)
    monkeypatch.setattr(r.pools,'relabel_current_NN',relabel)
    case.state.joint_steps=16900
    evidence=provider(case.state,'new-after-scheduled-refit')
    assert all(e.feedback_enabled for e in evidence.values())
    event=r.bound(provider.events[-1])
    assert event['recovery_target_mode']=='new_current_NN_validation'
    assert Path(event['target_actor']['path']).is_relative_to(provider.output)
    assert Path(event['target_package']['path']).is_relative_to(r.pools.TARGET_ROOT/'recovery')
    provider.verify()


def test_scheduled_refit_is_required_before_later_batch_validation(case):
    provider=case.restore()
    for index in range(3):
        provider(case.state,f'epoch-{index}')
    before=len(case.build_calls)
    case.state.joint_steps=16801
    with pytest.raises(RuntimeError,match='next scheduled extraction'):
        provider(case.state,'late-without-refit')
    assert len(case.build_calls)==before and provider.failed


def test_completed_fact_loader_replays_recorded_draws_and_report_without_forward_or_memory_certificate(tmp_path,monkeypatch):
    """Exercise the real new loader, using tiny arrays and a synthetic program."""
    def forbidden(*args,**kwargs):
        pytest.fail('completed-fact reconstruction must not mint a memory certificate or run an NN')
    monkeypatch.setattr(r.pools,'checked_targets',forbidden)
    monkeypatch.setattr(r.pools,'build_observed_evidence',forbidden)
    monkeypatch.setattr(r.pools.target_cache,'_seal_after_replay',forbidden)
    monkeypatch.setattr(r.pools.online,'runtime_probabilities',forbidden)
    monkeypatch.setattr(r.pools.online,'validate_rows',lambda x:None)
    monkeypatch.setattr(r.native,'role_hashes',lambda model:model.hashes)
    monkeypatch.setattr(r.pools,'source_files',lambda:{'synthetic-native':'same'})
    class Program:
        def __init__(self,encoded):
            self.encoded=encoded;self.role=encoded['role'];self.protection=encoded['protection']
            self.actor_role_sha256=encoded['target_actor_role_sha256']
        @classmethod
        def from_dict(cls,value,unused):return cls(value)
        def to_dict(self):return self.encoded
    monkeypatch.setattr(r.pools.evidence,'Program',Program)
    reports=[]
    def current_report(program,parts):
        reports.append(program.role)
        assert set(parts)==set(r.pools.online.SPLITS)
        assert all(a['sampled'].tolist()==[0] for a in parts.values())
        return dict(gate=dict(passed=True))
    monkeypatch.setattr(r.pools.online,'current_report',current_report)
    owner=put(tmp_path/'owner.json',dict(synthetic=True))
    actor=put(tmp_path/'actor.npz',dict(synthetic=True))
    source=put(tmp_path/'observed.json',dict(synthetic=True))
    parent=put(tmp_path/'parent.json',dict(synthetic=True))
    programs={role:put(tmp_path/(role+'-program.json'),dict(role=role,
        target_actor_role_sha256='a'*64,protection=dict(parent=parent))) for role in r.ROLES}
    parts={};actual={};refs=[]
    for i,split in enumerate(r.pools.online.SPLITS):
        mask=np.array([[1,0,0,0,1]],np.float32)
        x=np.zeros((1,r.pools.online.FEATURE_DIM),np.float32)
        x[:,r.pools.online.MASK_INDICES]=mask
        rows=[dict(episode='scene-'+split,base_seed=str(i),task=i%3+1,frame=0,action_seed=i,episode_id=1)]
        parts[split]=dict(x=np.stack((x,x),axis=1),m=np.stack((mask,mask),axis=1),
                          executed=np.zeros((1,2),np.int64),rows=rows)
        path=tmp_path/(split+'-actual.npz');np.savez_compressed(path,executed=parts[split]['executed'])
        actual[split]=r.pointer(path);refs.append(actual[split])
    pools={}
    for role in r.ROLES:
        role_parts={}
        for split,part in parts.items():
            path=tmp_path/(role+'-'+split+'.npz')
            np.savez_compressed(path,x=part['x'][:,0],m=part['m'][:,0],
                                p=np.array([[1,0,0,0,0]],np.float32),sampled=np.zeros(1,np.int64))
            rows=put(tmp_path/(role+'-'+split+'-rows.json'),part['rows'])
            role_parts[split]=dict(arrays=r.pointer(path),rows=rows);refs.extend(role_parts[split].values())
        pools[role]=put(tmp_path/(role+'-pool.json'),dict(schema=r.pools.online.MANIFEST_VERSION,
            scope='online_program_training_evidence',target_actor=actor,role=role,parts=role_parts))
        refs.append(pools[role])
    observed=SimpleNamespace(source_ref=source,audit_sha256='b'*64,observed_data_sha256='c'*64,
                             owner_ref=owner,parts=parts,verify=lambda:None)
    target=put(tmp_path/'targets/result.json',dict(schema=r.pools.TARGET_VERSION,
        observed_source=source,observed_audit_sha256=observed.audit_sha256,
        observed_data_sha256=observed.observed_data_sha256,owner=owner,source_files=r.pools.source_files(),
        target_actor=actor,pools=pools,actual_source_actions=actual,asset_refs=refs,
        counterfactual_difference_counts={role:{split:0 for split in parts} for role in r.ROLES},
        current_policy_trajectory=False,actual_actions_replaced=False,training_authority=False,
        counterfactual_draws_not_executed=True,actual_PPO_feedback_counts_from_real_rollouts_only=True,
        current_policy_task_quality_estimate=None,fresh_randomness_claimed=False,pilot_opened=False,
        uses_existing_recorded_seeds_for_counterfactual_numerical_replay=True,
        environment_resets=0,NN_updates=0,PPO_joint_steps=0))
    put(Path(target['path']).with_name('completed.json'),dict(result_sha256=target['sha256']))
    event=put(tmp_path/'event.json',dict(kind='validation',sequence_call=32,epoch_id='old-32',
                                     target_package=target,target_actor=actor))
    provider=r.RecoveryEvidenceProvider.__new__(r.RecoveryEvidenceProvider)
    provider.cached_call_range=(32,58);provider.original_events=[event]
    provider.observed=observed;provider.programs=programs
    state=SimpleNamespace(model=SimpleNamespace(hashes={role:'a'*64 for role in r.ROLES}))
    evidence,counts=provider._completed_native_evidence(state,event,'old-32')
    assert reports==list(r.ROLES) and set(evidence)==set(r.ROLES)
    assert counts['recorded_action_draws_replayed']==8
    assert counts['scalar_NN_forwards_recomputed']==0 and counts['scalar_NN_forwards_reused']==8
    assert counts['original_memory_certificate_issued'] is False
    # Changes to already recorded targets are rejected before report fitting.
    path=Path(refs[0]['path']);path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(RuntimeError,match='changed'):
        provider._completed_native_evidence(state,event,'old-32')
