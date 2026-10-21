"""Detached checkpoint equivalence, fake A/B, temporary soft-WAIT NN only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,random
import numpy as np
import pytest
import torch
from experiments.warehouse import v629_checkpoint_readback as r

ACTUAL_READINESS=r.owner.seeds.partitions.admission.load_readiness


def helper():
    spec=importlib.util.spec_from_file_location('readback_owner_fixture',Path(__file__).with_name('test_warehouse_v629_calibration_owner.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444);return r.pointer(path)


def test_actual_closed_gate_precedes_checkpoint_RNG_and_model_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(r.owner.seeds.partitions.admission,'load_readiness',ACTUAL_READINESS)
    monkeypatch.setattr(r.owner.seeds.partitions.admission.v,'OUTPUT',tmp_path/'missing-winner')
    def forbidden(*args,**kwargs):raise AssertionError('missing A/B cannot read/restore a checkpoint or create RNG')
    monkeypatch.setattr(r,'bound',forbidden);monkeypatch.setattr(r,'load_checkpoint',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        r.load(reference_directory=tmp_path/'reference',checkpoint_ref=None,expected_owner_ref=None)
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    folder=tmp_path_factory.mktemp('detached-readback-not-scientific');h=helper()
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as mp:
            original=h.export_actor
            def softer(model,*args,**kwargs):
                with torch.no_grad():
                    for actor in model.actors.values():actor.tree.leaf_logits[0,4]=6.
                return original(model,*args,**kwargs)
            mp.setattr(h,'export_actor',softer)
            preparation=h.preparation.__wrapped__(folder,mp)
            with pytest.raises(h.ProbeFinished):
                with h.enter(preparation) as owner:
                    state=owner.initialize('raw')
                    while state.joint_steps<5024:owner.collect_rollout('raw');owner.update('raw')
                    assert state.actor_updates==16 and state.critic_updates==80
                    before=owner.checkpoint('raw');first_role_hashes=r.owner.role_hashes(state.model)
                    trace=owner.collect_rollout('raw');owner.update('raw');after=owner.checkpoint('raw')
                    assert state.actor_updates==32 and state.critic_updates==96
                    assert r.owner.role_hashes(state.model)!=first_role_hashes
                    update=owner.updates['raw'][-1];owner_ref=owner.owners['raw']
                    raise h.ProbeFinished()
            yield dict(folder=folder,reference=preparation[1],before=before,after=after,trace=trace,
                update=update,owner=owner_ref,physical_training_steps=6048)
    finally:torch.set_num_threads(prior)


def load(recorded):
    return r.load(reference_directory=recorded['reference'],checkpoint_ref=recorded['before'],expected_owner_ref=recorded['owner'])


def test_restore_has_full_Actor_Adam_critic_beta_risk_RNG_and_live_environment(recorded):
    np_before=np.random.get_state();torch_before=torch.get_rng_state().clone();py_before=random.getstate()
    x=load(recorded)
    assert x.state.resumed_training_state and x.state.joint_steps==5024 and x.state.actor_updates==16 and x.state.critic_updates==80
    assert x.state.actor_optimizer.state and x.state.critic_optimizer.state
    assert all(e is not None for e in x.reader.environments)
    assert x.core.snapshot()==x.value['core'] and not x.consumed
    assert torch.equal(torch_before,torch.get_rng_state()) and py_before==random.getstate()
    np_after=np.random.get_state();assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]


def test_next_actual_recorded_rollout_and_Adam_updates_equal_continuous_run(recorded):
    x=load(recorded)
    result=r.replay_next_raw_update(x,recorded['trace'],expected_update_ref=recorded['update'],expected_checkpoint_ref=recorded['after'])
    assert result['full_state_and_update_bitwise_equal'] and result['physical_joint_frames']==1024
    assert result['new_environment_resets']==result['original_files_written']==result['actual_candidate_updates']==0
    assert not result['continuation_authority'] and x.state.actor_updates==32 and x.state.critic_updates==96
    with pytest.raises(RuntimeError,match='only one detached raw update'):
        r.replay_next_raw_update(x,recorded['trace'],expected_update_ref=recorded['update'],expected_checkpoint_ref=recorded['after'])


@pytest.mark.parametrize('change',['reset_cursor','private_RNG','reserved_steps','other_owner','update_receipts','partial_program_source'])
def test_checkpoint_consumption_aliases_fail_before_any_optimization(recorded,tmp_path,change):
    value=deepcopy(r.bound(recorded['before']))
    if change=='reset_cursor':value['reset_cursors'][0]-=1
    if change=='private_RNG':value['environments'][0]['payload']['episode_counter']+=1
    if change=='reserved_steps':value['reserved_joint_steps']+=8
    if change=='other_owner':value['owner']={'path':'another-owner','sha256':'0'*64}
    if change=='update_receipts':value['updates']=value['updates'][:-1]
    if change=='partial_program_source':value['program_source_cursor']=1
    ref=write(tmp_path/'altered-owner-state.json',value)
    with pytest.raises((RuntimeError,ValueError)):
        r.load(reference_directory=recorded['reference'],checkpoint_ref=ref,expected_owner_ref=recorded['owner'])


def test_next_trace_cannot_skip_a_consumed_batch(recorded,tmp_path):
    x=load(recorded);trace=deepcopy(r.bound(recorded['trace']));trace['start_joint_step']+=8
    ref=write(tmp_path/'skipped-batch.json',trace)
    with pytest.raises(RuntimeError,match='another owner/prefix'):
        r.replay_next_raw_update(x,ref,expected_update_ref=recorded['update'],expected_checkpoint_ref=recorded['after'])
    assert x.state.actor_updates==16 and x.core.update_sequence==5


def test_false_expected_update_is_detected_without_changing_original_files(recorded,tmp_path):
    x=load(recorded);value=deepcopy(r.bound(recorded['update']));value['shutdown_episodes']+=1
    ref=write(tmp_path/'false-update.json',value);original=r.pointer(recorded['update']['path'])
    with pytest.raises(RuntimeError,match='differs from continuous'):
        r.replay_next_raw_update(x,recorded['trace'],expected_update_ref=ref,expected_checkpoint_ref=recorded['after'])
    assert r.pointer(original['path'])==original


@pytest.mark.parametrize('next_refit',[False,True])
def test_saved_provider_prefix_rederives_failure_latch_and_only_scheduled_refit_reopens(tmp_path,monkeypatch,next_refit):
    # Pure event-chain control test. No claim of full regularized NN recovery.
    monkeypatch.setattr(r,'training_fingerprint',lambda state:'fixture-full-state')
    state=SimpleNamespace(initialization_seed=7);sources=r.provider_sources();config=r.asdict(r.PPOConfig())
    owner_ref=write(tmp_path/'owner.json',{'scope':'synthetic-provider-prefix-only'})
    header=dict(version=r.PROVIDER_VERSION,source_files=sources,owner=owner_ref,pair_seed=7,variant='regularized',
        config=config,training_authority=False)
    started=write(tmp_path/'started.json',header);events=[];installed=[];epochs=[];calls=0;latch=False;reason=None
    programs={'robot_1':{'fixture':1},'robot_2':{'fixture':2}}
    for kind,passed in [('install',True),('validation',False),('validation',True)]+([('install',True)] if next_refit else []):
        before=latch
        event=dict(version=r.PROVIDER_VERSION,training_authority=False,owner=owner_ref,source_sha256=r.digest(sources),
            sequence=len(events),previous=events[-1] if events else None,latch_before=before,
            passed_by_role={'robot_1':passed,'robot_2':True},kind=kind,programs=programs,asset_refs=[])
        if kind=='install':
            event.update(step=r.PPOConfig().extraction_steps[len(installed)],binding={'fixture':len(installed)})
            latch=not passed
        else:
            event.update(epoch_id=f'epoch-{calls}',sequence_call=calls,prior_failure=reason)
            epochs.append(event['epoch_id']);calls+=1;latch=latch or not passed
        event['latch_after']=latch;ref=write(tmp_path/f'event-{len(events)}.json',event);events.append(ref)
        if kind=='install':
            installed.append(dict(step=event['step'],binding=event['binding'],event=ref));reason=ref if latch else None
        elif not before and latch:reason=ref
    payload=dict(version=r.PROVIDER_VERSION,source_files=sources,owner=owner_ref,pair_seed=7,variant='regularized',
        config=config,training_authority=False,full_training_state_sha256='fixture-full-state',started=started,
        events=events,installed=installed,used_epochs=sorted(epochs),call_count=calls,latch=latch,reason=reason,programs=programs)
    snapshot=dict(payload=payload,payload_sha256=r.digest(payload))
    result=r.provider_prefix(snapshot,state,owner_ref)
    assert result['latch']==(not next_refit) and not result['executable_provider_returned'] and not result['training_authority']
    altered=deepcopy(payload);altered['latch']=not latch
    with pytest.raises(RuntimeError,match='not derived'):
        r.provider_prefix(dict(payload=altered,payload_sha256=r.digest(altered)),state,owner_ref)
    altered=deepcopy(payload);altered['pair_seed']=8
    with pytest.raises(RuntimeError,match='source, owner or NN differs'):
        r.provider_prefix(dict(payload=altered,payload_sha256=r.digest(altered)),state,owner_ref)
