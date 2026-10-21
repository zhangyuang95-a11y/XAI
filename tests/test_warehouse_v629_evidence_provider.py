"""Synthetic provider wiring only; no candidate rollout or finite authority."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_evidence_provider as p
from backend.training.warehouse_v622_actor_training import save_checkpoint,load_checkpoint
from env.warehouse.public_observation_v6 import OBSERVATION_DIM


def helper(name):
    spec=importlib.util.spec_from_file_location('provider_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444);return p.pointer(path)


def test_real_closed_prerequisites_precede_provider_directory_or_export(tmp_path,monkeypatch):
    monkeypatch.setattr(p.admission.v,'OUTPUT',tmp_path/'absent')
    def forbidden(*args,**kwargs):raise AssertionError('missing A/B cannot export or draw RNG')
    monkeypatch.setattr(p,'export_current_actor',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        p.CurrentNNEvidenceProvider(reference_directory=tmp_path/'unused',owner_ref=None,output=tmp_path/'unused-provider',pair_seed=0)
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    h=helper('test_warehouse_v629_observed_pool');generator=h.recording.__wrapped__(tmp_path_factory);a=next(generator)
    folder=a['folder']/'provider-setup';folder.mkdir();program_helper=helper('test_warehouse_v629');first,_=program_helper.fixture(folder)
    from backend.training.warehouse_v628_program import ParentProgram
    from backend.training.warehouse_v629_program import from_parent
    parent=ParentProgram.from_dict(p.bound(first.protection['parent']),first.actor_role_sha256)
    parent.role='robot_2';parent.actor_role_sha256='b'*64;second=from_parent(parent,write(folder/'parent-two.json',parent.to_dict()))
    programs={x.role:x for x in (first,second)}
    refs={role:write(folder/(role+'-program.json'),value.to_dict()) for role,value in programs.items()}
    ready=SimpleNamespace(value=dict(actor=p.pointer(a['state'].reference_path),program_pair={role:dict(language=p.admission.language_lock(value))
        for role,value in programs.items()}),payload_sha256='a'*64,verify=lambda:None)
    bindings=[]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p.admission,'load_readiness',lambda unused:ready)
        mp.setattr(p.pools,'TARGET_ROOT',folder/'targets');mp.setattr(p.pools.refit,'EXTRACTION_ROOT',folder/'extractions')
        for step in (15000,16800):
            a['state'].joint_steps=step
            target=p.pools.relabel_current_NN(a['observed'],a['current'],p.pools.TARGET_ROOT/str(step))
            result,binding=p.pools.refit_observed_pair_at_boundary(reference_directory=folder/'unused',owner_ref=a['observed'].owner_ref,
                state=a['state'],observed=a['observed'],targets_ref=target,program_refs=refs,output=p.pools.refit.EXTRACTION_ROOT/str(step),
                variant='regularized',epoch_id='synthetic-refit')
            bindings.append(binding);refs={role:result['roles'][role]['program'] for role in p.ROLES}
    a['state'].joint_steps=0
    yield {**a,'bindings':bindings,'ready':ready}
    generator.close()


@pytest.fixture
def run(setup,tmp_path,monkeypatch):
    monkeypatch.setattr(p.admission,'load_readiness',lambda unused:setup['ready'])
    monkeypatch.setattr(p,'PROVIDER_ROOT',tmp_path/'providers');monkeypatch.setattr(p.pools,'TARGET_ROOT',tmp_path/'targets')
    provider=p.CurrentNNEvidenceProvider(reference_directory=tmp_path/'unused',owner_ref=setup['observed'].owner_ref,
        output=p.PROVIDER_ROOT/'one',pair_seed=0)
    return provider,deepcopy(setup['state']),setup


def install(run,index):
    provider,state,a=run;state.joint_steps=(15000,16800)[index]
    return provider.install_extraction(a['bindings'][index],a['observed'],state)


def test_current_NN_export_and_epoch_specific_evidence_invalidate_after_change(run):
    provider,state,_=run;install(run,0);state.joint_steps=16500
    before=p.training_fingerprint(state);first=provider(state,'epoch-0')
    assert p.training_fingerprint(state)==before and not provider.latch
    for role,e in first.items():
        assert e.feedback_enabled and e.payload()['version']==p.EVIDENCE_VERSION
        e.verify(current_role_sha256=p.role_hashes(state.model)[role],epoch_id='epoch-0')
    with torch.no_grad():state.model.actors['robot_1'].tree.leaf_logits[0,4].add_(.001)
    with pytest.raises(RuntimeError,match='expired after Actor update'):
        first['robot_1'].verify(current_role_sha256=p.role_hashes(state.model)['robot_1'],epoch_id='epoch-0')
    second=provider(state,'epoch-1');assert second['robot_1'].target_role_sha256!=first['robot_1'].target_role_sha256
    assert len(provider.events)==3 and provider.call_count==2


def test_fidelity_failure_stays_disabled_after_recovery_until_next_scheduled_refit(run):
    provider,state,a=run;install(run,0);saved={role:state.model.actors[role].tree.leaf_logits.detach().clone() for role in p.ROLES}
    state.joint_steps=16500
    with torch.no_grad():
        for role in p.ROLES:state.model.actors[role].tree.leaf_logits.zero_()
    failed=provider(state,'failed-NN');assert provider.latch and all(not e.feedback_enabled for e in failed.values())
    with torch.no_grad():
        for role in p.ROLES:state.model.actors[role].tree.leaf_logits.copy_(saved[role])
    state.joint_steps=16600;recovered=provider(state,'recovered-NN')
    assert all(e.report['gate']['passed'] and not e.feedback_enabled for e in recovered.values())
    data=a['observed'].parts['fit']
    for i,role in enumerate(p.ROLES):
        out=recovered[role].loss_targets(data['x'][:,i],data['m'][:,i],current_role_sha256=p.role_hashes(state.model)[role],epoch_id='recovered-NN')
        assert not out['qualified'].any()
    install(run,1);assert not provider.latch and provider.reason is None
    state.joint_steps=16900;fresh=provider(state,'after-registered-refit');assert all(e.feedback_enabled for e in fresh.values())


def test_batch_ending_at_boundary_uses_previous_program_but_later_batch_requires_refit(run):
    provider,state,_=run;install(run,0);state.joint_steps=16800
    assert all(e.feedback_enabled for e in provider(state,'ending-at-84').values())
    state.joint_steps=16808
    with pytest.raises(RuntimeError,match='next scheduled extraction'):provider(state,'after-84-without-refit')
    assert provider.failed


def test_pre_window_and_duplicate_epochs_are_rejected_before_an_extra_export(run):
    provider,state,_=run;install(run,0);state.joint_steps=16000
    with pytest.raises(RuntimeError,match='before feedback window'):provider(state,'at-80')
    assert not (provider.output/'calls').exists()


def test_duplicate_optimizer_epoch_cannot_be_reused(run):
    provider,state,_=run;install(run,0);state.joint_steps=16500;provider(state,'one-epoch')
    with pytest.raises(RuntimeError,match='already consumed'):provider(state,'one-epoch')
    assert provider.call_count==1 and provider.failed


def test_checkpoint_restore_retains_failure_latch_and_checks_event_history(run,tmp_path):
    provider,state,a=run;install(run,0);state.joint_steps=16500
    with torch.no_grad():
        for role in p.ROLES:state.model.actors[role].tree.leaf_logits.zero_()
    provider(state,'failure');snapshot=provider.snapshot(state)
    saved=save_checkpoint(state,tmp_path/'state.pt');resumed=load_checkpoint(saved['path'],saved['sha256'])
    restored=p.CurrentNNEvidenceProvider.restore(snapshot,resumed,reference_directory=tmp_path/'unused',owner_ref=provider.owner_ref,observed=a['observed'])
    assert restored.latch and restored.reason==provider.reason and restored.used_epochs==provider.used_epochs
    bad=deepcopy(snapshot);bad['payload']['latch']=False;bad['payload_sha256']=p.digest(bad['payload'])
    with pytest.raises(RuntimeError,match='event-derived latch'):
        p.CurrentNNEvidenceProvider.restore(bad,resumed,reference_directory=tmp_path/'unused',owner_ref=provider.owner_ref,observed=a['observed'])
    current=p.bound(provider.events[-1])['target_actor'];path=Path(current['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='another NN export|history asset changed'):restored.verify()


def test_full_ppo_core_gets_fresh_provider_evidence_for_every_accepted_update(run,tmp_path):
    from backend.training.warehouse_v629_ppo_update import PPOUpdateCore,ReferenceTrainingBuffer,masked_log_probabilities
    from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA
    provider,state,a=run;install(run,0);install(run,1);data=a['observed'].parts['fit'];n=32
    x=torch.as_tensor(data['x'][:n].copy());m=torch.as_tensor(data['m'][:n].copy())
    with torch.no_grad():
        logits=state.model.outputs_encoded(x,m);actions=logits.argmax(-1);lp=masked_log_probabilities(logits,m).gather(-1,actions[...,None]).squeeze(-1)
    rows=[dict(obs=x[i,:,:OBSERVATION_DIM].numpy(),
        encoded=x[i].numpy(),mask=m[i].numpy(),actions=actions[i].numpy(),old_logp=lp[i].numpy(),
        reward_advantage=np.array([i/32,-i/32],np.float32),reward_return=np.array([.2,-.3],np.float32),episode_id=f'pure-numerical-PPO-{i//10}',
        frame=i%10,shutdown=False,terminated=False,truncated=False,cost_value=.3) for i in range(n)]
    # Numerical integration only; these probes are not asserted to be actual
    # admitted PPO trajectories or independent scientific evaluation scenes.
    file=tmp_path/'reference.npz';np.savez_compressed(file,encoded=data['x'],masks=data['m'],episode_ids=np.array([f'reference-probe-{i//10}' for i in range(len(data['x']))]),
        metadata_json=np.array(json.dumps(dict(schema=BUFFER_SCHEMA,split='train',reference_sha256=state.reference_sha256))))
    buffer=ReferenceTrainingBuffer.load(file,p.file_sha(file),state.reference_sha256)
    state.joint_steps=17968;core=PPOUpdateCore(state,buffer,variant='regularized',rng=np.random.default_rng(0),processed_joint_steps=17968)
    state.advance_joint_steps(32);result=core.update_batch(dict(start_joint_step=17968,end_joint_step=18000,rows=rows,complete_cost_rows=[]),
        batch_sha256='a'*64,evidence_provider=provider)
    assert provider.call_count==state.actor_updates==state.critic_updates==4
    assert result['accepted_positive_program_gradient_updates']=={role:4 for role in p.ROLES}
    assert len(provider.used_epochs)==4 and not provider.failed


def test_in_memory_program_change_cannot_bypass_registered_extraction(run):
    provider,state,_=run;install(run,0)
    provider.programs['robot_1']={'path':'unregistered-program','sha256':'0'*64}
    with pytest.raises(RuntimeError,match='control state differs'):provider.verify()


def test_snapshot_prefix_cannot_silently_resume_after_later_provider_work(run,tmp_path):
    provider,state,a=run;install(run,0);state.joint_steps=16500;snapshot=provider.snapshot(state)
    saved=save_checkpoint(state,tmp_path/'before.pt')
    provider(state,'work-after-snapshot')
    resumed=load_checkpoint(saved['path'],saved['sha256'])
    with pytest.raises(RuntimeError,match='post-snapshot'):
        p.CurrentNNEvidenceProvider.restore(snapshot,resumed,reference_directory=tmp_path/'unused',owner_ref=provider.owner_ref,observed=a['observed'])
