"""Full synthetic NN PPO updates; no environment or scientific candidate."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_ppo_update as p
from backend.training import warehouse_v629_online_evidence as o
from backend.training.warehouse_v622_actor_training import export_current_actor
from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA


@pytest.fixture(autouse=True)
def threads():
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(prior)


@pytest.fixture
def prepared(tmp_path):
    spec=importlib.util.spec_from_file_location('v629_ppo_full_helpers',Path(__file__).with_name('test_warehouse_v629_training_feedback.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    state,x,masks,sampled,evidence,_=helper.paired.__wrapped__(tmp_path)
    for group in state.actor_optimizer.param_groups:group['lr']=3e-5
    file=tmp_path/'reference-training.npz'
    np.savez_compressed(file,encoded=x.numpy(),masks=masks.numpy(),episode_ids=np.array([f'reference-train-{i//10}' for i in range(len(x))]),
        metadata_json=np.array(json.dumps(dict(schema=BUFFER_SCHEMA,split='train',reference_sha256=state.reference_sha256))))
    buffer=p.ReferenceTrainingBuffer.load(file,p.file_sha(file),state.reference_sha256)
    rows=[]
    with torch.no_grad():
        logits=state.model.outputs_encoded(x[:32],masks[:32]);logp=p.masked_log_probabilities(logits,masks[:32])
        actions=logits.argmax(-1);chosen=logp.gather(-1,actions[...,None]).squeeze(-1)
    for i in range(32):
        rows.append(dict(obs=x[i,:,:p.OBSERVATION_DIM].numpy().copy(),encoded=x[i].numpy().copy(),mask=masks[i].numpy().copy(),
            actions=actions[i].numpy().copy(),old_logp=chosen[i].numpy().copy(),reward_advantage=np.array([i/32,-i/32],np.float32),
            reward_return=np.array([.2,-.3],np.float32),episode_id=f'current-{i//10}',frame=i%10,
            shutdown=False,terminated=False,truncated=False,cost_value=.3))
    return state,buffer,rows,evidence


def batch_for(prepared,*,variant='raw',start=4000):
    state,buffer,rows,evidence=prepared
    state.joint_steps=start
    core=p.PPOUpdateCore(state,buffer,variant=variant,rng=np.random.default_rng(0),processed_joint_steps=start)
    state.advance_joint_steps(len(rows))
    batch=dict(start_joint_step=start,end_joint_step=state.joint_steps,rows=rows,complete_cost_rows=[])
    return core,batch,evidence


def test_exact_warmup_updates_critics_only_with_no_false_safe_outcome(prepared):
    core,batch,_=batch_for(prepared,start=0)
    before=p.role_hashes(core.state.model);beta=dict(core.state.beta)
    result=core.update_batch(batch,batch_sha256='a'*64)
    assert core.state.actor_updates==0 and core.state.critic_updates==4
    assert p.role_hashes(core.state.model)==before and core.state.beta==beta
    assert result['complete_episodes']==0 and result['risk']['accepted_complete_episodes']==0
    assert result['partial_episodes_labeled_negative']==0
    assert result['accepted_positive_program_gradient_updates']=={r:0 for r in p.ROLES}


def test_actor_updates_start_only_after_exact_4000_boundary(prepared):
    core,batch,_=batch_for(prepared,start=3968)
    core.update_batch(batch,batch_sha256='a'*64)
    assert core.state.joint_steps==4000 and core.state.actor_updates==0
    rows=deepcopy(batch['rows'])
    for row in rows:row['episode_id']='next-'+row['episode_id']
    core.state.advance_joint_steps(32)
    core.update_batch(dict(start_joint_step=4000,end_joint_step=4032,rows=rows,complete_cost_rows=[]),batch_sha256='b'*64)
    assert core.state.actor_updates==4


def test_raw_full_ppo_changes_both_actors_and_never_loads_programs(prepared):
    core,batch,_=batch_for(prepared)
    before=p.role_hashes(core.state.model)
    def forbidden(*args):raise AssertionError('raw inspected programs')
    result=core.update_batch(batch,batch_sha256='a'*64,evidence_provider=forbidden)
    assert core.state.actor_updates==core.state.critic_updates==4
    assert all(p.role_hashes(core.state.model)[r]!=before[r] for r in p.ROLES)
    assert result['accepted_positive_program_gradient_updates']=={r:0 for r in p.ROLES}
    assert all(update['program_loss']==0 for epoch in result['epochs'] for update in epoch['updates'])
    with pytest.raises(RuntimeError,match='receipt already consumed'):core.update_batch(batch,batch_sha256='a'*64)
    assert core.failed


def test_raw_and_regularized_before_window_match_full_optimizer_updates(prepared):
    state,buffer,rows,_=prepared
    from backend.training.warehouse_v622_actor_training import initialize_from_reference
    paired=initialize_from_reference(state.reference_path,state.reference_sha256,state.reference.role_sha256,
        initialization_seed=state.initialization_seed,actor_lr=3e-5)
    cores=[]
    for current,variant in ((state,'raw'),(paired,'regularized')):
        current.joint_steps=4000
        core=p.PPOUpdateCore(current,buffer,variant=variant,rng=np.random.default_rng(0),processed_joint_steps=4000)
        current.advance_joint_steps(32)
        core.update_batch(dict(start_joint_step=4000,end_joint_step=4032,rows=rows,complete_cost_rows=[]),batch_sha256='b'*64)
        cores.append(core)
    for key,value in cores[0].state.model.state_dict().items():assert torch.equal(value,cores[1].state.model.state_dict()[key]),key
    assert cores[0].state.beta==cores[1].state.beta


def test_real_complete_shutdown_updates_risk_once_and_partial_episode_is_rejected(prepared):
    core,batch,_=batch_for(prepared)
    rows=batch['rows']
    for i,row in enumerate(rows):
        row['episode_id']='complete-shutdown' if i<2 else f'partial-{(i-2)//10}'
        row['frame']=i if i<2 else (i-2)%10
    rows[1].update(shutdown=True,terminated=True)
    batch['complete_cost_rows']=deepcopy(rows[:2])
    result=core.update_batch(batch,batch_sha256='a'*64)
    assert result['complete_episodes']==result['shutdown_episodes']==1
    assert result['risk']['accepted_complete_episodes']==result['risk']['window_shutdowns']==1
    assert len(core.risk.completed)==1  # not four duplicate epoch outcomes
    assert all(u['complete_cost_frames']==2 and u['complete_cost_episodes']==1 for e in result['epochs'] for u in e['updates'])
    partial=deepcopy(rows[:1]);partial[0].update(shutdown=False,terminated=False,truncated=False)
    with pytest.raises(ValueError,match='completed physical episode'):p.complete_cost_rows(partial)


@pytest.mark.parametrize('change',['duplicate_frame','fake_cost','wrong_terminal','wrong_steps','cross_warmup'])
def test_bad_physical_rollout_fails_before_any_optimizer(prepared,change):
    core,batch,_=batch_for(prepared)
    if change=='duplicate_frame':batch['rows'][1]['frame']=0
    if change=='fake_cost':
        row=deepcopy(batch['rows'][0]);row.update(terminated=True,shutdown=True,cost_return=0)
        batch['complete_cost_rows']=[row]
    if change=='wrong_terminal':batch['rows'][0].update(terminated=True,shutdown=False)
    if change=='wrong_steps':batch['end_joint_step']+=1
    if change=='cross_warmup':
        core.processed_joint_steps=3992;batch['start_joint_step']=3992;batch['end_joint_step']=4024;core.state.joint_steps=4024
    before=p.role_hashes(core.state.model)
    with pytest.raises((ValueError,RuntimeError)):core.update_batch(batch,batch_sha256='a'*64)
    assert core.state.actor_updates==core.state.critic_updates==0 and p.role_hashes(core.state.model)==before
    assert core.failed


@pytest.mark.parametrize('shutdown',[True,False])
def test_terminal_episode_cannot_be_omitted_from_risk_or_cost_training(prepared,shutdown):
    core,batch,_=batch_for(prepared)
    row=batch['rows'][0]
    row.update(episode_id='missing-complete-cost-history',frame=0 if shutdown else 119,
        shutdown=shutdown,terminated=shutdown,truncated=not shutdown)
    assert batch['complete_cost_rows']==[]
    with pytest.raises(RuntimeError,match='terminal episodes missing complete cost history'):
        core.update_batch(batch,batch_sha256='a'*64)
    assert core.state.actor_updates==core.state.critic_updates==0 and len(core.risk.completed)==0


def test_complete_cost_cannot_substitute_a_new_critic_baseline(prepared):
    core,batch,_=batch_for(prepared)
    for i,row in enumerate(batch['rows'][:2]):row.update(episode_id='complete-shutdown',frame=i)
    batch['rows'][1].update(terminated=True,shutdown=True)
    batch['complete_cost_rows']=deepcopy(batch['rows'][:2]);batch['complete_cost_rows'][1]['cost_value']=.99
    with pytest.raises(RuntimeError,match='baseline differs from rollout-time critic'):
        core.update_batch(batch,batch_sha256='a'*64)
    assert core.state.actor_updates==core.state.critic_updates==0 and len(core.risk.completed)==0


def test_stale_online_evidence_is_rejected_after_first_accepted_ppo_update(prepared):
    core,batch,evidence=batch_for(prepared,variant='regularized',start=17968)
    from dataclasses import replace
    def provider(state,epoch_id):
        out={}
        for role,old in evidence.items():
            current=replace(old,epoch_id=epoch_id,envelope_digest='')
            out[role]=replace(current,envelope_digest=o.digest(current.payload()))
        return out
    with pytest.raises(RuntimeError,match='expired after Actor update'):
        core.update_batch(batch,batch_sha256='a'*64,evidence_provider=provider)
    assert core.state.actor_updates==1 and core.state.critic_updates==1 and core.failed
    assert all(value==1 for value in core.accepted_program_updates.values())


def test_regularized_full_ppo_accepts_positive_feedback_for_both_roles_with_fresh_evidence(prepared,tmp_path):
    core,batch,initial=batch_for(prepared,variant='regularized',start=17968)
    calls=[]
    def provider(state,epoch_id):
        folder=tmp_path/f'update-{len(calls)}';folder.mkdir();calls.append(epoch_id)
        exported=export_current_actor(state,folder/'current.npz',pair_seed=0,variant='regularized',candidate_id='synthetic-ppo-only')
        ref=o.pointer(exported['path']);actor=o.FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])
        result={}
        for role,original in initial.items():
            manifest=deepcopy(o.bound(original.pool_ref));manifest['target_actor']=ref
            for split,source in manifest['parts'].items():
                with np.load(source['arrays']['path'],allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                rows=o.bound(source['rows']);a['p']=o.runtime_probabilities(actor,role,a['x'],a['m'])
                a['sampled']=np.array([np.random.default_rng(o.independent_agent_seed(base_seed=r['action_seed'],episode_id=r['episode_id'],
                    frame=r['frame'],agent_id=role)).choice(5,p=q) for r,q in zip(rows,a['p'],strict=True)],np.int64)
                file=folder/f'{role}-{split}.npz';np.savez_compressed(file,**a);source['arrays']=o.pointer(file)
            file=folder/(role+'-manifest.json');file.write_text(json.dumps(manifest))
            result[role]=o.build_evidence(original.program_ref,ref,o.pointer(file),epoch_id=epoch_id)
        return result
    result=core.update_batch(batch,batch_sha256='c'*64,evidence_provider=provider)
    assert len(calls)==len(set(calls))==4 and core.state.actor_updates==4
    assert all(result['accepted_positive_program_gradient_updates'][r]==4 for r in p.ROLES)
    assert all(u['program_loss']>0 and u['common_gradient_clip_factor']>0 for e in result['epochs'] for u in e['updates'])
    assert core.snapshot()['processed_joint_steps']==18000 and not core.failed
    assert all(n==4 for n in core.snapshot()['accepted_program_updates'].values())


def test_full_checkpoint_resume_next_ppo_update_is_bitwise_equivalent(prepared,tmp_path):
    from backend.training.warehouse_v622_actor_training import save_checkpoint,load_checkpoint
    core,batch,_=batch_for(prepared)
    core.update_batch(batch,batch_sha256='a'*64)
    core.state.extra_rng_states['v629_core']=core.snapshot()
    checkpoint=save_checkpoint(core.state,tmp_path/'checkpoint.pt')
    restored_state=load_checkpoint(checkpoint['path'],checkpoint['sha256'],restore_rng=False)
    restored=p.PPOUpdateCore.restore(restored_state,core.buffer,restored_state.extra_rng_states['v629_core'],variant='raw',rng=np.random.default_rng(999))
    rows=deepcopy(batch['rows'])
    with torch.no_grad():
        x=p.tensor([r['encoded'] for r in rows]);m=p.tensor([r['mask'] for r in rows]);logits=core.state.model.outputs_encoded(x,m)
        actions=logits.argmax(-1);old=p.masked_log_probabilities(logits,m).gather(-1,actions[...,None]).squeeze(-1).numpy()
    for i,row in enumerate(rows):row.update(episode_id='next-'+row['episode_id'],actions=actions[i].numpy().copy(),old_logp=old[i].copy())
    reports=[]
    for current in (core,restored):
        current.state.advance_joint_steps(len(rows))
        next_batch=dict(start_joint_step=4032,end_joint_step=4064,rows=rows,complete_cost_rows=[])
        reports.append(current.update_batch(next_batch,batch_sha256='b'*64))
    assert reports[0]==reports[1]
    for key,value in core.state.model.state_dict().items():assert torch.equal(value,restored.state.model.state_dict()[key]),key
    assert core.state.beta==restored.state.beta and core.snapshot()==restored.snapshot()
    assert core.state.actor_updates==restored.state.actor_updates==8


def test_resume_rejects_a_partial_or_mixed_training_state(prepared,tmp_path):
    from backend.training.warehouse_v622_actor_training import save_checkpoint,load_checkpoint
    core,batch,_=batch_for(prepared);core.update_batch(batch,batch_sha256='a'*64)
    value=core.snapshot()
    checkpoint=save_checkpoint(core.state,tmp_path/'checkpoint.pt')
    state=load_checkpoint(checkpoint['path'],checkpoint['sha256'],restore_rng=False)
    partial=deepcopy(value);partial['safe_batch_boundary']=False
    with pytest.raises(RuntimeError,match='completed full-training checkpoint'):
        p.PPOUpdateCore.restore(state,core.buffer,partial,variant='raw',rng=np.random.default_rng(0))
    mixed=deepcopy(value);mixed['actor_updates']+=1
    with pytest.raises(RuntimeError,match='training-state checkpoint differ'):
        p.PPOUpdateCore.restore(state,core.buffer,mixed,variant='raw',rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError,match='completed full-training checkpoint'):
        p.PPOUpdateCore.restore(state,core.buffer,value,variant='regularized',rng=np.random.default_rng(0))
