"""Synthetic public traces only: no environment constructor/reset or PPO."""
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_rollout_replay as r
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor,export_actor
from backend.training.warehouse_v622_actor_training import initialize_from_reference,save_checkpoint,load_checkpoint
from env.warehouse.domain import WarehouseState,AgentState,DeliveryTask
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
from env.warehouse.public_observation_v6 import public_pre_action_snapshot


@pytest.fixture(autouse=True)
def threads():
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(prior)


@pytest.fixture
def prepared(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for role in r.ROLES:
            actor=model.actors[role]
            for p in actor.parameters():p.zero_()
            actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
            actor.tree.left_refs.zero_();actor.tree.right_refs.zero_();actor.tree.leaf_logits[0,4]=1000.
            actor.gate[-1].bias.fill_(1.)
    ref=export_actor(model,tmp_path/'reference.npz',pair_seed=0,variant='raw',candidate_id='synthetic-rollout-reader-only')
    state=initialize_from_reference(ref['path'],ref['sha256'],ref['role_sha256'],initialization_seed=0,actor_lr=3e-5)
    (tmp_path/'owner.json').write_text(json.dumps(dict(scope='synthetic-test-only-no-training-authority',stream='paired-zero',variant='raw')))
    return state,r.pointer(ref['path']),tmp_path


def public(frame,physical_id,*,shutdown=False):
    points=layout.passable_positions
    state=WarehouseState(episode_id=physical_id,frame=frame,
        agents=[AgentState(role,layout.robot_start_positions[i],battery=60.,heading='WAIT',
            active=not(shutdown and i==0),recent_positions=(layout.robot_start_positions[i],)) for i,role in enumerate(r.ROLES)],
        tasks=[DeliveryTask('first',points[1],points[-1]),DeliveryTask('second',points[2],points[-2])])
    return public_pre_action_snapshot(state)


def trace(prepared,start=0,*,shutdown_frame=None):
    state,actor_ref,folder=prepared;n=8*r.PPOConfig().next_collection_length(start);events=[]
    for index in range(n):
        slot=index%8;t=start//8+index//8;physical_id=t//120+1;frame=t%120
        # Optional real terminal event at the last vector avoids needing a
        # second synthetic reset branch in this fixture.
        shutdown=shutdown_frame is not None and t==shutdown_frame and slot==0
        before=public(frame,physical_id);after=public(frame+1,physical_id,shutdown=shutdown)
        case=dict(episode_id=f'slot-{slot}-episode-{physical_id}',scenario_seed=10000+physical_id*8+slot,
            task=slot%3+1,action_seed=20000+physical_id*8+slot,slot=slot,partition='train')
        events.append(dict(slot=slot,case=case,pre_state_public=before,post_state_public=after,
            proposed_actions={role:'WAIT' for role in r.ROLES},action_override_agents=[],rewards={role:.1 for role in r.ROLES},
            terminated=shutdown,truncated=frame==119 and not shutdown,shutdowns=['robot_1'] if shutdown else []))
    return dict(schema=r.TRACE_VERSION,partition='train',start_joint_step=start,end_joint_step=start+n,
        config=asdict(r.PPOConfig()),owner=r.pointer(folder/'owner.json'),actor=actor_ref,events=events,rollout_state=dict(actor_role_sha256=r.role_hashes(state.model),
            critic_sha256=r.critic_initialization_sha256(state.model),actor_updates=state.actor_updates,critic_updates=state.critic_updates))


def assemble(reader,prepared,value,name='trace.json'):
    state,_,folder=prepared;path=folder/name;path.write_text(json.dumps(value));state.joint_steps=value['end_joint_step']
    return reader.assemble(r.pointer(path),state)


def test_full_vector_rollout_reconstructs_public_nn_actions_and_only_complete_costs(prepared):
    reader=r.PublicRolloutReplay(r.pointer(prepared[2]/'owner.json'));value=trace(prepared);state,_,_=prepared
    before=r.role_hashes(state.model);batch,audit=assemble(reader,prepared,value)
    assert len(batch['rows'])==1024 and len(batch['complete_cost_rows'])==8*120
    assert audit['NN_action_checks']==2048 and audit['pending_frames']==[8]*8 and audit['complete_episodes']==8
    assert audit['partial_episodes_labeled_negative']==0 and not audit['training_authorized']
    assert r.role_hashes(state.model)==before and state.actor_updates==state.critic_updates==0
    assert all(row['old_logp'].tolist()==[0.,0.] for row in batch['rows'])
    assert all(row['cost_return']==0. and row['episode_length']==120 for row in batch['complete_cost_rows'])


def test_pending_episode_keeps_rollout_time_critic_across_updates_and_counts_real_shutdown(prepared):
    reader=r.PublicRolloutReplay(r.pointer(prepared[2]/'owner.json'));assemble(reader,prepared,trace(prepared))
    old=[row['cost_value'] for row in reader.pending[0]]
    state,_,_=prepared
    with torch.no_grad():
        for parameter in state.model.cost_critic.parameters():parameter.zero_()
        state.model.cost_critic[-1].bias.fill_(2.)
    second=trace(prepared,1024,shutdown_frame=255)
    batch,audit=assemble(reader,prepared,second,'second.json')
    episodes={row['episode_id'] for row in batch['complete_cost_rows']}
    assert len(episodes)==9 and audit['pending_frames'][0]==0 and audit['pending_frames'][1:]==[16]*7
    carried=[row for row in batch['complete_cost_rows'] if row['episode_id']=='slot-0-episode-2']
    assert [row['cost_value'] for row in carried[:8]]==old
    assert carried[8]['cost_value']!=old[0] and carried[8]['cost_value']>0.8
    terminal=[row for row in batch['complete_cost_rows'] if row['episode_id']=='slot-0-episode-3']
    assert len(terminal)==16 and all(row['cost_return']==1. for row in terminal)
    assert terminal[-1]['shutdown'] and terminal[-1]['terminated']


@pytest.mark.parametrize('change',['action','override','post_action','public_chain','slot_order','non_train','cost_event','critic','actor','length'])
def test_bad_physical_receipts_fail_without_optimizer_or_silent_retry(prepared,change):
    value=trace(prepared);event=value['events'][8]
    if change=='action':event['proposed_actions']['robot_1']='UP'
    if change=='override':event['action_override_agents']=['robot_1']
    if change=='post_action':event['post_state_public']['agents'][0]['last_action']='UP'
    if change=='public_chain':event['pre_state_public']['agents'][0]['battery']-=1
    if change=='slot_order':event['slot']=1
    if change=='non_train':event['case']['partition']='validation'
    if change=='cost_event':event['terminated']=True
    if change=='critic':value['rollout_state']['critic_sha256']='f'*64
    if change=='actor':value['rollout_state']['actor_role_sha256']['robot_1']='f'*64
    if change=='length':value['events']=value['events'][:16];value['end_joint_step']=16
    reader=r.PublicRolloutReplay(r.pointer(prepared[2]/'owner.json'));state,_,_=prepared
    with pytest.raises((RuntimeError,ValueError)):assemble(reader,prepared,value)
    assert reader.failed and state.actor_updates==state.critic_updates==0
    with pytest.raises(RuntimeError,match='cannot silently resume'):reader.assemble({},state)
    with pytest.raises(RuntimeError,match='failed or partially'):reader.snapshot(prepared[0])


def test_completed_reader_and_full_training_state_restore_next_batch_exactly(prepared):
    state,actor_ref,folder=prepared;reader=r.PublicRolloutReplay(r.pointer(prepared[2]/'owner.json'))
    assemble(reader,prepared,trace(prepared));snapshot=reader.snapshot(prepared[0])
    checkpoint=folder/'full.pt';save_checkpoint(state,checkpoint)
    resumed=load_checkpoint(checkpoint,r.file_sha(checkpoint))
    restored=r.PublicRolloutReplay.restore(snapshot,resumed,owner_ref=r.pointer(folder/'owner.json'))
    value=trace(prepared,1024);path=folder/'second.json';path.write_text(json.dumps(value))
    state.joint_steps=resumed.joint_steps=value['end_joint_step']
    original,a=reader.assemble(r.pointer(path),state);copy,b=restored.assemble(r.pointer(path),resumed)
    assert a==b and r.arrays_to_lists(original)==r.arrays_to_lists(copy) and reader.snapshot(prepared[0])==restored.snapshot(resumed)
    assert all(p.grad is None for p in state.model.parameters())


def test_restore_rejects_fresh_state_and_changed_physical_source(prepared):
    state,_,folder=prepared;reader=r.PublicRolloutReplay(r.pointer(prepared[2]/'owner.json'));assemble(reader,prepared,trace(prepared));value=reader.snapshot(prepared[0])
    with pytest.raises(RuntimeError,match='full completed training state'):r.PublicRolloutReplay.restore(value,state,owner_ref=r.pointer(folder/'owner.json'))
    checkpoint=folder/'full.pt';save_checkpoint(state,checkpoint);resumed=load_checkpoint(checkpoint,r.file_sha(checkpoint))
    (folder/'trace.json').write_text('{}')
    with pytest.raises(RuntimeError,match='prior physical trace'):r.PublicRolloutReplay.restore(value,resumed,owner_ref=r.pointer(folder/'owner.json'))


@pytest.mark.parametrize('change',['actor','critic','beta','optimizer','initialization','owner'])
def test_same_counter_other_training_state_or_stream_cannot_restore_pending_history(prepared,change):
    state,_,folder=prepared;owner_ref=r.pointer(folder/'owner.json');reader=r.PublicRolloutReplay(owner_ref)
    assemble(reader,prepared,trace(prepared));snapshot=reader.snapshot(state)
    if change=='actor':
        with torch.no_grad():state.model.actors['robot_1'].gate[-1].bias.add_(.1)
    if change=='critic':
        with torch.no_grad():state.model.reward_critic[-1].bias.add_(.1)
    if change=='beta':state.beta['robot_1']=2.
    if change=='optimizer':state.actor_optimizer.param_groups[0]['lr']=1.e-5
    if change=='initialization':state.initialization_seed=123
    if change=='owner':
        other=folder/'other-owner.json';other.write_text(json.dumps(dict(stream='another-real-pair')));owner_ref=r.pointer(other)
    checkpoint=folder/'mixed.pt';save_checkpoint(state,checkpoint);resumed=load_checkpoint(checkpoint,r.file_sha(checkpoint))
    assert resumed.joint_steps==snapshot['payload']['processed_joint_steps']
    with pytest.raises(RuntimeError,match='full completed training state'):
        r.PublicRolloutReplay.restore(snapshot,resumed,owner_ref=owner_ref)


@pytest.mark.parametrize('asset',['owner','actor'])
def test_source_replacement_during_frame_replay_invalidates_entire_batch(prepared,monkeypatch,asset):
    state,actor_ref,folder=prepared;reader=r.PublicRolloutReplay(r.pointer(folder/'owner.json'))
    value=trace(prepared);original=r.critic_values;changed=False
    def replaced(model,obs):
        nonlocal changed
        result=original(model,obs)
        if not changed:
            path=folder/'owner.json' if asset=='owner' else actor_ref['path']
            from pathlib import Path
            Path(path).write_bytes(b'concurrent replacement');changed=True
        return result
    monkeypatch.setattr(r,'critic_values',replaced)
    with pytest.raises(RuntimeError,match='changed during independent replay'):assemble(reader,prepared,value)
    assert reader.failed and state.actor_updates==state.critic_updates==0


def test_real_numeric_core_consumes_two_assembled_batches_without_losing_cost_history(prepared):
    from backend.training.warehouse_v629_ppo_update import PPOUpdateCore
    from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA,ReferenceTrainingBuffer
    state,_,folder=prepared;reader=r.PublicRolloutReplay(r.pointer(folder/'owner.json'))
    batch,audit=assemble(reader,prepared,trace(prepared))
    buffer_path=folder/'initial-reference-training.npz'
    np.savez_compressed(buffer_path,encoded=np.array([row['encoded'] for row in batch['rows'][:32]]),
        masks=np.array([row['mask'] for row in batch['rows'][:32]]),episode_ids=np.array(['synthetic-reference']*32),
        metadata_json=np.array(json.dumps(dict(schema=BUFFER_SCHEMA,split='train',reference_sha256=state.reference_sha256))))
    buffer=ReferenceTrainingBuffer.load(buffer_path,r.file_sha(buffer_path),state.reference_sha256)
    core=PPOUpdateCore(state,buffer,variant='raw',rng=np.random.default_rng(0))
    first=core.update_batch(batch,batch_sha256=r.file_sha(folder/'trace.json'))
    assert first['complete_episodes']==8 and first['risk']['accepted_complete_episodes']==8
    assert state.actor_updates==0 and state.critic_updates==16
    pending_values=[row['cost_value'] for row in reader.pending[0]]
    snapshot=reader.snapshot(state);checkpoint=folder/'after-first-ppo.pt';save_checkpoint(state,checkpoint)
    resumed=load_checkpoint(checkpoint,r.file_sha(checkpoint))
    restored=r.PublicRolloutReplay.restore(snapshot,resumed,owner_ref=r.pointer(folder/'owner.json'))
    second,_=assemble(reader,prepared,trace(prepared,1024),'second.json')
    resumed.joint_steps=state.joint_steps
    restored_second,_=restored.assemble(r.pointer(folder/'second.json'),resumed)
    assert r.arrays_to_lists(restored_second)==r.arrays_to_lists(second)
    carried=[row for row in second['complete_cost_rows'] if row['episode_id']=='slot-0-episode-2']
    assert [row['cost_value'] for row in carried[:8]]==pending_values
    result=core.update_batch(second,batch_sha256=r.file_sha(folder/'second.json'))
    assert result['risk']['accepted_complete_episodes']==8 and len(core.risk.completed)==16
    assert state.actor_updates==0 and state.critic_updates==32 and core.processed_joint_steps==2048
    assert core.accepted_program_updates=={role:0 for role in r.ROLES}
