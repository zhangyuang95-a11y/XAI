"""Explicit synthetic state fixtures only; no reset or scientific candidate run."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json,random
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_environment_replay as r
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor,export_actor
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v622_actor_training import initialize_from_reference,save_checkpoint,load_checkpoint
from backend.training.warehouse_v625_actor_training import critic_initialization_sha256
from env.warehouse.navigation import ACTIONS,MOVE_DELTAS,pickup_pairs,shortest_path_distance
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444);return r.pointer(path)


def synthetic_environment(case,physical_id):
    # No environment constructor/reset. Control RNG zero supplies a fully
    # explicit test state; it is not evidence about the declared scene seed.
    env=r.WarehouseMultiAgentEnv.__new__(r.WarehouseMultiAgentEnv)
    env.config=r.training_config(case['scenario_seed']);env.layout=layout;env._rng=random.Random(0);env._episode_counter=physical_id
    pickups=sorted({p for _,p in pickup_pairs(layout.layout_id)});endpoints=set();tasks=[]
    for pickup in pickups:
        if len(tasks)==2:break
        if pickup in endpoints:continue
        delivery=next(p for p in layout.passable_positions if p not in endpoints and p!=pickup
            and shortest_path_distance(pickup,p,layout.layout_id)>=4)
        endpoints.update((pickup,delivery));tasks.append(r.DeliveryTask(f'task_{len(tasks)+1}',pickup,delivery))
    state=r.WarehouseState(episode_id=physical_id,frame=0,
        agents=[r.AgentState(role,layout.robot_start_positions[i],heading='WAIT',recent_positions=(layout.robot_start_positions[i],))
            for i,role in enumerate(r.AGENTS)],tasks=tasks,next_task_index=3,participant_controlled_agent_id='robot_1')
    env.state=None;env.set_state(state);assert not env.validate_state(env.state)
    return env


def initialize(ref):
    return initialize_from_reference(ref['path'],ref['sha256'],ref['role_sha256'],initialization_seed=0,actor_lr=3e-5)


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    prior=torch.get_num_threads();torch.set_num_threads(1)
    folder=tmp_path_factory.mktemp('synthetic-physics-only')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for role in r.AGENTS:
            actor=model.actors[role]
            for p in actor.parameters():p.zero_()
            actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
            actor.tree.left_refs.zero_();actor.tree.right_refs.zero_();actor.tree.leaf_logits[0,4]=1000.;actor.gate[-1].bias.fill_(1.)
    ref=export_actor(model,folder/'reference.npz',pair_seed=0,variant='raw',candidate_id='synthetic-physics-audit-only')
    state=initialize(ref);frozen=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])
    cases=[dict(episode_id=f'slot-{slot}-episode-{episode}',scenario_seed=10000+episode*8+slot,
        task=slot%3+1,action_seed=20000+episode*8+slot,slot=slot,partition='train') for episode in (1,2,3) for slot in range(8)]
    inventory=write(folder/'inventory.json',dict(schema='warehouse-v629-finite-training-case-inventory.v1',cases=cases))
    owner=write(folder/'owner.json',dict(scope='synthetic-test-only-no-training-authority',train_case_inventory=inventory))
    environments=[None]*8;active=[None]*8;traces=[]
    for batch_index in range(2):
        events=[];resets={};ends=[None]*8
        for index in range(1024):
            slot=index%8
            if environments[slot] is None:
                episode=1+(batch_index*128+index//8)//120
                case=next(c for c in cases if c['episode_id']==f'slot-{slot}-episode-{episode}')
                env=synthetic_environment(case,episode);environments[slot]=env;active[slot]=case
                resets[case['episode_id']]=write(folder/(case['episode_id']+'-initial.json'),r.capture_environment(env))
            env=environments[slot];case=active[slot];pre=r.public_pre_action_snapshot(env.state)
            actions,_=frozen.act(env.state,base_seed=case['action_seed'],deterministic=False)
            _,rewards,terminated,truncated,info=env.step(actions)
            events.append(dict(slot=slot,case=case,pre_state_public=pre,post_state_public=r.public_pre_action_snapshot(env.state),
                proposed_actions=actions,action_override_agents=list(info['action_override_agents']),rewards=rewards,
                terminated=terminated,truncated=truncated,shutdowns=list(info['shutdowns'])))
            if index>=1016:ends[slot]=write(folder/f'batch-{batch_index}-slot-{slot}-end.json',r.capture_environment(env))
            if terminated or truncated:environments[slot]=None;active[slot]=None
        trace=dict(schema='warehouse-v629-collected-PPO-public-frames.v1',partition='train',owner=owner,config=asdict(r.PPOConfig()),
            start_joint_step=batch_index*1024,end_joint_step=(batch_index+1)*1024,actor=r.pointer(ref['path']),
            rollout_state=dict(actor_role_sha256=ref['role_sha256'],critic_sha256=critic_initialization_sha256(state.model),actor_updates=0,critic_updates=0),
            physics=dict(schema=r.PHYSICS_VERSION,resets=resets,end_snapshots=ends),events=events)
        traces.append(write(folder/f'batch-{batch_index}.json',trace))
    yield folder,ref,owner,inventory,traces
    torch.set_num_threads(prior)


def test_private_snapshot_roundtrip_never_calls_constructor_reset_or_set_state(recording,monkeypatch):
    _,_,_,_,traces=recording;trace=r.bound(traces[0]);snapshot=r.bound(next(iter(trace['physics']['resets'].values())))
    def forbidden(*args,**kwargs):raise AssertionError('audit must not create a fresh scenario or recompute goals')
    monkeypatch.setattr(r.WarehouseMultiAgentEnv,'__init__',forbidden);monkeypatch.setattr(r.WarehouseMultiAgentEnv,'reset',forbidden)
    monkeypatch.setattr(r.WarehouseMultiAgentEnv,'set_state',forbidden);monkeypatch.setattr(random.Random,'__init__',forbidden)
    env=r.restore_environment(snapshot)
    assert r.capture_environment(env)==snapshot and env._rng.getstate()==r.decode(snapshot['payload']['rng'])


def test_full_physics_and_public_nn_replay_agree_on_original_rewards_and_complete_episodes(recording):
    _,ref,owner,inventory,traces=recording;state=initialize(ref);reader=r.EnvironmentRolloutReplay(owner,inventory)
    state.advance_joint_steps(1024)
    batch,audit=reader.assemble(traces[0],state)
    assert len(batch['rows'])==1024 and audit['physical_frames']==1024 and audit['complete_episodes']==8
    assert len(batch['complete_cost_rows'])==960 and [len(x) for x in reader.public.pending]==[8]*8
    assert all(row['rewards'].tolist()==np.asarray([e['rewards'][role] for role in r.AGENTS],np.float32).tolist()
        for row,e in zip(batch['rows'],r.bound(traces[0])['events'],strict=True))
    assert not audit['training_authorized'] and not audit['private_state_used_as_NN_input']
    assert audit['environment_constructors']==audit['new_environment_resets']==audit['optimizer_steps']==0
    assert state.actor_updates==state.critic_updates==0


@pytest.mark.parametrize('change',['reward','public_post','override','unknown_case','missing_reset','wrong_slot','end_rng','reward_config'])
def test_false_physical_records_fail_before_any_optimizer(recording,tmp_path,change):
    _,ref,owner,inventory,traces=recording;value=r.bound(traces[0]);state=initialize(ref)
    if change=='reward':value['events'][0]['rewards']['robot_1']+=1
    if change=='public_post':value['events'][0]['post_state_public']['agents'][0]['battery']-=1
    if change=='override':value['events'][0]['action_override_agents']=['robot_1']
    if change=='unknown_case':value['events'][0]['case']['episode_id']='not-in-inventory'
    if change=='missing_reset':value['physics']['resets'].pop(next(iter(value['physics']['resets'])))
    if change=='wrong_slot':value['events'][0]['slot']=1
    if change in ('end_rng','reward_config'):
        original=value['physics']['end_snapshots'][0] if change=='end_rng' else next(iter(value['physics']['resets'].values()))
        snapshot=r.bound(original);env=r.restore_environment(snapshot)
        if change=='end_rng':env._rng.random()
        else:
            from dataclasses import replace
            env.config=replace(env.config,shutdown_points=-1.)
        changed=write(tmp_path/'changed-private.json',r.capture_environment(env))
        if change=='end_rng':value['physics']['end_snapshots'][0]=changed
        else:value['physics']['resets'][next(iter(value['physics']['resets']))]=changed
    trace=write(tmp_path/'changed.json',value);reader=r.EnvironmentRolloutReplay(owner,inventory);state.advance_joint_steps(1024)
    with pytest.raises((RuntimeError,ValueError,KeyError)):reader.assemble(trace,state)
    assert reader.failed and state.actor_updates==state.critic_updates==0
    with pytest.raises(RuntimeError,match='failed physics reader'):reader.assemble(traces[0],state)


def test_private_checkpoint_and_public_pending_restore_next_batch_exactly(recording,tmp_path):
    _,ref,owner,inventory,traces=recording;state=initialize(ref);reader=r.EnvironmentRolloutReplay(owner,inventory)
    state.advance_joint_steps(1024);reader.assemble(traces[0],state);snapshot=reader.snapshot(state)
    file=tmp_path/'full-training.pt';saved=save_checkpoint(state,file)
    resumed=load_checkpoint(file,saved['sha256']);restored=r.EnvironmentRolloutReplay.restore(snapshot,resumed,owner_ref=owner,inventory_ref=inventory)
    state.advance_joint_steps(1024);resumed.advance_joint_steps(1024)
    one,audit1=reader.assemble(traces[1],state);two,audit2=restored.assemble(traces[1],resumed)
    assert audit1==audit2 and len(one['rows'])==len(two['rows'])==1024
    for row1,row2 in zip(one['rows'],two['rows'],strict=True):
        for key in row1:
            if isinstance(row1[key],np.ndarray):assert np.array_equal(row1[key],row2[key])
            else:assert row1[key]==row2[key]
    assert reader.snapshot(state)==restored.snapshot(resumed)


def test_snapshot_rejects_tampered_private_rng_even_when_outer_checksum_is_recomputed(recording,tmp_path):
    _,ref,owner,inventory,traces=recording;state=initialize(ref);reader=r.EnvironmentRolloutReplay(owner,inventory)
    state.advance_joint_steps(1024);reader.assemble(traces[0],state);snapshot=reader.snapshot(state)
    env=r.restore_environment(snapshot['payload']['environments'][0]);env._rng.random()
    snapshot['payload']['environments'][0]=r.capture_environment(env);snapshot['payload_sha256']=r.digest(snapshot['payload'])
    file=tmp_path/'full-state.pt';saved=save_checkpoint(state,file);resumed=load_checkpoint(file,saved['sha256'])
    with pytest.raises(RuntimeError,match='recorded end state'):
        r.EnvironmentRolloutReplay.restore(snapshot,resumed,owner_ref=owner,inventory_ref=inventory)


@pytest.mark.parametrize('event',['collision','shutdown','delivery_refill'])
def test_full_private_state_replays_collision_shutdown_and_order_rng_without_reset(event):
    case=dict(scenario_seed=10008);env=synthetic_environment(case,1)
    positions=set(layout.passable_positions);first,second=env.state.agents
    actions={role:'WAIT' for role in r.AGENTS}
    if event in ('collision','shutdown'):
        start,action,target=next((p,a,(p[0]+d[0],p[1]+d[1])) for p in sorted(positions)
            for a,d in MOVE_DELTAS.items() if (p[0]+d[0],p[1]+d[1]) in positions)
        first.position=start;actions['robot_1']=action
        if event=='collision':
            second.position=target;actions['robot_2']={'UP':'DOWN','DOWN':'UP','LEFT':'RIGHT','RIGHT':'LEFT'}[action]
        else:
            second.position=next(p for p in sorted(positions) if p not in (start,target));first.battery=2.
    else:
        task=env.state.tasks[0];target=task.delivery_position
        start,action=next(((target[0]-d[0],target[1]-d[1]),a) for a,d in MOVE_DELTAS.items()
            if (target[0]-d[0],target[1]-d[1]) in positions)
        first.position=start;second.position=next(p for p in sorted(positions) if p not in (start,target))
        first.carrying_task_id=task.task_id;task.status='carried';task.carrier_agent_id=first.agent_id;task.claimed_frame=0
        actions['robot_1']=action
    env.set_state(env.state);before=r.capture_environment(env);clone=r.restore_environment(before)
    original=env.step(actions);replayed=clone.step(actions)
    assert original[1:4]==replayed[1:4] and r.capture_environment(env)==r.capture_environment(clone)
    assert not original[4]['action_override_agents']
    if event=='collision':assert env.state.robot_collision_events==1
    if event=='shutdown':assert original[2] and env.state.shutdown_count>=1
    if event=='delivery_refill':assert env.state.total_deliveries==1 and before['payload']['rng']!=r.capture_environment(env)['payload']['rng']


@pytest.mark.parametrize('change',['unknown_type','missing_field','nonfinite','invalid_rng'])
def test_data_only_snapshot_rejects_unknown_incomplete_or_nonfinite_state(recording,change):
    trace=r.bound(recording[-1][0]);snapshot=r.bound(next(iter(trace['physics']['resets'].values())))
    if change=='unknown_type':snapshot['payload']['state']['kind']='run_arbitrary_code'
    if change=='missing_field':snapshot['payload']['state']['fields'].pop('total_deliveries')
    if change=='nonfinite':snapshot['payload']['state']['fields']['user_score']=float('nan')
    if change=='invalid_rng':snapshot['payload']['rng']=r.encode((3,(0,),None))
    with pytest.raises((RuntimeError,ValueError)):
        snapshot['payload_sha256']=r.digest(snapshot['payload']);r.restore_environment(snapshot)


def test_asset_changed_during_public_nn_replay_cannot_deliver_training_rows(recording,tmp_path,monkeypatch):
    _,ref,owner,inventory,traces=recording;value=r.bound(traces[0]);state=initialize(ref)
    # Clone just one end asset so this mutation cannot affect shared fixtures.
    end=write(tmp_path/'end-private.json',r.bound(value['physics']['end_snapshots'][0]))
    value['physics']['end_snapshots'][0]=end;trace=write(tmp_path/'trace.json',value)
    reader=r.EnvironmentRolloutReplay(owner,inventory);original=reader.public.assemble
    def replaced(*args):
        result=original(*args);path=Path(end['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(reader.public,'assemble',replaced);state.advance_joint_steps(1024)
    with pytest.raises(RuntimeError,match='physics prerequisite asset missing or changed'):reader.assemble(trace,state)
    assert reader.failed and state.actor_updates==state.critic_updates==0
