"""Dynamics, provenance, deterministic replay and baseline acceptance."""
import copy
import json
import numpy as np
import pytest

from env.cooperative_kitchen import (
    ACTIONS, ACTOR_IDS, SCENARIOS, OBSERVATION_DIM, OBSERVATION_FEATURES,
    STATE_DIM, CooperativeKitchen, KitchenConfig, program_decision,
)
from env.cooperative_kitchen.environment import _valid_layout


def step(env, human="WAIT", ai="WAIT"):
    return env.step({"human": human, "ai": ai})


def give(env, id, kind):
    actor = env.state['actors'][ACTOR_IDS.index(id)]
    assert actor['holding'] is None
    actor['holding'] = kind
    actor['_held_id'] = env._new_item(kind, f'actor:{id}')
    return actor['_held_id']


def at_counter(env, id, row=2):
    actor = env.state['actors'][ACTOR_IDS.index(id)]
    actor.update(position=[row, 3 if id == 'human' else 5], facing='RIGHT' if id == 'human' else 'LEFT')


def auto(env, profiles=('efficient', 'efficient')):
    rng = np.random.default_rng(87)
    while not env.state['done']:
        env.step({id: program_decision(env, id, profile=p, rng=rng)['action'] for id,p in zip(ACTOR_IDS,profiles)})
    return env.state


def test_default_shape_and_geometry():
    env = CooperativeKitchen()
    state = env.public_view()
    assert state['maxSteps'] == 180 and state['targetOrders'] == 2
    assert state['map'] == ['#########','#I.X#X.D#','#...C...#','#H..#..A#','#...C...#','#S..#..P#','#########']
    assert state['actors'][0]['position'] == [3,1]
    assert state['actors'][1]['position'] == [3,7]
    assert state['score'] == 0
    assert _valid_layout(tuple(state['map']))


@pytest.mark.parametrize('scene', SCENARIOS)
def test_all_named_scenes_solvable(scene):
    env = CooperativeKitchen(scenario_id=scene)
    result = auto(env)
    assert result['orders'] == 2 and result['reason'] == 'success'
    assert result['turn'] <= 180
    env.restore(env.snapshot())


def test_seeded_variants_distinct_and_connected():
    states = []
    for seed in range(50):
        env = CooperativeKitchen(seed=seed, scenario_id='generated')
        s = env.public_view()
        assert _valid_layout(env.grid)
        states.append(json.dumps([s['map'],s['actors'],s['pot'],s['counters']],sort_keys=True))
        assert auto(env)['orders'] == 2
    assert len(set(states)) == 50


@pytest.mark.parametrize('profile', ['efficient','upper','lower','perturbed'])
def test_partner_profiles_are_usable(profile):
    env = CooperativeKitchen()
    assert auto(env, (profile, 'efficient'))['orders'] == 2


def test_turn_block_and_invalid_interaction_are_separate():
    env = CooperativeKitchen()
    env.state['actors'][0].update(position=[2,1],facing='LEFT')
    r = step(env,'UP')
    assert r['events'][0]['type'] == 'turn_in_place'
    assert env.state['actors'][0]['position'] == [2,1]
    assert step(env,'UP')['events'][0]['type'] == 'blocked'
    assert env.state['turn'] == 2
    # Point toward wall, not a source; interacting still consumes a joint step.
    step(env,'LEFT')
    r = step(env,'INTERACT')
    assert any(e['type']=='invalid_interaction' for e in r['events'])
    assert env.state['turn'] == 4


def test_source_one_hand_and_discard():
    env = CooperativeKitchen()
    actor = env.state['actors'][0]
    actor.update(position=[2,1],facing='UP')
    r = step(env,'INTERACT')
    item_id = actor['_held_id']
    assert actor['holding'] == 'onion'
    assert r['events'][-1]['item_id'] == item_id
    before = len(env.state['_items'])
    assert any(e.get('reason')=='hands_full' for e in step(env,'INTERACT')['events'])
    assert len(env.state['_items']) == before
    actor.update(position=[2,3],facing='UP')
    step(env,'INTERACT')
    assert actor['holding'] is None
    assert env.state['_items'][item_id]['location'] == 'discarded'


def test_simultaneous_drop_does_not_enable_pickup():
    env = CooperativeKitchen()
    at_counter(env,'human'); at_counter(env,'ai')
    item_id = give(env,'human','onion')
    r = step(env,'INTERACT','INTERACT')
    assert env.state['counters']['2,4'] == 'onion'
    assert env.state['actors'][1]['holding'] is None
    assert any(e.get('reason')=='counter_empty' for e in r['events'])
    step(env,'WAIT','INTERACT')
    assert env.state['actors'][1]['_held_id'] == item_id
    assert env.state['counters']['2,4'] is None


@pytest.mark.parametrize('turn, winner',[(0,'human'),(1,'ai')])
def test_contested_pickup_priority_never_duplicates(turn,winner):
    env = CooperativeKitchen()
    env.state['turn']=turn
    env._put_initial_counter('2,4','onion')
    item_id = env.state['_counter_item_ids']['2,4']
    at_counter(env,'human');at_counter(env,'ai')
    r = step(env,'INTERACT','INTERACT')
    assert sum(a['holding']=='onion' for a in env.state['actors']) == 1
    actor = env.state['actors'][ACTOR_IDS.index(winner)]
    assert actor['_held_id'] == item_id
    assert env.state['_items'][item_id]['location'] == f'actor:{winner}'
    assert len([e for e in r['events'] if e['type']=='conflict']) == 1


@pytest.mark.parametrize('turn,winner',[(0,'human'),(1,'ai')])
def test_contested_drop_retains_losing_item(turn,winner):
    env=CooperativeKitchen()
    env.state['turn']=turn
    at_counter(env,'human');at_counter(env,'ai')
    ids={id:give(env,id,'onion') for id in ACTOR_IDS}
    step(env,'INTERACT','INTERACT')
    assert env.state['_counter_item_ids']['2,4']==ids[winner]
    loser=next(id for id in ACTOR_IDS if id!=winner)
    assert env.state['actors'][ACTOR_IDS.index(loser)]['_held_id']==ids[loser]


def test_four_subsequent_cooking_steps_and_plating():
    env=CooperativeKitchen(scenario_id='base_inprogress')
    actor=env.state['actors'][1]
    actor.update(position=[4,7],facing='DOWN')
    give(env,'ai','onion')
    r=step(env,ai='INTERACT')
    assert env.state['pot']==dict(ingredients=3,remaining=4,ready=False)
    assert any(e['type']=='cooking_started' for e in r['events'])
    batch=env.state['_pot_batch']
    give(env,'ai','plate')
    for remaining in (3,2,1):
        step(env)
        assert env.state['pot']['remaining']==remaining
    # On the fourth subsequent step the soup is not ready at intent time.
    r=step(env,ai='INTERACT')
    assert actor['holding']=='plate' and env.state['pot']['ready']
    assert any(e.get('reason')=='pot_cooking' for e in r['events'])
    step(env,ai='INTERACT')
    assert actor['holding']=='soup'
    assert env.state['pot']==dict(ingredients=0,remaining=0,ready=False)
    assert env.state['_items'][actor['_held_id']]['batch_id']==batch
    assert len(env.state['_batches'][batch]['ingredient_ids'])==3
    env.restore(env.snapshot())


def test_handoff_serving_provenance_and_score():
    env=CooperativeKitchen()
    auto(env)
    s=env.snapshot()
    assert s['_first_serve_turn'] > 0
    served=[item for item in s['_items'].values() if item['location']=='served']
    assert len(served)==2 and len({i['batch_id'] for i in served})==2
    assert all(s['_batches'][i['batch_id']]['served_turn'] for i in served)
    assert env.public_view()['score']==200-s['turn']
    assert env.state['turn']==104


def test_congestion_wait_has_no_secret_deletion_and_recovers():
    env=CooperativeKitchen(scenario_id='base_congestion')
    original_items=copy.deepcopy(env.state['_items'])
    for _ in range(3):
        decision=program_decision(env,'ai')
        assert decision['action']=='WAIT' and decision['rule']=='wait_space'
        step(env,ai=decision['action'])
    assert env.state['_items']==original_items
    assert auto(env)['orders']==2


def test_terminal_idempotence_success_over_timeout():
    env=CooperativeKitchen(KitchenConfig(horizon=104))
    auto(env)
    assert env.state['reason']=='success' and env.state['turn']==104
    snapshot=env.snapshot()
    r=step(env,'INTERACT','UP')
    assert env.snapshot()==snapshot and r['rewards']=={'human':0.0,'ai':0.0}
    assert r['actual_actions']=={'human':'WAIT','ai':'WAIT'}


def test_timeout_and_wait_do_not_imply_error():
    env=CooperativeKitchen(KitchenConfig(horizon=2))
    step(env)
    result=step(env)
    assert env.state['reason']=='timeout' and env.public_view()['score']==-2
    assert [e['type'] for e in result['events']]==['wait','wait','timeout']


def test_same_seed_actions_snapshot_and_fork():
    env=CooperativeKitchen(seed=47,scenario_id='generated')
    twin=CooperativeKitchen(seed=47,scenario_id='generated')
    rng=np.random.default_rng(43)
    for _ in range(20):
        actions={id:str(rng.choice(ACTIONS)) for id in ACTOR_IDS}
        assert env.step(actions)==twin.step(actions)
    snapshot=env.snapshot()
    branch=env.fork()
    branch.step({'human':'UP','ai':'INTERACT'})
    assert env.snapshot()==snapshot
    recovered=CooperativeKitchen();recovered.restore(json.loads(json.dumps(snapshot)))
    assert recovered.snapshot()==snapshot
    assert recovered.step({'human':'WAIT','ai':'DOWN'})==env.step({'human':'WAIT','ai':'DOWN'})


def test_observations_are_finite_named_semantic_and_do_not_depend_on_commands():
    env=CooperativeKitchen()
    obs=env.observations()
    assert len(set(OBSERVATION_FEATURES))==OBSERVATION_DIM
    assert all(v.dtype==np.float32 and v.shape==(OBSERVATION_DIM,) and np.isfinite(v).all() for v in obs.values())
    assert env.global_state(obs).shape==(STATE_DIM,)
    assert not any('teacher' in n or 'program' in n or 'chosen_action' in n for n in OBSERVATION_FEATURES)
    before=env.snapshot()
    for action in ACTIONS:
        branch=env.fork()
        np.testing.assert_array_equal(branch.observations()['ai'],obs['ai'])
        branch.step({'human':action,'ai':'WAIT'})
    assert env.snapshot()==before


def test_direction_turn_not_action_masked():
    env=CooperativeKitchen()
    env.state['actors'][1].update(position=[4,7],facing='UP')
    obs=env.observations()['ai']
    assert obs[OBSERVATION_FEATURES.index('direction_down_walkable')]==0
    r=step(env,ai='DOWN')
    assert r['actual_actions']['ai']=='DOWN'
    assert env.state['actors'][1]['facing']=='DOWN'


def test_discounted_potential_cannot_reward_repeat_pickup_drop():
    env=CooperativeKitchen()
    env._put_initial_counter('2,4','onion')
    at_counter(env,'ai')
    total=0
    for _ in range(10):
        total+=step(env,ai='INTERACT')['rewards']['ai']
    assert total<0
    assert env.state['counters']['2,4']=='onion'


def test_public_state_hides_item_ids_but_preserves_visible_inventory():
    env=CooperativeKitchen(scenario_id='base_congestion')
    view=env.public_view()
    assert view['actors'][1]['holding']=='soup'
    assert not any(k.startswith('_') for k in view)
    assert all('_held_id' not in actor for actor in view['actors'])
    view['actors'][1]['holding']=None
    assert env.state['actors'][1]['holding']=='soup'


def test_restore_rejects_tampering_and_duplicate_items():
    env=CooperativeKitchen(scenario_id='base_congestion')
    original=env.snapshot()
    for mutate in (
        lambda s:s.update(turn=181),
        lambda s:s.update(reason='success'),
        lambda s:s['actors'][0].update(position=[0,0]),
        lambda s:s['actors'][0].update(holding='onion'),
        lambda s:s['pot'].update(remaining=2),
    ):
        invalid=copy.deepcopy(original);mutate(invalid)
        with pytest.raises((ValueError,KeyError,TypeError)): env.restore(invalid)
        assert env.snapshot()==original


def test_bad_action_no_mutation():
    env=CooperativeKitchen(); original=env.snapshot()
    for actions in ({'human':'JUMP','ai':'WAIT'},{'human':'UP'},{'human':'UP','ai':'WAIT','extra':'WAIT'}):
        with pytest.raises(ValueError):env.step(actions)
    assert env.snapshot()==original


def test_perturbations_use_caller_rng_and_do_not_mutate_env():
    env=CooperativeKitchen();before=env.snapshot()
    with pytest.raises(ValueError):program_decision(env,'human',profile='perturbed')
    a,b=np.random.default_rng(3),np.random.default_rng(3)
    assert [program_decision(env,'human','perturbed',a) for _ in range(30)]==[program_decision(env,'human','perturbed',b) for _ in range(30)]
    assert env.snapshot()==before

@pytest.mark.parametrize('scene',['base_empty','base_congestion','mirror_inprogress','generated'])
def test_freeplay_swap_preserves_item_identity_and_restores(scene):
    env=CooperativeKitchen(seed=47,scenario_id=scene)
    env.swap_roles()
    assert env.public_view()['preset']=='cook'
    assert env.state['actors'][0]['side']=='right'
    restored=CooperativeKitchen();restored.restore(env.snapshot())
    assert restored.snapshot()==env.snapshot()
    assert auto(env)['orders']==2
    with pytest.raises(ValueError):env.swap_roles()


def test_step_return_cannot_mutate_persisted_events():
    env=CooperativeKitchen()
    result=step(env)
    result['events'][0]['type']='invented'
    assert env.public_view()['events'][0]['type']=='wait'
