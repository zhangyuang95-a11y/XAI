"""Independent v6.2 rule boundaries; synthetic states are not human evidence."""
from copy import deepcopy
import json

import pytest

from domains.kitchen import engine as e
from tests.test_study_v3_kitchen import fixture, item, dish, tick
from tests.test_study_v3_kitchen_v5 import collect, prepare


def timed(ingredient='egg', prepared=50, iid='timed', order='order1'):
    food = item(ingredient, iid=iid, order=order)
    food.update(acquired_turn=prepared - 10, prepared_turn=prepared,
                freshness_started_turn=prepared, fresh_until=prepared + 20,
                expires_turn=prepared + 20, freshness_basis='prepared')
    return food


def locate(state, where, food):
    if where in ('human', 'ai'):
        state[where]['holding'] = food
    elif where == 'handoff':
        state['handoff'] = food
    elif where == 'human_buffer':
        state['buffers']['human'] = food
    else:
        state['buffers']['ai_raw'][int(where[-1])] = food


@pytest.mark.parametrize('ingredient', list(e.LABELS))
@pytest.mark.parametrize('where', ['human', 'ai', 'handoff', 'human_buffer', 'slot0', 'slot1'])
def test_same_prepared_clock_at_all_six_locations_and_19_20_boundary(ingredient, where):
    state = fixture()
    state['turn'] = 68
    food = timed(ingredient)
    # Old storage age is deliberately greater than 10: prepared portions
    # must obey only the completion timestamp in every physical location.
    food.update(storage_since_turn=50, storage_station='ai_raw' if where.startswith('slot') else 'human_buffer')
    locate(state, where, food)
    original_components = list(food['components'])
    state = tick(state)
    assert state['turn'] == 69
    actual = next(it for it in e._all_items(state) if it and it['id'] == food['id'])
    assert actual['stage'] == 'prepared'
    fresh = next(row for row in e.public_state(state)['food_freshness'] if row['item_id'] == food['id'])
    assert fresh['prepared_turn'] == 50 and fresh['expires_turn'] == 70
    assert fresh['remaining'] == 1 and fresh['warning']
    restored = json.loads(json.dumps(state))
    state = tick(restored)
    actual = next(it for it in e._all_items(state) if it and it['id'] == food['id'])
    assert actual['stage'] == 'spoiled' and actual['spoiled_turn'] == 70
    assert actual['components'] == original_components
    assert state['metrics']['spoiled'] == 1 and state['metrics']['discard_penalty'] == 0
    assert any(ev['type'] == 'ingredient_spoiled' and ev['item']['prepared_turn'] == 50
               and ev['item']['expires_turn'] == 70 for ev in state['events'])
    assert e.public_state(state)['food_freshness'][0]['remaining'] == 0


@pytest.mark.parametrize('ingredient', list(e.LABELS))
def test_preparation_completion_is_the_clock_origin_not_acquisition_or_first_e(ingredient):
    state = collect(fixture(), ingredient)
    acquired = state['human']['holding']['acquired_turn']
    while (e._front(state['human']) or {}).get('id') != 'prep':
        state = tick(state, e._approach(state, 'human', 'prep'))
    state = tick(state, 'interact')
    partial = state['human']['holding']['prepare_progress']
    assert partial == 1 and 'prepared_turn' not in state['human']['holding']
    for _ in range(3):
        state = tick(state)
        assert state['human']['holding']['prepare_progress'] == partial
    state = prepare(state)
    food = state['human']['holding']
    assert food['prepared_turn'] == state['turn'] > acquired
    assert food['fresh_until'] == food['expires_turn'] == state['turn'] + 20
    assert e.public_state(state)['food_freshness'][0]['remaining'] == 20


def test_multiple_real_transfers_and_json_restore_keep_preparation_deadline():
    state = fixture()
    state['human'].update(x=3, y=3, facing='right', holding=timed('egg', prepared=0))
    state['ai'].update(x=5, y=3, facing='left')
    original = deepcopy(state['human']['holding'])
    state = tick(state, 'interact')  # human -> handoff
    state = tick(state, ai='interact')  # handoff -> AI
    while (e._front(state['ai']) or {}).get('id') != 'ai_raw':
        state = tick(state, ai=e._approach(state, 'ai', 'ai_raw'))
    state = tick(state, ai='interact', slot=1)  # AI -> second slot
    state = tick(state, ai='interact', slot=1)  # second slot -> AI
    while (e._front(state['ai']) or {}).get('id') != 'handoff':
        state = tick(state, ai=e._approach(state, 'ai', 'handoff'))
    state = tick(state, ai='interact')
    state = tick(state, 'interact')  # handoff -> human
    state = json.loads(json.dumps(state))
    food = state['human']['holding']
    assert food['id'] == original['id'] and food['components'] == original['components']
    assert food['prepared_turn'] == 0 and food['expires_turn'] == food['fresh_until'] == 20
    assert e.public_state(state)['food_freshness'][0]['remaining'] == 20 - state['turn']
    while state['turn'] < 20:
        state = tick(state)
    assert state['human']['holding']['stage'] == 'spoiled'


@pytest.mark.parametrize('ingredient', list(e.LABELS))
def test_action_committed_at_expiry_cannot_load_and_does_not_mutate(ingredient):
    state = fixture()
    state['turn'] = 69
    order = 'order1' if e.INGREDIENT_RECIPE[ingredient] == 'egg_tomato' else 'order2'
    state['ai'].update(x=6, y=1, facing='right', holding=timed(ingredient, order=order))
    if ingredient in ('tomato', 'pepper'):
        state['pots'][0].update(phase='await_vegetable', order_id=order, recipe=e.INGREDIENT_RECIPE[ingredient])
    before = deepcopy(state)
    assert e._freshness(state['ai']['holding'], 69)['remaining'] == 1
    assert 'interact' not in e.legal_actions(state, 'ai')
    with pytest.raises(ValueError):
        tick(state, ai='interact')
    assert state == before
    after = tick(state)
    assert after['ai']['holding']['stage'] == 'spoiled'
    assert after['pots'][0]['item'] is None


@pytest.mark.parametrize('ingredient', ['egg', 'meat'])
def test_load_at_age_19_ends_freshness_but_exact_eight_ready_turn_burns(ingredient):
    state = fixture()
    state['turn'] = 68
    order = 'order1' if ingredient == 'egg' else 'order2'
    state['ai'].update(x=6, y=1, facing='right', holding=timed(ingredient, order=order))
    state = tick(state, ai='interact')
    pot = state['pots'][0]
    assert pot['remaining'] == e.COOK_TURNS[ingredient] and pot['item']['loaded_turn'] == 69
    assert pot['item']['prepared_turn'] == 50 and pot['item']['expires_turn'] == 70
    assert pot['item']['fresh_until'] is None and not e.public_state(state)['food_freshness']
    for _ in range(e.COOK_TURNS[ingredient]):
        state = tick(state)
    assert state['pots'][0]['status'] == 'ready' and state['pots'][0]['ready_age'] == 0
    for age in range(1, 8):
        state = tick(state)
        assert state['pots'][0]['status'] == 'ready' and state['pots'][0]['ready_age'] == age
    state = tick(state)
    assert state['pots'][0]['status'] == 'burnt'
    assert state['metrics']['spoiled'] == 0 and state['metrics']['discard_penalty'] == 0


def test_score_net_effects_deadline_inclusive_and_no_ordering_requirement():
    for waits, expected in ((0, 29), (1, 28)):
        state = fixture()
        food = dish(order='order3')
        food.update(stage='plated', container='serving_plate')
        state['orders'][2]['recipe'] = 'egg_tomato'
        state['orders'][2]['deadline'] = waits + 1
        state['human'].update(x=2, y=5, facing='right', holding=food)
        events = []
        for _ in range(waits):
            state = tick(state); events.extend(state['events'])
        state = tick(state, 'interact'); events.extend(state['events'])
        assert state['raw_score'] == expected
        assert state['orders'][2]['status'] == 'completed'
        assert state['orders'][0]['status'] == 'pending'  # order3 may finish first.
        assert sum(ev['delta'] for ev in events if ev['type'] == 'score_delta') == expected
        assert next(ev for ev in events if ev['type'] == 'served')['score_delta'] == 30
    for food, net in ((item(), -4), (dish(), -11)):
        state = fixture()
        state['human'].update(x=3, y=4, facing='right', holding=food)
        state = tick(state, 'interact')
        assert state['raw_score'] == net and state['human']['holding'] is None


def test_seeded_menus_frozen_distinct_across_tasks_no_triples_and_group_independent():
    all_menus = set()
    for seed in list(range(100)) + list(range(1000, 1024)) + list(range(2000, 2024)):
        menus = []
        for task in (1, 2, 3):
            group_a = e.initial_state(seed, task)
            group_b = e.initial_state(seed, task)
            assert group_a == group_b
            menu = tuple(order['recipe'] for order in group_a['orders'])
            assert len(menu) == 5 and len(set(menu)) == 2
            assert all(not (menu[i] == menu[i+1] == menu[i+2]) for i in range(3))
            assert [o['deadline'] for o in group_a['orders']] == [100, 140, 240, 280, 360]
            assert group_a['max_turns'] == 360
            restored = json.loads(json.dumps(group_a))
            assert restored['orders'] == group_a['orders']
            assert e.step(restored, 'wait')['orders'] == group_a['orders']
            assert restored['menu_version'] == 'kitchen-menu-v2'
            menus.append(menu); all_menus.add(menu)
        assert len(set(menus)) == 3
    assert len(all_menus) > 3  # No single fixed alternation template.


def test_versioned_metadata_matches_new_physics_and_public_state_is_pure():
    state = e.initial_state(1000, 1)
    original = deepcopy(state)
    view = e.public_state(state)
    meta = view['rule_metadata']
    assert meta == state['rule_metadata'] == e.rule_metadata()
    assert meta['engine_version'] == 'kitchen-v6.2.0'
    assert meta['menu_version'] == 'kitchen-menu-v2'
    assert meta['score'] == {'served': 30, 'step': -1, 'single_component_discard': -3, 'combined_dish_discard': -10}
    assert meta['prepared_fresh_turns'] == 20 and meta['max_consecutive_same_recipe'] == 2
    assert meta['heating']['egg_tomato']['total'] == 16 and meta['heating']['pepper_meat']['total'] == 20
    assert state == original
    view['rule_metadata']['score']['served'] = 999
    assert state == original
