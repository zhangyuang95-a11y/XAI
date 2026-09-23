"""16/20 heating times and physical two-pan scheduling regressions."""
from copy import deepcopy
import pytest

from domains.kitchen import engine as e
from tests.test_study_v3_kitchen import fixture, item, tick


def test_recipe_heating_totals_and_every_stage_boundary():
    assert e.COOK_TURNS == {'egg': 8, 'tomato': 6, 'meat': 10, 'pepper': 8}
    assert e.COOK_TURNS['egg'] + e.COOK_TURNS['tomato'] + e.MIX_TURNS == 16
    assert e.COOK_TURNS['meat'] + e.COOK_TURNS['pepper'] + e.MIX_TURNS == 20
    for ingredient, turns in [('egg', 8), ('tomato', 6), ('meat', 10), ('pepper', 8)]:
        state = fixture()
        recipe = e.INGREDIENT_RECIPE[ingredient]
        order = 'order1' if recipe == 'egg_tomato' else 'order2'
        pot = state['pots'][0]
        if ingredient in ('tomato', 'pepper'):
            pot.update(phase='await_vegetable', recipe=recipe, order_id=order)
        state['ai'].update(x=6, y=1, facing='right', holding=item(ingredient, order=order))
        state = tick(state, ai='interact')
        assert state['pots'][0]['remaining'] == turns  # Loading is not heating.
        for elapsed in range(1, turns):
            state = tick(state)
            assert state['pots'][0]['status'] == 'cooking'
            assert state['pots'][0]['remaining'] == turns - elapsed
        state = tick(state)
        assert state['pots'][0]['status'] == 'ready'
        assert state['pots'][0]['ready_age'] == 0


@pytest.mark.parametrize('first,second', [('egg', 'egg'), ('egg', 'meat'), ('meat', 'egg'), ('meat', 'meat')])
def test_fixed_ai_starts_second_pan_and_both_timers_really_advance(first, second):
    state = fixture()
    state['orders'][0]['recipe'] = e.INGREDIENT_RECIPE[first]
    state['orders'][1]['recipe'] = e.INGREDIENT_RECIPE[second]
    egg = item(first, pot='pot1')
    egg.update(fresh_until=None, freshness_basis='in_pan')
    state['pots'][0].update(phase='protein', status='cooking', item=egg,
                           recipe=e.INGREDIENT_RECIPE[first], order_id='order1', remaining=e.COOK_TURNS[first])
    state['ai'].update(x=5, y=3, facing='left')
    state['handoff'] = item(second, iid='second-protein', order='order2')
    frames = [deepcopy(state)]
    for _ in range(9):
        state = e.step(state, 'wait')  # Actual fixed AI; no forced AI actions.
        frames.append(state)
    pairs = [(a, b) for a, b in zip(frames, frames[1:])
             if all(p['status'] == 'cooking' for p in a['pots'] + b['pots'])]
    assert pairs
    for before, after in pairs:
        assert len({p['order_id'] for p in before['pots']}) == 2
        assert all(b['remaining'] == a['remaining'] - 1
                   for a, b in zip(before['pots'], after['pots']))
    assert state['metrics']['parallel_cooking_turns'] >= 2
    assert state['metrics']['burnt'] == 0


def test_second_pan_safety_includes_storage_then_return_to_new_vegetable():
    state = fixture()
    state['ai'].update(x=5, y=3, facing='left')
    state['pots'][0].update(phase='await_vegetable', recipe='egg_tomato', order_id='order1')
    egg = item(stage='cooked_protein', pot='pot1')
    egg['was_buffered'] = True
    state['buffers']['protein']['pot1'] = egg
    meat = item('meat', iid='cooking-meat', order='order2', pot='pot2')
    meat.update(fresh_until=None, freshness_basis='in_pan')
    state['pots'][1].update(phase='protein', status='cooking', item=meat,
                           recipe='pepper_meat', order_id='order2', remaining=5)
    state['handoff'] = item('tomato', iid='next-tomato', order='order1')
    # Loading tomato then removing/storing meat and fetching/returning egg
    # would burn the newly loaded tomato. First finish the meat's storage.
    assert e.decide(state)['reason_code'] == 'reserve_rescue_window'
    events = []
    for _ in range(35):
        state = e.step(state, 'wait')
        events.extend(state['events'])
    assert state['metrics']['burnt'] == 0
    stored = next(i for i, event in enumerate(events)
                  if event['type'] == 'item_placed' and event.get('station') == 'protein2')
    loaded = next(i for i, event in enumerate(events)
                  if event['type'] == 'pot_loaded' and event['item']['ingredient'] == 'tomato')
    assert stored < loaded


def test_public_advice_does_not_discard_valid_later_menu_ingredient():
    state = e.initial_state(1003, 2)  # This four-dish menu starts with two egg dishes.
    for target in ('meat', 'handoff'):
        while (e._front(state['human']) or {}).get('id') != target:
            state = e.step(state, e._approach(state, 'human', target))
        state = e.step(state, 'interact')
    state = e.step(state, 'interact')  # Retrieve the unprepared meat.
    original = deepcopy(state)
    assert e.human_advisor(state) == e._approach(state, 'human', 'prep')
    assert e.simulation_partner(state) == e._approach(state, 'human', 'trash')
    evidence = {row['id']: row for row in e.facts(state)}
    assert '肉' in evidence['next_input_ingredient']['zh']
    assert state == original
