"""Real trajectories must describe a finished dish as awaiting human delivery."""
import pytest
import re

from domains.kitchen import engine as kitchen


@pytest.mark.parametrize('task', [1, 2, 3])
def test_final_dish_waiting_reason_does_not_request_more_ingredients(task):
    state = kitchen.initial_state(1000, task)
    observed = set()
    while not state['terminal']:
        decision = kitchen.decide(state)
        held = state['human']['holding']
        if (held and held['stage'] in ('finished', 'plated')
                and kitchen._next_input_portion(state) is None
                and decision['reason_code'] == 'await_ingredient'):
            observed.add(held['stage'])
            assert 'no additional ingredient is needed' in decision['reason_en']
            assert '现在不需要再递原料' in decision['reason_zh']
            recipe = kitchen.RECIPES[held['recipe']]
            assert recipe['en'] in decision['reason_en']
            assert recipe['zh'] in decision['reason_zh']
            assert not re.search(r'\border\d+\b', decision['reason_en'])
            assert not re.search(r'\border\d+\b', decision['reason_zh'])
            assert held['order_id'] in {order['id'] for order in state['orders']}
            if held['stage'] == 'plated':
                assert 'ready to serve' in decision['reason_en']
                assert '可以去上菜' in decision['reason_zh']
                assert held['container'] == 'serving_plate'
            else:
                assert 'Transfer it to a serving plate' in decision['reason_en']
                assert '先转装正式餐盘再上菜' in decision['reason_zh']
                assert held['container'] == 'output_container'
        state = kitchen.step(state, kitchen.human_advisor(state))
    assert observed == {'finished', 'plated'}
    assert state['metrics']['completed_orders'] == state['total_orders']
