"""Current Kitchen evidence and counterfactual clocks; no semantic-model claims."""
from copy import deepcopy

import pytest

from domains.kitchen import engine as e
from domains.kitchen.build_qa_cases import regular_trace, HEATING_SEED
from study_v3.qa import _catalog, _digest, simulate
from tests.test_study_v3_qa import explainer_for_plan, plan


def prepared_portion(ingredient):
    state = e.initial_state(1000, 2)
    for target in (ingredient, 'prep'):
        while (e._front(state['human']) or {}).get('id') != target:
            state = e.step(state, e._approach(state, 'human', target))
        count = 1 if target == ingredient else e.PREPARE_TURNS[ingredient]
        for _ in range(count):
            state = e.step(state, 'interact')
    return state


def answer(state, ids, question, language):
    selected = plan(state, language=language, ids=ids)
    return explainer_for_plan(selected).answer(e, state, e.decide(state), question, language)


@pytest.mark.parametrize('ingredient', tuple(e.LABELS))
@pytest.mark.parametrize('language', ('en', 'zh'))
def test_current_portion_evidence_matches_prepared_twenty_turn_clock(ingredient, language):
    state = prepared_portion(ingredient)
    item = state['human']['holding']
    prepared, expiry, item_id = item['prepared_turn'], item['expires_turn'], item['id']
    assert prepared == state['turn'] and expiry == prepared + 20
    for age in (0, 19, 20):
        while state['turn'] < prepared + age:
            state = e.step(state, 'wait')
        snapshot = deepcopy(state)
        result = answer(state, ['freshness_' + ingredient],
                        'When was this prepared and when does it expire?' if language == 'en' else '哪回合备好，哪回合过期？', language)
        assert result['status'] == 'answered'
        assert result['evidence_ids'] == ['freshness_' + ingredient]
        assert str(prepared) in result['answer'] and str(expiry) in result['answer']
        row = next(row for row in e.public_state(state)['food_freshness'] if row['item_id'] == item_id)
        assert row['prepared_turn'] == prepared and row['expires_turn'] == expiry
        assert row['remaining'] == max(0, 20-age)
        assert state['human']['holding']['id'] == item_id
        assert state['human']['holding']['stage'] == ('spoiled' if age == 20 else 'prepared')
        if age == 20:
            assert ('already spoiled' if language == 'en' else '已经变质') in result['answer']
        assert state == snapshot


@pytest.mark.parametrize('language', ('en', 'zh'))
def test_general_freshness_and_serving_arithmetic_are_short_current_facts(language):
    state = e.initial_state(1000, 2)
    fresh = answer(state, ['prepared_freshness_rule'], 'Does moving it reset freshness?', language)
    question = 'Why +100 reward but only +99 net?' if language == 'en' else '上菜奖励100分，为什么净增99分？'
    points = answer(state, ['serving_score_rule'], question, language)
    assert fresh['status'] == points['status'] == 'answered'
    assert '20' in fresh['answer']
    assert all(str(n) in points['answer'] for n in (100, 99, 98))
    if language == 'en':
        assert 'reset' in fresh['answer'] and len(fresh['answer'].split()) < 80
        assert len(points['answer'].split()) < 85
    else:
        assert '重置' in fresh['answer'] and len(fresh['answer']) < 180
        assert len(points['answer']) < 180
    assert '+30' not in points['answer'] and '500' not in points['answer']


def test_counterfactual_preserves_actual_both_pan_decrements_without_live_mutation():
    frames = regular_trace(HEATING_SEED)
    state = next(state for state in frames if all(p['status'] == 'cooking' and p['remaining'] > 1 for p in state['pots']))
    before = deepcopy(state)
    result = simulate(e, state, e.decide(state), ['wait'], 1)
    expected = e.step(state, 'wait')
    assert len({p['order_id'] for p in state['pots']}) == 2
    assert result['pots'] == e.public_state(expected)['pots']
    assert all(b['remaining'] == a['remaining'] - 1 for a, b in zip(state['pots'], result['pots']))
    assert result['food_freshness'] == e.public_state(expected)['food_freshness']
    assert result['raw_score_delta'] == -1 and result['completed_orders_delta'] == 0
    assert state == before


def test_counterfactual_prepared_expiry_uses_the_executed_wait_boundary():
    state = prepared_portion('tomato')
    for _ in range(19):
        state = e.step(state, 'wait')
    before = deepcopy(state)
    result = simulate(e, state, e.decide(state), ['wait'], 1)
    row = next(row for row in result['food_freshness'] if row['item_id'] == state['human']['holding']['id'])
    assert row['remaining'] == 0 and row['status'] == 'spoiled'
    assert any(event['type'] == 'ingredient_spoiled' for event in result['events'])
    assert result['raw_score_delta'] == -1
    assert state == before


@pytest.mark.parametrize('language', ('en', 'zh'))
@pytest.mark.parametrize('age', (0, 18))
def test_wait_answer_states_held_food_outcome_from_final_simulated_facts(language, age):
    state = prepared_portion('meat')
    for _ in range(age):
        state = e.step(state, 'wait')
    before = deepcopy(state)
    selected = plan(state, language=language, intents=[{'kind': 'counterfactual',
        'subject': 'human', 'purpose': 'comparison', 'temporal_scope': 'current',
        'evidence_ids': [], 'actions': ['wait', 'wait'], 'horizon': 2}])
    question = ('If I wait for the next two turns, what changes to my food, score and completed orders?'
                if language == 'en' else '如果接下来等待两回合，我的食物、得分和完成订单数会怎样变化？')
    result = explainer_for_plan(selected).answer(e, state, e.decide(state), question, language)
    expected = e.step(e.step(state, 'wait'), 'wait')
    fact = next(row for row in e.facts(expected) if row['id'] == 'freshness_meat')
    simulation = result['audit']['simulations'][0]
    assert result['status'] == 'answered'
    assert simulation['final_kitchen_facts'] == [fact]
    assert fact[language] in result['answer']
    assert ('After these simulated turns:' if language == 'en' else '在这些模拟回合之后：') in result['answer']
    assert expected['human']['holding']['id'] == state['human']['holding']['id']
    assert expected['human']['holding']['stage'] == ('prepared' if age == 0 else 'spoiled')
    remaining = 18 if age == 0 else 0
    assert (f'{remaining} turns remain' if language == 'en' else f'剩余 {remaining} 回合') in result['answer']
    assert simulation['raw_score_delta'] == -2 and simulation['completed_orders_delta'] == 0
    assert state == before


@pytest.mark.parametrize('language', ('en', 'zh'))
def test_serve_answer_uses_empty_final_hand_not_old_food_clock(language):
    state = next(s for s in regular_trace() if s['human']['holding']
                 and s['human']['holding']['stage'] == 'plated'
                 and (e._front(s['human']) or {}).get('id') == 'serve')
    selected = plan(state, language=language, intents=[{'kind': 'counterfactual',
        'subject': 'human', 'purpose': 'comparison', 'evidence_ids': [],
        'actions': ['interact'], 'horizon': 1}])
    result = explainer_for_plan(selected).answer(e, state, e.decide(state), 'What if I serve?', language)
    expected = e.step(state, 'interact')
    final_fact = next(row for row in e.facts(expected) if row['id'] == 'human_holding')
    simulation = result['audit']['simulations'][0]
    assert simulation['final_kitchen_facts'] == [final_fact]
    assert final_fact[language] in result['answer']
    assert simulation['human']['holding'] is None and simulation['raw_score_delta'] == 99


@pytest.mark.parametrize('language', ('en', 'zh'))
@pytest.mark.parametrize('combined,penalty,net', ((False, 5, -6), (True, 20, -21)))
def test_disposal_answer_separates_item_penalty_from_action_cost(language, combined, penalty, net):
    state = (next(s for s in regular_trace() if s['human']['holding']
                  and s['human']['holding']['stage'] == 'finished')
             if combined else prepared_portion('egg'))
    while (e._front(state['human']) or {}).get('id') != 'trash':
        assert not state['terminal']
        state = e.step(state, e._approach(state, 'human', 'trash'))
    before = deepcopy(state)
    selected = plan(state, language=language, intents=[{'kind': 'counterfactual',
        'subject': 'human', 'purpose': 'comparison', 'temporal_scope': 'current',
        'evidence_ids': [], 'actions': ['interact'], 'horizon': 1}])
    question = ('If I put this in the bin now, what penalty and total score change will I get?'
                if language == 'en' else '现在把手里的东西丢进垃圾桶，丢弃罚分和总分变化各是多少？')
    result = explainer_for_plan(selected).answer(e, state, e.decide(state), question, language)
    assert result['status'] == 'answered'
    simulation = result['audit']['simulations'][0]
    discarded = [event for event in simulation['events'] if event['type'] == 'waste']
    assert len(discarded) == 1 and discarded[0]['penalty'] == penalty
    assert simulation['raw_score_delta'] == simulation['task_score_delta'] == net
    assert simulation['completed_orders_delta'] == 0 and simulation['human']['holding'] is None
    assert (f'Trash disposal: −{penalty} points.' if language == 'en' else f'垃圾桶丢弃：扣 {penalty} 分。') in result['answer']
    assert (f'Task score changes by {net} points' if language == 'en' else f'任务分数变化{net}分') in result['answer']
    assert state == before


def test_question_catalog_never_calls_test_partner_recovery(monkeypatch):
    state = prepared_portion('meat')
    expected = e.human_advisor(state)
    def forbidden(_):
        raise AssertionError('Test-only partner leaked into public QA')
    monkeypatch.setattr(e, 'simulation_partner', forbidden)
    original = deepcopy(state)
    catalog = _catalog(e, state, e.decide(state), [])
    assert catalog['system:human_advice']['action'] == expected
    assert expected in e.legal_actions(state)
    assert state == original


@pytest.mark.parametrize('language', ('en', 'zh'))
def test_completed_order_question_uses_count_even_when_score_is_negative(language):
    state = next(s for s in regular_trace() if s['metrics']['completed_orders'] == 1)
    while state['raw_score'] >= 0:
        assert not state['terminal']
        state = e.step(state, 'wait')
    assert state['raw_score'] < 0
    result = answer(state, ['system:completed_orders'], 'How many dishes have we completed?', language)
    assert result['status'] == 'answered'
    assert ('completed 1 of 3 orders' if language == 'en' else '已完成 1 道，共 3 道订单') in result['answer']
    assert str(state['raw_score']) not in result['answer']


@pytest.mark.parametrize('language', ('en', 'zh'))
def test_this_turn_reason_is_bound_to_executed_pre_action_state(language):
    frames = regular_trace()
    before, after = next((before, after) for before, after in zip(frames, frames[1:])
                         if e.decide(before)['reason_code'] == 'reserve_rescue_window'
                         and e.decide(before) != e.decide(after))
    decision = e.decide(before)
    human = e.human_advisor(before)
    assert e.step(before, human, decision) == after
    selected = plan(after, language=language, intents=[{'kind': 'facts', 'subject': 'ai',
        'purpose': 'reason', 'temporal_scope': 'performed', 'evidence_ids': ['performed:system:ai_reason']}])
    result = explainer_for_plan(selected).answer(e, after, e.decide(after),
        'Why did you choose this action?' if language == 'en' else '这一回合为什么这样做？', language,
        action_context={'state': before, 'decision': decision, 'human_action': human})
    assert result['status'] == 'answered'
    assert result['audit']['decision_turn'] == before['turn']
    assert result['audit']['performed_action']['state_hash'] == _digest(before)
    assert result['audit']['performed_action']['decision_hash'] == _digest(decision)
    assert result['evidence_ids'] == ['performed:system:ai_reason']
    assert 'My next action' not in result['answer'] and '我下一步' not in result['answer']
