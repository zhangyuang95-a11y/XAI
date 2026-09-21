"""Current Kitchen evidence and counterfactual clocks; no semantic-model claims."""
from copy import deepcopy

import pytest

from domains.kitchen import engine as e
from domains.kitchen.build_qa_cases import regular_trace
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
    points = answer(state, ['serving_score_rule'], 'Why +30 reward but only +29 net?', language)
    assert fresh['status'] == points['status'] == 'answered'
    assert '20' in fresh['answer']
    assert all(str(n) in points['answer'] for n in (30, 29, 28))
    if language == 'en':
        assert 'reset' in fresh['answer'] and len(fresh['answer'].split()) < 80
        assert len(points['answer'].split()) < 85
    else:
        assert '重置' in fresh['answer'] and len(fresh['answer']) < 180
        assert len(points['answer']) < 180
    assert '+100' not in points['answer'] and '500' not in points['answer']


def test_counterfactual_preserves_actual_both_pan_decrements_without_live_mutation():
    frames = regular_trace(1000)
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
    assert state['raw_score'] < 0
    result = answer(state, ['system:completed_orders'], 'How many dishes have we completed?', language)
    assert result['status'] == 'answered'
    assert ('completed 1 of 5 orders' if language == 'en' else '已完成 1 道，共 5 道订单') in result['answer']
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
