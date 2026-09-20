"""Pong target explanations use saved decisions and independently known geometry.

Injected intent plans test the evidence/rendering protocol, not LLM understanding.
"""
from copy import deepcopy

import pytest

from domains.pong import turnbased as pong
from study_v3.qa import _performed_catalog, simulate
from tests.test_study_v3_explanation_frame_v36 import ask, fact
from tests.test_study_v3_qa import ask as ask_current, explainer_for_plan, plan


def combination(*, remaining=3, ai=4):
    # Displayed lane 7 catches both small s-combo and the right end of t-combo;
    # displayed lane 3 is the human's separate, required team-ball contact.
    balls = [pong._ball('t-combo', 'cooperative', [2, 6], remaining),
             pong._ball('s-combo', 'ordinary', [6], remaining)]
    return pong._new_state([balls], seed=123, task=2, human=2, ai=ai)


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_performed_combination_reason_is_short_and_counts_from_displayed_frame(language):
    before = combination()
    decision = pong.decide(before)
    assert decision['action'] == 'right'
    after = pong.step(before, 'wait', decision)
    context = {'state': before, 'decision': decision, 'human_action': 'wait'}
    original = deepcopy((before, decision, after))
    result = ask(pong, after, context, [fact('performed')], language=language)
    assert result['status'] == 'answered'
    answer = result['answer'].split('\n\n', 1)[1]
    assert ('2 turns' if language == 'en' else '2回合') in answer
    assert ('lane 7' if language == 'en' else '第7道') in answer
    assert ('lane 3' if language == 'en' else '第3道') in answer
    assert 's-combo' in answer and 't-combo' in answer
    assert '本步行动前的判断' not in answer and 'reasoning before' not in answer
    assert 'highest' not in answer and '最高' not in answer
    assert '+ ' not in answer and '合计' not in answer
    assert answer.count('。' if language == 'zh' else '.') <= 2
    assert result['audit']['decision_turn'] == 0
    assert result['audit']['target_display_turn'] == 1
    assert result['evidence_ids'] == ['performed:system:ai_reason']
    assert (before, decision, after) == original


@pytest.mark.parametrize('language', ['en', 'zh'])
@pytest.mark.parametrize('human_action,caught', [('wait', True), ('left', False)])
def test_arrived_target_is_past_tense_and_never_invents_cooperative_success(language, human_action, caught):
    before = combination(remaining=1, ai=6)
    decision = pong.decide(before)
    after = pong.step(before, human_action, decision)
    events = {event['ball_id']: event['type'] for event in after['events'] if 'ball_id' in event}
    assert events['s-combo'] == 'caught'
    assert events['t-combo'] == ('caught' if caught else 'missed')
    result = ask(pong, after, {'state': before, 'decision': decision, 'human_action': human_action},
                 [fact('performed')], language=language)
    assert result['status'] == 'answered'
    answer = result['answer'].split('\n\n', 1)[1]
    assert 's-combo' in answer and 't-combo' in answer
    assert '0 turns' not in answer and '0回合后' not in answer
    assert '1 turns' not in answer and '1回合后' not in answer
    assert 'this turn' in answer.lower() if language == 'en' else '本回合' in answer
    if not caught:
        assert ('missed' in answer.lower() or 'not caught' in answer.lower()) if language == 'en' else ('没接住' in answer or '未接住' in answer or '漏接' in answer)
    assert answer.count('。' if language == 'zh' else '.') <= 2


def test_performed_reason_uses_saved_target_after_next_target_changes():
    before = combination(remaining=1, ai=6)
    decision = pong.decide(before)
    after = pong.step(before, 'wait', decision)
    assert after['terminal']
    assert pong.decide(after)['goal'] == 'task_complete'
    reason = _performed_catalog(pong, after, {'state': before, 'decision': decision, 'human_action': 'wait'})
    assert 's-combo' in reason['performed:system:ai_reason']['en']
    assert 't-combo' in reason['performed:system:ai_reason']['en']
    assert 'no next action' not in reason['performed:system:ai_reason']['en']


def test_position_counterfactual_reruns_real_policy_and_keeps_committed_combination():
    state = combination(remaining=3)
    state['policy_memory'] = deepcopy(pong.decide(state)['memory'])
    original = deepcopy(state)
    actual = pong.decide(state)
    branch = deepcopy(state)
    branch['human']['x'] = 8
    expected = pong.decide(branch)
    assert actual['action'] == expected['action'] == 'right'
    simulated = simulate(pong, state, actual, [], 0, intervention={'human_lane': 9})
    assert simulated['steps_completed'] == 0
    assert simulated['intervention']['ai_action'] == expected['action']
    assert simulated['intervention']['reason_en'] == expected['reason_en']
    assert simulated['intervention']['target_unchanged'] is True
    assert simulated['raw_score_delta'] == 0
    assert state == original


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_ai_motion_counterfactual_renders_only_requested_moves_and_selected_catches(language):
    state = combination(remaining=1, ai=6)
    # A second catch is public and real, but is unrelated to the AI's target.
    extra = pong._ball('human-only', 'ordinary', [2], 1)
    state['balls'].append(extra)
    state['_schedule'].append({'spawn_turn': 0, 'arrival_turn': 1, 'ball': extra})
    state['total_possible_points'] += 1
    intent = {'kind': 'counterfactual', 'subject': 'human', 'purpose': 'comparison',
              'evidence_ids': [], 'actions': ['wait'], 'horizon': 1,
              'answer_focus': 'ai_action'}
    result = ask_current(explainer_for_plan(plan(state, language=language, intents=[intent])), state,
                         question='If I wait, what will you do?')
    assert result['status'] == 'answered'
    simulation = result['audit']['simulations'][0]
    assert simulation['trace'][0]['ai_action'] == 'wait'
    assert simulation['raw_score_delta'] == 5
    assert len([event for event in simulation['events'] if event['type'] == 'caught']) == 3
    assert 's-combo' in result['answer'] and 't-combo' in result['answer']
    assert 'human-only' not in result['answer']
    assert 'task score changes' not in result['answer'].lower() and '任务得分变化' not in result['answer']
    assert ('lane 7' if language == 'en' else '第7道') in result['answer']


@pytest.mark.parametrize('bad_focus', ['anything', True, None])
def test_counterfactual_answer_focus_is_validated(bad_focus):
    state = combination()
    intent = {'kind': 'counterfactual', 'subject': 'human', 'purpose': 'comparison',
              'evidence_ids': [], 'actions': ['wait'], 'horizon': 1, 'answer_focus': bad_focus}
    result = ask_current(explainer_for_plan(plan(state, intents=[intent])), state)
    assert result['status'] == 'unavailable'
    assert result['audit']['failure_code'] == 'invalid_answer_focus'
