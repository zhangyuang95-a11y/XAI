"""Independent behavioral checks for the v6 AI-led Pong policy.

These synthetic fixtures are not evidence of a human explanation effect.
"""
from copy import deepcopy
import itertools
from domains.pong import turnbased as pong


def state_of(balls, human=2, ai=6):
    return pong._new_state([balls], seed=1, task=2, human=human, ai=ai)


def combo_state(turns=3):
    return state_of([pong._ball('team', 'cooperative', [2, 6], turns),
                     pong._ball('small', 'ordinary', [6], turns)], human=2, ai=4)


def test_combination_preference_can_override_larger_incompatible_small_reward():
    # The competing pile has four small points, but AI prioritizes its own
    # small+team contact. This explicitly is NOT a total-score optimal policy.
    balls = [pong._ball('team', 'cooperative', [3, 6], 1),
             pong._ball('overlap', 'ordinary', [6], 1)] + [
             pong._ball(f'alternative-{i}', 'ordinary', [2], 1) for i in range(4)]
    state = state_of(balls, human=2, ai=6)
    decision = pong.decide(state)
    assert decision['explanation_target']['small_ids'] == ['overlap']
    assert decision['explanation_target']['team_ids'] == ['team']
    result = pong.step(state, pong.human_advisor(state))
    assert result['raw_score'] == 4  # maximum raw points at this arrival is 5
    assert result['metrics']['cooperative_caught'] == 1


def test_initial_unreachable_partner_never_creates_combo_assignment():
    state = state_of([pong._ball('team', 'cooperative', [0, 6], 2),
                      pong._ball('small', 'ordinary', [6], 2)], human=3, ai=6)
    decision = pong.decide(state)
    assert decision['assignments'] == []
    assert decision['explanation_target']['team_ids'] == []
    assert decision['explanation_target']['small_ids'] == ['small']
    assert 'team ball' not in decision['reason_en']


def test_human_deviation_never_reverses_frozen_ai_route_and_preserves_small():
    state = combo_state()
    first = pong.decide(state)
    assert first['memory']['commitment']['ai_contact'] == 6
    state = pong.step(state, 'left')
    # Another step away makes the partner's end unreachable. Every possible
    # human continuation shares the originally frozen AI route to lane 7.
    state = pong.step(state, 'left')
    decision = pong.decide(state)
    assert decision['memory']['commitment']['ai_contact'] == 6
    assert not decision['assignment_changed'] and not decision['assignment_released']
    assert decision['explanation_target']['human_reachable'] is False
    assert 'can no longer reach' in decision['reason_en']
    end = pong.step(state, 'wait')
    assert end['ai']['x'] == 6
    assert {e['ball_id']: e['points'] for e in end['events']} == {'team': 0, 'small': 1}
    text = pong.performed_reason(state, decision, end)
    assert 'caught small, but missed team' in text['en']
    assert text['en'].count('.') == 2


def test_all_human_paths_share_fixed_ai_route_until_selected_arrival():
    initial = combo_state(4)
    expected = pong.decide(initial)['memory']['ai_route']
    for actions in itertools.product(('left', 'wait', 'right'), repeat=4):
        state, positions = deepcopy(initial), []
        for action in actions:
            if action not in pong.legal_actions(state):
                break
            state = pong.step(state, action)
            positions.append(state['ai']['x'])
        assert positions == expected[:len(positions)]


def test_single_small_goal_remains_stable_when_human_moves_toward_it():
    state = state_of([pong._ball('small', 'ordinary', [4], 3)], human=0, ai=6)
    before = pong.decide(state)
    assert before['explanation_target']['small_ids'] == ['small']
    next_state = pong.step(state, 'right')
    redirected = deepcopy(next_state)
    redirected['human']['x'] = 4
    assert pong.decide(next_state)['action'] == pong.decide(redirected)['action']
    assert pong.decide(next_state)['explanation_target']['small_ids'] == ['small']
    assert pong.decide(redirected)['explanation_target']['small_ids'] == ['small']


def test_initial_combo_only_sees_current_visible_balls_and_no_public_goal_leak():
    state = pong.initial_state(731100, 2)
    changed = deepcopy(state)
    changed['_schedule'] = []
    changed['seed'] = -10
    assert pong.decide(changed) == pong.decide(state)
    assert 'explanation_target' not in pong.public_state(state)
    assert 'policy_memory' not in pong.public_state(state)
    assert 'ai_route' not in pong.public_state(state)


def test_real_combo_performed_explanation_uses_displayed_countdown_and_saved_action():
    state = pong.initial_state(731100, 2)
    for _ in range(4):
        state = pong.step(state, pong.human_advisor(state))
    decision = pong.decide(state)
    assert decision['explanation_target']['small_ids'] and decision['explanation_target']['team_ids']
    after = pong.step(state, pong.human_advisor(state))
    text = pong.performed_reason(state, decision, after)
    assert 'I moved left' in text['en'] and 'in 1 turn' in text['en']
    assert '这一步我左移' in text['zh'] and '1回合后' in text['zh']
    assert text['en'].count('.') == text['zh'].count('。') == 2
    for lang in ('en', 'zh'):
        for banned in ('highest', 'points', 'commitment', '最高', '得分', '承诺'):
            assert banned not in text[lang]


def test_fallback_is_honest_about_sequential_not_simultaneous_targets():
    state = state_of([pong._ball('team', 'cooperative', [2, 6], 4),
                      pong._ball('small', 'ordinary', [5], 2)])
    decision = pong.decide(state)
    assert 'Then I cover team ball' in decision['reason_en']
    assert 'together' not in decision['reason_en']
    assert '同时' not in decision['reason_zh']


def test_completed_combo_target_expires_and_new_target_is_selected():
    state = combo_state(2)
    state = pong.step(state, pong.human_advisor(state))
    state = pong.step(state, pong.human_advisor(state))
    assert state['policy_memory'] == {}
    assert pong.decide(state)['goal'] == 'task_complete'
