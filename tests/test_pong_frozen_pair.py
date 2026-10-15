"""The rolling Pong release uses two trained actors and a frozen AI actor."""
from copy import deepcopy
import json

import pytest

from domains.pong import nn_policy, turnbased as pong


def test_pair_transition_accepts_two_external_actions_without_policy(monkeypatch):
    state = pong.initial_state(730100, 1)
    monkeypatch.setattr(pong, 'decide', lambda _state: (_ for _ in ()).throw(AssertionError('called policy')))
    monkeypatch.setattr(pong, '_plan', lambda _obs: (_ for _ in ()).throw(AssertionError('called planner')))
    result = pong.step_pair(state, 'right', 'left')
    assert result['human']['x'] == state['human']['x'] + 1
    assert result['ai']['x'] == state['ai']['x'] - 1
    assert result['turn'] == 1
    assert state['turn'] == 0


def test_frozen_ai_action_is_model_argmax_and_not_rule_planner(monkeypatch):
    state = pong.initial_state(730100, 2)
    monkeypatch.setattr(pong, '_plan', lambda _obs: (_ for _ in ()).throw(AssertionError('rule planner used')))
    decision = pong.decide(state)
    assert decision['action'] == nn_policy.choose(state, 'ai')
    assert decision['human_action'] == nn_policy.choose(state, 'human')
    assert decision['model_sha256'] == nn_policy.model_sha256()
    assert pong.step(state, 'wait', decision)['ai']['x'] == state['ai']['x'] + pong._DELTAS[decision['action']]
    payload = json.loads(nn_policy.MODEL_PATH.read_text())
    assert nn_policy.choose_from_payload(state, 'ai', payload) == decision['action']


def test_boundary_mask_and_hidden_schedule_independence():
    state = pong.initial_state(730101, 2)
    state['ai']['x'] = 0
    assert nn_policy.choose(state, 'ai') in ('wait', 'right')
    changed = deepcopy(state)
    changed['seed'] = -1
    for entry in changed['_schedule']:
        if entry['spawn_turn'] > state['turn']:
            entry['ball']['contacts'] = [8] if entry['ball']['kind'] == 'ordinary' else [0, 8]
    changed['experimental_group'] = 'B'
    changed['unsubmitted_human_action'] = 'left'
    assert pong.decide(state) == pong.decide(changed)


def test_human_takes_over_without_choosing_simultaneous_ai_action():
    state = pong.initial_state(730102, 1)
    decision = pong.decide(state)
    positions = {pong.step(state, action, decision)['ai']['x'] for action in pong.legal_actions(state)}
    assert positions == {state['ai']['x'] + pong._DELTAS[decision['action']]}


def test_replay_is_exact_and_forged_decision_rejected():
    a = pong.initial_state(730103, 1)
    b = deepcopy(a)
    for _ in range(10):
        action = pong.human_advisor(a)
        decision = pong.decide(a)
        a = pong.step(a, action, decision)
        b = pong.step(b, action, pong.decide(b))
        assert a == b
    bad = deepcopy(pong.decide(a))
    bad['action'] = 'left' if bad['action'] != 'left' else 'right'
    with pytest.raises(ValueError):
        pong.step(a, 'wait', bad)


def test_missing_or_mismatched_acceptance_prevents_play(monkeypatch, tmp_path):
    model = tmp_path / 'actor.json'
    model.write_bytes(nn_policy.MODEL_PATH.read_bytes())
    receipt = tmp_path / 'acceptance.json'
    receipt.write_text(json.dumps({'model_sha256': '0' * 64,
                                   'development_passed': True, 'heldout_passed': True}))
    monkeypatch.setattr(nn_policy, 'MODEL_PATH', model)
    monkeypatch.setattr(nn_policy, 'ACCEPTANCE_PATH', receipt)
    nn_policy.load_model.cache_clear()
    with pytest.raises(RuntimeError, match='acceptance'):
        pong.decide(pong.initial_state(730100, 1))
    nn_policy.load_model.cache_clear()


def test_explanations_report_physics_without_fabricated_assignment():
    state = pong.initial_state(730104, 2)
    decision = pong.decide(state)
    facts = {row['id']: row for row in pong.facts(state, decision)}
    for ball in state['balls']:
        assert facts['ball_plan:' + ball['id']]['plan_status'] == 'no_committed_assignment'
    assert 'not a promised catch assignment' in decision['reason_en']
    after = pong.step(state, 'wait', decision)
    performed = pong.performed_reason(state, decision, after)
    assert 'frozen teammate' in performed['en']
    assert 'selected side' not in performed['en']


def test_original_ball_physics_and_scoring_are_preserved():
    balls = [pong._ball('team', 'cooperative', [2, 6], 1),
             pong._ball('small-left', 'ordinary', [2], 1),
             pong._ball('small-right', 'ordinary', [6], 1)]
    state = pong._new_state([balls], seed=1, task=2, human=2, ai=6)
    after = pong.step_pair(state, 'wait', 'wait')
    assert after['raw_score'] == 5
    assert {event['ball_id']: event['points'] for event in after['events']} == {
        'team': 3, 'small-left': 1, 'small-right': 1}
    missed = pong.step_pair(pong._new_state([[pong._ball('team', 'cooperative', [2, 6], 1)]],
                                           seed=1, task=2, human=2, ai=2), 'wait', 'wait')
    assert missed['raw_score'] == 0
    assert pong.score(after)['task_score'] == 100


def test_fixed_schedule_public_boundary_and_finite_task():
    state = pong.initial_state(730100, 2)
    assert (state['max_turns'], state['height'], state['lanes']) == (90, 12, 9)
    assert state['total_possible_points'] == 180
    assert len(state['balls']) == 10
    public = pong.public_state(state)
    for private in ('_schedule', 'seed', 'policy_memory', 'model_sha256'):
        assert private not in public
    before = {ball['id']: ball for ball in public['balls']}
    next_state = pong.step_pair(state, 'wait', 'wait')
    for ball in next_state['balls']:
        assert ball['y'] == before[ball['id']]['y'] + before[ball['id']]['vy']
    second = pong.step_pair(next_state, 'wait', 'wait')
    assert any(event['type'] == 'balls_spawned' for event in second['events'])
    assert {entry['ball']['id'] for entry in state['_schedule']} == {
        entry['ball']['id'] for entry in second['_schedule']}


def test_acceptance_report_covers_both_splits_and_all_tasks():
    report = json.loads(nn_policy.EVALUATION_PATH.read_text())
    assert report['model_sha256'] == nn_policy.model_sha256()
    assert report['runtime_parity'] is True
    assert report['training']['selected_self_play_round'] >= 1
    assert report['training']['neural_joint_steps'] > 0
    training_rounds = report['training']['self_play_seed_schedule']
    assert len(training_rounds) == report['training']['self_play_rounds']
    selection_seeds = {row['seed'] for row in report['development']['rows'] + report['heldout']['rows']}
    assert all(set(round_seeds).isdisjoint(selection_seeds) for round_seeds in training_rounds)
    assert len(report['development']['rows']) == len(report['heldout']['rows']) == 72
    for split in ('development', 'heldout'):
        assert report[split]['passed']
        for task in report[split]['summaries']:
            assert task['mean_capacity_percent'] >= 90
            assert task['min_capacity_percent'] >= 80
            assert task['mean_ai_wait_rate'] <= .10
            assert task['max_ai_wait_rate'] <= .20
    assert len(report['human_replacement_proxies']['rows']) == 288
