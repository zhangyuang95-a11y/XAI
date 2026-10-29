"""Synthetic transition tests for temporal evidence and public non-disclosure.

These do not confer model admission or measure human collaboration.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from runtime_v642.contracts import RuntimeUnavailable
from runtime_v642.store import StudyError, encode
spec = importlib.util.spec_from_file_location('history_synthetic_core_fixture',
    Path(__file__).with_name('test_runtime_v642_store.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
configured = helper.configured
enrollment, demo_and_start, finish = helper.enrollment, helper.demo_and_start, helper.finish


def task2(store, *, group='A', participant='history-test'):
    token, iid, identity = enrollment(store, group=group, participant=participant)
    demo_and_start(store, token, iid)
    finish(store, token, iid, 1)
    store.start_task(token, iid)
    return token, iid, identity


def test_head_next_action_is_private_but_committed_history_and_last_action_are_public(configured):
    store, _, _, _, _ = configured
    token, iid, _ = enrollment(store)
    initial = demo_and_start(store, token, iid)
    next_action = initial['current']['decision']['action']
    public = store.participant_view(token, iid, initial)
    assert public['current']['decision'] == {}
    assert public['current']['decision_committed'] is False
    assert public['current']['last_executed_action'] is None
    assert next_action not in encode(public)
    after = store.action(token, iid, command_id='first', expected_turn=0, human_action='up')
    next_unexecuted = after['current']['decision']['action']
    public = store.participant_view(token, iid, after)
    assert public['current']['decision'] == {}
    assert public['current']['last_executed_action'] == dict(
        decision_turn=0, result_turn=1, human_action='up', ai_action=next_action)
    assert next_unexecuted not in encode(public)
    historical = store.participant_view(token, iid, store.frame(token, iid, task=1, turn=0))
    assert historical['decision']['action'] == next_action and historical['decision_committed']
    assert 'state' not in historical and 'transition' not in historical


def test_default_latest_and_explicit_alias_only_select_executed_decisions(configured):
    store, _, _, _, _ = configured
    token, iid, _ = task2(store)
    for coordinates in ({}, dict(turn=0), dict(decision_turn=0)):
        with pytest.raises(StudyError, match='requires_executed_action'):
            store.ask(token, iid, question='Why just now?', **coordinates)
    after = store.action(token, iid, command_id='task2-first', expected_turn=0, human_action='up')
    with pytest.raises(StudyError, match='requires_executed_action'):
        store.ask(token, iid, question='Next move?', decision_turn=1)
    with pytest.raises(StudyError, match='ambiguous_question_turn'):
        store.ask(token, iid, question='Why?', turn=0, decision_turn=0)
    latest = store.ask(token, iid, question='Why just now?')
    explicit = store.ask(token, iid, question='Why step 1?', turn=0)
    assert latest['decision_turn'] == explicit['decision_turn'] == 0
    assert latest['result_turn'] == 1 and latest['through_turn_at_question'] == 1
    assert latest['human_action'] == 'up'
    assert latest['nn_action'] != after['current']['decision']['action']
    assert latest['frame_sha256'] == explicit['frame_sha256']
    assert latest['result_frame_sha256'] == after['current']['frame_sha256']
    assert latest['nn_action'] == store.frame(token, iid, task=2, turn=0)['decision']['action']


def test_active_Task2_can_explain_real_Task1_with_same_bound_model_and_audited_reader(configured):
    store, adapters, _, _, _ = configured
    token, iid, identity = task2(store)
    adapters['warehouse'].evidence_mode = 'valid'
    answer = store.ask(token, iid, question='Why at the end of Task 1?', target_task=1)
    assert answer['target_task'] == 1 and answer['decision_turn'] == 1 and answer['result_turn'] == 2
    assert answer['status'] == 'grounded_pattern'
    assert answer['binding'] == identity['binding']
    assert answer['nn_action'] == store.frame(token, iid, task=1, turn=1)['decision']['action']
    assert answer['evidence']['nn_action'] == answer['nn_action']
    assert store.answers(token, iid)['questions'][0]['question_id'] == answer['question_id']
    assert store.export_instance(token, iid, admin=True)['records']['questions'][0]['turn'] == 1
    with pytest.raises(StudyError, match='target_must_be'):
        store.ask(token, iid, question='Why in Task 3?', target_task=3)


def test_counterfactual_cannot_simulate_past_the_commitment_at_question_time(configured):
    store, _, _, _, _ = configured
    token, iid, _ = task2(store)
    store.action(token, iid, command_id='task2-first', expected_turn=0, human_action='wait')
    with pytest.raises(StudyError, match='exceeds_committed_history'):
        store.ask(token, iid, question='What if?', decision_turn=0, human_actions=['up', 'wait'])
    assert store.db.execute('SELECT COUNT(*) FROM v642_questions').fetchone()[0] == 0
    accepted = store.ask(token, iid, question='What if?', decision_turn=0, human_actions=['up'])
    assert accepted['counterfactual']['initial_state']['turn'] == 0
    assert accepted['counterfactual']['final_state']['turn'] == 1
    assert store.resume(token, iid)['current']['turn'] == 1
    old = store.ask(token, iid, question='Task1 alternative?', target_task=1, decision_turn=0,
                    human_actions=['up', 'wait'])
    assert old['counterfactual']['final_state']['turn'] == old['through_turn_at_question'] == 2


@pytest.mark.parametrize('group', ['B', 'C'])
def test_B_C_cannot_query_any_history_or_retrieve_answers(configured, group):
    store, adapters, _, _, _ = configured
    token, iid, _ = task2(store, group=group)
    count = len(adapters['warehouse'].calls)
    for action in (lambda: store.ask(token, iid, question='Why?', target_task=1),
                   lambda: store.answers(token, iid)):
        with pytest.raises(StudyError, match='only_A'):
            action()
    assert not any(name == 'explanation' for name, _ in adapters['warehouse'].calls[count:])


def test_completed_Task2_and_Task3_revoke_even_a_previously_captured_answer(configured):
    store, _, _, _, _ = configured
    token, iid, _ = task2(store)
    store.action(token, iid, command_id='task2-first', expected_turn=0, human_action='wait')
    saved = store.ask(token, iid, question='Why?')
    store.action(token, iid, command_id='task2-last', expected_turn=1, human_action='wait')
    for transition in (False, True):
        if transition:
            store.start_task(token, iid)
        for action in (lambda: store.ask(token, iid, question='Why?', target_task=1),
                       lambda: store.answers(token, iid),
                       lambda: store.participant_view(token, iid, saved)):
            with pytest.raises(StudyError, match='only_A'):
                action()
    # A researcher retains historical data for auditing after participant access expires.
    assert store.export_instance(token, iid, admin=True)['records']['questions']


def test_audit_rejects_counterfactual_that_uses_later_commits_than_original_question(configured):
    store, _, registry, _, _ = configured
    token, iid, _ = task2(store)
    store.action(token, iid, command_id='task2-first', expected_turn=0, human_action='wait')
    saved = store.ask(token, iid, question='What if?', human_actions=['up'])
    store.action(token, iid, command_id='task2-last', expected_turn=1, human_action='wait')
    bound = registry.bind('warehouse', 'A', mode='test')
    before = store.frame(token, iid, task=2, turn=0)['state']
    saved['counterfactual'] = bound.adapter.counterfactual(before, ['up', 'up'], bound.binding)
    store.db.execute('DROP TRIGGER v642_questions_update_immutable')
    store.db.execute('UPDATE v642_questions SET result_json=? WHERE id=?', (encode(saved), saved['question_id']))
    with pytest.raises(RuntimeUnavailable, match='stored QA counterfactual'):
        store.export_instance(token, iid, admin=True)


def test_history_program_context_does_not_change_binding_or_grant_B_C_Task3_access(configured, monkeypatch):
    _, _, registry, _, _ = configured
    seen = []
    marker = object()
    monkeypatch.setattr(registry.catalog, 'program', lambda binding, role: seen.append((binding, role)) or marker)
    a = registry.bind('warehouse', 'A', mode='test')
    assert a.explanation_program(task=1) is None
    context = replace(a, question_context_task=2)
    assert context.binding.to_dict() == a.binding.to_dict()
    assert context.explanation_program(task=1) is marker
    for group in ('B', 'C'):
        assert replace(registry.bind('warehouse', group, mode='test'), question_context_task=2).explanation_program(task=1) is None
    assert context.explanation_program(task=3) is None
    with pytest.raises(RuntimeUnavailable, match='invalid explanation context'):
        replace(a, question_context_task=3).explanation_program(task=1)
    assert len(seen) == 1


# Reuse the local synthetic HTTP fixture; no live service or model is contacted.
http_spec = importlib.util.spec_from_file_location('history_synthetic_HTTP_fixture',
    Path(__file__).with_name('test_runtime_v642_server.py'))
http_helper = importlib.util.module_from_spec(http_spec)
http_spec.loader.exec_module(http_helper)
service = http_helper.service


def test_HTTP_history_coordinates_answer_revocation_and_no_head_action(service):
    _, _, call = service
    status, enrolled = call('/v642/instances', dict(domain='warehouse', participant_reference='history-HTTP',
                                                  group='A'), administrator=True)
    assert status == 200
    token, iid = enrolled['session_token'], enrolled['instance']['id']
    base = '/v642/instances/'+iid
    assert call(base+'/demonstration/complete', {}, token=token)[0] == 200
    status, initial = call(base+'/tasks', {}, token=token)
    assert status == 200 and initial['current']['decision'] == {}
    assert initial['current']['last_executed_action'] is None
    for turn in (0, 1):
        status, after = call(base+'/actions', dict(command_id=f'history-1-{turn}', expected_turn=turn,
                                                 human_action='wait'), token=token)
        assert status == 200 and after['current']['decision'] == {}
        assert after['current']['last_executed_action']['decision_turn'] == turn
    assert call(base+'/tasks', {}, token=token)[0] == 200
    assert call(base+'/answers', token=token) == (200, {'questions': []})
    status, old = call(base+'/questions', dict(question='Why in Task1?', target_task=1,
                         decision_turn=0, human_actions=['up']), token=token)
    assert status == 200 and old['target_task'] == 1 and old['result_turn'] == 1
    assert old['counterfactual']['contains_restorable_private_snapshot'] is False
    assert call(base+'/questions', dict(question='Why now?'), token=token)[0] == 409
    assert call(base+'/actions', dict(command_id='history-2-0', expected_turn=0, human_action='up'), token=token)[0] == 200
    status, latest = call(base+'/questions', dict(question='Why just now?'), token=token)
    assert status == 200 and latest['target_task'] == 2 and latest['decision_turn'] == 0
    assert call(base+'/questions', dict(question='Why?', turn=0, decision_turn=0), token=token)[0] == 400
    assert call(base+'/questions', dict(question='Next?', decision_turn=1), token=token)[0] == 409
    assert call(base+'/questions', dict(question='Future?', human_actions=['up', 'wait']), token=token)[0] == 409
    assert len(call(base+'/answers', token=token)[1]['questions']) == 2
    assert call(base+'/actions', dict(command_id='history-2-1', expected_turn=1, human_action='wait'), token=token)[0] == 200
    assert call(base+'/answers', token=token)[0] == 403
    assert call(base+'/tasks', {}, token=token)[0] == 200
    assert call(base+'/answers', token=token)[0] == 403
