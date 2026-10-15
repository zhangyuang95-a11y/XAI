"""Incoming-action binding versus current facts; injected plans are protocol tests."""
from copy import deepcopy
import json
from uuid import uuid4

import pytest

from study_v3.config import Settings
from study_v3.qa import Explainer, _catalog, _digest
from study_v3.registry import engine
from study_v3.store import Store, encode
from tests.test_study_v3_store import Flow, RecordingExplainer


def selected_plan(state, intents, language='en'):
    return {'language': language, 'binding': {'task': state['task'], 'turn': state['turn']},
            'premise': 'supported', 'clarification': None, 'intents': intents}


def fact(scope, purpose='reason', ids=None, subject='ai'):
    return {'kind': 'facts', 'subject': subject, 'purpose': purpose,
            'temporal_scope': scope, 'evidence_ids': ids or [
                ('performed:' if scope == 'performed' else '') + 'system:ai_' + purpose]}


def ask(eng, state, context, intents, *, language='en', capture=None):
    exp = Explainer(Settings(database=':memory:'))
    plan = selected_plan(state, intents, language)
    def request(payload):
        if capture is not None:
            capture.append(deepcopy(payload))
        return deepcopy(plan), {'reported_model': 'injected-protocol-plan-not-semantic-evaluation'}
    exp._request_plan = request
    result = exp.answer(eng, state, eng.decide(state), 'Why did you do that?', language,
                        public_history=[eng.public_state(state)], action_context=context)
    return result


def transition(domain):
    eng = engine(domain)
    before = eng.initial_state(730101, 2)
    # Pick a real boundary where the next chosen action differs. This would
    # expose the original one-step offset, rather than testing two equal waits.
    for _ in range(150):
        chosen = eng.decide(before)
        action = eng.human_advisor(before)
        after = eng.step(before, action, chosen)
        if not after['terminal'] and chosen['action'] != eng.decide(after)['action']:
            return eng, before, after, {'state': before, 'decision': chosen, 'human_action': action}
        before = after
        assert not before['terminal'], domain
    pytest.fail('No meaningful decision boundary found: ' + domain)


@pytest.mark.parametrize('domain', ('warehouse', 'pong', 'kitchen'))
@pytest.mark.parametrize('language', ('en', 'zh'))
def test_current_why_uses_saved_incoming_action_while_next_uses_displayed_frame(domain, language):
    eng, before, displayed, context = transition(domain)
    original = deepcopy((before, displayed, context))
    captured = []
    performed = ask(eng, displayed, context, [fact('performed')], language=language, capture=captured)
    future = ask(eng, displayed, context, [fact('current', 'action')], language=language)
    assert performed['status'] == future['status'] == 'answered'
    assert performed['audit']['decision_turn'] == displayed['turn'] - 1
    assert performed['audit']['target_display_turn'] == displayed['turn']
    assert performed['audit']['performed_action']['ai_action'] == context['decision']['action']
    assert performed['audit']['performed_action']['state_hash'] == _digest(before)
    assert performed['audit']['performed_action']['decision_hash'] == _digest(context['decision'])
    assert future['audit']['decision_turn'] == displayed['turn']
    assert performed['evidence_ids'] == ['performed:system:ai_reason']
    assert future['evidence_ids'] == ['system:ai_action']
    assert performed['answer'].split('\n')[0].endswith(str(displayed['turn']))
    assert '我下一步' not in performed['answer'] and 'My next action' not in performed['answer']
    assert captured[0]['public_observation'] == eng.public_state(displayed)
    assert captured[0]['performed_action']['result_turn'] == displayed['turn']
    assert original == (before, displayed, context)


@pytest.mark.parametrize('domain', ('warehouse', 'pong', 'kitchen'))
def test_current_advice_observations_and_counterfactual_do_not_shift_back(domain):
    eng, before, displayed, context = transition(domain)
    current_fact = {'warehouse': 'human_state', 'pong': 'position', 'kitchen': 'human_holding'}[domain]
    current_evidence = _catalog(eng, displayed, eng.decide(displayed), [])
    intents = [fact('current', 'observation', [current_fact], 'shared'),
               fact('current', 'advice', ['system:human_advice'], 'human'),
               {'kind': 'counterfactual', 'temporal_scope': 'current', 'subject': 'human',
                'purpose': 'comparison', 'evidence_ids': [], 'actions': ['wait'], 'horizon': 1}]
    result = ask(eng, displayed, context, intents)
    assert result['status'] == 'answered'
    assert current_evidence[current_fact]['en'] in result['answer']
    assert current_evidence['system:human_advice']['en'] in result['answer']
    simulated = result['audit']['simulations'][0]
    assert simulated['input_state_hash'] == _digest(displayed)
    assert simulated['trace'][0]['ai_action'] == eng.decide(displayed)['action']
    assert result['audit']['decision_turn'] == displayed['turn']


@pytest.mark.parametrize('domain', ('warehouse', 'pong', 'kitchen'))
def test_initial_frame_has_no_executed_action_but_next_action_is_available(domain):
    eng = engine(domain)
    state = eng.initial_state(730101, 2)
    context = {'state': None, 'decision': None, 'human_action': None}
    result = ask(eng, state, context, [fact('performed')])
    assert result['status'] == 'answered'
    assert 'No action has happened' in result['answer']
    assert result['audit']['decision_turn'] is None
    assert result['audit']['performed_action']['source'] == 'initial_frame_no_action'
    assert ask(eng, state, context, [fact('current', 'action')])['status'] == 'answered'


@pytest.mark.parametrize('domain', ('warehouse', 'pong', 'kitchen'))
def test_performed_comparison_uses_before_action_alternatives(domain):
    eng, before, displayed, context = transition(domain)
    alternative = next(row for row in eng.facts(before, context['decision']) if row['id'].startswith('alternative'))
    result = ask(eng, displayed, context, [fact('performed', 'comparison', ['performed:' + alternative['id']])])
    assert result['status'] == 'answered'
    assert alternative['en'] in result['answer']
    assert 'Before this turn' in result['answer']
    assert result['audit']['decision_turn'] == before['turn']


@pytest.mark.parametrize('bad', [
    fact('performed', ids=['system:ai_reason']),
    fact('current', ids=['performed:system:ai_reason']),
    {'kind': 'facts', 'subject': 'ai', 'purpose': 'reason', 'evidence_ids': ['system:ai_reason']},
    {'kind': 'counterfactual', 'subject': 'human', 'purpose': 'comparison', 'temporal_scope': 'performed',
     'actions': ['wait'], 'horizon': 1, 'evidence_ids': []},
])
def test_wrong_or_missing_temporal_scope_is_rejected_after_one_repair(bad):
    eng, before, displayed, context = transition('pong')
    captured = []
    result = ask(eng, displayed, context, [bad], capture=captured)
    assert result['status'] == 'unavailable'
    assert len(captured) == 2
    assert 'repair_request' in captured[-1]


def test_mixed_past_reason_and_next_action_retains_both_source_turns():
    eng, before, displayed, context = transition('pong')
    result = ask(eng, displayed, context, [fact('performed'), fact('current', 'action')])
    assert result['status'] == 'answered'
    assert result['audit']['selected_decision_turns'] == [before['turn'], displayed['turn']]
    assert result['audit']['decision_turn'] is None


def test_future_schedule_and_future_frames_are_absent_from_provider_context():
    eng, before, displayed, context = transition('pong')
    capture = []
    ask(eng, displayed, context, [fact('performed')], capture=capture)
    serialized = json.dumps(capture)
    for forbidden in ('_schedule', 'policy_memory', 'participant_id', 'state_hash', 'decision_hash'):
        assert forbidden not in serialized
    assert capture[0]['performed_action']['decision_turn'] < capture[0]['selected_frame']['turn']


@pytest.mark.parametrize('domain', ('warehouse', 'pong', 'kitchen'))
def test_store_supplies_actual_transition_for_replayed_frame_and_preserves_audit(tmp_path, domain):
    exp = RecordingExplainer()
    store = Store(Settings(database=str(tmp_path / (domain + '.sqlite3'))), exp)
    try:
        flow = Flow(store, domain)
        flow.start_task2()
        before = flow.internal_state()
        first_human_action = engine(domain).human_advisor(before)
        flow.step(first_human_action)
        displayed = flow.internal_state()
        flow.step()
        unchanged = flow.internal_state()
        payload = flow.question(turn=1)
        result = store.ask(flow.token, payload)
        assert result['status'] == 'answered'
        call = exp.calls[-1]
        assert call['state'] == displayed
        assert call['action_context'] == {'state': before,
            'decision': engine(domain).decide(before), 'human_action': first_human_action}
        assert [frame['turn'] for frame in call['history']] == [0, 1]
        assert flow.internal_state() == unchanged
        with store.db.transaction() as db:
            saved = json.loads(db.one('SELECT result_json FROM pl3_questions WHERE id=?', (payload['question_id'],))['result_json'])
        assert saved['audit']['target_display_turn'] == 1
        assert saved['audit']['performed_decision_turn'] == 0
        assert saved['audit']['incoming_transition_sha256']
        store.ask(flow.token, payload)
        assert len(exp.calls) == 1, 'Idempotent retries must reuse the saved answer'
    finally:
        store.db.close()


@pytest.mark.parametrize('language', ('en', 'zh'))
@pytest.mark.parametrize('purpose', ('action', 'reason'))
def test_cancelled_simultaneous_bin_interaction_is_not_presented_as_completed(language, purpose):
    from tests.test_study_v3_kitchen_v6 import DeliveryGrace
    eng = engine('kitchen')
    before = DeliveryGrace().blocked()
    for _ in range(15):
        decision = eng.decide(before)
        if decision.get('reason_code') == 'discard_blocked_output' and decision['action'] == 'interact':
            break
        before = eng.step(before, 'wait', decision)
    else:
        pytest.fail('Fixture never reached its real bin interaction')
    displayed = eng.step(before, 'interact', decision)
    outcome = next(event for event in displayed['events'] if event['type'] == 'delivery_resumed')
    assert outcome['outcome'] == 'cancelled'
    assert displayed['ai']['holding']['stage'] == 'finished'
    context = {'state': before, 'decision': decision, 'human_action': 'interact'}
    result = ask(eng, displayed, context, [fact('performed', purpose)], language=language)
    assert result['status'] == 'answered'
    assert outcome[language] in result['answer']
    assert ('没有丢弃扣分' if language == 'zh' else 'no disposal penalty') in result['answer']
    assert result['audit']['performed_action']['ai_action'] == 'interact'
    assert result['audit']['decision_turn'] == before['turn']
