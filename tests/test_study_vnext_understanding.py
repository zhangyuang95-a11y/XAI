"""Synthetic questionnaire checks; these do not validate the instrument in people."""
import inspect

import pytest

from study_vnext import understanding as u


# Public mechanics/control truth, used here for deterministic scoring checks.
KEYS = {
    'warehouse': dict(warehouse_charging='b', warehouse_endpoints='c',
                      warehouse_frozen_teammate='c', warehouse_past_actions='a'),
    'pong': dict(pong_ordinary_ball='a', pong_cooperative_ball='b',
                 pong_frozen_teammate='c', pong_past_actions='a'),
    'kitchen': dict(kitchen_interaction='b', kitchen_wait='c',
                    kitchen_frozen_teammate='c', kitchen_past_actions='a'),
}


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


@pytest.mark.parametrize('domain', KEYS)
def test_key_free_fixed_bilingual_bank_and_detached_return(domain):
    en, zh = (u.bank_for_participant(domain, lang) for lang in ('en', 'zh'))
    assert len(en['questions']) == len(zh['questions']) == 4
    assert en['sha256'] == zh['sha256'] == u.INSTRUMENT_SHA256 == u.instrument_sha256()
    assert len(en['sha256']) == 64
    assert en['version'] == u.VERSION == u.INSTRUMENT_ID
    assert en['status'] == 'candidate_instrument_requires_pilot_validation'
    assert en['administration'] == 'after_task3_only'
    assert not {'correct', 'answer', 'answer_key', 'solution', 'feedback'} & set(walk_keys(en))
    assert [q['id'] for q in en['questions']] == [q['id'] for q in zh['questions']]
    for left, right in zip(en['questions'], zh['questions']):
        assert left['prompt'] != right['prompt']
        assert [x['id'] for x in left['options']] == [x['id'] for x in right['options']]
    en['questions'][0]['options'][0]['text'] = 'mutated'
    assert u.bank_for_participant(domain, 'en')['questions'][0]['options'][0]['text'] != 'mutated'
    # No treatment, observed-performance or answers input can select a different bank.
    assert list(inspect.signature(u.bank_for_participant).parameters) == ['domain', 'language']


@pytest.mark.parametrize('domain', KEYS)
def test_aggregate_scoring_has_no_threshold_or_correct_answer_feedback(domain):
    answers = KEYS[domain].copy()
    result = u.score(domain, answers)
    assert result['correct_count'] == result['item_count'] == 4
    assert result['proportion_correct'] == 1
    assert result['validated_scale'] is result['efficacy_evidence'] is False
    assert not {'passed', 'threshold', 'answers', 'answer_key', 'questions', 'correct_options'} & set(result)
    assert answers == KEYS[domain]
    wrong = {key: ('a' if val != 'a' else 'b') for key, val in answers.items()}
    assert u.score(domain, wrong)['correct_count'] == 0
    wrong[next(iter(answers))] = answers[next(iter(answers))]
    assert u.score(domain, wrong)['proportion_correct'] == .25


@pytest.mark.parametrize('bad', [None, [], {}, {'foreign_item': 'a'}])
def test_missing_or_foreign_items_rejected(bad):
    with pytest.raises(u.UnderstandingError, match='exactly'):
        u.score('warehouse', bad)


@pytest.mark.parametrize('bad', [None, 0, True, '', 'd', ['a']])
def test_nonoption_response_rejected(bad):
    answers = KEYS['warehouse'].copy()
    answers['warehouse_charging'] = bad
    with pytest.raises(u.UnderstandingError, match='option'):
        u.score('warehouse', answers)


def test_extra_items_wrong_domain_and_languages_rejected():
    answers = KEYS['warehouse'] | {'extra': 'a'}
    with pytest.raises(u.UnderstandingError):
        u.score('warehouse', answers)
    with pytest.raises(u.UnderstandingError):
        u.score('pong', KEYS['warehouse'])
    for domain in ('other', None, True):
        with pytest.raises(u.UnderstandingError):
            u.bank_for_participant(domain, 'en')
    for language in ('fr', None, True, 'EN'):
        with pytest.raises(u.UnderstandingError):
            u.bank_for_participant('warehouse', language)


def test_items_are_control_rules_not_predicted_actions_or_inferred_motives():
    for domain in KEYS:
        questions = {q['id']: q for q in u.bank_for_participant(domain, 'en')['questions']}
        assert 'frozen NN' in next(x['text'] for x in questions[domain + '_frozen_teammate']['options'] if x['id'] == 'c')
        assert 'does not reveal the next unexecuted action' in next(
            x['text'] for x in questions[domain + '_past_actions']['options'] if x['id'] == 'a')
        assert 'motive' not in ' '.join(q['prompt'] for q in questions.values())
