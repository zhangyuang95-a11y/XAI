"""Synthetic enrollment/domain switching with real stores and fake answers.

Pilot remains closed until all v5 acceptance gates pass. These test-mode flows
exercise persistence and access control without admitting participants.
"""
from dataclasses import replace
import json
import threading
import uuid

import pytest

from study_v3 import RELEASE_ID, SUPPORTED_RELEASE_IDS
from study_v3.config import Settings
from study_v3.registry import engine
from study_v3.server import make_server
from study_v3.store import Store, StudyError, encode
from tests.test_study_v3_http import Client, ORIGIN
from tests.test_study_v3_store import (BOUND_DOMAINS, UNBOUND_BINDINGS, Flow,
                                       RecordingExplainer, assert_public, denied)


PREVIOUS_RELEASE = 'policylens-three-domain-20260920.v3.2'
UNSUPPORTED_RELEASE = 'policylens-three-domain-20260920.v3.1'


@pytest.fixture
def synthetic_store(tmp_path):
    settings = Settings(database=str(tmp_path / 'switching.sqlite3'), mode='test',
                        origin=ORIGIN, persistent=True, verified=True,
                        admin_token='switching-test-only-admin',
                        llm_base_url='https://model-test.invalid/v1', llm_model='test-only')
    store = Store(settings, RecordingExplainer())
    assert not store.ready
    try:
        yield store
    finally:
        store.db.close()


def enrollment(name, domain, group=None, **fields):
    payload = {'participant_id': name, 'domain': domain, 'mode': 'test', 'consent': True}
    if group is not None:
        payload['group'] = group
    return {**payload, **fields}


def synthetic_flow(store, domain='pong', group='A', name=None, token=None):
    flow = object.__new__(Flow)
    flow.store, flow.domain, flow.group = store, domain, group
    flow.name = name or 'switch-test-' + uuid.uuid4().hex
    flow.token, flow.view = store.create(enrollment(flow.name, domain, group), token, admin=True)
    flow.recovery = flow.view['recovery_code']
    return flow


def stable_view(view):
    return {key: value for key, value in view.items() if key != 'recovery_code'}


def assert_assignment(view, group, source='researcher_override'):
    assert view['group'] == group
    assert view['group_selection_locked'] is True
    assert view['group_assignment_source'] == source
    assert_public(view)


@pytest.mark.parametrize('domain', BOUND_DOMAINS)
@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_researcher_test_group_has_immutable_policy_binding(synthetic_store, domain, group):
    flow = synthetic_flow(synthetic_store, domain, group)
    assert flow.view['language'] == 'en' and flow.view['stage'] == 'demo'
    assert_assignment(flow.view, group)
    assert flow.view['can_ask'] is False
    assert_assignment(synthetic_store.recover_view(flow.token, domain), group)
    exported = synthetic_store.export()
    assert exported['participants'][0]['group_code'] == group
    assert exported['instances'][0]['group_code'] == group
    assert exported['enrollments'][0]['assignment_source'] == 'researcher_override'
    assert exported['policy_bindings'][0]['policy_id'] == flow.view['policy_id']
    assert exported['policy_bindings'][0]['actor_sha256'] == flow.view['actor_sha256']


@pytest.mark.parametrize('domain,group', UNBOUND_BINDINGS)
def test_unpinned_domain_cannot_enter_preview_or_test(synthetic_store, domain, group):
    for mode in ('preview', 'test'):
        denied(lambda mode=mode: synthetic_store.create(enrollment(
            'unbound-' + uuid.uuid4().hex, domain, group, mode=mode), admin=True),
            'policy_unavailable', 503)
    assert all(not rows for rows in synthetic_store.export().values())


def test_unaccepted_release_rejects_pilot_without_creating_records(synthetic_store):
    settings = replace(synthetic_store.settings, mode='pilot')
    pilot = Store(settings, RecordingExplainer())
    assert not pilot.ready
    denied(lambda: pilot.create(enrollment('pilot-closed', 'pong', mode='pilot')),
           'study_not_ready', 503)
    assert all(not rows for rows in pilot.export().values())


@pytest.mark.parametrize('mode', ['test', 'preview'])
def test_admin_synthetic_group_choice_keeps_researcher_provenance(synthetic_store, mode):
    _, view = synthetic_store.create(enrollment('synthetic-' + mode, 'pong', 'B', mode=mode), admin=True)
    assert_assignment(view, 'B', 'researcher_override')
    assert synthetic_store.export()['enrollments'][0]['assignment_source'] == 'researcher_override'


@pytest.mark.parametrize('domains', [BOUND_DOMAINS] if len(BOUND_DOMAINS) == 3 else [])
def test_three_unfinished_domains_keep_independent_progress_and_immutable_assignments(synthetic_store, domains):
    first = synthetic_flow(synthetic_store, domains[0], 'B')
    flows = [first]
    for domain, group in [(domains[1], 'A'), (domains[2], 'B')]:
        flows.append(synthetic_flow(synthetic_store, domain, group, first.name, first.token))
    for index, flow in enumerate(flows):
        assert flow.token == first.token
        flow.finish_demo()
        for _ in range(index + 1):
            flow.step('wait')
        if index == 1:
            flow.command('language', language='zh')
        assert flow.view['stage'] == 'task1'
    before = synthetic_store.export()
    for flow in reversed(flows):
        assert synthetic_store.recover_view(first.token, flow.domain) == flow.view
        denied(lambda flow=flow: synthetic_store.create(enrollment(first.name, flow.domain,
            'B' if flow.group == 'A' else 'A', language='en'), first.token, admin=True),
            'group_binding_immutable', 409)
        new_token, resumed = synthetic_store.create(enrollment(first.name, flow.domain,
            flow.group, language='en'), first.token, admin=True)
        assert new_token == first.token
        assert stable_view(resumed) == flow.view
        assert_assignment(resumed, flow.group)
    assert synthetic_store.export() == before
    assert before['participants'][0]['group_code'] == 'B'
    assert {row['domain']: row['group_code'] for row in before['instances']} == {
        domains[0]: 'B', domains[1]: 'A', domains[2]: 'B'}
    assert len(before['instances']) == len(before['runs']) == 3
    assert {row['assignment_source'] for row in before['enrollments']} == {'researcher_override'}


def test_omitted_group_is_balanced_and_existing_binding_is_preserved(synthetic_store, monkeypatch):
    monkeypatch.setattr('study_v3.store.secrets.choice', lambda options: options[0])
    token, first = synthetic_store.create(enrollment('default-choice-first', 'pong'), admin=True)
    assert_assignment(first, 'A', 'randomized_balanced')
    _, second = synthetic_store.create(enrollment('default-choice-second', 'pong'), admin=True)
    assert_assignment(second, 'B', 'randomized_balanced')
    _, resumed = synthetic_store.create(enrollment('default-choice-first', 'pong'), token, admin=True)
    assert_assignment(resumed, 'A', 'randomized_balanced')
    assert resumed['instance_id'] == first['instance_id']


def test_enrollment_guards_remain_and_rejected_requests_leave_no_records(synthetic_store):
    payload = enrollment('guarded-participant', 'pong', 'A')
    denied(lambda: synthetic_store.create({**payload, 'consent': False}), 'consent_required', 400)
    for mode in ('test', 'preview'):
        denied(lambda mode=mode: synthetic_store.create({**payload, 'mode': mode}),
               'researcher_access_required', 403)
    synthetic_store.qa_healthy = False
    denied(lambda: synthetic_store.create({**payload, 'mode': 'pilot', 'group': None}),
           'study_not_ready', 503)
    assert all(not rows for rows in synthetic_store.export().values())


@pytest.mark.parametrize('domains', [BOUND_DOMAINS] if len(BOUND_DOMAINS) == 3 else [])
def test_domain_switching_preserves_recovery_mode_and_identity_boundaries(synthetic_store, domains):
    first = synthetic_flow(synthetic_store, 'pong', 'A')
    other = next(domain for domain in domains if domain != 'pong')
    target = next(domain for domain in domains if domain not in ('pong', other))
    outsider = synthetic_flow(synthetic_store, other, 'B')
    payload = enrollment(first.name, target, 'B')
    before = synthetic_store.export()
    denied(lambda: synthetic_store.create(payload, admin=True), 'participant_exists_use_recovery', 409)
    denied(lambda: synthetic_store.create(payload, outsider.token, admin=True), 'participant_exists_use_recovery', 409)
    denied(lambda: synthetic_store.create({**payload, 'recovery_code': 'incorrect'}, admin=True),
           'participant_exists_use_recovery', 409)
    denied(lambda: synthetic_store.create({**payload, 'mode': 'preview'}, first.token, admin=True),
           'participant_mode_conflict', 409)
    denied(lambda: synthetic_store.view(outsider.token, first.view['instance_id']), 'study_not_found', 404)
    assert synthetic_store.export() == before
    recovered_token, warehouse = synthetic_store.create({**payload, 'recovery_code': first.recovery}, admin=True)
    assert recovered_token != first.token
    assert_assignment(warehouse, 'B')
    # Both valid sessions belong to the same identity, with per-domain groups.
    assert_assignment(synthetic_store.recover_view(recovered_token, 'pong'), 'A')
    assert_assignment(synthetic_store.recover_view(first.token, target), 'B')


@pytest.mark.parametrize('old_release', [PREVIOUS_RELEASE, UNSUPPORTED_RELEASE])
def test_old_gameplay_is_archived_and_new_enrollment_keeps_all_records(synthetic_store, old_release):
    assert old_release not in SUPPORTED_RELEASE_IDS and RELEASE_ID in SUPPORTED_RELEASE_IDS
    flow = synthetic_flow(synthetic_store)
    flow.start_task2()
    flow.step('wait')
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',
                   (old_release, flow.view['instance_id']))
    before = synthetic_store.export()
    reopened = Store(synthetic_store.settings, RecordingExplainer())
    try:
        welcome = reopened.recover_view(flow.token, 'pong')
        assert welcome == {'participant_id': flow.name, 'stage': 'welcome',
                           'previous_version_saved': True, 'archived_instances': [
                               {'id': flow.view['instance_id'], 'release_id': old_release,
                                'stage': flow.view['stage']}]}
        archived_view = reopened.view(flow.token, flow.view['instance_id'])
        assert archived_view['archived'] and archived_view['stage'] == 'archived'
        assert archived_view['actions'] == [] and not archived_view['can_ask']
        denied(lambda: reopened.ask(flow.token, flow.question()), 'release_changed', 409)
        token, new = reopened.create(enrollment(flow.name, 'pong', 'B'), flow.token, admin=True)
        assert token == flow.token and new['instance_id'] != flow.view['instance_id']
        assert new['release_id'] == RELEASE_ID and new['stage'] == 'demo'
        assert_assignment(new, 'B')
        assert reopened.recover_view(token, 'pong')['instance_id'] == new['instance_id']
        # Every historical row remains byte-for-byte available to research export.
        after = reopened.export()
        for table, records in before.items():
            assert all(row in after[table] for row in records)
        assert len(after['instances']) == 2 and len(after['runs']) == 2
    finally:
        reopened.db.close()


@pytest.mark.parametrize('domain', BOUND_DOMAINS)
@pytest.mark.parametrize('previous', [
    'policylens-three-domain-20260922.v3.8',
    'policylens-three-domain-20260922.v3.8.1',
    'policylens-three-domain-20260922.v3.8.2'])
def test_new_frozen_release_preserves_archives_and_restarts_each_domain(synthetic_store, domain, previous):
    flow = synthetic_flow(synthetic_store, domain, 'A')
    flow.finish_demo(); flow.step('wait')
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?', (previous, flow.view['instance_id']))
    before = synthetic_store.export(release_id=previous)
    reopened = Store(synthetic_store.settings, RecordingExplainer())
    try:
        restored = reopened.recover_view(flow.token, domain)
        archived = [{'id': flow.view['instance_id'], 'release_id': previous,
                     'stage': flow.view['stage']}]
        assert restored == {'participant_id': flow.name, 'stage': 'welcome',
                            'previous_version_saved': True, 'archived_instances': archived}
        archived_view = reopened.view(flow.token, flow.view['instance_id'])
        assert archived_view['archived'] is True
        assert archived_view['release_id'] == previous
        assert archived_view['state']['version'] == flow.view['state']['version']
        frame = reopened.frame(flow.token, flow.view['instance_id'], flow.view['run_id'], 0)
        assert frame['state']['version'] == flow.view['state']['version']
        denied(lambda: reopened.command(flow.token, 'action', {
            'instance_id': flow.view['instance_id'], 'command_id': uuid.uuid4().hex,
            'revision': flow.view['revision'], 'action': 'wait'}), 'release_changed', 409)
        token, fresh = reopened.create(enrollment(flow.name, domain, 'A'), flow.token, admin=True)
        assert token == flow.token and fresh['instance_id'] != flow.view['instance_id']
        assert fresh['release_id'] == RELEASE_ID and fresh['stage'] == 'demo'
        if domain == 'kitchen':
            assert fresh['rule_metadata']['score']['served'] == 100
        assert reopened.export(release_id=previous) == before
    finally:
        reopened.db.close()


def test_old_pong_answers_remain_exportable_without_cross_version_execution(synthetic_store):
    flow = synthetic_flow(synthetic_store, 'pong', 'A')
    flow.start_task2()
    answer = synthetic_store.ask(flow.token, flow.question(question='What is your next action?'))
    assert answer['status'] == 'answered'
    old_release = 'policylens-three-domain-20260922.v3.8'
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',
                   (old_release, flow.view['instance_id']))
    archived = synthetic_store.export(release_id=old_release)
    assert len(archived['questions']) == 1
    assert synthetic_store.recover_view(flow.token, 'pong')['previous_version_saved'] is True
    old_view = synthetic_store.view(flow.token, flow.view['instance_id'])
    assert old_view['archived'] and old_view['questions'][0]['id'] == answer['id']
    assert not old_view['can_ask']
    token, fresh = synthetic_store.create(enrollment(flow.name, 'pong', 'A'), flow.token, admin=True)
    assert token == flow.token and fresh['instance_id'] != flow.view['instance_id']
    assert synthetic_store.export(release_id=old_release) == archived


@pytest.mark.parametrize('stage', ['demo', 'completed'])
@pytest.mark.parametrize('other_domain', tuple(domain for domain in BOUND_DOMAINS if domain != 'pong'))
def test_unsupported_release_does_not_block_any_new_domain(synthetic_store, stage, other_domain):
    old = synthetic_flow(synthetic_store, 'pong', 'A')
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=?,stage=? WHERE id=?',
                   (UNSUPPORTED_RELEASE, stage, old.view['instance_id']))
    archived = synthetic_store.view(old.token, old.view['instance_id'])
    assert archived['archived'] and archived['stage'] == 'archived'
    assert synthetic_store.recover_view(old.token, 'pong')['previous_version_saved'] is True
    other = synthetic_flow(synthetic_store, other_domain, 'B', old.name, old.token)
    new_pong = synthetic_flow(synthetic_store, 'pong', 'B', old.name, old.token)
    assert other.view['release_id'] == new_pong.view['release_id'] == RELEASE_ID
    assert_assignment(other.view, 'B')
    assert_assignment(new_pong.view, 'B')
    instances = synthetic_store.export()['instances']
    assert len(instances) == 3
    assert next(row for row in instances if row['id'] == old.view['instance_id'])['release_id'] == UNSUPPORTED_RELEASE


@pytest.mark.parametrize('other_domain', tuple(domain for domain in BOUND_DOMAINS if domain != 'pong'))
def test_group_permissions_follow_instance_and_task_after_switching(synthetic_store, other_domain):
    group_a = synthetic_flow(synthetic_store, 'pong', 'A')
    group_a.start_task2()
    group_b = synthetic_flow(synthetic_store, other_domain, 'B', group_a.name, group_a.token)
    assert group_b.token == group_a.token
    denied(lambda: synthetic_store.ask(group_b.token, group_b.question(group='A')), 'explanations_unavailable', 403)
    group_b.finish_demo()
    denied(lambda: synthetic_store.ask(group_b.token, group_b.question(group='A')), 'explanations_unavailable', 403)
    group_b.finish_task()
    group_b.command('next')
    assert group_b.view['stage'] == 'task2' and group_b.view['can_ask'] is False
    denied(lambda: synthetic_store.ask(group_b.token, group_b.question(group='A', group_code='A')),
           'explanations_unavailable', 403)
    # A conflicting group cannot upgrade an immutable B enrollment.
    denied(lambda: synthetic_store.create(enrollment(group_a.name, other_domain, 'A'),
           group_a.token, admin=True), 'group_binding_immutable', 409)
    _, unchanged_b = synthetic_store.create(enrollment(group_a.name, other_domain, 'B'),
                                            group_a.token, admin=True)
    assert_assignment(unchanged_b, 'B') and not unchanged_b['can_ask']
    denied(lambda: synthetic_store.ask(group_a.token, group_a.question(target_run=group_b.view['run_id'], turn=0)),
           'frame_not_found', 404)
    answer = synthetic_store.ask(group_a.token, group_a.question())
    assert answer['status'] == 'answered'
    assert len(synthetic_store.explainer.calls) == 1
    assert synthetic_store.recover_view(group_b.token, other_domain)['questions'] == []
    group_a.finish_task()
    denied(lambda: synthetic_store.ask(group_a.token, group_a.question()), 'explanations_unavailable', 403)
    group_a.command('next')
    assert group_a.view['stage'] == 'task3' and group_a.view['can_ask'] is False
    denied(lambda: synthetic_store.ask(group_a.token, group_a.question()), 'explanations_unavailable', 403)
    assert len(synthetic_store.explainer.calls) == 1


@pytest.mark.parametrize('other_domain', tuple(domain for domain in BOUND_DOMAINS if domain != 'pong'))
def test_researcher_http_cookie_switches_domains_and_resumes_locked_group(synthetic_store, tmp_path, other_domain):
    settings = replace(synthetic_store.settings, database=str(tmp_path / 'http-switch.sqlite3'))
    server = make_server(settings, port=0, explainer=RecordingExplainer())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = Client(server.server_port)
    try:
        name = 'synthetic-http-switch'
        admin = {'Authorization': 'Bearer ' + settings.admin_token}
        status, pong, headers = client.request('POST', '/api/study/session', enrollment(name, 'pong', 'A'), admin)
        assert status == 200 and 'HttpOnly' in headers['Set-Cookie']
        assert_assignment(pong, 'A')
        cookie = client.cookie
        status, progress, _ = client.command('demo_next', pong)
        assert status == 200 and progress['revision'] == pong['revision'] + 1
        status, other, _ = client.request('POST', '/api/study/session', enrollment(name, other_domain, 'B'), admin)
        assert status == 200 and client.cookie == cookie
        assert_assignment(other, 'B')
        status, recovered, _ = client.request('GET', '/api/study/view?domain=pong')
        assert status == 200 and recovered == progress
        assert client.request('POST', '/api/study/session', enrollment(name, 'pong', 'B'), admin)[0:2] == (
            409, {'error': 'group_binding_immutable'})
        status, resumed, _ = client.request('POST', '/api/study/session', enrollment(name, 'pong', 'A'), admin)
        assert status == 200 and stable_view(resumed) == progress
        assert client.cookie == cookie
        outsider = Client(server.server_port)
        assert outsider.request('GET', '/api/study/view?instance_id=' + pong['instance_id'])[0] == 401
        assert outsider.request('POST', '/api/study/session', enrollment(name, 'pong', 'A'), admin)[0] == 409
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)

@pytest.mark.parametrize('domain', BOUND_DOMAINS)
def test_v35_archives_previous_gameplay_without_rewriting_records(synthetic_store, domain):
    old_release='policylens-three-domain-20260920.v3.4'
    assert old_release not in SUPPORTED_RELEASE_IDS
    flow=synthetic_flow(synthetic_store,domain,'A')
    flow.finish_demo();flow.step('wait')
    old_id=flow.view['instance_id']
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',(old_release,old_id))
    before=synthetic_store.export(release_id=old_release)
    restored=synthetic_store.recover_view(flow.token,domain)
    assert restored['stage']=='welcome' and restored['previous_version_saved']
    _,fresh=synthetic_store.create(enrollment(flow.name,domain,'B'),flow.token,admin=True)
    assert fresh['instance_id']!=old_id and fresh['group']=='B' and fresh['stage']=='demo'
    assert synthetic_store.export(release_id=old_release)==before


@pytest.mark.parametrize('domain', BOUND_DOMAINS)
@pytest.mark.parametrize('old_release', [
    'policylens-three-domain-20260920.v3.5',
    'policylens-three-domain-20260920.v3.5.1',
    'policylens-three-domain-20260920.v3.5.2',
    'policylens-three-domain-20260920.v3.6',
])
def test_current_release_archives_old_rules_without_rewriting_saved_records(synthetic_store, domain, old_release):
    assert old_release not in SUPPORTED_RELEASE_IDS
    flow=synthetic_flow(synthetic_store,domain,'A')
    flow.finish_demo();flow.step('wait')
    old_id=flow.view['instance_id']
    with synthetic_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',(old_release,old_id))
    before=synthetic_store.export(release_id=old_release)
    restored=synthetic_store.recover_view(flow.token,domain)
    assert restored['stage']=='welcome' and restored['previous_version_saved']
    _,fresh=synthetic_store.create(enrollment(flow.name,domain,'B'),flow.token,admin=True)
    assert fresh['instance_id']!=old_id and fresh['release_id']==RELEASE_ID
    assert fresh['group']=='B' and fresh['stage']=='demo'
    assert synthetic_store.export(release_id=old_release)==before


@pytest.mark.parametrize('domain', BOUND_DOMAINS)
def test_demo_skip_is_atomic_idempotent_and_cannot_skip_a_task(synthetic_store, domain):
    flow=synthetic_flow(synthetic_store,domain,'A')
    payload=flow.command('demo_skip')
    first=flow.view
    assert first['stage']==('practice' if domain=='kitchen' else 'task1')
    if domain != 'kitchen':
        assert first['state']['turn']==0
    assert not first['can_ask'] and first['questions']==[]
    retried=synthetic_store.command(flow.token,'demo_skip',payload)
    assert retried==first and len(retried['task_runs'])==(0 if domain=='kitchen' else 1)
    with pytest.raises(StudyError) as error:
        flow.command('demo_skip')
    assert error.value.code=='wrong_stage'
    if domain=='kitchen':
        flow.command('practice_skip')
    flow.step('wait')
    assert flow.view['state']['turn']==1


@pytest.mark.parametrize('domain', tuple(domain for domain in BOUND_DOMAINS if domain != 'kitchen'))
def test_continuous_demo_completion_does_not_start_game_clock(synthetic_store, domain):
    flow=synthetic_flow(synthetic_store,domain,'B')
    flow.command('demo_finish')
    assert flow.view['stage']=='demo' and flow.view['state'] is None
    assert flow.view['demo']['index']==len(flow.view['demo']['captions'])
    assert flow.view['task_runs']==[]
    flow.command('next')
    assert flow.view['stage']=='task1' and flow.view['state']['turn']==0
