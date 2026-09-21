"""Public enrollment/domain switching with real local stores and fake answers.

All identities/databases are test fixtures. Pilot configuration here exercises
admission rules only; it does not create real participants or use a model API.
"""
from dataclasses import replace
import threading
import uuid

import pytest

from study_v3 import RELEASE_ID, SUPPORTED_RELEASE_IDS
from study_v3.config import Settings
from study_v3.server import make_server
from study_v3.store import Store, StudyError
from tests.test_study_v3_http import Client, ORIGIN
from tests.test_study_v3_store import Flow, RecordingExplainer, assert_public, denied


PREVIOUS_RELEASE = 'policylens-three-domain-20260920.v3.2'
UNSUPPORTED_RELEASE = 'policylens-three-domain-20260920.v3.1'


@pytest.fixture
def pilot_store(tmp_path):
    settings = Settings(database=str(tmp_path / 'switching.sqlite3'), mode='pilot',
                        origin=ORIGIN, persistent=True, verified=True,
                        admin_token='switching-test-only-admin',
                        llm_base_url='https://model-test.invalid/v1', llm_model='test-only')
    store = Store(settings, RecordingExplainer())
    assert store.ready
    try:
        yield store
    finally:
        store.db.close()


def enrollment(name, domain, group=None, **fields):
    payload = {'participant_id': name, 'domain': domain, 'mode': 'pilot', 'consent': True}
    if group is not None:
        payload['group'] = group
    return {**payload, **fields}


def public_flow(store, domain='pong', group='A', name=None, token=None):
    flow = object.__new__(Flow)
    flow.store, flow.domain, flow.group = store, domain, group
    flow.name = name or 'switch-test-' + uuid.uuid4().hex
    flow.token, flow.view = store.create(enrollment(flow.name, domain, group), token)
    flow.recovery = flow.view['recovery_code']
    return flow


def stable_view(view):
    return {key: value for key, value in view.items() if key != 'recovery_code'}


def assert_assignment(view, group, source='participant_choice'):
    assert view['group'] == group
    assert view['group_selection_locked'] is True
    assert view['group_assignment_source'] == source
    assert_public(view)


@pytest.mark.parametrize('domain', ['warehouse', 'pong', 'kitchen'])
@pytest.mark.parametrize('group', ['A', 'B'])
def test_public_pilot_can_choose_group_without_researcher_auth(pilot_store, domain, group):
    flow = public_flow(pilot_store, domain, group)
    assert flow.view['language'] == 'en' and flow.view['stage'] == 'demo'
    assert_assignment(flow.view, group)
    assert flow.view['can_ask'] is False
    assert_assignment(pilot_store.recover_view(flow.token, domain), group)
    exported = pilot_store.export()
    assert exported['participants'][0]['group_code'] == group
    assert exported['instances'][0]['group_code'] == group
    assert exported['enrollments'][0]['assignment_source'] == 'participant_choice'


@pytest.mark.parametrize('mode', ['test', 'preview'])
def test_admin_synthetic_group_choice_keeps_researcher_provenance(pilot_store, mode):
    _, view = pilot_store.create(enrollment('synthetic-' + mode, 'pong', 'B', mode=mode), admin=True)
    assert_assignment(view, 'B', 'researcher_override')
    assert pilot_store.export()['enrollments'][0]['assignment_source'] == 'researcher_override'


def test_three_unfinished_domains_keep_independent_progress_and_immutable_assignments(pilot_store):
    first = public_flow(pilot_store, 'warehouse', 'B')
    flows = [first]
    for domain, group in [('pong', 'A'), ('kitchen', 'B')]:
        flows.append(public_flow(pilot_store, domain, group, first.name, first.token))
    for index, flow in enumerate(flows):
        assert flow.token == first.token
        flow.finish_demo()
        for _ in range(index + 1):
            flow.step('wait')
        if index == 1:
            flow.command('language', language='zh')
        assert flow.view['stage'] == 'task1'
    before = pilot_store.export()
    for flow in reversed(flows):
        assert pilot_store.recover_view(first.token, flow.domain) == flow.view
        new_token, resumed = pilot_store.create(enrollment(first.name, flow.domain,
            'B' if flow.group == 'A' else 'A', language='en'), first.token)
        assert new_token == first.token
        assert stable_view(resumed) == flow.view
        assert_assignment(resumed, flow.group)
    assert pilot_store.export() == before
    assert before['participants'][0]['group_code'] == 'B'
    assert {row['domain']: row['group_code'] for row in before['instances']} == {
        'warehouse': 'B', 'pong': 'A', 'kitchen': 'B'}
    assert len(before['instances']) == len(before['runs']) == 3
    assert {row['assignment_source'] for row in before['enrollments']} == {'participant_choice'}


def test_omitted_group_retains_balanced_and_existing_participant_defaults(pilot_store, monkeypatch):
    monkeypatch.setattr('study_v3.store.secrets.choice', lambda groups: 'A')
    token, first = pilot_store.create(enrollment('default-choice-first', 'pong'))
    assert_assignment(first, 'A', 'randomized_balanced')
    _, second = pilot_store.create(enrollment('default-choice-second', 'pong'))
    assert_assignment(second, 'B', 'randomized_balanced')
    _, other_domain = pilot_store.create(enrollment('default-choice-first', 'kitchen'), token)
    assert_assignment(other_domain, 'A', 'existing_participant')


def test_enrollment_guards_remain_and_rejected_requests_leave_no_records(pilot_store):
    payload = enrollment('guarded-participant', 'pong', 'A')
    denied(lambda: pilot_store.create({**payload, 'consent': False}), 'consent_required', 400)
    for mode in ('test', 'preview'):
        denied(lambda mode=mode: pilot_store.create({**payload, 'mode': mode}),
               'researcher_access_required', 403)
    pilot_store.qa_healthy = False
    denied(lambda: pilot_store.create(payload), 'study_not_ready', 503)
    assert all(not rows for rows in pilot_store.export().values())


def test_domain_switching_preserves_recovery_mode_and_identity_boundaries(pilot_store):
    first = public_flow(pilot_store, 'pong', 'A')
    outsider = public_flow(pilot_store, 'kitchen', 'B')
    payload = enrollment(first.name, 'warehouse', 'B')
    before = pilot_store.export()
    denied(lambda: pilot_store.create(payload), 'participant_exists_use_recovery', 409)
    denied(lambda: pilot_store.create(payload, outsider.token), 'participant_exists_use_recovery', 409)
    denied(lambda: pilot_store.create({**payload, 'recovery_code': 'incorrect'}),
           'participant_exists_use_recovery', 409)
    denied(lambda: pilot_store.create({**payload, 'mode': 'test'}, first.token, admin=True),
           'participant_mode_conflict', 409)
    denied(lambda: pilot_store.view(outsider.token, first.view['instance_id']), 'study_not_found', 404)
    assert pilot_store.export() == before
    recovered_token, warehouse = pilot_store.create({**payload, 'recovery_code': first.recovery})
    assert recovered_token != first.token
    assert_assignment(warehouse, 'B')
    # Both valid sessions belong to the same identity, with per-domain groups.
    assert_assignment(pilot_store.recover_view(recovered_token, 'pong'), 'A')
    assert_assignment(pilot_store.recover_view(first.token, 'warehouse'), 'B')


@pytest.mark.parametrize('old_release', [PREVIOUS_RELEASE, UNSUPPORTED_RELEASE])
def test_old_gameplay_is_archived_and_new_enrollment_keeps_all_records(pilot_store, old_release):
    assert old_release not in SUPPORTED_RELEASE_IDS and RELEASE_ID in SUPPORTED_RELEASE_IDS
    flow = public_flow(pilot_store)
    flow.start_task2()
    flow.step('wait')
    with pilot_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',
                   (old_release, flow.view['instance_id']))
    before = pilot_store.export()
    reopened = Store(pilot_store.settings, RecordingExplainer())
    try:
        welcome = reopened.recover_view(flow.token, 'pong')
        assert welcome == {'participant_id': flow.name, 'stage': 'welcome', 'previous_version_saved': True}
        denied(lambda: reopened.view(flow.token, flow.view['instance_id']), 'release_changed', 409)
        denied(lambda: reopened.ask(flow.token, flow.question()), 'release_changed', 409)
        token, new = reopened.create(enrollment(flow.name, 'pong', 'B'), flow.token)
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


@pytest.mark.parametrize('stage', ['demo', 'completed'])
def test_unsupported_release_does_not_block_any_new_domain(pilot_store, stage):
    old = public_flow(pilot_store, 'pong', 'A')
    with pilot_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=?,stage=? WHERE id=?',
                   (UNSUPPORTED_RELEASE, stage, old.view['instance_id']))
    denied(lambda: pilot_store.view(old.token, old.view['instance_id']), 'release_changed', 409)
    assert pilot_store.recover_view(old.token, 'pong')['previous_version_saved'] is True
    other = public_flow(pilot_store, 'kitchen', 'B', old.name, old.token)
    new_pong = public_flow(pilot_store, 'pong', 'B', old.name, old.token)
    assert other.view['release_id'] == new_pong.view['release_id'] == RELEASE_ID
    assert_assignment(other.view, 'B')
    assert_assignment(new_pong.view, 'B')
    instances = pilot_store.export()['instances']
    assert len(instances) == 3
    assert next(row for row in instances if row['id'] == old.view['instance_id'])['release_id'] == UNSUPPORTED_RELEASE


def test_group_permissions_follow_instance_and_task_after_switching(pilot_store):
    group_a = public_flow(pilot_store, 'pong', 'A')
    group_a.start_task2()
    group_b = public_flow(pilot_store, 'kitchen', 'B', group_a.name, group_a.token)
    assert group_b.token == group_a.token
    denied(lambda: pilot_store.ask(group_b.token, group_b.question(group='A')), 'explanations_unavailable', 403)
    group_b.finish_demo()
    denied(lambda: pilot_store.ask(group_b.token, group_b.question(group='A')), 'explanations_unavailable', 403)
    group_b.finish_task()
    group_b.command('next')
    assert group_b.view['stage'] == 'task2' and group_b.view['can_ask'] is False
    denied(lambda: pilot_store.ask(group_b.token, group_b.question(group='A', group_code='A')),
           'explanations_unavailable', 403)
    # Re-requesting A in an existing B enrollment cannot upgrade its access.
    _, unchanged_b = pilot_store.create(enrollment(group_a.name, 'kitchen', 'A'), group_a.token)
    assert_assignment(unchanged_b, 'B')
    assert unchanged_b['can_ask'] is False
    denied(lambda: pilot_store.ask(group_a.token, group_a.question(target_run=group_b.view['run_id'], turn=0)),
           'frame_not_found', 404)
    answer = pilot_store.ask(group_a.token, group_a.question())
    assert answer['status'] == 'answered'
    assert len(pilot_store.explainer.calls) == 1
    assert pilot_store.recover_view(group_b.token, 'kitchen')['questions'] == []
    group_a.finish_task()
    denied(lambda: pilot_store.ask(group_a.token, group_a.question()), 'explanations_unavailable', 403)
    group_a.command('next')
    assert group_a.view['stage'] == 'task3' and group_a.view['can_ask'] is False
    denied(lambda: pilot_store.ask(group_a.token, group_a.question()), 'explanations_unavailable', 403)
    assert len(pilot_store.explainer.calls) == 1


def test_public_http_cookie_switches_domains_and_resumes_locked_group(pilot_store, tmp_path):
    settings = replace(pilot_store.settings, database=str(tmp_path / 'http-switch.sqlite3'))
    server = make_server(settings, port=0, explainer=RecordingExplainer())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = Client(server.server_port)
    try:
        name = 'public-http-switch'
        status, pong, headers = client.request('POST', '/api/study/session', enrollment(name, 'pong', 'A'))
        assert status == 200 and 'HttpOnly' in headers['Set-Cookie']
        assert_assignment(pong, 'A')
        cookie = client.cookie
        status, progress, _ = client.command('demo_next', pong)
        assert status == 200 and progress['revision'] == pong['revision'] + 1
        status, kitchen, _ = client.request('POST', '/api/study/session', enrollment(name, 'kitchen', 'B'))
        assert status == 200 and client.cookie == cookie
        assert_assignment(kitchen, 'B')
        status, recovered, _ = client.request('GET', '/api/study/view?domain=pong')
        assert status == 200 and recovered == progress
        status, resumed, _ = client.request('POST', '/api/study/session', enrollment(name, 'pong', 'B'))
        assert status == 200 and stable_view(resumed) == progress
        assert client.cookie == cookie
        outsider = Client(server.server_port)
        assert outsider.request('GET', '/api/study/view?instance_id=' + pong['instance_id'])[0] == 401
        assert outsider.request('POST', '/api/study/session', enrollment(name, 'warehouse', 'A'))[0] == 409
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)

@pytest.mark.parametrize('domain', ['warehouse','pong','kitchen'])
def test_v35_archives_previous_gameplay_without_rewriting_records(pilot_store, domain):
    old_release='policylens-three-domain-20260920.v3.4'
    assert old_release not in SUPPORTED_RELEASE_IDS
    flow=public_flow(pilot_store,domain,'A')
    flow.command('demo_skip');flow.step('wait')
    old_id=flow.view['instance_id']
    with pilot_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',(old_release,old_id))
    before=pilot_store.export(release_id=old_release)
    restored=pilot_store.recover_view(flow.token,domain)
    assert restored['stage']=='welcome' and restored['previous_version_saved']
    _,fresh=pilot_store.create(enrollment(flow.name,domain,'B'),flow.token)
    assert fresh['instance_id']!=old_id and fresh['group']=='B' and fresh['stage']=='demo'
    assert pilot_store.export(release_id=old_release)==before


@pytest.mark.parametrize('domain', ['warehouse','pong','kitchen'])
@pytest.mark.parametrize('old_release', [
    'policylens-three-domain-20260920.v3.5',
    'policylens-three-domain-20260920.v3.5.1',
    'policylens-three-domain-20260920.v3.5.2',
    'policylens-three-domain-20260920.v3.6',
])
def test_current_release_archives_old_rules_without_rewriting_saved_records(pilot_store, domain, old_release):
    assert old_release not in SUPPORTED_RELEASE_IDS
    flow=public_flow(pilot_store,domain,'A')
    flow.command('demo_skip');flow.step('wait')
    old_id=flow.view['instance_id']
    with pilot_store.db.transaction() as db:
        db.execute('UPDATE pl3_instances SET release_id=? WHERE id=?',(old_release,old_id))
    before=pilot_store.export(release_id=old_release)
    restored=pilot_store.recover_view(flow.token,domain)
    assert restored['stage']=='welcome' and restored['previous_version_saved']
    _,fresh=pilot_store.create(enrollment(flow.name,domain,'B'),flow.token)
    assert fresh['instance_id']!=old_id and fresh['release_id']==RELEASE_ID
    assert fresh['group']=='B' and fresh['stage']=='demo'
    assert pilot_store.export(release_id=old_release)==before


@pytest.mark.parametrize('domain', ['warehouse','pong','kitchen'])
def test_demo_skip_is_atomic_idempotent_and_cannot_skip_a_task(pilot_store, domain):
    flow=public_flow(pilot_store,domain,'A')
    payload=flow.command('demo_skip')
    first=flow.view
    assert first['stage']=='task1' and first['state']['turn']==0
    assert not first['can_ask'] and first['questions']==[]
    retried=pilot_store.command(flow.token,'demo_skip',payload)
    assert retried==first and len(retried['task_runs'])==1
    with pytest.raises(StudyError) as error:
        flow.command('demo_skip')
    assert error.value.code=='wrong_stage'
    flow.step('wait')
    assert flow.view['state']['turn']==1


@pytest.mark.parametrize('domain', ['warehouse','pong'])
def test_continuous_demo_completion_does_not_start_game_clock(pilot_store, domain):
    flow=public_flow(pilot_store,domain,'B')
    flow.command('demo_finish')
    assert flow.view['stage']=='demo' and flow.view['state'] is None
    assert flow.view['demo']['index']==len(flow.view['demo']['captions'])
    assert flow.view['task_runs']==[]
    flow.command('next')
    assert flow.view['stage']=='task1' and flow.view['state']['turn']==0
