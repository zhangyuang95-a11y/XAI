"""Local JSON software tests only; synthetic adapters grant no admission."""
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import importlib.util
import json

import pytest

from runtime_v642 import RuntimeRegistry
from runtime_v642.server import make_server
from runtime_v642.store import Store


@pytest.fixture
def service(tmp_path):
    spec = importlib.util.spec_from_file_location('core_HTTP_synthetic_fixture',
        Path(__file__).with_name('test_runtime_v642_store.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, _, catalog = helper.package_at(tmp_path/'package')
    adapters = {d: helper.FakeAdapter(d) for d in ('warehouse', 'pong', 'kitchen')}
    registry = RuntimeRegistry(catalog, adapters=adapters)
    store = Store(tmp_path/'http.sqlite', registry, scenario_pools={d: (11,) for d in adapters},
                  demonstration_seeds={d: 10 for d in adapters}, mode='preview')
    admin = 'private-synthetic-unit-test-admin-token'
    server = make_server(store, admin_token=admin)
    # The factory never starts this thread; only this explicit test does.
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f'http://127.0.0.1:{server.server_port}'

    def call(path, payload=None, *, token=None, administrator=False, raw=None):
        headers = {}
        if token:
            headers['Authorization'] = 'Bearer '+token
        if administrator:
            headers['X-Admin-Token'] = admin
        if payload is not None or raw is not None:
            headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode() if payload is not None else raw
        request = Request(root+path, data=data, headers=headers)
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())
    yield store, adapters, call
    server.shutdown(); server.server_close(); worker.join(timeout=2); store.close()


def test_complete_core_HTTP_auth_Demo_threeTasks_QA_replay_and_export(service):
    store, _, call = service
    assert call('/health')[0] == 200
    assert call('/v642/status')[1]['old_UI_migrated'] is False
    assert call('/v642/instances', dict(domain='warehouse', participant_reference='software-only', group='A'))[0] == 403
    status, result = call('/v642/instances', dict(domain='warehouse', participant_reference='software-only', group='A'), administrator=True)
    assert status == 200
    token, iid = result['session_token'], result['instance']['id']
    base = '/v642/instances/'+iid
    assert call(base)[0] == 401
    assert call(base, token='wrong-token-000000000')[0] == 404
    assert call(base+'/demonstration', {}, token=token)[1]['two_frozen_NN']
    assert call(base+'/demonstration/complete', {}, token=token)[0] == 200
    for task in (1, 2, 3):
        assert call(base+'/tasks', {}, token=token)[0] == 200
        if task == 2:
            status, answer = call(base+'/questions', dict(question='why?', turn=0, language='en', human_actions=['up']), token=token)
            assert status == 200 and answer['status'] == 'insufficient_evidence'
            assert answer['counterfactual']['binding'] == result['instance']['binding']
        for turn in (0, 1):
            status, _ = call(base+'/actions', dict(command_id=f'{task}-{turn}', expected_turn=turn, human_action='wait'), token=token)
            assert status == 200
    assert call(base, token=token)[1]['instance']['stage'] == 'complete'
    assert call(base+'/frame?task=2&turn=1', token=token)[1]['turn'] == 1
    assert call(base+'/replay?task=2', token=token)[1]['all_AI_actions_recomputed']
    assert call(base+'/export', token=token)[0] == 403
    assert call(base+'/export', token=token, administrator=True)[1]['all_AI_frames_recomputed']
    assert call('/v642/research/instances')[0] == 403
    assert call('/v642/research/instances', administrator=True)[1]['instances'][0]['id'] == iid
    assert call('/v642/research/instances/'+iid+'/export', administrator=True)[1]['all_AI_frames_recomputed']
    # The old controller/HTML paths are not exposed by this new JSON API.
    assert call('/pong/')[0] == 404


def test_HTTP_pilot_requires_actual_catalog_acceptance_before_every_route(service):
    store, adapters, call = service
    store.mode = 'pilot'  # Synthetic test of HTTP fail-closed routing only.
    for path, payload in [('/v642/status', None), ('/v642/instances', dict(domain='warehouse', participant_reference='x', group='A')),
                          ('/v642/instances/missing', None), ('/v642/instances/missing/tasks', {})]:
        status, result = call(path, payload)
        assert status == 503 and result['pilot_opened'] is False
    assert store.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0] == 0
    assert all(not adapter.calls for adapter in adapters.values())


def test_HTTP_duplicate_JSON_unknown_fields_and_foreign_admin_rejected(service):
    _, _, call = service
    assert call('/v642/instances', raw=b'{"domain":"pong","domain":"warehouse","participant_reference":"x"}')[0] == 400
    assert call('/v642/instances', dict(domain='warehouse', participant_reference='x', admin=True))[0] == 400
    assert call('/v642/instances', dict(domain='warehouse', participant_reference='x', group='C'))[0] == 403
    assert call('/archive/old')[0] == 403


def test_HTTP_missing_bound_asset_becomes503_without_fallback(service):
    store, _, call = service
    status, result = call('/v642/instances', dict(domain='warehouse', participant_reference='x', group='A'), administrator=True)
    assert status == 200
    binding = result['instance']['binding']
    store.registry.catalog.assets.path(binding['actor_sha256']).unlink()
    status, result = call('/v642/instances/'+result['instance']['id'], token=result['session_token'])
    assert status == 503 and result['pilot_opened'] is False
