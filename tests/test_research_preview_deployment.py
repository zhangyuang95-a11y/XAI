"""Real isolated HTTP composition; no scientific acceptance or participant DB."""
import gzip
from http.client import HTTPConnection
import json

import pytest

from ui.researchpreviewserver import (MAX_BODY, VERSION, make_server,
    private_admin, running_backends)

ADMIN = 'synthetic-private-test-key-never-production-20261030'


def request(server, path, payload=None, headers=None):
    connection = HTTPConnection(*server.server_address, timeout=120)
    offered = dict(headers or {})
    if payload is not None:
        offered['Content-Type'] = 'application/json'
    connection.request('POST' if payload is not None else 'GET', path,
        body=None if payload is None else json.dumps(payload), headers=offered)
    response = connection.getresponse()
    raw = response.read()
    if response.getheader('Content-Encoding') == 'gzip':
        raw = gzip.decompress(raw)
    status, fields = response.status, dict(response.getheaders())
    connection.close()
    value = json.loads(raw) if fields['Content-Type'].startswith('application/json') else raw
    return status, fields, value


@pytest.fixture(scope='module')
def composed(tmp_path_factory):
    from study_vnext.server import compose_preview, make_server as study_server
    from ui.model_preview_server import Preview, make_server as manual_server
    root = tmp_path_factory.mktemp('isolated-composition')
    manual = Preview(str(root / 'manual.sqlite3'))
    study, preview = compose_preview(str(root / 'new-study.sqlite3'))
    backends = (manual_server(manual), study_server(study, admin_token=ADMIN))
    public = make_server(*backends, commit='synthetic-deploy-commit')
    with running_backends(*backends, public):
        yield public, study, root
    study.close()
    preview.close()
    manual.close()


def test_credential_is_explicit_and_never_reads_database_environment():
    assert private_admin({'POLICYLENS_STUDY_PREVIEW_ADMIN': ADMIN,
        'POLICYLENS_ADMIN_TOKEN': 'other'*10, 'POLICYLENS_DATABASE_URL': 'must-not-be-used'}) == ADMIN
    assert private_admin({'POLICYLENS_ADMIN_TOKEN': ADMIN}) == ADMIN
    with pytest.raises(ValueError, match='admin_required'):
        private_admin({'POLICYLENS_DATABASE_URL': ADMIN})
    with pytest.raises(ValueError):
        private_admin({'POLICYLENS_STUDY_PREVIEW_ADMIN': 'short', 'POLICYLENS_ADMIN_TOKEN': ADMIN})


def test_old_manual_and_new_study_pages_are_distinct_and_compressed(composed):
    server, _, _ = composed
    for path in ('/', '/warehouse/', '/pong/', '/kitchen/'):
        status, _, body = request(server, path)
        assert status == 200 and b'/preview-assets/app.js' in body
        assert b'href="/study/"' in body
    status, headers, body = request(server, '/study/', headers={'Accept-Encoding': 'gzip'})
    assert status == 200 and b'/study-assets/app.js' in body
    assert headers['Content-Encoding'] == 'gzip'
    assert 'preview_router;dur=' in headers['Server-Timing']
    for path in ('/study-assets/app.js', '/study-assets/board.js', '/study-assets/input-queue.js'):
        assert request(server, path)[0] == 200
    assert request(server, '/prolific/')[0] == 404


def test_actual_release_identity_stays_closed_and_models_unchanged(composed):
    server, _, _ = composed
    for path in ('/health', '/api/release'):
        status, _, value = request(server, path, headers={'Accept-Encoding': 'gzip'})
        assert status == 200
        assert value['deployment_runtime'] == VERSION
        assert value['deployment_commit'] == 'synthetic-deploy-commit'
        assert value['pilot_opened'] is value['recruitment_opened'] is False
        assert value['database_is_participant_database'] is False
    models = request(server, '/manual-api/models')[2]
    assert models['models']['warehouse']['acceptance'] == 'calibration_pair_failed'
    assert models['models']['pong']['acceptance'] == 'historical_v5_passed'
    assert models['models']['kitchen']['acceptance'] == 'historical_v5_passed'
    status = request(server, '/study-api/status')[2]
    assert status['mode'] == 'preview' and status['recruitment_opened'] is False
    assert status['question_service_mode'] == 'template_only'


def test_auth_csrf_and_separate_databases_preserved(composed):
    server, study, root = composed
    from study_vnext.store import CONSENT_VERSION
    enrollment = dict(domain='warehouse', participant_reference='synthetic-composition-only',
        consent=True, consent_version=CONSENT_VERSION, group='A')
    assert request(server, '/study-api/enroll', enrollment)[0] == 403
    assert request(server, '/study-api/enroll', enrollment,
        {'X-Admin-Token': ADMIN, 'Origin': 'https://untrusted.example'})[0] == 403
    status, _, result = request(server, '/study-api/enroll', enrollment,
        {'X-Admin-Token': ADMIN, 'Origin': 'http://%s:%s' % server.server_address})
    assert status == 200 and result['instance']['group'] == 'A'
    identifier, token = result['instance']['id'], result['session_token']
    assert request(server, f'/study-api/{identifier}/view')[0] == 401
    assert request(server, f'/study-api/{identifier}/view',
        headers={'Authorization': 'Bearer ' + token})[0] == 200
    # Old manual start/session/action remain usable with a separate token.
    status, _, manual = request(server, '/manual-api/start',
        dict(domain='warehouse', variant='regularized', task=1, seed=40786))
    assert status == 200
    auth = {'Authorization': 'Bearer ' + manual['session_token']}
    assert request(server, '/manual-api/action',
        dict(command_id='compose-one-step', expected_turn=0, human_action='wait'), auth)[2]['state']['turn'] == 1
    assert request(server, '/manual-api/session', headers=auth)[2]['state']['turn'] == 1
    assert (root / 'manual.sqlite3').exists() and (root / 'new-study.sqlite3').exists()
    assert study.mode == 'preview'


def test_bounded_body_and_absolute_target_rejected_without_forwarding(composed):
    server, _, _ = composed
    connection = HTTPConnection(*server.server_address, timeout=10)
    connection.request('POST', '/study-api/enroll', body=b'',
        headers={'Content-Length': str(MAX_BODY + 1)})
    response = connection.getresponse()
    assert response.status == 413
    response.read(); connection.close()
    assert request(server, 'http://untrusted.example/study-api/status')[0] == 400
