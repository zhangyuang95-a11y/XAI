"""Local tools exercise ID enrollment using isolated stores and HTTP mocks only."""
import json
from pathlib import Path
import stat
import sys

import httpx
import pytest
from sqlalchemy import select

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/cooperative_kitchen'
sys.path.insert(0, str(SCRIPTS))
import browser_fixture_server as fixture
import study_admin
from backend.cooperative_kitchen.study import KitchenStudy
from ui.cooperative_kitchen_store import StudyError, participants, runs


def test_fixed_browser_block_does_not_pre_register_ids(tmp_path):
    study = KitchenStudy(tmp_path, 'sqlite:///:memory:', namespace='test', allow_sqlite=True,
        test_mode=True, release={'study_ready': True, 'versions': {'fixture': 'tools-only'}}, enrollment_mode='internal_pilot')
    try:
        ids = fixture.configure_test_assignments(study)
        with study.store.transaction() as db:
            assert not db.execute(select(participants)).first()
            assert not db.execute(select(runs)).first()
        for expected in ('A', 'B'):
            token, view = study.join({'operation_id': 'create-' + expected, 'mode': 'pilot',
                                     'participant_id': ids[expected], 'language': 'zh'})
            with study.store.transaction() as db: actual = study.store.run(db, token)
            assert actual['condition'] == expected
            assert view['run']['participant_id'] == ids[expected] and view['run']['phase'] == 'consent'
        with pytest.raises(StudyError) as error:
            study.join({'operation_id': 'different-owner', 'mode': 'pilot', 'participant_id': ids['A']})
        assert error.value.code == 'participant_id_taken'
        with pytest.raises(StudyError) as error: study.create_invitations({'count': 4})
        assert error.value.status == 410
        with pytest.raises(ValueError): fixture.configure_test_assignments(study)
    finally: study.store.engine.dispose()


def test_fixed_browser_assignments_reject_non_test_namespace(tmp_path):
    study = KitchenStudy(tmp_path, 'sqlite:///:memory:', namespace='development', allow_sqlite=True)
    try:
        with pytest.raises(ValueError, match='isolated test'): fixture.configure_test_assignments(study)
    finally: study.store.engine.dispose()


def install_http_mock(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv('KITCHEN_URL', 'https://kitchen.example')
    monkeypatch.setenv('KITCHEN_ADMIN_KEY', 'test-only-admin')


def test_retired_invitation_cli_makes_no_request(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Retired command must not send a request.')
    monkeypatch.setattr(httpx, 'Client', forbidden)
    with pytest.raises(SystemExit) as result: study_admin.main(['invitations', '--count', '4', '--output', 'unused.json'])
    assert result.value.code == 2


def test_admin_status_uses_only_authenticated_endpoint(monkeypatch, capsys):
    def handler(request):
        assert request.url == 'https://kitchen.example/api/admin/status'
        assert request.headers['X-Kitchen-Admin-Key'] == 'test-only-admin'
        return httpx.Response(200, json={'service': {'enrollment': {'mode': 'internal_pilot', 'enabled': True}}})
    install_http_mock(monkeypatch, handler)
    study_admin.main(['status'])
    output = capsys.readouterr().out
    assert 'internal_pilot' in output and 'test-only-admin' not in output


def test_admin_retry_preserves_idempotent_payload_and_private_receipt(tmp_path, monkeypatch, capsys):
    saved = tmp_path / 'retry.json'
    def handler(request):
        assert request.url.path == '/api/admin/retry'
        assert json.loads(request.content) == {'operation_id': 'retry-operation', 'run_id': 'run-one', 'reason': 'Connection interrupted'}
        return httpx.Response(200, json={'run_id': 'run-two', 'retry_id': 1, 'participant_id': 'P001'})
    install_http_mock(monkeypatch, handler)
    study_admin.main(['retry', '--run-id', 'run-one', '--reason', 'Connection interrupted',
                      '--operation-id', 'retry-operation', '--output', str(saved)])
    assert json.loads(saved.read_text())['retry_id'] == 1
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert 'retry-operation' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        study_admin.main(['retry', '--run-id', 'run-one', '--reason', 'Connection interrupted',
                          '--operation-id', 'retry-operation', '--output', str(saved)])


def test_admin_redirect_never_forwards_key(monkeypatch, capsys):
    called = []
    def handler(request):
        called.append(str(request.url))
        return httpx.Response(302, headers={'Location': 'https://other.example/leak'})
    install_http_mock(monkeypatch, handler)
    with pytest.raises(SystemExit) as result: study_admin.main(['status'])
    assert result.value.code == 1 and called == ['https://kitchen.example/api/admin/status']
    assert 'test-only-admin' not in capsys.readouterr().err
