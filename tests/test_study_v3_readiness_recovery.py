"""Readiness retries use injected answers; these are not provider QA evidence."""
from dataclasses import replace
from http.client import HTTPConnection
import json
import threading

import pytest

from study_v3.config import Settings
from study_v3.server import QAReadinessMonitor, make_server
from study_v3.store import Store


def pilot_settings(tmp_path):
    return Settings(database=str(tmp_path / 'readiness.sqlite3'), mode='pilot',
        persistent=True, verified=True, admin_token='test-only-admin',
        llm_base_url='https://test-only-provider.invalid', llm_model='test-only')


@pytest.fixture
def store(tmp_path):
    instance = Store(pilot_settings(tmp_path))
    instance.qa_healthy = False
    try:
        yield instance
    finally:
        instance.db.close()


def test_failed_probes_recover_only_after_validated_response_and_rate_limit(store, caplog):
    now = [0.0]
    calls = []
    outcomes = [RuntimeError('private-api-key-must-not-be-logged'),
                {'status': 'unavailable'}, {'status': 'unknown'}, None,
                {'status': 'answered', 'answer': 'Verified fixture answer.'}]

    def probe():
        calls.append(now[0])
        value = outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monitor = QAReadinessMonitor(store, probe, interval=30, clock=lambda: now[0])
    try:
        for number in range(4):
            assert monitor.check() is False
            assert store.ready is False
            assert len(calls) == number + 1
            # Repeated checks cannot hammer the provider while it is failing.
            now[0] += 29.9
            assert monitor.check() is False
            assert len(calls) == number + 1
            now[0] += 0.1
        assert monitor.check() is True
        assert store.ready is True
        now[0] += 1000
        assert monitor.check() is False
        assert len(calls) == 5
        assert 'private-api-key' not in caplog.text
        assert 'retry in 30 seconds' in caplog.text
        # A later genuine QA failure also recovers without a process restart.
        store.qa_healthy = False
        outcomes.append({'status': 'answered'})
        assert monitor.check() is True
        assert store.ready is True and len(calls) == 6
    finally:
        monitor.stop()


@pytest.mark.parametrize('fields', [
    {'mode': 'preview'}, {'verified': False}, {'persistent': False},
    {'admin_token': ''}, {'llm_base_url': ''}, {'llm_model': ''},
])
def test_configuration_gates_cannot_be_bypassed_by_reprobe(tmp_path, fields):
    instance = Store(replace(pilot_settings(tmp_path), **fields))
    calls = []
    monitor = QAReadinessMonitor(instance, lambda: calls.append(True))
    try:
        instance.qa_healthy = True
        monitor.start()
        assert monitor.check() is False
        assert instance.ready is False and calls == []
        assert monitor._thread is None
    finally:
        monitor.stop()
        instance.db.close()


def test_single_flight_and_close_cancel_pending_result(store):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def probe():
        calls.append(True)
        entered.set()
        assert release.wait(3)
        finished.set()
        return {'status': 'answered'}

    monitor = QAReadinessMonitor(store, probe)
    monitor.start()
    try:
        assert entered.wait(3)
        for _ in range(3):
            monitor.start()
        results = []
        contenders = [threading.Thread(target=lambda: results.append(monitor.check())) for _ in range(8)]
        for worker in contenders:
            worker.start()
        for worker in contenders:
            worker.join(2)
            assert not worker.is_alive()
        assert results == [False] * 8 and len(calls) == 1
        monitor.stop()  # Does not wait for the blocked provider request.
        assert not finished.is_set()
        release.set()
        monitor._thread.join(3)
        assert not monitor._thread.is_alive()
        assert store.ready is False  # Discard the response received after close.
        monitor.start()
        assert monitor.check() is False and len(calls) == 1
    finally:
        release.set()
        monitor.stop()


@pytest.mark.parametrize('probe_success,newer_result', [(False, True), (True, False)])
def test_old_probe_cannot_replace_newer_participant_qa_health(store, probe_success, newer_result):
    entered, release = threading.Event(), threading.Event()

    def probe():
        entered.set()
        assert release.wait(3)
        return {'status': 'answered' if probe_success else 'unavailable'}

    monitor = QAReadinessMonitor(store, probe)
    worker = threading.Thread(target=monitor.check)
    try:
        worker.start()
        assert entered.wait(3)
        store.qa_healthy = newer_result
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert store.qa_healthy is newer_result
    finally:
        release.set()
        worker.join(3)
        monitor.stop()


def test_http_liveness_stays_available_during_probe_then_live_readiness_recovers(tmp_path, monkeypatch):
    # Demos are covered by cold-start tests; isolate the provider I/O here.
    monkeypatch.setattr('study_v3.server.demonstration', lambda domain: None)
    entered, release = threading.Event(), threading.Event()

    class BlockingExplainer:
        def answer(self, eng, state, decision, question, language, dialogue, history):
            assert state['domain'] == 'pong' and state['task'] == 2
            assert decision == eng.decide(state)
            assert history == [eng.public_state(state)]
            entered.set()
            assert release.wait(5)
            return {'status': 'answered', 'answer': 'Validated stub.'}

    server = make_server(pilot_settings(tmp_path), port=0, explainer=BlockingExplainer())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        assert entered.wait(3)
        for path in ('/health', '/api/release'):
            client.request('GET', path)
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read())['study_ready'] is False
        # No database transaction is held during the provider request.
        with server.store.db.transaction() as db:
            assert db.one('SELECT COUNT(*) AS n FROM pl3_participants')['n'] == 0
        release.set()
        for _ in range(200):
            if server.store.ready:
                break
            threading.Event().wait(0.01)
        assert server.store.ready
        for path in ('/health', '/api/release'):
            client.request('GET', path)
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read())['study_ready'] is True
        assert server.store.export()['participants'] == []
    finally:
        release.set()
        client.close()
        server.shutdown()
        worker.join(3)
        server.server_close()
        assert server.qa_monitor._stopped.is_set()


def test_shutdown_interrupts_retry_wait_without_another_provider_call(store):
    attempted = threading.Event()
    calls = []

    def probe():
        calls.append(True)
        attempted.set()
        return {'status': 'unavailable'}

    monitor = QAReadinessMonitor(store, probe, interval=3600)
    monitor.start()
    assert attempted.wait(3)
    monitor.stop()
    assert not monitor._thread.is_alive()
    assert len(calls) == 1 and store.ready is False
