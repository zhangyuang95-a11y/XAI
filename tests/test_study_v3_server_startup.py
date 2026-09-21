"""Cold-start computation and health probes must not hold enrollment locks.

These checks use real local domain demonstrations, SQLite and an HTTP health
request. They do not claim PostgreSQL or production deployment validation.
"""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
import threading

from study_v3.config import Settings
from study_v3.database import Database
from study_v3.registry import MODULES, demonstration, engine
from study_v3.server import make_server


def test_cold_demonstrations_finish_before_database_transactions(tmp_path, monkeypatch):
    demonstration.cache_clear()
    cold_calls = []
    transaction_depth = 0
    original_transaction = Database.transaction

    @contextmanager
    def tracked_transaction(self, *args, **kwargs):
        nonlocal transaction_depth
        assert set(cold_calls) == set(MODULES)-{'kitchen'}, "Database opened before demos finished"
        transaction_depth += 1
        try:
            with original_transaction(self, *args, **kwargs) as connection:
                yield connection
        finally:
            transaction_depth -= 1

    monkeypatch.setattr(Database, 'transaction', tracked_transaction)
    for domain in MODULES:
        module = engine(domain)
        original_demo = module.demonstration

        def cold_demo(domain=domain, original_demo=original_demo):
            assert transaction_depth == 0, "Cold demo ran inside a database transaction"
            result = original_demo()
            cold_calls.append(domain)
            return result

        monkeypatch.setattr(module, 'demonstration', cold_demo)

    server = None
    try:
        server = make_server(Settings(database=str(tmp_path / 'startup.sqlite3')), port=0, explainer=object())
        for domain in MODULES:
            _, view = server.store.create({'participant_id': 'startup-test-' + domain,
                'domain': domain, 'group': 'A', 'mode': 'test', 'consent': True}, admin=True)
            assert view['stage'] == 'demo'
            if domain=='kitchen':assert view['tutorial']['state'] and 'demo' not in view
            else:assert view['demo']['frames']
        assert cold_calls == [d for d in MODULES if d!='kitchen'], "Enrollment recomputed a cold demo"
    finally:
        if server:
            server.server_close()
        demonstration.cache_clear()


def test_health_remains_read_only_while_enrollment_writer_is_active(tmp_path, monkeypatch):
    server = make_server(Settings(database=str(tmp_path / 'health.sqlite3')), port=0, explainer=object())
    transaction_modes = []
    original_transaction = server.store.db.transaction

    @contextmanager
    def health_transaction(*args, **kwargs):
        transaction_modes.append(kwargs.get('read_only', False))
        assert kwargs.get('read_only') is True, "Health requested the global write lock"
        with original_transaction(*args, **kwargs) as connection:
            yield connection

    monkeypatch.setattr(server.store.db, 'transaction', health_transaction)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        # Hold the same Database writer lock used by enrollment. The HTTP
        # thread must complete a separate WAL read before this writer exits.
        with original_transaction() as writer:
            writer.execute('SELECT 1')
            client.request('GET', '/health')
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read())['status'] == 'ok'
        assert transaction_modes == [True]
    finally:
        client.close()
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
