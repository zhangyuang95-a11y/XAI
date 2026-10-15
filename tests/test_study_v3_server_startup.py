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
from study_v3.registry import MODULES
from study_v3.server import make_server
import study_v3.server as service


def test_accepted_pair_demonstrations_finish_before_database_transactions(tmp_path, monkeypatch):
    cold_calls = []
    transaction_depth = 0
    original_transaction = Database.transaction
    expected = [(domain, f"{domain}-{variant}-{'v6' if domain == 'warehouse' else 'v5'}")
                for domain in MODULES for variant in ('regularized', 'raw')]

    @contextmanager
    def tracked_transaction(self, *args, **kwargs):
        nonlocal transaction_depth
        assert cold_calls == expected, "Database opened before both NN demos per domain finished"
        transaction_depth += 1
        try:
            with original_transaction(self, *args, **kwargs) as connection:
                yield connection
        finally:
            transaction_depth -= 1

    monkeypatch.setattr(Database, 'transaction', tracked_transaction)
    monkeypatch.setattr(service, 'all_policies_accepted', lambda: True)
    monkeypatch.setattr(service, 'policy_id',
                        lambda domain, group: f"{domain}-{'raw' if group == 'C' else 'regularized'}-"
                        f"{'v6' if domain == 'warehouse' else 'v5'}")

    def cold_demo(domain, policy_id):
        assert transaction_depth == 0, "Cold demo ran inside a database transaction"
        cold_calls.append((domain, policy_id))
        return {'captions': [], 'frames': []}

    monkeypatch.setattr(service, 'demonstration', cold_demo)

    server = None
    try:
        server = make_server(Settings(database=str(tmp_path / 'startup.sqlite3')), port=0, explainer=object())
        assert cold_calls == expected
    finally:
        if server:
            server.server_close()


def test_incomplete_v5_assets_skip_cold_demonstrations_and_release_row(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'all_policies_accepted', lambda: False)

    def unexpected_demo(domain, policy_id):
        raise AssertionError(f'prewarmed an incomplete policy: {domain} {policy_id}')

    monkeypatch.setattr(service, 'demonstration', unexpected_demo)
    server = make_server(Settings(database=str(tmp_path / 'incomplete.sqlite3')),
                         port=0, explainer=object())
    try:
        assert server.release['policy_assets_accepted'] is False
        with server.store.db.transaction(read_only=True) as db:
            assert db.one('SELECT id FROM pl3_releases') is None
    finally:
        server.server_close()


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
