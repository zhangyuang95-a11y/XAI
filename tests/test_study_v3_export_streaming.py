"""Export memory/lifetime regressions using SQLite and a PostgreSQL test double.

No production requests or model calls are made. The PostgreSQL double checks
cursor use, not live PostgreSQL connectivity or production memory consumption.
"""
from contextlib import contextmanager
import csv
import io
import json
from types import SimpleNamespace

import pytest

from study_v3.config import Settings
from study_v3.database import Connection
from study_v3.store import Store
from study_v3 import server as server_module


TABLES = ('participants', 'enrollments', 'instances', 'runs', 'frames',
          'questions', 'questionnaires', 'timings', 'releases','tutorials','tutorial_events')
RELEASE = 'export-selected-release'
ADMIN = 'export-unit-test-only-token'
LEGACY_MARKER = 'EXCLUDED_LARGE_LEGACY_PAYLOAD'


@pytest.fixture
def export_store(tmp_path):
    store = Store(Settings(database=str(tmp_path / 'export.sqlite3'), admin_token=ADMIN))
    with store.db.transaction() as db:
        for release in (RELEASE, 'older-release'):
            db.execute('INSERT INTO pl3_releases VALUES(?,?,?)', (release, '{}', 1.25))
        # Reverse insertion order makes implicit row order an observable bug.
        for suffix, release, mode in (('z', RELEASE, 'test'),
                                      ('old', 'older-release', 'test'),
                                      ('preview', RELEASE, 'preview'),
                                      ('a', RELEASE, 'test')):
            pid, iid, rid = ('participant-' + suffix, 'instance-' + suffix, 'run-' + suffix)
            payload = {'unicode': '鸡蛋', 'quote': 'a,"b"\nc', 'suffix': suffix}
            if suffix in ('old', 'preview'):
                payload['large'] = LEGACY_MARKER + 'x' * (128 * 1024)
            encoded = json.dumps(payload, ensure_ascii=False)
            db.execute('INSERT INTO pl3_participants VALUES(?,?,?,?)',
                       (pid, 'A', 'private-recovery-' + suffix, 1.25))
            db.execute('INSERT INTO pl3_instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (iid, pid, 'kitchen', release, 'A', mode, 'complete', 9, rid, 6,
                        'en', 123, 1.25, 2.5))
            db.execute('INSERT INTO pl3_enrollments VALUES(?,?,?,?,?,?)',
                       (iid, 1, 'en', 'test', 'scenario-test', 1.25))
            db.execute('INSERT INTO pl3_runs VALUES(?,?,?,?,?,?,?,?,?)',
                       (rid, iid, 2, 123, encoded, 'complete', '{}', 1.25, 2.5))
            for turn in (2, 0, 1):
                db.execute('INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)',
                           (rid, turn, encoded, '{}', '{}', 'wait', 1.25))
            db.execute('INSERT INTO pl3_questions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       ('question-' + suffix, iid, rid, rid, 0, 'Where is the egg?',
                        'en', '{}', 'answered', 1.25, 1.5, 1.75))
            db.execute('INSERT INTO pl3_questionnaires VALUES(?,?,?,?)',
                       (iid, '{}', '{}', 2.5))
            db.execute('INSERT INTO pl3_timings VALUES(?,?,?,?,?,?)',
                       ('timing-' + suffix, iid, 2, 'active', 1.25, 2.5))
        db.execute('INSERT INTO pl3_participants VALUES(?,?,?,?)',
                   ('orphan-participant', 'B', 'private-orphan-recovery', 1.25))
    try:
        yield store
    finally:
        store.db.close()


def fail_materializing(*args, **kwargs):
    pytest.fail('Export used a materializing compatibility read')


def observe_transactions(monkeypatch, store):
    original = store.db.transaction
    tracker = SimpleNamespace(active=0, completed=0)

    @contextmanager
    def tracked(*args, **kwargs):
        with original(*args, **kwargs) as db:
            tracker.active += 1
            try:
                yield db
            finally:
                tracker.active -= 1
        tracker.completed += 1

    monkeypatch.setattr(store.db, 'transaction', tracked)
    return tracker


def test_selected_export_filters_large_rows_before_the_cursor_yields(export_store, monkeypatch):
    original = Connection.iterate
    crossed_cursor = []

    def observed(self, sql, values=(), batch_size=16):
        for row in original(self, sql, values, batch_size):
            assert LEGACY_MARKER not in json.dumps(row)
            assert 'recovery_hash' not in row
            crossed_cursor.append(row)
            yield row

    monkeypatch.setattr(Connection, 'iterate', observed)
    monkeypatch.setattr(Connection, 'all', fail_materializing)
    rows = list(export_store.iter_export(RELEASE, 'test', batch_size=2))
    assert len(crossed_cursor) == len(rows)
    by_table = {name: [r['record'] for r in rows if r['table'] == name] for name in TABLES}
    assert [r['id'] for r in by_table['participants']] == ['participant-a', 'participant-z']
    assert [r['id'] for r in by_table['instances']] == ['instance-a', 'instance-z']
    assert [r['id'] for r in by_table['releases']] == [RELEASE]
    assert [(r['run_id'], r['turn']) for r in by_table['frames']] == [
        (rid, turn) for rid in ('run-a', 'run-z') for turn in range(3)]
    for name in ('enrollments', 'runs', 'questions', 'questionnaires', 'timings'):
        assert len(by_table[name]) == 2
    assert list(export_store.iter_export(RELEASE + "' OR 1=1 --", 'test')) == []


@pytest.mark.parametrize('release,mode', [(None, None), (RELEASE, 'test'),
                                         (None, 'preview'), ('older-release', None)])
def test_export_order_and_compatibility_are_deterministic(export_store, release, mode, monkeypatch):
    original = Connection.iterate

    def no_credentials(self, *args, **kwargs):
        for row in original(self, *args, **kwargs):
            assert 'recovery_hash' not in row
            yield row

    monkeypatch.setattr(Connection, 'iterate', no_credentials)
    rows = list(export_store.iter_export(release, mode, batch_size=1))
    assert rows == list(export_store.iter_export(release, mode, batch_size=3))
    assert [TABLES.index(row['table']) for row in rows] == sorted(TABLES.index(row['table']) for row in rows)
    grouped = {name: [row['record'] for row in rows if row['table'] == name] for name in TABLES}
    for name, records in grouped.items():
        def key(record):
            if name == 'frames':
                return record['run_id'], record['turn']
            return record.get('id', record.get('instance_id'))
        assert records == sorted(records, key=key)
    assert export_store.export(release, mode) == grouped
    assert list(export_store.export(release, mode)) == list(TABLES)


@pytest.mark.parametrize('csv_format', [False, True])
def test_export_file_has_identical_records_and_closes_transaction_before_delivery(export_store, monkeypatch, csv_format):
    expected = list(export_store.iter_export(RELEASE, 'test'))
    tracker = observe_transactions(monkeypatch, export_store)
    monkeypatch.setattr(export_store, 'export', fail_materializing)
    with server_module.export_file(export_store, RELEASE, 'test', csv_format=csv_format) as stream:
        assert tracker.active == 0 and tracker.completed == 1
        assert not stream.closed
        content = stream.read()
        assert isinstance(content, bytes)
        if csv_format:
            reader = csv.DictReader(io.StringIO(content.decode('utf-8'), newline=''))
            assert reader.fieldnames == ['table', 'record_json']
            actual = [{'table': row['table'], 'record': json.loads(row['record_json'])} for row in reader]
        else:
            actual = [json.loads(line) for line in content.splitlines()]
        assert actual == expected
    assert stream.closed


@pytest.fixture
def handler_class(export_store, monkeypatch):
    # Build the real route closure without opening a socket or simulating demos.
    monkeypatch.setattr(server_module, 'Store', lambda *a, **k: export_store)
    monkeypatch.setattr(server_module, 'demonstration', lambda *a, **k: None)
    monkeypatch.setattr(server_module, 'manifest', lambda *a: {'source_sha256': 'test-only'})
    monkeypatch.setattr(server_module, 'StudyHTTPServer',
                        lambda address, handler: SimpleNamespace(RequestHandlerClass=handler))
    return server_module.make_server(export_store.settings, explainer=object()).RequestHandlerClass


def request_handler(handler_class, path, authorization=None):
    handler = object.__new__(handler_class)
    handler.path = path
    handler.command = 'GET'
    handler.headers = {} if authorization is None else {'Authorization': authorization}
    handler.wfile = io.BytesIO()
    handler.sent_headers = {}
    handler.send_response = lambda status: setattr(handler, 'sent_status', status)
    handler.send_header = lambda name, value: handler.sent_headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    return handler


@pytest.mark.parametrize('authorization', [None, 'Bearer wrong-token'])
def test_unauthorized_export_does_not_begin_generation(handler_class, export_store, monkeypatch, authorization):
    monkeypatch.setattr(server_module, 'export_file', fail_materializing)
    monkeypatch.setattr(export_store, 'iter_export', fail_materializing)
    monkeypatch.setattr(export_store, 'export', fail_materializing)
    handler = request_handler(handler_class, '/api/study/admin/export?format=csv', authorization)
    handler.do_GET()
    assert handler.sent_status == 403
    assert json.loads(handler.wfile.getvalue()) == {'error': 'researcher_access_required'}


@pytest.mark.parametrize('csv_format', [False, True])
def test_http_export_reads_bounded_chunks_after_database_release(handler_class, export_store, monkeypatch, csv_format):
    expected = list(export_store.iter_export())
    tracker = observe_transactions(monkeypatch, export_store)
    monkeypatch.setattr(export_store, 'export', fail_materializing)
    original_export_file = server_module.export_file
    reads, writes, files = [], [], []

    class TrackedReader:
        def __init__(self, file):
            self.file = file

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, size=-1):
            assert tracker.active == 0
            assert size == 64 * 1024
            reads.append(size)
            return self.file.read(size)

    @contextmanager
    def tracked_export(*args, **kwargs):
        with original_export_file(*args, **kwargs) as stream:
            assert tracker.active == 0
            files.append(stream)
            yield TrackedReader(stream)

    class TrackedWriter(io.BytesIO):
        def write(self, data):
            assert tracker.active == 0
            assert len(data) <= 64 * 1024
            writes.append(len(data))
            return super().write(data)

    monkeypatch.setattr(server_module, 'export_file', tracked_export)
    handler = request_handler(handler_class, '/api/study/admin/export?format=' + ('csv' if csv_format else 'jsonl'),
                              'Bearer ' + ADMIN)
    handler.wfile = TrackedWriter()
    handler.do_GET()
    assert handler.sent_status == 200
    assert len(reads) > 3 and len(writes) > 3
    assert tracker.active == 0 and tracker.completed == 1
    assert all(file.closed for file in files)
    content = handler.wfile.getvalue()
    assert int(handler.sent_headers['Content-Length']) == len(content)
    assert 'no-store' in handler.sent_headers['Cache-Control']
    if csv_format:
        assert handler.sent_headers['Content-Type'].startswith('text/csv')
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(1024 * 1024)
            actual = [{'table': row['table'], 'record': json.loads(row['record_json'])}
                      for row in csv.DictReader(io.StringIO(content.decode('utf-8'), newline=''))]
        finally:
            csv.field_size_limit(previous_limit)
    else:
        assert handler.sent_headers['Content-Type'].startswith('application/x-ndjson')
        actual = [json.loads(line) for line in content.splitlines()]
    assert actual == expected


@pytest.mark.parametrize('early_close', [False, True])
def test_postgres_iteration_uses_server_cursor_and_closes_it(early_close):
    class Cursor:
        def __init__(self):
            self.closed = False
            self.rows = iter([{'id': str(i)} for i in range(7)])
            self.batches = []

        def execute(self, sql, values=None):
            self.executed = sql, values

        def fetchmany(self, size):
            self.batches.append(size)
            batch = []
            for _ in range(size):
                try:
                    batch.append(next(self.rows))
                except StopIteration:
                    break
            return batch

        def fetchall(self):
            pytest.fail('PostgreSQL export buffered all rows')

        def close(self):
            self.closed = True

    class Postgres:
        def __init__(self):
            self.created = []

        def execute(self, *args, **kwargs):
            pytest.fail('PostgreSQL export used an unnamed buffering cursor')

        def cursor(self, *, name, withhold):
            assert isinstance(name, str) and name
            assert withhold is False
            cursor = Cursor()
            self.created.append((name, cursor))
            return cursor

    postgres = Postgres()
    connection = Connection(postgres, True)
    iterator = connection.iterate('SELECT id FROM example WHERE release_id=?', ('selected',), batch_size=2)
    assert postgres.created == []
    first = next(iterator)
    assert first == {'id': '0'}
    _, cursor = postgres.created[0]
    assert cursor.executed == ('SELECT id FROM example WHERE release_id=%s', ('selected',))
    assert cursor.batches == [2]
    assert not cursor.closed
    if early_close:
        iterator.close()
    else:
        assert [first, *iterator] == [{'id': str(i)} for i in range(7)]
    assert cursor.closed
    assert all(size == 2 for size in cursor.batches)
    second = connection.iterate('SELECT id FROM example', batch_size=2)
    next(second)
    second.close()
    assert postgres.created[1][0] != postgres.created[0][0]
    assert postgres.created[1][1].closed
