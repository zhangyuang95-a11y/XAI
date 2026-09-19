"""Real SQLite concurrency/recovery and explicit PostgreSQL adapter test doubles.

PostgreSQL tests verify generated SQL, pooling and failure paths against a fake
connection. They do not establish connectivity or behavior of a live PG server.
"""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import threading
import types

import pytest

from study_v3.database import Database, GLOBAL_ADVISORY_LOCK, TRANSACTION_TIMEOUTS, advisory_lock_key


@pytest.fixture
def database(tmp_path):
    db = Database(str(tmp_path / 'study.sqlite3'))
    yield db
    db.close()


def insert_participant(db, name='test-participant', created=1790000000.125):
    db.execute('INSERT INTO pl3_participants VALUES(?,?,?,?)', (name, 'A', 'test-only', created))


def test_sqlite_preserves_fractional_timestamps_across_restart(database):
    with database.transaction() as db:
        insert_participant(db)
    location = database.location
    database.close()
    reopened = Database(location)
    try:
        with reopened.transaction(read_only=True) as db:
            assert db.one('SELECT created FROM pl3_participants')['created'] == 1790000000.125
        with sqlite3.connect(location) as connection:
            assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    finally:
        reopened.close()


def test_sqlite_rolls_back_and_rejects_read_only_mutations(database):
    with pytest.raises(ValueError, match='abort'):
        with database.transaction(scope='instance-a') as db:
            insert_participant(db)
            raise ValueError('abort')
    with database.transaction(read_only=True) as db:
        assert db.one('SELECT COUNT(*) AS n FROM pl3_participants')['n'] == 0
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            insert_participant(db)


def test_sqlite_reader_keeps_snapshot_without_blocking_writer(database):
    with database.transaction() as db:
        insert_participant(db, 'before')
    with ThreadPoolExecutor(max_workers=1) as executor:
        with database.transaction(read_only=True) as db:
            assert db.one('SELECT COUNT(*) AS n FROM pl3_participants')['n'] == 1
            def write():
                with database.transaction(scope='independent-instance') as writer:
                    insert_participant(writer, 'during')
            executor.submit(write).result(timeout=3)
            assert db.one('SELECT COUNT(*) AS n FROM pl3_participants')['n'] == 1
    with database.transaction(read_only=True) as db:
        assert db.one('SELECT COUNT(*) AS n FROM pl3_participants')['n'] == 2


def test_sqlite_parallel_database_instances_do_not_lose_updates(database):
    second = Database(database.location)
    with database.transaction() as db:
        db.execute('CREATE TABLE concurrent_counter (value INTEGER NOT NULL)')
        db.execute('INSERT INTO concurrent_counter VALUES(0)')
    def increment(index):
        owner = database if index % 2 else second
        with owner.transaction(scope='same-instance') as db:
            value = db.one('SELECT value FROM concurrent_counter')['value']
            db.execute('UPDATE concurrent_counter SET value=?', (value + 1,))
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(increment, range(48)))
        with database.transaction(read_only=True) as db:
            assert db.one('SELECT value FROM concurrent_counter')['value'] == 48
    finally:
        second.close()


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = list(rows)
    def fetchone(self):
        return self.rows[0] if self.rows else None
    def fetchall(self):
        return self.rows


class FakePostgres:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.broken = False
        self.statements = []
        self.execution_options = []
        self.fail_health = False
        self.fail_write = False
        self.fail_commit = False
        self.fail_rollback = False
        self.in_transaction = False
        self.commits = 0
        self.rollbacks = 0
    def execute(self, sql, values=None, **kwargs):
        self.execution_options.append((sql, values, kwargs))
        self.statements.append((sql, values or ()))
        if sql == 'SELECT 1':
            assert not self.in_transaction, 'pool health probe must precede BEGIN'
            if self.fail_health:
                raise ConnectionError('idle connection died')
        if sql.startswith('BEGIN'):
            self.in_transaction = True
        if sql.startswith('INSERT') and self.fail_write:
            raise ConnectionError('write result uncertain')
        if 'information_schema.columns' in sql:
            return FakeCursor(self.owner.real_columns)
        return FakeCursor()
    def commit(self):
        self.commits += 1
        if self.fail_commit:
            self.broken = True
            raise ConnectionError('commit result uncertain')
        self.in_transaction = False
    def rollback(self):
        self.rollbacks += 1
        if self.fail_rollback:
            raise ConnectionError('rollback failed')
        self.in_transaction = False
    def close(self):
        self.closed = True


@pytest.fixture
def fake_postgres(monkeypatch):
    owner = types.SimpleNamespace(connections=[], connect_options=[], real_columns=[], fail_connect=False)
    def connect(location, **kwargs):
        owner.connect_options.append(kwargs)
        if owner.fail_connect:
            raise ConnectionError('cannot connect')
        connection = FakePostgres(owner)
        owner.connections.append(connection)
        return connection
    module = types.ModuleType('psycopg')
    module.connect = connect
    rows = types.ModuleType('psycopg.rows')
    rows.dict_row = object()
    monkeypatch.setitem(sys.modules, 'psycopg', module)
    monkeypatch.setitem(sys.modules, 'psycopg.rows', rows)
    return owner


def test_postgres_adapter_reuses_connection_and_scopes_locks(fake_postgres):
    database = Database('postgresql://fake.invalid/test')
    connection = fake_postgres.connections[0]
    connection.statements.clear()
    try:
        for scope in ('instance-a', 'instance-a', 'instance-b', None):
            with database.transaction(scope=scope) as db:
                db.execute('SELECT ? AS placeholder', (7,))
        locks = [values[0] for sql, values in connection.statements if 'pg_advisory' in sql]
        assert locks == [advisory_lock_key('instance-a'), advisory_lock_key('instance-a'),
                         advisory_lock_key('instance-b'), GLOBAL_ADVISORY_LOCK]
        assert locks[0] != locks[2]
        assert len(fake_postgres.connections) == 1
        assert ('SELECT %s AS placeholder', (7,)) in connection.statements
        assert all(begin == 'BEGIN' + TRANSACTION_TIMEOUTS for begin, _ in connection.statements if begin.startswith('BEGIN'))
        options = fake_postgres.connect_options[0]
        assert options['autocommit'] is True
        assert options['connect_timeout'] == 10
        assert all(kwargs.get('prepare') is False for sql, _, kwargs in connection.execution_options if sql.startswith('BEGIN'))
        assert 'options' not in options
        assert options['tcp_user_timeout'] == 15000
        assert "statement_timeout = '15s'" in TRANSACTION_TIMEOUTS
        assert "lock_timeout = '10s'" in TRANSACTION_TIMEOUTS
    finally:
        database.close()
    assert connection.closed


def test_postgres_read_only_uses_consistent_snapshot_without_advisory_lock(fake_postgres):
    database = Database('postgres://fake.invalid/test')
    connection = fake_postgres.connections[0]
    connection.statements.clear()
    try:
        with database.transaction(scope='ignored-for-read', read_only=True) as db:
            db.execute('SELECT 2')
        assert connection.statements[0] == ('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY' + TRANSACTION_TIMEOUTS, ())
        assert not any('advisory' in sql for sql, _ in connection.statements)
    finally:
        database.close()


def test_postgres_migration_widens_only_known_versioned_columns(fake_postgres):
    fake_postgres.real_columns = [
        {'table_name': 'pl3_instances', 'column_name': 'created'},
        {'table_name': 'pl3_timings', 'column_name': 'seconds'},
        {'table_name': 'legacy_participants', 'column_name': 'created'},
        {'table_name': 'pl3_instances', 'column_name': 'unexpected_column'},
    ]
    database = Database('postgresql://fake.invalid/test')
    try:
        alterations = [sql for sql, _ in fake_postgres.connections[0].statements if sql.startswith('ALTER')]
        assert alterations == [
            'ALTER TABLE pl3_instances ALTER COLUMN created TYPE DOUBLE PRECISION',
            'ALTER TABLE pl3_timings ALTER COLUMN seconds TYPE DOUBLE PRECISION',
        ]
        create_sql = '\n'.join(sql for sql, _ in fake_postgres.connections[0].statements if 'CREATE TABLE' in sql)
        assert ' REAL' not in create_sql
        assert all(values is None for sql, values, _ in fake_postgres.connections[0].execution_options if 'information_schema.columns' in sql)
    finally:
        database.close()


def test_postgres_replaces_dead_idle_connection_before_begin(fake_postgres):
    database = Database('postgresql://fake.invalid/test', health_check_after=0)
    stale = fake_postgres.connections[0]
    stale.statements.clear()
    stale.fail_health = True
    try:
        with database.transaction(scope='instance-a') as db:
            db.execute('INSERT INTO example VALUES(?)', (1,))
        assert stale.closed
        assert stale.statements == [('SELECT 1', ())]
        assert len(fake_postgres.connections) == 2
        assert sum(sql.startswith('INSERT') for c in fake_postgres.connections for sql, _ in c.statements) == 1
    finally:
        database.close()


@pytest.mark.parametrize('failure', ['fail_write', 'fail_commit'])
def test_postgres_does_not_retry_after_application_transaction_started(fake_postgres, failure):
    database = Database('postgresql://fake.invalid/test')
    connection = fake_postgres.connections[0]
    connection.statements.clear()
    setattr(connection, failure, True)
    try:
        with pytest.raises(ConnectionError, match='uncertain'):
            with database.transaction(scope='instance-a') as db:
                db.execute('INSERT INTO example VALUES(?)', (1,))
        assert len(fake_postgres.connections) == 1
        assert sum(sql.startswith('INSERT') for sql, _ in connection.statements) == 1
        assert connection.rollbacks == 1
    finally:
        database.close()


def test_postgres_pool_is_bounded_and_released_by_short_lived_threads(fake_postgres):
    database = Database('postgresql://fake.invalid/test', pool_size=2, pool_timeout=0.05)
    entered = threading.Barrier(3)
    release = threading.Event()
    def hold():
        with database.transaction(read_only=True):
            entered.wait(timeout=3)
            assert release.wait(3)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(hold) for _ in range(2)]
            entered.wait(timeout=3)
            try:
                with pytest.raises(TimeoutError, match='pool_timeout'):
                    with database.transaction(read_only=True):
                        pytest.fail('over-allocated the bounded pool')
                assert len(fake_postgres.connections) == 2
            finally:
                release.set()
            for future in futures:
                future.result(timeout=3)
        # Each fresh executor represents another short-lived request thread.
        for _ in range(8):
            with ThreadPoolExecutor(max_workers=1) as executor:
                def one_transaction():
                    with database.transaction(read_only=True):
                        pass
                executor.submit(one_transaction).result(timeout=3)
        assert len(fake_postgres.connections) == 2
    finally:
        release.set()
        database.close()
    assert all(connection.closed for connection in fake_postgres.connections)


def test_postgres_close_reclaims_in_flight_connection_on_return(fake_postgres):
    database = Database('postgresql://fake.invalid/test')
    with database.transaction(read_only=True):
        database.close()
    assert fake_postgres.connections[0].closed
    with pytest.raises(RuntimeError, match='database_closed'):
        with database.transaction():
            pass


def test_postgres_failed_new_connection_does_not_leak_pool_capacity(fake_postgres):
    database = Database('postgresql://fake.invalid/test', pool_size=1)
    fake_postgres.connections[0].closed = True
    fake_postgres.fail_connect = True
    try:
        with pytest.raises(ConnectionError, match='cannot connect'):
            with database.transaction():
                pass
        assert database._connection_count == 0
        fake_postgres.fail_connect = False
        with database.transaction():
            pass
        assert database._connection_count == 1
    finally:
        database.close()


def test_advisory_lock_key_is_stable_signed_64_bit_and_namespaced():
    assert advisory_lock_key(None) == GLOBAL_ADVISORY_LOCK
    # Hard-coded expected digest prevents accidental Python hash replacement.
    assert advisory_lock_key('instance-a') == -4155597128468280831
    assert -(2**63) <= advisory_lock_key('用户-' * 300) < 2**63
    assert advisory_lock_key('') != GLOBAL_ADVISORY_LOCK


def test_postgres_rollback_failure_discards_connection_without_retry(fake_postgres):
    database = Database('postgresql://fake.invalid/test')
    connection = fake_postgres.connections[0]
    connection.statements.clear()
    connection.fail_write = True
    connection.fail_rollback = True
    try:
        with pytest.raises(ConnectionError, match='write result uncertain'):
            with database.transaction(scope='instance-a') as db:
                db.execute('INSERT INTO example VALUES(?)', (1,))
        assert connection.closed
        assert database._connection_count == 0
        assert len(fake_postgres.connections) == 1
        assert sum(sql.startswith('INSERT') for sql, _ in connection.statements) == 1
        with database.transaction(scope='instance-a'):
            pass
        assert len(fake_postgres.connections) == 2
    finally:
        database.close()
