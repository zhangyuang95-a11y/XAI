"""Versioned storage, additive to legacy study tables, SQLite or PostgreSQL."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import hashlib
import queue
import sqlite3
import threading
import time
import uuid

SCHEMA = '''
CREATE TABLE IF NOT EXISTS pl3_participants (
 id TEXT PRIMARY KEY, group_code TEXT NOT NULL, recovery_hash TEXT NOT NULL,
 created DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_sessions (
 token_hash TEXT PRIMARY KEY, participant_id TEXT NOT NULL, created DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_instances (
 id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, domain TEXT NOT NULL,
 release_id TEXT NOT NULL, group_code TEXT NOT NULL, mode TEXT NOT NULL,
 stage TEXT NOT NULL, revision INTEGER NOT NULL, current_run TEXT,
 demo_index INTEGER NOT NULL, language TEXT NOT NULL, scenario_seed INTEGER NOT NULL,
 created DOUBLE PRECISION NOT NULL, completed DOUBLE PRECISION,
 UNIQUE(participant_id, domain, release_id)
);
CREATE TABLE IF NOT EXISTS pl3_enrollments (
 instance_id TEXT PRIMARY KEY, consent INTEGER NOT NULL, initial_language TEXT NOT NULL,
 assignment_source TEXT NOT NULL, scenario_version TEXT NOT NULL, created DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_runs (
 id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, task INTEGER NOT NULL,
 seed INTEGER NOT NULL, state_json TEXT NOT NULL, status TEXT NOT NULL,
 score_json TEXT NOT NULL, started DOUBLE PRECISION NOT NULL, ended DOUBLE PRECISION,
 UNIQUE(instance_id, task)
);
CREATE TABLE IF NOT EXISTS pl3_frames (
 run_id TEXT NOT NULL, turn INTEGER NOT NULL, state_json TEXT NOT NULL,
 public_json TEXT NOT NULL, decision_json TEXT NOT NULL, human_action TEXT,
 created DOUBLE PRECISION NOT NULL, PRIMARY KEY(run_id, turn)
);
CREATE TABLE IF NOT EXISTS pl3_questions (
 id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, authorized_run TEXT NOT NULL,
 target_run TEXT NOT NULL, target_turn INTEGER NOT NULL, question TEXT NOT NULL,
 language TEXT NOT NULL, result_json TEXT NOT NULL, status TEXT NOT NULL,
 requested DOUBLE PRECISION NOT NULL, finished DOUBLE PRECISION, displayed DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS pl3_questionnaires (
 instance_id TEXT PRIMARY KEY, answers_json TEXT NOT NULL,
 comprehension_json TEXT NOT NULL, submitted DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_commands (
 session_hash TEXT NOT NULL, command_id TEXT NOT NULL, request_hash TEXT NOT NULL,
 result_json TEXT NOT NULL, created DOUBLE PRECISION NOT NULL,
 PRIMARY KEY(session_hash, command_id)
);
CREATE TABLE IF NOT EXISTS pl3_timings (
 id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, task INTEGER,
 kind TEXT NOT NULL, seconds DOUBLE PRECISION NOT NULL, recorded DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_releases (
 id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL, created DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_tutorials (
 instance_id TEXT PRIMARY KEY, version TEXT NOT NULL, state_json TEXT NOT NULL,
 created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pl3_tutorial_events (
 id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, version TEXT NOT NULL,
 command TEXT NOT NULL, payload_json TEXT NOT NULL, before_json TEXT NOT NULL,
 after_json TEXT NOT NULL, recorded DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS pl3_active ON pl3_instances(participant_id, stage);
CREATE INDEX IF NOT EXISTS pl3_question_run ON pl3_questions(authorized_run, requested);
'''

class Connection:
    def __init__(self, connection, postgres):
        self.connection, self.postgres = connection, postgres
    def execute(self, sql, values=()):
        if self.postgres:
            # None bypasses parameter parsing for literal SQL (e.g. LIKE '%').
            return self.connection.execute(sql.replace('?', '%s'), values if values else None)
        return self.connection.execute(sql, values)
    def one(self, sql, values=()):
        row = self.execute(sql, values).fetchone()
        return dict(row) if row is not None else None
    def all(self, sql, values=()):
        return [dict(row) for row in self.execute(sql, values).fetchall()]
    def iterate(self, sql, values=(), batch_size=64):
        """Fetch a bounded batch, including on PostgreSQL/transaction poolers.

        A normal psycopg cursor buffers the entire result on execute(), even
        when fetchmany() follows. A named, non-holdable cursor keeps that result
        on the database server until this transaction consumes or closes it.
        """
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
            raise ValueError('invalid_export_batch_size')
        cursor = None
        try:
            if self.postgres:
                cursor = self.connection.cursor(name='pl3_export_' + uuid.uuid4().hex,
                                                withhold=False)
                cursor.execute(sql.replace('?', '%s'), values if values else None)
            else:
                cursor = self.execute(sql, values)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield dict(row)
        finally:
            if cursor is not None:
                cursor.close()

# Prefix-specific migration: never inspect or change legacy study tables.
FLOAT_COLUMNS = {
    'pl3_participants': ('created',),
    'pl3_sessions': ('created',),
    'pl3_instances': ('created', 'completed'),
    'pl3_enrollments': ('created',),
    'pl3_runs': ('started', 'ended'),
    'pl3_frames': ('created',),
    'pl3_questions': ('requested', 'finished', 'displayed'),
    'pl3_questionnaires': ('submitted',),
    'pl3_commands': ('created',),
    'pl3_timings': ('seconds', 'recorded'),
    'pl3_releases': ('created',),
    'pl3_tutorials': ('created','updated'),
    'pl3_tutorial_events': ('recorded',),
}
GLOBAL_ADVISORY_LOCK = 76320920
TRANSACTION_TIMEOUTS = (
    "; SET LOCAL statement_timeout = '15s'"
    "; SET LOCAL lock_timeout = '10s'"
    "; SET LOCAL idle_in_transaction_session_timeout = '30s'"
)


def advisory_lock_key(scope):
    """A process/restart-independent signed PostgreSQL bigint lock identifier."""
    if scope is None:
        return GLOBAL_ADVISORY_LOCK
    key = int.from_bytes(hashlib.sha256(
        ('policylens-study-v3:instance:' + str(scope)).encode('utf-8')
    ).digest()[:8], 'big', signed=True)
    return key if key != GLOBAL_ADVISORY_LOCK else -GLOBAL_ADVISORY_LOCK


class Database:
    """Additive schema and explicit transactions with bounded PostgreSQL reuse.

    A scope serializes writes for that study instance across processes. Default
    writes serialize with other default writes (e.g. enrollment); they do not
    take an exclusive barrier against separately scoped instance writes. Callers
    must use one consistent scope for every mutation of a given instance.

    A transaction is never replayed automatically. Only an idle connection may
    be replaced, before BEGIN. Retrying an application command after an ambiguous
    commit must use the same command id so the store can resolve its outcome.
    """
    def __init__(self, location, *, pool_size=4, pool_timeout=10, health_check_after=15):
        if not 1 <= pool_size <= 8:
            raise ValueError('database_pool_size_must_be_between_1_and_8')
        self.location = location
        self.postgres = location.startswith(('postgres://', 'postgresql://'))
        self.lock = threading.RLock()
        self._pool = queue.LifoQueue(maxsize=pool_size)
        self._pool_condition = threading.Condition()
        self._pool_size = pool_size
        self._pool_timeout = pool_timeout
        self._health_check_after = health_check_after
        self._connection_count = 0
        self._closed = False
        if not self.postgres:
            Path(location).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(location, timeout=20) as connection:
                connection.execute('PRAGMA journal_mode=WAL')
        try:
            with self.transaction() as db:
                for sql in SCHEMA.split(';'):
                    if sql.strip():
                        db.execute(sql)
                if self.postgres:
                    self._upgrade_timestamp_precision(db)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _upgrade_timestamp_precision(db):
        # PostgreSQL REAL is float32: contemporary Unix times lose many seconds.
        # Widening preserves existing values, but cannot recover precision that
        # an earlier float32 write already lost.
        columns = db.all("""SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema = current_schema() AND data_type = 'real'
            AND table_name LIKE 'pl3_%'""")
        for column in columns:
            table, name = column['table_name'], column['column_name']
            if name in FLOAT_COLUMNS.get(table, ()):
                db.execute(f'ALTER TABLE {table} ALTER COLUMN {name} TYPE DOUBLE PRECISION')

    def _connect_postgres(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.location, row_factory=dict_row, autocommit=True,
            connect_timeout=10, keepalives=1, keepalives_idle=30,
            keepalives_interval=10, keepalives_count=3, tcp_user_timeout=15000)

    @staticmethod
    def _unusable(connection):
        return bool(connection.closed or getattr(connection, 'broken', False))

    def _discard_postgres(self, connection):
        try:
            if connection is not None:
                connection.close()
        finally:
            with self._pool_condition:
                self._connection_count -= 1
                self._pool_condition.notify()

    def _take_postgres(self):
        deadline = time.monotonic() + self._pool_timeout
        # At most one replacement of an already pooled, stale connection. A new
        # connection failure propagates immediately; no transaction has begun.
        for attempt in range(2):
            with self._pool_condition:
                while True:
                    if self._closed:
                        raise RuntimeError('database_closed')
                    try:
                        connection, returned_at = self._pool.get_nowait()
                        break
                    except queue.Empty:
                        if self._connection_count < self._pool_size:
                            self._connection_count += 1
                            connection = None
                            returned_at = None
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('database_connection_pool_timeout')
                        self._pool_condition.wait(remaining)
            if connection is None:
                try:
                    return self._connect_postgres()
                except BaseException:
                    self._discard_postgres(None)
                    raise
            try:
                if self._unusable(connection):
                    raise ConnectionError('database_idle_connection_closed')
                if time.monotonic() - returned_at >= self._health_check_after:
                    # Autocommit keeps this probe outside the application tx.
                    connection.execute('SELECT 1')
                return connection
            except Exception:
                self._discard_postgres(connection)
                if attempt:
                    raise
        raise RuntimeError('database_connection_unavailable')

    def _return_postgres(self, connection):
        with self._pool_condition:
            discard = self._closed or self._unusable(connection)
            if not discard:
                self._pool.put_nowait((connection, time.monotonic()))
                self._pool_condition.notify()
        if discard:
            self._discard_postgres(connection)

    def close(self):
        """Close idle connections now and in-flight connections when returned."""
        with self._pool_condition:
            self._closed = True
            while True:
                try:
                    connection, _ = self._pool.get_nowait()
                except queue.Empty:
                    break
                try:
                    connection.close()
                finally:
                    self._connection_count -= 1
            self._pool_condition.notify_all()

    @contextmanager
    def transaction(self, scope=None, read_only=False):
        if self.postgres:
            connection = self._take_postgres()
            reusable = True
            try:
                # Session SET/startup options are unsafe with transaction poolers:
                # https://www.pgbouncer.org/features.html . SET LOCAL follows
                # BEGIN in one controlled, parameter-free protocol round trip.
                # Disable automatic preparation for this multi-statement query.
                if read_only:
                    connection.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY' + TRANSACTION_TIMEOUTS, prepare=False)
                else:
                    # Do not acquire a repeatable-read snapshot before waiting
                    # for the scope lock: the preceding writer may still commit.
                    connection.execute('BEGIN' + TRANSACTION_TIMEOUTS, prepare=False)
                    connection.execute('SELECT pg_advisory_xact_lock(%s)',
                                       (advisory_lock_key(scope),))
                yield Connection(connection, True)
                connection.commit()
            except BaseException:
                try:
                    connection.rollback()
                except Exception:
                    reusable = False
                raise
            finally:
                if reusable:
                    self._return_postgres(connection)
                else:
                    self._discard_postgres(connection)
        else:
            # WAL readers can retain a consistent snapshot while a writer works.
            # Python serialization remains for local writes; BEGIN IMMEDIATE
            # also serializes other Database instances/processes using this file.
            with self.lock if not read_only else nullcontext():
                if self._closed:
                    raise RuntimeError('database_closed')
                connection = sqlite3.connect(self.location, timeout=20)
                connection.row_factory = sqlite3.Row
                try:
                    connection.execute('PRAGMA foreign_keys=ON')
                    if read_only:
                        connection.execute('PRAGMA query_only=ON')
                    connection.execute('BEGIN' if read_only else 'BEGIN IMMEDIATE')
                    yield Connection(connection, False)
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    connection.close()
