"""Hermetic SQL/driver checks; these do not certify a live PostgreSQL service."""
from copy import deepcopy
import re

import pytest

from runtime_v642.store import _SCHEMA
from study_vnext.storage import (
    PostgresStorage, StorageError, MutationRule, compile_schema, qmark_sql,
    _REJECT_BODY, _REJECT_FUNCTION,
)

EXTRA = '''
CREATE TABLE IF NOT EXISTS vnext_enrollments(instance_id TEXT PRIMARY KEY REFERENCES v642_instances(id),created REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS vnext_one_domain ON v642_instances(participant_key,mode);
'''
PROTOCOL = 'a' * 64
APPEND = ('v642_demonstrations', 'v642_frames', 'v642_questions', 'v642_commands', 'vnext_enrollments')


class Cursor:
    def __init__(self, rows=()): self.rows = deepcopy(list(rows))
    def fetchone(self): return self.rows[0] if self.rows else None
    def fetchall(self): return self.rows


class Driver:
    def __init__(self):
        self.calls, self.tables, self.identities, self.functions, self.triggers = [], set(), [], [], {}
        self.closed = False
        self.failure = None
    def execute(self, sql, values=None):
        self.calls.append((sql, deepcopy(values)))
        if self.failure and self.failure[0](sql):
            failure, self.failure = self.failure[1], None
            raise failure
        if 'information_schema.tables' in sql:
            return Cursor([dict(table_name=t) for t in self.tables])
        if sql.startswith('SELECT * FROM vnext_storage_identity'):
            return Cursor(self.identities)
        if sql.startswith('INSERT INTO vnext_storage_identity'):
            self.identities = [dict(zip(('version', 'protocol_sha256', 'schema_sha256'), values))]
        if sql.startswith('CREATE TABLE IF NOT EXISTS '):
            self.tables.add(re.search(r'CREATE TABLE IF NOT EXISTS (\w+)', sql)[1])
        if sql.startswith('SELECT p.prosrc'):
            return Cursor(self.functions)
        if sql.startswith('CREATE FUNCTION '):
            self.functions = [dict(prosrc=_REJECT_BODY)]
        if sql.startswith('SELECT t.tgtype'):
            old = self.triggers.get(values[2])
            return Cursor([] if old is None else [old])
        if sql.startswith('CREATE TRIGGER '):
            name = re.search(r'CREATE TRIGGER "([^"]+)"', sql)[1]
            update = ' BEFORE UPDATE' in sql
            columns = re.search(r' BEFORE UPDATE OF (.*?) ON ', sql)
            columns = re.findall(r'"([^"]+)"', columns[1]) if columns else []
            schema = re.search(r' ON "([^"]+)"', sql)[1]
            self.triggers[name] = dict(type_bits=19 if update else 11, enabled='O', argument_count=0,
                condition=None, constraint_oid=0, function_name=_REJECT_FUNCTION, function_schema=schema, columns=columns)
        if sql.startswith('SELECT value FROM'):
            return Cursor([dict(value=values[0])])
        return Cursor()
    def close(self): self.closed = True


def storage(driver=None, **kwargs):
    driver = Driver() if driver is None else driver
    calls = []
    def connector(**kw):
        calls.append(kw)
        return driver
    result = PostgresStorage(dsn='explicit-test-DSN-never-connected', schema='study_protocol_20261029',
        schemas=[_SCHEMA, EXTRA], protocol_sha256=PROTOCOL, append_only_tables=APPEND,
        nondeletable_tables=('v642_instances', 'v642_runs'), connector=connector, **kwargs)
    return result, driver, calls


def test_compile_retains_constraints_and_portable_numeric_semantics():
    compiled = compile_schema([_SCHEMA, EXTRA], append_only_tables=APPEND,
                              nondeletable_tables=('v642_instances', 'v642_runs'))
    assert 'v642_instances' in compiled.tables and 'vnext_enrollments' in compiled.tables
    text = '\n'.join(compiled.statements)
    assert 'scenario_seed BIGINT NOT NULL' in text
    assert 'created DOUBLE PRECISION NOT NULL' in text
    assert 'UNIQUE(participant_key,domain,mode)' in text
    assert 'REFERENCES v642_instances(id)' in text
    assert not any('RAISE(ABORT' in statement for statement in compiled.statements)
    enrollment = next(r for r in compiled.mutation_rules if r.name == 'v642_immutable_binding')
    assert enrollment.columns[:3] == ('participant_key', 'token_hash', 'release_id')
    assert any(r.table == 'v642_frames' and r.operation == 'DELETE' for r in compiled.mutation_rules)
    literal = compile_schema(["CREATE TABLE IF NOT EXISTS vnext_literal(value TEXT DEFAULT 'REAL INTEGER', n integer, t real);"])
    assert "DEFAULT 'REAL INTEGER', n BIGINT, t DOUBLE PRECISION" in literal.statements[0]


def test_unknown_or_historical_schema_and_invented_immutable_tables_are_rejected():
    for sql in (
        'CREATE TABLE IF NOT EXISTS pl3_participants(id TEXT PRIMARY KEY);',
        'CREATE TABLE IF NOT EXISTS vnext_data(id TEXT); PRAGMA foreign_keys=ON;',
        'CREATE TABLE IF NOT EXISTS vnext_data(id TEXT); CREATE TRIGGER anything AFTER INSERT ON vnext_data BEGIN SELECT 1; END;',
        'DROP TABLE vnext_data;',
    ):
        with pytest.raises(StorageError): compile_schema([sql])
    with pytest.raises(StorageError, match='unknown_table'):
        compile_schema([EXTRA], append_only_tables=('vnext_absent',))
    with pytest.raises(StorageError): MutationRule('trigger', 'vnext_table', 'INSERT')
    with pytest.raises(StorageError): MutationRule('bad;DROP', 'vnext_table', 'UPDATE')


def test_qmark_conversion_preserves_strings_comments_and_percent_literals():
    sql = '''SELECT '?', "q?", ? -- comment ?
, '100%' /* ? */ WHERE note LIKE '%known%' AND value=?'''
    converted = qmark_sql(sql, 2)
    assert converted == '''SELECT '?', "q?", %s -- comment ?
, '100%%' /* ? */ WHERE note LIKE '%%known%%' AND value=%s'''
    assert qmark_sql("SELECT '100%'", 0) == "SELECT '100%'"
    with pytest.raises(StorageError): qmark_sql('SELECT ?', 0)
    with pytest.raises(StorageError): qmark_sql("SELECT 'unterminated", 0)


def test_driver_initialization_is_isolated_bounded_and_non_destructive():
    store, driver, calls = storage()
    assert calls == [dict(autocommit=True, connect_timeout=5,
        options='-c statement_timeout=30000 -c lock_timeout=5000 -c idle_in_transaction_session_timeout=60000')]
    statements = [s for s, _ in driver.calls]
    assert statements[0] == 'BEGIN ISOLATION LEVEL SERIALIZABLE'
    assert statements[1] == 'SELECT pg_advisory_xact_lock(%s)'
    assert 'CREATE SCHEMA IF NOT EXISTS "study_protocol_20261029"' in statements
    assert any('FOR EACH ROW EXECUTE FUNCTION' in s for s in statements)
    assert not any(re.search(r'\b(DROP|TRUNCATE|ALTER)\b', s) for s in statements)
    assert driver.identities[0]['protocol_sha256'] == PROTOCOL
    assert driver.identities[0]['schema_sha256'] == store.compiled.sha256
    store.close()
    assert driver.closed


def test_read_and_write_transaction_modes_and_mapping_rows():
    store, driver, _ = storage(); driver.calls.clear()
    with store.transaction() as db:
        assert db.execute('SELECT value FROM vnext_enrollments WHERE instance_id=?', ('x',)).fetchone() == dict(value='x')
    assert driver.calls[0][0] == 'BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY'
    assert driver.calls[-1][0] == 'COMMIT'
    assert not any('advisory' in s for s, _ in driver.calls)
    driver.calls.clear()
    with store.transaction(write=True) as db:
        db.execute('INSERT INTO vnext_enrollments VALUES(?,?)', ('x', 1.25))
    assert driver.calls[0][0] == 'BEGIN ISOLATION LEVEL SERIALIZABLE'
    assert driver.calls[1][0] == 'SELECT pg_advisory_xact_lock(%s)'
    assert driver.calls[2] == ('INSERT INTO vnext_enrollments VALUES(%s,%s)', ('x', 1.25))
    assert driver.calls[-1][0] == 'COMMIT'
    store.close()


def test_body_failure_rolls_back_and_does_not_retry_or_hide_business_error():
    store, driver, _ = storage(); driver.calls.clear()
    failure = ValueError('core-protocol-rejection')
    with pytest.raises(ValueError) as caught:
        with store.transaction(write=True) as db:
            db.execute('SELECT value FROM vnext_enrollments WHERE instance_id=?', ('one',))
            raise failure
    assert caught.value is failure
    assert [s for s, _ in driver.calls].count('ROLLBACK') == 1
    assert not any(s == 'COMMIT' for s, _ in driver.calls)
    assert len([s for s, _ in driver.calls if s.startswith('SELECT value')]) == 1
    store.close()


def test_uncertain_commit_closes_connection_and_never_replays():
    store, driver, _ = storage(); driver.calls.clear()
    class Disconnect(Exception): sqlstate = '08006'
    driver.failure = (lambda sql: sql == 'COMMIT', Disconnect('private DSN and participant data'))
    with pytest.raises(StorageError) as caught:
        with store.transaction(write=True) as db:
            db.execute('INSERT INTO vnext_enrollments VALUES(?,?)', ('one', 2.25))
    assert str(caught.value) == 'database_commit_outcome_unconfirmed'
    assert caught.value.sqlstate == '08006'
    assert driver.closed
    assert len([s for s, _ in driver.calls if s.startswith('INSERT INTO')]) == 1
    with pytest.raises(StorageError, match='closed_or_nested'):
        with store.transaction(write=True): pass


def test_reopen_checks_original_identity_and_does_not_replace_mutation_guards():
    first, driver, _ = storage()
    driver.calls.clear()
    second, same, _ = storage(driver)
    assert first.compiled == second.compiled
    assert not any(s.startswith('CREATE TRIGGER') or s.startswith('CREATE FUNCTION') for s, _ in driver.calls)
    driver.identities[0]['protocol_sha256'] = 'b' * 64
    with pytest.raises(StorageError, match='schema_or_protocol_differs'): storage(driver)
    assert driver.closed


def test_preexisting_unstamped_or_unregistered_schema_is_not_adopted():
    for tables in ({'pl3_participants'}, {'v642_instances'}):
        driver = Driver(); driver.tables = tables
        with pytest.raises(StorageError): storage(driver)
        assert driver.closed
        assert not any(s.startswith('INSERT INTO') for s, _ in driver.calls)


def test_tampered_existing_trigger_or_function_is_refused_instead_of_repaired():
    _, driver, _ = storage()
    name = next(iter(driver.triggers)); driver.triggers[name]['condition'] = '(false)'
    driver.calls.clear()
    with pytest.raises(StorageError, match='immutable_trigger_differs'): storage(driver)
    assert not any(s.startswith('CREATE TRIGGER') for s, _ in driver.calls)
    _, driver, _ = storage(); driver.functions = [dict(prosrc='BEGIN RETURN NEW; END;')]
    with pytest.raises(StorageError, match='trigger_function_differs'): storage(driver)


def test_bad_namespace_timeouts_and_connection_failures_are_sanitized():
    for schema in ('public', 'pg_catalog', 'information_schema', 'study;DROP TABLE x'):
        with pytest.raises(StorageError):
            PostgresStorage(dsn='unused', schema=schema, schemas=[_SCHEMA], protocol_sha256=PROTOCOL)
    with pytest.raises(StorageError, match='bounded_storage_timeouts'):
        storage(connect_timeout=0)
    def fail(**_kwargs): raise RuntimeError('postgresql://secret-password@private-host/participant-db')
    with pytest.raises(StorageError) as caught:
        PostgresStorage(dsn='explicit', schema='study_test', schemas=[_SCHEMA], protocol_sha256=PROTOCOL, connector=fail)
    assert str(caught.value) == 'postgresql_connection_failed'
