"""Optional isolated PostgreSQL persistence for the study's existing SQL API.

No DSN is discovered from the environment and no connection is made on import.
A real production restore/permissions/concurrency receipt is still mandatory.
Fake-driver tests verify translation and transaction semantics, not deployment.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import threading

VERSION = 'study-vnext-postgresql-storage.v1'
_IDENTIFIER = re.compile(r'^[a-z_][a-z0-9_]{0,62}$')
_SQLITE_TRIGGER = re.compile(
    r"CREATE\s+TRIGGER\s+IF\s+NOT\s+EXISTS\s+(?P<name>[a-z_][a-z0-9_]*)\s+"
    r"BEFORE\s+(?P<operation>UPDATE(?:\s+OF\s+[a-z_][a-z0-9_]*(?:\s*,\s*[a-z_][a-z0-9_]*)*)?|DELETE)\s+"
    r"ON\s+(?P<table>[a-z_][a-z0-9_]*)\s+BEGIN\s+SELECT\s+RAISE\s*\(\s*ABORT\s*,\s*'[^']*'\s*\)\s*;\s*END\s*;",
    re.IGNORECASE)
_REJECT_FUNCTION = 'vnext_reject_evidence_mutation'
_REJECT_BODY = "BEGIN RAISE EXCEPTION USING ERRCODE='23514', MESSAGE='immutable study evidence'; END;"


class StorageError(RuntimeError):
    """A sanitized error code, never a DSN or participant SQL argument."""
    def __init__(self, code, sqlstate=None):
        self.code, self.sqlstate = code, sqlstate
        super().__init__(code)


def _check(ok, code):
    if not ok:
        raise StorageError(code)


def _identifier(value):
    _check(type(value) is str and bool(_IDENTIFIER.fullmatch(value)), 'invalid_storage_identifier')
    return '"' + value + '"'


def _split_sql(text):
    """Split trusted schema SQL while preserving semicolons inside SQL strings."""
    result, current, index, quote, comment = [], [], 0, None, None
    while index < len(text):
        ch, nxt = text[index], text[index:index+2]
        if comment == 'line':
            current.append(ch)
            if ch == '\n': comment = None
        elif comment == 'block':
            current.append(ch)
            if nxt == '*/':
                current.append('/'); index += 1; comment = None
        elif quote:
            current.append(ch)
            if ch == quote:
                if index + 1 < len(text) and text[index+1] == quote:
                    current.append(quote); index += 1
                else: quote = None
        elif nxt in ('--', '/*'):
            current.append(nxt); index += 1; comment = 'line' if nxt == '--' else 'block'
        elif ch in ("'", '"'):
            current.append(ch); quote = ch
        elif ch == ';':
            if ''.join(current).strip(): result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
        index += 1
    _check(not quote and comment != 'block', 'unterminated_schema_SQL')
    if ''.join(current).strip(): result.append(''.join(current).strip())
    return result


def qmark_sql(sql, parameter_count):
    """Translate SQLite-style placeholders, excluding quoted/comment question marks.

    Percent characters in source SQL are escaped for psycopg only when bound
    parameters are present. Native PostgreSQL percent placeholders are not this
    adapter's input format.
    """
    _check(type(sql) is str and type(parameter_count) is int and parameter_count >= 0,
           'invalid_parameterized_SQL')
    output, index, found, quote, comment = [], 0, 0, None, None
    while index < len(sql):
        ch, nxt = sql[index], sql[index:index+2]
        escaped = '%%' if ch == '%' and parameter_count else ch
        if comment == 'line':
            output.append(escaped)
            if ch == '\n': comment = None
        elif comment == 'block':
            output.append(escaped)
            if nxt == '*/': output.append('/'); index += 1; comment = None
        elif quote:
            output.append(escaped)
            if ch == quote:
                if index + 1 < len(sql) and sql[index+1] == quote:
                    output.append(quote); index += 1
                else: quote = None
        elif nxt in ('--', '/*'):
            output.append(nxt); index += 1; comment = 'line' if nxt == '--' else 'block'
        elif ch in ("'", '"'):
            output.append(ch); quote = ch
        elif ch == '?':
            output.append('%s'); found += 1
        else:
            output.append(escaped)
        index += 1
    _check(not quote and comment != 'block' and found == parameter_count, 'SQL_parameter_count_mismatch')
    return ''.join(output)


@dataclass(frozen=True, slots=True)
class MutationRule:
    name: str
    table: str
    operation: str
    columns: tuple[str, ...] = ()

    def __post_init__(self):
        _identifier(self.name); _identifier(self.table)
        _check(self.operation in ('UPDATE', 'DELETE'), 'unsupported_immutable_operation')
        _check(type(self.columns) is tuple and len(set(self.columns)) == len(self.columns)
               and all(_IDENTIFIER.fullmatch(c) for c in self.columns)
               and (not self.columns or self.operation == 'UPDATE'), 'invalid_immutable_columns')

    def statement(self, schema):
        columns = ' OF ' + ', '.join(_identifier(c) for c in self.columns) if self.columns else ''
        return (f'CREATE TRIGGER {_identifier(self.name)} BEFORE {self.operation}{columns} '
                f'ON {_identifier(schema)}.{_identifier(self.table)} FOR EACH ROW '
                f'EXECUTE FUNCTION {_identifier(schema)}.{_identifier(_REJECT_FUNCTION)}()')


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    statements: tuple[str, ...]
    mutation_rules: tuple[MutationRule, ...]
    tables: tuple[str, ...]
    sha256: str


def _postgres_types(statement):
    token = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*[\s\S]*?\*/|\b(?:INTEGER|REAL)\b", re.IGNORECASE)
    return token.sub(lambda m: {'INTEGER':'BIGINT', 'REAL':'DOUBLE PRECISION'}.get(m[0].upper(), m[0]), statement)


def compile_schema(schemas, *, append_only_tables=(), immutable_columns=None, nondeletable_tables=()):
    """Translate the closed SQLite core DDL; unknown triggers fail, never vanish."""
    _check(type(schemas) in (tuple, list) and schemas and all(type(s) is str for s in schemas),
           'explicit_schema_sources_required')
    _check(type(append_only_tables) in (tuple, list) and type(nondeletable_tables) in (tuple, list),
           'explicit_immutable_table_inventory_required')
    immutable_columns = {} if immutable_columns is None else immutable_columns
    _check(type(immutable_columns) is dict, 'invalid_immutable_column_inventory')
    rules, statements, tables = [], [], set()
    for source in schemas:
        _check(not re.search(r'\bpl3_[a-z_]', source, flags=re.I), 'historical_schema_must_remain_read_only')
        def extract(match):
            operation = re.split(r'\s+OF\s+', match['operation'], maxsplit=1, flags=re.IGNORECASE)
            columns = tuple(c.strip() for c in operation[1].split(',')) if len(operation) == 2 else ()
            rules.append(MutationRule(match['name'], match['table'], operation[0].upper(), columns))
            return ''
        remaining = _SQLITE_TRIGGER.sub(extract, source)
        _check(not re.search(r'\bTRIGGER\b|\bPRAGMA\b|\bAUTOINCREMENT\b', remaining, flags=re.I),
               'untranslated_SQLite_schema_construct')
        for statement in _split_sql(remaining):
            _check(bool(re.match(r'^CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+IF\s+NOT\s+EXISTS\b',
                                 statement, flags=re.I)), 'unsupported_schema_statement')
            table_match = re.match(r'CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([a-z_][a-z0-9_]*)\b', statement, flags=re.I)
            if table_match:
                table = table_match[1]
                _check(table.startswith(('v642_', 'vnext_')), 'unregistered_study_table_namespace')
                tables.add(table)
            # All core INTEGER values use SQLite's signed 64-bit semantics,
            # including seeds. REAL timestamps require double precision in PG.
            statements.append(_postgres_types(statement))
    for table in append_only_tables:
        _identifier(table)
        for operation in ('UPDATE', 'DELETE'):
            rules.append(MutationRule(f'{table}_{operation.lower()}_immutable', table, operation))
    for table in nondeletable_tables:
        _identifier(table); rules.append(MutationRule(f'{table}_delete_immutable', table, 'DELETE'))
    for table, columns in immutable_columns.items():
        _identifier(table)
        _check(type(columns) in (tuple, list), 'invalid_immutable_column_inventory')
        rules.append(MutationRule(f'{table}_origin_immutable', table, 'UPDATE', tuple(columns)))
    _check(all(rule.table in tables for rule in rules), 'immutability_rule_references_unknown_table')
    _check(len({r.name for r in rules}) == len(rules), 'duplicate_immutability_rule')
    identity = json.dumps(dict(statements=statements, rules=[dict(name=r.name, table=r.table,
        operation=r.operation, columns=r.columns) for r in rules]), sort_keys=True, separators=(',', ':'))
    return CompiledSchema(tuple(statements), tuple(rules), tuple(sorted(tables)), sha256(identity.encode()).hexdigest())


class QmarkConnection:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, values=()):
        _check(type(values) in (tuple, list), 'SQL_parameters_must_be_positional')
        converted = qmark_sql(sql, len(values))
        try:
            return self.raw.execute(converted, values if values else None)
        except Exception as exc:
            # psycopg errors can embed SQL values, user input or connection data.
            raise StorageError('database_statement_failed', getattr(exc, 'sqlstate', None)) from None


class PostgresStorage:
    """One explicit isolated schema; no live-data/credential discovery or retries.

    ``connector`` is only for hermetic driver tests. Normal construction uses
    installed psycopg with mapping rows. A connection/commit failure is surfaced;
    uncertain transactions are never replayed automatically.
    """
    def __init__(self, *, dsn, schema, schemas, protocol_sha256, append_only_tables=(),
                 immutable_columns=None, nondeletable_tables=(), connect_timeout=5,
                 statement_timeout_ms=30000, lock_timeout_ms=5000,
                 idle_transaction_timeout_ms=60000, connector=None):
        _identifier(schema)
        _check(schema not in ('public', 'information_schema') and not schema.startswith('pg_'),
               'isolated_study_schema_required')
        _check(type(dsn) is str and bool(dsn.strip()), 'explicit_postgresql_DSN_required')
        _check(type(protocol_sha256) is str and bool(re.fullmatch('[0-9a-f]{64}', protocol_sha256)),
               'explicit_protocol_hash_required')
        _check(type(connect_timeout) is int and 1 <= connect_timeout <= 30
               and all(type(v) is int and 1 <= v <= 120000 for v in
                       (statement_timeout_ms, lock_timeout_ms, idle_transaction_timeout_ms)), 'bounded_storage_timeouts_required')
        compiled = compile_schema(schemas, append_only_tables=append_only_tables,
            immutable_columns=immutable_columns, nondeletable_tables=nondeletable_tables)
        self.schema, self.compiled, self.protocol_sha256 = schema, compiled, protocol_sha256
        self.lock, self._active, self._failed = threading.RLock(), False, False
        self._lock_id = int.from_bytes(sha256(('study-vnext:'+schema).encode()).digest()[:8], 'big', signed=True)
        if connector is None:
            import psycopg
            from psycopg.rows import dict_row
            connector = lambda **kw: psycopg.connect(dsn, row_factory=dict_row, **kw)
        options = (f'-c statement_timeout={statement_timeout_ms} -c lock_timeout={lock_timeout_ms} '
                   f'-c idle_in_transaction_session_timeout={idle_transaction_timeout_ms}')
        try:
            self._raw = connector(autocommit=True, connect_timeout=connect_timeout, options=options)
        except Exception:
            raise StorageError('postgresql_connection_failed') from None
        self.connection = QmarkConnection(self._raw)
        try:
            self._initialize()
        except BaseException:
            self._raw.close(); self._failed = True
            raise

    def _initialize(self):
        # Advisory lock also serializes schema initialization across processes.
        self._raw.execute('BEGIN ISOLATION LEVEL SERIALIZABLE')
        try:
            self._raw.execute('SELECT pg_advisory_xact_lock(%s)', (self._lock_id,))
            self._raw.execute(f'CREATE SCHEMA IF NOT EXISTS {_identifier(self.schema)}')
            self._raw.execute(f'SET LOCAL search_path TO {_identifier(self.schema)}, pg_catalog')
            names = {r['table_name'] for r in self._raw.execute(
                'SELECT table_name FROM information_schema.tables WHERE table_schema=%s', (self.schema,)).fetchall()}
            expected = set(self.compiled.tables) | {'vnext_storage_identity'}
            _check(names <= expected, 'isolated_schema_contains_unregistered_tables')
            _check(not names or 'vnext_storage_identity' in names, 'existing_schema_has_no_storage_identity')
            self._raw.execute('CREATE TABLE IF NOT EXISTS vnext_storage_identity('
                              'version TEXT PRIMARY KEY,protocol_sha256 TEXT NOT NULL,schema_sha256 TEXT NOT NULL)')
            identity = self._raw.execute('SELECT * FROM vnext_storage_identity').fetchall()
            expected_identity = dict(version=VERSION, protocol_sha256=self.protocol_sha256, schema_sha256=self.compiled.sha256)
            if identity:
                _check(identity == [expected_identity], 'existing_storage_schema_or_protocol_differs')
            else:
                _check(not names, 'existing_study_schema_has_missing_identity')
                self._raw.execute('INSERT INTO vnext_storage_identity VALUES(%s,%s,%s)', tuple(expected_identity.values()))
            for statement in self.compiled.statements:
                self._raw.execute(statement)
            function = self._raw.execute('SELECT p.prosrc FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace '
                'WHERE n.nspname=%s AND p.proname=%s AND p.pronargs=0', (self.schema, _REJECT_FUNCTION)).fetchall()
            if function:
                _check(function == [dict(prosrc=_REJECT_BODY)], 'immutable_trigger_function_differs')
            else:
                self._raw.execute(f'CREATE FUNCTION {_identifier(self.schema)}.{_identifier(_REJECT_FUNCTION)}() '
                    f'RETURNS trigger LANGUAGE plpgsql AS $body${_REJECT_BODY}$body$')
            rules = self.compiled.mutation_rules + tuple(MutationRule(f'vnext_storage_identity_{op.lower()}',
                'vnext_storage_identity', op) for op in ('UPDATE', 'DELETE'))
            for rule in rules:
                old = self._raw.execute('SELECT t.tgtype AS type_bits,t.tgenabled AS enabled,t.tgnargs AS argument_count,'
                    't.tgqual::text AS condition,t.tgconstraint AS constraint_oid,p.proname AS function_name,'
                    'fn.nspname AS function_schema,ARRAY(SELECT a.attname FROM '
                    'unnest(t.tgattr::smallint[]) WITH ORDINALITY u(attnum,ord) '
                    'JOIN pg_attribute a ON a.attrelid=t.tgrelid AND a.attnum=u.attnum ORDER BY u.ord) AS columns '
                    'FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace '
                    'JOIN pg_proc p ON p.oid=t.tgfoid JOIN pg_namespace fn ON fn.oid=p.pronamespace '
                    'WHERE n.nspname=%s AND c.relname=%s AND t.tgname=%s AND NOT t.tgisinternal',
                    (self.schema, rule.table, rule.name)).fetchall()
                statement = rule.statement(self.schema)
                if old:
                    expected_rule = dict(type_bits=19 if rule.operation=='UPDATE' else 11, enabled='O',
                        argument_count=0, condition=None, constraint_oid=0, function_name=_REJECT_FUNCTION,
                        function_schema=self.schema, columns=list(rule.columns))
                    _check(old == [expected_rule], 'existing_immutable_trigger_differs')
                else:
                    self._raw.execute(statement)
            self._raw.execute('COMMIT')
            self._raw.execute(f'SET search_path TO {_identifier(self.schema)}, pg_catalog')
        except BaseException as original:
            try: self._raw.execute('ROLLBACK')
            except Exception: pass
            if isinstance(original, StorageError): raise
            raise StorageError('postgresql_schema_initialization_failed', getattr(original, 'sqlstate', None)) from None

    @contextmanager
    def transaction(self, *, write=False):
        _check(type(write) is bool, 'transaction_write_flag_required')
        with self.lock:
            _check(not self._failed and not self._active, 'storage_closed_or_nested_transaction')
            self._active = True
            try:
                try:
                    self._raw.execute('BEGIN ISOLATION LEVEL SERIALIZABLE' + ('' if write else ' READ ONLY'))
                    if write: self._raw.execute('SELECT pg_advisory_xact_lock(%s)', (self._lock_id,))
                except Exception as exc:
                    raise StorageError('database_transaction_start_failed', getattr(exc, 'sqlstate', None)) from None
                yield self.connection
            except BaseException:
                try: self._raw.execute('ROLLBACK')
                except Exception:
                    self._failed = True; self._raw.close()
                raise
            else:
                try: self._raw.execute('COMMIT')
                except Exception as exc:
                    # The caller cannot assume a failed acknowledgement means
                    # commit did not happen. Do not replay; force explicit recovery.
                    self._failed = True; self._raw.close()
                    raise StorageError('database_commit_outcome_unconfirmed', getattr(exc, 'sqlstate', None)) from None
            finally:
                self._active = False

    def close(self):
        with self.lock:
            self._raw.close(); self._failed = True
