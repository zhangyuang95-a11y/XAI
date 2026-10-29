#!/usr/bin/env python3
"""Real, isolated local PostgreSQL drill using only synthetic toy study records.

Creates and stops its own temporary cluster over a private Unix socket; it never
connects to an existing cluster or discovers credentials. The subprocess has a
clean environment. The resulting receipt is software evidence only, not a
production persistence/restore receipt, model result or pilot authorization.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ('initdb', 'postgres', 'pg_ctl', 'pg_dump', 'pg_restore')


def prerequisites():
    """Check binaries/driver availability without opening a database."""
    found = {name: shutil.which(name) or str(Path('/opt/homebrew/bin') / name) for name in TOOLS}
    missing = [name for name, path in found.items() if not Path(path).is_file()]
    if importlib.util.find_spec('psycopg') is None:
        missing.append('psycopg')
    return dict(available=not missing, missing=missing, binaries=found)


def _run(command, *, timeout=30, env=None):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=timeout, env=env)
    if result.returncode:
        # This environment is synthetic; still omit SQL/paths/connection details
        # from raised errors and the machine-readable public receipt.
        raise RuntimeError('local_drill_subprocess_failed:' + Path(command[0]).name)
    return result.stdout.strip()


def _load_fixture():
    path = ROOT / 'tests/test_runtime_v642_store.py'
    spec = importlib.util.spec_from_file_location('local_postgres_synthetic_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_difference(left, right, path='exports'):
    if type(left) is not type(right):
        return path + ':type'
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ':keys'
        for key in left:
            difference = _first_difference(left[key], right[key], path + '.' + str(key))
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ':length'
        for index, (before, after) in enumerate(zip(left, right)):
            difference = _first_difference(before, after, path + '[' + str(index) + ']')
            if difference:
                return difference
    elif left != right:
        return path + ':value'
    return None


@contextmanager
def _cluster(binaries, base):
    """Only owned subprocesses, a fresh data dir, no TCP listener or password."""
    import psycopg
    data, sock = base / 'data', base / 'socket'
    sock.mkdir(mode=0o700)
    passfile = base / 'empty.pgpass'
    passfile.touch(mode=0o600)
    env = dict(PATH=os.defpath, LC_ALL='C', LANG='C', PGPASSFILE=str(passfile),
               PGSERVICEFILE=str(base / 'no-service-file'))
    _run([binaries['initdb'], '-D', str(data), '--username=study_drill',
          '--auth-local=trust', '--auth-host=reject', '--encoding=UTF8', '--locale=C',
          '--no-instructions'], env=env)
    log = (base / 'postgres.log').open('w')
    process = subprocess.Popen([binaries['postgres'], '-D', str(data), '-k', str(sock), '-p', '55432',
                                '-c', 'listen_addresses=', '-c', 'fsync=on',
                                '-c', 'max_connections=20'], stdout=log, stderr=subprocess.STDOUT, env=env)
    dsn = f'host={sock} port=55432 user=study_drill dbname=postgres sslmode=disable passfile={passfile}'
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise RuntimeError('owned_postgres_exited_before_ready')
            try:
                with psycopg.connect(dsn, connect_timeout=1):
                    break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('owned_postgres_start_timeout') from None
                time.sleep(.05)
        yield dsn, env
    finally:
        try:
            if process.poll() is None:
                _run([binaries['pg_ctl'], '-D', str(data), '-m', 'fast', '-w', 'stop'], env=env)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            log.close()


def _prepare(store, token, instance):
    codes = dict(warehouse=['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'],
                 pong=['KeyA', 'KeyD', 'Space'],
                 kitchen=['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'KeyE'])
    domain = store.view(token, instance)['instance']['domain']
    store.complete_preparation(token, instance, kind='controls', observed_codes=codes[domain])
    store.demonstration(token, instance)
    saved = store.view(token, instance)['study']['preparation']
    store.complete_preparation(token, instance, kind='demonstration',
        bundle_sha256=saved['demonstration_sha256'], final_frame_count=saved['demonstration_frame_count'])
    store.acknowledge_demonstration(token, instance)


def _finish_questionnaires(store, token, instance):
    from study_vnext.store import RATINGS
    if hasattr(store, 'understanding'):
        bank = store.understanding(token, instance)
        # First options are arbitrary synthetic responses, not an efficacy goal.
        store.submit_understanding(token, instance,
            answers={q['id']: q['options'][0]['id'] for q in bank['questions']},
            version=bank['version'], sha256=bank['sha256'])
    store.submit_survey(token, instance, answers={key: 4 for key in RATINGS}, feedback='synthetic local PG drill')
    assert store.view(token, instance)['study']['phase'] == 'finished'


def _duplicate_action(first, second, token, instance):
    from study_vnext.storage import StorageError
    barrier = threading.Barrier(2)
    def call(store):
        barrier.wait(timeout=5)
        try:
            return dict(result=store.action(token, instance, command_id='duplicate-command',
                        expected_turn=0, human_action='wait'))
        except StorageError as exc:
            if exc.sqlstate != '40001':
                raise
            return dict(serialization_conflict=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, store) for store in (first, second)]
        results = [future.result(timeout=15) for future in futures]
    successful = [item['result'] for item in results if 'result' in item]
    assert successful, 'neither concurrent action committed'
    assert all(item == successful[0] for item in successful)
    # Explicit caller recovery after both outcomes are known. The backend never
    # retries transactions; the original command identity is retained here.
    recovered = second.action(token, instance, command_id='duplicate-command', expected_turn=0, human_action='wait')
    assert recovered == successful[0]
    with first._transaction() as db:
        assert db.execute('SELECT COUNT(*) AS n FROM v642_commands WHERE instance_id=?',
                          (instance,)).fetchone()['n'] == 1
    assert first.view(token, instance)['current']['turn'] == 1
    return dict(committed_commands=1, serialization_conflicts=sum('serialization_conflict' in x for x in results),
                explicit_same_command_recovery=True, automatic_backend_retries=0)


def _read_during_provider(first, second, token, instance):
    from study_vnext.qa import EvidenceQuestionComposer
    entered, release = threading.Event(), threading.Event()
    class BlockingSyntheticProvider:
        configured = True
        consented = True
        def select(self, _):
            entered.set()
            if not release.wait(10):
                raise RuntimeError('synthetic_provider_wait_timeout')
            return dict(intent='reason', fact_ids=['recorded_action'])
    first.question_composer = EvidenceQuestionComposer(BlockingSyntheticProvider())
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(first.ask, token, instance, question='Why did the recorded action happen?',
                             target_task=1, decision_turn=0)
        try:
            assert entered.wait(5), 'synthetic provider was not reached'
            started = time.monotonic()
            view = pool.submit(second.view, token, instance).result(timeout=3)
            elapsed = time.monotonic() - started
            assert view['instance']['stage'] == 'task_2' and view['current']['state']['task'] == 2
        finally:
            release.set()
        result = future.result(timeout=10)
    assert 'language_rendering' in result
    first.question_composer = None
    return dict(read_completed_while_provider_blocked=True, read_seconds=elapsed,
                actual_external_provider_calls=0)


def _guards(store, token, instance):
    from study_vnext.storage import StorageError
    snapshot = store.export_instance(token, instance, admin=True)
    statements = [
        ('binding_update', 'UPDATE v642_instances SET binding_json=binding_json WHERE id=?'),
        ('frame_update', 'UPDATE v642_frames SET state_json=state_json WHERE run_id IN '
                         '(SELECT id FROM v642_runs WHERE instance_id=?)'),
        ('frame_delete', 'DELETE FROM v642_frames WHERE run_id IN '
                         '(SELECT id FROM v642_runs WHERE instance_id=?)'),
        ('consent_update', 'UPDATE vnext_enrollments SET language=language WHERE instance_id=?'),
        ('consent_delete', 'DELETE FROM vnext_enrollments WHERE instance_id=?'),
    ]
    for name, sql in statements:
        try:
            with store._transaction(write=True) as db:
                db.execute(sql, (instance,))
        except StorageError as exc:
            assert exc.sqlstate == '23514', (name, exc.sqlstate)
        else:
            raise AssertionError('immutable guard did not reject:' + name)
    assert store.export_instance(token, instance, admin=True) == snapshot
    return dict(guarded_operations=[name for name, _ in statements], post_rollback_export_unchanged=True)


def _execute(output):
    sys.path.insert(0, str(ROOT))
    import psycopg
    from runtime_v642 import RuntimeRegistry
    from study_vnext.store import StudyStore, CONSENT_VERSION
    from study_vnext.protocol import protocol_sha256
    preflight = prerequisites()
    if not preflight['available']:
        raise RuntimeError('local_postgres_prerequisites_unavailable')
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    fixture = _load_fixture()
    with tempfile.TemporaryDirectory(prefix='pl-study-pg-') as temporary:
        base = Path(temporary)
        _, _, catalog = fixture.package_at(base / 'synthetic-package')
        class WaitingAdapter(fixture.FakeAdapter):
            def decide(self, state, binding):
                result = super().decide(state, binding)
                result['action'] = result['policy_actions']['ai'] = 'wait'
                return result
        registry = RuntimeRegistry(catalog, adapters={domain: WaitingAdapter(domain)
            for domain in ('warehouse', 'pong', 'kitchen')})
        kwargs = dict(schema='vnext_local_drill', registry=registry, mode='preview',
                      scenario_pools={domain: (11,) for domain in registry.adapters},
                      demonstration_seeds={domain: 10 for domain in registry.adapters})
        with _cluster(preflight['binaries'], base) as (dsn, env):
            first = second = restored = None
            try:
                first = StudyStore.from_postgres(dsn=dsn, **kwargs)
                second = StudyStore.from_postgres(dsn=dsn, **kwargs)
                sessions = []
                checks = {}
                for domain in registry.adapters:
                    for group in ('A', 'B', 'C'):
                        enrolled = first.enroll(domain=domain, group=group, admin=True,
                            participant_reference=f'synthetic-local-pg-{domain}-{group}',
                            consent=True, consent_version=CONSENT_VERSION, question_transmission_consent=True)
                        token, instance = enrolled['session_token'], enrolled['instance']['id']
                        sessions.append((token, instance))
                        _prepare(first, token, instance)
                        for task in (1, 2, 3):
                            first.start_task(token, instance)
                            if domain == 'warehouse' and group == 'A' and task == 1:
                                checks['concurrent_duplicate_command'] = _duplicate_action(first, second, token, instance)
                            if domain == 'warehouse' and group == 'A' and task == 2:
                                checks['read_while_provider_outside_database'] = _read_during_provider(first, second, token, instance)
                            for turn in (0, 1):
                                if domain == 'warehouse' and group == 'A' and task == 1 and turn == 0:
                                    continue
                                first.action(token, instance, command_id=f'{task}-{turn}', expected_turn=turn, human_action='wait')
                        _finish_questionnaires(first, token, instance)
                checks['immutability'] = _guards(first, *sessions[0])
                before = [first.export_instance(token, instance, admin=True) for token, instance in sessions]
                with first._transaction() as db:
                    columns = db.execute('SELECT data_type FROM information_schema.columns WHERE table_schema=? '
                        'AND table_name=? AND column_name=?', (kwargs['schema'], 'vnext_enrollments', 'created')).fetchall()
                    assert columns == [dict(data_type='double precision')]
                    counts = {table: db.execute(f'SELECT COUNT(*) AS n FROM {table}').fetchone()['n']
                        for table in ('v642_instances', 'v642_runs', 'v642_frames', 'v642_commands',
                                      'vnext_preparation', 'vnext_enrollments', 'vnext_surveys')}
                assert counts['v642_instances'] == 9 and counts['v642_runs'] == 27
                assert counts['v642_frames'] == 81 and counts['v642_commands'] == 54
                checks['round_trip_timestamp_type'] = 'double precision'
                first.close(); first = None
                second.close(); second = None
                first = StudyStore.from_postgres(dsn=dsn, **kwargs)
                assert [first.export_instance(t, i, admin=True) for t, i in sessions] == before
                checks['reopen_exact_exports'] = True
                dump = output / 'synthetic-only.backup'
                _run([preflight['binaries']['pg_dump'], '--format=custom', '--no-owner', '--no-privileges',
                      '--schema=' + kwargs['schema'], '--file=' + str(dump), '--dbname=' + dsn], env=env)
                with psycopg.connect(dsn, autocommit=True) as connection:
                    connection.execute('CREATE DATABASE study_restore TEMPLATE template0')
                restored_dsn = dsn.replace('dbname=postgres', 'dbname=study_restore')
                _run([preflight['binaries']['pg_restore'], '--exit-on-error', '--no-owner', '--no-privileges',
                      '--dbname=' + restored_dsn, str(dump)], env=env)
                restored = StudyStore.from_postgres(dsn=restored_dsn, **kwargs)
                after = [restored.export_instance(t, i, admin=True) for t, i in sessions]
                assert after == before, _first_difference(before, after)
                checks['pg_dump_restore_fresh_database_exact_exports'] = True
                checks['restored_guards'] = _guards(restored, *sessions[0])
                receipt = dict(version='study-vnext-local-postgres-drill.v1',
                    status='passed_synthetic_local_software_drill', synthetic_only=True,
                    production_storage_receipt=False, pilot_authorization=False, human_results=False,
                    existing_clusters_accessed=False, external_provider_calls=0,
                    protocol_sha256=protocol_sha256(), postgres_version=_run([preflight['binaries']['postgres'], '--version']),
                    psycopg_version=psycopg.__version__, checks=checks, counts=counts,
                    all_instance_groups_domains=[[d, g] for d in registry.adapters for g in ('A', 'B', 'C')],
                    export_sha256=sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    synthetic_backup_sha256=sha256(dump.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in (
                        Path(__file__), ROOT/'study_vnext/storage.py', ROOT/'study_vnext/store.py',
                        ROOT/'runtime_v642/store.py', ROOT/'tests/test_runtime_v642_store.py')},
                    elapsed_seconds=time.monotonic()-started)
            finally:
                for store in (restored, second, first):
                    if store is not None:
                        store.close()
        receipt['owned_cluster_stopped'] = True
    receipt['temporary_cluster_removed'] = True
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--isolated-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (args.output or ROOT/'output/study_vnext'/('local-postgres-'+uuid.uuid4().hex[:12])).resolve()
    if not args.isolated_worker:
        # Exclude inherited PG* configuration, passfiles and service definitions.
        # No credential values are inspected, printed or copied into this env.
        clean = dict(PATH=os.defpath, LC_ALL='C', LANG='C')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated-worker',
                                 '--output', str(output)], env=clean, timeout=120)
        return result.returncode
    try:
        receipt = _execute(output)
    except Exception as exc:
        # A failed drill is not a scientific/model failure. Keep a small,
        # sanitized failure artifact without secret or SQL exception text.
        output.mkdir(parents=True, exist_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(version='study-vnext-local-postgres-drill.v1',
            status='local_software_drill_failed', error_type=type(exc).__name__,
            production_storage_receipt=False, pilot_authorization=False))+'\n')
        print('Local synthetic PostgreSQL drill failed; see failure.json.', file=sys.stderr)
        raise
    print(json.dumps(dict(status=receipt['status'], receipt=str(output/'receipt.json'),
                          elapsed_seconds=receipt['elapsed_seconds'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
