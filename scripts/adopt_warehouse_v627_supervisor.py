#!/usr/bin/env python3
"""Observe orphaned, still-running jobs and finish only unstarted v627 jobs.

No solver state is restored and no job is retried. Orphan process exits are
observed, not falsely reported as waitpid results. The scientific source and
single-use plan remain unchanged; this controller has its own bound receipt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import v627_protocol as v
from scripts.run_warehouse_v627 import checked_plan, finalize
from scripts.run_warehouse_v625 import write_new


def processes():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,stat=,command='],
                            check=True, text=True, capture_output=True)
    out = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) != 5:
            continue
        pid, parent, group, stat, command = fields
        if str(ROOT / 'scripts/run_warehouse_v627.py') not in command:
            continue
        out[int(pid)] = dict(pid=int(pid), parent=int(parent), group=int(group),
                             stat=stat, argv=shlex.split(command))
    return out


def is_job(process, stage, job_id):
    args = process['argv']
    script = str(ROOT / 'scripts/run_warehouse_v627.py')
    return (script in args and '--stage' in args and '--job' in args
            and args[args.index('--stage') + 1] == stage
            and args[args.index('--job') + 1] == job_id)


def check_result(folder, plan_path, spec):
    result = json.loads((folder / 'result.json').read_bytes())
    v.require(json.loads((folder / 'completed.json').read_bytes()) ==
              dict(result_sha256=v.file_sha(folder / 'result.json')),
              'missing final worker checksum')
    v.require(result['job'] == spec and result['plan_sha256'] == v.file_sha(plan_path),
              'worker result is not bound to the same single-use plan')
    v.require(result['solver']['checkpoint']['pending_calls'] == 0,
              'worker left an unpaid or unsettled solver call')
    return v.pointer(folder / 'result.json')


def execute(stage='S64'):
    path, plan = checked_plan(stage)
    control = path.parent / 'execution'
    v.require(control.is_dir() and not (control / 'completed.json').exists()
              and not (control / 'failed.json').exists()
              and not (path.parent / 'summary.json').exists(),
              'cannot replace a completed or failed experiment')
    listing = processes()
    for process in listing.values():
        args = process['argv']
        if (str(ROOT / 'scripts/run_warehouse_v627.py') in args
                and '--execute' in args and '--stage' in args
                and args[args.index('--stage') + 1] == stage):
            raise RuntimeError('original supervisor is still alive')
    active, pending, settled = [], [], []
    for spec in plan['jobs']:
        folder = path.parent / 'jobs' / spec['job_id']
        found = [p for p in listing.values() if is_job(p, stage, spec['job_id'])]
        v.require(len(found) <= 1, 'duplicate running job')
        if found:
            p = found[0]
            v.require(p['parent'] == 1 and p['group'] == p['pid'],
                      'only orphaned original worker process groups can be adopted')
            v.require((folder / 'started.json').is_file() and not (folder / 'failed.json').exists(),
                      'unregistered or failed worker cannot be adopted')
            active.append(dict(spec=spec, process=p, child=None, log=None))
        elif folder.exists():
            # Never restore a crashed/dirty worker. A finished worker must carry
            # its final result and checksum; absence is a hard failure.
            ref = check_result(folder, path, spec)
            settled.append(dict(job=spec['job_id'], result=ref,
                                exit_evidence='already_finished_before_adoption',
                                waitpid_returncode=None))
        else:
            pending.append(spec)
    v.require(len(active) <= 3, 'registered parallelism exceeded')
    claim = dict(schema='warehouse-v627-supervisor-adoption.v1',
                 plan=v.pointer(path), scientific_source_sha256=plan['source_sha256'],
                 controller=v.pointer(Path(__file__).resolve()),
                 observed_original_supervisor_absent=True,
                 orphan_workers=[dict(job=i['spec']['job_id'], **i['process']) for i in active],
                 already_settled=settled, unstarted=[s['job_id'] for s in pending],
                 no_worker_restore=True, no_job_retry=True, **v.ZERO)
    write_new(control / 'adoption.claim.json', claim)
    events = control / 'adoption.events.jsonl'
    with events.open('x'):
        pass
    sequence, tail = 0, '0' * 64

    def event(kind, data):
        nonlocal sequence, tail
        row = dict(sequence=sequence + 1, previous_sha256=tail,
                   event=kind, monotonic=time.monotonic(), data=data)
        row['record_sha256'] = v.digest(row)
        with events.open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
        sequence, tail = row['sequence'], row['record_sha256']

    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    owned_waited = 0
    old_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise RuntimeError('replacement supervisor interrupted')

    signal.signal(signal.SIGTERM, interrupt)
    event('adoption_started', dict(claim=v.pointer(control / 'adoption.claim.json')))
    try:
        while active or pending:
            while pending and len(active) < 3:
                spec = pending.pop(0)
                checked_plan(stage)
                folder = path.parent / 'jobs' / spec['job_id']
                v.require(not folder.exists(), 'never launch an existing worker directory')
                log = (control / (spec['job_id'] + '.log')).open('x')
                child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_warehouse_v627.py'),
                                          '--stage', stage, '--job', spec['job_id']],
                                         cwd=ROOT, env=env, stdout=log, stderr=log,
                                         start_new_session=True)
                process = dict(pid=child.pid, group=child.pid)
                active.append(dict(spec=spec, process=process, child=child, log=log))
                event('unstarted_job_launched', dict(job=spec['job_id'], pid=child.pid))
            listing = processes()
            for item in list(active):
                spec, info, child = item['spec'], item['process'], item['child']
                folder = path.parent / 'jobs' / spec['job_id']
                if child is not None:
                    code = child.poll()
                    alive = code is None
                else:
                    p = listing.get(info['pid'])
                    alive = p is not None and not p['stat'].startswith('Z')
                    if alive:
                        v.require(is_job(p, stage, spec['job_id']), 'adopted PID identity changed')
                    code = None
                start = folder / 'started.json'
                if alive and start.exists():
                    a = json.loads(start.read_bytes())
                    deadline = a['search_deadline_monotonic']
                    final = folder / 'final-started.json'
                    if final.exists():
                        deadline = min(a['deadline_monotonic'], json.loads(final.read_bytes())['deadline_monotonic'])
                    if time.monotonic() >= deadline:
                        raise RuntimeError('registered external phase deadline: ' + spec['job_id'])
                if not alive:
                    if child is not None:
                        code = child.wait()
                        owned_waited += 1
                        item['log'].close()
                        v.require(code == 0, 'newly owned worker failed')
                    ref = check_result(folder, path, spec)
                    row = dict(job=spec['job_id'], result=ref,
                               exit_evidence='waitpid' if child is not None else 'orphan_exit_observed_with_final_worker_seal',
                               waitpid_returncode=code)
                    settled.append(row)
                    event('worker_settled', row)
                    active.remove(item)
            if active:
                time.sleep(.5)
        v.require(len(settled) == 6, 'not all finite jobs settled')
        receipt = dict(schema='warehouse-v627-supervisor-adoption-completed.v1',
                       claim=v.pointer(control / 'adoption.claim.json'),
                       all_six_processes_settled=True, settled=settled,
                       newly_owned_workers_waitpid_count=owned_waited,
                       originally_orphaned_workers_waitpid_count=0,
                       events=v.pointer(events), no_worker_restore=True, no_job_retry=True, **v.ZERO)
        write_new(control / 'adoption.completed.json', receipt)
        # "owned" here means only the replacement's OS child processes. The
        # separate adoption receipt explicitly records the original orphans.
        write_new(control / 'completed.json', dict(all_owned_workers_waited=True, jobs=6))
    except BaseException as error:
        failures = []
        for item in active:
            pid, child = item['process']['pid'], item['child']
            try:
                current = processes().get(pid)
                if current is not None and is_job(current, stage, item['spec']['job_id']):
                    os.killpg(pid, signal.SIGTERM)
                if child is not None:
                    try:
                        code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid, signal.SIGKILL)
                        code = child.wait()
                    item['log'].close()
                else:
                    code = None
                failures.append(dict(job=item['spec']['job_id'], waitpid_returncode=code,
                                     was_OS_child=child is not None))
            except ProcessLookupError:
                pass
        write_new(control / 'adoption.failed.json', dict(error_type=type(error).__name__,
                  error=str(error), workers=failures, automatic_retry=False, **v.ZERO))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
    return finalize(stage)


if __name__ == '__main__':
    print(json.dumps(execute(), sort_keys=True))
