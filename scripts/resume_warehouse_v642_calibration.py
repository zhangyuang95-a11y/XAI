#!/usr/bin/env python3
"""Prepare or run a separately claimed, source-bound v6.42 continuation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import v642_calibration_recovery as recovery

SUPERVISOR_VERSION = 'warehouse-v642-calibration-recovery-supervisor.v1'


def _atomic_receipt(path, value):
    """Publish one fsynced, read-only receipt atomically without replacement."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    recovery.write_new(temporary, value)
    temporary.chmod(0o444)
    # link fails if the final path exists; no prior receipt is overwritten.
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return recovery.pointer(path)


def supervise(recovery_plan_ref):
    """Observe the actual child wait status, independently of tool sessions."""
    # Validate authorization and the registered source closure before output
    # creation, NN/RNG construction, subprocess creation or experimental work.
    plan = recovery.checked_plan(recovery_plan_ref)
    folder = Path(recovery_plan_ref['path']).parent
    cli = Path(__file__).resolve()
    source = recovery.pointer(cli)
    recovery.require(plan['source_files']['scripts/resume_warehouse_v642_calibration.py'] == source['sha256'],
                     'supervisor executable is not the registered recovery CLI')
    recovery.require(not (folder / 'supervisor-started.json').exists()
                     and not (folder / 'supervisor-exit.json').exists()
                     and not recovery.claim_path(plan['original_plan']).exists(),
                     'recovery supervision or its single-use worker was already started')
    command = [str(Path(sys.executable).resolve()), str(cli), '--run', '--recovery-plan', recovery_plan_ref['path']]
    started_utc = datetime.now(timezone.utc).isoformat()
    header = dict(version=SUPERVISOR_VERSION, observed_by='source_bound_subprocess_launch',
        recovery_plan=recovery_plan_ref, command=command, supervisor_pid=os.getpid(), started_utc=started_utc,
        supervisor_source=source, log_path=str(folder / 'run.log'))
    _atomic_receipt(folder / 'supervisor-started.json', header)
    started_ref = recovery.pointer(folder / 'supervisor-started.json')
    stage = 'spawn'
    try:
        with (folder / 'run.log').open('xb') as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, cwd=str(ROOT), close_fds=True)
            stage = 'wait'
            exit_code = worker.wait()
            recovery.require(type(exit_code) is int, 'subprocess wait did not return an integer exit status')
            log.flush()
            os.fsync(log.fileno())
    except BaseException as error:
        # A spawn/wait exception is not a terminal observation. Preserve that
        # distinction instead of inventing a subprocess return code.
        _atomic_receipt(folder / 'supervisor-failed.json', dict(version=SUPERVISOR_VERSION,
            observed_by='source_bound_subprocess_' + stage + '_failure', started=started_ref,
            recovery_plan=recovery_plan_ref, command=command, supervisor_pid=os.getpid(),
            supervisor_source=source, error_type=type(error).__name__, error=str(error)))
        raise
    (folder / 'run.log').chmod(0o444)
    value = dict(version=SUPERVISOR_VERSION, observed_by='source_bound_subprocess_wait',
        recovery_plan=recovery_plan_ref, started=started_ref, command=command,
        worker_pid=worker.pid, supervisor_pid=os.getpid(), started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(), supervisor_source=source,
        exit_code=exit_code, log=recovery.pointer(folder / 'run.log'))
    binding_error = None
    if exit_code == 0:
        try:
            result = recovery.pointer(folder / 'result.json')
            completed = recovery.pointer(folder / 'completed.json')
            payload = recovery.bound(result)
            recovery.require(recovery.bound(completed) == dict(result_sha256=result['sha256'])
                and payload['version'] == recovery.VERSION and payload['completed'] is True
                and payload['numerical_training_completed'] is True and payload['recovery_plan'] == recovery_plan_ref
                and payload['pilot_opened'] is False, 'successful worker lacks its exact completed result binding')
            value.update(result=result, completed=completed)
        except BaseException as error:
            # Still report the child's real exit 0, but fail the supervisor and
            # omit success pointers so an independent auditor rejects it.
            binding_error = error
            value['result_binding_error'] = dict(type=type(error).__name__, message=str(error))
    receipt = _atomic_receipt(folder / 'supervisor-exit.json', value)
    if binding_error is not None:
        raise binding_error
    if exit_code != 0:
        # Preserve the exact signed return code in the receipt; POSIX shells
        # use 128+signal when the child was terminated by a signal.
        raise SystemExit(exit_code if exit_code > 0 else 128 - exit_code)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--supervise', action='store_true')
    parser.add_argument('--reference-directory', type=Path)
    parser.add_argument('--original-plan', type=Path)
    parser.add_argument('--recovery-id')
    parser.add_argument('--human-authorization')
    parser.add_argument('--recovery-plan', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if any(x is None for x in (args.reference_directory, args.original_plan, args.recovery_id, args.human_authorization)):
            parser.error('--prepare requires reference, original plan, recovery ID and explicit human authorization')
        recovery.old.seeds.partitions.admission.load_readiness(args.reference_directory).verify()
        result = recovery.prepare(reference_directory=args.reference_directory, plan_ref=recovery.pointer(args.original_plan),
            recovery_id=args.recovery_id, authorization=args.human_authorization)
    else:
        if args.recovery_plan is None:
            parser.error('--run/--supervise require --recovery-plan')
        ref = recovery.pointer(args.recovery_plan)
        result = supervise(ref) if args.supervise else recovery.run(recovery_plan_ref=ref)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
