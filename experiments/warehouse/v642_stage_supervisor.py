"""Observe actual child exits for the two post-calibration stages.

The supervisor grants no scientific authority. Each child retains its own
preconditions, finite inventory and once-only claim. A PID disappearing or a
result file existing is never substituted for Popen.wait().
"""
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys

from .v626_protocol import bound, file_sha, pointer, require
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-post-calibration-subprocess-supervisor.v1'
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = 'scripts/supervise_warehouse_v642_stage.py'


def sources():
    from . import v642_quality_collection as collection
    return collection.source_files()


def invocation(stage, arguments):
    """Return only a fixed, source-owned entry point, never a shell command."""
    from . import v642_final_programs as final, v642_quality_collection as quality
    require(type(arguments) is dict, 'stage arguments must be explicit')
    if stage == 'final-programs':
        require(set(arguments) == {'completed_assets', 'reference_directory', 'run_id'},
                'unknown final-program arguments')
        quality.training.owner.seeds.confirmation.run_name(arguments['run_id'])
        require(pointer(arguments['completed_assets']['path']) == arguments['completed_assets'],
                'completed-assets pointer changed')
        reference = str(Path(arguments['reference_directory']).resolve())
        require(reference == arguments['reference_directory'], 'reference path must be absolute')
        command = [sys.executable, '-u', str(ROOT / 'scripts/finalize_warehouse_v642_programs.py'),
                   '--completed-assets', arguments['completed_assets']['path'],
                   '--reference-directory', reference, '--run-id', arguments['run_id']]
        output = final.OUTPUT / arguments['run_id']
    elif stage == 'calibration-quality':
        require(set(arguments) == {'plan'} and pointer(arguments['plan']['path']) == arguments['plan'],
                'quality plan pointer changed or unknown arguments')
        plan_path = Path(arguments['plan']['path']).resolve()
        require(plan_path.parent.is_relative_to(quality.OUTPUT.resolve())
                and plan_path.name == 'plan.json', 'quality plan is outside its registered root')
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_warehouse_v642_calibration_quality.py'),
                   '--collect', '--plan', str(plan_path)]
        output = plan_path.parent / 'collection'
    else:
        raise RuntimeError('unknown post-calibration stage')
    return command, output.resolve()


def _readonly(path, value):
    write_new(path, value)
    path.chmod(0o444)


def supervise(stage, arguments):
    command, output = invocation(stage, arguments)
    require(not output.exists(), 'stage output already exists; no automatic retry')
    supervision = output.with_name(output.name + '.supervision')
    supervision.mkdir(parents=True, exist_ok=False)
    started = dict(version=VERSION, stage=stage, arguments=arguments, command=command,
                   output=str(output), source_files=sources(), supervisor_pid=os.getpid(),
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   scientific_authority=False, automatic_retry=False, pilot_opened=False)
    started_path = supervision / 'started.json'
    _readonly(started_path, started)
    log = supervision / 'run.log'
    environment = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    try:
        with log.open('xb') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, env=environment)
            code = child.wait()
        log.chmod(0o444)
        receipt = dict(version=VERSION, stage=stage, observed_by='source_bound_subprocess_wait',
                       exit_code=code, worker_pid=child.pid, supervisor_pid=os.getpid(),
                       started=pointer(started_path), log=pointer(log), command=command,
                       finished_utc=datetime.now(timezone.utc).isoformat(),
                       scientific_acceptance=False, pilot_opened=False)
        if code == 0:
            result = pointer(output / 'result.json')
            require(bound(pointer(output / 'completed.json')) == dict(result=result)
                    and not (output / 'failed.json').exists(),
                    'child exited zero without a valid completion seal')
            receipt.update(result=result, completed=pointer(output / 'completed.json'))
        _readonly(supervision / 'exit.json', receipt)
        if output.is_dir():
            _readonly(output / 'supervisor-exit.json', receipt)
        return receipt
    except BaseException as error:
        # Do not invent an exit code if spawning or waiting did not complete.
        _readonly(supervision / 'supervisor-failed.json', dict(
            version=VERSION, started=pointer(started_path), error_type=type(error).__name__,
            error=str(error), automatic_retry=False, pilot_opened=False))
        raise


def verify_execution(execution_ref, result_ref, *, stage):
    """Validate this explicit receipt schema; return its immutable evidence refs."""
    execution = bound(execution_ref)
    require(execution.get('version') == VERSION
            and execution.get('observed_by') == 'source_bound_subprocess_wait'
            and type(execution.get('exit_code')) is int and execution['exit_code'] == 0
            and execution.get('stage') == stage and execution.get('result') == result_ref,
            'unknown or unsuccessful post-calibration subprocess receipt')
    started = bound(execution['started'])
    command, output = invocation(stage, started['arguments'])
    supervision = output.with_name(output.name + '.supervision')
    require(started['version'] == VERSION and started['stage'] == stage
            and started['output'] == str(output) and started['command'] == execution['command'] == command
            and started['source_files'] == sources()
            and started['scientific_authority'] is started['automatic_retry'] is started['pilot_opened'] is False
            and execution['scientific_acceptance'] is execution['pilot_opened'] is False
            and type(execution['worker_pid']) is int and execution['worker_pid'] > 0
            and type(execution['supervisor_pid']) is int and execution['supervisor_pid'] > 0
            and execution['supervisor_pid'] == started['supervisor_pid']
            and execution_ref == pointer(output / 'supervisor-exit.json')
            and execution['started'] == pointer(supervision / 'started.json')
            and execution['log'] == pointer(supervision / 'run.log')
            and execution['completed'] == pointer(output / 'completed.json')
            and result_ref == pointer(output / 'result.json')
            and bound(execution['completed']) == dict(result=result_ref)
            and not (output / 'failed.json').exists()
            and not (supervision / 'supervisor-failed.json').exists()
            and bound(pointer(supervision / 'exit.json')) == execution,
            'post-calibration supervision source, command or completion binding differs')
    before = datetime.fromisoformat(started['started_utc'])
    after = datetime.fromisoformat(execution['finished_utc'])
    require(before.tzinfo and after.tzinfo and before <= after, 'invalid supervision timestamps')
    refs = [execution_ref, execution['started'], execution['log'], execution['completed'],
            pointer(supervision / 'exit.json')]
    for ref in refs:
        path = Path(ref['path'])
        require(not path.is_symlink() and not path.stat().st_mode & 0o222
                and file_sha(path) == ref['sha256'], 'writable, relocated or changed supervision evidence')
    return refs
