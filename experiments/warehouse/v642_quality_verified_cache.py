"""Separately versioned calibration reader with bracketed evidence verification.

The frozen native collector/auditor code objects are reused in an isolated
namespace. Their module and globals are never patched. Only repeated prerequisite
validation is hoisted; all NN actions, physics, counters, scoring and gates remain
the native implementations. This is a new once-only fresh evaluation attempt.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType, SimpleNamespace
import json
import sys

from . import v642_quality_collection as native
from . import v642_calibration_admission as native_admission
from .v626_protocol import bound, pointer, require, file_sha
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-matched-quality-bracketed-verification.v3'
OUTPUT = native.ROOT / 'output/warehouse_v642/calibration-quality-cached'
FILES = ('experiments/warehouse/v642_quality_verified_cache.py',
         'scripts/run_warehouse_v642_verified_quality.py',
         'tests/test_warehouse_v642_quality_verified_cache.py',
         'docs/warehouse_v642_quality_verified_cache.md')


def source_files():
    return dict(sorted({**native.source_files(),
        **{name: file_sha(native.ROOT / name) for name in FILES}}.items()))


def claim_path():
    return OUTPUT / 'once-only.claim.json'


def _save(path, value):
    write_new(path, value)
    path.chmod(0o444)
    return pointer(path)


class VerifiedPrerequisites:
    """Full byte verification before/after; critical execution assets each reset.

    No stat-only result grants authority. Full native verification initializes
    the cache; a second full native verification must reproduce it before an
    evaluation can acquire a successful process exit.
    """
    def __init__(self, full_check, sources):
        self.full_check, self.sources = full_check, sources
        self.saved = None
        self.full_checks = self.fast_checks = 0

    def full(self, ref):
        value = self.full_check(ref)
        self.full_checks += 1
        if self.saved is not None:
            require(ref == self.ref and value == self.saved,
                    'complete prerequisite verification changed')
        else:
            self.ref, self.saved = deepcopy(ref), deepcopy(value)
        return deepcopy(value)

    def checked(self, ref):
        if self.saved is None:
            return self.full(ref)
        require(ref == self.ref and pointer(ref['path']) == ref,
                'cached prerequisite plan changed')
        plan, folder = self.saved
        require(plan['source_files'] == self.sources(), 'cached source changed')
        refs = [plan['exact_plan'], *plan['fixed']['actors'].values()]
        for roles in plan['final_programs']['programs'].values():
            refs.extend(roles.values())
        for asset in refs:
            path = Path(asset['path'])
            require(pointer(path) == asset and not path.is_symlink()
                    and not path.stat().st_mode & 0o222,
                    'critical cached execution asset changed')
        # Source snapshots are execution inputs, so recheck them every reset.
        for name, sha in plan['source_files'].items():
            path = folder / 'source' / name
            require(file_sha(path) == sha and not path.stat().st_mode & 0o222,
                    'cached source snapshot changed')
        self.fast_checks += 1
        return deepcopy(self.saved)


def isolated_api():
    """Rebind the declared profile globals without mutating the frozen module."""
    globals_ = dict(vars(native))
    for name, value in vars(native).items():
        if isinstance(value, FunctionType) and value.__module__ == native.__name__:
            clone = FunctionType(value.__code__, globals_, value.__name__,
                                 value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            globals_[name] = clone
    globals_.update(VERSION=VERSION, OUTPUT=OUTPUT,
                    source_files=source_files, claim_path=claim_path)
    cache = VerifiedPrerequisites(globals_['checked'], source_files)
    globals_['checked'] = cache.checked
    return SimpleNamespace(**globals_), cache


def require_prior_terminal():
    """Never overlap or relabel the original once-only evaluation attempt."""
    if not native.claim_path().exists():
        return None
    original = bound(pointer(native.claim_path()))
    folder = Path(original['plan']['path']).parent / 'collection'
    ref = pointer(folder / 'supervisor-exit.json')
    terminal = bound(ref)
    require(terminal.get('observed_by') == 'source_bound_subprocess_wait'
            and type(terminal.get('exit_code')) is int
            and terminal['exit_code'] != 0,
            'original evaluation must have an actually observed unsuccessful terminal exit')
    from . import v642_stage_supervisor as supervisor
    supervision = folder.with_name(folder.name + '.supervision')
    started = bound(terminal['started'])
    command, output = supervisor.invocation('calibration-quality', started['arguments'])
    require(terminal['version'] == started['version'] == supervisor.VERSION
            and terminal['stage'] == started['stage'] == 'calibration-quality'
            and output == folder and started['output'] == str(folder)
            and started['source_files'] == native.source_files()
            and terminal['command'] == started['command'] == command
            and terminal['started'] == pointer(supervision / 'started.json')
            and terminal['log'] == pointer(supervision / 'run.log')
            and bound(pointer(supervision / 'exit.json')) == terminal
            and terminal['pilot_opened'] is started['pilot_opened'] is False,
            'original unsuccessful terminal lost its source-bound supervision evidence')
    for asset in (ref, terminal['started'], terminal['log'], pointer(supervision / 'exit.json')):
        path = Path(asset['path'])
        require(not path.is_symlink() and not path.stat().st_mode & 0o222
                and pointer(path) == asset, 'original terminal evidence changed')
    return ref


def prepare(*args, **kwargs):
    require_prior_terminal()
    api, _ = isolated_api()
    return api.prepare(*args, **kwargs)


def audit_seeds(ref):
    require_prior_terminal()
    api, cache = isolated_api()
    result = api.audit_seeds(ref)
    cache.full(ref)
    return result


def collect(ref):
    prior = require_prior_terminal()
    api, cache = isolated_api()
    folder = Path(ref['path']).parent / 'collection'
    try:
        result = api.collect(ref)
        cache.full(ref)
        require(cache.full_checks >= 2, 'both full verification brackets are required')
        _save(folder / 'verification-bracket.json', dict(version=VERSION,
            plan=ref, result=result, prior_terminal=prior, source_files=source_files(),
            full_native_prerequisite_checks=cache.full_checks,
            critical_asset_checks=cache.fast_checks,
            all_native_NN_frame_checks_retained=True,
            NN_updates=0, program_fit_calls=0, qualification_changes=0,
            pilot_opened=False))
        return result
    except BaseException as error:
        if folder.exists() and not (folder / 'failed.json').exists():
            _save(folder / 'failed.json', dict(version=VERSION,
                error_type=type(error).__name__, error=str(error),
                single_use_claim_retained=True, pilot_opened=False))
        raise


def audit(result_ref, exit_ref):
    api, cache = isolated_api()
    result = bound(result_ref)
    execution = bound(exit_ref)
    require(execution.get('implementation_version') == VERSION
            and execution.get('observed_by') == 'exec_tool_terminal_result'
            and execution.get('exit_code') == 0
            and execution.get('source_files') == source_files()
            and execution.get('command') == [sys.executable, '-u',
                str(native.ROOT / 'scripts/run_warehouse_v642_verified_quality.py'),
                'collect', '--plan', result['plan']['path']],
            'verified profile requires its actual observed, source-bound CLI exit')
    bracket_ref = pointer(Path(result_ref['path']).parent / 'verification-bracket.json')
    bracket = bound(bracket_ref)
    require(bracket['version'] == VERSION and bracket['result'] == result_ref
            and bracket['plan'] == result['plan']
            and bracket['source_files'] == source_files()
            and bracket['full_native_prerequisite_checks'] >= 2
            and bracket['prior_terminal'] == require_prior_terminal()
            and bracket['all_native_NN_frame_checks_retained'] is True,
            'completed full verification bracket required')
    globals_ = dict(vars(native_admission), collection=api)
    auditor = FunctionType(native_admission.audit.__code__, globals_, 'audit')
    value = auditor(result_ref, exit_ref)
    cache.full(result['plan'])
    require(pointer(bracket_ref['path']) == bracket_ref, 'verification bracket changed')
    value.update(implementation_version=VERSION, implementation_source_files=source_files(),
                 verification_bracket=bracket_ref,
                 full_native_prerequisite_checks=cache.full_checks)
    return value
