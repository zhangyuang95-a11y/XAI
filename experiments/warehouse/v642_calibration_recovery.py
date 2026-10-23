"""Explicit, single-use continuation of one interrupted 20k calibration.

The original owner, claim, journals and provider files remain immutable.
Cached physical trajectories are replayed, never reset or recollected. This
module is a continuation capability, not a new experiment or release gate.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

import numpy as np

from . import v642_calibration_owner as old
from . import v642_calibration_pipeline as pipeline
from .v626_protocol import require, pointer, bound, digest, file_sha
from backend.training.warehouse_v622_actor_training import load_checkpoint, save_checkpoint
from backend.training.warehouse_v629_ppo_update import PPOUpdateCore
from backend.training.warehouse_v629_environment_replay import EnvironmentRolloutReplay, restore_environment, capture_environment
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v625_actor_training import ReferenceTrainingBuffer
from env.warehouse.reset_audit_v614 import capture_stage1_resets
from scripts.run_warehouse_v625 import write_new

VERSION = 'warehouse-v642-explicit-calibration-recovery.v2'
FAILED_VERSION = 'warehouse-v642-explicit-calibration-recovery.v1'
REPAIR_VERSION = 'warehouse-v642-zero-work-engineering-repair.v1'
RECOVERY_CLAIM_NAME = 'recovery-v2.claim.json'
FAILED_RECOVERY_ID = 'warehouse-v642-calibration-02-recovery-01'
FAILED_ATTEMPT_SHA256 = {
    'plan': 'f581300229f2aaaf5058fddcce306fed7d6c1532fbdfbbd4f60bfc780246281d',
    'claim': '8cf2b237a20cc9675bffc9ff209243ee3fea2f54b79f69cef2afcf366c84405e',
    'failure': '30f97915fe180ec568c4a0bf61ca1fa653d0ce9e5ab7b1d071d556ee76c0f124',
    'supervisor_started': 'dbe5a9b97c00acbb4f03d29a72f4b9ae9a884a868dd2b7417f7de8b4dc377a1a',
    'supervisor_exit': '59045ea58a6f5e3a13ab71ea7a24baf192c212ae4c6e5a7e161bd403c88f541a',
    'journal': '559604290f632f3b7f8d695fc49fde81e9b50a945030553a19201e9edf98b031',
    'reset_journal': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'log': '1a17dd4ed79e3cdc22dc2768d02307f9392e0ab0ea69fa6dae054879be52baab',
    'launcher': 'ac830957c0a60c42293464d2469de43e2dc4a39f3783d29034d3c67e404b705d',
    'supervisor_console': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}
AUTHORIZATION = '继续，完成验收'
STARTS = {'raw': 18600, 'regularized': 16800}
ROLES = ('robot_1', 'robot_2')
NEW_SOURCE_NAMES = (
    'experiments/warehouse/v642_calibration_recovery.py',
    'scripts/resume_warehouse_v642_calibration.py',
    'tests/test_warehouse_v642_calibration_recovery.py',
    'backend/training/warehouse_v642_recovery_evidence.py',
    'tests/test_warehouse_v642_recovery_evidence.py',
    'docs/warehouse_v642_calibration_recovery.md',
)


def source_files():
    return {name: file_sha(old.ROOT / name) for name in NEW_SOURCE_NAMES}


def claim_path(original_plan_ref):
    return Path(original_plan_ref['path']).parent / RECOVERY_CLAIM_NAME


def checked_engineering_repair(original_base_path):
    """Pure offline v1 failure/unchanged-input proof, without lease or PID checks."""
    base = Path(original_base_path).resolve()
    failed_plan = base / 'recovery' / FAILED_RECOVERY_ID / 'plan.json'
    require(file_sha(failed_plan) == FAILED_ATTEMPT_SHA256['plan'],
            'only the pinned v1 initialization failure has a v2 repair capability')
    p = bound(pointer(failed_plan))
    require(Path(p['original_plan']['path']) == base / 'plan.json'
            and checked_tree(p['original_roots']) == p['original_input_hashes'],
            'original training inputs changed after the failed v1 engineering attempt')
    for key in ('original_plan', 'original_claim', 'original_descriptor', 'original_controller_contract'):
        require(pointer(p[key]['path']) == p[key], 'original training identity changed after v1 failure')
    for name, sha in p['original_source_files'].items():
        require(file_sha(old.ROOT / name) == sha, 'original training source changed after v1 failure')
    return checked_failed_v1_repair(base, p)


def checked_failed_v1_repair(base, original):
    """Authorize only the pinned pre-work v1 initialization failure, not a retry."""
    base = Path(base)
    folder = base / 'recovery' / FAILED_RECOVERY_ID
    locations = dict(plan=folder / 'plan.json', claim=base / 'recovery.claim.json',
        failure=folder / 'failed.json', supervisor_started=folder / 'supervisor-started.json',
        supervisor_exit=folder / 'supervisor-exit.json', journal=folder / 'owner-journal.jsonl',
        reset_journal=folder / 'actual-resets.jsonl', log=folder / 'run.log',
        launcher=folder / 'launcher.json', supervisor_console=folder / 'supervisor-console.log')
    refs = {}
    for name, path in locations.items():
        require(path.is_file() and not path.is_symlink() and path.resolve() == path,
                'missing or relocated pinned failed recovery evidence')
        refs[name] = pointer(path)
        require(refs[name]['sha256'] == FAILED_ATTEMPT_SHA256[name],
                'pinned zero-work v1 failure evidence changed: ' + name)
    p, claim, failure = (bound(refs[name]) for name in ('plan', 'claim', 'failure'))
    started, exited = (bound(refs[name]) for name in ('supervisor_started', 'supervisor_exit'))
    require(p['version'] == claim['version'] == failure['version'] == FAILED_VERSION
            and p['recovery_id'] == FAILED_RECOVERY_ID and p['authorization'] == AUTHORIZATION
            and claim['recovery_plan'] == failure['recovery_plan'] == refs['plan']
            and failure['recovery_claim'] == refs['claim']
            and failure['error_type'] == 'AttributeError'
            and failure['error'] == "'RecoveryOwner' object has no attribute 'refits'"
            and failure['original_claim_retained'] is True and failure['automatic_retry'] is False
            and failure['pilot_opened'] is False and claim['single_use_even_if_interrupted'] is True,
            'prior recovery was not the pinned missing-refits initialization failure')
    for name in ('original_plan', 'original_claim', 'original_descriptor', 'original_controller_contract',
                 'original_source_files', 'original_calibration', 'original_roots', 'checkpoints',
                 'cached_rollouts', 'original_updates', 'owners', 'keys', 'consumption',
                 'journals', 'reset_journals', 'expires_utc', 'original_input_hashes'):
        require(exact_value_equal(p[name], original[name]),
                'v2 repair differs from unchanged original training history: ' + name)
    snapshots = {}
    require(set(p['source_files']) == set(NEW_SOURCE_NAMES), 'failed v1 source inventory is incomplete')
    for name, sha in p['source_files'].items():
        path = folder / 'source' / name
        require(not path.is_symlink() and file_sha(path) == sha and not path.stat().st_mode & 0o222,
                'failed v1 source snapshot bytes changed')
        snapshots[name] = pointer(path)
    source = dict(path=str(old.ROOT / 'scripts/resume_warehouse_v642_calibration.py'),
                  sha256=p['source_files']['scripts/resume_warehouse_v642_calibration.py'])
    # Verify the old executable through its sealed source copy. The current
    # source is intentionally v2 and must not be presented as the old binary.
    expected_argv = [started['command'][0], source['path'], '--run', '--recovery-plan', refs['plan']['path']]
    require(started['version'] == exited['version'] == 'warehouse-v642-calibration-recovery-supervisor.v1'
            and started['observed_by'] == 'source_bound_subprocess_launch'
            and exited['observed_by'] == 'source_bound_subprocess_wait'
            and type(exited['exit_code']) is int and exited['exit_code'] == 1
            and exited['started'] == refs['supervisor_started'] and exited['log'] == refs['log']
            and started['recovery_plan'] == exited['recovery_plan'] == refs['plan']
            and started['command'] == exited['command'] == expected_argv
            and started['supervisor_source'] == exited['supervisor_source'] == source
            and started['supervisor_pid'] == exited['supervisor_pid']
            and exited['worker_pid'] == claim['owner_pid']
            and 'result' not in exited and 'completed' not in exited,
            'failed v1 has no exact observed nonzero subprocess wait')
    times = [datetime.fromisoformat(value) for value in
             (started['started_utc'], claim['started_utc'], exited['finished_utc'])]
    require(all(t.tzinfo for t in times) and times == sorted(times),
            'failed v1 launch/claim/wait timestamps differ')
    journal = json_lines(locations['journal'])
    require(len(journal) == 1 and journal[0]['kind'] == 'explicit_recovery_claimed'
            and journal[0]['sequence'] == len(json_lines(original['journals'][0]['path']))
            and journal[0]['claim'] == journal[0]['original_claim'] == original['original_claim']
            and journal[0]['recovery_claim'] == refs['claim']
            and journal[0]['recovery_plan'] == refs['plan']
            and locations['reset_journal'].read_bytes() == b'',
            'failed v1 performed restoration, optimization, a reset or extra owner work')
    expected = {str(path): refs[name]['sha256'] for name, path in locations.items() if path.parent == folder}
    expected.update({ref['path']: ref['sha256'] for ref in snapshots.values()})
    require(checked_tree([folder]) == expected
            and {path.name for path in folder.iterdir()} ==
                {Path(path).name for path in expected if Path(path).parent == folder} | {'source'},
            'failed v1 contains unexpected optimizer/checkpoint/provider outputs')
    for root in (pipeline.PROVIDER_ROOT / 'recovery' / FAILED_RECOVERY_ID,
                 pipeline.EXTRACTION_ROOT / 'recovery' / FAILED_RECOVERY_ID,
                 old.observed.TARGET_ROOT / 'recovery' / FAILED_RECOVERY_ID,
                 old.observed.TARGET_ROOT / 'recovery' / digest(refs['claim'])):
        require(not root.exists(), 'failed v1 created a provider or target output; repair is refused')
    return dict(version=REPAIR_VERSION, reason='missing_refits_owner_initialization_before_restoration',
        failed_attempt=refs, failed_source_snapshots=snapshots, failed_tree_files=expected,
        original_input_hashes_unchanged=True, observed_failed_worker_exit_code=1,
        accepted_optimizer_updates=0, new_environment_resets=0, new_provider_calls=0)


def pid_exists(pid):
    require(type(pid) is int and pid > 0, 'original owner PID is invalid')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def json_lines(path):
    raw = Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'), 'journal ends with an uncommitted partial line')
    return [json.loads(line) for line in raw.splitlines()]


def checked_tree(roots):
    """Read hashes only; no model, RNG, environment, outputs or chmod."""
    values = {}
    for root in roots:
        root = Path(root)
        require(root.is_dir() and not root.is_symlink(), 'missing original recovery input tree')
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink(), 'recovery input contains a symlink')
            if path.is_file():
                values[str(path.resolve())] = file_sha(path)
    return values


def verify_files(values):
    for name, sha in values.items():
        require(file_sha(name) == sha, 'original recovery input changed: ' + name)


def exact_value_equal(first, second):
    """Match JSON types as well as values: True is not the integer 1."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact_value_equal(first[k], second[k]) for k in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(exact_value_equal(a, b) for a, b in zip(first, second, strict=True))
    if isinstance(first, float):
        return first.hex() == second.hex()
    return first == second


def original_paths(plan):
    folder = Path(plan.ref['path']).parent
    calibration = folder / 'calibration'
    owners = {v: pointer(calibration / v / 'owner.json') for v in old.seeds.VARIANTS}
    keys = {v: digest(ref) for v, ref in owners.items()}
    roots = [calibration]
    for variant in old.seeds.VARIANTS:
        key = keys[variant]
        roots += [old.ROOT / 'output/warehouse_v642/training-extraction' / key]
        # Raw has no minibatch provider. Its refit targets still exist.
        targets = old.observed.TARGET_ROOT / key
        if targets.exists():
            roots.append(targets)
    roots.append(pipeline.PROVIDER_ROOT / keys['regularized'])
    return folder, calibration, owners, keys, roots


def check_consumption(events, reset_rows, train_inventory, evidence_inventory):
    """Reconstruct constructor cursors from the actual old reset journal."""
    require([row['sequence'] for row in events] == list(range(len(events))), 'original owner journal sequence differs')
    streams = {slot: [c for c in train_inventory['cases'] if c['slot'] == slot] for slot in range(8)}
    cursors = {v: [0] * 8 for v in old.seeds.VARIANTS}
    evidence = {v: 0 for v in old.seeds.VARIANTS}
    pending = None
    count = 0
    for row in events:
        if row['kind'] == 'environment_constructor_and_reset_predebit':
            require(pending is None, 'a reset predebit was not committed before another constructor')
            pending = row
            v, case = row['variant'], row['case']
            require(v in cursors, 'unknown original reset variant')
            if row['scope'] == 'train':
                slot = case['slot']
                require(case == streams[slot][cursors[v][slot]], 'original training resets skipped their finite inventory')
                cursors[v][slot] += 1
            else:
                require(row['scope'] == 'program_evidence' and case == evidence_inventory['cases'][evidence[v]],
                        'original program resets skipped their finite inventory')
                evidence[v] += 1
        elif row['kind'] == 'actual_reset_bound':
            require(pending is not None and row['variant'] == pending['variant'] and row['scope'] == pending['scope']
                    and row['reset_hook_index'] == count and row['reset'] == reset_rows[count]
                    and row['reset']['seed'] == pending['case']['scenario_seed'], 'original reset hook/predebit mismatch')
            pending = None
            count += 1
    require(pending is None and count == len(reset_rows), 'ambiguous interrupted physical reset; recovery is refused')
    return dict(training_cursors=cursors, evidence_cursors=evidence, actual_reset_count=count)


def inspect_original(*, reference_directory, plan_ref):
    # Actual A/B readiness precedes user-supplied plan or output reads.
    plan = old.load_seed_plan(reference_directory, plan_ref)
    contract = pipeline.checked_contract(plan)
    folder, calibration, owners, keys, roots = original_paths(plan)
    claim = pointer(old.claim_path())
    claim_value = bound(claim)
    descriptor = pointer(calibration / 'descriptor.json')
    require(claim_value['plan'] == plan.ref and claim_value['descriptor'] == descriptor
            and claim_value['single_use_even_if_interrupted'] is True and claim_value['automatic_resume'] is False,
            'original claim is not the registered interrupted calibration')
    require(not pid_exists(claim_value['owner_pid']), 'original owner PID is still alive')
    require(not any((calibration / name).exists() for name in ('completed.json', 'failed.json', 'controller-completed.json')),
            'completed or failed original attempt cannot acquire this interruption-only capability')
    clearance = old.seeds.checked_clearance(plan, execution=False)
    require(datetime.now(timezone.utc) < datetime.fromisoformat(clearance['value']['expires_utc']),
            'original finite seed reservation lease expired; no renewal is implicit')
    require(claim_value['clearance'] == clearance['clearance'], 'original claim clearance differs')
    events = json_lines(calibration / 'owner-journal.jsonl')
    resets = json_lines(calibration / 'actual-resets.jsonl')
    consumption = check_consumption(events, resets, bound(plan.proposal.value['train_case_inventory']),
                                    bound(plan.proposal.value['program_evidence_inventory']))
    require(all(n == 1650 for n in consumption['evidence_cursors'].values()), 'program source collection must already be complete')
    checkpoints = {v: pointer(calibration / v / 'checkpoints' / f'{step:06d}' / 'owner-state.json')
                   for v, step in STARTS.items()}
    for variant, ref in checkpoints.items():
        value = bound(ref)
        require(value['claim'] == claim and value['owner'] == owners[variant]
                and value['source_files'] == old.source_files()
                and value['core']['processed_joint_steps'] == value['reserved_joint_steps'] == STARTS[variant]
                and value['core']['safe_batch_boundary'] is True and value['core']['failed'] is False,
                'recovery requires the exact registered safe checkpoint')
        require(not (calibration / variant / 'checkpoints/020000').exists(), 'already materialized final checkpoint refuses recovery')
    cached = {v: [] for v in old.seeds.VARIANTS}
    original_updates = {v: [] for v in old.seeds.VARIANTS}
    for variant in old.seeds.VARIANTS:
        for rollout in sorted((calibration / variant / 'rollouts').iterdir()):
            require(rollout.is_dir() and rollout.name.isdigit(), 'unknown original rollout directory')
            trace = pointer(rollout / 'trace.json')
            audit = pointer(rollout / 'audit.json')
            row = bound(trace)
            require(row['start_joint_step'] == int(rollout.name), 'cached rollout directory identity differs')
            if (rollout / 'update.json').exists():
                original_updates[variant].append(pointer(rollout / 'update.json'))
            if row['start_joint_step'] >= STARTS[variant]:
                cached[variant].append(dict(trace=trace, audit=audit,
                    update=pointer(rollout / 'update.json') if (rollout / 'update.json').exists() else None))
    require(not cached['raw'] and [(bound(r['trace'])['start_joint_step'], bound(r['trace'])['end_joint_step'], r['update'] is not None)
            for r in cached['regularized']] == [(16800, 17824, True), (17824, 18600, False)],
            'this explicit capability only covers the observed 16800/17824 interruption')
    pending = {(r['variant'], r['step']) for r in events if r['kind'] == 'optimizer_update_predebit'}
    pending -= {(r['variant'], r['step']) for r in events if r['kind'] == 'optimizer_update_accepted'}
    require(pending == {('regularized', 18600)}, 'uncommitted optimizer history differs from the declared interruption')
    return plan, dict(original_plan=plan.ref, original_claim=claim, original_descriptor=descriptor,
        original_controller_contract=contract, original_source_files=bound(contract)['source_files'],
        original_calibration=str(calibration), original_roots=list(map(str, roots)),
        checkpoints=checkpoints, cached_rollouts=cached, original_updates=original_updates,
        owners=owners, keys=keys, consumption=consumption,
        journals=[pointer(calibration / 'owner-journal.jsonl')], reset_journals=[pointer(calibration / 'actual-resets.jsonl')],
        expires_utc=clearance['value']['expires_utc'], original_input_hashes=checked_tree(roots))


def prepare(*, reference_directory, plan_ref, recovery_id, authorization):
    require(authorization == AUTHORIZATION, 'explicit human continuation authorization is required')
    plan, original = inspect_original(reference_directory=reference_directory, plan_ref=plan_ref)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', recovery_id) is not None, 'unsafe recovery ID')
    base = Path(plan.ref['path']).parent
    require(not claim_path(plan.ref).exists(), 'this interruption already consumed its explicit v2 recovery capability')
    repair = checked_engineering_repair(base)
    previous = bound(repair['failed_attempt']['plan'])
    require(all(exact_value_equal(previous[name], value) for name, value in original.items()),
            'v2 prepared original history differs from the pinned v1 failure')
    folder = base / 'recovery' / recovery_id
    sources = source_files()
    folder.mkdir(parents=True, exist_ok=False)
    # Freeze the stopped process's complete input inventory without changing
    # any original bytes or their SHA identity. Native started/evidence/budget
    # JSON can still be writable even after their final content was committed.
    for name in original['original_input_hashes']:
        Path(name).chmod(0o444)
    for ref in repair['failed_attempt'].values():
        Path(ref['path']).chmod(0o444)
    for name in repair['failed_tree_files']:
        Path(name).chmod(0o444)
    for name, sha in sources.items():
        raw = (old.ROOT / name).read_bytes()
        require(file_sha(old.ROOT / name) == sha, 'recovery source changed before snapshot')
        target = folder / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o444)
    value = dict(version=VERSION, recovery_id=recovery_id, authorization=authorization, source_files=sources,
        reference_directory=str(Path(reference_directory).resolve()), **original,
        config=asdict(old.PPOConfig()), restores_training_state_not_new_initialization=True,
        cached_frames_are_numerical_replay_not_new_samples=True, maximum_joint_steps_per_variant=20000,
        original_claim_retained=True, original_files_read_only=True, automatic_retry=False,
        scientific_acceptance=False, pilot_opened=False, repair_of=repair)
    write_new(folder / 'plan.json', value)
    (folder / 'plan.json').chmod(0o444)
    return pointer(folder / 'plan.json')


def checked_plan(ref):
    value = bound(ref)
    require(value['version'] == VERSION and value['authorization'] == AUTHORIZATION
            and value['source_files'] == source_files() and value['config'] == asdict(old.PPOConfig()),
            'recovery authorization, source or registered configuration differs')
    folder = Path(ref['path']).parent
    expected = Path(value['original_plan']['path']).parent / 'recovery' / value['recovery_id']
    require(folder == expected.resolve() and Path(ref['path']).name == 'plan.json', 'recovery plan location differs')
    # Full original input bytes are checked by prepare, the claimed entry and
    # its final boundary. Recheck the small pinned failure archive here; doing
    # another full source-episode scan per live operation adds no new samples.
    require(value['repair_of'] == checked_failed_v1_repair(Path(value['original_plan']['path']).parent, value),
            'v2 engineering repair chain changed after preparation')
    require(all(not Path(ref['path']).stat().st_mode & 0o222
                for ref in value['repair_of']['failed_attempt'].values()),
            'v1 failure evidence was not sealed read-only during v2 preparation')
    for name, sha in value['source_files'].items():
        snap = folder / 'source' / name
        require(file_sha(snap) == sha and not snap.stat().st_mode & 0o222, 'recovery source snapshot changed')
    for name, sha in value['original_source_files'].items():
        require(file_sha(old.ROOT / name) == sha, 'original 407-file source closure changed')
    for key in ('original_plan', 'original_claim', 'original_descriptor', 'original_controller_contract'):
        require(pointer(value[key]['path']) == value[key], 'original plan/claim/descriptor/contract changed')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(value['expires_utc']), 'recovery seed lease expired')
    require(not pid_exists(bound(value['original_claim'])['owner_pid']), 'original owner PID is still alive')
    calibration = Path(value['original_calibration'])
    require(not any((calibration / name).exists() for name in ('completed.json', 'failed.json', 'controller-completed.json')),
            'original terminal state changed after recovery preparation')
    return value


class RecoveryOwner(old.CalibrationOwner):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('use the explicitly authorized single-use recovery context')

    def live(self, *, full=False):
        require(old.ACTIVE.get() is self and not self.failed and not self.closed, 'recovery owner is not active')
        require(pointer(self.recovery_claim_ref['path']) == self.recovery_claim_ref, 'recovery claim changed')
        require(datetime.now(timezone.utc) < datetime.fromisoformat(self.recovery_plan['expires_utc']), 'recovery seed lease expired')
        if full:
            require(checked_plan(self.recovery_plan_ref) == self.recovery_plan, 'recovery plan changed')
            require(self.sources == old.source_files(), 'original owner source changed during recovery')

    def event(self, kind, **values):
        self.live()
        row = dict(version=VERSION, sequence=self.sequence, kind=kind, claim=self.claim_ref,
            recovery_claim=self.recovery_claim_ref, utc=datetime.now(timezone.utc).isoformat(), **values)
        self.journal.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.sequence += 1
        if kind in ('full_training_checkpoint_restored', 'cached_rollout_replayed', 'optimizer_update_accepted',
                    'scheduled_refit_bound', 'checkpoint', 'recovery_pair_steps_finished'):
            print(json.dumps(dict(recovery_progress=kind, sequence=row['sequence'],
                variant=values.get('variant'), step=values.get('step', self.states[values['variant']].joint_steps
                    if values.get('variant') in self.states else None)),
                sort_keys=True, separators=(',', ':'), allow_nan=False), flush=True)
        return row

    def update(self, variant, *, evidence_provider=None):
        from backend.training.warehouse_v642_recovery_evidence import RecoveryEvidenceProvider
        self.live(full=True)
        require(variant in self.pending, 'no verified recovery rollout to optimize')
        state, pending = self.states[variant], self.pending[variant]
        require(training_fingerprint(state) == pending['state_sha256'], 'training state changed since recovery replay')
        provider = evidence_provider or self.providers.get(variant)
        require((variant == 'raw' and provider is None) or (variant == 'regularized'
            and type(provider) is RecoveryEvidenceProvider and provider.owner_ref == self.owners[variant]),
            'recovery feedback must use its exact owned current-NN evidence provider')
        self.event('optimizer_update_predebit', variant=variant, trace=pending['trace'], step=state.joint_steps)
        result = self.cores[variant].update_batch(pending['batch'], batch_sha256=pending['trace']['sha256'], evidence_provider=provider)
        folder = self.folder / variant / 'rollouts' / f'{pending["batch"]["start_joint_step"]:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / 'update.json'
        write_new(file, result)
        file.chmod(0o444)
        ref = pointer(file)
        if pending.get('original_update') is not None:
            require(exact_value_equal(result, bound(pending['original_update'])), 'recomputed committed update differs; continuation is refused')
            proof = dict(original_update=pending['original_update'], recomputed_update=ref, matched=True,
                exact_recursive_value_equality=True, new_environment_resets=0,
                replayed_physical_frames=len(pending['batch']['rows']), added_scientific_joint_steps=0)
            self.replay_proofs.append(proof)
            # Preserve byte-identical old committed update as the result prefix.
        else:
            self.updates[variant].append(ref)
        del self.pending[variant]
        self.state_fingerprints[variant] = training_fingerprint(state)
        self.event('optimizer_update_accepted', variant=variant, receipt=ref, step=state.joint_steps,
            replay_of_existing_commit=pending.get('original_update') is not None)
        return result

    def replay_cached_rollout(self, variant, row):
        self.live(full=True)
        require(variant not in self.pending, 'pending recovery batch must be completed first')
        state = self.states[variant]
        trace = bound(row['trace'])
        require(state.joint_steps == trace['start_joint_step'] == self.reserved[variant], 'cached rollout is not the exact continuation boundary')
        require(training_fingerprint(state) == self.state_fingerprints[variant], 'state changed before cached replay')
        state.advance_joint_steps(len(trace['events']))
        batch, audit = old.seeds.partitions.assemble_training_batch(self.plan.proposal, self.readers[variant], row['trace'], state)
        require(audit == bound(row['audit']), 'cached physics/NN audit does not reproduce exactly')
        # Increment the original per-slot cursor only for resets already recorded
        # in this cached trace; no constructor, reset or fresh draw occurs.
        reset_ids = set(trace['physics']['resets'])
        seen = set()
        for event in trace['events']:
            case = event['case']
            if case['episode_id'] in reset_ids and case['episode_id'] not in seen:
                slot = case['slot']
                require(case == self.train_streams[slot][self.cursors[variant][slot]], 'cached reset cursor skipped its original inventory')
                self.cursors[variant][slot] += 1
                seen.add(case['episode_id'])
        require(seen == reset_ids, 'cached reset inventory is incomplete')
        self.environments[variant] = [None if env is None else restore_environment(capture_environment(env))
                                      for env in self.readers[variant].environments]
        self.active_cases[variant] = deepcopy(self.readers[variant].active)
        self.reserved[variant] = state.joint_steps
        self.pending[variant] = dict(batch=batch, trace=row['trace'], audit=row['audit'],
            state_sha256=training_fingerprint(state), original_update=row['update'])
        self.state_fingerprints[variant] = training_fingerprint(state)
        self.event('cached_rollout_replayed', variant=variant, trace=row['trace'], audit=row['audit'],
            original_update=row['update'], original_physical_frames=len(trace['events']), new_environment_resets=0)

    def checkpoint(self, variant):
        self.live(full=True)
        require(variant not in self.pending, 'cannot checkpoint a partially updated recovery batch')
        state = self.states[variant]
        require(self.cores[variant].processed_joint_steps == self.reserved[variant] == state.joint_steps
                and training_fingerprint(state) == self.state_fingerprints[variant], 'unsafe recovery checkpoint boundary')
        folder = self.folder / variant / 'checkpoints' / f'{state.joint_steps:06d}'
        folder.mkdir(parents=True, exist_ok=False)
        saved = save_checkpoint(state, folder / 'state.pt')
        Path(saved['path']).chmod(0o444)
        live = [e is not None and not (e.state.terminated or e.state.truncated) for e in self.environments[variant]]
        value = dict(version=VERSION, claim=self.claim_ref, recovery_claim=self.recovery_claim_ref,
            owner=self.owners[variant], source_files=self.sources, recovery_source_files=self.recovery_plan['source_files'],
            training_state=pointer(saved['path']), core=self.cores[variant].snapshot(), reader=self.readers[variant].snapshot(state),
            environments=[capture_environment(e) if live[i] else None for i, e in enumerate(self.environments[variant])],
            active_cases=[c if live[i] else None for i, c in enumerate(self.active_cases[variant])],
            reset_cursors=self.cursors[variant], program_source_cursor=self.evidence_cursor[variant],
            reserved_joint_steps=self.reserved[variant], refits=self.refits[variant], updates=self.updates[variant],
            provider=None if variant not in self.providers else self.providers[variant].snapshot(state),
            safe_snapshot_not_resume_authority=True, automatic_resume=False)
        file = folder / 'owner-state.json'
        write_new(file, value)
        file.chmod(0o444)
        ref = pointer(file)
        self.checkpoints[variant].append(ref)
        self.event('checkpoint', variant=variant, snapshot=ref)
        return ref


def restore_variant(active, variant):
    from backend.training.warehouse_v642_recovery_evidence import RecoveryEvidenceProvider
    p = active.recovery_plan
    snapshot = bound(p['checkpoints'][variant])
    saved = snapshot['training_state']
    state = load_checkpoint(saved['path'], saved['sha256'])
    buffer_ref = active.plan.proposal.readiness.value['reference']['buffer']
    buffer = ReferenceTrainingBuffer.load(buffer_ref['path'], buffer_ref['sha256'], state.reference_sha256)
    old.seeds.partitions.check_reference_buffer(active.plan.proposal, buffer)
    # Constructor creates a carrier generator, immediately replaced with the
    # exact checkpoint bit-generator state. No new experimental seed is used.
    core = PPOUpdateCore.restore(state, buffer, snapshot['core'], variant=variant,
        rng=np.random.Generator(np.random.PCG64(0)), config=active.config)
    reader = EnvironmentRolloutReplay.restore(snapshot['reader'], state, owner_ref=active.owners[variant],
        inventory_ref=active.plan.proposal.value['train_case_inventory'], config=active.config)
    active.states[variant], active.cores[variant], active.readers[variant] = state, core, reader
    active.environments[variant] = [None if row is None else restore_environment(row) for row in snapshot['environments']]
    active.active_cases[variant] = deepcopy(snapshot['active_cases'])
    active.cursors[variant] = deepcopy(snapshot['reset_cursors'])
    active.evidence_cursor[variant] = snapshot['program_source_cursor']
    active.reserved[variant] = snapshot['reserved_joint_steps']
    active.refits[variant] = list(snapshot['refits'])
    active.state_fingerprints[variant] = training_fingerprint(state)
    active.pools[variant] = old.observed.audit_observed_pool(active.program_sources[variant], reference_directory=active.reference_directory)
    if variant == 'regularized':
        active.providers[variant] = RecoveryEvidenceProvider.restore_prefix(snapshot['provider'], state,
            reference_directory=active.reference_directory, owner_ref=active.owners[variant], observed=active.pools[variant],
            output=pipeline.PROVIDER_ROOT / 'recovery' / p['recovery_id'], recovery_claim_ref=active.recovery_claim_ref,
            config=active.config)
    active.event('full_training_checkpoint_restored', variant=variant, checkpoint=p['checkpoints'][variant],
        step=state.joint_steps, numerical_initialization_replaced_by_exact_saved_state=True, new_environment_resets=0)


def scheduled_refit(active, variant):
    active.live(full=True)
    state = active.states[variant]
    require(state.joint_steps == 18600 and active.refits[variant] == [15000, 16800], 'only the remaining scheduled 93% extraction is allowed')
    folder = active.folder / variant / 'refit-targets/18600'
    folder.mkdir(parents=True, exist_ok=False)
    actor_ref, _ = old.frozen_export(state, folder / 'current-NN.npz', variant)
    key = digest(active.owners[variant])
    targets = old.observed.relabel_current_NN(active.pools[variant], actor_ref,
        old.observed.TARGET_ROOT / 'recovery' / active.recovery_plan['recovery_id'] / key / 'refit-18600')
    previous = bound(bound(active.scheduled_refits[variant][-1])['extraction'])
    programs = {role: previous['roles'][role]['program'] for role in ROLES}
    _, binding = old.observed.refit_observed_pair_at_boundary(reference_directory=active.reference_directory,
        owner_ref=active.owners[variant], state=state, observed=active.pools[variant], targets_ref=targets,
        program_refs=programs, output=pipeline.EXTRACTION_ROOT / 'recovery' / active.recovery_plan['recovery_id'] / key / 'step-18600',
        variant=variant, epoch_id='scheduled-18600', config=active.config)
    require(variant == 'regularized', 'raw already completed its 93% extraction')
    active.providers[variant].install_extraction(binding, active.pools[variant], state)
    active.refits[variant].append(18600)
    active.scheduled_refits[variant].append(binding)
    active.event('scheduled_refit_bound', variant=variant, step=18600, binding=binding)


@contextmanager
def claimed_recovery(*, recovery_plan_ref):
    p = checked_plan(recovery_plan_ref)
    require(old.ACTIVE.get() is None, 'owners cannot nest')
    require(checked_tree(p['original_roots']) == p['original_input_hashes'], 'original recovery tree changed after preparation')
    plan = old.load_seed_plan(p['reference_directory'], p['original_plan'])
    folder = Path(recovery_plan_ref['path']).parent
    claim_file = claim_path(p['original_plan'])
    require(not claim_file.exists(), 'explicit recovery capability already consumed')
    claim = dict(version=VERSION, recovery_plan=recovery_plan_ref, original_claim=p['original_claim'],
        authorization=p['authorization'], owner_pid=os.getpid(), single_use_even_if_interrupted=True,
        source_files=p['source_files'], started_utc=datetime.now(timezone.utc).isoformat(), expires_utc=p['expires_utc'],
        maximum_joint_steps_per_variant=20000, pilot_opened=False, repair_of=p['repair_of'])
    write_new(claim_file, claim)
    claim_file.chmod(0o444)
    active = RecoveryOwner.__new__(RecoveryOwner)
    active.recovery_plan_ref, active.recovery_plan = recovery_plan_ref, p
    active.recovery_claim_ref, active.claim_ref = pointer(claim_file), p['original_claim']
    active.descriptor_ref = p['original_descriptor']
    active.plan, active.sources, active.config = plan, old.source_files(), old.PPOConfig()
    active.folder, active.reference_directory = folder, p['reference_directory']
    active.owners = deepcopy(p['owners'])
    active.failed = active.closed = False
    active.sequence = len(json_lines(p['journals'][0]['path']))
    active.states = {}; active.cores = {}; active.readers = {}; active.pending = {}; active.pools = {}; active.providers = {}
    active.state_fingerprints = {}; active.collecting = {}; active.collecting_source = set()
    active.reserved = {}; active.cursors = {}; active.evidence_cursor = {}; active.environments = {}; active.active_cases = {}
    active.refits = {}
    active.updates = deepcopy(p['original_updates'])
    active.replay_proofs = []
    original = Path(p['original_calibration'])
    active.program_sources = {v: pointer(original / v / 'program-source/source.json') for v in old.seeds.VARIANTS}
    active.checkpoints = {v: [pointer(original / v / 'checkpoints' / f'{step:06d}' / 'owner-state.json')
        for step in (4000, 15000, 16800, 18600) if step <= STARTS[v]] for v in old.seeds.VARIANTS}
    active.scheduled_refits = {v: [pointer(pipeline.EXTRACTION_ROOT / p['keys'][v] / f'step-{step}' / 'observed-source-binding.json')
        for step in (15000, 16800, 18600) if step <= STARTS[v]] for v in old.seeds.VARIANTS}
    training = bound(plan.proposal.value['train_case_inventory'])['cases']
    active.train_streams = {slot: [c for c in training if c['slot'] == slot] for slot in range(8)}
    active.evidence_cases = bound(plan.proposal.value['program_evidence_inventory'])['cases']
    token = old.ACTIVE.set(active)
    try:
        with (folder / 'owner-journal.jsonl').open('x') as journal, capture_stage1_resets(folder / 'actual-resets.jsonl') as actual:
            active.journal, active.actual_resets = journal, actual
            active.event('explicit_recovery_claimed', recovery_plan=recovery_plan_ref, original_claim=p['original_claim'])
            yield active
            active.live(full=True)
            require(not active.pending and all(active.states[v].joint_steps == active.cores[v].processed_joint_steps == 20000
                    and active.refits[v] == [15000, 16800, 18600] and not active.cores[v].failed for v in old.seeds.VARIANTS),
                    'recovery pair is incomplete or numerically failed')
            require(active.cores['raw'].accepted_program_updates == {r: 0 for r in ROLES}
                    and all(active.cores['regularized'].accepted_program_updates[r] > 0 for r in ROLES),
                    'raw zero feedback / regularized both-role positive feedback requirement failed')
            require(checked_tree(p['original_roots']) == p['original_input_hashes'], 'original recovery tree changed during continuation')
    except BaseException as error:
        active.failed = True
        write_new(folder / 'failed.json', dict(version=VERSION, recovery_plan=recovery_plan_ref,
            recovery_claim=active.recovery_claim_ref, error_type=type(error).__name__, error=str(error),
            original_claim_retained=True, automatic_retry=False, pilot_opened=False))
        raise
    finally:
        active.closed = True
        old.ACTIVE.reset(token)


def run(*, recovery_plan_ref):
    with claimed_recovery(recovery_plan_ref=recovery_plan_ref) as active:
        p = active.recovery_plan
        for variant in old.seeds.VARIANTS:
            restore_variant(active, variant)
        for row in p['cached_rollouts']['regularized']:
            active.replay_cached_rollout('regularized', row)
            active.update('regularized')
        require(active.cursors == p['consumption']['training_cursors'], 'restored suffix cursors differ from original actual reset history')
        scheduled_refit(active, 'regularized')
        active.checkpoint('regularized')
        while any(s.joint_steps < 20000 for s in active.states.values()):
            for variant in old.seeds.VARIANTS:
                if active.states[variant].joint_steps >= 20000:
                    continue
                active.collect_rollout(variant)
                active.update(variant)
                if active.states[variant].joint_steps == 20000:
                    active.checkpoint(variant)
        active.event('recovery_pair_steps_finished', per_variant_joint_steps={v: active.states[v].joint_steps for v in old.seeds.VARIANTS})
    folder = active.folder
    for name in ('owner-journal.jsonl', 'actual-resets.jsonl'):
        (folder / name).chmod(0o444)
    result = dict(version=VERSION, completed=True, numerical_training_completed=True,
        original_plan=p['original_plan'], original_claim=p['original_claim'], original_descriptor=p['original_descriptor'],
        original_controller_contract=p['original_controller_contract'], original_source_files=p['original_source_files'],
        recovery_plan=recovery_plan_ref, recovery_claim=active.recovery_claim_ref, source_files=p['source_files'],
        repair_of=p['repair_of'],
        checkpoints=active.checkpoints, updates=active.updates,
        journals=[*p['journals'], pointer(folder / 'owner-journal.jsonl')],
        reset_journals=[*p['reset_journals'], pointer(folder / 'actual-resets.jsonl')],
        owners=active.owners, program_sources=active.program_sources, scheduled_refits=active.scheduled_refits,
        replay_proofs=active.replay_proofs, per_variant_joint_steps={v: s.joint_steps for v, s in active.states.items()},
        accepted_program_updates={v: dict(core.accepted_program_updates) for v, core in active.cores.items()},
        original_files_preserved=True, actual_original_process_exit_code_unknown=True,
        model_acceptance_passed=False, quality_and_program_calibration_evaluation_still_required=True,
        publishable=False, pilot_opened=False)
    write_new(folder / 'result.json', result)
    (folder / 'result.json').chmod(0o444)
    write_new(folder / 'completed.json', dict(result_sha256=file_sha(folder / 'result.json')))
    (folder / 'completed.json').chmod(0o444)
    return pointer(folder / 'result.json')
