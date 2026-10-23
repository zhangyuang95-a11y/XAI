"""Audit an explicitly recovered pair without rewriting its interrupted run.

Recovery is a new, source-bound execution of the same finite training stream.
Its terminal observation must never be attributed to the vanished process.
This reader grants neither scientific acceptance nor permission to publish.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import sys

from . import v642_calibration_pipeline as training
from .v626_protocol import require, pointer, bound, file_sha, digest
from .v642_update_receipt_audit import audit_updates
from backend.training.warehouse_v642_checkpoint_export import export_checkpoint_actor
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from scripts.run_warehouse_v625 import write_new

ROOT = training.owner.ROOT
RECOVERY_VERSION = 'warehouse-v642-explicit-calibration-recovery.v2'
RECOVERY_CLAIM_NAME = 'recovery-v2.claim.json'
VERSION = 'warehouse-v642-completed-recovered-calibration-assets.v1'
V3_RECOVERY_VERSION = 'warehouse-v642-accelerated-calibration-recovery.v3'
V3_VERSION = 'warehouse-v642-completed-accelerated-calibration-assets.v1'
ROLES = ('robot_1', 'robot_2')
VARIANTS = ('raw', 'regularized')
STEPS = (4000, 15000, 16800, 18600, 20000)
STARTS = {'raw': 18600, 'regularized': 16800}
EXECUTOR_FILES = (
    'experiments/warehouse/v642_calibration_recovery.py',
    'scripts/resume_warehouse_v642_calibration.py',
    'tests/test_warehouse_v642_calibration_recovery.py',
    'backend/training/warehouse_v642_recovery_evidence.py',
    'tests/test_warehouse_v642_recovery_evidence.py',
    'docs/warehouse_v642_calibration_recovery.md',
)
FILES = ('experiments/warehouse/v642_recovered_assets.py',
         'scripts/audit_warehouse_v642_recovery_results.py',
         'tests/test_warehouse_v642_recovered_assets.py')


@dataclass(frozen=True)
class RecoveryProfile:
    executor: str
    report: str
    claim: str
    supervisor: str
    cli: str
    carrier: str = RECOVERY_VERSION


V2 = RecoveryProfile(RECOVERY_VERSION, VERSION, RECOVERY_CLAIM_NAME,
    'warehouse-v642-calibration-recovery-supervisor.v1', 'scripts/resume_warehouse_v642_calibration.py')
V3 = RecoveryProfile(V3_RECOVERY_VERSION, V3_VERSION, 'recovery-v3.claim.json',
    'warehouse-v642-accelerated-recovery-supervisor.v1', 'scripts/resume_warehouse_v642_accelerated.py')


def recovery_profile(version):
    require(version in (V2.executor, V3.executor), 'unknown explicit recovered executor version')
    return V2 if version == V2.executor else V3


def report_profile(version):
    require(version in (V2.report, V3.report), 'unknown explicit recovered asset report version')
    return V2 if version == V2.report else V3


def source_files():
    return {name: file_sha(ROOT / name) for name in FILES}


def immutable(ref, inputs):
    p = Path(ref['path'])
    require(p.is_absolute() and p.resolve() == p and pointer(p) == ref
            and not p.is_symlink() and not p.stat().st_mode & 0o222,
            'recovery audit input changed, writable or symlinked')
    inputs.append(deepcopy(ref))
    return p


def _journal(refs, inputs):
    rows = []
    require(type(refs) is list and len(refs) == 2,
            'recovery must retain both original and continuation journals')
    for ref in refs:
        path = immutable(ref, inputs)
        rows.extend(json.loads(s) for s in path.read_text().splitlines())
    return rows


def audit_terminal(result_ref, supervisor_exit_ref, inputs, *, profile=V2):
    """Require an actual tool exit or a registered supervisor's real OS wait."""
    immutable(supervisor_exit_ref, inputs)
    e = bound(supervisor_exit_ref)
    require(type(e.get('exit_code')) is int and e['exit_code'] == 0
            and e.get('result') == result_ref,
            'actual recovery terminal exit0 bound to this result is required')
    if e.get('observed_by') == 'exec_tool_terminal_result':
        return e
    require(e.get('observed_by') == 'source_bound_subprocess_wait'
            and e.get('version') == profile.supervisor
            and 'result_binding_error' not in e,
            'recovery exit was not observed by a registered terminal supervisor')
    folder = Path(result_ref['path']).parent
    require(Path(supervisor_exit_ref['path']) == folder / 'supervisor-exit.json',
            'recovery wait receipt was relocated')
    immutable(e['started'], inputs)
    require(e['started'] == pointer(folder / 'supervisor-started.json'),
            'recovery wait receipt lacks the exclusive supervisor launch commitment')
    started = bound(e['started'])
    source = pointer(ROOT / profile.cli)
    if profile == V2:
        immutable(dict(path=str(folder / 'source' / profile.cli), sha256=source['sha256']), inputs)
    else:
        result = bound(result_ref); contract = bound(result['source_contract'])
        immutable(contract['source_snapshots'][profile.cli], inputs)
        require(contract['source_snapshots'][profile.cli]['sha256'] == source['sha256'],
                'accelerated supervisor source snapshot differs')
    command = [str(Path(sys.executable).resolve()), source['path'], '--run',
               '--recovery-plan', str(folder / 'plan.json')]
    require(started['version'] == e['version']
            and started['observed_by'] == 'source_bound_subprocess_launch'
            and e['command'] == started['command'] == command
            and e['supervisor_source'] == started['supervisor_source'] == source
            and e['recovery_plan'] == started['recovery_plan'] == pointer(folder / 'plan.json')
            and type(e['worker_pid']) is int and e['worker_pid'] > 0
            and type(e['supervisor_pid']) is int and e['supervisor_pid'] > 0
            and e['supervisor_pid'] == started['supervisor_pid'] != e['worker_pid']
            and e['started_utc'] == started['started_utc']
            and _time(e['started_utc']) <= _time(e['finished_utc']),
            'registered recovery wait source, command, PID or timestamps differ')
    require(e['completed'] == pointer(folder / 'completed.json')
            and bound(e['completed']) == dict(result_sha256=result_ref['sha256']),
            'registered recovery wait does not bind the actual result seal')
    immutable(e['completed'], inputs)
    immutable(e['log'], inputs)
    log_path = Path(started['log_path'])
    require(log_path == folder / 'run.log' and e['log'] == pointer(log_path),
            'registered recovery wait log differs from its exclusive launch commitment')
    result = bound(result_ref)
    require(result['recovery_plan'] == e['recovery_plan'],
            'recovery wait result belongs to another registered plan')
    claim = bound(result['recovery_claim'])
    require(claim['owner_pid'] == e['worker_pid']
            and _time(e['started_utc']) <= _time(claim['started_utc'])
            <= _time(e['finished_utc']),
            'recovery worker PID or claim time differs from the observed subprocess')
    return e


def executor_sources(*, profile=V2):
    """Read the complete registered executor closure, without executing it."""
    if profile == V3:
        from . import v642_accelerated_recovery as recovery
        require(recovery.VERSION == V3.executor and recovery.CLAIM_NAME == V3.claim,
                'unknown accelerated executor inventory')
        return recovery.source_files()
    require(profile == V2, 'unknown recovery executor profile')
    from . import v642_calibration_recovery as recovery
    require(recovery.VERSION == RECOVERY_VERSION
            and recovery.RECOVERY_CLAIM_NAME == RECOVERY_CLAIM_NAME
            and recovery.AUTHORIZATION == '继续，完成验收'
            and tuple(recovery.NEW_SOURCE_NAMES) == EXECUTOR_FILES,
            'unknown recovery executor source inventory or authorization')
    expected = {name: file_sha(ROOT / name) for name in EXECUTOR_FILES}
    require(recovery.source_files() == expected,
            'recovery executor omitted or added a source dependency')
    return expected


def engineering_repair(base, inputs):
    """Recheck the one pinned zero-work failure; never authorize a retry."""
    from . import v642_calibration_recovery as recovery
    repair = recovery.checked_engineering_repair(base)
    require(repair['version'] == 'warehouse-v642-zero-work-engineering-repair.v1'
            and repair['reason'] == 'missing_refits_owner_initialization_before_restoration'
            and type(repair['observed_failed_worker_exit_code']) is int
            and repair['observed_failed_worker_exit_code'] == 1
            and repair['original_input_hashes_unchanged'] is True
            and all(type(repair[k]) is int and repair[k] == 0 for k in
                    ('accepted_optimizer_updates', 'new_environment_resets', 'new_provider_calls')),
            'engineering replacement did not stop before all new scientific work')
    for ref in repair['failed_attempt'].values():
        immutable(ref, inputs)
    for ref in repair['failed_source_snapshots'].values():
        immutable(ref, inputs)
    for name, sha in repair['failed_tree_files'].items():
        immutable(dict(path=name, sha256=sha), inputs)
    return repair


def _time(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'recovery timestamps must include a timezone')
    return parsed


def exact_value(left, right):
    """Preserve integer types and the sign/bit pattern of JSON float values."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return left.hex() == right.hex()
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_value(x, y) for x, y in zip(left, right, strict=True))
    return left == right


def audit_recovery_plan(r, *, folder, plan, descriptor, original_claim,
                        reference_directory, inputs, profile=V2):
    """Offline plan/claim validation; do not consult current lease or PID state."""
    p = bound(r['recovery_plan'])
    claim = bound(r['recovery_claim'])
    base = Path(plan.ref['path']).parent
    original = base / 'calibration'
    require(isinstance(p.get('recovery_id'), str)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', p['recovery_id'])
            and folder == base / 'recovery' / p['recovery_id']
            and r['recovery_plan'] == pointer(folder / 'plan.json')
            and r['recovery_claim'] == pointer(base / profile.claim),
            'recovery plan or single-use claim is outside its registered location')
    require(p['version'] == profile.executor and p['authorization'] == '继续，完成验收'
            and p['reference_directory'] == str(Path(reference_directory).resolve())
            and p['config'] == asdict(training.owner.PPOConfig())
            and p['maximum_joint_steps_per_variant'] == 20000
            and p['restores_training_state_not_new_initialization'] is True
            and p['cached_frames_are_numerical_replay_not_new_samples'] is True
            and p['original_claim_retained'] is True and p['original_files_read_only'] is True
            and p['automatic_retry'] is False and p['scientific_acceptance'] is False
            and p['pilot_opened'] is False,
            'recovery authorization, training configuration or scope differs')
    for key in ('original_plan', 'original_claim', 'original_descriptor',
                'original_controller_contract', 'original_source_files', 'owners'):
        require(p[key] == r[key], 'recovery result differs from its sealed plan: ' + key)
    require(p['original_plan'] == plan.ref and p['original_calibration'] == str(original)
            and p['journals'] == [pointer(original / 'owner-journal.jsonl')]
            and p['reset_journals'] == [pointer(original / 'actual-resets.jsonl')],
            'recovery plan omitted or relocated original training journals')
    expected_sources = executor_sources() if profile == V2 else executor_sources(profile=profile)
    require(r['source_files'] == p['source_files'] == expected_sources,
            'recovery executor source contract is incomplete or changed')
    repair = engineering_repair(base, inputs)
    require(exact_value(r.get('repair_of'), repair)
            and exact_value(p.get('repair_of'), repair)
            and exact_value(claim.get('repair_of'), repair),
            'recovery v2 lost or replaced the pinned zero-work failure evidence')
    require(claim['version'] == profile.executor and claim['recovery_plan'] == r['recovery_plan']
            and claim['original_claim'] == r['original_claim']
            and claim['authorization'] == p['authorization']
            and claim['source_files'] == expected_sources
            and type(claim['owner_pid']) is int and claim['owner_pid'] > 0
            and claim['single_use_even_if_interrupted'] is True
            and claim['maximum_joint_steps_per_variant'] == 20000
            and claim['expires_utc'] == p['expires_utc'] and claim['pilot_opened'] is False,
            'recovery single-use claim authorization, source or limits differ')
    if profile == V3:
        audit_accelerated_contract(r, p, claim, inputs)
    require(original_claim['clearance'] == descriptor['clearance'],
            'original claim uses a different finite-seed clearance')
    immutable(original_claim['clearance'], inputs)
    clearance = bound(original_claim['clearance'])
    require(p['expires_utc'] == clearance['expires_utc']
            and _time(clearance['issued_utc']) <= _time(claim['started_utc'])
            < _time(clearance['expires_utc']),
            'recovery started outside the original finite-seed lease')
    # Recompute the sealed original inventory, including unconsumed cached
    # traces and provider snapshots; comparing pointers alone is insufficient.
    owners = {v: pointer(original / v / 'owner.json') for v in VARIANTS}
    keys = {v: digest(ref) for v, ref in owners.items()}
    roots = [original]
    for variant in VARIANTS:
        roots.append(training.EXTRACTION_ROOT / keys[variant])
        targets = training.owner.observed.TARGET_ROOT / keys[variant]
        if targets.exists():
            roots.append(targets)
    roots.append(training.PROVIDER_ROOT / keys['regularized'])
    require(p['owners'] == owners and p['keys'] == keys
            and p['original_roots'] == list(map(str, roots)),
            'recovery original owner or evidence-tree inventory differs')
    actual = {}
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), 'missing original evidence tree')
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink(), 'original recovery tree contains a symlink')
            if path.is_file():
                actual[str(path)] = file_sha(path)
    require(actual == p['original_input_hashes'],
            'original recovery file inventory or bytes changed')
    for name, sha in actual.items():
        immutable(dict(path=name, sha256=sha), inputs)
    for variant in VARIANTS:
        expected_checkpoint = pointer(original / variant / 'checkpoints'
                                      / f'{STARTS[variant]:06d}' / 'owner-state.json')
        require(p['checkpoints'][variant] == expected_checkpoint,
                'recovery starts from a different original safe checkpoint')
        original_updates = [pointer(path) for path in sorted(
            (original / variant / 'rollouts').glob('*/update.json'))]
        require(p['original_updates'][variant] == original_updates
                and r['updates'][variant][:len(original_updates)] == original_updates,
                'recovery replaced the original committed optimizer prefix')
    expected_cached = {'raw': [], 'regularized': [dict(
        trace=pointer(original / 'regularized' / 'rollouts' / f'{step:06d}' / 'trace.json'),
        audit=pointer(original / 'regularized' / 'rollouts' / f'{step:06d}' / 'audit.json'),
        update=(pointer(original / 'regularized' / 'rollouts' / '016800' / 'update.json')
                if step == 16800 else None)) for step in (16800, 17824)]}
    require(p['cached_rollouts'] == expected_cached,
            'recovery changed the two declared cached rollout boundaries')
    for row in expected_cached['regularized']:
        trace = bound(row['trace'])
        require((trace['start_joint_step'], trace['end_joint_step']) in
                ((16800, 17824), (17824, 18600)), 'cached physical trace boundary differs')
    return p


def audit_accelerated_contract(r,p,claim,inputs):
    from . import v642_accelerated_recovery as execution
    from . import v642_checkpoint_cutover as cutover
    from backend.training import warehouse_v642_accelerated_recovery_evidence as adapter
    immutable(r['source_contract'],inputs);contract=execution.checked_sources(r['source_contract'])
    require(r['source_contract']==p['source_contract']==claim['source_contract']
        and r['cutover']==p['cutover']==claim['cutover']
        and claim['acceleration_authorization']==p['acceleration_authorization']==execution.AUTHORIZATION
        and r['checkpoint_serialization_schema']==r['event_serialization_schema']==V2.executor,
        'accelerated executor lost its explicit source, authorization or carrier binding')
    stopped=cutover.read_completed(r['cutover'])
    require(stopped['parent_v2_plan']==p['parent_recovery_plan']==r['parent_recovery_plan']
        and stopped['mode']=='original-safe-replay' and stopped['source_contract']==r['source_contract'],
        'accelerated handoff dropped new physical samples or committed updates')
    for ref in stopped['inputs']:immutable(ref,inputs)
    refs,roots,files,manifests=adapter.checked_prefix_manifests(p['prefix_manifest_refs'])
    for ref in (*refs,*files):immutable(ref,inputs)
    require(manifests[0]['files']==p['original_input_hashes'], 'accelerated original prefix inventory differs')
    for ref in contract['source_snapshots'].values():immutable(ref,inputs)


def audit_replay_proofs(proofs, *, original, folder, inputs):
    require(type(proofs) is list and len(proofs) == 1,
            'exactly the committed regularized 16800-to-17824 replay is required')
    proof = proofs[0]
    old = pointer(original / 'regularized' / 'rollouts' / '016800' / 'update.json')
    new = pointer(folder / 'regularized' / 'rollouts' / '016800' / 'update.json')
    require(proof['original_update'] == old and proof['recomputed_update'] == new
            and old['path'] != new['path'] and proof['matched'] is True
            and proof['exact_recursive_value_equality'] is True
            and proof['new_environment_resets'] == proof['added_scientific_joint_steps'] == 0
            and proof['replayed_physical_frames'] == 1024,
            'reconstruction proof skipped, duplicated or replaced the committed update')
    immutable(old, inputs)
    immutable(new, inputs)
    left, right = bound(old), bound(new)
    require(left['start_joint_step'] == right['start_joint_step'] == 16800
            and left['end_joint_step'] == right['end_joint_step'] == 17824
            and exact_value(left, right),
            'reconstructed committed optimizer update is not exactly equal')


def audit_update_chain(r, *, original_events, continuation, original, folder, inputs):
    """Link accepted commits to predebited physical traces and NN hash chains."""
    for variant in VARIANTS:
        old = [e['receipt'] for e in original_events
               if e['kind'] == 'optimizer_update_accepted' and e['variant'] == variant]
        require(r['updates'][variant][:len(old)] == old,
                'result replaced optimizer receipts accepted by the original journal')
        new = [e['receipt'] for e in continuation if e['kind'] == 'optimizer_update_accepted'
               and e['variant'] == variant and e.get('replay_of_existing_commit') is False]
        require(r['updates'][variant][len(old):] == new,
                'continuation accepted optimizer receipts differ from final history')
    pending, replayed = {}, []
    for event in continuation:
        if event['kind'] == 'optimizer_update_predebit':
            variant = event['variant']
            require(variant not in pending, 'continuation duplicated an unresolved optimizer predebit')
            pending[variant] = event
        elif event['kind'] == 'optimizer_update_accepted':
            variant = event['variant']
            require(variant in pending, 'continuation accepted an optimizer update without a predebit')
            before = pending.pop(variant)
            row = bound(event['receipt'])
            trace_ref = before['trace']
            immutable(trace_ref, inputs)
            trace = bound(trace_ref)
            require(before['step'] == event['step'] == row['end_joint_step'] == trace['end_joint_step']
                    and row['start_joint_step'] == trace['start_joint_step']
                    and row['batch_sha256'] == trace_ref['sha256']
                    and trace['owner'] == r['owners'][variant] and trace['partition'] == 'train',
                    'accepted continuation update differs from its predebited physical batch')
            start = row['start_joint_step']
            location = original if variant == 'regularized' and start in (16800, 17824) else folder
            require(trace_ref == pointer(location / variant / 'rollouts' / f'{start:06d}' / 'trace.json'),
                    'continuation optimizer used a relocated physical trace')
            checked_ref = pointer(Path(trace_ref['path']).with_name('audit.json'))
            immutable(checked_ref, inputs)
            checked = bound(checked_ref)
            require(checked['environment_resets'] == checked['optimizer_steps'] == 0
                    and checked['program_evidence_frames_in_gradient_buffers'] == 0
                    and checked['physics_and_NN_replay']['trace'] == trace_ref
                    and checked['physics_and_NN_replay']['owner'] == r['owners'][variant]
                    and checked['physics_and_NN_replay']['public_NN_audit']['trace'] == trace_ref,
                    'continuation physical/NN replay is not bound to its actual optimizer batch')
            if event.get('replay_of_existing_commit') is True:
                replayed.append(event['receipt'])
    require(not pending and replayed == [r['replay_proofs'][0]['recomputed_update']],
            'continuation optimizer predebits or committed replay events are incomplete')


def audit_reset_chain(continuation, resets, *, inventory, initial):
    """Account only genuinely new physical resets against the remaining inventory."""
    streams = {slot: [case for case in inventory['cases'] if case['slot'] == slot] for slot in range(8)}
    cursors = deepcopy(initial)
    pending, count = None, 0
    for event in continuation:
        if event['kind'] == 'environment_constructor_and_reset_predebit':
            require(pending is None and event['scope'] == 'train',
                    'continuation introduced an ambiguous or program-source physical reset')
            variant, case = event['variant'], event['case']
            slot = case['slot']
            require(case == streams[slot][cursors[variant][slot]],
                    'continuation reset skipped its remaining finite case inventory')
            cursors[variant][slot] += 1
            pending = event
        elif event['kind'] == 'actual_reset_bound':
            require(pending is not None and count < len(resets)
                    and event['variant'] == pending['variant'] and event['scope'] == 'train'
                    and event['reset_hook_index'] == count and event['reset'] == resets[count]
                    and event['reset']['seed'] == pending['case']['scenario_seed'],
                    'continuation physical reset differs from its predebited remaining case')
            count += 1
            pending = None
    require(pending is None and count == len(resets), 'continuation reset history is incomplete')
    return cursors


def audit_NN_hash_chain(updates, checkpoint_values):
    current = deepcopy(checkpoint_values[0]['core']['actor_role_sha256'])
    at_step = {v['core']['processed_joint_steps']: v['core']['actor_role_sha256']
               for v in checkpoint_values}
    for ref in updates:
        row = bound(ref)
        for epoch in row['epochs']:
            for update in epoch['updates']:
                if update['actor_updated']:
                    require(update['role_sha256_before'] == current
                            and set(update['role_sha256_after']) == set(ROLES),
                            'optimizer NN role hashes do not form a continuous checkpoint chain')
                    current = deepcopy(update['role_sha256_after'])
        if row['end_joint_step'] in at_step:
            require(current == at_step[row['end_joint_step']],
                    'registered checkpoint NN weights differ from its accepted optimizer suffix')


def audit_provider(r, *, recovery_plan, final_checkpoint, inputs, profile=V2):
    """Read provider facts and bind each current-NN validation to its optimizer."""
    from backend.training import warehouse_v642_recovery_evidence as provider
    snapshot = final_checkpoint['provider']
    p = snapshot['payload']
    if profile == V3:
        audit_accelerated_provider(snapshot,r,recovery_plan,inputs)
    output = training.PROVIDER_ROOT / 'recovery' / recovery_plan['recovery_id']
    require(snapshot['payload_sha256'] == digest(p) and p['version'] == provider.VERSION
            and p['source_files'] == provider.source_files()
            and p['native_source_files'] == provider.native.source_files()
            and p['owner'] == r['owners']['regularized'] and p['recovery_claim'] == r['recovery_claim']
            and p['output'] == str(output) and p['variant'] == 'regularized'
            and p['config'] == recovery_plan['config'] and p['training_authority'] is False,
            'recovered provider checkpoint source, owner or claim differs')
    immutable(p['recovery_started'], inputs)
    require(p['recovery_started'] == pointer(output / 'recovery-started.json'),
            'recovery provider registration was relocated')
    header = bound(p['recovery_started'])
    start = bound(recovery_plan['checkpoints']['regularized'])['provider']
    original_output = training.PROVIDER_ROOT / recovery_plan['keys']['regularized']
    original_events = [pointer(path) for path in sorted((original_output / 'events').glob('*.json'))]
    original_files = {name: dict(path=name, sha256=sha)
                      for name, sha in recovery_plan['original_input_hashes'].items()
                      if Path(name).is_relative_to(original_output)}
    require(header['version'] == provider.VERSION and header['checkpoint'] == start
            and header['recovery_claim'] == r['recovery_claim']
            and header['source_files'] == p['source_files']
            and header['native_source_files'] == p['native_source_files']
            and header['original_events'] == original_events
            and header['original_files'] == original_files
            and p['events'][:len(header['original_events'])] == header['original_events'],
            'provider reconstruction lost its original committed event prefix')
    for name, ref in header['original_files'].items():
        require(ref['path'] == name
                and recovery_plan['original_input_hashes'].get(name) == ref['sha256'],
                'provider used an original file outside the frozen recovery inventory')
        immutable(ref, inputs)
    # Shared arrays occur in many validation envelopes. Validate every unique
    # byte object here and again at the final seal, without hashing the same
    # source episode thousands of times merely because its pointer repeats.
    seen = {ref['path']: ref['sha256'] for ref in inputs}
    def asset(ref):
        if ref['path'] in seen:
            require(seen[ref['path']] == ref['sha256'], 'conflicting provider asset content hashes')
        else:
            immutable(ref, inputs)
            seen[ref['path']] = ref['sha256']
    control = provider._control(p['events'], owner_ref=p['owner'],
                                sources=p['native_source_files'], config=training.owner.PPOConfig(),
                                check_assets=False)
    require(control == dict(installed=p['installed'], used_epochs=set(p['used_epochs']),
                            call_count=p['call_count'], latch=p['latch'], reason=p['reason'], programs=p['programs']),
            'provider latch or call history differs from its immutable events')
    validations = {}
    for ref in p['events']:
        immutable(ref, inputs)
        event = bound(ref)
        for item in event['asset_refs']:
            asset(item)
        if event['kind'] == 'validation':
            validations[event['epoch_id']] = (ref, event)
    expected_calls = list(range(start['payload']['call_count'], p['call_count']))
    require(len(p['audit_refs']) == len(expected_calls)
            and sorted((output / 'replay-audit').glob('*.json')) ==
                [Path(ref['path']) for ref in p['audit_refs']],
            'provider replay-audit calls are missing, duplicated or reordered')
    for index, (sequence, ref) in enumerate(zip(expected_calls, p['audit_refs'], strict=True)):
        immutable(ref, inputs)
        a = bound(ref)
        event_ref, event = validations[a['epoch_id']]
        require(ref == pointer(output / 'replay-audit' / f'{index:06d}.json')
                and a['version'] == provider.VERSION and a['recovery_started'] == p['recovery_started']
                and a['event'] == event_ref and a['sequence_call'] == event['sequence_call'] == sequence
                and a['target_actor'] == event['target_actor'] and a['target_package'] == event['target_package']
                and a['report_sha256'] == event['reports']
                and a['environment_resets'] == a['NN_updates'] == a['PPO_joint_steps'] == 0
                and a['training_authority'] is False,
                'provider replay-audit is not bound to its current-NN event and targets')
        target = bound(a['target_package'])
        immutable(a['target_actor'], inputs)
        immutable(a['target_package'], inputs)
        require(target['target_actor'] == a['target_actor'] and target['owner'] == p['owner'],
                'provider replay-audit target belongs to another NN or owner')
        for item in target['asset_refs']:
            asset(item)
        accounting = a['replay_accounting']
        if a['event'] in header['original_events']:
            require(a['mode'] == 'completed_native_validation_rebuilt_from_committed_facts'
                    and accounting['completed_native_target_facts_reused'] is True
                    and accounting['scalar_NN_forwards_recomputed'] == 0
                    and accounting['scalar_NN_forwards_reused'] == accounting['recorded_action_draws_replayed']
                    and accounting['original_memory_certificate_issued'] is False
                    and accounting['new_scientific_evaluation'] is False,
                    'completed native provider replay was mislabeled as new validation')
        else:
            require(a['mode'] in ('interrupted_native_targets_replayed', 'new_current_NN_validation')
                    and accounting['completed_native_target_facts_reused'] is False
                    and accounting['original_full_scalar_NN_target_replay_required'] is True,
                    'partial or new provider validation omitted full NN target replay')
    consumed = []
    for ref in r['updates']['regularized']:
        row = bound(ref)
        if row['end_joint_step'] <= 16000:
            continue
        for epoch in row['epochs']:
            for update in epoch['updates']:
                event_ref, event = validations[update['epoch_id']]
                require(event['step'] == row['end_joint_step'],
                        'provider current-NN epoch belongs to another optimizer batch')
                folder = Path(event.get('recovery_validation_folder',
                    str(Path(event_ref['path']).parent.parent / 'calls' / f'{event["sequence_call"]:06d}')))
                for role in ROLES:
                    evidence_ref = pointer(folder / (role + '-evidence.json'))
                    immutable(evidence_ref, inputs)
                    evidence = bound(evidence_ref)
                    require(evidence['current_target_role_sha256'] == update['role_sha256_before'][role]
                            and evidence['current_target_actor'] == event['target_actor']
                            and evidence['target_package'] == event['target_package'],
                            'program evidence was reused for a different current NN optimizer state')
                consumed.append(update['epoch_id'])
    require(consumed == [event['epoch_id'] for _, event in validations.values()],
            'provider validation epochs do not match the actual accepted optimizer history')


def audit_completion(result_ref, supervisor_exit_ref, *, reference_directory):
    """Recount exact histories; all input evidence is read without RNG or NN updates."""
    inputs = []
    result_path = immutable(result_ref, inputs)
    r = bound(result_ref); profile = recovery_profile(r.get('version'))
    audit_terminal(result_ref, supervisor_exit_ref, inputs,profile=profile)
    require(r.get('version') == profile.executor and r.get('completed') is True
            and r.get('numerical_training_completed') is True
            and r.get('pilot_opened') is False and r.get('model_acceptance_passed') is False
            and r.get('publishable') is False,
            'recovery is incomplete, unknown or incorrectly claims model acceptance')
    for key in ('original_plan', 'original_claim', 'original_descriptor',
                'original_controller_contract', 'recovery_plan', 'recovery_claim'):
        immutable(r[key], inputs)
    plan = training.owner.load_seed_plan(reference_directory, r['original_plan'])
    require(training.checked_contract(plan) == r['original_controller_contract'],
            'original controller source or seed plan changed')
    contract = bound(r['original_controller_contract'])
    claim = bound(r['original_claim'])
    descriptor = bound(r['original_descriptor'])
    original = Path(plan.ref['path']).parent / 'calibration'
    require(r['original_claim'] == pointer(training.owner.claim_path())
            and claim['plan'] == plan.ref and claim['descriptor'] == r['original_descriptor']
            and Path(r['original_descriptor']['path']) == original / 'descriptor.json'
            and descriptor['seed_plan'] == plan.ref and descriptor['per_variant_joint_step_limit'] == 20000
            and descriptor['reference_actor'] == plan.proposal.readiness.value['actor']
            and descriptor['source_files'] == training.owner.source_files()
            and claim['single_use_even_if_interrupted'] is True
            and claim['automatic_resume'] is False and claim['pilot_opened'] is False
            and not (original / 'completed.json').exists()
            and not (original / 'controller-completed.json').exists()
            and not (original / 'failed.json').exists(),
            'recovery replaced an old claim or reuses a terminal training attempt')
    folder = result_path.parent
    require(result_path == folder / 'result.json'
            and Path(supervisor_exit_ref['path']) == folder / 'supervisor-exit.json'
            and not (folder / 'failed.json').exists()
            and not (folder / 'supervisor-failed.json').exists()
            and r.get('original_files_preserved') is True
            and r.get('actual_original_process_exit_code_unknown') is True
            and r.get('per_variant_joint_steps') == {v: 20000 for v in VARIANTS},
            'recovery result location, original process status or final steps differ')
    completion = pointer(folder / 'completed.json')
    immutable(completion, inputs)
    require(bound(completion) == dict(result_sha256=result_ref['sha256']),
            'recovery completion seal does not bind this result')
    recovery_plan = audit_recovery_plan(r, folder=folder, plan=plan,
        descriptor=descriptor, original_claim=claim,
        reference_directory=reference_directory, inputs=inputs,profile=profile)
    require(r['original_source_files'] == contract['source_files'] == training.source_files(),
            'original recovery source contract differs')
    for name, sha in r['source_files'].items():
        require(file_sha(ROOT / name) == sha, 'recovery executor source changed')
        snapshot = (folder / 'source' / name if profile == V2 else Path(bound(r['source_contract'])['source_snapshots'][name]['path']))
        require(file_sha(snapshot) == sha and not snapshot.stat().st_mode & 0o222,
                'recovery executor source snapshot changed')
        inputs.append(dict(path=str(snapshot), sha256=sha))
    events = _journal(r['journals'], inputs)
    require([e['sequence'] for e in events] == list(range(len(events))),
            'recovered owner journal skipped, duplicated or reordered events')
    require(r['journals'] == [pointer(original / 'owner-journal.jsonl'), pointer(folder / 'owner-journal.jsonl')]
            and r['reset_journals'] == [pointer(original / 'actual-resets.jsonl'), pointer(folder / 'actual-resets.jsonl')],
            'original or continuation immutable journals were relocated')
    continuation = [json.loads(s) for s in (folder / 'owner-journal.jsonl').read_text().splitlines()]
    require(continuation and continuation[0]['kind'] == 'explicit_recovery_claimed'
            and continuation[0]['recovery_plan'] == r['recovery_plan']
            and continuation[0]['original_claim'] == r['original_claim']
            and continuation[-1]['kind'] == 'recovery_pair_steps_finished'
            and continuation[-1]['per_variant_joint_steps'] == {v: 20000 for v in VARIANTS},
            'recovery continuation lacks its authorized start or completion event')
    recovery_claim = bound(r['recovery_claim'])
    require(all(e['claim'] == r['original_claim'] and e['recovery_claim'] == r['recovery_claim']
                and _time(recovery_claim['started_utc']) <= _time(e['utc'])
                < _time(recovery_plan['expires_utc']) for e in continuation),
            'continuation event claim or finite-seed lease differs')
    restored = [e for e in continuation if e['kind'] == 'full_training_checkpoint_restored']
    require(len(restored) == 2 and {e['variant'] for e in restored} == set(VARIANTS)
            and all(e['checkpoint'] == recovery_plan['checkpoints'][e['variant']]
                    and e['step'] == STARTS[e['variant']]
                    and e['new_environment_resets'] == 0 for e in restored),
            'recovery did not restore the two declared safe checkpoints')
    resets = _journal(r['reset_journals'], inputs)
    require(len(resets) == sum(e['kind'] == 'actual_reset_bound' for e in events),
            'recovered reset hook and native owner event counts differ')
    starts = [e for e in events if e['kind'] == 'initialized']
    require(len(starts) == 2 and {e['variant'] for e in starts} == set(VARIANTS),
            'recovery reinitialized a scientific Actor or critic')
    starts = {e['variant']: e for e in starts}
    require(starts['raw']['roles'] == starts['regularized']['roles']
            and starts['raw']['critic_sha256'] == starts['regularized']['critic_sha256'],
            'paired initialization evidence differs')
    # Reconstruction is extra compute on already recorded frames, never new
    # observations or another accepted application of the original update.
    audit_replay_proofs(r.get('replay_proofs'), original=original, folder=folder, inputs=inputs)
    original_events = [json.loads(s) for s in (original / 'owner-journal.jsonl').read_text().splitlines()]
    original_resets = [json.loads(s) for s in (original / 'actual-resets.jsonl').read_text().splitlines()]
    train_inventory = bound(plan.proposal.value['train_case_inventory'])
    evidence_inventory = bound(plan.proposal.value['program_evidence_inventory'])
    from . import v642_calibration_recovery as recovery
    require(recovery.check_consumption(original_events, original_resets,
                                      train_inventory, evidence_inventory) == recovery_plan['consumption'],
            'recovery plan reset cursors differ from the original committed reset history')
    new_resets = [json.loads(s) for s in (folder / 'actual-resets.jsonl').read_text().splitlines()]
    final_cursors = audit_reset_chain(continuation, new_resets, inventory=train_inventory,
                                     initial=recovery_plan['consumption']['training_cursors'])
    audit_update_chain(r, original_events=original_events, continuation=continuation,
                       original=original, folder=folder, inputs=inputs)
    snapshots, reports = {}, {}
    for variant in VARIANTS:
        immutable(r['owners'][variant], inputs)
        immutable(r['program_sources'][variant], inputs)
        require(r['owners'][variant] == pointer(original / variant / 'owner.json')
                and r['program_sources'][variant] == pointer(original / variant / 'program-source/source.json'),
                'recovery replaced its original finite owner or source episodes')
        refs = r['checkpoints'][variant]
        require(len(refs) == len(STEPS), 'all five registered checkpoints are required')
        values = []
        for step, ref in zip(STEPS, refs, strict=True):
            location = (original if step <= STARTS[variant] else folder)
            checkpoint_folder = location / variant / 'checkpoints' / f'{step:06d}'
            require(ref == pointer(checkpoint_folder / 'owner-state.json'),
                    'recovery checkpoint replaced a fixed old prefix or relocated a new snapshot')
            immutable(ref, inputs)
            value = bound(ref)
            immutable(value['training_state'], inputs)
            core = value['core']
            require(value['owner'] == r['owners'][variant]
                    and value['claim'] == r['original_claim']
                    and value['source_files'] == descriptor['source_files']
                    and value['training_state'] == pointer(checkpoint_folder / 'state.pt')
                    and core['processed_joint_steps'] == value['reserved_joint_steps'] == step
                    and core['variant'] == variant and core['safe_batch_boundary'] is True
                    and core['failed'] is False
                    and value['refits'] == [s for s in (15000, 16800, 18600) if s <= step],
                    'recovery checkpoint identity, safety or refit steps differ')
            if location == folder:
                require(value['version'] == profile.carrier
                        and value['recovery_claim'] == r['recovery_claim']
                        and value['recovery_source_files'] == r['source_files'],
                        'new checkpoint does not bind the actual recovery executor')
            values.append(value)
        warmup, final = values[0], values[-1]
        require(warmup['core']['actor_updates'] == 0
                and warmup['core']['actor_role_sha256'] == starts[variant]['roles']
                and final['updates'] == r['updates'][variant],
                'warmup or final optimizer history differs')
        require(final['reset_cursors'] == final_cursors[variant]
                and final['program_source_cursor'] == 1650,
                'final checkpoint finite reset or program-evidence cursor differs')
        for earlier in values[:-1]:
            require(earlier['updates'] == final['updates'][:len(earlier['updates'])],
                    'recovery discarded or replaced a committed optimizer history')
        for ref in final['updates']:
            immutable(ref, inputs)
        original_updates = recovery_plan['original_updates'][variant]
        for ref in final['updates'][len(original_updates):]:
            row = bound(ref)
            require(ref == pointer(folder / variant / 'rollouts'
                                   / f'{row["start_joint_step"]:06d}' / 'update.json'),
                    'new optimizer receipt is outside the authorized continuation')
        reports[variant] = audit_updates(final['updates'], variant=variant,
                                        expected_joint_steps=20000, core=final['core'])
        audit_NN_hash_chain(final['updates'], values)
        refits = [e for e in events if e['kind'] == 'scheduled_refit_bound' and e['variant'] == variant]
        require([e['step'] for e in refits] == [15000, 16800, 18600]
                and [e['binding'] for e in refits] == r['scheduled_refits'][variant],
                'recovery omitted or duplicated a scheduled extraction')
        for e in refits:
            immutable(e['binding'], inputs)
            binding = bound(e['binding'])
            require(binding['owner'] == r['owners'][variant]
                    and binding['observed_source'] == r['program_sources'][variant],
                    'scheduled program refit belongs to a different owner or episode pool')
        snapshots[variant] = (warmup, final)
    require(snapshots['raw'][0]['core']['critic_sha256'] == snapshots['regularized'][0]['core']['critic_sha256'],
            'paired critic-only warmup checkpoint differs')
    require(r['accepted_program_updates'] ==
            {v: reports[v]['accepted_program_updates'] for v in VARIANTS},
            'recovery completion counters differ from accepted optimizer receipts')
    audit_provider(r, recovery_plan=recovery_plan,
                   final_checkpoint=snapshots['regularized'][-1], inputs=inputs,profile=profile)
    unique = {}
    for ref in inputs:
        require(ref['path'] not in unique or unique[ref['path']] == ref,
                'recovery audit encountered conflicting content addresses')
        unique[ref['path']] = ref
    inputs = list(unique.values())
    for ref in inputs:
        immutable(ref, [])
    return dict(result=r, snapshots=snapshots, receipt_audit=reports,
                input_asset_refs=inputs, actual_reset_count=len(resets))


def run(*,reference_directory, result_ref, supervisor_exit_ref, output):
    checked = audit_completion(result_ref, supervisor_exit_ref, reference_directory=reference_directory)
    sources = source_files()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exports = {}
    for variant, (warmup, final) in checked['snapshots'].items():
        exports[variant] = {}
        for label, value, step in (('warmup', warmup, 4000), ('final', final, 20000)):
            export = export_checkpoint_actor(value['training_state'], output / f'{variant}-{label}.actor.npz',
                variant=variant, candidate_id=training.owner.seeds.partitions.admission.v.CANDIDATE,
                expected_joint_steps=step)
            require(export['role_sha256'] == value['core']['actor_role_sha256'],
                    'exported recovered Actor tensors differ from the native checkpoint')
            exports[variant][label] = export
    for role in ROLES:
        anchor = exports['raw']['warmup']['role_sha256'][role]
        require(exports['regularized']['warmup']['role_sha256'][role] == anchor
                and len({anchor, exports['raw']['final']['role_sha256'][role],
                         exports['regularized']['final']['role_sha256'][role]}) == 3,
                'recovered reference/raw/regularized role weights are not independently bound')
    require(source_files() == sources, 'recovery result audit source changed during export')
    r = dict(version=recovery_profile(checked['result']['version']).report, recovery_completed=result_ref, supervisor_exit=supervisor_exit_ref,
        reference_directory=str(Path(reference_directory).resolve()), audit_source_files=sources,
        input_asset_refs=checked['input_asset_refs'], receipt_audit=checked['receipt_audit'], exports=exports,
        actual_reset_count=checked['actual_reset_count'], numerical_training_completed=True,
        independent_quality_evaluation_required=True,
        independent_current_final_NN_program_evaluation_required=True,
        original_process_terminal_exit_observed=False, recovery_process_terminal_exit_observed=True,
        new_RNG_constructors=0, environment_resets=0, optimizer_steps=0,
        model_acceptance_passed=False, publishable=False, pilot_opened=False)
    write_new(output / 'report.json', r)
    (output / 'report.json').chmod(0o444)
    return pointer(output / 'report.json')


def read_assets(report_ref):
    """Versioned downstream reader; never silently accepts a recovery flag."""
    r = bound(report_ref)
    profile=report_profile(r['version'])
    require(r['version'] == profile.report and r['audit_source_files'] == source_files()
            and r['numerical_training_completed'] is True
            and r['model_acceptance_passed'] is r['publishable'] is r['pilot_opened'] is False,
            'recovered calibration assets are missing, changed or incorrectly published')
    checked = audit_completion(r['recovery_completed'], r['supervisor_exit'],
                               reference_directory=r['reference_directory'])
    require(checked['receipt_audit'] == r['receipt_audit']
            and checked['input_asset_refs'] == r['input_asset_refs']
            and checked['actual_reset_count'] == r['actual_reset_count'],
            'recovered calibration audit no longer reproduces')
    result = checked['result']
    actors = {'reference': bound(result['original_descriptor'])['reference_actor']}
    roles = {}
    inputs = [report_ref, *r['input_asset_refs']]
    for variant in VARIANTS:
        final = checked['snapshots'][variant][-1]
        export = r['exports'][variant]['final']
        require(export['checkpoint'] == final['training_state']
                and export['role_sha256'] == final['core']['actor_role_sha256'],
                'recovered final frozen export differs from its training checkpoint')
        actors[variant] = export['actor']
        roles[variant] = export['role_sha256']
        immutable(export['actor'], inputs)
        FrozenWarehouseV6Actor.load(export['actor']['path'], expected_sha256=export['actor']['sha256'],
                                   expected_role_sha256=roles[variant])
    anchor = FrozenWarehouseV6Actor.load(actors['reference']['path'], expected_sha256=actors['reference']['sha256'])
    roles['reference'] = deepcopy(anchor.role_sha256)
    inputs.append(actors['reference'])
    require(all(len({roles[v][role] for v in actors}) == 3 for role in ROLES),
            'reference/raw/regularized actual role weights must be different')
    return dict(report=report_ref, actors=actors, actor_role_sha256=roles, inputs=inputs)


def finalization_context(report_ref, reference_directory):
    r = bound(report_ref)
    checked = audit_completion(r['recovery_completed'], r['supervisor_exit'],
                               reference_directory=reference_directory)
    result = checked['result']
    plan = training.owner.load_seed_plan(reference_directory, result['original_plan'])
    events = _journal(result['journals'], [])
    return dict(plan=plan, events=events, journal_refs=result['journals'],
                owners=result['owners'], program_sources=result['program_sources'])


def audit_accelerated_provider(snapshot,r,plan,inputs):
    from backend.training import warehouse_v642_accelerated_recovery_evidence as adapter
    from backend.training import warehouse_v642_canonical_relabel as producer
    from backend.training import warehouse_v642_canonical_batch as batch
    import numpy as np
    p=snapshot['payload'];wrapper=p.get('accelerated_wrapper')
    require(type(wrapper) is dict and wrapper['version']==adapter.VERSION
        and wrapper['source_files']==adapter.source_files()
        and wrapper['prefix_manifests']==plan['prefix_manifest_refs']
        and wrapper['every_distinct_bound_file_rehashed'] is wrapper['no_stat_cache']
            is wrapper['current_NN_metrics_recomputed'] is True,'accelerated provider wrapper changed')
    immutable(wrapper['acceleration_started'],inputs);header=bound(wrapper['acceleration_started'])
    require(header['version']==adapter.VERSION and header['mode']=='original_prefix'
        and header['source_files']==adapter.source_files() and header['recovery_claim']==r['recovery_claim']
        and header['prefix_manifests']==plan['prefix_manifest_refs']
        and header['canonical_producer_enabled'] is True
        and header['NN_or_optimizer_method_changed'] is header['training_authority'] is False,
        'accelerated evidence registration differs from its actual source')
    for event_ref in p['events']:
        event=bound(event_ref)
        if event['kind']!='validation':continue
        target=bound(event['target_package']);ref=target.get('canonical_producer_provenance')
        if ref is None:continue
        require(event.get('canonical_producer_provenance')==ref and ref in event['asset_refs'],
            'canonical producer omitted from actual native event')
        immutable(ref,inputs);a=bound(ref)
        require(a['version']==producer.VERSION and a['source_files']==producer.source_files()
            and a['source_sha256']==digest(a['source_files'])
            and a['source_guard']==dict(owner=p['owner'],claim=r['recovery_claim'],source_files=r['source_files'],executor=V3.executor)
            and a['target_actor']==event['target_actor'] and a['epoch_id']==event['epoch_id']
            and a['sequence']==event['sequence_call'] and a['owner']==p['owner']
            and a['observed_source']==target['observed_source']
            and a['observed_audit_sha256']==target['observed_audit_sha256']
            and a['observed_data_sha256']==target['observed_data_sha256']
            and a['full_native_scalar_audit_required_before_feedback'] is True
            and a['native_scalar_audit_performed'] is a['replayed_target_certificate_minted']
                is a['independent_validation'] is a['training_authority'] is a['pilot_opened'] is False,
            'canonical appendix falsely claims audit authority or changes the actual current NN')
        actor=FrozenWarehouseV6Actor.load(a['target_actor']['path'],expected_sha256=a['target_actor']['sha256'])
        require(actor.role_sha256==a['actor_role_sha256'],'canonical appendix NN role hashes differ')
        for role in ROLES:
            manifest=bound(target['pools'][role])
            for split,part in manifest['parts'].items():
                immutable(part['arrays'],inputs)
                with np.load(part['arrays']['path'],allow_pickle=False) as arrays:
                    require({key:batch.array_sha(arrays[key]) for key in ('x','m','p','sampled')}==a['outputs'][role][split],
                        'canonical numeric outputs differ from their source-bound appendix')
                    witness=a['witnesses'][role][split]
                    require(witness['indices']==producer.witness_indices(len(arrays['x']))
                        and witness['all_witness_values_bitwise_equal'] is True and witness['full_scalar_audit'] is False
                        and producer._float_contract(arrays['p'][witness['indices']])==witness['probabilities'],
                        'canonical scalar witness scope or stored probabilities differ')
