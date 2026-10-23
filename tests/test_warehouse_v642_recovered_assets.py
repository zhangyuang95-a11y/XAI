"""Fail closed before any scientific model load or export."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from experiments.warehouse import v642_recovered_assets as audit


def put(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    return audit.pointer(path)


@pytest.mark.parametrize('change', ['nonzero', 'unobserved', 'other_result', 'partial', 'published'])
def test_no_unobserved_or_partial_recovery_exports(tmp_path, monkeypatch, change):
    result = dict(version=audit.RECOVERY_VERSION, completed=True,
                  numerical_training_completed=True, pilot_opened=False,
                  model_acceptance_passed=False, publishable=False)
    if change == 'partial':
        result['completed'] = False
    if change == 'published':
        result['model_acceptance_passed'] = True
    ref = put(tmp_path / 'result.json', result)
    terminal = dict(exit_code=0, result=ref, observed_by='exec_tool_terminal_result')
    if change == 'nonzero': terminal['exit_code'] = 1
    if change == 'unobserved': terminal['observed_by'] = 'PID_disappeared'
    if change == 'other_result': terminal['result'] = dict(path='elsewhere', sha256='0' * 64)
    exit_ref = put(tmp_path / 'supervisor-exit.json', terminal)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid recovery reached scientific source/model loading')
    monkeypatch.setattr(audit.training.owner, 'load_seed_plan', forbidden)
    monkeypatch.setattr(audit, 'export_checkpoint_actor', forbidden)
    monkeypatch.setattr(audit, 'source_files', forbidden)
    monkeypatch.setattr(audit, 'executor_sources', forbidden)
    with pytest.raises(RuntimeError):
        audit.run(reference_directory=tmp_path / 'reference', result_ref=ref,
                  supervisor_exit_ref=exit_ref, output=tmp_path / 'exports')
    assert not (tmp_path / 'exports').exists()


def test_writable_recovery_evidence_rejected(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='writable'):
        audit.immutable(audit.pointer(path), [])


def test_journal_preserves_sequence_boundary(tmp_path):
    a = put(tmp_path / 'old.jsonl', dict(sequence=0, kind='original'))
    b = put(tmp_path / 'new.jsonl', dict(sequence=1, kind='recovery'))
    assert [row['sequence'] for row in audit._journal([a, b], [])] == [0, 1]
    with pytest.raises(RuntimeError, match='both original'):
        audit._journal([a], [])


def replace(path, value):
    """Change synthetic test metadata only; never a scientific asset."""
    path.chmod(0o644)
    return put(path, value)


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    """Pure files and dictionaries: no NN, optimizer, RNG or environment."""
    base = tmp_path / 'base'
    original = base / 'calibration'
    folder = base / 'recovery' / 'fixture'
    folder.mkdir(parents=True)
    plan_ref = put(base / 'plan.json', {'synthetic': True})
    for variant in audit.VARIANTS:
        owner = original / variant / 'owner.json'
        owner.parent.mkdir(parents=True)
        put(owner, {'variant': variant})
        snapshot = original / variant / 'checkpoints' / f'{audit.STARTS[variant]:06d}' / 'owner-state.json'
        snapshot.parent.mkdir(parents=True)
        put(snapshot, {'synthetic': 'safe checkpoint'})
    for name in ('owner-journal.jsonl', 'actual-resets.jsonl'):
        put(original / name, {})
    owners = {v: audit.pointer(original / v / 'owner.json') for v in audit.VARIANTS}
    keys = {v: audit.digest(ref) for v, ref in owners.items()}
    extraction, provider, targets = (tmp_path / name for name in ('extraction', 'provider', 'targets'))
    monkeypatch.setattr(audit.training, 'EXTRACTION_ROOT', extraction)
    monkeypatch.setattr(audit.training, 'PROVIDER_ROOT', provider)
    monkeypatch.setattr(audit.training.owner.observed, 'TARGET_ROOT', targets)
    for path in (extraction / keys['raw'], extraction / keys['regularized'], provider / keys['regularized']):
        path.mkdir(parents=True)
    raw_update = original / 'raw/rollouts/017824/update.json'
    raw_update.parent.mkdir(parents=True)
    put(raw_update, {'start_joint_step': 17824, 'end_joint_step': 18600})
    cached = []
    for start, end in ((16800, 17824), (17824, 18600)):
        location = original / 'regularized/rollouts' / f'{start:06d}'
        location.mkdir(parents=True)
        trace = put(location / 'trace.json', dict(start_joint_step=start, end_joint_step=end))
        checked = put(location / 'audit.json', {'synthetic': 'physical replay'})
        update = put(location / 'update.json', dict(start_joint_step=start, end_joint_step=end)) if start == 16800 else None
        cached.append(dict(trace=trace, audit=checked, update=update))
    originals = {key: put(base / (key + '.json'), {'synthetic': key})
                 for key in ('original_claim', 'original_descriptor', 'original_controller_contract')}
    clearance = put(base / 'clearance.json', dict(issued_utc='2000-01-01T00:00:00+00:00',
                                               expires_utc='2000-01-02T00:00:00+00:00'))
    sources = {'synthetic-executor.py': 'a' * 64}
    monkeypatch.setattr(audit, 'executor_sources', lambda: deepcopy(sources))
    checkpoints = {v: audit.pointer(original / v / 'checkpoints' / f'{audit.STARTS[v]:06d}' / 'owner-state.json')
                   for v in audit.VARIANTS}
    updates = {v: [audit.pointer(p) for p in sorted((original / v / 'rollouts').glob('*/update.json'))]
               for v in audit.VARIANTS}
    roots = [original, extraction / keys['raw'], extraction / keys['regularized'], provider / keys['regularized']]
    original_input_hashes = {str(p): audit.file_sha(p) for root in roots for p in sorted(root.rglob('*')) if p.is_file()}
    repair = dict(version='warehouse-v642-zero-work-engineering-repair.v1', synthetic_fixture=True,
                  observed_failed_worker_exit_code=1, accepted_optimizer_updates=0,
                  new_environment_resets=0, new_provider_calls=0)
    monkeypatch.setattr(audit, 'engineering_repair', lambda base, inputs: deepcopy(repair))
    r = dict(original_plan=plan_ref, **originals, original_source_files={'original.py': 'b' * 64},
             owners=owners, updates=deepcopy(updates), source_files=deepcopy(sources), repair_of=deepcopy(repair))
    p = dict(version=audit.RECOVERY_VERSION, recovery_id='fixture', authorization='继续，完成验收',
             reference_directory=str(tmp_path / 'reference'), config=asdict(audit.training.owner.PPOConfig()),
             maximum_joint_steps_per_variant=20000, restores_training_state_not_new_initialization=True,
             cached_frames_are_numerical_replay_not_new_samples=True, original_claim_retained=True,
             original_files_read_only=True, automatic_retry=False, scientific_acceptance=False, pilot_opened=False,
             **{k:deepcopy(r[k]) for k in ('original_plan', 'original_claim', 'original_descriptor',
                                         'original_controller_contract', 'original_source_files', 'owners')},
             source_files=deepcopy(sources), original_calibration=str(original), keys=keys, repair_of=deepcopy(repair),
             original_roots=list(map(str, roots)), original_input_hashes=original_input_hashes,
             checkpoints=checkpoints, original_updates=updates, cached_rollouts={'raw': [], 'regularized': cached},
             journals=[audit.pointer(original / 'owner-journal.jsonl')],
             reset_journals=[audit.pointer(original / 'actual-resets.jsonl')], expires_utc='2000-01-02T00:00:00+00:00')
    r['recovery_plan'] = put(folder / 'plan.json', p)
    claim = dict(version=audit.RECOVERY_VERSION, recovery_plan=r['recovery_plan'], original_claim=r['original_claim'],
                 authorization=p['authorization'], owner_pid=123, single_use_even_if_interrupted=True,
                 source_files=deepcopy(sources), maximum_joint_steps_per_variant=20000,
                 started_utc='2000-01-01T12:00:00+00:00', expires_utc=p['expires_utc'], pilot_opened=False,
                 repair_of=deepcopy(repair))
    r['recovery_claim'] = put(base / audit.RECOVERY_CLAIM_NAME, claim)
    return dict(r=r, p=p, claim=claim, folder=folder, original=original,
                plan=SimpleNamespace(ref=plan_ref), descriptor={'clearance': clearance},
                original_claim={'clearance': clearance}, reference_directory=tmp_path / 'reference', sources=sources)


def check_plan(f):
    return audit.audit_recovery_plan(f['r'], folder=f['folder'], plan=f['plan'],
                                    descriptor=f['descriptor'], original_claim=f['original_claim'],
                                    reference_directory=f['reference_directory'], inputs=[])


def reseal_plan(f):
    f['r']['recovery_plan'] = replace(f['folder'] / 'plan.json', f['p'])
    f['claim']['recovery_plan'] = f['r']['recovery_plan']
    f['r']['recovery_claim'] = replace(Path(f['r']['recovery_claim']['path']), f['claim'])


@pytest.mark.parametrize('side', ['r', 'p', 'claim'])
def test_v2_requires_same_zero_work_repair_evidence_everywhere(plan_fixture, side):
    f = plan_fixture
    f[side]['repair_of']['accepted_optimizer_updates'] = 1
    if side != 'r':
        reseal_plan(f)
    with pytest.raises(RuntimeError, match='zero-work failure evidence'):
        check_plan(f)


def test_offline_plan_accepts_historical_lease_without_current_clock_or_pid(plan_fixture, monkeypatch):
    f = plan_fixture
    # The completed lease is deliberately in 2000. A post-run read must validate
    # the actual start time rather than refuse a valid archive after expiration.
    assert check_plan(f) == f['p']


@pytest.mark.parametrize('change', ['empty_sources', 'extra_sources', 'original_pointer', 'original_bytes',
                                    'extra_original_file', 'prefix', 'checkpoint', 'cached_trace',
                                    'claim_location', 'claim_reusable', 'claim_source', 'claim_authorization',
                                    'start_before', 'start_at_expiry', 'expiry_changed', 'naive_time'])
def test_recovery_plan_rejects_unbound_history_and_claim(plan_fixture, change):
    f = plan_fixture
    if change == 'empty_sources':
        f['p']['source_files'] = f['r']['source_files'] = {}; reseal_plan(f)
    elif change == 'extra_sources':
        f['p']['source_files']['unregistered.py'] = '0' * 64
        f['r']['source_files'] = deepcopy(f['p']['source_files']); reseal_plan(f)
    elif change == 'original_pointer':
        f['p']['original_claim'] = {'path': '/elsewhere', 'sha256': '0' * 64}; reseal_plan(f)
    elif change == 'original_bytes':
        replace(f['original'] / 'raw/owner.json', {'variant': 'changed'})
    elif change == 'extra_original_file':
        put(f['original'] / 'additional.json', {})
    elif change == 'prefix':
        f['r']['updates']['regularized'] = []
    elif change == 'checkpoint':
        f['p']['checkpoints']['raw'] = f['p']['checkpoints']['regularized']; reseal_plan(f)
    elif change == 'cached_trace':
        f['p']['cached_rollouts']['regularized'] = []; reseal_plan(f)
    elif change == 'claim_location':
        f['r']['recovery_claim'] = put(f['folder'] / 'replacement-claim.json', f['claim'])
    else:
        if change == 'claim_reusable': f['claim']['single_use_even_if_interrupted'] = False
        elif change == 'claim_source': f['claim']['source_files'] = {}
        elif change == 'claim_authorization': f['claim']['authorization'] = 'automatic_resume'
        elif change == 'start_before': f['claim']['started_utc'] = '1999-12-31T23:59:59+00:00'
        elif change == 'start_at_expiry': f['claim']['started_utc'] = f['p']['expires_utc']
        elif change == 'expiry_changed': f['claim']['expires_utc'] = '2000-01-03T00:00:00+00:00'
        elif change == 'naive_time': f['claim']['started_utc'] = '2000-01-01T12:00:00'
        f['r']['recovery_claim'] = replace(Path(f['r']['recovery_claim']['path']), f['claim'])
    with pytest.raises((RuntimeError, FileNotFoundError)):
        check_plan(f)


@pytest.mark.parametrize('change', ['self_proof', 'wrong_original', 'changed_update', 'missing', 'duplicate', 'new_samples',
                                    'signed_zero', 'integer_type'])
def test_reconstruction_proof_cannot_skip_or_replace_committed_update(tmp_path, change):
    original, folder = tmp_path / 'original', tmp_path / 'recovery'
    suffix = Path('regularized/rollouts/016800/update.json')
    for base in (original, folder):
        (base / suffix).parent.mkdir(parents=True)
    row = {'start_joint_step': 16800, 'end_joint_step': 17824, 'synthetic': True, 'value': 0.0}
    left, right = put(original / suffix, row), put(folder / suffix, row)
    proof = dict(original_update=left, recomputed_update=right, matched=True,
                 exact_recursive_value_equality=True, new_environment_resets=0,
                 added_scientific_joint_steps=0, replayed_physical_frames=1024)
    audit.audit_replay_proofs([proof], original=original, folder=folder, inputs=[])
    proofs = [proof]
    if change == 'self_proof': proof['recomputed_update'] = left
    elif change == 'wrong_original': proof['original_update'] = right
    elif change == 'changed_update':
        proof['recomputed_update'] = replace(folder / suffix, {**row, 'synthetic': 'different'})
    elif change == 'missing': proofs = []
    elif change == 'duplicate': proofs = [proof, deepcopy(proof)]
    elif change == 'new_samples': proof['added_scientific_joint_steps'] = 1024
    elif change == 'signed_zero':
        proof['recomputed_update'] = replace(folder / suffix, {**row, 'value': -0.0})
    elif change == 'integer_type':
        proof['recomputed_update'] = replace(folder / suffix, {**row, 'value': 0})
    with pytest.raises(RuntimeError):
        audit.audit_replay_proofs(proofs, original=original, folder=folder, inputs=[])


def test_executor_source_inventory_may_not_omit_dependencies(monkeypatch):
    from experiments.warehouse import v642_calibration_recovery as recovery
    monkeypatch.setattr(recovery, 'NEW_SOURCE_NAMES', ())
    with pytest.raises(RuntimeError, match='source inventory'):
        audit.executor_sources()


@pytest.fixture
def supervisor_fixture(tmp_path, monkeypatch):
    """Synthetic durable OS-wait facts; no subprocess is created here."""
    root, folder = tmp_path / 'root', tmp_path / 'base/recovery/fixture'
    source = root / 'scripts/resume_warehouse_v642_calibration.py'
    source.parent.mkdir(parents=True)
    source.write_text('# synthetic supervisor source, never executed\n')
    monkeypatch.setattr(audit, 'ROOT', root)
    snapshot = folder / 'source/scripts/resume_warehouse_v642_calibration.py'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes()); snapshot.chmod(0o444)
    plan = put(folder / 'plan.json', {'synthetic': 'plan'})
    claim = put(folder / 'claim.json', {'owner_pid': 12, 'started_utc': '2000-01-01T12:00:01+00:00'})
    result = put(folder / 'result.json', dict(recovery_plan=plan, recovery_claim=claim))
    completed = put(folder / 'completed.json', dict(result_sha256=result['sha256']))
    log = put(folder / 'run.log', {'synthetic': 'worker log'})
    source_ref = audit.pointer(source)
    command = [str(Path(sys.executable).resolve()), str(source), '--run', '--recovery-plan', plan['path']]
    started = dict(version='warehouse-v642-calibration-recovery-supervisor.v1',
                   observed_by='source_bound_subprocess_launch', recovery_plan=plan, command=command,
                   supervisor_pid=11, started_utc='2000-01-01T12:00:00+00:00',
                   supervisor_source=source_ref, log_path=log['path'])
    started_ref = put(folder / 'supervisor-started.json', started)
    exit_value = dict(version=started['version'], observed_by='source_bound_subprocess_wait', recovery_plan=plan,
                      started=started_ref, command=command, worker_pid=12, supervisor_pid=11,
                      started_utc=started['started_utc'], finished_utc='2000-01-01T12:00:05+00:00',
                      supervisor_source=source_ref, exit_code=0, log=log, result=result, completed=completed)
    exit_ref = put(folder / 'supervisor-exit.json', exit_value)
    return dict(folder=folder, result=result, claim=claim, exit=exit_value, exit_ref=exit_ref,
                started=started, started_ref=started_ref)


def test_registered_subprocess_wait_is_distinct_from_original_process_exit(supervisor_fixture):
    f = supervisor_fixture
    assert audit.audit_terminal(f['result'], f['exit_ref'], [])['observed_by'] == 'source_bound_subprocess_wait'


@pytest.mark.parametrize('change', ['argv', 'source', 'worker', 'supervisor', 'time', 'unobserved',
                                    'binding_error', 'seal', 'log', 'launch', 'nonzero', 'boolean_code'])
def test_registered_wait_cannot_be_forged_by_files_or_pid(supervisor_fixture, change):
    f = supervisor_fixture; e = f['exit']
    if change == 'argv': e['command'] = [sys.executable, '--run']
    elif change == 'source': e['supervisor_source'] = {'path': '/unregistered.py', 'sha256': '0' * 64}
    elif change == 'worker': e['worker_pid'] = 99
    elif change == 'supervisor': e['supervisor_pid'] = e['worker_pid']
    elif change == 'time': e['finished_utc'] = '2000-01-01T11:59:59+00:00'
    elif change == 'unobserved': e['observed_by'] = 'PID_disappeared'
    elif change == 'binding_error': e['result_binding_error'] = 'exit0 without valid scientific result'
    elif change == 'seal':
        e['completed'] = replace(f['folder'] / 'completed.json', {'result_sha256': '0' * 64})
    elif change == 'log':
        e['log'] = put(f['folder'] / 'other.log', {'synthetic': 'different log'})
    elif change == 'launch':
        f['started']['observed_by'] = 'file_already_exists'
        e['started'] = replace(f['folder'] / 'supervisor-started.json', f['started'])
    elif change == 'nonzero': e['exit_code'] = -9
    elif change == 'boolean_code': e['exit_code'] = False
    ref = replace(f['folder'] / 'supervisor-exit.json', e)
    with pytest.raises(RuntimeError):
        audit.audit_terminal(f['result'], ref, [])


@pytest.mark.parametrize('change', ['before_hash', 'checkpoint_hash'])
def test_optimizer_role_hashes_must_form_continuous_checkpoint_chain(tmp_path, change):
    roles = lambda value: {role: value * 64 for role in audit.ROLES}
    checkpoints = [dict(core=dict(processed_joint_steps=step, actor_role_sha256=roles(value)))
                   for step, value in ((4000, 'a'), (15000, 'b'), (20000, 'c'))]
    refs = []
    for start, end, before, after in ((4000, 15000, 'a', 'b'), (15000, 20000, 'b', 'c')):
        refs.append(put(tmp_path / f'{start}.json', dict(start_joint_step=start, end_joint_step=end,
            epochs=[dict(updates=[dict(actor_updated=True, role_sha256_before=roles(before), role_sha256_after=roles(after))])])) )
    audit.audit_NN_hash_chain(refs, checkpoints)
    if change == 'before_hash':
        row = audit.bound(refs[-1]); row['epochs'][0]['updates'][0]['role_sha256_before'] = roles('d')
        refs[-1] = replace(Path(refs[-1]['path']), row)
    else:
        checkpoints[-1]['core']['actor_role_sha256'] = roles('d')
    with pytest.raises(RuntimeError, match='NN'):
        audit.audit_NN_hash_chain(refs, checkpoints)


@pytest.mark.parametrize('change', ['seed', 'scope', 'missing_predebit', 'uncommitted', 'skip_case'])
def test_recovery_resets_use_only_remaining_predebited_finite_cases(change):
    case = {'slot': 0, 'scenario_seed': 123, 'episode_id': 1}
    later = {'slot': 0, 'scenario_seed': 456, 'episode_id': 2}
    inventory = {'cases': [case, later]}
    initial = {v: [0] * 8 for v in audit.VARIANTS}
    pre = dict(kind='environment_constructor_and_reset_predebit', variant='regularized', scope='train', case=case)
    reset = {'seed': 123}
    commit = dict(kind='actual_reset_bound', variant='regularized', scope='train', reset_hook_index=0, reset=reset)
    events = [pre, commit]
    cursors = audit.audit_reset_chain(events, [reset], inventory=inventory, initial=initial)
    assert cursors['regularized'][0] == 1 and initial['regularized'][0] == 0
    if change == 'seed': commit['reset'] = {'seed': 999}
    elif change == 'scope': pre['scope'] = 'program_evidence'
    elif change == 'missing_predebit': events = [commit]
    elif change == 'uncommitted': events = [pre]
    elif change == 'skip_case': pre['case'] = later
    with pytest.raises(RuntimeError):
        audit.audit_reset_chain(events, [reset], inventory=inventory, initial=initial)
