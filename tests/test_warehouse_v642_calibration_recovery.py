"""Recovery safety and orchestration with synthetic records, no real run."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest

from experiments.warehouse import v642_calibration_recovery as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    return r.pointer(path)


def test_unauthorized_prepare_never_reads_inputs_or_constructs_models(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'inspect_original', lambda **kw: pytest.fail('no authorization must not inspect inputs'))
    with pytest.raises(RuntimeError, match='human continuation'):
        r.prepare(reference_directory=tmp_path, plan_ref=None, recovery_id='no', authorization='implicit')
    assert not list(tmp_path.iterdir())


def test_real_A_B_closed_gate_precedes_original_plan_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(r.old.seeds.partitions.admission.v, 'OUTPUT', tmp_path / 'missing')
    monkeypatch.setattr(r.old, 'bound', lambda *a: pytest.fail('closed A/B cannot read a plan'))
    monkeypatch.setattr(r, 'load_checkpoint', lambda *a, **kw: pytest.fail('closed A/B cannot construct a model'))
    with pytest.raises(RuntimeError, match='complete reaped'):
        r.inspect_original(reference_directory=tmp_path, plan_ref=None)
    assert not list(tmp_path.iterdir())


def test_same_original_pid_is_live_and_invalid_pid_is_refused():
    assert r.pid_exists(os.getpid())
    with pytest.raises(RuntimeError, match='PID is invalid'):
        r.pid_exists(0)


def test_journal_partial_line_refuses_ambiguous_interruption(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_bytes(b'{"sequence":0}')
    with pytest.raises(RuntimeError, match='partial line'):
        r.json_lines(path)


def test_tree_rejects_symlinks_and_hash_rechecks_missing_or_mutated_input(tmp_path):
    folder = tmp_path / 'original'
    folder.mkdir()
    file = folder / 'evidence.json'
    file.write_text('{}')
    values = r.checked_tree([folder])
    r.verify_files(values)
    file.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='original recovery input changed'):
        r.verify_files(values)
    (folder / 'link').symlink_to(file)
    with pytest.raises(RuntimeError, match='symlink'):
        r.checked_tree([folder])


def test_exact_update_comparison_rejects_value_type_and_deep_field_changes():
    value = dict(accepted=True, score=1.0, epochs=[dict(gradient=0.001, count=1)])
    assert r.exact_value_equal(value, deepcopy(value))
    assert not r.exact_value_equal(0.0, -0.0)
    for changed in (dict(accepted=1, score=1.0, epochs=value['epochs']),
                    dict(accepted=True, score=1, epochs=value['epochs']),
                    dict(accepted=True, score=1.0, epochs=[dict(gradient=0.0010000001, count=1)])):
        assert not r.exact_value_equal(value, changed)


def reset_history():
    case = dict(episode_id='train-0', slot=0, scenario_seed=101)
    reset = dict(seed=101, episode_id=1)
    events = [dict(sequence=0, kind='environment_constructor_and_reset_predebit', variant='raw', scope='train', case=case),
              dict(sequence=1, kind='actual_reset_bound', variant='raw', scope='train', reset_hook_index=0, reset=reset)]
    return events, [reset], dict(cases=[case]), dict(cases=[])


def test_reset_consumption_uses_real_hook_and_cannot_reuse_or_skip_case():
    events, resets, training, evidence = reset_history()
    value = r.check_consumption(events, resets, training, evidence)
    assert value['actual_reset_count'] == 1
    assert value['training_cursors']['raw'] == [1, 0, 0, 0, 0, 0, 0, 0]
    for mutate in ('hook', 'variant', 'partial', 'sequence'):
        bad = deepcopy(events)
        if mutate == 'hook': bad[-1]['reset']['seed'] = 102
        elif mutate == 'variant': bad[-1]['variant'] = 'regularized'
        elif mutate == 'partial': bad = bad[:-1]
        else: bad[-1]['sequence'] = 2
        with pytest.raises(RuntimeError):
            r.check_consumption(bad, resets, training, evidence)


class State:
    def __init__(self, step): self.joint_steps = step
    def advance_joint_steps(self, count): self.joint_steps += count


def test_cached_physics_replay_uses_existing_resets_without_constructor(tmp_path, monkeypatch):
    case = dict(episode_id='existing-episode', slot=0)
    trace = write(tmp_path / 'trace.json', dict(start_joint_step=16800, end_joint_step=16808,
        physics=dict(resets={'existing-episode': {'path': 'already-recorded', 'sha256': 'x'}}),
        events=[dict(case=case) for _ in range(8)]))
    audit = write(tmp_path / 'audit.json', dict(physical_frames=8, new_environment_resets=0))
    state = State(16800)
    active = r.RecoveryOwner.__new__(r.RecoveryOwner)
    active.live = lambda **kw: None
    active.pending = {}; active.states = {'regularized': state}
    active.reserved = {'regularized': 16800}; active.state_fingerprints = {'regularized': 'same'}
    active.cursors = {'regularized': [0] * 8}; active.train_streams = {i: [case] if i == 0 else [] for i in range(8)}
    active.readers = {'regularized': SimpleNamespace(environments=[None] * 8, active=[None] * 8)}
    active.environments = {}; active.active_cases = {}; active.plan = SimpleNamespace(proposal='finite-proposal')
    active.event = lambda *a, **kw: None
    batch = dict(start_joint_step=16800, end_joint_step=16808, rows=[{}] * 8)
    monkeypatch.setattr(r, 'training_fingerprint', lambda state: 'same')
    monkeypatch.setattr(r.old.seeds.partitions, 'assemble_training_batch', lambda *a: (batch, r.bound(audit)))
    monkeypatch.setattr(r.old, 'WarehouseMultiAgentEnv', lambda *a, **kw: pytest.fail('cached replay cannot construct/reset'))
    active.replay_cached_rollout('regularized', dict(trace=trace, audit=audit, update=None))
    assert state.joint_steps == active.reserved['regularized'] == 16808
    assert active.cursors['regularized'][0] == 1
    assert active.pending['regularized']['trace'] == trace


@pytest.mark.parametrize('changed', [False, True])
def test_replayed_committed_update_matches_every_field_before_continuation(tmp_path, monkeypatch, changed):
    result = dict(start_joint_step=16800, end_joint_step=17824,
        accepted_positive_program_gradient_updates={'robot_1': 15, 'robot_2': 15}, epochs=[dict(loss=0.125)])
    original = write(tmp_path / 'original-update.json', result)
    recomputed = deepcopy(result)
    if changed: recomputed['epochs'][0]['loss'] += 1e-12
    active = r.RecoveryOwner.__new__(r.RecoveryOwner)
    active.live = lambda **kw: None
    active.states = {'raw': State(17824)}
    active.pending = {'raw': dict(state_sha256='same', trace={'path': 'cached', 'sha256': 'a' * 64},
        batch=dict(start_joint_step=16800, rows=[{}] * 8), original_update=original)}
    active.providers = {}; active.cores = {'raw': SimpleNamespace(update_batch=lambda *a, **kw: recomputed)}
    active.event = lambda *a, **kw: None
    active.folder = tmp_path / 'recovery'; active.updates = {'raw': [original]}; active.replay_proofs = []
    active.state_fingerprints = {}
    monkeypatch.setattr(r, 'training_fingerprint', lambda state: 'same')
    if changed:
        with pytest.raises(RuntimeError, match='committed update differs'):
            active.update('raw')
        assert not active.replay_proofs
    else:
        active.update('raw')
        assert active.updates['raw'] == [original]
        assert active.replay_proofs[0]['matched'] is True
        assert r.bound(active.replay_proofs[0]['recomputed_update']) == result
        assert not active.pending


def test_recovery_is_explicit_only_and_budget_remains_20k():
    with pytest.raises(RuntimeError, match='explicitly authorized'):
        r.RecoveryOwner()
    assert r.STARTS == dict(raw=18600, regularized=16800)
    assert r.old.PPOConfig().joint_step_budget == 20000


@pytest.fixture
def supervisor_case(tmp_path, monkeypatch):
    from scripts import resume_warehouse_v642_calibration as cli
    original = write(tmp_path / 'base/plan.json', {})
    ref = write(tmp_path / 'base/recovery/synthetic/plan.json', {})
    folder = Path(ref['path']).parent
    source = r.pointer(Path(cli.__file__).resolve())
    plan = dict(original_plan=original, source_files={'scripts/resume_warehouse_v642_calibration.py': source['sha256']})
    monkeypatch.setattr(r, 'checked_plan', lambda checked: plan if checked == ref else pytest.fail('different plan'))
    return cli, ref, folder, source


def test_supervisor_validates_before_spawn_or_output(supervisor_case, monkeypatch):
    cli, ref, folder, _ = supervisor_case
    def refuse(*args, **kwargs): raise RuntimeError('source/authorization rejected')
    monkeypatch.setattr(r, 'checked_plan', refuse)
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('closed source gate spawned a worker'))
    with pytest.raises(RuntimeError, match='source/authorization'):
        cli.supervise(ref)
    assert sorted(p.name for p in folder.iterdir()) == ['plan.json']


def test_supervisor_binds_actual_wait_zero_and_is_single_use(supervisor_case, monkeypatch):
    cli, ref, folder, source = supervisor_case
    calls = []
    class Worker:
        pid = 43210
        def wait(self):
            calls.append('actual_mock_wait_returned')
            result = write(folder / 'result.json', dict(version=r.VERSION, completed=True,
                numerical_training_completed=True, recovery_plan=ref, pilot_opened=False))
            write(folder / 'completed.json', dict(result_sha256=result['sha256']))
            return 0
    def spawn(command, **kwargs):
        calls.append(command)
        assert command == [str(Path(cli.sys.executable).resolve()), str(Path(cli.__file__).resolve()),
                           '--run', '--recovery-plan', ref['path']]
        assert kwargs['stderr'] == cli.subprocess.STDOUT and kwargs['stdin'] == cli.subprocess.DEVNULL
        kwargs['stdout'].write(b'synthetic worker log\n')
        return Worker()
    monkeypatch.setattr(cli.subprocess, 'Popen', spawn)
    receipt = cli.supervise(ref)
    value = r.bound(receipt)
    assert value['observed_by'] == 'source_bound_subprocess_wait' and value['exit_code'] == 0
    assert value['worker_pid'] == 43210 and value['supervisor_source'] == source
    assert value['result'] == r.pointer(folder / 'result.json')
    assert value['completed'] == r.pointer(folder / 'completed.json')
    assert value['started'] == r.pointer(folder / 'supervisor-started.json')
    assert value['log'] == r.pointer(folder / 'run.log')
    assert calls[-1] == 'actual_mock_wait_returned'
    assert not (folder / 'run.log').stat().st_mode & 0o222
    with pytest.raises(RuntimeError, match='already started'):
        cli.supervise(ref)
    assert len(calls) == 2


@pytest.mark.parametrize('worker_exit,cli_exit', [(7, 7), (-9, 137)])
def test_supervisor_preserves_actual_wait_failure_status(supervisor_case, monkeypatch, worker_exit, cli_exit):
    cli, ref, folder, _ = supervisor_case
    waited = []
    class Worker:
        pid = 43211
        def wait(self): waited.append(True); return worker_exit
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *args, **kwargs: Worker())
    with pytest.raises(SystemExit) as error:
        cli.supervise(ref)
    assert error.value.code == cli_exit and waited == [True]
    receipt = r.bound(r.pointer(folder / 'supervisor-exit.json'))
    assert receipt['exit_code'] == worker_exit and receipt['observed_by'] == 'source_bound_subprocess_wait'
    assert 'result' not in receipt and 'completed' not in receipt


def test_spawn_failure_never_fabricates_a_wait_receipt(supervisor_case, monkeypatch):
    cli, ref, folder, _ = supervisor_case
    def spawn(*args, **kwargs): raise OSError('synthetic spawn refusal')
    monkeypatch.setattr(cli.subprocess, 'Popen', spawn)
    with pytest.raises(OSError, match='spawn refusal'):
        cli.supervise(ref)
    assert not (folder / 'supervisor-exit.json').exists()
    failed = r.bound(r.pointer(folder / 'supervisor-failed.json'))
    assert failed['observed_by'] == 'source_bound_subprocess_spawn_failure'
    assert 'exit_code' not in failed


def test_wait_zero_without_result_is_recorded_truthfully_but_rejected(supervisor_case, monkeypatch):
    cli, ref, folder, _ = supervisor_case
    class Worker:
        pid = 43212
        def wait(self): return 0
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *args, **kwargs: Worker())
    with pytest.raises(RuntimeError, match='missing/symlink scientific asset'):
        cli.supervise(ref)
    receipt = r.bound(r.pointer(folder / 'supervisor-exit.json'))
    assert receipt['exit_code'] == 0 and 'result_binding_error' in receipt
    assert 'result' not in receipt and 'completed' not in receipt
