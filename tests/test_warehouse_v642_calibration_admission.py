"""Synthetic receipt replay contracts, not NN/physics or scientific evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.warehouse import v642_calibration_admission as a


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    return a.pointer(path)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    c = a.collection
    folder = tmp_path / 'synthetic-quality'
    out = folder / 'collection'
    out.mkdir(parents=True)
    cases = c.quality.cases(90300000, 9030000000)
    actors = {v: dict(path=str(tmp_path / (v + '.npz')), sha256=str(i) * 64)
              for i, v in enumerate(c.quality.VARIANTS, 1)}
    plan = dict(cases=cases, fixed=dict(actors=actors, actor_role_sha256={
        variant: dict(robot_1='a' * 64, robot_2='b' * 64) for variant in actors}), final_programs={})
    plan_ref = save(folder / 'plan.json', plan)
    now = datetime.now(timezone.utc)
    issued, started, authorized, expires = [now + timedelta(seconds=n) for n in (-10, -5, -1, 60)]
    clearance = dict(issued_utc=issued.isoformat(), expires_utc=expires.isoformat())
    clearance_ref = save(folder / 'clearance.json', clearance)
    claim_ref = save(folder / 'claim.json', dict(version=c.VERSION, plan=plan_ref,
        clearance=clearance_ref, started_utc=started.isoformat(), single_use_even_if_interrupted=True, pilot_opened=False))
    monkeypatch.setattr(c, 'checked', lambda ref: (deepcopy(plan), folder))
    monkeypatch.setattr(c, 'clearance', lambda ref, execution: (clearance_ref, deepcopy(clearance)))
    monkeypatch.setattr(c, 'claim_path', lambda: Path(claim_ref['path']))
    monkeypatch.setattr(c.FrozenWarehouseV6Actor, 'load', lambda *args, **kwargs: object())
    monkeypatch.setattr(c, 'verify_frame', lambda *args: 2)
    monkeypatch.setattr(c, 'replay_public_episode', lambda case, events, frames, sha:
        dict(case=case, deliveries=15, robot_collisions=2, shutdown=False, physical_frames=frames))
    program_metrics = dict(passed=True, synthetic_scope='metadata_replay_only')
    monkeypatch.setattr(a, 'score_programs', lambda *args: deepcopy(program_metrics))
    entries, outcomes, auth, resets = [], [], [], []
    for case in cases:
        for variant in c.quality.VARIANTS:
            actor = actors[variant]
            stem = out / (variant + '-' + case['case_id'])
            trace = stem.with_suffix('.jsonl.gz')
            with gzip.open(trace, 'wt') as h:
                for _ in range(120):
                    h.write('{"action_override_agents":[]}\n')
            trace.chmod(0o444)
            array = stem.with_suffix('.npz')
            np.savez_compressed(array, **{key: np.zeros((120, 1)) for key in ('x', 'm', 'p', 'sampled')})
            array.chmod(0o444)
            entries.append(dict(case=case, variant=variant, trace=a.pointer(trace), arrays=a.pointer(array)))
            outcomes.append(dict(case=case, deliveries=15, robot_collisions=2, shutdown=False, physical_frames=120,
                variant=variant, actor=actor, complete=True, NN_action_checks=240, action_overrides=0))
            auth.append(dict(case=case, variant=variant, plan=plan_ref, actor=actor, authorized_utc=authorized.isoformat()))
            resets.append(dict(seed=case['scenario_seed'], episode_id=1, phase='quality',
                **{key: a.ACTION_PHYSICS_CONFIG[key] for key in ('map_layout_id', 'move_battery_cost', 'charge_per_wait')}))
    def lines(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        path.chmod(0o444)
        return a.pointer(path)
    result = dict(version=c.VERSION, plan=plan_ref, claim=claim_ref, clearance=clearance_ref,
        actual_resets=lines(out / 'actual-resets.jsonl', resets),
        reset_preauthorization=lines(out / 'reset-preauthorization.jsonl', auth), entries=entries, outcomes=outcomes,
        actual_reset_count=225, metrics=c.quality.evaluate(outcomes, cases, actors), program_metrics=program_metrics,
        task_retention_passed=True, calibration_precheck_passed=True, model_acceptance_passed=False, pilot_opened=False)
    def persist():
        ref = save(out / 'result.json', result)
        save(out / 'completed.json', dict(result=ref))
        exit_ref = save(out / 'supervisor-exit.json', dict(exit_code=0, observed_by='exec_tool_terminal_result', result=ref))
        return ref, exit_ref
    return result, program_metrics, persist, out


def test_missing_completion_refuses_without_environment_or_rng(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('missing completed evidence reached environment or RNG')
    monkeypatch.setattr(a.collection.WarehouseMultiAgentEnv, '__init__', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    with pytest.raises(RuntimeError):
        a.audit(dict(path=str(tmp_path / 'missing'), sha256='x'), {})
    assert not list(tmp_path.iterdir())


def test_replayed_contract_is_calibration_only(completed):
    _, _, persist, _ = completed
    report = a.audit(*persist())
    assert report['calibration_precheck_passed'] and report['replayed_role_actions'] == 54000
    assert report['new_environment_resets'] == report['NN_updates'] == report['program_fit_calls'] == 0
    assert not any(report[key] for key in ('training_authority', 'sealed_authority', 'model_acceptance_passed', 'pilot_opened'))


@pytest.mark.parametrize('change', ['exit', 'seal', 'writable', 'trace_hash', 'reset_order', 'outcome', 'task_metric', 'program_metric', 'failed'])
def test_forged_or_incomplete_evidence_cannot_pass(completed, change):
    result, _, persist, out = completed
    ref, exit_ref = persist()
    if change == 'exit':
        exit_ref = save(Path(exit_ref['path']), dict(exit_code=True, observed_by='exec_tool_terminal_result', result=ref))
    elif change == 'seal':
        save(out / 'completed.json', dict(result={}))
    elif change == 'writable':
        Path(result['entries'][0]['trace']['path']).chmod(0o644)
    elif change == 'trace_hash':
        p = Path(result['entries'][0]['trace']['path']);p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    elif change == 'reset_order':
        p = Path(result['actual_resets']['path']);p.chmod(0o644)
        rows = [json.loads(x) for x in p.read_text().splitlines()];rows[0]['seed'] += 1
        p.write_text(''.join(json.dumps(x) + '\n' for x in rows));p.chmod(0o444)
        result['actual_resets'] = a.pointer(p);ref, exit_ref = persist()
    elif change == 'outcome':
        result['outcomes'][0]['deliveries'] += 1;ref, exit_ref = persist()
    elif change == 'task_metric':
        result['metrics']['quality']['raw']['deliveries'] += 1;ref, exit_ref = persist()
    elif change == 'program_metric':
        result['program_metrics'] = dict(passed=True, forged=True);ref, exit_ref = persist()
    else:
        save(out / 'failed.json', dict(error='interrupted'))
    with pytest.raises(RuntimeError):
        a.audit(ref, exit_ref)


def test_matching_failed_program_verdict_still_closes_gate(completed):
    result, programs, persist, _ = completed
    programs['passed'] = False
    result['calibration_precheck_passed'] = False
    with pytest.raises(RuntimeError, match='four program gates failed'):
        a.audit(*persist())
