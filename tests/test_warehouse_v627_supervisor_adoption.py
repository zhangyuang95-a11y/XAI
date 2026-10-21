"""Synthetic process ownership and no-retry contracts; no science is run."""
import json
from pathlib import Path
import time

import pytest

from scripts import adopt_warehouse_v627_supervisor as controller


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    root = tmp_path / 'S64'
    (root / 'execution').mkdir(parents=True)
    specs = [dict(job_id=f'job-{i}', stage='S64') for i in range(6)]
    plan = dict(jobs=specs, source_sha256='a' * 64)
    path = root / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(controller, 'checked_plan', lambda stage: (path, plan))
    monkeypatch.setattr(controller, 'finalize', lambda stage: {'finalized': stage})
    monkeypatch.setattr(controller.time, 'sleep', lambda seconds: None)

    def finished(spec):
        folder = root / 'jobs' / spec['job_id']
        folder.mkdir(parents=True, exist_ok=True)
        start = dict(search_deadline_monotonic=time.monotonic() + 100,
                     deadline_monotonic=time.monotonic() + 200)
        (folder / 'started.json').write_text(json.dumps(start))
        result = dict(job=spec, plan_sha256=controller.v.file_sha(path),
                      solver={'checkpoint': {'pending_calls': 0}})
        (folder / 'result.json').write_text(json.dumps(result))
        (folder / 'completed.json').write_text(json.dumps(
            dict(result_sha256=controller.v.file_sha(folder / 'result.json'))))
        return folder

    def process(spec, pid):
        return dict(pid=pid, parent=1, group=pid, stat='S',
                    argv=['python', str(tmp_path / 'scripts/run_warehouse_v627.py'),
                          '--stage', 'S64', '--job', spec['job_id']])

    return root, specs, path, finished, process


def test_adoption_observes_orphans_but_only_waitpids_new_children(monkeypatch, tmp_path):
    root, specs, path, finished, process = fixture(monkeypatch, tmp_path)
    for spec in specs[:3]:
        finished(spec)
    calls = 0

    def listing():
        nonlocal calls
        calls += 1
        return {101: process(specs[0], 101), 102: process(specs[1], 102)} if calls < 3 else {}

    monkeypatch.setattr(controller, 'processes', listing)
    launched, waits = [], []

    class Child:
        def __init__(self, args, **kwargs):
            job = args[args.index('--job') + 1]
            assert job not in {'job-0', 'job-1', 'job-2'}
            assert not (root / 'jobs' / job).exists()
            self.pid = 300 + len(launched)
            launched.append(job)
            finished(next(s for s in specs if s['job_id'] == job))

        def poll(self):
            return 0

        def wait(self, **kwargs):
            waits.append(self.pid)
            return 0

    monkeypatch.setattr(controller.subprocess, 'Popen', Child)
    assert controller.execute() == {'finalized': 'S64'}
    receipt = json.loads((root / 'execution/adoption.completed.json').read_bytes())
    assert launched == ['job-3', 'job-4', 'job-5'] and len(waits) == 3
    assert receipt['newly_owned_workers_waitpid_count'] == 3
    assert receipt['originally_orphaned_workers_waitpid_count'] == 0
    assert receipt['all_six_processes_settled'] is True
    for row in receipt['settled']:
        assert (row['waitpid_returncode'] == 0) == (row['exit_evidence'] == 'waitpid')
    with pytest.raises(RuntimeError):
        controller.execute()


def test_dirty_exited_worker_is_never_restored_or_relaunched(monkeypatch, tmp_path):
    root, specs, path, finished, process = fixture(monkeypatch, tmp_path)
    (root / 'jobs/job-0').mkdir(parents=True)
    monkeypatch.setattr(controller, 'processes', lambda: {})
    with pytest.raises(FileNotFoundError):
        controller.execute()
    assert not (root / 'execution/adoption.claim.json').exists()


def test_adoption_rejects_original_live_supervisor(monkeypatch, tmp_path):
    root, specs, path, finished, process = fixture(monkeypatch, tmp_path)
    p = process(specs[0], 101)
    p['argv'] = ['python', str(tmp_path / 'scripts/run_warehouse_v627.py'), '--stage', 'S64', '--execute']
    monkeypatch.setattr(controller, 'processes', lambda: {101: p})
    with pytest.raises(RuntimeError, match='supervisor is still alive'):
        controller.execute()
    assert not (root / 'execution/adoption.claim.json').exists()


def test_final_seal_cannot_hide_unsettled_optimization(monkeypatch, tmp_path):
    root, specs, path, finished, process = fixture(monkeypatch, tmp_path)
    folder = finished(specs[0])
    result = json.loads((folder / 'result.json').read_bytes())
    result['solver']['checkpoint']['pending_calls'] = 1
    (folder / 'result.json').write_text(json.dumps(result))
    (folder / 'completed.json').write_text(json.dumps(
        dict(result_sha256=controller.v.file_sha(folder / 'result.json'))))
    with pytest.raises(RuntimeError, match='unpaid or unsettled'):
        controller.check_result(folder, path, specs[0])
