"""Synthetic terminal-metric contracts; no old worker or environment calls."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import pytest
from scripts import measure_warehouse_v628_terminal as m


def test_waitpid_events_do_not_treat_started_or_partial_records_as_exit(tmp_path):
    file=tmp_path/'events.jsonl'
    events=[dict(event='started',job=dict(job_id='j'),pid=42),
        dict(event='exited',job=dict(job_id='j'),pid=42,returncode=0,waited=True)]
    file.write_text(json.dumps(events[0])+'\n'+json.dumps(events[1])+'\n'+ '{"event":"exited"')
    assert m.waited_jobs(file)=={'j':events[1]}


def test_terminal_measurement_never_refits_or_turns_failed_job_into_admission(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('v629_synthetic_helpers_terminal',Path(__file__).with_name('test_warehouse_v629.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    model,parts=helper.fixture(tmp_path)
    job=dict(job_id='C64-fold-0-robot_1',fold=0,role='robot_1',capacity=64)
    root=tmp_path/'old';folder=root/'jobs'/job['job_id'];folder.mkdir(parents=True)
    result=dict(job=job,status='scientific_budget_exhausted',gate=dict(passed=False))
    (folder/'result.json').write_text(json.dumps(result))
    (folder/'completed.json').write_text(json.dumps(dict(result_sha256=m.v.file_sha(folder/'result.json'))))
    plan=dict(actor_role_sha256={'robot_1':model.actor_role_sha256},parents={'fold-0-robot_1':m.v.pointer(tmp_path/'parent.json')})
    plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(m.v,'OUTPUT',root)
    monkeypatch.setattr(m,'initial_program',lambda *args:(deepcopy(model),dict(synthetic='snapshot'),{}))
    monkeypatch.setattr(m.data,'load_parts',lambda *args,**kw:parts)
    import backend.training.warehouse_v628_program as kernel
    def forbidden(*args,**kw):raise AssertionError('terminal observer cannot fit or calibrate')
    monkeypatch.setattr(kernel,'fit_leaf',forbidden);monkeypatch.setattr(kernel,'calibrate',forbidden)
    source={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    row=m.measure_job(plan_path,plan,job,dict(job=job,waited=True,pid=42,returncode=0),tmp_path/'observation')
    assert row['metric_only_gate']['passed']
    assert not row['original_admission'] and not row['candidate_admission']
    assert row['fitting_calls']==row['LP_calls']==row['environment_resets']==0
    assert row['coefficients_and_topology_unchanged']
    assert source=={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError,match='waitpid'):
        m.measure_job(plan_path,plan,job,dict(job=job,waited=False),tmp_path/'unowned')
