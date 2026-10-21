"""Observer-only synthetic files; never fit, infer, allocate or release."""
from pathlib import Path
import json,time
import pytest
from scripts import report_warehouse_v630_progress as r
from backend.training.warehouse_v628_checkpoint import CheckpointSession


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return r.v.pointer(path)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(r.v,'OUTPUT',tmp_path/'candidate')
    job=dict(job_id='P-fold-0-robot_1',name='P',scope='protected',fold=0,role='robot_1')
    plan=dict(jobs=[job],actor_role_sha256={'robot_1':'test-role'},dependency_process=dict(pid=99,identity='old driver'))
    path=r.v.OUTPUT/'plan.json';write(path,plan)
    monkeypatch.setattr(r.v,'process_identity',lambda pid:'old driver' if pid==99 else None)
    return path,plan,job


def test_waiting_reports_no_invented_zero_metrics_or_model_pass(setup):
    path,plan,job=setup;value=r.collect(path,plan,supervisor_pid=100)
    assert value['dependency']=='waiting_verified_v629_driver' and value['settled']==0 and not value['supervisor_verified']
    assert value['rows'][0]['status']=='not_started' and 'metrics' not in value['rows'][0]
    assert '—' in r.markdown(value) and 'not_started' in r.markdown(value)


def metrics():
    return {k:dict(coverage=.46,expected_actionable_coverage=.44,agreement=.99,kl=.1,
        whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0) for k in ('feedback','explanation')}


def test_terminal_measurements_require_matching_plan_job_and_seal(setup):
    path,plan,job=setup;folder=r.v.OUTPUT/'jobs'/job['job_id']
    result=dict(job=job,plan=r.v.pointer(path),status='completed',metrics=dict(validation=metrics()),gate=dict(passed=True),leaf_count=50,depth=8)
    ref=write(folder/'result.json',result);write(folder/'completed.json',dict(result_sha256=ref['sha256']))
    value=r.collect(path,plan);assert value['settled']==1 and value['rows'][0]['metrics']==metrics()
    write(folder/'completed.json',dict(result_sha256='0'*64));value=r.collect(path,plan)
    assert value['settled']==0 and value['errors'] and 'metrics' not in value['rows'][0]


def test_failed_native_job_preserves_reason_without_zero_filling(setup):
    path,plan,job=setup;folder=r.v.OUTPUT/'jobs'/job['job_id']
    ref=write(folder/'result.json',dict(job=job,plan=r.v.pointer(path),status='scientific_budget_exhausted',
        error='fixed search budget',gate=dict(passed=False),solver=None))
    write(folder/'completed.json',dict(result_sha256=ref['sha256']))
    value=r.collect(path,plan);row=value['rows'][0]
    assert row['stop_reason']=='fixed search budget' and row['metrics'] is None and row['gate']['passed'] is False
    assert 'fixed search budget' in r.markdown(value)


def test_worker_identity_missing_remains_unsettled_and_partial_event_tail_is_ignored(setup):
    path,plan,job=setup;folder=r.v.OUTPUT/'jobs'/job['job_id'];write(folder/'started.json',{})
    events=r.v.OUTPUT/'execution/events.jsonl';events.parent.mkdir();events.write_bytes((json.dumps(dict(event='started',job=job,pid=101))+'\n').encode()+b'{"event":')
    value=r.collect(path,plan);assert value['settled']==0 and value['rows'][0]['status']=='unsettled_worker_not_verified'


def test_live_checkpoint_copies_existing_scores_without_new_scoring(setup,monkeypatch):
    path,plan,job=setup;folder=r.v.OUTPUT/'jobs'/job['job_id'];source=write(path.parent/'source.json',dict(leaves=[{},{}]))
    started=dict(job=job,plan=r.v.pointer(path),source=dict(program=source));write(folder/'started.json',started)
    identity=dict(plan_sha256=r.v.pointer(path)['sha256'],job=job,source=started['source'],limits=r.v.recipe()['limits'])
    session=CheckpointSession(folder/'checkpoint.json',identity,max_calls=256,wall_seconds=1800)
    best=dict(role='robot_1',actor_role_sha256='test-role',maximum_leaves=256,leaves=[{},{}])
    session.commit(dict(stage='allocation',inspected_leaves=[0,1],solver_counts=dict(leaf=0,node=1,proof=0),
        candidates=[dict(metrics=metrics(),reliable_selection_counts=dict(E=50,F=50,expected_F=45),reliable_selection_passed=True)],
        best_program=best,best_candidate=0),status='running');session.close()
    events=r.v.OUTPUT/'execution/events.jsonl';events.parent.mkdir();events.write_text(json.dumps(dict(event='started',job=job,pid=101))+'\n')
    monkeypatch.setattr(r.v,'process_identity',lambda pid:'synthetic python run_warehouse_v630.py --job P-fold-0-robot_1' if pid==101 else None)
    before=r.v.pointer(folder/'checkpoint.json');value=r.collect(path,plan);p=value['rows'][0]['partial']
    assert value['rows'][0]['status']=='running' and p['metrics']==metrics() and p['inspected_leaves']==2
    assert p['scope'].startswith('fit_only') and r.v.pointer(folder/'checkpoint.json')==before
