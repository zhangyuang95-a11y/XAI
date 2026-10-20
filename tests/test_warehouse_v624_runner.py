"""Runner provenance and cleanup tests without training or scientific fits."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest
from scripts import run_warehouse_v624_upgrade as runner
from experiments.warehouse import v624_protocol as v


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def synthetic_results(tmp_path):
    path=tmp_path/'plan.json';save(path,{'jobs':v.jobs()});plan={'jobs':v.jobs()};sha=v.file_sha(path)
    for spec in plan['jobs']:
        p=path.parent/'jobs'/spec['job_id']
        r=dict(v.ZERO,schema='warehouse-v624-upgrade-result.v1',job=spec,stage='A',plan_sha256=sha,gate={'passed':False})
        save(p/'result.json',r);save(p/'started.json',dict(v.ZERO,job=spec,stage='A',plan_sha256=sha))
        save(p/'completed.json',{'plan_sha256':sha,'result_sha256':v.file_sha(p/'result.json'),'passed':False})
    return path,plan


def test_collect_rejects_missing_job_or_bad_completion_binding(tmp_path):
    path,plan=synthetic_results(tmp_path)
    assert len(runner.collect_results(path,plan))==12
    p=path.parent/'jobs'/plan['jobs'][-1]['job_id']/'completed.json'
    good=json.loads(p.read_bytes());bad=deepcopy(good);bad['result_sha256']='0'*64;save(p,bad)
    with pytest.raises(RuntimeError):runner.collect_results(path,plan)
    save(p,good);p.unlink()
    with pytest.raises(FileNotFoundError):runner.collect_results(path,plan)


def test_engineering_failure_never_enters_scientific_summary(tmp_path):
    path,plan=synthetic_results(tmp_path)
    p=path.parent/'jobs'/plan['jobs'][0]['job_id']/'failed.json';save(p,{'error':'interrupted'})
    with pytest.raises(RuntimeError):runner.collect_results(path,plan)


def test_write_once_and_exclusive_worker_claim(tmp_path):
    path=tmp_path/'receipt.json';runner.write_new(path,{'value':1})
    with pytest.raises(FileExistsError):runner.write_new(path,{'value':2})
    first=runner._claim(tmp_path)
    try:
        with pytest.raises(BlockingIOError):runner._claim(tmp_path)
    finally:first.close()
    runner._claim(tmp_path).close()


def test_stop_workers_reaps_all_owned_processes_before_return(tmp_path):
    active={};processes=[]
    try:
        for i in range(2):
            p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
            processes.append(p);log=(tmp_path/f'{i}.log').open('w');active[str(i)]=(p,log,time.monotonic())
        rows=runner.stop_workers(active)
        assert not active and len(rows)==2 and all(r['waited'] for r in rows)
        assert all(p.poll() is not None for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait()


def test_execute_rejects_invalid_worker_count_before_any_prepare():
    for value in (0,4,True):
        with pytest.raises(RuntimeError):runner.execute(value)


def test_actual_new_search_structure_and_report_serialize_without_partial_numpy_receipts(tmp_path):
    import numpy as np
    from backend.training.warehouse_v6_actor import FEATURE_DIM,FEATURE_NAMES
    from backend.training.warehouse_v622_program import program_structure_sha256
    from backend.training.warehouse_v624_search import fit_structure,UpgradeBudget
    mask_indices=[FEATURE_NAMES.index('legal_'+a) for a in ('up','down','left','right','wait')]
    x=np.zeros((80,FEATURE_DIM),np.float32);m=np.ones((80,5),np.float32);x[:,mask_indices]=m
    x[:,FEATURE_NAMES.index('self_row')]=np.linspace(0,1,80)
    p=np.full((80,5),.01,np.float32);p[:,0]=.96
    episodes=np.asarray(['synthetic-'+str(i//10) for i in range(80)])
    w=np.zeros((1,FEATURE_DIM),np.float32);w[0,mask_indices[0]]=1
    initial={'root_ref':0,'leaf_count':2,'feature_dim':FEATURE_DIM,'split_weights':w,
        'split_bias':np.asarray([-.5],np.float32),'left_refs':np.asarray([-1]),'right_refs':np.asarray([-2])}
    structure,report=fit_structure('P0',x,m,p,episodes,feature_names=FEATURE_NAMES,
        actor_role_sha256='a'*64,initial_structure=initial,
        initial_structure_sha256=program_structure_sha256(initial),
        budget=UpgradeBudget(maximum_new_leaf_calls=40,wall_seconds=660,maximum_rounds=1))
    runner.write_new(tmp_path/'structure.json',structure);runner.write_new(tmp_path/'search-report.json',report)
    loaded=json.loads((tmp_path/'structure.json').read_bytes())
    assert program_structure_sha256(loaded)==program_structure_sha256(structure)
    assert json.loads((tmp_path/'search-report.json').read_bytes())==runner.plain(report)
    runner.write_new(tmp_path/'scalars.json',{'integer':np.int64(3),'bool':np.bool_(True)})
    assert json.loads((tmp_path/'scalars.json').read_bytes())=={'integer':3,'bool':True}
    invalid=tmp_path/'invalid.json'
    with pytest.raises(ValueError):runner.write_new(invalid,{'invalid':np.float32('nan')})
    assert not invalid.exists()
