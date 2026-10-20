import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import run_warehouse_v625 as runner
from scripts import finalize_warehouse_v625 as finalizer


def test_prepare_refuses_without_engineering_receipt_before_loading_actor(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.v,'OUTPUT',tmp_path/'stage-A'/'paired-01')
    monkeypatch.setattr(runner.v,'source_sha',lambda:'a'*64)
    monkeypatch.setattr(runner.v,'checked_baseline',lambda:({},{}))
    monkeypatch.setattr(runner.v.old,'checked_actor',lambda *a:pytest.fail('Actor must not load before engineering gate'))
    with pytest.raises(RuntimeError,match='engineering preflight'):
        runner.prepare()
    assert not runner.v.OUTPUT.exists()


def test_prepare_rejects_generic_pass_flag_before_loading_actor(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.v,'OUTPUT',tmp_path/'stage-A'/'paired-01')
    monkeypatch.setattr(runner.v,'source_sha',lambda:'a'*64)
    monkeypatch.setattr(runner.v,'checked_baseline',lambda:({},{}))
    monkeypatch.setattr(runner.v.old,'checked_actor',lambda *a:pytest.fail('Actor must not load before native evidence'))
    path=tmp_path/'engineering'/runner.v.ENGINEERING_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'passed':True,'source_sha256':'a'*64}))
    with pytest.raises(RuntimeError,match='complete registered native-04'):
        runner.prepare()
    assert not runner.v.OUTPUT.exists()


def test_owned_worker_is_terminated_and_waited_before_failure_receipt(tmp_path):
    log=(tmp_path/'child.log').open('w')
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
                             stdout=log,stderr=log,start_new_session=True)
    active={'job':(process,log,0.)}
    try:
        result=runner.stop_workers(active)
        assert not active and process.poll() is not None
        assert result==[dict(job_id='job',pid=process.pid,returncode=process.returncode,waited=True)]
        assert log.closed
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_interruption_seal_refuses_unwaited_workers(monkeypatch,tmp_path):
    path=tmp_path/'plan.json';path.write_text('{}')
    (tmp_path/'execution').mkdir()
    (tmp_path/'execution/failed.json').write_text(json.dumps({'all_owned_workers_waited':False}))
    monkeypatch.setattr(runner,'checked_plan',lambda *a:(path,{}))
    with pytest.raises(RuntimeError,match='wait all owned'):
        finalizer.finalize(path,interruption=True)
    assert not (tmp_path/'finalizations').exists()


def test_receipts_are_single_use_and_nonfinite_never_creates_partial_file(tmp_path):
    path=tmp_path/'result.json'
    with pytest.raises(ValueError):runner.write_new(path,{'value':float('nan')})
    assert not path.exists()
    runner.write_new(path,{'value':0})
    with pytest.raises(FileExistsError):runner.write_new(path,{'value':1})
    assert json.loads(path.read_bytes())=={'value':0}


def test_runner_summary_can_be_sealed_and_asset_changes_are_reaudited(monkeypatch,tmp_path):
    from scripts import audit_warehouse_v625 as auditor
    folder=tmp_path/'stage-A'/'paired-01';folder.mkdir(parents=True)
    (tmp_path/'experiments/warehouse/evidence').mkdir(parents=True)
    planpath=folder/'plan.json';planpath.write_text('{}')
    plan={'source_sha256':'a'*64,'jobs':runner.v.jobs()}
    rows=[]
    for spec in plan['jobs']:
        metric=dict(coverage=.3,expected_actionable_coverage=.2,agreement=.99,kl=.1,
            whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0,illegal_count=0,path_uncertified_count=0)
        m=dict(feedback=dict(metric),explanation=dict(metric))
        row=dict(job=spec,status='completed',metrics={'validation':m},gate=runner.v.program_gate(m),leaf_count=32)
        path=folder/'jobs'/spec['job_id']/'result.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row));rows.append(row)
    monkeypatch.setattr(runner,'checked_plan',lambda *a:(planpath,plan))
    monkeypatch.setattr(runner,'collect_results',lambda *a:rows)
    monkeypatch.setattr(finalizer,'ROOT',tmp_path)
    calls=[]
    def audit(path):
        calls.append(path)
        return dict(passed=True,summary_sha256=runner.v.file_sha(folder/'summary.json'),
            source_sha256='a'*64,plan_sha256=runner.v.file_sha(planpath),jobs=rows,
            evidence_sha256={str(folder/'jobs'/r['job']['job_id']/'result.json'):
                runner.v.file_sha(folder/'jobs'/r['job']['job_id']/'result.json') for r in rows})
    monkeypatch.setattr(auditor,'audit',audit)
    result=runner.finalize()
    assert len(calls)==2  # A saved pass flag cannot bypass live-asset re-audit.
    assert result['finalization']['status']=='closed_failed_program_precheck'
    assert (folder/'finalizations/completed-development-01/receipt.json').is_file()
    with pytest.raises(RuntimeError,match='already sealed'):
        finalizer.finalize(planpath)
