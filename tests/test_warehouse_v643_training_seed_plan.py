"""Synthetic original-argument and real exact-auditor checks; no science run."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util
import json

import pytest

from experiments.warehouse import v643_training_seed_plan as s


def helpers():
    spec=importlib.util.spec_from_file_location('v643_synthetic_partition_fixture',
        Path(__file__).with_name('test_warehouse_v643_training_partitions.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def partition(tmp_path,monkeypatch):
    return helpers().partition.__wrapped__(tmp_path,monkeypatch)


@pytest.fixture
def prepared(partition,tmp_path,monkeypatch):
    # Small source closure only in isolated software fixtures. This never
    # creates or authenticates a candidate execution claim.
    path=Path(s.__file__).resolve()
    sources={str(path.relative_to(s.ROOT)):s.file_sha(path)}
    monkeypatch.setattr(s,'source_files',lambda:sources)
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path/'seed-plans')
    return s.prepare(reference_directory=tmp_path/'reference',partition_ref=partition.ref,run_id='synthetic')


def test_exact_call_plan_accounts_for_quality_and_paired_streams(partition,tmp_path):
    declaration,cases=s.declared_calls(partition)
    path=tmp_path/'exact-original-arguments.json';path.write_text(json.dumps(declaration))
    assert s.exact.read_exact_plan(path)[0]==declaration
    assert len(cases)==2*(168+1650)+225
    assert len(declaration['uses'])==4*len(cases)+4
    raw=[{k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='raw' and c['scope']!='quality']
    reg=[{k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='regularized' and c['scope']!='quality']
    assert raw==reg
    quality=[c for c in cases if c['scope']=='quality']
    assert len(quality)==225 and {c['variant'] for c in quality}=={'reference','raw','regularized'}
    assert declaration['candidate_id']==s.CANDIDATE and declaration['authorization_granted'] is False


def test_seed_plan_freezes_quality_before_training_and_has_zero_execution_counts(prepared):
    prepared.verify()
    assert prepared.value['quality_policy_reservations']==225
    assert prepared.value['maximum_explicit_resets']==3861
    assert prepared.value['training_authority'] is False
    assert prepared.value['new_environment_resets']==prepared.value['new_Actor_updates']==0


@pytest.mark.parametrize('change',['case','argument','snapshot','seal'])
def test_tampered_finite_plan_cannot_reverify(prepared,change):
    if change=='case':path=Path(prepared.value['case_reservations']['path'])
    elif change=='argument':path=Path(prepared.value['exact_plan']['path'])
    elif change=='snapshot':
        name=next(iter(prepared.value['source_files']))
        path=Path(prepared.ref['path']).parent/'source'/prepared.value['source_sha256']/name
    else:path=Path(prepared.ref['path']).with_name('completed.json')
    path.chmod(0o644);path.write_text('{}');path.chmod(0o444)
    with pytest.raises((RuntimeError,ValueError)):
        prepared.verify()


def empty_roots(tmp_path,monkeypatch):
    root=tmp_path/'synthetic-history'/'XAI-v6-integration'
    for name in s.history.HISTORY_NAMES:(root.parent/name).mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root)
    path=root.parent/'registered-traces.json';path.write_text(json.dumps(dict(registered_trace_paths=[])))
    monkeypatch.setattr(s.history,'INVENTORY_REF',s.pointer(path))
    return root


@pytest.fixture
def audited(prepared,tmp_path,monkeypatch):
    root=empty_roots(tmp_path,monkeypatch)
    return prepared,s.audit_seeds(prepared),root


def test_real_metadata_triage_exact_pipeline_on_four_empty_fixture_roots(audited):
    plan,clearance,_=audited
    assert s.checked_clearance(plan)==clearance
    assert clearance['value']['candidate_id']==s.CANDIDATE
    assert clearance['value']['selection']==[] and clearance['value']['actual_resets']==0
    assert clearance['value']['training_authority'] is False


@pytest.mark.parametrize('change',['expired','history_delta','root_removed','old_candidate','false_pass','registry'])
def test_clearance_refuses_expiry_delta_old_candidate_or_fabricated_authority(audited,change):
    plan,clearance,root=audited
    ref=clearance['clearance'];path=Path(ref['path']);value=deepcopy(clearance['value'])
    if change=='expired':
        issued=datetime.now(timezone.utc)-timedelta(days=2)
        value['issued_utc']=issued.isoformat();value['expires_utc']=(issued+timedelta(hours=24)).isoformat()
    if change=='history_delta':(root/'warehouse-new-receipt.json').write_text(json.dumps(dict(seed=60000)))
    if change=='root_removed':(root.parent/s.history.HISTORY_NAMES[0]).rmdir()
    if change=='old_candidate':value['candidate_id']='warehouse-v642-depth9-depth12-01'
    if change=='false_pass':value['training_authority']=True
    if change=='registry':Path(s.history.INVENTORY_REF['path']).write_text('{"registered_trace_paths":["new"]}')
    if change in ('expired','old_candidate','false_pass'):
        path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)):
        s.checked_clearance(plan)


def test_spent_quality_argument_blocks_clearance_too(prepared,tmp_path,monkeypatch):
    root=empty_roots(tmp_path,monkeypatch)
    # Exact quality argument, not a TRAIN/program value. No environment is made.
    (root/'warehouse-prior-quality-seed-receipt.json').write_text(json.dumps(dict(schema='warehouse-training-result.v1',scenario_seed=500001)))
    with pytest.raises(RuntimeError,match='conflict|unresolved'):
        s.audit_seeds(prepared)
    assert not (Path(prepared.ref['path']).parent/'seed-ledger/clearance.json').exists()
