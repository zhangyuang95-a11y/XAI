"""Explicit seed declarations only; no scientific RNG or episode allocation."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timedelta,timezone
import importlib.util,json
import numpy as np
import pytest
from experiments.warehouse import v642_training_seed_plan as s


def helper():
    spec=importlib.util.spec_from_file_location('seed_plan_partition_fixtures',Path(__file__).with_name('test_warehouse_v642_training_partitions.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def partition(tmp_path,monkeypatch):
    return helper().proposal.__wrapped__(tmp_path,monkeypatch)[0]


def test_real_gate_precedes_proposal_read_run_ID_and_any_output(tmp_path,monkeypatch):
    monkeypatch.setattr(s.partitions.admission.v,'OUTPUT',tmp_path/'missing-winner')
    def forbidden(*args,**kwargs):raise AssertionError('closed A/B must not read new input or draw RNG')
    monkeypatch.setattr(s,'bound',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='complete reaped'):
        s.prepare(reference_directory=tmp_path/'reference',proposal_ref=None,run_id=None)
    assert not list(tmp_path.iterdir())


def test_exact_parser_accepts_paired_reuse_and_both_constructor_reset_arguments(partition,tmp_path):
    plan,cases=s.declared_calls(partition)
    path=tmp_path/'typed-original-arguments.json';path.write_text(json.dumps(plan))
    parsed,_=s.exact.read_exact_plan(path)
    assert parsed==plan and len(cases)==344 and len(plan['uses'])==1380
    raw=[{k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='raw']
    reg=[{k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='regularized']
    assert raw==reg
    calls=[r for r in plan['uses'] if r['group']=='raw:train-0']
    assert {r['argument'] for r in calls}=={'task_expansion_base','random.Random(config.seed)','reset(seed)',
        'independent_agent_seed(base_seed,episode_id=1,frame,agent_id)'}
    assert len({r['value'] for r in calls if r['kind']=='env'})==1
    assert plan['paired_case_reuse_not_independent_evidence'] and not plan['authorization_granted']


def test_same_numeric_stream_cannot_hide_cross_purpose_reuse(partition,tmp_path):
    plan,_=s.declared_calls(partition);plan=deepcopy(plan)
    row=next(r for r in plan['uses'] if r['group']=='raw:evidence-validation' and r['kind']=='action_rng')
    other=next(r for r in plan['uses'] if r['group']=='raw:train-0' and r['kind']=='action_rng')
    row['value']=other['value'];row['reuse_tag']=other['reuse_tag']
    path=tmp_path/'false-reuse.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='shared across purposes'):s.exact.read_exact_plan(path)


def test_loaded_partition_must_reconstruct_all_A_B_asset_bindings(partition,tmp_path):
    loaded=s.load_partition(tmp_path/'reference',partition.ref)
    assert loaded.ref==partition.ref and loaded.refs==partition.refs and loaded.value==partition.value
    Path(partition.readiness.refs[0]['path']).chmod(0o644)
    Path(partition.readiness.refs[0]['path']).write_text('{}')
    with pytest.raises(RuntimeError,match='asset changed'):s.load_partition(tmp_path/'reference',partition.ref)


@pytest.fixture
def prepared(partition,tmp_path,monkeypatch):
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path/'seed-plans')
    return s.prepare(reference_directory=tmp_path/'reference',proposal_ref=partition.ref,run_id='synthetic-case-only')


def test_prepare_snapshots_source_and_explicit_case_reservations(prepared):
    prepared.verify();value=prepared.value
    assert value['maximum_environment_constructors']==value['maximum_explicit_resets']==344
    assert value['variants']==['raw','regularized']
    assert value['new_critic_initializations']==value['optimizer_RNG_constructors']==2
    assert value['new_environment_resets']==value['Actor_updates']==value['PPO_joint_steps']==0
    assert not value['history_freshness_verified'] and not value['training_authority']


@pytest.mark.parametrize('change',['seed_argument','case_task','source_snapshot','completion_seal'])
def test_changed_original_calls_cases_sources_or_seal_are_rejected(prepared,change):
    path=Path(prepared.ref['path']);value=prepared.value
    if change=='seed_argument':
        path=Path(value['exact_plan']['path']);body=json.loads(path.read_text());body['uses'][0]['value']+=100
        payload=json.dumps(body).encode()
    elif change=='case_task':
        path=Path(value['case_reservations']['path']);body=json.loads(path.read_text());body[0]['task']=2
        payload=json.dumps(body).encode()
    elif change=='source_snapshot':
        name=next(iter(value['source_files']));path=path.parent/'source'/value['source_sha256']/name;payload=b'# changed\n'
    else:path=path.with_name('completed.json');payload=b'{}'
    path.chmod(0o644);path.write_bytes(payload);path.chmod(0o444)
    with pytest.raises((RuntimeError,ValueError)):prepared.verify()


def test_second_prepare_does_not_overwrite_existing_plan(prepared,tmp_path):
    old=prepared.ref
    with pytest.raises(FileExistsError):
        s.prepare(reference_directory=tmp_path/'reference',proposal_ref=prepared.proposal.ref,run_id=prepared.value['run_id'])
    assert s.pointer(old['path'])==old


def empty_history(tmp_path,monkeypatch):
    # Real metadata -> triage -> exact/context machinery against four EMPTY
    # temporary roots. No real repository history audit is claimed here.
    root=tmp_path/'synthetic-four-roots'/'XAI-v6-integration'
    for name in s.confirmation.HISTORY_NAMES:(root.parent/name).mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root)
    registry=root.parent/'registry.json';registry.write_text(json.dumps(dict(registered_trace_paths=[])))
    monkeypatch.setattr(s.confirmation,'INVENTORY_REF',s.pointer(registry))
    return root


@pytest.fixture
def audited(prepared,tmp_path,monkeypatch):
    root=empty_history(tmp_path,monkeypatch)
    return prepared,s.audit_seeds(prepared),root


def test_four_root_historical_pipeline_replays_without_issuing_training(audited):
    plan,clearance,_=audited;value=clearance['value']
    assert s.checked_clearance(plan)==clearance and value['selection']==[]
    assert not value['training_authority'] and not value['sealed_authority'] and value['actual_resets']==0
    assert value['maximum_finite_reset_reservations']==344


@pytest.mark.parametrize('change',['expired','new_history_file','missing_root','false_exact_summary','registry','grant_training'])
def test_expired_changed_or_incomplete_history_cannot_clear_training_inputs(audited,change):
    plan,clearance,root=audited
    if change=='expired':
        path=Path(clearance['clearance']['path']);value=clearance['value'];issued=datetime.now(timezone.utc)-timedelta(days=2)
        value['issued_utc']=issued.isoformat();value['expires_utc']=(issued+timedelta(hours=24)).isoformat()
        path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
        assert s.checked_clearance(plan,execution=False)['value']['expires_utc']==value['expires_utc']
    if change=='new_history_file':(root/'new-seed-receipt.json').write_text(json.dumps(dict(note='new selected metadata file')))
    if change=='missing_root':(root.parent/s.confirmation.HISTORY_NAMES[0]).rmdir()
    if change=='false_exact_summary':
        path=Path(clearance['value']['exact']['path']);value=json.loads(path.read_text());value['summary']['unresolved_paths']=['invented']
        path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
        clearance['value']['exact']=s.pointer(path)
        clear_path=Path(clearance['clearance']['path']);clear_path.chmod(0o644)
        clear_path.write_text(json.dumps(clearance['value']));clear_path.chmod(0o444)
    if change=='registry':Path(s.confirmation.INVENTORY_REF['path']).write_text(json.dumps(dict(registered_trace_paths=['other-trace'])))
    if change=='grant_training':
        path=Path(clearance['clearance']['path']);value=clearance['value'];value['training_authority']=True
        path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)):s.checked_clearance(plan)


def test_existing_seed_reference_blocks_clearance_even_without_model_execution(prepared,tmp_path,monkeypatch):
    root=empty_history(tmp_path,monkeypatch)
    (root/'warehouse-seed-receipt.json').write_text(json.dumps(dict(schema='warehouse-training-result.v1',scenario_seed=100001)))
    with pytest.raises(RuntimeError,match='historical seed conflict|unresolved metadata'):s.audit_seeds(prepared)
    assert not (Path(prepared.ref['path']).parent/'seed-ledger/clearance.json').exists()
