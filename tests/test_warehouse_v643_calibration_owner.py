"""Data-only owner/journal fixtures: no real claim, NN, RNG or environment."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import pytest

from experiments.warehouse import v643_calibration_owner as o
from experiments.warehouse import v643_calibration_recovery as r
from experiments.warehouse import v643_training_seed_plan as s


def helper():
    spec=importlib.util.spec_from_file_location('v643_owner_seed_fixtures',
        Path(__file__).with_name('test_warehouse_v643_training_seed_plan.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan(tmp_path,monkeypatch):
    h=helper();partition=h.partition.__wrapped__(tmp_path,monkeypatch)
    return h.prepared.__wrapped__(partition,tmp_path,monkeypatch)


def test_disabled_owner_and_resume_refuse_before_input_or_output(tmp_path,monkeypatch):
    def forbidden(*_a,**_k):raise AssertionError('unreviewed execution must not inspect an input')
    monkeypatch.setattr(o,'pointer',forbidden)
    monkeypatch.setattr(s,'load',forbidden)
    with pytest.raises(RuntimeError,match='no claim or training'):
        o.claimed_owner(plan_ref=tmp_path/'unused')
    with pytest.raises(RuntimeError,match='explicit new candidate plan'):
        r.resume(reference_directory=tmp_path/'unused',plan_ref=None,checkpoint_ref={'path':'unused','sha256':'a'*64})
    assert not list(tmp_path.iterdir())


def test_exclusive_durable_writer_cannot_replace_even_a_partial_claim_file(tmp_path):
    path=tmp_path/'synthetic-only-exclusive-record.json'
    def attempt(index):
        try:return o._exclusive_json(path,dict(synthetic=True,index=index))
        except FileExistsError:return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(attempt,range(4)))
    assert sum(x is not None for x in results)==1
    original=path.read_bytes()
    assert path.stat().st_mode&0o222==0
    with pytest.raises(FileExistsError):o._exclusive_json(path,{'replacement':True})
    assert path.read_bytes()==original
    partial=tmp_path/'partial.claim';partial.write_bytes(b'{')
    with pytest.raises(FileExistsError):o._exclusive_json(partial,dict(synthetic=True))
    assert partial.read_bytes()==b'{'


def test_descriptor_freezes_pair_reference_program_and_quality_without_claim(plan,tmp_path,monkeypatch):
    monkeypatch.setattr(o,'OWNER_ROOT',tmp_path/'no-real-candidate-claims')
    ref=o.prepare_descriptor(plan);value=o.bound(ref)
    assert value['reference_actor']['sha256']==o.REFERENCE_SHA256
    assert value['program_templates']==plan.partition.value['program_templates']
    assert value['quality_case_inventory']==plan.partition.value['quality_case_inventory']
    assert value['training_authority'] is False
    assert value['public_admission_contract']=='original_BC_finite_inventories_exact_history_single_use_candidate_claim'
    assert value['collector_implementation']=='warehouse-v643-finite-calibration-pipeline.v1'
    assert not o.claim_path().exists()
    with pytest.raises(FileExistsError):o.prepare_descriptor(plan)


def test_claim_compiler_requires_exact_new_history_clearance_but_does_not_write(plan,tmp_path,monkeypatch):
    h=helper();h.empty_roots(tmp_path,monkeypatch)
    s.audit_seeds(plan)
    monkeypatch.setattr(o,'OWNER_ROOT',tmp_path/'no-real-candidate-claims')
    ref=o.prepare_descriptor(plan)
    value=o.compile_claim_record(plan,ref)
    assert value['schema']==o.CLAIM_VERSION and value['candidate_id']==o.CANDIDATE
    assert value['partition']==plan.partition.ref and value['single_use_even_if_interrupted'] is True
    assert value['training_authority'] is False and not o.claim_path().exists()


@pytest.fixture
def ledger(plan,tmp_path):
    ledger=o.ResetLedger(plan,tmp_path/'synthetic-predebits.jsonl')
    yield ledger
    ledger.close()


def test_reset_predebits_are_finite_ordered_unique_and_bound_to_exact_actual_seed(ledger,tmp_path):
    with pytest.raises(RuntimeError,match='initialization'):
        ledger.predebit_reset('raw',ledger.streams[0][0],scope='train')
    ledger.predebit_initialization('raw')
    with pytest.raises(RuntimeError,match='cannot repeat'):ledger.predebit_initialization('raw')
    with pytest.raises(RuntimeError,match='finite case order'):
        ledger.predebit_reset('raw',ledger.streams[0][1],scope='train')
    record=ledger.predebit_reset('raw',ledger.streams[0][0],scope='train')
    with pytest.raises(RuntimeError,match='previous actual reset'):
        ledger.predebit_reset('raw',ledger.streams[1][0],scope='train')
    with pytest.raises(RuntimeError,match='safe boundary'):ledger.snapshot()
    initial=helper().helpers().write(tmp_path/'synthetic-not-a-physics-proof.json',dict(synthetic_only=True))
    with pytest.raises(RuntimeError,match='seed'):
        ledger.bind_actual_reset('raw',record['sha256'],dict(seed=0,episode_id=1),initial_environment_ref=initial)
    actual=dict(seed=ledger.streams[0][0]['scenario_seed'],episode_id=1)
    ledger.bind_actual_reset('raw',record['sha256'],actual,initial_environment_ref=initial)
    with pytest.raises(RuntimeError,match='pending'):
        ledger.bind_actual_reset('raw',record['sha256'],actual,initial_environment_ref=initial)
    snapshot=ledger.snapshot()
    assert snapshot['reset_cursors']['raw']==[1,0,0,0,0,0,0,0]
    assert snapshot['training_authority'] is snapshot['resume_authority'] is False
    assert r.validate_ledger_snapshot(snapshot,ledger.records,plan=ledger.plan)


def test_immutable_saved_journal_prefix_survives_append_but_not_tamper(ledger):
    ledger.predebit_initialization('raw')
    snapshot=ledger.snapshot();records=deepcopy(ledger.records)
    ledger.predebit_initialization('regularized')
    assert r.validate_ledger_snapshot(snapshot,records,plan=ledger.plan)
    ledger.close()
    raw=ledger.path.read_bytes();ledger.path.write_bytes(b'X'+raw[1:])
    with pytest.raises(RuntimeError,match='prefix'):
        r.validate_ledger_snapshot(snapshot,records,plan=ledger.plan)


def test_missing_journal_rows_or_false_saved_cursors_fail(ledger):
    ledger.predebit_initialization('raw');snapshot=ledger.snapshot()
    with pytest.raises(RuntimeError,match='identity'):
        r.validate_ledger_snapshot(snapshot,[],plan=ledger.plan)
    snapshot['reset_cursors']['raw'][0]=1
    with pytest.raises(RuntimeError,match='cursors'):
        r.validate_ledger_snapshot(snapshot,ledger.records,plan=ledger.plan)


@pytest.fixture
def boundary(plan,ledger,tmp_path,monkeypatch):
    """Typed data-only envelope fixture, not a numerical resume certificate."""
    h=helper().helpers()
    monkeypatch.setattr(o,'OWNER_ROOT',tmp_path/'synthetic-owner-root')
    o.OWNER_ROOT.mkdir()
    descriptor_ref=o.prepare_descriptor(plan)
    claim_ref=h.write(o.claim_path(),dict(schema=o.CLAIM_VERSION,candidate_id=o.CANDIDATE,plan=plan.ref,
        partition=plan.partition.ref,source_sha256=plan.value['source_sha256'],descriptor=descriptor_ref,
        single_use_even_if_interrupted=True,automatic_resume=False,training_authority=False))
    owner_ref=h.write(Path(plan.ref['path']).parent/'calibration/raw/owner.json',
        dict(schema=o.OWNER_VERSION,candidate_id=o.CANDIDATE,variant='raw',claim=claim_ref,plan=plan.ref,
            descriptor=descriptor_ref,source_sha256=plan.value['source_sha256'],source_files=plan.value['source_files'],
            pair_seed=plan.partition.value['initialization_seed'],**{key:plan.partition.value[key] for key in
                ('train_case_inventory','program_evidence_inventory','quality_case_inventory','program_templates')}))
    ledger.predebit_initialization('raw')
    public=dict(version=r.public_reader.VERSION,source_sha256=r.file_sha(r.public_reader.__file__),
        owner=owner_ref,config=plan.partition.value['config'],full_training_state_sha256='d'*64,
        processed_joint_steps=0,safe_assembly_boundary=True,authorizes_training=False,
        pending=[[] for _ in range(8)],active=[None]*8,last_public=[None]*8)
    wrap=lambda value:dict(payload=value,payload_sha256=o.digest(value))
    reader=wrap(dict(version=r.physics.VERSION,source_files=r.physics.source_files(),owner=owner_ref,
        inventory=plan.partition.value['train_case_inventory'],public=wrap(public),environments=[None]*8,
        active=[None]*8,training_authorized=False))
    value=dict(version=r.VERSION,candidate_id=o.CANDIDATE,plan=plan.ref,partition=plan.partition.ref,
        owner=owner_ref,claim=claim_ref,source_files=plan.value['source_files'],source_sha256=plan.value['source_sha256'],
        variant='raw',numerical_boundary=dict(path='not-read-in-data-only-validation',sha256='e'*64),
        reader=reader,environments=[None]*8,active_cases=[None]*8,reset_ledger=ledger.snapshot(),
        reset_journal_records=deepcopy(ledger.records),program_source=None,provider=None,refits=[],updates=[],
        reserved_joint_steps=0,training_authority=False,resume_authority=False)
    numerical=dict(version=r.numerical.BOUNDARY_VERSION,candidate_id=o.CANDIDATE,
        reference_sha256=plan.partition.value['reference_actor']['sha256'],
        reference_buffer_sha256=plan.partition.value['reference_buffer']['sha256'],
        initialization_seed=plan.partition.value['initialization_seed'],joint_steps=0,training_state_sha256='d'*64,
        core=dict(PPO=dict(variant='raw',update_sequence=0,accepted_program_updates=dict(robot_1=0,robot_2=0))))
    return value,numerical


def test_full_data_envelope_binds_eight_slots_reader_numerical_journal_and_claim(boundary,plan):
    value,numerical=boundary
    result=r.validate_boundary(value,plan=plan,numerical_value=numerical)
    assert result==value and result is not value
    assert result['resume_authority'] is result['training_authority'] is False


@pytest.mark.parametrize('change',['omit_reader','slot_count','numerical_step','critic_fingerprint',
    'raw_gradient','claim_alias','refit_early','missing_journal','grant_resume','bool_step'])
def test_incomplete_or_mismatched_full_boundary_rejected(boundary,plan,change):
    value,numerical=deepcopy(boundary)
    if change=='omit_reader':value.pop('reader')
    if change=='slot_count':value['environments'].pop()
    if change=='numerical_step':numerical['joint_steps']=8
    if change=='critic_fingerprint':numerical['training_state_sha256']='a'*64
    if change=='raw_gradient':numerical['core']['PPO']['accepted_program_updates']['robot_1']=1
    if change=='claim_alias':value['claim']=value['owner']
    if change=='refit_early':value['refits']=[15000]
    if change=='missing_journal':value['reset_journal_records']=[]
    if change=='grant_resume':value['resume_authority']=True
    if change=='bool_step':numerical['joint_steps']=False
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        r.validate_boundary(value,plan=plan,numerical_value=numerical)


def test_named_private_slot_without_complete_environment_rng_is_rejected(boundary,plan):
    value,numerical=deepcopy(boundary)
    case=o.bound(plan.partition.value['train_case_inventory'])['cases'][0]
    fake=dict(payload={'synthetic_incomplete':True},payload_sha256=o.digest({'synthetic_incomplete':True}))
    value['environments'][0]=fake;value['active_cases'][0]=case
    reader=value['reader']['payload'];public=reader['public']['payload']
    reader['environments'][0]=fake;reader['active'][0]=case;public['active'][0]=case
    public['pending'][0]=[dict(frame=0,episode_id=case['episode_id'],slot=0,terminated=False,truncated=False)]
    public['last_public'][0]='f'*64
    reader['public']['payload_sha256']=o.digest(public)
    value['reader']['payload_sha256']=o.digest(reader)
    with pytest.raises(RuntimeError,match='complete environment/RNG'):
        r.validate_boundary(value,plan=plan,numerical_value=numerical)


def test_checkpoint_writer_rejects_declarations_before_creating_output(plan,tmp_path):
    with pytest.raises(RuntimeError,match='actual matching'):
        r.save_components(plan=plan,owner_ref=None,claim_ref=None,variant='raw',core={},reader={},ledger={},
            environments=[],active_cases=[],program_source=None,provider=None,refits=[],updates=[])
    assert not (Path(plan.ref['path']).parent/'calibration').exists()
