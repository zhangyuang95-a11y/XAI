"""Synthetic permission and temporary inventories only; no scientific runs."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from experiments.warehouse import v642_long_training_seed_plan as s


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    s.write_new(path,value);path.chmod(0o444);return s.pointer(path)


def inventories(budget):
    config=s.PPOConfig(joint_step_budget=budget);per_slot=(budget//8+119)//120
    train=[];program=[];index=0
    for slot in range(8):
        for number in range(per_slot):
            base=10000+index;task=index%3+1
            train.append(dict(episode_id=f'synthetic-train-{slot}-{number}',scenario_seed=10*base+task,
                              task=task,action_seed=1000000+index,slot=slot,partition='train'));index+=1
    counts=dict(fit=900,calibration=150,eligibility=300,validation=300)
    for split,count in counts.items():
        for number in range(count):
            base=10000+index;task=index%3+1
            program.append(dict(episode=f'synthetic-program-{split}-{number}',base_seed=base,
                scenario_seed=10*base+task,task=task,action_seed=1000000+index,split=split));index+=1
    return dict(schema=s.old_partitions.TRAIN_VERSION,cases=train),dict(schema=s.old_partitions.observed.INVENTORY_VERSION,
        scope=s.old_partitions.observed.SCOPE,expected_counts=counts,cases=program)


@pytest.fixture
def allowed(tmp_path,monkeypatch):
    ref=put(tmp_path/'fake-calibration-authorization.json',dict(synthetic_only=True))
    ready_ref=put(tmp_path/'fake-readiness.json',dict(synthetic_only=True))
    registration=SimpleNamespace(value=dict(calibration_precheck=ref),refs=(ref,),payload_sha256='a'*64,verify=lambda:None)
    ready=SimpleNamespace(value=dict(actor=dict(path='not-loaded',sha256=s.protocol.REFERENCE_SHA256)),
                          refs=(ready_ref,),payload_sha256='b'*64,verify=lambda:None)
    gated=[]
    def gate(receipt):
        assert receipt==ref;gated.append(receipt);return registration
    monkeypatch.setattr(s.protocol,'register_preparation',gate)
    monkeypatch.setattr(s.old_partitions.admission,'load_readiness',lambda _:ready)
    monkeypatch.setattr(s,'_known_spent',lambda *args:(set(),set()))
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path/'prepared-plans')
    source='experiments/warehouse/v642_long_training_seed_plan.py'
    monkeypatch.setattr(s,'source_files',lambda:{source:s.file_sha(s.SOURCE_ROOT/source)})
    train,program=inventories(200000)
    train_ref=put(tmp_path/'inventories/training.json',train)
    program_ref=put(tmp_path/'inventories/program.json',program)
    proposal=put(tmp_path/'inventories/proposal.json',dict(version=s.PROPOSAL_VERSION,phase='development',
        stream_id='development-0',joint_step_budget=200000,initialization_seed=500,optimizer_seed=600,
        train_case_inventory=train_ref,program_evidence_inventory=program_ref))
    return SimpleNamespace(ref=ref,proposal=proposal,train=train,program=program,gated=gated,
                           directory=tmp_path,registration=registration,ready=ready)


@pytest.fixture
def prepared(allowed):
    plan=s.prepare(calibration_precheck_ref=allowed.ref,reference_directory=allowed.directory/'unused',
                   proposal_ref=allowed.proposal,run_id='synthetic-pair-only')
    return plan,allowed


def test_real_gate_precedes_plan_proposal_reads_run_name_source_or_output(tmp_path,monkeypatch):
    seen=[]
    def denied(*args):seen.append('gate');raise RuntimeError('synthetic closed precheck')
    def forbidden(*args,**kwargs):pytest.fail('closed gate accessed proposal, source, seed or output')
    monkeypatch.setattr(s.protocol,'register_preparation',denied)
    monkeypatch.setattr(s,'bound',forbidden);monkeypatch.setattr(s,'_asset',forbidden)
    monkeypatch.setattr(s,'source_files',forbidden)
    with pytest.raises(RuntimeError,match='closed precheck'):
        s.prepare(calibration_precheck_ref=None,reference_directory=None,proposal_ref=None,run_id=None)
    with pytest.raises(RuntimeError,match='closed precheck'):
        s.check(None,calibration_precheck_ref=None,reference_directory=None)
    assert seen==['gate','gate'] and not list(tmp_path.iterdir())


@pytest.mark.parametrize('budget',[200000,300000,400000])
def test_all_registered_budgets_have_finite_slots_and_exact_1650_program_cases(budget):
    train,program=inventories(budget)
    checked=s.validate_inventories(train,program,config=s.PPOConfig(joint_step_budget=budget),
                                    initialization_seed=500,optimizer_seed=600)
    assert sum(checked['evidence_counts'].values())==1650
    assert len(checked['episodes_per_slot'])==8 and checked['training_episodes']<=budget
    assert checked['exhausted_finite_inventory_fails_closed'] and not checked['history_freshness_verified']


@pytest.mark.parametrize('change',['whole_scene','sealed','partition','counts','slot','optimizer_collision'])
def test_invalid_scene_or_streams_are_rejected(change):
    train,program=inventories(200000);init,opt=500,600
    if change=='whole_scene':
        first=train['cases'][0];program['cases'][0].update(base_seed=first['scenario_seed']//10,
            scenario_seed=first['scenario_seed'],task=first['task'])
    elif change=='sealed':
        train['cases'][0]['scenario_seed']=2401001
    elif change=='partition':train['cases'][0]['partition']='validation'
    elif change=='counts':program['cases'].pop()
    elif change=='slot':train['cases'][0]['slot']=8
    else:opt=init
    with pytest.raises(RuntimeError):
        s.validate_inventories(train,program,config=s.PPOConfig(joint_step_budget=200000),
                               initialization_seed=init,optimizer_seed=opt)


def test_prepare_freezes_exact_calls_common_start_and_unconsumed_cursors(prepared):
    plan,a=prepared;plan.verify()
    assert a.gated==[a.ref] and plan.proposal is plan and plan.config.joint_step_budget==200000
    p=plan.value;cases=s.bound(p['case_reservations']);declaration=s.bound(p['exact_plan'])
    parsed,_=s.exact.read_exact_plan(p['exact_plan']['path']);assert parsed==declaration
    assert p['maximum_environment_constructors']==p['maximum_explicit_resets']==2*(1672+1650)
    assert p['maximum_original_reset_invocations']==p['maximum_explicit_resets']
    assert p['maximum_original_environment_seed_operations']==2*p['maximum_explicit_resets']
    assert [{k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='raw']==[
            {k:v for k,v in c.items() if k!='variant'} for c in cases if c['variant']=='regularized']
    assert len(declaration['uses'])==4+4*len(cases)
    assert p['unconsumed_cursor']['program']=={'raw':0,'regularized':0}
    assert p['paired_initialization']['empty_Actor_and_critic_optimizers_required']
    assert not p['training_authority'] and not p['history_freshness_verified']
    assert Path(p['execution_output']).parent==Path(plan.ref['path']).parent
    assert Path(p['global_claim_path']) in s.exclusions(plan)
    assert Path(p['global_claim_path']).parent not in s.exclusions(plan)


def test_plan_reader_replays_gate_and_changed_inventory_is_refused(prepared):
    plan,a=prepared
    loaded=s.check(plan.ref,calibration_precheck_ref=a.ref,reference_directory=a.directory/'unused')
    assert loaded.ref==plan.ref and len(a.gated)==2
    path=Path(plan.value['train_case_inventory']['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='changed'):
        plan.verify()


def test_second_run_cannot_overwrite_prepared_plan(prepared):
    plan,a=prepared
    with pytest.raises(FileExistsError):
        s.prepare(calibration_precheck_ref=a.ref,reference_directory=a.directory/'unused',proposal_ref=a.proposal,
                   run_id=plan.value['run_id'])
    assert s.pointer(plan.ref['path'])==plan.ref


def test_global_claim_cannot_be_bypassed_by_a_different_run_id(prepared):
    plan,a=prepared
    put(Path(plan.value['global_claim_path']),dict(consumed_even_if_interrupted=True))
    with pytest.raises(RuntimeError,match='already consumed even if interrupted'):
        s.prepare(calibration_precheck_ref=a.ref,reference_directory=a.directory/'unused',
                  proposal_ref=a.proposal,run_id='different-synthetic-run')
    assert not (s.OUTPUT_ROOT/'different-synthetic-run').exists()


def test_spent_data_rejected_before_output(allowed,monkeypatch):
    monkeypatch.setattr(s,'_known_spent',lambda *args:({10000},set()))
    with pytest.raises(RuntimeError,match='overlaps spent'):
        s.prepare(calibration_precheck_ref=allowed.ref,reference_directory=allowed.directory/'unused',
                   proposal_ref=allowed.proposal,run_id='synthetic-spent')
    assert not s.OUTPUT_ROOT.exists()


def test_program_evidence_cannot_enter_replayed_ppo_or_cost_buffers(prepared,monkeypatch):
    plan,a=prepared
    class Reader:
        inventory_ref=plan.value['train_case_inventory'];config=plan.config
        owner_ref=put(a.directory/'synthetic-owner.json',dict(training_partition_proposal=plan.ref,
            train_case_inventory=plan.value['train_case_inventory'],program_evidence_inventory=plan.value['program_evidence_inventory']))
        def assemble(self,ref,state):
            return dict(rows=[dict(episode_id='foreign-program-case')],complete_cost_rows=[]),dict(synthetic_only=True)
    monkeypatch.setattr(s,'EnvironmentRolloutReplay',Reader)
    state=SimpleNamespace(initialization_seed=500,reference_sha256=s.protocol.REFERENCE_SHA256)
    program=a.program['cases'][0]
    wrong_case=dict(episode_id=program['episode'],scenario_seed=program['scenario_seed'],task=program['task'],
                    action_seed=program['action_seed'],slot=0,partition='train')
    trace=put(a.directory/'program-trace.json',dict(events=[dict(case=wrong_case)]))
    with pytest.raises(RuntimeError,match='program evidence or foreign cases'):
        s.assemble_training_batch(plan,Reader(),trace,state)
    trace=put(a.directory/'train-trace.json',dict(events=[dict(case=a.train['cases'][0])]))
    with pytest.raises(RuntimeError,match='non-training scenes'):
        s.assemble_training_batch(plan,Reader(),trace,state)


@pytest.fixture
def audited(prepared,monkeypatch):
    plan,a=prepared;root=a.directory/'four-empty-worktrees'/'XAI-v6-integration'
    for name in s.HISTORY_NAMES:(root.parent/name).mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root)
    registry=put(a.directory/'synthetic-registry.json',dict(registered_trace_paths=[]))
    monkeypatch.setattr(s,'REGISTRY_REF',registry)
    return plan,s.audit(plan),root


def test_real_indexed_triage_exact_pipeline_on_four_empty_temporary_roots(audited):
    plan,clearance,_=audited
    assert s.checked_clearance(plan)==clearance
    c=clearance['value'];e=s.bound(c['exact']);inv=s.bound(e['bounded_or_triaged_inventory'])
    assert c['actual_resets']==0 and not c['training_authority'] and not c['sealed_authority']
    assert len(e['scan_roots'])==4 and not inv['missing_roots']
    assert inv['auditor_sha256']==s.file_sha(s.metadata.__file__)
    assert inv['interval_index_source_sha256']==s.file_sha(s.interval_source.__file__)


@pytest.mark.parametrize('change',['new_history','expired','missing_root','registry'])
def test_final_delta_and_registry_lifetime_four_root_checks(audited,change):
    plan,clearance,root=audited
    if change=='new_history':
        put(root/'output/warehouse-new-report.json',dict(seed=999999))
    elif change=='expired':
        ref=clearance['clearance'];p=Path(ref['path']);value=clearance['value']
        issued=datetime.now(timezone.utc)-timedelta(days=2)
        value['issued_utc']=issued.isoformat();value['expires_utc']=(issued+timedelta(hours=24)).isoformat()
        p.chmod(0o644);p.write_text(json.dumps(value));p.chmod(0o444)
    elif change=='missing_root':(root.parent/s.HISTORY_NAMES[0]).rmdir()
    else:
        p=Path(s.REGISTRY_REF['path']);p.chmod(0o644);p.write_text('{}');p.chmod(0o444)
    with pytest.raises((RuntimeError,KeyError,ValueError)):
        s.checked_clearance(plan)
