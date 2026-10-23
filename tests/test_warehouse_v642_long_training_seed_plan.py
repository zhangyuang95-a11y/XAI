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


def test_restart_tags_repeat_only_common_start_and_keep_scene_tags_budget_specific():
    train,program=inventories(200000)
    v=dict(phase='development',stream_id='development-0',joint_step_budget=200000,initialization_seed=500,optimizer_seed=600)
    first,_=s.declared_calls(v,train,program);v['joint_step_budget']=300000;second,_=s.declared_calls(v,train,program)
    for a,b in zip(first['uses'],second['uses'],strict=True):
        if a['purpose'] in ('v642_long_imported_actors_new_critic_initialization','v642_long_PPO_shuffle_and_reference_mix'):
            assert a['reuse_tag']==b['reuse_tag']
        else:assert a['reuse_tag']!=b['reuse_tag']


def test_unproven_later_budget_cannot_claim_a_shared_restart():
    with pytest.raises(RuntimeError,match='independently verified common-start restart'):
        s._verified_restart(dict(phase='development',joint_step_budget=300000))


def test_restart_context_cannot_whitelist_scene_or_unknown_original_use(tmp_path):
    ref=put(tmp_path/'original-seed-report.json',dict(initialization_seed=500,optimizer_seed=600,
        scene=dict(scenario_seed=500,action_seed=700)))
    families=[dict(name=name,kind=kind,values=[value],intervals=[(value,value+1)],count=1,query_only=False)
              for name,kind,value in [('init','init',500),('optimizer','action_rng',600),('scene','env',700)]]
    matches=[dict(path=path,value=value,semantics=semantics,families=[family]) for path,value,semantics,family in
        [(['initialization_seed'],500,'init','init'),(['optimizer_seed'],600,'action_rng','optimizer'),
         (['scene','scenario_seed'],500,'env','init'),(['scene','action_seed'],700,'action_rng','scene')]]
    row=dict(path=ref['path'],sha256=ref['sha256'],resolved=True,classification='receipt_or_metadata_report_not_independently_replayed',
             detail=dict(candidate_matches=matches,reservation_overlaps=[]))
    evidence=dict(inventory=[row],families=families,missing_roots=[],selection_stats={})
    proof=dict(allowed_metadata_refs=[ref],reference_only_audit_refs=[],initialization_seed=500,optimizer_seed=600)
    reviewed,removed=s._restart_context(evidence,proof)
    assert len(removed)==2 and len(reviewed['inventory'][0]['detail']['candidate_matches'])==2
    with pytest.raises(RuntimeError,match='historical seed conflict'):
        s.legacy.conflict_free(reviewed)
    untrusted=dict(proof,allowed_metadata_refs=[])
    unchanged,removed=s._restart_context(evidence,untrusted)
    assert not removed and len(unchanged['inventory'][0]['detail']['candidate_matches'])==4


def test_restart_parent_requires_real_asset_reader_and_exact_same_stream_start(tmp_path,monkeypatch):
    from experiments.warehouse import v642_long_training_assets as assets
    proposal=put(tmp_path/'previous-proposal.json',dict(initialization_seed=500,optimizer_seed=600))
    plan_ref=put(tmp_path/'previous-plan.json',dict(initialization_seed=500,optimizer_seed=600))
    prior_assets=put(tmp_path/'previous-assets.json',dict(reference_directory=str(tmp_path)))
    bounded=put(tmp_path/'bounded.json',dict(synthetic_only=True))
    triaged=put(tmp_path/'triaged.json',dict(bounded_inventory=bounded,proposal=proposal))
    exact=put(tmp_path/'exact.json',dict(bounded_or_triaged_inventory=triaged))
    identity=dict(roles=dict(s.protocol.REFERENCE_ROLES),critic_sha256='a'*64,full_initial_training_state_sha256='b'*64,
        actor_optimizer_sha256='c'*64,critic_optimizer_sha256='d'*64,initial_joint_steps=0,initial_actor_updates=0,initial_critic_updates=0)
    prior=dict(phase='development',stream_id='development-0',initialization_seed=500,optimizer_seed=600,
        reference_actor=dict(sha256=s.protocol.REFERENCE_SHA256),proposal_ref=proposal,
        checked=dict(all_seed_arguments=[500,600,10000],episode_ids=['old-scene']),restart_proof=None)
    context=dict(plan=SimpleNamespace(ref=plan_ref,value=prior,config=s.PPOConfig(joint_step_budget=200000)),
        events=[dict(kind='initialized',variant=variant,**identity) for variant in s.VARIANTS],assets=dict(inputs=[]))
    called=[]
    def read(ref,directory):called.append(ref);return context
    monkeypatch.setattr(assets,'finalization_context',read)
    monkeypatch.setattr(s,'checked_clearance',lambda *args,**kwargs:dict(value=dict(exact=exact,restart_context=None)))
    decl=put(tmp_path/'restart.json',dict(version=s.RESTART_VERSION,phase='development',stream_id='development-0',
        from_budget=200000,to_budget=300000,initialization_seed=500,optimizer_seed=600,previous_assets=prior_assets))
    p=dict(phase='development',stream_id='development-0',joint_step_budget=300000,initialization_seed=500,
           optimizer_seed=600,restart_declaration=decl)
    proof=s._verified_restart(p)
    assert called==[prior_assets] and proof['expected_initial_identity']==identity
    assert proof['spent_seed_arguments']==[500,600,10000] and proof['scene_and_action_arguments_must_remain_fresh']
    prior['stream_id']='development-1'
    with pytest.raises(RuntimeError,match='same actual common Actor/critic/optimizer start'):
        s._verified_restart(p)


def formal_inputs(allowed,monkeypatch,*,budget=200000,stream='formal-0'):
    from tests.test_warehouse_v642_formal_training_admission import install_fake_development_auditor
    ref,_,calls=install_fake_development_auditor(allowed.directory,monkeypatch,budget=budget)
    p=s.bound(allowed.proposal)
    p.update(version=s.formal.PROPOSAL_VERSION,phase='formal',stream_id=stream,joint_step_budget=budget,development_budget_lock=ref)
    if budget!=200000:
        t,e=inventories(budget)
        p['train_case_inventory']=put(allowed.directory/'formal-training.json',t)
        p['program_evidence_inventory']=put(allowed.directory/'formal-program.json',e)
    proposal_ref=put(allowed.directory/'formal-proposal.json',p)
    return ref,proposal_ref,calls


def test_formal_missing_budget_lock_precedes_every_other_input_or_output(tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('missing formal lock reached other input or output')
    monkeypatch.setattr(s.protocol,'register_preparation',forbidden)
    monkeypatch.setattr(s,'_asset',forbidden);monkeypatch.setattr(s,'source_files',forbidden)
    for function in (lambda:s.prepare_formal(budget_lock_ref=None,calibration_precheck_ref=None,
            reference_directory=None,proposal_ref=None,run_id=None),
        lambda:s.check_formal(None,budget_lock_ref=None,calibration_precheck_ref=None,reference_directory=None)):
        with pytest.raises(RuntimeError,match='before other inputs'):function()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('budget',[200000,300000,400000])
def test_formal_prepare_and_check_use_real_token_from_explicit_FAKE_lower_development_auditor(allowed,monkeypatch,budget):
    lock,proposal_ref,calls=formal_inputs(allowed,monkeypatch,budget=budget)
    plan=s.prepare_formal(budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
        reference_directory=allowed.directory/'unused',proposal_ref=proposal_ref,run_id='synthetic-formal-only')
    assert type(plan.formal_authorization) is s.formal.FormalTrainingAdmission
    assert plan.value['formal_admission']==plan.formal_authorization.value and plan.value['development_budget_lock']==lock
    assert plan.config.joint_step_budget==budget and plan.value['restart_proof'] is plan.value['restart_declaration'] is None
    assert plan.value['phase']=='formal' and plan.value['stream_id']=='formal-0'
    assert plan.value['maximum_explicit_resets']==2*(plan.value['checked']['training_episodes']+1650)
    assert lock in plan.refs and not plan.value['training_authority'] and not plan.value['sealed_authority']
    restored=s.check_formal(plan.ref,budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
        reference_directory=allowed.directory/'unused')
    assert restored.value==plan.value and len(calls)==2  # actual lock issue+read, genuine-token cache thereafter.
    with pytest.raises(RuntimeError,match='ordinary development reader'):
        s.check(plan.ref,calibration_precheck_ref=allowed.ref,reference_directory=allowed.directory/'unused')


@pytest.mark.parametrize('change',['different_budget','fourth_stream','restart','another_lock'])
def test_formal_proposal_constraints_fail_before_reading_seed_inventories(allowed,monkeypatch,change):
    lock,proposal_ref,_=formal_inputs(allowed,monkeypatch)
    p=s.bound(proposal_ref)
    if change=='different_budget':p['joint_step_budget']=300000
    elif change=='fourth_stream':p['stream_id']='formal-3'
    elif change=='restart':p['restart_declaration']=dict(passed=True)
    else:p['development_budget_lock']=dict(lock,sha256='f'*64)
    invalid=put(allowed.directory/'invalid-formal.json',p)
    original=s.validate_inventories
    def forbidden(*a,**kw):pytest.fail('invalid formal phase read/validated seeds')
    monkeypatch.setattr(s,'validate_inventories',forbidden)
    with pytest.raises(RuntimeError,match='exact real budget lock'):
        s.prepare_formal(budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
            reference_directory=allowed.directory/'unused',proposal_ref=invalid,run_id='must-not-exist')
    assert not (s.OUTPUT_ROOT/'must-not-exist').exists()


def test_formal_global_single_use_and_four_root_audit_have_no_restart_freshness_exception(allowed,monkeypatch):
    lock,proposal_ref,_=formal_inputs(allowed,monkeypatch)
    plan=s.prepare_formal(budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
        reference_directory=allowed.directory/'unused',proposal_ref=proposal_ref,run_id='formal-exact-audit')
    root=allowed.directory/'formal-four-empty-worktrees'/'XAI-v6-integration'
    for name in s.HISTORY_NAMES:(root.parent/name).mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root)
    registry=put(allowed.directory/'formal-empty-registry.json',dict(registered_trace_paths=[]))
    monkeypatch.setattr(s,'REGISTRY_REF',registry)
    clearance=s.audit(plan);exact=s.bound(clearance['value']['exact'])
    assert len(exact['scan_roots'])==4 and clearance['value']['restart_context'] is None
    assert not clearance['value']['training_authority'] and plan.value['restart_proof'] is None
    put(Path(plan.value['global_claim_path']),dict(synthetic_consumed=True))
    with pytest.raises(RuntimeError,match='already consumed even if interrupted'):
        s.prepare_formal(budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
            reference_directory=allowed.directory/'unused',proposal_ref=proposal_ref,run_id='formal-second-run')
    assert not (s.OUTPUT_ROOT/'formal-second-run').exists()


@pytest.mark.parametrize('kind',['initialization','optimizer','training_scene','program_scene'])
def test_formal_four_root_exact_audit_rejects_previously_used_arguments(allowed,monkeypatch,kind):
    lock,proposal_ref,_=formal_inputs(allowed,monkeypatch)
    plan=s.prepare_formal(budget_lock_ref=lock,calibration_precheck_ref=allowed.ref,
        reference_directory=allowed.directory/'unused',proposal_ref=proposal_ref,run_id='formal-conflict-only')
    root=allowed.directory/'formal-conflicting-worktrees'/'XAI-v6-integration'
    for name in s.HISTORY_NAMES:(root.parent/name).mkdir(parents=True)
    monkeypatch.setattr(s,'ROOT',root)
    registry=put(allowed.directory/'formal-conflict-registry.json',dict(registered_trace_paths=[]))
    monkeypatch.setattr(s,'REGISTRY_REF',registry)
    value=(dict(initialization_seed=plan.value['initialization_seed']) if kind=='initialization' else
        dict(optimizer_seed=plan.value['optimizer_seed']) if kind=='optimizer' else
        dict(scenario_seed=allowed.train['cases'][0]['scenario_seed']) if kind=='training_scene' else
        dict(scenario_seed=allowed.program['cases'][0]['scenario_seed']))
    put(root/'output/warehouse-spent-formal-report.json',value)
    with pytest.raises(RuntimeError,match='historical seed conflict'):s.audit(plan)
    assert not (Path(plan.ref['path']).parent/'seed-ledger/clearance.json').exists()
