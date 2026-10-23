"""Protocol arithmetic and mocked admission only; no real runs or seeds."""
from copy import deepcopy
from dataclasses import FrozenInstanceError,asdict
import pytest
from experiments.warehouse import v642_long_training_protocol as p


def pair(budget=200000,index=0,phase='development'):
    init=dict(reference_actor_sha256=p.REFERENCE_SHA256,actor_role_sha256=dict(p.REFERENCE_ROLES),
              critic_sha256='a'*64,Actor_optimizer_sha256='b'*64,critic_optimizer_sha256='c'*64,
              initialization_random_stream_sha256=str(index+1)*64,optimizer_random_stream_sha256=str(index+4)*64,
              scenario_sequence_sha256=str(index+7)*64)
    q=dict(episodes=100,deliveries=1500,collisions=280,shutdown_episodes=1)
    program=dict(feedback_coverage=.5,explanation_coverage=.5,expected_feedback_coverage=.45,
                 feedback_agreement=.98,explanation_agreement=.99,feedback_KL=.08,explanation_KL=.08,
                 illegal_actions=0,unproved_paths=0)
    return dict(stream_id=(p.DEVELOPMENT_STREAMS if phase=='development' else p.FORMAL_STREAMS)[index],
        checkpoint_id='final',joint_step_budget=budget,config=asdict(p.PPOConfig(joint_step_budget=budget)),
        start_joint_step=0,end_joint_steps={v:budget for v in p.VARIANTS},initialization={v:deepcopy(init) for v in p.VARIANTS},
        quality_cases_sha256='f'*64,quality={v:deepcopy(q) for v in p.VARIANTS},reference_quality=deepcopy(q),
        programs={v:{role:deepcopy(program) for role in p.ROLES} for v in p.VARIANTS},
        program_gradient_updates={'raw':{role:0 for role in p.ROLES},'regularized':{role:7 for role in p.ROLES}})


def group(budget=200000):
    return [pair(budget,index) for index in range(3)]


def test_immutable_recipe_copies_and_exact_budget_specific_windows():
    recipe=p.Protocol();first=recipe.payload();first['program']['maximum_leaves']=1
    assert recipe.payload()['program']==dict(method='warehouse-v642-fixed-language-protected-descendant-refit.v1',
        maximum_depth=12,maximum_leaves=256,maximum_public_sources_per_node_or_leaf=3,
        exact_confirmed_fold_zero_structure_and_support_lock_required=True,program_never_selects_environment_actions=True)
    with pytest.raises(FrozenInstanceError):recipe.version='modified'
    for budget in p.BUDGETS:
        config=p.PPOConfig(joint_step_budget=budget)
        assert config.extraction_steps==tuple(budget*n//100 for n in (75,84,93))
        assert config.program_coefficient(.80*budget)==0
        assert config.program_coefficient(.85*budget)==pytest.approx(.01)
    assert p.Protocol().payload()['streams_are_continuations_from_one_complete_v620_BC_not_independent_BC_reproductions']


@pytest.mark.parametrize('broken',['critic_sha256','Actor_optimizer_sha256','scenario_sequence_sha256','optimizer_random_stream_sha256'])
def test_raw_reg_must_share_complete_initialization_and_scenario(broken):
    row=pair();row['initialization']['regularized'][broken]='d'*64
    with pytest.raises(RuntimeError,match='paired Actor, critic'):
        p.validate_pair_declaration(row,budget=200000,phase='development')


def test_reference_continuation_cannot_be_renamed_independent_bc_or_resume_lower_budget():
    row=pair(300000);row['start_joint_step']=200000
    with pytest.raises(RuntimeError,match='no cross-budget continuation'):
        p.validate_pair_declaration(row,budget=300000,phase='development')
    row=pair();row['initialization']['raw']['reference_actor_sha256']='e'*64
    row['initialization']['regularized']=deepcopy(row['initialization']['raw'])
    with pytest.raises(RuntimeError,match='exact complete v620'):
        p.validate_pair_declaration(row,budget=200000,phase='development')


def test_actual_gradient_requirements_are_per_role_and_raw_strict_zero():
    row=pair();assert p.validate_pair_declaration(row,budget=200000,phase='development')['all_declared_gates_passed']
    row['program_gradient_updates']['raw']['robot_1']=1
    result=p.validate_pair_declaration(row,budget=200000,phase='development')
    assert not result['all_declared_gates_passed']
    row=pair();row['program_gradient_updates']['regularized']['robot_2']=0
    assert not p.validate_pair_declaration(row,budget=200000,phase='development')['all_declared_gates_passed']


def test_budget_order_and_all_three_rule_prevent_pooled_pass():
    assert p.development_budget_decision([])['next_budget']==200000
    failed=group();failed[2]['quality']['regularized']['shutdown_episodes']=6
    result=p.development_budget_decision([failed]);assert result['next_budget']==300000
    result=p.development_budget_decision([failed,group(300000)])
    assert result['declared_budget']==300000 and result['formal_preparation_authority'] is False
    with pytest.raises(RuntimeError,match='stop at the first'):
        p.development_budget_decision([group(),group(300000)])
    with pytest.raises(RuntimeError,match='budget must restart'):
        p.development_budget_decision([group(300000)])
    with pytest.raises(RuntimeError,match='all three complete'):
        p.development_budget_decision([group()[:2]])


def test_stream_independence_and_quality_cohort_checked():
    rows=group();rows[1]['initialization']=deepcopy(rows[0]['initialization'])
    with pytest.raises(RuntimeError,match='independently registered'):
        p.development_budget_decision([rows])
    rows=group();rows[2]['quality_cases_sha256']='e'*64
    with pytest.raises(RuntimeError,match='same quality cohort'):
        p.development_budget_decision([rows])


def test_each_budget_restarts_same_critic_optimizer_and_rng_not_just_same_actor():
    failed=group();failed[2]['quality']['regularized']['shutdown_episodes']=6
    next_budget=group(300000)
    for variant in p.VARIANTS:next_budget[0]['initialization'][variant]['critic_sha256']='d'*64
    with pytest.raises(RuntimeError,match='same stream-specific Actor, critic'):
        p.development_budget_decision([failed,next_budget])


def test_reference_retention_and_program_gates_are_not_replaced_by_absolute_quality():
    row=pair();row['quality']['raw']['deliveries']=1400  # >5.5, but <95% of15.
    assert not p.validate_pair_declaration(row,budget=200000,phase='development')['all_declared_gates_passed']
    row=pair();row['programs']['regularized']['robot_2']['explanation_coverage']=.4499
    assert not p.validate_pair_declaration(row,budget=200000,phase='development')['all_declared_gates_passed']


def quality_rows():
    rows=[]
    for index in range(3):
        full=pair(index=index,phase='formal')
        row={k:full[k] for k in ('stream_id','checkpoint_id','quality_cases_sha256','quality')}
        rows.append(row)
    return rows


def test_formal_selector_uses_worse_variant_and_never_reads_programs():
    rows=quality_rows()
    rows[0]['quality']['raw']['deliveries']=3000
    rows[0]['quality']['regularized']['deliveries']=1000
    rows[2]['quality']['raw']['deliveries']=1600
    rows[2]['quality']['regularized']['deliveries']=1600
    winner=p.select_quality_only(rows)
    assert winner['stream_id']=='formal-2' and winner['program_prescreen_may_change_selection'] is False
    assert winner['sealed_authority'] is False
    rows[0]['program_passed']=True
    with pytest.raises(RuntimeError,match='no program/preaudit inputs'):
        p.select_quality_only(rows)


def test_formal_selector_rejects_quality_failure_instead_of_rescuing_by_program():
    rows=quality_rows()
    for row in rows:row['quality']['raw']['shutdown_episodes']=11
    with pytest.raises(RuntimeError,match='no pair passes formal quality'):
        p.select_quality_only(rows)


def test_formal_streams_need_independent_initialization_optimizer_and_scenarios():
    rows=[pair(index=index,phase='formal') for index in range(3)]
    result=p.validate_formal_group_declaration(rows,budget=200000)
    assert result['program_diagnostics_may_not_reselect_pair'] and result['formal_preparation_authority'] is False
    for variant in p.VARIANTS:rows[1]['initialization'][variant]=deepcopy(rows[0]['initialization'][variant])
    with pytest.raises(RuntimeError,match='streams must be independent'):
        p.validate_formal_group_declaration(rows,budget=200000)


def test_execution_is_closed_even_with_fully_passing_declarations():
    decision=p.development_budget_decision([group()]);assert decision['declared_budget']==200000
    with pytest.raises(RuntimeError,match='not implemented; execution remains closed'):
        p.execution_authority(decision)
    with pytest.raises(RuntimeError,match='actual independent calibration replay'):
        p.PreparationRegistration()


def test_calibration_failed_gate_precedes_any_auditor_or_source_lookup(monkeypatch):
    ref=dict(path='synthetic-not-production',sha256='a'*64)
    monkeypatch.setattr(p,'pointer',lambda path:ref)
    monkeypatch.setattr(p,'bound',lambda _:dict(version=p.calibration.VERSION,calibration_precheck_passed=False))
    def forbidden(*args,**kwargs):pytest.fail('failed calibration cannot reach preparation or source work')
    monkeypatch.setattr(p.calibration,'audit',forbidden);monkeypatch.setattr(p,'source_files',forbidden)
    with pytest.raises(RuntimeError,match='before preparation'):
        p.register_preparation(ref)


def test_calibration_flag_alone_cannot_open_registration(monkeypatch):
    ref=dict(path='synthetic-not-production',sha256='a'*64)
    body=dict(version=p.calibration.VERSION,calibration_precheck_passed=True,model_acceptance_passed=False,
              pilot_opened=False,quality_result={},supervisor_exit={},input_asset_refs=[])
    monkeypatch.setattr(p,'pointer',lambda path:ref);monkeypatch.setattr(p,'bound',lambda _:body)
    monkeypatch.setattr(p,'source_files',lambda:{})
    monkeypatch.setattr(p.calibration,'audit',lambda *args:dict(body,changed=True))
    with pytest.raises(RuntimeError,match='did not exactly replay'):
        p.register_preparation(ref)


def test_mocked_replayed_registration_grants_only_symbolic_preparation(monkeypatch):
    ref=dict(path='synthetic-only-no-receipt-written',sha256='a'*64)
    body=dict(version=p.calibration.VERSION,calibration_precheck_passed=True,model_acceptance_passed=False,
              pilot_opened=False,quality_result=ref,supervisor_exit=ref,input_asset_refs=[])
    monkeypatch.setattr(p,'pointer',lambda path:ref);monkeypatch.setattr(p,'bound',lambda _:body)
    monkeypatch.setattr(p,'source_files',lambda:{})
    replayed=[]
    def replay(*args):replayed.append(args);return deepcopy(body)
    monkeypatch.setattr(p.calibration,'audit',replay)
    registered=p.register_preparation(ref)
    assert len(replayed)==1 and registered.value['symbolic_preparation_registered']
    assert registered.value['training_authority'] is registered.value['seed_allocation_authority'] is False
    registered.value['training_authority']=True
    with pytest.raises(RuntimeError,match='registration, source or authority changed'):
        registered.verify()
