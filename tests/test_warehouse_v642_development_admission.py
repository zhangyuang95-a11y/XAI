"""Real-budget admission boundaries; fixtures are not experimental outcomes."""
import pytest
from types import SimpleNamespace
from copy import deepcopy
import json
from experiments.warehouse import v642_development_admission as a


def test_lock_cannot_be_constructed_from_passed_metadata():
    with pytest.raises(RuntimeError):a.BudgetLock(passed=True)


def test_missing_or_writable_budget_lock_rejected_before_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'audit_attempts',lambda *args:pytest.fail('bad input reached numerical replay'))
    with pytest.raises(RuntimeError):a.read_budget_lock(dict(path=str(tmp_path/'missing'),sha256='missing'))
    p=tmp_path/'fake.json';p.write_text('{"passed":true}')
    with pytest.raises(RuntimeError,match='writable'):a.read_budget_lock(a.pointer(p))


def test_gate_totals_are_complete_integer_episodes_not_rounded_means():
    rows=[dict(variant='raw',complete=True,shutdown=i==0,deliveries=14,robot_collisions=3) for i in range(75)]
    assert a._quality_totals(rows,'raw')==dict(episodes=75,deliveries=1050,collisions=225,shutdown_episodes=1)
    rows.pop()
    with pytest.raises(RuntimeError):a._quality_totals(rows,'raw')


def test_failed_attempt_does_not_create_a_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'OUTPUT',tmp_path/'locks')
    monkeypatch.setattr(a,'audit_attempts',lambda attempts:dict(all_three_streams_passed=False))
    with pytest.raises(RuntimeError,match='one or more'):a.issue_budget_lock([])
    assert not a.OUTPUT.exists()


def test_group_order_and_complete_three_streams_required(monkeypatch):
    monkeypatch.setattr(a,'source_files',lambda:{'synthetic':'only schema test'})
    for value in ([],[dict(joint_step_budget=300000,streams={})],
                  [dict(joint_step_budget=200000,streams={'development-0':{}})]):
        with pytest.raises(RuntimeError):a.audit_attempts(value)


def test_program_metrics_include_whole_tree_illegal_and_unproved_counts():
    m=dict(coverage=.5,expected_actionable_coverage=.48,agreement=.99,kl=.02,
           whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0)
    row={'metrics':{'feedback':dict(m),'explanation':dict(m)}}
    row['metrics']['explanation']['whole_tree_raw_illegal_count']=1
    assert a._program_metrics(row)['illegal_actions']==1


@pytest.fixture
def replayed_components(tmp_path,monkeypatch):
    """Explicit fake lower auditors: exercise orchestration, never real gates."""
    from backend.training.warehouse_v625_actor_training import PPOConfig
    from experiments.warehouse import v642_long_training_assets as assets
    from experiments.warehouse import v642_long_training_final_programs as final
    from experiments.warehouse import v642_long_training_quality as quality
    contexts={};gates={};pool_verdicts={};calls=[]
    monkeypatch.setattr(a,'source_files',lambda:{'synthetic':'no scientific evidence'})
    monkeypatch.setattr(a,'OUTPUT',tmp_path/'locks')
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));p.chmod(0o444);return a.pointer(p)
    def group(budget,*,failed_quality=False,failed_program=False):
        cohort=write(f'cohort-{budget}.json',{'synthetic':budget});streams={}
        for i,stream in enumerate(a.protocol.DEVELOPMENT_STREAMS):
            inventory=write(f'train-{budget}-{i}.json',dict(cases=[dict(scenario_seed=budget+10*i,
                action_seed=budget+100+i,slot=0)]))
            seed=write(f'seed-{budget}-{i}.json',dict(synthetic=True,stream=stream,budget=budget))
            plan=SimpleNamespace(ref=seed,config=PPOConfig(joint_step_budget=budget),value=dict(phase='development',
                stream_id=stream,train_case_inventory=inventory,initialization_seed=1000+i,optimizer_seed=2000+i))
            events=[dict(kind='initialized',variant=v,roles=dict(a.protocol.REFERENCE_ROLES),
                critic_sha256=str(i+1)*64,actor_optimizer_sha256='a'*64,critic_optimizer_sha256='b'*64)
                for v in a.protocol.VARIANTS]
            asset=write(f'assets-{budget}-{i}.json',dict(reference_directory=str(tmp_path),
                synthetic_stream=stream,synthetic_budget=budget))
            contexts[asset['sha256']]=dict(plan=plan,events=events,input_asset_refs=[],receipt_audit={
                'raw':dict(accepted_program_updates={r:0 for r in a.protocol.ROLES}),
                'regularized':dict(accepted_program_updates={r:1 for r in a.protocol.ROLES})})
            pool_pass=not(failed_program and i==1)
            fit=write(f'final-{budget}-{i}.json',dict(program_precheck_passed=pool_pass,synthetic_stream=i))
            pool_verdicts[fit['sha256']]=pool_pass
            gate=None
            if not failed_program:
                passed=not(failed_quality and i==1)
                outcomes=[dict(variant=v,complete=True,deliveries=15,robot_collisions=3,
                    shutdown=(not passed and v=='regularized' and j<4))
                    for v in ('reference','raw','regularized') for j in range(75)]
                collected=write(f'collected-{budget}-{i}.json',dict(outcomes=outcomes,
                    synthetic_stream=stream,synthetic_budget=budget))
                exitref=write(f'exit-{budget}-{i}.json',dict(synthetic=True))
                metric=dict(rows=90,coverage=.5,expected_actionable_coverage=.48,agreement=.99,kl=.02,
                    whole_tree_raw_illegal_count=0,whole_tree_path_uncertified_count=0)
                report=dict(quality_result=collected,supervisor_exit=exitref,completed_assets=asset,
                    seed_plan=seed,phase='development',stream_id=stream,joint_step_budget=budget,cohort=cohort,
                    quality_cases_sha256=a.digest(dict(synthetic_common_cohort=budget)),
                    development_pair_gate_passed=passed,input_asset_refs=[],program_metrics=dict(roles={
                        v:{r:dict(metrics=dict(feedback=deepcopy(metric),explanation=deepcopy(metric)))
                           for r in a.protocol.ROLES} for v in a.protocol.VARIANTS}))
                gate=write(f'gate-{budget}-{i}.json',report);gates[collected['sha256']]=report
            streams[stream]=dict(completed_assets=asset,final_programs=fit,quality_gate=gate)
        return dict(joint_step_budget=budget,streams=streams)
    monkeypatch.setattr(assets,'finalization_context',lambda ref,directory:contexts[ref['sha256']])
    monkeypatch.setattr(final,'_read_result',lambda *args,**kwargs:dict(inputs=[]))
    def replay(ref,exitref):calls.append(ref);return deepcopy(gates[ref['sha256']])
    monkeypatch.setattr(quality,'audit',replay)
    monkeypatch.setattr(final,'audit_program_pool',lambda ref,*args,**kwargs:dict(
        program_pool_passed=pool_verdicts[ref['sha256']],input_asset_refs=[]))
    return group,calls


def test_three_replayed_streams_issue_one_replayable_bound_lock(replayed_components):
    group,calls=replayed_components;attempts=[group(200000)]
    ref=a.issue_budget_lock(attempts);assert len(calls)==3
    lock=a.read_budget_lock(ref)
    assert len(calls)==6 and lock.value['selected_joint_step_budget']==200000
    assert lock.value['model_acceptance_passed'] is lock.value['pilot_opened'] is False
    lock.value['selected_joint_step_budget']=300000
    with pytest.raises(RuntimeError,match='budget lock changed'):lock.verify()


def test_one_failed_stream_is_not_hidden_by_two_passing_streams(replayed_components):
    group,_=replayed_components;report=a.audit_attempts([group(200000,failed_quality=True)])
    assert report['all_three_streams_passed'] is False and report['next_joint_step_budget']==300000
    assert report['formal_preparation_authorized'] is False


def test_program_failure_preserves_unmeasured_task_quality(replayed_components):
    group,calls=replayed_components;report=a.audit_attempts([group(200000,failed_program=True)])
    assert calls==[] and report['groups'][0]['task_quality'] is None
    assert not report['formal_preparation_authorized']


def test_higher_budget_cannot_replace_an_already_passing_budget(replayed_components):
    group,_=replayed_components
    with pytest.raises(RuntimeError,match='first all-three passing budget'):
        a.audit_attempts([group(200000),group(300000)])


def test_failed_lower_budget_then_passing_higher_budget_retains_same_starts(replayed_components):
    group,_=replayed_components
    report=a.audit_attempts([group(200000,failed_quality=True),group(300000)])
    assert report['selected_joint_step_budget']==300000 and len(report['groups'])==2
