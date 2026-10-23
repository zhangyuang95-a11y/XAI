"""Synthetic shared-cohort gates and finite replay, never real NN/scenes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from experiments.warehouse import v642_long_training_quality as q


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);q.write_new(path,value);path.chmod(0o444);return q.pointer(path)


def test_actual_training_gate_precedes_any_scene_arithmetic_output_or_model(tmp_path,monkeypatch):
    def denied(ref):raise RuntimeError('synthetic missing independently audited long training')
    def forbidden(*a,**kw):pytest.fail('closed long gate reached scene arithmetic, output or NN')
    monkeypatch.setattr(q,'assets',denied);monkeypatch.setattr(q.quality,'cases',forbidden)
    monkeypatch.setattr(q.FrozenWarehouseV6Actor,'load',forbidden)
    with pytest.raises(RuntimeError,match='missing independently'):
        q.prepare_cohort(streams=[dict(completed_assets=None,final_programs=None)]*3,run_id='x',base_start=None,action_start=None)
    assert not list(tmp_path.iterdir())


def test_final_program_gate_precedes_any_scene_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'assets',lambda r:{})
    def denied(*a):raise RuntimeError('synthetic one final program failed')
    monkeypatch.setattr(q,'final_programs',denied)
    monkeypatch.setattr(q.quality,'cases',lambda *a:pytest.fail('failed program allocated quality scenes'))
    with pytest.raises(RuntimeError,match='one final program failed'):
        q.prepare_cohort(streams=[dict(completed_assets=None,final_programs=None)]*3,run_id='x',base_start=None,action_start=None)
    assert not list(tmp_path.iterdir())


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    roots=[tmp_path/'empty-history'/str(i) for i in range(4)]
    for root in roots:root.mkdir(parents=True)
    monkeypatch.setattr(q,'OUTPUT',roots[0]/'output/warehouse_v642/long-quality')
    sources={name:q.file_sha(q.ROOT/name) for name in q.FILES}
    monkeypatch.setattr(q,'source_files',lambda:dict(sources))
    fixed={};programs={};streams=[]
    reference=put(tmp_path/'fakes/reference.json',dict(fake_NN='reference'))
    for stream in q.seeds.protocol.DEVELOPMENT_STREAMS:
        seed=put(tmp_path/'fakes'/stream/'seed-plan.json',dict(synthetic_only=True))
        report=put(tmp_path/'fakes'/stream/'assets.json',dict(phase='development',stream_id=stream,joint_step_budget=200000,seed_plan=seed))
        actors=dict(reference=reference,**{v:put(tmp_path/'fakes'/stream/(v+'.json'),dict(fake_NN=stream+v)) for v in ('raw','regularized')})
        roles={v:dict(robot_1=v+'-role-1',robot_2=v+'-role-2') for v in actors}
        fixed[report['sha256']]=dict(report=report,actors=actors,actor_role_sha256=roles,inputs=[report,*actors.values()])
        final=put(tmp_path/'fakes'/stream/'final.json',dict(synthetic_only=True))
        ps={v:{r:put(tmp_path/'fakes'/stream/(v+'-'+r+'-program.json'),dict(fake_program=True))
               for r in ('robot_1','robot_2')} for v in ('raw','regularized')}
        programs[final['sha256']]=dict(report=final,programs=ps,inputs=[final,*[p for pair in ps.values() for p in pair.values()]])
        streams.append(dict(completed_assets=report,final_programs=final))
    def read(ref):
        assert q.pointer(ref['path'])==ref
        for item in fixed[ref['sha256']]['inputs']:assert q.pointer(item['path'])==item
        return deepcopy(fixed[ref['sha256']])
    monkeypatch.setattr(q,'assets',read)
    monkeypatch.setattr(q,'final_programs',lambda ref,assets:deepcopy(programs[ref['sha256']]))
    value=q.prepare_cohort(streams=streams,run_id='synthetic-only',base_start=90100000,action_start=9010000000)
    monkeypatch.setattr(q,'history_roots',lambda:roots)
    registry=put(tmp_path/'fake-registry.json',dict(registered_trace_paths=[]));monkeypatch.setattr(q,'REGISTRY_REF',registry)
    return SimpleNamespace(value=value,roots=roots,fixed=fixed,programs=programs,streams=streams,directory=tmp_path)


def test_shared_cohort_three_streams_same_cases_and_exact_675_original_arguments(prepared):
    plans=[q.checked(ref)[0] for ref in prepared.value['plans'].values()]
    assert len({q.digest(p['cases']) for p in plans})==1
    assert all(len(p['cases'])==75 for p in plans)
    value=q.bound(plans[0]['exact_plan']);parsed,_=q.exact.read_exact_plan(plans[0]['exact_plan']['path'])
    assert parsed==value and len(value['uses'])==2700
    assert value['planned_resets']==675 and value['independent_scenes']==75
    assert len({r['group'] for r in value['uses']})==675
    assert len({r['value'] for r in value['uses'] if r['argument']=='reset(seed)'})==75
    assert all(prepared.value['cohort']['sha256'] in r['reuse_tag'] for r in value['uses'])
    with pytest.raises(RuntimeError,match='single-stream'):q.prepare(None)


@pytest.mark.parametrize('change',['source','unknown_inside_exclusion','case','formal'])
def test_changed_sources_unmanaged_exclusion_content_or_identity_fail_before_reset(prepared,change):
    ref=prepared.value['plans']['development-0'];p,folder=q.checked(ref)
    if change=='source':
        root=folder.parent;name=next(iter(p['source_files']));path=root/'source'/name
        path.chmod(0o644);path.write_text('# changed source')
        with pytest.raises(RuntimeError,match='source snapshot'):q.checked(ref)
    elif change=='unknown_inside_exclusion':
        (folder.parent/'unregistered-warehouse-seeds.json').write_text('{"scenario_seed":123}')
        with pytest.raises(RuntimeError,match='unregistered file'):q.managed_tree(p)
    elif change=='case':
        p['cases'][0]['action_seed']+=1
        path=Path(ref['path']);path.chmod(0o644);path.write_text(json.dumps(p));path.chmod(0o444)
        with pytest.raises(RuntimeError):q.checked(q.pointer(path))
    else:
        with pytest.raises(RuntimeError,match='formal'):q.identity_name(dict(phase='formal',stream_id='formal-0',joint_step_budget=200000))
    assert not q.claim_path(p).exists()


def test_one_shared_exact_clearance_accepts_only_registered_cohort_outputs(prepared):
    ref=prepared.value['plans']['development-0'];clear=q.audit_seeds(ref)
    for plan in prepared.value['plans'].values():assert q.clearance(plan,execution=True)[0]==clear
    p,_=q.checked(ref);assert q.bound(clear)['environment_reset_ceiling']==675
    assert q.cohort_claim_path(200000).is_file() and not q.claim_path(p).exists()
    (prepared.roots[0]/'new-warehouse-receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='historical metadata changed'):q.clearance(ref,execution=True)


def test_prior_used_seed_is_never_reauthorized_by_a_new_cohort(prepared):
    (prepared.roots[0]/'prior-warehouse-receipt.json').write_text('{"scenario_seed":901000001}')
    with pytest.raises(RuntimeError,match='historical seed conflict|unresolved metadata'):
        q.audit_seeds(prepared.value['plans']['development-0'])
    assert not q.cohort_claim_path(200000).exists()


def install_fakes(monkeypatch,program_passed):
    monkeypatch.setattr(q,'FEATURE_DIM',1)
    from env.warehouse.reset_audit_v614 import record_successful_reset
    class Env:
        def __init__(self,config):self.config=config
        def reset(self,*,seed):
            self.state=SimpleNamespace(frame=0,terminated=False,truncated=False,total_deliveries=6,robot_collision_events=0,shutdown_count=0)
            c=q.ACTION_PHYSICS_CONFIG
            record_successful_reset(seed,1,c['map_layout_id'],c['move_battery_cost'],c['charge_per_wait'])
        def step(self,actions):
            self.state.frame=1;self.state.truncated=True
            return None,dict(robot_1=0.,robot_2=0.),False,True,dict(proposed_actions=actions,action_override_agents=[],shutdowns=[])
    actor=SimpleNamespace(act=lambda state,**kw:(dict(robot_1='WAIT',robot_2='WAIT'),None),
                          logits=lambda obs,m:np.zeros((2,5)))
    monkeypatch.setattr(q,'WarehouseMultiAgentEnv',Env);monkeypatch.setattr(q.FrozenWarehouseV6Actor,'load',lambda *a,**kw:actor)
    monkeypatch.setattr(q,'_set_dual_nn_mode',lambda env:None)
    monkeypatch.setattr(q,'capture_environment',lambda env:dict(synthetic_only=True,frame=env.state.frame))
    monkeypatch.setattr(q,'public_pre_action_snapshot',lambda state:dict(frame=state.frame))
    monkeypatch.setattr(q,'encode_public_observation',lambda state:np.zeros((2,1),np.float32))
    monkeypatch.setattr(q,'action_masks',lambda state:np.ones((2,5),np.float32))
    monkeypatch.setattr(q,'encode_actor_tree_features',lambda obs,m:np.zeros((2,1),np.float32))
    monkeypatch.setattr(q,'verify_frame',lambda actor,case,event,a,i:2)
    monkeypatch.setattr(q,'verify_physics',lambda *a:dict(synthetic_only=True))
    monkeypatch.setattr(q,'replay_public_episode',lambda case,rows,n,sha:dict(case=case,physical_frames=n,
        deliveries=6,robot_collisions=0,shutdown=False))
    monkeypatch.setattr(q,'score_programs',lambda *a:dict(synthetic_only=True,passed=program_passed))


@pytest.mark.parametrize('program_passed',[True,False])
def test_full_synthetic_225_episode_collection_and_repeatable_independent_audit(prepared,monkeypatch,program_passed):
    install_fakes(monkeypatch,program_passed)
    plan_ref=prepared.value['plans']['development-0'];q.audit_seeds(plan_ref)
    result_ref=q.collect(plan_ref);result=q.bound(result_ref)
    assert result['actual_reset_count']==len(result['entries'])==225
    assert result['development_pair_gate_passed'] is program_passed and result['task_retention_passed']
    exit_ref=put(Path(result_ref['path']).parent/'supervisor-exit.json',dict(synthetic_only=True,exit_code=0,result=result_ref,worker_pid=q.os.getpid(),finished_utc=q.datetime.now(q.timezone.utc).isoformat()))
    from experiments.warehouse import v642_long_stage_supervisor as supervisor
    def wait_proof(exit,result,*,stage):
        assert stage=='long-quality' and exit==exit_ref and result==result_ref;return [exit_ref]
    monkeypatch.setattr(supervisor,'verify_execution',wait_proof)
    audited=q.audit(result_ref,exit_ref);assert q.audit(result_ref,exit_ref)==audited
    assert audited['version']==q.AUDIT_VERSION and audited['complete_episodes']==225
    assert audited['independent_scenes']==75 and audited['replayed_role_actions']==450
    assert audited['development_pair_gate_passed'] is program_passed
    assert audited['quality_cases_sha256']==q.digest(q.checked(plan_ref)[0]['cases'])
    assert not audited['model_acceptance_passed'] and not audited['training_authority']
    with pytest.raises(RuntimeError,match='once-only'):q.collect(plan_ref)


def test_entire_cohort_675_finite_slots_share_one_clearance_and_keep_reference_repeats_explicit(prepared,monkeypatch):
    install_fakes(monkeypatch,True)
    first=prepared.value['plans']['development-0'];clear=q.audit_seeds(first);results=[]
    for ref in prepared.value['plans'].values():
        assert q.clearance(ref,execution=True)[0]==clear
        result=q.bound(q.collect(ref));results.append(result)
        assert result['development_pair_gate_passed'] and result['actual_reset_count']==225
    assert sum(r['actual_reset_count'] for r in results)==675
    assert len({q.digest(r['metrics']['quality']['reference']) for r in results})==1
    assert len({q.digest([entry['case'] for entry in r['entries'] if entry['variant']=='reference']) for r in results})==1
    assert len({r['cohort']['sha256'] for r in results})==1
    assert len({r['claim']['path'] for r in results})==3
    with pytest.raises(RuntimeError,match='single-use clearance'):q.audit_seeds(first)


def test_private_physics_replay_uses_recorded_state_without_new_constructor_or_reset(tmp_path,monkeypatch):
    case=dict(scenario_seed=101)
    state=SimpleNamespace(frame=0,episode_id=1,participant_controlled_agent_id='robot_1',terminated=False,truncated=False)
    actions=dict(robot_1='WAIT',robot_2='WAIT');rewards=dict(robot_1=0.,robot_2=0.)
    def step(request):
        assert request==actions;state.frame+=1;state.truncated=True
        return None,rewards,False,True,dict(shutdowns=[],action_override_agents=[])
    env=SimpleNamespace(state=state,config=q.participant_study_config(seed=101),step=step)
    initial=put(tmp_path/'initial.json',dict(synthetic_only=True,frame=0))
    final=put(tmp_path/'final.json',dict(synthetic_only=True,frame=1))
    monkeypatch.setattr(q,'restore_environment',lambda snapshot:env)
    monkeypatch.setattr(q,'public_pre_action_snapshot',lambda state:dict(frame=state.frame))
    monkeypatch.setattr(q,'capture_environment',lambda env:dict(synthetic_only=True,frame=env.state.frame))
    monkeypatch.setattr(q,'WarehouseMultiAgentEnv',lambda *a:pytest.fail('replay constructed or reset a new environment'))
    event=dict(pre_state_public=dict(frame=0),post_state_public=dict(frame=1),proposed_actions=actions,
        rewards=rewards,terminated=False,truncated=True,shutdowns=[])
    assert q.verify_physics(case,[event],initial,final)==dict(physical_frames=1,environment_constructors=0,new_environment_resets=0)


def test_task_strata_remain_reports_and_do_not_add_an_unregistered_per_task_gate():
    cases=q.quality.cases(90100000,9010000000)
    actors={v:dict(path=v,sha256=v) for v in q.quality.VARIANTS};rows=[]
    for v in actors:
        for c in cases:
            rows.append(dict(case=c,variant=v,actor=actors[v],complete=True,physical_frames=120,NN_action_checks=240,
                action_overrides=0,deliveries=6,robot_collisions=0,shutdown=v!='reference' and c['task']==3 and c['case_id'].endswith(('002','005'))))
    metrics=q.task_metrics(rows,cases,actors)
    assert metrics['quality']['raw']['shutdown_rate']==2/75
    assert metrics['by_task']['raw']['3']['shutdown_rate']==2/25
    assert metrics['task_retention_passed']
