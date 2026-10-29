"""Temporary synthetic arithmetic/NNs and restored engineering snapshots only."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pytest
import torch

from experiments.warehouse import v643_calibration_quality as q
from experiments.warehouse import v643_program_provider as p
from experiments.warehouse import v643_program_namespace as n


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(before)


def helper(name):
    spec=importlib.util.spec_from_file_location('v643_quality_test_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));path.chmod(0o444);return n.pointer(path)


def quality_cases():
    cases=q.arithmetic.cases(90100000,9010000000)
    return [{**c,'case_id':n.CANDIDATE+'-'+c['case_id']} for c in cases]


def outcomes():
    cases=quality_cases();actors={v:dict(path=v,sha256=v) for v in q.VARIANTS}
    rows=[dict(variant=v,case=c,actor=actors[v],complete=True,physical_frames=120,NN_action_checks=240,
        action_overrides=0,deliveries=15,robot_collisions=3,shutdown=False) for v in q.VARIANTS for c in cases]
    return cases,actors,rows


def test_unchanged_retention_thresholds_and_task_strata_do_not_claim_audit():
    cases,actors,rows=outcomes()
    for row in rows:
        if row['variant']=='raw':row['deliveries']={1:14,2:15,3:16}[row['case']['task']]
    result=q.evaluate_outcomes(rows,cases,actors)
    assert result['task_retention_passed']
    assert result['gates']['raw']['minimum_deliveries']==14.25
    assert result['gates']['raw']['maximum_collisions']==4
    assert result['by_task']['raw']['1']['delivery_reference_fraction']==pytest.approx(14/15)
    assert result['by_task']['raw']['3']['delivery_difference_from_reference']==1
    assert result['independently_audited_collection'] is result['model_acceptance_passed'] is result['pilot_opened'] is False


@pytest.mark.parametrize('change',['deliveries','collisions','shutdown','missing','duplicate','override','wrong_actor'])
def test_failed_or_incomplete_variant_cannot_be_pooled_away(change):
    cases,actors,rows=outcomes();reg=[r for r in rows if r['variant']=='regularized']
    if change=='deliveries':
        for row in reg:row['deliveries']=14
    elif change=='collisions':
        for row in reg:row['robot_collisions']=5
    elif change=='shutdown':
        for row in reg[:4]:row['shutdown']=True
    elif change=='missing':rows.pop()
    elif change=='duplicate':rows.append(deepcopy(rows[-1]))
    elif change=='override':reg[0]['action_overrides']=1
    else:reg[0]['actor']=actors['raw']
    if change in ('deliveries','collisions','shutdown'):
        result=q.evaluate_outcomes(rows,cases,actors)
        assert result['gates']['raw']['passed'] and not result['gates']['regularized']['passed']
        assert not result['task_retention_passed']
    else:
        with pytest.raises(RuntimeError):q.evaluate_outcomes(rows,cases,actors)


def test_old_scene_namespace_and_old_20k_pass_refused_before_physics(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='new candidate namespace'):
        q.validate_cases(q.arithmetic.cases(90100000,9010000000))
    old=write(tmp_path/'old-passed.json',dict(version='warehouse-v642-completed-calibration-assets.v1',
        training_completed=True,passed=True))
    monkeypatch.setattr(q,'_require_plan',lambda plan:plan)
    def forbidden(*args,**kwargs):pytest.fail('old 20k pass entered NN/physics/RNG')
    monkeypatch.setattr(q.FrozenWarehouseV6Actor,'load',forbidden)
    monkeypatch.setattr(q.physics,'restore_environment',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='actual new source-bound 20k pair'):q.audit_completed_assets(None,old)
    assert list(tmp_path.iterdir())==[Path(old['path'])]


@pytest.fixture
def tensors(tmp_path):
    before=torch.get_num_threads();torch.set_num_threads(1)
    folder=tmp_path/'tensor-fixture';folder.mkdir()
    try:yield helper('test_warehouse_v642_online_evidence').paired.__wrapped__(folder)
    finally:torch.set_num_threads(before)


def test_complete_checkpoint_role_hashes_match_actual_frozen_forward(tensors):
    state,*_=tensors
    value=dict(model=state.model.state_dict())
    assert q.checkpoint_role_hashes(value)==p.role_hashes(state.model)
    changed=deepcopy(value);changed['model']['actors.robot_1.gate.2.bias'][0]+=.1
    assert q.checkpoint_role_hashes(changed)['robot_1']!=q.checkpoint_role_hashes(value)['robot_1']


@pytest.mark.parametrize('fault',['support','nonfinite'])
def test_invalid_checkpoint_actor_tensors_cannot_supply_role_identity(tensors,fault):
    state,*_=tensors;value=dict(model=deepcopy(state.model.state_dict()))
    if fault=='support':value['model']['actors.robot_1.tree.support_fixed'][0]=0
    else:value['model']['actors.robot_1.gate.0.weight'][0,0]=float('nan')
    with pytest.raises(RuntimeError):q.checkpoint_role_hashes(value)


@pytest.mark.parametrize('fault',[None,'disconnected','wrong_final','old_method'])
def test_update_chain_binds_actual_initial_and_final_role_hashes(tmp_path,fault):
    initial={r:'a'*64 for r in q.ROLES};middle={r:'b'*64 for r in q.ROLES};final={r:'c'*64 for r in q.ROLES}
    changes=[dict(actor_updated=True,role_sha256_before=initial,role_sha256_after=middle),
        dict(actor_updated=True,role_sha256_before=middle,role_sha256_after=final)]
    if fault=='disconnected':changes[1]['role_sha256_before']=initial
    version='old-v642' if fault=='old_method' else q.numerical.VERSION
    ref=write(tmp_path/'synthetic-update.json',dict(version=version,separate_reward_and_cost_optimizers=True,
        epochs=[dict(updates=changes)]))
    expected=middle if fault=='wrong_final' else final
    if fault:
        with pytest.raises(RuntimeError):q._update_chain([ref],initial,expected)
    else:q._update_chain([ref],initial,expected)


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    generator=helper('test_warehouse_v629_observed_pool').recording.__wrapped__(tmp_path_factory)
    try:yield next(generator)
    finally:generator.close()


@pytest.fixture
def quality_episode(recorded,tmp_path):
    """Build a native-reward trace from a saved snapshot without any reset."""
    source=n.bound(recorded['source']);old_inventory=n.bound(source['inventory']);old=old_inventory['cases'][0]
    original=source['cases'][old['episode']]
    case=dict(case_id=n.CANDIDATE+'-recorded-quality',base=old['base_seed'],task=old['task'],
        scenario_seed=old['scenario_seed'],action_seed=old['action_seed'])
    actor_ref=original['behavior_actor'];actor=q.FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'])
    env=q.physics.restore_environment(n.bound(original['initial_environment']))
    env.config=q.participant_study_config(seed=case['scenario_seed'])
    initial=write(tmp_path/'initial.json',q.physics.capture_environment(env));events=[]
    while not(env.state.terminated or env.state.truncated):
        before=q.public_pre_action_snapshot(env.state);actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
        _,rewards,t,u,info=env.step(actions)
        events.append(dict(pre_state_public=before,post_state_public=q.public_pre_action_snapshot(env.state),
            proposed_actions=actions,action_override_agents=[],rewards=rewards,terminated=t,truncated=u,shutdowns=info['shutdowns']))
    final=write(tmp_path/'final.json',q.physics.capture_environment(env))
    trace=write(tmp_path/'trace.json',dict(schema=q.EPISODE_VERSION,case=case,behavior_actor=actor_ref,events=events))
    entry=dict(variant='reference',case=case,behavior_actor=actor_ref,trace=trace,
        initial_environment=initial,final_environment=final,reset_binding={})
    return actor,actor_ref,case,entry


def test_quality_audit_replays_actual_saved_physics_and_both_NNs_without_reset(quality_episode,monkeypatch):
    actor,ref,case,entry=quality_episode
    def forbidden(*args,**kwargs):pytest.fail('auditor constructed/reset an environment')
    monkeypatch.setattr(q.physics.WarehouseMultiAgentEnv,'__init__',forbidden)
    monkeypatch.setattr(q.physics.WarehouseMultiAgentEnv,'reset',forbidden)
    outcome,arrays=q._replay_episode(actor,ref,case,entry)
    assert outcome['complete'] and outcome['NN_action_checks']==2*outcome['physical_frames']
    assert outcome['action_overrides']==0 and arrays['x'].shape[1]==2 and arrays['p'].shape[-1]==5
    assert all(np.array_equal(arrays['p'][:,i],p.numerical.legacy.runtime_probabilities(actor,r,
        arrays['x'][:,i],arrays['m'][:,i])) for i,r in enumerate(q.ROLES))


@pytest.mark.parametrize('fault',['partial','reward','action','end_rng','config'])
def test_forged_quality_actions_outcomes_private_state_or_partial_episode_refused(quality_episode,tmp_path,fault):
    actor,ref,case,entry=quality_episode;entry=deepcopy(entry)
    if fault in ('partial','reward','action'):
        trace=n.bound(entry['trace'])
        if fault=='partial':trace['events'].pop()
        elif fault=='reward':trace['events'][0]['rewards']['robot_1']+=1
        else:trace['events'][0]['proposed_actions']['robot_1']='UP'
        entry['trace']=write(tmp_path/'wrong-trace.json',trace)
    else:
        key='final_environment' if fault=='end_rng' else 'initial_environment'
        env=q.physics.restore_environment(n.bound(entry[key]))
        if fault=='end_rng':env._rng.random()
        else:env.config=replace(env.config,robot_collision_points=-1.)
        entry[key]=write(tmp_path/'wrong-environment.json',q.physics.capture_environment(env))
    with pytest.raises((RuntimeError,ValueError)):q._replay_episode(actor,ref,case,entry)


@pytest.mark.parametrize('fault',[None,'missing_hook','changed_hook','wrong_binding'])
def test_all_225_quality_owner_predebits_match_physical_hook_rows(tmp_path,fault):
    from experiments.warehouse import v643_calibration_owner as owner
    cases=quality_cases()
    train=write(tmp_path/'train.json',dict(cases=[]));program=write(tmp_path/'program.json',dict(cases=[]))
    quality=write(tmp_path/'quality.json',dict(cases=cases));plan_ref=write(tmp_path/'synthetic-plan.json',dict(synthetic=True))
    plan=SimpleNamespace(ref=plan_ref,verify=lambda:None,partition=SimpleNamespace(value=dict(
        train_case_inventory=train,program_evidence_inventory=program,quality_case_inventory=quality,
        initialization_seed=0,optimizer_seed=1,reference_actor=dict(sha256=q.REFERENCE_SHA256))))
    ledger=owner.ResetLedger(plan,tmp_path/'synthetic-predebits.jsonl');hooks=[];entries={}
    initial=write(tmp_path/'synthetic-not-a-physical-snapshot.json',dict(synthetic=True))
    for case in cases:
        for variant in q.VARIANTS:
            debit=ledger.predebit_reset(variant,case,scope='quality')
            reset=dict(phase='quality',seed=case['scenario_seed'],episode_id=1,
                **{k:q.ACTION_PHYSICS_CONFIG[k] for k in ('map_layout_id','move_battery_cost','charge_per_wait')})
            record=ledger.bind_actual_reset(variant,debit['sha256'],reset,initial_environment_ref=initial)
            hooks.append(reset);entries[variant+':'+case['case_id']]=dict(variant=variant,case=case,
                reset_binding=record,initial_environment=initial)
    snapshot=ledger.snapshot();records=deepcopy(ledger.records);ledger.close()
    export=write(tmp_path/'ledger-export.json',dict(version=q.LEDGER_VERSION,candidate_id=n.CANDIDATE,
        plan=plan_ref,ledger=snapshot,records=records))
    if fault=='missing_hook':hooks.pop()
    elif fault=='changed_hook':hooks[0]={**hooks[0],'seed':hooks[0]['seed']+10}
    elif fault=='wrong_binding':entries[next(iter(entries))]['reset_binding']=records[-1]
    hook=tmp_path/'physical-hook-rows.jsonl';hook.write_text(''.join(json.dumps(r)+'\n' for r in hooks));hook.chmod(0o444)
    collection=dict(owner_ledger_snapshot=export,reset_journal=n.pointer(hook),cases=entries)
    if fault:
        with pytest.raises(RuntimeError):q._quality_resets(plan,collection,cases)
    else:q._quality_resets(plan,collection,cases)


def test_final_projection_targets_actual_20k_roles_and_cannot_be_repeated(tmp_path,monkeypatch):
    """Actual NN/solver numerics; the finite-admission seam is synthetic only."""
    fixture=helper('test_warehouse_v643_program').sample.__wrapped__(tmp_path,monkeypatch)
    regular=fixture['namespace'];raw=n.ProgramNamespace.from_owner(fixture['raw_owner']())
    namespaces=dict(raw=raw,regularized=regular);states={v:deepcopy(fixture['state']) for v in namespaces}
    pools={};actors={'reference':n.pointer(fixture['state'].reference_path)};roles={};boundaries={};owners={}
    roles['reference']=p.role_hashes(fixture['state'].model)
    for index,(variant,namespace) in enumerate(namespaces.items()):
        state=states[variant];state.joint_steps=20000
        with torch.no_grad():
            for role in q.ROLES:state.model.actors[role].gate[-1].bias.add_(.1*(index+1))
        exported=p.export_current_actor(state,tmp_path/(variant+'-20k.npz'),pair_seed=0,variant=variant,candidate_id=n.CANDIDATE)
        Path(exported['path']).chmod(0o444);actors[variant]=n.pointer(exported['path']);roles[variant]=p.role_hashes(state.model)
        owners[variant]=namespace.owner_ref;boundaries[variant]=dict(training_state_sha256=p.training_fingerprint(state))
        source=write(tmp_path/(variant+'-synthetic-program-source.json'),dict(synthetic=True))
        original=fixture['observed'];refs=tuple([source,*namespace.value['refs']])
        audit=dict(source=source,owner=namespace.owner_ref,observed_data_sha256=n.data_sha(original.parts),
            asset_manifest_sha256=n.digest(refs),synthetic_software_tensors_only=True,training_authority=False)
        pools[variant]=n.ObservedPool(namespace,source,original.parts,refs,audit,n.digest(audit))
        object.__setattr__(pools[variant],'_replayed_seal',n._REPLAYED)
    completed=write(tmp_path/'synthetic-not-an-actual-completion.json',dict(synthetic=True))
    fixed=dict(ref=completed,value=dict(owners=owners,actors=actors,actor_role_sha256=roles,
        program_sources={v:pool.source_ref for v,pool in pools.items()}),namespaces=namespaces,boundaries=boundaries,inputs=[completed])
    monkeypatch.setattr(q,'audit_completed_assets',lambda plan,ref:fixed)
    plan=SimpleNamespace(ref=write(tmp_path/'synthetic-final-plan.json',dict(synthetic=True)))
    before={v:p.training_fingerprint(s) for v,s in states.items()}
    ref=q.finalize_programs(plan,completed,namespaces=namespaces,states=states,observed=pools,
        last_program_refs={v:fixture['templates'] for v in states},output=fixture['root']/'final-programs/one')
    result=q._final_programs(plan,ref,fixed,observed=pools)
    assert result['all_training_program_gates_passed']
    assert result['quality_frames_used_for_fitting']==result['NN_updates']==result['environment_resets']==0
    assert result['model_acceptance_passed'] is result['pilot_opened'] is False
    assert all(n.bound(result['programs'][v][r])['target_actor_role_sha256']==roles[v][r] for v in states for r in q.ROLES)
    assert before=={v:p.training_fingerprint(s) for v,s in states.items()}
    with pytest.raises(RuntimeError,match='once-only'):
        q.finalize_programs(plan,completed,namespaces=namespaces,states=states,observed=pools,
            last_program_refs={v:fixture['templates'] for v in states},output=fixture['root']/'final-programs/two')
