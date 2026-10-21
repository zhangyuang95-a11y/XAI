"""Recorded synthetic complete states; no candidate scenes/reset/training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_observed_pool as r
from backend.training.warehouse_v622_actor_training import initialize_from_reference,export_current_actor


def helper(name):
    spec=importlib.util.spec_from_file_location('observed_pool_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444);return r.pointer(path)


def test_closed_A_B_prevents_any_source_replay_or_rng(tmp_path,monkeypatch):
    monkeypatch.setattr(r.admission.v,'OUTPUT',tmp_path/'missing')
    def forbidden(*args,**kwargs):raise AssertionError('missing A/B must precede source replay')
    monkeypatch.setattr(r.physics,'restore_environment',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):r.audit_observed_pool(None,reference_directory=tmp_path/'unused')
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        r.refit_observed_pair_at_boundary(reference_directory=tmp_path/'unused',owner_ref=None,state=None,observed=None,targets_ref=None,
            program_refs=None,output=tmp_path/'must-not-create',variant='regularized',epoch_id='closed')
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    old=helper('test_warehouse_v629_environment_replay');generator=old.recording.__wrapped__(tmp_path_factory)
    folder,actor,_,_,traces=next(generator);folder=folder/'observed';folder.mkdir()
    actor_ref=r.pointer(actor['path']);Path(actor_ref['path']).chmod(0o444)
    raw=[r.bound(ref) for ref in traces];episodes={};initial={}
    for batch in raw:
        initial.update(batch['physics']['resets'])
        for e in batch['events']:episodes.setdefault(e['case']['episode_id'],[]).append(e)
    selected=[(key,events) for key,events in episodes.items() if len(events)==120]
    assert len(selected)==16
    cases=[];records={}
    for i,(key,events) in enumerate(selected):
        split=r.online.SPLITS[i//4];task=i%3+1;base=230000+i
        case=dict(episode=f'observed-{i}',base_seed=base,scenario_seed=10*base+task,task=task,action_seed=330000+i,split=split)
        cases.append(case);env=r.physics.restore_environment(r.bound(initial[key]));env.config=r.physics.training_config(case['scenario_seed'])
        start=write(folder/(case['episode']+'-initial.json'),r.physics.capture_environment(env));saved=[]
        for event in events:
            event={k:deepcopy(event[k]) for k in ('pre_state_public','post_state_public','proposed_actions','action_override_agents','rewards','terminated','truncated','shutdowns')}
            # Explicit all-WAIT synthetic NN has probability one. Changing the
            # test's declared case seed does not create or reset any scene.
            _,rewards,t,u,info=env.step(event['proposed_actions'])
            assert rewards==event['rewards'] and t==event['terminated'] and u==event['truncated']
            saved.append(event)
        records[case['episode']]=dict(behavior_actor=actor_ref,initial_environment=start,
            final_environment=write(folder/(case['episode']+'-final.json'),r.physics.capture_environment(env)),
            trace=write(folder/(case['episode']+'-trace.json'),dict(schema=r.EPISODE_VERSION,case=case,behavior_actor=actor_ref,events=saved)))
    inventory=write(folder/'inventory.json',dict(schema=r.INVENTORY_VERSION,scope=r.SCOPE,expected_counts={s:4 for s in r.online.SPLITS},cases=cases))
    owner=write(folder/'owner.json',dict(scope='synthetic-test-only-no-training-authority',program_evidence_inventory=inventory))
    source=write(folder/'source.json',dict(schema=r.VERSION,scope=r.SCOPE,owner=owner,inventory=inventory,cases=records))
    ready=SimpleNamespace(verify=lambda:None)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(r.admission,'load_readiness',lambda unused:ready)
        observed=r.audit_observed_pool(source,reference_directory=folder/'unused')
    # Current soft target is a different NN from the probability-one source.
    state=initialize_from_reference(actor_ref['path'],actor_ref['sha256'],actor['role_sha256'],initialization_seed=0,actor_lr=3e-5)
    with torch.no_grad():
        for role in r.AGENTS:
            state.model.actors[role].tree.leaf_logits[0]=torch.tensor([-3.,-3.,-3.,-3.,2.9])
            state.model.actors[role].gate[-1].bias.fill_(10.)
    current=export_current_actor(state,folder/'current.npz',pair_seed=0,variant='regularized',candidate_id='synthetic-observed-target-only')
    Path(current['path']).chmod(0o444)
    yield dict(folder=folder,observed=observed,source=source,current=r.pointer(current['path']),state=state,ready=ready)
    generator.close()


@pytest.fixture(autouse=True)
def root(tmp_path,monkeypatch):monkeypatch.setattr(r,'TARGET_ROOT',tmp_path/'targets')


def make_target(a,name='current'):
    return r.relabel_current_NN(a['observed'],a['current'],r.TARGET_ROOT/name)


def test_source_audit_keeps_behavior_actions_full_physics_and_complete_episodes(recording):
    pool=recording['observed'];pool.verify()
    assert pool.audit['complete_source_episodes']==16 and pool.audit['joint_physical_frames']==1920
    assert pool.audit['current_policy_task_quality_estimate'] is None and pool.audit['source_policy_outcomes_only']
    assert pool.audit['new_environment_resets']==0 and not pool.audit['training_authority']
    assert all(np.all(p['executed']==4) and not p['x'].flags.writeable for p in pool.parts.values())


def test_relabelling_does_not_step_any_environment_or_overwrite_source_actions(recording,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('counterfactual labels must never be executed')
    monkeypatch.setattr(r.physics.WarehouseMultiAgentEnv,'__init__',forbidden)
    monkeypatch.setattr(r.physics.WarehouseMultiAgentEnv,'reset',forbidden);monkeypatch.setattr(r.physics.WarehouseMultiAgentEnv,'step',forbidden)
    ref=make_target(recording);value,refs=r.checked_targets(recording['observed'],ref)
    assert value['current_policy_trajectory'] is value['actual_actions_replaced'] is False
    assert value['counterfactual_draws_not_executed'] and value['current_policy_task_quality_estimate'] is None
    assert sum(n for role in value['counterfactual_difference_counts'].values() for n in role.values())>0
    assert all(np.all(p['executed']==4) for p in recording['observed'].parts.values())
    assert value['environment_resets']==value['NN_updates']==value['PPO_joint_steps']==0


@pytest.mark.parametrize('change',['scope','partial','behavior_action','reward','terminal_type','initial_config','end_rng'])
def test_source_replay_rejects_wrong_scope_incomplete_or_forged_physics(recording,tmp_path,monkeypatch,change):
    source=r.bound(recording['source']);case=next(iter(source['cases']));item=source['cases'][case]
    if change=='scope':source['scope']='confirmation'
    if change in ('partial','behavior_action','reward','terminal_type'):
        trace=r.bound(item['trace'])
        if change=='partial':trace['events'].pop()
        if change=='behavior_action':trace['events'][0]['proposed_actions']['robot_1']='UP'
        if change=='reward':trace['events'][0]['rewards']['robot_1']+=1
        if change=='terminal_type':trace['events'][0]['terminated']=0
        item['trace']=write(tmp_path/'false-trace.json',trace)
    if change in ('initial_config','end_rng'):
        key='initial_environment' if change=='initial_config' else 'final_environment'
        env=r.physics.restore_environment(r.bound(item[key]))
        if change=='initial_config':env.config=replace(env.config,robot_collision_points=-1.)
        else:env._rng.random()
        item[key]=write(tmp_path/'false-environment.json',r.physics.capture_environment(env))
    ref=write(tmp_path/'false-source.json',source);monkeypatch.setattr(r.admission,'load_readiness',lambda unused:recording['ready'])
    with pytest.raises((RuntimeError,ValueError)):r.audit_observed_pool(ref,reference_directory=tmp_path/'unused')


def test_cached_observed_arrays_cannot_be_relabelled_after_memory_mutation(recording):
    old=recording['observed'];parts=deepcopy(old.parts);changed=replace(old,parts=parts)
    parts['fit']['x'].flags.writeable=True;parts['fit']['x'][0,0,0]+=.01
    with pytest.raises(RuntimeError,match='pool, audit, binding or source mutated'):make_target({**recording,'observed':changed})


@pytest.mark.parametrize('change',['source_binding','owner_binding','omitted_reference'])
def test_cached_pool_keeps_complete_source_identity_and_reference_manifest(recording,change):
    old=recording['observed']
    if change=='source_binding':changed=replace(old,source_ref={'path':old.source_ref['path'],'sha256':'0'*64})
    if change=='owner_binding':changed=replace(old,owner_ref={'path':old.owner_ref['path'],'sha256':'0'*64})
    if change=='omitted_reference':changed=replace(old,refs=old.refs[:-1])
    with pytest.raises(RuntimeError,match='binding or source mutated'):changed.verify()


@pytest.mark.parametrize('change',['trajectory_claim','actual_actions','difference_count','dropped_part','omitted_asset','reset_claim','fresh_randomness'])
def test_counterfactual_package_replays_data_and_cannot_claim_executed_actions(recording,tmp_path,change):
    ref=make_target(recording);value=r.bound(ref)
    if change=='trajectory_claim':value['current_policy_trajectory']=True
    if change=='reset_claim':value['environment_resets']=1
    if change=='fresh_randomness':value['fresh_randomness_claimed']=True
    if change=='actual_actions':
        key='fit';original=value['actual_source_actions'][key];file=Path(original['path']);file.chmod(0o644)
        np.savez_compressed(file,executed=np.zeros_like(recording['observed'].parts[key]['executed']));file.chmod(0o444)
        updated=r.pointer(file);value['actual_source_actions'][key]=updated
        value['asset_refs']=[updated if item==original else item for item in value['asset_refs']]
    if change=='difference_count':value['counterfactual_difference_counts']['robot_1']['fit']+=1
    if change=='dropped_part':value['actual_source_actions'].pop('validation')
    if change=='omitted_asset':value['asset_refs'].pop()
    altered=write(tmp_path/'forged/result.json',value);write(tmp_path/'forged/completed.json',dict(result_sha256=altered['sha256']))
    with pytest.raises(RuntimeError):r.checked_targets(recording['observed'],altered)


def test_bound_current_evidence_has_explicit_scope_and_feeds_both_role_program_gradients(recording,tmp_path):
    h=helper('test_warehouse_v629');program,_=h.fixture(tmp_path)
    from backend.training.warehouse_v628_program import ParentProgram
    from backend.training.warehouse_v629_program import from_parent
    old=ParentProgram.from_dict(r.bound(program.protection['parent']),program.actor_role_sha256)
    old.role='robot_2';old.actor_role_sha256='b'*64
    second=from_parent(old,write(tmp_path/'parent-two.json',old.to_dict()))
    package=make_target(recording);evidence={}
    for p in (program,second):
        ref=write(tmp_path/(p.role+'-program.json'),p.to_dict())
        e=r.build_observed_evidence(ref,recording['observed'],package,epoch_id='synthetic-current')
        assert e.payload()['version']==r.EVIDENCE_VERSION and e.payload()['validation_actions_are_counterfactual']
        saved=write(tmp_path/(p.role+'-evidence.json'),e.payload())
        with pytest.raises(RuntimeError,match='unknown online evidence schema'):
            r.online.load_evidence(saved,current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)
        loaded=r.load_observed_evidence(saved,recording['observed'],current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)
        assert loaded.payload()==e.payload();evidence[p.role]=e
    from backend.training.warehouse_v629_training_feedback import feedback_loss
    part=recording['observed'].parts['fit'];target=r.bound(package)
    # These inputs are synthetic numerical probes, not actual PPO receipts;
    # this test only checks typed loss compatibility and positive gradients.
    sampled=[]
    for role in r.AGENTS:
        pool=r.bound(target['pools'][role]);file=pool['parts']['fit']['arrays']['path']
        with np.load(file,allow_pickle=False) as z:sampled.append(z['sampled'].copy())
    before=r.physics.training_fingerprint(recording['state'])
    loss=feedback_loss(recording['state'].model,torch.as_tensor(part['x'].copy()),torch.as_tensor(part['m'].copy()),
        torch.as_tensor(np.stack(sampled,axis=1)),joint_steps=18000,total_steps=20000,variant='regularized',epoch_id='synthetic-current',evidence=evidence)
    assert all(v>0 for v in loss.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert r.physics.training_fingerprint(recording['state'])==before and recording['state'].actor_updates==0


def test_scheduled_refit_preserves_counterfactual_source_binding_in_its_receipt(recording,tmp_path,monkeypatch):
    h=helper('test_warehouse_v629');first,_=h.fixture(tmp_path)
    from backend.training.warehouse_v628_program import ParentProgram
    from backend.training.warehouse_v629_program import from_parent
    parent=ParentProgram.from_dict(r.bound(first.protection['parent']),first.actor_role_sha256)
    parent.role='robot_2';parent.actor_role_sha256='b'*64
    second=from_parent(parent,write(tmp_path/'parent-role-two.json',parent.to_dict()))
    programs={p.role:p for p in (first,second)}
    refs={role:write(tmp_path/(role+'-initial-program.json'),p.to_dict()) for role,p in programs.items()}
    state=recording['state'];old_steps=state.joint_steps;state.joint_steps=15000
    ready=SimpleNamespace(value=dict(actor=r.pointer(state.reference_path),program_pair={role:dict(language=r.admission.language_lock(p))
        for role,p in programs.items()}),payload_sha256='a'*64,verify=lambda:None)
    monkeypatch.setattr(r.admission,'load_readiness',lambda unused:ready)
    monkeypatch.setattr(r.refit,'EXTRACTION_ROOT',tmp_path/'extraction')
    package=make_target(recording)
    try:
        result,ref=r.refit_observed_pair_at_boundary(reference_directory=tmp_path/'unused',owner_ref=recording['observed'].owner_ref,
            state=state,observed=recording['observed'],targets_ref=package,program_refs=refs,output=r.refit.EXTRACTION_ROOT/'fixed',
            variant='regularized',epoch_id='synthetic-refit')
        value=r.bound(ref)
        assert value['target_package']==package and value['observed_source']==recording['source']
        assert value['actual_PPO_feedback_counts_from_real_rollouts_only'] and value['current_policy_task_quality_estimate'] is None
        assert value['validation_actionable_coverage_scope']=='counterfactual_current_NN_draws_on_fixed_observed_states'
        assert result['NN_updates']==value['NN_updates']==0 and not value['training_authority']
    finally:state.joint_steps=old_steps
