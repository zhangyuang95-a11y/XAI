"""Finite partition and real replay wiring; all model/state fixtures synthetic."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
import torch
from experiments.warehouse import v642_training_partitions as p
from backend.training.warehouse_v6_actor import FEATURE_DIM
from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444)
    return p.pointer(path)


def inputs():
    train=dict(schema=p.TRAIN_VERSION,cases=[])
    for i in range(168):
        task=i%3+1
        train['cases'].append(dict(episode_id=f'train-{i}',scenario_seed=10*(10000+i)+task,
            task=task,action_seed=20000+i,slot=i%8,partition='train'))
    evidence=dict(schema=p.observed.INVENTORY_VERSION,scope=p.observed.SCOPE,
        expected_counts={s:1 for s in p.observed.online.SPLITS},cases=[])
    for i,split in enumerate(p.observed.online.SPLITS):
        task=i%3+1
        evidence['cases'].append(dict(episode=f'evidence-{split}',base_seed=30000+i,
            scenario_seed=10*(30000+i)+task,task=task,action_seed=40000+i,split=split))
    return train,evidence


def test_real_missing_winner_rejects_before_inventory_read_output_or_RNG(tmp_path,monkeypatch):
    monkeypatch.setattr(p.admission.v,'OUTPUT',tmp_path/'no-winner')
    def forbidden(*args,**kwargs):raise AssertionError('no A/B must not read a proposal or draw RNG')
    monkeypatch.setattr(p,'bound',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='complete reaped'):
        p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=None,evidence_inventory_ref=None,
            initialization_seed=0,optimizer_seed=1,output=tmp_path/'must-not-exist')
    assert not list(tmp_path.iterdir())


def test_finite_paired_schedule_and_four_evidence_splits_are_disjoint():
    train,evidence=inputs();result=p.validate_inventories(train,evidence,initialization_seed=0,optimizer_seed=1)
    assert result['training_episodes']==168 and len(result['complete_scene_values'])==172
    assert result['episodes_per_slot']=={str(i):21 for i in range(8)}
    assert result['raw_regularized_share_identical_training_and_evidence_inventories']
    assert not result['history_freshness_verified'] and not result['environment_authority']


@pytest.mark.parametrize('change',['same_scene_other_task','evidence_scene_alias','action_scene_alias','duplicate_ID',
    'initializer_alias','optimizer_alias','missing_slot','non_train','bool_task','missing_partition','wrong_count','sealed'])
def test_leakage_and_ambiguous_seed_partitions_rejected(change):
    train,evidence=inputs();c=evidence['cases'][0]
    if change=='same_scene_other_task':
        train['cases'][1]['task']=2;train['cases'][1]['scenario_seed']=100002
    if change=='evidence_scene_alias':c['base_seed']=10000;c['scenario_seed']=100001
    if change=='action_scene_alias':c['action_seed']=train['cases'][0]['scenario_seed']
    if change=='duplicate_ID':c['episode']=train['cases'][0]['episode_id']
    if change=='initializer_alias':c['action_seed']=0
    if change=='optimizer_alias':c['action_seed']=1
    if change=='missing_slot':train['cases']=[c for c in train['cases'] if c['slot']!=7]
    if change=='non_train':train['cases'][0]['partition']='validation'
    if change=='bool_task':c['task']=True
    if change=='missing_partition':evidence['cases'].pop()
    if change=='wrong_count':evidence['expected_counts']['fit']=2
    if change=='sealed':c['base_seed']=240100;c['scenario_seed']=2401001
    with pytest.raises(RuntimeError):p.validate_inventories(train,evidence,initialization_seed=0,optimizer_seed=1)


def test_schedule_that_cannot_cover_budget_even_at_full_horizon_is_rejected():
    train,evidence=inputs();train['cases']=train['cases'][:160]
    with pytest.raises(RuntimeError,match='bounded finite training stream'):
        p.validate_inventories(train,evidence,initialization_seed=0,optimizer_seed=1)


@pytest.mark.parametrize('budget',[200000,300000,400000])
def test_development_budget_cannot_bypass_initial_20k_calibration(budget):
    train,evidence=inputs()
    with pytest.raises(RuntimeError,match='only the 20k calibration'):
        p.validate_inventories(train,evidence,config=p.PPOConfig(joint_step_budget=budget),initialization_seed=0,optimizer_seed=1)


@pytest.fixture
def proposal(tmp_path,monkeypatch):
    training,evidence=inputs()
    train_ref=write(tmp_path/'train.json',training);evidence_ref=write(tmp_path/'evidence.json',evidence)
    prior=write(tmp_path/'old-confirmation.json',dict(cases=[dict(case_id='spent-confirm',base=50000,scenario_seed=500001,action_seed=60000)]))
    ready=SimpleNamespace(value=dict(confirmation=dict(plan=prior),actor=dict(sha256='a'*64),
        recipe=dict(program_evidence_counts=dict(evidence['expected_counts']))),refs=(prior,),
        payload_sha256='b'*64,verify=lambda:None)
    monkeypatch.setattr(p.admission,'load_readiness',lambda unused:ready);monkeypatch.setattr(p,'OUTPUT_ROOT',tmp_path/'proposals')
    result=p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=train_ref,evidence_inventory_ref=evidence_ref,
        initialization_seed=0,optimizer_seed=1,output=p.OUTPUT_ROOT/'one')
    return result,training,evidence


def test_preparation_binds_exact_same_inventory_refs_for_both_versions(proposal):
    plan,training,evidence=proposal;plan.verify()
    assert p.bound(plan.value['train_case_inventory'])==training and p.bound(plan.value['program_evidence_inventory'])==evidence
    assert plan.value['training_authority'] is False and plan.value['new_environment_resets']==0


def test_changed_inventory_is_rejected_even_if_its_new_contents_still_valid(proposal):
    plan,training,_=proposal;path=Path(plan.value['train_case_inventory']['path'])
    training['cases'][0]['episode_id']='renamed';path.chmod(0o644);path.write_text(json.dumps(training));path.chmod(0o444)
    with pytest.raises(RuntimeError,match='asset changed'):plan.verify()


def test_spent_confirmation_alias_prevents_any_output(proposal,tmp_path):
    plan,train,evidence=proposal
    c=evidence['cases'][0];c['base_seed']=50000;c['scenario_seed']=500001
    evidence_ref=write(tmp_path/'aliases-confirmation.json',evidence);output=p.OUTPUT_ROOT/'bad'
    with pytest.raises(RuntimeError,match='overlaps spent'):
        p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=plan.value['train_case_inventory'],
            evidence_inventory_ref=evidence_ref,initialization_seed=0,optimizer_seed=1,output=output)
    assert not output.exists()


def test_evidence_case_cannot_be_relabeled_to_train_before_physics(proposal,tmp_path):
    plan,training,evidence=proposal
    owner=write(tmp_path/'owner.json',dict(train_case_inventory=plan.value['train_case_inventory'],
        program_evidence_inventory=plan.value['program_evidence_inventory'],training_partition_proposal=plan.ref))
    reader=p.EnvironmentRolloutReplay(owner,plan.value['train_case_inventory'])
    source=evidence['cases'][0];false=dict(episode_id=source['episode'],scenario_seed=source['scenario_seed'],
        task=source['task'],action_seed=source['action_seed'],slot=0,partition='train')
    trace=write(tmp_path/'false-trace.json',dict(events=[dict(case=false)]))
    state=SimpleNamespace(initialization_seed=0,reference_sha256='a'*64)
    with pytest.raises(RuntimeError,match='cannot enter PPO gradients'):p.assemble_training_batch(plan,reader,trace,state)
    assert reader.public.processed_joint_steps==0 and not reader.used_episodes


def test_owner_must_bind_both_inventories_and_the_proposal(proposal,tmp_path):
    plan,*_=proposal
    owner=write(tmp_path/'bad-owner.json',dict(train_case_inventory=plan.value['train_case_inventory'],
        program_evidence_inventory=plan.value['program_evidence_inventory'],training_partition_proposal={'path':'wrong','sha256':'0'*64}))
    reader=p.EnvironmentRolloutReplay(owner,plan.value['train_case_inventory'])
    state=SimpleNamespace(initialization_seed=0,reference_sha256='a'*64)
    with pytest.raises(RuntimeError,match='owner does not bind'):p.assemble_training_batch(plan,reader,None,state)


@pytest.mark.parametrize('change',[None,'same_shape_other_source','renamed_episodes','changed_features'])
def test_reference_KL_requires_admitted_common_fit_bytes_and_row_identity(proposal,tmp_path,change):
    plan,*_=proposal;file=tmp_path/'common-fit.npz'
    data=dict(encoded=np.zeros((2,2,FEATURE_DIM),np.float32),masks=np.ones((2,2,5),np.float32),
        episode_ids=np.array(['old-fit-1','old-fit-2']),metadata_json=np.array(json.dumps(dict(schema=BUFFER_SCHEMA,
            split='train',reference_sha256='a'*64))))
    np.savez_compressed(file,**data);file.chmod(0o444);ref=p.pointer(file)
    plan.readiness.value['reference']=dict(buffer=ref)
    buffer=p.ReferenceTrainingBuffer.load(ref['path'],ref['sha256'],'a'*64)
    if change=='same_shape_other_source':
        other=tmp_path/'same-data-other-path.npz';other.write_bytes(file.read_bytes())
        buffer=p.ReferenceTrainingBuffer.load(other,p.file_sha(other),'a'*64)
    if change=='renamed_episodes':buffer.episode_ids=('program-validation-1','program-validation-2')
    if change=='changed_features':buffer.encoded=buffer.encoded.copy();buffer.encoded[0,0,0]=1
    if change is None:assert p.check_reference_buffer(plan,buffer) is buffer
    else:
        with pytest.raises(RuntimeError,match='exact admitted|relabeled or changed'):p.check_reference_buffer(plan,buffer)


def test_successful_wrapper_uses_actual_environment_and_NN_replay(proposal,tmp_path):
    plan,cases,_=proposal
    spec=importlib.util.spec_from_file_location('partition_physics_fixture',Path(__file__).with_name('test_warehouse_v629_environment_replay.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0);model=helper.WarehouseV6Actor()
        with torch.no_grad():
            for role in helper.r.AGENTS:
                actor=model.actors[role]
                for parameter in actor.parameters():parameter.zero_()
                actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
                actor.tree.left_refs.zero_();actor.tree.right_refs.zero_();actor.tree.leaf_logits[0,4]=1000.;actor.gate[-1].bias.fill_(1.)
        ref=helper.export_actor(model,tmp_path/'reference.npz',pair_seed=0,variant='raw',candidate_id='synthetic-partitions-only')
        # The fake prerequisite was used only to exercise pure proposal schema;
        # bind its Actor pointer now for this explicit synthetic replay.
        plan.readiness.value['actor']['sha256']=ref['sha256']
        state=helper.initialize(ref);actor=helper.FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])
        owner=write(tmp_path/'real-reader-owner.json',dict(train_case_inventory=plan.value['train_case_inventory'],
            program_evidence_inventory=plan.value['program_evidence_inventory'],training_partition_proposal=plan.ref))
        events=[];resets={};ends=[None]*8;envs=[None]*8
        for index in range(1024):
            slot=index%8;ordinal=(index//8)//120;case=cases['cases'][ordinal*8+slot]
            if envs[slot] is None:
                envs[slot]=helper.synthetic_environment(case,ordinal+1)
                resets[case['episode_id']]=write(tmp_path/(case['episode_id']+'-initial.json'),helper.r.capture_environment(envs[slot]))
            env=envs[slot];pre=helper.r.public_pre_action_snapshot(env.state)
            actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
            _,rewards,terminated,truncated,info=env.step(actions)
            events.append(dict(slot=slot,case=case,pre_state_public=pre,post_state_public=helper.r.public_pre_action_snapshot(env.state),
                proposed_actions=actions,action_override_agents=info['action_override_agents'],rewards=rewards,
                terminated=terminated,truncated=truncated,shutdowns=list(info['shutdowns'])))
            if index>=1016:ends[slot]=write(tmp_path/f'slot-{slot}-end.json',helper.r.capture_environment(env))
            if terminated or truncated:envs[slot]=None
        trace=write(tmp_path/'physical-trace.json',dict(schema='warehouse-v629-collected-PPO-public-frames.v1',
            partition='train',owner=owner,config=asdict(p.PPOConfig()),start_joint_step=0,end_joint_step=1024,actor=p.pointer(ref['path']),
            rollout_state=dict(actor_role_sha256=ref['role_sha256'],critic_sha256=helper.critic_initialization_sha256(state.model),actor_updates=0,critic_updates=0),
            physics=dict(schema=helper.r.PHYSICS_VERSION,resets=resets,end_snapshots=ends),events=events))
        state.advance_joint_steps(1024);reader=p.EnvironmentRolloutReplay(owner,plan.value['train_case_inventory'])
        batch,audit=p.assemble_training_batch(plan,reader,trace,state)
        assert len(batch['rows'])==1024 and len(batch['complete_cost_rows'])==960
        assert audit['physics_and_NN_replay']['public_NN_audit']['NN_action_checks']==2048
        assert audit['program_evidence_frames_in_gradient_buffers']==0 and not audit['training_authority']
        assert state.actor_updates==state.critic_updates==0
    finally:torch.set_num_threads(prior)


def test_scientific_evidence_counts_cannot_be_reduced_after_readiness(proposal,tmp_path):
    plan,_,_=proposal
    plan.readiness.value['recipe']['program_evidence_counts']={s:10 for s in p.observed.online.SPLITS}
    target=p.OUTPUT_ROOT/'undersized'
    with pytest.raises(RuntimeError,match='preregistered'):
        p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=plan.value['train_case_inventory'],
            evidence_inventory_ref=plan.value['program_evidence_inventory'],initialization_seed=0,optimizer_seed=1,output=target)
    assert not target.exists()
