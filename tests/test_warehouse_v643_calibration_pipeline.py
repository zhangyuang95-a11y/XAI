"""Real collector/physics/PPO wiring on synthetic states, no science run.

The physical fixtures are built with __new__, a control RNG0 and explicit
WarehouseState data. The real environment constructor/reset are forbidden.
The controlled reset stand-in only emits the real hook over that explicit
fixture; it never claims a sample from a declared scientific scenario seed.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pytest
import torch

from experiments.warehouse import v643_calibration_pipeline as c
from experiments.warehouse import v643_training_partitions as p
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor, export_actor
from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA


def helper(name):
    spec=importlib.util.spec_from_file_location('v643_pipeline_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_public_entry_refuses_before_plan_read_claim_output_or_RNG(tmp_path,monkeypatch):
    def forbidden(*_a,**_k):raise AssertionError('unreviewed public entry must not perform work')
    monkeypatch.setattr(c,'pointer',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    monkeypatch.setattr(c,'Environment',forbidden)
    with pytest.raises(RuntimeError,match='explicit new candidate plan'):
        c.run(reference_directory=tmp_path/'unused',plan_ref=None)
    assert not list(tmp_path.iterdir())


class FakeOwner:
    def __init__(self):
        self.config=c.PPOConfig();self.states={};self.events=[];self.refits={v:[] for v in c.seeds.VARIANTS}
    def live(self,**_):pass
    def initialize(self,v):self.states[v]=SimpleNamespace(joint_steps=0);self.events.append(('init',v))
    def collect_rollout(self,v):
        state=self.states[v]
        assert self.refits[v]==[s for s in self.config.extraction_steps if s<=state.joint_steps]
        self.events.append(('collect',v,state.joint_steps))
        state.joint_steps+=8*self.config.next_collection_length(state.joint_steps)
    def update(self,v):self.events.append(('update',v,self.states[v].joint_steps))
    def scheduled_refit(self,v,previous):
        step=self.states[v].joint_steps;self.refits[v].append(step);self.events.append(('refit',v,step,deepcopy(previous)))
        return {role:f'{v}-{step}-{role}' for role in previous}
    def checkpoint(self,v):self.events.append(('checkpoint',v,self.states[v].joint_steps))
    def pair_checkpoint(self):self.events.append(('pair-checkpoint',self.states['raw'].joint_steps))
    def final_assets(self):self.events.append(('final-assets',));return {'synthetic_only':True}


def test_finite_driver_preserves_all_extraction_warmup_and_checkpoint_boundaries():
    owner=FakeOwner();initial=dict(robot_1='one',robot_2='two')
    assert c.drive(owner,initial)=={'synthetic_only':True}
    assert initial==dict(robot_1='one',robot_2='two')
    assert [row[:3] for row in owner.events if row[0]=='refit']==[
        ('refit',v,s) for s in (15000,16800,18600) for v in ('raw','regularized')]
    for v in c.seeds.VARIANTS:
        assert [row[2] for row in owner.events if row[:2]==('checkpoint',v)]==[4000,15000,16800,18600,20000]
        assert owner.states[v].joint_steps==20000
    assert owner.events[-1]==('final-assets',)


def test_refit_failure_is_not_retried_and_no_later_batch_is_collected():
    owner=FakeOwner();calls=[]
    def fail(v,previous):calls.append((v,owner.states[v].joint_steps));raise RuntimeError('synthetic refit failed')
    owner.scheduled_refit=fail
    with pytest.raises(RuntimeError,match='refit failed'):c.drive(owner,dict(robot_1='one',robot_2='two'))
    assert calls==[('raw',15000)]
    assert owner.states['raw'].joint_steps==15000 and owner.states['regularized'].joint_steps<15000
    assert not any(row[0]=='final-assets' for row in owner.events)


def test_other_training_budget_cannot_be_selected_by_the_controller():
    owner=FakeOwner();owner.config=c.PPOConfig(joint_step_budget=200000)
    with pytest.raises(RuntimeError,match='fixed paired20k'):c.drive(owner,{})
    assert not owner.events


def test_quality_cannot_consume_a_scene_before_independent_final_program_validation(tmp_path):
    active=c.FiniteCalibrationOwner.__new__(c.FiniteCalibrationOwner)
    active.live=lambda **_:None;active.training_complete=lambda:True
    active.final_assets_ref={'synthetic_only':True};active.final_programs_ref=None
    active.quality_started=False;active.failed=False
    with pytest.raises(RuntimeError,match='fresh quality is once-only'):
        active.collect_quality()
    assert active.failed and not list(tmp_path.iterdir())


def test_program_observation_source_is_not_collected_from_an_earlier_nn(tmp_path):
    active=c.FiniteCalibrationOwner.__new__(c.FiniteCalibrationOwner)
    active.live=lambda **_:None;active.failed=False;active.pending={};active.pools={}
    active.ledger=SimpleNamespace(program_cursors={'raw':0})
    active.states={'raw':SimpleNamespace(joint_steps=14992)}
    active.cores={'raw':SimpleNamespace(processed_joint_steps=14992)};active.reserved={'raw':14992}
    with pytest.raises(RuntimeError,match='once-only at completed75percent'):
        active.collect_program_source('raw')
    assert active.failed and not list(tmp_path.iterdir())


@pytest.mark.parametrize('passed',[False,True])
def test_actual_measurement_verdict_is_preserved_without_becoming_model_admission(tmp_path,monkeypatch,passed):
    from experiments.warehouse import v643_calibration_quality as audit
    plan_ref=dict(path='synthetic-plan',sha256='a'*64);claim=dict(path='synthetic-claim',sha256='b'*64)
    measured=c._write(tmp_path/'synthetic-measured.json',dict(version=audit.VERSION,candidate_id=c.CANDIDATE,
        source_files=audit.source_files(),plan=plan_ref,claim=claim,calibration_precheck_passed=passed,
        independently_replayed_saved_physics_and_current_NN=True,
        quality_metrics=dict(task_retention_passed=passed),program_metrics=dict(passed=True)))
    refit=c._write(tmp_path/'synthetic-refit.json',dict(roles={r:dict(program={}) for r in c.feedback.ROLES}))
    active=SimpleNamespace(plan=SimpleNamespace(ref=plan_ref,value={'run_id':'synthetic'}),folder=tmp_path/'run',claim_ref=claim,
        refit_refs={v:[refit] for v in c.seeds.VARIANTS},namespaces={},states={},pools={},
        bind_final_programs=lambda _:None,collect_quality=lambda:{'synthetic_only':True},event=lambda *_a,**_k:None)
    monkeypatch.setattr(audit,'finalize_programs',lambda *_a,**_k:{'synthetic_only':True})
    monkeypatch.setattr(audit,'audit_collection',lambda *_a,**_k:measured)
    result=c.bound(c._evaluate_completed_pair(active,{'synthetic_only':True}))
    assert result['calibration_precheck_passed'] is passed
    assert result['candidate_closed_after_failed_calibration'] is (not passed)
    assert result['model_acceptance_passed'] is result['pilot_opened'] is False
    assert (active.folder/'calibration-closed.json').exists() is (not passed)


@pytest.fixture
def engineering_plan(tmp_path,monkeypatch):
    h=helper('test_warehouse_v643_training_partitions')
    original=h.partition.__wrapped__(tmp_path,monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for actor in model.actors.values():
            for parameter in actor.parameters():parameter.zero_()
            actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
            actor.tree.left_refs.zero_();actor.tree.right_refs.zero_();actor.tree.leaf_logits[0,4]=1000.
            actor.gate[-1].bias.fill_(1.)
    result=export_actor(model,tmp_path/'synthetic-control-reference.npz',pair_seed=0,variant='raw',candidate_id='synthetic-only')
    Path(result['path']).chmod(0o444);reference=c.pointer(result['path'])
    for module in (p,c.numerical,c.programs,c.feedback,c.identity):
        monkeypatch.setattr(module,'REFERENCE_SHA256',reference['sha256'])
    ready=original.reference_evidence;ready.value['actor']=reference
    ready.value['program_pair']={role:dict(program=h.write(tmp_path/(role+'-synthetic-template.json'),{'synthetic_only':True}))
                                 for role in ('robot_1','robot_2')}
    physics_fixture=helper('test_warehouse_v629_environment_replay')
    case=c.bound(original.value['train_case_inventory'])['cases'][0]
    manual=physics_fixture.synthetic_environment(case,1)
    from env.warehouse.public_observation_v6 import encode_public_observation
    from env.warehouse.pure_nn import action_masks
    from backend.training.warehouse_v6_actor import encode_actor_tree_features
    masks=action_masks(manual.state);encoded=encode_actor_tree_features(encode_public_observation(manual.state),masks)
    buffer=tmp_path/'synthetic-reference-buffer.npz'
    np.savez_compressed(buffer,encoded=np.stack([encoded,encoded]),masks=np.stack([masks,masks]),
        episode_ids=np.array(['synthetic-fit-0','synthetic-fit-1']),metadata_json=np.array(json.dumps(dict(
            schema=BUFFER_SCHEMA,split='train',reference_sha256=reference['sha256']))))
    buffer.chmod(0o444);ready.value['reference']=dict(buffer=c.pointer(buffer))
    partition=p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=original.value['train_case_inventory'],
        program_inventory_ref=original.value['program_evidence_inventory'],quality_inventory_ref=original.value['quality_case_inventory'],
        initialization_seed=0,optimizer_seed=1,output=p.OUTPUT_ROOT/'physical-fixture')
    candidate=tmp_path/'synthetic-candidate'
    monkeypatch.setattr(c.identity,'OWNER_ROOT',candidate);monkeypatch.setattr(c.programs,'OWNER_ROOT',candidate)
    monkeypatch.setattr(c.seeds,'OUTPUT_ROOT',candidate/'seed-plans')
    plan=c.seeds.prepare(reference_directory=tmp_path/'reference',partition_ref=partition.ref,run_id='physical-fixture')
    seed_fixture=helper('test_warehouse_v643_training_seed_plan')
    seed_fixture.empty_roots(tmp_path,monkeypatch);c.seeds.audit_seeds(plan)
    return plan,physics_fixture


class ProbeComplete(Exception):pass


def controlled_physics(active,physical_fixture,monkeypatch):
    from env.warehouse.reset_audit_v614 import record_successful_reset
    cases={case['scenario_seed']:case for case in active.ledger.training}
    def factory(config):
        case=cases[config.seed]
        assert any(record['payload']['case']==case for record in active.ledger.pending.values())
        # Explicit synthetic control state, not an environment reset from seed.
        return physical_fixture.synthetic_environment(case,1)
    def reset(env,*,seed):
        assert seed==env.config.seed and any(record['payload']['case']['scenario_seed']==seed
                                            for record in active.ledger.pending.values())
        record_successful_reset(seed,env.state.episode_id,env.config.map_layout_id,
                                env.config.move_battery_cost,env.config.charge_per_wait)
        return {},{}
    def forbidden(*_a,**_k):raise AssertionError('real environment constructor forbidden in engineering fixture')
    monkeypatch.setattr(c,'Environment',factory)
    monkeypatch.setattr(c.physics.WarehouseMultiAgentEnv,'__init__',forbidden)
    monkeypatch.setattr(c.physics.WarehouseMultiAgentEnv,'reset',reset)


def test_actual_collector_replay_critic_update_and_full_checkpoint_on_explicit_fixture(engineering_plan,monkeypatch):
    plan,fixture=engineering_plan
    with pytest.raises(ProbeComplete):
        with c._claimed_owner(plan) as active:
            raw=active.initialize('raw');regularized=active.initialize('regularized')
            assert c.role_hashes(raw.model)==c.role_hashes(regularized.model)
            assert active.cores['raw'].rng.bit_generator.state==active.cores['regularized'].rng.bit_generator.state
            controlled_physics(active,fixture,monkeypatch)
            trace=active.collect_rollout('raw')
            pending=active.pending['raw'];audit=c.bound(pending['audit'])
            assert raw.joint_steps==1024 and len(active.actual_resets)==16
            assert audit['public_NN_audit']['NN_action_checks']==2048
            assert audit['program_evidence_frames_in_gradient_buffers']==audit['quality_frames_in_gradient_buffers']==0
            assert len(pending['batch']['complete_cost_rows'])==960
            result=active.update('raw')
            assert raw.actor_updates==0 and raw.critic_updates>0
            assert len(active.cores['raw'].risk_training.episodes)==8
            assert active.cores['raw'].risk_training.critic_updates>0
            assert all(not episode['shutdown'] for episode in active.cores['raw'].risk_training.episodes.values())
            assert active.cores['raw'].accepted_program_updates==dict(robot_1=0,robot_2=0)
            wrapper=active.checkpoint('raw');saved=c.bound(wrapper)
            assert saved['core_snapshot_is_projection'] is True
            full=c.recovery.read_boundary(saved['full_owner_boundary'],plan=plan)
            assert full['reserved_joint_steps']==1024
            assert len(full['reader']['payload']['public']['payload']['pending'][0])==8
            assert c.pointer(trace['path'])==trace
            raise ProbeComplete()
    assert c.identity.claim_path().exists() and c.ACTIVE.get() is None
    assert (active.folder/'failed.json').exists() and not (active.folder/'completed.json').exists()


def test_interrupted_or_incomplete_pair_cannot_reuse_the_same_candidate_claim(engineering_plan):
    plan,_=engineering_plan
    with pytest.raises(RuntimeError,match='pair incomplete'):
        with c._claimed_owner(plan):pass
    claim=c.pointer(c.identity.claim_path())
    with pytest.raises(RuntimeError,match='already consumed'):
        with c._claimed_owner(plan):raise AssertionError('candidate was retried')
    assert c.pointer(c.identity.claim_path())==claim


def test_raw_feedback_attempt_fails_owner_without_extra_step(engineering_plan,monkeypatch):
    plan,fixture=engineering_plan
    with pytest.raises(RuntimeError,match='not active'):
        with c._claimed_owner(plan) as active:
            state=active.initialize('raw');controlled_physics(active,fixture,monkeypatch)
            active.collect_rollout('raw')
            with pytest.raises(RuntimeError,match='raw external'):
                active.update('raw',evidence_provider=lambda *_:None)
            assert active.failed and state.actor_updates==0
    assert (active.folder/'failed.json').exists()
