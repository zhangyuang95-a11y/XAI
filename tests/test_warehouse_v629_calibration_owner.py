"""Scoped fake A/B, empty temporary history, synthetic NN; no real candidate."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
import torch
from experiments.warehouse import v629_calibration_owner as o
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor,export_actor
from backend.training.warehouse_v6_actor import FEATURE_DIM
from backend.training.warehouse_v625_actor_training import BUFFER_SCHEMA


class ProbeFinished(Exception):pass


def helper(name):
    spec=importlib.util.spec_from_file_location('owner_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_closed_gate_precedes_any_input_file_directory_constructor_or_RNG(tmp_path,monkeypatch):
    monkeypatch.setattr(o.seeds.partitions.admission.v,'OUTPUT',tmp_path/'missing-winner')
    def forbidden(*args,**kwargs):raise AssertionError('missing actual A/B cannot consume collection or initialization')
    monkeypatch.setattr(o,'bound',forbidden);monkeypatch.setattr(o,'initialize_from_reference',forbidden)
    monkeypatch.setattr(o.WarehouseMultiAgentEnv,'__init__',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        with o.claimed_owner(reference_directory=tmp_path/'missing-reference',plan_ref=None):raise AssertionError('armed')
    assert not list(tmp_path.iterdir())


def test_bare_owner_is_never_a_training_capability():
    with pytest.raises(RuntimeError,match='single-use claimed_owner'):o.CalibrationOwner()


@pytest.fixture
def preparation(tmp_path,monkeypatch):
    h=helper('test_warehouse_v629_training_seed_plan')
    proposal=h.helper().proposal.__wrapped__(tmp_path,monkeypatch)[0]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for actor in model.actors.values():
            for parameter in actor.parameters():parameter.zero_()
            actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
            actor.tree.left_refs.zero_();actor.tree.right_refs.zero_();actor.tree.leaf_logits[0,4]=1000.;actor.gate[-1].bias.fill_(1.)
    actor=export_actor(model,tmp_path/'synthetic-reference.npz',pair_seed=0,variant='raw',candidate_id='synthetic-owner-only')
    ready=proposal.readiness
    ready.value['actor']=o.pointer(actor['path'])
    ready.value['program_pair']={r:dict(language=dict(anchor_actor_role_sha256=sha)) for r,sha in actor['role_sha256'].items()}
    physics_helper=helper('test_warehouse_v629_environment_replay')
    case=o.bound(proposal.value['train_case_inventory'])['cases'][0]
    manual=physics_helper.synthetic_environment(case,1)  # Explicit state; no constructor/reset.
    masks=o.observed.action_masks(manual.state)
    encoded=o.observed.encode_actor_tree_features(o.observed.encode_public_observation(manual.state),masks)
    file=tmp_path/'synthetic-common-fit.npz'
    np.savez_compressed(file,encoded=np.stack([encoded,encoded]),masks=np.stack([masks,masks]),
        episode_ids=np.array(['synthetic-common-fit-0','synthetic-common-fit-1']),metadata_json=np.array(json.dumps(dict(
            schema=BUFFER_SCHEMA,split='train',reference_sha256=actor['sha256']))))
    file.chmod(0o444);ready.value['reference']=dict(buffer=o.pointer(file))
    monkeypatch.setattr(o.seeds,'OUTPUT_ROOT',tmp_path/'seed-plans')
    plan=o.seeds.prepare(reference_directory=tmp_path/'reference',proposal_ref=proposal.ref,run_id='synthetic-owner-only')
    h.empty_history(tmp_path,monkeypatch)
    o.prepare_sources(reference_directory=tmp_path/'reference',plan_ref=plan.ref)
    o.audit_source_bound_seeds(reference_directory=tmp_path/'reference',plan_ref=plan.ref)
    monkeypatch.setattr(o,'claim_path',lambda:tmp_path/'single-use-claim.json')
    return plan,tmp_path/'reference'


def enter(preparation):
    plan,reference=preparation;return o.claimed_owner(reference_directory=reference,plan_ref=plan.ref)


def test_pair_import_preserves_both_actor_and_new_critic_initialization(preparation):
    with pytest.raises(ProbeFinished):
        with enter(preparation) as owner:
            raw=owner.initialize('raw');reg=owner.initialize('regularized')
            assert o.role_hashes(raw.model)==o.role_hashes(reg.model)
            assert o.critic_initialization_sha256(raw.model)==o.critic_initialization_sha256(reg.model)
            assert owner.cores['raw'].rng.bit_generator.state==owner.cores['regularized'].rng.bit_generator.state
            assert raw.actor_updates==reg.actor_updates==raw.critic_updates==reg.critic_updates==0
            raise ProbeFinished()
    assert o.claim_path().is_file() and o.ACTIVE.get() is None
    assert (owner.folder/'failed.json').is_file() and not (owner.folder/'completed.json').exists()


def test_partial_pair_is_failure_and_same_claim_cannot_be_reissued(preparation):
    with pytest.raises(RuntimeError,match='pair incomplete'):
        with enter(preparation) as owner:owner.initialize('raw')
    claim=o.pointer(o.claim_path())
    with pytest.raises(RuntimeError,match='already consumed'):
        with enter(preparation):raise AssertionError('reissued')
    assert o.pointer(o.claim_path())==claim


def test_actual_finite_reset_hook_replay_and_critic_only_optimizer_wiring(preparation):
    with pytest.raises(ProbeFinished):
        with enter(preparation) as owner:
            state=owner.initialize('raw');before=o.training_fingerprint(state)
            trace=owner.collect_rollout('raw')
            assert state.joint_steps==1024 and owner.reserved['raw']==1024 and len(owner.actual_resets)==16
            assert all(r['episode_id']==1 for r in owner.actual_resets)
            pending=owner.pending['raw'];audit=o.bound(pending['audit'])
            assert audit['physics_and_NN_replay']['public_NN_audit']['NN_action_checks']==2048
            assert len(pending['batch']['complete_cost_rows'])==960 and audit['program_evidence_frames_in_gradient_buffers']==0
            result=owner.update('raw')
            assert state.actor_updates==0 and state.critic_updates==16
            assert result['accepted_positive_program_gradient_updates']=={'robot_1':0,'robot_2':0}
            snapshot=owner.checkpoint('raw');stored=o.bound(snapshot)
            assert stored['core']['safe_batch_boundary'] and stored['reader']['payload']['public']['payload']['processed_joint_steps']==1024
            assert not stored['automatic_resume'] and stored['safe_snapshot_not_resume_authority']
            assert o.pointer(trace['path'])==trace and o.training_fingerprint(state)!=before
            journal=[json.loads(row) for row in (owner.folder/'owner-journal.jsonl').read_text().splitlines()]
            assert sum(r['kind']=='environment_constructor_and_reset_predebit' for r in journal)==16
            assert sum(r['kind']=='actual_reset_bound' for r in journal)==16
            raise ProbeFinished()


def test_raw_feedback_attempt_and_caught_errors_close_owner(preparation):
    with pytest.raises(RuntimeError,match='not active'):
        with enter(preparation) as owner:
            owner.initialize('raw');owner.collect_rollout('raw')
            with pytest.raises(RuntimeError,match='raw program feedback'):owner.update('raw',evidence_provider=lambda *a:None)
            assert owner.failed
    assert (owner.folder/'failed.json').exists()


def test_second_rollout_before_optimizer_is_rejected_and_cannot_reset_more(preparation):
    with pytest.raises(RuntimeError,match='previous rollout'):
        with enter(preparation) as owner:
            owner.initialize('raw');owner.collect_rollout('raw');resets=len(owner.actual_resets)
            try:owner.collect_rollout('raw')
            finally:assert len(owner.actual_resets)==resets==16


@pytest.mark.parametrize('operation',['program_source_before_75','unknown_variant','duplicate_initialization','forged_step_prefix','unreserved_reset'])
def test_unregistered_operation_fails_closed_before_new_reset(preparation,operation):
    with pytest.raises((RuntimeError,KeyError)):
        with enter(preparation) as owner:
            owner.initialize('raw')
            if operation=='program_source_before_75':owner.collect_program_source('raw')
            if operation=='unknown_variant':owner.initialize('other')
            if operation=='duplicate_initialization':owner.initialize('raw')
            if operation=='forged_step_prefix':owner.states['raw'].joint_steps=15000;owner.collect_program_source('raw')
            if operation=='unreserved_reset':owner.new_environment('raw',owner.train_streams[0][0],scope='train')
    assert len(owner.actual_resets)==0 and owner.failed and not (owner.folder/'completed.json').exists()


def test_unowned_NN_change_is_rejected_before_collection(preparation):
    with pytest.raises(RuntimeError,match='changed outside owned optimizer'):
        with enter(preparation) as owner:
            state=owner.initialize('raw')
            with torch.no_grad():state.model.actors['robot_1'].tree.leaf_logits[0,4].add_(1)
            owner.collect_rollout('raw')
    assert len(owner.actual_resets)==0


def test_changed_source_or_claim_rejects_before_next_constructor(preparation):
    with pytest.raises(RuntimeError,match='claim or descriptor changed'):
        with enter(preparation) as owner:
            owner.initialize('raw');path=Path(owner.descriptor_ref['path']);path.chmod(0o644);path.write_text('{}');path.chmod(0o444)
            owner.collect_rollout('raw')
    assert len(owner.actual_resets)==0


def test_owner_source_snapshot_change_is_rejected_before_single_use_claim(preparation):
    plan,_=preparation;contract=o.bound(o.checked_sources(plan));name='experiments/warehouse/v629_calibration_owner.py'
    path=Path(plan.ref['path']).parent/'owner-source'/contract['source_sha256']/name
    path.chmod(0o644);path.write_text('# changed executor\n');path.chmod(0o444)
    with pytest.raises(RuntimeError,match='source snapshot changed'):
        with enter(preparation):raise AssertionError('armed with changed code')
    assert not o.claim_path().exists()


def test_output_is_nested_under_the_predeclared_audit_exclusion(preparation):
    plan,_=preparation
    with pytest.raises(ProbeFinished):
        with enter(preparation) as owner:
            assert owner.folder==Path(plan.ref['path']).parent/'calibration'
            assert any(owner.folder.is_relative_to(path) for path in o.seeds.exclusions(plan))
            raise ProbeFinished()


def test_actual_owned_75_percent_prefix_then_once_only_program_source_replay(preparation):
    # Synthetic WAIT NN and fake A/B only. These are actual numerical updates
    # and test-scene resets, not a v620 candidate or scientific calibration.
    with pytest.raises(ProbeFinished):
        with enter(preparation) as owner:
            state=owner.initialize('raw')
            while state.joint_steps<15000:
                owner.collect_rollout('raw');owner.update('raw')
            assert owner.reserved['raw']==owner.cores['raw'].processed_joint_steps==15000
            assert len(owner.updates['raw'])==15 and state.actor_updates>0 and state.critic_updates>0
            before=o.training_fingerprint(state);resets=len(owner.actual_resets)
            pool=owner.collect_program_source('raw')
            assert len(owner.actual_resets)==resets+4 and owner.evidence_cursor['raw']==4
            assert o.training_fingerprint(state)==before and pool.owner_ref==owner.owners['raw']
            assert {s:len({r['episode'] for r in part['rows']}) for s,part in pool.parts.items()}=={s:1 for s in o.observed.online.SPLITS}
            assert sum(len(part['x']) for part in pool.parts.values())==480
            with pytest.raises(RuntimeError,match='scheduled refit'):owner.collect_rollout('raw')
            assert owner.failed
            raise ProbeFinished()
