"""Recorded synthetic states, no new experimental scenes or actual training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json

import numpy as np
import pytest
import torch

from backend.training import warehouse_v630_observed_pool as r
from backend.training.warehouse_v630_program import from_previous,split_identical
from backend.training.warehouse_v6_actor import FEATURE_NAMES


def helper():
    spec=importlib.util.spec_from_file_location('v630_observed_recording_helper',
                Path(__file__).with_name('test_warehouse_v629_observed_pool.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.r=r
    return module


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    generator=helper().recording.__wrapped__(tmp_path_factory)
    try:yield next(generator)
    finally:generator.close()


@pytest.fixture(autouse=True)
def targets(tmp_path,monkeypatch):monkeypatch.setattr(r,'TARGET_ROOT',tmp_path/'targets')


def test_real_missing_A_B_rejects_before_private_replay_or_RNG(tmp_path,monkeypatch):
    monkeypatch.setattr(r.admission.v,'OUTPUT',tmp_path/'missing')
    def forbidden(*args,**kwargs):pytest.fail('closed A/B entered source replay or random draws')
    monkeypatch.setattr(r.physics,'restore_environment',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no complete locked audited winner'):
        r.audit_observed_pool(None,reference_directory=tmp_path/'unused')
    assert list(tmp_path.iterdir())==[]


def test_complete_source_and_counterfactual_draws_keep_distinct_provenance(recording,monkeypatch):
    h=helper();h.test_source_audit_keeps_behavior_actions_full_physics_and_complete_episodes(recording)
    h.test_relabelling_does_not_step_any_environment_or_overwrite_source_actions(recording,monkeypatch)


@pytest.mark.parametrize('fault',['scope','partial','behavior_action','reward','end_rng'])
def test_wrong_scope_incomplete_episode_or_forged_physics_is_rejected(recording,tmp_path,monkeypatch,fault):
    helper().test_source_replay_rejects_wrong_scope_incomplete_or_forged_physics(recording,tmp_path,monkeypatch,fault)


@pytest.mark.parametrize('fault',['source_binding','owner_binding','omitted_reference'])
def test_cached_pool_cannot_drop_source_identity_or_consumed_asset(recording,fault):
    helper().test_cached_pool_keeps_complete_source_identity_and_reference_manifest(recording,fault)


@pytest.mark.parametrize('fault',['trajectory_claim','actual_actions','difference_count','omitted_asset','fresh_randomness'])
def test_counterfactual_data_cannot_be_claimed_as_executed_current_policy(recording,tmp_path,fault):
    helper().test_counterfactual_package_replays_data_and_cannot_claim_executed_actions(recording,tmp_path,fault)


def native_programs(tmp_path):
    h=helper().helper('test_warehouse_v629');first,_=h.fixture(tmp_path)
    from backend.training.warehouse_v628_program import ParentProgram
    from backend.training.warehouse_v629_program import from_parent
    parent=ParentProgram.from_dict(r.bound(first.protection['parent']),first.actor_role_sha256)
    parent.role='robot_2';parent.actor_role_sha256='b'*64
    second=from_parent(parent,helper().write(tmp_path/'second-parent.json',parent.to_dict()))
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_column')]=1.
    programs={}
    for p in (first,second):
        out=from_previous(p);out.maximum_leaves=256;out,_=split_identical(out,0,w,-.5)
        programs[out.role]=out
    return programs


def test_observed_new_assets_feed_loss_without_relabelling_executed_history(recording,tmp_path):
    programs=native_programs(tmp_path)
    package=r.relabel_current_NN(recording['observed'],recording['current'],r.TARGET_ROOT/'current')
    evidence={}
    for role,p in programs.items():
        ref=helper().write(tmp_path/(role+'-program.json'),p.to_dict())
        e=r.build_observed_evidence(ref,recording['observed'],package,epoch_id='synthetic-refined-observed')
        assert e.payload()['version']==r.EVIDENCE_VERSION and e.payload()['validation_actions_are_counterfactual']
        assert e.payload()['actual_PPO_feedback_counts_from_real_rollouts_only']
        saved=helper().write(tmp_path/(role+'-evidence.json'),e.payload())
        assert r.load_observed_evidence(saved,recording['observed'],current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id).payload()==e.payload()
        evidence[role]=e
    from backend.training.warehouse_v629_training_feedback import feedback_loss
    part=recording['observed'].parts['fit'];target=r.bound(package);sampled=[]
    for role in r.AGENTS:
        pool=r.bound(target['pools'][role])
        with np.load(pool['parts']['fit']['arrays']['path'],allow_pickle=False) as z:sampled.append(z['sampled'].copy())
    before=r.physics.training_fingerprint(recording['state'])
    value=feedback_loss(recording['state'].model,torch.as_tensor(part['x'].copy()),torch.as_tensor(part['m'].copy()),
        torch.as_tensor(np.stack(sampled,axis=1)),joint_steps=18000,total_steps=20000,variant='regularized',
        epoch_id='synthetic-refined-observed',evidence=evidence)
    assert all(v>0 for v in value.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert before==r.physics.training_fingerprint(recording['state'])
    assert all(np.all(a['executed']==4) for a in recording['observed'].parts.values())


def test_refit_keeps_observed_source_and_counterfactual_scope_in_binding(recording,tmp_path,monkeypatch):
    programs=native_programs(tmp_path)
    refs={role:helper().write(tmp_path/(role+'-initial.json'),p.to_dict()) for role,p in programs.items()}
    state=recording['state'];old=state.joint_steps;state.joint_steps=15000
    ready=SimpleNamespace(value=dict(actor=r.pointer(state.reference_path),program_pair={role:dict(language=r.refit.language_lock(p)) for role,p in programs.items()}),
                          payload_sha256='a'*64,verify=lambda:None)
    monkeypatch.setattr(r.admission,'load_readiness',lambda unused:ready)
    monkeypatch.setattr(r.refit,'EXTRACTION_ROOT',tmp_path/'extraction')
    package=r.relabel_current_NN(recording['observed'],recording['current'],r.TARGET_ROOT/'current')
    try:
        result,ref=r.refit_observed_pair_at_boundary(reference_directory=tmp_path/'unused',owner_ref=recording['observed'].owner_ref,
            state=state,observed=recording['observed'],targets_ref=package,program_refs=refs,output=r.refit.EXTRACTION_ROOT/'fixed',
            variant='regularized',epoch_id='synthetic-refit')
        value=r.bound(ref)
        assert value['schema']=='warehouse-v630-observed-refit-source-binding.v1'
        assert value['target_package']==package and value['observed_source']==recording['source']
        assert value['current_policy_task_quality_estimate'] is None and value['actual_PPO_feedback_counts_from_real_rollouts_only']
        assert result['NN_updates']==0 and not value['training_authority']
    finally:state.joint_steps=old
