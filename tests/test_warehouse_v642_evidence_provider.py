"""Synthetic owner/state integration, no actual candidate training or scenes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json

import numpy as np
import pytest

from backend.training import warehouse_v642_evidence_provider as p


def module(name):
    spec=importlib.util.spec_from_file_location('v642_provider_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def old_tests():
    value=module('test_warehouse_v629_evidence_provider');value.p=p
    return value


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444)
    return p.pointer(path)


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    h=module('test_warehouse_v642_observed_pool')
    generator=h.recording.__wrapped__(tmp_path_factory);a=next(generator)
    folder=a['folder']/'refined-provider-setup';folder.mkdir()
    programs=h.native_programs(folder)
    refs={role:write(folder/(role+'-program.json'),value.to_dict()) for role,value in programs.items()}
    ready=SimpleNamespace(value=dict(actor=p.pointer(a['state'].reference_path),program_pair={role:dict(language=p.pools.refit.language_lock(value))
        for role,value in programs.items()}),payload_sha256='a'*64,verify=lambda:None)
    bindings=[]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p.admission,'load_readiness',lambda unused:ready)
        mp.setattr(p.pools,'TARGET_ROOT',folder/'targets')
        mp.setattr(p.pools.refit,'EXTRACTION_ROOT',folder/'extractions')
        for step in (15000,16800):
            a['state'].joint_steps=step
            target=p.pools.relabel_current_NN(a['observed'],a['current'],p.pools.TARGET_ROOT/str(step))
            result,binding=p.pools.refit_observed_pair_at_boundary(reference_directory=folder/'unused',owner_ref=a['observed'].owner_ref,
                state=a['state'],observed=a['observed'],targets_ref=target,program_refs=refs,output=p.pools.refit.EXTRACTION_ROOT/str(step),
                variant='regularized',epoch_id='synthetic-refit')
            bindings.append(binding);refs={role:result['roles'][role]['program'] for role in p.ROLES}
    a['state'].joint_steps=0
    try:yield {**a,'bindings':bindings,'ready':ready}
    finally:generator.close()


@pytest.fixture
def run(setup,tmp_path,monkeypatch):
    monkeypatch.setattr(p.admission,'load_readiness',lambda unused:setup['ready'])
    monkeypatch.setattr(p,'PROVIDER_ROOT',tmp_path/'providers')
    monkeypatch.setattr(p.pools,'TARGET_ROOT',tmp_path/'targets')
    provider=p.CurrentNNEvidenceProvider(reference_directory=tmp_path/'unused',owner_ref=setup['observed'].owner_ref,
                output=p.PROVIDER_ROOT/'one',pair_seed=0)
    return provider,deepcopy(setup['state']),setup


def test_real_closed_prerequisites_precede_every_export_directory_or_RNG(tmp_path,monkeypatch):
    monkeypatch.setattr(p.admission.v,'OUTPUT',tmp_path/'missing')
    def forbidden(*args,**kwargs):pytest.fail('unqualified candidate exported NN or constructed RNG')
    monkeypatch.setattr(p,'export_current_actor',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='complete reaped'):
        p.CurrentNNEvidenceProvider(reference_directory=tmp_path/'unused',owner_ref=None,output=tmp_path/'unused-provider',pair_seed=0)
    assert list(tmp_path.iterdir())==[]


def test_every_update_uses_current_NN_hash_and_new_epoch(run):
    old_tests().test_current_NN_export_and_epoch_specific_evidence_invalidate_after_change(run)


def test_validation_failure_cannot_reopen_until_registered_refit(run):
    old_tests().test_fidelity_failure_stays_disabled_after_recovery_until_next_scheduled_refit(run)


def test_boundary_requires_next_refit_before_later_rollout_update(run):
    old_tests().test_batch_ending_at_boundary_uses_previous_program_but_later_batch_requires_refit(run)


def test_epoch_is_consumed_once(run):
    old_tests().test_duplicate_optimizer_epoch_cannot_be_reused(run)


def test_restore_preserves_failure_latch_and_rejects_tampered_history(run,tmp_path):
    old_tests().test_checkpoint_restore_retains_failure_latch_and_checks_event_history(run,tmp_path)


def test_actual_synthetic_PPO_accepts_four_positive_updates_for_each_role(run,tmp_path):
    old_tests().test_full_ppo_core_gets_fresh_provider_evidence_for_every_accepted_update(run,tmp_path)


def test_in_memory_program_replacement_is_rejected(run):
    old_tests().test_in_memory_program_change_cannot_bypass_registered_extraction(run)


def test_snapshot_prefix_cannot_resume_after_later_provider_work(run,tmp_path):
    old_tests().test_snapshot_prefix_cannot_silently_resume_after_later_provider_work(run,tmp_path)
