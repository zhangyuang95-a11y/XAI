"""Exact replay reuse on synthetic recorded states, no experimental resets."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import importlib.util

import pytest

from backend.training import warehouse_v630_observed_pool as r
from backend.training import warehouse_v630_target_cache as cache


def helper():
    spec=importlib.util.spec_from_file_location('v630_cache_helper',
        Path(__file__).with_name('test_warehouse_v630_observed_pool.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@pytest.fixture(scope='module')
def recording(tmp_path_factory):
    generator=helper().recording.__wrapped__(tmp_path_factory)
    try:yield next(generator)
    finally:generator.close()


@pytest.fixture
def prepared(recording,tmp_path,monkeypatch):
    monkeypatch.setattr(r,'TARGET_ROOT',tmp_path/'targets')
    # A new owner-view has no certificate, even if its source is identical.
    observed=replace(recording['observed'])
    ref=r.relabel_current_NN(observed,recording['current'],r.TARGET_ROOT/'first')
    return observed,ref


def count_replays(monkeypatch):
    calls=[];original=r.online.runtime_probabilities
    def counted(actor,role,x,m):
        calls.append((role,len(x)));return original(actor,role,x,m)
    monkeypatch.setattr(r.online,'runtime_probabilities',counted)
    return calls


def test_no_certificate_without_the_validator():
    with pytest.raises(RuntimeError,match='complete target replay'):cache.ReplayedTargets()


def test_first_validation_replays_every_role_and_split_then_reuses_exact_package(prepared,monkeypatch):
    observed,ref=prepared;calls=count_replays(monkeypatch)
    first=r.checked_targets(observed,ref)
    assert len(calls)==8 and sum(n for _,n in calls)==3840
    assert {role:sum(n for role2,n in calls if role2==role) for role in r.AGENTS}==dict(robot_1=1920,robot_2=1920)
    assert r.checked_targets(observed,ref)==first and len(calls)==8
    cert=observed._replayed_target_cache
    assert all(not a[k].flags.writeable for parts in cert.parts.values() for a in parts.values()
        for k in ('x','m','p','sampled'))
    # Returned dictionaries cannot mutate the verified resident package.
    first[0]['NN_updates']=123
    assert r.checked_targets(observed,ref)[0]['NN_updates']==0
    restored=replace(observed)
    assert restored._replayed_target_cache is None
    r.checked_targets(restored,ref)
    assert len(calls)==16


def test_both_role_envelopes_equal_uncached_full_replay_and_epoch_is_fresh(prepared,tmp_path,monkeypatch):
    observed,ref=prepared;targets,refs=r.checked_targets(observed,ref)
    calls=count_replays(monkeypatch)
    (tmp_path/'programs').mkdir()
    programs=helper().native_programs(tmp_path/'programs');write=helper().helper().write
    for role,p in programs.items():
        p_ref=write(tmp_path/(role+'.json'),p.to_dict())
        cached=r.build_observed_evidence(p_ref,observed,ref,epoch_id='epoch-one')
        assert len(calls)==(0 if role=='robot_1' else 4)
        base=r.evidence.build_evidence(p_ref,targets['target_actor'],targets['pools'][role],epoch_id='epoch-one')
        args={**base.__dict__,'asset_refs':tuple(deepcopy([*base.asset_refs,*refs])),
            'observed':observed,'target_package_ref':deepcopy(ref),'envelope_digest':''}
        direct=r.ObservedOnlineEvidence(**args)
        direct=r.ObservedOnlineEvidence(**{**args,'envelope_digest':r.digest(direct.payload())})
        assert cached.payload()==direct.payload()
        next_epoch=r.build_observed_evidence(p_ref,observed,ref,epoch_id='epoch-two')
        assert len(calls)==(4 if role=='robot_1' else 8)
        assert next_epoch.report==cached.report and next_epoch.envelope_digest!=cached.envelope_digest
        with pytest.raises(RuntimeError):cached.verify(current_role_sha256=cached.target_role_sha256,epoch_id='epoch-two')
        with pytest.raises(RuntimeError):next_epoch.verify(current_role_sha256='0'*64,epoch_id='epoch-two')
    assert len(calls)==8  # All eight calls belong only to the direct control.


@pytest.mark.parametrize('fault',['array','rows','payload','role_hash','writable'])
def test_cached_memory_tampering_fails_without_silent_revalidation(prepared,monkeypatch,fault):
    observed,ref=prepared;r.checked_targets(observed,ref);cert=observed._replayed_target_cache
    if fault=='array':
        a=cert.parts['robot_1']['fit']['p'];a.flags.writeable=True;a[0,0]+=.01;a.flags.writeable=False
    if fault=='rows':cert.parts['robot_1']['fit']['rows'][0]['task']=0
    if fault=='payload':cert.payload['NN_updates']=1
    if fault=='role_hash':cert.role_hashes['robot_1']='0'*64
    if fault=='writable':cert.parts['robot_1']['fit']['x'].flags.writeable=True
    def forbidden(*args,**kwargs):pytest.fail('tampering must fail, not reissue a certificate')
    monkeypatch.setattr(r,'_replay_targets',forbidden)
    with pytest.raises(RuntimeError):r.checked_targets(observed,ref)


def test_cached_asset_change_fails_before_reuse(prepared,monkeypatch):
    observed,ref=prepared;r.checked_targets(observed,ref)
    path=Path(ref['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ');path.chmod(0o444)
    def forbidden(*args,**kwargs):pytest.fail('changed asset must fail, not reissue a certificate')
    monkeypatch.setattr(r,'_replay_targets',forbidden)
    with pytest.raises(RuntimeError):r.checked_targets(observed,ref)


def test_new_target_package_evicts_previous_and_requires_full_replay(prepared,recording,monkeypatch):
    observed,ref=prepared;r.checked_targets(observed,ref)
    newer=r.relabel_current_NN(observed,recording['current'],r.TARGET_ROOT/'second')
    calls=count_replays(monkeypatch);r.checked_targets(observed,newer)
    assert len(calls)==8 and observed._replayed_target_cache.target_ref==newer
    r.checked_targets(observed,ref)
    assert len(calls)==16 and observed._replayed_target_cache.target_ref==ref


def test_changed_observed_binding_is_rejected(prepared):
    observed,ref=prepared;r.checked_targets(observed,ref)
    changed=replace(observed,observed_data_sha256='0'*64)
    object.__setattr__(changed,'_replayed_target_cache',observed._replayed_target_cache)
    with pytest.raises(RuntimeError):r.checked_targets(changed,ref)
