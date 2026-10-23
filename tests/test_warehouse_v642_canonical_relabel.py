"""Synthetic producer integration only; no environment/new research seed."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from backend.training import warehouse_v642_canonical_relabel as r
from tests.test_warehouse_v642_canonical_batch import numerical


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    r.write_new(path, value); path.chmod(0o444); return r.pointer(path)


@pytest.fixture
def synthetic(tmp_path, monkeypatch, numerical):
    role, x = numerical
    x = x[:7]
    owner = put(tmp_path/'owner.json', dict(synthetic_only=True))
    source = put(tmp_path/'source.json', dict(synthetic_only=True))
    actor_ref = put(tmp_path/'actor.npz', dict(synthetic_tensor_file=True))
    actor = r.native.FrozenWarehouseV6Actor(dict(robot_1=role, robot_2=role), actor_ref['sha256'])
    monkeypatch.setattr(r.native.FrozenWarehouseV6Actor, 'load', lambda *a,**kw:actor)
    monkeypatch.setattr(r.native, 'TARGET_ROOT', tmp_path/'targets')
    # Actual ObservedPool record with software-only readiness and audit. No
    # audit_observed_pool or environments are invoked or purported to pass.
    ready = SimpleNamespace(verify=lambda:None)
    parts = {}
    for i, split in enumerate(r.native.online.SPLITS):
        xx = np.stack([x, x], axis=1)
        m = xx[:,:,r.batch.MASK_INDICES].copy()
        p = np.full(m.shape, .2, np.float32)
        rows = [dict(episode='synthetic-'+split, base_seed='synthetic-scene-'+split,
            task=i%3+1, frame=j, action_seed=100+i, episode_id=1) for j in range(len(x))]
        parts[split] = dict(x=xx, m=m, behavior_p=p, executed=np.full((len(x),2),4,np.int64), rows=rows)
        for name in ('x','m','behavior_p','executed'): parts[split][name].flags.writeable=False
    sources = r.native.source_files(); refs = (owner,source); data_sha = r.native.data_sha(parts)
    audit = dict(source=source, owner=owner, source_sha256=r.digest(sources),
        observed_data_sha256=data_sha, asset_manifest_sha256=r.digest(refs))
    observed = r.native.ObservedPool(source, owner, parts, refs, sources, ready, data_sha, audit, r.digest(audit))
    guard_value = dict(owner=owner, claim=owner, source_files=r.source_files(), executor='synthetic-only')
    return SimpleNamespace(pool=observed,actor=actor_ref,guard=lambda:deepcopy(guard_value),guard_value=guard_value,tmp=tmp_path)


def test_missing_owner_guard_rejects_before_model_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(r.native.FrozenWarehouseV6Actor,'load',lambda *a,**kw:pytest.fail('model read'))
    with pytest.raises(RuntimeError):
        r.relabel_current_NN(object(),{},tmp_path/'never',epoch_id='fake',sequence=0,source_guard=None)
    assert not list(tmp_path.iterdir())


def test_producer_matches_native_and_retains_complete_scalar_reader(synthetic, monkeypatch):
    s = synthetic
    ref = r.relabel_current_NN(s.pool,s.actor,r.native.TARGET_ROOT/'new',epoch_id='fake-epoch',sequence=3,source_guard=s.guard)
    p = r.bound(ref)
    assert s.pool._replayed_target_cache is None  # producer minted no authority
    calls = []
    original = r.native.online.runtime_probabilities
    def scalar(*args):
        calls.append((args[1],len(args[2])))
        return original(*args)
    monkeypatch.setattr(r.native.online,'runtime_probabilities',scalar)
    _,_,parts,_ = r.native._replay_targets(s.pool,ref)
    assert len(calls) == 8 and sum(n for _,n in calls) == 56
    for role in r.native.AGENTS:
        for split,a in parts[role].items():
            assert r.batch.bits_equal(a['p'],original(r.native.FrozenWarehouseV6Actor.load(None),role,a['x'],a['m']))
    appendix = r.read_provenance(s.pool,ref,epoch_id='fake-epoch',sequence=3,source_guard=s.guard)
    assert appendix['native_scalar_audit_still_required'] is True
    assert appendix['provenance']['native_scalar_audit_performed'] is False
    assert p['source_files'] == r.native.source_files()
    assert p['canonical_producer_provenance'] not in p['asset_refs']


@pytest.mark.parametrize('kind',['epoch','sequence','owner','source','actor'])
def test_binding_rejections(synthetic,kind):
    s=synthetic
    ref=r.relabel_current_NN(s.pool,s.actor,r.native.TARGET_ROOT/'new',epoch_id='fake-epoch',sequence=3,source_guard=s.guard)
    kwargs=dict(epoch_id='fake-epoch',sequence=3,source_guard=s.guard)
    if kind=='epoch':kwargs['epoch_id']='other'
    if kind=='sequence':kwargs['sequence']=4
    if kind=='owner':s.guard_value['owner']={'path':'other','sha256':'0'*64}
    if kind=='source':s.guard_value['source_files']={}
    if kind=='actor':
        path=Path(s.actor['path']);path.chmod(0o644);path.write_bytes(b'changed');path.chmod(0o444)
    with pytest.raises(RuntimeError):r.read_provenance(s.pool,ref,**kwargs)


def test_one_ulp_witness_difference_fails_before_package_creation(synthetic,monkeypatch):
    s=synthetic; original=r.batch.forward_trace
    def corrupt(*a,**kw):
        out=original(*a,**kw);out['probabilities']=out['probabilities'].copy()
        out['probabilities'][0,4]=np.nextafter(out['probabilities'][0,4],np.float32(2))
        return out
    monkeypatch.setattr(r.batch,'forward_trace',corrupt)
    with pytest.raises(RuntimeError,match='scalar witness differs bitwise'):
        r.relabel_current_NN(s.pool,s.actor,r.native.TARGET_ROOT/'bad',epoch_id='fake',sequence=1,source_guard=s.guard)
    assert not r.native.TARGET_ROOT.exists()


def test_corruption_outside_witness_is_rejected_by_full_native_scalar_reader(synthetic,monkeypatch):
    s=synthetic
    # Deliberately sparse software witness demonstrates why the full original
    # reader is still mandatory. No production schedule is changed on disk.
    monkeypatch.setattr(r,'witness_indices',lambda n:[0])
    original=r.batch.forward_trace
    def corrupt(*a,**kw):
        out=original(*a,**kw);out['probabilities']=out['probabilities'].copy()
        out['probabilities'][1,4]=np.nextafter(out['probabilities'][1,4],np.float32(2))
        return out
    monkeypatch.setattr(r.batch,'forward_trace',corrupt)
    ref=r.relabel_current_NN(s.pool,s.actor,r.native.TARGET_ROOT/'bad',epoch_id='fake',sequence=1,source_guard=s.guard)
    with pytest.raises(RuntimeError,match='not the current complete masked NN'):
        r.native._replay_targets(s.pool,ref)
    assert s.pool._replayed_target_cache is None


def test_fixed_schedule_no_randomness_and_inclusive_endpoints():
    assert r.witness_indices(1)==[0]
    assert r.witness_indices(7)==list(range(7))
    assert len(r.witness_indices(1000))==64
    assert r.witness_indices(1000)[0]==0 and r.witness_indices(1000)[-1]==999
    assert r.source_files()['backend/training/warehouse_v642_canonical_relabel.py']==r.file_sha(r.__file__)
