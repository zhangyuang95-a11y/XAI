"""Synthetic-only numerical/storage tests; no candidate or environment."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json,os,sys
import numpy as np
import pytest
from backend.training import warehouse_v642_long_targets as t


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);t.write_new(path,value);path.chmod(0o444);return t.pointer(path)


def test_old_20k_unclaimed_or_arbitrary_object_rejected_before_model_read_or_output(tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('unclaimed producer read model, NN or created stage')
    monkeypatch.setattr(t.rows,'new_stage',forbidden)
    monkeypatch.setattr(t.native.FrozenWarehouseV6Actor,'load',forbidden)
    with pytest.raises(RuntimeError,match='active newly claimed'):
        t.relabel_current_NN(object(),'regularized',None,None,stage_id='call-000000')
    assert not list(tmp_path.iterdir())


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    monkeypatch.setattr(t.rows,'STORAGE_ROOT',tmp_path/'future-only')
    monkeypatch.setattr(t.native,'TARGET_ROOT',tmp_path/'native-control')
    owner=put(tmp_path/'fake-owner.json',dict(synthetic_only=True));source=put(tmp_path/'fake-source.json',dict(synthetic_only=True))
    actor_ref=put(tmp_path/'fake-actor.npz',dict(synthetic_only=True))
    class Pool:
        def verify(self):return self
    pool=Pool();pool.owner_ref=owner;pool.source_ref=source;pool.audit_sha256='fake-audit';pool.observed_data_sha256='fake-data'
    pool.parts={};pool._replayed_target_cache=None
    rng=np.random.default_rng(91)
    for index,split in enumerate(t.native.online.SPLITS):
        n=120
        x=rng.normal(0,.1,(n,2,t.native.online.FEATURE_DIM)).astype(np.float32)
        m=np.ones((n,2,5),np.float32);x[:,:,t.native.online.MASK_INDICES]=m
        rows=[dict(episode=f'synthetic-{index}',base_seed=f'synthetic-scene-{index}',task=index%3+1,frame=i,
                   action_seed=100+index,episode_id=1) for i in range(n)]
        pool.parts[split]=dict(x=x,m=m,executed=np.full((n,2),4,np.int64),rows=rows)
    class Actor:
        model_version=t.native.MODEL_VERSION
        role_sha256=dict(robot_1='fake-role-1',robot_2='fake-role-2')
        @staticmethod
        def load(*args,**kwargs):return Actor()
    def probabilities(actor,role,x,m):
        # Named fake NN, stable float32 arithmetic in both producers.
        z=x[:,:5]+np.array([.1,.2,.3,.4,.5],np.float32)
        e=np.exp(z-z.max(1,keepdims=True)).astype(np.float32)*m
        return (e/e.sum(1,keepdims=True)).astype(np.float32)
    monkeypatch.setattr(t.native,'ObservedPool',Pool)
    monkeypatch.setattr(t.native,'FrozenWarehouseV6Actor',Actor)
    monkeypatch.setattr(t.native.online,'runtime_probabilities',probabilities)
    monkeypatch.setattr(t.native.online,'validate_rows',lambda x:None)
    monkeypatch.setattr(t,'_active_owner',lambda active,variant,value: t.require(value is pool,'fake foreign pool'))
    monkeypatch.setattr(t.native,'source_files',lambda:{'synthetic-native':'fake'})
    return SimpleNamespace(pool=pool,owner=owner,actor=actor_ref,tmp=tmp_path,active=object())


@pytest.mark.skipif(sys.platform!='darwin',reason='real APFS clone integration has no ordinary-copy fallback')
def test_real_APFS_and_hardlinks_match_native_all_arrays_bytes_draws_and_public_manifests(synthetic):
    s=synthetic
    old=t.native.relabel_current_NN(s.pool,s.actor,t.native.TARGET_ROOT/'control')
    new=t.relabel_current_NN(s.active,'regularized',s.pool,s.actor,stage_id='refit-150000')
    before=t.native._replay_targets(s.pool,old)
    after=t.native._replay_targets(s.pool,new)
    a,b=t.bound(old),t.bound(new)
    assert a['counterfactual_difference_counts']==b['counterfactual_difference_counts']
    inventory=t.read_materialization(new,expected_owner=s.owner)
    provenance=inventory['provenance'];base=t.bound(provenance['npz_baselines'])
    first_identity={k:(Path(v['path']).read_bytes(),Path(v['path']).stat().st_mode,Path(v['path']).stat().st_mtime_ns)
                    for k,v in base['baselines'].items()}
    for name,ref in provenance['final_arrays'].items():
        control=Path(old['path']).parent/name
        assert control.read_bytes()==Path(ref['path']).read_bytes()
        with np.load(control,allow_pickle=False) as x,np.load(ref['path'],allow_pickle=False) as y:
            assert x.files==y.files and all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape
                and x[k].tobytes()==y[k].tobytes() for k in x.files)
    for role in t.native.AGENTS:
        for split in t.native.online.SPLITS:
            for key in ('x','m','p','sampled'):
                assert before[2][role][split][key].tobytes()==after[2][role][split][key].tobytes()
    second=t.relabel_current_NN(s.active,'regularized',s.pool,s.actor,stage_id='call-000000')
    p2=t.read_materialization(second,expected_owner=s.owner)['provenance']
    assert p2['npz_baselines']==provenance['npz_baselines']
    assert first_identity=={k:(Path(v['path']).read_bytes(),Path(v['path']).stat().st_mode,Path(v['path']).stat().st_mtime_ns)
                           for k,v in base['baselines'].items()}
    for item in inventory['inputs']:
        assert Path(item['path']).is_file() and t.pointer(item['path'])==item
    consumed=t.bound(provenance['temporal_cleanup'])['consumed_temporal_inputs']
    assert len(consumed)==12 and all(not (Path(provenance['stage']['path']).parent/r['stage_relative_path']).exists() for r in consumed)
    assert all('target' not in r for r in t.bound(provenance['npz_materialization'])['NPZ'])
    assert all(Path(ref['path']).stat().st_ino==Path(p2['final_rows'][name]['path']).stat().st_ino
               for name,ref in provenance['final_rows'].items())
    assert inventory['canonical_NN_replay_still_required'] is True


@pytest.mark.skipif(sys.platform!='darwin',reason='real APFS only')
@pytest.mark.parametrize('tamper',['missing_baseline','missing_CAS','provenance_source','cleanup_map','semantic_pool','old_scratch_reappears'])
def test_independent_storage_reader_rejects_missing_persistent_or_tampered_lifetime(synthetic,tamper):
    s=synthetic;new=t.relabel_current_NN(s.active,'regularized',s.pool,s.actor,stage_id='call-000000')
    p=t.bound(t.bound(new)['long_materialization_provenance']);folder=Path(p['stage']['path']).parent
    if tamper=='missing_baseline':Path(next(iter(t.bound(p['npz_baselines'])['baselines'].values()))['path']).unlink()
    elif tamper=='missing_CAS':Path(t.bound(p['rows_deduplication'])['rows'][0]['CAS']).unlink()
    elif tamper=='old_scratch_reappears':(folder/'unbound-target'/next(iter(t.rows.NPZ_NAMES))).write_bytes(b'not-original')
    else:
        ref=p['npz_materialization'] if tamper=='cleanup_map' else (t.bound(new)['pools']['robot_1'] if tamper=='semantic_pool'
             else t.bound(new)['long_materialization_provenance'])
        path=Path(ref['path']);path.chmod(0o644);path.write_text('{}');path.chmod(0o444)
    with pytest.raises((RuntimeError,KeyError,FileNotFoundError)):
        t.read_materialization(new,expected_owner=s.owner)


@pytest.mark.skipif(sys.platform!='darwin',reason='real APFS only')
def test_clone_failure_preserves_first_CAS_and_canonical_scratch_without_binding_or_retry(synthetic,monkeypatch):
    def unavailable(*args,**kwargs):raise t.clones.CloneUnavailable('synthetic unavailable kernel; no copy fallback')
    monkeypatch.setattr(t.clones,'clone_patch_npz',unavailable)
    with pytest.raises(RuntimeError,match='no copy fallback'):
        t.relabel_current_NN(synthetic.active,'regularized',synthetic.pool,synthetic.actor,stage_id='call-000000')
    root=t.rows.STORAGE_ROOT/t.digest(synthetic.owner);stage=root/'stages/call-000000'
    assert (root/'npz-baselines.json').is_file() and (stage/'storage-stage-failed.json').is_file()
    assert len(list((stage/'unbound-target').glob('*.npz')))==12
    assert not (stage/'published-target/result.json').exists()
    with pytest.raises(RuntimeError,match='single-use'):
        t.relabel_current_NN(synthetic.active,'regularized',synthetic.pool,synthetic.actor,stage_id='call-000000')


@pytest.mark.skipif(sys.platform!='darwin',reason='real APFS only')
def test_every_clone_and_rows_publication_precedes_pool_manifest_result_and_seal(synthetic,monkeypatch):
    original=t.clones.clone_patch_npz;calls=[]
    def before_binding(base,target,output,**kw):
        folder=Path(output).parent
        assert not list(folder.glob('*-pool.json')) and not (folder/'result.json').exists()
        assert not (folder/'completed.json').exists()
        calls.append(Path(output).name)
        return original(base,target,output,**kw)
    monkeypatch.setattr(t.clones,'clone_patch_npz',before_binding)
    ref=t.relabel_current_NN(synthetic.active,'raw',synthetic.pool,synthetic.actor,stage_id='refit-150000')
    assert len(calls)==12 and set(calls)==t.rows.NPZ_NAMES
    assert t.read_materialization(ref,expected_owner=synthetic.owner)['provenance']['variant']=='raw'
