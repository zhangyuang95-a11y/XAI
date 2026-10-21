"""Synthetic artifacts only; no candidate NN update or environment resets."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_online_evidence as o
from backend.training.warehouse_v6_actor_training import WarehouseV6Actor,export_actor
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v622_program import masked_probabilities,digest
from env.warehouse.decision_protocol import independent_agent_seed


@pytest.fixture(autouse=True)
def threads():
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(prior)


@pytest.fixture
def assets(tmp_path):
    spec=importlib.util.spec_from_file_location('v629_online_test_helpers',Path(__file__).with_name('test_warehouse_v629.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    program,parts=helpers.fixture(tmp_path)
    physical_spec=importlib.util.spec_from_file_location('v622_online_physical_helpers',Path(__file__).with_name('test_warehouse_v622_actor_training.py'))
    physical=importlib.util.module_from_spec(physical_spec);physical_spec.loader.exec_module(physical)
    encoded,masks=physical.physical_rows()
    program_path=tmp_path/'program.json';program_path.write_text(json.dumps(program.to_dict()))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    with torch.no_grad():
        for role in ('robot_1','robot_2'):
            actor=model.actors[role]
            for param in actor.parameters():param.zero_()
            actor.tree.node_count[0]=0;actor.tree.leaf_count[0]=1
            actor.tree.left_refs.zero_();actor.tree.right_refs.zero_()
            actor.tree.leaf_logits[0]=torch.tensor([-3.,-3.,-3.,-3.,2.9])
            actor.gate[-1].bias.fill_(10.)
    exported=export_actor(model,tmp_path/'target.npz',pair_seed=0,variant='raw',candidate_id='synthetic-online-evidence-only')
    actor_ref=o.pointer(exported['path']);actor=FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'])
    refs={}
    for split_index,(split,a) in enumerate(parts.items()):
        a['x']=np.tile(encoded[:,0].numpy(),(60,1));a['m']=np.tile(masks[:,0].numpy(),(60,1))
        a['p']=o.runtime_probabilities(actor,program.role,a['x'],a['m'])
        for i,row in enumerate(a['rows']):
            row.update(frame=i%10,action_seed=split_index*1000+i//10,episode_id=1)
        a['sampled']=np.array([np.random.default_rng(independent_agent_seed(base_seed=r['action_seed'],episode_id=r['episode_id'],
            frame=r['frame'],agent_id=program.role)).choice(5,p=p) for r,p in zip(a['rows'],a['p'],strict=True)],np.int64)
        array_path=tmp_path/(split+'.npz');np.savez_compressed(array_path,**{k:a[k] for k in ('x','m','p','sampled')})
        rows_path=tmp_path/(split+'-rows.json');rows_path.write_text(json.dumps(a['rows']))
        refs[split]=dict(arrays=o.pointer(array_path),rows=o.pointer(rows_path))
    manifest=dict(schema=o.MANIFEST_VERSION,scope='online_program_training_evidence',target_actor=actor_ref,role=program.role,parts=refs)
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    return dict(program=program,program_ref=o.pointer(program_path),actor_ref=actor_ref,actor=actor,
        parts=parts,manifest=manifest,manifest_path=manifest_path,pool_ref=o.pointer(manifest_path))


def build(a):return o.build_evidence(a['program_ref'],a['actor_ref'],a['pool_ref'],epoch_id='synthetic-epoch-0')


def rewrite_manifest(a):
    a['manifest_path'].write_text(json.dumps(a['manifest']));a['pool_ref']=o.pointer(a['manifest_path'])


def test_anchor_hash_remains_immutable_current_target_gets_separate_binding(assets):
    a=assets;before=digest(a['program'].to_dict());e=build(a)
    assert e.program.actor_role_sha256=='a'*64
    assert e.target_role_sha256==a['actor'].role_sha256['robot_1']!='a'*64
    assert e.report['gate']['passed'] and e.report['validation']['explanation']['coverage']==1
    assert digest(a['program'].to_dict())==before and not a['program'].feedback_leaves
    assert e.report['feedback_leaves'] and sum(e.report['fit_leaf_counts'])==240
    targets=e.loss_targets(a['parts']['fit']['x'],a['parts']['fit']['m'],current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)
    assert targets['qualified'].all()
    assert not e.payload()['execution_actions_from_program'] and not e.payload()['authorizes_training']


@pytest.mark.parametrize('change',['actor','epoch','report','program','asset'])
def test_evidence_invalidates_after_nn_epoch_or_evidence_change(assets,change):
    a=assets;e=build(a);role_sha=e.target_role_sha256;epoch=e.epoch_id
    if change=='actor':role_sha='b'*64
    if change=='epoch':epoch='next-optimizer-epoch'
    if change=='report':e.report['gate']['passed']=False
    if change=='program':e.program.feedback_leaves=frozenset({0})
    if change=='asset':Path(e.target_actor_ref['path']).write_bytes(b'changed')
    with pytest.raises((RuntimeError,ValueError)):
        e.loss_targets(a['parts']['fit']['x'],a['parts']['fit']['m'],current_role_sha256=role_sha,epoch_id=epoch)


@pytest.mark.parametrize('kind',['targets','sampled','mask'])
def test_pool_requires_complete_current_nn_distributions_and_true_sample_stream(assets,kind):
    a=assets;part=a['parts']['eligibility']
    if kind=='targets':part['p'][0]=[.2]*5
    if kind=='sampled':part['sampled'][0]=(part['sampled'][0]+1)%5
    if kind=='mask':part['m'][0,0]=1-part['m'][0,0]
    file=Path(a['manifest']['parts']['eligibility']['arrays']['path'])
    np.savez_compressed(file,**{k:part[k] for k in ('x','m','p','sampled')})
    a['manifest']['parts']['eligibility']['arrays']=o.pointer(file);rewrite_manifest(a)
    with pytest.raises((RuntimeError,ValueError)):build(a)


def test_fit_and_validation_cannot_share_scenes_even_with_different_episode_ids(assets):
    a=assets;rows=deepcopy(a['parts']['validation']['rows'])
    for i,row in enumerate(rows):row['base_seed']=a['parts']['fit']['rows'][i]['base_seed']
    path=Path(a['manifest']['parts']['validation']['rows']['path']);path.write_text(json.dumps(rows))
    a['manifest']['parts']['validation']['rows']=o.pointer(path);rewrite_manifest(a)
    with pytest.raises(RuntimeError,match='scenes overlap'):build(a)


def test_confirmation_or_validation_pool_scope_cannot_enter_online_feedback(assets):
    assets['manifest']['scope']='confirmation';rewrite_manifest(assets)
    with pytest.raises(RuntimeError,match='scope'):build(assets)


def test_serialized_evidence_rebuilds_from_raw_arrays_and_rejects_qualified_flag_tamper(assets,tmp_path):
    e=build(assets);path=tmp_path/'evidence.json';path.write_text(json.dumps(e.payload()))
    loaded=o.load_evidence(o.pointer(path),current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)
    assert loaded.payload()==e.payload()
    modified=e.payload();modified['report']['feedback_leaves']=[];path.write_text(json.dumps(modified))
    with pytest.raises(RuntimeError,match='independently replay'):
        o.load_evidence(o.pointer(path),current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)


def test_independent_validation_can_disable_otherwise_qualified_feedback(assets,monkeypatch):
    a=assets
    original=o.current_report
    def invalid(program,parts):
        report=original(program,parts)
        assert report['feedback_leaves'] and report['gate']['passed']
        # Exercise strict validation gating without changing evidence membership.
        report['validation']['feedback']['agreement']=.94
        report['gate']=o.gate(report['validation']);return report
    monkeypatch.setattr(o,'current_report',invalid)
    e=build(a);assert not e.report['gate']['passed']
    targets=e.loss_targets(a['parts']['fit']['x'],a['parts']['fit']['m'],current_role_sha256=e.target_role_sha256,epoch_id=e.epoch_id)
    assert not targets['qualified'].any()


def test_nonzero_nn_uses_runtime_single_frame_shape_for_mixed_physical_rows(assets,tmp_path,monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0);model=WarehouseV6Actor()
    exported=export_actor(model,tmp_path/'nonzero.npz',pair_seed=0,variant='raw',candidate_id='synthetic-nonzero-shape-replay')
    actor=FrozenWarehouseV6Actor.load(exported['path'],expected_sha256=exported['sha256'])
    x=assets['parts']['fit']['x'][:8];m=assets['parts']['fit']['m'][:8]
    role=actor.roles['robot_1'];original=role.logits_encoded;shapes=[]
    expected=np.stack([masked_probabilities(original(row),mask) for row,mask in zip(x,m,strict=True)])
    # Frozen role is immutable; instrument its class method without mutation.
    cls=type(role);unbound=cls.logits_encoded
    def observed(self,value):
        shapes.append(value.shape);return unbound(self,value)
    monkeypatch.setattr(cls,'logits_encoded',observed)
    actual=o.runtime_probabilities(actor,'robot_1',x,m)
    assert np.array_equal(actual,expected) and shapes==[(x.shape[1],)]*len(x)
