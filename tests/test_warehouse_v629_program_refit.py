"""Toy fixed-language fits only; no real admission, candidate or environment."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_program_refit as r
from backend.training import warehouse_v629_online_evidence as o
from backend.training.warehouse_v629_program import from_parent
from backend.training.warehouse_v628_program import ParentProgram
from backend.training.warehouse_v622_actor_training import initialize_from_reference,export_current_actor
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
from env.warehouse.public_action_features_v617 import encode_actor_tree_features
from env.warehouse.layouts import STUDY_MAP_LAYOUT as layout
from env.warehouse.navigation import ACTIONS,MOVE_DELTAS


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));path.chmod(0o444);return r.pointer(path)


def physical_design(n):
    names={name:i for i,name in enumerate(OBSERVATION_FEATURE_NAMES)};points=list(layout.passable_positions)
    rows=[];masks=[]
    for i in range(n):
        own=points[i%(len(points)-1)];row=np.zeros(len(names),np.float32)
        for endpoint,position in [('self',own),('partner',points[-1]),('charger',layout.charger_position),
                ('order_0_pickup',points[0]),('order_0_delivery',points[-1]),('order_1_pickup',points[1]),('order_1_delivery',points[-2])]:
            row[names[endpoint+'_row']]=position[0]/(layout.rows-1);row[names[endpoint+'_column']]=position[1]/(layout.cols-1)
        row[names['self_active']]=1.;row[names['self_battery_fraction']]=.8
        for y,line in enumerate(layout.tiles):
            for x,tile in enumerate(line):row[names[f'wall_row_{y}_column_{x}']]=tile=='#'
        rows.append(row);masks.append([float(a=='WAIT' or (own[0]+MOVE_DELTAS[a][0],own[1]+MOVE_DELTAS[a][1]) in points) for a in ACTIONS])
    masks=np.asarray(masks,np.float32)
    return encode_actor_tree_features(np.asarray(rows),masks),masks


def test_real_missing_A_or_B_refuses_before_fitting_rng_or_any_output(tmp_path,monkeypatch):
    monkeypatch.setattr(r.admission.v,'OUTPUT',tmp_path/'missing-candidate')
    def forbidden(*args,**kwargs):raise AssertionError('failed A/B must not fit, load a pool or create RNG')
    monkeypatch.setattr(r,'complete_program',forbidden);monkeypatch.setattr(r,'load_pool',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no locked audited winner'):
        r.refit_pair_at_boundary(reference_directory=tmp_path/'reference',owner_ref=None,state=None,
            target_actor_ref=None,program_refs=None,pool_refs=None,output=tmp_path/'unused',variant='regularized',epoch_id='closed')
    assert not list(tmp_path.iterdir())


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('refit_online_helpers',Path(__file__).with_name('test_warehouse_v629_online_evidence.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a=helper.assets.__wrapped__(tmp_path)
    first=a['program'];parent=ParentProgram.from_dict(r.bound(first.protection['parent']),first.actor_role_sha256)
    parent.role='robot_2';parent.actor_role_sha256='b'*64
    second=from_parent(parent,write(tmp_path/'parent-robot_2.json',parent.to_dict()),64)
    programs={first.role:first,second.role:second};program_refs={role:write(tmp_path/(role+'-program.json'),p.to_dict()) for role,p in programs.items()}
    pool_refs={};pools={}
    for role,p in programs.items():
        refs={};pools[role]={}
        for split_index,(split,part) in enumerate(a['parts'].items()):
            part=deepcopy(part);n=len(part['x'])
            # Explicit encoded toy inputs exercise all three paths; they are
            # not claimed as collected joint public scene transitions.
            part['x'],part['m']=physical_design(n)
            part['p']=o.runtime_probabilities(a['actor'],role,part['x'],part['m'])
            part['sampled']=np.array([np.random.default_rng(independent_agent_seed(base_seed=row['action_seed'],episode_id=row['episode_id'],
                frame=row['frame'],agent_id=role)).choice(5,p=prob) for row,prob in zip(part['rows'],part['p'],strict=True)],np.int64)
            path=tmp_path/(role+'-'+split+'.npz')
            np.savez_compressed(path,**{k:part[k] for k in ('x','m','p','sampled')});path.chmod(0o444)
            refs[split]=dict(arrays=r.pointer(path),rows=write(tmp_path/(role+'-'+split+'-rows.json'),part['rows']))
            pools[role][split]=part
        pool_refs[role]=write(tmp_path/(role+'-pool.json'),dict(schema=o.MANIFEST_VERSION,scope='online_program_training_evidence',
            target_actor=a['actor_ref'],role=role,parts=refs))
    ready=SimpleNamespace(value=dict(actor=a['actor_ref'],program_pair={role:dict(language=r.admission.language_lock(p)) for role,p in programs.items()}),
        payload_sha256='f'*64,verify=lambda:None)
    monkeypatch.setattr(r.admission,'load_readiness',lambda unused:ready)
    monkeypatch.setattr(r,'EXTRACTION_ROOT',tmp_path/'extraction')
    state=initialize_from_reference(a['actor_ref']['path'],a['actor_ref']['sha256'],a['actor'].role_sha256,initialization_seed=0,actor_lr=3e-5)
    state.joint_steps=15000
    return dict(state=state,ready=ready,actor_ref=a['actor_ref'],programs=programs,program_refs=program_refs,pool_refs=pool_refs,pools=pools,
        owner=write(tmp_path/'owner.json',dict(scope='synthetic-test-only-no-training-authority')),folder=tmp_path)


def run(a,name='refit',variant='regularized'):
    return r.refit_pair_at_boundary(reference_directory=a['folder']/'unused-reference',owner_ref=a['owner'],state=a['state'],
        target_actor_ref=a['actor_ref'],program_refs=a['program_refs'],pool_refs=a['pool_refs'],output=r.EXTRACTION_ROOT/name,
        variant=variant,epoch_id='synthetic-epoch-0')


def test_both_roles_refit_current_full_NN_but_preserve_topology_supports_and_protected_outputs(prepared):
    a=prepared;before=r.training_fingerprint(a['state']);result=run(a)
    assert r.training_fingerprint(a['state'])==before and result['NN_updates']==result['PPO_joint_steps']==0
    assert result['training_authority'] is result['external_program_selects_execution_action'] is False
    for role,record in result['roles'].items():
        p=r.Program.from_dict(r.bound(record['program']),a['programs'][role].actor_role_sha256)
        assert r.admission.language_lock(p)==a['ready'].value['program_pair'][role]['language']
        assert all(r.payload(p.leaves[i])==r.payload(a['programs'][role].leaves[i]) for i in r.protected_indices(p))
        assert any(p.leaves[i]['bias']!=a['programs'][role].leaves[i]['bias'] for i in range(len(p.leaves)) if i not in r.protected_indices(p))
        assert record['solver']['counts']['leaf']>0 and record['solver']['counts']['node']==0
        assert record['anchor_actor_role_sha256']!=record['target_actor_role_sha256']
        assert record['feedback_enabled'] and record['current']['gate']['passed']
        # The rare path has <20 qualification frames and must stay excluded,
        # despite matching actions. The real strict gate still passes.
        assert .45<=record['current']['validation']['explanation']['coverage']<1.
        assert any(record['current']['eligibility'][str(i)]['explanation']['rows']<20
            for i in range(len(p.leaves)) if i not in p.explanation_leaves)
        assert len(record['current']['split_episode_ids']['fit'])==24
    assert len(list((r.EXTRACTION_ROOT/'claims').glob('*.json')))==1
    with pytest.raises(RuntimeError,match='already consumed'):run(a,'must-not-retry')
    assert not (r.EXTRACTION_ROOT/'must-not-retry').exists()


@pytest.mark.parametrize('change',['boundary','current_NN','lock','role_scene_overlap'])
def test_stale_target_or_unregistered_language_data_or_boundary_cannot_start_fits(prepared,change):
    a=prepared
    if change=='boundary':a['state'].joint_steps=14992
    if change=='current_NN':
        with torch.no_grad():a['state'].model.actors['robot_1'].tree.leaf_logits[0,4].add_(.001)
    if change=='lock':a['ready'].value['program_pair']['robot_1']['language']['structure_sha256']='0'*64
    if change=='role_scene_overlap':
        manifest=r.bound(a['pool_refs']['robot_2']);part=manifest['parts']['validation'];rows=r.bound(part['rows'])
        for row in rows:row['base_seed']='fit-0'
        path=Path(part['rows']['path']);path.chmod(0o644);path.write_text(json.dumps(rows));path.chmod(0o444)
        part['rows']=r.pointer(path);pool=Path(a['pool_refs']['robot_2']['path']);pool.chmod(0o644);pool.write_text(json.dumps(manifest));pool.chmod(0o444)
        a['pool_refs']['robot_2']=r.pointer(pool)
    with pytest.raises((RuntimeError,ValueError)):run(a)
    assert not r.EXTRACTION_ROOT.exists() and a['state'].actor_updates==a['state'].critic_updates==0


def test_failed_solver_call_is_predebited_and_cannot_retry_in_a_new_output(prepared,monkeypatch):
    from backend.training import warehouse_v626_program as kernel
    from scipy.optimize import OptimizeResult
    monkeypatch.setattr(kernel,'minimize',lambda *args,**kwargs:OptimizeResult(success=False,status=9,fun=1.,nit=1))
    with pytest.raises(ValueError,match='did not converge'):run(prepared)
    output=r.EXTRACTION_ROOT/'refit';assert (output/'failed.json').is_file() and not (output/'completed.json').exists()
    events=[json.loads(line) for line in (output/'robot_1/checkpoint.json.ledger.jsonl').read_bytes().splitlines()]
    assert len([e for e in events if e['event']=='call_started'])==1
    settlements=[e for e in events if e['event']=='call_completed'];assert len(settlements)==1 and not settlements[0]['data']['success']
    with pytest.raises(RuntimeError,match='already consumed'):run(prepared,'forbidden-retry')


def test_actual_validation_failure_disables_feedback_without_choosing_other_coefficients(prepared):
    # Uniform current NN ties prefer action zero; these WAIT-witness paths
    # cannot explain it. Recompute all genuine targets/draws for that NN,
    # rather than injecting a fabricated failure flag or validation metric.
    a=prepared
    with torch.no_grad():
        for role in r.ROLES:a['state'].model.actors[role].tree.leaf_logits.zero_()
    exported=export_current_actor(a['state'],a['folder']/'uniform-current.npz',pair_seed=0,variant='regularized',candidate_id='synthetic-refit-failure-only')
    a['actor_ref']=r.pointer(exported['path']);actor=r.FrozenWarehouseV6Actor.load(exported['path'],expected_sha256=exported['sha256'])
    for role in r.ROLES:
        manifest=r.bound(a['pool_refs'][role]);manifest['target_actor']=a['actor_ref']
        for split,part in a['pools'][role].items():
            part['p']=o.runtime_probabilities(actor,role,part['x'],part['m'])
            part['sampled']=np.array([np.random.default_rng(independent_agent_seed(base_seed=row['action_seed'],episode_id=row['episode_id'],
                frame=row['frame'],agent_id=role)).choice(5,p=prob) for row,prob in zip(part['rows'],part['p'],strict=True)],np.int64)
            path=Path(manifest['parts'][split]['arrays']['path']);path.chmod(0o644)
            np.savez_compressed(path,**{k:part[k] for k in ('x','m','p','sampled')});path.chmod(0o444)
            manifest['parts'][split]['arrays']=r.pointer(path)
        path=Path(a['pool_refs'][role]['path']);path.chmod(0o644);path.write_text(json.dumps(manifest));path.chmod(0o444)
        a['pool_refs'][role]=r.pointer(path)
    result=run(prepared)
    assert all(not record['feedback_enabled'] for record in result['roles'].values())
    assert all(record['solver']['counts']['leaf']>0 for record in result['roles'].values())
    assert prepared['state'].actor_updates==prepared['state'].critic_updates==0


@pytest.mark.parametrize('change',['owner','NN_during_fit','stored_program'])
def test_changed_asset_or_training_state_during_fit_cannot_produce_completed_pair(prepared,monkeypatch,change):
    a=prepared;original=r.complete_program;calls=[]
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(args[0].role)
        if change=='owner' and len(calls)==1:
            path=Path(a['owner']['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
        if change=='NN_during_fit' and len(calls)==1:
            with torch.no_grad():a['state'].model.actors['robot_1'].tree.leaf_logits[0,4].add_(.001)
        if change=='stored_program' and len(calls)==2:
            path=r.EXTRACTION_ROOT/'refit/robot_1/program.json';path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(r,'complete_program',changed)
    with pytest.raises(RuntimeError,match='asset changed during fitting|changed the NN/critics/optimizers'):run(a)
    assert (r.EXTRACTION_ROOT/'refit/failed.json').is_file() and not (r.EXTRACTION_ROOT/'refit/completed.json').exists()


def test_raw_refit_produces_evidence_without_any_external_NN_gradient(prepared):
    state=prepared['state'];before=r.training_fingerprint(state);result=run(prepared,variant='raw')
    assert result['identity']['variant']=='raw' and result['raw_external_program_gradient']==0
    assert before==r.training_fingerprint(state) and all(p.grad is None for p in state.model.actors.parameters())


def test_new_refit_programs_feed_current_NN_evidence_and_both_role_gradients(prepared):
    from backend.training.warehouse_v629_training_feedback import feedback_loss
    a=prepared;result=run(a);epoch='new-synthetic-epoch'
    evidence={role:o.build_evidence(record['program'],a['actor_ref'],a['pool_refs'][role],epoch_id=epoch)
        for role,record in result['roles'].items()}
    x=torch.as_tensor(np.stack([a['pools'][role]['fit']['x'] for role in r.ROLES],axis=1))
    m=torch.as_tensor(np.stack([a['pools'][role]['fit']['m'] for role in r.ROLES],axis=1))
    sampled=torch.as_tensor(np.stack([a['pools'][role]['fit']['sampled'] for role in r.ROLES],axis=1))
    before=r.training_fingerprint(a['state'])
    feedback=feedback_loss(a['state'].model,x,m,sampled,joint_steps=18000,total_steps=20000,
        variant='regularized',epoch_id=epoch,evidence=evidence)
    assert all(feedback.diagnostics['role_program_gradient_norm_before_shared_clip'][role]>0 for role in r.ROLES)
    assert all(feedback.diagnostics['selected_frames'][role]>0 for role in r.ROLES)
    feedback.verify_before_optimizer_step(a['state'].model,current_epoch_id=epoch)
    raw=feedback_loss(a['state'].model,x,m,sampled,joint_steps=18000,total_steps=20000,
        variant='raw',epoch_id=epoch,evidence=None)
    assert all(raw.diagnostics['role_program_gradient_norm_before_shared_clip'][role]==0 for role in r.ROLES)
    assert before==r.training_fingerprint(a['state']) and a['state'].actor_updates==a['state'].critic_updates==0
