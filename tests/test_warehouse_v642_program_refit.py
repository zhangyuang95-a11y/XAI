"""Admitted-owner plumbing with synthetic targets, not actual training."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json

import numpy as np
import pytest
import torch

from backend.training import warehouse_v642_program_refit as r
from backend.training import warehouse_v642_online_evidence as o
from backend.training.warehouse_v630_program import from_previous,split_identical as old_split_identical
from backend.training.warehouse_v642_program import import_previous

def split_identical(program,*args):
    refined,record=old_split_identical(program,*args)
    return import_previous(refined,12),record
from backend.training.warehouse_v6_actor import FEATURE_NAMES


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(before)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('v642_scheduled_refit_fixture',
                  Path(__file__).with_name('test_warehouse_v629_program_refit.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a=helper.prepared.__wrapped__(tmp_path,monkeypatch)
    w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_column')]=1.
    programs={};refs={}
    for role,p in a['programs'].items():
        p=from_previous(p);p.maximum_leaves=256;p,_=split_identical(p,0,w,-.5)
        path=tmp_path/(role+'-refined-program.json');path.write_text(json.dumps(p.to_dict()));path.chmod(0o444)
        programs[role]=p;refs[role]=r.pointer(path)
    a['programs'],a['program_refs']=programs,refs
    a['ready'].value['program_pair']={role:dict(language=r.language_lock(p)) for role,p in programs.items()}
    monkeypatch.setattr(r.admission,'load_readiness',lambda unused:a['ready'])
    monkeypatch.setattr(r,'EXTRACTION_ROOT',tmp_path/'v642-extraction')
    return a


def run(a,name='refit',variant='regularized'):
    return r.refit_pair_at_boundary(reference_directory=a['folder']/'unused-reference',owner_ref=a['owner'],
        state=a['state'],target_actor_ref=a['actor_ref'],program_refs=a['program_refs'],pool_refs=a['pool_refs'],
        output=r.EXTRACTION_ROOT/name,variant=variant,epoch_id='synthetic-refit-epoch')


def test_missing_actual_A_B_blocks_every_fit_pool_rng_and_output(tmp_path,monkeypatch):
    monkeypatch.setattr(r.admission.v,'OUTPUT',tmp_path/'missing')
    def forbidden(*args,**kwargs):pytest.fail('closed A/B touched fitter, evidence pool or RNG')
    monkeypatch.setattr(r,'complete_program',forbidden);monkeypatch.setattr(r,'load_pool',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='complete reaped'):
        r.refit_pair_at_boundary(reference_directory=tmp_path/'reference',owner_ref=None,state=None,
             target_actor_ref=None,program_refs=None,pool_refs=None,output=tmp_path/'forbidden',variant='regularized',epoch_id='no')
    assert list(tmp_path.iterdir())==[]


def test_scheduled_dual_role_refit_preserves_all_protected_children_and_full_NN_state(prepared):
    a=prepared;before=r.training_fingerprint(a['state']);result=run(a)
    assert r.training_fingerprint(a['state'])==before
    assert not result['external_program_selects_execution_action'] and not result['training_authority']
    for role,record in result['roles'].items():
        p=r.Program.from_dict(r.bound(record['program']),record['target_actor_role_sha256'])
        protected=r.protected_indices(p)
        assert len(protected)==2 and record['protected_leaves_unchanged']==sorted(protected)
        assert r.language_lock(p)==a['ready'].value['program_pair'][role]['language']
        assert all(r.payload(p.leaves[i])==r.payload(a['programs'][role].leaves[i]) for i in protected)
        assert record['solver']['counts']['leaf']>0 and record['solver']['counts']['node']==0
        assert record['anchor_actor_role_sha256']!=record['target_actor_role_sha256']
        assert set(record['current']['feedback_leaves'])==set(p.feedback_leaves)
        assert record['feedback_enabled']==record['current']['gate']['passed']
    with pytest.raises(RuntimeError,match='already consumed'):run(a,'forbidden-retry')
    assert not (r.EXTRACTION_ROOT/'forbidden-retry').exists()


@pytest.mark.parametrize('fault',['boundary','current_NN','language'])
def test_unregistered_boundary_stale_NN_or_changed_language_stops_before_claim(prepared,fault):
    a=prepared
    if fault=='boundary':a['state'].joint_steps=14992
    elif fault=='current_NN':
        with torch.no_grad():a['state'].model.actors['robot_1'].tree.leaf_logits[0,4].add_(.001)
    else:a['ready'].value['program_pair']['robot_1']['language']['structure_sha256']='0'*64
    with pytest.raises((RuntimeError,ValueError)):run(a)
    assert not r.EXTRACTION_ROOT.exists()


def test_failed_solve_is_charged_and_cannot_be_retried_with_another_output(prepared,monkeypatch):
    from backend.training import warehouse_v626_program as kernel
    from scipy.optimize import OptimizeResult
    monkeypatch.setattr(kernel,'minimize',lambda *args,**kwargs:OptimizeResult(success=False,status=9,fun=1.,nit=1))
    with pytest.raises(ValueError,match='did not converge'):run(prepared)
    folder=r.EXTRACTION_ROOT/'refit'
    rows=[json.loads(line) for line in (folder/'robot_1/checkpoint.json.ledger.jsonl').read_bytes().splitlines()]
    assert sum(row['event']=='call_started' for row in rows)==1
    assert (folder/'failed.json').exists() and not (folder/'completed.json').exists()
    with pytest.raises(RuntimeError,match='already consumed'):run(prepared,'retry')


def test_refitted_assets_feed_fresh_current_NN_gradients_without_changing_training_state(prepared):
    from backend.training.warehouse_v629_training_feedback import feedback_loss
    a=prepared;result=run(a);epoch='after-refit-synthetic-only'
    evidence={role:o.build_evidence(row['program'],a['actor_ref'],a['pool_refs'][role],epoch_id=epoch)
              for role,row in result['roles'].items()}
    x=torch.as_tensor(np.stack([a['pools'][role]['fit']['x'] for role in r.ROLES],axis=1))
    m=torch.as_tensor(np.stack([a['pools'][role]['fit']['m'] for role in r.ROLES],axis=1))
    sampled=torch.as_tensor(np.stack([a['pools'][role]['fit']['sampled'] for role in r.ROLES],axis=1))
    before=r.training_fingerprint(a['state'])
    regularized=feedback_loss(a['state'].model,x,m,sampled,joint_steps=18000,total_steps=20000,
               variant='regularized',epoch_id=epoch,evidence=evidence)
    assert all(v>0 for v in regularized.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    raw=feedback_loss(a['state'].model,x,m,None,joint_steps=18000,total_steps=20000,
               variant='raw',epoch_id=epoch)
    assert all(v==0 for v in raw.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert r.training_fingerprint(a['state'])==before


def test_genuine_changed_NN_validation_failure_closes_feedback(prepared):
    spec=importlib.util.spec_from_file_location('v642_refit_failure_scenario',
               Path(__file__).with_name('test_warehouse_v629_program_refit.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.r=r
    # Recompute genuine uniform-NN targets and sampled actions, rather than
    # modifying a stored gate. No Warehouse environment is constructed.
    helper.test_actual_validation_failure_disables_feedback_without_choosing_other_coefficients(prepared)


@pytest.mark.parametrize('fault',['owner','NN_during_fit'])
def test_mutation_during_refit_cannot_produce_completed_pair(prepared,monkeypatch,fault):
    a=prepared;original=r.complete_program;calls=[]
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(args[0].role)
        if len(calls)==1:
            if fault=='owner':
                path=Path(a['owner']['path']);path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ')
            else:
                with torch.no_grad():a['state'].model.actors['robot_1'].tree.leaf_logits[0,4].add_(.001)
        return result
    monkeypatch.setattr(r,'complete_program',changed)
    with pytest.raises(RuntimeError,match='asset changed during fitting|changed the NN/critics/optimizers'):
        run(a)
    assert (r.EXTRACTION_ROOT/'refit/failed.json').exists()
    assert not (r.EXTRACTION_ROOT/'refit/completed.json').exists()
