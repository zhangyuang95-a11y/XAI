"""No-RNG export agrees bitwise with native Torch export, including -0."""
import random
import numpy as np
import pytest
import torch
from backend.training import warehouse_v642_checkpoint_export as x
from backend.training.warehouse_v622_actor_training import initialize_from_reference,save_checkpoint,export_current_actor
from backend.training.warehouse_v6_actor_training import role_hashes
from experiments.warehouse import v630_protocol as v


def test_checkpoint_arrays_equal_native_export_without_model_initialization_or_RNG(tmp_path,monkeypatch):
    _,plan=v.checked_plan();state=initialize_from_reference(plan['actor']['path'],plan['actor']['sha256'],plan['actor_role_sha256'],initialization_seed=0)
    with torch.no_grad():
        for actor in state.model.actors.values():
            actor.tree.split.weight.add_(.003)
            actor.tree.split.weight[~actor.tree.support_mask]=-.003
            actor.action_residual[-1].bias.add_(.01)
    state.joint_steps=20000
    cp=save_checkpoint(state,tmp_path/'synthetic.pt');ref=x.pointer(cp['path'])
    native=export_current_actor(state,tmp_path/'native.npz',pair_seed=0,variant='raw',candidate_id='synthetic-export-only')
    old_torch=torch.get_rng_state().clone();old_numpy=np.random.get_state();old_python=random.getstate()
    def forbidden(*args,**kwargs):pytest.fail('read-only export constructed an RNG or training model')
    from backend.training.warehouse_v6_actor_training import WarehouseV6Actor
    monkeypatch.setattr(WarehouseV6Actor,'__init__',forbidden)
    monkeypatch.setattr(torch,'manual_seed',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    out=x.export_checkpoint_actor(ref,tmp_path/'direct.npz',variant='raw',candidate_id='synthetic-export-only',expected_joint_steps=20000)
    assert out['role_sha256']==role_hashes(state.model)==native['role_sha256']
    with np.load(out['actor']['path']) as a,np.load(native['path']) as b:
        assert set(a.files)==set(b.files)
        for k in a.files:
            if k!='metadata_json':assert a[k].tobytes()==b[k].tobytes()
    assert torch.equal(old_torch,torch.get_rng_state()) and random.getstate()==old_python
    new_numpy=np.random.get_state();assert old_numpy[0]==new_numpy[0] and np.array_equal(old_numpy[1],new_numpy[1]) and old_numpy[2:]==new_numpy[2:]
    assert not out['new_RNG_constructors'] and not out['optimizer_steps']
    with pytest.raises(FileExistsError):x.export_checkpoint_actor(ref,tmp_path/'direct.npz',variant='raw',candidate_id='synthetic',expected_joint_steps=20000)
    with pytest.raises(RuntimeError):x.export_checkpoint_actor(ref,tmp_path/'wrong-step.npz',variant='raw',candidate_id='synthetic',expected_joint_steps=18600)
    assert not (tmp_path/'wrong-step.npz').exists()
