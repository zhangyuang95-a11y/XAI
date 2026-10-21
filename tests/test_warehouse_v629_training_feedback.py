"""Pure tensor feedback; synthetic NN only, no candidate/environment steps."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
import torch
from backend.training import warehouse_v629_training_feedback as f
from backend.training import warehouse_v629_online_evidence as o
from backend.training.warehouse_v622_actor_training import initialize_from_reference
from backend.training.warehouse_v626_program import Program as Parent
from backend.training.warehouse_v629_program import from_parent


@pytest.fixture(autouse=True)
def threads():
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(prior)


@pytest.fixture
def paired(tmp_path):
    spec=importlib.util.spec_from_file_location('v629_feedback_online_helpers',Path(__file__).with_name('test_warehouse_v629_online_evidence.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a=helper.assets.__wrapped__(tmp_path)
    state=initialize_from_reference(a['actor_ref']['path'],a['actor_ref']['sha256'],a['actor'].role_sha256,initialization_seed=0)
    epoch='synthetic-feedback-0'
    first=o.build_evidence(a['program_ref'],a['actor_ref'],a['pool_ref'],epoch_id=epoch)
    parent_value=o.bound(a['program'].protection['parent']);parent_value['role']='robot_2'
    parent=Parent.from_dict(parent_value,'a'*64);parent_path=tmp_path/'role2-parent.json';parent_path.write_text(json.dumps(parent.to_dict()))
    program=from_parent(parent,o.pointer(parent_path));program_path=tmp_path/'role2-program.json';program_path.write_text(json.dumps(program.to_dict()))
    manifest=deepcopy(a['manifest']);manifest['role']='robot_2';second_sample=None
    for split,part in a['parts'].items():
        p=o.runtime_probabilities(a['actor'],'robot_2',part['x'],part['m'])
        sampled=np.array([np.random.default_rng(o.independent_agent_seed(base_seed=r['action_seed'],episode_id=r['episode_id'],
            frame=r['frame'],agent_id='robot_2')).choice(5,p=q) for r,q in zip(part['rows'],p,strict=True)],np.int64)
        path=tmp_path/('role2-'+split+'.npz');np.savez_compressed(path,x=part['x'],m=part['m'],p=p,sampled=sampled)
        manifest['parts'][split]['arrays']=o.pointer(path)
        if split=='fit':second_sample=sampled
    manifest_path=tmp_path/'role2-manifest.json';manifest_path.write_text(json.dumps(manifest))
    second=o.build_evidence(o.pointer(program_path),a['actor_ref'],o.pointer(manifest_path),epoch_id=epoch)
    x=torch.from_numpy(np.stack([a['parts']['fit']['x']]*2,axis=1))
    masks=torch.from_numpy(np.stack([a['parts']['fit']['m']]*2,axis=1))
    sampled=torch.from_numpy(np.stack([a['parts']['fit']['sampled'],second_sample],axis=1))
    return state,x,masks,sampled,dict(robot_1=first,robot_2=second),epoch


@pytest.mark.parametrize('step,expected',[(0,0),(4000,0),(15999,0),(16000,0),(16500,.005),(17000,.01),(20000,.01)])
def test_program_schedule_exact_80_85_percent_boundaries_and_raw_zero(step,expected):
    assert f.coefficient(step,20000,'regularized')==pytest.approx(expected)
    assert f.coefficient(step,20000,'raw')==0


@pytest.mark.parametrize('variant,step',[('raw',20000),('regularized',15999)])
def test_raw_and_pre_window_do_not_inspect_evidence_or_get_program_gradient(paired,variant,step):
    state,x,masks,_,_,epoch=paired
    class Unreadable:
        def __getattribute__(self,name):raise AssertionError('program evidence inspected without feedback')
    result=f.feedback_loss(state.model,x,masks,None,joint_steps=step,total_steps=20000,
        variant=variant,epoch_id=epoch,evidence=Unreadable())
    assert result.program_loss.loss.item()==0 and not result.evidence
    assert all(value==0 for value in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert not any(result.diagnostics['selected_frames'].values())
    assert result.diagnostics['optimizer_steps']==0


def test_regularized_actual_positive_program_gradient_reaches_both_roles(paired):
    state,x,masks,sampled,evidence,epoch=paired
    result=f.feedback_loss(state.model,x,masks,sampled,joint_steps=18000,total_steps=20000,
        variant='regularized',epoch_id=epoch,evidence=evidence)
    assert result.diagnostics['coefficient']==.01
    assert all(value>0 for value in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert all(value>0 for value in result.diagnostics['selected_frames'].values())
    assert result.program_loss.physical_frames==len(x)
    result.program_loss.loss.backward()
    for role in ('robot_1','robot_2'):
        norm=sum(p.grad.detach().double().square().sum() for p in state.model.actors[role].parameters() if p.grad is not None).sqrt()
        assert float(norm)==pytest.approx(result.diagnostics['role_program_gradient_norm_before_shared_clip'][role])
    # A real synthetic Adam update changes both roles and invalidates old proof.
    # This never touches the scientific candidate or claims an admitted update.
    state.actor_optimizer.step()
    with pytest.raises(RuntimeError,match='Actor changed'):
        result.verify_before_optimizer_step(state.model,current_epoch_id=epoch)
    with pytest.raises(RuntimeError,match='expired after Actor update'):
        f.feedback_loss(state.model,x,masks,sampled,joint_steps=18001,total_steps=20000,
            variant='regularized',epoch_id=epoch,evidence=evidence)


def test_another_role_or_epoch_cannot_borrow_current_nn_qualification(paired):
    state,x,masks,sampled,evidence,epoch=paired
    with pytest.raises(ValueError,match='another role'):
        f.feedback_loss(state.model,x,masks,sampled,joint_steps=18000,total_steps=20000,
            variant='regularized',epoch_id=epoch,evidence={r:evidence['robot_1'] for r in evidence})
    with pytest.raises(RuntimeError,match='epoch change'):
        f.feedback_loss(state.model,x,masks,sampled,joint_steps=18000,total_steps=20000,
            variant='regularized',epoch_id='later-epoch',evidence=evidence)
    valid=f.feedback_loss(state.model,x,masks,sampled,joint_steps=18000,total_steps=20000,
        variant='regularized',epoch_id=epoch,evidence=evidence)
    with pytest.raises(RuntimeError,match='optimizer epoch changed'):
        valid.verify_before_optimizer_step(state.model,current_epoch_id='later-epoch')
