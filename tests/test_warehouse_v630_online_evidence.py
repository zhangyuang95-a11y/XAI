"""Synthetic full-NN tensors only; no candidate updates or environment use."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json

import numpy as np
import pytest
import torch

from backend.training import warehouse_v630_online_evidence as o
from backend.training import warehouse_v629_training_feedback as feedback
from backend.training.warehouse_v630_program import from_previous,split_identical
from backend.training.warehouse_v6_actor import FEATURE_NAMES


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(before)


@pytest.fixture
def paired(tmp_path):
    spec=importlib.util.spec_from_file_location('v630_online_tensor_fixture',
              Path(__file__).with_name('test_warehouse_v629_training_feedback.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    state,x,m,sampled,old,epoch=helper.paired.__wrapped__(tmp_path)
    evidence={};w=np.zeros(len(FEATURE_NAMES),np.float32);w[FEATURE_NAMES.index('self_column')]=1.
    for role,ev in old.items():
        p=from_previous(ev.program);p.maximum_leaves=256
        p,_=split_identical(p,0,w,-.5)
        target=tmp_path/(role+'-v630.json');target.write_text(json.dumps(p.to_dict()))
        evidence[role]=o.build_evidence(o.pointer(target),ev.target_actor_ref,ev.pool_ref,epoch_id=epoch)
    return state,x,m,sampled,evidence,epoch


def test_refined_program_has_separate_anchor_and_current_NN_bindings(paired):
    state,x,m,sampled,evidence,epoch=paired
    for role,ev in evidence.items():
        assert ev.program.actor_role_sha256=='a'*64 and ev.target_role_sha256!='a'*64
        assert ev.program.maximum_leaves==256 and ev.report['gate']['passed']
        assert ev.payload()['version']==o.VERSION
        assert ev.payload()['immutable_protected_parent'] in ev.asset_refs
        assert not ev.payload()['execution_actions_from_program'] and not ev.payload()['authorizes_training']
        assert not ev.program.feedback_leaves and ev.report['feedback_leaves']


def test_actual_program_gradients_reach_both_synthetic_Actors_then_old_evidence_expires(paired):
    state,x,m,sampled,evidence,epoch=paired
    result=feedback.feedback_loss(state.model,x,m,sampled,joint_steps=18000,total_steps=20000,
        variant='regularized',epoch_id=epoch,evidence=evidence)
    assert all(v>0 for v in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())
    assert all(v>0 for v in result.diagnostics['selected_frames'].values())
    assert result.program_loss.physical_frames==len(x)
    before=feedback.role_hashes(state.model)
    result.program_loss.loss.backward();state.actor_optimizer.step()
    assert all(before[r]!=feedback.role_hashes(state.model)[r] for r in before)
    with pytest.raises(RuntimeError,match='expired after Actor update'):
        feedback.feedback_loss(state.model,x,m,sampled,joint_steps=18001,total_steps=20000,
            variant='regularized',epoch_id=epoch,evidence=evidence)


def test_raw_never_reads_new_program_evidence_and_receives_zero_program_gradient(paired):
    state,x,m,sampled,_,epoch=paired
    class Unreadable:
        def __getattribute__(self,name):pytest.fail('raw inspected external program evidence')
    result=feedback.feedback_loss(state.model,x,m,None,joint_steps=18000,total_steps=20000,
        variant='raw',epoch_id=epoch,evidence=Unreadable())
    assert result.program_loss.loss.item()==0 and not result.evidence
    assert all(v==0 for v in result.diagnostics['role_program_gradient_norm_before_shared_clip'].values())


def test_independent_current_NN_validation_failure_disables_refined_feedback(paired,monkeypatch):
    *_,evidence,epoch=paired;before=evidence['robot_1'];original=o.legacy.current_report
    def failed(program,parts):
        report=original(program,parts)
        assert report['feedback_leaves']
        report['validation']['feedback']['agreement']=.94
        report['gate']=o.legacy.gate(report['validation'])
        return report
    monkeypatch.setattr(o.legacy,'current_report',failed)
    ev=o.build_evidence(before.program_ref,before.target_actor_ref,before.pool_ref,epoch_id=epoch)
    targets=ev.loss_targets(paired[1][:,0].numpy(),paired[2][:,0].numpy(),
                           current_role_sha256=ev.target_role_sha256,epoch_id=epoch)
    assert not ev.report['gate']['passed'] and not targets['qualified'].any()


def test_owner_source_closure_includes_native_reader_and_actual_gradient_kernel():
    from experiments.warehouse import v630_protocol as v
    files=o.source_files()
    for name in ('backend/training/warehouse_v630_online_evidence.py',
                 'backend/training/warehouse_v630_program.py',
                 'backend/training/warehouse_v629_training_feedback.py',
                 'backend/training/warehouse_v622_continuation_safety.py'):
        assert files[name]==v.file_sha(v.ROOT/name)


def test_serialized_new_schema_replays_and_old_reader_cannot_reinterpret_it(paired,tmp_path):
    *_,evidence,epoch=paired;ev=evidence['robot_1']
    path=tmp_path/'v630-evidence.json';path.write_text(json.dumps(ev.payload()));ref=o.pointer(path)
    assert o.load_evidence(ref,current_role_sha256=ev.target_role_sha256,epoch_id=epoch).payload()==ev.payload()
    with pytest.raises(RuntimeError,match='unknown online evidence schema'):
        o.legacy.load_evidence(ref,current_role_sha256=ev.target_role_sha256,epoch_id=epoch)
    value=ev.payload();value['report']['feedback_leaves']=[];path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='independently replay'):
        o.load_evidence(o.pointer(path),current_role_sha256=ev.target_role_sha256,epoch_id=epoch)


@pytest.mark.parametrize('fault',['current_hash','epoch','anchor_parent'])
def test_changed_current_NN_epoch_or_immutable_parent_cannot_supply_feedback(paired,fault):
    *_,evidence,epoch=paired;ev=evidence['robot_1'];sha=ev.target_role_sha256
    if fault=='current_hash':sha='b'*64
    elif fault=='epoch':epoch='another-update'
    else:Path(ev.program.protection['parent']['path']).write_text('{}')
    with pytest.raises((RuntimeError,ValueError)):
        ev.loss_targets(paired[1][:,0].numpy(),paired[2][:,0].numpy(),current_role_sha256=sha,epoch_id=epoch)
