"""Synthetic terminal/checkpoint/receipt tests, no long scientific run."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import sys
import numpy as np
import torch
import pytest
from experiments.warehouse import v642_long_training_assets as a


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);a.write_new(path,value);path.chmod(0o444);return a.pointer(path)


@pytest.fixture
def terminal(tmp_path,monkeypatch):
    root=tmp_path/'source';monkeypatch.setattr(a,'ROOT',root)
    script=root/'scripts/run_warehouse_v642_long_training.py';script.parent.mkdir(parents=True);script.write_text('# synthetic source, never executed\n')
    base=tmp_path/'long-plan';ref=put(base/'plan.json',dict(synthetic_only=True))
    precheck=put(tmp_path/'precheck.json',dict(synthetic_only=True))
    done=put(base/'training/completed.json',dict(synthetic_only=True))
    result=put(base/'training/controller-completed.json',dict(synthetic_only=True))
    source=a.pointer(script);now=datetime.now(timezone.utc)
    command=[str(Path(sys.executable).resolve()),source['path'],'--run','--reference-directory',str(tmp_path/'reference'),
             '--calibration-precheck',precheck['path'],'--plan',ref['path']]
    started=put(base/'long-supervisor-started.json',dict(version='warehouse-v642-long-development-supervisor.v1',
        observed_by='source_bound_subprocess_launch',seed_plan=ref,calibration_precheck=precheck,command=command,
        supervisor_pid=10,started_utc=now.isoformat(),supervisor_source=source,log_path=str(base/'long-run.log')))
    log=base/'long-run.log';log.write_text('synthetic-only, no process launched');log.chmod(0o444)
    exit_value=dict(version='warehouse-v642-long-development-supervisor.v1',observed_by='source_bound_subprocess_wait',
        exit_code=0,result=result,completed=done,started=started,seed_plan=ref,command=command,
        worker_pid=11,supervisor_pid=10,started_utc=now.isoformat(),finished_utc=(now+timedelta(seconds=1)).isoformat(),
        supervisor_source=source,log=a.pointer(log))
    exit_ref=put(base/'long-supervisor-exit.json',exit_value)
    plan=SimpleNamespace(ref=ref,value=dict(execution_output=str(base/'training')))
    return SimpleNamespace(plan=plan,result=result,exit=exit_ref,value=exit_value,precheck=precheck,root=tmp_path)


def test_bound_supervisor_receipt_schema_is_recognized(terminal):
    t=terminal;inputs=[]
    assert a.audit_terminal(t.result,t.exit,plan=t.plan,reference_directory=t.root/'reference',
                            calibration_precheck_ref=t.precheck,inputs=inputs)==t.value
    assert len(inputs)==4


@pytest.mark.parametrize('change',['nonzero','boolean_exit','pid','clock','binding_error','fake_tool_receipt','command'])
def test_unobserved_or_mismatched_terminal_is_rejected(terminal,change):
    t=terminal;body=deepcopy(t.value)
    if change=='nonzero':body['exit_code']=1
    elif change=='boolean_exit':body['exit_code']=False
    elif change=='pid':body['worker_pid']=body['supervisor_pid']
    elif change=='clock':body['finished_utc']='2020-01-01T00:00:00+00:00'
    elif change=='binding_error':body['result_binding_error']='missing result'
    elif change=='fake_tool_receipt':body['observed_by']='exec_tool_terminal_result'
    else:body['command']=['unbound alternate worker']
    path=Path(t.exit['path']);path.chmod(0o644);path.unlink();put(path,body)
    with pytest.raises(RuntimeError):
        a.audit_terminal(t.result,a.pointer(path),plan=t.plan,reference_directory=t.root/'reference',
                         calibration_precheck_ref=t.precheck,inputs=[])


def test_closed_real_precheck_precedes_any_result_or_output_read(tmp_path,monkeypatch):
    def denied(*args):raise RuntimeError('closed independent calibration')
    def forbidden(*args,**kwargs):pytest.fail('closed calibration reached result or output')
    monkeypatch.setattr(a.owner.seeds.protocol,'register_preparation',denied)
    monkeypatch.setattr(a,'asset',forbidden)
    with pytest.raises(RuntimeError,match='closed independent'):
        a.run(reference_directory=None,calibration_precheck_ref=None,result_ref=None,supervisor_exit_ref=None,output=tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def native_receipt(tmp_path,*,start=4000,end=4512,batch=0):
    config=a.owner.PPOConfig(joint_step_budget=200000);n=end-start
    epochs=[dict(epoch=i,updates=[dict(epoch_id=f'batch-{batch}/epoch-{i}/minibatch-{offset}',
        physical_frames=min(256,n-offset)) for offset in range(0,n,256)]) for i in range(4)]
    return put(tmp_path/f'update-{batch}.json',dict(start_joint_step=start,end_joint_step=end,epochs=epochs,
                                                 batch_sha256='a'*64))


def test_native_layout_rejects_missing_epochs_and_wrong_normalization(tmp_path):
    ref=native_receipt(tmp_path);config=a.owner.PPOConfig(joint_step_budget=200000)
    a.audit_native_layout([ref],config)
    row=a.bound(ref);row['epochs'][1]['updates'][0]['physical_frames']=255
    bad=put(tmp_path/'bad.json',row)
    with pytest.raises(RuntimeError,match='normalization differs'):
        a.audit_native_layout([bad],config)
    row=a.bound(ref);row['epochs'].pop();bad=put(tmp_path/'missing.json',row)
    with pytest.raises(RuntimeError,match='epoch count changed'):
        a.audit_native_layout([bad],config)


def test_native_commits_require_exact_predebit_and_same_sequence(tmp_path):
    ref=native_receipt(tmp_path)
    events=[dict(kind='optimizer_update_predebit',variant='raw',step=4512,trace=dict(sha256='a'*64)),
            dict(kind='optimizer_update_accepted',variant='raw',step=4512,receipt=ref)]
    a.audit_commit_journal(events,dict(raw=[ref],regularized=[]))
    with pytest.raises(RuntimeError,match='lacks a predebit'):
        a.audit_commit_journal(events[1:],dict(raw=[ref],regularized=[]))
    with pytest.raises(RuntimeError,match='incomplete or reordered'):
        a.audit_commit_journal(events,dict(raw=[ref,ref],regularized=[]))


def test_checkpoint_actor_hashes_include_exact_masked_support_and_signed_zero():
    model={}
    for role in a.ROLES:
        prefix='actors.'+role+'.'
        for name in a._tensor_names(a.MODEL_VERSION):model[prefix+name]=torch.zeros((1,),dtype=torch.float32)
        model[prefix+'tree.split.weight']=torch.tensor([[-0.0,4.0]],dtype=torch.float32)
        model[prefix+'tree.support_mask']=torch.tensor([[True,False]])
        model[prefix+'tree.support_fixed']=torch.tensor([True])
    first=a.checkpoint_actor_hashes(dict(model=model))
    for role in a.ROLES:model['actors.'+role+'.tree.split.weight'][0,1]=999
    assert a.checkpoint_actor_hashes(dict(model=model))==first
    model['actors.robot_2.tree.support_fixed'][0]=False
    with pytest.raises(RuntimeError,match='fixed Actor feature support'):
        a.checkpoint_actor_hashes(dict(model=model))


def test_twenty_k_recovery_cannot_be_misread_as_long_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(a.owner.seeds.protocol,'register_preparation',lambda _:SimpleNamespace(verify=lambda:None))
    ref=put(tmp_path/'recovery.json',dict(version='warehouse-v642-explicit-calibration-recovery.v2',
        completed=True,numerical_training_completed=True,model_acceptance_passed=False,pilot_opened=False,publishable=False))
    with pytest.raises(RuntimeError,match='only completed nonpublication long-development'):
        a.audit_completion(reference_directory=None,calibration_precheck_ref=None,result_ref=ref,supervisor_exit_ref=None)


def test_report_reader_rejects_changed_sources_or_acceptance_claim(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'source_files',lambda:{})
    ref=put(tmp_path/'claimed.json',dict(version=a.VERSION,phase='development',audit_source_files={},
        model_acceptance_passed=True,pilot_opened=False,publishable=False))
    with pytest.raises(RuntimeError,match='nonpublication development artifact'):
        a.read_assets(ref)


@pytest.mark.parametrize('change',[None,'fingerprint','rng','critic','warmup_optimizer'])
def test_full_checkpoint_reader_never_constructs_model_optimizer_or_rng(tmp_path,monkeypatch,change):
    model={}
    for role in a.ROLES:
        prefix='actors.'+role+'.'
        for name in a._tensor_names(a.MODEL_VERSION):model[prefix+name]=torch.zeros(1)
        model[prefix+'tree.split.weight']=torch.zeros((1,1))
        model[prefix+'tree.support_mask']=torch.ones((1,1),dtype=torch.bool)
        model[prefix+'tree.support_fixed']=torch.ones(1,dtype=torch.bool)
    critic=sha256()
    for name in ('reward_critic','cost_critic'):
        array=np.zeros((1,1),dtype='<f4');key=name+'.weight';model[key]=torch.from_numpy(array)
        critic.update(key.encode());critic.update(np.asarray(array.shape,dtype='<i8').tobytes());critic.update(array.tobytes())
    reference=put(tmp_path/'synthetic-reference.json',dict(synthetic_only=True))
    roles={role:'a'*64 for role in a.ROLES}
    value=dict(schema=a.CHECKPOINT_SCHEMA,model_version=a.MODEL_VERSION,reference_path=reference['path'],
        reference_sha256=reference['sha256'],reference_role_sha256=roles,initialization_seed=1,model=model,
        actor_optimizer=dict(state={},param_groups=[]),critic_optimizer=dict(state={},param_groups=[]),
        beta={role:1. for role in a.ROLES},joint_steps=4000,actor_updates=0,critic_updates=16,
        rng=dict(python=(3,(),None),numpy=dict(name='MT19937',words=torch.zeros(624,dtype=torch.int64),
            position=0,gaussian=0,cached=0.),torch_cpu=torch.zeros(8,dtype=torch.uint8)),
        extra_rng_states={},requires_grad={'reward_critic.weight':True,'cost_critic.weight':True})
    plan=SimpleNamespace(value=dict(reference_actor=reference,reference_role_sha256=roles,initialization_seed=1))
    record=dict(full_training_state_sha256='expected-full-state',core=dict(actor_updates=0,critic_updates=16,
        beta=value['beta'],actor_role_sha256=a.checkpoint_actor_hashes(value),critic_sha256=critic.hexdigest()))
    def fingerprint(state):
        assert state.model.state_dict() is not None
        assert state.actor_optimizer.state_dict()['param_groups']==[]
        return 'expected-full-state'
    monkeypatch.setattr(a,'training_fingerprint',fingerprint)
    def forbidden(*args,**kwargs):pytest.fail('checkpoint readback constructed a training RNG or model')
    monkeypatch.setattr(torch,'manual_seed',forbidden);monkeypatch.setattr(np.random,'default_rng',forbidden)
    if change=='fingerprint':record['full_training_state_sha256']='tampered'
    elif change=='rng':value['rng'].pop('torch_cpu')
    elif change=='critic':record['core']['critic_sha256']='f'*64
    elif change=='warmup_optimizer':value['actor_optimizer']['state']={1:{}}
    path=tmp_path/'synthetic-checkpoint.pt';torch.save(value,path);path.chmod(0o444);ref=a.pointer(path)
    if change is None:
        assert a.checkpoint_state(ref,record,variant='raw',step=4000,plan=plan,inputs=[])['joint_steps']==4000
    else:
        with pytest.raises(RuntimeError):a.checkpoint_state(ref,record,variant='raw',step=4000,plan=plan,inputs=[])
