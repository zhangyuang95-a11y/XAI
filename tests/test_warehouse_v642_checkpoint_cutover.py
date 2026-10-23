"""Synthetic cutover only. Signals, process inspection and OS wait are fakes."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import signal
import pytest,torch
from experiments.warehouse import v642_checkpoint_cutover as c


def put(path,value,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True);c.write_new(path,value);path.chmod(mode);return c.pointer(path)


@pytest.fixture
def controlled(tmp_path,monkeypatch):
    """Named fake identity/state gate. No signal ever reaches a real PID."""
    old=tmp_path/'fake-v2';old.mkdir();now=datetime.now(timezone.utc)
    recovery=put(old/'plan.json',dict(synthetic_only=True));claim=put(old/'claim.json',dict(synthetic_only=True))
    command=['/synthetic/python','/synthetic/registered-recovery.py','--run','--recovery-plan',recovery['path']]
    launch=put(old/'supervisor-started.json',dict(version='warehouse-v642-calibration-recovery-supervisor.v1',
        observed_by='source_bound_subprocess_launch',recovery_plan=recovery,command=command,
        supervisor_pid=900001,started_utc=(now-timedelta(minutes=1)).isoformat(),supervisor_source=dict(path='synthetic',sha256='a'*64)))
    checkpoint=put(old/'original-safe-owner-state.json',dict(synthetic_only=True));state=put(old/'state.pt',dict(synthetic_only=True))
    logs={name:put(old/name,dict(synthetic_only=True),mode=0o644) for name in ('owner-journal.jsonl','actual-resets.jsonl','run.log')}
    b=dict(recovery_plan=recovery,recovery_claim=claim,supervisor_started=launch,worker_pid=900002,supervisor_pid=900001,
        command=command,target_checkpoint=checkpoint['path'],mode='original-safe-replay',expires_utc=(now+timedelta(hours=1)).isoformat(),
        frozen_v2_source_files={},original_source_files={})
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'external-only')
    def fake_binding(*a,**kw):return deepcopy(b)
    monkeypatch.setattr(c,'_binding',fake_binding)
    process=dict(pid=900002,ppid=900001,status='R',command=' '.join(command));signals=[]
    monkeypatch.setattr(c,'_process',lambda pid:dict(process))
    plan=c.prepare(recovery_plan_ref=recovery,supervisor_started_ref=launch,worker_pid=900002,
        target_checkpoint_path=checkpoint['path'],mode=b['mode'],cutover_id='synthetic-only')
    safe=dict(version=c.VERSION,plan=plan,binding=b,checkpoint=checkpoint,checkpoint_facts=dict(training_state=state),
        logs=logs,journal_last_sequence=7,reset_consumption=dict(synthetic_only=True),
        discarded_work='only_uncommitted_numerical_PPO_computation_no_new_physical_samples_or_commits',
        no_new_seed_or_environment_authority=True,model_acceptance_passed=False,pilot_opened=False)
    monkeypatch.setattr(c,'assert_safe',lambda ref,**kw:deepcopy(safe))
    source_name='experiments/warehouse/v642_checkpoint_cutover.py';sha=c.file_sha(c.ROOT/source_name)
    snapshot=tmp_path/'source-snapshot.py';snapshot.write_bytes((c.ROOT/source_name).read_bytes());snapshot.chmod(0o444)
    sources={source_name:sha}
    source=put(tmp_path/'successor-source-contract.json',dict(version='warehouse-v642-accelerated-recovery-source-contract.v1',
        source_files=sources,source_sha256=c.digest(sources),source_snapshots={source_name:c.pointer(snapshot)},training_authority=False))
    def fake_kill(pid,sig):
        assert pid==900002;signals.append(sig)
        if sig==signal.SIGSTOP:process['status']='T'
        elif sig==signal.SIGTERM:process['terminating']=True
        elif sig==signal.SIGCONT:
            process['status']='R'
            if process.get('terminating'):
                value=dict(version=c.bound(launch)['version'],observed_by='source_bound_subprocess_wait',recovery_plan=recovery,
                    started=launch,command=command,worker_pid=900002,supervisor_pid=900001,
                    supervisor_source=c.bound(launch)['supervisor_source'],exit_code=-15,log=logs['run.log'],
                    finished_utc=datetime.now(timezone.utc).isoformat())
                put(old/'supervisor-exit.json',value)
    monkeypatch.setattr(c.os,'kill',fake_kill)
    values=dict(authorization=c.v2.AUTHORIZATION,successor_source_contract_ref=source,worker_pid=900002,
        target_checkpoint_path=checkpoint['path'],supervisor_started_ref=launch)
    return SimpleNamespace(plan=plan,b=b,old=old,safe=safe,values=values,signals=signals,source=source)


def test_prepare_and_assert_never_signal_and_execute_is_explicit(controlled):
    x=controlled
    assert not x.signals and c.checked_plan(x.plan)['default_action']=='assert_only_no_signals'
    c.assert_safe(x.plan);assert not x.signals
    with pytest.raises(RuntimeError,match='explicit execute'):c.execute(x.plan,**dict(x.values,authorization=''))
    assert not x.signals


def test_explicit_verified_stop_term_cont_requires_fake_actual_wait_minus15_and_offline_reader(controlled):
    x=controlled;result=c.execute(x.plan,**x.values)
    assert x.signals==[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT]
    done=c.bound(result)
    assert done['actual_exit_code']==-15 and not done['training_completed']
    assert not done['scientific_success_claimed'] and not done['scientific_failure_claimed']
    assert all(not Path(ref['path']).stat().st_mode&0o222 for ref in x.safe['logs'].values())
    report=c.read_completed(result)
    assert report['parent_v2_plan']==x.b['recovery_plan'] and report['parent_v2_claim']==x.b['recovery_claim']
    assert report['source_contract']==x.source and report['mode']=='original-safe-replay'
    assert all(c.pointer(ref['path'])==ref for ref in report['inputs'])


@pytest.mark.parametrize('field,value',[('worker_pid',999999),('target_checkpoint_path','/unrelated/checkpoint'),
    ('supervisor_started_ref',dict(path='/unrelated/launch',sha256='a'*64))])
def test_explicit_mismatched_PID_target_or_supervisor_is_rejected_before_any_signal(controlled,field,value):
    x=controlled
    with pytest.raises(RuntimeError,match='explicit execute'):c.execute(x.plan,**dict(x.values,**{field:value}))
    assert not x.signals


def test_missing_or_passed_only_successor_source_never_stops_worker(controlled):
    x=controlled;bad=put(Path(x.plan['path']).parent/'passed-only.json',dict(version='passed',ready=True,passed=True))
    with pytest.raises((RuntimeError,KeyError)):c.execute(x.plan,**dict(x.values,successor_source_contract_ref=bad))
    assert not x.signals


def test_post_checkpoint_or_new_commit_race_resumes_worker_and_never_TERMs(controlled,monkeypatch):
    x=controlled
    def refused(*a,**kw):raise RuntimeError('new 18600 commit or physical suffix raced with stop')
    monkeypatch.setattr(c,'assert_safe',refused)
    with pytest.raises(RuntimeError,match='raced'):c.execute(x.plan,**x.values)
    assert x.signals==[signal.SIGSTOP,signal.SIGCONT]
    folder=Path(x.plan['path']).parent
    assert (folder/'cutover-refused.json').exists() and not (folder/'prestop-verified.json').exists()
    assert not (x.old/'supervisor-exit.json').exists()


def test_true_exit1_or_missing_result_cannot_be_presented_as_a_cutover(controlled,monkeypatch):
    x=controlled;original=c.os.kill
    def exit1(pid,sig):
        original(pid,sig)
        if sig==signal.SIGCONT and (x.old/'supervisor-exit.json').exists():
            path=x.old/'supervisor-exit.json';p=c.bound(c.pointer(path));p['exit_code']=1
            path.chmod(0o644);path.write_text(__import__('json').dumps(p));path.chmod(0o444)
    monkeypatch.setattr(c.os,'kill',exit1)
    with pytest.raises(RuntimeError,match='actual OS wait'):c.execute(x.plan,**x.values)
    assert not (Path(x.plan['path']).parent/'cutover-completed.json').exists()


def test_arbitrary_unpinned_plan_is_rejected_by_real_identity_gate_without_signals(tmp_path,monkeypatch):
    ref=put(tmp_path/'unrelated-plan.json',dict(passed=True))
    def forbidden(*a):pytest.fail('arbitrary PID was signaled')
    monkeypatch.setattr(c.os,'kill',forbidden)
    with pytest.raises(RuntimeError,match='pinned current v2'):
        c.prepare(recovery_plan_ref=ref,supervisor_started_ref=ref,worker_pid=42,
            target_checkpoint_path='/anything',mode='original-safe-replay',cutover_id='bad')


def safe_state(tmp_path,*,step=16800):
    reference=tmp_path/'reference.npz';reference.write_bytes(b'synthetic-reference-only');reference.chmod(0o444)
    state=dict(schema=c.CHECKPOINT_SCHEMA,model_version=c.MODEL_VERSION,reference_path=str(reference),reference_sha256=c.file_sha(reference),
        initialization_seed=1,beta=dict(robot_1=1.,robot_2=1.),joint_steps=step,actor_updates=2,critic_updates=3,
        model={'synthetic.weight':torch.tensor([1.,2.])},actor_optimizer={'state':{},'param_groups':[]},critic_optimizer={'state':{},'param_groups':[]},
        requires_grad={'synthetic.weight':True},rng=dict(python=[1],numpy={},torch_cpu=torch.tensor([1],dtype=torch.uint8)),extra_rng_states={})
    path=tmp_path/'state.pt';torch.save(state,path);path.chmod(0o444)
    carrier=SimpleNamespace(**{k:state[k] for k in ('reference_path','reference_sha256','initialization_seed','beta','joint_steps','actor_updates','critic_updates')})
    carrier.model=SimpleNamespace(state_dict=lambda:state['model'],named_parameters=lambda:[('synthetic.weight',SimpleNamespace(requires_grad=True))])
    carrier.actor_optimizer=SimpleNamespace(state_dict=lambda:state['actor_optimizer']);carrier.critic_optimizer=SimpleNamespace(state_dict=lambda:state['critic_optimizer'])
    h=c.training_fingerprint(carrier);owner=put(tmp_path/'owner.json',dict(synthetic_only=True))
    public=dict(processed_joint_steps=step,safe_assembly_boundary=True,full_training_state_sha256=h,owner=owner)
    reader=dict(public=dict(payload=public,payload_sha256=c.digest(public)),owner=owner,environments=[None]*8,active=[None]*8)
    provider=dict(version='warehouse-v642-current-NN-provider.v1',full_training_state_sha256=h,call_count=32,
        installed=[dict(step=n) for n in (15000,16800)],config={'synthetic':True},variant='regularized',latch=False,
        used_epochs=[str(i) for i in range(32)],events=[],programs={},owner=owner)
    return dict(training_state=c.pointer(path),reserved_joint_steps=step,core=dict(processed_joint_steps=step,safe_batch_boundary=True,failed=False,
        actor_updates=2,critic_updates=3,beta=state['beta'],rng_state=dict(bit_generator='PCG64'),seen_batch_receipts=[],update_sequence=0,
        variant='regularized',config={'synthetic':True},reference_sha256=state['reference_sha256'],risk={},seen_complete_episodes=[],
        accepted_program_updates={},source_sha256=c.file_sha(c.ROOT/'backend/training/warehouse_v629_ppo_update.py')),
        reader=dict(payload=reader,payload_sha256=c.digest(reader)),provider=dict(payload=provider,payload_sha256=c.digest(provider)),
        environments=[None]*8,active_cases=[None]*8,reset_cursors=[0]*8,program_source_cursor=1650,refits=[15000,16800],owner=owner)


def test_full_tensor_core_reader_provider_fingerprint_check_uses_no_model_or_RNG_construction(tmp_path):
    s=safe_state(tmp_path)
    result=c._saved_state(s,step=16800,provider_version='warehouse-v642-current-NN-provider.v1',provider_calls=32,installs=[15000,16800])
    assert result['restores_full_RNG_without_new_draws'] and result['step']==16800
    s['core']['safe_batch_boundary']=False
    with pytest.raises(RuntimeError,match='unsafe'):c._saved_state(s,step=16800,
        provider_version='warehouse-v642-current-NN-provider.v1',provider_calls=32,installs=[15000,16800])
