"""Pure synthetic ownership/wiring tests; no candidate NN, seeds or environment.

All admission callbacks, NN/critic/optimizer/RNG and physical kernels below
are named fakes. A passing test is never scientific calibration evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import ast,importlib.util,json

import pytest
from experiments.warehouse import v642_long_training_owner as o


class ProbeFinished(Exception):pass


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    o.write_new(path,value);path.chmod(0o444);return o.pointer(path)


def cli_module():
    path=o.ROOT/'scripts/run_warehouse_v642_long_training.py'
    spec=importlib.util.spec_from_file_location('synthetic_long_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_real_gate_precedes_input_directory_source_constructor_and_RNG(tmp_path,monkeypatch):
    seen=[]
    def denied(*args):seen.append('independent-calibration-replay');raise RuntimeError('fake closed calibration precheck')
    def forbidden(*args,**kwargs):pytest.fail('closed gate reached long input, output, environment, model or RNG')
    monkeypatch.setattr(o.seeds.protocol,'register_preparation',denied)
    monkeypatch.setattr(o,'bound',forbidden);monkeypatch.setattr(o,'source_files',forbidden)
    monkeypatch.setattr(o.seeds,'bound',forbidden);monkeypatch.setattr(o,'initialize_from_reference',forbidden)
    monkeypatch.setattr(o,'WarehouseMultiAgentEnv',forbidden);monkeypatch.setattr(o.np.random,'default_rng',forbidden)
    for call in (o.prepare_sources,o.audit_source_bound_seeds,o.run):
        with pytest.raises(RuntimeError,match='closed calibration'):
            call(reference_directory=tmp_path/'missing',calibration_precheck_ref=None,plan_ref=None)
    assert seen==['independent-calibration-replay']*3 and not list(tmp_path.iterdir())


def test_bare_owner_and_unclaimed_factory_grant_nothing():
    with pytest.raises(RuntimeError,match='single-use claimed_owner'):o.LongTrainingOwner()
    with pytest.raises(RuntimeError,match='source-bound claimed context'):
        o._make_owner(None,None,None,None,None,None,None,None,token=object())


@dataclass
class FakeState:
    joint_steps:int=0
    actor_updates:int=0
    critic_updates:int=0
    initialization_seed:int=41
    reference_sha256:str='reference-sha'
    resumed_training_state:bool=False
    revision:int=0
    def advance_joint_steps(self,n):self.joint_steps+=n


class FakeOptimizer:
    def state_dict(self):return dict(state={},param_groups=[dict(lr=0.1)])


def fake_fingerprint(state):
    return o.digest(dict(step=state.joint_steps,actor=state.actor_updates,critic=state.critic_updates,
        revision=state.revision,initialization_seed=state.initialization_seed,reference=state.reference_sha256))


class FakeCore:
    def __init__(self,state,buffer,*,variant,rng,config):
        self.state=state;self.variant=variant;self.rng=rng;self.config=config
        self.processed_joint_steps=0;self.failed=False;self.accepted_program_updates=dict(robot_1=0,robot_2=0)
    def update_batch(self,batch,*,batch_sha256,evidence_provider):
        actor=self.state.joint_steps>4000
        self.state.critic_updates+=1;self.state.actor_updates+=int(actor)
        self.processed_joint_steps=self.state.joint_steps
        coefficient=(0 if self.variant=='raw' else self.config.program_coefficient(self.state.joint_steps))
        counts={r:int(coefficient>0 and evidence_provider is not None) for r in self.accepted_program_updates}
        for r,n in counts.items():self.accepted_program_updates[r]+=n
        return dict(synthetic_only=True,accepted_positive_program_gradient_updates=counts,
                    actor_enabled=actor,batch_sha256=batch_sha256,program_coefficient=coefficient)
    def snapshot(self):
        return dict(synthetic_only=True,safe_batch_boundary=True,config=asdict(self.config),
            processed_joint_steps=self.processed_joint_steps,accepted_program_updates=self.accepted_program_updates)


class FakeReader:
    def __init__(self,owner,inventory,*,config):self.owner=owner;self.inventory=inventory;self.config=config
    def snapshot(self,state):return dict(synthetic_only=True,processed_joint_steps=state.joint_steps,complete_cost_history='fake')


@pytest.fixture
def prepared(tmp_path,monkeypatch,request):
    folder=tmp_path/'long-plans'/'synthetic-only';folder.mkdir(parents=True)
    reference=put(tmp_path/'fake-reference.json',dict(synthetic_only=True))
    train=[]
    for slot in range(8):
        for i in range(2):train.append(dict(episode_id=f'fake-train-{slot}-{i}',slot=slot,scenario_seed=1000+slot*10+i,
            action_seed=2000+slot*10+i,task=1,partition='train'))
    program=[dict(episode=f'fake-program-{i}',scenario_seed=3000+i,action_seed=4000+i,split='fit') for i in range(4)]
    train_ref=put(tmp_path/'fake-training.json',dict(cases=train));program_ref=put(tmp_path/'fake-program.json',dict(cases=program))
    plan_ref=put(folder/'plan.json',dict(synthetic_only=True))
    buffer_ref=put(tmp_path/'fake-buffer.json',dict(synthetic_only=True))
    precheck=put(tmp_path/'fake-precheck.json',dict(synthetic_only=True))
    budget=getattr(request,"param",200000)
    config=o.PPOConfig(joint_step_budget=budget)
    ready=SimpleNamespace(value=dict(actor=dict(path=reference['path'],sha256='reference-sha'),
        reference=dict(buffer=buffer_ref),program_pair={r:dict(language=dict(anchor_actor_role_sha256=r+'-sha'),
            program=dict(path='unused-'+r,sha256='not-loaded')) for r in ('robot_1','robot_2')}))
    value=dict(phase='development',stream_id='development-0',joint_step_budget=budget,
        initialization_seed=41,optimizer_seed=43,train_case_inventory=train_ref,program_evidence_inventory=program_ref,
        config=asdict(config),execution_output=str(folder/'training'),restart_proof=None,
        execution_lease_days={200000:7,300000:10,400000:14}[budget],seed_clearance_start_window_hours=24)
    plan=SimpleNamespace(ref=plan_ref,value=value,config=config,readiness=ready,verify=lambda:None,
        registration=SimpleNamespace(value=dict(synthetic_only=True)))
    plan.proposal=plan
    monkeypatch.setattr(o.seeds,'OUTPUT_ROOT',tmp_path/'long-plans')
    value['global_claim_path']=str(o.claim_path(plan))
    calls=[]
    def checked(ref,*,calibration_precheck_ref,reference_directory):
        assert ref==plan.ref and calibration_precheck_ref==precheck;calls.append('fake-real-precheck');return plan
    monkeypatch.setattr(o.seeds,'check',checked)
    sources={name:o.file_sha(o.ROOT/name) for name in o.FILES}
    monkeypatch.setattr(o,'source_files',lambda:dict(sources))
    clearance_ref=put(folder/'fake-clearance.json',dict(synthetic_only=True))
    now=datetime.now(timezone.utc)
    clearance=dict(clearance=clearance_ref,value=dict(issued_utc=(now-timedelta(hours=1)).isoformat(),
        expires_utc=(now+timedelta(hours=23)).isoformat()))
    monkeypatch.setattr(o.seeds,'audit',lambda p:clearance)
    monkeypatch.setattr(o.seeds,'checked_clearance',lambda p,**kw:clearance)
    values=dict(reference_directory=tmp_path/'unused-reference',calibration_precheck_ref=precheck,plan_ref=plan.ref)
    o.prepare_sources(**values);o.audit_source_bound_seeds(**values)
    return SimpleNamespace(plan=plan,values=values,clearance=clearance,calls=calls,folder=folder,source_files=sources)


@pytest.fixture
def kernels(monkeypatch):
    initialized=[];random_seeds=[];checked_buffers=[]
    def initialize(path,sha,roles,**kw):
        state=FakeState(initialization_seed=kw['initialization_seed'],reference_sha256=sha)
        state.model=SimpleNamespace(role_sha256=dict(roles));state.actor_optimizer=FakeOptimizer();state.critic_optimizer=FakeOptimizer()
        initialized.append(state);return state
    monkeypatch.setattr(o,'initialize_from_reference',initialize)
    monkeypatch.setattr(o,'role_hashes',lambda model:dict(model.role_sha256))
    monkeypatch.setattr(o,'critic_initialization_sha256',lambda model:'fake-common-critic')
    monkeypatch.setattr(o,'training_fingerprint',fake_fingerprint)
    def rng(seed):random_seeds.append(seed);return SimpleNamespace(seed=seed)
    monkeypatch.setattr(o.np.random,'default_rng',rng)
    monkeypatch.setattr(o,'ReferenceTrainingBuffer',SimpleNamespace(load=lambda *a:SimpleNamespace(synthetic_only=True)))
    monkeypatch.setattr(o.seeds,'check_reference_buffer',lambda p,b:checked_buffers.append((p,b)))
    monkeypatch.setattr(o,'PPOUpdateCore',FakeCore);monkeypatch.setattr(o,'EnvironmentRolloutReplay',FakeReader)
    return SimpleNamespace(initialized=initialized,random_seeds=random_seeds,checked_buffers=checked_buffers)


def enter(p):return o.claimed_owner(**p.values)


def test_actual_claim_context_initializes_all_owner_fields_and_identical_fresh_states(prepared,kernels):
    with pytest.raises(ProbeFinished):
        with enter(prepared) as active:
            raw=active.initialize('raw');reg=active.initialize('regularized')
            assert raw is not reg and fake_fingerprint(raw)==fake_fingerprint(reg)
            assert active.refits==dict(raw=[],regularized=[])
            assert kernels.random_seeds==[43,43] and len(kernels.checked_buffers)==2
            assert active.initial_identity['full_initial_training_state_sha256']==fake_fingerprint(raw)
            tree=ast.parse((o.ROOT/'experiments/warehouse/v642_long_training_owner.py').read_text())
            klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LongTrainingOwner')
            reads={n.attr for n in ast.walk(klass) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
                   and n.value.id=='self' and isinstance(n.ctx,ast.Load)}
            # Context-created handles and methods are present as well; catches
            # initialization omissions such as the earlier recovery.refits bug.
            assert all(hasattr(active,key) for key in reads)
            assert o.ACTIVE.get() is active
            raise ProbeFinished()
    assert o.ACTIVE.get() is None and active.closed and active.failed
    assert o.claim_path(prepared.plan).is_file() and (active.folder/'failed.json').is_file()
    assert not (active.folder/'completed.json').exists()


@pytest.mark.parametrize('change',['resumed','nonzero','actor','optimizer','critic'])
def test_fresh_initialization_and_pair_identity_fail_before_any_reset(prepared,kernels,monkeypatch,change):
    original=o.initialize_from_reference
    def invalid(*a,**kw):
        state=original(*a,**kw)
        if change=='resumed':state.resumed_training_state=True
        elif change=='nonzero':state.joint_steps=20000
        elif change=='actor':state.model.role_sha256['robot_1']='wrong-role'
        elif change=='optimizer':state.actor_optimizer=SimpleNamespace(state_dict=lambda:dict(state={'old':'momentum'}))
        elif change=='critic' and len(kernels.initialized)==2:state.revision=1
        return state
    monkeypatch.setattr(o,'initialize_from_reference',invalid)
    with pytest.raises(RuntimeError,match='step zero|Actor roles|initial Actor or critic'):
        with enter(prepared) as active:
            active.initialize('raw');active.initialize('regularized')
    assert not active.actual_resets


@pytest.mark.parametrize('change',['formal','config','old20k'])
def test_formal_budget_or_config_rejected_before_claim(prepared,change):
    if change=='formal':prepared.plan.value['phase']='formal'
    elif change=='config':prepared.plan.value['config']['actor_learning_rate']=1.
    else:
        prepared.plan.config=o.PPOConfig();prepared.plan.value['config']=asdict(prepared.plan.config)
    with pytest.raises(RuntimeError,match='formal long training|unknown long development'):
        with enter(prepared):pytest.fail('unadmitted stage claimed')
    assert not o.claim_path(prepared.plan).exists() and not (prepared.folder/'training').exists()


def test_source_snapshot_tamper_rejects_before_claim(prepared):
    contract=o.bound(o.checked_sources(prepared.plan));name=o.FILES[0]
    path=prepared.folder/'long-owner-source'/contract['source_sha256']/name
    path.chmod(0o644);path.write_text('# synthetic tamper');path.chmod(0o444)
    with pytest.raises(RuntimeError,match='source snapshot changed'):
        with enter(prepared):pytest.fail('changed executor claimed')
    assert not o.claim_path(prepared.plan).exists()


def test_partial_attempt_consumes_global_stream_budget_even_across_run_ids(prepared,kernels):
    with pytest.raises(RuntimeError,match='pair is incomplete'):
        with enter(prepared) as active:active.initialize('raw')
    original=o.pointer(o.claim_path(prepared.plan))
    other=deepcopy(prepared.plan);other.ref=put(prepared.folder.parent/'renamed-run'/'plan.json',dict(synthetic_only=True))
    assert o.claim_path(other)==o.claim_path(prepared.plan)
    with pytest.raises(RuntimeError,match='already consumed'):
        with enter(prepared):pytest.fail('single-use attempt repeated')
    assert o.pointer(o.claim_path(prepared.plan))==original


@pytest.mark.parametrize('budget,days',[(200000,7),(300000,10),(400000,14)])
def test_seed_start_deadline_and_budget_bound_nonrenewable_execution_lease(budget,days):
    now=datetime.now(timezone.utc)
    plan=SimpleNamespace(config=o.PPOConfig(joint_step_budget=budget),value=dict(
        execution_lease_days=days,seed_clearance_start_window_hours=24))
    clear=dict(value=dict(issued_utc=(now-timedelta(hours=1)).isoformat(),expires_utc=(now+timedelta(hours=23)).isoformat()))
    lease=o.execution_lease(plan,clear,started=now)
    assert datetime.fromisoformat(lease['expires_utc'])-now==timedelta(days=days)
    assert lease['seed_clearance_start_deadline_utc']==clear['value']['expires_utc'] and not lease['automatic_renewal']
    plan.value['execution_lease_days']=365000
    with pytest.raises(RuntimeError,match='lease protocol differs'):o.execution_lease(plan,clear,started=now)


def test_expired_start_clearance_cannot_create_claim_folder_or_model(prepared,kernels):
    now=datetime.now(timezone.utc)
    prepared.clearance['value'].update(issued_utc=(now-timedelta(hours=25)).isoformat(),expires_utc=(now-timedelta(hours=1)).isoformat())
    with pytest.raises(RuntimeError,match='starting only before'):
        with enter(prepared):pytest.fail('expired seed clearance started a long run')
    assert not o.claim_path(prepared.plan).exists() and not (prepared.folder/'training').exists()
    assert not kernels.initialized and not kernels.random_seeds


@pytest.mark.parametrize('days_elapsed,allowed',[(2,True),(7,False)])
def test_active_long_run_uses_execution_lease_not_expired_24h_start_window(prepared,kernels,monkeypatch,days_elapsed,allowed):
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return simulated
    with pytest.raises(ProbeFinished if allowed else RuntimeError,match=None if allowed else 'execution lease expired'):
        with enter(prepared) as active:
            start=datetime.fromisoformat(o.bound(active.claim_ref)['started_utc'])
            simulated=start+timedelta(days=days_elapsed,seconds=1)
            monkeypatch.setattr(o,'datetime',Clock)
            if allowed:
                active.live(full=True);assert simulated>datetime.fromisoformat(prepared.clearance['value']['expires_utc'])
                raise ProbeFinished()
            active.live()
    assert active.closed and (active.folder/'failed.json').is_file()


def test_claim_cannot_be_rewritten_to_extend_the_execution_lease(prepared,kernels):
    with pytest.raises(RuntimeError,match='claim or descriptor changed'):
        with enter(prepared) as active:
            claim=o.bound(active.claim_ref);claim['expires_utc']=(datetime.now(timezone.utc)+timedelta(days=365)).isoformat()
            path=Path(active.claim_ref['path']);path.chmod(0o644);path.write_text(json.dumps(claim));path.chmod(0o444)
            active.live()


def test_larger_budget_restart_must_match_actual_full_initial_identity(prepared,kernels):
    prepared.plan.value['restart_proof']=dict(expected_initial_identity=dict(not_the_real_critic_or_optimizers=True))
    with pytest.raises(RuntimeError,match='identical full initial state'):
        with enter(prepared) as active:active.initialize('raw')
    assert len(kernels.initialized)==1 and not kernels.random_seeds and not active.actual_resets


def install_physics_fakes(monkeypatch):
    from env.warehouse.reset_audit_v614 import record_successful_reset
    class Env:
        def __init__(self,seed):self.seed=seed
        def reset(self,*,seed):
            assert seed==self.seed
            self.state=SimpleNamespace(frame=0,episode_id=1,terminated=False,truncated=False)
            record_successful_reset(seed,1,'synthetic-only-layout',1.,1.)
        def step(self,actions):
            self.state.frame+=1;self.state.truncated=self.state.frame==120
            return None,dict(robot_1=0.,robot_2=0.),False,self.state.truncated,dict(
                proposed_actions=actions,action_override_agents=[],shutdowns=[])
    monkeypatch.setattr(o,'WarehouseMultiAgentEnv',Env);monkeypatch.setattr(o,'training_config',lambda seed:seed)
    monkeypatch.setattr(o,'_set_dual_nn_mode',lambda env:None)
    monkeypatch.setattr(o,'capture_environment',lambda env:dict(synthetic_only=True,frame=env.state.frame,seed=env.seed))
    monkeypatch.setattr(o,'public_pre_action_snapshot',lambda state:dict(synthetic_only=True,frame=state.frame))
    def frozen(state,path,variant):
        ref=put(path,dict(synthetic_NN_only=True,variant=variant,state_sha256=fake_fingerprint(state)))
        actor=SimpleNamespace(act=lambda state,**kw:(dict(robot_1='WAIT',robot_2='WAIT'),None))
        return ref,actor
    monkeypatch.setattr(o,'frozen_export',frozen)
    def batch(plan,reader,trace,state):
        data=o.bound(trace);allowed={c['episode_id'] for c in o.bound(plan.value['train_case_inventory'])['cases']}
        assert all(row['case']['episode_id'] in allowed for row in data['events'])
        return dict(synthetic_only=True),dict(synthetic_only=True,program_evidence_frames_in_gradient_buffers=0,
                                            replayed_NN_actions=2*len(data['events']))
    monkeypatch.setattr(o.seeds,'assemble_training_batch',batch)
    def save(state,path):return put(path,dict(synthetic_only=True,state_sha256=fake_fingerprint(state)))
    monkeypatch.setattr(o,'save_checkpoint',save)


def test_real_claim_factory_collect_update_and_full_checkpoint_wiring(prepared,kernels,monkeypatch):
    install_physics_fakes(monkeypatch)
    with pytest.raises(ProbeFinished):
        with enter(prepared) as active:
            for variant in o.VARIANTS:
                state=active.initialize(variant);trace=active.collect_rollout(variant)
                assert state.joint_steps==1024 and active.reserved[variant]==1024
                assert o.bound(active.pending[variant]['audit'])['replayed_NN_actions']==2048
                result=active.update(variant)
                assert result['actor_enabled'] is False and state.actor_updates==0 and state.critic_updates==1
                assert result['accepted_positive_program_gradient_updates']==dict(robot_1=0,robot_2=0)
                checkpoint=o.bound(active.checkpoint(variant))
                assert checkpoint['safe_batch_boundary'] and checkpoint['pending_optimizer_batches']==0
                assert checkpoint['core']['processed_joint_steps']==checkpoint['reserved_joint_steps']==1024
                assert checkpoint['reader']['complete_cost_history']=='fake'
                assert checkpoint['reset_cursors']==[2]*8 and len(checkpoint['environments'])==8
                assert checkpoint['full_training_state_sha256']==fake_fingerprint(state)
                assert checkpoint['safe_snapshot_not_resume_authority'] and not checkpoint['automatic_resume']
                assert o.pointer(trace['path'])==trace
            assert len(active.actual_resets)==32
            raise ProbeFinished()


def test_raw_feedback_and_unowned_weight_change_latch_owner_closed(prepared,kernels,monkeypatch):
    install_physics_fakes(monkeypatch)
    with pytest.raises(RuntimeError,match='raw program feedback'):
        with enter(prepared) as active:
            active.initialize('raw');active.collect_rollout('raw');active.update('raw',evidence_provider=object())
    assert active.failed and (active.folder/'failed.json').is_file()


@pytest.mark.parametrize('operation',['source_before75','reset_without_rollout','second_collection','changed_weights'])
def test_out_of_order_operations_reject_before_additional_synthetic_reset(prepared,kernels,monkeypatch,operation):
    install_physics_fakes(monkeypatch)
    with pytest.raises(RuntimeError):
        with enter(prepared) as active:
            state=active.initialize('raw')
            if operation=='source_before75':active.collect_program_source('raw')
            elif operation=='reset_without_rollout':active.new_environment('raw',active.train_streams[0][0],scope='train')
            elif operation=='second_collection':
                active.collect_rollout('raw');resets=len(active.actual_resets)
                try:active.collect_rollout('raw')
                finally:assert len(active.actual_resets)==resets
            else:state.revision+=1;active.collect_rollout('raw')
    assert active.failed


@pytest.mark.parametrize('budget',[200000,300000,400000])
def test_driver_all_registered_boundaries_restarts_both_actors_and_never_reuses_small_budget_state(monkeypatch,budget):
    # Jump only in a synthetic driver double. Production collection is the
    # frame-by-frame NN/physics path tested separately above.
    config=o.PPOConfig(joint_step_budget=budget);events=[]
    class Driver:
        def __init__(self):self.config=config;self.states={};self.refits={v:[] for v in o.VARIANTS}
        def initialize(self,v):self.states[v]=FakeState();events.append(('init',v,0))
        def collect_rollout(self,v):
            state=self.states[v];step=next(s for s in o.checkpoint_steps(config) if s>state.joint_steps)
            state.joint_steps=step;events.append(('collect',v,step))
        def update(self,v):events.append(('update',v,self.states[v].joint_steps))
        def checkpoint(self,v):
            step=self.states[v].joint_steps;events.append(('checkpoint',v,step));return dict(step=step)
    def refit(active,v,programs):
        step=active.states[v].joint_steps;active.refits[v].append(step);events.append(('refit',v,step));return programs
    monkeypatch.setattr(o,'scheduled_refit',refit)
    active=Driver();stored=o.drive(active,dict(robot_1='fake1',robot_2='fake2'))
    assert events[:2]==[('init','raw',0),('init','regularized',0)]
    for v in o.VARIANTS:
        assert active.states[v].joint_steps==budget
        assert active.refits[v]==list(config.extraction_steps)
        assert [r['step'] for r in stored[v]]==list(o.checkpoint_steps(config))
    assert config.program_coefficient(80*budget//100)==0
    assert config.program_coefficient(85*budget//100)==pytest.approx(.01)


@pytest.mark.parametrize('bad_gradients',[False,True])
def test_actual_context_completion_requires_full_pair_all_refits_and_actual_role_gradients(prepared,kernels,bad_gradients):
    def fill(active):
        for v in o.VARIANTS:
            state=active.initialize(v);state.joint_steps=active.config.joint_step_budget
            active.cores[v].processed_joint_steps=state.joint_steps;active.reserved[v]=state.joint_steps
            active.refits[v]=list(active.config.extraction_steps);active.evidence_cursor[v]=len(active.evidence_cases)
        active.cores['regularized'].accepted_program_updates=dict(robot_1=1,robot_2=int(not bad_gradients))
    if bad_gradients:
        with pytest.raises(RuntimeError,match='positive program gradient'):
            with enter(prepared) as active:fill(active)
        assert not (active.folder/'completed.json').exists()
    else:
        with enter(prepared) as active:fill(active)
        result=o.bound(o.pointer(active.folder/'completed.json'))
        assert result['per_variant_joint_steps']==dict(raw=200000,regularized=200000)
        assert result['accepted_program_updates']['raw']==dict(robot_1=0,robot_2=0)
        assert result['accepted_program_updates']['regularized']==dict(robot_1=1,robot_2=1)
        assert result['training_finished'] and not result['publishable'] and not result['budget_lock_issued']
        assert not result['model_acceptance_passed'] and not result['formal_training_authorized']


@pytest.mark.parametrize('prepared',[200000,300000,400000],indirect=True)
def test_full_fake_pipeline_real_context_initialization_updates_refits_checkpoints_and_receipts(prepared,kernels,monkeypatch):
    """Exercise the real state factory and controller through all boundaries.

    Physical chunks and program fitting are explicit synthetic shortcuts, not
    scientific trajectories. Actual owner update/refit/checkpoint/context and
    result binding run unchanged, so missing owner containers cannot hide.
    """
    install_physics_fakes(monkeypatch)
    budget=prepared.plan.config.joint_step_budget
    counts=dict(fit=900,calibration=150,eligibility=300,validation=300)
    program_cases=[dict(episode=f'synthetic-program-{split}-{i}',split=split)
                   for split,count in counts.items() for i in range(count)]
    class Provider:
        def __init__(self,owner_ref):self.owner_ref=owner_ref;self.steps=[]
        def install_extraction(self,binding,pool,state):self.steps.append(state.joint_steps)
        def snapshot(self,state):return dict(synthetic_only=True,installed_steps=self.steps,step=state.joint_steps)
    class Callback:
        def __init__(self,active,provider):self.native=provider;self.active=active
        def verify(self):assert self.active.providers['regularized'] is self.native
        def snapshot(self,state):return dict(synthetic_only=True,explicit_long_callback=True,step=state.joint_steps)
    monkeypatch.setattr(o,'CurrentNNEvidenceProvider',Provider)
    monkeypatch.setattr(o,'LongEvidenceCallback',Callback)
    monkeypatch.setattr(o.observed,'source_files',lambda:dict(synthetic_only='fake-source'))
    collected=[];accepted=[];critic_only=[]
    def fake_collection(active,v):
        active.live(full=True);state=active.states[v];start=state.joint_steps
        boundaries=sorted({4000,*active.config.extraction_steps,80*budget//100,85*budget//100,budget})
        end=next(n for n in boundaries if n>start)
        trace=put(active.folder/v/'synthetic-chunks'/str(start)/'trace.json',dict(synthetic_only=True,start=start,end=end))
        audit=put(Path(trace['path']).with_name(f'{start}-audit.json'),dict(synthetic_only=True,
            program_evidence_frames_in_gradient_buffers=0))
        state.joint_steps=end;active.reserved[v]=end
        active.pending[v]=dict(batch=dict(synthetic_only=True),trace=trace,audit=audit,state_sha256=fake_fingerprint(state))
        active.state_fingerprints[v]=fake_fingerprint(state);collected.append((v,start,end))
    monkeypatch.setattr(o.LongTrainingOwner,'collect_rollout',fake_collection)
    original_update=o.LongTrainingOwner.update
    def observed_update(active,v):
        receipt=original_update(active,v);accepted.append((v,active.states[v].joint_steps,receipt))
        if active.states[v].joint_steps==4000:critic_only.append((v,active.states[v].actor_updates))
        return receipt
    monkeypatch.setattr(o.LongTrainingOwner,'update',observed_update)
    def fake_refit(active,v,programs):
        state=active.states[v];step=state.joint_steps
        if not active.refits[v]:
            active.evidence_cases=deepcopy(program_cases);active.evidence_cursor[v]=len(program_cases)
            source=put(active.folder/v/'fake-program-source.json',dict(synthetic_only=True,cases=len(program_cases)))
            active.pools[v]=SimpleNamespace(source_ref=source,audit_sha256='fake-program-audit')
        extraction=put(active.folder/v/f'fake-refit-{step}.json',dict(synthetic_only=True,
            identity=dict(owner=active.owners[v],variant=v,step=step),full_training_state_sha256=fake_fingerprint(state)))
        binding=put(active.folder/v/f'bindings-{step}'/'binding.json',dict(schema='warehouse-v642-observed-refit-source-binding.v1',
            source_files=o.observed.source_files(),owner=active.owners[v],observed_source=active.pools[v].source_ref,extraction=extraction))
        put(Path(binding['path']).with_name('observed-source-binding.completed.json'),dict(binding_sha256=binding['sha256']))
        provider=(active.providers.get(v) or Provider(active.owners[v])) if v=='regularized' else None
        active.record_refit(v,binding,evidence_provider=provider)
        return programs
    monkeypatch.setattr(o,'scheduled_refit',fake_refit)
    result_ref=o.run(**prepared.values);result=o.bound(result_ref);completion=o.bound(result['owner_completed'])
    assert result['numerical_training_completed'] and not result['publishable'] and not result['model_acceptance_passed']
    assert completion['per_variant_joint_steps']==dict(raw=budget,regularized=budget)
    assert completion['accepted_program_updates']['raw']==dict(robot_1=0,robot_2=0)
    assert all(n>0 for n in completion['accepted_program_updates']['regularized'].values())
    assert critic_only==[('raw',0),('regularized',0)]
    assert [(a,b) for v,a,b in collected if v=='raw']==[(a,b) for v,a,b in collected if v=='regularized']
    for v in o.VARIANTS:
        checkpoints=[o.bound(ref) for ref in result['checkpoints'][v]]
        assert [r['reserved_joint_steps'] for r in checkpoints]==list(o.checkpoint_steps(prepared.plan.config))
        assert all(r['safe_batch_boundary'] and not r['automatic_resume'] for r in checkpoints)
        assert checkpoints[-1]['refits']==list(prepared.plan.config.extraction_steps)
        assert checkpoints[-1]['program_source_cursor']==1650
        assert checkpoints[0]['paired_initial_identity']==checkpoints[-1]['paired_initial_identity']
        if v=='raw':assert all(r['provider'] is r['long_storage_callback'] is None for r in checkpoints)
        else:
            assert checkpoints[-1]['provider']['installed_steps']==list(prepared.plan.config.extraction_steps)
            assert checkpoints[-1]['long_storage_callback']['explicit_long_callback']
    journal=[json.loads(line) for line in (prepared.folder/'training'/'owner-journal.jsonl').read_text().splitlines()]
    assert [r['sequence'] for r in journal]==list(range(len(journal)))
    assert len([r for r in journal if r['kind']=='scheduled_refit_bound'])==6
    assert o.ACTIVE.get() is None and not (prepared.folder/'training'/'failed.json').exists()


def test_long_bridge_rejects_program_evidence_before_reader_or_gradient_buffer(prepared,monkeypatch):
    # The real bridge checks train membership before invoking even a fake
    # replay reader. No evidence scene can acquire a PPO or cost label.
    plan=prepared.plan
    class Reader:
        def __init__(self):
            self.owner_ref=put(prepared.folder/'fake-bridge-owner.json',dict(training_partition_proposal=plan.ref,
                train_case_inventory=plan.value['train_case_inventory'],program_evidence_inventory=plan.value['program_evidence_inventory']))
            self.inventory_ref=plan.value['train_case_inventory'];self.config=plan.config
        def assemble(self,*args):pytest.fail('foreign program scene reached a gradient replay reader')
    monkeypatch.setattr(o.seeds,'EnvironmentRolloutReplay',Reader)
    bad=o.bound(plan.value['program_evidence_inventory'])['cases'][0]
    trace=put(prepared.folder/'fake-foreign-trace.json',dict(events=[dict(case=dict(episode_id=bad['episode']))]))
    state=SimpleNamespace(initialization_seed=plan.value['initialization_seed'],reference_sha256=o.seeds.protocol.REFERENCE_SHA256)
    with pytest.raises(RuntimeError,match='program evidence or foreign cases'):
        o.seeds.assemble_training_batch(plan,Reader(),trace,state)


def test_sealing_only_changes_own_new_tree_permissions_and_rejects_symlink(tmp_path,monkeypatch):
    roots=[tmp_path/name for name in ('provider','targets','extraction','new-long-storage')]
    monkeypatch.setattr(o,'PROVIDER_ROOT',roots[0]);monkeypatch.setattr(o.observed,'TARGET_ROOT',roots[1])
    monkeypatch.setattr(o,'EXTRACTION_ROOT',roots[2])
    monkeypatch.setattr(o.long_targets.rows,'STORAGE_ROOT',roots[3])
    active=SimpleNamespace(owners={v:dict(path='synthetic-'+v,sha256=v+'-sha') for v in o.VARIANTS})
    files=[]
    for root in roots:
        for v in o.VARIANTS:
            path=root/o.digest(active.owners[v])/'nested'/'receipt.json';path.parent.mkdir(parents=True)
            path.write_text('{"synthetic_only":true}');files.append((path,o.file_sha(path)))
    historical=roots[0]/'historical-parent.json';historical.write_text('{}')
    assert len(o.seal_owned_artifacts(active))==8
    assert all(o.file_sha(path)==sha and not path.stat().st_mode&0o222 for path,sha in files)
    assert historical.stat().st_mode&0o222
    link=roots[0]/o.digest(active.owners['raw'])/'forbidden-link';link.symlink_to(historical)
    with pytest.raises(RuntimeError,match='symlink'):o.seal_owned_artifacts(active)


def test_cli_gate_before_plan_hash_or_dispatch(tmp_path,monkeypatch):
    c=cli_module();precheck=put(tmp_path/'fake-precheck.json',dict(synthetic_only=True))
    real_pointer=c.pointer;reads=[]
    def ref(path):reads.append(str(path));return real_pointer(path)
    def denied(receipt):raise RuntimeError('fake closed actual precheck')
    monkeypatch.setattr(c,'pointer',ref);monkeypatch.setattr(c.owner.seeds.protocol,'register_preparation',denied)
    with pytest.raises(RuntimeError,match='closed actual precheck'):
        c.main(['--run','--reference-directory',str(tmp_path),'--calibration-precheck',precheck['path'],
                '--plan',str(tmp_path/'must-not-be-read.json')])
    assert reads==[precheck['path']]


@pytest.mark.parametrize('actual_exit,has_completion',[(0,True),(0,False),(1,False),(-15,False)])
def test_supervisor_records_actual_wait_and_rejects_exit0_without_binding(prepared,monkeypatch,actual_exit,has_completion):
    c=cli_module();waited=[]
    class Worker:
        pid=987654
        def __init__(self,command,**kw):self.command=command
        def wait(self):
            waited.append(actual_exit)
            if has_completion:
                training=prepared.folder/'training';training.mkdir()
                descriptor=put(training/'descriptor.json',dict(seed_plan=prepared.plan.ref))
                completed=put(training/'completed.json',dict(version=o.VERSION,training_finished=True,descriptor=descriptor,publishable=False))
                put(training/'controller-completed.json',dict(version=o.VERSION,seed_plan=prepared.plan.ref,
                    owner_completed=completed,numerical_training_completed=True,publishable=False))
            return actual_exit
    monkeypatch.setattr(c.subprocess,'Popen',Worker)
    if actual_exit==0 and has_completion:result=c.supervise(**prepared.values)
    else:
        with pytest.raises(RuntimeError,match='failed or its exit-zero'):c.supervise(**prepared.values)
    exit_record=o.bound(o.pointer(prepared.folder/'long-supervisor-exit.json'))
    assert waited==[actual_exit] and exit_record['exit_code']==actual_exit
    assert exit_record['observed_by']=='source_bound_subprocess_wait' and exit_record['worker_pid']==987654
    assert '--run' in exit_record['command'] and '--supervise' not in exit_record['command']
    assert not Path(exit_record['log']['path']).stat().st_mode&0o222
    if actual_exit==0 and not has_completion:assert 'result_binding_error' in exit_record and 'result' not in exit_record
    elif actual_exit==0:assert result==o.pointer(prepared.folder/'long-supervisor-exit.json')
    else:assert 'result' not in exit_record and 'completed' not in exit_record


@pytest.fixture
def formal_prepared(prepared,monkeypatch):
    """Actual FormalAdmission/lock readers, explicitly fake lower dev auditor."""
    from tests.test_warehouse_v642_formal_training_admission import install_fake_development_auditor
    lock,_,calls=install_fake_development_auditor(prepared.folder.parent,monkeypatch,
        budget=prepared.plan.config.joint_step_budget)
    authority=o.seeds.formal.read_admission(lock)
    plan=deepcopy(prepared.plan);folder=prepared.folder.parent/'synthetic-formal-only';folder.mkdir()
    plan.ref=put(folder/'plan.json',dict(synthetic_only=True,not_real_seed_plan=True))
    plan.formal_authorization=authority
    plan.value.update(phase='formal',stream_id='formal-0',development_budget_lock=lock,
        formal_admission=deepcopy(authority.value),restart_proof=None,execution_output=str(folder/'training'))
    plan.value['global_claim_path']=str(o.claim_path(plan));plan.proposal=plan
    def checked(ref,*,budget_lock_ref,calibration_precheck_ref,reference_directory):
        assert ref==plan.ref and budget_lock_ref==lock
        assert o.seeds.formal.read_admission(budget_lock_ref) is authority
        return plan
    monkeypatch.setattr(o.seeds,'check_formal',checked)
    values=dict(prepared.values,plan_ref=plan.ref,phase='formal',formal_budget_lock_ref=lock)
    o.prepare_sources(**values);o.audit_source_bound_seeds(**values)
    return SimpleNamespace(plan=plan,values=values,folder=folder,clearance=prepared.clearance,lock=lock,calls=calls)


def test_explicit_formal_missing_lock_precedes_plan_seed_calibration_model_RNG_and_output(tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('missing formal lock reached another input/model/output')
    monkeypatch.setattr(o.seeds.protocol,'register_preparation',forbidden)
    monkeypatch.setattr(o,'bound',forbidden);monkeypatch.setattr(o,'source_files',forbidden)
    monkeypatch.setattr(o,'initialize_from_reference',forbidden);monkeypatch.setattr(o,'WarehouseMultiAgentEnv',forbidden)
    values=dict(reference_directory=None,calibration_precheck_ref=None,plan_ref=None,phase='formal',formal_budget_lock_ref=None)
    for call in (o.prepare_sources,o.audit_source_bound_seeds,o.run):
        with pytest.raises(RuntimeError,match='before other inputs'):call(**values)
    assert not list(tmp_path.iterdir())


def test_formal_owner_claim_binding_fresh_pair_and_single_use_scope(formal_prepared,kernels):
    p=formal_prepared
    with pytest.raises(ProbeFinished):
        with enter(p) as active:
            for variant in o.VARIANTS:active.initialize(variant)
            assert active.plan.formal_authorization.budget==active.config.joint_step_budget
            descriptor=o.bound(active.descriptor_ref);claim=o.bound(active.claim_ref)
            assert descriptor['phase']==claim['phase']=='formal'
            assert descriptor['development_budget_lock']==claim['development_budget_lock']==p.lock
            assert descriptor['formal_admission']==claim['formal_admission']==p.plan.value['formal_admission']
            assert descriptor['stage']=='paired_long_formal_training_not_quality_or_sealed_admission'
            assert not descriptor['model_acceptance_passed'] and not descriptor['sealed_authority'] and not descriptor['pilot_opened']
            assert active.initial_identity['initial_joint_steps']==0 and active.states['raw'].actor_updates==0
            raise ProbeFinished()
    with pytest.raises(RuntimeError,match='already consumed'):
        with enter(p):pytest.fail('formal interruption was rearmed')
    assert len(p.calls)==2 and not (p.folder/'training/completed.json').exists()


def test_formal_full_fake_five_checkpoint_pipeline_keeps_phase_lock_and_no_model_success(formal_prepared,kernels,monkeypatch):
    # The existing full factory/driver fake path executes; no real environment,
    # seed preparation or optimizer is used. Lower development auditor is fake.
    test_full_fake_pipeline_real_context_initialization_updates_refits_checkpoints_and_receipts(formal_prepared,kernels,monkeypatch)
    result=o.bound(o.pointer(formal_prepared.folder/'training/controller-completed.json'))
    done=o.bound(result['owner_completed'])
    for row in (result,done):
        assert row['phase']=='formal' and row['development_budget_lock']==formal_prepared.lock
        assert row['formal_admission']==formal_prepared.plan.value['formal_admission']
        assert row['formal_training_authorized'] and not row['sealed_authority'] and not row['model_acceptance_passed'] and not row['pilot_opened']
    assert all(len(result['checkpoints'][v])==5 for v in o.VARIANTS)


def test_formal_CLI_missing_lock_is_rejected_before_hashing_any_other_input(tmp_path,monkeypatch):
    c=cli_module()
    def forbidden(*a,**kw):pytest.fail('formal CLI touched precheck/plan before lock')
    monkeypatch.setattr(c,'pointer',forbidden)
    with pytest.raises(RuntimeError,match='before other inputs'):
        c.main(['--run','--phase','formal','--reference-directory',str(tmp_path),
                '--calibration-precheck',str(tmp_path/'unread-precheck'),'--plan',str(tmp_path/'unread-plan')])
    assert not list(tmp_path.iterdir())


def test_formal_supervisor_exact_argv_actual_wait_and_same_lock_receipts(formal_prepared,monkeypatch):
    c=cli_module();p=formal_prepared;commands=[]
    class Worker:
        pid=987655
        def __init__(self,command,**kw):commands.append(command)
        def wait(self):
            folder=p.folder/'training';folder.mkdir()
            fields=dict(phase='formal',development_budget_lock=p.lock,formal_admission=p.plan.value['formal_admission'])
            descriptor=put(folder/'descriptor.json',dict(seed_plan=p.plan.ref,**fields))
            completed=put(folder/'completed.json',dict(version=o.VERSION,training_finished=True,descriptor=descriptor,publishable=False,**fields))
            put(folder/'controller-completed.json',dict(version=o.VERSION,seed_plan=p.plan.ref,owner_completed=completed,
                numerical_training_completed=True,publishable=False,**fields))
            return 0
    monkeypatch.setattr(c.subprocess,'Popen',Worker)
    ref=c.supervise(**p.values);record=o.bound(ref);started=o.bound(record['started'])
    assert commands[0][-4:]==['--phase','formal','--formal-budget-lock',p.lock['path']]
    assert record['command']==started['command']==commands[0] and record['exit_code']==0
    assert record['phase']==started['phase']=='formal' and record['development_budget_lock']==started['development_budget_lock']==p.lock
    assert record['formal_admission']==started['formal_admission']==p.plan.value['formal_admission']
