"""Independent finite long-budget development owner for the v6.42 method.

Only a source-bound long seed plan following a replayed real 20k calibration
can enter the exclusive claim. Every budget restarts from the full v6.20 NN;
this module does not change or re-arm the old calibration owner. Programs
supply gradients/evidence only. Quality, budget locking and release remain
separate later gates. Ordinary imports and constructors authorize no work.

The collection/update bodies preserve the existing audited physical replay
and PPO contracts, with their own owner identity, source closure and budget.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
import gzip,json,os,re,time
import numpy as np
from . import v642_long_training_seed_plan as seeds
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training import warehouse_v642_observed_pool as observed
from backend.training.warehouse_v629_environment_replay import (EnvironmentRolloutReplay,capture_environment,
    training_config,WarehouseMultiAgentEnv,public_pre_action_snapshot,PHYSICS_VERSION)
from backend.training.warehouse_v629_rollout_replay import training_fingerprint,TRACE_VERSION
from backend.training.warehouse_v642_evidence_provider import CurrentNNEvidenceProvider,PROVIDER_ROOT
from backend.training.warehouse_v642_program_refit import EXTRACTION_ROOT
from backend.training.warehouse_v629_ppo_update import PPOUpdateCore
from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer,critic_initialization_sha256
from backend.training.warehouse_v622_actor_training import initialize_from_reference,export_current_actor,save_checkpoint
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v6_actor_training import role_hashes
from backend.training.warehouse_v6_runner import _set_dual_nn_mode
from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
from scripts.run_warehouse_v625 import write_new

ROOT=seeds.ROOT
VERSION='warehouse-v642-finite-long-development-owner.v1'
CANDIDATE='warehouse-v642-depth9-depth12-01'
VARIANTS=('raw','regularized')
ACTIVE=ContextVar('warehouse_v642_finite_long_training_owner',default=None)
_FACTORY_TOKEN=object()
FILES=('experiments/warehouse/v642_long_training_owner.py',
       'scripts/run_warehouse_v642_long_training.py',
       'tests/test_warehouse_v642_long_training_owner.py',
       'docs/warehouse_v642_long_training_owner.md')


def source_files():
    files=seeds.source_files()
    files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def load_seed_plan(reference_directory,calibration_precheck_ref,plan_ref):
    # The seed-plan reader replays the actual independent calibration first,
    # before reading caller-supplied plan data or constructing a model/RNG.
    plan=seeds.check(plan_ref,calibration_precheck_ref=calibration_precheck_ref,
                     reference_directory=reference_directory)
    plan.verify()
    require(plan.value['phase']=='development',
        'formal long training requires the not-yet-issued independently audited development budget lock')
    require(plan.config.joint_step_budget in (200000,300000,400000)
        and asdict(plan.config)==plan.value['config'], 'unknown long development budget or config')
    return plan


def checkpoint_steps(config):
    return (4000,*config.extraction_steps,config.joint_step_budget)


def execution_lease(plan,clearance,*,started=None):
    """A fresh 24h seed check starts one finite, nonrenewable long run."""
    started=datetime.now(timezone.utc) if started is None else started
    days=plan.value['execution_lease_days']
    require(days==seeds.EXECUTION_LEASE_DAYS_BY_BUDGET[plan.config.joint_step_budget]
        and plan.value['seed_clearance_start_window_hours']==24,'long execution lease protocol differs')
    issued=datetime.fromisoformat(clearance['value']['issued_utc'])
    deadline=datetime.fromisoformat(clearance['value']['expires_utc'])
    require(issued.tzinfo and deadline.tzinfo and started.tzinfo
        and issued<=started<deadline and deadline-issued==timedelta(hours=24),
        '24h long seed clearance permits starting only before its exact deadline')
    return dict(started_utc=started.isoformat(),expires_utc=(started+timedelta(days=days)).isoformat(),
        execution_lease_days=days,seed_clearance_start_deadline_utc=deadline.isoformat(),automatic_renewal=False)


def claim_path(plan):
    name=f"{plan.value['phase']}-{plan.value['stream_id']}-{plan.config.joint_step_budget}.json"
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',name) is not None,'unsafe long-run claim identity')
    return seeds.OUTPUT_ROOT/'claims'/name


def contract_value(plan):
    files=source_files()
    return dict(version=VERSION,seed_plan=plan.ref,source_files=files,source_sha256=digest(files),
        execution_output=str(Path(plan.ref['path']).parent/'training'),global_claim_path=str(claim_path(plan)),
        config=asdict(plan.config),checkpoint_joint_steps=list(checkpoint_steps(plan.config)),
        execution_lease_days=plan.value['execution_lease_days'],
        seed_clearance_start_window_hours=plan.value['seed_clearance_start_window_hours'],
        variants=list(VARIANTS),training_authority=False,sealed_authority=False,pilot_opened=False)


def checked_sources(plan):
    plan.verify();folder=Path(plan.ref['path']).parent
    ref=pointer(folder/'long-owner-source-contract.json');value=bound(ref)
    require(value==contract_value(plan),'long owner source/config contract differs')
    for name,sha in value['source_files'].items():
        file=folder/'long-owner-source'/value['source_sha256']/name
        require(file_sha(file)==sha and not file.stat().st_mode&0o222,'long owner source snapshot changed')
    return ref


def prepare_sources(*,reference_directory,calibration_precheck_ref,plan_ref):
    plan=load_seed_plan(reference_directory,calibration_precheck_ref,plan_ref)
    folder=Path(plan.ref['path']).parent
    require(not (folder/'seed-ledger').exists(),'register full long owner before exact seed clearance')
    value=contract_value(plan)
    for name,sha in value['source_files'].items():
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==sha,'long owner source changed preparing snapshot')
        target=folder/'long-owner-source'/value['source_sha256']/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        target.chmod(0o444)
    target=folder/'long-owner-source-contract.json';write_new(target,value);target.chmod(0o444)
    return checked_sources(plan)


def audit_source_bound_seeds(*,reference_directory,calibration_precheck_ref,plan_ref):
    plan=load_seed_plan(reference_directory,calibration_precheck_ref,plan_ref);contract=checked_sources(plan)
    clearance=seeds.audit(plan)
    require(checked_sources(plan)==contract,'long owner source changed during seed audit')
    target=Path(plan.ref['path']).parent/'long-owner-source-clearance.json'
    write_new(target,dict(version=VERSION,owner_source_contract=contract,
        seed_clearance=clearance['clearance'],training_authority=False))
    target.chmod(0o444);return pointer(target)


def frozen_export(state,path,variant):
    exported=export_current_actor(state,path,pair_seed=state.initialization_seed,variant=variant,
        candidate_id=CANDIDATE)
    Path(exported['path']).chmod(0o444);ref=pointer(exported['path'])
    actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=role_hashes(state.model))
    return ref,actor


def seal_owned_artifacts(owner):
    """Seal only this new owner's provider/target/refit trees, never parents.

    The unchanged numerical helpers leave a few receipts writable. Sealing
    changes permissions only after completed work; no scientific bytes or
    historical program assets are edited.
    """
    sealed=[]
    for variant in VARIANTS:
        key=digest(owner.owners[variant])
        for root in (PROVIDER_ROOT,observed.TARGET_ROOT,EXTRACTION_ROOT):
            folder=root/key
            require(not folder.is_symlink(),'owned artifact root cannot be a symlink')
            if not folder.exists():continue
            require(folder.is_dir(),'owned artifact root is not a directory')
            for path in folder.rglob('*'):
                require(not path.is_symlink() and (path.is_file() or path.is_dir()),
                    'owned artifact tree contains a symlink or non-file')
                if path.is_file():path.chmod(0o444)
            sealed.append(str(folder))
    return sealed


def owned_operation(method):
    def call(self,*args,**kwargs):
        try:return method(self,*args,**kwargs)
        except BaseException:
            self.failed=True
            raise
    return call


class LongTrainingOwner:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('use the single-use claimed_owner context after real calibration replay and exact seed clearance')

    def live(self,*,full=False):
        require(ACTIVE.get() is self and not self.failed and not self.closed,'long-training owner is not active')
        require(pointer(self.claim_ref['path'])==self.claim_ref and pointer(self.descriptor_ref['path'])==self.descriptor_ref,
            'long-training claim or descriptor changed')
        require(datetime.now(timezone.utc)<self.execution_expires_utc,
            'finite long-training execution lease expired; automatic renewal is forbidden')
        if full:
            self.plan.verify();require(self.sources==source_files(),'long-training source changed')
            require(checked_sources(self.plan)==self.source_contract_ref,'registered owner source changed')
            require(pointer(self.clearance['clearance']['path'])==self.clearance['clearance'],
                'long-training history clearance changed')
            for ref in self.clearance_assets:require(pointer(ref['path'])==ref,'long-training history input asset changed')

    def event(self,kind,**values):
        self.live()
        value=dict(version=VERSION,sequence=self.sequence,kind=kind,claim=self.claim_ref,
            phase=self.plan.value['phase'],stream_id=self.plan.value['stream_id'],budget=self.config.joint_step_budget,
            utc=datetime.now(timezone.utc).isoformat(),**values)
        self.journal.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        self.journal.flush();os.fsync(self.journal.fileno());self.sequence+=1
        return value

    @owned_operation
    def initialize(self,variant):
        self.live(full=True);require(variant in VARIANTS and variant not in self.states,'paired variant already initialized or unknown')
        p=self.plan.proposal;actor=p.readiness.value['actor'];config=self.config
        self.event('initialize_predebit',variant=variant,initialization_seed=p.value['initialization_seed'],reference_actor=actor)
        # Mark before RNG/model construction; failure cannot silently retry.
        self.states[variant]=None
        roles={role:row['language']['anchor_actor_role_sha256'] for role,row in p.readiness.value['program_pair'].items()}
        state=initialize_from_reference(actor['path'],actor['sha256'],roles,
            initialization_seed=p.value['initialization_seed'],actor_lr=config.actor_learning_rate,critic_lr=config.critic_learning_rate)
        require(state.joint_steps==state.actor_updates==state.critic_updates==0 and not state.resumed_training_state
            and state.reference_sha256==actor['sha256'] and state.initialization_seed==p.value['initialization_seed']
            and not state.actor_optimizer.state_dict()['state'] and not state.critic_optimizer.state_dict()['state'],
            'every budget must import the complete reference at step zero with fresh critics and empty optimizers')
        identity=dict(roles=role_hashes(state.model),critic_sha256=critic_initialization_sha256(state.model),
            full_initial_training_state_sha256=training_fingerprint(state),
            actor_optimizer_sha256=digest(state.actor_optimizer.state_dict()),
            critic_optimizer_sha256=digest(state.critic_optimizer.state_dict()),
            initial_joint_steps=0,initial_actor_updates=0,initial_critic_updates=0)
        require(identity['roles']==roles,'initial Actor roles differ from the confirmed complete reference')
        require(not self.initial_identity or identity==self.initial_identity,'raw/reg initial Actor or critic differs')
        restart=p.value.get('restart_proof')
        require(restart is None or identity==restart['expected_initial_identity'],
            'larger budget did not restart from the independently audited identical full initial state')
        self.initial_identity=identity
        self.event('optimizer_RNG_predebit',variant=variant,seed=p.value['optimizer_seed'])
        rng=np.random.default_rng(p.value['optimizer_seed'])
        ref=p.readiness.value['reference']['buffer']
        buffer=ReferenceTrainingBuffer.load(ref['path'],ref['sha256'],state.reference_sha256)
        seeds.check_reference_buffer(p,buffer)
        self.states[variant]=state;self.cores[variant]=PPOUpdateCore(state,buffer,variant=variant,rng=rng,config=config)
        self.readers[variant]=EnvironmentRolloutReplay(self.owners[variant],p.value['train_case_inventory'],config=config)
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('initialized',variant=variant,**identity,Actor_updates=0,critic_updates=0)
        return state

    @owned_operation
    def new_environment(self,variant,case,*,scope):
        self.live(full=True);require(variant in self.states and self.states[variant] is not None,'initialize variant before collection')
        if scope=='train':
            require(variant in self.collecting and self.collecting[variant][0]<=self.states[variant].joint_steps<self.collecting[variant][1],
                'training reset needs an active predebited rollout')
            slot=case['slot'];cursor=self.cursors[variant][slot];stream=self.train_streams[slot]
            require(cursor<len(stream) and case==stream[cursor],'training reset outside its finite per-slot order')
            self.cursors[variant][slot]+=1;episode=case['episode_id']
        else:
            cursor=self.evidence_cursor[variant]
            require(scope=='program_evidence' and variant in self.collecting_source and self.states[variant].joint_steps==self.config.extraction_steps[0]
                and cursor<len(self.evidence_cases) and case==self.evidence_cases[cursor],
                'program-source reset outside its once-only 75% inventory')
            self.evidence_cursor[variant]+=1;episode=case['episode']
        self.event('environment_constructor_and_reset_predebit',variant=variant,scope=scope,case=case)
        prior=len(self.actual_resets)
        # These hook phase names are inherited labels; the claim and owner
        # record explicitly identify this as long-training, never formal train.
        if scope=='train':
            phase='formal_'+variant+'_rollout'
            metadata=dict(start_joint_step=self.states[variant].joint_steps,episode_index=cursor)
        else:phase='program';metadata={}
        with stage1_reset_phase(phase,**metadata):
            env=WarehouseMultiAgentEnv(training_config(case['scenario_seed']));env.reset(seed=case['scenario_seed'])
        _set_dual_nn_mode(env)
        require(len(self.actual_resets)==prior+1 and self.actual_resets[-1]['seed']==case['scenario_seed']
            and self.actual_resets[-1]['episode_id']==env.state.episode_id==1,'actual reset hook differs from its predebit')
        row=self.actual_resets[-1]
        self.event('actual_reset_bound',variant=variant,scope=scope,episode=episode,reset_hook_index=prior,reset=row,
            initial_environment_sha256=digest(capture_environment(env)),native_hook_phase_is_legacy_label_not_formal_admission=True)
        return env

    @owned_operation
    def collect_rollout(self,variant):
        self.live(full=True);require(variant in self.states and self.states[variant] is not None,'variant is not initialized')
        state=self.states[variant];require(variant not in self.pending,'previous rollout must be optimized before another collection')
        require(training_fingerprint(state)==self.state_fingerprints[variant],'training state changed outside owned optimizer updates')
        start=state.joint_steps;length=self.config.next_collection_length(start);n=8*length
        require(n>0 and self.reserved[variant]==start and self.cores[variant].processed_joint_steps==start,
            'rollout reservation, optimizer or finite budget differs')
        due=[step for step in self.config.extraction_steps if step<=start]
        require(self.refits[variant]==due,'scheduled refit is required before next training collection')
        self.event('reserve_rollout_predebit',variant=variant,start_joint_step=start,joint_frames=n)
        self.reserved[variant]+=n
        folder=self.folder/variant/'rollouts'/f'{start:06d}';folder.mkdir(parents=True,exist_ok=False)
        before=training_fingerprint(state);ref,actor=frozen_export(state,folder/'current-NN.npz',variant)
        identity=dict(actor_role_sha256=role_hashes(state.model),critic_sha256=critic_initialization_sha256(state.model),
            actor_updates=state.actor_updates,critic_updates=state.critic_updates)
        self.collecting[variant]=(start,start+n)
        events=[];resets={};end_refs=[None]*8
        for index in range(n):
            self.live();slot=index%8;env=self.environments[variant][slot]
            if env is None or env.state.terminated or env.state.truncated:
                stream=self.train_streams[slot];cursor=self.cursors[variant][slot]
                require(cursor<len(stream),'finite training scene inventory exhausted; no automatic extension')
                case=stream[cursor];env=self.new_environment(variant,case,scope='train')
                self.environments[variant][slot]=env;self.active_cases[variant][slot]=case
                file=folder/(case['episode_id']+'-initial.json')
                write_new(file,capture_environment(env));file.chmod(0o444);resets[case['episode_id']]=pointer(file)
            case=self.active_cases[variant][slot];pre=public_pre_action_snapshot(env.state)
            actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
            _,rewards,terminated,truncated,info=env.step(actions)
            require(info['proposed_actions']==actions and not info['action_override_agents'],'post-NN controller override')
            events.append(dict(slot=slot,case=case,pre_state_public=pre,post_state_public=public_pre_action_snapshot(env.state),
                proposed_actions=actions,action_override_agents=list(info['action_override_agents']),rewards=rewards,
                terminated=terminated,truncated=truncated,shutdowns=list(info['shutdowns'])))
            state.advance_joint_steps(1)
            if index>=n-8:
                file=folder/f'slot-{slot}-end.json';write_new(file,capture_environment(env));file.chmod(0o444);end_refs[slot]=pointer(file)
        require(training_fingerprint(replace(state,joint_steps=start))==before,'NN, critic or optimizer changed during physical collection')
        del self.collecting[variant]
        trace=dict(schema=TRACE_VERSION,partition='train',owner=self.owners[variant],config=asdict(self.config),
            start_joint_step=start,end_joint_step=state.joint_steps,actor=ref,rollout_state=identity,
            physics=dict(schema=PHYSICS_VERSION,resets=resets,end_snapshots=end_refs),events=events)
        file=folder/'trace.json';write_new(file,trace);file.chmod(0o444);trace_ref=pointer(file)
        batch,audit=seeds.assemble_training_batch(self.plan.proposal,self.readers[variant],trace_ref,state)
        audit_file=folder/'audit.json';write_new(audit_file,audit);audit_file.chmod(0o444)
        self.pending[variant]=dict(batch=batch,trace=trace_ref,audit=pointer(audit_file),state_sha256=training_fingerprint(state))
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('rollout_replayed',variant=variant,trace=trace_ref,audit=pointer(audit_file),joint_frames=n)
        return trace_ref

    @owned_operation
    def update(self,variant,*,evidence_provider=None):
        self.live(full=True);require(variant in self.pending,'no owned replayed rollout to optimize')
        state=self.states[variant];pending=self.pending[variant]
        require(training_fingerprint(state)==pending['state_sha256'],'training state changed since owned rollout replay')
        if variant=='regularized' and evidence_provider is None:evidence_provider=self.providers.get(variant)
        if variant=='raw':require(evidence_provider is None,'raw program feedback must remain zero')
        elif evidence_provider is not None:
            require(type(evidence_provider) is CurrentNNEvidenceProvider and evidence_provider.owner_ref==self.owners[variant],
                'regularized updates require the actual current-NN provider owned by this variant')
            require(self.providers.get(variant) is evidence_provider,'provider was not installed at the owned scheduled refit')
        require(pointer(pending['trace']['path'])==pending['trace'] and pointer(pending['audit']['path'])==pending['audit'],
            'owned rollout changed before optimizer update')
        self.event('optimizer_update_predebit',variant=variant,trace=pending['trace'],step=state.joint_steps)
        result=self.cores[variant].update_batch(pending['batch'],batch_sha256=pending['trace']['sha256'],evidence_provider=evidence_provider)
        file=Path(pending['trace']['path']).with_name('update.json');write_new(file,result);file.chmod(0o444)
        self.updates[variant].append(pointer(file));del self.pending[variant]
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('optimizer_update_accepted',variant=variant,receipt=pointer(file),step=state.joint_steps)
        return result

    @owned_operation
    def collect_program_source(self,variant):
        self.live(full=True);require(variant in self.states and variant not in self.pending and self.evidence_cursor[variant]==0,
            'program source collection cannot restart or overlap an optimizer batch')
        state=self.states[variant]
        require(state.joint_steps==self.config.extraction_steps[0] and self.cores[variant].processed_joint_steps==state.joint_steps
            and self.reserved[variant]==state.joint_steps and training_fingerprint(state)==self.state_fingerprints[variant],
            'program source is collected once at the actual completed 75% training prefix')
        folder=self.folder/variant/'program-source';folder.mkdir(parents=True,exist_ok=False)
        before=training_fingerprint(state);actor_ref,actor=frozen_export(state,folder/'behavior-NN.npz',variant);records={}
        self.collecting_source.add(variant)
        for case in self.evidence_cases:
            env=self.new_environment(variant,case,scope='program_evidence');episode=case['episode'];events=[]
            initial=folder/(episode+'-initial.json');write_new(initial,capture_environment(env));initial.chmod(0o444)
            while not(env.state.terminated or env.state.truncated):
                self.live();pre=public_pre_action_snapshot(env.state);actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
                _,rewards,terminated,truncated,info=env.step(actions)
                require(not info['action_override_agents'] and info['proposed_actions']==actions,'program-source NN override')
                events.append(dict(pre_state_public=pre,post_state_public=public_pre_action_snapshot(env.state),
                    proposed_actions=actions,action_override_agents=list(info['action_override_agents']),rewards=rewards,
                    terminated=terminated,truncated=truncated,shutdowns=list(info['shutdowns'])))
            final=folder/(episode+'-final.json');write_new(final,capture_environment(env));final.chmod(0o444)
            trace=folder/(episode+'-trace.json.gz')
            with gzip.open(trace,'xt') as handle:json.dump(dict(schema=observed.EPISODE_VERSION,case=case,behavior_actor=actor_ref,events=events),handle)
            trace.chmod(0o444);records[episode]=dict(behavior_actor=actor_ref,trace=pointer(trace),initial_environment=pointer(initial),final_environment=pointer(final))
        require(training_fingerprint(state)==before,'program evidence collection changed the NN training state')
        self.collecting_source.remove(variant)
        source=folder/'source.json';write_new(source,dict(schema=observed.VERSION,scope=observed.SCOPE,owner=self.owners[variant],
            inventory=self.plan.proposal.value['program_evidence_inventory'],cases=records));source.chmod(0o444)
        pool=observed.audit_observed_pool(pointer(source),reference_directory=self.reference_directory)
        self.pools[variant]=pool;self.event('program_source_replayed',variant=variant,source=pointer(source),audit_sha256=pool.audit_sha256,
            physical_frames=sum(len(part['x']) for part in pool.parts.values()),NN_joint_training_steps_added=0)
        return pool

    @owned_operation
    def record_refit(self,variant,binding_ref,*,evidence_provider=None):
        self.live(full=True);state=self.states[variant];step=state.joint_steps
        require(variant not in self.pending and len(self.refits[variant])<3 and step==self.config.extraction_steps[len(self.refits[variant])]
            and variant in self.pools and self.cores[variant].processed_joint_steps==self.reserved[variant]==step
            and training_fingerprint(state)==self.state_fingerprints[variant],'refit must follow its exact registered completed NN state')
        binding=bound(binding_ref);result=bound(binding['extraction']);identity=result['identity']
        require(binding['schema']=='warehouse-v642-observed-refit-source-binding.v1' and binding['source_files']==observed.source_files()
            and binding['owner']==self.owners[variant] and binding['observed_source']==self.pools[variant].source_ref
            and identity['owner']==self.owners[variant] and identity['variant']==variant and identity['step']==step
            and result['full_training_state_sha256']==training_fingerprint(state)
            and bound(pointer(Path(binding_ref['path']).with_name('observed-source-binding.completed.json')))==dict(binding_sha256=binding_ref['sha256']),
            'refit lacks owned physical source, NN state or completion binding')
        if variant=='regularized':
            require(type(evidence_provider) is CurrentNNEvidenceProvider and evidence_provider.owner_ref==self.owners[variant],
                'scheduled regularized refit must install its actual owned provider')
            evidence_provider.install_extraction(binding_ref,self.pools[variant],state);self.providers[variant]=evidence_provider
        else:require(evidence_provider is None,'raw refit cannot install feedback')
        self.refits[variant].append(step);self.event('scheduled_refit_bound',variant=variant,step=step,binding=binding_ref)

    @owned_operation
    def checkpoint(self,variant):
        self.live(full=True);require(variant in self.states and variant not in self.pending,'checkpoint only after a completed optimizer batch')
        require(training_fingerprint(self.states[variant])==self.state_fingerprints[variant],
            'checkpoint training state changed outside owned optimizer updates')
        state=self.states[variant]
        self.event('checkpoint_predebit',variant=variant,step=state.joint_steps)
        started=time.monotonic()
        folder=self.folder/variant/'checkpoints'/f'{state.joint_steps:06d}';folder.mkdir(parents=True,exist_ok=False)
        exported=save_checkpoint(state,folder/'state.pt')
        Path(exported['path']).chmod(0o444)
        live=[e is not None and not(e.state.terminated or e.state.truncated) for e in self.environments[variant]]
        data=dict(version=VERSION,claim=self.claim_ref,owner=self.owners[variant],source_files=self.sources,
            training_state=pointer(exported['path']),core=self.cores[variant].snapshot(),reader=self.readers[variant].snapshot(state),
            environments=[capture_environment(e) if live[i] else None for i,e in enumerate(self.environments[variant])],
            active_cases=[c if live[i] else None for i,c in enumerate(self.active_cases[variant])],
            reset_cursors=self.cursors[variant],program_source_cursor=self.evidence_cursor[variant],
            reserved_joint_steps=self.reserved[variant],refits=self.refits[variant],updates=self.updates[variant],
            provider=None if variant not in self.providers else self.providers[variant].snapshot(state),
            observed_source=None if variant not in self.pools else self.pools[variant].source_ref,
            observed_audit_sha256=None if variant not in self.pools else self.pools[variant].audit_sha256,
            full_training_state_sha256=training_fingerprint(state),paired_initial_identity=self.initial_identity,
            safe_batch_boundary=self.cores[variant].processed_joint_steps==state.joint_steps==self.reserved[variant],
            pending_optimizer_batches=0,safe_snapshot_not_resume_authority=True,automatic_resume=False)
        file=folder/'owner-state.json';write_new(file,data);file.chmod(0o444)
        self.event('checkpoint',variant=variant,snapshot=pointer(file),step=state.joint_steps,
            checkpoint_wall_seconds=time.monotonic()-started);return pointer(file)




def _make_owner(plan,clearance,sources,claim_ref,descriptor_ref,contract,folder,reference_directory,*,token):
    """Build every owner container in one place, before any scientific state."""
    require(token is _FACTORY_TOKEN,'owner state factory requires the source-bound claimed context')
    owner=LongTrainingOwner.__new__(LongTrainingOwner)
    owner.plan=plan;owner.clearance=clearance;owner.sources=deepcopy(sources)
    owner.claim_ref=claim_ref;owner.descriptor_ref=descriptor_ref;owner.source_contract_ref=contract
    claim=bound(claim_ref);start=datetime.fromisoformat(claim['started_utc'])
    owner.execution_expires_utc=datetime.fromisoformat(claim['expires_utc'])
    require(execution_lease(plan,clearance,started=start)=={key:claim[key] for key in
        ('started_utc','expires_utc','execution_lease_days','seed_clearance_start_deadline_utc','automatic_renewal')},
        'long execution claim has an unregistered duration, renewal or start time')
    owner.clearance_assets=[clearance['clearance'],contract]
    owner.config=plan.config;owner.folder=folder;owner.reference_directory=reference_directory
    owner.failed=False;owner.closed=False;owner.sequence=0
    owner.states={};owner.cores={};owner.readers={};owner.pending={};owner.pools={};owner.providers={}
    owner.initial_identity={};owner.owners={};owner.state_fingerprints={};owner.collecting={};owner.collecting_source=set()
    owner.reserved={v:0 for v in VARIANTS};owner.cursors={v:[0]*8 for v in VARIANTS}
    owner.evidence_cursor={v:0 for v in VARIANTS};owner.environments={v:[None]*8 for v in VARIANTS}
    owner.active_cases={v:[None]*8 for v in VARIANTS};owner.updates={v:[] for v in VARIANTS}
    owner.refits={v:[] for v in VARIANTS}
    inventory=bound(plan.value['train_case_inventory'])
    owner.train_streams={slot:[case for case in inventory['cases'] if case['slot']==slot] for slot in range(8)}
    owner.evidence_cases=bound(plan.value['program_evidence_inventory'])['cases']
    return owner


@contextmanager
def claimed_owner(*,reference_directory,calibration_precheck_ref,plan_ref):
    plan=load_seed_plan(reference_directory,calibration_precheck_ref,plan_ref)
    require(ACTIVE.get() is None,'long training owners cannot nest')
    clearance=seeds.checked_clearance(plan,execution=True)
    execution_lease(plan,clearance)  # No output/model/RNG for an expired start window.
    sources=source_files();source_sha=digest(sources);contract=checked_sources(plan)
    clearance_ref=pointer(Path(plan.ref['path']).parent/'long-owner-source-clearance.json')
    require(bound(clearance_ref)==dict(version=VERSION,owner_source_contract=contract,
        seed_clearance=clearance['clearance'],training_authority=False),
        'long seed clearance does not bind the registered actual executor')
    claim_file=claim_path(plan)
    require(not claim_file.exists(),'this stream/budget already consumed its single long-run attempt')
    folder=Path(plan.ref['path']).parent/'training'
    require(plan.value['execution_output']==str(folder) and plan.value['global_claim_path']==str(claim_file),
        'long execution output or global claim differs from exact seed reservation')
    for case in bound(plan.value['program_evidence_inventory'])['cases']:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',case['episode']) is not None,
            'unsafe long program episode ID')
    for case in bound(plan.value['train_case_inventory'])['cases']:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',case['episode_id']) is not None,
            'unsafe long training episode ID')
    folder.mkdir(parents=True,exist_ok=False)
    descriptor=dict(version=VERSION,stage='paired_long_development_training_not_formal_or_model_acceptance',
        seed_plan=plan.ref,calibration_precheck=calibration_precheck_ref,protocol_registration=plan.registration.value,
        phase=plan.value['phase'],stream_id=plan.value['stream_id'],source_files=sources,source_sha256=source_sha,
        clearance=clearance['clearance'],config=asdict(plan.config),owner_source_contract=contract,
        reference_actor=plan.readiness.value['actor'],training_partition_proposal=plan.ref,
        train_case_inventory=plan.value['train_case_inventory'],program_evidence_inventory=plan.value['program_evidence_inventory'],
        variants=list(VARIANTS),per_variant_joint_step_limit=plan.config.joint_step_budget,
        checkpoint_joint_steps=list(checkpoint_steps(plan.config)),every_budget_restarts_at_step_zero=True,
        sealed_authority=False,model_acceptance_passed=False,pilot_opened=False)
    target=folder/'descriptor.json';write_new(target,descriptor);target.chmod(0o444);descriptor_ref=pointer(target)
    require(seeds.checked_clearance(plan,execution=True)==clearance and source_files()==sources,
        'long source or exact history clearance changed immediately before claim')
    lease=execution_lease(plan,clearance)
    claim_file.parent.mkdir(parents=True,exist_ok=True)
    write_new(claim_file,dict(version=VERSION,descriptor=descriptor_ref,plan=plan.ref,clearance=clearance['clearance'],
        phase=plan.value['phase'],stream_id=plan.value['stream_id'],joint_step_budget=plan.config.joint_step_budget,
        owner_pid=os.getpid(),**lease,single_use_even_if_interrupted=True,
        automatic_resume=False,automatic_retry=False,sealed_authority=False,pilot_opened=False))
    claim_file.chmod(0o444)
    owner=_make_owner(plan,clearance,sources,pointer(claim_file),descriptor_ref,contract,folder,reference_directory,token=_FACTORY_TOKEN)
    owner.clearance_assets.append(clearance_ref)
    for variant in VARIANTS:
        target=folder/variant/'owner.json';target.parent.mkdir()
        write_new(target,dict(version=VERSION,variant=variant,claim=owner.claim_ref,descriptor=descriptor_ref,
            train_case_inventory=plan.value['train_case_inventory'],program_evidence_inventory=plan.value['program_evidence_inventory'],
            training_partition_proposal=plan.ref));target.chmod(0o444);owner.owners[variant]=pointer(target)
    current=ACTIVE.set(owner)
    try:
        with (folder/'owner-journal.jsonl').open('x') as journal,capture_stage1_resets(folder/'actual-resets.jsonl') as actual:
            owner.journal=journal;owner.actual_resets=actual
            owner.event('long_training_claimed',descriptor=descriptor_ref,source_contract=contract)
            yield owner
            owner.live(full=True)
            require(not owner.pending and not owner.collecting and not owner.collecting_source
                and set(owner.states)==set(VARIANTS)
                and all(owner.states[v].joint_steps==owner.cores[v].processed_joint_steps==plan.config.joint_step_budget
                    and owner.reserved[v]==plan.config.joint_step_budget
                    and owner.refits[v]==list(plan.config.extraction_steps)
                    and owner.evidence_cursor[v]==len(owner.evidence_cases) and not owner.cores[v].failed for v in VARIANTS),
                'long pair is incomplete; partial work is not a completed run')
            require(owner.cores['raw'].accepted_program_updates=={'robot_1':0,'robot_2':0}
                and all(n>0 for n in owner.cores['regularized'].accepted_program_updates.values()),
                'raw zero or regularized actual both-role positive program gradient requirement failed')
            sealed=seal_owned_artifacts(owner)
            owner.event('paired_long_steps_finished',per_variant_joint_steps={v:owner.states[v].joint_steps for v in VARIANTS})
            result=dict(version=VERSION,claim=owner.claim_ref,descriptor=descriptor_ref,updates=owner.updates,
                actual_reset_count=len(actual),per_variant_joint_steps={v:owner.states[v].joint_steps for v in VARIANTS},
                accepted_program_updates={v:dict(owner.cores[v].accepted_program_updates) for v in VARIANTS},
                sealed_owned_artifact_directories=sealed,
                training_finished=True,task_quality_and_final_NN_program_evaluation_required=True,
                budget_lock_issued=False,formal_training_authorized=False,sealed_authority=False,
                model_acceptance_passed=False,pilot_opened=False,publishable=False)
            write_new(folder/'completed.json',result);(folder/'completed.json').chmod(0o444)
    except BaseException as error:
        owner.failed=True
        write_new(folder/'failed.json',dict(version=VERSION,claim=owner.claim_ref,error_type=type(error).__name__,error=str(error),
            single_use_claim_retained=True,automatic_retry=False,automatic_resume=False,pilot_opened=False))
        (folder/'failed.json').chmod(0o444)
        raise
    finally:
        owner.closed=True;ACTIVE.reset(current)
        for name in ('owner-journal.jsonl','actual-resets.jsonl'):
            if (folder/name).exists():(folder/name).chmod(0o444)


def scheduled_refit(active,variant,programs):
    state=active.states[variant];step=state.joint_steps
    pool=active.collect_program_source(variant) if not active.refits[variant] else active.pools[variant]
    key=digest(active.owners[variant]);folder=active.folder/variant/'refit-targets'/str(step)
    folder.mkdir(parents=True,exist_ok=False)
    actor_ref,_=frozen_export(state,folder/'current-NN.npz',variant)
    targets=observed.relabel_current_NN(pool,actor_ref,observed.TARGET_ROOT/key/f'refit-{step}')
    result,binding=observed.refit_observed_pair_at_boundary(reference_directory=active.reference_directory,
        owner_ref=active.owners[variant],state=state,observed=pool,targets_ref=targets,program_refs=programs,
        output=EXTRACTION_ROOT/key/f'step-{step}',variant=variant,epoch_id=f'scheduled-{step}',config=active.config)
    provider=active.providers.get(variant)
    if variant=='regularized' and provider is None:
        provider=CurrentNNEvidenceProvider(reference_directory=active.reference_directory,owner_ref=active.owners[variant],
            output=PROVIDER_ROOT/key,pair_seed=state.initialization_seed,config=active.config)
    active.record_refit(variant,binding,evidence_provider=provider)
    return {role:result['roles'][role]['program'] for role in ('robot_1','robot_2')}


def drive(active,initial_programs):
    programs={v:deepcopy(initial_programs) for v in VARIANTS};checkpoints={v:[] for v in VARIANTS}
    for variant in VARIANTS:active.initialize(variant)
    while any(state.joint_steps<active.config.joint_step_budget for state in active.states.values()):
        for variant in VARIANTS:
            if active.states[variant].joint_steps>=active.config.joint_step_budget:continue
            active.collect_rollout(variant);active.update(variant)
            if active.states[variant].joint_steps in active.config.extraction_steps:
                programs[variant]=scheduled_refit(active,variant,programs[variant])
            if active.states[variant].joint_steps in checkpoint_steps(active.config):
                checkpoints[variant].append(active.checkpoint(variant))
    return checkpoints


def run(*,reference_directory,calibration_precheck_ref,plan_ref):
    with claimed_owner(reference_directory=reference_directory,calibration_precheck_ref=calibration_precheck_ref,plan_ref=plan_ref) as active:
        initial={role:row['program'] for role,row in active.plan.readiness.value['program_pair'].items()}
        checkpoints=drive(active,initial)
    result=dict(version=VERSION,owner_completed=pointer(active.folder/'completed.json'),
        source_contract=active.source_contract_ref,seed_plan=active.plan.ref,checkpoints=checkpoints,
        phase=active.plan.value['phase'],stream_id=active.plan.value['stream_id'],
        numerical_training_completed=True,quality_and_program_evaluation_required=True,
        model_acceptance_passed=False,pilot_opened=False,publishable=False)
    path=active.folder/'controller-completed.json';write_new(path,result);path.chmod(0o444)
    return pointer(path)
