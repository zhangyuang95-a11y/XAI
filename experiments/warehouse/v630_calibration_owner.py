"""Single-use 20k pair owner, only after actual A/B and exact seed clearance.

No CLI, imports or bare constructors can arm collection. Optimizer updates
must consume this owner's full physics/NN replay and registered current-NN
evidence. This is calibration, not development-budget or release admission.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import gzip,json,os,re
import numpy as np
from . import v630_training_seed_plan as seeds
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training import warehouse_v630_observed_pool as observed
from backend.training.warehouse_v629_environment_replay import (EnvironmentRolloutReplay,capture_environment,
    training_config,WarehouseMultiAgentEnv,public_pre_action_snapshot,PHYSICS_VERSION)
from backend.training.warehouse_v629_rollout_replay import training_fingerprint,TRACE_VERSION
from backend.training.warehouse_v630_evidence_provider import CurrentNNEvidenceProvider
from backend.training.warehouse_v629_ppo_update import PPOUpdateCore
from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer,critic_initialization_sha256
from backend.training.warehouse_v622_actor_training import initialize_from_reference,export_current_actor,save_checkpoint
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v6_actor_training import role_hashes
from backend.training.warehouse_v6_runner import _set_dual_nn_mode
from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
from scripts.run_warehouse_v625 import write_new

ROOT=seeds.ROOT
VERSION='warehouse-v630-finite-paired-calibration-owner.v1'
ACTIVE=ContextVar('warehouse_v630_paired_calibration_owner',default=None)


def source_files():
    files=seeds.source_files();files['experiments/warehouse/v630_calibration_owner.py']=file_sha(__file__)
    for name in ('tests/test_warehouse_v630_calibration_owner.py','docs/warehouse_v630_owner_bridge.md'):
        files[name]=file_sha(ROOT/name)
    return dict(sorted(files.items()))


def claim_path():return seeds.partitions.admission.v.OUTPUT/'training-calibration-20k.claim.json'


def load_seed_plan(reference_directory,plan_ref):
    # Keep the actual science gate first even when the caller passes None.
    ready=seeds.partitions.admission.load_readiness(reference_directory);ready.verify()
    value=bound(plan_ref);path=Path(plan_ref['path']).resolve()
    require(path==seeds.OUTPUT_ROOT/value['run_id']/'plan.json','calibration seed plan location differs')
    proposal=seeds.load_partition(reference_directory,value['partition_proposal'])
    return seeds.CalibrationSeedPlan(plan_ref,value,proposal).verify()


def prepare_sources(*,reference_directory,plan_ref):
    """Register the additional executor source before its seed audit."""
    plan=load_seed_plan(reference_directory,plan_ref);folder=Path(plan.ref['path']).parent
    require(not (folder/'seed-ledger').exists(),'register owner source before any calibration history audit')
    sources=source_files();sha=digest(sources)
    for name,expected in sources.items():
        target=folder/'owner-source'/sha/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==expected,'owner source changed before registration')
        with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        target.chmod(0o444)
    file=folder/'owner-source-contract.json'
    write_new(file,dict(version=VERSION,seed_plan=plan.ref,source_files=sources,source_sha256=sha,
        execution_output=str(folder/'calibration'),config=asdict(PPOConfig()),variants=list(seeds.VARIANTS),training_authority=False))
    file.chmod(0o444);return checked_sources(plan)


def checked_sources(plan):
    plan.verify();folder=Path(plan.ref['path']).parent;ref=pointer(folder/'owner-source-contract.json');value=bound(ref)
    require(value==dict(version=VERSION,seed_plan=plan.ref,source_files=source_files(),source_sha256=digest(source_files()),
        execution_output=str(folder/'calibration'),config=asdict(PPOConfig()),variants=list(seeds.VARIANTS),training_authority=False),
        'owner source contract or registered configuration differs')
    for name,sha in value['source_files'].items():
        path=folder/'owner-source'/value['source_sha256']/name
        require(file_sha(path)==sha and not path.stat().st_mode&0o222,'owner source snapshot changed')
    return ref


def audit_source_bound_seeds(*,reference_directory,plan_ref):
    plan=load_seed_plan(reference_directory,plan_ref);contract=checked_sources(plan)
    clearance=seeds.audit_seeds(plan)
    require(checked_sources(plan)==contract,'owner source changed during exact seed audit')
    file=Path(plan.ref['path']).with_name('owner-source-clearance.json')
    write_new(file,dict(version=VERSION,owner_source_contract=contract,seed_clearance=clearance['clearance'],training_authority=False))
    file.chmod(0o444);return pointer(file)


def frozen_export(state,path,variant):
    exported=export_current_actor(state,path,pair_seed=state.initialization_seed,variant=variant,
        candidate_id=seeds.partitions.admission.v.CANDIDATE)
    Path(exported['path']).chmod(0o444);ref=pointer(exported['path'])
    actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=role_hashes(state.model))
    return ref,actor


def owned_operation(method):
    def call(self,*args,**kwargs):
        try:return method(self,*args,**kwargs)
        except BaseException:
            self.failed=True
            raise
    return call


class CalibrationOwner:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('use the single-use claimed_owner context after actual A/B and seed clearance')

    def live(self,*,full=False):
        require(ACTIVE.get() is self and not self.failed and not self.closed,'calibration owner is not active')
        require(pointer(self.claim_ref['path'])==self.claim_ref and pointer(self.descriptor_ref['path'])==self.descriptor_ref,
            'calibration claim or descriptor changed')
        require(datetime.now(timezone.utc)<datetime.fromisoformat(self.clearance['value']['expires_utc']),
            'calibration seed clearance expired')
        if full:
            self.plan.verify();require(self.sources==source_files(),'calibration source changed')
            require(checked_sources(self.plan)==self.source_contract_ref,'registered owner source changed')
            require(pointer(self.clearance['clearance']['path'])==self.clearance['clearance'],
                'calibration history clearance changed')
            for ref in self.clearance_assets:require(pointer(ref['path'])==ref,'calibration history input asset changed')

    def event(self,kind,**values):
        self.live()
        value=dict(version=VERSION,sequence=self.sequence,kind=kind,claim=self.claim_ref,
            utc=datetime.now(timezone.utc).isoformat(),**values)
        self.journal.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        self.journal.flush();os.fsync(self.journal.fileno());self.sequence+=1
        return value

    @owned_operation
    def initialize(self,variant):
        self.live(full=True);require(variant in seeds.VARIANTS and variant not in self.states,'paired variant already initialized or unknown')
        p=self.plan.proposal;actor=p.readiness.value['actor'];config=self.config
        self.event('initialize_predebit',variant=variant,initialization_seed=p.value['initialization_seed'],reference_actor=actor)
        # Mark before RNG/model construction; failure cannot silently retry.
        self.states[variant]=None
        roles={role:row['language']['anchor_actor_role_sha256'] for role,row in p.readiness.value['program_pair'].items()}
        state=initialize_from_reference(actor['path'],actor['sha256'],roles,
            initialization_seed=p.value['initialization_seed'],actor_lr=config.actor_learning_rate,critic_lr=config.critic_learning_rate)
        identity=dict(roles=role_hashes(state.model),critic_sha256=critic_initialization_sha256(state.model))
        require(not self.initial_identity or identity==self.initial_identity,'raw/reg initial Actor or critic differs')
        self.initial_identity=identity
        self.event('optimizer_RNG_predebit',variant=variant,seed=p.value['optimizer_seed'])
        rng=np.random.default_rng(p.value['optimizer_seed'])
        ref=p.readiness.value['reference']['buffer']
        buffer=ReferenceTrainingBuffer.load(ref['path'],ref['sha256'],state.reference_sha256)
        seeds.partitions.check_reference_buffer(p,buffer)
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
        # record explicitly identify this as calibration, never formal train.
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
        batch,audit=seeds.partitions.assemble_training_batch(self.plan.proposal,self.readers[variant],trace_ref,state)
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
        require(binding['schema']=='warehouse-v630-observed-refit-source-binding.v1' and binding['source_files']==observed.source_files()
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
        state=self.states[variant];folder=self.folder/variant/'checkpoints'/f'{state.joint_steps:06d}';folder.mkdir(parents=True,exist_ok=False)
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
            safe_snapshot_not_resume_authority=True,automatic_resume=False)
        file=folder/'owner-state.json';write_new(file,data);file.chmod(0o444)
        self.event('checkpoint',variant=variant,snapshot=pointer(file));return pointer(file)


@contextmanager
def claimed_owner(*,reference_directory,plan_ref):
    plan=load_seed_plan(reference_directory,plan_ref)
    require(ACTIVE.get() is None,'calibration owners cannot nest')
    clearance=seeds.checked_clearance(plan,execution=True);sources=source_files();source_sha=digest(sources)
    contract=checked_sources(plan);source_clearance_ref=pointer(Path(plan.ref['path']).with_name('owner-source-clearance.json'))
    require(bound(source_clearance_ref)==dict(version=VERSION,owner_source_contract=contract,
        seed_clearance=clearance['clearance'],training_authority=False),'seed audit does not bind the registered owner source')
    require(not claim_path().exists(),'this candidate already consumed its single 20k calibration attempt')
    for case in bound(plan.value['case_reservations']):
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',case['episode']) is not None,
            'episode ID is unsafe for immutable artifact names')
    # The seed-plan directory is already an explicit, audited exclusion. A
    # new external output tree here would invalidate our own final stat delta.
    folder=Path(plan.ref['path']).parent/'calibration';folder.mkdir(parents=True,exist_ok=False)
    for name,sha in sources.items():
        target=folder/'source'/source_sha/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==sha,'source changed claiming calibration owner')
        target.write_bytes(raw);target.chmod(0o444)
    descriptor=dict(version=VERSION,stage='paired_20k_calibration_not_formal_training',seed_plan=plan.ref,
        source_files=sources,source_sha256=source_sha,clearance=clearance['clearance'],config=asdict(PPOConfig()),
        owner_source_contract=contract,owner_source_clearance=source_clearance_ref,
        reference_actor=plan.proposal.readiness.value['actor'],training_partition_proposal=plan.proposal.ref,
        train_case_inventory=plan.proposal.value['train_case_inventory'],program_evidence_inventory=plan.proposal.value['program_evidence_inventory'],
        variants=list(seeds.VARIANTS),per_variant_joint_step_limit=20000,sealed_authority=False,pilot_opened=False)
    write_new(folder/'descriptor.json',descriptor);(folder/'descriptor.json').chmod(0o444);descriptor_ref=pointer(folder/'descriptor.json')
    # Recheck history immediately before the exclusive, global candidate claim.
    require(seeds.checked_clearance(plan,execution=True)==clearance and source_files()==sources,'clearance/source changed before single use')
    write_new(claim_path(),dict(version=VERSION,descriptor=descriptor_ref,plan=plan.ref,clearance=clearance['clearance'],
        owner_pid=os.getpid(),single_use_even_if_interrupted=True,history_delta_checked_at_claim_not_continuously=True,
        automatic_resume=False,pilot_opened=False))
    claim_path().chmod(0o444);owner=CalibrationOwner.__new__(CalibrationOwner)
    owner.plan=plan;owner.clearance=clearance;owner.sources=sources;owner.claim_ref=pointer(claim_path());owner.descriptor_ref=descriptor_ref
    owner.source_contract_ref=contract
    evidence=bound(clearance['value']['exact']);owner.clearance_assets=[clearance['value']['exact'],evidence['exact_plan'],evidence['bounded_or_triaged_inventory'],clearance['value']['registry']]
    owner.clearance_assets.extend([contract,source_clearance_ref])
    if evidence.get('context_annotations'):owner.clearance_assets.append(evidence['context_annotations'])
    owner.config=PPOConfig();owner.folder=folder;owner.reference_directory=reference_directory;owner.failed=False;owner.closed=False;owner.sequence=0
    owner.states={};owner.cores={};owner.readers={};owner.pending={};owner.pools={};owner.providers={};owner.initial_identity={};owner.owners={}
    owner.state_fingerprints={};owner.collecting={};owner.collecting_source=set()
    owner.reserved={v:0 for v in seeds.VARIANTS};owner.cursors={v:[0]*8 for v in seeds.VARIANTS};owner.evidence_cursor={v:0 for v in seeds.VARIANTS}
    owner.environments={v:[None]*8 for v in seeds.VARIANTS};owner.active_cases={v:[None]*8 for v in seeds.VARIANTS}
    owner.updates={v:[] for v in seeds.VARIANTS};owner.refits={v:[] for v in seeds.VARIANTS}
    training=bound(plan.proposal.value['train_case_inventory'])['cases'];owner.train_streams={i:[c for c in training if c['slot']==i] for i in range(8)}
    owner.evidence_cases=bound(plan.proposal.value['program_evidence_inventory'])['cases']
    for variant in seeds.VARIANTS:
        target=folder/variant/'owner.json';target.parent.mkdir();write_new(target,dict(version=VERSION,variant=variant,claim=owner.claim_ref,
            descriptor=descriptor_ref,train_case_inventory=descriptor['train_case_inventory'],program_evidence_inventory=descriptor['program_evidence_inventory'],
            training_partition_proposal=plan.proposal.ref));target.chmod(0o444);owner.owners[variant]=pointer(target)
    token=ACTIVE.set(owner)
    try:
        with (folder/'owner-journal.jsonl').open('x') as journal,capture_stage1_resets(folder/'actual-resets.jsonl') as actual:
            owner.journal=journal;owner.actual_resets=actual;yield owner
            owner.live(full=True)
            require(not owner.pending and set(owner.states)==set(seeds.VARIANTS) and all(owner.states[v].joint_steps==20000
                and owner.cores[v].processed_joint_steps==20000 and owner.refits[v]==list(owner.config.extraction_steps) for v in seeds.VARIANTS),
                'calibration pair incomplete; partial work cannot become a successful receipt')
            require(not any(core.failed for core in owner.cores.values()),'failed numerical PPO core cannot complete calibration')
            require(owner.cores['raw'].accepted_program_updates=={'robot_1':0,'robot_2':0},'raw program gradients must be zero')
            require(all(n>0 for n in owner.cores['regularized'].accepted_program_updates.values()),
                'both regularized Actors require actual accepted positive program gradients')
            write_new(folder/'completed.json',dict(version=VERSION,claim=owner.claim_ref,descriptor=descriptor_ref,
                updates=owner.updates,actual_reset_count=len(actual),calibration_training_finished=True,
                quality_and_program_calibration_evaluation_still_required=True,pilot_opened=False,publishable=False))
            owner.closed=True
    except BaseException as error:
        owner.failed=True
        write_new(folder/'failed.json',dict(version=VERSION,claim=owner.claim_ref,error_type=type(error).__name__,error=str(error),
            actual_reset_journal=str(folder/'actual-resets.jsonl'),single_use_claim_retained=True,automatic_retry=False,pilot_opened=False))
        raise
    finally:owner.closed=True;ACTIVE.reset(token)
