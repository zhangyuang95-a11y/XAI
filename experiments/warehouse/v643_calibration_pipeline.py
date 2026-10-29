"""Actual finite v6.43 collector/PPO owner, awaiting public admission review.

The private context below performs real initialization, controlled resets and
updates only after a source-closed finite plan and final exact history check.
The public ``run`` deliberately remains closed: implementing a collector is
not a completed source/claim safety review. No old candidate global is patched.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import gzip
import json
import os
import sys

import numpy as np

from . import v643_calibration as numerical
from . import v643_calibration_owner as identity
from . import v643_calibration_recovery as recovery
from . import v643_training_seed_plan as seeds
from . import v643_program_namespace as programs
from . import v643_program_provider as feedback
from .v643_risk_recipe import CANDIDATE
from .v626_protocol import require, pointer, bound, digest
from backend.training import warehouse_v629_environment_replay as physics
from backend.training.warehouse_v629_rollout_replay import training_fingerprint, TRACE_VERSION
from backend.training.warehouse_v625_actor_training import PPOConfig, ReferenceTrainingBuffer
from backend.training.warehouse_v622_actor_training import initialize_from_reference, export_current_actor
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from backend.training.warehouse_v6_actor_training import role_hashes
from backend.training.warehouse_v6_runner import _set_dual_nn_mode
from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
from env.warehouse.domain import participant_study_config

VERSION='warehouse-v643-finite-calibration-pipeline.v1'
ASSETS_VERSION='warehouse-v643-completed-calibration-assets.v1'
WRAPPER_VERSION='warehouse-v643-completed-checkpoint-wrapper.v1'
QUALITY_VERSION='warehouse-v643-matched-quality-source.v1'
QUALITY_EPISODE_VERSION='warehouse-v643-quality-episode.v1'
ACTIVE=ContextVar('warehouse_v643_finite_calibration_owner',default=None)
Environment=physics.WarehouseMultiAgentEnv


def runtime_identity():
    names=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
    return dict(python_version=sys.version.split()[0],numpy_version=str(np.__version__),
        torch_version=str(numerical.torch.__version__),torch_threads=numerical.torch.get_num_threads(),
        numeric_thread_environment={name:os.environ.get(name) for name in names})


def source_files():
    return seeds.source_files()


def run(*,reference_directory,plan_ref,resume_ref=None):
    """Explicit new-plan entry; every scientific capability is checked below.

    No enable/pass flag can authorize this. Fresh execution requires original
    reference validation, the complete current source snapshot, exact finite
    inventories/history clearance and the unused candidate claim. Continuation
    instead proves a clean checkpoint under that same consumed claim.
    """
    require(type(plan_ref) is dict and set(plan_ref)=={'path','sha256'},'explicit new candidate plan reference required')
    if resume_ref is not None:
        require(type(resume_ref) is dict and set(resume_ref)=={'path','sha256'},'explicit clean-pair checkpoint reference required')
    plan=seeds.load(Path(reference_directory).resolve(),plan_ref)
    return _execute_reviewed_plan(plan) if resume_ref is None else _continue_reviewed_pair(plan,resume_ref)


def resume(*,reference_directory,plan_ref,checkpoint_ref):
    require(checkpoint_ref is not None,'explicit clean-pair checkpoint reference required')
    return run(reference_directory=reference_directory,plan_ref=plan_ref,resume_ref=checkpoint_ref)


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    return identity._exclusive_json(path,value)


def frozen_export(state,path,variant):
    result=export_current_actor(state,path,pair_seed=state.initialization_seed,variant=variant,candidate_id=CANDIDATE)
    Path(result['path']).chmod(0o444);ref=pointer(result['path'])
    actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=role_hashes(state.model))
    return ref,actor


def owned(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        try:return method(self,*args,**kwargs)
        except BaseException:
            self.failed=True
            raise
    return call


class FiniteCalibrationOwner:
    def __init__(self,*_args,**_kwargs):
        raise RuntimeError('finite collector requires the private source-checked claimed context')

    def live(self,*,full=False):
        require(ACTIVE.get() is self and not self.failed and not self.closed,'finite v643 owner is not active')
        require(datetime.now(timezone.utc)<datetime.fromisoformat(self.clearance['value']['expires_utc']),
                'v643 exact seed clearance expired')
        if full:
            self.plan.verify()
            require(source_files()==self.sources and pointer(self.claim_ref['path'])==self.claim_ref
                and pointer(self.descriptor_ref['path'])==self.descriptor_ref,
                'active v643 source/claim/descriptor changed')
            for ref in self.clearance_assets:
                require(pointer(ref['path'])==ref,'active v643 history evidence changed')

    def event(self,kind,**values):
        self.live()
        p=dict(version=VERSION,candidate_id=CANDIDATE,sequence=len(self.events),
            previous_sha256=None if not self.events else digest(self.events[-1]),
            claim=self.claim_ref,kind=kind,**values)
        raw=(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        self.journal.write(raw);self.journal.flush();os.fsync(self.journal.fileno());self.events.append(p)
        return p

    @owned
    def initialize(self,variant):
        self.live(full=True)
        require(variant in seeds.VARIANTS and variant not in self.states,'unknown or already initialized paired variant')
        p=self.plan.partition.value;self.ledger.predebit_initialization(variant);self.states[variant]=None
        self.event('initialization_predebit',variant=variant,seed=p['initialization_seed'])
        original=FrozenWarehouseV6Actor.load(p['reference_actor']['path'],expected_sha256=p['reference_actor']['sha256'])
        state=initialize_from_reference(p['reference_actor']['path'],p['reference_actor']['sha256'],original.role_sha256,
            initialization_seed=p['initialization_seed'],actor_lr=self.config.actor_learning_rate,critic_lr=self.config.critic_learning_rate)
        buffer=ReferenceTrainingBuffer.load(p['reference_buffer']['path'],p['reference_buffer']['sha256'],state.reference_sha256)
        common=numerical.validate_common_start(state,buffer)
        require(not self.initial_identity or common==self.initial_identity,'raw/regularized complete zero-update starts differ')
        self.initial_identity=deepcopy(common)
        self.event('optimizer_RNG_predebit',variant=variant,seed=p['optimizer_seed'])
        rng=np.random.default_rng(p['optimizer_seed'])
        core=numerical.V643PPOUpdateCore(state,buffer,variant=variant,rng=rng,config=self.config)
        if self.initial_rng is None:self.initial_rng=deepcopy(rng.bit_generator.state)
        require(self.initial_rng==rng.bit_generator.state,'paired optimizer/risk RNG starts differ')
        self.states[variant]=state;self.cores[variant]=core
        self.readers[variant]=physics.EnvironmentRolloutReplay(self.owners[variant],p['train_case_inventory'],config=self.config)
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('initialized',variant=variant,common_start=common,full_state_sha256=self.state_fingerprints[variant])
        return state

    @owned
    def new_environment(self,variant,case,*,scope,initial_path):
        self.live(full=True)
        if scope=='train':
            require(variant in self.collecting and self.collecting[variant][0]<=self.states[variant].joint_steps<self.collecting[variant][1],
                    'TRAIN reset requires its active predebited physical batch')
            phase='formal_'+variant+'_rollout'
            metadata=dict(start_joint_step=self.collecting[variant][0],episode_index=self.ledger.cursors[variant][case['slot']])
        elif scope=='program_evidence':
            require(variant in self.collecting_program and self.states[variant].joint_steps==15000,
                    'program-only source is collected exactly once at75percent')
            phase='program';metadata={}
        else:
            require(scope=='quality' and self.collecting_quality and self.training_complete(),
                    'quality reset requires the actual completed pair')
            phase='quality';metadata={}
        debit=self.ledger.predebit_reset(variant,case,scope=scope)
        self.event('environment_constructor_and_reset_predebit',variant=variant,scope=scope,
                   case=case,reset_predebit_sha256=debit['sha256'])
        before=len(self.actual_resets)
        with stage1_reset_phase(phase,**metadata):
            config=participant_study_config(seed=case['scenario_seed']) if scope=='quality' else physics.training_config(case['scenario_seed'])
            env=Environment(config)
            env.reset(seed=case['scenario_seed'])
        _set_dual_nn_mode(env)
        require(len(self.actual_resets)==before+1,'real reset hook count differs from one explicit physical reset')
        actual=self.actual_resets[-1]
        require(actual['seed']==case['scenario_seed'] and actual['episode_id']==env.state.episode_id==1,
                'actual reset hook identity differs from reserved original argument')
        initial=_write(initial_path,physics.capture_environment(env))
        binding=self.ledger.bind_actual_reset(variant,debit['sha256'],actual,initial_environment_ref=initial)
        self.event('actual_reset_bound',variant=variant,scope=scope,reset_hook_index=before,binding_sha256=binding['sha256'])
        return env,initial,binding

    @owned
    def collect_rollout(self,variant):
        self.live(full=True)
        require(variant in self.states and self.states[variant] is not None and variant not in self.pending,
                'initialize variant and optimize preceding physical batch before collecting')
        state=self.states[variant];start=state.joint_steps;n=8*self.config.next_collection_length(start)
        require(n>0 and self.reserved[variant]==self.cores[variant].processed_joint_steps==start
            and self.refits[variant]==[step for step in self.config.extraction_steps if step<=start]
            and training_fingerprint(state)==self.state_fingerprints[variant],
            'finite counter, scheduled extraction or owned state differs')
        self.event('rollout_predebit',variant=variant,start_joint_step=start,joint_frames=n)
        self.reserved[variant]+=n
        folder=self.folder/variant/'rollouts'/f'{start:06d}';folder.mkdir(parents=True,exist_ok=False)
        before=training_fingerprint(state);actor_ref,actor=frozen_export(state,folder/'current-NN.npz',variant)
        identity_before=dict(actor_role_sha256=role_hashes(state.model),
            critic_sha256=numerical.previous.critic_initialization_sha256(state.model),
            actor_updates=state.actor_updates,critic_updates=state.critic_updates)
        self.collecting[variant]=(start,start+n);events=[];resets={};ends=[None]*8
        for index in range(n):
            self.live();slot=index%8;env=self.environments[variant][slot]
            if env is None or env.state.terminated or env.state.truncated:
                cursor=self.ledger.cursors[variant][slot];stream=self.ledger.streams[slot]
                require(cursor<len(stream),'finite TRAIN inventory exhausted; no automatic extension')
                case=stream[cursor]
                env,initial,_=self.new_environment(variant,case,scope='train',initial_path=folder/(case['episode_id']+'-initial.json'))
                self.environments[variant][slot]=env;self.active_cases[variant][slot]=case;resets[case['episode_id']]=initial
            case=self.active_cases[variant][slot]
            events.append(self._step(env,actor,case));state.advance_joint_steps(1)
            if index>=n-8:ends[slot]=_write(folder/f'slot-{slot}-end.json',physics.capture_environment(env))
        require(training_fingerprint(replace(state,joint_steps=start))==before,'collection mutated NN/critics/optimizers')
        del self.collecting[variant]
        trace=dict(schema=TRACE_VERSION,partition='train',owner=self.owners[variant],config=asdict(self.config),
            start_joint_step=start,end_joint_step=state.joint_steps,actor=actor_ref,rollout_state=identity_before,
            physics=dict(schema=physics.PHYSICS_VERSION,resets=resets,end_snapshots=ends),
            events=[dict(slot=index%8,case=self._event_case(event),**{k:v for k,v in event.items() if k!='_case'})
                    for index,event in enumerate(events)])
        trace_ref=_write(folder/'trace.json',trace)
        allowed={c['episode_id']:c for c in self.ledger.training}
        require(all(allowed.get(e['case']['episode_id'])==e['case'] for e in trace['events']),
                'program/quality case cannot enter TRAIN replay')
        batch,audit=self.readers[variant].assemble(trace_ref,state)
        require(all(row['episode_id'] in allowed for key in ('rows','complete_cost_rows') for row in batch[key]),
                'non-TRAIN observation reached an Actor/reward/cost buffer')
        audit.update(program_evidence_frames_in_gradient_buffers=0,quality_frames_in_gradient_buffers=0)
        audit_ref=_write(folder/'audit.json',audit)
        self.pending[variant]=dict(batch=batch,trace=trace_ref,audit=audit_ref,state_sha256=training_fingerprint(state))
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('physical_rollout_replayed',variant=variant,trace=trace_ref,audit=audit_ref)
        self.live(full=True)
        return trace_ref

    @staticmethod
    def _event_case(event):return event['_case']

    @staticmethod
    def _step(env,actor,case):
        pre=physics.public_pre_action_snapshot(env.state)
        actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
        _,rewards,terminated,truncated,info=env.step(actions)
        require(info['proposed_actions']==actions and not info['action_override_agents'],'post-NN action override')
        return dict(_case=case,pre_state_public=pre,post_state_public=physics.public_pre_action_snapshot(env.state),
            proposed_actions=actions,action_override_agents=[],rewards=rewards,terminated=terminated,
            truncated=truncated,shutdowns=list(info['shutdowns']))

    @owned
    def update(self,variant,*,evidence_provider=None):
        self.live(full=True);require(variant in self.pending,'no owned physical batch to optimize')
        state=self.states[variant];pending=self.pending[variant]
        require(training_fingerprint(state)==pending['state_sha256'],'training state changed after physical replay')
        if variant=='raw':require(evidence_provider is None,'raw external program feedback must be zero')
        else:
            installed=self.providers.get(variant)
            if evidence_provider is None:evidence_provider=installed
            require(evidence_provider is installed and (installed is None or type(installed) is feedback.CurrentNNEvidenceProvider),
                    'only this owned current-NN provider may supply regularized feedback')
        require(pointer(pending['trace']['path'])==pending['trace'] and pointer(pending['audit']['path'])==pending['audit'],
                'physical batch/audit changed before optimization')
        self.event('optimizer_update_predebit',variant=variant,trace=pending['trace'])
        result=self.cores[variant].update_batch(pending['batch'],batch_sha256=pending['trace']['sha256'],evidence_provider=evidence_provider)
        ref=_write(Path(pending['trace']['path']).with_name('update.json'),result)
        self.updates[variant].append(ref);del self.pending[variant]
        self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('optimizer_update_accepted',variant=variant,receipt=ref,joint_steps=state.joint_steps)
        self.live(full=True);return result

    @owned
    def collect_program_source(self,variant):
        self.live(full=True);state=self.states[variant]
        require(variant not in self.pending and variant not in self.pools and self.ledger.program_cursors[variant]==0
            and state.joint_steps==self.cores[variant].processed_joint_steps==self.reserved[variant]==15000
            and training_fingerprint(state)==self.state_fingerprints[variant], 'program source is once-only at completed75percent')
        folder=self.folder/variant/'program-source';folder.mkdir(parents=True,exist_ok=False)
        before=training_fingerprint(state);actor_ref,actor=frozen_export(state,folder/'behavior-NN.npz',variant)
        self.collecting_program.add(variant);records={}
        for case in self.ledger.program:
            initial_path=folder/(case['episode']+'-initial.json')
            env,initial,_=self.new_environment(variant,case,scope='program_evidence',initial_path=initial_path)
            events=[]
            while not(env.state.terminated or env.state.truncated):
                self.live();require(len(events)<self.config.episode_horizon,'program episode exceeded its finite physical horizon')
                events.append({k:v for k,v in self._step(env,actor,case).items() if k!='_case'})
            final=_write(folder/(case['episode']+'-final.json'),physics.capture_environment(env))
            path=folder/(case['episode']+'-trace.json.gz')
            with path.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
                    zipped.write(json.dumps(dict(schema=programs.EPISODE_VERSION,case=case,behavior_actor=actor_ref,events=events),
                        sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                raw.flush();os.fsync(raw.fileno())
            path.chmod(0o444)
            records[case['episode']]=dict(behavior_actor=actor_ref,trace=pointer(path),initial_environment=initial,final_environment=final)
        self.collecting_program.remove(variant)
        source=_write(folder/'source.json',dict(schema=programs.VERSION,scope=programs.SCOPE,candidate_id=CANDIDATE,
            owner=self.owners[variant],inventory=self.plan.partition.value['program_evidence_inventory'],cases=records))
        require(training_fingerprint(state)==before,'program observations changed the NN training state')
        pool=programs.audit_observed_pool(self.namespaces[variant],source);self.pools[variant]=pool
        self.event('program_source_physically_replayed',variant=variant,source=source,audit_sha256=pool.audit_sha256)
        self.live(full=True);return pool

    @owned
    def scheduled_refit(self,variant,previous):
        self.live(full=True);state=self.states[variant];step=state.joint_steps
        require(variant not in self.pending and len(self.refits[variant])<3
            and step==self.config.extraction_steps[len(self.refits[variant])]
            and self.cores[variant].processed_joint_steps==self.reserved[variant]==step,
            'program refit must follow its exact completed physical/update boundary')
        pool=self.collect_program_source(variant) if not self.refits[variant] else self.pools[variant]
        namespace=self.namespaces[variant];folder=namespace.output_root/'targets'/str(step)
        actor_folder=self.folder/variant/'refit-actors'/str(step);actor_folder.mkdir(parents=True,exist_ok=False)
        actor_ref,_=frozen_export(state,actor_folder/'current-NN.npz',variant)
        target=feedback.relabel_current_NN(pool,actor_ref,folder)
        ref=feedback.refit_pair_at_boundary(namespace,state,pool,target,previous,
            namespace.output_root/'refits'/str(step),epoch_id=f'scheduled-{step}',config=self.config)
        result=bound(ref)
        require(result['identity']['step']==step and result['identity']['owner']==self.owners[variant]
            and result['identity']['full_training_state_sha256']==training_fingerprint(state), 'refit changed owner/current NN')
        if variant=='regularized':
            provider=self.providers.get(variant)
            if provider is None:provider=feedback.CurrentNNEvidenceProvider(namespace,output=namespace.output_root/'provider',config=self.config)
            provider.install_extraction(ref,pool,state);self.providers[variant]=provider
        self.refits[variant].append(step);self.refit_refs[variant].append(ref)
        self.event('scheduled_refit_bound',variant=variant,step=step,result=ref)
        self.live(full=True)
        return {role:result['roles'][role]['program'] for role in feedback.ROLES}

    @owned
    def checkpoint(self,variant):
        self.live(full=True);require(variant not in self.pending,'checkpoint only after an accepted optimizer boundary')
        state=self.states[variant]
        require(training_fingerprint(state)==self.state_fingerprints[variant],'checkpoint state changed outside owned updates')
        alive=[env is not None and not(env.state.terminated or env.state.truncated) for env in self.environments[variant]]
        ref=recovery.save_components(plan=self.plan,owner_ref=self.owners[variant],claim_ref=self.claim_ref,variant=variant,
            core=self.cores[variant],reader=self.readers[variant],ledger=self.ledger,
            environments=[physics.capture_environment(env) if alive[i] else None for i,env in enumerate(self.environments[variant])],
            active_cases=[case if alive[i] else None for i,case in enumerate(self.active_cases[variant])],
            program_source=None if variant not in self.pools else self.pools[variant].source_ref,
            provider=self.providers.get(variant),refits=self.refits[variant],updates=self.updates[variant])
        saved=bound(ref)
        wrapper=_write(Path(ref['path']).with_name('completed-wrapper.json'),dict(version=WRAPPER_VERSION,candidate_id=CANDIDATE,
            owner=self.owners[variant],claim=self.claim_ref,training_state=saved['numerical_boundary'],
            core=dict(version=numerical.VERSION,PPO=self.cores[variant].snapshot()['PPO'],
                      risk_training_state_in_training_state=True),core_snapshot_is_projection=True,
            updates=self.updates[variant],full_owner_boundary=ref))
        self.checkpoints[variant].append(wrapper);self.event('checkpoint',variant=variant,wrapper=wrapper)
        return wrapper

    @owned
    def pair_checkpoint(self):
        return recovery.save_clean_pair(self)

    def training_complete(self):
        return not self.pending and set(self.states)==set(seeds.VARIANTS) and all(self.states[v] is not None
            and self.states[v].joint_steps==self.cores[v].processed_joint_steps==self.reserved[v]==20000
            and self.refits[v]==list(self.config.extraction_steps) and not self.cores[v].failed for v in seeds.VARIANTS)

    @owned
    def final_assets(self):
        self.live(full=True);require(self.training_complete(),'actual full pair must finish before exporting final assets')
        require(self.cores['raw'].accepted_program_updates==dict(robot_1=0,robot_2=0)
            and all(n>0 for n in self.cores['regularized'].accepted_program_updates.values()),
            'raw must be zero; both regularized roles need accepted positive program gradients')
        actors=dict(reference=deepcopy(self.plan.partition.value['reference_actor']))
        reference=FrozenWarehouseV6Actor.load(actors['reference']['path'],expected_sha256=actors['reference']['sha256'])
        roles=dict(reference=reference.role_sha256)
        for variant in seeds.VARIANTS:
            require(self.checkpoints[variant] and self.states[variant].joint_steps==20000
                and bound(self.checkpoints[variant][-1])['core']['PPO']==self.cores[variant].snapshot()['PPO'],
                'missing or outdated final checkpoint')
            actors[variant],_=frozen_export(self.states[variant],self.folder/variant/'final-NN.npz',variant)
            roles[variant]=role_hashes(self.states[variant].model)
        ref=_write(self.folder/'completed-assets.json',dict(version=ASSETS_VERSION,candidate_id=CANDIDATE,
            plan=self.plan.ref,claim=self.claim_ref,descriptor=self.descriptor_ref,owners=self.owners,actors=actors,
            actor_role_sha256=roles,checkpoints={v:self.checkpoints[v][-1] for v in seeds.VARIANTS},updates=self.updates,
            program_sources={v:self.pools[v].source_ref for v in seeds.VARIANTS},refits=self.refit_refs,
            source_files=self.sources,source_sha256=digest(self.sources),training_completed=True,
            model_acceptance_passed=False,pilot_opened=False))
        self.final_assets_ref=ref;return ref

    @owned
    def bind_final_programs(self,ref):
        self.live(full=True)
        require(self.training_complete() and self.final_assets_ref is not None and self.final_programs_ref is None,
                'final programs can bind once after this completed pair')
        from . import v643_calibration_quality as audit
        audit.validate_final_programs(self.plan,self.final_assets_ref,ref,observed=self.pools)
        self.final_programs_ref=deepcopy(ref)
        self.event('final_programs_independently_validated',result=ref)
        return ref

    @owned
    def collect_quality(self):
        self.live(full=True)
        require(self.training_complete() and self.final_assets_ref is not None and self.final_programs_ref is not None
                and not self.quality_started,
                'fresh quality is once-only after actual completed frozen pair assets')
        from . import v643_calibration_quality as audit
        audit.validate_final_programs(self.plan,self.final_assets_ref,self.final_programs_ref,observed=self.pools)
        self.quality_started=True;assets=bound(self.final_assets_ref)
        require(pointer(self.final_assets_ref['path'])==self.final_assets_ref and assets['training_completed'] is True,
                'completed pair asset manifest changed')
        before={v:training_fingerprint(self.states[v]) for v in seeds.VARIANTS}
        actors=assets['actors'];folder=self.folder/'quality';folder.mkdir(exist_ok=False)
        self.collecting_quality=True;records={}
        for variant in ('reference',*seeds.VARIANTS):
            actor=FrozenWarehouseV6Actor.load(actors[variant]['path'],expected_sha256=actors[variant]['sha256'],
                                            expected_role_sha256=assets['actor_role_sha256'][variant])
            for case in self.ledger.quality:
                key=variant+':'+case['case_id'];base=folder/variant/case['case_id'];base.mkdir(parents=True,exist_ok=False)
                env,initial,binding=self.new_environment(variant,case,scope='quality',initial_path=base/'initial.json')
                events=[]
                while not(env.state.terminated or env.state.truncated):
                    self.live();require(len(events)<self.config.episode_horizon,'quality episode exceeded its finite physical horizon')
                    events.append({k:v for k,v in self._step(env,actor,case).items() if k!='_case'})
                final=_write(base/'final.json',physics.capture_environment(env))
                trace=_write(base/'trace.json',dict(schema=QUALITY_EPISODE_VERSION,case=case,
                    behavior_actor=actors[variant],events=events))
                records[key]=dict(variant=variant,case=case,behavior_actor=actors[variant],trace=trace,
                    initial_environment=initial,final_environment=final,reset_binding=binding)
        self.collecting_quality=False
        require(before=={v:training_fingerprint(self.states[v]) for v in seeds.VARIANTS},
                'quality evaluation modified a training state')
        self.ledger.handle.flush();os.fsync(self.ledger.handle.fileno());self.ledger.path.chmod(0o444)
        reset_path=self.folder/'actual-resets.jsonl';reset_path.chmod(0o444)
        ledger_ref=_write(folder/'owner-ledger.json',dict(version='warehouse-v643-owner-ledger-export.v1',candidate_id=CANDIDATE,
            plan=self.plan.ref,ledger=self.ledger.snapshot(),records=self.ledger.records))
        source=_write(folder/'source.json',dict(schema=QUALITY_VERSION,candidate_id=CANDIDATE,plan=self.plan.ref,
            claim=self.claim_ref,descriptor=self.descriptor_ref,inventory=self.plan.partition.value['quality_case_inventory'],
            actors=actors,cases=records,reset_journal=pointer(reset_path),owner_ledger_snapshot=ledger_ref))
        self.event('quality_collected_not_yet_audited',source=source,completed_policy_episodes=len(records),
                   training_steps_added=0,model_acceptance_passed=False)
        self.live(full=True);return source


def drive(active,initial_programs):
    """Bounded round-robin execution; no retries, alternative budget or actions."""
    require(active.config==PPOConfig(),'only the fixed paired20k recipe is registered')
    current={variant:deepcopy(initial_programs) for variant in seeds.VARIANTS}
    for variant in seeds.VARIANTS:active.live(full=True);active.initialize(variant)
    return _drive_initialized(active,current)


def _drive_initialized(active,current):
    require(active.config==PPOConfig() and set(active.states)==set(seeds.VARIANTS),
            'continuation needs the complete fixed paired state')
    while any(state.joint_steps<20000 for state in active.states.values()):
        for variant in seeds.VARIANTS:
            state=active.states[variant]
            if state.joint_steps>=20000:continue
            active.live(full=True);active.collect_rollout(variant);active.update(variant)
            if state.joint_steps in active.config.extraction_steps:current[variant]=active.scheduled_refit(variant,current[variant])
            if state.joint_steps in (4000,15000,16800,18600,20000):active.checkpoint(variant)
        values={state.joint_steps for state in active.states.values()}
        if len(values)==1 and next(iter(values)) in (4000,15000,16800,18600,20000):active.pair_checkpoint()
    return active.final_assets()


def _continue_reviewed_pair(plan,checkpoint_ref):
    """Continue a verified clean boundary under its original consumed claim."""
    with recovery._resume_clean_pair(plan,checkpoint_ref) as active:
        current={v:({role:bound(active.refit_refs[v][-1])['roles'][role]['program'] for role in feedback.ROLES}
                    if active.refit_refs[v] else deepcopy(plan.partition.value['program_templates'])) for v in seeds.VARIANTS}
        assets=_drive_initialized(active,current)
        return _evaluate_completed_pair(active,assets)


def _execute_reviewed_plan(plan):
    """Actual complete finite workflow, deliberately not a public admission.

    Root/CLI wiring must follow independent source/claim review. This function
    contains no parameter search, implicit retries or alternate seed streams.
    """
    from . import v643_calibration_quality as audit
    plan.verify()
    with _claimed_owner(plan) as active:
        assets=drive(active,plan.partition.value['program_templates'])
        return _evaluate_completed_pair(active,assets)


def _evaluate_completed_pair(active,assets):
    from . import v643_calibration_quality as audit
    plan=active.plan
    latest={v:{role:bound(active.refit_refs[v][-1])['roles'][role]['program']
               for role in feedback.ROLES} for v in seeds.VARIANTS}
    final_programs=audit.finalize_programs(plan,assets,namespaces=active.namespaces,states=active.states,
        observed=active.pools,last_program_refs=latest,
        output=identity.OWNER_ROOT/'final-programs'/plan.value['run_id'])
    active.bind_final_programs(final_programs)
    quality_source=active.collect_quality()
    measured=audit.audit_collection(plan,assets,quality_source,final_programs,
        output=identity.OWNER_ROOT/'quality'/plan.value['run_id'],observed=active.pools)
    verdict=bound(measured)
    require(verdict.get('version')==audit.VERSION and verdict.get('candidate_id')==CANDIDATE
        and type(verdict.get('calibration_precheck_passed')) is bool
        and verdict.get('plan')==plan.ref and verdict.get('claim')==active.claim_ref
        and verdict.get('independently_replayed_saved_physics_and_current_NN') is True
        and verdict.get('source_files')==audit.source_files()
        and type(verdict['quality_metrics']['task_retention_passed']) is bool
        and type(verdict['program_metrics']['passed']) is bool
        and verdict['calibration_precheck_passed']==(verdict['quality_metrics']['task_retention_passed']
                                                    and verdict['program_metrics']['passed']),
        'independent calibration verdict is missing or differently bound')
    active.calibration_receipt=measured;active.calibration_passed=verdict['calibration_precheck_passed']
    active.event('calibration_independently_measured',assets=assets,final_programs=final_programs,
        quality_source=quality_source,measurement=measured,calibration_precheck_passed=active.calibration_passed,
        candidate_closed_after_failed_calibration=not active.calibration_passed,model_acceptance_passed=False,pilot_opened=False)
    if not active.calibration_passed:
        _write(active.folder/'calibration-closed.json',dict(version=VERSION,candidate_id=CANDIDATE,claim=active.claim_ref,
            plan=plan.ref,measurement=measured,calibration_precheck_passed=False,candidate_closed_after_failed_calibration=True,
            training_completed=True,automatic_retry=False,pilot_opened=False))
    result=_write(active.folder/'pipeline-result.json',dict(version=VERSION,candidate_id=CANDIDATE,
        plan=plan.ref,claim=active.claim_ref,assets=assets,final_programs=final_programs,
        quality_source=quality_source,measurement=measured,calibration_precheck_passed=active.calibration_passed,
        candidate_closed_after_failed_calibration=not active.calibration_passed,model_acceptance_passed=False,pilot_opened=False))
    return result


@contextmanager
def _claimed_owner(plan):
    """Real private owner, callable only after independent entry/claim review.

    This helper is deliberately not wired to either public run/CLI. Its tests
    replace the entire reference/history namespace with temporary fixtures.
    """
    require(ACTIVE.get() is None,'finite v643 owners cannot nest')
    require(not identity.claim_path().exists(),'v643 candidate already consumed its only calibration attempt')
    plan.verify();sources=source_files();clearance=seeds.checked_clearance(plan,execution=True)
    descriptor=(identity.prepare_descriptor(plan) if not (Path(plan.ref['path']).parent/'calibration').exists()
                else pointer(Path(plan.ref['path']).parent/'calibration/descriptor.json'))
    require(bound(descriptor)==identity.descriptor_document(plan),'new owner descriptor changed')
    record=identity.compile_claim_record(plan,descriptor)
    # Last history/source check immediately before exclusive global candidate claim.
    require(seeds.checked_clearance(plan,execution=True)==clearance and source_files()==sources,
            'history/source changed immediately before candidate claim')
    identity.claim_path().parent.mkdir(parents=True,exist_ok=True)
    claim=identity._exclusive_json(identity.claim_path(),record)
    active=FiniteCalibrationOwner.__new__(FiniteCalibrationOwner)
    active.plan=plan;active.sources=sources;active.clearance=clearance;active.claim_ref=claim;active.descriptor_ref=descriptor
    active.folder=Path(descriptor['path']).parent;active.config=PPOConfig();active.failed=False;active.closed=False
    evidence=bound(clearance['value']['exact'])
    active.clearance_assets=[clearance['clearance'],clearance['value']['exact'],clearance['value']['registry'],
        evidence['exact_plan'],evidence['bounded_or_triaged_inventory']]
    if evidence.get('context_annotations'):active.clearance_assets.append(evidence['context_annotations'])
    active.states={};active.cores={};active.readers={};active.pending={};active.providers={};active.pools={};active.namespaces={}
    active.owners={};active.events=[];active.state_fingerprints={};active.initial_identity=None;active.initial_rng=None
    active.collecting={};active.collecting_program=set();active.collecting_quality=False
    active.environments={v:[None]*8 for v in seeds.VARIANTS};active.active_cases={v:[None]*8 for v in seeds.VARIANTS}
    active.reserved={v:0 for v in seeds.VARIANTS};active.updates={v:[] for v in seeds.VARIANTS}
    active.refits={v:[] for v in seeds.VARIANTS};active.refit_refs={v:[] for v in seeds.VARIANTS}
    active.checkpoints={v:[] for v in seeds.VARIANTS};active.pair_checkpoints=[]
    active.final_assets_ref=None;active.final_programs_ref=None;active.quality_started=False
    active.calibration_receipt=None;active.calibration_passed=None
    active.ledger=None;token=None
    try:
        for variant in seeds.VARIANTS:
            active.owners[variant]=_write(active.folder/variant/'owner.json',identity.variant_owner_document(plan,descriptor,claim,variant))
            active.namespaces[variant]=programs.ProgramNamespace.from_owner(active.owners[variant])
        active.ledger=identity.ResetLedger(plan,active.folder/'reset-predebits.jsonl')
        token=ACTIVE.set(active)
        with (active.folder/'owner-journal.jsonl').open('xb') as journal,capture_stage1_resets(active.folder/'actual-resets.jsonl') as resets:
            import fcntl
            fcntl.flock(journal.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            active.journal=journal;active.actual_resets=resets
            active.event('runtime_bound',runtime=runtime_identity())
            yield active
            active.live(full=True)
            require(active.training_complete() and active.final_assets_ref is not None,'finite pair incomplete; no success receipt')
            _write(active.folder/'completed.json',dict(version=VERSION,candidate_id=CANDIDATE,claim=claim,
                assets=active.final_assets_ref,training_completed=True,calibration_measurement=active.calibration_receipt,
                calibration_precheck_passed=active.calibration_passed,calibration_evaluation_still_required=active.calibration_passed is None,
                candidate_closed_after_failed_calibration=active.calibration_passed is False,
                model_acceptance_passed=False,pilot_opened=False))
    except BaseException as error:
        active.failed=True
        _write(active.folder/'failed.json',dict(version=VERSION,candidate_id=CANDIDATE,claim=claim,
            error_type=type(error).__name__,single_use_claim_retained=True,automatic_retry=False,pilot_opened=False))
        raise
    finally:
        if active.ledger is not None:active.ledger.close()
        active.closed=True
        if token is not None:ACTIVE.reset(token)
