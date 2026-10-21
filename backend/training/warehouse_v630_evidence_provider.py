"""Current-NN minibatch evidence and a durable fidelity-failure latch.

No collection, NN optimizer or training capability is issued here. A future
finite owner installs only scheduled, source-bound extractions and supplies
actual audited PPO frames separately. Raw/pre-window updates never need this
provider. Its diagnostics are not accepted-gradient/update counters.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from . import warehouse_v630_observed_pool as pools
from .warehouse_v629_training_feedback import coefficient
from .warehouse_v622_actor_training import TrainingState,export_current_actor
from .warehouse_v6_actor_training import role_hashes
from .warehouse_v625_actor_training import PPOConfig
from .warehouse_v629_rollout_replay import training_fingerprint
from experiments.warehouse import v630_training_admission as admission
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v630-current-NN-provider.v1'
EVIDENCE_VERSION='warehouse-v630-provider-controlled-evidence.v3'
PROVIDER_ROOT=admission.v.ROOT/'output/warehouse_v630/training-evidence-provider'
ROLES=('robot_1','robot_2')


def source_files():
    files=pools.source_files();files['backend/training/warehouse_v630_evidence_provider.py']=file_sha(__file__)
    for name in ('tests/test_warehouse_v630_evidence_provider.py','docs/warehouse_v630_provider_bridge.md',
                 'tests/test_warehouse_v629_evidence_provider.py'):
        files[name]=file_sha(admission.v.ROOT/name)
    return dict(sorted(files.items()))


@dataclass(frozen=True)
class ProviderEvidence(pools.ObservedOnlineEvidence):
    provider_receipt_ref:dict
    feedback_enabled:bool

    def payload(self):
        value=super().payload();value.update(version=EVIDENCE_VERSION,provider_receipt=self.provider_receipt_ref,
            provider_feedback_enabled=self.feedback_enabled,only_scheduled_extraction_can_reset_failure_latch=True)
        return value

    def verify(self,*,current_role_sha256,epoch_id):
        super().verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)
        r=bound(self.provider_receipt_ref)
        require(r['kind']=='validation' and r['epoch_id']==self.epoch_id and r['target_actor']==self.target_actor_ref
            and r['programs'][self.program.role]==self.program_ref and r['reports'][self.program.role]==digest(self.report)
            and self.feedback_enabled==(not r['latch_after']),'provider evidence differs from its validation/latch receipt')
        return self

    def loss_targets(self,x,m,*,current_role_sha256,epoch_id):
        out=super().loss_targets(x,m,current_role_sha256=current_role_sha256,epoch_id=epoch_id)
        if not self.feedback_enabled:out['qualified']=np.zeros_like(out['qualified'],dtype=bool)
        return out


class CurrentNNEvidenceProvider:
    def __init__(self,*,reference_directory,owner_ref,output,pair_seed,variant='regularized',config=PPOConfig()):
        ready=admission.load_readiness(reference_directory)  # Before output or NN export.
        ready.verify()
        require(variant=='regularized' and type(pair_seed) is int and 0<=pair_seed<2**63,
            'raw updates must not construct a program evidence provider')
        require(pointer(owner_ref['path'])==owner_ref,'provider owner changed')
        output=Path(output).resolve();require(output.is_relative_to(PROVIDER_ROOT.resolve()) and output!=PROVIDER_ROOT.resolve(),
            'provider requires a new isolated owner output directory')
        output.mkdir(parents=True,exist_ok=False)
        self.reference_directory=Path(reference_directory);self.ready=ready;self.owner_ref=deepcopy(owner_ref);self.output=output
        self.pair_seed,self.variant,self.config=pair_seed,variant,config;self.sources=source_files()
        self.events=[];self.installed=[];self.used_epochs=set();self.call_count=0;self.latch=False;self.reason=None
        self.observed=None;self.programs=None;self.failed=False
        write_new(output/'started.json',dict(version=VERSION,source_files=self.sources,owner=self.owner_ref,pair_seed=pair_seed,
            variant=variant,config=asdict(config),readiness_sha256=ready.payload_sha256,training_authority=False))
        self.started_ref=pointer(output/'started.json')

    def verify(self):
        self.ready.verify()
        require(not self.failed and self.sources==source_files() and pointer(self.owner_ref['path'])==self.owner_ref
            and pointer(self.started_ref['path'])==self.started_ref,'failed provider or changed source/owner')
        assets={};installed=[];epochs=set();calls=0;latch=False;reason=None;programs=None;previous=None
        paths=sorted((self.output/'events').glob('*')) if (self.output/'events').exists() else []
        require(paths==[Path(ref['path']) for ref in self.events],'provider has post-snapshot, missing or incomplete events')
        for sequence,ref in enumerate(self.events):
            require(pointer(ref['path'])==ref,'provider history changed')
            event=bound(ref)
            require(event['sequence']==sequence and event['previous']==previous and event['owner']==self.owner_ref
                and event['source_sha256']==digest(self.sources) and event['latch_before']==latch,'provider event chain changed')
            if event['kind']=='install':
                require(len(installed)<3 and event['step']==self.config.extraction_steps[len(installed)],'provider history contains an unscheduled extraction')
                latch=not all(event['passed_by_role'].values());reason=ref if latch else None;programs=event['programs']
                installed.append(dict(step=event['step'],binding=event['binding'],event=ref))
            else:
                require(event['kind']=='validation' and event['epoch_id'] not in epochs and event['sequence_call']==calls
                    and event['programs']==programs and event['prior_failure']==reason,'provider validation/program history changed')
                epochs.add(event['epoch_id']);calls+=1;disabled=latch or not all(event['passed_by_role'].values())
                if not latch and disabled:reason=ref
                latch=disabled
                folder=self.output/'calls'/f'{event["sequence_call"]:06d}'
                require(pointer(folder/'current-NN.npz')==event['target_actor'],'provider recorded another NN export')
                for role in ROLES:
                    e=bound(pointer(folder/(role+'-evidence.json')))
                    require(e['version']==EVIDENCE_VERSION and e['provider_receipt']==ref and e['epoch_id']==event['epoch_id']
                        and e['current_target_actor']==event['target_actor'] and e['anchor_program']==programs[role]
                        and digest(e['report'])==event['reports'][role] and e['provider_feedback_enabled']==(not latch),
                        'stored provider evidence differs from its event')
            require(event['latch_after']==latch,'provider event reopened feedback without scheduled extraction');previous=ref
            for asset in event['asset_refs']:
                require(asset['path'] not in assets or assets[asset['path']]==asset,'conflicting provider history asset')
                assets[asset['path']]=asset
        require(installed==self.installed and epochs==self.used_epochs and calls==self.call_count and latch==self.latch
            and reason==self.reason and programs==self.programs,'provider control state differs from immutable event history')
        call_paths=sorted((self.output/'calls').glob('*')) if (self.output/'calls').exists() else []
        require(call_paths==[self.output/'calls'/f'{i:06d}' for i in range(calls)],'provider has post-snapshot or incomplete validation calls')
        if installed:
            event=bound(installed[-1]['event'])
            require(self.observed is not None and self.observed.source_ref==event['observed_source']
                and self.observed.audit_sha256==event['observed_audit_sha256'],'active observed pool differs from installed source')
        for ref in assets.values():require(pointer(ref['path'])==ref,'provider history asset changed or missing')

    def check_state(self,state):
        require(isinstance(state,TrainingState) and state.reference_sha256==self.ready.value['actor']['sha256']
            and state.initialization_seed==self.pair_seed,'provider does not own this confirmed reference/initialization')

    def append(self,value):
        value=dict(version=VERSION,owner=self.owner_ref,sequence=len(self.events),previous=self.events[-1] if self.events else None,
            source_sha256=digest(self.sources),training_authority=False,**value)
        file=self.output/'events'/f'{len(self.events):06d}.json';file.parent.mkdir(exist_ok=True)
        write_new(file,value);file.chmod(0o444);ref=pointer(file);self.events.append(ref);return ref

    def install_extraction(self,binding_ref,observed,state):
        try:
            self.verify();self.check_state(state);observed.verify();before=training_fingerprint(state)
            b=bound(binding_ref);folder=Path(binding_ref['path']).parent
            require(bound(pointer(folder/'observed-source-binding.completed.json'))==dict(binding_sha256=binding_ref['sha256'])
                and b['schema']=='warehouse-v630-observed-refit-source-binding.v1' and b['source_files']==pools.source_files()
                and b['owner']==self.owner_ref and b['observed_source']==observed.source_ref and b['observed_audit_sha256']==observed.audit_sha256,
                'scheduled extraction lacks its observed-source completion binding')
            result=bound(b['extraction']);identity=result['identity'];step=state.joint_steps
            require(len(self.installed)<3 and step==self.config.extraction_steps[len(self.installed)]
                and identity['step']==step and identity['owner']==self.owner_ref and identity['variant']==self.variant
                and identity['config']==asdict(self.config) and result['full_training_state_sha256']==before,
                'only next scheduled extraction at its exact completed NN state can reset feedback')
            programs={role:result['roles'][role]['program'] for role in ROLES}
            target,_=pools.checked_targets(observed,b['target_package'])
            require(target['target_actor']==identity['current_target_actor'],'extraction target NN changed')
            evidence={role:pools.build_observed_evidence(programs[role],observed,b['target_package'],epoch_id=f'install-{step}') for role in ROLES}
            require(all(e.target_role_sha256==role_hashes(state.model)[role]
                and e.report==result['roles'][role]['current'] for role,e in evidence.items()),'scheduled extraction current-NN validation differs')
            passed={role:e.report['gate']['passed'] for role,e in evidence.items()}
            new_latch=not all(passed.values())
            event=self.append(dict(kind='install',step=step,binding=binding_ref,observed_source=observed.source_ref,
                observed_audit_sha256=observed.audit_sha256,target_package=b['target_package'],programs=programs,
                asset_refs=[binding_ref,b['extraction'],*observed.refs,*[ref for e in evidence.values() for ref in e.asset_refs]],
                passed_by_role=passed,latch_before=self.latch,latch_after=new_latch,optimizer_steps=0))
            self.installed.append(dict(step=step,binding=deepcopy(binding_ref),event=event))
            self.observed=observed;self.programs=programs;self.latch=new_latch;self.reason=event if new_latch else None
            require(training_fingerprint(state)==before,'program installation changed training state');return event
        except BaseException:
            self.failed=True;raise

    def __call__(self,state,epoch_id):
        try:
            self.verify();self.check_state(state)
            require(coefficient(state.joint_steps,self.config.joint_step_budget,self.variant)>0,'provider called before feedback window')
            require(type(epoch_id) is str and epoch_id and epoch_id not in self.used_epochs,'optimizer epoch was missing or already consumed')
            # A batch ENDING at 84/93% uses the preceding program. The owner
            # installs the new extraction after that batch's PPO update and
            # before collecting the next batch, matching the original driver.
            due=[s for s in self.config.extraction_steps if s<state.joint_steps]
            require(self.installed and self.installed[-1]['step']>=max(due) and self.observed is not None,
                'next scheduled extraction is required before later rollout updates')
            self.used_epochs.add(epoch_id);sequence=self.call_count;self.call_count+=1
            before=training_fingerprint(state);folder=self.output/'calls'/f'{sequence:06d}';folder.mkdir(parents=True,exist_ok=False)
            exported=export_current_actor(state,folder/'current-NN.npz',pair_seed=self.pair_seed,variant=self.variant,candidate_id=admission.v.CANDIDATE)
            Path(exported['path']).chmod(0o444);actor_ref=pointer(exported['path'])
            targets=pools.relabel_current_NN(self.observed,actor_ref,pools.TARGET_ROOT/digest(self.owner_ref)/f'call-{sequence:06d}')
            evidence={role:pools.build_observed_evidence(self.programs[role],self.observed,targets,epoch_id=epoch_id) for role in ROLES}
            passed={role:e.report['gate']['passed'] for role,e in evidence.items()};new_latch=self.latch or not all(passed.values())
            require(training_fingerprint(state)==before,'evidence provider changed NN/critic/optimizer/RNG state')
            event=self.append(dict(kind='validation',epoch_id=epoch_id,step=state.joint_steps,sequence_call=sequence,
                target_actor=actor_ref,target_package=targets,programs=self.programs,
                reports={role:digest(e.report) for role,e in evidence.items()},passed_by_role=passed,
                asset_refs=[self.installed[-1]['binding'],*self.observed.refs,*[ref for e in evidence.values() for ref in e.asset_refs]],
                latch_before=self.latch,latch_after=new_latch,prior_failure=self.reason,
                validation_actions_are_counterfactual=True,accepted_gradient_updates='recorded_only_by_actual_PPO_update_core',optimizer_steps=0))
            if not self.latch and new_latch:self.reason=event
            self.latch=new_latch;result={}
            for role,e in evidence.items():
                args={**e.__dict__,'asset_refs':tuple([*e.asset_refs,event,self.installed[-1]['binding'],self.owner_ref]),
                    'provider_receipt_ref':event,'feedback_enabled':not self.latch,'envelope_digest':''}
                out=ProviderEvidence(**args);out=ProviderEvidence(**{**args,'envelope_digest':digest(out.payload())})
                result[role]=out.verify(current_role_sha256=role_hashes(state.model)[role],epoch_id=epoch_id)
                write_new(folder/(role+'-evidence.json'),out.payload())
            self.verify();return result
        except BaseException:
            self.failed=True;raise

    def snapshot(self,state):
        self.verify();self.check_state(state)
        value=dict(version=VERSION,source_files=self.sources,owner=self.owner_ref,started=self.started_ref,
            pair_seed=self.pair_seed,variant=self.variant,config=asdict(self.config),output=str(self.output),
            full_training_state_sha256=training_fingerprint(state),events=deepcopy(self.events),installed=deepcopy(self.installed),
            used_epochs=sorted(self.used_epochs),call_count=self.call_count,latch=self.latch,reason=deepcopy(self.reason),
            observed_source=None if self.observed is None else self.observed.source_ref,
            observed_audit_sha256=None if self.observed is None else self.observed.audit_sha256,programs=deepcopy(self.programs),training_authority=False)
        return dict(payload=value,payload_sha256=digest(value))

    @classmethod
    def restore(cls,snapshot,state,*,reference_directory,owner_ref,observed,config=PPOConfig()):
        ready=admission.load_readiness(reference_directory);ready.verify();p=snapshot['payload']
        require(snapshot['payload_sha256']==digest(p) and p['version']==VERSION and p['source_files']==source_files()
            and p['owner']==owner_ref and p['config']==asdict(config) and state.resumed_training_state
            and p['full_training_state_sha256']==training_fingerprint(state) and p['training_authority'] is False,
            'provider restore requires exact full TrainingState, owner and source')
        require(p['variant']=='regularized' and type(p['pair_seed']) is int and 0<=p['pair_seed']<2**63,'invalid restored provider variant/seed')
        observed.verify();require(p['observed_source']==observed.source_ref and p['observed_audit_sha256']==observed.audit_sha256,
            'restored provider observed states differ')
        output=Path(p['output']).resolve();require(output.is_relative_to(PROVIDER_ROOT.resolve()) and output!=PROVIDER_ROOT.resolve(),
            'provider resume output is not isolated')
        events=[];installed=[];epochs=set();calls=0;latch=False;reason=None;programs=None
        for sequence,ref in enumerate(p['events']):
            require(pointer(ref['path'])==ref and Path(ref['path'])==output/'events'/f'{sequence:06d}.json','provider event history missing or reordered')
            event=bound(ref)
            require(event['version']==VERSION and event['sequence']==sequence and event['owner']==owner_ref
                and event['source_sha256']==digest(p['source_files']) and event['previous']==(events[-1] if events else None)
                and event['latch_before']==latch and set(event['passed_by_role'])==set(ROLES)
                and all(type(v) is bool for v in event['passed_by_role'].values()),'provider event chain or fidelity status differs')
            if event['kind']=='install':
                require(len(installed)<3 and event['step']==config.extraction_steps[len(installed)],'provider installed an unscheduled extraction')
                latch=not all(event['passed_by_role'].values());reason=ref if latch else None;programs=event['programs']
                installed.append(dict(step=event['step'],binding=event['binding'],event=ref))
                require(pointer(event['binding']['path'])==event['binding'],'installed source binding changed')
            else:
                require(event['kind']=='validation' and event['epoch_id'] not in epochs and event['sequence_call']==calls
                    and event['prior_failure']==reason and event['programs']==programs,'provider repeated an epoch or changed failure/program history')
                epochs.add(event['epoch_id']);calls+=1;new_latch=latch or not all(event['passed_by_role'].values())
                if not latch and new_latch:reason=ref
                latch=new_latch
            require(event['latch_after']==latch,'provider latch was reopened without scheduled extraction');events.append(ref)
        require(installed==p['installed'] and sorted(epochs)==p['used_epochs'] and calls==p['call_count']
            and latch==p['latch'] and reason==p['reason'] and programs==p['programs'],'provider snapshot changed its event-derived latch or programs')
        out=cls.__new__(cls);out.reference_directory=Path(reference_directory);out.ready=ready;out.owner_ref=deepcopy(owner_ref)
        out.output=output;out.pair_seed=p['pair_seed'];out.variant=p['variant'];out.config=config;out.sources=p['source_files']
        out.events=events;out.installed=installed;out.used_epochs=epochs;out.call_count=calls;out.latch=latch;out.reason=reason
        out.programs=programs;out.observed=observed;out.failed=False;out.started_ref=p['started'];out.verify();out.check_state(state)
        header=bound(out.started_ref)
        require(header==dict(version=VERSION,source_files=out.sources,owner=owner_ref,pair_seed=out.pair_seed,variant=out.variant,
            config=asdict(config),readiness_sha256=ready.payload_sha256,training_authority=False),'provider resume header differs')
        return out
