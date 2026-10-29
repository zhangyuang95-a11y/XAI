"""v643 fixed-language refit and epoch-specific current-NN evidence.

Only numerical v642 algorithms are reused. This component requires an already
claimed new finite owner and never issues a training or recovery capability.
It constructs no environment and never supplies an execution action.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import json

import numpy as np

from backend.training import warehouse_v642_online_evidence as numerical
from backend.training.warehouse_v642_program import Program
from backend.training.warehouse_v642_training_program import complete_program, language_lock, limits
from backend.training.warehouse_v626_budget import Budget
from backend.training.warehouse_v625_actor_training import PPOConfig
from backend.training.warehouse_v622_actor_training import TrainingState, export_current_actor
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION
from backend.training.warehouse_v6_actor_training import role_hashes
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v629_training_feedback import coefficient
from env.warehouse.decision_protocol import independent_agent_seed
from experiments.warehouse.v626_protocol import require, pointer, bound, digest, file_sha
from scripts.audit_warehouse_v642_program import forward
from scripts.audit_warehouse_v628 import ledger
from . import v643_program_namespace as ns
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256

VERSION = 'warehouse-v643-current-NN-program-provider.v1'
TARGET_VERSION = 'warehouse-v643-current-NN-program-targets.v1'
REFIT_VERSION = 'warehouse-v643-fixed-language-current-NN-refit.v1'
EVIDENCE_VERSION = 'warehouse-v643-epoch-current-NN-program-evidence.v1'
ROLES = ('robot_1','robot_2')
source_files = ns.source_files


def write_new(path, value):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,'wb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return pointer(path)


def _check_state(namespace, state):
    namespace.verify()
    require(isinstance(state,TrainingState) and state.reference_sha256 == REFERENCE_SHA256
        and file_sha(state.reference_path) == REFERENCE_SHA256
        and state.initialization_seed == namespace.value['owner']['pair_seed'],
        'program provider requires the exact new paired reference/initialization')


def _current_actor(namespace, ref, expected_roles=None):
    ns.immutable(ref,decode=False)
    with np.load(ref['path'],allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata_json'].item()))
    require(metadata.get('candidate_id') == CANDIDATE and metadata.get('variant') == namespace.variant
        and metadata.get('pair_seed') == namespace.value['owner']['pair_seed']
        and metadata.get('reference_sha256') == REFERENCE_SHA256,
        'old or differently owned Actor export cannot supply v643 program targets')
    actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=expected_roles)
    require(actor.model_version == MODEL_VERSION, 'current target must retain the full v620 forward architecture')
    return actor


def _parts_match_inventory(namespace, parts):
    inventory={case['episode']:case for case in namespace.value['inventory']['cases']}
    require(set(parts) == set(numerical.legacy.SPLITS), 'four current-NN program evidence splits required')
    seen=set()
    for split,part in parts.items():
        for row in part['rows']:
            case=inventory.get(row['episode'])
            require(case is not None and case['split'] == split and str(case['base_seed']) == row['base_seed']
                and case['task'] == row['task'] and case['action_seed'] == row['action_seed'],
                'current-NN program rows are outside the claimed finite split')
            seen.add(row['episode'])
    require(seen == set(inventory), 'current-NN program pool omits finite evidence episodes')


def relabel_current_NN(observed, target_actor_ref, output):
    """Fresh full masked distributions on fixed, audited physical states.

    Sampled targets replay recorded action RNG identities. They are explicitly
    counterfactual, never executed outcomes or independent task-quality data.
    """
    require(type(observed) is ns.ObservedPool, 'v643 audited observed states required')
    observed.verify();namespace=observed.namespace
    actor=_current_actor(namespace,target_actor_ref)
    output=namespace.output(output);output.mkdir(parents=True,exist_ok=False)
    pools={};assets=[]
    for i,role in enumerate(ROLES):
        splits={}
        for split,part in observed.parts.items():
            x,m=part['x'][:,i],part['m'][:,i]
            p=numerical.legacy.runtime_probabilities(actor,role,x,m)
            sampled=np.array([np.random.default_rng(independent_agent_seed(base_seed=row['action_seed'],
                episode_id=row['episode_id'],frame=row['frame'],agent_id=role)).choice(5,p=prob)
                for row,prob in zip(part['rows'],p,strict=True)],np.int64)
            file=output/(role+'-'+split+'.npz')
            with file.open('xb') as handle:np.savez_compressed(handle,x=x,m=m,p=p,sampled=sampled)
            file.chmod(0o444)
            rows=write_new(output/(role+'-'+split+'-rows.json'),part['rows'])
            splits[split]=dict(arrays=pointer(file),rows=rows);assets.extend(splits[split].values())
        manifest=dict(schema=numerical.legacy.MANIFEST_VERSION,scope='online_program_training_evidence',
            target_actor=target_actor_ref,role=role,parts=splits)
        pools[role]=write_new(output/(role+'-pool.json'),manifest);assets.append(pools[role])
    observed.verify();ns.immutable(target_actor_ref,decode=False)
    payload=dict(version=TARGET_VERSION,candidate_id=CANDIDATE,owner=namespace.owner_ref,
        observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
        observed_data_sha256=observed.audit['observed_data_sha256'],target_actor=target_actor_ref,
        target_actor_role_sha256=actor.role_sha256,pools=pools,asset_refs=assets,
        namespace_sha256=namespace.payload_sha256,source_files=source_files(),
        counterfactual_targets_not_executed=True,actual_source_actions_unchanged=True,
        current_policy_task_quality_estimate=None,environment_resets=0,NN_updates=0,training_authority=False,pilot_opened=False)
    ref=write_new(output/'result.json',payload)
    write_new(output/'completed.json',dict(result_sha256=ref['sha256']))
    return ref


def checked_targets(observed, target_ref):
    observed.verify();namespace=observed.namespace;target=ns.immutable(target_ref)
    require(target.get('version') == TARGET_VERSION and target.get('candidate_id') == CANDIDATE
        and target['owner'] == namespace.owner_ref and target['observed_source'] == observed.source_ref
        and target['observed_audit_sha256'] == observed.audit_sha256
        and target['observed_data_sha256'] == observed.audit['observed_data_sha256']
        and target['namespace_sha256'] == namespace.payload_sha256 and target['source_files'] == source_files()
        and set(target['pools']) == set(ROLES) and target['counterfactual_targets_not_executed'] is True
        and target['actual_source_actions_unchanged'] is True and target['current_policy_task_quality_estimate'] is None
        and target['environment_resets'] == target['NN_updates'] == 0
        and target['training_authority'] is target['pilot_opened'] is False,
        'program targets are not bound to this new candidate, recorded pool and current NN')
    require(ns.immutable(pointer(Path(target_ref['path']).with_name('completed.json'))) == dict(result_sha256=target_ref['sha256']),
            'current-NN target completion seal differs')
    actor=_current_actor(namespace,target['target_actor'],target['target_actor_role_sha256'])
    parts={};refs=[target_ref,target['target_actor'],pointer(Path(target_ref['path']).with_name('completed.json'))]
    for i,role in enumerate(ROLES):
        parts[role],items=numerical.legacy.load_pool(target['pools'][role],target['target_actor'],actor,role)
        _parts_match_inventory(namespace,parts[role]);refs.extend(items)
        for split,part in parts[role].items():
            recorded=observed.parts[split]
            require(np.array_equal(part['x'],recorded['x'][:,i]) and np.array_equal(part['m'],recorded['m'][:,i])
                and part['rows'] == recorded['rows'], 'relabeling changed recorded public states or provenance')
    expected={digest(ref) for ref in refs[3:]}
    require(expected == {digest(ref) for ref in target['asset_refs']}, 'current-NN target asset closure differs')
    for ref in refs:ns.immutable(ref,decode=False)
    return target,parts,refs


@dataclass(frozen=True)
class ProgramEvidence(numerical.OnlineEvidence):
    namespace: ns.ProgramNamespace
    provider_receipt_ref: dict
    feedback_enabled: bool

    def payload(self):
        value=super().payload()
        value.update(version=EVIDENCE_VERSION,candidate_id=CANDIDATE,owner=self.namespace.owner_ref,
            namespace_sha256=self.namespace.payload_sha256,provider_receipt=self.provider_receipt_ref,
            provider_feedback_enabled=self.feedback_enabled,external_program_selects_execution_action=False)
        return value

    def verify(self, *, current_role_sha256, epoch_id):
        self.namespace.verify();super().verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)
        event=ns.immutable(self.provider_receipt_ref)
        require(event['version'] == VERSION and event['candidate_id'] == CANDIDATE and event['kind'] == 'validation'
            and event['owner'] == self.namespace.owner_ref and event['epoch_id'] == self.epoch_id
            and event['target_actor'] == self.target_actor_ref
            and event['target_actor_role_sha256'][self.program.role] == current_role_sha256
            and event['programs'][self.program.role] == self.program_ref
            and event['reports'][self.program.role] == digest(self.report)
            and self.feedback_enabled == (not event['latch_after']), 'v643 epoch evidence differs from current-NN receipt')
        return self

    def loss_targets(self,x,m,*,current_role_sha256,epoch_id):
        value=super().loss_targets(x,m,current_role_sha256=current_role_sha256,epoch_id=epoch_id)
        if not self.feedback_enabled:value['qualified']=np.zeros_like(value['qualified'],dtype=bool)
        return value


def refit_pair_at_boundary(namespace, state, observed, target_ref, program_refs, output, *, epoch_id, config=PPOConfig()):
    """Same v642 fixed-language fitter, with a new owner and asset wrapper."""
    require(type(namespace) is ns.ProgramNamespace and type(observed) is ns.ObservedPool,
            'v643 claimed owner and observed pool required before fitting')
    _check_state(namespace,state);observed.verify()
    require(config == PPOConfig() and observed.namespace.payload_sha256 == namespace.payload_sha256 and state.joint_steps in config.extraction_steps
        and type(epoch_id) is str and epoch_id and set(program_refs) == set(ROLES), 'unscheduled or unowned v643 refit')
    target,parts,refs=checked_targets(observed,target_ref)
    require(target['target_actor_role_sha256'] == role_hashes(state.model), 'refit target is not this exact current NN')
    before=training_fingerprint(state);sources=source_files();output=namespace.output(output)
    claim=namespace.output_root/'refit-claims'/f'{state.joint_steps}.json'
    require(not claim.exists(), 'v643 scheduled refit is once-only even after interruption')
    output.mkdir(parents=True,exist_ok=False)
    identity=dict(version=REFIT_VERSION,candidate_id=CANDIDATE,owner=namespace.owner_ref,variant=namespace.variant,
        step=state.joint_steps,epoch_id=epoch_id,config=asdict(config),namespace_sha256=namespace.payload_sha256,
        target_package=target_ref,target_actor=target['target_actor'],actor_role_sha256=target['target_actor_role_sha256'],
        programs=program_refs,full_training_state_sha256=before,source_files=sources)
    claim_ref=write_new(claim,dict(identity=identity,output=str(output),single_use_even_if_interrupted=True,training_authority=False))
    write_new(output/'started.json',dict(identity=identity,claim=claim_ref));results={}
    try:
        templates=namespace.value['owner'].get('program_templates')
        require(type(templates) is dict and set(templates) == set(ROLES), 'v643 finite owner must bind both starting program languages')
        for role in ROLES:
            original=ns.immutable(templates[role]);template=Program.from_dict(original,original['target_actor_role_sha256'])
            encoded=ns.immutable(program_refs[role]);program=Program.from_dict(encoded,encoded['target_actor_role_sha256'])
            require(program.role == role and language_lock(program) == language_lock(template)
                and program.maximum_depth <= 12 and program.maximum_leaves <= 256, 'refit changed the registered v642 program language')
            folder=output/role;folder.mkdir();budget=Budget(folder/'checkpoint.json',{**identity,'role':role},limits(program),1200)
            budget.phase='final'
            try:
                with budget.session.hard_deadline():
                    fitted,measurements=complete_program(program,parts[role],budget,refit=True,
                        target_actor_role_sha256=target['target_actor_role_sha256'][role])
                    report=numerical.legacy.current_report(fitted,parts[role])
                    for part in parts[role].values():
                        direct=fitted.distributions(part['x'],part['m']);independent=forward(fitted,part['x'],part['m'])
                        require(all(np.array_equal(a,b) for a,b in zip(direct,independent,strict=True)), 'independent program forward differs')
                    program_ref=write_new(folder/'program.json',fitted.to_dict())
                    budget.checkpoint(dict(program=fitted.to_dict(),current=report,measurements=measurements),status='completed')
                solver=budget.summary()
                require(ledger(folder,dict(solver=solver)) == solver['counts'] and solver['counts']['node'] == 0
                    and all(solver['counts'][key] <= solver['limits'][key] for key in solver['counts']),
                        'v643 refit budget ledger changed')
                results[role]=dict(program=program_ref,current=report,solver=solver,
                    target_actor_role_sha256=target['target_actor_role_sha256'][role],
                    full_training_state_sha256=before,qualification_inherited=False)
            finally:budget.session.close()
            for key,name in (('checkpoint','checkpoint.json'),('ledger','checkpoint.json.ledger.jsonl')):
                file=folder/name;file.chmod(0o444);results[role][key]=pointer(file)
        namespace.verify();observed.verify()
        require(source_files() == sources and training_fingerprint(state) == before, 'program refit changed source or NN training state')
        result=dict(version=REFIT_VERSION,identity=identity,claim=claim_ref,roles=results,
            observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
            target_package=target_ref,asset_refs=[*refs,claim_ref,*program_refs.values(),*templates.values(),
                *[result[key] for result in results.values() for key in ('program','checkpoint','ledger')]],
            NN_updates=0,environment_resets=0,external_program_selects_execution_action=False,training_authority=False,pilot_opened=False)
        ref=write_new(output/'result.json',result);write_new(output/'completed.json',dict(result_sha256=ref['sha256']))
        return ref
    except BaseException as error:
        write_new(output/'failed.json',dict(version=REFIT_VERSION,error_type=type(error).__name__,completed_roles=sorted(results),
            single_use_claim_retained=True,automatic_retry=False,training_authority=False,pilot_opened=False))
        raise


class CurrentNNEvidenceProvider:
    def __init__(self, namespace, *, output, config=PPOConfig()):
        require(type(namespace) is ns.ProgramNamespace, 'explicit v643 claimed namespace required')
        namespace.verify()
        require(namespace.variant == 'regularized', 'raw must not construct or call an external program provider')
        require(config == PPOConfig(), 'v643 program schedule/numerical configuration changed')
        output=namespace.output(output);output.mkdir(parents=True,exist_ok=False)
        self.namespace,self.output,self.config=namespace,output,config
        self.events=[];self.installed=[];self.used_epochs=set();self.latch=False;self.failed=False
        self.observed=None;self.programs=None;self.sources=source_files()
        self.started=write_new(output/'started.json',dict(version=VERSION,candidate_id=CANDIDATE,owner=namespace.owner_ref,
            namespace_sha256=namespace.payload_sha256,config=asdict(config),source_files=self.sources,training_authority=False))

    def verify(self):
        self.namespace.verify()
        started=ns.immutable(self.started)
        require(started == dict(version=VERSION,candidate_id=CANDIDATE,owner=self.namespace.owner_ref,
            namespace_sha256=self.namespace.payload_sha256,config=asdict(self.config),source_files=self.sources,training_authority=False),
            'v643 provider starting identity differs')
        require(not self.failed and self.sources == source_files(), 'v643 provider failed or source changed')
        paths=sorted((self.output/'events').glob('*.json')) if (self.output/'events').exists() else []
        require(paths == [Path(ref['path']) for ref in self.events], 'provider history has missing or post-snapshot events')
        previous=None;installed=[];epochs=set();latch=False;programs=None;calls=0
        for sequence,ref in enumerate(self.events):
            event=ns.immutable(ref)
            require(event['version'] == VERSION and event['candidate_id'] == CANDIDATE
                and event['owner'] == self.namespace.owner_ref and event['sequence'] == sequence
                and event['namespace_sha256'] == self.namespace.payload_sha256
                and event['previous'] == previous and event['latch_before'] == latch, 'v643 provider event chain differs')
            passed=event['passed_by_role']
            require(set(passed) == set(ROLES) and all(type(v) is bool for v in passed.values()), 'provider gate status malformed')
            if event['kind'] == 'install':
                require(len(installed) < 3 and event['step'] == self.config.extraction_steps[len(installed)], 'provider installed an unscheduled refit')
                installed.append(dict(step=event['step'],refit=event['refit'],event=ref));programs=event['programs'];latch=not all(passed.values())
            else:
                require(event['kind'] == 'validation' and event['epoch_id'] not in epochs and event['programs'] == programs,
                        'provider repeated epoch or changed installed program')
                epochs.add(event['epoch_id']);latch=latch or not all(passed.values())
                folder=self.output/'calls'/f'{calls:06d}'
                require(pointer(folder/'current-NN.npz') == event['target_actor'], 'provider current-NN export changed')
                for role in ROLES:
                    evidence=ns.immutable(pointer(folder/(role+'-evidence.json')))
                    require(evidence['version'] == EVIDENCE_VERSION and evidence['provider_receipt'] == ref
                        and evidence['epoch_id'] == event['epoch_id'] and evidence['current_target_actor'] == event['target_actor']
                        and evidence['anchor_program'] == programs[role] and digest(evidence['report']) == event['reports'][role]
                        and evidence['provider_feedback_enabled'] == (not latch), 'stored current-NN provider evidence differs')
                calls += 1
            require(event['latch_after'] == latch, 'feedback failure reopened without scheduled refit')
            for asset in event['asset_refs']:ns.immutable(asset,decode=False)
            previous=ref
        require(installed == self.installed and epochs == self.used_epochs and latch == self.latch and programs == self.programs,
                'v643 provider in-memory state differs from immutable events')
        call_paths=sorted((self.output/'calls').glob('*')) if (self.output/'calls').exists() else []
        require(call_paths == [self.output/'calls'/f'{i:06d}' for i in range(calls)],
                'provider has incomplete or post-snapshot validation work')
        if installed:
            event=ns.immutable(installed[-1]['event'])
            require(self.observed is not None and self.observed.source_ref == event['observed_source']
                and self.observed.audit_sha256 == event['observed_audit_sha256'], 'active observed pool differs from installed v643 source')
        return self

    def _append(self, **values):
        event=dict(version=VERSION,candidate_id=CANDIDATE,owner=self.namespace.owner_ref,
            namespace_sha256=self.namespace.payload_sha256,sequence=len(self.events),
            previous=self.events[-1] if self.events else None,training_authority=False,**values)
        ref=write_new(self.output/'events'/f'{len(self.events):06d}.json',event);self.events.append(ref);return ref

    def install_extraction(self, refit_ref, observed, state):
        try:
            self.verify();_check_state(self.namespace,state);observed.verify();value=ns.immutable(refit_ref)
            require(value['version'] == REFIT_VERSION and value['identity']['candidate_id'] == CANDIDATE
                and value['identity']['owner'] == self.namespace.owner_ref
                and value['identity']['namespace_sha256'] == self.namespace.payload_sha256
                and value['identity']['full_training_state_sha256'] == training_fingerprint(state)
                and len(self.installed) < 3 and state.joint_steps == self.config.extraction_steps[len(self.installed)]
                and value['identity']['step'] == state.joint_steps
                and value['observed_source'] == observed.source_ref and value['observed_audit_sha256'] == observed.audit_sha256,
                'only actual next scheduled current-NN refit can reset v643 feedback')
            require(ns.immutable(pointer(Path(refit_ref['path']).with_name('completed.json'))) == dict(result_sha256=refit_ref['sha256']),
                    'v643 refit has no completion seal')
            target,parts,refs=checked_targets(observed,value['target_package'])
            require(target['target_actor_role_sha256'] == role_hashes(state.model), 'installed refit is not this current NN')
            programs={r:value['roles'][r]['program'] for r in ROLES};passed={}
            for role in ROLES:
                encoded=ns.immutable(programs[role]);program=Program.from_dict(encoded,target['target_actor_role_sha256'][role])
                report=numerical.legacy.current_report(program,parts[role])
                require(report == value['roles'][role]['current'], 'installed refit current-NN qualification does not replay')
                passed[role]=report['gate']['passed']
            new_latch=not all(passed.values())
            event=self._append(kind='install',step=state.joint_steps,refit=refit_ref,programs=programs,
                observed_source=observed.source_ref,observed_audit_sha256=observed.audit_sha256,
                passed_by_role=passed,latch_before=self.latch,latch_after=new_latch,
                asset_refs=[refit_ref,*refs,*programs.values()])
            self.installed.append(dict(step=state.joint_steps,refit=refit_ref,event=event))
            self.observed,self.programs,self.latch=observed,programs,new_latch
            return event
        except BaseException:
            self.failed=True;raise

    def __call__(self, state, epoch_id):
        try:
            self.verify();_check_state(self.namespace,state)
            require(coefficient(state.joint_steps,self.config.joint_step_budget,'regularized') > 0
                and type(epoch_id) is str and 0 < len(epoch_id) <= 128 and epoch_id not in self.used_epochs,
                'program feedback requires a fresh in-window optimizer epoch')
            maximum=((self.config.joint_step_budget+self.config.minibatch_joint_frames-1)
                //self.config.minibatch_joint_frames)*self.config.epochs
            require(len(self.used_epochs) < maximum, 'finite v643 optimizer-epoch evidence budget exhausted')
            due=[step for step in self.config.extraction_steps if step < state.joint_steps]
            require(self.installed and self.installed[-1]['step'] >= max(due) and self.observed is not None,
                    'next scheduled extraction is required before later NN updates')
            self.used_epochs.add(epoch_id);before=training_fingerprint(state)
            folder=self.output/'calls'/f'{len(self.used_epochs)-1:06d}';folder.mkdir(parents=True,exist_ok=False)
            exported=export_current_actor(state,folder/'current-NN.npz',pair_seed=state.initialization_seed,
                variant='regularized',candidate_id=CANDIDATE)
            Path(exported['path']).chmod(0o444);actor_ref=pointer(exported['path'])
            targets=relabel_current_NN(self.observed,actor_ref,folder/'targets')
            target,parts,refs=checked_targets(self.observed,targets);roles=role_hashes(state.model)
            require(target['target_actor_role_sha256'] == roles, 'current exported NN role hashes differ')
            base={};passed={}
            for role in ROLES:
                encoded=ns.immutable(self.programs[role]);program=Program.from_dict(encoded,encoded['target_actor_role_sha256'])
                report=numerical.legacy.current_report(program,parts[role]);passed[role]=report['gate']['passed']
                base[role]=numerical.OnlineEvidence(program,report,deepcopy(self.programs[role]),actor_ref,roles[role],
                    target['pools'][role],epoch_id,tuple([*refs,self.programs[role],program.protection['parent']]),'')
            require(training_fingerprint(state) == before, 'provider changed NN, critic, optimizer or RNG state')
            new_latch=self.latch or not all(passed.values())
            event=self._append(kind='validation',epoch_id=epoch_id,step=state.joint_steps,programs=self.programs,
                target_actor=actor_ref,target_actor_role_sha256=roles,reports={r:digest(base[r].report) for r in ROLES},
                passed_by_role=passed,latch_before=self.latch,latch_after=new_latch,
                asset_refs=[*refs,*self.programs.values()],validation_actions_are_counterfactual=True,NN_updates=0)
            self.latch=new_latch;result={}
            for role,evidence in base.items():
                args={**evidence.__dict__,'namespace':self.namespace,'provider_receipt_ref':event,
                    'feedback_enabled':not self.latch,'envelope_digest':'',
                    'asset_refs':tuple([*evidence.asset_refs,event,self.namespace.owner_ref])}
                value=ProgramEvidence(**args);args['envelope_digest']=digest(value.payload());value=ProgramEvidence(**args)
                result[role]=value.verify(current_role_sha256=roles[role],epoch_id=epoch_id)
                write_new(folder/(role+'-evidence.json'),value.payload())
            self.verify();return result
        except BaseException:
            self.failed=True;raise

    def snapshot(self, state):
        self.verify();_check_state(self.namespace,state)
        payload=dict(version=VERSION,candidate_id=CANDIDATE,namespace_sha256=self.namespace.payload_sha256,
            owner=self.namespace.owner_ref,source_files=self.sources,config=asdict(self.config),output=str(self.output),
            events=deepcopy(self.events),installed=deepcopy(self.installed),used_epochs=sorted(self.used_epochs),
            programs=deepcopy(self.programs),latch=self.latch,full_training_state_sha256=training_fingerprint(state),
            observed_source=None if self.observed is None else self.observed.source_ref,
            observed_audit_sha256=None if self.observed is None else self.observed.audit_sha256,
            training_authority=False,resume_authority=False)
        return dict(payload=payload,payload_sha256=digest(payload))

    @classmethod
    def restore(cls, snapshot, state, *, namespace, observed, config=PPOConfig()):
        """Numerical/provider reconstruction only; no owner recovery permission."""
        namespace.verify();_check_state(namespace,state);value=snapshot['payload']
        require(state.resumed_training_state, 'provider restore requires a fully restored TrainingState')
        if value['observed_source'] is None:
            require(observed is None and not value['installed'], 'uninstalled provider cannot restore observed evidence')
        else:
            require(type(observed) is ns.ObservedPool, 'restored provider requires its replayed observed pool')
            observed.verify()
        require(value['version'] == VERSION and value['candidate_id'] == CANDIDATE
            and snapshot['payload_sha256'] == digest(value) and value['namespace_sha256'] == namespace.payload_sha256
            and value['owner'] == namespace.owner_ref and value['source_files'] == source_files()
            and value['config'] == asdict(config) and value['full_training_state_sha256'] == training_fingerprint(state)
            and value['observed_source'] == (None if observed is None else observed.source_ref)
            and value['observed_audit_sha256'] == (None if observed is None else observed.audit_sha256)
            and value['training_authority'] is value['resume_authority'] is False,
            'v643 provider checkpoint differs from exact current NN, owner, pool or source')
        require(namespace.variant == 'regularized', 'raw cannot restore external feedback')
        out=cls.__new__(cls);out.namespace=namespace;out.output=namespace.output(value['output']);out.config=config
        out.started=pointer(out.output/'started.json');out.sources=value['source_files'];out.events=deepcopy(value['events'])
        out.installed=deepcopy(value['installed']);out.used_epochs=set(value['used_epochs']);out.programs=deepcopy(value['programs'])
        out.latch=value['latch'];out.failed=False;out.observed=observed
        out.verify();return out
