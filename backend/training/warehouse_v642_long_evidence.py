"""Explicit long-owner callback composed with the unchanged native provider.

The native provider remains the real state/latch/evidence object. Only the
pre-binding target storage writer differs. This module does not replace any
production function or mutate the original provider class/module.
"""
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
from . import warehouse_v642_evidence_provider as native
from . import warehouse_v642_long_targets as targets
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-explicit-long-storage-evidence-callback.v1'
ROOT=Path(__file__).resolve().parents[2]
FILES=('backend/training/warehouse_v642_long_evidence.py','tests/test_warehouse_v642_long_evidence.py',
       'docs/warehouse_v642_long_evidence.md')


def source_files():
    files=targets.source_files();files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def _put(path,value):
    write_new(path,value);path.chmod(0o444);return pointer(path)


class LongEvidenceCallback:
    def __init__(self,active,provider):
        require(type(provider) is native.CurrentNNEvidenceProvider,'long callback requires the real native provider')
        targets._active_owner(active,'regularized',provider.observed)
        require(active.providers.get('regularized') is provider and provider.owner_ref==active.owners['regularized']
            and provider.call_count==0,'only a new owned native provider may install the explicit callback')
        self.active=active;self.native=provider;self.sources=source_files();self.events=[];self.failed=False
        require(all(active.sources.get(k)==v for k,v in self.sources.items()),'owner source omitted explicit long callback')
        self.folder=provider.output/'long-storage-callback';self.folder.mkdir(exist_ok=False)
        self.started=_put(self.folder/'started.json',dict(version=VERSION,source_files=self.sources,
            owner=provider.owner_ref,native_provider_started=provider.started_ref,native_sources=provider.sources,
            materialization='new_unsealed_stage_only',NN_method_changed=False,pilot_opened=False))

    def verify(self,*,native_verify=True):
        require(not self.failed and self.sources==source_files() and pointer(self.started['path'])==self.started,
            'long callback failed or source/header changed')
        targets._immutable(self.started)
        targets._active_owner(self.active,'regularized',self.native.observed)
        require(self.active.providers.get('regularized') is self.native,'callback native provider changed')
        if native_verify:self.native.verify()
        header=bound(self.started)
        require(header==dict(version=VERSION,source_files=self.sources,owner=self.native.owner_ref,
            native_provider_started=self.native.started_ref,native_sources=self.native.sources,
            materialization='new_unsealed_stage_only',NN_method_changed=False,pilot_opened=False),'callback header changed')
        validation=[ref for ref in self.native.events if bound(ref)['kind']=='validation']
        require(len(self.events)==len(validation)==self.native.call_count,'native calls bypassed explicit long callback')
        files=sorted((self.folder/'calls').glob('*.json')) if (self.folder/'calls').exists() else []
        require(files==[Path(r['path']) for r in self.events],'callback history has incomplete or extra calls')
        for i,ref in enumerate(self.events):
            require(pointer(ref['path'])==ref,'callback ledger changed')
            targets._immutable(ref)
            item=bound(ref);event=bound(validation[i]);package=bound(event['target_package'])
            require(item['version']==VERSION and item['sequence_call']==i and item['native_validation']==validation[i]
                and item['previous']==(self.events[i-1] if i else None) and item['source_sha256']==digest(self.sources)
                and item['target_package']==event['target_package'] and item['provenance']==package['long_materialization_provenance']
                and event['long_storage_callback_started']==self.started and event['long_materialization_provenance']==item['provenance'],
                'callback/native validation/materialization chain differs')

    def __call__(self,state,epoch_id):
        p=self.native
        try:
            self.verify();p.check_state(state)
            require(native.coefficient(state.joint_steps,p.config.joint_step_budget,p.variant)>0,'provider called before feedback window')
            require(type(epoch_id) is str and epoch_id and epoch_id not in p.used_epochs,'optimizer epoch was missing or already consumed')
            due=[s for s in p.config.extraction_steps if s<state.joint_steps]
            require(p.installed and p.installed[-1]['step']>=max(due) and p.observed is not None,
                'next scheduled extraction is required before later rollout updates')
            p.used_epochs.add(epoch_id);sequence=p.call_count;p.call_count+=1
            before=native.training_fingerprint(state)
            folder=p.output/'calls'/f'{sequence:06d}';folder.mkdir(parents=True,exist_ok=False)
            exported=native.export_current_actor(state,folder/'current-NN.npz',pair_seed=p.pair_seed,
                variant=p.variant,candidate_id=native.admission.v.CANDIDATE)
            Path(exported['path']).chmod(0o444);actor_ref=pointer(exported['path'])
            package=targets.relabel_current_NN(self.active,'regularized',p.observed,actor_ref,stage_id=f'call-{sequence:06d}')
            materialized=targets.read_materialization(package,expected_owner=p.owner_ref)
            evidence={role:native.pools.build_observed_evidence(p.programs[role],p.observed,package,epoch_id=epoch_id)
                      for role in native.ROLES}
            passed={role:e.report['gate']['passed'] for role,e in evidence.items()}
            new_latch=p.latch or not all(passed.values())
            require(native.training_fingerprint(state)==before,'evidence provider changed NN/critic/optimizer/RNG state')
            event=p.append(dict(kind='validation',epoch_id=epoch_id,step=state.joint_steps,sequence_call=sequence,
                target_actor=actor_ref,target_package=package,programs=p.programs,
                reports={role:digest(e.report) for role,e in evidence.items()},passed_by_role=passed,
                asset_refs=[p.installed[-1]['binding'],*p.observed.refs,*[ref for e in evidence.values() for ref in e.asset_refs],
                    self.started,*materialized['inputs']],latch_before=p.latch,latch_after=new_latch,prior_failure=p.reason,
                validation_actions_are_counterfactual=True,accepted_gradient_updates='recorded_only_by_actual_PPO_update_core',optimizer_steps=0,
                long_storage_callback_started=self.started,long_materialization_provenance=materialized['provenance_ref']))
            if not p.latch and new_latch:p.reason=event
            p.latch=new_latch;result={}
            for role,e in evidence.items():
                args={**e.__dict__,'asset_refs':tuple([*e.asset_refs,event,p.installed[-1]['binding'],p.owner_ref]),
                    'provider_receipt_ref':event,'feedback_enabled':not p.latch,'envelope_digest':''}
                out=native.ProviderEvidence(**args)
                out=native.ProviderEvidence(**{**args,'envelope_digest':digest(out.payload())})
                result[role]=out.verify(current_role_sha256=native.role_hashes(state.model)[role],epoch_id=epoch_id)
                _put(folder/(role+'-evidence.json'),out.payload())
            (self.folder/'calls').mkdir(exist_ok=True)
            ref=_put(self.folder/'calls'/f'{sequence:06d}.json',dict(version=VERSION,sequence_call=sequence,
                source_sha256=digest(self.sources),previous=self.events[-1] if self.events else None,
                native_validation=event,target_package=package,provenance=materialized['provenance_ref'],
                native_NN_and_qualification_method_unchanged=True,optimizer_updates=0))
            self.events.append(ref);self.verify();return result
        except BaseException:
            self.failed=True;p.failed=True;raise

    def snapshot(self,state):
        self.verify();self.native.check_state(state)
        value=dict(version=VERSION,source_files=self.sources,owner=self.native.owner_ref,started=self.started,
            native_provider_started=self.native.started_ref,events=deepcopy(self.events),call_count=self.native.call_count,
            full_training_state_sha256=native.training_fingerprint(state),safe_snapshot_not_resume_authority=True)
        return dict(payload=value,payload_sha256=digest(value))
