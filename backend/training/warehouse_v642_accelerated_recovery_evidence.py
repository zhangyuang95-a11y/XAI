"""Explicit source-bound continuation; never modifies the frozen recovery.

All event/control checks remain native.  Duplicate immutable file references
are checked once per verification, with their complete actual bytes hashed.
No stat cache, private NN-target certificate or new evidence authority exists.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from . import warehouse_v642_recovery_evidence as original
from . import warehouse_v642_prepared_program as prepared
from . import warehouse_v642_static_metrics as static_metrics

native=original.native
pools=native.pools
require,pointer,bound,digest=original.require,original.pointer,original.bound,original.digest
write_new=original.write_new
_immutable=original._immutable
ROLES=original.ROLES
ROOT=Path(__file__).resolve().parents[2]
VERSION='warehouse-v642-accelerated-continuation-evidence.v1'
MANIFEST_VERSION='warehouse-v642-v3-immutable-prefix-manifest.v1'
FILES=('backend/training/warehouse_v642_accelerated_recovery_evidence.py',
       'tests/test_warehouse_v642_accelerated_recovery_evidence.py',
       'docs/warehouse_v642_accelerated_recovery_evidence.md')
FROZEN_RECOVERY_FILES=('experiments/warehouse/v642_calibration_recovery.py',
    'scripts/resume_warehouse_v642_calibration.py','tests/test_warehouse_v642_calibration_recovery.py',
    'backend/training/warehouse_v642_recovery_evidence.py','tests/test_warehouse_v642_recovery_evidence.py',
    'docs/warehouse_v642_calibration_recovery.md')


def source_files():
    from . import warehouse_v642_canonical_relabel as canonical
    result={}
    for source in (native.source_files(),prepared.source_files(),static_metrics.source_files(),canonical.source_files()):
        for name,sha in source.items():
            require(name not in result or result[name]==sha,'conflicting accelerated recovery source')
            result[name]=sha
    result.update({name:native.file_sha(ROOT/name) for name in (*FROZEN_RECOVERY_FILES,*FILES)})
    return dict(sorted(result.items()))


def _canonical_ref(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str
        and type(ref['sha256']) is str and len(ref['sha256'])==64
        and all(c in '0123456789abcdef' for c in ref['sha256']),'complete canonical asset reference required')
    path=Path(ref['path'])
    require(path.is_absolute() and path.resolve()==path
        and all(not p.is_symlink() for p in (path,*path.parents)),'asset path is noncanonical or symlinked')
    return ref


def unique_references(refs):
    """Reject conflicts before reading any payload; retain every distinct file."""
    unique={}
    for ref in refs:
        require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str,
            'complete canonical asset reference required')
        path=ref['path']
        require(path not in unique or unique[path]==ref,'same asset path has conflicting SHA bindings')
        if path not in unique:
            _canonical_ref(ref);unique[path]=deepcopy(ref)
    return tuple(unique[name] for name in sorted(unique))


def check_unique_assets(refs):
    unique=unique_references(refs)
    for ref in unique:original._immutable(ref)
    return unique


def event_asset_references(events):
    return [ref for event_ref in events for ref in bound(event_ref)['asset_refs']]


def checked_event_control(events,*,owner_ref,sources,config,extra_refs=()):
    """Native event/epoch/latch/evidence logic, complete de-duplicated bytes."""
    # Conflict rejection precedes the native logic and any repeated reads.
    assets=unique_references([*event_asset_references(events),*extra_refs])
    control=original._control(events,owner_ref=owner_ref,sources=sources,config=config,check_assets=False)
    for ref in assets:original._immutable(ref)
    return control


def _put(path,value):
    original.write_new(path,value);path.chmod(0o444);return pointer(path)


def _readonly(ref):
    _canonical_ref(ref);original._immutable(ref)
    require(not Path(ref['path']).stat().st_mode&0o222,'continuation prefix asset is writable')
    return ref


def _tree_paths(roots):
    paths=set()
    for name in roots:
        root=Path(name)
        require(root.is_absolute() and root.resolve()==root and root.exists()
            and not root.is_symlink(),'prefix root missing or noncanonical')
        items=[root] if root.is_file() else root.rglob('*')
        for path in items:
            require(not path.is_symlink(),'prefix tree contains a symlink')
            if path.is_file():paths.add(str(path))
    return tuple(sorted(paths))


def checked_prefix_manifests(refs):
    require(type(refs) in (tuple,list) and len(refs)==2,'both original and recovery-v2 prefix manifests required')
    files={};roots=[];kinds=set();manifests=[]
    for ref in refs:
        _readonly(ref);m=bound(ref)
        require(m['version']==MANIFEST_VERSION and m['kind'] in ('original407','recovery-v2')
            and m['kind'] not in kinds and type(m['files']) is dict and type(m['roots']) is list
            and m['roots'] and type(m['source_files']) is dict and type(m['source_snapshots']) is dict,
            'unknown or incomplete immutable prefix manifest')
        kinds.add(m['kind']);manifests.append(m)
        for name,sha in m['source_files'].items():
            require(native.file_sha(ROOT/name)==sha and m['source_snapshots'][name]['sha256']==sha,
                'frozen prefix semantic source changed')
            _readonly(m['source_snapshots'][name])
        for path,sha in m['files'].items():
            current=dict(path=path,sha256=sha);_canonical_ref(current)
            require(path not in files or files[path]==sha,'prefix manifests conflict on a file')
            files[path]=sha
        require(_tree_paths(m['roots'])==tuple(sorted(m['files'])),'prefix manifest omitted or added a tree file')
        roots.extend(m['roots'])
        terminal=m['terminal_observation']
        if m['kind']=='recovery-v2':
            require(terminal is not None,'v2 continuation requires its actual terminal observation')
            _readonly(terminal);t=bound(terminal)
            require(t['observed_by']=='source_bound_subprocess_wait' and type(t['exit_code']) is int
                and t['exit_code']!=0,'v2 was not actually stopped by its source-bound supervisor')
        elif terminal is not None:_readonly(terminal)
    require(kinds=={'original407','recovery-v2'},'ambiguous continuation prefix kinds')
    references=tuple(dict(path=p,sha256=files[p]) for p in sorted(files))
    for ref in references:_readonly(ref)
    return tuple(deepcopy(refs)),tuple(sorted(set(roots))),references,tuple(manifests)


class AcceleratedRecoveryEvidenceProvider(original.RecoveryEvidenceProvider):
    """New suffix of a real v2 checkpoint; no old update or scene is repeated."""
    @classmethod
    def restore_original_prefix(cls,snapshot,state,*,reference_directory,owner_ref,observed,output,
                                recovery_claim_ref,prefix_manifest_refs,config=native.PPOConfig(),source_guard=None):
        require(cls is AcceleratedRecoveryEvidenceProvider,'unknown accelerated recovery provider subclass')
        require(type(observed) is pools.ObservedPool,'genuine audited native observed pool required')
        manifests,roots,files,_=checked_prefix_manifests(prefix_manifest_refs)
        output=Path(output).resolve()
        require(all(not output.is_relative_to(Path(root)) for root in roots),
            'accelerated output cannot mutate an immutable prefix tree')
        # Invoke the actual frozen factory once.  No object/private certificate
        # is fabricated, no base class/global is replaced, and the original
        # output/claim/event chain remains untouched.
        base=original.RecoveryEvidenceProvider.restore_prefix(snapshot,state,
            reference_directory=reference_directory,owner_ref=owner_ref,observed=observed,
            output=output,recovery_claim_ref=recovery_claim_ref,config=config)
        out=object.__new__(cls);out.__dict__.update(base.__dict__)
        out.acceleration_sources=source_files();out.prefix_manifest_refs=manifests
        out.prefix_roots=roots;out.prefix_files=files;out._prior_snapshot_sha256=digest(snapshot)
        out._programs={};out._metric_caches={};out.acceleration_mode='original_prefix'
        require(source_guard is None or callable(source_guard),'canonical producer requires an explicit live source guard')
        out._source_guard=source_guard;out._pending_target_provenance={}
        inventory={r['path']:r['sha256'] for r in files}
        for ref in [owner_ref,*out.original_events,*out.original_files.values()]:
            require(inventory.get(ref['path'])==ref['sha256'],'original prefix input omitted from sealed manifests')
        out.acceleration_started_ref=_put(output/'acceleration-started.json',dict(version=VERSION,
            mode=out.acceleration_mode,source_files=out.acceleration_sources,
            frozen_recovery_source_files=out.recovery_sources,native_source_files=out.sources,
            recovery_claim=recovery_claim_ref,prior_snapshot_sha256=out._prior_snapshot_sha256,
            prior_recovery_started=out.recovery_started_ref,prior_audit_refs=[],prefix_manifests=list(manifests),
            prefix_roots=list(roots),prefix_files_sha256=digest(files),complete_native_prefix_events=out.original_events,
            prior_joint_steps=state.joint_steps,prior_call_count=out.call_count,
            duplicate_file_hashes_removed=True,every_distinct_file_fully_hashed_each_verify=True,
            static_preparation=prepared.VERSION,static_metrics=static_metrics.VERSION,
            canonical_producer_enabled=source_guard is not None,
            NN_or_optimizer_method_changed=False,training_authority=False))
        out.verify();return out

    @classmethod
    def restore_continuation(cls,snapshot,state,*,reference_directory,owner_ref,observed,output,
                             recovery_claim_ref,prefix_manifest_refs,config=native.PPOConfig(),source_guard=None):
        require(cls is AcceleratedRecoveryEvidenceProvider,'unknown accelerated recovery provider subclass')
        ready=native.admission.load_readiness(reference_directory);ready.verify()
        require(type(observed) is pools.ObservedPool,'genuine audited native observed pool required')
        observed.verify();p=snapshot['payload'];native_sources=native.source_files()
        require(snapshot['payload_sha256']==digest(p) and p['version']==original.VERSION
            and 'accelerated_wrapper' not in p and p['source_files']==original.source_files()
            and p['native_source_files']==native_sources and p['owner']==owner_ref
            and p['config']==asdict(config) and p['variant']=='regularized' and state.resumed_training_state
            and p['full_training_state_sha256']==native.training_fingerprint(state)
            and state.joint_steps==config.extraction_steps[-1] and p['training_authority'] is False,
            'continuation requires the exact final-refit v2 checkpoint and complete NN/optimizer state')
        require(p['observed_source']==observed.source_ref and p['observed_audit_sha256']==observed.audit_sha256,
            'continuation observed source/audit differs')
        manifests,roots,files,values=checked_prefix_manifests(prefix_manifest_refs)
        inventory={r['path']:r['sha256'] for r in files}
        for ref in [owner_ref,p['recovery_started'],p['recovery_claim'],*p['events'],*p['audit_refs']]:
            require(inventory.get(ref['path'])==ref['sha256'],'checkpoint prefix binding omitted from manifests')
        output=Path(output).resolve();previous_output=Path(p['output'])
        require(output.is_relative_to(native.PROVIDER_ROOT.resolve()/'recovery') and not output.exists()
            and previous_output.is_relative_to(native.PROVIDER_ROOT.resolve()/'recovery')
            and all(not output.is_relative_to(Path(root)) for root in roots),
            'continuation needs an independent new provider suffix directory')
        sources=source_files();_readonly(recovery_claim_ref)
        control=checked_event_control(p['events'],owner_ref=owner_ref,sources=native_sources,config=config)
        expected=dict(installed=p['installed'],used_epochs=set(p['used_epochs']),call_count=p['call_count'],
            latch=p['latch'],reason=p['reason'],programs=p['programs'])
        require(prepared._typed_sha(control)==prepared._typed_sha(expected)
            and p['installed'][-1]['step']==config.extraction_steps[-1],
            'checkpoint control/latch does not equal its immutable full event prefix')
        # A complete prefix cannot hide an uncommitted later event/call.
        suffix=[ref for ref in p['events'] if Path(ref['path']).is_relative_to(previous_output/'events')]
        require(sorted(str(path) for path in (previous_output/'events').glob('*.json'))==sorted(r['path'] for r in suffix),
            'v2 directory contains post-checkpoint provider events')
        expected_calls=[bound(ref)['recovery_validation_folder'] for ref in suffix if bound(ref)['kind']=='validation']
        require(sorted(str(path) for path in (previous_output/'calls').glob('*'))==sorted(expected_calls),
            'v2 directory contains uncommitted validation calls')
        out=object.__new__(cls)
        out.reference_directory=Path(reference_directory);out.ready=ready;out.owner_ref=deepcopy(owner_ref)
        out.output=output;out.original_output=previous_output;out.pair_seed=p['pair_seed'];out.variant=p['variant'];out.config=config
        out.sources=native_sources;out.recovery_sources=original.source_files();out.acceleration_sources=sources
        for key,value in control.items():setattr(out,key,deepcopy(value))
        out.events=deepcopy(p['events']);out.original_events=deepcopy(p['events']);out.original_event_count=len(p['events'])
        out.original_files={r['path']:r for r in files if Path(r['path']).is_relative_to(previous_output)}
        out.observed=observed;out.failed=False;out.partial=None;out.started_ref=deepcopy(bound(p['recovery_started'])['checkpoint']['payload']['started'])
        out.recovery_claim_ref=deepcopy(recovery_claim_ref);out.cached_call_range=(p['call_count'],p['call_count']-1)
        out.prefix_manifest_refs=manifests;out.prefix_roots=roots;out.prefix_files=files;out.audit_refs=[]
        out._programs={};out._metric_caches={};out._prior_snapshot_sha256=digest(snapshot)
        out.acceleration_mode='v2_continuation'
        require(source_guard is None or callable(source_guard),'canonical producer requires an explicit live source guard')
        out._source_guard=source_guard;out._pending_target_provenance={}
        output.mkdir(parents=True,exist_ok=False)
        out.acceleration_started_ref=_put(output/'acceleration-started.json',dict(version=VERSION,mode=out.acceleration_mode,
            source_files=sources,frozen_recovery_source_files=out.recovery_sources,native_source_files=native_sources,
            recovery_claim=recovery_claim_ref,prior_snapshot_sha256=out._prior_snapshot_sha256,
            prior_recovery_started=p['recovery_started'],prior_audit_refs=p['audit_refs'],prefix_manifests=list(manifests),
            prefix_roots=list(roots),prefix_files_sha256=digest(files),complete_native_prefix_events=p['events'],
            prior_joint_steps=state.joint_steps,prior_call_count=p['call_count'],
            duplicate_file_hashes_removed=True,every_distinct_file_fully_hashed_each_verify=True,
            static_preparation=prepared.VERSION,static_metrics=static_metrics.VERSION,
            canonical_producer_enabled=source_guard is not None,
            NN_or_optimizer_method_changed=False,training_authority=False))
        out.recovery_started_ref=_put(output/'recovery-started.json',dict(version=original.VERSION,checkpoint=snapshot,
            original_events=out.original_events,original_files=out.original_files,partial=None,
            cached_completed_call_range=list(out.cached_call_range),recovery_claim=recovery_claim_ref,
            source_files=out.recovery_sources,native_source_files=native_sources,
            original_assets_never_overwritten=True,replayed_validation_is_not_new_data=True,training_authority=False,
            acceleration_started=out.acceleration_started_ref))
        out.check_state(state);out.verify();return out

    def verify(self):
        self.ready.verify()
        require(not self.failed and self.sources==native.source_files()
            and self.recovery_sources==original.source_files() and self.acceleration_sources==source_files(),
            'failed accelerated provider or changed semantic sources')
        for ref in (self.owner_ref,self.recovery_claim_ref,self.recovery_started_ref,self.acceleration_started_ref,*self.prefix_manifest_refs):
            _readonly(ref)
        header=bound(self.acceleration_started_ref);registration=bound(self.recovery_started_ref)
        require(header['version']==VERSION and header['source_files']==self.acceleration_sources
            and header['mode']==self.acceleration_mode
            and header['canonical_producer_enabled']==(self._source_guard is not None)
            and header['native_source_files']==self.sources and header['frozen_recovery_source_files']==self.recovery_sources
            and header['recovery_claim']==self.recovery_claim_ref and header['prefix_manifests']==list(self.prefix_manifest_refs)
            and header['prefix_roots']==list(self.prefix_roots) and header['prefix_files_sha256']==digest(self.prefix_files)
            and header['complete_native_prefix_events']==self.original_events
            and header['prior_snapshot_sha256']==self._prior_snapshot_sha256
            and registration['version']==original.VERSION
            and registration['original_events']==self.original_events and registration['original_files']==self.original_files
            and registration['source_files']==self.recovery_sources and registration['native_source_files']==self.sources
            and registration['recovery_claim']==self.recovery_claim_ref and registration['partial']==self.partial
            and registration['cached_completed_call_range']==list(self.cached_call_range)
            and digest(registration['checkpoint'])==self._prior_snapshot_sha256,
            'accelerated continuation registration or immutable prefix binding changed')
        if self.acceleration_mode=='v2_continuation':
            require(registration['acceleration_started']==self.acceleration_started_ref and self.partial is None,
                'continuation acceleration attachment changed')
        else:
            require(self.acceleration_mode=='original_prefix' and header['prior_recovery_started']==self.recovery_started_ref,
                'unknown accelerated restoration mode')
        require(_tree_paths(self.prefix_roots)==tuple(r['path'] for r in self.prefix_files),
            'immutable prefix tree has added/missing files')
        n=min(len(self.events),self.original_event_count)
        require(self.original_event_count==len(self.original_events)
            and self.events[:n]==self.original_events[:n],'accelerated provider changed its committed prefix')
        new_events=self.events[self.original_event_count:]
        require(sorted((self.output/'events').glob('*.json'))==[Path(ref['path']) for ref in new_events],
            'accelerated provider has uncommitted/missing new events')
        validations=[bound(ref) for ref in new_events if bound(ref)['kind']=='validation']
        require(sorted((self.output/'calls').glob('*'))==[Path(e['recovery_validation_folder']) for e in validations],
            'accelerated provider has an incomplete validation directory')
        for event in validations:
            provenance=bound(event['target_package']).get('canonical_producer_provenance')
            require(event.get('canonical_producer_provenance')==provenance
                and (provenance is None or provenance in event['asset_refs']),
                'canonical producer provenance missing from native validation event assets')
        # Prefix assets are represented by an immutable full file inventory;
        # native _control still rechecks every historical event and envelope.
        if len(self.events)<self.original_event_count:
            original._control(self.original_events,owner_ref=self.owner_ref,sources=self.sources,
                config=self.config,check_assets=False)
        extra=[*self.prefix_files,*self.audit_refs]
        control=checked_event_control(self.events,owner_ref=self.owner_ref,sources=self.sources,config=self.config,extra_refs=extra)
        require(all(prepared._typed_sha(control[k])==prepared._typed_sha(getattr(self,k)) for k in control),
            'accelerated control differs from native immutable event history')
        if self.installed:
            event=bound(self.installed[-1]['event'])
            require(self.observed.source_ref==event['observed_source']
                and self.observed.audit_sha256==event['observed_audit_sha256'],'accelerated observed source changed')
        require(sorted((self.output/'replay-audit').glob('*.json'))==[Path(ref['path']) for ref in self.audit_refs],
            'accelerated audit suffix is missing, reordered or uncommitted')

    def _program(self,ref):
        encoded=bound(ref);key=(ref['path'],ref['sha256'],encoded['target_actor_role_sha256'])
        if key not in self._programs:
            self._programs[key]=prepared.PreparedProgram.from_reference(ref,target_actor_role_sha256=key[2])
        program=self._programs[key]
        require(type(program) is prepared.PreparedProgram and program._prepared.references[0]==key[:2],
            'accelerated program identity changed')
        return program.verify_binding()

    def _report(self,ref,parts):
        key=(ref['path'],ref['sha256'])
        if key not in self._metric_caches:self._metric_caches[key]=static_metrics.StaticMetricsCache(ref,parts)
        return self._metric_caches[key].current_report(parts)

    def _build_observed_evidence(self,program_ref,targets_ref,*,epoch_id):
        require(type(self.observed) is pools.ObservedPool,'genuine native observed pool required')
        targets,refs=pools.checked_targets(self.observed,targets_ref);cached=self.observed._replayed_target_cache
        require(type(cached) is pools.target_cache.ReplayedTargets,'native checked_targets private certificate required')
        cached.verify(self.observed);program=self._program(program_ref);role=program.role
        report=self._report(program_ref,cached.parts[role]);role_refs=[targets['pools'][role]];manifest=bound(targets['pools'][role])
        for split in pools.online.SPLITS:role_refs.extend([manifest['parts'][split]['arrays'],manifest['parts'][split]['rows']])
        base=pools.evidence.OnlineEvidence(program,report,deepcopy(program_ref),deepcopy(targets['target_actor']),
            cached.role_hashes[role],deepcopy(targets['pools'][role]),epoch_id,
            tuple(deepcopy([program_ref,targets['target_actor'],program.protection['parent'],*role_refs])), '')
        base=pools.evidence.OnlineEvidence(**{**base.__dict__,'envelope_digest':digest(base.payload())})
        base.verify(current_role_sha256=base.target_role_sha256,epoch_id=epoch_id)
        args={**base.__dict__,'asset_refs':tuple(deepcopy([*base.asset_refs,*refs])),
            'observed':self.observed,'target_package_ref':deepcopy(targets_ref),'envelope_digest':''}
        out=pools.ObservedOnlineEvidence(**args);out=pools.ObservedOnlineEvidence(**{**args,'envelope_digest':digest(out.payload())})
        return out.verify(current_role_sha256=out.target_role_sha256,epoch_id=epoch_id)

    def _target_package(self,actor_ref,*,epoch_id,sequence):
        output=pools.TARGET_ROOT/'recovery'/digest(self.recovery_claim_ref)/f'call-{sequence:06d}'
        if self._source_guard is None:return pools.relabel_current_NN(self.observed,actor_ref,output)
        from . import warehouse_v642_canonical_relabel as canonical
        def guard():
            value=self._source_guard()
            require(value['owner']==self.owner_ref and value['claim']==self.recovery_claim_ref
                and all(value['source_files'].get(k)==v for k,v in self.acceleration_sources.items())
                and self.acceleration_sources==source_files(),'canonical producer live owner/claim/source guard differs')
            return value
        result=canonical.relabel_current_NN(self.observed,actor_ref,output,epoch_id=epoch_id,
            sequence=sequence,source_guard=guard)
        provenance=canonical.read_provenance(self.observed,result,epoch_id=epoch_id,sequence=sequence,source_guard=guard)
        # The native package has a strict original asset inventory.  The new
        # source-bound provenance is bound by the new event, never hidden in
        # that original inventory or a forged ReplayedTargets certificate.
        ref=bound(result)['canonical_producer_provenance']
        _readonly(ref);self._pending_target_provenance[sequence]=deepcopy(ref)
        return result

    def __call__(self, state, epoch_id):
        started=perf_counter()
        try:
            self.verify(); self.check_state(state)
            require(native.coefficient(state.joint_steps, self.config.joint_step_budget, self.variant)>0,
                    'provider called before feedback window')
            require(type(epoch_id) is str and epoch_id and epoch_id not in self.used_epochs,
                    'optimizer epoch was missing or already consumed')
            due=[s for s in self.config.extraction_steps if s<state.joint_steps]
            require(self.installed and self.installed[-1]['step']>=max(due) and self.observed is not None,
                    'next scheduled extraction is required before later rollout updates')
            before=native.training_fingerprint(state); sequence=self.call_count
            old_ref=self.original_events[len(self.events)] if len(self.events)<self.original_event_count else None
            if old_ref is not None:
                event=bound(old_ref)
                require(event['kind']=='validation' and event['epoch_id']==epoch_id
                        and event['step']==state.joint_steps and event['sequence_call']==sequence
                        and event['programs']==self.programs and event['prior_failure']==self.reason
                        and event['latch_before'] is self.latch,
                        'recovery epoch, step or failure state differs from the next recorded call')
                actor_ref=event['target_actor']; targets=event['target_package']
                mode='completed_native_validation_replayed'
            elif self.partial is not None and self.partial['sequence_call']==sequence:
                actor_ref=self.partial['actor']; targets=self.partial['targets']
                mode='interrupted_native_targets_replayed'
            else:
                folder=self.output/'calls'/f'{sequence:06d}'; folder.mkdir(parents=True, exist_ok=False)
                exported=native.export_current_actor(state, folder/'current-NN.npz', pair_seed=self.pair_seed,
                            variant=self.variant, candidate_id=native.admission.v.CANDIDATE)
                Path(exported['path']).chmod(0o444); actor_ref=pointer(exported['path'])
                targets=self._target_package(actor_ref,epoch_id=epoch_id,sequence=sequence)
                mode='new_current_NN_validation'
            self._same_actor(state, actor_ref)
            # Completed native calls reuse only their committed target facts;
            # every incomplete/new call keeps the original full NN replay.
            # Both routes recompute qualification for each role/program.
            if old_ref is not None:
                evidence, replay_accounting=self._completed_native_evidence(state,old_ref,epoch_id)
                mode='completed_native_validation_rebuilt_from_committed_facts'
            else:
                evidence={role:self._build_observed_evidence(self.programs[role], targets,
                                                              epoch_id=epoch_id) for role in ROLES}
                replay_accounting=dict(completed_native_target_facts_reused=False,
                    original_full_scalar_NN_target_replay_required=True,original_memory_certificate_rule_unchanged=True)
            passed={role:e.report['gate']['passed'] for role,e in evidence.items()}
            new_latch=self.latch or not all(passed.values())
            require(native.training_fingerprint(state)==before, 'recovery validation changed training state')
            if old_ref is not None:
                require(event['reports']=={role:digest(e.report) for role,e in evidence.items()}
                        and event['passed_by_role']==passed and event['latch_after'] is new_latch,
                        'recomputed recovery report differs from the recorded validation')
                event_ref=old_ref; self.events.append(old_ref)
            else:
                folder=self.output/'calls'/f'{sequence:06d}'
                folder.mkdir(parents=True, exist_ok=True)
                event_ref=self.append(dict(kind='validation', epoch_id=epoch_id, step=state.joint_steps,
                    sequence_call=sequence, target_actor=actor_ref, target_package=targets, programs=self.programs,
                    reports={role:digest(e.report) for role,e in evidence.items()}, passed_by_role=passed,
                    asset_refs=[self.installed[-1]['binding'], *self.observed.refs,
                                *[ref for e in evidence.values() for ref in e.asset_refs],
                                *([self._pending_target_provenance[sequence]] if sequence in self._pending_target_provenance else [])],
                    latch_before=self.latch, latch_after=new_latch, prior_failure=self.reason,
                    validation_actions_are_counterfactual=True,
                    accepted_gradient_updates='recorded_only_by_actual_PPO_update_core', optimizer_steps=0,
                    recovery_validation_folder=str(folder), recovery_target_mode=mode,
                    **({'canonical_producer_provenance':self._pending_target_provenance[sequence]}
                       if sequence in self._pending_target_provenance else {})))
            if not self.latch and new_latch:
                self.reason=event_ref
            self.latch=new_latch; self.call_count+=1; self.used_epochs.add(epoch_id)
            result=self._wrap(evidence, event_ref, epoch_id, state)
            for role,e in result.items():
                if old_ref is not None:
                    old_folder=Path(old_ref['path']).parent.parent/'calls'/f'{sequence:06d}'
                    require(prepared._typed_sha(e.payload())==prepared._typed_sha(bound(pointer(old_folder/(role+'-evidence.json')))),
                            'recovery evidence envelope differs from the original accepted update')
                else:
                    path=folder/(role+'-evidence.json'); write_new(path,e.payload()); path.chmod(0o444)
            self.verify()
            self._audit(dict(sequence_call=sequence, epoch_id=epoch_id, mode=mode, event=event_ref,
                        target_actor=actor_ref, target_package=targets,
                        replay_accounting=replay_accounting,
                        revalidation_wall_seconds=perf_counter()-started,
                        report_sha256={role:digest(e.report) for role,e in evidence.items()}))
            return result
        except BaseException:
            self.failed=True
            raise

    def snapshot(self,state):
        value=super().snapshot(state)['payload']
        value['accelerated_wrapper']=dict(version=VERSION,source_files=self.acceleration_sources,
            acceleration_started=self.acceleration_started_ref,prefix_manifests=list(self.prefix_manifest_refs),
            every_distinct_bound_file_rehashed=True,no_stat_cache=True,current_NN_metrics_recomputed=True)
        return dict(payload=value,payload_sha256=digest(value))

    def _completed_native_evidence(self, state, event_ref, epoch_id):
        """Rebuild an already committed fact; never issue a target-memory cert.

        A completed native event plus BOTH original role evidence envelopes
        proves that its exact immutable target bytes passed the old scalar NN
        replay. Recovery separately rechecks all those bytes, observed inputs,
        recorded draws and qualification, tied to the identical current NN.
        This is deliberately not pools.checked_targets or _seal_after_replay:
        neither a fresh forward nor a fresh scientific audit is claimed.
        """
        event=bound(event_ref); sequence=event['sequence_call']
        require(self.cached_call_range[0]<=sequence<=self.cached_call_range[1]
                and event_ref in self.original_events and event['kind']=='validation'
                and event['epoch_id']==epoch_id, 'only a completed original validation may reuse target facts')
        observed=self.observed; observed.verify()
        target_ref=event['target_package']; p=bound(target_ref)
        require(p['schema']==pools.TARGET_VERSION and p['observed_source']==observed.source_ref
                and p['observed_audit_sha256']==observed.audit_sha256
                and p['observed_data_sha256']==observed.observed_data_sha256
                and p['owner']==observed.owner_ref and p['source_files']==pools.source_files()
                and p['target_actor']==event['target_actor'] and set(p['pools'])==set(ROLES)
                and p['current_policy_trajectory'] is p['actual_actions_replaced'] is p['training_authority'] is False
                and p['counterfactual_draws_not_executed'] is p['actual_PPO_feedback_counts_from_real_rollouts_only'] is True
                and p['current_policy_task_quality_estimate'] is None
                and p['fresh_randomness_claimed'] is p['pilot_opened'] is False
                and p['uses_existing_recorded_seeds_for_counterfactual_numerical_replay'] is True
                and all(type(p[k]) is int and p[k]==0 for k in ('environment_resets','NN_updates','PPO_joint_steps')),
                'completed native target scope, source or NN binding differs')
        completed=pointer(Path(target_ref['path']).with_name('completed.json'))
        require(bound(completed)==dict(result_sha256=target_ref['sha256']), 'completed native target seal changed')
        refs=[target_ref,completed,*p['asset_refs'],p['target_actor']]
        for ref in refs:
            _immutable(ref)
        splits=pools.online.SPLITS
        require(set(p['actual_source_actions'])==set(splits)
                and set(p['counterfactual_difference_counts'])==set(ROLES)
                and all(set(v)==set(splits) for v in p['counterfactual_difference_counts'].values()),
                'completed native target split inventory differs')
        expected_refs=[]; parts={}; role_refs={}; replayed_draws=0
        for split,ref in p['actual_source_actions'].items():
            expected_refs.append(ref)
            with np.load(ref['path'],allow_pickle=False) as z:
                require(z.files==['executed'] and np.array_equal(z['executed'],observed.parts[split]['executed']),
                        'completed native targets replaced source actual actions')
        episodes={}; scenes={}
        for i,role in enumerate(ROLES):
            manifest_ref=p['pools'][role]; manifest=bound(manifest_ref)
            require(set(manifest)=={'schema','scope','target_actor','role','parts'}
                    and manifest['schema']==pools.online.MANIFEST_VERSION
                    and manifest['scope']=='online_program_training_evidence'
                    and manifest['target_actor']==p['target_actor'] and manifest['role']==role
                    and set(manifest['parts'])==set(splits), 'completed native role manifest differs')
            role_refs[role]=[manifest_ref]; parts[role]={}
            for split in splits:
                source=manifest['parts'][split]
                require(set(source)=={'arrays','rows'}, 'completed native part inventory differs')
                role_refs[role].extend([source['arrays'],source['rows']])
                _immutable(source['arrays']); _immutable(source['rows'])
                with np.load(source['arrays']['path'],allow_pickle=False) as z:
                    require(set(z.files)=={'x','m','p','sampled'}, 'completed native target arrays differ')
                    a={key:z[key].copy() for key in z.files}
                a['rows']=bound(source['rows']); old=observed.parts[split]; n=len(a['x'])
                require(n>0 and a['x'].shape==(n,pools.online.FEATURE_DIM)
                        and a['m'].shape==a['p'].shape==(n,5) and a['sampled'].shape==(n,)
                        and a['sampled'].dtype==np.int64 and len(a['rows'])==n
                        and all(a[key].dtype==np.float32 and np.isfinite(a[key]).all() for key in ('x','m','p')),
                        'invalid completed native target arrays')
                pools.online.validate_rows(a['x'])
                require(np.array_equal(a['x'],old['x'][:,i]) and np.array_equal(a['m'],old['m'][:,i])
                        and a['rows']==old['rows'] and np.isin(a['m'],(0,1)).all()
                        and (a['m'][:,4]==1).all()
                        and np.array_equal(a['x'][:,pools.online.MASK_INDICES],a['m'])
                        and np.all(a['p']>=0) and np.all(a['p']<=1)
                        and np.all(a['p'][a['m']==0]==0)
                        and np.allclose(a['p'].sum(-1),1,rtol=0,atol=1e-6),
                        'completed native target public inputs, masks or distributions differ')
                seen=set(); frames={}
                for row,prob,action in zip(a['rows'],a['p'],a['sampled'],strict=True):
                    key=(row['episode'],row['frame'])
                    require(key not in seen, 'completed native target repeated a physical frame')
                    seen.add(key); frames.setdefault(row['episode'],[]).append(row['frame'])
                    seed=pools.independent_agent_seed(base_seed=row['action_seed'],episode_id=row['episode_id'],
                                                     frame=row['frame'],agent_id=role)
                    proposed=int(np.random.default_rng(seed).choice(5,p=prob))
                    require(int(action)==proposed and a['m'][len(seen)-1,proposed]==1,
                            'completed native sampled action differs from its recorded stream')
                    replayed_draws+=1
                require(all(sorted(v)==list(range(len(v))) for v in frames.values())
                        and p['counterfactual_difference_counts'][role][split]==
                        int(np.count_nonzero(a['sampled']!=old['executed'][:,i])),
                        'completed native episodes or recorded action differences changed')
                episodes[split]={row['episode'] for row in a['rows']}
                scenes[split]={row['base_seed'] for row in a['rows']}
                parts[role][split]=a
            require(all(not episodes[a]&episodes[b] and not scenes[a]&scenes[b]
                        for a in splits for b in splits if a<b), 'completed native complete-scene partitions overlap')
            expected_refs.extend(role_refs[role])
        require({digest(ref) for ref in expected_refs}=={digest(ref) for ref in p['asset_refs']},
                'completed native target asset manifest omitted consumed files')
        hashes=native.role_hashes(state.model); evidence={}
        for role in ROLES:
            program=self._program(self.programs[role])
            report=self._report(self.programs[role],parts[role])
            base=pools.evidence.OnlineEvidence(program,report,deepcopy(self.programs[role]),deepcopy(p['target_actor']),
                hashes[role],deepcopy(p['pools'][role]),epoch_id,
                tuple(deepcopy([self.programs[role],p['target_actor'],program.protection['parent'],*role_refs[role]])), '')
            base=pools.evidence.OnlineEvidence(**{**base.__dict__,'envelope_digest':digest(base.payload())})
            base.verify(current_role_sha256=hashes[role],epoch_id=epoch_id)
            args={**base.__dict__,'asset_refs':tuple(deepcopy([*base.asset_refs,*refs])),
                  'observed':observed,'target_package_ref':deepcopy(target_ref),'envelope_digest':''}
            out=pools.ObservedOnlineEvidence(**args)
            out=pools.ObservedOnlineEvidence(**{**args,'envelope_digest':digest(out.payload())})
            evidence[role]=out.verify(current_role_sha256=hashes[role],epoch_id=epoch_id)
        observed.verify()
        for ref in refs:
            _immutable(ref)
        return evidence,dict(completed_native_target_facts_reused=True,scalar_NN_forwards_recomputed=0,
                             scalar_NN_forwards_reused=replayed_draws,recorded_action_draws_replayed=replayed_draws,
                             original_memory_certificate_issued=False,new_scientific_evaluation=False)
