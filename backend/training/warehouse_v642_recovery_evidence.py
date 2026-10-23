"""Explicit, append-only recovery of interrupted v642 evidence validation.

This adapter changes no scientific gate or optimizer. Completed validations
are numerically rebuilt from their immutable targets and must reproduce the
old evidence envelope exactly. An incomplete validation may reuse only a
complete, replay-checked target package whose two Actor hashes match the
restored NN. Its missing event/evidence are written in a new recovery folder.
Recomputed validation frames are neither new samples nor training steps.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np

from . import warehouse_v642_evidence_provider as native
from .warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION

pools = native.pools
require, pointer, bound, digest = native.require, native.pointer, native.bound, native.digest
write_new = native.write_new
ROLES = native.ROLES
VERSION = 'warehouse-v642-explicit-evidence-recovery.v1'


def source_files():
    root = native.admission.v.ROOT
    names = ('backend/training/warehouse_v642_recovery_evidence.py',
             'tests/test_warehouse_v642_recovery_evidence.py')
    return {name: native.file_sha(root/name) for name in names}


def _immutable(ref):
    path = Path(ref['path'])
    # The recovery owner seals old assets read-only without changing bytes.
    # Content addressing, the original inventory and the event chain supply
    # this adapter's mutation checks; the owner checks the archival mode.
    require(pointer(path) == ref and not path.is_symlink(),
            'recovery source asset changed, missing or symlinked')


def _control(events, *, owner_ref, sources, config, check_assets=True):
    """Check the original chain without requiring its directory to end here."""
    installed, epochs, calls = [], set(), 0
    latch, reason, programs, previous = False, None, None, None
    for sequence, ref in enumerate(events):
        _immutable(ref)
        event = bound(ref)
        require(event['version'] == native.VERSION and event['sequence'] == sequence
                and event['previous'] == previous and event['owner'] == owner_ref
                and event['source_sha256'] == digest(sources) and event['latch_before'] is latch
                and set(event['passed_by_role']) == set(ROLES)
                and all(type(v) is bool for v in event['passed_by_role'].values()),
                'recovery provider event chain or fidelity status differs')
        if event['kind'] == 'install':
            require(len(installed) < len(config.extraction_steps)
                    and event['step'] == config.extraction_steps[len(installed)],
                    'recovery history contains an unscheduled extraction')
            latch = not all(event['passed_by_role'].values())
            reason = ref if latch else None
            programs = event['programs']
            installed.append(dict(step=event['step'], binding=event['binding'], event=ref))
        else:
            require(event['kind'] == 'validation' and event['epoch_id'] not in epochs
                    and event['sequence_call'] == calls and event['prior_failure'] == reason
                    and event['programs'] == programs,
                    'recovery repeated epoch, call, program or failure history')
            epochs.add(event['epoch_id']); calls += 1
            new_latch = latch or not all(event['passed_by_role'].values())
            if not latch and new_latch:
                reason = ref
            latch = new_latch
            folder = Path(event.get('recovery_validation_folder',
                                   str(Path(ref['path']).parent.parent/'calls'/f'{calls-1:06d}')))
            _immutable(event['target_actor'])
            for role in ROLES:
                evidence_ref = pointer(folder/(role+'-evidence.json'))
                _immutable(evidence_ref)
                e = bound(evidence_ref)
                require(e['version'] == native.EVIDENCE_VERSION and e['provider_receipt'] == ref
                        and e['epoch_id'] == event['epoch_id'] and e['current_target_actor'] == event['target_actor']
                        and e['anchor_program'] == programs[role] and digest(e['report']) == event['reports'][role]
                        and e['provider_feedback_enabled'] is (not latch),
                        'recovery stored evidence differs from its validation/latch event')
        require(event['latch_after'] is latch,
                'recovery history reopened feedback without a scheduled extraction')
        if check_assets:
            for asset in event['asset_refs']:
                _immutable(asset)
        previous = ref
    return dict(installed=installed, used_epochs=epochs, call_count=calls,
                latch=latch, reason=reason, programs=programs)


class RecoveryEvidenceProvider(native.CurrentNNEvidenceProvider):
    """Native ProviderEvidence and latch semantics, new durable recovery tail."""

    @classmethod
    def restore_prefix(cls, snapshot, state, *, reference_directory, owner_ref,
                       observed, output, recovery_claim_ref, config=native.PPOConfig()):
        ready = native.admission.load_readiness(reference_directory)
        ready.verify(); observed.verify()
        p = snapshot['payload']; sources = native.source_files()
        require(snapshot['payload_sha256'] == digest(p) and p['version'] == native.VERSION
                and p['source_files'] == sources and p['owner'] == owner_ref
                and p['config'] == asdict(config) and p['training_authority'] is False
                and state.resumed_training_state
                and p['full_training_state_sha256'] == native.training_fingerprint(state),
                'recovery requires the exact checkpoint TrainingState, owner and source')
        require(p['variant'] == 'regularized' and type(p['pair_seed']) is int
                and 0 <= p['pair_seed'] < 2**63 and pointer(owner_ref['path']) == owner_ref,
                'recovery requires a bound regularized owner and initialization')
        require(p['observed_source'] == observed.source_ref
                and p['observed_audit_sha256'] == observed.audit_sha256,
                'recovery observed states differ from the checkpoint')
        original = Path(p['output']).resolve()
        require(original.is_relative_to(native.PROVIDER_ROOT.resolve())
                and original != native.PROVIDER_ROOT.resolve(), 'invalid original provider output')
        header = bound(p['started']); _immutable(p['started'])
        require(header == dict(version=native.VERSION, source_files=sources, owner=owner_ref,
                              pair_seed=p['pair_seed'], variant=p['variant'], config=asdict(config),
                              readiness_sha256=ready.payload_sha256, training_authority=False),
                'original provider header changed')
        all_events = [pointer(path) for path in sorted((original/'events').glob('*.json'))]
        require(all_events[:len(p['events'])] == p['events']
                and [Path(ref['path']) for ref in all_events] ==
                [original/'events'/f'{i:06d}.json' for i in range(len(all_events))],
                'checkpoint is not an exact prefix of the original provider events')
        control = _control(p['events'], owner_ref=owner_ref, sources=sources, config=config)
        require(control == dict(installed=p['installed'], used_epochs=set(p['used_epochs']),
                                call_count=p['call_count'], latch=p['latch'], reason=p['reason'],
                                programs=p['programs']), 'checkpoint control differs from event-derived latch')
        full = _control(all_events, owner_ref=owner_ref, sources=sources, config=config)
        call_dirs = sorted((original/'calls').glob('*'))
        complete_dirs = [original/'calls'/f'{i:06d}' for i in range(full['call_count'])]
        require(call_dirs[:len(complete_dirs)] == complete_dirs
                and len(call_dirs) in (len(complete_dirs), len(complete_dirs)+1),
                'original provider has an ambiguous incomplete call suffix')
        partial = None
        if len(call_dirs) > len(complete_dirs):
            folder = original/'calls'/f'{full["call_count"]:06d}'
            require(call_dirs[-1] == folder and sorted(path.name for path in folder.iterdir()) == ['current-NN.npz'],
                    'incomplete validation contains unexpected or partial evidence')
            actor_ref = pointer(folder/'current-NN.npz'); _immutable(actor_ref)
            target_ref = pointer(pools.TARGET_ROOT/digest(owner_ref)/f'call-{full["call_count"]:06d}'/'result.json')
            _immutable(target_ref)
            completed = pointer(Path(target_ref['path']).with_name('completed.json')); _immutable(completed)
            require(bound(completed) == dict(result_sha256=target_ref['sha256'])
                    and bound(target_ref)['target_actor'] == actor_ref,
                    'incomplete call has no complete, matching target package')
            partial = dict(sequence_call=full['call_count'], actor=actor_ref, targets=target_ref)
        require(pointer(recovery_claim_ref['path']) == recovery_claim_ref, 'recovery claim changed')
        output = Path(output).resolve()
        require(output.is_relative_to(native.PROVIDER_ROOT.resolve()/'recovery')
                and output != native.PROVIDER_ROOT.resolve()/'recovery'
                and not output.exists(), 'recovery provider requires a new isolated recovery directory')
        out = cls.__new__(cls)
        out.reference_directory=Path(reference_directory); out.ready=ready; out.owner_ref=deepcopy(owner_ref)
        out.output=output; out.original_output=original; out.pair_seed=p['pair_seed']; out.variant=p['variant']
        out.config=config; out.sources=sources; out.events=deepcopy(p['events'])
        for key, value in control.items():
            setattr(out, key, deepcopy(value))
        out.observed=observed; out.failed=False; out.started_ref=deepcopy(p['started'])
        out.original_events=deepcopy(all_events); out.original_event_count=len(all_events)
        out.partial=partial; out.recovery_claim_ref=deepcopy(recovery_claim_ref)
        out.cached_call_range=(p['call_count'],full['call_count']-1)
        out.recovery_sources=source_files(); out.audit_refs=[]
        # Freeze all old files, including the interrupted current-NN export.
        out.original_files={str(path):pointer(path) for path in sorted(original.rglob('*')) if path.is_file()}
        output.mkdir(parents=True, exist_ok=False)
        write_new(output/'recovery-started.json', dict(version=VERSION, checkpoint=snapshot,
            original_events=all_events, original_files=out.original_files, partial=partial,
            cached_completed_call_range=list(out.cached_call_range),
            recovery_claim=recovery_claim_ref, source_files=out.recovery_sources,
            native_source_files=sources, original_assets_never_overwritten=True,
            replayed_validation_is_not_new_data=True, training_authority=False))
        (output/'recovery-started.json').chmod(0o444)
        out.recovery_started_ref=pointer(output/'recovery-started.json')
        out.check_state(state); out.verify()
        return out

    def verify(self):
        self.ready.verify()
        require(not self.failed and self.sources == native.source_files()
                and self.recovery_sources == source_files()
                and pointer(self.owner_ref['path']) == self.owner_ref
                and pointer(self.recovery_claim_ref['path']) == self.recovery_claim_ref
                and pointer(self.recovery_started_ref['path']) == self.recovery_started_ref,
                'failed recovery provider or changed code, owner, claim or header')
        header=bound(self.recovery_started_ref)
        require(header['version']==VERSION and header['original_events']==self.original_events
                and header['original_files']==self.original_files and header['partial']==self.partial
                and header['recovery_claim']==self.recovery_claim_ref
                and header['source_files']==self.recovery_sources
                and header['native_source_files']==self.sources
                and header['cached_completed_call_range']==list(self.cached_call_range)
                and self.original_event_count==len(self.original_events),
                'recovery registration, immutable old-file inventory or source binding changed')
        require({str(path):pointer(path) for path in sorted(self.original_output.rglob('*')) if path.is_file()}
                == self.original_files, 'original provider directory changed after recovery registration')
        require(self.events[:min(len(self.events), self.original_event_count)]
                == self.original_events[:min(len(self.events), self.original_event_count)],
                'recovery did not consume the exact original event prefix')
        _control(self.original_events, owner_ref=self.owner_ref, sources=self.sources, config=self.config)
        control = _control(self.events, owner_ref=self.owner_ref, sources=self.sources, config=self.config)
        require(all(control[key] == getattr(self, key) for key in control),
                'recovery control state differs from immutable event history')
        new_paths=sorted((self.output/'events').glob('*.json')) if (self.output/'events').exists() else []
        require(new_paths == [Path(ref['path']) for ref in self.events[self.original_event_count:]],
                'recovery has unconsumed, missing or incomplete new events')
        calls=[bound(ref) for ref in self.events[self.original_event_count:] if bound(ref)['kind']=='validation']
        require(sorted((self.output/'calls').glob('*')) ==
                [Path(event['recovery_validation_folder']) for event in calls],
                'recovery has an incomplete or unexpected validation call')
        if self.installed:
            event=bound(self.installed[-1]['event'])
            require(self.observed.source_ref == event['observed_source']
                    and self.observed.audit_sha256 == event['observed_audit_sha256'],
                    'active recovery observed pool changed')
        for ref in self.audit_refs:
            _immutable(ref)
        require(sorted((self.output/'replay-audit').glob('*.json'))==
                [Path(ref['path']) for ref in self.audit_refs],
                'recovery revalidation audit was omitted, reordered or incomplete')

    def append(self, value):
        require(len(self.events) >= self.original_event_count,
                'old scheduled provider events must be replayed before creating new events')
        value=dict(version=native.VERSION, owner=self.owner_ref, sequence=len(self.events),
                   previous=self.events[-1] if self.events else None, source_sha256=digest(self.sources),
                   training_authority=False, recovery_started=self.recovery_started_ref, **value)
        path=self.output/'events'/f'{len(self.events):06d}.json'; path.parent.mkdir(exist_ok=True)
        write_new(path, value); path.chmod(0o444)
        ref=pointer(path); self.events.append(ref)
        return ref

    def _same_actor(self, state, actor_ref):
        _immutable(actor_ref)
        actor=FrozenWarehouseV6Actor.load(actor_ref['path'], expected_sha256=actor_ref['sha256'])
        require(actor.model_version == MODEL_VERSION and actor.role_sha256 == native.role_hashes(state.model),
                'recovery current NN differs from the cached dual-Actor export')

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
            encoded=bound(self.programs[role])
            program=pools.evidence.Program.from_dict(encoded,encoded['target_actor_role_sha256'])
            report=pools.online.current_report(program,parts[role])
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

    def _wrap(self, evidence, event_ref, epoch_id, state):
        result={}
        for role, e in evidence.items():
            args={**e.__dict__, 'asset_refs':tuple([*e.asset_refs, event_ref,
                  self.installed[-1]['binding'], self.owner_ref]),
                  'provider_receipt_ref':event_ref, 'feedback_enabled':not self.latch, 'envelope_digest':''}
            out=native.ProviderEvidence(**args)
            out=native.ProviderEvidence(**{**args, 'envelope_digest':digest(out.payload())})
            result[role]=out.verify(current_role_sha256=native.role_hashes(state.model)[role], epoch_id=epoch_id)
        return result

    def _audit(self, value):
        path=self.output/'replay-audit'/f'{len(self.audit_refs):06d}.json'; path.parent.mkdir(exist_ok=True)
        write_new(path, dict(version=VERSION, recovery_started=self.recovery_started_ref,
                  replay_is_not_new_or_independent_data=True, environment_resets=0,
                  NN_updates=0, PPO_joint_steps=0, training_authority=False, **value))
        path.chmod(0o444); self.audit_refs.append(pointer(path))

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
                targets=pools.relabel_current_NN(self.observed, actor_ref,
                    pools.TARGET_ROOT/'recovery'/digest(self.recovery_claim_ref)/f'call-{sequence:06d}')
                mode='new_current_NN_validation'
            self._same_actor(state, actor_ref)
            # Completed native calls reuse only their committed target facts;
            # every incomplete/new call keeps the original full NN replay.
            # Both routes recompute qualification for each role/program.
            if old_ref is not None:
                evidence, replay_accounting=self._completed_native_evidence(state,old_ref,epoch_id)
                mode='completed_native_validation_rebuilt_from_committed_facts'
            else:
                evidence={role:pools.build_observed_evidence(self.programs[role], self.observed, targets,
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
                                *[ref for e in evidence.values() for ref in e.asset_refs]],
                    latch_before=self.latch, latch_after=new_latch, prior_failure=self.reason,
                    validation_actions_are_counterfactual=True,
                    accepted_gradient_updates='recorded_only_by_actual_PPO_update_core', optimizer_steps=0,
                    recovery_validation_folder=str(folder), recovery_target_mode=mode))
            if not self.latch and new_latch:
                self.reason=event_ref
            self.latch=new_latch; self.call_count+=1; self.used_epochs.add(epoch_id)
            result=self._wrap(evidence, event_ref, epoch_id, state)
            for role,e in result.items():
                if old_ref is not None:
                    old_folder=Path(old_ref['path']).parent.parent/'calls'/f'{sequence:06d}'
                    require(e.payload()==bound(pointer(old_folder/(role+'-evidence.json'))),
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

    def snapshot(self, state):
        self.verify(); self.check_state(state)
        value=dict(version=VERSION, source_files=self.recovery_sources, native_source_files=self.sources,
            owner=self.owner_ref, recovery_started=self.recovery_started_ref, recovery_claim=self.recovery_claim_ref,
            full_training_state_sha256=native.training_fingerprint(state), output=str(self.output),
            events=deepcopy(self.events), installed=deepcopy(self.installed), used_epochs=sorted(self.used_epochs),
            call_count=self.call_count, latch=self.latch, reason=deepcopy(self.reason),
            programs=deepcopy(self.programs), observed_source=self.observed.source_ref,
            observed_audit_sha256=self.observed.audit_sha256, audit_refs=deepcopy(self.audit_refs),
            pair_seed=self.pair_seed, variant=self.variant, config=asdict(self.config), training_authority=False)
        return dict(payload=value, payload_sha256=digest(value))
