"""Read-only calibration checkpoint reconstruction and raw numerical replay.

Never opens a training owner, consumes a new scene, or issues continuation
authority. Replaying a stored raw batch updates detached in-memory tensors
only. In particular this cannot revive a failed candidate or write into an
original regularized provider's immutable event history.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from . import v629_calibration_owner as owner
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training.warehouse_v622_actor_training import load_checkpoint
from backend.training.warehouse_v629_environment_replay import EnvironmentRolloutReplay,capture_environment
from backend.training.warehouse_v629_ppo_update import PPOUpdateCore,ReferenceTrainingBuffer
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v629_evidence_provider import VERSION as PROVIDER_VERSION,source_files as provider_sources
from backend.training.warehouse_v625_actor_training import PPOConfig
from scripts.run_warehouse_v625 import plain

VERSION='warehouse-v629-calibration-checkpoint-readback.v1'


def source_files():
    files=owner.source_files();files['experiments/warehouse/v629_checkpoint_readback.py']=file_sha(__file__)
    return dict(sorted(files.items()))


def provider_prefix(snapshot,state,owner_ref):
    """Verify saved immutable prefix without activating or writing a provider.

    Unlike an actual continuation, forensic reading may inspect an old prefix
    even if the original run later wrote additional events. The returned
    latch is derived from that prefix; no executable feedback object escapes.
    """
    p=snapshot['payload'];require(snapshot['payload_sha256']==digest(p) and p['version']==PROVIDER_VERSION
        and p['source_files']==provider_sources() and p['owner']==owner_ref and p['variant']=='regularized'
        and p['config']==asdict(PPOConfig()) and p['pair_seed']==state.initialization_seed
        and p['training_authority'] is False and p['full_training_state_sha256']==training_fingerprint(state),
        'provider checkpoint source, owner or NN differs')
    header=bound(p['started'])
    require(header['version']==PROVIDER_VERSION and header['source_files']==p['source_files']
        and header['owner']==owner_ref and header['pair_seed']==p['pair_seed'] and header['variant']==p['variant']
        and header['config']==p['config'] and header['training_authority'] is False,
        'provider prefix header differs from saved configuration')
    events=[];installed=[];epochs=set();calls=0;latch=False;reason=None;programs=None;assets={}
    for sequence,ref in enumerate(p['events']):
        event=bound(ref)
        require(event['version']==PROVIDER_VERSION and event['training_authority'] is False
            and event['sequence']==sequence and event['previous']==(events[-1] if events else None)
            and event['owner']==owner_ref and event['source_sha256']==digest(p['source_files'])
            and event['latch_before']==latch and set(event['passed_by_role'])=={'robot_1','robot_2'}
            and all(type(v) is bool for v in event['passed_by_role'].values()),'saved provider event prefix differs')
        if event['kind']=='install':
            require(len(installed)<3 and event['step']==PPOConfig().extraction_steps[len(installed)],'unscheduled provider install in checkpoint')
            latch=not all(event['passed_by_role'].values());reason=ref if latch else None;programs=event['programs']
            installed.append(dict(step=event['step'],binding=event['binding'],event=ref))
        else:
            require(event['kind']=='validation' and event['epoch_id'] not in epochs and event['sequence_call']==calls
                and event['programs']==programs and event['prior_failure']==reason,'provider checkpoint epoch/program prefix differs')
            epochs.add(event['epoch_id']);calls+=1;disabled=latch or not all(event['passed_by_role'].values())
            if not latch and disabled:reason=ref
            latch=disabled
        require(event['latch_after']==latch,'saved provider reopened feedback without its next refit')
        for ref_asset in event['asset_refs']:
            require(ref_asset['path'] not in assets or assets[ref_asset['path']]==ref_asset,'conflicting provider prefix asset')
            assets[ref_asset['path']]=ref_asset
        events.append(ref)
    for ref in assets.values():require(pointer(ref['path'])==ref,'provider prefix asset changed or missing')
    require(installed==p['installed'] and sorted(epochs)==p['used_epochs'] and calls==p['call_count'] and latch==p['latch']
        and reason==p['reason'] and programs==p['programs'],'provider checkpoint latch was not derived from saved events')
    return dict(latch=latch,reason=reason,programs=programs,validation_calls=calls,installed_steps=[x['step'] for x in installed],
        executable_provider_returned=False,training_authority=False)


@dataclass
class CheckpointReadback:
    checkpoint_ref:dict
    value:dict
    state:object
    core:object
    reader:object
    proposal:object
    refs:tuple
    source_sha256:str
    provider:dict|None
    consumed:bool=False

    def verify(self):
        self.proposal.verify()
        require(self.source_sha256==digest(source_files()),'checkpoint readback source changed')
        for ref in self.refs:require(pointer(ref['path'])==ref,'checkpoint readback asset changed or missing')
        require(bound(self.checkpoint_ref)==self.value,'checkpoint readback payload changed')
        return self


def load(*,reference_directory,checkpoint_ref,expected_owner_ref):
    # The actual A/B gate precedes checkpoint reads or tensor/RNG construction.
    ready=owner.seeds.partitions.admission.load_readiness(reference_directory);ready.verify()
    value=bound(checkpoint_ref);owner_value=bound(expected_owner_ref);variant=owner_value['variant']
    require(value['version']==owner.VERSION and value['owner']==expected_owner_ref and value['source_files']==owner.source_files()
        and variant in owner.seeds.VARIANTS and value['automatic_resume'] is False and value['safe_snapshot_not_resume_authority'] is True,
        'checkpoint does not bind this current finite owner')
    claim=bound(value['claim']);descriptor_ref=claim['descriptor'];descriptor=bound(descriptor_ref)
    require(owner_value['claim']==value['claim'] and owner_value['descriptor']==descriptor_ref
        and claim['version']==descriptor['version']==owner.VERSION and descriptor['source_files']==owner.source_files()
        and descriptor['stage']=='paired_20k_calibration_not_formal_training','owner/claim/descriptor provenance differs')
    plan=owner.load_seed_plan(reference_directory,descriptor['seed_plan'])
    contract=owner.checked_sources(plan)
    require(descriptor['owner_source_contract']==contract
        and bound(descriptor['owner_source_clearance'])==dict(version=owner.VERSION,owner_source_contract=contract,
            seed_clearance=descriptor['clearance'],training_authority=False)
        and descriptor['source_sha256']==digest(owner.source_files()) and descriptor['config']==asdict(PPOConfig()),
        'checkpoint owner source registration or audited configuration differs')
    require(plan.proposal.readiness.payload_sha256==ready.payload_sha256 and descriptor['training_partition_proposal']==plan.proposal.ref
        and owner_value['train_case_inventory']==plan.proposal.value['train_case_inventory']
        and owner_value['program_evidence_inventory']==plan.proposal.value['program_evidence_inventory'],
        'checkpoint owner no longer belongs to confirmed data partitions')
    ref=ready.value['actor']
    state=load_checkpoint(value['training_state']['path'],value['training_state']['sha256'],reference_path=ref['path'],restore_rng=False)
    require(state.reference_sha256==ref['sha256'] and state.initialization_seed==plan.proposal.value['initialization_seed']
        and 0<=state.joint_steps<=20000,'checkpoint restored another reference, initialization or finite budget')
    buffer_ref=ready.value['reference']['buffer'];buffer=ReferenceTrainingBuffer.load(buffer_ref['path'],buffer_ref['sha256'],ref['sha256'])
    owner.seeds.partitions.check_reference_buffer(plan.proposal,buffer)
    # Zero is a fixed constructor control, overwritten by the saved PCG state
    # before any draw. It is not a fresh optimizer or scenario stream.
    rng=np.random.default_rng(0)
    core=PPOUpdateCore.restore(state,buffer,value['core'],variant=variant,rng=rng)
    reader=EnvironmentRolloutReplay.restore(value['reader'],state,owner_ref=expected_owner_ref,
        inventory_ref=plan.proposal.value['train_case_inventory'])
    require(value['reserved_joint_steps']==state.joint_steps==reader.public.processed_joint_steps
        and value['active_cases']==reader.active and len(value['environments'])==len(value['reset_cursors'])==8,
        'checkpoint physical, reserved or processed counters differ')
    cases=bound(plan.proposal.value['train_case_inventory'])['cases'];used=set()
    for slot in range(8):
        stream=[c for c in cases if c['slot']==slot];cursor=value['reset_cursors'][slot]
        require(type(cursor) is int and 0<=cursor<=len(stream),'invalid restored finite reset cursor')
        used.update(c['episode_id'] for c in stream[:cursor]);env=reader.environments[slot]
        require(value['environments'][slot]==(None if env is None else capture_environment(env)),
            'owner environment or RNG differs from physical replay checkpoint')
        if env is not None:require(cursor>0 and value['active_cases'][slot]==stream[cursor-1],
            'active restored episode is outside consumed slot prefix')
    refit_prefixes=([s for s in PPOConfig().extraction_steps if s<state.joint_steps],
        [s for s in PPOConfig().extraction_steps if s<=state.joint_steps])
    require(used==reader.used_episodes and type(value['refits']) is list and all(type(s) is int for s in value['refits'])
        and value['refits'] in refit_prefixes,
        'checkpoint reset consumption or scheduled refit prefix differs')
    refs=[checkpoint_ref,expected_owner_ref,value['claim'],descriptor_ref,descriptor['seed_plan'],value['training_state'],buffer_ref,
        contract,descriptor['owner_source_clearance'],descriptor['clearance'],
        *value['updates'],*reader.asset_refs]
    refs.extend([plan.ref,plan.proposal.ref,*plan.proposal.refs])
    for ref_asset in refs:require(pointer(ref_asset['path'])==ref_asset,'checkpoint dependency changed')
    require(len(value['updates'])==core.update_sequence and [bound(r)['batch_sha256'] for r in value['updates']]
        ==[r['sha256'] for r in reader.public.receipts],'checkpoint update receipts differ from replayed physical batches')
    saved_provider=None if value['provider'] is None else provider_prefix(value['provider'],state,expected_owner_ref)
    require(variant!='raw' or saved_provider is None,'raw checkpoint contains program feedback state')
    require(variant!='regularized' or (not value['refits'] if saved_provider is None
        else saved_provider['installed_steps']==value['refits']),
        'regularized checkpoint lost its scheduled provider or installed prefix')
    evidence_count=len(bound(plan.proposal.value['program_evidence_inventory'])['cases'])
    cursor=value['program_source_cursor']
    require(type(cursor) is int and cursor in (0,evidence_count)
        and (cursor==0 or state.joint_steps>=PPOConfig().extraction_steps[0])
        and (not value['refits'] or cursor==evidence_count),
        'checkpoint program source cursor is partial, premature or missing after refit')
    if cursor:
        source_ref=pointer(Path(expected_owner_ref['path']).parent/'program-source/source.json')
        pool=owner.observed.audit_observed_pool(source_ref,reference_directory=reference_directory)
        require(pool.owner_ref==expected_owner_ref,'checkpoint program source belongs to another owner')
        refs.extend(pool.refs)
    if value['provider'] is not None:
        refs.extend([value['provider']['payload']['started'],*value['provider']['payload']['events']])
        for event_ref in value['provider']['payload']['events']:refs.extend(bound(event_ref)['asset_refs'])
    return CheckpointReadback(checkpoint_ref,value,state,core,reader,plan.proposal,tuple(deepcopy(refs)),digest(source_files()),saved_provider).verify()


def replay_next_raw_update(restored,trace_ref,*,expected_update_ref,expected_checkpoint_ref):
    """Forensic equivalence check, no original outputs or fresh scenes written."""
    restored.verify();require(not restored.consumed and restored.core.variant=='raw',
        'only one detached raw update may be replayed; regularized provider history is read-only')
    restored.consumed=True;trace=bound(trace_ref);expected=bound(expected_checkpoint_ref)
    require(expected['owner']==restored.value['owner'] and trace['owner']==restored.value['owner']
        and trace['start_joint_step']==restored.state.joint_steps,'next trace or target checkpoint belongs to another owner/prefix')
    n=trace['end_joint_step']-trace['start_joint_step'];restored.state.advance_joint_steps(n)
    batch,audit=owner.seeds.partitions.assemble_training_batch(restored.proposal,restored.reader,trace_ref,restored.state)
    report=restored.core.update_batch(batch,batch_sha256=trace_ref['sha256'])
    require(plain(report)==bound(expected_update_ref),'restored numerical update differs from continuous execution')
    require(training_fingerprint(restored.state)==expected['reader']['payload']['public']['payload']['full_training_state_sha256']
        and plain(restored.core.snapshot())==expected['core'] and plain(restored.reader.snapshot(restored.state))==expected['reader'],
        'restored NN, Adam, beta, risk, RNG or pending physics differs after replay')
    restored.verify()
    return dict(version=VERSION,checkpoint=restored.checkpoint_ref,replayed_trace=trace_ref,expected_update=expected_update_ref,
        expected_checkpoint=expected_checkpoint_ref,physical_joint_frames=n,full_state_and_update_bitwise_equal=True,
        new_environment_resets=0,original_files_written=0,actual_candidate_updates=0,continuation_authority=False,pilot_opened=False)
