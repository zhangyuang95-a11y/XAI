"""Explicit numerical continuation with deduplicated verification.

The original 407 files and both earlier recovery attempts remain immutable.
This owner uses their exact PPO, physical replay and checkpoint carriers.  A
new claim and source contract identify the changed verification implementation;
no earlier execution is relabelled and no scientific threshold is changed.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,os,re
from time import perf_counter
import numpy as np

from . import v642_calibration_recovery as previous
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training.warehouse_v622_actor_training import load_checkpoint
from backend.training.warehouse_v629_ppo_update import PPOUpdateCore
from backend.training.warehouse_v629_environment_replay import EnvironmentRolloutReplay,restore_environment
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v625_actor_training import ReferenceTrainingBuffer
from env.warehouse.reset_audit_v614 import capture_stage1_resets
from scripts.run_warehouse_v625 import write_new

ROOT=previous.old.ROOT
VERSION='warehouse-v642-accelerated-calibration-recovery.v3'
SOURCE_VERSION='warehouse-v642-accelerated-recovery-source-contract.v1'
CLAIM_NAME='recovery-v3.claim.json'
AUTHORIZATION='那就解决这些重复计算的问题，在不降低验收标准的前提下加快速度，尽快完成验收，给我各项实测指标。'
FILES=('experiments/warehouse/v642_accelerated_recovery.py','scripts/resume_warehouse_v642_accelerated.py',
       'tests/test_warehouse_v642_accelerated_recovery.py','docs/warehouse_v642_accelerated_recovery.md')


def source_files():
    from backend.training import warehouse_v642_accelerated_recovery_evidence as evidence
    from . import v642_checkpoint_cutover as cutover
    files=previous.pipeline.source_files();files.update(previous.source_files());files.update(evidence.source_files())
    files.update(cutover.source_files())
    files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def readonly(path,value):
    write_new(path,value);Path(path).chmod(0o444);return pointer(path)


def prepare_sources(output):
    """Concrete reviewable source snapshot only; no run or RNG permission."""
    folder=Path(output).resolve();require(not folder.exists(),'accelerated source snapshot already exists')
    sources=source_files();folder.mkdir(parents=True);snapshots={}
    for name,sha in sources.items():
        path=folder/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==sha,'source changed before acceleration snapshot')
        with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        path.chmod(0o444);snapshots[name]=pointer(path)
    return readonly(folder/'source-contract.json',dict(version=SOURCE_VERSION,source_files=sources,
        source_sha256=digest(sources),source_snapshots=snapshots,training_authority=False,pilot_opened=False))


def checked_sources(ref):
    value=bound(ref);sources=source_files()
    require(value['version']==SOURCE_VERSION and value['source_files']==sources
        and value['source_sha256']==digest(sources) and set(value['source_snapshots'])==set(sources)
        and value['training_authority'] is value['pilot_opened'] is False,
        'accelerated source contract changed or grants another scope')
    require(not Path(ref['path']).stat().st_mode&0o222,'source contract must be immutable')
    for name,sha in sources.items():
        item=value['source_snapshots'][name]
        require(item['sha256']==sha and pointer(item['path'])==item
            and not Path(item['path']).is_symlink() and not Path(item['path']).stat().st_mode&0o222,
            'accelerated source snapshot missing, changed or writable')
    return value


def claim_path(original_plan_ref):return Path(original_plan_ref['path']).parent/CLAIM_NAME


def _cutover(ref):
    from . import v642_checkpoint_cutover as cutover
    return cutover.read_completed(ref)


def prepare(*,parent_plan_ref,cutover_ref,source_contract_ref,recovery_id,authorization):
    require(authorization==AUTHORIZATION,'explicit acceleration authorization required')
    contract=checked_sources(source_contract_ref)
    stopped=_cutover(cutover_ref)
    parent=previous.checked_plan(parent_plan_ref)
    require(stopped['parent_v2_plan']==parent_plan_ref and stopped['mode']=='original-safe-replay'
        and stopped['source_contract']==source_contract_ref,'cutover is not the registered unchanged numerical replay')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',recovery_id) is not None,'unsafe acceleration ID')
    base=Path(parent['original_plan']['path']).parent
    require(not claim_path(parent['original_plan']).exists(),'accelerated recovery claim already consumed')
    require(previous.checked_tree(parent['original_roots'])==parent['original_input_hashes'],
        'original numerical training inputs changed before acceleration')
    folder=base/'recovery'/recovery_id;folder.mkdir(parents=True,exist_ok=False)
    from backend.training import warehouse_v642_accelerated_recovery_evidence as evidence
    prefix_manifests=[]
    for kind,roots,sources,terminal in (
        ('original407',parent['original_roots'],parent['original_source_files'],None),
        ('recovery-v2',[str(Path(parent_plan_ref['path']).parent),
            str(previous.pipeline.PROVIDER_ROOT/'recovery'/parent['recovery_id']),
            stopped['parent_v2_claim']['path']],parent['source_files'],stopped['actual_supervisor_exit'])):
        if kind=='recovery-v2':
            targets=previous.old.observed.TARGET_ROOT/'recovery'/digest(stopped['parent_v2_claim'])
            if targets.exists():roots=[*roots,str(targets)]
        paths=evidence._tree_paths(roots)
        files={path:file_sha(path) for path in paths}
        for path in paths:require(not Path(path).stat().st_mode&0o222,'immutable prefix contains a writable file')
        if kind=='original407':require(files==parent['original_input_hashes'],'original prefix manifest changed')
        prefix_manifests.append(readonly(folder/(kind+'-prefix-manifest.json'),dict(
            version=evidence.MANIFEST_VERSION,kind=kind,roots=roots,files=files,source_files=sources,
            source_snapshots={name:contract['source_snapshots'][name] for name in sources},terminal_observation=terminal)))
    value=dict(deepcopy(parent),version=VERSION,recovery_id=recovery_id,source_files=contract['source_files'],
        source_contract=source_contract_ref,parent_recovery_plan=parent_plan_ref,cutover=cutover_ref,
        acceleration_authorization=authorization,checkpoint_serialization_schema=previous.VERSION,
        current_NN_qualification_recomputed_for_every_optimizer_update=True,
        every_distinct_asset_still_hashed=True,program_never_selects_environment_actions=True,
        raw_observed_pool_not_needed_for_remaining_training=True,
        prefix_manifest_refs=prefix_manifests,
        source_snapshots=contract['source_snapshots'],scientific_acceptance=False,pilot_opened=False)
    ref=readonly(folder/'plan.json',value);checked_plan(ref);return ref


def checked_plan(ref):
    p=bound(ref)
    require(p['version']==VERSION and p['source_files']==source_files()
        and p['acceleration_authorization']==AUTHORIZATION and p['authorization']==previous.AUTHORIZATION
        and p['checkpoint_serialization_schema']==previous.VERSION
        and p['maximum_joint_steps_per_variant']==20000 and p['automatic_retry'] is False
        and p['scientific_acceptance'] is p['pilot_opened'] is False,'accelerated recovery contract differs')
    contract=checked_sources(p['source_contract']);require(p['source_snapshots']==contract['source_snapshots'],'source snapshots differ')
    parent=previous.checked_plan(p['parent_recovery_plan'])
    unchanged=('original_plan','original_claim','original_descriptor','original_controller_contract',
        'original_source_files','original_calibration','original_roots','checkpoints','cached_rollouts',
        'original_updates','owners','keys','consumption','journals','reset_journals','expires_utc','original_input_hashes','config','repair_of')
    require(all(previous.exact_value_equal(p[name],parent[name]) for name in unchanged),
        'accelerated recovery changed the initial state, seed inventory, budget or finite deadline')
    stopped=_cutover(p['cutover'])
    require(stopped['parent_v2_plan']==p['parent_recovery_plan'] and stopped['source_contract']==p['source_contract']
        and stopped['mode']=='original-safe-replay','accelerated recovery cutover no longer verifies')
    require(len(p['prefix_manifest_refs'])==2 and all(pointer(item['path'])==item
        and not Path(item['path']).stat().st_mode&0o222 for item in p['prefix_manifest_refs']),
        'accelerated immutable prefix manifest changed')
    require(Path(ref['path'])==Path(p['original_plan']['path']).parent/'recovery'/p['recovery_id']/'plan.json',
        'accelerated recovery plan escaped its exact namespace')
    return p


class AcceleratedOwner(previous.RecoveryOwner):
    def _target_guard(self):
        self.live(full=True)
        return dict(owner=self.owners['regularized'],claim=self.recovery_claim_ref,
            source_files=source_files(),executor=VERSION)

    def live(self,*,full=False):
        require(previous.old.ACTIVE.get() is self and not self.failed and not self.closed,'accelerated owner is not active')
        require(pointer(self.recovery_claim_ref['path'])==self.recovery_claim_ref,'accelerated claim changed')
        require(datetime.now(timezone.utc)<datetime.fromisoformat(self.recovery_plan['expires_utc']),
                'original finite deadline expired; acceleration does not renew it')
        if full:
            require(checked_plan(self.recovery_plan_ref)==self.recovery_plan and self.sources==previous.old.source_files(),
                'accelerated source or original numerical owner changed')

    def update(self,variant,*,evidence_provider=None):
        from backend.training.warehouse_v642_accelerated_recovery_evidence import AcceleratedRecoveryEvidenceProvider
        self.live(full=True);require(variant in self.pending,'no replayed rollout to optimize')
        state=self.states[variant];pending=self.pending[variant]
        require(training_fingerprint(state)==pending['state_sha256'],'state changed since physical replay')
        provider=evidence_provider or self.providers.get(variant)
        require((variant=='raw' and provider is None) or (variant=='regularized'
            and type(provider) is AcceleratedRecoveryEvidenceProvider and provider.owner_ref==self.owners[variant]),
            'accelerated update requires its exact owned evidence provider')
        self.event('optimizer_update_predebit',variant=variant,trace=pending['trace'],step=state.joint_steps)
        result=self.cores[variant].update_batch(pending['batch'],batch_sha256=pending['trace']['sha256'],evidence_provider=provider)
        folder=self.folder/variant/'rollouts'/f'{pending["batch"]["start_joint_step"]:06d}';folder.mkdir(parents=True,exist_ok=True)
        ref=readonly(folder/'update.json',result)
        if pending.get('original_update') is not None:
            require(previous.exact_value_equal(result,bound(pending['original_update'])),
                    'accelerated replay differs from the original committed optimizer update')
            self.replay_proofs.append(dict(original_update=pending['original_update'],recomputed_update=ref,matched=True,
                exact_recursive_value_equality=True,new_environment_resets=0,replayed_physical_frames=len(pending['batch']['rows']),
                added_scientific_joint_steps=0))
        else:self.updates[variant].append(ref)
        del self.pending[variant];self.state_fingerprints[variant]=training_fingerprint(state)
        self.event('optimizer_update_accepted',variant=variant,receipt=ref,step=state.joint_steps,
                   replay_of_existing_commit=pending.get('original_update') is not None)
        return result


def restore_variant(active,variant):
    from backend.training.warehouse_v642_accelerated_recovery_evidence import AcceleratedRecoveryEvidenceProvider
    started=perf_counter()
    p=active.recovery_plan;snapshot=bound(p['checkpoints'][variant]);saved=snapshot['training_state']
    state=load_checkpoint(saved['path'],saved['sha256'])
    ref=active.plan.proposal.readiness.value['reference']['buffer']
    buffer=ReferenceTrainingBuffer.load(ref['path'],ref['sha256'],state.reference_sha256)
    previous.old.seeds.partitions.check_reference_buffer(active.plan.proposal,buffer)
    core=PPOUpdateCore.restore(state,buffer,snapshot['core'],variant=variant,
        rng=np.random.Generator(np.random.PCG64(0)),config=active.config)
    reader=EnvironmentRolloutReplay.restore(snapshot['reader'],state,owner_ref=active.owners[variant],
        inventory_ref=active.plan.proposal.value['train_case_inventory'],config=active.config)
    active.states[variant],active.cores[variant],active.readers[variant]=state,core,reader
    active.environments[variant]=[None if row is None else restore_environment(row) for row in snapshot['environments']]
    active.active_cases[variant]=deepcopy(snapshot['active_cases']);active.cursors[variant]=deepcopy(snapshot['reset_cursors'])
    active.evidence_cursor[variant]=snapshot['program_source_cursor'];active.reserved[variant]=snapshot['reserved_joint_steps']
    active.refits[variant]=list(snapshot['refits']);active.state_fingerprints[variant]=training_fingerprint(state)
    # Raw has no remaining extraction, no provider and no program loss. Its
    # 1650 old program episodes are audited later for final-program fitting.
    if variant=='regularized':
        print(json.dumps(dict(acceleration_progress='native_observed_pool_audit_started',variant=variant,
            step=state.joint_steps),sort_keys=True),flush=True)
        active.pools[variant]=previous.old.observed.audit_observed_pool(active.program_sources[variant],
            reference_directory=active.reference_directory)
        print(json.dumps(dict(acceleration_progress='native_observed_pool_audit_completed',variant=variant,
            elapsed_seconds=perf_counter()-started),sort_keys=True),flush=True)
        active.providers[variant]=AcceleratedRecoveryEvidenceProvider.restore_original_prefix(snapshot['provider'],state,
            reference_directory=active.reference_directory,owner_ref=active.owners[variant],observed=active.pools[variant],
            output=previous.pipeline.PROVIDER_ROOT/'recovery'/p['recovery_id'],
            recovery_claim_ref=active.recovery_claim_ref,prefix_manifest_refs=p['prefix_manifest_refs'],
            source_guard=active._target_guard,config=active.config)
    print(json.dumps(dict(acceleration_progress='restore_timing',variant=variant,
        elapsed_seconds=perf_counter()-started,raw_unused_program_pool_skipped=variant=='raw'),sort_keys=True),flush=True)
    active.event('full_training_checkpoint_restored',variant=variant,checkpoint=p['checkpoints'][variant],step=state.joint_steps,
        numerical_initialization_replaced_by_exact_saved_state=True,new_environment_resets=0)


@contextmanager
def claimed_recovery(*,recovery_plan_ref):
    p=checked_plan(recovery_plan_ref);old=previous.old
    require(old.ACTIVE.get() is None,'owners cannot nest')
    require(previous.checked_tree(p['original_roots'])==p['original_input_hashes'],'original recovery inputs changed')
    plan=old.load_seed_plan(p['reference_directory'],p['original_plan']);folder=Path(recovery_plan_ref['path']).parent
    claim=claim_path(p['original_plan']);require(not claim.exists(),'accelerated single-use claim already consumed')
    claim_ref=readonly(claim,dict(version=VERSION,recovery_plan=recovery_plan_ref,original_claim=p['original_claim'],
        authorization=p['authorization'],acceleration_authorization=p['acceleration_authorization'],owner_pid=os.getpid(),
        source_files=p['source_files'],source_contract=p['source_contract'],cutover=p['cutover'],
        repair_of=p['repair_of'],
        started_utc=datetime.now(timezone.utc).isoformat(),expires_utc=p['expires_utc'],
        single_use_even_if_interrupted=True,maximum_joint_steps_per_variant=20000,pilot_opened=False))
    active=AcceleratedOwner.__new__(AcceleratedOwner)
    active.recovery_plan_ref,active.recovery_plan=recovery_plan_ref,p
    active.recovery_claim_ref,active.claim_ref=claim_ref,p['original_claim'];active.descriptor_ref=p['original_descriptor']
    active.plan,active.sources,active.config=plan,old.source_files(),old.PPOConfig()
    active.folder,active.reference_directory=folder,p['reference_directory'];active.owners=deepcopy(p['owners'])
    active.failed=active.closed=False;active.sequence=len(previous.json_lines(p['journals'][0]['path']))
    for name in ('states','cores','readers','pending','pools','providers','state_fingerprints','collecting','reserved',
                 'cursors','evidence_cursor','environments','active_cases','refits'):setattr(active,name,{})
    active.collecting_source=set();active.updates=deepcopy(p['original_updates']);active.replay_proofs=[]
    original=Path(p['original_calibration'])
    active.program_sources={v:pointer(original/v/'program-source/source.json') for v in old.seeds.VARIANTS}
    active.checkpoints={v:[pointer(original/v/'checkpoints'/f'{step:06d}'/'owner-state.json')
        for step in (4000,15000,16800,18600) if step<=previous.STARTS[v]] for v in old.seeds.VARIANTS}
    active.scheduled_refits={v:[pointer(previous.pipeline.EXTRACTION_ROOT/p['keys'][v]/f'step-{step}'/'observed-source-binding.json')
        for step in (15000,16800,18600) if step<=previous.STARTS[v]] for v in old.seeds.VARIANTS}
    training=bound(plan.proposal.value['train_case_inventory'])['cases']
    active.train_streams={slot:[c for c in training if c['slot']==slot] for slot in range(8)}
    active.evidence_cases=bound(plan.proposal.value['program_evidence_inventory'])['cases'];token=old.ACTIVE.set(active)
    try:
        with (folder/'owner-journal.jsonl').open('x') as journal,capture_stage1_resets(folder/'actual-resets.jsonl') as actual:
            active.journal,active.actual_resets=journal,actual
            active.event('explicit_recovery_claimed',recovery_plan=recovery_plan_ref,original_claim=p['original_claim'],
                accelerated_executor=VERSION,source_contract=p['source_contract'],cutover=p['cutover'])
            yield active
            active.live(full=True)
            require(not active.pending and all(active.states[v].joint_steps==active.cores[v].processed_joint_steps==20000
                and active.refits[v]==[15000,16800,18600] and not active.cores[v].failed for v in old.seeds.VARIANTS),
                'accelerated pair is not numerically complete')
            require(active.cores['raw'].accepted_program_updates=={role:0 for role in previous.ROLES}
                and all(active.cores['regularized'].accepted_program_updates[role]>0 for role in previous.ROLES),
                'raw zero/reg both-role actual positive feedback requirement failed')
            require(previous.checked_tree(p['original_roots'])==p['original_input_hashes'],'original inputs changed during acceleration')
    except BaseException as error:
        active.failed=True
        readonly(folder/'failed.json',dict(version=VERSION,recovery_plan=recovery_plan_ref,recovery_claim=claim_ref,
            error_type=type(error).__name__,error=str(error),original_claim_retained=True,automatic_retry=False,pilot_opened=False))
        raise
    finally:
        active.closed=True;old.ACTIVE.reset(token)


def run(*,recovery_plan_ref):
    with claimed_recovery(recovery_plan_ref=recovery_plan_ref) as active:
        p=active.recovery_plan
        for variant in previous.old.seeds.VARIANTS:restore_variant(active,variant)
        for row in p['cached_rollouts']['regularized']:
            active.replay_cached_rollout('regularized',row);active.update('regularized')
        require(active.cursors==p['consumption']['training_cursors'],'restored cached reset cursors differ')
        previous.scheduled_refit(active,'regularized');active.checkpoint('regularized')
        while any(state.joint_steps<20000 for state in active.states.values()):
            for variant in previous.old.seeds.VARIANTS:
                if active.states[variant].joint_steps>=20000:continue
                active.collect_rollout(variant);active.update(variant)
                if active.states[variant].joint_steps==20000:active.checkpoint(variant)
        active.event('recovery_pair_steps_finished',per_variant_joint_steps={v:s.joint_steps for v,s in active.states.items()})
    folder=active.folder
    for name in ('owner-journal.jsonl','actual-resets.jsonl'):(folder/name).chmod(0o444)
    result=dict(version=VERSION,completed=True,numerical_training_completed=True,
        original_plan=p['original_plan'],original_claim=p['original_claim'],original_descriptor=p['original_descriptor'],
        original_controller_contract=p['original_controller_contract'],original_source_files=p['original_source_files'],
        recovery_plan=recovery_plan_ref,recovery_claim=active.recovery_claim_ref,source_files=p['source_files'],
        source_contract=p['source_contract'],cutover=p['cutover'],parent_recovery_plan=p['parent_recovery_plan'],repair_of=p['repair_of'],
        checkpoint_serialization_schema=previous.VERSION,event_serialization_schema=previous.VERSION,
        checkpoints=active.checkpoints,updates=active.updates,
        journals=[*p['journals'],pointer(folder/'owner-journal.jsonl')],
        reset_journals=[*p['reset_journals'],pointer(folder/'actual-resets.jsonl')],
        owners=active.owners,program_sources=active.program_sources,scheduled_refits=active.scheduled_refits,
        replay_proofs=active.replay_proofs,per_variant_joint_steps={v:s.joint_steps for v,s in active.states.items()},
        accepted_program_updates={v:dict(core.accepted_program_updates) for v,core in active.cores.items()},
        original_files_preserved=True,actual_original_process_exit_code_unknown=True,
        model_acceptance_passed=False,quality_and_program_calibration_evaluation_still_required=True,publishable=False,pilot_opened=False)
    ref=readonly(folder/'result.json',result);readonly(folder/'completed.json',dict(result_sha256=ref['sha256']));return ref
