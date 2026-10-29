"""Full-owner checkpoint envelope validation, without scientific resume authority.

This binds the real numerical bridge checkpoint to a physics-reader boundary,
eight private slot snapshots, reset journal/cursors, program pool/provider and
refits. It creates no Actor, environment, RNG or optimizer. The future reviewed
owner must additionally replay all physical assets and classify post-boundary
work before an explicit continuation; this module does not grant that right.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
from contextlib import contextmanager

from . import v643_calibration as numerical
from . import v643_training_partitions as partitions
from . import v643_calibration_owner as owner
from .v643_risk_recipe import CANDIDATE
from .v626_protocol import require, pointer, bound, digest
from backend.training import warehouse_v629_environment_replay as physics
from backend.training import warehouse_v629_rollout_replay as public_reader
from .v626_protocol import file_sha

VERSION='warehouse-v643-full-owner-boundary-envelope.v1'
PAIR_VERSION='warehouse-v643-clean-paired-resume-boundary.v1'


def _prefix(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'existing owner journal required')
    raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=sha256(raw).hexdigest())


def _prefix_bytes(ref):
    path=Path(ref['path'])
    require(path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and ref['bytes']>=0,
            'owner journal prefix is missing')
    with path.open('rb') as handle:raw=handle.read(ref['bytes'])
    require(len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['sha256'],'owner journal prefix changed or truncated')
    return raw


def _artifact_tree(folder):
    """Complete immutable owned outputs; journals get exact prefix checks."""
    roots=(Path(folder),owner.OWNER_ROOT/'program',owner.OWNER_ROOT/'final-programs',owner.OWNER_ROOT/'quality')
    skip={'owner-journal.jsonl','reset-predebits.jsonl','actual-resets.jsonl','failed.json'}
    rows=[]
    for root in roots:
        if not root.exists():continue
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink(),'symlink in owned recovery output')
            if not path.is_file():continue
            relative=path.relative_to(root)
            if root==Path(folder) and (relative.parts[0] in ('pair-checkpoints','resume-failures')
                or str(relative) in skip):continue
            require(not path.stat().st_mode&0o222,'unsealed artifact at clean recovery boundary')
            rows.append(dict(path=str(path.resolve()),sha256=file_sha(path)))
    return rows


def save_clean_pair(active):
    """Save a genuinely clean pair, before any final program/quality work."""
    active.live(full=True)
    from . import v643_training_seed_plan as seeds
    require(not active.pending and set(active.states)=={'raw','regularized'}
        and len({s.joint_steps for s in active.states.values()})==1 and not active.quality_started
        and active.final_assets_ref is active.final_programs_ref is None,'pair checkpoint requires a clean pre-finalization boundary')
    step=active.states['raw'].joint_steps
    require(step in seeds.optimizer_boundaries() and all(active.checkpoints[v]
        and bound(active.checkpoints[v][-1])['core']['PPO']==active.cores[v].snapshot()['PPO']
        for v in ('raw','regularized')),'both actual full checkpoints must match the current pair')
    path=active.folder/'pair-checkpoints'/f'{step:06d}'/'boundary.json'
    require(not path.exists(),'clean paired checkpoint already exists')
    active.event('paired_clean_checkpoint_predebit',joint_steps=step)
    value=dict(version=PAIR_VERSION,candidate_id=CANDIDATE,plan=active.plan.ref,partition=active.plan.partition.ref,
        claim=active.claim_ref,descriptor=active.descriptor_ref,owners=active.owners,source_files=active.sources,
        source_sha256=digest(active.sources),joint_steps=step,
        checkpoints={v:active.checkpoints[v][-1] for v in ('raw','regularized')},
        ledger=active.ledger.snapshot(),ledger_records=deepcopy(active.ledger.records),
        execution_journal=_prefix(active.folder/'owner-journal.jsonl'),execution_records=deepcopy(active.events),
        reset_journal=_prefix(active.folder/'actual-resets.jsonl'),reset_records=deepcopy(active.actual_resets),
        artifacts=_artifact_tree(active.folder),initial_identity=deepcopy(active.initial_identity),
        initial_rng=deepcopy(active.initial_rng),refit_refs=deepcopy(active.refit_refs),
        training_authority=False,resume_authority=False)
    path.parent.mkdir(parents=True,exist_ok=False)
    ref=owner._exclusive_json(path,value)
    active.pair_checkpoints.append(ref)
    active.event('paired_clean_checkpoint_saved',checkpoint=ref,joint_steps=step)
    return ref


def _execution_rows(raw):
    require(not raw or raw.endswith(b'\n'),'partial execution journal cannot resume')
    rows=[json.loads(line) for line in raw.splitlines()]
    for index,row in enumerate(rows):
        require(type(row) is dict and row['sequence']==index
            and row['previous_sha256']==(None if index==0 else digest(rows[index-1])),
            'execution journal chain differs')
    return rows


def _external_history_unchanged(plan):
    from . import v643_training_seed_plan as seeds
    cleared=seeds.checked_clearance(plan,execution=False)
    traces=cleared['value']['registered_traces']
    selected,weights=seeds.history_selection(plan,traces)
    owned=(owner.OWNER_ROOT/'program',owner.OWNER_ROOT/'final-programs',owner.OWNER_ROOT/'quality')
    def external(row):
        path=Path(row['path'])
        return path!=owner.claim_path() and not any(path.is_relative_to(root) for root in owned)
    require([r for r in selected if external(r)]==cleared['value']['selection']
        and [r for r in weights if external(r)]==cleared['value']['weight_paths'],
        'external scientific history changed since the original finite claim')
    from datetime import datetime,timezone
    require(datetime.now(timezone.utc)<datetime.fromisoformat(cleared['value']['expires_utc']),
            'original finite claim history clearance expired')
    return cleared


def validate_clean_pair(plan,ref):
    """Read-only clean-boundary proof; post-boundary work fails closed."""
    from . import v643_training_seed_plan as seeds
    from . import v643_calibration_pipeline as pipeline
    plan.verify();p=partitions._immutable(ref);folder=Path(plan.ref['path']).parent/'calibration'
    require(p['version']==PAIR_VERSION and p['candidate_id']==CANDIDATE and p['plan']==plan.ref
        and p['partition']==plan.partition.ref and p['source_files']==owner.source_files()==plan.value['source_files']
        and p['source_sha256']==plan.value['source_sha256'] and p['joint_steps'] in seeds.optimizer_boundaries()
        and Path(ref['path'])==folder/'pair-checkpoints'/f"{p['joint_steps']:06d}"/'boundary.json'
        and p['training_authority'] is p['resume_authority'] is False,'clean paired checkpoint identity/source differs')
    require(not (folder/'calibration-closed.json').exists() and not (folder/'completed.json').exists(),
            'completed or scientifically failed calibration cannot resume from an older checkpoint')
    require(Path(p['claim']['path'])==owner.claim_path() and pointer(owner.claim_path())==p['claim']
        and partitions._immutable(p['descriptor'])==owner.descriptor_document(plan),'original consumed claim/descriptor changed')
    require(_artifact_tree(folder)==p['artifacts'],'owned output changed after clean checkpoint; explicit reconciliation unavailable')
    before=_execution_rows(_prefix_bytes(p['execution_journal']))
    require(before==p['execution_records'],'saved execution records differ from journal prefix')
    runtime=[row['runtime'] for row in before if row['kind']=='runtime_bound']
    require(len(runtime)==1 and runtime[0]==pipeline.runtime_identity(),
            'original numerical runtime/thread identity differs before recovery allocation')
    current=_execution_rows(Path(p['execution_journal']['path']).read_bytes())
    tail=current[len(before):]
    require(len(tail)==1 and tail[0]['kind']=='paired_clean_checkpoint_saved'
        and tail[0]['checkpoint']==ref and tail[0]['joint_steps']==p['joint_steps'],
        'post-boundary initialization, reset, rollout, update or prior recovery attempt cannot be replayed')
    raw=_prefix_bytes(p['reset_journal'])
    require(Path(p['reset_journal']['path']).stat().st_size==p['reset_journal']['bytes']
        and [json.loads(line) for line in raw.splitlines()]==p['reset_records'],
        'post-boundary physical reset or changed hook journal')
    require(Path(p['ledger']['journal']['path']).stat().st_size==p['ledger']['journal_bytes'],
            'post-boundary reset predebit cannot be retried')
    validate_ledger_snapshot(p['ledger'],p['ledger_records'],plan=plan)
    clearance=_external_history_unchanged(plan);boundaries={}
    for variant in ('raw','regularized'):
        wrapper=partitions._immutable(p['checkpoints'][variant])
        require(wrapper['version']==pipeline.WRAPPER_VERSION and wrapper['candidate_id']==CANDIDATE
            and wrapper['owner']==p['owners'][variant] and wrapper['claim']==p['claim'], 'clean pair wrapper changed')
        full=read_boundary(wrapper['full_owner_boundary'],plan=plan)
        require(full['variant']==variant and full['reserved_joint_steps']==p['joint_steps']
            and full['updates']==wrapper['updates'],'clean pair physical/numerical boundaries differ')
        boundaries[variant]=full
    return p,boundaries,clearance,current


def restore_numerical_without_sampling(plan,full):
    """Meta allocation plus exact saved weights/RNG; no sampled initialization."""
    import torch
    import numpy as np
    from backend.training.warehouse_v6_actor_training import WarehouseV6Actor
    from backend.training.warehouse_v622_actor_training import TrainingState
    from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,MODEL_VERSION
    nref=full['numerical_boundary'];n=numerical.read_boundary(nref['path'],nref['sha256'])
    p=plan.partition.value;config=PPOConfig()
    require(torch.get_default_dtype()==torch.float32,'registered CPU float32 restoration required')
    before=torch.get_rng_state().clone()
    with torch.device('meta'):model=WarehouseV6Actor(model_version=MODEL_VERSION)
    model.to_empty(device='cpu');model.load_state_dict(n['model'],strict=True)
    # Meta -> CPU allocation can replace Parameter objects. Restore the same
    # three registered role hooks on the actual CPU tensors before gradients.
    for actor in model.actors.values():
        for parameter in (actor.tree.split.weight,actor.tree.split.bias,actor.tree.leaf_logits):
            if parameter._backward_hooks is not None:parameter._backward_hooks.clear()
        actor.tree.split.weight.register_hook(actor.tree._mask_guard_weight_gradient)
        actor.tree.split.bias.register_hook(actor.tree._mask_guard_bias_gradient)
        actor.tree.leaf_logits.register_hook(actor._core_leaf_logit_gradient)
    reference=FrozenWarehouseV6Actor.load(p['reference_actor']['path'],expected_sha256=p['reference_actor']['sha256'])
    with np.load(p['reference_actor']['path'],allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata_json'].item()))
    state=TrainingState(model,reference,str(Path(p['reference_actor']['path']).resolve()),p['reference_actor']['sha256'],
        p['initialization_seed'],torch.optim.Adam(model.actors.parameters(),lr=config.actor_learning_rate,eps=1e-5),
        torch.optim.Adam(model.reward_critic.parameters(),lr=config.critic_learning_rate,eps=1e-5),metadata)
    # Allocation only from the ORIGINAL registered argument. No value from
    # this initialization state may be drawn before full state replacement.
    bitgen=np.random.PCG64(p['optimizer_seed']);bitgen.state=deepcopy(n['core']['PPO']['rng_state'])
    require(bitgen.state==n['core']['PPO']['rng_state'],'restored optimizer RNG state differs before exposure')
    rng=np.random.Generator(bitgen)
    buffer=ReferenceTrainingBuffer.load(p['reference_buffer']['path'],p['reference_buffer']['sha256'],p['reference_actor']['sha256'])
    core=numerical.V643PPOUpdateCore(state,buffer,variant=full['variant'],rng=rng,config=config)
    numerical.restore_numerical_boundary(core,n)
    require(core.risk_training.rng is core.rng and core.rng.bit_generator.state==n['core']['PPO']['rng_state']
        and torch.equal(before,torch.get_rng_state()),'restoration sampled an initialization or broke shared optimizer/risk RNG')
    return core


def _restore_ledger(plan,snapshot,records):
    validate_ledger_snapshot(snapshot,records,plan=plan)
    out=owner.ResetLedger.__new__(owner.ResetLedger)
    out.plan=plan;out.failed=False;out.closed=False
    out.training=bound(plan.partition.value['train_case_inventory'])['cases']
    out.program=bound(plan.partition.value['program_evidence_inventory'])['cases']
    out.quality=bound(plan.partition.value['quality_case_inventory'])['cases']
    out.streams={slot:[case for case in out.training if case['slot']==slot] for slot in range(8)}
    out.cursors=deepcopy(snapshot['reset_cursors']);out.program_cursors=deepcopy(snapshot['program_cursors'])
    out.quality_cursors=deepcopy(snapshot['quality_cursors']);out.initialized=set(snapshot['initialized'])
    out.pending={};out.records=deepcopy(records);out.path=Path(snapshot['journal']['path']);out.handle=out.path.open('ab')
    return out


@contextmanager
def _append_reset_capture(path,records):
    from env.warehouse import reset_audit_v614 as audit
    require(audit._AUDIT_SINK.get() is None,'reset audit capture already active')
    with Path(path).open('ab') as handle:
        token=audit._AUDIT_SINK.set((records,handle))
        try:yield records
        finally:
            handle.flush();os.fsync(handle.fileno());audit._AUDIT_SINK.reset(token)


@contextmanager
def _resume_clean_pair(plan,ref):
    """Explicit once-per-checkpoint continuation under the SAME consumed claim.

    Holds an OS lock on the existing execution journal; creates no claim or
    lease file. The durable attempt event is appended BEFORE reconstruction.
    A nonclean boundary or previous attempt is rejected without an RNG draw.
    """
    import fcntl
    from datetime import datetime,timezone
    from . import v643_calibration_pipeline as pipeline
    from . import v643_program_namespace as program_ns
    from . import v643_program_provider as provider_code
    from . import v643_training_seed_plan as seeds
    require(pipeline.ACTIVE.get() is None,'cannot resume inside another finite owner')
    folder=Path(plan.ref['path']).parent/'calibration';journal_path=folder/'owner-journal.jsonl'
    require(journal_path.is_file() and not journal_path.is_symlink(),'existing claimed execution journal required')
    with journal_path.open('r+b') as journal:
        fcntl.flock(journal.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        p,boundaries,clearance,events=validate_clean_pair(plan,ref)
        active=pipeline.FiniteCalibrationOwner.__new__(pipeline.FiniteCalibrationOwner)
        active.plan=plan;active.sources=deepcopy(p['source_files']);active.clearance=clearance
        active.claim_ref=p['claim'];active.descriptor_ref=p['descriptor'];active.folder=folder
        active.config=partitions.PPOConfig();active.failed=False;active.closed=False;active.journal=journal
        active.events=events;active.owners=deepcopy(p['owners']);active.initial_identity=deepcopy(p['initial_identity'])
        active.initial_rng=deepcopy(p['initial_rng']);active.states={};active.cores={};active.readers={};active.pending={}
        active.providers={};active.pools={};active.namespaces={};active.environments={};active.active_cases={}
        active.collecting={};active.collecting_program=set();active.collecting_quality=False;active.quality_started=False
        active.reserved={v:p['joint_steps'] for v in seeds.VARIANTS};active.state_fingerprints={}
        active.updates={v:deepcopy(boundaries[v]['updates']) for v in seeds.VARIANTS}
        active.refits={v:deepcopy(boundaries[v]['refits']) for v in seeds.VARIANTS};active.refit_refs=deepcopy(p['refit_refs'])
        active.checkpoints={v:[p['checkpoints'][v]] for v in seeds.VARIANTS};active.pair_checkpoints=[ref]
        active.final_assets_ref=active.final_programs_ref=active.calibration_receipt=active.calibration_passed=None
        evidence=bound(clearance['value']['exact'])
        active.clearance_assets=[clearance['clearance'],clearance['value']['exact'],clearance['value']['registry'],
            evidence['exact_plan'],evidence['bounded_or_triaged_inventory']]
        if evidence.get('context_annotations'):active.clearance_assets.append(evidence['context_annotations'])
        active.ledger=None;token=pipeline.ACTIVE.set(active);journal.seek(0,os.SEEK_END)
        try:
            active.event('resume_attempt_predebit',checkpoint=ref,joint_steps=p['joint_steps'],
                         original_claim_retained=True,initialization_samples=0,automatic_retry=False)
            for variant in seeds.VARIANTS:
                active.event('restore_optimizer_rng_allocation_predebit',checkpoint=ref,variant=variant,
                    context='restore_optimizer_rng',seed=plan.partition.value['optimizer_seed'],
                    allocation_only=True,draws_before_state_overwrite=0)
                core=restore_numerical_without_sampling(plan,boundaries[variant]);active.cores[variant]=core
                state=core.state;active.states[variant]=state
                active.readers[variant]=physics.EnvironmentRolloutReplay.restore(boundaries[variant]['reader'],state,
                    owner_ref=active.owners[variant],inventory_ref=plan.partition.value['train_case_inventory'],config=active.config)
                active.environments[variant]=[None if snapshot is None else physics.restore_environment(snapshot)
                                               for snapshot in boundaries[variant]['environments']]
                active.active_cases[variant]=deepcopy(boundaries[variant]['active_cases'])
                active.state_fingerprints[variant]=public_reader.training_fingerprint(state)
                namespace=program_ns.ProgramNamespace.from_owner(active.owners[variant]);active.namespaces[variant]=namespace
                source=boundaries[variant]['program_source']
                if source is not None:
                    pool=program_ns.audit_observed_pool(namespace,source);active.pools[variant]=pool
                    if variant=='regularized':
                        active.providers[variant]=provider_code.CurrentNNEvidenceProvider.restore(boundaries[variant]['provider'],state,
                            namespace=namespace,observed=pool,config=active.config)
            active.ledger=_restore_ledger(plan,p['ledger'],p['ledger_records'])
            active.actual_resets=deepcopy(p['reset_records'])
            with _append_reset_capture(p['reset_journal']['path'],active.actual_resets):
                active.event('clean_pair_restored',checkpoint=ref,joint_steps=p['joint_steps'],new_environment_resets=0,
                             new_NN_initialization_samples=0,original_claim_retained=True)
                yield active
                active.live(full=True)
                require(active.training_complete() and active.final_assets_ref is not None,
                        'resumed pair incomplete; original claim remains consumed')
                owner._exclusive_json(folder/'completed.json',dict(version=pipeline.VERSION,candidate_id=CANDIDATE,
                    claim=p['claim'],resumed_from=ref,assets=active.final_assets_ref,training_completed=True,
                    calibration_measurement=active.calibration_receipt,calibration_precheck_passed=active.calibration_passed,
                    calibration_evaluation_still_required=active.calibration_passed is None,
                    candidate_closed_after_failed_calibration=active.calibration_passed is False,
                    model_acceptance_passed=False,pilot_opened=False))
        except BaseException as error:
            active.failed=True
            path=folder/'resume-failures'/f"{ref['sha256']}.json";path.parent.mkdir(exist_ok=True)
            if not path.exists():owner._exclusive_json(path,dict(version=PAIR_VERSION,candidate_id=CANDIDATE,claim=p['claim'],
                checkpoint=ref,error_type=type(error).__name__,automatic_retry=False,original_claim_retained=True))
            raise
        finally:
            if active.ledger is not None:active.ledger.close()
            active.closed=True;pipeline.ACTIVE.reset(token)


def _payload(snapshot, label):
    require(type(snapshot) is dict and set(snapshot)=={'payload','payload_sha256'}
        and type(snapshot['payload']) is dict and snapshot['payload_sha256']==digest(snapshot['payload']),
        label+' snapshot hash or schema differs')
    return snapshot['payload']


def validate_ledger_snapshot(snapshot, records, *, plan):
    """Replay data-only predebits/cursors; no supplied row proves its origin."""
    require(type(snapshot) is dict and snapshot.get('version')==owner.VERSION
        and snapshot.get('candidate_id')==CANDIDATE and snapshot.get('training_authority') is False
        and snapshot.get('resume_authority') is False and snapshot.get('unbound_predebits')==0
        and type(records) is list and type(snapshot.get('journal_records')) is int
        and len(records)==snapshot.get('journal_records'),
        'reset ledger boundary identity differs')
    plan.verify()
    require(snapshot['plan']==plan.ref,'journal plan differs')
    training=bound(plan.partition.value['train_case_inventory'])['cases']
    streams={slot:[c for c in training if c['slot']==slot] for slot in range(8)}
    program_cases=bound(plan.partition.value['program_evidence_inventory'])['cases']
    quality_cases=bound(plan.partition.value['quality_case_inventory'])['cases']
    previous=None;pending={};initialized=set()
    cursors={v:[0]*8 for v in ('raw','regularized')}
    program={v:0 for v in cursors};quality={v:0 for v in ('reference',*cursors)}
    journal=Path(snapshot['journal']['path']);size=snapshot.get('journal_bytes')
    require(type(size) is int and size>=0 and journal.is_file() and not journal.is_symlink()
        and journal.stat().st_size>=size,'reset journal prefix is missing or truncated')
    with journal.open('rb') as handle:prefix=handle.read(size)
    require(sha256(prefix).hexdigest()==snapshot['journal']['sha256']
        and prefix==b''.join((json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                            for r in records),'reset journal bytes differ from the saved complete prefix')
    for index,record in enumerate(records):
        require(type(record) is dict and set(record)=={'payload','sha256'},'journal record fields differ')
        p=record['payload']
        require(type(p) is dict and digest(p)==record['sha256']
            and p['version']==owner.JOURNAL_VERSION and p['candidate_id']==CANDIDATE
            and p['plan']==snapshot['plan'] and type(p['sequence']) is int and p['sequence']==index
            and p['previous_sha256']==previous,
            'reset journal sequence/hash/provenance differs')
        variant=p.get('variant')
        if p['kind']=='initialization_predebit':
            require(variant in cursors and variant not in initialized
                and type(p['initialization_seed']) is type(p['optimizer_seed']) is int
                and p['initialization_seed']==plan.partition.value['initialization_seed']
                and p['optimizer_seed']==plan.partition.value['optimizer_seed']
                and p['reference_actor_sha256']==plan.partition.value['reference_actor']['sha256'],
                'repeated, foreign or differently seeded initialization')
            initialized.add(variant)
        elif p['kind']=='environment_constructor_and_reset_predebit':
            scope=p['scope'];case=p['case']
            require(variant not in pending and scope in ('train','program_evidence','quality')
                and type(p['cursor']) is int,
                    'overlapping or unknown reset predebit')
            if scope=='train':
                slot=p['slot']
                require(variant in initialized and type(slot) is int and 0<=slot<8
                    and case['slot']==slot and p['cursor']==cursors[variant][slot]
                    and p['cursor']<len(streams[slot]) and case==streams[slot][p['cursor']],
                    'training reset cursor or exact finite case differs')
                cursors[variant][slot]+=1
            elif scope=='program_evidence':
                require(variant in initialized and p['cursor']==program[variant] and p['slot'] is None,
                        'program reset cursor differs')
                require(p['cursor']<len(program_cases) and case==program_cases[p['cursor']],
                        'program reset outside its finite case inventory')
                program[variant]+=1
            else:
                require(variant in quality and p['cursor']==quality[variant] and p['slot'] is None,
                        'quality reset cursor differs')
                require(p['cursor']<len(quality_cases) and case==quality_cases[p['cursor']],
                        'quality reset outside its finite case inventory')
                quality[variant]+=1
            pending[variant]=record
        else:
            require(p['kind']=='actual_reset_bound' and variant in pending
                and p['predebit_sha256']==pending[variant]['sha256']
                and p['scope']==pending[variant]['payload']['scope']
                and type(p['reset']['seed']) is int
                and p['reset']['seed']==pending[variant]['payload']['case']['scenario_seed']
                and type(p['reset']['episode_id']) is int and p['reset']['episode_id']==1,
                'actual reset does not match its unique finite predebit')
            partitions._immutable(p['initial_environment'])
            del pending[variant]
        previous=record['sha256']
    require(not pending and previous==snapshot['journal_head_sha256'] and sorted(initialized)==snapshot['initialized']
        and cursors==snapshot['reset_cursors'] and program==snapshot['program_cursors']
        and quality==snapshot['quality_cursors'], 'reset journal replay and saved finite cursors differ')
    return True


def validate_boundary(value, *, plan, numerical_value):
    """Cross-check complete stored components; no reconstruction or resume."""
    plan.verify()
    fields={'version','candidate_id','plan','partition','owner','claim','source_files','source_sha256',
        'variant','numerical_boundary','reader','environments','active_cases','reset_ledger','reset_journal_records',
        'program_source','provider','refits','updates','reserved_joint_steps','training_authority','resume_authority'}
    require(type(value) is dict and set(value)==fields and value['version']==VERSION and value['candidate_id']==CANDIDATE
        and value['plan']==plan.ref and value['partition']==plan.partition.ref
        and value['source_files']==plan.value['source_files']==owner.source_files()
        and value['source_sha256']==digest(value['source_files'])==plan.value['source_sha256']
        and value['variant'] in ('raw','regularized')
        and value['training_authority'] is value['resume_authority'] is False,
        'full v643 owner boundary identity/source/authority differs')
    claim=partitions._immutable(value['claim']);bound_owner=partitions._immutable(value['owner'])
    require(Path(value['claim']['path'])==owner.claim_path() and claim['schema']==owner.CLAIM_VERSION
        and claim['candidate_id']==CANDIDATE and claim['plan']==plan.ref
        and claim['partition']==plan.partition.ref and claim['source_sha256']==value['source_sha256']
        and claim['single_use_even_if_interrupted'] is True and claim['automatic_resume'] is False
        and claim['training_authority'] is False
        and Path(value['owner']['path'])==Path(plan.ref['path']).parent/'calibration'/value['variant']/'owner.json'
        and bound_owner['schema']==owner.OWNER_VERSION and bound_owner['candidate_id']==CANDIDATE
        and bound_owner['variant']==value['variant'] and bound_owner['claim']==value['claim']
        and bound_owner['plan']==plan.ref and bound_owner['source_sha256']==value['source_sha256']
        and bound_owner['source_files']==value['source_files']
        and bound_owner['pair_seed']==plan.partition.value['initialization_seed']
        and all(bound_owner[key]==plan.partition.value[key] for key in
            ('train_case_inventory','program_evidence_inventory','quality_case_inventory','program_templates')),
        'checkpoint has an old or differently bound owner/claim')
    descriptor_ref=claim['descriptor']
    require(bound_owner['descriptor']==descriptor_ref
        and Path(descriptor_ref['path'])==Path(plan.ref['path']).parent/'calibration/descriptor.json'
        and partitions._immutable(descriptor_ref)==owner.descriptor_document(plan),
        'full boundary descriptor does not bind the exact new source/data/start')
    n=numerical_value
    require(n['version']==numerical.BOUNDARY_VERSION and n['candidate_id']==CANDIDATE
        and n['core']['PPO']['variant']==value['variant']
        and n['reference_sha256']==plan.partition.value['reference_actor']['sha256']
        and n['reference_buffer_sha256']==plan.partition.value['reference_buffer']['sha256']
        and n['initialization_seed']==plan.partition.value['initialization_seed']
        and type(n['joint_steps']) is int and 0<=n['joint_steps']<=20000 and n['joint_steps']%8==0
        and type(value['reserved_joint_steps']) is int and n['joint_steps']==value['reserved_joint_steps'],
        'numerical full state and finite owner boundary differ')
    reader=_payload(value['reader'],'physics reader');public=_payload(reader['public'],'public reader')
    require(reader['version']==physics.VERSION and reader['source_files']==physics.source_files()
        and public['version']==public_reader.VERSION and public['source_sha256']==file_sha(public_reader.__file__)
        and public['config']==plan.partition.value['config']
        and reader['owner']==public['owner']==value['owner']
        and reader['inventory']==plan.partition.value['train_case_inventory']
        and public['full_training_state_sha256']==n['training_state_sha256']
        and public['processed_joint_steps']==n['joint_steps']
        and public['safe_assembly_boundary'] is True and public['authorizes_training'] is False
        and reader['training_authorized'] is False
        and len(reader['environments'])==len(reader['active'])==len(public['pending'])
            ==len(public['active'])==len(public['last_public'])==8
        and reader['environments']==value['environments'] and reader['active']==public['active']==value['active_cases'],
        'checkpoint omits or mismatches private slots and pending rollout histories')
    training=bound(plan.partition.value['train_case_inventory'])['cases']
    cases={c['episode_id']:c for c in training}
    for slot,(environment,case,pending) in enumerate(zip(value['environments'],value['active_cases'],public['pending'])):
        require((environment is None)==(case is None)==(not pending), 'private and pending slot liveness differs')
        if case is not None:
            require(cases.get(case['episode_id'])==case and case['slot']==slot
                and len(pending)<120 and [r['frame'] for r in pending]==list(range(len(pending)))
                and all(r['episode_id']==case['episode_id'] and r['slot']==slot
                        and not r['terminated'] and not r['truncated'] for r in pending),
                'pending private episode is not the exact finite TRAIN history')
            private=_payload(environment,'private environment')
            require(set(private)=={'schema','config','state','rng','episode_counter'}
                and private['schema']==physics.SNAPSHOT_VERSION and type(private['episode_counter']) is int
                and private['episode_counter']==1,'private slot is not a complete environment/RNG snapshot')
            config=physics.decode(private['config']);state=physics.decode(private['state']);rng=physics.decode(private['rng'])
            require(config==physics.training_config(case['scenario_seed']) and state.episode_id==1
                and state.frame==len(pending) and not(state.terminated or state.truncated)
                and state.participant_controlled_agent_id=='robot_1'
                and public['last_public'][slot]==digest(physics.public_pre_action_snapshot(state))
                and type(rng) is tuple and len(rng)==3 and rng[0]==3
                and type(rng[1]) is tuple and len(rng[1])==625
                and all(type(word) is int and 0<=word<2**32 for word in rng[1]),
                'private state, physical config, public history or complete environment RNG differs')
    ledger=value['reset_ledger']
    require(ledger['plan']==plan.ref and value['variant'] in ledger['initialized'],
            'checkpoint ledger belongs to another run or uninitialized variant')
    validate_ledger_snapshot(ledger,value['reset_journal_records'],plan=plan)
    for v,indices in ledger['reset_cursors'].items():
        require(len(indices)==8 and all(type(index) is int and 0<=index<=sum(c['slot']==slot for c in training)
            for slot,index in enumerate(indices)), 'checkpoint reset cursor exceeds its finite slot inventory')
    require(all(type(index) is int and 0<=index<=len(bound(plan.partition.value['program_evidence_inventory'])['cases'])
                for index in ledger['program_cursors'].values())
        and all(type(index) is int and 0<=index<=75 for index in ledger['quality_cursors'].values()),
        'checkpoint program/quality cursor exceeds the finite inventory')
    if value['variant']=='raw':
        require(not any(n['core']['PPO']['accepted_program_updates'].values()), 'raw feedback must be zero at every boundary')
    step=n['joint_steps'];scheduled=[s for s in (15000,16800,18600) if s<=step]
    require(value['refits']==scheduled and type(value['updates']) is list
        and len(value['updates'])==n['core']['PPO']['update_sequence'], 'checkpoint scheduled refits/update history differs')
    for ref in value['updates']:partitions._immutable(ref)
    if not scheduled:
        require(value['program_source'] is value['provider'] is None,'program data/provider appears before first extraction')
    else:
        partitions._immutable(value['program_source'])
        if value['variant']=='raw':
            require(value['provider'] is None and not any(n['core']['PPO']['accepted_program_updates'].values()),
                    'raw cannot restore external feedback/provider state')
        else:
            provider=_payload(value['provider'],'program provider')
            require(provider['owner']==value['owner'] and provider['full_training_state_sha256']==n['training_state_sha256']
                and provider['candidate_id']==CANDIDATE and provider['observed_source']==value['program_source']
                and provider['training_authority'] is provider['resume_authority'] is False,
                'program provider is not the exact current numerical boundary')
    return deepcopy(value)


def read_boundary(ref, *, plan):
    value=partitions._immutable(ref)
    numerical_ref=value['numerical_boundary']
    numerical_value=numerical.read_boundary(numerical_ref['path'],numerical_ref['sha256'])
    return validate_boundary(value,plan=plan,numerical_value=numerical_value)


def save_components(*, plan, owner_ref, claim_ref, variant, core, reader, ledger,
                    environments, active_cases, program_source, provider, refits, updates):
    """Save actual caller-owned components at one completed optimizer boundary.

    No components are constructed here. This is a full data boundary, not a
    continuation capability; physical replay and post-boundary reconciliation
    remain mandatory before any future explicit owner can resume it.
    """
    require(type(core) is numerical.V643PPOUpdateCore and type(reader) is physics.EnvironmentRolloutReplay
        and type(ledger) is owner.ResetLedger and variant==core.variant
        and not core.failed and core.processed_joint_steps==core.state.joint_steps,
        'full owner checkpoint requires actual matching numerical/reader/ledger components')
    plan.verify()
    step=core.state.joint_steps
    folder=Path(plan.ref['path']).parent/'calibration'/variant/'checkpoints'/f'{step:06d}'
    folder.mkdir(parents=True,exist_ok=False)
    numerical_ref=numerical.save_boundary(core,folder/'numerical.pt')
    value=dict(version=VERSION,candidate_id=CANDIDATE,plan=deepcopy(plan.ref),partition=deepcopy(plan.partition.ref),
        owner=deepcopy(owner_ref),claim=deepcopy(claim_ref),source_files=owner.source_files(),
        source_sha256=plan.value['source_sha256'],variant=variant,numerical_boundary=numerical_ref,
        reader=reader.snapshot(core.state),environments=deepcopy(environments),active_cases=deepcopy(active_cases),
        reset_ledger=ledger.snapshot(),reset_journal_records=deepcopy(ledger.records),program_source=deepcopy(program_source),
        provider=None if provider is None else provider.snapshot(core.state),refits=list(refits),updates=deepcopy(updates),
        reserved_joint_steps=step,training_authority=False,resume_authority=False)
    numerical_value=numerical.read_boundary(numerical_ref['path'],numerical_ref['sha256'])
    validate_boundary(value,plan=plan,numerical_value=numerical_value)
    return owner._exclusive_json(folder/'owner-state.json',value)


def resume(*,reference_directory,plan_ref,checkpoint_ref):
    """Public explicit continuation; never acquires or replaces a claim."""
    from . import v643_calibration_pipeline as pipeline
    return pipeline.resume(reference_directory=reference_directory,plan_ref=plan_ref,checkpoint_ref=checkpoint_ref)
