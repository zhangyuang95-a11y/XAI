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

from . import v643_calibration as numerical
from . import v643_training_partitions as partitions
from . import v643_calibration_owner as owner
from .v643_risk_recipe import CANDIDATE
from .v626_protocol import require, pointer, bound, digest
from backend.training import warehouse_v629_environment_replay as physics
from backend.training import warehouse_v629_rollout_replay as public_reader
from .v626_protocol import file_sha

VERSION='warehouse-v643-full-owner-boundary-envelope.v1'


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


def resume(*_args, **_kwargs):
    raise RuntimeError('v643 explicit physical collector/recovery integration is unavailable; no reconstruction or resume')
