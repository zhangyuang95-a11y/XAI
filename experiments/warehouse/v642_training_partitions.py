"""Finite data partition proposals after A/B, never training authority.

The future owner must additionally audit seed history, issue a single-use
capability, and journal actual resets. This layer binds shared raw/reg scene
inventories and excludes program evidence from all NN gradient buffers.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
from collections import Counter
import numpy as np
from . import v642_training_admission as admission
from .v626_protocol import require,pointer,bound,digest,file_sha
from .v628_data import cases as spent_development_cases
from backend.training import warehouse_v642_observed_pool as observed
from backend.training import warehouse_v642_evidence_provider as provider
from backend.training.warehouse_v629_environment_replay import EnvironmentRolloutReplay
from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-training-partition-proposal.v1'
TRAIN_VERSION='warehouse-v629-finite-training-case-inventory.v1'
OUTPUT_ROOT=admission.v.ROOT/'output/warehouse_v642/training-proposals'


def source_files():
    files=provider.source_files()
    files['experiments/warehouse/v642_training_partitions.py']=file_sha(__file__)
    for name in ('tests/test_warehouse_v642_training_partitions.py','docs/warehouse_v642_finite_training_plans.md'):
        files[name]=file_sha(admission.v.ROOT/name)
    return dict(sorted(files.items()))


def seed(value):
    require(type(value) is int and 0<=value<2**63,'explicit uint63 seed required')
    require(not is_sealed_scenario_seed(value),'sealed argument cannot enter a training proposal')
    return value


def validate_inventories(training,evidence,*,config=PPOConfig(),initialization_seed,optimizer_seed):
    """Pure finite schema checks. No freshness, RNG or collection is implied."""
    require(config.joint_step_budget==20000,'initial admission permits only the 20k calibration pair')
    initialization_seed=seed(initialization_seed);optimizer_seed=seed(optimizer_seed)
    require(initialization_seed!=optimizer_seed,'initialization and optimizer streams must be distinct')
    require(set(training)=={'schema','cases'} and training['schema']==TRAIN_VERSION
        and type(training['cases']) is list and 8<=len(training['cases'])<=config.joint_step_budget,
        'invalid finite eight-slot training inventory')
    require(set(evidence)=={'schema','scope','expected_counts','cases'}
        and evidence['schema']==observed.INVENTORY_VERSION and evidence['scope']==observed.SCOPE
        and set(evidence['expected_counts'])==set(observed.online.SPLITS)
        and all(type(n) is int and n>0 for n in evidence['expected_counts'].values())
        and type(evidence['cases']) is list,'invalid program-only evidence inventory')
    ids=set();scenes=set();values={initialization_seed,optimizer_seed};slots=Counter();counts=Counter()
    def accept(episode,base,scenario,action,task):
        require(type(episode) is str and episode and episode not in ids,'episode identity reused across partitions')
        require(type(task) is int and task in (1,2,3),'unknown task')
        base,scenario,action=seed(base),seed(scenario),seed(action)
        require(scenario==10*base+task,'task expansion does not match its whole scene')
        require(base not in scenes,'whole scene reused across partitions or tasks')
        arguments={base,scenario,action}
        require(len(arguments)==3 and not arguments&values,'scene/action/initialization/optimizer namespaces overlap')
        ids.add(episode);scenes.add(base);values.update(arguments)
    for case in training['cases']:
        require(set(case)=={'episode_id','scenario_seed','task','action_seed','slot','partition'}
            and case['partition']=='train' and type(case['slot']) is int and 0<=case['slot']<8,
            'training inventory contains a non-train case or invalid slot')
        scenario=seed(case['scenario_seed'])
        accept(case['episode_id'],scenario//10,scenario,case['action_seed'],case['task']);slots[case['slot']]+=1
    minimum=(config.joint_step_budget//8+config.episode_horizon-1)//config.episode_horizon
    require(set(slots)==set(range(8)) and min(slots.values())>=minimum
        and max(slots.values())<=config.joint_step_budget//8,
        'each slot requires a bounded finite training stream')
    for case in evidence['cases']:
        require(set(case)=={'episode','base_seed','scenario_seed','task','action_seed','split'}
            and case['split'] in observed.online.SPLITS,'program evidence split or schema differs')
        accept(case['episode'],case['base_seed'],case['scenario_seed'],case['action_seed'],case['task']);counts[case['split']]+=1
    require(dict(counts)==evidence['expected_counts'],'program evidence partition counts differ')
    return dict(training_episodes=len(training['cases']),episodes_per_slot={str(k):v for k,v in sorted(slots.items())},evidence_counts=dict(counts),
        complete_scene_values=sorted(scenes),all_seed_arguments=sorted(values),episode_ids=sorted(ids),
        raw_regularized_share_identical_training_and_evidence_inventories=True,
        scientific_sample_counts_must_be_locked_by_future_owner=True,history_freshness_verified=False,
        environment_authority=False,optimizer_authority=False)


def known_spent_arguments(ready):
    """Explicit prior development/confirmation exclusions, not a history audit."""
    ready.verify();confirmation=bound(ready.value['confirmation']['plan'])
    values=set();ids=set()
    for case in spent_development_cases():
        values.update(case[k] for k in ('base','scenario_seed','action_seed'));ids.add(case['case_id'])
    for case in confirmation['cases']:
        values.update(case[k] for k in ('base','scenario_seed','action_seed'));ids.add(case['case_id'])
    return values,ids


@dataclass(frozen=True)
class PartitionProposal:
    ref:dict
    value:dict
    readiness:object
    refs:tuple

    def verify(self):
        self.readiness.verify();p=self.value
        require(pointer(self.ref['path'])==self.ref and bound(self.ref)==p and p['version']==VERSION
            and p['source_files']==source_files() and p['readiness_sha256']==self.readiness.payload_sha256
            and p['training_authority'] is False,'training partition proposal or prerequisite/source changed')
        require(bound(pointer(Path(self.ref['path']).with_name('completed.json')))==dict(proposal_sha256=self.ref['sha256']),
            'training partition proposal lacks its completion seal')
        for ref in self.refs:require(pointer(ref['path'])==ref,'training partition asset changed')
        require(p['asset_manifest_sha256']==digest(self.refs),'training partition asset manifest changed')
        return self


def prepare(*,reference_directory,training_inventory_ref,evidence_inventory_ref,initialization_seed,optimizer_seed,output):
    ready=admission.load_readiness(reference_directory)  # Before inventory reads/output.
    ready.verify();sources=source_files()
    require(pointer(training_inventory_ref['path'])==training_inventory_ref
        and pointer(evidence_inventory_ref['path'])==evidence_inventory_ref,'input inventory hash changed')
    require(all(not Path(ref['path']).is_symlink() and not Path(ref['path']).stat().st_mode&0o222
        for ref in (training_inventory_ref,evidence_inventory_ref)),'finite inventories must be immutable regular files')
    training,evidence=bound(training_inventory_ref),bound(evidence_inventory_ref)
    checked=validate_inventories(training,evidence,initialization_seed=initialization_seed,optimizer_seed=optimizer_seed)
    require(checked['evidence_counts']==ready.value['recipe']['program_evidence_counts'],
        'program evidence counts differ from the preregistered training readiness recipe')
    spent,ids=known_spent_arguments(ready)
    require(not spent&set(checked['all_seed_arguments']) and not ids&set(checked['episode_ids']),
        'proposed train/evidence data overlaps spent development or confirmation')
    refs=tuple(deepcopy([*ready.refs,training_inventory_ref,evidence_inventory_ref]))
    for ref in refs:require(pointer(ref['path'])==ref,'input changed while preparing training proposal')
    ready.verify();require(source_files()==sources,'source changed preparing training proposal')
    output=Path(output).resolve();require(output.is_relative_to(OUTPUT_ROOT.resolve()) and output!=OUTPUT_ROOT.resolve(),
        'training proposal requires a new isolated output')
    output.mkdir(parents=True,exist_ok=False)
    value=dict(version=VERSION,candidate_id=admission.v.CANDIDATE,source_files=sources,source_sha256=digest(sources),
        readiness_sha256=ready.payload_sha256,config=asdict(PPOConfig()),initialization_seed=initialization_seed,
        optimizer_seed=optimizer_seed,train_case_inventory=deepcopy(training_inventory_ref),
        program_evidence_inventory=deepcopy(evidence_inventory_ref),checked=checked,asset_manifest_sha256=digest(refs),
        future_owner_must_bind_proposal_ref=True,exact_history_audit_and_single_use_owner_still_required=True,
        training_authority=False,new_environment_resets=0,Actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
    write_new(output/'proposal.json',value);(output/'proposal.json').chmod(0o444)
    ref=pointer(output/'proposal.json');write_new(output/'completed.json',dict(proposal_sha256=ref['sha256']))
    (output/'completed.json').chmod(0o444)
    return PartitionProposal(ref,value,ready,refs).verify()


def assemble_training_batch(proposal,reader,trace_ref,state):
    """Only physically replayed inventory members reach an NN update caller.

    The wrapper checks source separation and owner binding; the future owner
    is still responsible for exact history clearance and real reset journals.
    It grants no optimizer capability and makes no model update itself.
    """
    proposal.verify()
    require(type(reader) is EnvironmentRolloutReplay and reader.inventory_ref==proposal.value['train_case_inventory'],
        'training reader uses another inventory')
    require(asdict(reader.config)==proposal.value['config'] and state.initialization_seed==proposal.value['initialization_seed']
        and state.reference_sha256==proposal.readiness.value['actor']['sha256'],
        'training reader or state differs from paired proposal')
    owner=bound(reader.owner_ref)
    require(owner['training_partition_proposal']==proposal.ref
        and owner['program_evidence_inventory']==proposal.value['program_evidence_inventory']
        and owner['train_case_inventory']==proposal.value['train_case_inventory'],'owner does not bind the paired data partitions')
    trace=bound(trace_ref);allowed={c['episode_id']:c for c in bound(reader.inventory_ref)['cases']}
    require(all(allowed.get(e['case']['episode_id'])==e['case'] for e in trace['events']),
        'program evidence or another scene cannot enter PPO gradients')
    batch,audit=reader.assemble(trace_ref,state)
    require(all(row['episode_id'] in allowed for key in ('rows','complete_cost_rows') for row in batch[key]),
        'reward or completed cost buffer contains non-training episodes')
    proposal.verify()
    return batch,dict(physics_and_NN_replay=audit,partition_proposal=deepcopy(proposal.ref),
        program_evidence_frames_in_gradient_buffers=0,environment_resets=0,optimizer_steps=0,training_authority=False)


def check_reference_buffer(proposal,buffer):
    """The initial half of reference KL is the exact audited common-fit file.

    The current half must come from assemble_training_batch. The PPO core
    rechecks its buffer bytes before updates; the future owner must use both
    bindings rather than trusting caller-assigned episode IDs or split tags.
    """
    proposal.verify();expected=proposal.readiness.value['reference']['buffer']
    require(isinstance(buffer,ReferenceTrainingBuffer) and buffer.source_path is not None
        and pointer(buffer.source_path)==expected and buffer.source_sha256==expected['sha256'],
        'reference KL must use the exact admitted common-fit buffer')
    original=ReferenceTrainingBuffer.load(expected['path'],expected['sha256'],proposal.readiness.value['actor']['sha256'])
    require(np.array_equal(buffer.encoded,original.encoded) and np.array_equal(buffer.masks,original.masks)
        and buffer.episode_ids==original.episode_ids,'reference buffer was relabeled or changed in memory')
    proposal.verify();return buffer
