"""Candidate-bound program evidence namespace, not a training capability.

The v642 numerical program language and physical trace replay are unchanged.
No v642 readiness, permit, claim or output path is accepted as v643 authority.
This reader verifies an already claimed finite owner; it never initializes an
Actor/environment, collects data, claims seeds, or grants recovery authority.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
import gzip
import json
import re

import numpy as np

from backend.training import warehouse_v629_environment_replay as physics
from backend.training import warehouse_v629_online_evidence as online
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION, encode_actor_tree_features
from backend.training.warehouse_v622_program import masked_probabilities
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_observation_v6 import public_pre_action_snapshot, encode_public_observation
from env.warehouse.pure_nn import AGENTS, action_masks
from experiments.warehouse.v626_protocol import require, pointer, bound, digest, file_sha
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256

ROOT = Path(__file__).resolve().parents[2]
OWNER_ROOT = ROOT/'output/warehouse_v643'/CANDIDATE
VERSION = 'warehouse-v643-observed-program-states.v1'
OWNER_VERSION = 'warehouse-v643-finite-calibration-variant-owner.v1'
CLAIM_VERSION = 'warehouse-v643-finite-calibration-single-use-claim.v1'
PARTITION_VERSION = 'warehouse-v643-finite-calibration-partitions.v1'
PLAN_VERSION = 'warehouse-v643-finite-calibration-seed-plan.v1'
INVENTORY_VERSION = 'warehouse-v643-program-evidence-case-inventory.v1'
EPISODE_VERSION = 'warehouse-v643-program-source-episode.v1'
SCOPE = 'program_evidence_only_not_NN_gradient_data'
COUNTS = dict(fit=900, calibration=150, eligibility=300, validation=300)
_REPLAYED = object()


def source_files():
    # Hash inherited numerical dependencies; never invoke an old admission.
    from .v643_calibration import source_files as numerical_sources
    files = numerical_sources()
    for name in ('backend/training/warehouse_v642_program.py',
                 'backend/training/warehouse_v642_training_program.py',
                 'backend/training/warehouse_v642_online_evidence.py',
                 'backend/training/warehouse_v642_guards.py',
                 'backend/training/warehouse_v642_certificate.py',
                 'backend/training/warehouse_v629_online_evidence.py',
                 'scripts/audit_warehouse_v642_program.py',
                 'experiments/warehouse/v643_program_namespace.py',
                 'experiments/warehouse/v643_program_provider.py',
                 'tests/test_warehouse_v643_program.py',
                 'docs/warehouse_v643_program_bridge.md'):
        files[name] = file_sha(ROOT/name)
    return dict(sorted(files.items()))


def immutable(ref, *, decode=True):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'immutable program asset reference required')
    path = Path(ref['path'])
    require(path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222
        and pointer(path) == ref, 'v643 program asset missing, writable or changed')
    return bound(ref) if decode else deepcopy(ref)


def _seed(value):
    require(type(value) is int and 0 <= value < 2**63 and not is_sealed_scenario_seed(value),
            'program evidence cannot consume an unknown or sealed seed')
    return value


def _seed_values(value):
    if type(value) is list:
        return set().union(*(_seed_values(v) for v in value)) if value else set()
    if type(value) is dict:
        result = {v for k,v in value.items() if k in ('base','base_seed','scenario_seed','action_seed') and type(v) is int}
        result.update(v//10 for k,v in value.items() if k == 'scenario_seed' and type(v) is int)
        for v in value.values():
            if type(v) in (dict,list): result.update(_seed_values(v))
        return result
    return set()


def _episode_values(value):
    if type(value) is list:
        return set().union(*(_episode_values(v) for v in value)) if value else set()
    if type(value) is dict:
        result={v for k,v in value.items() if k in ('episode','episode_id','case_id') and type(v) is str}
        for v in value.values():
            if type(v) in (dict,list):result.update(_episode_values(v))
        return result
    return set()


def validate_inventory(value):
    require(type(value) is dict and set(value) == {'schema','scope','candidate_id','expected_counts','cases'}
        and value['schema'] == INVENTORY_VERSION and value['scope'] == SCOPE
        and value['candidate_id'] == CANDIDATE and value['expected_counts'] == COUNTS
        and all(type(n) is int for n in value['expected_counts'].values()) and type(value['cases']) is list,
        'new candidate finite program inventory required')
    ids, scenes, arguments, counts = set(), set(), set(), {s:0 for s in online.SPLITS}
    for case in value['cases']:
        require(type(case) is dict and set(case) == {'episode','base_seed','scenario_seed','task','action_seed','split'}
            and type(case['episode']) is str and case['episode'].startswith(CANDIDATE+'-')
            and re.fullmatch(r'[A-Za-z0-9_.-]+',case['episode'])
            and case['split'] in online.SPLITS and type(case['task']) is int and case['task'] in (1,2,3),
            'program case is outside the v643 candidate/partition namespace')
        base, scenario, action = (_seed(case[k]) for k in ('base_seed','scenario_seed','action_seed'))
        values = {base,scenario,action}
        require(scenario == 10*base+case['task'] and case['episode'] not in ids and base not in scenes
            and len(values) == 3 and not values & arguments, 'program whole scenes, action streams or IDs overlap')
        ids.add(case['episode']); scenes.add(base); arguments.update(values); counts[case['split']] += 1
    require(counts == COUNTS, 'program fit/calibration/qualification/validation counts differ')
    return dict(episodes=sorted(ids), scenes=sorted(scenes), seed_arguments=sorted(arguments), counts=counts)


def _load_owner(owner_ref):
    owner = immutable(owner_ref)
    require(owner.get('schema') == OWNER_VERSION and owner.get('candidate_id') == CANDIDATE
        and owner.get('variant') in ('raw','regularized'), 'v643 requires its own claimed variant owner')
    claim_ref, plan_ref, descriptor_ref = (owner.get(k) for k in ('claim','plan','descriptor'))
    claim, plan, descriptor = (immutable(r) for r in (claim_ref,plan_ref,descriptor_ref))
    require(Path(claim_ref['path']) == OWNER_ROOT/'single-use-20k.claim.json'
        and Path(plan_ref['path']).is_relative_to(OWNER_ROOT/'seed-plans')
        and Path(plan_ref['path']).name == 'plan.json'
        and Path(descriptor_ref['path']) == Path(plan_ref['path']).parent/'calibration/descriptor.json'
        and Path(owner_ref['path']) == Path(plan_ref['path']).parent/'calibration'/owner['variant']/'owner.json',
        'old or unowned claim/output namespace cannot authorize v643 programs')
    require(claim.get('schema') == CLAIM_VERSION and claim.get('candidate_id') == CANDIDATE
        and claim.get('plan') == plan_ref and claim.get('descriptor') == descriptor_ref
        and claim.get('single_use_even_if_interrupted') is True
        and plan.get('version') == PLAN_VERSION and plan.get('candidate_id') == CANDIDATE
        and descriptor.get('candidate_id') == CANDIDATE and descriptor.get('plan') == plan_ref,
        'old claim/plan/descriptor cannot be reinterpreted as a v643 owner')
    partition_ref = plan['partition']; partition = immutable(partition_ref)
    require(claim.get('partition') == partition_ref and descriptor.get('partition') == partition_ref
        and partition.get('version') == PARTITION_VERSION and partition.get('candidate_id') == CANDIDATE,
        'v643 program partition differs from the actual finite claim')
    sources = owner.get('source_files'); expected = source_files()
    require(type(sources) is dict and all(sources.get(k) == v for k,v in expected.items())
        and digest(sources) == owner.get('source_sha256') == claim.get('source_sha256')
        == descriptor.get('source_sha256') == plan.get('source_sha256')
        and descriptor.get('source_files') == plan.get('source_files') == sources,
        'v643 program numerical/source closure differs from its claimed owner')
    refs = [owner_ref,claim_ref,plan_ref,descriptor_ref,partition_ref]
    inventories = {}
    for key in ('train_case_inventory','program_evidence_inventory','quality_case_inventory'):
        ref = owner.get(key)
        require(ref == descriptor.get(key) == partition.get(key), 'v643 owner data inventory changed')
        inventories[key] = immutable(ref); refs.append(ref)
    program = inventories['program_evidence_inventory']; checked = validate_inventory(program)
    other = _seed_values(inventories['train_case_inventory']) | _seed_values(inventories['quality_case_inventory'])
    require(not set(checked['seed_arguments']) & other,
            'program-only evidence overlaps NN training or independent quality arguments')
    other_ids=_episode_values(inventories['train_case_inventory']) | _episode_values(inventories['quality_case_inventory'])
    require(not set(checked['episodes']) & other_ids, 'program-only episode identity reused by NN training or quality')
    templates=owner.get('program_templates')
    require(type(templates) is dict and set(templates) == set(AGENTS)
        and templates == descriptor.get('program_templates') == partition.get('program_templates'),
        'v643 owner does not bind both original program languages')
    for ref in templates.values():immutable(ref);refs.append(ref)
    reference = partition.get('reference_actor')
    require(type(reference) is dict and reference.get('sha256') == REFERENCE_SHA256,
            'program owner is not bound to the original complete v620 BC Actor')
    immutable(reference,decode=False); refs.append(reference)
    clearance = immutable(claim['clearance']); refs.append(claim['clearance'])
    require(clearance.get('candidate_id') == CANDIDATE and clearance.get('plan') == plan_ref,
            'old candidate clearance cannot bind v643 program evidence')
    return dict(owner=owner, claim=claim_ref, plan=plan_ref, partition=partition_ref,
        program_inventory=owner['program_evidence_inventory'], inventory=program,
        source_files=sources, refs=refs, checked=checked,training_authority=False,
        seed_history_provenance_recomputed=False,program_numerics_only=True)


@dataclass(frozen=True)
class ProgramNamespace:
    owner_ref: dict
    value: dict
    payload_sha256: str

    @classmethod
    def from_owner(cls, owner_ref):
        value = _load_owner(owner_ref)
        return cls(deepcopy(owner_ref),deepcopy(value),digest(value)).verify()

    def verify(self):
        require(digest(self.value) == self.payload_sha256 and _load_owner(self.owner_ref) == self.value,
                'program namespace or claimed finite owner changed')
        return self

    @property
    def variant(self): return self.value['owner']['variant']

    @property
    def output_root(self): return OWNER_ROOT/'program'/self.variant

    def output(self, path):
        self.verify(); path = Path(path).resolve()
        require(path.is_relative_to(self.output_root.resolve()) and path != self.output_root.resolve(),
                'v643 program output must be isolated under its own candidate and variant')
        return path


def data_sha(parts):
    h = sha256()
    for split in online.SPLITS:
        part = parts[split]; h.update(split.encode()); h.update(digest(part['rows']).encode())
        for key in ('x','m','behavior_p','executed'):
            a=part[key]; h.update(key.encode()); h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ObservedPool:
    namespace: ProgramNamespace
    source_ref: dict
    parts: dict
    refs: tuple
    audit: dict
    audit_sha256: str
    _replayed_seal: object = field(default=None,init=False,repr=False,compare=False)

    def verify(self):
        self.namespace.verify()
        require(self._replayed_seal is _REPLAYED, 'program pool must be produced by actual physical trace replay')
        require(digest(self.audit) == self.audit_sha256 and self.audit['observed_data_sha256'] == data_sha(self.parts)
            and self.audit['owner'] == self.namespace.owner_ref and self.audit['source'] == self.source_ref
            and self.audit['asset_manifest_sha256'] == digest(self.refs), 'v643 observed program pool mutated')
        for ref in self.refs: immutable(ref,decode=False)
        return self


def audit_observed_pool(namespace, source_ref):
    """Replay completed recorded episodes; no reset/constructor is called."""
    require(type(namespace) is ProgramNamespace, 'explicit new candidate program namespace required')
    namespace.verify(); source = immutable(source_ref)
    require(set(source) == {'schema','scope','candidate_id','owner','inventory','cases'}
        and source['schema'] == VERSION and source['scope'] == SCOPE and source['candidate_id'] == CANDIDATE
        and source['owner'] == namespace.owner_ref and source['inventory'] == namespace.value['program_inventory'],
        'program source belongs to another candidate, owner or data partition')
    cases=namespace.value['inventory']['cases']
    require(set(source['cases']) == {case['episode'] for case in cases}, 'program source is missing complete finite cases')
    refs=[source_ref,*namespace.value['refs']]
    parts={s:{k:[] for k in ('x','m','behavior_p','executed','rows')} for s in online.SPLITS}
    actors={}; outcomes=[]
    for case in cases:
        item=source['cases'][case['episode']]
        require(set(item) == {'behavior_actor','trace','initial_environment','final_environment'}, 'unknown physical program source record')
        for ref in item.values(): immutable(ref,decode=False)
        refs.extend(deepcopy(list(item.values()))); actor_ref=item['behavior_actor']; key=digest(actor_ref)
        if key not in actors:
            actor=FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'])
            require(actor.model_version == MODEL_VERSION, 'program behavior NN architecture changed'); actors[key]=actor
        actor=actors[key]; env=physics.restore_environment(immutable(item['initial_environment']))
        require(asdict(env.config) == asdict(physics.training_config(case['scenario_seed'])) and env.state.frame == 0
            and not(env.state.terminated or env.state.truncated) and env.state.participant_controlled_agent_id == 'robot_1',
            'program source has a different physical configuration or starting state')
        path=Path(item['trace']['path'])
        if path.suffix == '.gz':
            with gzip.open(path,'rt') as handle: trace=json.load(handle)
        else: trace=immutable(item['trace'])
        require(set(trace) == {'schema','case','behavior_actor','events'} and trace['schema'] == EPISODE_VERSION
            and trace['case'] == case and trace['behavior_actor'] == actor_ref and 0 < len(trace['events']) <= 120,
            'program trace candidate, case or complete-frame inventory differs')
        part=parts[case['split']]
        for frame,event in enumerate(trace['events']):
            require(not(env.state.terminated or env.state.truncated) and env.state.frame == frame
                and public_pre_action_snapshot(env.state) == event['pre_state_public'], 'program source public history disconnected')
            require(type(event['terminated']) is type(event['truncated']) is bool, 'program source requires real terminal booleans')
            obs=encode_public_observation(env.state);m=action_masks(env.state);x=encode_actor_tree_features(obs,m)
            p=masked_probabilities(actor.logits(obs,m),m); actions=[]
            for i,role in enumerate(AGENTS):
                seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=env.state.episode_id,frame=frame,agent_id=role)
                actions.append(int(np.random.default_rng(seed).choice(5,p=p[i])))
            proposed={role:ACTIONS[actions[i]] for i,role in enumerate(AGENTS)}
            require(event['proposed_actions'] == proposed and event['action_override_agents'] == [],
                    'program source actions differ from the actual behavior NN')
            physical_id=env.state.episode_id;_,rewards,terminated,truncated,info=env.step(proposed)
            require(event['rewards'] == rewards and event['terminated'] == terminated and event['truncated'] == truncated
                and sorted(event['shutdowns']) == sorted(info['shutdowns']) and not info['action_override_agents']
                and public_pre_action_snapshot(env.state) == event['post_state_public'], 'program source outcomes/physics differ')
            for name,value in (('x',x),('m',m),('behavior_p',p),('executed',actions)): part[name].append(value)
            part['rows'].append(dict(episode=case['episode'],base_seed=str(case['base_seed']),task=case['task'],frame=frame,
                action_seed=case['action_seed'],episode_id=physical_id))
        require((env.state.terminated or env.state.truncated) and physics.capture_environment(env) == immutable(item['final_environment']),
                'program source is partial or final environment/RNG differs')
        outcomes.append(dict(episode=case['episode'],physical_frames=len(trace['events']),source_policy_outcomes_only=True))
    for part in parts.values():
        for key in ('x','m','behavior_p','executed'):
            part[key]=np.asarray(part[key],np.int64 if key == 'executed' else np.float32);part[key].flags.writeable=False
    audit=dict(version=VERSION,candidate_id=CANDIDATE,source=source_ref,owner=namespace.owner_ref,
        observed_data_sha256=data_sha(parts),asset_manifest_sha256=digest(tuple(refs)),
        complete_source_episodes=len(cases),joint_physical_frames=sum(len(p['x']) for p in parts.values()),
        outcomes=outcomes,current_policy_task_quality_estimate=None,environment_resets=0,
        owner_must_prove_reset_authority_and_exclusion_from_NN_gradients=True,training_authority=False,pilot_opened=False)
    observed=ObservedPool(namespace,deepcopy(source_ref),parts,tuple(refs),audit,digest(audit))
    object.__setattr__(observed,'_replayed_seal',_REPLAYED)
    return observed.verify()
