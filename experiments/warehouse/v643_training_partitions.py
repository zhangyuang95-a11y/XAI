"""Finite v6.43 TRAIN/program/quality inventories, never execution authority.

The historical v6.42 readiness reader is used ONLY to authenticate the original
v6.20 BC Actor/reference buffer and their already-spent extraction evidence.
It does not require the failed v6.42 20k result to pass. No old claim is loaded.
No environment, RNG, model, optimizer or scientific episode is constructed.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from . import v642_training_admission as historical_reference
from . import v642_calibration_quality as quality
from . import v643_calibration as numerical
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256
from .v626_protocol import require, pointer, bound, digest, file_sha
from backend.training.warehouse_v625_actor_training import PPOConfig
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
from scripts.run_warehouse_v625 import write_new

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'warehouse-v643-finite-calibration-partitions.v1'
TRAIN_VERSION = 'warehouse-v629-finite-training-case-inventory.v1'
PROGRAM_VERSION = 'warehouse-v643-program-evidence-case-inventory.v1'
PROGRAM_SCOPE = 'program_evidence_only_not_NN_gradient_data'
QUALITY_VERSION = 'warehouse-v643-quality-case-inventory.v1'
PROGRAM_COUNTS = dict(fit=900, calibration=150, eligibility=300, validation=300)
OUTPUT_ROOT = ROOT/'output/warehouse_v643'/CANDIDATE/'partitions'
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z')


def source_files():
    files = numerical.source_files()
    for name in ('experiments/warehouse/v643_training_partitions.py',
                 'tests/test_warehouse_v643_training_partitions.py',
                 'docs/warehouse_v643_finite_owner.md'):
        files[name] = file_sha(ROOT/name)
    return dict(sorted(files.items()))


def seed(value):
    require(type(value) is int and 0 <= value < 2**63 and not is_sealed_scenario_seed(value),
            'explicit unsealed uint63 original seed argument required')
    return value


def validate_inventories(training, program, evaluation, *, initialization_seed, optimizer_seed,
                         config=PPOConfig()):
    """Exact membership and namespace validation; no history freshness claim."""
    require(asdict(config) == asdict(PPOConfig()), 'v643 is a fixed 20k calibration recipe')
    initialization_seed, optimizer_seed = seed(initialization_seed), seed(optimizer_seed)
    require(initialization_seed != optimizer_seed, 'initialization and optimizer streams overlap')
    require(type(training) is dict and set(training) == {'schema', 'cases'}
            and training['schema'] == TRAIN_VERSION and type(training['cases']) is list,
            'invalid finite TRAIN inventory')
    require(type(program) is dict and set(program) == {'schema','scope','candidate_id','expected_counts','cases'}
            and program['schema'] == PROGRAM_VERSION and program['scope'] == PROGRAM_SCOPE
            and program['candidate_id'] == CANDIDATE and program['expected_counts'] == PROGRAM_COUNTS
            and all(type(n) is int for n in program['expected_counts'].values())
            and type(program['cases']) is list, 'invalid new candidate program-only inventory')
    require(type(evaluation) is dict and set(evaluation) == {'schema','candidate_id','cases'}
            and evaluation['schema'] == QUALITY_VERSION and evaluation['candidate_id'] == CANDIDATE,
            'invalid new candidate quality-only inventory')
    quality.validate_cases(evaluation['cases'])
    identities, scenes = set(), set()
    values = {initialization_seed, optimizer_seed}
    counts, slots = Counter(), Counter()

    def accept(identifier, base, scenario, action, task):
        require(type(identifier) is str and _NAME.fullmatch(identifier)
                and identifier.startswith(CANDIDATE+'-') and identifier not in identities,
                'unsafe or duplicate episode identity across partitions')
        require(type(task) is int and task in (1,2,3), 'invalid task')
        base, scenario, action = seed(base), seed(scenario), seed(action)
        require(scenario == 10*base+task and base not in scenes, 'whole scene reused or task expansion differs')
        current = {base, scenario, action}
        require(len(current) == 3 and not current & values,
                'TRAIN/program/quality/initialization/optimizer original arguments overlap')
        identities.add(identifier); scenes.add(base); values.update(current)

    for case in training['cases']:
        require(type(case) is dict and set(case) == {'episode_id','scenario_seed','task','action_seed','slot','partition'}
                and case['partition'] == 'train' and type(case['slot']) is int and 0 <= case['slot'] < 8,
                'TRAIN case schema or eight-slot membership differs')
        scenario = seed(case['scenario_seed'])
        accept(case['episode_id'], scenario//10, scenario, case['action_seed'], case['task'])
        slots[case['slot']] += 1
    minimum = (config.joint_step_budget//8+config.episode_horizon-1)//config.episode_horizon
    require(set(slots) == set(range(8)) and minimum <= min(slots.values())
            and max(slots.values()) <= config.joint_step_budget//8,
            'each finite slot requires bounded capacity for the 20k budget')
    for case in program['cases']:
        require(type(case) is dict and set(case) == {'episode','base_seed','scenario_seed','task','action_seed','split'}
                and case['split'] in PROGRAM_COUNTS, 'program-only case schema or split differs')
        accept(case['episode'], case['base_seed'], case['scenario_seed'], case['action_seed'], case['task'])
        counts[case['split']] += 1
    require(dict(counts) == PROGRAM_COUNTS, 'program evidence episode counts differ')
    require(all(Counter(c['task'] for c in program['cases'] if c['split'] == split)
                == {task: number//3 for task in (1,2,3)} for split, number in PROGRAM_COUNTS.items()),
            'program evidence must balance all three tasks within every split')
    for case in evaluation['cases']:
        accept(case['case_id'], case['base'], case['scenario_seed'], case['action_seed'], case['task'])
    return dict(training_episodes=len(training['cases']), program_counts=dict(counts), quality_episodes=75,
                episodes_per_slot={str(k): slots[k] for k in range(8)},
                all_seed_arguments=sorted(values), complete_scene_values=sorted(scenes),
                episode_ids=sorted(identities), history_freshness_verified=False,
                same_raw_regularized_training_and_program_inventories=True,
                same_reference_raw_regularized_quality_scenes=True,
                program_or_quality_frames_permitted_in_gradient_buffers=0,
                environment_authority=False, optimizer_authority=False)


def _immutable(ref):
    require(type(ref) is dict and set(ref) == {'path','sha256'} and pointer(ref['path']) == ref,
            'partition input hash changed')
    path = Path(ref['path'])
    require(path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222,
            'partition inputs must be immutable regular files')
    return bound(ref)


@dataclass(frozen=True)
class Partition:
    ref: dict
    value: dict
    reference_evidence: object

    def verify(self):
        self.reference_evidence.verify()
        p = self.value
        require(_immutable(self.ref) == p and p['version'] == VERSION and p['candidate_id'] == CANDIDATE
                and p['source_files'] == source_files() and p['source_sha256'] == digest(source_files())
                and p['reference_actor']['sha256'] == REFERENCE_SHA256
                and p['reference_actor'] == self.reference_evidence.value['actor']
                and p['reference_buffer'] == self.reference_evidence.value['reference']['buffer']
                and p['program_templates'] == {role: row['program'] for role, row
                    in self.reference_evidence.value['program_pair'].items()}
                and p['historical_reference_evidence_sha256'] == self.reference_evidence.payload_sha256
                and p['training_authority'] is False, 'v643 partition/source/reference binding differs')
        checked = validate_inventories(*[_immutable(p[key]) for key in
            ('train_case_inventory','program_evidence_inventory','quality_case_inventory')],
            initialization_seed=p['initialization_seed'], optimizer_seed=p['optimizer_seed'])
        require(checked == p['checked'] and _immutable(pointer(Path(self.ref['path']).with_name('completed.json')))
                == dict(partition_sha256=self.ref['sha256']), 'partition semantics or completion seal differs')
        return self


def prepare(*, reference_directory, training_inventory_ref, program_inventory_ref,
            quality_inventory_ref, initialization_seed, optimizer_seed, output):
    # Historical original-reference authentication, not an old 20k pass gate.
    ready = historical_reference.load_readiness(reference_directory)
    ready.verify()
    require(ready.value['actor']['sha256'] == REFERENCE_SHA256,
            'only the complete original v620 BC artifact can be the new paired start')
    refs = (training_inventory_ref, program_inventory_ref, quality_inventory_ref)
    inventories = [_immutable(ref) for ref in refs]
    checked = validate_inventories(*inventories, initialization_seed=initialization_seed,
                                  optimizer_seed=optimizer_seed)
    sources = source_files()
    output = Path(output).resolve()
    require(output.is_relative_to(OUTPUT_ROOT.resolve()) and output != OUTPUT_ROOT.resolve(),
            'v643 partitions require their new candidate namespace')
    ready.verify()
    require(source_files() == sources, 'source changed before partition freeze')
    output.mkdir(parents=True, exist_ok=False)
    value = dict(version=VERSION, candidate_id=CANDIDATE, source_files=sources, source_sha256=digest(sources),
        reference_actor=deepcopy(ready.value['actor']), reference_buffer=deepcopy(ready.value['reference']['buffer']),
        program_templates={role:deepcopy(row['program']) for role,row in ready.value['program_pair'].items()},
        historical_reference_evidence_sha256=ready.payload_sha256,
        historical_confirmation_is_not_v643_calibration_pass=True, old_20k_pass_required=False,
        train_case_inventory=deepcopy(training_inventory_ref), program_evidence_inventory=deepcopy(program_inventory_ref),
        quality_case_inventory=deepcopy(quality_inventory_ref), initialization_seed=initialization_seed,
        optimizer_seed=optimizer_seed, config=asdict(PPOConfig()), checked=checked,
        training_authority=False, new_environment_resets=0, new_Actor_updates=0,
        exact_history_and_complete_owner_required=True, pilot_opened=False)
    write_new(output/'partition.json', value); (output/'partition.json').chmod(0o444)
    ref = pointer(output/'partition.json')
    write_new(output/'completed.json', dict(partition_sha256=ref['sha256'])); (output/'completed.json').chmod(0o444)
    return Partition(ref, value, ready).verify()


def load(reference_directory, ref):
    ready = historical_reference.load_readiness(reference_directory)
    ready.verify()
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(OUTPUT_ROOT.resolve()) and path.name == 'partition.json',
            'v643 partition is outside its registered namespace')
    return Partition(ref, _immutable(ref), ready).verify()
