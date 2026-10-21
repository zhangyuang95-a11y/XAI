"""Read-only v630 A/B admission, not an environment or optimizer capability.

Reuse the original audited reference buffer without changing its source or
receipts. The future finite owner still needs its own exact seed clearance,
single-use claim and full training-state binding before any actual training.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from . import v630_confirmation as c, v630_protocol as v
from . import v629_training_admission as legacy
from backend.training import warehouse_v630_training_program as program_training
from scripts import audit_warehouse_v630_confirmation as confirmation_audit

VERSION='warehouse-v630-training-readiness.v1'
PROGRAM_FOLD=0
FILES=('experiments/warehouse/v630_training_admission.py',
       'backend/training/warehouse_v630_training_program.py',
       'scripts/check_warehouse_v630_training_admission.py',
       'tests/test_warehouse_v630_training_admission.py',
       'tests/test_warehouse_v630_training_program.py',
       'docs/warehouse_v630_training_bridge.md')
reference=legacy.reference
sealed_plan_refs=legacy.sealed_plan_refs
# Pure source/readback verifier of the already audited v629 preparation. It
# does not load a v629 winner or initialize models/optimizers/scene RNGs.
reference_evidence=legacy.reference_evidence


def source_files():
    files=c.source_files()
    for name,sha in legacy.source_files().items():
        v.require(name not in files or files[name]==sha,'conflicting inherited training source')
        files[name]=sha
    for name in FILES:files[name]=v.file_sha(v.ROOT/name)
    return dict(sorted(files.items()))


def recipe():
    value=legacy.recipe()
    value.update(version=VERSION,program_language='v630_fixed_structure_and_complete_protected_descendants',
        program_refit_method=program_training.VERSION,program_pair_fold=PROGRAM_FOLD,
        original_reference_buffer_reused_without_regeneration=True,
        current_target_NN_hash_distinct_from_immutable_anchor_hash=True)
    return value


def program_pair(bundle):
    v.require(PROGRAM_FOLD==0,'training fold must be fixed before confirmation results')
    units=[u for u in bundle.units if u.fold==PROGRAM_FOLD]
    v.require(len(units)==2 and {u.role for u in units}==set(c.metrics.ROLES),
              'fixed fold zero must provide exactly both roles')
    return {u.role:dict(program=deepcopy(u.program_ref),reference=deepcopy(u.reference_ref),
        language=program_training.language_lock(u.program)) for u in units}


def expected_summary(path,passed):
    value=legacy.expected_summary(path,passed)
    value['schema']='warehouse-v630-fixed-confirmation-summary.v1'
    return value


def winner_evidence(winner):
    path,plan=v.checked_plan()
    v.require(winner.provenance['plan']==v.pointer(path),'winner belongs to another extraction plan')
    refs=[*winner.asset_refs,*winner.provenance.values(),*sealed_plan_refs(path,plan)]
    # Preserve source snapshots and seals along the original experiment chain.
    for ref in (plan['previous_plan'],v.bound(plan['previous_plan'])['previous_plan']):
        refs.extend(sealed_plan_refs(Path(ref['path']),v.bound(ref)))
    winner.verify()
    return refs


def clearance_evidence(clearance):
    e=v.bound(clearance['exact'])
    refs=[deepcopy(c.INVENTORY_REF),clearance['exact'],e['exact_plan'],e['bounded_or_triaged_inventory']]
    if e.get('context_annotations'):refs.append(e['context_annotations'])
    return refs


def confirmation_evidence(winner):
    marker=v.OUTPUT/'confirmation.passed.json'
    v.require(marker.is_file() and not (v.OUTPUT/'confirmation.failed.json').exists(),
              'no exclusive passing v630 fixed confirmation; training remains closed')
    marker_ref=v.pointer(marker);m=v.bound(marker_ref);summary_ref=m['summary'];summary=v.bound(summary_ref)
    v.require(m==dict(summary=summary_ref,confirmation_passed=True),'confirmation passed marker schema differs')
    path,plan,bundle=c.checked_plan(summary['plan']['path'])
    v.require(summary_ref==v.pointer(path.parent/'summary.json') and summary['plan']==v.pointer(path)
              and bundle.provenance==winner.provenance and bundle.actor==winner.actor,
              'confirmation summary is not the locked v630 winner')
    datasets,collection,audit=confirmation_audit.load_datasets(path,plan)
    clearance=c.checked_clearance(path,plan,execution=False)
    v.require(audit['clearance']==v.pointer(path.parent/'seed-ledger/clearance.json')
              and audit['exact_evidence']==clearance['exact'],'confirmation seed evidence differs')
    dependencies=sealed_plan_refs(path,plan)+clearance_evidence(clearance)
    expected=c.metrics.evaluate_fixed(bundle,datasets,plan['cases'],collection['outcomes'])
    expected.update(plan=v.pointer(path),collection_audit=v.pointer(path.parent/'collection-audit.json'),
                    collection=audit['collection'])
    v.require(v.bound(v.pointer(path.parent/'metrics.json'))==expected
              and expected['metrics_passed'] is True and audit['integrity_passed'] is True,
              'confirmation must independently pass strict metrics and all four paired endpoints')
    v.require(summary==expected_summary(path,True),'confirmation summary authority or next stage differs')
    refs=[marker_ref,summary_ref,v.pointer(path),v.pointer(path.parent/'metrics.json'),
          v.pointer(path.parent/'collection-audit.json'),audit['collection']]
    for key in ('claim','clearance','exact_evidence'):refs.append(audit[key])
    for row in collection['cases'].values():refs.extend(row.values())
    refs.extend(dependencies)
    return dict(summary=summary_ref,plan=v.pointer(path),metrics=v.pointer(path.parent/'metrics.json'),
                collection_audit=refs[4]),refs


@dataclass(frozen=True)
class TrainingReadiness:
    value:dict
    refs:tuple
    payload_sha256:str

    def verify(self):
        v.require(v.digest(self.value)==self.payload_sha256
                  and self.value['source_files']==source_files()
                  and self.value['recipe']==recipe()
                  and self.value['asset_manifest_sha256']==v.digest(self.refs),
                  'v630 training prerequisite payload or sources changed')
        for ref in self.refs:v.require(v.pointer(ref['path'])==ref,'training prerequisite asset changed')
        return self


def load_readiness(reference_directory):
    # Before reference I/O, seed proposals, model/critic initialization or RNGs.
    winner=c.metrics.load_locked_bundle()
    sources=source_files();refs=winner_evidence(winner)
    confirmed,confirmation_refs=confirmation_evidence(winner);refs.extend(confirmation_refs)
    buffer,buffer_refs=reference_evidence(winner,reference_directory);refs.extend(buffer_refs)
    pair=program_pair(winner);refs.extend([winner.actor,*winner.provenance.values()]);unique={}
    for ref in refs:
        v.require(ref['path'] not in unique or unique[ref['path']]==ref,'conflicting readiness asset binding')
        unique[ref['path']]=deepcopy(ref)
    refs=tuple(unique[name] for name in sorted(unique))
    v.require(source_files()==sources,'readiness source changed during evidence replay')
    value=dict(version=VERSION,candidate_id=v.CANDIDATE,prerequisites_passed=True,recipe=recipe(),
        actor=deepcopy(winner.actor),program_pair=pair,all_six_programs_confirmed=True,
        confirmation=confirmed,reference=buffer,source_files=sources,asset_manifest_sha256=v.digest(refs),
        next_stage='finite_20k_pair_plan_and_exact_seed_audit',environment_authority=False,
        optimizer_authority=False,seed_allocation_authority=False,new_environment_resets=0,
        Actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)
    return TrainingReadiness(value,refs,v.digest(value)).verify()
