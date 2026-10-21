"""Read-only training prerequisites, not a finite training capability.

No seed allocation, model initialization, environment construction or updates.
The future Stage C owner must additionally bind its own complete source,
finite scenario schedule, exact historical audit and single-use capability.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
from . import v629_confirmation as c
from . import v629_protocol as v
from backend.training import warehouse_v629_reference_buffer as reference
from backend.training.warehouse_v625_actor_training import PPOConfig,ReferenceTrainingBuffer
from backend.training.warehouse_v628_certificate import structure_info
from backend.training.warehouse_v622_program import plain
from scripts import audit_warehouse_v629_confirmation as confirmation_audit

VERSION='warehouse-v629-training-readiness.v1'
PROGRAM_FOLD=0  # Fixed before any confirmation outcome; never select by score.
FILES=('experiments/warehouse/v629_training_admission.py','scripts/check_warehouse_v629_training_admission.py',
    'backend/training/warehouse_v629_online_evidence.py','backend/training/warehouse_v629_training_feedback.py',
    'backend/training/warehouse_v629_ppo_update.py','backend/training/warehouse_v629_rollout_replay.py',
    'backend/training/warehouse_v622_actor_training.py','backend/training/warehouse_v622_continuation_safety.py',
    'backend/training/warehouse_v6_shutdown_cost.py')


def source_files():
    files=c.source_files();files.update(reference.source_files())
    for name in FILES:files[name]=v.file_sha(v.ROOT/name)
    return dict(sorted(files.items()))


def recipe():
    return dict(version=VERSION,initial_stage='paired_20k_calibration_only',config=asdict(PPOConfig()),
        initial_Actor='complete_frozen_v620_both_roles',initial_critic_and_optimizers='new_identical_within_pair',
        program_pair_fold=PROGRAM_FOLD,program_fold_rule='fixed_zero_not_quality_or_program_score_selection',
        reference_buffer='all_300_spent_argmax_common_fit_only_shared_SHA',reference_mix=[.5,.5],
        program_language='B_locked_structure_supports_witnesses_and_protected_parent_contract',
        raw_external_program_gradient=0,regularized_only_current_NN_qualified_agreeing_program_gradient=True,
        development_budgets_after_calibration=[200000,300000,400000],development_streams=3,
        streams_restart_from_same_reference_not_three_BC_reproductions=True,
        environment_authority=False,optimizer_authority=False,seed_allocation_authority=False,pilot_opened=False)


def structure_sha256(structure,maximum_leaves):
    tree,_=structure_info(structure,maximum_leaves)
    return v.digest(dict(version='warehouse-v629-training-structure.v1',maximum_leaves=maximum_leaves,maximum_depth=9,tree=plain(tree)))


def language_lock(program):
    program.validate()
    return dict(version=program.to_dict()['version'],maximum_leaves=program.maximum_leaves,
        structure_sha256=structure_sha256(program.structure,program.maximum_leaves),
        supports=[list(leaf['support']) for leaf in program.leaves],witnesses=[leaf['witness'] for leaf in program.leaves],
        protection_sha256=v.digest(program.protection),anchor_actor_role_sha256=program.actor_role_sha256,
        coefficient_updates='fixed_support_refit_then_recertify_never_execution_actions')


def program_pair(bundle):
    v.require(PROGRAM_FOLD==0,'fixed training program fold changed without a new method')
    units=[u for u in bundle.units if u.fold==PROGRAM_FOLD]
    v.require(len(units)==2 and {u.role for u in units}==set(c.metrics.ROLES),'fixed fold zero must provide exactly both roles')
    return {u.role:dict(program=deepcopy(u.program_ref),reference=deepcopy(u.reference_ref),language=language_lock(u.program)) for u in units}


def expected_summary(path,passed):
    return dict(schema='warehouse-v629-fixed-confirmation-summary.v1',confirmation_passed=passed,
        plan=v.pointer(path),metrics=v.pointer(path.parent/'metrics.json'),collection_audit=v.pointer(path.parent/'collection-audit.json'),
        method_and_six_assets_fixed=True,next_stage='paired_20k_training_calibration' if passed else 'candidate_closed_failed_confirmation',
        independent_final_audit_used=False,heldout_used=False,actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)


def sealed_plan_refs(path,plan):
    """Bind consumed plan seals, snapshots and declared fixed prerequisites.

    A parent JSON hash does not bind the current bytes of the files it names.
    The owning checked_plan still establishes its schema and scientific gates.
    """
    path=Path(path);claim=path.with_name('plan.claim.json')
    v.require(v.bound(v.pointer(claim))==dict(sha256=v.file_sha(path),single_use=True),'prerequisite plan seal differs')
    refs=[v.pointer(path),v.pointer(claim)]
    for name,sha in plan['source_files'].items():
        refs.append(dict(path=str(path.parent/'source'/plan['source_sha256']/name),sha256=sha))
    refs.extend(dict(path=name,sha256=sha) for name,sha in plan.get('prerequisites',{}).items())
    refs.extend(plan.get('parents',{}).values())
    for name in ('actor','previous_plan','collection'):
        if name in plan:refs.append(plan[name])
    if 'allocation' in plan:
        allocation=plan['allocation'];allocation_claim=Path(allocation['path']).with_name('allocation.claim.json')
        v.require(v.bound(v.pointer(allocation_claim))==dict(sha256=allocation['sha256']),'confirmation allocation seal differs')
        refs.extend([allocation,v.pointer(allocation_claim)])
    return refs


def winner_evidence(winner):
    """All settled jobs, including failed arms, determine smallest-cap selection."""
    path,plan=v.checked_plan()
    v.require(winner.provenance['plan']==v.pointer(path),'winner plan binding differs')
    refs=sealed_plan_refs(path,plan)
    for job in plan['jobs']:
        folder=v.OUTPUT/'jobs'/job['job_id'];result_ref=v.pointer(folder/'result.json');result=v.bound(result_ref)
        seal_ref=v.pointer(folder/'completed.json')
        v.require(v.bound(seal_ref)==dict(result_sha256=result_ref['sha256']) and result['job']==job,'winner job seal or identity differs')
        refs.extend([result_ref,seal_ref])
        if result['status']=='completed':refs.append(result['search_report'])
    # Recheck the actual selection after capturing its complete read set;
    # final verify then detects changes during any later metric replay.
    checked=c.metrics.load_locked_bundle()
    v.require(checked.provenance==winner.provenance and checked.actor==winner.actor
        and [(u.role,u.fold,u.program_ref,u.reference_ref) for u in checked.units]
            ==[(u.role,u.fold,u.program_ref,u.reference_ref) for u in winner.units],'winner changed while binding job evidence')
    return refs


def clearance_evidence(clearance):
    evidence=v.bound(clearance['exact'])
    refs=[deepcopy(c.INVENTORY_REF),clearance['exact'],evidence['exact_plan'],evidence['bounded_or_triaged_inventory']]
    if evidence.get('context_annotations'):refs.append(evidence['context_annotations'])
    return refs


def confirmation_evidence(winner):
    marker=v.OUTPUT/'confirmation.passed.json'
    v.require(marker.is_file() and not (v.OUTPUT/'confirmation.failed.json').exists(),'no exclusive passing fixed confirmation; training remains closed')
    marker_ref=v.pointer(marker);m=v.bound(marker_ref);summary_ref=m['summary'];summary=v.bound(summary_ref)
    v.require(m==dict(summary=summary_ref,confirmation_passed=True),'confirmation passed marker schema differs')
    path,plan,bundle=c.checked_plan(summary['plan']['path'])
    v.require(summary_ref==v.pointer(path.parent/'summary.json') and summary['plan']==v.pointer(path)
        and bundle.provenance==winner.provenance and bundle.actor==winner.actor,'confirmation summary is not the locked winner')
    datasets,collection,audit=confirmation_audit.load_datasets(path,plan)
    clearance=c.checked_clearance(path,plan,execution=False)
    v.require(audit['clearance']==v.pointer(path.parent/'seed-ledger/clearance.json')
        and audit['exact_evidence']==clearance['exact'],'confirmation historical seed evidence differs')
    dependencies=sealed_plan_refs(path,plan)+clearance_evidence(clearance)
    # Recompute every strict gate and scene bootstrap from bound arrays. A
    # true flag in a JSON receipt cannot replace the confirmation evidence.
    expected=c.metrics.evaluate_fixed(bundle,datasets,plan['cases'],collection['outcomes'])
    expected.update(plan=v.pointer(path),collection_audit=v.pointer(path.parent/'collection-audit.json'),collection=audit['collection'])
    stored=v.bound(v.pointer(path.parent/'metrics.json'))
    v.require(stored==expected and expected['metrics_passed'] is True and audit['integrity_passed'] is True,
        'confirmation metrics do not independently pass for fixed assets')
    v.require(summary==expected_summary(path,True),'confirmation summary authority, scope or next stage differs')
    refs=[marker_ref,summary_ref,v.pointer(path),v.pointer(path.parent/'metrics.json'),v.pointer(path.parent/'collection-audit.json'),audit['collection']]
    for name in ('claim','clearance','exact_evidence'):refs.append(audit[name])
    for item in collection['cases'].values():refs.extend(item.values())
    refs.extend(dependencies)
    return dict(summary=summary_ref,plan=v.pointer(path),metrics=v.pointer(path.parent/'metrics.json'),collection_audit=refs[4]),refs


def reference_evidence(winner,output):
    output=Path(output).resolve();base=v.ROOT/'output/warehouse_v629/training-preparation'
    v.require(output.is_relative_to(base) and output!=base,'reference package must be the recorded isolated preparation')
    result_ref=v.pointer(output/'result.json');audit_ref=v.pointer(output/'independent-audit.json')
    r=v.bound(result_ref);audit=v.bound(audit_ref)
    v.require(json.loads((output/'completed.json').read_bytes())==dict(result_sha256=result_ref['sha256']), 'reference completion seal differs')
    path,plan,cases,collection_ref,receipt,outcomes,actor=reference.source_bundle()
    v.require(r['version']==audit['version']==reference.VERSION and r['source_files']==audit['source_files']==reference.source_files()
        and r['source_plan']==audit['source_plan']==v.pointer(path) and r['shared_collection']==audit['shared_collection']==collection_ref
        and r['actor']==plan['actor']==winner.actor and r['actor_role_sha256']==plan['actor_role_sha256']
        and audit['independent_replay_passed'] is True and audit['result']==result_ref and audit['buffer']==r['buffer'],
        'reference preparation not independently bound to the selected complete Actor')
    v.require(r['buffer']==v.pointer(output/'reference-training.npz') and [row['case'] for row in r['case_provenance']]==cases,
        'reference buffer path or predetermined common-fit case inventory differs')
    buffer=ReferenceTrainingBuffer.load(r['buffer']['path'],r['buffer']['sha256'],winner.actor['sha256'])
    with np.load(r['buffer']['path'],allow_pickle=False) as archive:
        v.require(json.loads(str(archive['metadata_json'].item()))==reference.buffer_metadata(path,plan,collection_ref,r['source_files']),
            'initial reference buffer metadata differs')
    offset=0;refs=[result_ref,audit_ref,v.pointer(output/'completed.json'),r['buffer'],v.pointer(path),collection_ref,
        v.pointer(Path(collection_ref['path']).with_name('completed.json')),*sealed_plan_refs(path,plan)]
    for name,sha in r['source_files'].items():
        source=output/'source'/v.digest(r['source_files'])/name
        v.require(v.file_sha(source)==sha and not source.stat().st_mode&0o222,'reference code snapshot changed')
        refs.append(v.pointer(source))
    for row,case in zip(r['case_provenance'],cases,strict=True):
        original=receipt['cases'][case['case_id']];refs.extend(original.values())
        v.require(row['source']==original and row['offset']==offset and row['public_counter']==outcomes[case['case_id']],
            'reference row does not belong to the original common-fit collection')
        for ref in original.values():v.require(v.pointer(ref['path'])==ref,'original common-fit source changed')
        with np.load(original['arrays']['path'],allow_pickle=False) as z:
            n=len(z['x'])
            v.require(row['physical_frames']==n and np.array_equal(buffer.encoded[offset:offset+n],z['x'])
                and np.array_equal(buffer.masks[offset:offset+n],z['m']) and buffer.episode_ids[offset:offset+n]==(case['case_id'],)*n,
                'initial reference buffer includes other or reordered physical frames')
        offset+=n
    v.require(offset==len(buffer.encoded)==r['joint_frames']==audit['joint_frames']
        and r['complete_episodes']==audit['complete_episodes']==300 and r['NN_action_checks']==audit['NN_action_checks']==2*offset
        and r['used_qualification_or_validation'] is False and audit['qualification_validation_or_confirmation_frames']==0
        and r['training_authority'] is audit['training_authority'] is False,'reference count, scope or declared authority differs')
    v.require(all(not Path(ref['path']).stat().st_mode&0o222 for ref in [r['buffer'],*[ref for row in r['case_provenance'] for ref in row['source'].values()]]),
        'reference physical data must stay immutable')
    return dict(buffer=r['buffer'],preparation=result_ref,independent_source_readback=audit_ref,complete_episodes=300,joint_frames=offset),refs


@dataclass(frozen=True)
class TrainingReadiness:
    value:dict
    refs:tuple
    payload_sha256:str

    def verify(self):
        v.require(v.digest(self.value)==self.payload_sha256 and self.value['source_files']==source_files()
            and self.value['asset_manifest_sha256']==v.digest(self.refs) and self.value['recipe']==recipe(),
            'training readiness payload, references or source changed')
        for ref in self.refs:v.require(v.pointer(ref['path'])==ref,'training prerequisite asset missing or changed')
        return self


def load_readiness(reference_directory):
    # Must remain first: failed/partial A cannot touch a training proposal,
    # reference initialization, scene RNG, or environment constructor.
    winner=c.metrics.load_locked_bundle()
    sources=source_files()
    winner_refs=winner_evidence(winner)
    confirmed,refs=confirmation_evidence(winner)
    reference_buffer,reference_refs=reference_evidence(winner,reference_directory)
    refs.extend(reference_refs);refs.extend(winner_refs);refs.extend([winner.actor,*winner.provenance.values()])
    for unit in winner.units:refs.extend([unit.program_ref,unit.reference_ref,unit.program.protection['parent'],unit.reference.protection['parent']])
    pair=program_pair(winner);unique={}
    for ref in refs:
        v.require(ref['path'] not in unique or unique[ref['path']]==ref,'conflicting training prerequisite binding')
        unique[ref['path']]=deepcopy(ref)
    refs=tuple(unique[k] for k in sorted(unique))
    v.require(source_files()==sources,'training prerequisite source changed during replay')
    value=dict(version=VERSION,candidate_id=v.CANDIDATE,prerequisites_passed=True,recipe=recipe(),
        actor=deepcopy(winner.actor),program_pair=pair,all_six_programs_confirmed=True,
        confirmation=confirmed,reference=reference_buffer,source_files=sources,asset_manifest_sha256=v.digest(refs),
        next_stage='finite_20k_pair_plan_and_exact_seed_audit',environment_authority=False,optimizer_authority=False,
        seed_allocation_authority=False,new_environment_resets=0,Actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)
    return TrainingReadiness(value,refs,v.digest(value)).verify()
