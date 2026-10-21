"""Source-registered one-attempt 20k pair orchestration, conditional on A/B.

The finite owner supplies all NN updates, resets and feedback capabilities.
This controller adds no action chooser, retries, continuation or release gate.
"""
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
import os
from . import v629_calibration_owner as owner
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training.warehouse_v629_evidence_provider import CurrentNNEvidenceProvider,PROVIDER_ROOT
from backend.training.warehouse_v629_program_refit import EXTRACTION_ROOT
from backend.training.warehouse_v625_actor_training import PPOConfig
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v629-paired-calibration-controller.v1'


def source_files():
    sources=owner.source_files()
    for name in ('experiments/warehouse/v629_calibration_pipeline.py','scripts/run_warehouse_v629_calibration.py'):
        sources[name]=file_sha(owner.ROOT/name)
    return dict(sorted(sources.items()))


def contract_value(plan,owner_contract):
    return dict(version=VERSION,seed_plan=plan.ref,source_files=source_files(),source_sha256=digest(source_files()),
        owner_source_contract=owner_contract,config=asdict(PPOConfig()),variants=list(owner.seeds.VARIANTS),
        raw_then_regularized_round_robin=True,checkpoint_joint_steps=[4000,15000,16800,18600,20000],
        automatic_retry=False,automatic_resume=False,training_authority=False,pilot_opened=False)


def checked_contract(plan):
    plan.verify();folder=Path(plan.ref['path']).parent;ref=pointer(folder/'controller-source-contract.json')
    value=bound(ref)
    require(value==contract_value(plan,owner.checked_sources(plan)),'calibration controller source/config contract differs')
    for name,sha in value['source_files'].items():
        snap=folder/'controller-source'/value['source_sha256']/name
        require(file_sha(snap)==sha and not snap.stat().st_mode&0o222,'calibration controller source snapshot changed')
    return ref


def prepare(*,reference_directory,plan_ref):
    plan=owner.load_seed_plan(reference_directory,plan_ref)  # Actual A/B first.
    folder=Path(plan.ref['path']).parent
    require(not (folder/'seed-ledger').exists(),'register complete controller source before calibration history audit')
    owner_contract=(owner.checked_sources(plan) if (folder/'owner-source-contract.json').exists()
        else owner.prepare_sources(reference_directory=reference_directory,plan_ref=plan_ref))
    value=contract_value(plan,owner_contract)
    for name,sha in value['source_files'].items():
        raw=(owner.ROOT/name).read_bytes();require(file_sha(owner.ROOT/name)==sha,'controller source changed preparing snapshot')
        target=folder/'controller-source'/value['source_sha256']/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        target.chmod(0o444)
    path=folder/'controller-source-contract.json';write_new(path,value);path.chmod(0o444)
    return checked_contract(plan)


def audit_seeds(*,reference_directory,plan_ref):
    plan=owner.load_seed_plan(reference_directory,plan_ref);contract=checked_contract(plan)
    clearance=owner.audit_source_bound_seeds(reference_directory=reference_directory,plan_ref=plan_ref)
    require(checked_contract(plan)==contract,'controller source changed during exact seed audit')
    path=Path(plan.ref['path']).with_name('controller-source-clearance.json')
    write_new(path,dict(version=VERSION,controller_source_contract=contract,owner_source_clearance=clearance,training_authority=False))
    path.chmod(0o444);return pointer(path)


def scheduled_refit(active,variant,programs):
    state=active.states[variant];step=state.joint_steps
    pool=active.collect_program_source(variant) if not active.refits[variant] else active.pools[variant]
    key=digest(active.owners[variant]);folder=active.folder/variant/'refit-targets'/str(step)
    folder.mkdir(parents=True,exist_ok=False)
    actor_ref,_=owner.frozen_export(state,folder/'current-NN.npz',variant)
    targets=owner.observed.relabel_current_NN(pool,actor_ref,owner.observed.TARGET_ROOT/key/f'refit-{step}')
    result,binding=owner.observed.refit_observed_pair_at_boundary(reference_directory=active.reference_directory,
        owner_ref=active.owners[variant],state=state,observed=pool,targets_ref=targets,program_refs=programs,
        output=EXTRACTION_ROOT/key/f'step-{step}',variant=variant,epoch_id=f'scheduled-{step}',config=active.config)
    provider=active.providers.get(variant)
    if variant=='regularized' and provider is None:
        provider=CurrentNNEvidenceProvider(reference_directory=active.reference_directory,owner_ref=active.owners[variant],
            output=PROVIDER_ROOT/key,pair_seed=state.initialization_seed,config=active.config)
    active.record_refit(variant,binding,evidence_provider=provider)
    return {role:result['roles'][role]['program'] for role in ('robot_1','robot_2')}


def drive(active,initial_programs,verify):
    """All physical work is delegated to the already claimed finite owner."""
    require(active.config==PPOConfig(),'controller is only registered for the fixed 20k calibration budget')
    programs={variant:deepcopy(initial_programs) for variant in owner.seeds.VARIANTS}
    for variant in owner.seeds.VARIANTS:verify();active.initialize(variant)
    checkpoints={variant:[] for variant in owner.seeds.VARIANTS}
    while any(state.joint_steps<20000 for state in active.states.values()):
        for variant in owner.seeds.VARIANTS:
            state=active.states[variant]
            if state.joint_steps>=20000:continue
            verify();active.collect_rollout(variant);active.update(variant)
            if state.joint_steps in active.config.extraction_steps:
                verify();programs[variant]=scheduled_refit(active,variant,programs[variant])
            if state.joint_steps in (4000,15000,16800,18600,20000):
                verify();checkpoints[variant].append(active.checkpoint(variant))
    return checkpoints


def run(*,reference_directory,plan_ref):
    plan=owner.load_seed_plan(reference_directory,plan_ref);contract=checked_contract(plan)
    file=Path(plan.ref['path']).with_name('controller-source-clearance.json');clearance_ref=pointer(file)
    expected=dict(version=VERSION,controller_source_contract=contract,
        owner_source_clearance=pointer(file.with_name('owner-source-clearance.json')),training_authority=False)
    require(bound(clearance_ref)==expected,'exact seed audit does not bind the registered controller source')
    def verify():
        require(checked_contract(plan)==contract and pointer(file)==clearance_ref,'controller or pre-execution clearance changed')
    verify()
    with owner.claimed_owner(reference_directory=reference_directory,plan_ref=plan_ref) as active:
        active.event('controller_bound',contract=contract,clearance=clearance_ref)
        programs={role:row['program'] for role,row in plan.proposal.readiness.value['program_pair'].items()}
        checkpoints=drive(active,programs,verify)
        verify();active.event('controller_pair_steps_finished',checkpoints=checkpoints,
            task_quality_and_program_calibration_evaluation_still_required=True)
    completed_ref=pointer(active.folder/'completed.json')
    result=dict(version=VERSION,controller_contract=contract,controller_clearance=clearance_ref,owner_completed=completed_ref,
        checkpoints=checkpoints,calibration_training_finished=True,model_acceptance_passed=False,
        quality_and_program_calibration_evaluation_still_required=True,pilot_opened=False,publishable=False)
    path=active.folder/'controller-completed.json';write_new(path,result);path.chmod(0o444)
    return pointer(path)
