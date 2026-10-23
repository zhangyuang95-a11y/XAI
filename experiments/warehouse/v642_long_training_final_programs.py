"""Bind fixed programs to a genuinely completed long-budget development pair.

No new scenes, Actor updates or topology selection. This is a different
protocol from the spent 20k finalization claim and cannot read its receipts.
"""
from copy import deepcopy
from pathlib import Path

from . import v642_final_programs as shared
from .v626_protocol import require, pointer, bound, digest, file_sha
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-long-pair-final-programs.v1'
ROOT=shared.q.ROOT
OUTPUT=ROOT/'output/warehouse_v642/long-training-final-programs'
ROLES=shared.ROLES
VARIANTS=('raw','regularized')
FILES=('experiments/warehouse/v642_long_training_final_programs.py',
       'tests/test_warehouse_v642_long_training_final_programs.py',
       'docs/warehouse_v642_long_training_final_programs.md',
       'experiments/warehouse/v642_long_stage_supervisor.py',
       'scripts/finalize_warehouse_v642_long_programs.py',
       'scripts/supervise_warehouse_v642_long_stage.py',
       'tests/test_warehouse_v642_long_stage_supervisor.py')


def source_files():
    from . import v642_long_training_assets as assets
    files=assets.source_files()
    files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def identity(plan):
    p=plan.value
    require(p['phase']=='development' and p['joint_step_budget'] in (200000,300000,400000),
            'long-program finalization currently admits development only')
    require(p['stream_id'] in ('development-0','development-1','development-2'),'unknown development stream')
    return f'{p["phase"]}-{p["stream_id"]}-{p["joint_step_budget"]}'


def last_programs(events,*,variant,owner_ref,source_ref,config):
    fits=[e for e in events if e['kind']=='scheduled_refit_bound' and e['variant']==variant]
    require([e['step'] for e in fits]==list(config.extraction_steps),
            'all and only registered long-budget refits are required')
    binding_ref=fits[-1]['binding'];binding=bound(binding_ref);result=bound(binding['extraction'])
    require(binding['schema']=='warehouse-v642-observed-refit-source-binding.v1'
            and binding['owner']==owner_ref and binding['observed_source']==source_ref
            and binding['source_files']==shared.pools.source_files()
            and result['identity']['owner']==owner_ref and result['identity']['variant']==variant
            and result['identity']['step']==config.extraction_steps[-1]
            and set(result['roles'])==set(ROLES), 'last long-program source or budget differs')
    return {role:result['roles'][role]['program'] for role in ROLES},[binding_ref,binding['extraction']]


def fit_role(program_ref,parts,*,role,target_sha,language,folder,plan_ref):
    """One bounded fixed-language fit; all calls and reproofs stay on ledger."""
    encoded=bound(program_ref)
    program=shared.Program.from_dict(encoded,encoded['target_actor_role_sha256'])
    require(program.role==role and shared.language_lock(program)==language,
            'long final fit changed the confirmed role language')
    original=deepcopy(program);protected=shared.protected_indices(program)
    folder.mkdir(parents=True,exist_ok=False)
    budget=shared.Budget(folder/'checkpoint.json',dict(plan=plan_ref,role=role),shared.limits(program),1200)
    budget.phase='final'
    try:
        with budget.session.hard_deadline():
            final,measurements=shared.complete_program(program,parts,budget,refit=True,
                                                       target_actor_role_sha256=target_sha)
            require(shared.language_lock(final)==shared.language_lock(original)
                    and all(shared.payload(final.leaves[i])==shared.payload(original.leaves[i]) for i in protected),
                    'long final fit modified a split, support, witness or protected output')
            current=shared.current_report(final,parts)
            require(current['validation']=={k:measurements['validation'][k] for k in ('feedback','explanation')}
                    and set(current['feedback_leaves'])==set(final.feedback_leaves)
                    and set(current['explanation_leaves'])==set(final.explanation_leaves),
                    'long final qualification does not reproduce')
            for part in parts.values():
                require(all(shared.np.array_equal(x,y) for x,y in zip(
                    final.distributions(part['x'],part['m']),shared.forward(final,part['x'],part['m']),strict=True)),
                    'independent long final program forward differs')
            path=folder/'program.json';write_new(path,final.to_dict());path.chmod(0o444)
            require(shared.Program.from_dict(bound(pointer(path)),target_sha).to_dict()==final.to_dict(),
                    'stored long final program or legality certificate differs')
            budget.checkpoint(dict(program=final.to_dict(),measurements=measurements,current=current),status='completed')
        solver=budget.summary()
        require(shared.ledger(folder,dict(solver=solver))==solver['counts'] and solver['counts']['node']==0,
                'long final fit performed unregistered structure search or unaccounted calls')
        result=dict(status='completed',program=pointer(path),previous_program=program_ref,
                    current=current,metrics=measurements,solver=solver,
                    target_actor_role_sha256=target_sha,program_precheck_passed=current['gate']['passed'],
                    checkpoint=pointer(folder/'checkpoint.json'),ledger=pointer(folder/'checkpoint.json.ledger.jsonl'))
        return result,[program_ref,deepcopy(program.protection['parent']),result['program'],result['checkpoint'],result['ledger']]
    finally:
        budget.session.close()


def run(*,completed_assets_ref,reference_directory):
    from . import v642_long_training_assets as assets
    fixed=assets.read_assets(completed_assets_ref)
    context=assets.finalization_context(completed_assets_ref,reference_directory)
    plan=context['plan'];name=identity(plan);folder=OUTPUT/name
    claim=OUTPUT/'claims'/(name+'.json')
    require(not claim.exists(),'long final programs are once-only even after interruption')
    sources=source_files()
    for i,e in enumerate(context['events']):
        require(e['sequence']==i,'long training journal has missing or reordered events')
    folder.mkdir(parents=True,exist_ok=False)
    declaration=dict(version=VERSION,completed_assets=completed_assets_ref,seed_plan=plan.ref,
        phase=plan.value['phase'],stream_id=plan.value['stream_id'],joint_step_budget=plan.config.joint_step_budget,
        source_files=sources,fixed_language=True,source_evidence_only=True,new_NN_updates=0,
        new_environment_resets=0,program_never_selects_actions=True,model_acceptance_passed=False,pilot_opened=False)
    for path,sha in sources.items():
        raw=(ROOT/path).read_bytes();require(file_sha(ROOT/path)==sha,'long final source changed before claim')
        destination=folder/'source'/path;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw);destination.chmod(0o444)
    write_new(folder/'plan.json',declaration);(folder/'plan.json').chmod(0o444)
    claim.parent.mkdir(parents=True,exist_ok=True)
    write_new(claim,dict(version=VERSION,plan=pointer(folder/'plan.json'),single_use_even_if_interrupted=True,pilot_opened=False))
    claim.chmod(0o444)
    refs=[*fixed['inputs'],pointer(folder/'plan.json'),pointer(claim)];results={}
    try:
        for variant in VARIANTS:
            owner=context['owners'][variant];source=context['program_sources'][variant]
            previous,bindings=last_programs(context['events'],variant=variant,owner_ref=owner,
                                           source_ref=source,config=plan.config)
            observed=shared.pools.audit_observed_pool(source,reference_directory=reference_directory)
            targets=shared.pools.relabel_current_NN(observed,fixed['actors'][variant],
                shared.pools.TARGET_ROOT/'long-finalization'/digest(dict(claim=pointer(claim),variant=variant)))
            shared.pools.checked_targets(observed,targets)
            cached=observed._replayed_target_cache;cached.verify(observed)
            refs.extend([owner,source,*bindings,*observed.refs,targets,*cached.asset_refs]);results[variant]={}
            for role in ROLES:
                result,inputs=fit_role(previous[role],cached.parts[role],role=role,
                    target_sha=fixed['actor_role_sha256'][variant][role],
                    language=plan.readiness.value['program_pair'][role]['language'],
                    folder=folder/variant/role,plan_ref=pointer(folder/'plan.json'))
                results[variant][role]=result;refs.extend(inputs)
        require(source_files()==sources,'long finalization sources changed')
        for ref in refs:require(pointer(ref['path'])==ref,'long finalization input bytes changed')
        result=dict(version=VERSION,plan=pointer(folder/'plan.json'),claim=pointer(claim),source_files=sources,
            completed_assets=completed_assets_ref,seed_plan=plan.ref,input_asset_refs=refs,
            actors=fixed['actors'],actor_role_sha256=fixed['actor_role_sha256'],roles=results,
            program_precheck_passed=all(row['program_precheck_passed'] for roles in results.values() for row in roles.values()),
            new_NN_updates=0,new_environment_resets=0,fresh_task_and_program_evaluation_required=True,
            model_acceptance_passed=False,pilot_opened=False)
        write_new(folder/'result.json',result);(folder/'result.json').chmod(0o444)
        write_new(folder/'completed.json',dict(result=pointer(folder/'result.json')));(folder/'completed.json').chmod(0o444)
        return pointer(folder/'result.json')
    except BaseException as error:
        write_new(folder/'failed.json',dict(version=VERSION,error_type=type(error).__name__,error=str(error),
            completed_roles=results,single_use_claim_retained=True,pilot_opened=False))
        raise


def read_completed(ref,completed_assets_ref):
    from . import v642_long_training_assets as assets
    from .v642_long_stage_supervisor import verify_execution
    require(type(ref) is dict,'completed long final-program receipt is required')
    r=bound(ref);fixed=assets.read_assets(completed_assets_ref)
    plan=bound(r['plan']);seed_plan=bound(r['seed_plan'])
    name=identity(type('Identity',(),{'value':seed_plan})());folder=OUTPUT/name
    claim=OUTPUT/'claims'/(name+'.json')
    require(ref==pointer(folder/'result.json') and r['version']==VERSION
        and r['completed_assets']==completed_assets_ref and r['source_files']==source_files()
        and r['actors']==fixed['actors'] and r['actor_role_sha256']==fixed['actor_role_sha256']
        and r['program_precheck_passed'] is True and r['new_NN_updates']==r['new_environment_resets']==0
        and r['model_acceptance_passed'] is r['pilot_opened'] is False
        and r['plan']==pointer(folder/'plan.json') and r['claim']==pointer(claim)
        and bound(r['claim'])==dict(version=VERSION,plan=r['plan'],single_use_even_if_interrupted=True,pilot_opened=False)
        and plan['version']==VERSION and plan['seed_plan']==r['seed_plan']
        and plan['completed_assets']==completed_assets_ref and plan['source_files']==r['source_files']
        and plan['phase']==seed_plan['phase'] and plan['stream_id']==seed_plan['stream_id']
        and plan['joint_step_budget']==seed_plan['joint_step_budget']
        and bound(pointer(folder/'completed.json'))==dict(result=ref)
        and not (folder/'failed.json').exists(),'long final-program identity, source or actual NN binding changed')
    execution_ref=pointer(folder/'supervisor-exit.json')
    inputs=[*fixed['inputs'],*r['input_asset_refs'],r['plan'],r['claim'],
            *verify_execution(execution_ref,ref,stage='long-final-programs')]
    require(set(r['roles'])==set(VARIANTS),'both long variants require final programs')
    programs={}
    for variant in VARIANTS:
        require(set(r['roles'][variant])==set(ROLES),'both long roles require final programs')
        programs[variant]={}
        for role in ROLES:
            row=r['roles'][variant][role];sha=fixed['actor_role_sha256'][variant][role]
            program=shared.Program.from_dict(bound(row['program']),sha)
            require(row['status']=='completed' and row['program_precheck_passed'] is True
                and row['target_actor_role_sha256']==program.target_actor_role_sha256==sha and program.role==role
                and shared.gate(row['current']['validation'])==row['current']['gate']
                and row['current']['gate']['passed'] is True
                and set(row['current']['feedback_leaves'])==set(program.feedback_leaves)
                and set(row['current']['explanation_leaves'])==set(program.explanation_leaves)
                and row['current']['validation']=={k:row['metrics']['validation'][k] for k in ('feedback','explanation')}
                and row['solver']['limits']==shared.limits(program) and row['solver']['counts']['node']==0
                and shared.ledger(Path(row['program']['path']).parent,dict(solver=row['solver']))==row['solver']['counts']
                and all(0<=row['solver']['counts'][k]<=row['solver']['limits'][k] for k in ('leaf','node','proof')),
                'long final program qualification, target, legality or budget changed')
            programs[variant][role]=row['program'];inputs.extend([row['program'],row['checkpoint'],row['ledger']])
    for path,sha in r['source_files'].items():
        saved=folder/'source'/path
        require(file_sha(saved)==sha and not saved.stat().st_mode&0o222,'long final source snapshot changed')
    for item in inputs:
        path=Path(item['path'])
        require(pointer(path)==item and not path.is_symlink() and not path.stat().st_mode&0o222,
                'long final evidence bytes changed or writable')
    return dict(report=ref,programs=programs,inputs=inputs)
