"""Finalize fixed-language programs against the actual completed20k NN.

This uses only the owner's previously collected program-only episodes. It
adds no NN update or new scene and cannot replace any independent audit.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from . import v642_quality_collection as q
from .v626_protocol import require,pointer,bound,digest,file_sha
from backend.training import warehouse_v642_observed_pool as pools
from backend.training.warehouse_v642_training_program import complete_program,language_lock,protected_indices,limits
from backend.training.warehouse_v642_program import Program,payload
from backend.training.warehouse_v626_budget import Budget
from backend.training.warehouse_v629_online_evidence import current_report,gate
from scripts.audit_warehouse_v642_program import forward
from scripts.audit_warehouse_v628 import ledger
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-post-training-fixed-language-programs.v1'
OUTPUT=q.ROOT/'output/warehouse_v642/training-final-programs'
ROLES=('robot_1','robot_2')


def claim_path():return q.training.owner.seeds.partitions.admission.v.OUTPUT/'calibration-final-programs.claim.json'


def source_files():
    result=q.source_files()
    for name in ('experiments/warehouse/v642_final_programs.py','tests/test_warehouse_v642_final_programs.py',
        'docs/warehouse_v642_final_program_plan.md'):
        result[name]=file_sha(q.ROOT/name)
    return dict(sorted(result.items()))


def last_programs(journal,*,variant,owner_ref,source_ref):
    events=[e for e in journal if e['kind']=='scheduled_refit_bound' and e['variant']==variant]
    require([e['step'] for e in events]==[15000,16800,18600], 'all three scheduled refits must precede finalization')
    binding_ref=events[-1]['binding'];b=bound(binding_ref);r=bound(b['extraction'])
    require(b['schema']=='warehouse-v642-observed-refit-source-binding.v1'
        and b['owner']==owner_ref and b['observed_source']==source_ref and b['source_files']==pools.source_files()
        and r['identity']['owner']==owner_ref and r['identity']['variant']==variant and r['identity']['step']==18600
        and set(r['roles'])==set(ROLES), 'last-fit role/owner/source identity differs')
    return {role:r['roles'][role]['program'] for role in ROLES},[binding_ref,b['extraction']]


def run(*,completed_assets_ref,reference_directory,run_id):
    fixed=q.assets(completed_assets_ref)  # Before output, fitting or replay.
    report=bound(completed_assets_ref)
    if report.get('version')==q.recovery_assets.VERSION:
        context=q.recovery_assets.finalization_context(completed_assets_ref,reference_directory)
        plan=context['plan'];events=context['events'];journal_refs=context['journal_refs']
        owners=context['owners'];program_sources=context['program_sources']
        require(all(ref in report['input_asset_refs'] for ref in journal_refs),
            'recovered assets do not bind both optimizer/refit journals')
    else:
        controller=bound(report['controller_completed']);contract=bound(controller['controller_contract'])
        plan=q.training.owner.load_seed_plan(reference_directory,contract['seed_plan'])
        require(q.training.checked_contract(plan)==controller['controller_contract'],'training source/plan changed')
        folder=Path(report['controller_completed']['path']).parent
        journal_refs=[pointer(folder/'owner-journal.jsonl')]
        require(journal_refs[0] in report['input_asset_refs'],'completed training report does not bind its refit journal')
        events=[json.loads(line) for line in Path(journal_refs[0]['path']).read_text().splitlines()]
        owners={v:pointer(folder/v/'owner.json') for v in ('raw','regularized')}
        program_sources={v:pointer(folder/v/'program-source/source.json') for v in ('raw','regularized')}
    q.training.owner.seeds.confirmation.run_name(run_id)
    for i,event in enumerate(events):require(event['sequence']==i,'training journal has missing or reordered events')
    sources=source_files();source_sha=digest(sources);output=OUTPUT/run_id
    claim=claim_path()
    require(not claim.exists(),'post-training program finalization is once-only even after failure')
    output.mkdir(parents=True,exist_ok=False)
    identity=dict(version=VERSION,completed_assets=completed_assets_ref,seed_plan=plan.ref,
        source_files=sources,source_sha256=source_sha,variants=['raw','regularized'],roles=list(ROLES),
        final_joint_steps=20000,source_episode_inventory=plan.proposal.value['program_evidence_inventory'],
        fixed_topology_supports_witnesses_and_protected_payloads=True,per_role_seconds=1200,
        new_NN_gradients=0,new_environment_resets=0,new_independent_randomness=False,pilot_opened=False)
    for name,sha in sources.items():
        raw=(q.ROOT/name).read_bytes();require(file_sha(q.ROOT/name)==sha,'finalization source changed before claim')
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444)
    write_new(output/'plan.json',identity);(output/'plan.json').chmod(0o444)
    write_new(claim,dict(version=VERSION,plan=pointer(output/'plan.json'),output=str(output),single_use_even_if_failed=True,pilot_opened=False))
    claim.chmod(0o444);results={};refs=[*fixed['inputs'],*journal_refs,pointer(output/'plan.json'),pointer(claim)]
    try:
        for variant in ('raw','regularized'):
            owner_ref=owners[variant];source_ref=program_sources[variant]
            source=bound(source_ref)
            require(source['owner']==owner_ref and source['inventory']==plan.proposal.value['program_evidence_inventory'],
                'finalization may use only this owner\'s admitted program-only episode inventory')
            previous,bindings=last_programs(events,variant=variant,owner_ref=owner_ref,source_ref=source_ref)
            observed=pools.audit_observed_pool(source_ref,reference_directory=reference_directory)
            targets_ref=pools.relabel_current_NN(observed,fixed['actors'][variant],
                pools.TARGET_ROOT/digest(dict(finalization=pointer(claim),variant=variant)))
            targets,_=pools.checked_targets(observed,targets_ref);cached=observed._replayed_target_cache
            cached.verify(observed);refs.extend([owner_ref,source_ref,*bindings,*observed.refs,targets_ref,*cached.asset_refs])
            results[variant]={}
            for role in ROLES:
                encoded=bound(previous[role]);program=Program.from_dict(encoded,encoded['target_actor_role_sha256'])
                require(program.role==role and language_lock(program)==plan.proposal.readiness.value['program_pair'][role]['language'],
                    'finalization cannot change the confirmed fixed program language')
                original=deepcopy(program);protected=protected_indices(program)
                refs.append(deepcopy(program.protection['parent']))
                rolefolder=output/variant/role;rolefolder.mkdir(parents=True)
                budget=Budget(rolefolder/'checkpoint.json',dict(plan=pointer(output/'plan.json'),variant=variant,role=role),limits(program),1200)
                budget.phase='final'
                try:
                    with budget.session.hard_deadline():
                        final,measurements=complete_program(program,cached.parts[role],budget,refit=True,
                            target_actor_role_sha256=fixed['actor_role_sha256'][variant][role])
                        require(language_lock(final)==language_lock(original)
                            and all(payload(final.leaves[i])==payload(original.leaves[i]) for i in protected),
                            'finalization changed a fixed split/support/witness/protected output')
                        current=current_report(final,cached.parts[role])
                        require(current['validation']=={key:measurements['validation'][key] for key in ('feedback','explanation')}
                            and set(current['feedback_leaves'])==set(final.feedback_leaves)
                            and set(current['explanation_leaves'])==set(final.explanation_leaves),'final current-NN qualification differs')
                        for split,a in cached.parts[role].items():
                            require(all(np.array_equal(x,y) for x,y in zip(final.distributions(a['x'],a['m']),forward(final,a['x'],a['m']),strict=True)),
                                'independent final program routing/distribution forward differs')
                        path=rolefolder/'program.json';write_new(path,final.to_dict());path.chmod(0o444)
                        require(Program.from_dict(bound(pointer(path)),fixed['actor_role_sha256'][variant][role]).to_dict()==final.to_dict(),
                            'stored final current-NN program/certificate changed')
                        budget.checkpoint(dict(program=final.to_dict(),measurements=measurements,current=current),status='completed')
                    solver=budget.summary()
                    require(ledger(rolefolder,dict(solver=solver))==solver['counts'] and solver['counts']['node']==0,
                        'post-training fit ledger differs or performed a topology search')
                    result=dict(status='completed',program=pointer(path),previous_program=previous[role],current=current,metrics=measurements,
                        solver=solver,calibration_program_precheck_passed=current['gate']['passed'],
                        target_actor_role_sha256=fixed['actor_role_sha256'][variant][role],
                        checkpoint=pointer(rolefolder/'checkpoint.json'),ledger=pointer(rolefolder/'checkpoint.json.ledger.jsonl'))
                    refs.extend([previous[role],result['program'],result['checkpoint'],result['ledger']])
                finally:budget.session.close()
                results[variant][role]=result
                print(json.dumps(dict(final_program_variant=variant,role=role,passed=result['calibration_program_precheck_passed'])),flush=True)
        require(source_files()==sources,'final program source changed')
        for ref in refs:require(pointer(ref['path'])==ref,'final program prerequisite changed during processing')
        value=dict(version=VERSION,plan=pointer(output/'plan.json'),claim=pointer(claim),source_files=sources,input_asset_refs=refs,
            completed_assets=completed_assets_ref,actors=fixed['actors'],actor_role_sha256=fixed['actor_role_sha256'],roles=results,
            calibration_program_precheck_passed=all(r['calibration_program_precheck_passed'] for roles in results.values() for r in roles.values()),
            scope='existing_training_program_pool_only_not_fresh_confirmation_or_sealed_audit',
            counterfactual_validation_actions_not_executed=True,new_NN_updates=0,new_environment_resets=0,
            new_independent_randomness=False,fresh_task_quality_required=True,model_acceptance_passed=False,pilot_opened=False)
        write_new(output/'result.json',value);(output/'result.json').chmod(0o444)
        write_new(output/'completed.json',dict(result=pointer(output/'result.json')));(output/'completed.json').chmod(0o444)
        return pointer(output/'result.json')
    except BaseException as error:
        write_new(output/'failed.json',dict(version=VERSION,error_type=type(error).__name__,error=str(error),completed_roles=results,
            single_use_claim_retained=True,automatic_retry=False,pilot_opened=False));raise


def read_completed(ref,completed_assets_ref):
    """Load fixed programs only after observed successful finalization."""
    require(isinstance(ref,dict),'completed final-NN program report is required before fresh collection')
    r=bound(ref);folder=Path(ref['path']).parent
    require(folder.is_relative_to(OUTPUT.resolve()) and folder!=OUTPUT.resolve()
        and Path(ref['path']).name=='result.json' and not (folder/'failed.json').exists()
        and r['version']==VERSION and r['source_files']==source_files()
        and r['completed_assets']==completed_assets_ref
        and r['calibration_program_precheck_passed'] is True
        and r['new_NN_updates']==r['new_environment_resets']==0
        and r['model_acceptance_passed'] is r['pilot_opened'] is False,
        'missing, failed, changed or differently bound final programs')
    p=bound(r['plan']);c=bound(r['claim']);execution_ref=pointer(folder/'supervisor-exit.json');execution=bound(execution_ref)
    require(r['plan']==pointer(folder/'plan.json') and r['claim']==pointer(claim_path())
        and c['plan']==r['plan'] and c['output']==str(folder)
        and p['version']==VERSION and p['completed_assets']==completed_assets_ref
        and p['source_files']==r['source_files'] and p['source_sha256']==digest(r['source_files'])
        and bound(pointer(folder/'completed.json'))==dict(result=ref)
        and execution['result']==ref and execution['exit_code']==0
        and execution['observed_by']=='exec_tool_terminal_result',
        'final-program claim, completion or observed terminal exit differs')
    fixed=q.assets(completed_assets_ref)
    require(r['actors']==fixed['actors'] and r['actor_role_sha256']==fixed['actor_role_sha256']
        and set(r['roles'])=={'raw','regularized'},'final programs belong to another frozen NN pair')
    programs={};inputs=[ref,r['plan'],r['claim'],execution_ref,pointer(folder/'completed.json'),*r['input_asset_refs']]
    for variant in ('raw','regularized'):
        require(set(r['roles'][variant])==set(ROLES),'both final program roles required')
        programs[variant]={}
        for role in ROLES:
            item=r['roles'][variant][role];role_sha=fixed['actor_role_sha256'][variant][role]
            program=Program.from_dict(bound(item['program']),role_sha)
            require(item['status']=='completed' and item['calibration_program_precheck_passed'] is True
                and item['target_actor_role_sha256']==program.target_actor_role_sha256==role_sha
                and program.role==role and gate(item['current']['validation'])==item['current']['gate']
                and item['current']['gate']['passed'] is True
                and set(item['current']['feedback_leaves'])==set(program.feedback_leaves)
                and set(item['current']['explanation_leaves'])==set(program.explanation_leaves)
                and item['current']['validation']=={key:item['metrics']['validation'][key] for key in ('feedback','explanation')}
                and item['solver']['limits']==limits(program) and item['solver']['counts']['node']==0
                and all(0<=item['solver']['counts'][key]<=item['solver']['limits'][key] for key in ('leaf','node','proof')),
                'final program target hash, qualification, gate or fixed-search budget differs')
            programs[variant][role]=item['program'];inputs.extend([item['program'],item['checkpoint'],item['ledger']])
    for name,sha in r['source_files'].items():
        path=folder/'source'/name;require(file_sha(path)==sha and not path.stat().st_mode&0o222,'final program source snapshot changed')
    for item in inputs:require(pointer(item['path'])==item,'final program evidence asset changed')
    return dict(report=ref,programs=programs,inputs=inputs)
