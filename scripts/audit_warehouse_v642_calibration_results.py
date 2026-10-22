#!/usr/bin/env python3
"""Read-only tensor export and receipt recount after an observed 20k exit.

No model/optimizer/environment is constructed. This produces frozen assets
for a separately declared quality evaluation, never a model acceptance.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_calibration_pipeline as pipeline
from experiments.warehouse.v642_update_receipt_audit import audit_updates
from experiments.warehouse.v626_protocol import require,pointer,bound,file_sha
from backend.training.warehouse_v642_checkpoint_export import export_checkpoint_actor
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-completed-calibration-assets.v1'
STEPS=(4000,15000,16800,18600,20000)


def run(*,reference_directory,completed_ref,supervisor_exit_ref,output):
    completed=bound(completed_ref);exit_record=bound(supervisor_exit_ref)
    require(exit_record['exit_code']==0 and exit_record['controller_completed']==completed_ref
        and exit_record['observed_by']=='exec_tool_terminal_result',
        'an actual terminal exit0 bound to this controller is required')
    require(completed['version']==pipeline.VERSION and completed['calibration_training_finished'] is True
        and completed['model_acceptance_passed'] is False and completed['publishable'] is False
        and completed['pilot_opened'] is False,'completed calibration is not model acceptance')
    contract=bound(completed['controller_contract'])
    plan=pipeline.owner.load_seed_plan(reference_directory,contract['seed_plan'])
    require(pipeline.checked_contract(plan)==completed['controller_contract'],
        'registered calibration controller source differs')
    folder=Path(plan.ref['path']).parent/'calibration'
    require(Path(completed_ref['path']).resolve()==folder/'controller-completed.json'
        and not (folder/'failed.json').exists(),'failed or misplaced calibration cannot export completed assets')
    done=bound(completed['owner_completed']);claim=bound(done['claim']);descriptor=bound(done['descriptor'])
    require(completed['owner_completed']==pointer(folder/'completed.json')
        and done['claim']==pointer(pipeline.owner.claim_path()) and claim['plan']==plan.ref
        and claim['descriptor']==done['descriptor'] and descriptor['seed_plan']==plan.ref
        and descriptor['reference_actor']==plan.proposal.readiness.value['actor'],
        'owner completion, single-use claim or original Actor binding differs')
    initial={};snapshots={};reports={};inputs=[completed_ref,supervisor_exit_ref,completed['owner_completed'],done['claim'],done['descriptor']]
    for variant in pipeline.owner.seeds.VARIANTS:
        refs=completed['checkpoints'][variant]
        require(len(refs)==len(STEPS),'all registered checkpoints must be present')
        records=[]
        for step,ref in zip(STEPS,refs,strict=True):
            value=bound(ref);core=value['core'];owner=bound(value['owner'])
            require(Path(ref['path']).resolve()==folder/variant/'checkpoints'/f'{step:06d}'/'owner-state.json'
                and value['claim']==done['claim'] and value['source_files']==descriptor['source_files']
                and owner['variant']==variant and owner['claim']==done['claim']
                and core['processed_joint_steps']==step and core['variant']==variant
                and core['safe_batch_boundary'] is True and core['failed'] is False
                and value['reserved_joint_steps']==step
                and value['refits']==[s for s in (15000,16800,18600) if s<=step],
                'unsafe or mismatched checkpoint identity/step/refit history')
            inputs.extend([ref,value['training_state'],*value['updates']]);records.append(value)
        warmup,final=records[0],records[-1]
        initial[variant]=warmup['core']
        require(warmup['core']['actor_updates']==0
            and warmup['core']['accepted_program_updates']=={r:0 for r in ('robot_1','robot_2')},
            'critic-only warmup changed an Actor or accepted program feedback')
        require(final['updates']==done['updates'][variant],'final checkpoint update receipt list differs')
        for earlier in records[:-1]:
            require(earlier['updates']==final['updates'][:len(earlier['updates'])],
                'checkpoint optimizer histories are not prefixes of the final history')
        reports[variant]=audit_updates(final['updates'],variant=variant,expected_joint_steps=20000,core=final['core'])
        snapshots[variant]=(warmup,final)
    require(initial['raw']['actor_role_sha256']==initial['regularized']['actor_role_sha256']
        and initial['raw']['critic_sha256']==initial['regularized']['critic_sha256'],
        'paired critic-only warmup states differ')
    # Same seeds do not imply equal trained critics: divergent physical rollouts
    # after 4k are expected. Only the two initializations must match.
    events=[json.loads(line) for line in (folder/'owner-journal.jsonl').read_text().splitlines()]
    starts={e['variant']:e for e in events if e['kind']=='initialized'}
    require(set(starts)=={'raw','regularized'} and
        starts['raw']['roles']==starts['regularized']['roles']==initial['raw']['actor_role_sha256']
        and starts['raw']['critic_sha256']==starts['regularized']['critic_sha256'],
        'actual pair initialization evidence differs')
    require(sum(e['kind']=='initialized' for e in events)==2,'pair was initialized more than once')
    resets=(folder/'actual-resets.jsonl').read_text().splitlines()
    require(len(resets)==done['actual_reset_count']==sum(e['kind']=='actual_reset_bound' for e in events),
        'physical reset hook and owner reset counts differ')
    inputs.extend([pointer(folder/'owner-journal.jsonl'),pointer(folder/'actual-resets.jsonl')])
    for ref in inputs:require(pointer(ref['path'])==ref,'completed calibration input asset changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    exports={}
    anchor_roles=initial['raw']['actor_role_sha256']
    for variant,(warmup,final) in snapshots.items():
        exports[variant]={}
        for label,value,step in (('warmup',warmup,4000),('final',final,20000)):
            exported=export_checkpoint_actor(value['training_state'],output/f'{variant}-{label}.actor.npz',
                variant=variant,candidate_id=pipeline.owner.seeds.partitions.admission.v.CANDIDATE,
                expected_joint_steps=step)
            require(exported['role_sha256']==value['core']['actor_role_sha256'],
                'checkpoint tensors and live core role hashes differ')
            if label=='warmup':require(exported['role_sha256']==anchor_roles,'warmup changed original Actor tensors')
            else:require(all(exported['role_sha256'][r]!=anchor_roles[r] for r in anchor_roles),
                'a final Actor did not change from the common reference')
            exports[variant][label]=exported
    require(all(exports['raw']['final']['role_sha256'][r]!=exports['regularized']['final']['role_sha256'][r]
        for r in anchor_roles),'final raw and regularized roles must have independent hashes')
    for ref in inputs:require(pointer(ref['path'])==ref,'calibration inputs changed during read-only export')
    report=dict(version=VERSION,controller_completed=completed_ref,supervisor_exit=supervisor_exit_ref,
        source_sha256=file_sha(__file__),receipt_audit=reports,exports=exports,actual_reset_count=len(resets),
        numerical_training_completed=True,independent_quality_evaluation_required=True,
        independent_current_final_NN_program_evaluation_required=True,
        last_fit_programs_are_not_automatically_certified_for_final_NN=True,
        new_RNG_constructors=0,environment_resets=0,optimizer_steps=0,
        model_acceptance_passed=False,publishable=False,pilot_opened=False)
    write_new(output/'report.json',report);(output/'report.json').chmod(0o444)
    return pointer(output/'report.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('reference-directory','controller-completed','supervisor-exit','output'):
        parser.add_argument('--'+field,required=True,type=Path)
    args=parser.parse_args(argv)
    result=run(reference_directory=args.reference_directory,completed_ref=pointer(args.controller_completed),
        supervisor_exit_ref=pointer(args.supervisor_exit),output=args.output)
    print(json.dumps(result));return result


if __name__=='__main__':main()
