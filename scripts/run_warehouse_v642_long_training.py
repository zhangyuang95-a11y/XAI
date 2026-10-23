#!/usr/bin/env python3
"""Finite single-attempt long pair, with explicit budget-locked formal entry.

No stage opens pilot, selects a policy, issues a budget lock, or authorizes a
sealed experiment. Formal training additionally requires the real development
budget-lock reader before other inputs. The supervisor records a real wait;
the existence of a result file is never treated as a successful process exit.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_long_training_owner as owner
from experiments.warehouse.v626_protocol import pointer,bound,require
from scripts.run_warehouse_v625 import write_new

SUPERVISOR_VERSION='warehouse-v642-budget-locked-long-supervisor.v2'


def readonly_new(path,value):
    write_new(path,value);Path(path).chmod(0o444);return pointer(path)


def supervise(*,reference_directory,calibration_precheck_ref,plan_ref,phase='development',formal_budget_lock_ref=None):
    plan=owner.load_seed_plan(reference_directory,calibration_precheck_ref,plan_ref,phase=phase,
        formal_budget_lock_ref=formal_budget_lock_ref)
    # This independently checks admission and the exact current source/history
    # before launching the sole worker; the worker repeats them before claim.
    owner.checked_sources(plan);owner.seeds.checked_clearance(plan,execution=True)
    require(not owner.claim_path(plan).exists(),'long pair already consumed its single attempt')
    folder=Path(plan.ref['path']).parent
    cli=pointer(Path(__file__).resolve());interpreter=str(Path(sys.executable).resolve())
    command=[interpreter,cli['path'],'--run','--reference-directory',str(Path(reference_directory).resolve()),
        '--calibration-precheck',str(Path(calibration_precheck_ref['path']).resolve()),'--plan',str(Path(plan_ref['path']).resolve())]
    if phase=='formal':command.extend(['--phase','formal','--formal-budget-lock',str(Path(formal_budget_lock_ref['path']).resolve())])
    started=readonly_new(folder/'long-supervisor-started.json',dict(version=SUPERVISOR_VERSION,
        observed_by='source_bound_subprocess_launch',seed_plan=plan.ref,calibration_precheck=calibration_precheck_ref,
        phase=phase,development_budget_lock=plan.value.get('development_budget_lock'),formal_admission=plan.value.get('formal_admission'),
        command=command,supervisor_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        supervisor_source=cli,log_path=str(folder/'long-run.log'),pilot_opened=False))
    launch=bound(started);exit_ref=None
    try:
        with (folder/'long-run.log').open('xb') as log:
            worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=False)
            exit_code=worker.wait()  # Actual OS-observed wait, never inferred.
            require(type(exit_code) is int,'subprocess wait did not return an actual integer exit code')
            log.flush();os.fsync(log.fileno())
        (folder/'long-run.log').chmod(0o444)
        value=dict(version=SUPERVISOR_VERSION,observed_by='source_bound_subprocess_wait',seed_plan=plan.ref,
            phase=phase,development_budget_lock=plan.value.get('development_budget_lock'),formal_admission=plan.value.get('formal_admission'),
            started=started,command=command,worker_pid=worker.pid,supervisor_pid=os.getpid(),
            started_utc=launch['started_utc'],finished_utc=datetime.now(timezone.utc).isoformat(),supervisor_source=cli,
            exit_code=exit_code,log=pointer(folder/'long-run.log'),model_acceptance_passed=False,pilot_opened=False)
        if exit_code==0:
            try:
                result=pointer(Path(plan.value['execution_output'])/'controller-completed.json');r=bound(result)
                completion=pointer(Path(plan.value['execution_output'])/'completed.json');c=bound(completion)
                require(r['version']==owner.VERSION and r['seed_plan']==plan.ref and r['owner_completed']==completion
                    and r['numerical_training_completed'] is True and r['publishable'] is False
                    and c['version']==owner.VERSION and c['training_finished'] is True
                    and bound(c['descriptor'])['seed_plan']==plan.ref and c['publishable'] is False,
                    'exit-zero long worker lacks bound, complete, nonpublication receipts')
                if phase=='formal':
                    descriptor=bound(c['descriptor'])
                    require(all(row['phase']=='formal' and row['development_budget_lock']==formal_budget_lock_ref
                        and row['formal_admission']==plan.value['formal_admission'] for row in (r,c,descriptor)),
                        'formal worker result/descriptor does not bind the same actual budget lock')
                value.update(result=result,completed=completion)
            except BaseException as error:
                # Preserve true exit 0, but explicitly reject missing or invalid
                # result binding; never manufacture a successful completion.
                value['result_binding_error']=dict(type=type(error).__name__,message=str(error))
        exit_ref=readonly_new(folder/'long-supervisor-exit.json',value)
        require(exit_code==0 and 'result_binding_error' not in value,'long worker failed or its exit-zero completion did not bind')
        return exit_ref
    except BaseException as error:
        readonly_new(folder/'long-supervisor-failed.json',dict(version=SUPERVISOR_VERSION,started=started,
            actual_exit=exit_ref,error_type=type(error).__name__,error=str(error),automatic_retry=False,
            model_acceptance_passed=False,pilot_opened=False))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-sources','audit-seeds','run','supervise'):stages.add_argument('--'+name,action='store_true')
    parser.add_argument('--reference-directory',required=True,type=Path)
    parser.add_argument('--calibration-precheck',required=True,type=Path)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--phase',choices=('development','formal'),default='development')
    parser.add_argument('--formal-budget-lock',type=Path)
    args=parser.parse_args(argv)
    formal_ref=None
    if args.phase=='formal':
        require(args.formal_budget_lock is not None,'formal entry requires its real development budget lock before other inputs')
        formal_ref=pointer(args.formal_budget_lock)
        owner.seeds.formal.read_admission(formal_ref)  # Before hashing calibration or plan.
    else:require(args.formal_budget_lock is None,'development entry cannot infer formal permission from a lock argument')
    precheck=pointer(args.calibration_precheck)
    # The actual independent calibration must replay before even hashing a
    # caller's long seed-plan file. `passed: true` JSON is not a capability.
    owner.seeds.protocol.register_preparation(precheck).verify()
    plan=pointer(args.plan)
    values=dict(reference_directory=args.reference_directory,calibration_precheck_ref=precheck,plan_ref=plan)
    if args.phase=='formal':values.update(phase='formal',formal_budget_lock_ref=formal_ref)
    call=(owner.prepare_sources if args.prepare_sources else owner.audit_source_bound_seeds if args.audit_seeds
          else supervise if args.supervise else owner.run)
    result=call(**values)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


if __name__=='__main__':main()
