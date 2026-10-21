#!/usr/bin/env python3
"""Finite v628 preparation, collection and 18-unit capacity comparison."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v,v628_data as data
from scripts.run_warehouse_v625 import write_new
from scripts.run_warehouse_v626 import checked_plan as checked_parent,stop_workers
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from scripts.verify_warehouse_v625_native import real_deadline
from backend.training.warehouse_v628_program import from_parent,Program,ParentProgram,complete_program,protected_indices
from backend.training.warehouse_v628_search import Budget,Context,optimize
from backend.training.warehouse_v628_certificate import structure_info
from backend.training.warehouse_v625_checkpoint import BudgetExhausted,DeadlineExceeded


def prepare():
    _,parent=checked_parent();files=v.source_files();source=v.digest(files)
    v.require(json.loads((ROOT/'configs/warehouse_v628_recipe.json').read_bytes())==v.recipe(),'recipe differs')
    folder=v.OUTPUT;folder.mkdir(parents=True,exist_ok=False)
    registration=[]
    for key,ref in v.parents().items():
        role=key.split('fold-')[1].split('-',1)[1]
        p=ParentProgram.from_dict(v.bound(ref),v.old.ROLE_SHA256[role]);_,info=structure_info(p.structure)
        q=from_parent(p,ref,64)
        registration.append(dict(key=key,parent=ref,leaves=len(p.leaves),depth=info['depth'],protected_leaves=len(protected_indices(q))))
    v.require([r['protected_leaves'] for r in registration]==[13,14,17,10,13,10],'unexpected v626 protected union')
    for name,sha in files.items():
        f=folder/'source'/source/name;f.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();v.require(v.file_sha(ROOT/name)==sha,'source changed while preparing')
        f.write_bytes(raw);f.chmod(0o444)
    plan=dict(schema='warehouse-v628-plan.v1',candidate_id=v.CANDIDATE,source_sha256=source,source_files=files,
        recipe=v.recipe(),jobs=v.jobs(),parents=v.parents(),parent_registration=registration,
        actor=parent['actor'],actor_role_sha256=v.old.ROLE_SHA256,cases=data.cases(),folds=data.folds(),
        target_arithmetic='canonical_frozen_NN_single_physical_frame_numpy',
        environment_reset_ceiling=1650,training_authority=False,pilot_opened=False,sealed_authority=False,runtime=runtime_identity())
    write_new(folder/'plan.json',plan);write_new(folder/'plan.claim.json',dict(sha256=v.file_sha(folder/'plan.json'),single_use=True))
    return folder/'plan.json'


def run_job(job_id):
    path,plan=v.checked_plan();spec=next(j for j in plan['jobs'] if j['job_id']==job_id)
    folder=v.OUTPUT/'jobs'/job_id;folder.mkdir(parents=True,exist_ok=False)
    key=f"fold-{spec['fold']}-{spec['role']}";ref=plan['parents'][key]
    parent=ParentProgram.from_dict(v.bound(ref),v.old.ROLE_SHA256[spec['role']]);initial=from_parent(parent,ref,spec['capacity'])
    # Each arm loads the same pure frozen targets. No learned cache crosses jobs.
    inner=data.load_parts(plan,spec['fold'],spec['role'],inner=True)
    begin=time.monotonic();write_new(folder/'started.json',dict(job=spec,started_monotonic=begin,
        search_deadline_monotonic=begin+4800,deadline_monotonic=begin+6000,plan_sha256=v.file_sha(path)))
    budget=Budget(folder/'checkpoint.json',dict(plan_sha256=v.file_sha(path),job=spec),started_at=begin)
    ctx=Context(inner,initial,budget)
    try:
        with real_deadline(begin,4790):program,search=optimize(ctx)
        search['actual_search_seconds']=time.monotonic()-begin
        write_new(folder/'search-best.json',program.to_dict());write_new(folder/'search-report.json',search)
        del ctx,inner
        budget.phase='final';final_begin=time.monotonic()
        write_new(folder/'final-started.json',dict(deadline_monotonic=min(begin+6000,final_begin+1200)))
        with real_deadline(final_begin,min(1190,begin+5990-final_begin)):
            parts=data.load_parts(plan,spec['fold'],spec['role'])
            program,measurements=complete_program(program,parts,budget)
            write_new(folder/'program.json',program.to_dict())
            # Original v626 output bytes are unchanged; qualification sidecar is new.
            baseline=from_parent(parent,ref,spec['capacity'])
            baseline,reference=complete_program(baseline,parts,budget,refit=False)
            write_new(folder/'reference.json',baseline.to_dict())
            budget.checkpoint(dict(stage='completed',program_sha256=v.file_sha(folder/'program.json')),status='completed')
        gate=program_gate(measurements['validation'],search)
        _,info=structure_info(program.structure,spec['capacity'])
        result=dict(schema='warehouse-v628-result.v1',job=spec,status='completed',plan_sha256=v.file_sha(path),
            initial_parent=ref,program=v.pointer(folder/'program.json'),reference=v.pointer(folder/'reference.json'),
            search_report=v.pointer(folder/'search-report.json'),search_best=v.pointer(folder/'search-best.json'),
            leaf_count=len(program.leaves),depth=info['depth'],protected_leaves=len(protected_indices(program)),
            core_invariance=True,metrics=measurements,reference_metrics=reference,gate=gate,solver=budget.summary(),
            elapsed_seconds=time.monotonic()-begin,Actor_updates=0,PPO_joint_steps=0,pilot_opened=False,
            engineering_performance='unmeasured_independent_release_gate')
    except (BudgetExhausted,DeadlineExceeded) as error:
        result=dict(schema='warehouse-v628-result.v1',job=spec,status='scientific_budget_exhausted',plan_sha256=v.file_sha(path),
            gate=dict(passed=False),error=str(error),solver=budget.summary(),elapsed_seconds=time.monotonic()-begin)
        budget.session.close()
    except BaseException as error:
        write_new(folder/'failed.json',dict(error_type=type(error).__name__,error=str(error),solver=budget.summary()))
        budget.session.close();raise
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'Actor changed')
    write_new(folder/'result.json',result);write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    print(json.dumps(dict(job=spec,status=result['status'],passed=result['gate']['passed'])),flush=True)
    return result


def program_gate(metrics,search):
    from backend.training.warehouse_v622_program import gate_metrics
    g=gate_metrics(metrics,phase='development');g['checks']['all_editable_regions_explored']=search['mandatory_complete']
    g['passed']=all(g['checks'].values());return g


def execute():
    path,plan=v.checked_plan();v.require((v.OUTPUT/'collection/completed.json').exists(),'common collection incomplete')
    control=v.OUTPUT/'execution';control.mkdir(exist_ok=False)
    v.require(not (v.OUTPUT/'jobs').exists(),'existing batch cannot be retried')
    pending=list(plan['jobs']);active=[];exits=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    events=(control/'events.jsonl').open('x',buffering=1)
    def interrupted(signum,frame):raise RuntimeError('v628 supervisor interrupted')
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        while pending or active:
            while pending and len(active)<3:
                spec=pending.pop(0);log=(control/(spec['job_id']+'.log')).open('x')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',spec['job_id']],cwd=ROOT,
                    env=env,stdout=log,stderr=log,start_new_session=True)
                active.append((spec,process,log));events.write(json.dumps(dict(event='started',job=spec,pid=process.pid))+'\n')
            for item in list(active):
                spec,process,log=item;folder=v.OUTPUT/'jobs'/spec['job_id'];started=folder/'started.json'
                if process.poll() is None and started.exists():
                    s=json.loads(started.read_bytes());deadline=s['search_deadline_monotonic']
                    final=folder/'final-started.json'
                    if final.exists():deadline=min(s['deadline_monotonic'],json.loads(final.read_bytes())['deadline_monotonic'])
                    if time.monotonic()>=deadline:os.killpg(process.pid,signal.SIGTERM)
                code=process.poll()
                if code is not None:
                    process.wait();log.close();active.remove(item)
                    row=dict(job=spec,pid=process.pid,returncode=code,waited=True);exits.append(row)
                    events.write(json.dumps(dict(event='exited',**row))+'\n')
            if active:time.sleep(.25)
        write_new(control/'completed.json',dict(all_owned_workers_waited=True,jobs=18,exits=exits))
    except BaseException as e:
        stopped=stop_workers(active);write_new(control/'failed.json',dict(error_type=type(e).__name__,error=str(e),workers=stopped,
            all_owned_workers_waited=True,automatic_retry=False));raise
    finally:events.close();signal.signal(signal.SIGTERM,previous)
    return finalize()


def finalize():
    from scripts.audit_warehouse_v628 import audit,report
    path,plan=v.checked_plan();results=[]
    for spec in plan['jobs']:
        folder=v.OUTPUT/'jobs'/spec['job_id'];done=folder/'completed.json'
        if done.exists():
            v.require(json.loads(done.read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),'result seal differs')
            r=json.loads((folder/'result.json').read_bytes());v.require(r['job']==spec,'wrong scientific unit')
        else:r=dict(job=spec,status='failed_or_unfinished',gate=dict(passed=False),metrics={})
        results.append(r)
    integrity=audit(plan,results);write_new(v.OUTPUT/'independent-audit.json',integrity)
    passing=[n for n in (64,128,256) if all(r['status']=='completed' and r['gate']['passed'] for r in results if r['job']['capacity']==n)]
    all_completed=all(r['status']=='completed' for r in results)
    selected=min(passing) if passing and all_completed and integrity['integrity_passed'] else None
    summary=dict(schema='warehouse-v628-summary.v1',candidate_id=v.CANDIDATE,all_18_settled=True,
        all_18_completed=all(r['status']=='completed' for r in results),selected_capacity=selected,
        scientific_pass=selected is not None,independent_audit=v.pointer(v.OUTPUT/'independent-audit.json'),
        next_stage='fixed_asset_confirmation' if selected else 'closed_failed_program_precheck',
        engineering_performance='not_yet_independently_measured',actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)
    write_new(v.OUTPUT/'summary.json',summary);report(results,summary)
    if selected:
        v.require(summary['all_18_completed'],'all18 must complete before any capacity selection')
        write_new(v.OUTPUT/'method-lock.json',dict(method=v.recipe()['method'],capacity=selected,
            programs=[r['program'] for r in results if r['job']['capacity']==selected],summary=v.pointer(v.OUTPUT/'summary.json')))
    else:
        write_new(v.OUTPUT/'candidate.closed.json',dict(status='closed_failed_program_precheck',summary=v.pointer(v.OUTPUT/'summary.json'),
            scientific_pass=False,fresh_confirmation_used=False,PPO_joint_steps=0,pilot_opened=False))
    return summary


def pipeline():
    path=prepare();print('prepared '+str(path),flush=True)
    data.audit_seeds();print('seed history cleared',flush=True)
    data.collect();print('common collection completed',flush=True)
    return execute()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','seed_audit','collect','execute','finalize','pipeline'):g.add_argument('--'+name.replace('_','-'),action='store_true')
    g.add_argument('--job');a=parser.parse_args()
    if a.job:r=run_job(a.job)
    else:r=next(fn for key,fn in (('prepare',prepare),('seed_audit',data.audit_seeds),('collect',data.collect),('execute',execute),('finalize',finalize),('pipeline',pipeline)) if getattr(a,key))()
    print(json.dumps(r if isinstance(r,dict) else str(r),sort_keys=True),flush=True)
