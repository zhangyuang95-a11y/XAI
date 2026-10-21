#!/usr/bin/env python3
"""Source-pinned v627 six-unit stages; old target data, no NN training."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v627_protocol as v
from scripts.run_warehouse_v625 import write_new
from scripts.run_warehouse_v626 import checked_plan as checked_parent,load_parts,stop_workers
from scripts.verify_warehouse_v625_native import real_deadline
from backend.training.warehouse_v627_program import Budget,Context,CertifiedSemanticProgram
from backend.training.warehouse_v627_search import optimize
from backend.training.warehouse_v626_program import complete_program
from backend.training.warehouse_v625_checkpoint import BudgetExhausted,DeadlineExceeded


def prepare(stage):
    v.checked_recipe();path0,parent=checked_parent()
    closed=v.bound(v.pointer(v.parent.OUTPUT/'candidate.closed.json'))
    v.require(closed['status']=='closed_failed_program_precheck' and closed['scientific_pass'] is False,'v626 closed evidence required')
    predecessor=v.pointer(v.parent.OUTPUT/'candidate.closed.json')
    if stage=='S64':
        p,_=checked_plan('S32');s=v.bound(v.pointer(p.parent/'summary.json'))
        v.require(s['all_six_settled'] and s['scientific_pass'] is False and v.bound(s['independent_audit'])['integrity_passed'],
                  'S64 only follows complete audited S32 failure')
        predecessor=v.pointer(p.parent/'summary.json')
    files=v.source_files();source=v.digest(files);folder=v.OUTPUT/stage
    folder.mkdir(parents=True,exist_ok=False)
    for name,sha in files.items():
        target=folder/'source'/source/name;target.parent.mkdir(parents=True,exist_ok=True)
        v.require(v.file_sha(ROOT/name)==sha,'source changed during snapshot')
        target.write_bytes((ROOT/name).read_bytes());target.chmod(0o444)
    plan=dict(schema='warehouse-v627-plan.v1',candidate_id=v.CANDIDATE,stage=stage,
        source_sha256=source,source_files=files,recipe=v.recipe(),runtime=v.runtime(),jobs=v.jobs(stage),
        maximum_leaves=32 if stage=='S32' else 64,parent_plan=v.pointer(path0),predecessor=predecessor,
        actor=parent['actor'],actor_role_sha256=parent['actor_role_sha256'],
        initial_S1=parent['initial_S1'],datasets=parent['datasets'],**v.ZERO)
    write_new(folder/'plan.json',plan);write_new(folder/'plan.claim.json',dict(sha256=v.file_sha(folder/'plan.json'),single_use=True))
    return folder/'plan.json'


def checked_plan(stage):
    p=v.OUTPUT/stage/'plan.json';a=json.loads(p.read_bytes())
    v.require(json.loads(p.with_name('plan.claim.json').read_bytes())==dict(sha256=v.file_sha(p),single_use=True),'v627 plan seal changed')
    files=v.source_files()
    v.require(a['candidate_id']==v.CANDIDATE and a['stage']==stage and a['jobs']==v.jobs(stage)
        and a['source_sha256']==v.digest(files) and a['source_files']==files and a['runtime']==v.runtime()
        and a['recipe']==v.recipe() and a['maximum_leaves']==(32 if stage=='S32' else 64),'v627 source/recipe/runtime changed')
    _,parent=checked_parent()
    v.require(a['actor']==parent['actor'] and a['actor_role_sha256']==v.old.ROLE_SHA256 and a['datasets']==parent['datasets']
        and a['initial_S1']==parent['initial_S1'],'v627 inputs changed')
    for key in ('parent_plan','predecessor','actor'):v.require(v.pointer(a[key]['path'])==a[key],'changed bound dependency')
    for key,value in v.ZERO.items():v.require(type(a.get(key)) is type(value) and a[key]==value,'unregistered execution authority')
    v.checked_recipe()
    for name,sha in files.items():
        f=p.parent/'source'/a['source_sha256']/name
        v.require(v.file_sha(f)==sha and not f.stat().st_mode&0o222,'v627 source snapshot changed')
    return p,a


def job(stage,job_id):
    path,plan=checked_plan(stage);spec=next((j for j in plan['jobs'] if j['job_id']==job_id),None)
    v.require(spec is not None,'unknown v627 unit')
    folder=path.parent/'jobs'/job_id;folder.mkdir(parents=True,exist_ok=False)
    _,parent=checked_parent();parts=load_parts(parent,job_id)
    initial=v.bound(plan['initial_S1'][job_id]['structure'])
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    begin=time.monotonic();write_new(folder/'started.json',dict(job=spec,started_monotonic=begin,
        search_deadline_monotonic=begin+4800,deadline_monotonic=begin+5400,plan_sha256=v.file_sha(path),**v.ZERO))
    budget=Budget(folder/'checkpoint.json',dict(plan_sha256=v.file_sha(path),job=spec),v.recipe()['limits'],5400,
        maximum_leaves=plan['maximum_leaves'],started_at=begin)
    ctx=Context(parts['fit'],spec['role'],v.old.ROLE_SHA256[spec['role']],plan['maximum_leaves'],budget)
    try:
        with real_deadline(begin,4790):program,search=optimize(ctx,initial)
        search['actual_search_seconds']=time.monotonic()-begin
        v.require(search['actual_search_seconds']<4800,'v627 search settlement overran')
        write_new(folder/'search-best.json',program.to_dict());write_new(folder/'search-report.json',search)
        budget.phase='final';final_begin=time.monotonic()
        write_new(folder/'final-started.json',dict(deadline_monotonic=min(begin+5400,final_begin+600)))
        with real_deadline(final_begin,min(600,begin+5400-final_begin)):
            program.kernel,measurements=complete_program(program.kernel,parts,budget)
            write_new(folder/'program.json',program.to_dict())
            budget.checkpoint(dict(stage='completed',program_sha256=v.file_sha(folder/'program.json')),status='completed')
        gate=v.gate(measurements['validation'])
        gate['checks']['four_starts_completed_mandatory_exploration']=search['mandatory_completed']==[0,1,2,3]
        gate['passed']=all(gate['checks'].values())
        result=dict(schema='warehouse-v627-result.v1',job=spec,status='completed',plan_sha256=v.file_sha(path),
            program=v.pointer(folder/'program.json'),search_best=v.pointer(folder/'search-best.json'),search_report=v.pointer(folder/'search-report.json'),
            leaf_count=len(program.leaves),maximum_leaves=plan['maximum_leaves'],metrics=measurements,gate=gate,
            solver=budget.summary(),elapsed_seconds=time.monotonic()-begin,
            performance_status='full_runner_overhead_not_independently_measured',**v.ZERO)
        v.require(time.monotonic()-final_begin<600,'final v627 result exceeded phase budget')
    except (BudgetExhausted,DeadlineExceeded) as error:
        result=dict(schema='warehouse-v627-result.v1',job=spec,status='scientific_budget_exhausted',
            plan_sha256=v.file_sha(path),error=str(error),gate={'passed':False},solver=budget.summary(),elapsed_seconds=time.monotonic()-begin,**v.ZERO)
        budget.session.close()
    except BaseException as error:
        write_new(folder/'failed.json',dict(error_type=type(error).__name__,error=str(error),solver=budget.summary(),**v.ZERO))
        budget.session.close();raise
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen Actor changed')
    write_new(folder/'result.json',result);write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    print(json.dumps(dict(stage=stage,job=job_id,status=result['status'],passed=result['gate']['passed'])),flush=True)
    return result


def execute(stage):
    path,plan=checked_plan(stage);control=path.parent/'execution';control.mkdir(exist_ok=False)
    v.require(not (path.parent/'jobs').exists(),'v627 never retries an existing batch')
    pending=list(plan['jobs']);active=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise RuntimeError('v627 supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        while pending or active:
            while pending and len(active)<3:
                spec=pending.pop(0);log=(control/(spec['job_id']+'.log')).open('x')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage',stage,'--job',spec['job_id']],
                    cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
                active.append((spec,process,log))
            for item in list(active):
                spec,process,log=item;folder=path.parent/'jobs'/spec['job_id'];started=folder/'started.json'
                if started.exists():
                    a=json.loads(started.read_bytes());deadline=a['search_deadline_monotonic'];final=folder/'final-started.json'
                    if final.exists():deadline=min(a['deadline_monotonic'],json.loads(final.read_bytes())['deadline_monotonic'])
                    if process.poll() is None and time.monotonic()>=deadline:raise DeadlineExceeded('external phase deadline '+spec['job_id'])
                code=process.poll()
                if code is not None:
                    process.wait();log.close();active.remove(item)
                    if code:raise RuntimeError('v627 worker failed '+spec['job_id'])
            if active:time.sleep(.25)
        write_new(control/'completed.json',dict(all_owned_workers_waited=True,jobs=6))
    except BaseException as error:
        workers=stop_workers(active);write_new(control/'failed.json',dict(error_type=type(error).__name__,error=str(error),
            workers=workers,all_owned_workers_waited=True,automatic_retry=False,**v.ZERO));raise
    finally:signal.signal(signal.SIGTERM,previous)
    return finalize(stage)


def finalize(stage):
    from scripts.audit_warehouse_v627 import audit,report
    path,plan=checked_plan(stage);results=[]
    for spec in plan['jobs']:
        folder=path.parent/'jobs'/spec['job_id'];r=json.loads((folder/'result.json').read_bytes())
        v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),'missing unit seal')
        v.require(r['job']==spec and r['plan_sha256']==v.file_sha(path),'unbound unit');results.append(r)
    integrity=audit(path,results);v.require(integrity['integrity_passed'],'v627 arithmetic replay failed')
    write_new(path.parent/'independent-audit.json',integrity)
    passed=all(r['status']=='completed' and r['gate']['passed'] for r in results)
    summary=dict(schema='warehouse-v627-summary.v1',candidate_id=v.CANDIDATE,stage=stage,
        source_sha256=plan['source_sha256'],all_six_settled=True,scientific_pass=passed,
        next_stage='B' if passed else 'S64' if stage=='S32' else 'closed_failed_program_precheck',
        independent_audit=v.pointer(path.parent/'independent-audit.json'),results=[v.pointer(path.parent/'jobs'/j['job_id']/'result.json') for j in plan['jobs']],
        engineering_performance='unmeasured_not_scientific_blocker',**v.ZERO)
    write_new(path.parent/'summary.json',summary);report(path.parent,results,summary)
    write_new(ROOT/f'experiments/warehouse/evidence/warehouse-v627-{stage}.json',dict(summary=v.pointer(path.parent/'summary.json'),
        scientific_pass=passed,next_stage=summary['next_stage'],**v.ZERO))
    if passed:write_new(v.OUTPUT/'method-lock.json',dict(method='semantic_compiled_qualification_consistent.v1',
        capacity=plan['maximum_leaves'],plan=v.pointer(path),summary=v.pointer(path.parent/'summary.json'),**v.ZERO))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('S32','S64'),required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');g.add_argument('--finalize',action='store_true');g.add_argument('--job')
    a=p.parse_args()
    if a.prepare:print(prepare(a.stage))
    elif a.job:job(a.stage,a.job)
    elif a.finalize:print(json.dumps(finalize(a.stage),sort_keys=True))
    else:print(json.dumps(execute(a.stage),sort_keys=True))
