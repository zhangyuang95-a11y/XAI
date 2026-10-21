#!/usr/bin/env python3
"""Conditional single-use program search after the entire v629 batch closes."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training import warehouse_v630_partition_search as search
from backend.training.warehouse_v628_checkpoint import CheckpointSession
from backend.training.warehouse_v626_budget import Budget as BaseBudget
from backend.training.warehouse_v625_checkpoint import BudgetExhausted,DeadlineExceeded
from backend.training.warehouse_v628_program import measure
from scripts.run_warehouse_v625 import write_new
from scripts.run_warehouse_v626 import stop_workers
from scripts.verify_warehouse_v625_native import real_deadline


class Budget(BaseBudget):
    def __init__(self,path,identity,*,started_at):
        self.limits=v.recipe()['limits'];self.counts={k:0 for k in self.limits};self.pending={};self.phase='fit';self.final_leaf_reserve=0
        self.session=CheckpointSession(path,{**identity,'limits':self.limits},max_calls=256,wall_seconds=1800,started_at=started_at)
    def check(self):
        super().check()
        if self.phase!='final' and self.session.elapsed_seconds>=1200:raise DeadlineExceeded('v630 cumulative search limit')


def wait_dependency(plan):
    identity=plan['dependency_process'];start=time.monotonic()
    write_new(v.OUTPUT/'dependency-wait-started.json',dict(process=identity,started_monotonic=start))
    while identity is not None and v.process_identity(identity['pid'])==identity['identity']:time.sleep(10)
    completion=v.old_completion()
    if completion['skip_new_candidate']:
        write_new(v.OUTPUT/'candidate.skipped.json',completion);return False
    sources={}
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            _,binding=v.source_program(plan,fold,role);sources[f'{fold}-{role}']=binding
    write_new(v.OUTPUT/'dependency-cleared.json',dict(completion=completion,sources=sources,
        waited_seconds=time.monotonic()-start,old_jobs_interrupted=False,old_jobs_restarted=False))
    return True


def gate(metrics,report,scope):
    checks={}
    for purpose,agreement in [('feedback',.95),('explanation',.97)]:
        m=metrics[purpose];checks[purpose+'_coverage']=m['coverage']>=.45;checks[purpose+'_agreement']=m['agreement']>=agreement
        checks[purpose+'_KL']=m['kl'] is not None and m['kl']<=.20
        for key in ('illegal_count','path_uncertified_count','whole_tree_raw_illegal_count','whole_tree_path_uncertified_count'):
            checks[purpose+'_'+key]=m[key]==0
    checks['expected_feedback']=metrics['feedback']['expected_actionable_coverage']>=.40
    checks['mandatory_complete']=report.get('mandatory_complete') is True
    checks['reliable_fit_only_proxy']=scope=='control' or report.get('reliable_selection_passed') is True
    return dict(passed=all(checks.values()),checks=checks)


def run_job(job_id):
    path,plan=v.checked_plan();spec=next(j for j in plan['jobs'] if j['job_id']==job_id)
    dependency=v.bound(v.pointer(v.OUTPUT/'dependency-cleared.json'));v.require(v.old_completion()==dependency['completion'],'upstream completion changed')
    source,binding=v.source_program(plan,spec['fold'],spec['role'])
    v.require(binding==dependency['sources'][f'{spec["fold"]}-{spec["role"]}'],'bound source program changed')
    folder=v.OUTPUT/'jobs'/job_id;folder.mkdir(parents=True,exist_ok=False);begin=time.monotonic();budget=None
    write_new(folder/'started.json',dict(job=spec,source=binding,plan=v.pointer(path),started_monotonic=begin,
        search_deadline_monotonic=begin+1200,deadline_monotonic=begin+1800))
    try:
        arrays,cases,array_refs=v.fit_data(plan,spec['fold'],spec['role']);parts=search.lane_parts(arrays,cases)
        budget=Budget(folder/'checkpoint.json',dict(plan_sha256=v.file_sha(path),job=spec,source=binding),started_at=begin)
        with real_deadline(begin,1190):program,report=search.run(source,parts,budget,scope=spec['scope'])
        tables=dict(groups=report.pop('option_groups',{}),local=report.pop('local_details',{}))
        write_new(folder/'option-tables.json',tables);report['option_tables']=v.pointer(folder/'option-tables.json')
        report['input_arrays']=array_refs
        write_new(folder/'search-program.json',program.to_dict());write_new(folder/'search-report.json',report)
        del arrays,parts
        budget.phase='final';start=time.monotonic();deadline=min(begin+1800,start+600)
        write_new(folder/'final-started.json',dict(deadline_monotonic=deadline))
        with real_deadline(start,max(.01,deadline-start-5)):
            old=v.bound(plan['previous_plan']);data_plan=v.bound(old['previous_plan'])
            outer=data.load_parts(data_plan,spec['fold'],spec['role'])
            program=search.qualify_current(program,outer)
            reference=search.from_previous(source);reference.maximum_leaves=256;search.qualify_current(reference,outer)
            metrics={s:measure(program,a) for s,a in outer.items()};reference_metrics={s:measure(reference,a) for s,a in outer.items()}
            for s,a in outer.items():
                old_output=reference.distributions(a['x'],a['m']);new_output=program.distributions(a['x'],a['m'])
                import numpy as np
                v.require(np.array_equal(old_output[1],new_output[1]) and np.array_equal(old_output[2],new_output[2]),'frozen source distribution changed')
            write_new(folder/'program.json',program.to_dict());write_new(folder/'reference.json',reference.to_dict())
            budget.checkpoint(dict(stage='completed',program=v.pointer(folder/'program.json'),search_report=v.pointer(folder/'search-report.json')),status='completed')
        result=dict(schema='warehouse-v630-result.v1',status='completed',job=spec,plan=v.pointer(path),source=binding,
            program=v.pointer(folder/'program.json'),reference=v.pointer(folder/'reference.json'),search_report=v.pointer(folder/'search-report.json'),
            search_program=v.pointer(folder/'search-program.json'),metrics=metrics,reference_metrics=reference_metrics,
            leaf_count=len(program.leaves),depth=search.structure_info(program.structure,256)[1]['depth'],
            all_source_surrogate_distributions_preserved=True,gate=gate(metrics['validation'],report,spec['scope']),
            solver=budget.summary(),elapsed_seconds=time.monotonic()-begin,NN_updates=0,PPO_joint_steps=0,environment_resets=0,pilot_opened=False)
    except BaseException as error:
        result=dict(schema='warehouse-v630-result.v1',status='scientific_budget_exhausted' if isinstance(error,(BudgetExhausted,DeadlineExceeded)) else 'failed',
            job=spec,plan=v.pointer(path),source=binding,error_type=type(error).__name__,error=str(error),gate=dict(passed=False),
            elapsed_seconds=time.monotonic()-begin,solver=None if budget is None else budget.summary(),pilot_opened=False)
        if budget is not None:budget.session.close()
        write_new(folder/'failed.json',result)
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'frozen NN bytes changed')
    write_new(folder/'result.json',result);write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    return result


def execute():
    _,plan=v.checked_plan();v.bound(v.pointer(v.OUTPUT/'dependency-cleared.json'));control=v.OUTPUT/'execution';control.mkdir(exist_ok=False)
    v.require(not (v.OUTPUT/'jobs').exists(),'existing v630 jobs cannot restart')
    pending=list(plan['jobs']);active=[];exits=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def interrupt(signum,frame):raise RuntimeError('v630 supervisor interrupted')
    handler=signal.signal(signal.SIGTERM,interrupt);events=(control/'events.jsonl').open('x',buffering=1)
    try:
        while pending or active:
            while pending and len(active)<3:
                job=pending.pop(0);log=(control/(job['job_id']+'.log')).open('x')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',job['job_id']],cwd=ROOT,env=env,
                    stdout=log,stderr=log,start_new_session=True);active.append((job,process,log))
                events.write(json.dumps(dict(event='started',job=job,pid=process.pid))+'\n')
            for item in list(active):
                job,process,log=item;folder=v.OUTPUT/'jobs'/job['job_id']
                if process.poll() is None and (folder/'started.json').exists():
                    started=v.bound(v.pointer(folder/'started.json'));deadline=started['search_deadline_monotonic']
                    if (folder/'final-started.json').exists():deadline=min(started['deadline_monotonic'],v.bound(v.pointer(folder/'final-started.json'))['deadline_monotonic'])
                    if time.monotonic()>=deadline:os.killpg(process.pid,signal.SIGTERM)
                code=process.poll()
                if code is not None:
                    process.wait();log.close();active.remove(item);row=dict(job=job,pid=process.pid,returncode=code,waited=True);exits.append(row)
                    events.write(json.dumps(dict(event='exited',**row))+'\n')
            if active:time.sleep(.25)
        write_new(control/'completed.json',dict(all_owned_workers_waited=True,jobs=18,exits=exits))
    except BaseException as error:
        stopped=stop_workers(active);write_new(control/'failed.json',dict(error_type=type(error).__name__,error=str(error),workers=stopped,
            all_owned_workers_waited=True,automatic_retry=False));raise
    finally:events.close();signal.signal(signal.SIGTERM,handler)
    return finalize()


def finalize():
    from scripts.audit_warehouse_v630 import audit,report
    _,plan=v.checked_plan();results=[]
    for job in plan['jobs']:
        folder=v.OUTPUT/'jobs'/job['job_id'];ref=v.pointer(folder/'result.json');result=v.bound(ref)
        v.require(v.bound(v.pointer(folder/'completed.json'))==dict(result_sha256=ref['sha256']) and result['job']==job,'v630 result seal differs')
        results.append(result)
    integrity=audit(plan,results);write_new(v.OUTPUT/'independent-audit.json',integrity)
    all_settled=len(results)==18;selected=next((name for name in ('P','A') if all(r['status']=='completed' and r['gate']['passed']
        for r in results if r['job']['name']==name)),None)
    quality=v.common_quality(plan);quality_ok=all(m['deliveries']>=5.5 and m['collisions']<=6 and m['shutdown_rate']<=.05 for m in quality.values())
    all_completed=all(r['status']=='completed' for r in results)
    passed=selected is not None and all_settled and all_completed and integrity['integrity_passed'] and quality_ok
    summary=dict(schema='warehouse-v630-summary.v1',candidate_id=v.CANDIDATE,all_18_settled=all_settled,
        selected_method=selected if passed else None,scientific_pass=passed,independent_audit=v.pointer(v.OUTPUT/'independent-audit.json'),
        all_18_completed=all_completed,
        common_frozen_NN_development_quality_passed=quality_ok,common_quality=plan['common_quality'],
        next_stage='fixed_asset_fresh_confirmation' if passed else 'closed_failed_program_precheck',
        NN_updates=0,PPO_joint_steps=0,fresh_confirmation_used=False,sealed_audit_used=False,heldout_used=False,
        training_authority=False,pilot_opened=False,publishable=False)
    write_new(v.OUTPUT/'summary.json',summary);report(results,summary)
    if passed:write_new(v.OUTPUT/'method-lock.json',dict(method=v.recipe()['method'],arm=selected,
        programs=[r['program'] for r in results if r['job']['name']==selected],summary=v.pointer(v.OUTPUT/'summary.json')))
    else:write_new(v.OUTPUT/'candidate.closed.json',dict(status=summary['next_stage'],summary=v.pointer(v.OUTPUT/'summary.json'),pilot_opened=False))
    return summary


def pipeline(wait_pid):
    plan=v.prepare(wait_pid)
    if not wait_dependency(plan):return v.bound(v.pointer(v.OUTPUT/'candidate.skipped.json'))
    return execute()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pipeline',action='store_true');modes.add_argument('--job');modes.add_argument('--finalize',action='store_true')
    parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    try:
        result=run_job(args.job) if args.job else finalize() if args.finalize else pipeline(args.wait_pid)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    except BaseException as error:
        if v.OUTPUT.exists() and not (v.OUTPUT/'driver.failed.json').exists():write_new(v.OUTPUT/'driver.failed.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
