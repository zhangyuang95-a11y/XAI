#!/usr/bin/env python3
"""Single-use paired v6.25 Stage A on spent development only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v625_protocol as v
from scripts.run_warehouse_v622_capacity import arrays
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from scripts.verify_warehouse_v625_native import checked_receipt as checked_native_receipt


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def write_new(path,value):
    # Serialize first so a rejected numeric value cannot leave a partial receipt.
    payload=json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    try:
        with temporary.open('xb') as h:
            h.write(payload);h.flush();os.fsync(h.fileno())
        os.link(temporary,path)  # Atomic publication with no overwrite.
        fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def prepare():
    source=v.source_sha();baseline,initial=v.checked_baseline()
    engineering_path=v.OUTPUT.parent.parent/'engineering'/v.ENGINEERING_RECEIPT
    v.require(engineering_path.is_file(),'run the registered v625 engineering preflight before prepare')
    engineering=checked_native_receipt(engineering_path,expected_source_sha256=source)
    v.require(engineering.get('passed') is True and engineering.get('source_sha256')==source,
              'bound real-state engineering preflight required before scientific preparation')
    actor,actorpath=v.old.checked_actor(ROOT);episodes,manifest=v.old.checked_inputs(ROOT)
    plan=dict(v.ZERO,schema='warehouse-v625-sparse-logistic-plan.v1',candidate_id=v.CANDIDATE,stage='A',
        source_sha256=source,source_files=v.source_files(),recipe_sha256=v.recipe(),runtime=runtime_identity(),
        actor=v.pointer(actorpath),actor_role_sha256=v.old.ROLE_SHA256,inputs=manifest,
        targets=v.old.target_identity(actor,episodes),folds=v.old.grouped_folds(),jobs=v.jobs(),
        baseline=baseline,initial_S1=initial,budget=v.BUDGET,gates=v.GATES,engineering=v.pointer(engineering_path),
        fresh_confirmation_authorized=False,automatic_retry=False,scientific_retry=False)
    v.require(v.source_sha()==source,'source changed during preparation')
    v.OUTPUT.mkdir(parents=True,exist_ok=False)
    snapshot=v.OUTPUT/'source'/source
    for relative,expected in plan['source_files'].items():
        raw=(ROOT/relative).read_bytes();v.require(v.file_sha(ROOT/relative)==expected,'snapshot source changed')
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as h:h.write(raw)
        target.chmod(0o444)
    write_new(v.OUTPUT/'plan.json',plan)
    write_new(v.OUTPUT/'plan.claim.json',{'sha256':v.file_sha(v.OUTPUT/'plan.json'),'single_use':True})
    v.require(v.source_sha()==source,'source changed while snapshotting')
    return v.OUTPUT/'plan.json'


def checked_plan(plan_path=None):
    path=v.OUTPUT/'plan.json' if plan_path is None else Path(plan_path).resolve()
    v.require(path==v.OUTPUT/'plan.json','plan outside registered candidate')
    plan=json.loads(path.read_bytes());claim=json.loads((path.parent/'plan.claim.json').read_bytes())
    v.require(claim=={'sha256':v.file_sha(path),'single_use':True},'plan claim changed')
    v.require(plan.get('schema')=='warehouse-v625-sparse-logistic-plan.v1' and plan.get('candidate_id')==v.CANDIDATE
        and plan.get('stage')=='A' and plan.get('source_sha256')==v.source_sha()
        and plan.get('source_files')==v.source_files() and plan.get('recipe_sha256')==v.recipe()
        and plan.get('runtime')==runtime_identity() and plan.get('jobs')==v.jobs()
        and plan.get('folds')==v.old.grouped_folds() and plan.get('budget')==v.BUDGET
        and plan.get('gates')==v.GATES and plan.get('actor_role_sha256')==v.old.ROLE_SHA256
        and plan.get('actor',{}).get('sha256')==v.old.ACTOR_SHA256,'scientific plan identity changed')
    for k,val in dict(v.ZERO,fresh_confirmation_authorized=False,automatic_retry=False,scientific_retry=False).items():
        v.require(type(plan.get(k)) is type(val) and plan[k]==val,'authority changed '+k)
    v.require(v.pointer(plan['actor']['path'])==plan['actor'],'Actor missing/mismatched')
    for p,h in plan['inputs']['files_sha256'].items():v.require(v.file_sha(p)==h,'input missing/mismatched '+p)
    engineering=v.bound(plan['engineering'])
    checked_native_receipt(plan['engineering']['path'],expected_source_sha256=plan['source_sha256'])
    v.require(engineering.get('passed') is True and engineering.get('source_sha256')==plan['source_sha256'],
              'engineering preflight binding changed')
    for relative,expected in plan['source_files'].items():
        v.require(v.file_sha(path.parent/'source'/plan['source_sha256']/relative)==expected,'readonly source snapshot changed')
    baseline,initial=v.checked_baseline()
    v.require(plan.get('baseline')==baseline and plan.get('initial_S1')==initial,'v623 baseline binding changed')
    return path,plan


def _claim(folder):
    h=(folder/'worker.lock').open('a+')
    try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:h.close();raise
    h.seek(0);h.truncate();h.write(str(os.getpid()));h.flush()
    return h


def job(job_id,resume=False):
    from backend.training.warehouse_v625_search import fit_structure,UpgradeBudget,SearchIncompleteError
    from backend.training.warehouse_v625_program import fit_program,program_metrics,program_structure_sha256
    from backend.training.warehouse_v6_actor import role_sha256
    path,plan=checked_plan();spec=next((r for r in plan['jobs'] if r['job_id']==job_id),None)
    v.require(spec is not None,'unknown role/fold/arm')
    folder=path.parent/'jobs'/job_id
    if resume:
        v.require(folder.is_dir() and not any((folder/p).exists() for p in ('failed.json','completed.json','result.json')),
                  'resume requires incomplete safe job, never failed/completed retry')
        started=json.loads((folder/'started.json').read_bytes())
        v.require(started.get('job')==spec and started.get('plan_sha256')==v.file_sha(path),'resume start binding changed')
    else:folder.mkdir(parents=True,exist_ok=False)
    lock=_claim(folder);begin=time.monotonic();search=None;fit_begin=None;search_return=None;fitting_wall=None
    try:
        try:
            if resume:
                checkpoint=folder/'checkpoint.json'
                v.require(checkpoint.is_file() and not Path(str(checkpoint)+'.dirty').exists(),'no safe checkpoint')
                resume_number=len(list(folder.glob("resume-*.json")))+1
                write_new(folder/f'resume-{resume_number:03d}.json',
                    dict(v.ZERO,job=spec,plan_sha256=v.file_sha(path),checkpoint=v.pointer(checkpoint),explicit_resume=True))
            else:write_new(folder/'started.json',dict(v.ZERO,job=spec,stage='A',plan_sha256=v.file_sha(path)))
            actor,actorpath=v.old.checked_actor(ROOT);episodes,manifest=v.old.checked_inputs(ROOT)
            v.require(manifest==plan['inputs'] and v.old.target_identity(actor,episodes)==plan['targets'],'frozen targets changed')
            fold=plan['folds'][spec['fold']]
            parts={split:[row for case in fold['splits'][split] for row in episodes[case['base']][spec['role']]] for split in v.SPLITS}
            initial_ref=plan['initial_S1'][f"fold-{spec['fold']}-{spec['role']}"]['structure']
            initial=v.bound(initial_ref);initialsha=program_structure_sha256(initial);budget=UpgradeBudget()
            v.require(budget.production_configuration is True,'diagnostic search budget cannot run scientifically')
            v.require(budget.maximum_new_leaf_calls==20000 and budget.final_leaf_reserve==32
                and budget.wall_seconds==5400 and budget.final_wall_reserve==600,'search budget differs from registered recipe')
            fit_begin=time.monotonic()
            phase_path=folder/(f'search-resume-{resume_number:03d}.json' if resume else 'search-started.json')
            write_new(phase_path,{'deadline_monotonic':fit_begin+4800.,'started_monotonic':fit_begin,
                'cumulative_budget_enforced_by_checkpoint':True})
            structure,search=fit_structure(spec['arm'],*arrays(actor,spec['role'],parts['fit']),
                feature_names=actor.roles[spec['role']].feature_names,actor_role_sha256=v.old.ROLE_SHA256[spec['role']],
                initial_structure=initial,initial_structure_sha256=initialsha,budget=budget,
                checkpoint_path=folder/'checkpoint.json',resume=resume)
            write_new(folder/'structure.json',structure);write_new(folder/'search-report.json',search)
            registered=json.loads((ROOT/v.RECIPE).read_bytes())
            v.require(search.get('configuration')==registered['search_configuration']
                and search.get('production_configuration') is True,'search report algorithm configuration differs')
            cumulative_search=search['elapsed_seconds']
            v.require(type(cumulative_search) in (int,float) and 0<=cumulative_search<=4800,'search cumulative time invalid')
            search_return=time.monotonic()
            v.require(search_return-fit_begin<=4800,'search export settlement exceeded deadline')
            final_deadline=search_return+600.
            write_new(folder/'final-started.json',{'started_monotonic':search_return,'deadline_monotonic':final_deadline})
            program=fit_program(spec['role'],actor,*(parts[s] for s in v.SPLITS),structure=structure,
                mode=spec['arm'],initial_leaves=structure['initial_leaf_models'],
                fit_budget={'deadline_monotonic':final_deadline,'maximum_leaf_api_calls':32,
                    'prior_leaf_optimization_calls':search['new_leaf_calls'],'max_total_leaf_optimization_calls':20000,
                    'provenance_sha256':v.file_sha(folder/'search-report.json'),
                    'checkpoint_path':str(folder/'final-checkpoint.json')})
            metrics={split:program_metrics(program,actor,rows) for split,rows in parts.items()}
            fitting_wall=cumulative_search+(time.monotonic()-search_return)
            if time.monotonic()>=final_deadline:
                raise TimeoutError('final metrics exceeded the registered 600 second deadline')
            program.save(folder/'program.json');finalcalls=program.report['leaf_optimization_calls_outside_search']
            checks={'search_calls_with_reserved_final':search['new_leaf_calls']<=19968,
                'final_calls_at_most_32':finalcalls<=32,'total_new_calls_at_most_20000':search['new_leaf_calls']+finalcalls<=20000,
                'all_final_leaves_refitted':finalcalls==structure['leaf_count'],
                'search_wall_at_most_4800':search_return-fit_begin<=4800,
                'final_wall_at_most_600':time.monotonic()<final_deadline,
                'checkpoint_overhead_at_most_10_percent':search.get('checkpoint_fraction',1.)<=.10,
                'no_actor_change':v.file_sha(actorpath)==v.old.ACTOR_SHA256 and all(
                    role_sha256(actor.roles[r].weights,model_version=actor.roles[r].model_version)==v.old.ROLE_SHA256[r] for r in v.ROLES)}
            v.require(all(checks.values()),'scientific budget/Actor checks failed')
            result=dict(v.ZERO,schema='warehouse-v625-sparse-logistic-result.v1',job=spec,stage='A',status='completed',
                plan_sha256=v.file_sha(path),actor=plan['actor'],initial_structure=initial_ref,initial_structure_sha256=initialsha,
                structure=v.pointer(folder/'structure.json'),search_report=v.pointer(folder/'search-report.json'),
                program=v.pointer(folder/'program.json'),leaf_count=structure['leaf_count'],metrics=metrics,
                gate=v.program_gate(metrics['validation']),budget_checks=checks,search_calls=search['new_leaf_calls'],
                final_calls=finalcalls,fitting_wall_seconds=fitting_wall,elapsed_seconds=cumulative_search+(fit_begin-begin)+(time.monotonic()-search_return),
                runner_segment_elapsed_seconds=time.monotonic()-begin,
                time_scope='cumulative_search_active_plus_final_segment; resume downtime excluded; preprocessing outside fit budget',
                explicitly_resumed=resume)
            v.require(time.monotonic()<final_deadline,'final serialization exceeded deadline')
        except SearchIncompleteError as exc:
            result=dict(v.ZERO,schema='warehouse-v625-sparse-logistic-result.v1',job=spec,stage='A',status='scientific_budget_exhausted',
                plan_sha256=v.file_sha(path),error_type=type(exc).__name__,failure=exc.report,gate={'passed':False},
                fitting_wall_seconds=exc.report.get('elapsed_seconds',time.monotonic()-fit_begin),
                elapsed_seconds=time.monotonic()-begin,fitting_wall_limit_seconds=5400,explicitly_resumed=resume)
            write_new(folder/'incomplete-search-report.json',exc.report)
            result['search_report']=v.pointer(folder/'incomplete-search-report.json')
        except BaseException as exc:
            write_new(folder/'failed.json',dict(v.ZERO,error_type=type(exc).__name__,error=str(exc),automatic_retry=False))
            raise
        else:pass
        checked_plan()
        if result['status']=='completed':
            v.require(time.monotonic()<final_deadline,'final identity verification exceeded deadline')
        write_new(folder/'result.json',result)
        if result['status']=='completed' and time.monotonic()>=final_deadline:
            raise RuntimeError('final result commit exceeded hard deadline; no completion receipt')
        write_new(folder/'completed.json',{'plan_sha256':v.file_sha(path),'result_sha256':v.file_sha(folder/'result.json'),'passed':result['gate']['passed']})
        if result['status']=='completed' and time.monotonic()>=final_deadline:
            write_new(folder/'failed.json',dict(v.ZERO,error_type='DeadlineExceeded',
                error='completion fsync exceeded final deadline',automatic_retry=False))
            raise RuntimeError('completion fsync exceeded final deadline')
        print(json.dumps({'event':'job_completed','job':job_id,'status':result['status'],'passed':result['gate']['passed']}),flush=True)
        return result
    finally:lock.close()


def collect_results(path,plan):
    results=[]
    for spec in plan['jobs']:
        folder=path.parent/'jobs'/spec['job_id']
        v.require(not (folder/'failed.json').exists(),'engineering failure cannot enter comparison')
        result=json.loads((folder/'result.json').read_bytes());completion=json.loads((folder/'completed.json').read_bytes())
        started=json.loads((folder/'started.json').read_bytes())
        v.require(all(type(result.get(k)) is type(value) and result[k]==value for k,value in v.ZERO.items()),'job authority changed')
        v.require(result.get('schema')=='warehouse-v625-sparse-logistic-result.v1'
            and result.get('job')==spec and result.get('stage')=='A' and result.get('plan_sha256')==v.file_sha(path)
            and started==dict(v.ZERO,job=spec,stage='A',plan_sha256=v.file_sha(path))
            and completion=={'plan_sha256':v.file_sha(path),'result_sha256':v.file_sha(folder/'result.json'),'passed':result['gate']['passed']},
            'incomplete/unbound job '+spec['job_id'])
        results.append(result)
    return results


def finalize():
    path,plan=checked_plan();results=collect_results(path,plan);selection=v.comparison(results)
    for arm in v.ARMS:
        for row in selection['arms'][arm]['role_folds']:
            row['result']=v.pointer(path.parent/'jobs'/row['job_id']/'result.json')
    summary=dict(v.ZERO,schema='warehouse-v625-sparse-logistic-summary.v1',candidate_id=v.CANDIDATE,stage='A',
        plan_sha256=v.file_sha(path),source_sha256=plan['source_sha256'],
        jobs={spec['job_id']:v.pointer(path.parent/'jobs'/spec['job_id']/'result.json') for spec in plan['jobs']},**selection)
    write_new(path.parent/'summary.json',summary)
    from scripts.audit_warehouse_v625 import audit
    receipt=audit(path);v.require(receipt.get('passed') is True,'independent audit failed')
    write_new(path.parent/'independent-audit.json',receipt)
    from scripts.report_warehouse_v625 import report
    report(path.parent)
    from scripts.finalize_warehouse_v625 import finalize as seal
    sealed=seal(path)
    return {'summary':summary,'independent_audit':receipt,'finalization':sealed}


def stop_workers(active):
    """Terminate every child group, then wait before declaring run failure."""
    rows=[]
    for name,(process,log,begin) in active.items():
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for name,(process,log,begin) in active.items():
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        finally:log.close()
        rows.append({'job_id':name,'pid':process.pid,'returncode':process.returncode,'waited':True})
    active.clear();return rows


def execute(max_workers=3):
    v.require(type(max_workers) is int and 1<=max_workers<=3,'at most three single-thread workers')
    path=v.OUTPUT/'plan.json'
    if not path.exists():path=prepare()
    path,plan=checked_plan(path)
    v.require(not (path.parent/'jobs').exists(),'execute never retries or stitches preexisting jobs')
    control=path.parent/'execution';control.mkdir(exist_ok=False)
    journal=(control/'events.jsonl').open('x',buffering=1);active={}
    prior_sigterm=signal.getsignal(signal.SIGTERM)
    def on_termination(signum,frame):raise RuntimeError('supervisor received SIGTERM; stopping owned workers')
    signal.signal(signal.SIGTERM,on_termination)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def event(kind,**fields):
        record={'event':kind,'at_utc':datetime.now(timezone.utc).isoformat(),**fields}
        journal.write(json.dumps(record,sort_keys=True)+'\n');print(json.dumps(record,sort_keys=True),flush=True)
    try:
        pending=[j['job_id'] for j in plan['jobs']];last=time.monotonic()
        event('stage_started',source_sha256=plan['source_sha256'],total_jobs=12)
        while pending or active:
            while pending and len(active)<max_workers:
                name=pending.pop(0);log=(path.parent/(name+'.log')).open('x')
                try:process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',name],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:log.close();raise
                active[name]=(process,log,time.monotonic());event('job_started',job=name,pid=process.pid)
            for name,(process,log,begin) in list(active.items()):
                rc=process.poll()
                if rc is None:
                    jobdir=path.parent/'jobs'/name
                    phase='final' if (jobdir/'final-started.json').exists() else 'search'
                    clock_path=jobdir/(phase+'-started.json')
                    if clock_path.exists():
                        clock=json.loads(clock_path.read_bytes())
                        v.require(time.monotonic()<clock['deadline_monotonic'],
                                  'external hard '+phase+' deadline expired '+name)
                    else:
                        v.require(time.monotonic()-begin<=900,'worker input preparation hard timeout '+name)
                    continue
                log.close();del active[name];event('job_finished',job=name,returncode=rc)
                v.require(rc==0 and (path.parent/'jobs'/name/'completed.json').is_file(),'worker failed without scientific completion '+name)
            if time.monotonic()-last>=45:
                event('heartbeat',completed=12-len(pending)-len(active),pending=len(pending),active=list(active));last=time.monotonic()
            if active:time.sleep(.25)
        result=finalize();event('stage_completed',selected_arm=result['summary']['selected_arm'],next_stage=result['summary']['next_stage'])
        write_new(control/'completed.json',dict(v.ZERO,status='completed',summary=v.pointer(path.parent/'summary.json'),
            independent_audit=v.pointer(path.parent/'independent-audit.json'),selected_arm=result['summary']['selected_arm'],
            next_stage=result['summary']['next_stage'],automatic_retry=False))
        return result
    except BaseException as exc:
        stopped=stop_workers(active)
        write_new(control/'failed.json',dict(v.ZERO,error_type=type(exc).__name__,error=str(exc),
            automatic_retry=False,all_owned_workers_waited=True,stopped_workers=stopped))
        event('execution_failed',error_type=type(exc).__name__,error=str(exc),all_owned_workers_waited=True)
        raise
    finally:
        journal.close();signal.signal(signal.SIGTERM,prior_sigterm)


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true')
    group.add_argument('--finalize',action='store_true');group.add_argument('--job')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-workers',type=int,choices=(1,2,3),default=3)
    a=p.parse_args()
    if a.resume and not a.job:p.error('--resume is an explicit single safe-job operation')
    if a.prepare:print(prepare())
    elif a.job:job(a.job,a.resume)
    elif a.finalize:print(json.dumps(finalize(),sort_keys=True))
    else:execute(a.max_workers)

if __name__=='__main__':main()
