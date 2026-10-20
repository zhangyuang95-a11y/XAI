#!/usr/bin/env python3
"""Sequentially admitted six-unit TAO32 / TAO64, fit-only selection."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v626_protocol as v
from scripts.run_warehouse_v626 import checked_plan as checked_A0,load_parts,stop_workers
from scripts.run_warehouse_v625 import write_new
from scripts.verify_warehouse_v625_native import real_deadline
from backend.training.warehouse_v625_checkpoint import BudgetExhausted,DeadlineExceeded
from backend.training.warehouse_v626_search import Context,SearchBudget,optimize
from backend.training.warehouse_v626_program import Program,complete_program
from backend.training.warehouse_v626_certificate import structure_info

ADDITIONAL=('backend/training/warehouse_v626_search.py','backend/training/warehouse_v626_checkpoint.py','scripts/run_warehouse_v626_search.py',
    'scripts/audit_warehouse_v626_search.py','configs/warehouse_v626_search_recipe.json')


def recipe():
    return dict(schema='warehouse-v626-TAO-recipe.v1',candidate_id=v.CANDIDATE,
        order=['TAO32','TAO64'],maximum_rounds=14,stagnant_sweeps=2,
        limits=dict(leaf=20000,node=1024,proof=100000),search_seconds=4800,search_settlement_reserve=10,
        final_seconds=600,final_leaf_calls_per_leaf=8,final_proof_calls_per_leaf=52,
        capacities={'TAO32':dict(leaves=32,depth=6),'TAO64':dict(leaves=64,depth=7)},
        node_optimizer=dict(method='L-BFGS-B',maxiter=100,ftol=1e-7,L2=.001,coefficient_bound=32),
        node_supports=['incumbent_support','fit_cost_residual_ranked_support'],legal_guards_immutable=True,
        leaf_cache_entries=512,cache_eviction='FIFO',checkpoint='CAS_leaves_cache_complete_queue_cursor_state_machine',
        selection='fit_internal_80_20_scenario_split_explanation_expected_feedback_negative_KL_negative_leaves',
        growth='first_fit_improvement_from_four_worst_fit_leaves_legal_guard_or_axis_plus_guard',
        maximum_growth_acceptances_per_sweep=4,
        initial_TAO32='original_corresponding_S1_topology',initial_TAO64='TAO32_fit_selected_structure_before_calibration',
        validation_selects_topology=False,performance_blocks_science=False,maximum_workers=3,BLAS_threads=1,**v.ZERO)


def files():
    result=v.source_files();result.update({name:v.file_sha(ROOT/name) for name in ADDITIONAL});return dict(sorted(result.items()))


def expected_jobs(stage):
    return [dict(job_id=f'fold-{fold}-{role}',fold=fold,role=role,arm='C3',stage=stage)
        for fold in range(3) for role in v.old.ROLES]


def prepare(stage):
    if stage not in ('TAO32','TAO64'):raise ValueError('unknown stage')
    path0,a0=checked_A0();summary0=v.bound(v.pointer(path0.parent/'summary.json'))
    v.require(summary0['all_24_completed'] and not summary0['scientific_pass'],'A0 must be complete and C3 failed before TAO')
    audit0=v.bound(summary0['independent_audit']);v.require(audit0['integrity_passed'],'A0 integrity audit required')
    v.require(json.loads((ROOT/'configs/warehouse_v626_search_recipe.json').read_bytes())==recipe(),'search recipe differs')
    source_files=files();source=v.digest(source_files);parent=v.pointer(path0.parent/'summary.json')
    jobs=[];initial={}
    if stage=='TAO64':
        parent_path,old=checked_stage('TAO32');s=v.bound(v.pointer(parent_path.parent/'summary.json'))
        v.require(s['all_six_completed'] and not s['scientific_pass'] and v.bound(s['independent_audit'])['integrity_passed'],
            '64 leaves require the complete audited 32-leaf failure')
        parent=v.pointer(parent_path.parent/'summary.json')
    for fold in range(3):
        for role in v.old.ROLES:
            key=f'fold-{fold}-{role}';jobs.append(dict(job_id=key,fold=fold,role=role,arm='C3',stage=stage))
            if stage=='TAO32':initial[key]=a0['initial_S1'][key]['structure']
            else:
                r=json.loads((parent_path.parent/'jobs'/key/'result.json').read_bytes())
                v.require(r['status']=='completed','64-leaf warm start requires a valid fit-selected 32-leaf artifact')
                initial[key]=r['search_best'];Program.from_dict(v.bound(initial[key]),v.old.ROLE_SHA256[role])
    folder=v.OUTPUT/stage;folder.mkdir(parents=True,exist_ok=False)
    for name,sha in source_files.items():
        p=folder/'source'/source/name;p.parent.mkdir(parents=True,exist_ok=True)
        v.require(v.file_sha(ROOT/name)==sha,'source changed during search preparation');p.write_bytes((ROOT/name).read_bytes());p.chmod(0o444)
    plan=dict(schema='warehouse-v626-search-plan.v1',candidate_id=v.CANDIDATE,stage=stage,
        source_sha256=source,source_files=source_files,recipe=recipe(),jobs=jobs,initial=initial,
        A0_plan=v.pointer(path0),parent=parent,maximum_leaves=32 if stage=='TAO32' else 64,
        actor=a0['actor'],actor_role_sha256=v.old.ROLE_SHA256,automatic_retry=False,**v.ZERO)
    write_new(folder/'plan.json',plan);write_new(folder/'plan.claim.json',dict(sha256=v.file_sha(folder/'plan.json'),single_use=True))
    return folder/'plan.json'


def checked_stage(stage):
    path=v.OUTPUT/stage/'plan.json';plan=json.loads(path.read_bytes())
    v.require(json.loads(path.with_name('plan.claim.json').read_bytes())==dict(sha256=v.file_sha(path),single_use=True),'search plan claim changed')
    v.require(stage in ('TAO32','TAO64') and plan['stage']==stage and plan['source_sha256']==v.digest(files()) and plan['source_files']==files()
        and plan['recipe']==recipe() and plan['actor_role_sha256']==v.old.ROLE_SHA256
        and plan['jobs']==expected_jobs(stage) and type(plan['maximum_leaves']) is int
        and plan['maximum_leaves']==(32 if stage=='TAO32' else 64)
        and plan['actor']['sha256']==v.old.ACTOR_SHA256,'search source/algorithm/Actor binding changed')
    for key,value in v.ZERO.items():v.require(type(plan.get(key)) is type(value) and plan[key]==value,'search execution authority changed')
    checked_A0()
    for key in ('A0_plan','parent','actor'):v.require(v.pointer(plan[key]['path'])==plan[key],'parent/NN asset changed')
    for name,sha in plan['source_files'].items():
        p=path.parent/'source'/plan['source_sha256']/name
        v.require(v.file_sha(p)==sha and not p.stat().st_mode&0o222,'search source snapshot changed')
    return path,plan


def job(stage,job_id):
    path,plan=checked_stage(stage);spec=next((j for j in plan['jobs'] if j['job_id']==job_id),None)
    v.require(spec is not None,'unknown search unit');folder=path.parent/'jobs'/job_id;folder.mkdir(parents=True,exist_ok=False)
    _,a0=checked_A0();parts=load_parts(a0,job_id)
    initial=v.bound(plan['initial'][job_id])
    if stage=='TAO64':initial=Program.from_dict(initial,v.old.ROLE_SHA256[spec['role']]).structure
    initial=structure_info(initial,plan['maximum_leaves'])[0]
    started=time.monotonic();write_new(folder/'started.json',dict(job=spec,plan_sha256=v.file_sha(path),
        started_monotonic=started,search_deadline_monotonic=started+4800,deadline_monotonic=started+5400,**v.ZERO))
    budget=SearchBudget(folder/'checkpoint.json',dict(plan_sha256=v.file_sha(path),job=spec),
        recipe()['limits'],5400,started_at=started,maximum_leaves=plan['maximum_leaves'])
    ctx=Context(parts,{**spec,'actor_role_sha256':v.old.ROLE_SHA256[spec['role']]},budget,plan['maximum_leaves'])
    try:
        try:
            with real_deadline(started,4790):program,search=optimize(ctx,initial)
        except (DeadlineExceeded,BudgetExhausted) as error:
            if not hasattr(ctx,'best'):raise
            program=ctx.best;search=dict(score=list(ctx.best_score),initial_score=ctx.initial_score,stopped=type(error).__name__,trace=ctx.trace,
                evaluations=ctx.evaluations,cache_hits=ctx.cache_hits,cache_evictions=ctx.cache_evictions,
                selection_uses_only_fit=True,validation_used_for_search=False)
        search['actual_search_seconds']=time.monotonic()-started
        v.require(search['actual_search_seconds']<4800,'search settlement exceeded registered phase')
        write_new(folder/'search-best.json',program.to_dict());write_new(folder/'search-report.json',search)
        budget.phase='final';begin=time.monotonic()
        write_new(folder/'final-started.json',dict(deadline_monotonic=min(started+5400,begin+600)))
        with real_deadline(begin,min(600,started+5400-begin)):
            program,metrics=complete_program(program,parts,budget)
            write_new(folder/'program.json',program.to_dict())
            budget.checkpoint(dict(stage='completed',program_sha256=v.file_sha(folder/'program.json')),status='completed')
        result=dict(schema='warehouse-v626-search-result.v1',job=spec,status='completed',plan_sha256=v.file_sha(path),
            maximum_leaves=plan['maximum_leaves'],leaf_count=len(program.leaves),search_best=v.pointer(folder/'search-best.json'),
            search_report=v.pointer(folder/'search-report.json'),program=v.pointer(folder/'program.json'),metrics=metrics,
            gate=v.gate(metrics['validation']),solver=budget.summary(),elapsed_seconds=time.monotonic()-started,
            performance_pass=False,performance_status='full_runner_overhead_not_independently_measured',**v.ZERO)
        v.require(time.monotonic()-begin<600,'final serialization exceeded registered phase')
    except (DeadlineExceeded,BudgetExhausted) as error:
        result=dict(schema='warehouse-v626-search-result.v1',job=spec,status='scientific_budget_exhausted',plan_sha256=v.file_sha(path),
            error=str(error),gate={'passed':False},solver=budget.summary(),elapsed_seconds=time.monotonic()-started,**v.ZERO)
        budget.session.close()
    except BaseException as error:
        write_new(folder/'failed.json',dict(error_type=type(error).__name__,error=str(error),solver=budget.summary(),**v.ZERO))
        budget.session.close();raise
    v.require(v.file_sha(plan['actor']['path'])==v.old.ACTOR_SHA256,'NN changed during surrogate search')
    write_new(folder/'result.json',result);write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    print(json.dumps(dict(job=job_id,stage=stage,status=result['status'],passed=result['gate']['passed'])),flush=True)
    return result


def execute(stage):
    path,plan=checked_stage(stage);control=path.parent/'execution';control.mkdir(exist_ok=False)
    v.require(not (path.parent/'jobs').exists(),'never retry existing search jobs')
    pending=list(plan['jobs']);active=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise RuntimeError('search supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        while pending or active:
            while pending and len(active)<3:
                spec=pending.pop(0);log=(control/(spec['job_id']+'.log')).open('x')
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',spec['job_id'],'--stage',stage],cwd=ROOT,
                    env=env,stdout=log,stderr=log,start_new_session=True)
                active.append((spec,p,log))
            for item in list(active):
                spec,p,log=item;folder=path.parent/'jobs'/spec['job_id'];start=folder/'started.json'
                if start.exists():
                    a=json.loads(start.read_bytes());deadline=a['search_deadline_monotonic'];final=folder/'final-started.json'
                    if final.exists():deadline=min(a['deadline_monotonic'],json.loads(final.read_bytes())['deadline_monotonic'])
                    if p.poll() is None and time.monotonic()>=deadline:raise DeadlineExceeded('external search deadline '+spec['job_id'])
                code=p.poll()
                if code is not None:
                    p.wait();log.close();active.remove(item)
                    if code:raise RuntimeError('search worker failed '+spec['job_id'])
            if active:time.sleep(.25)
        write_new(control/'completed.json',dict(all_owned_workers_waited=True,jobs=6))
    except BaseException as error:
        workers=stop_workers(active);write_new(control/'failed.json',dict(error_type=type(error).__name__,error=str(error),
            all_owned_workers_waited=True,workers=workers,automatic_retry=False,**v.ZERO));raise
    finally:signal.signal(signal.SIGTERM,previous)
    return finalize(stage)


def finalize(stage):
    from scripts.audit_warehouse_v626_search import audit,report
    path,plan=checked_stage(stage);results=[]
    for spec in plan['jobs']:
        folder=path.parent/'jobs'/spec['job_id'];r=json.loads((folder/'result.json').read_bytes())
        v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),'incomplete search unit')
        v.require(r['job']==spec and r['plan_sha256']==v.file_sha(path),'unbound search unit');results.append(r)
    integrity=audit(path,results);v.require(integrity['integrity_passed'],'search arithmetic audit failed')
    write_new(path.parent/'independent-audit.json',integrity)
    passed=all(r['status']=='completed' and r['gate']['passed'] for r in results)
    summary=dict(schema='warehouse-v626-search-summary.v1',candidate_id=v.CANDIDATE,stage=stage,
        source_sha256=plan['source_sha256'],all_six_completed=True,scientific_pass=passed,
        next_stage='B' if passed else 'TAO64' if stage=='TAO32' else 'closed_failed_program_precheck',
        independent_audit=v.pointer(path.parent/'independent-audit.json'),
        results=[v.pointer(path.parent/'jobs'/j['job_id']/'result.json') for j in plan['jobs']],
        engineering_performance_pass=False,**v.ZERO)
    write_new(path.parent/'summary.json',summary);report(path.parent,results,summary)
    write_new(ROOT/f'experiments/warehouse/evidence/warehouse-v626-{stage}.json',
        dict(summary=v.pointer(path.parent/'summary.json'),scientific_pass=passed,next_stage=summary['next_stage'],**v.ZERO))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('TAO32','TAO64'),required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');g.add_argument('--finalize',action='store_true');g.add_argument('--job')
    a=p.parse_args()
    if a.prepare:print(prepare(a.stage))
    elif a.job:job(a.stage,a.job)
    elif a.finalize:print(json.dumps(finalize(a.stage),sort_keys=True))
    else:print(json.dumps(execute(a.stage),sort_keys=True))
