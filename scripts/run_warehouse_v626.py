#!/usr/bin/env python3
"""Single-use A0 comparison on spent episodes. No collection or NN updates."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v626_protocol as v
from scripts.run_warehouse_v625 import write_new
from backend.training.warehouse_v622_program import _arrays,masked_probabilities
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from backend.training.warehouse_v626_domain import validate_rows,domain_sha
from backend.training.warehouse_v626_certificate import region,structure_info
from backend.training.warehouse_v626_program import Program,fit_leaf,complete_program
from backend.training.warehouse_v626_budget import Budget
from backend.training.warehouse_v625_checkpoint import BudgetExhausted,DeadlineExceeded
from env.warehouse.navigation import ACTIONS


def internal_split(rows):
    scenes={str(r['base_seed']):int(r['task']) for r in rows};groups=defaultdict(list)
    for scene,task in scenes.items():groups[task].append(scene)
    selected=set()
    for task,values in sorted(groups.items()):
        ordered=sorted(values,key=lambda scene:v.digest([v.CANDIDATE+':inner-fit-v1',task,scene]))
        selected.update(ordered[:max(1,int(np.ceil(.2*len(ordered))))])
    choose=np.array([str(r['base_seed']) in selected for r in rows])
    v.require(choose.any() and (~choose).any(),'empty inner fit partition')
    return ~choose,choose


def prepare():
    folder=v.OUTPUT/'A0';v.require(not folder.exists(),'A0 attempt already exists; no automatic retry')
    recipe=v.checked_recipe();source=v.source_sha();files=v.source_files()
    baseline,initial=v.previous.checked_baseline()
    actor,actorpath=v.old.checked_actor(ROOT);episodes,manifest=v.old.checked_inputs(ROOT)
    folder.mkdir(parents=True,exist_ok=False)
    snapshots=folder/'source'/source
    for name,expected in files.items():
        raw=(ROOT/name).read_bytes();v.require(v.file_sha(ROOT/name)==expected,'source changed during snapshot')
        p=snapshots/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
    datasets={};folds=v.old.grouped_folds();canonical={}
    for fold in folds:
        for role in v.old.ROLES:
            data={};meta={}
            for split in v.old.SPLITS:
                blocks=[];rows=[]
                for case in fold['splits'][split]:
                    key0=(case['base'],role);case_rows=episodes[case['base']][role];rows.extend(case_rows)
                    if key0 not in canonical:
                        x0,m0,_=_arrays(case_rows);validate_rows(x0)
                        # Exact deployed single-frame arithmetic. The historical
                        # target cannot be silently changed by a BLAS batch size.
                        p0=np.concatenate([masked_probabilities(actor.roles[role].logits_encoded(x0[i:i+1]),m0[i:i+1])
                            for i in range(len(x0))])
                        v.require(np.array_equal(p0,np.asarray([r['nn_probabilities'] for r in case_rows],np.float32)),
                            'canonical single-frame NN differs from its bound historical recording')
                        canonical[key0]=(x0,m0,p0)
                    blocks.append(canonical[key0])
                x,m,p=(np.concatenate([block[i] for block in blocks]) for i in range(3))
                data.update({split+'_x':x,split+'_m':m,split+'_p':p,
                    split+'_sampled':np.array([ACTIONS.index(r['sampled_action']) for r in rows],np.int64)})
                meta[split]=[dict(episode=str(r['episode']),base_seed=str(r['base_seed']),task=int(r['task'])) for r in rows]
            key=f"fold-{fold['fold']}-{role}";directory=folder/'datasets'/key;directory.mkdir(parents=True)
            with (directory/'arrays.npz').open('xb') as h:np.savez_compressed(h,**data);h.flush();os.fsync(h.fileno())
            (directory/'arrays.npz').chmod(0o444);write_new(directory/'metadata.json',meta)
            tr,se=internal_split(meta['fit'])
            datasets[key]=dict(arrays=v.pointer(directory/'arrays.npz'),metadata=v.pointer(directory/'metadata.json'),
                inner_training_scenarios=sorted({r['base_seed'] for r,keep in zip(meta['fit'],tr,strict=True) if keep}),
                inner_selection_scenarios=sorted({r['base_seed'] for r,keep in zip(meta['fit'],se,strict=True) if keep}))
    plan=dict(v.ZERO,schema='warehouse-v626-A0-plan.v1',candidate_id=v.CANDIDATE,stage='A0',
        source_sha256=source,source_files=files,recipe=recipe,runtime=runtime_identity(),
        actor=v.pointer(actorpath),actor_role_sha256=v.old.ROLE_SHA256,inputs=manifest,
        initial_S1=initial,baseline=baseline,folds=folds,datasets=datasets,jobs=v.jobs(),
        domain_sha256=domain_sha(),target_arithmetic='canonical_frozen_NN_single_physical_frame_numpy',
        automatic_retry=False,performance_blocks_science=False)
    v.require(v.source_sha()==source,'source changed while preparing A0')
    write_new(folder/'plan.json',plan);write_new(folder/'plan.claim.json',dict(sha256=v.file_sha(folder/'plan.json'),single_use=True))
    return folder/'plan.json'


def checked_plan():
    path=v.OUTPUT/'A0/plan.json';plan=json.loads(path.read_bytes())
    claim=json.loads(path.with_name('plan.claim.json').read_bytes())
    v.require(claim==dict(sha256=v.file_sha(path),single_use=True),'A0 claim changed')
    v.require(plan.get('schema')=='warehouse-v626-A0-plan.v1' and plan.get('source_sha256')==v.source_sha()
        and plan.get('source_files')==v.source_files() and plan.get('jobs')==v.jobs()
        and plan.get('recipe')==v.checked_recipe() and plan.get('runtime')==runtime_identity()
        and plan.get('domain_sha256')==domain_sha() and plan.get('actor_role_sha256')==v.old.ROLE_SHA256
        and plan.get('performance_blocks_science') is False,'A0 source/recipe/model binding changed')
    for k,value in v.ZERO.items():v.require(type(plan.get(k)) is type(value) and plan[k]==value,'A0 authority changed')
    v.require(v.pointer(plan['actor']['path'])==plan['actor'] and plan['actor']['sha256']==v.old.ACTOR_SHA256,'complete NN asset changed')
    for name,expected in plan['source_files'].items():
        p=path.parent/'source'/plan['source_sha256']/name
        v.require(v.file_sha(p)==expected and not p.stat().st_mode&0o222,'source snapshot changed')
    return path,plan


def load_parts(plan,key):
    dataset=plan['datasets'][key]
    for ref in (dataset['arrays'],dataset['metadata']):v.require(v.pointer(ref['path'])==ref,'bound scientific inputs changed')
    meta=json.loads(Path(dataset['metadata']['path']).read_bytes())
    with np.load(dataset['arrays']['path'],allow_pickle=False) as z:
        parts={s:{k:z[s+'_'+k].copy() for k in ('x','m','p','sampled')} for s in v.old.SPLITS}
    for s,a in parts.items():
        a['rows']=meta[s];validate_rows(a['x'])
        v.require(np.all((a['m']==0)|(a['m']==1)) and np.all(a['p'][a['m']==0]==0)
            and np.allclose(a['p'].sum(1),1,atol=1e-6,rtol=0),'invalid complete NN targets or physical masks')
    return parts


def initial_models(spec,structure,parts,budget,maximum_leaves=32):
    p=Program(spec['role'],v.old.ROLE_SHA256[spec['role']],structure,[],spec['arm'],maximum_leaves)
    fit=parts['fit'];tr,se=internal_split(fit['rows']);routes=p.leaf_indices(fit['x']);proofs=p.certificates()
    for i in range(structure['leaf_count']):
        train=(routes==i)&tr;selection=(routes==i)&se
        p.leaves.append(fit_leaf(fit['x'][train],fit['m'][train],fit['p'][train],arm=spec['arm'],
            path=region(structure,i,maximum_leaves),legal=proofs[i],budget=budget,
            selection=(fit['x'][selection],fit['m'][selection],fit['p'][selection])))
        if (i+1)%8==0:
            budget.checkpoint(dict(cursor=i+1,selected_leaves=p.leaves,structure=structure,
                stage='initial_fit',algorithm_deterministic=True,all_queue_items_preserved=True))
    return p


def job(job_id):
    path,plan=checked_plan();spec=next((j for j in plan['jobs'] if j['job_id']==job_id),None)
    v.require(spec is not None,'unregistered A0 unit')
    folder=path.parent/'jobs'/job_id;folder.mkdir(parents=True,exist_ok=False)
    key=f"fold-{spec['fold']}-{spec['role']}";parts=load_parts(plan,key)
    structure=structure_info(v.bound(plan['initial_S1'][key]['structure']))[0]
    started=time.monotonic();write_new(folder/'started.json',dict(job=spec,plan_sha256=v.file_sha(path),
        started_monotonic=started,deadline_monotonic=started+1200.,**v.ZERO))
    budget=Budget(folder/'checkpoint.json',dict(plan_sha256=v.file_sha(path),job=spec),
        v.recipe()['limits_A0'],1200,started_at=started);budget.final_leaf_reserve=256
    try:
        with budget.session.hard_deadline():
            program=initial_models(spec,structure,parts,budget)
            budget.phase='final';final_begin=time.monotonic()
            write_new(folder/'final-started.json',dict(deadline_monotonic=min(started+1200.,final_begin+600.)))
            program,result_metrics=complete_program(program,parts,budget)
            if time.monotonic()-final_begin>=600:raise DeadlineExceeded('A0 final 600 second budget exhausted')
            write_new(folder/'program.json',program.to_dict())
            budget.checkpoint(dict(stage='completed',structure=structure,program_sha256=v.file_sha(folder/'program.json')),status='completed')
        result=dict(schema='warehouse-v626-A0-result.v1',job=spec,plan_sha256=v.file_sha(path),status='completed',
            program=v.pointer(folder/'program.json'),metrics=result_metrics,gate=v.gate(result_metrics['validation']),
            solver=budget.summary(),elapsed_seconds=time.monotonic()-started,
            performance_pass=budget.summary()['checkpoint']['checkpoint_fraction']<=.10,
            scientific_gate_ignores_performance=True,**v.ZERO)
        v.require(v.file_sha(plan['actor']['path'])==v.old.ACTOR_SHA256,'NN asset changed during scientific fit')
    except (BudgetExhausted,DeadlineExceeded) as error:
        result=dict(schema='warehouse-v626-A0-result.v1',job=spec,plan_sha256=v.file_sha(path),
            status='scientific_budget_exhausted',gate={'passed':False},error=str(error),
            solver=budget.summary(),elapsed_seconds=time.monotonic()-started,**v.ZERO)
        budget.session.close()
    except BaseException as error:
        write_new(folder/'failed.json',dict(error_type=type(error).__name__,error=str(error),solver=budget.summary(),**v.ZERO))
        budget.session.close();raise
    write_new(folder/'result.json',result)
    v.require(time.monotonic()-started<=1200,'A0 result commit exceeded total deadline')
    write_new(folder/'completed.json',dict(result_sha256=v.file_sha(folder/'result.json')))
    print(json.dumps(dict(event='completed',job=job_id,status=result['status'],passed=result['gate']['passed'])),flush=True)
    return result


def stop_workers(active):
    rows=[]
    for spec,process,log in active:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for spec,process,log in active:
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        log.close();rows.append(dict(job=spec,pid=process.pid,returncode=process.returncode,waited=True))
    active.clear();return rows


def execute():
    path,plan=checked_plan();control=path.parent/'execution';control.mkdir(exist_ok=False)
    v.require(not (path.parent/'jobs').exists(),'A0 never retries existing scientific units')
    pending=list(plan['jobs']);active=[];events=(control/'events.jsonl').open('x',buffering=1)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise RuntimeError('A0 supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        while pending or active:
            while pending and len(active)<3:
                spec=pending.pop(0);log=(control/(spec['job_id']+'.log')).open('x')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',spec['job_id']],
                    cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
                active.append((spec,process,log));events.write(json.dumps(dict(event='started',job=spec,pid=process.pid))+'\n')
            for item in list(active):
                spec,process,log=item;folder=path.parent/'jobs'/spec['job_id'];marker=folder/'started.json'
                if marker.exists():
                    a=json.loads(marker.read_bytes());deadline=a['deadline_monotonic']
                    final=folder/'final-started.json'
                    if final.exists():deadline=min(deadline,json.loads(final.read_bytes())['deadline_monotonic'])
                    if process.poll() is None and time.monotonic()>=deadline:
                        raise DeadlineExceeded('external A0 deadline: '+spec['job_id'])
                code=process.poll()
                if code is not None:
                    process.wait();log.close();active.remove(item)
                    events.write(json.dumps(dict(event='exited',job=spec,returncode=code))+'\n')
                    if code:raise RuntimeError('scientific worker failed: '+spec['job_id'])
            if active:time.sleep(.25)
        write_new(control/'completed.json',dict(all_owned_workers_waited=True,jobs=24))
    except BaseException as error:
        workers=stop_workers(active);write_new(control/'failed.json',dict(error_type=type(error).__name__,error=str(error),
            all_owned_workers_waited=True,workers=workers,automatic_retry=False,**v.ZERO));raise
    finally:
        events.close();signal.signal(signal.SIGTERM,previous)
    return finalize()


def finalize():
    from scripts.audit_warehouse_v626 import audit,report
    path,plan=checked_plan();results=[]
    for spec in plan['jobs']:
        folder=path.parent/'jobs'/spec['job_id'];a=json.loads((folder/'result.json').read_bytes())
        v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),'incomplete scientific unit')
        v.require(a['job']==spec and a['plan_sha256']==v.file_sha(path),'unbound scientific unit')
        results.append(a)
    receipt=audit(path,results);v.require(receipt['integrity_passed'],'independent numerical audit failed')
    write_new(path.parent/'independent-audit.json',receipt)
    all_pass=all(r['status']=='completed' and r['gate']['passed'] for r in results if r['job']['arm']=='C3')
    summary=dict(schema='warehouse-v626-A0-summary.v1',candidate_id=v.CANDIDATE,all_24_completed=True,
        scientific_pass=all_pass,next_stage='B' if all_pass else 'TAO32',
        engineering_performance_pass=all(r.get('performance_pass',False) for r in results),
        results=[v.pointer(path.parent/'jobs'/j['job_id']/'result.json') for j in plan['jobs']],
        independent_audit=v.pointer(path.parent/'independent-audit.json'),source_sha256=plan['source_sha256'],**v.ZERO)
    write_new(path.parent/'summary.json',summary);report(path.parent,results,summary)
    write_new(ROOT/'experiments/warehouse/evidence/warehouse-v626-A0.json',
        dict(summary=v.pointer(path.parent/'summary.json'),scientific_pass=all_pass,next_stage=summary['next_stage'],**v.ZERO))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true');mode.add_argument('--job');mode.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.prepare:print(prepare())
    elif args.job:job(args.job)
    elif args.finalize:print(json.dumps(finalize(),sort_keys=True))
    else:print(json.dumps(execute(),sort_keys=True))
