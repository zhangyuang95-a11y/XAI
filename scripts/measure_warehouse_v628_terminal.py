#!/usr/bin/env python3
"""Measure terminal budget-stop checkpoints, never resume or admit them.

Outer DEVELOPMENT data only; no sealed data, fitting, LP, NN or environment
calls. New eligibility is recomputed for unchanged checkpoint coefficients.
These observations cannot override the original job's failed admission.
"""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v,v628_data as data
from backend.training.warehouse_v628_program import qualify,measure,payload,ParentProgram,from_parent
from backend.training.warehouse_v622_program import digest,gate_metrics
from scripts.run_warehouse_v629 import initial_program,process_identity
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic,same_metrics
from scripts.run_warehouse_v625 import write_new


def waited_jobs(path):
    if not path.is_file():return {}
    raw=path.read_bytes();rows={}
    for line in raw.splitlines():
        try:record=json.loads(line)
        except json.JSONDecodeError:continue
        if record.get('event')=='exited' and record.get('waited') is True:rows[record['job']['job_id']]=record
    return rows


def measure_job(plan_path,plan,job,exit_event,output):
    folder=v.OUTPUT/'jobs'/job['job_id'];result=v.bound(v.pointer(folder/'result.json'))
    v.require(exit_event['job']==job and exit_event['waited'] is True,'must observe owned-worker waitpid before terminal measurement')
    v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),'terminal result seal changed')
    v.require(result['job']==job and result['status']=='scientific_budget_exhausted','only a sealed terminal budget-stop can use this diagnostic')
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'started.json',dict(terminal_result=v.pointer(folder/'result.json'),worker_exit=exit_event,observer_sha256=v.file_sha(__file__)))
    source_plan=dict(previous_plan=v.pointer(plan_path),actor_role_sha256=plan['actor_role_sha256'])
    p,source,_=initial_program(source_plan,job)
    before=digest([payload(leaf) for leaf in p.leaves]);structure=digest(p.structure)
    parts=data.load_parts(plan,job['fold'],job['role'])
    qualify(p,parts)
    v.require(before==digest([payload(leaf) for leaf in p.leaves]) and structure==digest(p.structure),'terminal observer changed coefficients or topology')
    a=parts['validation'];pred=forward(p,a['x'],a['m'])
    v.require(all(np.array_equal(x,y) for x,y in zip(pred,p.distributions(a['x'],a['m']),strict=True)),'independent terminal forward differs')
    m=measure(p,a)
    for purpose,selected in (('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)):
        same_metrics(basic(p,a,pred,selected,purpose),m[purpose])
    parent_ref=plan['parents'][f"fold-{job['fold']}-{job['role']}"]
    parent=ParentProgram.from_dict(v.bound(parent_ref),plan['actor_role_sha256'][job['role']])
    baseline=from_parent(parent,parent_ref,job['capacity']);qualify(baseline,parts);b=measure(baseline,a)
    write_new(output/'observed-program.json',p.to_dict());write_new(output/'parent-reference.json',baseline.to_dict())
    observation=dict(schema='warehouse-v628-terminal-outer-observation.v1',job=job,
        source=source,terminal_result=v.pointer(folder/'result.json'),worker_exit=exit_event,
        observed_program=v.pointer(output/'observed-program.json'),parent_reference=v.pointer(output/'parent-reference.json'),
        source_plan=v.pointer(plan_path),observer_sha256=v.file_sha(__file__),
        metric_scope='outer_development_validation_after_budget_stop_not_final_job_admission_or_sealed_test',
        metrics=m,parent_metrics=b,metric_only_gate=gate_metrics(m,phase='development'),
        original_status=result['status'],original_admission=False,candidate_admission=False,
        qualification_recomputed_on_outer_eligibility=True,coefficients_and_topology_unchanged=True,
        leaf_count=len(p.leaves),delta_explanation_pp=100*(m['explanation']['coverage']-b['explanation']['coverage']),
        fitting_calls=0,LP_calls=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
    write_new(output/'result.json',observation);return observation


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+str(os.getpid())+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def main(watch=False,driver_pid=None):
    plan_path,plan=v.checked_plan();output=v.OUTPUT/'progress/terminal-outer'
    output.mkdir(parents=True,exist_ok=True);driver=process_identity(driver_pid) if driver_pid else None
    if watch:v.require(driver and 'continue_warehouse_v628_seed_audit_bounded.py' in driver,'verify actual driver before watching')
    rows={};errors=[]
    while True:
        exits=waited_jobs(v.OUTPUT/'execution/events.jsonl')
        for job in plan['jobs']:
            folder=output/job['job_id'];source=v.OUTPUT/'jobs'/job['job_id']
            if (folder/'result.json').is_file():rows[job['job_id']]=json.loads((folder/'result.json').read_bytes());continue
            if job['job_id'] not in exits or not (source/'completed.json').is_file():continue
            r=json.loads((source/'result.json').read_bytes())
            if r['status']!='scientific_budget_exhausted':continue
            if folder.exists():continue  # interrupted measurement is reported, never silently replaced
            try:
                rows[job['job_id']]=measure_job(plan_path,plan,job,exits[job['job_id']],folder)
                print(json.dumps(dict(job=job['job_id'],metrics=rows[job['job_id']]['metrics'])),flush=True)
            except Exception as error:
                errors.append(dict(job=job['job_id'],error_type=type(error).__name__,error=str(error)))
                errors=errors[-30:]
                if folder.is_dir() and not (folder/'failed.json').exists():write_new(folder/'failed.json',errors[-1])
        summary=dict(observed_utc=datetime.now(timezone.utc).isoformat(),scope='post_budget_stop_outer_development_only',
            rows=list(rows.values()),errors=errors,candidate_admission=False,pilot_opened=False)
        atomic(output/'latest.json',summary)
        pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
        num=lambda x:'—' if x is None else f'{x:.4f}'
        lines=['# v6.28 停止后固定检查点的外层开发指标','',
            '**只读诊断：不恢复旧搜索、不补做拟合、不改写原失败结论，不是封存审计或留出。**','',
            '| 容量 | 折/角色 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 原实验准入 |',
            '|---|---|---|---|---|---|---|---|---|']
        for job in plan['jobs']:
            r=rows.get(job['job_id'],{});m=r.get('metrics',{});f=m.get('feedback',{});e=m.get('explanation',{})
            lines.append('| '+' | '.join([str(job['capacity']),str(job['fold'])+'/'+job['role'],pct(f.get('coverage')),
                pct(e.get('coverage')),pct(f.get('expected_actionable_coverage')),pct(f.get('agreement'))+'/'+pct(e.get('agreement')),
                num(f.get('kl'))+'/'+num(e.get('kl')),str(f.get('whole_tree_raw_illegal_count','—'))+'/'+str(f.get('whole_tree_path_uncertified_count','—')),
                '失败，不由此表更改' if r else '待原作业结算'])+' |')
        (output/'latest.md').write_text('\n'.join(lines)+'\n')
        if not watch or process_identity(driver_pid)!=driver:return
        time.sleep(60)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--driver-pid',type=int)
    args=parser.parse_args();main(args.watch,args.driver_pid)
