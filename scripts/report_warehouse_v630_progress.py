#!/usr/bin/env python3
"""Read-only progress: live worker claims, settled metrics, verified handles.

No data collection, fitting, DP, model inference or candidate selection.
This observer is outside the immutable scientific source closure.
"""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v
from backend.training.warehouse_v625_checkpoint import ContentStore


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.writing')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def checkpoint(folder,job,plan_ref,role_sha):
    raw=(folder/'checkpoint.json').read_bytes();manifest=json.loads(raw);started=v.bound(v.pointer(folder/'started.json'))
    expected=dict(plan_sha256=plan_ref['sha256'],job=job,source=started['source'],limits=v.recipe()['limits'])
    v.require(started['job']==job and started['plan']==plan_ref and manifest['identity_sha256']==v.digest(expected),
        'progress checkpoint is not this plan/job/source')
    store=ContentStore(str(folder/'checkpoint.json')+'.objects');store.load_registry(manifest['object_pack_registry'])
    state=store.get_json(manifest['state']);stage=state.get('stage')
    result=dict(stage=stage,checkpoint_sha256=sha256(raw).hexdigest(),solver_counts=state.get('solver_counts'),
        inspected_leaves=len(state.get('inspected_leaves',[])),candidate_count=len(state.get('candidates',[])))
    source=v.bound(started['source']['program']);result['source_leaf_count']=len(source['leaves'])
    if stage=='allocation' and 'best_candidate' in state:
        best=state['best_program'];index=state['best_candidate'];candidate=state['candidates'][index]
        v.require(best['role']==job['role'] and best['actor_role_sha256']==role_sha and best['maximum_leaves']==256,
            'live selected program role/anchor differs')
        result.update(scope='fit_only_150_scene_worker_score_not_outer_validation_or_admission',
            metrics=candidate['metrics'],reliable_selection_counts=candidate['reliable_selection_counts'],
            reliable_selection_passed=candidate['reliable_selection_passed'],leaves=len(best['leaves']),
            best_candidate=index,program_sha256=v.digest(best))
    return result


def worker_handles(output):
    path=output/'execution/events.jsonl'
    if not path.exists():return {}
    # A concurrently appended final line is not a terminal-state signal.
    rows=[json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n') and line.strip()];active={}
    for row in rows:
        key=row['job']['job_id']
        if row['event']=='started':active[key]=row['pid']
        elif row['event']=='exited':active.pop(key,None)
    return active


def collect(path,plan,*,supervisor_pid=None,cache=None):
    ref=v.pointer(path);rows=[];errors=[];active=worker_handles(v.OUTPUT);cache={} if cache is None else cache
    for job in plan['jobs']:
        folder=v.OUTPUT/'jobs'/job['job_id'];started=folder/'started.json';sealed=folder/'completed.json'
        row=dict(job=job,status='not_started')
        try:
            if sealed.exists():
                result_ref=v.pointer(folder/'result.json');result=v.bound(result_ref)
                v.require(v.bound(v.pointer(sealed))==dict(result_sha256=result_ref['sha256']) and result['job']==job and result['plan']==ref,
                    'settled result seal, job or plan differs')
                row.update(status=result['status'],result=result_ref,metrics=result.get('metrics',{}).get('validation'),
                    gate=result['gate'],leaves=result.get('leaf_count'),depth=result.get('depth'),
                    reference_metrics=result.get('reference_metrics',{}).get('validation'),stop_reason=result.get('error'),
                    solver_counts=(result.get('solver') or {}).get('counts'))
            elif started.exists():
                pid=active.get(job['job_id']);identity=v.process_identity(pid)
                verified=identity is not None and f'run_warehouse_v630.py --job {job["job_id"]}' in identity
                row.update(status='running' if verified else 'unsettled_worker_not_verified',pid=pid,verified_process_identity=identity if verified else None)
                file=folder/'checkpoint.json'
                if file.exists():
                    stamp=(file.stat().st_mtime_ns,file.stat().st_size)
                    previous=cache.get(job['job_id'])
                    if previous is None or previous[0]!=stamp:
                        partial=checkpoint(folder,job,ref,plan['actor_role_sha256'][job['role']]);cache[job['job_id']]=(stamp,partial)
                    row['partial']=cache[job['job_id']][1]
        except Exception as error:
            errors.append(dict(job=job['job_id'],error_type=type(error).__name__,error=str(error)))
            row['observation_error']=str(error)
        rows.append(row)
    if (v.OUTPUT/'candidate.skipped.json').exists():dependency='skipped_for_upstream_winner'
    elif (v.OUTPUT/'dependency-cleared.json').exists():dependency='cleared'
    else:
        expected=plan['dependency_process'];identity=None if expected is None else v.process_identity(expected['pid'])
        dependency='waiting_verified_v629_driver' if expected is not None and identity==expected['identity'] else 'waiting_for_terminal_dependency_evidence'
    supervisor=v.process_identity(supervisor_pid) if supervisor_pid is not None else None
    result=dict(schema='warehouse-v630-read-only-progress.v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        plan=ref,phase='frozen_NN_program_extraction_only',dependency=dependency,supervisor_pid=supervisor_pid,supervisor_verified=supervisor is not None
            and 'run_warehouse_v630.py --pipeline' in supervisor,rows=rows,errors=errors,
        settled=sum('result' in row for row in rows),total=len(plan['jobs']),
        actor_updates=0,PPO_joint_steps=0,new_environment_resets=0,observer_fits=0,observer_DP_calls=0,pilot_opened=False)
    return result


def markdown(result):
    pct=lambda value:'—' if value is None else f'{100*value:.2f}%'
    number=lambda value:'—' if value is None else f'{value:.4f}'
    lines=['# Warehouse v6.30 实测进度','',f"依赖：{result['dependency']}；已结算 {result['settled']}/{result['total']}。",'',
        '## 搜索前缀（工作进程的拟合集内部评分，不是独立验证）','',
        '| 方法/折/角色 | 阶段 | 已检查旧叶 | 分配候选 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL |',
        '|---|---|---|---|---|---|---|---|---|']
    partials=0
    for row in result['rows']:
        p=row.get('partial')
        if p is None:continue
        partials+=1;f=p.get('metrics',{}).get('feedback',{});e=p.get('metrics',{}).get('explanation',{});j=row['job']
        lines.append('| '+' | '.join([f"{j['name']}/{j['fold']}/{j['role']}",str(p['stage']),f"{p['inspected_leaves']}/{p['source_leaf_count']}",
            str(p['candidate_count']),pct(f.get('coverage')),pct(e.get('coverage')),pct(f.get('expected_actionable_coverage')),
            pct(f.get('agreement'))+'/'+pct(e.get('agreement')),number(f.get('kl'))+'/'+number(e.get('kl'))])+' |')
    if not partials:lines+=['','尚无已提交的新搜索前缀；未测值不记0。']
    lines+=['','## 外层共同开发集（不是新独立确认、封存或留出）','',
        '| 方法/折/角色 | 状态 | 叶/深 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 通过 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for row in result['rows']:
        j=row['job'];m=row.get('metrics') or {};f=m.get('feedback',{});e=m.get('explanation',{})
        lines.append('| '+' | '.join([f"{j['name']}/{j['fold']}/{j['role']}",row['status'],str(row.get('leaves','—'))+'/'+str(row.get('depth','—')),
            pct(f.get('coverage')),pct(e.get('coverage')),pct(f.get('expected_actionable_coverage')),
            pct(f.get('agreement'))+'/'+pct(e.get('agreement')),number(f.get('kl'))+'/'+number(e.get('kl')),
            str(e.get('whole_tree_raw_illegal_count','—'))+'/'+str(e.get('whole_tree_path_uncertified_count','—')),
            str(row.get('gate',{}).get('passed','待测'))])+' |')
        if row.get('stop_reason'):lines+=['',f"{j['job_id']} 停止原因：{row['stop_reason']}"]
    lines+=['','门槛保持F/E≥45%、expected F≥40%、一致≥95%/97%、KL≤0.20、非法/未证0；必须逐角色/折通过。',
        'NN和原程序的全树输出冻结；覆盖率变化不等于NN任务能力或全树保真提升。观察器不拟合、不分配叶预算，不影响准入。']
    return '\n'.join(lines)+'\n'


def main(*,watch=False,supervisor_pid=None):
    path,plan=v.checked_plan();output=v.OUTPUT/'progress';cache={}
    while True:
        result=collect(path,plan,supervisor_pid=supervisor_pid,cache=cache);atomic(output/'latest.json',result)
        output.mkdir(parents=True,exist_ok=True);temp=output/f'latest.md.{os.getpid()}.writing';temp.write_text(markdown(result));os.replace(temp,output/'latest.md')
        if not watch or any((v.OUTPUT/name).exists() for name in ('summary.json','driver.failed.json','candidate.skipped.json')):return result
        time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--supervisor-pid',type=int)
    args=parser.parse_args();main(watch=args.watch,supervisor_pid=args.supervisor_pid)
