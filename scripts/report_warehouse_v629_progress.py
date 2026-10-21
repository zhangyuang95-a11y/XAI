#!/usr/bin/env python3
"""Read-only numeric progress, separate inner diagnostics and outer results."""
from pathlib import Path
import argparse, json, os, sys, time
from hashlib import sha256
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v629_protocol as v,v628_data as data
from backend.training.warehouse_v625_checkpoint import ContentStore
from backend.training.warehouse_v629_program import Program,measure
from scripts.run_warehouse_v629 import initial_program


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def load_snapshot(folder,job,plan_sha,role_sha):
    raw=(folder/'checkpoint.json').read_bytes();manifest=json.loads(raw)
    start=json.loads((folder/'started.json').read_bytes())
    expected=v.digest(dict(plan_sha256=plan_sha,job=job,initial=start['initial'],
        solver_limits=v.recipe()['limits'],budget_contract='v629-cumulative-common-max256-reserve.v1'))
    v.require(start['plan_sha256']==plan_sha and manifest['identity_sha256']==expected,'live checkpoint binding differs')
    root=Path(str(folder/'checkpoint.json')+'.objects');v.require(root.is_dir(),'missing CAS store')
    store=ContentStore(root);store.load_registry(manifest['object_pack_registry']);state=store.get_json(manifest['state'])
    if state.get('algorithm')!='cumulative_local_certified_repair_v1':return None
    ref=state['best']['program'];p=Program.from_dict(store.get_json(ref),role_sha)
    v.require(p.role==job['role'] and p.maximum_leaves==job['capacity'],'checkpoint role/capacity differs')
    return dict(program=p,program_sha256=ref['sha256'],evaluations=state['evaluations'],
        solver_counts=state['solver_counts'],reused_leaves=state['reused_leaves'],
        mandatory_checked=len(state['machine']['mandatory_checked']),mandatory_total=len(state['machine']['mandatory_paths']),
        checkpoint_sha256=sha256(raw).hexdigest())


def main(watch=False):
    path,plan=v.checked_plan();oldplan=v.bound(plan['previous_plan']);rows={};loaded={};seen={};errors=[]
    output=v.OUTPUT/'progress'
    while True:
        for job in plan['jobs']:
            folder=v.OUTPUT/'jobs'/job['job_id'];checkpoint=folder/'checkpoint.json'
            if not checkpoint.is_file() or (folder/'completed.json').is_file():continue
            stamp=checkpoint.stat().st_mtime_ns
            if seen.get(job['job_id'])==stamp:continue
            try:
                snap=load_snapshot(folder,job,v.file_sha(path),plan['actor_role_sha256'][job['role']])
                seen[job['job_id']]=stamp
                if snap is None:continue
                if rows.get(job['job_id'],{}).get('program_sha256')==snap['program_sha256']:
                    rows[job['job_id']].update({k:value for k,value in snap.items() if k!='program'});continue
                key=(job['fold'],job['role'])
                if key not in loaded:
                    # Keep at most two 900-episode inner partitions in memory.
                    if len(loaded)>=2:loaded.pop(next(iter(loaded)))
                    loaded[key]=data.load_parts(oldplan,job['fold'],job['role'],inner=True)
                parts=loaded[key];initial,ref,_=initial_program(plan,job)
                m=measure(snap['program'],parts['validation']);b=measure(initial,parts['validation'])
                row=dict(job=job,scope='inner_180_scene_score_not_outer_validation_or_admission',
                    metrics=m,initial_metrics=b,initial=ref,
                    **{k:value for k,value in snap.items() if k!='program'},
                    delta_explanation_pp=100*(m['explanation']['coverage']-b['explanation']['coverage']))
                rows[job['job_id']]=row
                atomic(output/'inner-score'/job['job_id']/(snap['program_sha256']+'.json'),row)
            except Exception as error:
                errors.append(dict(job=job['job_id'],error_type=type(error).__name__,error=str(error)))
                errors=errors[-50:]
        terminal=[];pending=[]
        for job in plan['jobs']:
            folder=v.OUTPUT/'jobs'/job['job_id']
            if (folder/'completed.json').is_file():
                v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),
                    'observed final result seal differs')
                terminal.append(json.loads((folder/'result.json').read_bytes()))
            else:pending.append(dict(job=job,status='running' if (folder/'started.json').is_file() else 'not_started'))
        result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),inner_rows=list(rows.values()),
            outer_results=terminal,pending=pending,errors=errors,
            dependency='cleared' if (v.OUTPUT/'dependency-cleared.json').is_file() else 'waiting_verified_v628_driver',
            actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
        atomic(output/'latest.json',result)
        pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
        num=lambda x:'—' if x is None else f'{x:.4f}'
        lines=['# Warehouse v6.29 实测进度','',f"依赖状态：{result['dependency']}；结算 {len(terminal)}/24。",'',
            '## 内部搜索指标（非独立验证或验收）','',
            '| 臂 | 折/角色 | 候选数 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 相对起点E |',
            '|---|---|---|---|---|---|---|---|---|']
        for row in rows.values():
            j=row['job'];f=row['metrics']['feedback'];e=row['metrics']['explanation']
            lines.append('| '+' | '.join([j['name'],str(j['fold'])+'/'+j['role'],str(row['evaluations']),
                pct(f['coverage']),pct(e['coverage']),pct(f['expected_actionable_coverage']),
                pct(f['agreement'])+'/'+pct(e['agreement']),num(f['kl'])+'/'+num(e['kl']),f"{row['delta_explanation_pp']:+.2f}pp"] )+' |')
        if not rows:lines+=['','尚无已提交优化程序；未测值不记作0，代码测试不算效果指标。']
        lines+=['','## 外层共同开发集（非新独立确认）','',
            '| 臂 | 折/角色 | 状态 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 | 通过 |',
            '|---|---|---|---|---|---|---|---|---|---|']
        byid={r['job']['job_id']:r for r in terminal+pending}
        for job in plan['jobs']:
            r=byid[job['job_id']];m=r.get('metrics',{}).get('validation',{});f=m.get('feedback',{});e=m.get('explanation',{})
            lines.append('| '+' | '.join([job['name'],str(job['fold'])+'/'+job['role'],r['status'],
                pct(f.get('coverage')),pct(e.get('coverage')),pct(f.get('expected_actionable_coverage')),
                pct(f.get('agreement'))+'/'+pct(e.get('agreement')),num(f.get('kl'))+'/'+num(e.get('kl')),
                str(f.get('whole_tree_raw_illegal_count','—'))+'/'+str(f.get('whole_tree_path_uncertified_count','—')),
                str(r.get('gate',{}).get('passed','待测'))])+' |')
        lines+=['','门槛保持F/E≥45%、预期F≥40%、一致≥95%/97%、KL≤.20、非法/未证0；不得用平均值掩盖某角色/折失败。']
        (output/'latest.md').write_text('\n'.join(lines)+'\n')
        if not watch or (v.OUTPUT/'summary.json').is_file() or (v.OUTPUT/'driver.failed.json').is_file():return
        time.sleep(60)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    main(parser.parse_args().watch)
