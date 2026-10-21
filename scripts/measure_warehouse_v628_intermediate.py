#!/usr/bin/env python3
"""Read-only, hash-bound measurements of published search checkpoints.

Only the predeclared 180-episode FIT-INTERNAL score partition is used. These
are search diagnostics, never outer validation, admission or independent
confirmation. No live worker state, model or budget is modified.
"""
from pathlib import Path
import argparse,json,os,sys,time
from datetime import datetime,timezone
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v,v628_data as data
from backend.training.warehouse_v625_checkpoint import ContentStore
from backend.training.warehouse_v628_program import Program,ParentProgram,from_parent,qualify,measure
from backend.training.warehouse_v622_program import gate_metrics


def load_snapshot(folder,spec,plan_sha,role_sha):
    """Only already committed CAS packs; no CheckpointSession or resume call."""
    file=folder/'checkpoint.json'
    raw=file.read_bytes();manifest=json.loads(raw)
    expected=v.digest(dict(plan_sha256=plan_sha,job=spec,solver_limits=v.recipe()['limits'],
        budget_contract='v628-common-max256-reserve.v1'))
    if manifest['identity_sha256']!=expected:raise ValueError('checkpoint job/plan/budget identity mismatch')
    root=Path(str(file)+'.objects')
    if not root.is_dir():raise ValueError('committed CAS store missing')
    store=ContentStore(root);store.load_registry(manifest['object_pack_registry'])
    state=store.get_json(manifest['state'])
    if state.get('algorithm')!='protected_region_subtree_refinement_v1':return None
    ref=state['best']['program'];model=Program.from_dict(store.get_json(ref),role_sha)
    if model.role!=spec['role'] or model.maximum_leaves!=spec['capacity']:raise ValueError('checkpoint role/capacity mismatch')
    return dict(program=model,program_sha256=ref['sha256'],checkpoint_sha256=sha256(raw).hexdigest(),
        checkpoint_sequence=manifest['ledger_sequence'],evaluations=state['evaluations'],solver_counts=state['solver_counts'])


def calculate(model,parts,baseline):
    m=measure(model,parts['validation']);b=measure(baseline,parts['validation'])
    return dict(scope='fit_internal_180_episode_score_not_outer_validation_or_admission',
        metrics=m,same_partition_parent_metrics=b,
        delta_explanation_percentage_points=100*(m['explanation']['coverage']-b['explanation']['coverage']),
        delta_expected_feedback_percentage_points=100*(m['feedback']['expected_actionable_coverage']-b['feedback']['expected_actionable_coverage']),
        inner_diagnostic_gate=gate_metrics(m,phase='development'),
        outer_validation_used=False,fit_calls=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0)


def atomic(path,value):
    path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    path,plan=v.checked_plan();plan_sha=v.file_sha(path);rows={};loaded={};seen={}
    folder=v.OUTPUT/'progress/inner-score';folder.mkdir(exist_ok=True,parents=True)
    while True:
        errors=[]
        if (v.OUTPUT/'collection/completed.json').exists():
            for job in plan['jobs']:
                directory=v.OUTPUT/'jobs'/job['job_id'];checkpoint=directory/'checkpoint.json'
                if not checkpoint.exists() or (directory/'completed.json').exists():continue
                # Commit files change atomically; a partially written current
                # candidate is never a measurement target.
                stamp=checkpoint.stat().st_mtime_ns
                if seen.get(job['job_id'])==stamp:continue
                try:
                    snap=load_snapshot(directory,job,plan_sha,plan['actor_role_sha256'][job['role']])
                    seen[job['job_id']]=stamp
                    if snap is None:continue
                    if job['job_id'] in rows and rows[job['job_id']]['program_sha256']==snap['program_sha256']:
                        rows[job['job_id']].update(checkpoint_sequence=snap['checkpoint_sequence'],evaluations=snap['evaluations'],solver_counts=snap['solver_counts']);continue
                    key=(job['fold'],job['role'])
                    if key not in loaded:
                        parts=data.load_parts(plan,job['fold'],job['role'],inner=True)
                        ref=plan['parents'][f"fold-{job['fold']}-{job['role']}"]
                        parent=ParentProgram.from_dict(v.bound(ref),plan['actor_role_sha256'][job['role']])
                        baseline=from_parent(parent,ref,256);qualify(baseline,parts)
                        loaded[key]=(parts,baseline)
                    parts,baseline=loaded[key]
                    result=dict(job=job,observed_utc=datetime.now(timezone.utc).isoformat(),
                        **{k:value for k,value in snap.items() if k!='program'},**calculate(snap['program'],parts,baseline))
                    rows[job['job_id']]=result
                    atomic(folder/job['job_id']/(snap['program_sha256']+'.json'),result)
                    print(json.dumps(dict(job=job['job_id'],E=result['metrics']['explanation']['coverage'],
                        F=result['metrics']['feedback']['coverage'],delta_E_pp=result['delta_explanation_percentage_points'])),flush=True)
                except Exception as e:errors.append(dict(job=job['job_id'],error_type=type(e).__name__,error=str(e)))
        result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),
            scope='fit_internal_search_diagnostics_not_final_validation',rows=list(rows.values()),errors=errors)
        atomic(folder/'latest.json',result)
        lines=['# v6.28 搜索中间指标（拟合集内部评分集）','',
            '**本表不是外层验证、独立确认或验收结论。最终准入仍使用每项 result.json。**','',
            '| 容量 | 折 | 角色 | 已评估候选 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 同分区原E | E变化(pp) |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows.values():
            j=r['job'];f=r['metrics']['feedback'];e=r['metrics']['explanation'];base=r['same_partition_parent_metrics']['explanation']
            pct=lambda x:f'{100*x:.2f}%';num=lambda x:'无合格帧' if x is None else f'{x:.4f}'
            lines.append('| '+' | '.join([str(j['capacity']),str(j['fold']),j['role'],str(r['evaluations']),pct(f['coverage']),pct(e['coverage']),
                pct(f['expected_actionable_coverage']),pct(f['agreement'])+'/'+pct(e['agreement']),num(f['kl'])+'/'+num(e['kl']),
                pct(base['coverage']),f"{r['delta_explanation_percentage_points']:+.2f}"])+' |')
        if not rows:lines+=['','尚无已提交搜索检查点；不把未测记为0。']
        (folder/'latest.md').write_text('\n'.join(lines)+'\n')
        if not a.watch or (v.OUTPUT/'summary.json').exists():return
        driver=json.loads((v.OUTPUT/'driver-started.json').read_bytes())
        try:os.kill(driver['pid'],0)
        except ProcessLookupError:return
        time.sleep(120)


if __name__=='__main__':main()
