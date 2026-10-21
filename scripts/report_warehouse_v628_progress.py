#!/usr/bin/env python3
"""Observe finite experiment outputs; never fit, collect, retry or choose models.

Operational status and final validation measurements remain distinct. Missing
measurements stay null. Historical v626 values are explicitly historical, not
substituted for new, matched-scene measurements.
"""
from pathlib import Path
import argparse,csv,json,os,sys,time
from datetime import datetime,timezone
from hashlib import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/warehouse_v628/warehouse-v628-v626-capacity-01'
BASE=ROOT/'experiments/warehouse/evidence/warehouse-v626-TAO64-metrics.csv'


def read(path):
    try:return json.loads(path.read_bytes())
    except (FileNotFoundError,json.JSONDecodeError):return None


def historical():
    with BASE.open() as h:rows=list(csv.DictReader(h))
    result={}
    for role in ('robot_1','robot_2','all'):
        values=[r for r in rows if role=='all' or r['role']==role]
        result[role]={}
        for key,threshold in (('feedback_coverage',.45),('explanation_coverage',.45),('expected_feedback_coverage',.40)):
            data=[float(r[key]) for r in values];worst=min(data)
            result[role][key]=dict(mean=sum(data)/len(data),worst=worst,threshold=threshold,
                worst_gap_percentage_points=max(0.,threshold-worst)*100)
        result[role]['feedback_agreement_min']=min(float(r['feedback_agreement']) for r in values)
        result[role]['explanation_agreement_min']=min(float(r['explanation_agreement']) for r in values)
        result[role]['qualified_KL_max']=max(float(r[k]) for r in values for k in ('feedback_KL','explanation_KL'))
    return dict(scope='historical_spent_development_not_new_matched_scene_baseline',values=result,
        source=dict(path=str(BASE),sha256=sha256(BASE.read_bytes()).hexdigest()))


def snapshot():
    plan=read(OUT/'plan.json')
    if not plan:raise RuntimeError('no sealed v628 plan')
    jobs=[]
    for spec in plan['jobs']:
        folder=OUT/'jobs'/spec['job_id'];r=read(folder/'result.json');seal=read(folder/'completed.json')
        valid=bool(r and seal and seal.get('result_sha256')==sha256((folder/'result.json').read_bytes()).hexdigest())
        status=r['status'] if valid else 'failed' if (folder/'failed.json').exists() else 'running' if (folder/'started.json').exists() else 'not_started'
        m=r.get('metrics',{}).get('validation') if valid else None
        jobs.append(dict(**spec,status=status,validation_metrics=m,gate=r.get('gate') if valid else None,
            elapsed_seconds=r.get('elapsed_seconds') if valid else None))
    collection=read(OUT/'collection/result.json');summary=read(OUT/'summary.json')
    stage='complete' if summary else 'capacity_experiments' if any(j['status']!='not_started' for j in jobs) else 'collecting' if (OUT/'collection/started.json').exists() else 'seed_history_audit'
    return dict(schema='warehouse-v628-live-metrics.v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        stage=stage,collected_episodes=len(list((OUT/'collection').glob('dev-*.npz'))),scheduled_episodes=1650,
        completed_jobs=sum(j['status']=='completed' for j in jobs),started_jobs=sum(j['status']!='not_started' for j in jobs),scheduled_jobs=18,
        scientific_pass=summary.get('scientific_pass') if summary else None,
        new_metrics_available=any(j['validation_metrics'] is not None for j in jobs),jobs=jobs,
        new_task_quality=collection.get('outcomes') if collection else None,
        historical_baseline=historical(),
        method_modified=False,fit_calls=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0)


def markdown(s):
    lines=['# Warehouse v6.28 实测进度','',f"更新时间（UTC）：{s['observed_utc']}",
        f"阶段：{s['stage']}；轨迹 {s['collected_episodes']}/1650；实验完成 {s['completed_jobs']}/18。",'',
        '**空项表示尚无实测结果，不表示0。测试通过和启动成功不属于效果指标。**','',
        '| 容量 | 折 | 角色 | 状态 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 准入 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    pct=lambda x:'待测' if x is None else f'{100*x:.2f}%'
    num=lambda x:'待测' if x is None else f'{x:.4f}'
    for j in s['jobs']:
        m=j['validation_metrics'] or {};f=m.get('feedback',{});e=m.get('explanation',{})
        lines.append('| '+' | '.join([str(j['capacity']),str(j['fold']),j['role'],j['status'],pct(f.get('coverage')),
            pct(e.get('coverage')),pct(f.get('expected_actionable_coverage')),
            pct(f.get('agreement'))+'/'+pct(e.get('agreement')),num(f.get('kl'))+'/'+num(e.get('kl')),
            str(j['gate']['passed']) if j['gate'] else '待测'])+' |')
    lines+=['','## 历史起点及差距','',
        '下表仅为v6.26旧开发结果；不能代替v6.28新场景的配对基线。',
        '| 角色 | 最差F覆盖 / 差距 | 最差E覆盖 / 差距 | 最差预期F / 差距 |',
        '|---|---|---|---|']
    for role in ('robot_1','robot_2'):
        r=s['historical_baseline']['values'][role]
        cells=[pct(r[k]['worst'])+f" / {r[k]['worst_gap_percentage_points']:.2f}个百分点" for k in ('feedback_coverage','explanation_coverage','expected_feedback_coverage')]
        lines.append('| '+' | '.join([role,*cells])+' |')
    lines+=['','门槛：每角色每折F/E覆盖45%，预期F40%，一致95%/97%，KL≤0.20，整树非法/未证0。',
        '只有18项完整结束才选择容量。此观察器不会拟合、改变预算或中断低分作业。','']
    return '\n'.join(lines)


def save(s):
    folder=OUT/'progress';folder.mkdir(exist_ok=True)
    for name,text in (('latest.json',json.dumps(s,ensure_ascii=False,indent=2)+'\n'),('latest.md',markdown(s))):
        tmp=folder/(name+'.writing');tmp.write_text(text);os.replace(tmp,folder/name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        s=snapshot();save(s)
        print(json.dumps({k:s[k] for k in ('observed_utc','stage','collected_episodes','completed_jobs','new_metrics_available')},ensure_ascii=False),flush=True)
        if not a.watch or (OUT/'summary.json').exists():return
        driver=read(OUT/'driver-started.json')
        try:
            if not driver:raise ProcessLookupError
            os.kill(driver['pid'],0)
        except ProcessLookupError:return
        time.sleep(30)


if __name__=='__main__':main()
