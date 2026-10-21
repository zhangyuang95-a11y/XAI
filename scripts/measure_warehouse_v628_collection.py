#!/usr/bin/env python3
"""Report completed development episodes from immutable public traces only.

All metrics are descriptive interim data. Argmax fit episodes and sampled
episodes remain separate. Public counters are rechecked against the producer's
complete outcome receipt when that receipt becomes available.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v


def episode(case,events,frames,actor_sha):
    if len(events)!=frames or not 1<=frames<=120:raise ValueError('incomplete episode')
    deliveries=collisions=0;shutdown=False;previous=None
    for frame,r in enumerate(events):
        before,after=r['pre_state_public'],r['post_state_public']
        if (r['case']!=case or r['actor_sha256']!=actor_sha or r['frame']!=frame
                or before['frame']!=frame or after['frame']!=frame+1
                or previous is not None and previous!=before):raise ValueError('public trace identity or chain mismatch')
        removed={t['task_id'] for t in before['tasks']}-{t['task_id'] for t in after['tasks']}
        added={t['task_id'] for t in after['tasks']}-{t['task_id'] for t in before['tasks']}
        if len(removed)!=len(added):raise ValueError('delivery replacement mismatch')
        for task in before['tasks']:
            if task['task_id'] in removed:
                carrier=next((a['agent_id'] for a in before['agents'] if a['carrying_task_id']==task['task_id']),None)
                if carrier is None or next(a for a in after['agents'] if a['agent_id']==carrier)['position']!=task['delivery_position']:
                    raise ValueError('removed task was not physically delivered')
        terminal=any(a['active'] and not b['active'] for a,b in zip(before['agents'],after['agents'],strict=True))
        if terminal and frame!=frames-1:raise ValueError('continued after shutdown')
        deliveries+=len(removed);collisions+=bool(after['last_robot_collision_event']);shutdown|=terminal;previous=after
    if frames<120 and not shutdown:raise ValueError('partial episode cannot be counted as zero shutdown')
    return dict(case=case,deliveries=deliveries,robot_collisions=collisions,shutdown=shutdown,physical_frames=frames)


def aggregate(rows):
    out={}
    for mode in ('argmax','sampled'):
        sample=[r for r in rows if r['case']['sampling_mode']==mode]
        if not sample:out[mode]=None;continue
        out[mode]=dict(episodes=len(sample),deliveries=float(np.mean([r['deliveries'] for r in sample])),
            collisions=float(np.mean([r['robot_collisions'] for r in sample])),shutdown_episodes=sum(r['shutdown'] for r in sample),
            shutdown_rate=float(np.mean([r['shutdown'] for r in sample])),
            per_task={str(t):dict(episodes=sum(r['case']['task']==t for r in sample)) for t in (1,2,3)})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    _,plan=v.checked_plan();rows={};folder=v.OUTPUT/'progress/collection-metrics';folder.mkdir(exist_ok=True,parents=True)
    while True:
        for case in plan['cases']:
            if case['case_id'] in rows:continue
            base=v.OUTPUT/'collection'/case['case_id'];array=base.with_suffix('.npz');trace=base.with_suffix('.jsonl.gz')
            if not array.exists() or not trace.exists() or array.stat().st_mode&0o222 or trace.stat().st_mode&0o222:continue
            with np.load(array,allow_pickle=False) as z:frames=len(z['x'])
            with gzip.open(trace,'rt') as h:events=[json.loads(line) for line in h]
            rows[case['case_id']]=episode(case,events,frames,plan['actor']['sha256'])
        receipt=v.OUTPUT/'collection/result.json'
        checked=False
        if receipt.exists():
            expected=json.loads(receipt.read_bytes())['outcomes']
            if len(rows)==len(expected):
                if rows!={r['case']['case_id']:r for r in expected}:raise ValueError('public counters differ from producer outcomes')
                checked=True
        quality=aggregate(list(rows.values()));result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),
            scope='completed_development_episodes_descriptive_not_heldout_or_capacity_improvement',completed_episodes=len(rows),
            quality=quality,public_counters_match_complete_producer_receipt=checked,
            source_actor=plan['actor'],actor_updates=0,PPO_joint_steps=0,fit_calls=0,environment_resets=0)
        path=folder/'latest.json';temp=folder/'latest.json.writing';temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
        lines=['# v6.28 新轨迹任务指标','',f"已完成episode：{len(rows)}/1650。",'',
            '仅已完成episode；argmax是拟合数据，sampled是开发数据，均非最终留出。NN冻结，不能把不同场景的均值变化写成模型进步。','',
            '| 动作方式 | 完整局数 | 平均配送 | 平均碰撞 | 停机局数 | 停机率 |','|---|---|---|---|---|---|']
        for mode,q in quality.items():
            if q:lines.append(f"| {mode} | {q['episodes']} | {q['deliveries']:.2f} | {q['collisions']:.2f} | {q['shutdown_episodes']} | {100*q['shutdown_rate']:.2f}% |")
        (folder/'latest.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(completed_episodes=len(rows),quality=quality)),flush=True)
        if not args.watch or checked:return
        driver=json.loads((v.OUTPUT/'driver-started.json').read_bytes())
        try:os.kill(driver['pid'],0)
        except ProcessLookupError:return
        time.sleep(30)


if __name__=='__main__':main()
