#!/usr/bin/env python3
"""Fixed-qualification task/action/public-context report on spent scenes.

No refit, requalification, new candidate choice, episode or NN forward pass.
Scene counts, not consecutive frame counts, describe evaluation exposure.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v, v628_data as data
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import measure
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,plan=v.checked_plan();dp=v.bound(v.bound(plan['previous_plan'])['previous_plan'])
    a=data.load_parts(dp,spec['fold'],spec['role'])['validation']
    sr=v.bound(spec['source_result']);baseline=Program.from_dict(v.bound(sr['program']),plan['actor_role_sha256'][spec['role']])
    p=Program.from_dict(v.bound(r['program']),baseline.actor_role_sha256)
    tasks={}
    for task in (1,2,3):
        take=np.array([row['task']==task for row in a['rows']],bool)
        b={k:np.asarray(a[k])[take] for k in ('x','m','p','sampled')}
        b['rows']=[row for row,keep in zip(a['rows'],take,strict=True) if keep]
        scene_count=len({row['base_seed'] for row in b['rows']})
        v.require(scene_count==100,'each validation task must contain 100 original scenes')
        tasks[str(task)]=dict(scene_count=scene_count,physical_frames=len(b['x']),
                             baseline=measure(baseline,b),candidate=measure(p,b))
    # Reuse already computed aggregate by-action/public-context tables. Their
    # exact numeric provenance was checked by each full-program outer replay.
    whole={purpose:dict(baseline_action=r['baseline_metrics']['validation'][purpose]['per_nn_action'],
                       candidate_action=r['metrics']['validation'][purpose]['per_nn_action'],
                       baseline_context=r['baseline_metrics']['validation'][purpose]['public_strata'],
                       candidate_context=r['metrics']['validation'][purpose]['public_strata'])
           for purpose in ('feedback','explanation')}
    return dict(job=spec['id'],input=ref,program=r['program'],source_program=sr['program'],tasks=tasks,
                whole_action_and_context=whole,qualification_changed=False)


def run(folder,output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref);v.require(summary['all_twelve_settled'],'complete all units first')
    refs=summary['units'];v.require(all(v.bound(ref)['status']=='completed' for ref in refs),'strata require twelve actual programs')
    output.mkdir(parents=True,exist_ok=False)
    config=dict(schema='warehouse-v632-fixed-qualification-strata.v1',source=v.pointer(__file__),summary=summaryref,
                inputs=refs,maximum_workers=3,qualification_changed=False,NN_updates=0,new_environment_resets=0)
    write_new(output/'plan.json',config);(output/'plan.json').chmod(0o444)
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(one,ref) for ref in refs]
        for f in as_completed(fs):
            row=f.result();rows.append(row);print(row['job'],'task/action/context report completed',flush=True)
    rows.sort(key=lambda r:r['job'])
    for ref in refs:v.require(v.pointer(ref['path'])==ref,'fixed qualification input changed')
    v.require(v.pointer(__file__)==config['source'],'strata source changed')
    result=dict(schema=config['schema'],plan=v.pointer(output/'plan.json'),units=rows,
        diagnostic_only_not_additional_admission_gate=True,fit_calls=0,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    lines=['# v6.32固定资格：三特征方法按task分层','',
           '| 折/角色 | task | 原解释覆盖 | 新解释覆盖 | 变化(pp) | 新反馈覆盖 | 场景数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if not row['job'].startswith('oblique'):continue
        for task,t in row['tasks'].items():
            b=t['baseline']['explanation']['coverage'];e=t['candidate']['explanation']['coverage'];f=t['candidate']['feedback']['coverage']
            lines.append(f"| {row['job'][8:]} | {task} | {b:.5%} | {e:.5%} | {100*(e-b):+.5f} | {f:.5%} | {t['scene_count']} |")
    lines+=['','固定原资格集合，未按task重新判定或选择模型。每task100场景是旧开发集分层，绝非最终留出。',
            '完整JSON含全部12项、各task五动作与低电量／冲突等公开状态分层；连续帧不当作独立场景。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve())
