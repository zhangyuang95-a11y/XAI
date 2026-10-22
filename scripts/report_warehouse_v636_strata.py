#!/usr/bin/env python3
"""Task/action/context report for the bounded-guard program language."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v,v628_data as data
from backend.training.warehouse_v630_program import Program as Previous
from backend.training.warehouse_v635_program import Program,measure
from scripts.run_warehouse_v625 import write_new


def one(ref):
    r=v.bound(ref);spec=r['spec'];_,old=v.checked_plan();dp=v.bound(v.bound(old['previous_plan'])['previous_plan'])
    a=data.load_parts(dp,spec['fold'],spec['role'])['validation'];sr=v.bound(spec['source_result'])
    before=Previous.from_dict(v.bound(sr['program']),old['actor_role_sha256'][spec['role']])
    after=Program.from_dict(v.bound(r['program']),old['actor_role_sha256'][spec['role']]);tasks={}
    for task in (1,2,3):
        take=np.array([row['task']==task for row in a['rows']],bool);part={k:a[k][take] for k in ('x','m','p','sampled')}
        part['rows']=[row for row,yes in zip(a['rows'],take,strict=True) if yes]
        count=len({row['base_seed'] for row in part['rows']});v.require(count==100,'old task pool must contain100 scenarios')
        tasks[str(task)]=dict(scenarios=count,baseline=measure(before,part),candidate=measure(after,part))
    context={purpose:dict(baseline_action=r['baseline_metrics']['validation'][purpose]['per_nn_action'],candidate_action=r['metrics']['validation'][purpose]['per_nn_action'],
        baseline_context=r['baseline_metrics']['validation'][purpose]['public_strata'],candidate_context=r['metrics']['validation'][purpose]['public_strata']) for purpose in ('feedback','explanation')}
    return dict(job=spec['id'],input=ref,tasks=tasks,whole_action_and_context=context,qualification_changed=False)


def main(folder,output):
    summaryref=v.pointer(folder/'result.json');s=v.bound(summaryref)
    v.require(s['all_six_settled'] and all(v.bound(ref)['status']=='completed' for ref in s['units']),'need six actual programs')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    config=dict(source=source,summary=summaryref,inputs=s['units'],qualification_changed=False,maximum_workers=3)
    write_new(output/'plan.json',config);(output/'plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,ref) for ref in s['units']]):
            r=f.result();rows.append(r);print(r['job'],'fixed task/action/context complete',flush=True)
    for ref in [source,*s['units']]:v.require(v.pointer(ref['path'])==ref,'report source/input changed')
    result=dict(schema='warehouse-v636-fixed-qualification-strata.v1',plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),
        qualification_changed=False,new_fits=0,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    lines=['# v6.36旧开发集固定资格task分层','',
        '| 单元 | task | 原E | 新E | 变化(pp) | 新F | 旧场景数 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['job']):
        for task,t in r['tasks'].items():
            b=t['baseline']['explanation']['coverage'];e=t['candidate']['explanation']['coverage'];f=t['candidate']['feedback']['coverage']
            lines.append(f"| {r['job']} | {task} | {b:.5%} | {e:.5%} | {100*(e-b):+.5f} | {f:.5%} | {t['scenarios']} |")
    lines+=['','固定原资格集合，不按task重判或选模型。task×100均为旧开发切片，绝非最终留出。',
        'JSON包含完整五动作及低电量／冲突等公开状态；上下文可重叠，连续帧不是独立样本。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
