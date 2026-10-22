#!/usr/bin/env python3
"""Six-unit v633 task/action/context report using fixed qualified programs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v
from scripts.diagnose_warehouse_v632_strata import one
from scripts.run_warehouse_v625 import write_new


def run(folder,output):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    v.require(summary['all_six_settled'] and all(v.bound(r)['status']=='completed' for r in summary['units']),
              'need six actual completed programs')
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='warehouse-v633-fixed-task-action-context.v1',source=v.pointer(__file__),
        numerical_kernel=v.pointer(ROOT/'scripts/diagnose_warehouse_v632_strata.py'),summary=summaryref,
        inputs=summary['units'],qualification_changed=False,NN_updates=0,new_environment_resets=0)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one,ref) for ref in summary['units']]):
            r=future.result();rows.append(r);print(r['job'],'fixed qualification task/action/context report complete',flush=True)
    for ref in [*summary['units'],plan['source'],plan['numerical_kernel']]:v.require(v.pointer(ref['path'])==ref,'stratification input/source changed')
    out=dict(schema=plan['schema'],plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),
        qualification_changed=False,new_fits=0,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',out);(output/'result.json').chmod(0o444)
    lines=['# v6.33固定资格，旧开发集task分层','',
        '| 折/角色 | task | v6.32解释覆盖 | v6.33解释覆盖 | 变化(pp) | v6.33反馈覆盖 | 旧场景数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['job']):
        for task,t in r['tasks'].items():
            b=t['baseline']['explanation']['coverage'];e=t['candidate']['explanation']['coverage'];f=t['candidate']['feedback']['coverage']
            lines.append(f"| {r['job']} | {task} | {b:.5%} | {e:.5%} | {100*(e-b):+.5f} | {f:.5%} | {t['scene_count']} |")
    lines+=['','原资格集合固定，未按task重新判定或选模型。task×100均为旧开发切片，不是最终留出。',
            'JSON含五动作、低电量、冲突等公开状态；分层可重叠，连续帧不是独立样本。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve())
