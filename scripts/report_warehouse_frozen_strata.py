#!/usr/bin/env python3
"""Shared reporter for completed frozen-NN warehouse development matrices.

Reuses the immutable numerical kernel; trained/new-target NN matrices must
provide their own current target provenance instead of using this reporter.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v
from scripts.diagnose_warehouse_v632_strata import one
from scripts.run_warehouse_v625 import write_new


def run(folder,output,display_arm):
    summaryref=v.pointer(folder/'result.json');summary=v.bound(summaryref)
    refs=summary['units'];v.require((summary.get('all_twelve_settled') or summary.get('all_six_settled')) and
        len(refs) in (6,12) and all(v.bound(ref)['status']=='completed' for ref in refs),'need a complete actual frozen-NN matrix')
    specs={v.bound(ref)['spec']['id']:v.bound(ref)['spec'] for ref in refs}
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='warehouse-frozen-NN-fixed-qualification-strata.v1',source=v.pointer(__file__),
        numerical_kernel=v.pointer(ROOT/'scripts/diagnose_warehouse_v632_strata.py'),summary=summaryref,
        inputs=refs,display_arm=display_arm,maximum_workers=3,qualification_changed=False,NN_updates=0,new_environment_resets=0)
    write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444);rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,ref) for ref in refs]):
            r=f.result();rows.append(r);print(r['job'],'fixed task/action/context report complete',flush=True)
    for ref in [*refs,plan['source'],plan['numerical_kernel']]:v.require(v.pointer(ref['path'])==ref,'stratification source/input changed')
    result=dict(schema=plan['schema'],plan=v.pointer(output/'plan.json'),units=sorted(rows,key=lambda r:r['job']),
        qualification_changed=False,new_fits=0,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    lines=['# 固定资格的旧开发集task分层','',
        '| 单元 | task | 来源E覆盖 | 当前E覆盖 | 变化(pp) | 当前F覆盖 | 旧场景数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['job']):
        if display_arm is not None and specs[r['job']].get('arm')!=display_arm:continue
        for task,t in r['tasks'].items():
            b=t['baseline']['explanation']['coverage'];e=t['candidate']['explanation']['coverage'];f=t['candidate']['feedback']['coverage']
            lines.append(f"| {r['job']} | {task} | {b:.5%} | {e:.5%} | {100*(e-b):+.5f} | {f:.5%} | {t['scene_count']} |")
    lines+=['','原资格固定，不按task重判或选模型；全部任务×100均为旧开发切片，不是正式留出。',
        '完整JSON包含所有单元、五动作和低电量／冲突等公开状态；上下文可重叠，连续帧不当作独立场景。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--display-arm')
    a=p.parse_args();run(a.run.resolve(),a.output.resolve(),a.display_arm)
