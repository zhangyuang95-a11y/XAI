#!/usr/bin/env python3
"""Frozen initial-program metrics on predeclared, completed inner partitions.

No fitting and no outer validation. Input hashes are retained and checked
against the complete collection receipt when it becomes available. These
reference values never grant admission or change the search.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v
from backend.training.warehouse_v628_program import ParentProgram,from_parent,qualify,measure
from scripts.run_warehouse_v625 import write_new


def run():
    path,plan=v.checked_plan();folder=v.OUTPUT/'progress/initial-programs';folder.mkdir(exist_ok=True,parents=True)
    byid={c['case_id']:c for c in plan['cases']};reports=[]
    for f in plan['folds']:
        refs={};ready=True
        for key in sum(f['inner'].values(),[]):
            p=v.OUTPUT/'collection'/(key+'.npz');trace=v.OUTPUT/'collection'/(key+'.jsonl.gz')
            if not p.exists() or not trace.exists() or p.stat().st_mode&0o222 or trace.stat().st_mode&0o222:ready=False;break
            refs[key]=v.pointer(p)
        if not ready:continue
        for index,role in enumerate(v.old.ROLES):
            target=folder/f"fold-{f['fold']}-{role}.json"
            if target.exists():reports.append(json.loads(target.read_bytes()));continue
            parts={}
            for split,ids in f['inner'].items():
                arrays={k:[] for k in ('x','m','p','sampled')};rows=[]
                for key in ids:
                    with np.load(refs[key]['path'],allow_pickle=False) as z:
                        for k in arrays:arrays[k].append(z[k][:,index].copy())
                        rows.extend(dict(episode=key,base_seed=str(byid[key]['base']),task=byid[key]['task']) for _ in range(len(z['x'])))
                parts[split]={k:np.concatenate(a) for k,a in arrays.items()};parts[split]['rows']=rows
            parent=plan['parents'][f"fold-{f['fold']}-{role}"]
            p=from_parent(ParentProgram.from_dict(v.bound(parent),plan['actor_role_sha256'][role]),parent,256)
            qualify(p,parts);m=measure(p,parts['validation'])
            result=dict(fold=f['fold'],role=role,scope='initial_frozen_reference_fit_internal_score_not_capacity_result_or_admission',
                original_parent=parent,plan=v.pointer(path),input_arrays=refs,metrics=m,
                all_three_capacities_share_this_initial_output=True,outer_validation_used=False,
                new_eligibility_recomputed=True,fit_calls=0,environment_resets=0,actor_updates=0,PPO_joint_steps=0)
            write_new(target,result);reports.append(result)
            print(json.dumps(dict(fold=f['fold'],role=role,F=m['feedback']['coverage'],E=m['explanation']['coverage'],expectedF=m['feedback']['expected_actionable_coverage'])),flush=True)
    receipt=v.OUTPUT/'collection/result.json'
    if receipt.exists():
        complete=json.loads(receipt.read_bytes())
        for report in reports:
            for key,ref in report['input_arrays'].items():v.require(ref==complete['cases'][key]['arrays'],'early reference targets differ from complete collection')
    lines=['# v6.28 新数据上的优化前参考指标','',
        '**冻结v6.26父程序重新判定新数据资格。拟合集内部180场景评分，不是容量优化结果或最终验收。三种容量的初始输出相同。**','',
        '| 折 | 角色 | F覆盖 | E覆盖 | 预期F | F/E一致 | F/E KL | 非法/未证 |',
        '|---|---|---|---|---|---|---|---|']
    for r in reports:
        f,e=r['metrics']['feedback'],r['metrics']['explanation'];pct=lambda a:f'{100*a:.2f}%';num=lambda a:'无合格帧' if a is None else f'{a:.4f}'
        lines.append('| '+' | '.join([str(r['fold']),r['role'],pct(f['coverage']),pct(e['coverage']),pct(f['expected_actionable_coverage']),
            pct(f['agreement'])+'/'+pct(e['agreement']),num(f['kl'])+'/'+num(e['kl']),str(f['whole_tree_raw_illegal_count'])+'/'+str(f['whole_tree_path_uncertified_count'])])+' |')
    (folder/'latest.md').write_text('\n'.join(lines)+'\n')
    return reports


if __name__=='__main__':run()
