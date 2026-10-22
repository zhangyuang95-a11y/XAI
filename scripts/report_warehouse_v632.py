#!/usr/bin/env python3
"""Finish frozen fit tables after a baseline *metadata* comparison failure.

Never refits, retries, searches, chooses another candidate, or modifies the
original run. Both actual purpose dictionaries must exactly match baseline.
The original gate summary is independently reconstructed as well.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json, os, sys, traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.warehouse import v630_protocol as v, v628_data as data
from backend.training import warehouse_v632_search as search
from backend.training import warehouse_v630_training_program as fixed
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v628_program import measure
from scripts.audit_warehouse_v630 import audited_program
from scripts.audit_warehouse_v632 import ledger
from scripts.run_warehouse_v632 import strict
from scripts.run_warehouse_v625 import write_new


def evaluate(output, spec):
    folder=Path(output)/spec['id'];folder.mkdir(exist_ok=False)
    planref=v.pointer(Path(output)/'plan.json');plan=v.bound(planref)
    recovery=v.bound(v.pointer(Path(output)/'report-plan.json'));binding=recovery['inputs'][spec['id']]
    r=v.bound(binding['result']);report=v.bound(binding['fit_report']);sr=v.bound(spec['source_result'])
    _,p=v.checked_plan();dp=v.bound(v.bound(p['previous_plan'])['previous_plan'])
    source=Program.from_dict(v.bound(sr['program']),p['actor_role_sha256'][spec['role']])
    base=Program.from_dict(report['collapsed_base'],source.actor_role_sha256)
    chosen=report['candidates'][report['selected']]
    partitions={int(g):report['local'][g][i]['structure'] for g,i in chosen['allocation']['selected'].items()}
    program,_=search.old.assemble(base,partitions)
    parts=data.load_parts(dp,spec['fold'],spec['role']);baseline={s:measure(source,a) for s,a in parts.items()}
    for split in parts:
        v.require(set(sr['metrics'][split])=={'feedback','explanation','gate'},'unexpected baseline metadata shape')
        for purpose in ('feedback','explanation'):
            v.require(baseline[split][purpose]==sr['metrics'][split][purpose],'actual baseline metric dictionary differs')
        v.require(fixed.gate(baseline[split])==sr['metrics'][split]['gate'],'baseline gate summary differs')
    search.old.qualify_current(program,parts);metrics={s:measure(program,a) for s,a in parts.items()}
    independent=audited_program(program,parts,metrics)
    for split,a in parts.items():
        original=source.distributions(a['x'],a['m'])
        v.require(all(np.array_equal(x,y) for x,y in zip(original[1:],independent[split][1:])), 'surrogate q changed')
    calls=ledger(Path(binding['ledger']['path']),r['counts'])
    v.require(r['status']=='failed' and r['error']=='registered fixed-refit baseline differs',
              'this reporter may only recover the exact registered metadata failure')
    write_new(folder/'program.json',program.to_dict());(folder/'program.json').chmod(0o444)
    # Preserve a byte-identical ledger for the generic independent replay.
    (folder/'ledger.jsonl').write_bytes(Path(binding['ledger']['path']).read_bytes());(folder/'ledger.jsonl').chmod(0o444)
    result=dict(status='completed',spec=spec,plan=planref,report_plan=v.pointer(Path(output)/'report-plan.json'),
        original_failure=binding['result'],program=v.pointer(folder/'program.json'),fit_report=binding['fit_report'],
        baseline_metrics=baseline,metrics=metrics,counts=r['counts'],strict_numeric_passed=strict(metrics['validation']),
        reliable_fit_only_passed=report['reliable_selection_passed'],
        development_passed=strict(metrics['validation']) and report['reliable_selection_passed'],
        independent_program_frames=sum(len(a['x']) for a in parts.values()),independent_leaf_certificates=len(program.leaves),
        distributions_preserved=True,additional_optimizer_calls=0,new_fits=0,new_candidate_choices=0,
        NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'result.json',result);(folder/'result.json').chmod(0o444)
    return v.pointer(folder/'result.json')


def main(run,output):
    summaryref=v.pointer(run/'result.json');summary=v.bound(summaryref);planref=v.pointer(run/'plan.json');plan=v.bound(planref)
    v.require(summary['all_twelve_settled'] and len(summary['units'])==12,'wait for all twelve original jobs')
    inputs={}
    for ref in summary['units']:
        r=v.bound(ref);folder=Path(ref['path']).parent
        v.require(r['status']=='failed' and r['error']=='registered fixed-refit baseline differs','unexpected scientific or execution failure')
        inputs[r['spec']['id']]=dict(result=ref,fit_report=v.pointer(folder/'fit-report.json'),ledger=v.pointer(folder/'ledger.jsonl'))
    output.mkdir(parents=True,exist_ok=False)
    # Byte-identical scientific plan, stored under the completion report's own
    # directory. The separate report plan binds this clerical correction.
    (output/'plan.json').write_bytes(Path(planref['path']).read_bytes());(output/'plan.json').chmod(0o444)
    sources=dict(plan['source_files'])
    for p in (Path(__file__),ROOT/'scripts/audit_warehouse_v632.py'):sources[str(p)]=v.file_sha(p)
    reportplan=dict(schema='warehouse-v632-fixed-table-report-completion.v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        original_plan=planref,original_summary=summaryref,inputs=inputs,source_files=sources,
        change='compare exact feedback/explanation dictionaries and independently verify extra gate field',
        fitting_or_selection_changed=False,new_optimizer_calls=0,NN_updates=0,new_environment_resets=0,pilot_opened=False)
    write_new(output/'report-plan.json',reportplan);(output/'report-plan.json').chmod(0o444)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(evaluate,str(output),spec) for spec in plan['jobs']]
        for f in as_completed(fs):
            ref=f.result();results.append(ref);r=v.bound(ref);m=r['metrics']['validation']
            print(json.dumps(dict(job=r['spec']['id'],E=m['explanation']['coverage'],F=m['feedback']['coverage'],
                numeric=r['strict_numeric_passed'],reliable=r['reliable_fit_only_passed'])),flush=True)
    for name,sha in sources.items():v.require(v.file_sha(Path(name) if Path(name).is_absolute() else ROOT/name)==sha,'registered source changed')
    for binding in inputs.values():
        for ref in binding.values():v.require(v.pointer(ref['path'])==ref,'original fixed fit result changed')
    rows=sorted((v.bound(r) for r in results),key=lambda r:r['spec']['id'])
    methods={arm:dict(completed=sum(r['status']=='completed' for r in rows if r['spec']['arm']==arm),
        numeric_passes=sum(r['strict_numeric_passed'] for r in rows if r['spec']['arm']==arm),
        development_passes=sum(r['development_passed'] for r in rows if r['spec']['arm']==arm)) for arm in ('axis','oblique')}
    selected=next((arm for arm in ('axis','oblique') if methods[arm]['development_passes']==6),None)
    result=dict(schema=plan['schema'],plan=v.pointer(output/'plan.json'),report_plan=v.pointer(output/'report-plan.json'),
        original_failed_summary=summaryref,units=sorted(results,key=lambda r:r['path']),methods=methods,
        selected_method=selected,all_twelve_settled=True,original_failures_preserved=True,
        new_fits=0,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    lines=['# v6.32完整12项开发对照','',
        '| 方法 | 折/角色 | 反馈覆盖 | 解释覆盖 | 预期反馈 | F/E一致 | F/E KL | 非法/未证 | 数值/可靠 |',
        '|---|---|---:|---:|---:|---|---|---|---|']
    for r in rows:
        f,e=r['metrics']['validation']['feedback'],r['metrics']['validation']['explanation']
        lines.append(f"| {r['spec']['arm']} | {r['spec']['fold']}/{r['spec']['role']} | {f['coverage']:.5%} | {e['coverage']:.5%} | {f['expected_actionable_coverage']:.5%} | {f['agreement']:.5%}/{e['agreement']:.5%} | {f['kl']:.6f}/{e['kl']:.6f} | {e['whole_tree_raw_illegal_count']}/{e['whole_tree_path_uncertified_count']} | {r['strict_numeric_passed']}/{r['reliable_fit_only_passed']} |")
    lines+=['',f'选中方法：{selected}。保留原12项报告层失败，复用已固定的拟合与选择表，不新增拟合或候选选择。',
        '同一冻结NN、同一旧开发场景、同一完整程序输出；不是新确认、PPO、留出或真人效果。pilot关闭。']
    (output/'results.md').write_text('\n'.join(lines)+'\n');(output/'results.md').chmod(0o444)
    print(json.dumps(v.pointer(output/'result.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();main(a.run.resolve(),a.output.resolve())
