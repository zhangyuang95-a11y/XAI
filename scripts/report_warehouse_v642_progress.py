#!/usr/bin/env python3
"""Read settled receipts only; pending units never count as successful zeros."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v
from backend.training.warehouse_v642_certificate import structure_info
from scripts.run_warehouse_v625 import write_new


def report(folder,output):
    planref=v.pointer(folder/'plan.json');plan=v.bound(planref);rows=[];refs=[]
    for spec in plan['jobs']:
        ref=folder/spec['id']/'result.json';parent=v.bound(spec['source_result']);old=parent['metrics']['validation']['explanation']['coverage']
        row=dict(job=spec['id'],depth_limit=spec['maximum_depth'],fold=spec['fold'],role=spec['role'],source_E=old,status='pending')
        if ref.exists():
            pointer=v.pointer(ref);result=v.bound(pointer);refs.append(pointer);row.update(status=result['status'],result=pointer)
            v.require(result['spec']==spec and result['plan']==planref,'settled receipt not bound to registered job')
            if result['status']=='completed':
                sr=v.bound(result['search_report']);c=sr['candidates'][sr['selected']];n=sr['selection_frames'];a=v.bound(result['program'])
                _,size=structure_info(a['structure'],256,spec['maximum_depth']);f,e=result['metrics']['validation']['feedback'],result['metrics']['validation']['explanation']
                row.update(E=e['coverage'],F=f['coverage'],expected_F=f['expected_actionable_coverage'],F_agreement=f['agreement'],E_agreement=e['agreement'],F_KL=f['kl'],E_KL=e['kl'],
                    raw_illegal=e['whole_tree_raw_illegal_count'],unproved=e['whole_tree_path_uncertified_count'],delta_E_pp=100*(e['coverage']-old),
                    reliable={k:value/n for k,value in c['reliable_counts'].items()},numerical_passed=result['numerical_passed'],reliable_passed=result['reliable_fit_only_passed'],
                    development_passed=result['development_passed'],actual_depth=size['depth'],leaves=size['leaf_count'],counts=result['counts'],elapsed_seconds=result['elapsed_seconds'])
            else:row.update(error=result.get('error'),counts=result['counts'])
        rows.append(row)
    arms={}
    for depth in (9,12):
        selected=[r for r in rows if r['depth_limit']==depth];completed=[r for r in selected if r['status']=='completed']
        arms[str(depth)]=dict(settled=sum(r['status']!='pending' for r in selected),completed=len(completed),
            numerical_pass_count=sum(r['numerical_passed'] for r in completed),development_pass_count=sum(r['development_passed'] for r in completed),
            all_six_development_passed=len(completed)==6 and all(r['development_passed'] for r in completed),
            mean_E=None if len(completed)!=6 else sum(r['E'] for r in completed)/6,
            minimum_E=None if len(completed)!=6 else min(r['E'] for r in completed))
    evidence=dict(schema='warehouse-v642-settled-receipt-progress.v1',observed_utc=datetime.now(timezone.utc).isoformat(),plan=planref,
        source=v.pointer(__file__),units=rows,arms=arms,partial_means_suppressed=True,NN_updates=0,new_environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
    lines=['# 深9／深12容量对照：已结算结果','','仅读取完成或失败收据；待定不计作0分或通过。完整六项完成前不报告该臂均值。',
        '同一冻结NN，配送／碰撞／停机未重新评估，本轮不构成NN任务能力提升证据。','',
        '| 深度上限 | 折/角色 | 状态 | 来源E | 新E / 变化pp | F / 预期F | 内部可靠E / F / 预期F | F/E一致 | F/E KL | 非法/未证 | 实深/叶数 | 数值/可靠 |',
        '|---|---|---|---:|---|---|---|---|---|---|---|---|']
    for r in rows:
        prefix=f"| {r['depth_limit']} | {r['fold']}/{r['role']} | {r['status']} | {r['source_E']:.4%} |"
        if r['status']!='completed':lines.append(prefix+' — | — | — | — | — | — | — | — |');continue
        rel=r['reliable'];lines.append(prefix+f" {r['E']:.4%} / {r['delta_E_pp']:+.4f} | {r['F']:.4%} / {r['expected_F']:.4%} | {rel['E']:.4%} / {rel['F']:.4%} / {rel['expected_F']:.4%} | {r['F_agreement']:.4%}/{r['E_agreement']:.4%} | {r['F_KL']:.6f}/{r['E_KL']:.6f} | {r['raw_illegal']}/{r['unproved']} | {r['actual_depth']}/{r['leaves']} | {r['numerical_passed']}/{r['reliable_passed']} |")
    lines+=['','门槛：覆盖F/E各45%，预期F40%，一致95%/97%，KL≤.20，非法/未证0，内部可靠预检亦须通过。','']
    output.mkdir(parents=True,exist_ok=False);write_new(output/'metrics.json',evidence);(output/'metrics.json').chmod(0o444)
    (output/'metrics.md').write_text('\n'.join(lines));(output/'metrics.md').chmod(0o444)
    print(json.dumps(dict(output=str(output),arms=arms,settled=sum(r['status']!='pending' for r in rows))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();report(args.run.resolve(),args.output.resolve())
