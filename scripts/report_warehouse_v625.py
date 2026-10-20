#!/usr/bin/env python3
"""Render independently audited Stage A values; never selects a model."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v625_protocol as v


def report(folder=v.OUTPUT):
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_bytes())
    audit=json.loads((folder/'independent-audit.json').read_bytes())
    v.require(audit.get('passed') is True and audit.get('summary_sha256')==v.file_sha(folder/'summary.json'),
              'report requires independently audited complete Stage A')
    rows=[]
    for job in v.jobs():
        result=v.bound(summary['jobs'][job['job_id']])
        values=result.get('metrics',{}).get('validation',{})
        f,e=values.get('feedback',{}),values.get('explanation',{})
        rows.append(dict(role=job['role'],fold=job['fold'],arm=job['arm'],status=result['status'],
            passed=result['gate']['passed'],feedback_coverage=f.get('coverage'),
            explanation_coverage=e.get('coverage'),expected_feedback_coverage=f.get('expected_actionable_coverage'),
            actual_sample_usable_coverage=f.get('actionable_coverage'),feedback_agreement=f.get('agreement'),
            explanation_agreement=e.get('agreement'),feedback_KL=f.get('kl'),explanation_KL=e.get('kl'),
            raw_illegal_count=f.get('whole_tree_raw_illegal_count'),
            uncertified_count=f.get('whole_tree_path_uncertified_count'),
            search_calls=result.get('search_calls'),final_calls=result.get('final_calls'),
            Actor_updates=0,actual_program_gradient=0))
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_path=folder/'audited-metrics.csv'
    with csv_path.open('x',newline='') as h:h.write(output.getvalue())
    def percent(value):return '—' if value is None else f'{100*value:.2f}%'
    def number(value):return '—' if value is None else f'{value:.4f}'
    lines=['# Warehouse v6.25 Stage A 结果','',
        f"候选：`{v.CANDIDATE}`。P1 六单元通过：**{summary['arms']['P1']['all_six_pass']}**。",'',
        'P0 是常量叶程序；P1 是稀疏 logistic 叶程序。两者拟合相同冻结 v6.20 NN；不是 raw/reg 双 NN 训练组。', '',
        '| 角色 | 折 | 程序 | 反馈叶覆盖 | 解释叶覆盖 | 预期反馈覆盖 | 实际采样可用 | 反馈一致 | 解释一致 | F/E KL | 通过 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        lines.append('| '+' | '.join([r['role'],str(r['fold']),r['arm'],percent(r['feedback_coverage']),
            percent(r['explanation_coverage']),percent(r['expected_feedback_coverage']),
            percent(r['actual_sample_usable_coverage']),percent(r['feedback_agreement']),
            percent(r['explanation_agreement']),number(r['feedback_KL'])+'/'+number(r['explanation_KL']),
            '是' if r['passed'] else '否'])+' |')
    lines.extend(['','门槛：两种叶覆盖各≥45%，预期反馈≥40%，反馈/解释一致≥95%/97%，KL各≤0.20，非法及未证路径均为0。准入逐角色/折检查；不池化失败。','',
        '叶覆盖是全部物理帧中落入独立资格合格叶的比例；预期反馈按冻结 NN 动作概率计算；实际采样可用覆盖还要求记录中的采样动作匹配。三者都不是已接受梯度比例。', '',
        '本阶段没有环境重置、配送质量重测、Actor 更新或实际程序训练梯度。没有新开发确认、PPO、独立封存审计、留出测试或 pilot 授权。','',
        '以上为已花费开发数据上的描述性比较，不作统计显著改善或真人协作收益声明。','',
        f"独立审计：`{v.file_sha(folder/'independent-audit.json')}`。源码：`{summary['source_sha256']}`。",''])
    md_path=folder/'audited-results.md'
    with md_path.open('x') as h:h.write('\n'.join(lines))
    return dict(csv=v.pointer(csv_path),markdown=v.pointer(md_path),rows=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,default=v.OUTPUT)
    print(json.dumps(report(parser.parse_args().folder),sort_keys=True))
