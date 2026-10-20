#!/usr/bin/env python3
"""Seal a fully audited, scientifically failed v626 candidate without retries.

This is a reporting/closure tool. It never changes a scientific recipe, fits a
program, collects a trajectory, trains an Actor or changes pilot authority.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.warehouse import v626_protocol as v
from scripts.run_warehouse_v625 import write_new
from scripts.run_warehouse_v626 import checked_plan
from scripts.run_warehouse_v626_search import checked_stage


def read(path):
    return json.loads(Path(path).read_bytes())


def checked_results(path, plan, expected):
    completed = read(path.parent / 'execution/completed.json')
    v.require(completed.get('all_owned_workers_waited') is True,
              'supervisor has not waited for all owned workers')
    results = []
    for spec in plan['jobs']:
        folder = path.parent / 'jobs' / spec['job_id']
        result = read(folder / 'result.json')
        v.require(read(folder / 'completed.json') ==
                  dict(result_sha256=v.file_sha(folder / 'result.json')),
                  'incomplete result seal')
        v.require(result['job'] == spec and result['plan_sha256'] == v.file_sha(path)
                  and result['status'] == 'completed', 'unfinished or unbound unit')
        for key, value in v.ZERO.items():
            v.require(type(result.get(key)) is type(value) and result[key] == value,
                      'closure cannot authorize new experiments')
        results.append(result)
    v.require(len(results) == expected, 'incomplete batch')
    return results


def summary_row(name, results):
    metrics = [r['metrics']['validation'] for r in results]
    f = [m['feedback'] for m in metrics]
    e = [m['explanation'] for m in metrics]
    return dict(method=name, units=len(results),
                feedback_coverage_mean=statistics.mean(m['coverage'] for m in f),
                feedback_coverage_min=min(m['coverage'] for m in f),
                explanation_coverage_mean=statistics.mean(m['coverage'] for m in e),
                explanation_coverage_min=min(m['coverage'] for m in e),
                expected_feedback_mean=statistics.mean(m['expected_actionable_coverage'] for m in f),
                expected_feedback_min=min(m['expected_actionable_coverage'] for m in f),
                feedback_agreement_min=min(m['agreement'] for m in f),
                explanation_agreement_min=min(m['agreement'] for m in e),
                qualified_KL_max=max(m['kl'] for m in f + e),
                illegal_sum=sum(m['whole_tree_raw_illegal_count'] for m in f),
                uncertified_sum=sum(m['whole_tree_path_uncertified_count'] for m in f),
                passing_units=sum(r['gate']['passed'] for r in results),
                maximum_used_leaves=max(r.get('leaf_count', 32) for r in results),
                leaf_calls=sum(r['solver']['counts']['leaf'] for r in results),
                node_calls=sum(r['solver']['counts']['node'] for r in results),
                proof_calls=sum(r['solver']['counts']['proof'] for r in results))


def close():
    root = v.OUTPUT
    v.require(not (root / 'candidate.closed.json').exists(), 'candidate already sealed')
    v.require(not (root / 'closure-manifest.json').exists(), 'never overwrite an interrupted closeout')
    path0, plan0 = checked_plan()
    path32, plan32 = checked_stage('TAO32')
    path64, plan64 = checked_stage('TAO64')
    stages = [(path0, plan0, 24), (path32, plan32, 6), (path64, plan64, 6)]
    batches, summaries = [], []
    from scripts.audit_warehouse_v626 import audit as audit_A0
    from scripts.audit_warehouse_v626_search import audit as audit_A1
    for index, (path, plan, count) in enumerate(stages):
        results = checked_results(path, plan, count)
        summary = read(path.parent / 'summary.json')
        stored_audit = v.bound(summary['independent_audit'])
        replay = (audit_A0 if index == 0 else audit_A1)(path, results)
        v.require(stored_audit == replay and replay['integrity_passed'] is True,
                  'independent arithmetic replay changed')
        v.require(summary['scientific_pass'] is False,
                  'this tool only closes a candidate that exhausted both routes')
        batches.append(results)
        summaries.append(v.pointer(path.parent / 'summary.json'))
    v.require(read(path64.parent / 'summary.json')['next_stage'] ==
              'closed_failed_program_precheck', '64-leaf failure required')
    # No B allocation, trajectory collection or Actor-training authority exists.
    v.require(not any((root / name).exists() for name in ('B', 'confirmation', 'PPO', 'training', 'release')),
              'unexpected later-stage data; manual investigation required')
    actor = plan0['actor']
    v.require(v.pointer(actor['path']) == actor and actor['sha256'] == v.old.ACTOR_SHA256,
              'original complete Actor changed')
    tests_path = ROOT / 'output/warehouse_v626/engineering-search-tests-02.xml'
    tests = ET.parse(tests_path).getroot()
    suites = list(tests) if tests.tag == 'testsuites' else [tests]
    test_counts = {key: sum(int(s.get(key, 0)) for s in suites)
                   for key in ('tests', 'failures', 'errors', 'skipped')}
    v.require(test_counts['tests'] == 67 and test_counts['failures'] == test_counts['errors'] == 0,
              'registered numerical/budget/recovery test receipt failed')
    rows = [summary_row(name, [r for r in batches[0] if r['job']['arm'] == name])
            for name in ('C0', 'C1', 'C2', 'C3')]
    rows += [summary_row(name, rs) for name, rs in zip(('TAO32', 'TAO64'), batches[1:], strict=True)]
    csv_out = io.StringIO(newline='')
    writer = csv.DictWriter(csv_out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    with (root / 'comparison.csv').open('x', newline='') as f:
        f.write(csv_out.getvalue())
    pct = lambda x: f'{100 * x:.2f}%'
    lines = ['# Warehouse v6.26 执行结果', '',
             f'候选：`{v.CANDIDATE}`。**程序开发预检失败，候选关闭；未获得模型完整验收。**', '',
             '## 实测对照', '',
             '以下平均值为六个角色／折的等权平均；准入逐单元检查，不能以均值替代。', '',
             '| 方法 | 平均解释覆盖 | 最差解释覆盖 | 最差反馈覆盖 | 最差预期反馈 | 最低F/E一致 | 最高合格KL | 非法/未证 | 通过单元 |',
             '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append('| ' + ' | '.join([r['method'], pct(r['explanation_coverage_mean']),
            pct(r['explanation_coverage_min']), pct(r['feedback_coverage_min']),
            pct(r['expected_feedback_min']), pct(r['feedback_agreement_min']) + '/' +
            pct(r['explanation_agreement_min']), f"{r['qualified_KL_max']:.4f}",
            f"{r['illegal_sum']}/{r['uncertified_sum']}", f"{r['passing_units']}/6"]) + ' |')
    lines += ['', '门槛保持：每个角色／折反馈和解释覆盖各≥45%、预期反馈≥40%、一致≥95%/97%、KL≤0.20，整树非法和未证明路径均0。', '',
              'A0全部24项及TAO32、TAO64各6项已完成，均经过独立实现的确定性算术重放（并非外部评审或新的封存审计）。',
              '完整表见 [A0](warehouse_v626_A0_results.md)、[TAO32](warehouse_v626_TAO32_results.md)、[TAO64](warehouse_v626_TAO64_results.md)。', '',
              '## 方法实施与局限', '',
              '已实现173维版本化公开域和物理归一化、残差驱动稀疏logistic叶支持选择、路径条件LP及精确有理数双证书校验、硬路由交替优化和64叶容量增长。',
              '32/64分别为容量上限；搜索不强迫使用全部叶，实际叶数和停止原因保存在每项search-report/result中。',
              '所有支持和拓扑选择仅用fit内部80%/20%场景分区，最终全fit重拟合；校准、资格和验证未参与搜索。',
              '这些结果表明：本轮表示与优化组合仍未达到预设覆盖率，不能声称已解决。合法性与已合格区域保真通过，不等于整棵树均已保真。', '',
              '## NN、训练与准入状态', '',
              '| 项目 | 结果 |', '|---|---|',
              '| 双Actor | 完整v6.20权重与角色哈希保持不变 |',
              '| 配送／碰撞／停机 | 14.93／2.84／1/75：仅历史开发基线，本轮没有新测 |',
              '| 新环境轨迹／NN更新／PPO联合步 | 0／0／0 |',
              '| raw／regularized实际外部程序梯度 | 0／0；未进入配对训练 |',
              '| 新数据确认、正式审计和留出测试 | 均未启动，封存种子未使用 |',
              '| 工程开销≤10% | 完整科学运行器开销未独立测量，单列为待验工程门槛 |',
              '| v6发布／pilot | 关闭 |', '',
              '没有向执行接口添加分配、避让或救场规则；程序没有选择环境动作。Pong/Kitchen未重跑已用留出集。', '',
              '## 调用预算与可恢复证据', '',
              '| 阶段 | 叶优化 | 节点优化 | 证明LP |', '|---|---:|---:|---:|']
    for name, batch in zip(('A0-recovery-02', 'TAO32', 'TAO64'), batches, strict=True):
        counts = {k: sum(r['solver']['counts'][k] for r in batch) for k in ('leaf', 'node', 'proof')}
        lines.append(f"| {name} | {counts['leaf']} | {counts['node']} | {counts['proof']} |")
    lines += ['| 早期A0工程中断（不拼接结果） | 265 | 0 | 0 |', '',
              '所有实际求解预扣、失败调用计数，未用余额不转移。早期零拟合准备失败和A0工程中断均原样留存；随后24项全部重新独立运行。',
              f"数值／预算／证书／安全恢复相关测试：{test_counts['tests']}通过，失败和错误均0。", '',
              '完整源码快照、NN/数据/程序哈希、队列、缓存、求解账本和36项结果在output/warehouse_v626/warehouse-v626-domain-complete-01保留；closure-manifest.json封存全部既有文件SHA256。',
              '候选关闭不消耗最终审计/留出资格；不在同一候选下继续调参、训练或发布。', '']
    with (root / 'results.md').open('x') as f:
        f.write('\n'.join(lines))
    for phase, path in (('TAO64', path64),):
        for name, target in [('results-six.md', ROOT / f'docs/warehouse_v626_{phase}_results.md'),
                             ('metrics-six.csv', ROOT / f'experiments/warehouse/evidence/warehouse-v626-{phase}-metrics.csv')]:
            with target.open('xb') as f:
                f.write((path.parent / name).read_bytes())
    for name, target in [('results.md', ROOT / 'docs/warehouse_v626_results.md'),
                         ('comparison.csv', ROOT / 'experiments/warehouse/evidence/warehouse-v626-comparison.csv')]:
        with target.open('xb') as f:
            f.write((root / name).read_bytes())
    original_files = {str(p.relative_to(root)): dict(sha256=v.file_sha(p), bytes=p.stat().st_size)
                      for p in sorted(root.rglob('*')) if p.is_file()}
    manifest = dict(schema='warehouse-v626-closed-inventory.v1', candidate_id=v.CANDIDATE,
                    files=original_files, file_count=len(original_files),
                    bytes=sum(f['bytes'] for f in original_files.values()),
                    scope='all_existing_run_files_before_manifest_and_closure', **v.ZERO)
    write_new(root / 'closure-manifest.json', manifest)
    receipt = dict(schema='warehouse-v626-candidate-closed.v1', candidate_id=v.CANDIDATE,
                   status='closed_failed_program_precheck', scientific_pass=False,
                   failure='Both 32-leaf and 64-leaf development routes failed strict per-role/fold gates',
                   engineering_performance='not_independently_measured_not_scientific_blocker',
                   source_A0=plan0['source_sha256'], source_A1=plan64['source_sha256'],
                   summaries=summaries, actor=actor, actor_role_sha256=v.old.ROLE_SHA256,
                   tests=v.pointer(tests_path), test_counts=test_counts,
                   manifest=v.pointer(root / 'closure-manifest.json'),
                   result_report=v.pointer(root / 'results.md'),
                   closeout_code=v.pointer(Path(__file__).resolve()),
                   permits_new_confirmation=False, permits_PPO=False,
                   permits_same_candidate_retry=False, **v.ZERO)
    write_new(root / 'candidate.closed.json', receipt)
    write_new(ROOT / 'experiments/warehouse/evidence/warehouse-v626.closed.json',
              dict(closure=v.pointer(root / 'candidate.closed.json'), manifest=receipt['manifest'],
                   status=receipt['status'], scientific_pass=False, **v.ZERO))
    # Read-only failed evidence, without deleting or moving any historical path.
    for path in root.rglob('*'):
        if path.is_file():
            os.chmod(path, path.stat().st_mode & ~0o222)
    print(json.dumps(dict(status=receipt['status'], rows=rows, tests=test_counts,
                          files=manifest['file_count'], manifest=receipt['manifest']), ensure_ascii=False))
    return receipt


if __name__ == '__main__':
    close()
