#!/usr/bin/env python3
"""Read-only scientific replays followed by append-only failure sealing."""
from __future__ import annotations
import json,os,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v627_protocol as v
from scripts.run_warehouse_v625 import write_new


def close():
    from scripts.run_warehouse_v627 import checked_plan
    from scripts.audit_warehouse_v627 import audit
    v.require(not (v.OUTPUT/'candidate.closed.json').exists() and not (v.OUTPUT/'method-lock.json').exists(),
              'only an unsealed failed candidate can close')
    batches={};summaries=[]
    for stage in ('S32','S64'):
        path,plan=checked_plan(stage);s=v.bound(v.pointer(path.parent/'summary.json'))
        v.require(s['all_six_settled'] and s['scientific_pass'] is False,'both complete failed capacity routes required')
        v.require(json.loads((path.parent/'execution/completed.json').read_bytes())==dict(all_owned_workers_waited=True,jobs=6),
                  'owned workers not all waited')
        rs=[]
        for spec in plan['jobs']:
            folder=path.parent/'jobs'/spec['job_id'];r=json.loads((folder/'result.json').read_bytes())
            v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),
                      'unit result seal changed')
            rs.append(r)
        v.require(audit(path,rs)==v.bound(s['independent_audit']),'independent arithmetic replay changed')
        summaries.append(v.pointer(path.parent/'summary.json'));batches[stage]=rs
        for source,target in [('results-six.md',ROOT/f'docs/warehouse_v627_{stage}_results.md'),
                              ('metrics-six.csv',ROOT/f'experiments/warehouse/evidence/warehouse-v627-{stage}-metrics.csv')]:
            with target.open('xb') as f:f.write((path.parent/source).read_bytes())
    v.require(s['next_stage']=='closed_failed_program_precheck','unexpected final stage')
    v.require(not any((v.OUTPUT/name).exists() for name in ('B','confirmation','training','PPO','release')),'later-stage data unexpectedly exist')
    from study_v3 import policies
    from study_v3.release_catalog import RELEASE_ID
    refusals={}
    for group in ('A','B','C'):
        try:policies.binding_for_group('warehouse',group,release_id=RELEASE_ID)
        except policies.PolicyUnavailable as e:refusals[group]=str(e)
        else:raise RuntimeError('failed candidate cannot bind a participant group')
    lines=['# Warehouse v6.27执行结果','',f'候选`{v.CANDIDATE}`：**开发程序预检失败，候选关闭。**','',
        '| 方法 | 完成数 | 平均解释覆盖 | 最差解释覆盖 | 最差反馈覆盖 | 最差预期反馈 | 通过数 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| v6.26 C0 | 6 | 36.68% | 34.06% | 34.06% | 32.93% | 0/6 |',
        '| v6.26 TAO64 | 6 | 37.58% | 33.34% | 37.95% | 36.56% | 0/6 |']
    for name,rs in batches.items():
        complete=[r for r in rs if r['status']=='completed']
        if complete:
            e=[r['metrics']['validation']['explanation']['coverage'] for r in complete]
            f=[r['metrics']['validation']['feedback']['coverage'] for r in complete]
            a=[r['metrics']['validation']['feedback']['expected_actionable_coverage'] for r in complete]
            fmt=lambda x:f'{100*x:.2f}%'
            numbers=[fmt(statistics.mean(e)),fmt(min(e)),fmt(min(f)),fmt(min(a))]
        else:numbers=['—']*4
        lines.append('| '+' | '.join([name,str(len(complete))]+numbers+[str(sum(r['gate']['passed'] for r in rs))+'/6'])+' |')
    lines+=['','各项逐角色/折门槛保持：F/E覆盖45%、预期F40%、一致95%/97%、KL≤.20、非法/未证0；还须完成四起点的必需探索。',
        '若有预算终止，均值仅描述已完成单元，不能补零或据此声称完整方法有效。','',
        '实现了语义划分与显式合法性编译、四起点全树硬路由梯度候选、子树重建/预算交换、有限beam和与最终相同的资格评分。',
        '32/64限制包含所有guard、fallback、空叶；输出资产绑定语义结构、编译器、旧数值证明内核及Actor哈希。',
        '本轮没有获得全部合格的候选；有限预算局部/梯度搜索不是对整个程序类的不可行性证明。','',
        '原v6.20完整双Actor始终不变；配送14.93、碰撞2.84、停机1/75仅为旧开发基线，本轮没有新测。',
        '新环境轨迹、Actor更新、PPO联合步及实际外部程序训练梯度均0。程序自身的梯度拟合不计作Actor训练。',
        '未启动新数据确认、正式独立审计或留出测试；Pong/Kitchen未重跑已用留出。',
        '工程开销≤10%仍待独立测量，未据此阻断科学指标。v6发布/pilot保持关闭，A/B/C实际绑定入口均拒绝。','',
        '逐项表见warehouse_v627_S32_results.md和warehouse_v627_S64_results.md；原始结果、逐次预扣、源码及状态对象均封存。','']
    raw='\n'.join(lines).encode()
    with (v.OUTPUT/'results.md').open('xb') as f:f.write(raw)
    with (ROOT/'docs/warehouse_v627_results.md').open('xb') as f:f.write(raw)
    manifest={str(p.relative_to(v.OUTPUT)):dict(sha256=v.file_sha(p),bytes=p.stat().st_size)
              for p in sorted(v.OUTPUT.rglob('*')) if p.is_file()}
    write_new(v.OUTPUT/'closure-manifest.json',dict(schema='warehouse-v627-failure-inventory.v1',
        candidate_id=v.CANDIDATE,files=manifest,file_count=len(manifest),**v.ZERO))
    receipt=dict(schema='warehouse-v627-closed.v1',candidate_id=v.CANDIDATE,status='closed_failed_program_precheck',
        scientific_pass=False,source_sha256=plan['source_sha256'],summaries=summaries,
        actor=plan['actor'],actor_role_sha256=plan['actor_role_sha256'],
        tests=v.pointer(ROOT/'output/warehouse_v627/tests-01.xml'),
        manifest=v.pointer(v.OUTPUT/'closure-manifest.json'),report=v.pointer(v.OUTPUT/'results.md'),
        participant_binding_refusals=refusals,permits_same_candidate_retry=False,**v.ZERO)
    write_new(v.OUTPUT/'candidate.closed.json',receipt)
    write_new(ROOT/'experiments/warehouse/evidence/warehouse-v627.closed.json',dict(closure=v.pointer(v.OUTPUT/'candidate.closed.json'),
        status=receipt['status'],manifest=receipt['manifest'],**v.ZERO))
    for p in v.OUTPUT.rglob('*'):
        if p.is_file():os.chmod(p,p.stat().st_mode&~0o222)
    print(json.dumps(dict(status=receipt['status'],manifest=receipt['manifest'],files=len(manifest))))
    return receipt


if __name__=='__main__':close()
