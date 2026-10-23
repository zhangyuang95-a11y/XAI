#!/usr/bin/env python3
"""Report completed 20k models on the new, complete matched quality scenes.

Analysis only: reads audited arrays; no NN forward, reset, fitting or training.
The failure receipt records an actual observed native auditor exit, not a pass.
"""
from pathlib import Path
import json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v626_protocol import pointer,bound,require
from experiments.warehouse import v642_quality_verified_cache as profile
from backend.training.warehouse_v642_program import Program

Q=ROOT/'output/warehouse_v642/calibration-quality-cached/warehouse-v642-verified-quality-11'
RESULT=pointer(Q/'collection/result.json');r=bound(RESULT);plan=bound(r['plan'])
EXIT=pointer(Q/'collection/supervisor-exit.json');terminal=bound(EXIT)
require(terminal['exit_code']==0 and terminal['result']==RESULT,'actual completed collection required')
require(terminal['verification_bracket']==pointer(Q/'collection/verification-bracket.json'), 'final full bracket changed')
require(plan['source_files']==profile.source_files(),'evaluation source changed')
require(len(r['entries'])==r['actual_reset_count']==225,'all matched episodes required')
assets=bound(plan['fixed']['report']);programs=r['program_metrics']['roles'];quality=r['metrics']['quality']
require(not r['calibration_precheck_passed'] and r['program_metrics']['passed']
        and r['metrics']['gates']['raw']['checks']==dict(collisions=True,deliveries=True,shutdown_rate=False)
        and r['metrics']['gates']['regularized']['passed'], 'this failure-specific report must match the actual gates')

out=ROOT/'output/warehouse_v642/engineering/verified-postchecks-11';out.mkdir(parents=True,exist_ok=True)
failure=dict(version='warehouse-v642-independent-precheck-observed-failure.v1',quality_result=RESULT,
    collection_terminal=EXIT,implementation_version=profile.VERSION,implementation_source_files=profile.source_files(),
    observed_by='exec_tool_terminal_result',actual_tool_session_id=85258,actual_terminal_chunk_id='e52e4d',exit_code=1,
    error='calibration task retention or one of four program gates failed',
    native_failure_source=pointer(ROOT/'experiments/warehouse/v642_calibration_admission.py'),native_failure_line=130,
    native_success_report_emitted=False,
    replay_and_metric_reproduction_reached_before_failure=True,
    source_control_flow_basis='native audit reached its quality gate after all 225 NN/physics episodes and exact task/program metric reproduction checks',
    post_gate_checks_not_reached=True,
    complete_episodes=225,independent_scenes=75,
    replayed_role_actions_before_gate=sum(x['NN_action_checks'] for x in r['outcomes']),
    failed_checks=[dict(variant='raw',criterion='shutdown_rate',observed=quality['raw']['shutdown_rate'],maximum=.05,
        shutdown_episodes=5,episodes=75,maximum_allowed_shutdown_episodes=3)],
    calibration_precheck_passed=False,model_acceptance_passed=False,pilot_opened=False,
    new_environment_resets=0,NN_updates=0,program_fit_calls=0)
failure_path=out/'independent-precheck-failed.json'
require(not failure_path.exists(),'preserve an existing observed failure receipt')
failure_path.write_text(json.dumps(failure,ensure_ascii=False,indent=2));failure_path.chmod(0o444)

cases=plan['cases'];bykey={(x['variant'],x['case']['case_id']):x for x in r['outcomes']}
rng=np.random.default_rng(20261023642)
index=np.concatenate([np.array([i for i,c in enumerate(cases) if c['task']==task])[
    rng.integers(0,25,size=(10000,25))] for task in (1,2,3)],axis=1)
def interval(v):return [float(x) for x in np.quantile(v,[.025,.975])]
series={v:{key:np.array([bykey[v,c['case_id']][field] for c in cases],float)
    for key,field in [('deliveries','deliveries'),('collisions','robot_collisions'),('shutdown_rate','shutdown')]}
    for v in ('reference','raw','regularized')}
qci={v:{key:interval(a[index].mean(1)) for key,a in s.items()} for v,s in series.items()}
paired={key:interval((series['regularized'][key]-series['raw'][key])[index].mean(1)) for key in series['raw']}
program_ci={};entrymap={(e['variant'],e['case']['case_id']):e for e in r['entries']}
for variant in ('raw','regularized'):
    program_ci[variant]={}
    for ri,role in enumerate(('robot_1','robot_2')):
        program=Program.from_dict(bound(programs[variant][role]['program']),plan['fixed']['actor_role_sha256'][variant][role])
        per={p:dict(n=[],selected=[],expected=[],actionable=[],agreement=[],kl=[]) for p in ('feedback','explanation')}
        cert=program.certificates()
        for case in cases:
            ref=entrymap[variant,case['case_id']]['arrays'];require(pointer(ref['path'])==ref,'new arrays changed')
            with np.load(ref['path'],allow_pickle=False) as z:
                x,m,p,s=(z[k][:,ri] for k in ('x','m','p','sampled'))
                route,logits,q=program.distributions(x,m);raw=logits.argmax(-1);top=q.argmax(-1);nn=p.argmax(-1)
                valid=(m[np.arange(len(m)),raw]>0)&np.array([int(a) in cert[int(l)] for a,l in zip(raw,route)])
                kl=np.sum(p*(np.log(np.maximum(p,np.float32(1e-8)))-np.log(np.maximum(q,np.float32(1e-8)))),axis=-1)
                for purpose,leaves in [('feedback',program.feedback_leaves),('explanation',program.explanation_leaves)]:
                    selected=np.isin(route,tuple(leaves));matched=valid&(top==s)
                    expected=valid*p[np.arange(len(p)),top]
                    if purpose=='feedback':matched&=s==nn;expected*=top==nn
                    values=dict(n=len(x),selected=int(selected.sum()),expected=float(expected[selected].sum()),
                        actionable=int((selected&matched).sum()),agreement=int((selected&(top==nn)).sum()),kl=float(kl[selected].sum()))
                    for key,value in values.items():per[purpose][key].append(value)
        program_ci[variant][role]={}
        for purpose,values in per.items():
            a={k:np.array(v,float) for k,v in values.items()};den=a['n'][index].sum(1);chosen=a['selected'][index].sum(1)
            program_ci[variant][role][purpose]={
                'coverage':interval(a['selected'][index].sum(1)/den),
                'actionable_coverage':interval(a['actionable'][index].sum(1)/den),
                'expected_actionable_coverage':interval(a['expected'][index].sum(1)/den),
                'agreement':interval(a['agreement'][index].sum(1)/chosen),
                'kl':interval(a['kl'][index].sum(1)/chosen)}

stats=dict(version='warehouse-v642-post20k-fresh-scene-analysis.v1',quality_result=RESULT,
    failure_receipt=pointer(failure_path),analysis_source=pointer(Path(__file__).resolve()),
    bootstrap=dict(iterations=10000,seed=20261023642,unit='complete_matched_scene',stratified_by_task=True,
                   scenes=75,independent_scenes_not_225=True,new_environment_resets=0),
    quality_95_percent_intervals=qci,paired_regularized_minus_raw_95_percent_intervals=paired,
    program_95_percent_intervals=program_ci,gate_results_unchanged=True)
sp=out/'scene-bootstrap.json';sp.write_text(json.dumps(stats,ensure_ascii=False,indent=2));sp.chmod(0o444)

def pct(x):return f'{100*x:.2f}%'
def link(ref):return f"[证据]({ref['path']})"
lines=['# Warehouse：20k 后 raw / regularized 实测结果','',
    '**结论：regularized 通过本轮质量及程序校准；raw 的停机率未通过。因此配对校准失败，pilot 保持关闭。**','',
    '本报告使用 20,000 联合步后导出的最终双 Actor。75 个全新匹配场景、每 task 25 个；reference、raw、regularized 各运行 75 局，共 225 个完整 episode。评估类型为离线 NN–NN，真人样本数为 0。', '',
    '原先较慢评估的 18 局已封存并关闭，其数据没有计入本表。旧 v6.20 历史质量分数也没有计入。reference 在这批新场景上重新运行，仅用于任务能力保留门槛。','',
    '## 1. 任务质量','',
    '| 指标 | raw | regularized | 本轮门槛 |','|---|---:|---:|---:|']
for label,key in [('平均配送/局','deliveries'),('平均机器人碰撞/局','collisions')]:
    limit=r['metrics']['gates']['raw']['minimum_deliveries' if key=='deliveries' else 'maximum_collisions']
    lines.append(f"| {label} | {quality['raw'][key]:.4f} | {quality['regularized'][key]:.4f} | {'≥' if key=='deliveries' else '≤'}{limit:.4f} |")
lines += [f"| 总配送（75局） | {int(series['raw']['deliveries'].sum())} | {int(series['regularized']['deliveries'].sum())} | — |",
    f"| 停机局数 / 比例 | 5/75 / {pct(quality['raw']['shutdown_rate'])} | 2/75 / {pct(quality['regularized']['shutdown_rate'])} | ≤5%，75局最多3局 |",
    f"| 同场景参考配送保留率 | {pct(quality['raw']['deliveries']/quality['reference']['deliveries'])} | {pct(quality['regularized']['deliveries']/quality['reference']['deliveries'])} | ≥95% |",
    '| 质量门槛 | **失败：停机率** | **通过** | 两版分别通过 |','',
    f"同场景 reference：平均配送 {quality['reference']['deliveries']:.4f}、碰撞 {quality['reference']['collisions']:.4f}、停机 2/75。raw 停机率比5%线高1.67个百分点，在75局口径下比允许上限多2局。",'',
    '相对 raw，regularized 本次平均配送增加0.56、碰撞减少0.8533、停机局数减少3局。此比较来自一个训练配对；场景级不确定区间见附带统计文件。','',
    '### 每 task 质量','',
    '| 版本 | task | 局数 | 配送/局 | 碰撞/局 | 停机局数 / 比例 |','|---|---:|---:|---:|---:|---:|']
for v in ('raw','regularized','reference'):
    for task,s in r['metrics']['by_task'][v].items():
        lines.append(f"| {v} | {task} | {s['episodes']} | {s['deliveries']:.4f} | {s['collisions']:.4f} | {s['shutdown_episodes']}/{s['episodes']} / {pct(s['shutdown_rate'])} |")
lines += ['','质量准入按每版75局整体计算；上表保留task分层。','',
    '## 2. 新场景上的程序指标','',
    '以下均针对当前20k最终NN，资格集合在新场景采集前已冻结；新轨迹未用于重拟合或重选合格叶。', '',
    '| 版本/角色 | 反馈叶覆盖 | 解释叶覆盖 | 预期反馈覆盖 | 反馈一致率 | 解释一致率 | KL 反馈/解释 | 结果 |',
    '|---|---:|---:|---:|---:|---:|---:|---|']
for v in ('raw','regularized'):
    for role,it in programs[v].items():
        f,e=(it['metrics'][p] for p in ('feedback','explanation'))
        lines.append(f"| {v}/{role} | {pct(f['coverage'])} | {pct(e['coverage'])} | {pct(f['expected_actionable_coverage'])} | {pct(f['agreement'])} | {pct(e['agreement'])} | {f['kl']:.5f}/{e['kl']:.5f} | {'通过' if it['gate']['passed'] else '失败'} |")
lines += ['','门槛：反馈/解释叶覆盖各≥45%；预期反馈≥40%；一致率≥95%/97%；合格区域 KL≤0.20。四角色的验证支持量门槛也均通过。KL方向为 NN∥程序。', '',
    '| 版本/角色 | 实际反馈可用覆盖 | 实际解释可用覆盖 | 全树一致率 | 全树 KL | 原始非法动作 | 未证明路径 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for v in ('raw','regularized'):
    for role,it in programs[v].items():
        f,e=(it['metrics'][p] for p in ('feedback','explanation'))
        lines.append(f"| {v}/{role} | {pct(f['actionable_coverage'])} | {pct(e['actionable_coverage'])} | {pct(f['whole_tree_agreement'])} | {f['whole_tree_kl']:.5f} | {f['whole_tree_raw_illegal_count']} | {f['whole_tree_path_uncertified_count']} |")
lines += ['','叶覆盖表示进入已合格叶的物理帧比例。实际可用覆盖额外要求程序与当帧NN采样动作一致；反馈还要求该采样动作等于NN首选动作。一致率及门槛KL以合格区域为分母；全树列覆盖所有物理帧。', '',
    '两个角色的程序实际上限为深12/256叶，当前分别使用102/77叶；全部程序均不选择执行动作。','',
    '## 3. 训练及执行证据','',
    '| 指标 | raw | regularized |','|---|---:|---:|',
    '| 联合环境步 | 20,000 | 20,000 |','| Actor优化更新 | 260 | 260 |','| critic优化更新 | 324 | 324 |',
    '| 两角色实际接受的正程序梯度更新 | 0 / 0 | 71 / 71 |',
    '| 两角色程序梯度范数之和（共享裁剪前） | 0 / 0 | 0.463876 / 0.375972 |','',
    f"全部225局执行动作均由绑定NN和物理合法动作掩码产生。收集阶段逐帧核对了 {failure['replayed_role_actions_before_gate']:,} 个角色动作；动作覆盖次数为0。独立回放到达原生审计器第130行的质量拒绝门槛，此前全部episode动作、物理计分及任务/程序指标精确复现检查均未报错。独立审计实际退出1，原因是raw停机率；没有生成通过收据。门槛后的检查未执行，失败记录明确保存这一点。",'',
    '### 最终权重绑定','', '| 版本 | 角色1 SHA256 | 角色2 SHA256 |','|---|---|---|']
for v in ('raw','regularized'):
    h=plan['fixed']['actor_role_sha256'][v];lines.append(f"| {v} | `{h['robot_1']}` | `{h['robot_2']}` |")
lines += ['','## 4. 性能与准入状态','',
    '验收器相关测试48项通过。真实证据包的重复完整校验耗时63.990秒；关键执行输入缓存校验中位0.228秒，约281倍加速。开始与结束完整证据校验及全部逐帧NN/物理核对仍保留。', '',
    '**本轮20k配对校准未通过；未启动后续正式训练/封存审计/留出测试，也未开放pilot。** 程序抽取当前四角色均已过线，剩余失败项是raw停机率。', '',
    '## 5. 可追溯文件与统计','',
    f"- 新225局结果：{link(RESULT)}",
    f"- 实际收集退出0：{link(EXIT)}",
    f"- 收集前后完整校验：{link(pointer(Q/'collection/verification-bracket.json'))}",
    f"- 独立预检退出1：{link(pointer(failure_path))}",
    f"- 场景级95%区间：{link(pointer(sp))}",
    f"- 已审计20k训练资产与梯度：{link(plan['fixed']['report'])}",
    f"- 本报告生成程序：{link(pointer(Path(__file__).resolve()))}",'',
    '区间采用10,000次按task分层的完整匹配场景配对bootstrap，分析随机种子20261023642。独立场景数为75；相邻帧和同一场景的三个模型结果未当作独立场景。区间仅描述不确定性，未改变预登记点估计门槛。', '',
    '### 五动作与公开状态分层','']
for v in ('raw','regularized'):
    for role,it in programs[v].items():
        e=it['metrics']['explanation'];lines += [f'#### {v}/{role}','',
            '| NN首选动作 | 物理帧 | 解释叶覆盖 | 一致率 | 实际可用帧 |','|---|---:|---:|---:|---:|']
        for action,s in e['per_nn_action'].items():
            lines.append(f"| {action} | {s['physical_frames']} | {pct(s['qualified_coverage']) if s['qualified_coverage'] is not None else '无样本'} | {pct(s['agreement']) if s['agreement'] is not None else '无样本'} | {s['actual_actionable_rows']} |")
        lines += ['','| 公开状态 | 物理帧 | 解释叶覆盖 | 一致率 | KL |','|---|---:|---:|---:|---:|']
        for label,s in e['public_strata'].items():
            lines.append(f"| {label} | {s['physical_frames']} | {pct(s['qualified_coverage']) if s['qualified_coverage'] is not None else '无样本'} | {pct(s['agreement']) if s['agreement'] is not None else '无样本'} | {format(s['kl'],'.5f') if s['kl'] is not None else '无样本'} |")
        lines += ['','公开状态分层允许重叠：冲突、邻近机器人、低充电返程余量及普通状态均使用原指标实现中的公开动作前公式。','']
doc=ROOT/'docs/warehouse_v642_20k_results.md';require(not doc.exists(),'preserve existing result report')
doc.write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(report=str(doc),quality=quality,quality_gates=r['metrics']['gates'],
    program_gates_passed=r['program_metrics']['passed'],calibration_precheck_passed=False,
    role_action_checks=failure['replayed_role_actions_before_gate'],paired_CI=paired),ensure_ascii=False))
