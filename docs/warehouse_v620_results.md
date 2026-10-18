# Warehouse v6.20：已完成结果与当前验证状态

状态日期：2026-10-19（Asia/Shanghai）。`cv-02` 和种子 6160 的两种共同 BC 配方比较已经完成。选中的冻结模型完成新的 275 episode 预检，两个角色均因覆盖不足失败。候选 warehouse-v620-01 关闭，未进入条件 216 draw 或 PPO。v6 participant pilot 继续关闭。

## 1. 旧数据 CV02：提取搜索仍未达到开发门槛

在已使用的 v6.19 开发轨迹上完成 3 折 × 2 角色 × 3 方法，共 18 项程序拟合。Actor 全程冻结，环境 reset、Actor 更新和 PPO 步数均为零。独立复算核对了完整 NN 分布、程序路由、逐帧指标、资产绑定与预算。

| 方法 | 平均解释覆盖 | 最差角色／折解释覆盖 | 最差预期反馈覆盖 | 全部角色／折通过 |
|---|---:|---:|---:|---|
| M0：历史方法，同折重拟合 | 26.63% | 17.50% | 21.14% | 否 |
| M1：持续队列搜索 | 25.12% | 16.79% | 16.04% | 否 |
| M2：稳健解释覆盖搜索 | 24.88% | 21.84% | 21.52% | 否 |

该阶段要求每个角色／折解释覆盖至少 40%、预期反馈覆盖至少 35%，以及既定一致率、KL 和合法性门槛。三种方法均未通过，原生 CV 未选出可准入方法，按预声明路线进入共同 BC 结构学习。M2 的最差覆盖较高，平均覆盖没有提高；有限搜索失败不能证明满足门槛的树不存在。M0 使用历史的每初始化最多 30 轮，M1/M2 使用整次角色拟合合计最多 30 轮，不能据此声称同算力效率优势。已关闭的 `cv-01` 不参与上述结论。

可跟踪收据：[完整 CV02 与 18 项哈希链](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-complete-cv-02.json)。

## 2. 共同 BC 比较：γ=0.03 已锁定

γ 是 BC 内部残差惩罚权重。两种配方共享种子 6160 的双 Actor 初始化、场景／动作参数序列与训练预算；每种均完成 1270 个 BC episode 和 2 次 critic reset，完整 journal 与收据一致。两角色权重均已更新。BC 阶段的外部程序反馈、PPO 步数和 fresh validation episode 均为零。

这里的 γ=0.01／0.03 是共同 BC 配方比较，不对应正式 raw／regularized 分组。正式配对 self-play、封存审计与发布准入仍未完成。

比较阶段对每种冻结 Actor 使用同一套已用的 124 个 v6.19 开发场景：fit 50（25 sampled、25 argmax）、calibration 25、eligibility 25、validation 24。质量来自同一批 24 条 sampled validation 轨迹，并进行同轨迹物理回放；每配方为 124 次采集 reset＋24 次回放 reset，统计质量样本数仍为 24。

| BC γ | 平均配送 | 平均机器人碰撞 | 停机 episode | 旧开发质量合格 | 两角色程序一致性合格 |
|---|---:|---:|---:|---|---|
| 0.01 | 15.67 | 1.875 | 0/24（0%） | 是 | 是 |
| 0.03 | 15.33 | 3.25 | 1/24（4.17%） | 是 | 是 |

质量资格线为配送 ≥5.5、碰撞 ≤6、停机率 ≤5%。程序一致性资格要求两角色反馈／解释一致率分别 ≥93%／95%、合格区 KL ≤0.25、全树原始非法动作与无路径证明动作均为零。四个程序的独立复算确认了保存指标、路由、预算与绑定一致；这些质量与程序指标的评估范围仍是已用开发场景。

### 程序覆盖与选择依据

解释覆盖的分母是该角色全部记录物理帧；分子是落入已合格解释叶的帧数。预期反馈覆盖还计入完整 NN 给程序动作的概率质量。

| BC γ | 角色 | 反馈叶覆盖 | 解释叶覆盖 | 预期反馈覆盖 | 解释一致率 | 解释 KL |
|---|---|---:|---:|---:|---:|---:|
| 0.01 | robot_1 | 19.72% | 19.72% | 19.14% | 99.12% | 0.0552 |
| 0.01 | robot_2 | 24.20% | 24.20% | 23.82% | 99.57% | 0.0294 |
| 0.03 | robot_1 | 26.53% | 24.37% | 25.82% | 99.27% | 0.0433 |
| 0.03 | robot_2 | 23.98% | 23.98% | 23.23% | 99.12% | 0.0492 |

两种配方都满足质量资格与程序一致性偏好，随后按预声明排序最大化最差角色解释覆盖（minE）：γ=0.03 的 23.98% 高于 γ=0.01 的 19.72%，因此锁定 γ=0.03。该排序在最差预期反馈覆盖、KL 和质量分数之前比较 minE；γ=0.03 并非因配送或碰撞成绩更优而入选。

开发比较没有覆盖率硬资格线。选中模型的解释覆盖 24.37%／23.98% 仍低于随后 fresh 预检的各角色 35% 门槛。不得据旧开发结果声称已通过新的程序验收。

原生结果：[γ=0.01](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_bc_pilot/comparison-01/gamma-001/result.json)、[γ=0.03](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_bc_pilot/comparison-01/gamma-003/result.json)、[预声明选择锁](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_bc_pilot/comparison-01/gamma.lock.json)。配方与选择规则见 [共同 BC 比较契约](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/docs/warehouse_v620_bc_pilot_comparison.md)；此前可跟踪的 [实现核验收据](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-shared-bc-implementation-01.json) 只记录工程核验。

## 3. 冻结身份与已完成的 275 预检

选中的模型与方法由 [model-method.lock.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_bc_pilot/comparison-01/model-method.lock.json) 绑定：

| 项目 | 冻结值 |
|---|---|
| γ／方法 | 0.03／M2 `episode_robust_explanation_tree_v1` |
| BC、比较及当前 frozen gate source | `4fba3c172feeaefa11e4c0c2b435e21457f4f27c60c11e5b2cd0c24c618f229c` |
| Actor 资产 SHA256 | `1b6310f6d93309fb17e64516519a71680ad3c30e73f222be27df4dff10da81ec` |
| robot_1 权重 SHA256 | `7aee44859397b8cbe1bb9cd3b8541e82f2a080d884c83404fe1e4fd2a998eaf0` |
| robot_2 权重 SHA256 | `17f2f75877421854c680fab07557f031389800a9503b14f060db5ff4f862fcf0` |

CV02 使用其历史 source `5d8236dee814f6d8b83343a3d2d9ddd453d4aee9dfe4d0376448b424a8b085ed`；上述 `4fba…` 是后续 BC／比较／当前冻结流程的 source。

root 已出具 [模型／源码／配方／精确种子绑定的受限清场](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_ledger/cleared-frozen6160-01/clearance.json)（SHA256 `ffaff9e8dc0dad5f726aa67f4b8818a291acddf8c02042163526acacd7ad6941`）。权限限于 6160 当前单次 frozen gate；6161–6163、PPO、后续开发配对、正式训练、封存／留出与 participant pilot 仍关闭。

本次单次 frozen gate 的预声明顺序为：

1. 275 个互不重叠的 base／物理场景：fit 100（先 50 sampled、后 50 argmax）、calibration 25、eligibility 75、validation 75，基于固定 NN 重新拟合角色程序。
2. 内部程序门槛要求各角色反馈与解释覆盖 ≥35%、预期反馈覆盖 ≥30%、反馈／解释一致率 ≥93%／95%、KL ≤0.25，且全树原始非法／无路径证明动作均为零。
3. 仅在内部预检通过后使用独立的 24 base ×3 task ×3 action stream＝216 draw；共 72 个物理场景、24 个 base 簇。额外 216 次完整物理回放复现相同轨迹，不增加统计样本。独立阶段要求覆盖及实际可用覆盖各 ≥30%，并保持既定一致率、KL、合法性与质量门槛。

实际完成 275 次采集 reset，未启动条件 216 轨迹及其完整物理回放。fresh 预检失败后没有改选另一 γ，没有复用旧比较程序，也没有启动 PPO、后续开发配对、正式训练、封存审计或留出测试。

## 4. 新数据结果：BC 预检失败并关闭候选

四个分区分别含 100／25／75／75 局，物理帧数为 11,988／2,942／8,874／8,978（每角色），不排除停机截短帧。下表仅使用 75 局 validation 的全部 8,978 帧；叶资格由独立 eligibility 分区确定。两棵程序均重新拟合于固定完整 NN 的掩码后分布。

| 指标 | robot_1 | robot_2 | 本次 BC 预检门槛 |
|---|---:|---:|---:|
| 反馈叶覆盖 | 21.01% | 20.96% | ≥35% |
| 解释叶覆盖 | 21.01% | 20.96% | ≥35% |
| 预期可反馈覆盖 | 20.20% | 20.43% | ≥30% |
| 反馈／解释动作一致率 | 97.88% | 99.15% | ≥93%／95% |
| 合格区域分布 KL | 0.0882 | 0.0404 | ≤0.25 |
| 全树原始非法动作数 | 0 | 0 | 0 |
| 无合法路径证明帧数 | 0 | 0 | 0 |

两角色失败项均为反馈覆盖、解释覆盖及预期反馈覆盖。其余硬检查通过。这里不能用正式阶段 20% 的覆盖线替换已锁定的 BC 35% 预检线；正式阶段尚未启动。

每个角色均完成预定 8 初始化、合计 30 搜索轮数。冻结 Actor 权重与双角色 SHA 全程不变，PPO 联合步和 Actor 更新数为零。独立算术复核从原始轨迹重建完整 NN 分布、硬路由、资格集合、指标及门槛决定，确认原生失败结果；复算没有重新采样或拟合。

### 同一批 75 局的描述性任务统计

| 范围 | 局数 | 平均配送 | 平均碰撞 | 停机局数 |
|---|---:|---:|---:|---:|
| 全部 | 75 | 14.93 | 2.84 | 1/75（1.33%） |
| Task 1 | 25 | 14.48 | 3.16 | 0/25 |
| Task 2 | 25 | 15.36 | 2.72 | 0/25 |
| Task 3 | 25 | 14.96 | 2.64 | 1/25 |

共配送 1120 次，单局 10–19 次。上述统计仅从已有预检记录汇总，未增加环境运行；它不是原定独立 216 条压力轨迹的验收结果，也不是留出或真人效果。任务统计不能抵消程序覆盖门槛失败。

### 失败区域的事后诊断

仅用本次已保存数据作事后描述：robot_1 最大未合格叶占验证帧 27.98%，程序／NN 首选动作一致率 41.76%、KL 0.791；robot_2 最大未合格叶占 37.84%，一致率 38.45%、KL 0.942。对应资格数据各有 2581／3234 帧，均跨 75 个 episode，合法路径与样本支持检查通过；失败项是动作一致率、Wilson 下界及 KL。这些大叶混合了不同的 NN 决策，不能通过补充样本数量直接视为合格。

按完整 NN 首选动作分层，WAIT 帧的解释覆盖为 66.16%／56.69%，UP 帧只有 1.39%／0%，DOWN 帧只有 0.124%／0.068%。因此“五个动作”并未使当前小树足以区分产生这些动作的状态。该诊断不改变门槛、γ 或候选选择，也不证明有限树搜索已找到最优树。原始逐叶、逐动作统计保存在关闭证据绑定的 posthoc 诊断中。

结果支持的结论是：该共同 BC 配方仍未使足够多状态进入合格程序区域。合格区域本身有较高一致率，不能据此把未覆盖的约 79% 帧视为已有解释。有限搜索和两个配方的失败也不能证明不存在满足限制的树。

[原生结果](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v620_frozen/warehouse-v620-01-gamma003-precheck-01/result.json)；[关闭候选及独立复算证据](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-frozen-precheck-01.json)。新数据已记录为本候选已用预检数据，不能在下一候选中重新声称为全新验证。条件 216、封存审计与留出未被消费；重试当前候选、改门槛或补充执行动作规则均未发生。

## 5. Pong、Kitchen 与研究范围

Pong、Kitchen 保留现有已验收收据与资产身份，本轮进行的是已有证据、资产 SHA 与绑定复核，没有重跑它们已使用的留出集。A/B 使用相同 regularized 绑定，C 使用独立 raw 绑定；raw 外部程序反馈为零。参见可跟踪的 [三域既有资产复核](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-pong-kitchen-reverification-01.json) 与 [归档证据清单](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-carried-domain-evidence-01.json)。

Kitchen 现有 [24 项三 task 工程核验收据](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/experiments/warehouse/evidence/warehouse-v620-kitchen-all-task-tests-01.json) 覆盖开发 fixture 中的冻结 NN／物理掩码动作、真人接管接口、恢复／回放及反事实绑定；其范围是工程一致性。离线人类代理成绩始终标为代理结果，没有测得新的真人协作收益。Warehouse 尚未完成准入，三域 v6 pilot 继续关闭。
