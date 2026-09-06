# 双机器人协作配送 XAI 用户实验

参与者固定控制 `robot_1`，与 MAPPO 控制的 `robot_2` 在 6×7 紧凑错位货架仓库中完成两个持续补充的共享 A→B 配送任务。服务器分配后、演示与任务开始前，界面会明确告知参与者属于 A 组（Task 1 可即时询问 Robot 2）或 B 组（无即时解释）。实验流程为：AI–AI 操作演示（可完整观看或提前结束）→ Task 1 → Task 2 → 简短问卷；不存在独立解释阶段。提前结束演示会记录已观看帧数、剩余帧数和完成比例，然后以独立 seed、新状态和双方满电开始 Task 1。

本仓库也包含独立的 **PolicyLens 协作厨房**研究环境、神经队友、解释服务、实验网页、训练与验收脚本。厨房目前只开放内部预实验，使用参与者自填的匿名用户 ID，并继续执行 A/B × XY/YX 四人随机区组；入口、运行命令、数据边界和正式发布门槛见 [协作厨房研究说明](docs/cooperative_kitchen_research.md)。冻结 Actor、抽取程序和题库答案通过 Render Secret File 注入，不存放在公开 Git 历史中；厨房使用独立的 [`render-kitchen.yaml`](render-kitchen.yaml)，不会复用物流仓库服务进程。

## 当前实验契约

- 两台机器人、120 个联合决策步，初始电量均为 100。
- 任一空载机器人进入 A 点后认领任务；只有承运机器人能在 B 点完成交付。
- 成功移动消耗 2 电量；在充电站执行等待恢复 10；阻塞动作不耗电。
- 正式地图是 `warehouse_staggered_aisles_6x7_v2_three_cell_exit_no_cross`：左右作业通道错位，不形成四向十字路口。机器人从 `(5,2)`、`(5,4)` 出发，充电站位于 `(5,3)`；正上方 `(4,2)`、`(4,3)`、`(4,4)` 是三个连续且真实可通行的出口格。取货点、交付点、充电站、路径规划、观测、碰撞判定和浏览器全部读取同一个 `MapLayout`。
- 每一步先冻结共同状态 `S_t`，两个共享参数 Actor 分别只读取各自的 `S_t` 本地观察并独立产生动作，最后只调用一次 `env.step({robot_1: a1, robot_2: a2})`。参与者命令只在两个分布都计算完成后替换 `robot_1`；`robot_2` 永远看不到 `robot_1` 本帧动作。
- 机器人冲突时双方本步均等待，计一次碰撞但不终止。除参与者在 Task 1/2 中替换 `robot_1` 的输入外，AI 命令由共享 MAPPO Actor 直接输出并原样提交给环境；系统不存在运行时通行权规则、coordination shield、教师策略或决策树动作改写。撞墙、货架阻塞、同格争抢、交换位置和进入未离开的队友位置，均只由环境动力学解析。网络必须自行学习任务分工、充电、让行和避免绕路。
- 用户得分保持为：`100×配送数 − 200×机器人碰撞事件 − 50×断电事件 − 步数 − 2×参与者绕路单位`。断电提前结束时补扣到 120 步。

## 强化学习 Reward

参与者得分和训练 Reward 是两个独立层次。界面、数据库和实验分析只使用未缩放的用户得分；训练专用势能不会进入参与者得分。

每个联合步保留共享任务结果，但效率归因按机器人独立计算：

```text
reward_i = user_score_delta / 100
         + 0.01 * frozen_safe_cost_reduction_i
         + clipped_shared_coordination_progress
         - 0.02 * clipped_counterfactual_regret_i
         - repeated_avoidable_wait_penalty_i
```

每步开始时先冻结每台机器人的安全目标：载货配送、必要充电、已承诺 pickup，或一次性安全任务匹配。安全任务成本包含到 A、A→B、交付后返充、安全余量，以及必要的充电等待；同一步内重新匹配或生成替补任务不能制造虚假进展。每减少一个安全动作奖励 `+0.01`，所以必要充电 `WAIT` 大致抵消 `−0.01` 时间成本。

- 断电终止步的势能塑形严格为 0，不会产生“死亡奖励”。
- 达到安全电量即可离开充电站，不要求充满至 100。
- 满电等待、无必要充电和充放电循环不能刷取正奖励。
- 本步新生成的替补任务不进入本步 `potential_after`，从下一步起再同时进入势能两端。
- 固定队友动作后枚举本机器人安全动作；所选下一步距离比最佳安全距离多出的部分截断到 `[0,2]`，每单位扣 `0.02`。必要充电、真实让行、排队、charger clearance 和没有安全进展动作时不扣。
- 第一次可避免 `WAIT` 只由 regret 处理；从第二次起额外依次扣 `0.01/0.02/0.03/0.04`，上限 `0.04`。合法等待立即清零 streak。
- coordination delay 按实际清障步数计量，默认上限 4 个成本单位，单步共享 shaping 截断到 `±0.04`；往返循环净收益不为正。
- 旧的全局 mission regression 默认关闭。日志保存每台机器人的 progress、regret、WAIT streak/penalty、普通/载货绕路、协调奖励和路径效率。

## 训练方法与版本

正式策略采用共享 Actor、集中式 Critic 的 MAPPO。Actor 是独立、去中心化的同时决策结构；Critic 训练时可以读取全局状态。30% 训练回合启用代理人类，其中约 20% 时间步会在两个 Actor 输出完成后替换机器人 1 的动作；替换样本不进入 Actor loss，但保留给 Critic。

训练开始前可使用可审计的行为克隆样本初始化神经网络权重；这些标签不会作为运行时动作，也不会在 MAPPO rollout 中替换 Actor 输出。主体训练的每一步均执行当前 Actor 动作。早期少量训练回合使用单机器人低电量课程，课程在后期衰减到 0；正式评估和用户任务始终从电量 100 开始。RCPD 只单向读取神经网络真实 rollout 并在训练后抽取程序，不向 Actor 提供 target、loss 或梯度，也不参与动作选择。

当前部署版本：

- 模型：`warehouse_mappo_v68_causal_coordination_6x7_actor`
- 环境：`warehouse_collaborative_delivery_v43_compact6_live_human_ai`
- Reward：`warehouse_safe_mission_reward_v29_temporal_consistency`
- 观测：`collaborative_observation_v38_frozen_plan_mask`
- 训练 checkpoint：`warehouse_mappo_training_v59_human_ai_live_6x7`
- RCPD 合同：`warehouse_rcpd_v60_live_human_ai_trace`
- 地图：`warehouse_staggered_aisles_6x7_v2_three_cell_exit_no_cross`
- seed 库合同：`warehouse_parallel_seed_pairs_v60_compact6`
- 人机时间线合同：`warehouse_human_ai_timeline_v60`
- 动作执行：`frozen_joint_plan_atomic_actor_v14`
- 运行时控制器：`mappo_frozen_state_actor_atomic_joint_execution`
- 日志：`human-study-log.v30`

PyTorch checkpoint 位于 `output/deployment/warehouse_mappo_v68_6x7.pt`。Render 使用从该 checkpoint 精确导出的 `output/deployment/warehouse_mappo_v68_6x7_actor.npz`，以 NumPy 执行同一神经网络；测试逐 logit 比对两种运行时，允许误差不超过 `1e-4`。两个 Actor 只读取同一个冻结决策前状态；同格、换位和通道冲突由一次联合审计原子解析。v68 使用无四向十字的 6×7 错位通道、三格机器人出口、2–4步瓶颈预约、载货优先和经反事实验证的精简解释。正式验收分别覆盖 AI–AI 与 Human–AI 的100个固定种子和100个随机种子；完整指标见 `output/deployment/warehouse_mappo_v68_6x7_acceptance.json`。

共享 Actor 内部包含五类神经任务意图（两个任务槽、交付、充电、等待）。任务所有权仍只在到达 A 点后产生；持久目标仅锁定跨帧规划意图。冻结状态导出的安全/联合计划掩码直接进入 Actor 的 masked logits，不在 Actor 输出后重写动作。离线关键状态覆盖充电离站、任务连续未认领、两机器人同目标、狭窄通道避让和碰撞后恢复；评估 rollout、参考轨迹和 UI 使用相同执行路径。

成功移动固定消耗 2 点电量；撞墙、货架阻塞、机器人冲突和普通等待不耗电；在充电站等待仍恢复 10 点。错位地图的正式配置使用四步安全余量，避免两台低电量机器人排队时以 0 电量到站。新增的可避免等待和任务成本回退项仅用于训练，参与者最终得分不包含任何训练塑形。

浏览器使用连续插值动画，在同一动画帧插值两个机器人的联合移动结果。地图机器人图标直接显示电量和承运货物。AI–AI 轨迹只用于最开始的操作演示；Task 1 的 A 组即时解释直接读取参与者刚产生的真实 Human–AI 时间线，Task 2 与 B 组均不显示提问面板。

即时提问先锚定最近动作、等待或碰撞事件，再读取该事件及之前最多四步；不得读取未来帧。语言无关的结构化证据同时渲染英文和中文，切换语言不推进环境、不重置状态也不丢失回答。回答只保留与动作、电量、队友、分工或碰撞问题直接相关的事实，公共载荷继续隐藏策略目标、神经分布和内部字段。

## 环境安装

推荐 Windows 10/11 和 Python 3.11。以下命令块均使用 `bash` 标记，每格只有一个命令，可由 PyCharm 显示左侧三角运行按钮。

```bash
py -3.11 -m pip install torch numpy matplotlib scikit-learn joblib transformers sentencepiece pytest
```

## 训练与评估

运行隔离目录中的短训练测试：

```bash
py -3.11 train_rl.py --smoke-test --device cuda --use-rcpd
```

从零运行推荐的 2,800 回合 MAPPO + RCPD 正式训练：

```bash
py -3.11 train_rl.py --use-rcpd
```

如果一个从零训练出的同版本候选只在正式门槛中暴露稀有 Actor 错误，可运行隔离的 1,000 回合续训入口；它保留源候选并写入新的候选目录：

```bash
py -3.11 continue_train_rl.py
```

重新执行 AI–AI、20% 噪声队友和随机策略各 200 个 seed 的评估：

```bash
py -3.11 evaluate_rl.py
```

训练前执行离线教师 200-seed 门控，并运行 reward/teacher 四组消融：

```bash
py -3.11 evaluate_teacher.py --episodes 200 --seed-start 15000
py -3.11 evaluate_reward_teacher_ablation.py --seeds 41,42,43 --episodes 200 --eval-episodes 20 --device cuda
```

完整公式、已运行的 30 回合 quick proxy 和正式可复现命令见 `docs/reward_ablation_v19.md`。quick proxy 只验证管线，不作为策略改进结论。

`train_rl.py`、`continue_train_rl.py` 和 `evaluate_rl.py` 都有 `main()` 入口，可直接使用 PyCharm 左侧绿色三角运行。项目解释器必须选择安装了 CUDA 版 PyTorch 的 Python 3.11。

既有正式指标作为只读基线保留。v26 新增神经任务意图准确率、充电离站返回循环、任务连续 40 步无人认领、直接 Actor 动作来源、队友上下文和 seed=42027 五帧绕路回归门槛；在全部新门槛为 true 前，不在此处发布 v26 指标，也不切换正式启动默认模型。

训练摘要会写入保守的 `warehouse-training-seed-ledger.v1` 区间清单，正式评估只有在全部评估区间与训练、课程、重标记、参考轨迹、旧失败候选诊断和后验蒸馏区间不重叠时才通过独立性门槛。RCPD 文件必须声明 `posthoc_explanation_only`、禁止反馈和运行时控制；参考轨迹与平行 seed 库必须声明 Actor 直接执行且干预次数为 0。只有这些产物契约和行为门槛全部通过，`run.py` 才接受该候选。

## 启动用户实验

日常界面测试使用开发入口。`run.py` 默认启用条件选择器、默认选择 A 组，并写入独立的 `development` namespace：

```bash
py -3.11 run.py
```

正式 pilot/confirmatory 必须使用独立入口，该入口不显示条件选择器，并使用服务器区组分配：

```bash
py -3.11 run_formal_ui.py
```

本地启动与 Render 相同的公开预览（服务器默认加载上述当前 NumPy Actor）：

```bash
python -m ui.development_preview_server --host 0.0.0.0 --port 8000
```

正式 RCPD/语言模型解释入口只能在另行生成并验证同版本的后验 RCPD 制品后，再通过 `run.py --checkpoint ... --program ... --transformer-model ...` 启动。当前 Render 公开预览不伪造该制品，仅展示可复现的 Actor 轨迹与确定性证据解释。

浏览器端每次方向键、WASD、方向按钮或“等待”输入只推进一个联合步。服务器使用 operation ID 和状态版本保证幂等、恢复和防重复。

免费 Render 部署使用 `render.yaml` 启动公共实验服务，并直接加载同一 checkpoint 导出的 NumPy Actor：

```text
https://policylens-warehouse-study.onrender.com/
```

## 数据与分析

SQLite 是实验日志的唯一运行时事实源。pilot 和 confirmatory 必须使用独立 namespace；旧数据保持只读。

导出 pilot 数据：

```bash
py -3.11 -m ui.study_data --db output/collaborative/safe_mission_v26_neural_mission_intent/collaborative_study.sqlite3 --namespace pilot output/collaborative/safe_mission_v26_neural_mission_intent/pilot_events.jsonl
```

分析 `ΔScore = Task2 − Task1`、区组置换检验和 bootstrap 95% 置信区间：

```bash
py -3.11 evaluate_trace_ablation.py analyze-study --input output/collaborative/safe_mission_v26_neural_mission_intent/pilot_events.jsonl --study-phase pilot --output output/collaborative/safe_mission_v26_neural_mission_intent/pilot_analysis.json
```

## 测试

```bash
py -3.11 -m pytest -q
```

测试覆盖地图连通性、拓扑死端端点排除、共享任务、计分、充电/断电、满电离站、三类碰撞、紧急充电禁止反向、载货交付承诺、必要单步让行、Reward 分项、不可刷分循环、任务补充隔离、代理人类 Actor mask、版本拒绝、RCPD 证据、121 帧本地 AI–AI 解释轨迹、实验状态机和 Web 契约。

## 代码架构

模块职责、允许的依赖方向、兼容入口、产物保留规则和迁移说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。持久化版本统一定义在 `env/warehouse/contracts.py`，正式产物路径统一由 `backend/artifacts.py` 生成；业务代码不得自行拼接版本字符串或使用浮动的 `current` 目录。

## 开发条件测试

在 PyCharm 中直接运行 `run_test_ui.py`（文件左侧三角按钮），登记页会显示“自动区组分配／解释组／对照组”选择器。该模式按运行次数而不是参与者编号推进自动分配，并将记录写入独立的 `development` namespace，不得用于正式实验。

```bash
py -3.11 run_test_ui.py
```

正式 pilot 或 confirmatory 继续使用不带测试选择器的 `run_formal_ui.py`；服务器会拒绝强制条件请求，并保持同一正式参与者的分组不变。

```bash
py -3.11 run_formal_ui.py
```
