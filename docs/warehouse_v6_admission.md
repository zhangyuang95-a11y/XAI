# Warehouse v6 模型准入

当前文档说明准入流程，不代表任一候选已经通过。发布 ID 从
`policylens-three-domain-20261016.v6.0` 开始；v5.0 继续关闭并保留失败收据。

截至 2026-10-18，探索候选 12469、12470、12471 均已关闭。12471 的
完整 BC 阶段通过了 Torch／冻结 Actor 的逐动作路由一致性检查，但
四分集独立程序预检仅覆盖两角色约 8% 的帧，未达到 35% 的内部
开发门槛。详见 `warehouse_v614_stage1_12471_failure_20261017.md`。
后续仅在已关闭模型上进行的合法性保护树探针也未达到质量门槛，见
`warehouse_v615_guarded_tree_probe_20261017.md`；它未注册为新候选。
正式配对 6101–6103、独立程序审计和留出测试仍未启动；pilot 继续关闭。

当前实现候选为 `warehouse-v618-quality-selective-01`，采用完整 256 宽 NN
加零初始化可学习树支路、独立选择性程序；移除受保护叶夹紧和 PPO 隐含
树→NN 对齐。BC 结束锁定特征支持，24%/48%/72% 仅作只读诊断，75% 冻结
分裂，叶输出继续学习。详见 `warehouse_v618_quality_selective.md`。
下文旧 v6.17 结果是失败历史，不是当前候选效果。

v6.16 候选 `warehouse-v616-joint-certified-01` 在 BC 独立预检得到两角色 0% 合格覆盖，未启动 PPO。后续候选为 `warehouse-v617-risk-protected-02`，采用保护掩码与证书分离的风险分层 Actor，见 `warehouse_v617_risk_protected.md`。下表已为新候选分配未用开发种子，旧候选收据不改写。

该 v6.17 候选已在 6140 完成完整 BC 后关闭：24 局开发质量筛查平均配送 1.083、碰撞 30.042、停机 0%，未启动 PPO。全帧程序诊断覆盖为 34.72%／67.85%，一致率 100%；这不是封存审计。原 BC 采样预检沿用了碰撞去重，其覆盖不得写成全帧覆盖。关闭后 `.r3` 配方已修正独立预检为全帧并加入丢帧拒绝检查；该修正尚无新的训练结果。详见 `warehouse_v617_risk_protected.md`。

## 不可变实验边界

- 两个独立 Actor 从同一双角色 BC 权重开始训练。raw 与 regularized 使用相同架构、配对场景序列及锁定预算；正式训练两个角色的动作均来自各自 NN。raw 的程序反馈行数和梯度为零，regularized 的两角色反馈梯度必须为正。
- 程序仅拟合完整 Actor 的物理掩码后动作分布，用于训练反馈及经审计的解释；不参与执行动作选择。程序拟合、校准、叶资格、内部验证按 episode 分离。
- 独立程序审计同时报告合格叶覆盖与实际动作可用覆盖：在同一冻结 NN–NN 轨迹的动作前帧记录采样动作，审计器重算绑定 Actor 的掩码后分布。解释仅计程序动作等于该帧采样 NN 动作；反馈还须采样动作等于当帧 NN 最大概率动作。训练期实际反馈另由 `feedback_rows / feedback_window_actor_rows` 报告，不能以冻结审计覆盖率替代。
- 每个配对种子的正式程序提取另用固定、互不重叠的 50／25／25／24 个 episode 分别拟合、校准、判定叶资格、内部验证。种子列表及 SHA 写入配对配置与预算锁；候选 ID 不能改变它们。最终程序在选定 checkpoint 后重新拟合，在线程序反馈每个 PPO 更新轮次按当前 Actor 重新验证，不能把较早 Actor 的一致性证书沿用到新权重。
- 最终程序拟合的四分集各保存一份内容寻址的双角色逐帧 JSONL，记录动作前／后公开状态、两角色观察与掩码、冻结 Actor 的分布与采样动作。打包时对全部开发与正式配对完整重放物理轨迹、以固定 Actor 重新拟合程序，并比对程序哈希。运行时首次核验全部文件哈希及打包时的重拟合收据，并对最终选中的正式配对重新完整重放轨迹、叶资格和验证统计；其他配对的逐帧重放以内容寻址的打包收据为依据。此运行时路径不依赖训练用 PyTorch。
- 规范 `run_spec_sha256` 对预注册配对、预算、场景、检查点种子、显式程序拟合方法、1,220 个教师／NN 访问 BC episode 的分阶段种子表，以及碰撞 critic 同一 seed 的额外 2 次环境 reset 寻址。`training_config_sha256` 再对固定 BC 损失、优化参数、碰撞 critic、风险乘子配置及源码闭包寻址，用于 checkpoint 恢复，并由发布验收独立重算。训练源闭包 SHA 写入 checkpoint、双变体报告和配对报告。准入的所有开发与正式配对必须与锁定 SHA 相同。训练期间修改源闭包会使运行或恢复失败，不能将前后代码混合的轨迹当作完整预检。BC 收据与 checkpoint 的一致性校验不是对历史训练过程的独立执行证明。
- 预算按 200k、300k、400k 联合环境步递增尝试；仅锁定三个开发配对种子全部通过的最小预算。正式配对固定为 6101、6102、6103，同一预算。质量选配对只读 240048–240071，程序预审不能改选。
- 开发质量门槛：raw 和 regularized 均交付≥5.5、碰撞≤6、停机 episode≤5%。每个角色的程序反馈与解释叶覆盖率以及实际 NN 采样动作可用覆盖率均≥30%、一致率≥93%／95%、KL≤0.25、非法原始程序动作零。正式门槛：两版本交付≥5、碰撞≤8、停机 episode≤10%；两角色程序反馈的两种覆盖率均≥20%、一致率≥85%，AI 角色解释的两种覆盖率均≥20%、一致率≥90%，KL≤0.35、非法动作零。实际覆盖率以所有独立审计帧为分母；反馈还要求采样动作等于当前 NN 分布最大概率动作。

## 种子和收据

| 阶段 | 预注册种子 | 使用方式 |
| --- | --- | --- |
| 开发配对 | 6150–6152 | 每个预算各训练 raw、regularized |
| 开发 checkpoint | 264100–264123 | 训练内选 checkpoint |
| 开发压力集 | base 264200–264455 | 每 task 三个独立动作 RNG，共 2304 轨迹／模型 |
| 开发程序审计 | 264500–264523 | 与拟合 episode、checkpoint 和压力集分离 |
| 正式配对 | 6101–6103 | 锁定预算后运行 |
| 正式质量选择 | 240048–240071 | 只按两版本较差质量成绩选一对 |
| 程序预审 | 240072–240095 | 只审被选中的一对，不改选 |
| 一次性独立程序审计 | 240100–240123 | 通过预审后，先写 claim 再运行 |
| 一次性留出 | base 250000–250099 | 审计通过后，先写 claim；每 task 100，合计 300 episode／模型 |

每个阶段写不可覆盖的 JSON 收据。开发压力、正式质量选择和留出均保留逐 episode 原始结果、场景种子、动作 RNG 种子、冻结 Actor SHA，以及逐帧两动作的 base64 编码与公开状态 SHA 链；均值从原始结果重算。打包时，独立模块加载发布的冻结 Actor，对这些动作与真实环境转移逐帧重放，核对终局交付、碰撞和停机，再写内容寻址的 `quality_replays` 收据。线上验收验证收据、全部输入行和 Actor 哈希绑定；不在请求线程重跑 2304 轨迹／模型的开发压力集。留出轨迹的重放只作同一次已记录评估的完整性核验，不另选 checkpoint 或形成新的统计测试。一次性 claim 固定放在仓库的 `output/warehouse_v6_sealed_claims/<release-id>/`，与调用者选择的运行输出目录无关；中途崩溃或失败也视作已使用。程序审计失败会关闭该候选，不运行留出；留出失败也关闭候选。下一候选使用递增 v6.x 发布版本和全新封存种子（每增加一个 minor 版本，封存基础种子增加 100000）。不得修改或删除失败收据后重用旧封存集。

配对训练按相同联合步数、同一场景种子区块安排 raw 与 regularized；报告记录每次实际 `reset` 的种子、episode 和步数。两策略若在不同时间提前终止，实际 reset 次数可能不同，不能把配对区块说成完全相同的已访问轨迹。程序审计另外保存逐帧动作前公开状态、观察、物理掩码、NN 分布、抽样动作、动作后公开状态及内容哈希。发布核验从冻结 Actor 重算每次抽样，并沿保存帧做完整物理环境重放；摘要和逐帧数据不一致时拒绝发布。

`policy_id` 同发布版本绑定：v6.0 使用 `warehouse-{raw|regularized}-v6`，v6.1 起使用 `warehouse-{raw|regularized}-v6.1` 等对应后缀；非规范的前导零版本号会被拒绝，避免两个 release ID 指向同一封存种子。

## 命令

在仓库根目录执行，`RUN_ROOT` 指向单独保存的实验输出根目录，`CANDIDATE` 是未使用过的候选 ID。`METHOD` 必须是该候选明确实现的程序拟合方法，由开发结果决定能否进入正式阶段；不要把探索阶段填写的名称视为已注册发布方法。119 v2 的 `actor_sparse_topology_v1` 四分程序开发预检覆盖率未达门槛，因此不能凭该预检锁定正式训练。当前 pilot 保持关闭。

```bash
export RUN_ROOT=/absolute/path/to/warehouse-v6-runs
export CANDIDATE=warehouse-v6-candidate-03
export RELEASE_ID=policylens-three-domain-20261016.v6.0
export METHOD=replace-with-implemented-candidate-method

python3 -m experiments.warehouse.admission develop \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
python3 -m experiments.warehouse.admission formal-train \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
python3 -m experiments.warehouse.admission preaudit \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
python3 -m experiments.warehouse.admission final-audit \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
python3 -m experiments.warehouse.admission heldout \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
python3 -m experiments.warehouse.admission acceptance \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID" --program-fit-method "$METHOD"
```

`budget.lock` 只会记录全部开发门槛通过的候选方法；后续每阶段都核对这一锁定方法及 BC 配方。`formal-train` 缺少通过的预算锁会拒绝；`preaudit` 缺少正式质量选择会拒绝；`final-audit` 缺少通过的预审会拒绝；`heldout` 缺少通过的最终审计会拒绝。`acceptance` 会复核全部训练、阶段收据、封存 claim、资产哈希、程序证据以及留出原始行；只有核验成功才写 `passed=true`。该收据本身不会开放 pilot。发布者还须核验 Pong、Kitchen 已有收据和迁移后的逐帧执行一致性，生成完整 v6 内容寻址发布包后才能开放研究。

失败候选及探索运行均保留其原始输出。曾在 6110 上跨训练代码变更的 `candidate-01` 是探索诊断，不计入锁定预检。`candidate-02` 在 6110、200k 步上完成配对训练但两版本质量未过门槛；它的开发选择与程序审计种子（260300–260323、260400–260423）已使用。失败收据和探索性数值诊断分别保存在 `experiments/warehouse/evidence/warehouse-v6-candidate-02.failure.json` 与 `experiments/warehouse/evidence/warehouse-v6-candidate-02.diagnosis.md`。新候选使用上表互不重叠的开发种子；旧收据中的种子不得改写。

## 训练器目录

新 Warehouse 研究只从 `experiments/warehouse/admission.py` 调用 `backend/training/warehouse_v6_runner.py`。旧 v5、v5.4 的九个实现（双 Actor 训练器、纯 NN 训练器、稀疏／选择性程序、结构化 Actor、开发 gate 和变体 runner）存放在 `backend/training/legacy/warehouse_v5/`。原模块名保留薄层供旧文档、检查点重现、测试和历史运行时读取；这些薄层不会进入 v6 准入的 Actor 动作链。

正式服务依赖 `requirements-render.txt`，不安装 PyTorch。v6 冻结 Actor 的 NumPy 推理留在 `backend/training/warehouse_v6_actor.py`，Torch 训练定义单独放在 `warehouse_v6_actor_training.py`；程序的 Torch 拟合只在训练函数内部导入。旧 v5.4 结构化 Actor 的冻结推理另置于 `backend/inference/warehouse_structured_v5.py`，不修改旧训练实现。`tests/test_study_runtime_no_torch.py` 在阻断 PyTorch 导入的子进程里检查三域 manifest 和冻结 Actor 执行动作。

旧版 `tests/test_study_v5_frame_invariants.py` 的整份测试及 `tests/test_study_v5_policy_bindings.py` 的原始完整测试已归档在 `tests/legacy_task2/archived_*`。前者的临时 v5 Actor、旧 release ID 和 active gate 假设不再适用于 v6；后者仍有价值的 8 项绑定、随机分组、历史收据一致性测试继续留在原测试模块运行。新流程的逐帧 NN 动作、演示、真人接管、回放、问答和恢复检查分别由 `tests/test_study_v6_warehouse_frame_invariants.py`、`tests/test_study_v6_pong_kitchen_frames.py` 及 v6 发布包测试承担；旧数据库、帧与收据只读留存，不改写为 v6 证据。

旧 v5.4 结构化候选的 active promotion 路由断言也在 `tests/legacy_task2/archived_test_warehouse_structured_runtime_v54.py` 留存；该模块仍可验证的 5 项格式、特征 schema 和独立收据契约留在 active 测试。v4 兼容入口的回归测试使用临时合成 Actor，因而无需把已缺失的旧忽略模型文件复制进新主线。
