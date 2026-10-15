# Warehouse v6 模型准入

当前文档说明准入流程，不代表任一候选已经通过。发布 ID 从
`policylens-three-domain-20261016.v6.0` 开始；v5.0 继续关闭并保留失败收据。

## 不可变实验边界

- 两个独立 Actor 从同一双角色 BC 权重开始训练。raw 与 regularized 使用相同架构、配对场景序列及锁定预算；正式训练两个角色的动作均来自各自 NN。raw 的程序反馈行数和梯度为零，regularized 的两角色反馈梯度必须为正。
- 程序仅拟合完整 Actor 的物理掩码后动作分布，用于训练反馈及经审计的解释；不参与执行动作选择。程序拟合、校准、叶资格、内部验证按 episode 分离。
- 训练源闭包 SHA 写入 checkpoint、双变体报告和配对报告。准入的所有开发与正式配对必须与锁定 SHA 相同。训练期间修改源闭包会使运行或恢复失败，不能将前后代码混合的轨迹当作完整预检。
- 预算按 200k、300k、400k 联合环境步递增尝试；仅锁定三个开发配对种子全部通过的最小预算。正式配对固定为 6101、6102、6103，同一预算。质量选配对只读 240048–240071，程序预审不能改选。
- 开发质量门槛：raw 和 regularized 均交付≥5.5、碰撞≤6、停机 episode≤5%。每个角色的程序反馈与解释覆盖率≥30%、一致率≥93%／95%、KL≤0.25、非法原始程序动作零。正式门槛：两版本交付≥5、碰撞≤8、停机 episode≤10%；两角色程序反馈覆盖率≥20%、一致率≥85%，AI 角色解释覆盖率≥20%、一致率≥90%，KL≤0.35，非法动作零。

## 种子和收据

| 阶段 | 预注册种子 | 使用方式 |
| --- | --- | --- |
| 开发配对 | 6120–6122 | 每个预算各训练 raw、regularized |
| 开发 checkpoint | 261100–261123 | 训练内选 checkpoint |
| 开发压力集 | base 261200–261455 | 每 task 三个独立动作 RNG，共 2304 轨迹／模型 |
| 开发程序审计 | 261500–261523 | 与拟合 episode、checkpoint 和压力集分离 |
| 正式配对 | 6101–6103 | 锁定预算后运行 |
| 正式质量选择 | 240048–240071 | 只按两版本较差质量成绩选一对 |
| 程序预审 | 240072–240095 | 只审被选中的一对，不改选 |
| 一次性独立程序审计 | 240100–240123 | 通过预审后，先写 claim 再运行 |
| 一次性留出 | base 250000–250099 | 审计通过后，先写 claim；每 task 100，合计 300 episode／模型 |

每个阶段写不可覆盖的 JSON 收据。开发压力与留出均保留逐 episode 原始结果、场景种子、哈希及从原始结果重算的均值。一次性 claim 固定放在仓库的 `output/warehouse_v6_sealed_claims/<release-id>/`，与调用者选择的运行输出目录无关；中途崩溃或失败也视作已使用。程序审计失败会关闭该候选，不运行留出；留出失败也关闭候选。下一候选使用递增 v6.x 发布版本和全新封存种子（每增加一个 minor 版本，封存基础种子增加 100000）。不得修改或删除失败收据后重用旧封存集。

`policy_id` 同发布版本绑定：v6.0 使用 `warehouse-{raw|regularized}-v6`，v6.1 起使用 `warehouse-{raw|regularized}-v6.1` 等对应后缀；非规范的前导零版本号会被拒绝，避免两个 release ID 指向同一封存种子。

## 命令

在仓库根目录执行，`RUN_ROOT` 指向单独保存的实验输出根目录，`CANDIDATE` 是未使用过的候选 ID：

```bash
export RUN_ROOT=/absolute/path/to/warehouse-v6-runs
export CANDIDATE=warehouse-v6-candidate-03
export RELEASE_ID=policylens-three-domain-20261016.v6.0

python3 -m experiments.warehouse.admission develop \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
python3 -m experiments.warehouse.admission formal-train \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
python3 -m experiments.warehouse.admission preaudit \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
python3 -m experiments.warehouse.admission final-audit \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
python3 -m experiments.warehouse.admission heldout \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
python3 -m experiments.warehouse.admission acceptance \
  --output-root "$RUN_ROOT" --candidate-id "$CANDIDATE" --release-id "$RELEASE_ID"
```

`formal-train` 缺少通过的预算锁会拒绝；`preaudit` 缺少正式质量选择会拒绝；`final-audit` 缺少通过的预审会拒绝；`heldout` 缺少通过的最终审计会拒绝。`acceptance` 会复核全部训练、阶段收据、封存 claim、资产哈希、程序证据以及留出原始行；只有核验成功才写 `passed=true`。该收据本身不会开放 pilot。发布者还须核验 Pong、Kitchen 已有收据和迁移后的逐帧执行一致性，生成完整 v6 内容寻址发布包后才能开放研究。

失败候选及探索运行均保留其原始输出。曾在 6110 上跨训练代码变更的 `candidate-01` 是探索诊断，不计入锁定预检。`candidate-02` 在 6110、200k 步上完成配对训练但两版本质量未过门槛；它的开发选择与程序审计种子（260300–260323、260400–260423）已使用，失败收据固定保存在 `experiments/warehouse/evidence/warehouse-v6-candidate-02.failure.json`。新候选使用上表互不重叠的开发种子；旧收据中的种子不得改写。

## 训练器目录

新 Warehouse 研究只从 `experiments/warehouse/admission.py` 调用 `backend/training/warehouse_v6_runner.py`。旧 v5、v5.4 的九个实现（双 Actor 训练器、纯 NN 训练器、稀疏／选择性程序、结构化 Actor、开发 gate 和变体 runner）存放在 `backend/training/legacy/warehouse_v5/`。原模块名保留薄层供旧文档、检查点重现、测试和历史运行时读取；这些薄层不会进入 v6 准入的 Actor 动作链。

旧版 `tests/test_study_v5_frame_invariants.py` 的整份测试及 `tests/test_study_v5_policy_bindings.py` 的原始完整测试已归档在 `tests/legacy_task2/archived_*`。前者的临时 v5 Actor、旧 release ID 和 active gate 假设不再适用于 v6；后者仍有价值的 8 项绑定、随机分组、历史收据一致性测试继续留在原测试模块运行。新流程的逐帧 NN 动作、演示、真人接管、回放、问答和恢复检查分别由 `tests/test_study_v6_warehouse_frame_invariants.py`、`tests/test_study_v6_pong_kitchen_frames.py` 及 v6 发布包测试承担；旧数据库、帧与收据只读留存，不改写为 v6 证据。

旧 v5.4 结构化候选的 active promotion 路由断言也在 `tests/legacy_task2/archived_test_warehouse_structured_runtime_v54.py` 留存；该模块仍可验证的 5 项格式、特征 schema 和独立收据契约留在 active 测试。v4 兼容入口的回归测试使用临时合成 Actor，因而无需把已缺失的旧忽略模型文件复制进新主线。
