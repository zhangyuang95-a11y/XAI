# PolicyLens 三领域架构与发布契约

本文件描述当前三领域研究流程。早期 Warehouse 单领域架构说明保存在
[历史文档](docs/archive/ARCHITECTURE_pre_v6_20261016.md)。当前候选是
`policylens-three-domain-20261016.v6.0`；Warehouse 尚未通过模型验收，
因此 pilot 报名保持关闭。版本号表示候选身份，不表示通过。

## 运行边界

```text
物理环境 + 动作前公开观察
            ↓
      内容哈希绑定的冻结双 Actor
            ↓
  演示：两个 Actor 执行 / 真人任务：人接管一个角色
            ↓
       环境处理合法性、碰撞和计分

独立程序 → 训练反馈 / 通过审计后的解释证据
独立程序 ↛ 执行动作选择
```

- `env/warehouse/` 提供 Warehouse 状态、公开观察、合法动作掩码及物理转移；
  `domains/pong/` 和 `domains/kitchen/` 提供另外两个领域的环境与适配器。
- `domains/warehouse/neural_turnbased.py` 是 Warehouse 的参与者适配器。
  演示由两名冻结 Actor 行动；真人任务由人替换 `robot_1`，`robot_2`
  的动作仍由绑定 Actor 产生。恢复、回放和问答使用相同的发布绑定。
- `study_v3/domain_contracts.py` 定义三领域共同的物理环境、观察、
  冻结 Actor、程序和参与者适配接口。`study_v3/policies.py`、
  `release_catalog.py` 与 `registry.py` 验证分组、模型与发布身份；
  `store.py` 保留实例、帧与历史数据。
- `backend/training/warehouse_v6_runner.py` 是当前 Warehouse 双 Actor
  BC、PPO 和开发评估的唯一正式训练管线。模型、程序、公开教师及训练专用
  critic 分别位于同目录的 `warehouse_v6_actor*.py`、
  `warehouse_v6_program.py`、`warehouse_v6_teacher*.py` 和
  `warehouse_v6_collision_critic.py`。旧训练器保留为历史读取与兼容入口，
  不得作为 v6 正式运行的替代入口。
- `experiments/warehouse/protocol.py` 固定配对种子、场景序列、预算、
  开发/压力/预审/封存种子边界和门槛；`admission.py`、
  `acceptance.py` 核验成套收据。开发产物在按候选与 run ID 隔离的
  `output/` 中；它是 Git 忽略目录，不能整目录提交。
- `releases/<release_id>/assets/<sha256>/` 保存发布资产的内容寻址副本，
  `catalog.json` 固定双 Actor、程序、训练报告、收据与运行配置哈希。
  运行时只从通过验收的发布清单加载；开发输出不是运行时回退路径。

## 模型、程序与实验边界

每个领域的两个角色各有独立 Actor。raw 与 regularized 使用配对初始化及
相同场景序列；两组的执行动作都由各自 NN 采样并经物理合法掩码处理。
Warehouse regularized 的程序梯度只在规定训练窗口、独立验证通过且程序
动作与 NN 动作一致时施加；raw 的外部程序反馈为零。停机 cost critic 只
通过 PPO 更新权重，不在执行时接管动作。

程序拟合完整 NN 的掩码后分布。拟合、校准、叶资格和内部验证按 episode
分离；正式独立审计和留出集有一次性 claim。候选封存后若失败，该版本关闭，
下一候选递增版本并使用全新封存种子。质量、程序保真度、解释保真度和逐帧
NN 动作一致性全部通过后才能开放 pilot。Pong、Kitchen 沿用已验收的
v5 模型和收据，迁移时核对内容哈希与执行一致性，不重跑已使用的留出集。
Kitchen 的离线人类代理成绩只称为代理评估。

## 兼容与仓库维护

- `core/` 保持环境无关的策略与程序契约；`backend/inference/` 是冻结
  推理实现，研究训练依赖不进入正式运行时。
- `ui/` 和旧根目录入口提供既有调用兼容。历史 v4/v5 实例、帧、收据
  与失败候选只读保留，不改写成 v6 结果。
- `docs/git_v6_branch_inventory_20261016.md` 记录原脏工作树的可恢复快照、
  已归档分支与仍在使用的检出。不得重置协作者分支、强推或对大型实验输出
  运行 `git clean -xfd`。
- 先合入 pilot 关闭的主线与架构 PR；模型证据全通过后再单独合入 v6 发布
  PR。Render 仍引用的分支必须先核实并迁移部署引用，之后才可处理旧分支。

架构及发布回归测试位于 `tests/`，尤其是
`test_architecture_contracts.py`、`test_layer_boundaries.py`、Warehouse v6
协议/验收测试及三领域运行帧测试。架构门禁从当前服务器、训练、验收、发布
入口追踪本地依赖，对实际运行模块检查导入期循环和 2,000 行上限；历史候选
不再冒充当前入口。共享物理环境 `env/warehouse/environment.py` 已有 2,171
行技术债被冻结，不能继续增长。
