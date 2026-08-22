# 项目架构与维护契约

## 依赖方向

项目采用单向依赖，而不是让环境、训练、解释和界面互相调用：

```text
core
  ↑
env/warehouse
  ↑
backend/adapters ──→ backend/nlp + backend/simulation
  ↑                         ↑
backend/training            │
  ↑                         │
ui ─────────────────────────┘
  ↑
root entrypoints
```

- `core/`：环境无关的策略输出契约、可执行程序模型与后验 RCPD 蒸馏。程序表示、配置、树工具和蒸馏流程分别位于 `program.py`、`rcpd_config.py`、`rcpd_tree.py` 与 `rcpd.py`；正式仓库训练不会把程序输出反馈给 Actor。
- `env/warehouse/`：领域状态、地图导航、动力学、观测、Reward、神经策略和离线训练场景；禁止依赖 `backend`、`ui` 或 `evaluation`。
- `backend/adapters/`：把仓库状态翻译为通用模拟/解释契约。协作、电量和任务上下文在 `warehouse_context.py`，适配器不再重复计算。
- `backend/nlp/`：查询 IR、解释 IR 和语言实现；不依赖具体环境。解释数据模型、消息渲染和证据编译分别位于 `explanation_model.py`、`explanation_rendering.py` 和 `explanation_ir.py`。
- `backend/simulation/`：查询编排、反事实与证据组装；`query_engine.py` 负责用例编排，纯证据函数位于 `query_evidence.py`。
- `backend/training/`：可以依赖环境、策略、适配器和持久化的应用层训练服务。
- `ui/`：`warehouse_view.py` 只做公开状态序列化；`web_session.py` 管理单会话状态机；`web_application.py` 管理共享模型、并发、会话和数据库；`web_runtime.py` 仅为旧导入提供兼容导出。
- `evaluation/`：离线评估，不参与在线生成路径；claim 验证编排与纯对齐工具分别位于 `claim_grounding.py` 和 `claim_alignment.py`。

依赖方向、循环依赖、版本字符串唯一性、默认产物目录和生产模块 2000 行上限由 `tests/test_architecture_contracts.py` 与 `tests/test_layer_boundaries.py` 自动检查。

## 仓库领域分层

- `domain.py`：`WarehouseConfig`、任务、机器人和完整状态数据结构。
- `layouts.py`：地图布局数据。
- `navigation.py`：纯地图查询、动作掩码和最短路。
- `environment.py`：联合动作解析和状态转移，并兼容导出历史公共符号。
- `observations.py`：局部观测与集中式 Critic 状态。
- `rewards.py`：用户分数增量与训练势能，二者保持隔离。
- `coordination.py`：仅供行为克隆和 Actor 失败状态重标记使用的离线教师。它可以在隔离的数据生成环境中构造监督标签与教师状态序列，但不得进入 MAPPO rollout、生产 UI、参考轨迹或正式评估；这些正式路径提交的 AI 动作只能来自 Actor。
- `policy.py`：只含网络、推理和模型 checkpoint 验证；UI 不导入训练器。
- `mappo.py`：PPO rollout、优化与离线评估。Actor 输出原样进入环境，碰撞、阻塞、耗电和任务认领仅由环境动力学解析；训练流程本身位于 `backend/training/warehouse.py`。

## 单一事实源

- 所有环境、观测、Reward、模型、checkpoint、RCPD、seed、参考轨迹和日志版本只在 `env/warehouse/contracts.py` 定义。
- 所有候选版本文件位置通过 `CollaborativeArtifactPaths` 获得。
- 运行代码不得默认读取 `output/collaborative/current/`。该目录仅作为只读历史，不是别名或发布通道。
- 用户得分公式、公开 API 和当前 checkpoint 契约由回归测试保护。

## 兼容与迁移

- `env.warehouse.environment` 继续导出配置、状态和导航符号，已有调用无需修改；新代码应直接从 `domain` 或 `navigation` 导入。
- `env.warehouse.mappo` 继续导出 `MAPPOPolicy`、配置和训练器；在线调用应改为 `from env.warehouse.policy import MAPPOPolicy`，避免加载训练依赖。
- `ui.web_runtime` 继续导出应用、会话和序列化函数；新代码应分别导入 `web_application`、`web_session` 或 `warehouse_view`。
- 训练模块从 `env.warehouse.train_rl` 迁移为 `backend.training.warehouse`。根目录 `train_rl.py` 是稳定的推荐入口。
- `core.rcpd` 继续兼容导出程序类；新代码可从 `core.program` 导入可执行程序和 trace 数据结构。
- 任何未来的不兼容持久化调整必须先提升 `contracts.py` 中相应版本，并提供明确拒绝旧产物的错误，而不是静默加载。

## 数据与清理规则

- 当前候选写入：`safe_mission_v26_neural_mission_intent`。只有 AI–AI、20% 噪声队友和随机策略各自独立 200-seed 门控、后验 RCPD 和参考轨迹全部通过后才可成为正式默认模型。
- 保留：v24 与更早模型、失败候选、训练 checkpoint、正式评估和实验 SQLite，均作为只读历史，不得被 v25 覆盖。
- 可清理：明确标记为 `smoke`、临时截图、缓存和可由测试重新生成的 probe 文件。
- 不得用清理脚本递归删除 `output/collaborative`、任一 SQLite 数据库或未知候选目录。删除前必须解析并核对每个绝对路径。

## 变更验收

每次架构或行为变更至少运行：语法编译、架构测试、完整 pytest、正式 checkpoint 加载与版本校验、API/UI 冒烟。修改浏览器交互时还需运行浏览器端到端测试。训练动力学变化才需要重新训练；仅重排模块且序列化契约不变时不得覆盖正式模型。
