# Warehouse v642 显式运行适配层

## 当前范围

`runtime_v642/warehouse.py` 实现 `WarehouseAdapter(catalog)`，依赖显式的
`PolicyCatalog` 和不可变 `PolicyBinding`。它没有改动旧 runtime、407 个训练源文件、
恢复执行闭包或全局 registry，也没有启动服务、报名流程或 pilot。

正式 v642 验收 reader 尚未实现；未知验收 schema 默认拒绝 pilot。软件测试通过
不代表模型正式验收通过。Pong／Kitchen 的历史验收迁移仍需要独立的版本化 reader，
不能要求其伪造 Warehouse 新训练阶段。

## 接口

所有动作入口必须传入 catalog 登记的同一个完整 `PolicyBinding`：

```python
adapter = WarehouseAdapter(catalog)
state = adapter.initial_state(seed, task, binding)
decision = adapter.decide(state, binding)
state = adapter.step(state, human_command, binding, decision=decision)
```

`initial_state` 是唯一进行新场景 `reset` 的入口。本轮测试通过自动 fixture 禁止
`WarehouseMultiAgentEnv.__init__` 和 `reset`，没有调用此入口。新运行场景的报名、
随机分组、配额与封存授权必须由后续 Store／Server 层提供，本模块不创建这些权限。

恢复与旧数据软件检查使用 `recorded_state(snapshot, binding, task=...,
scenario_seed=..., action_seed=..., provenance=...)`。输入是完整已记录环境、
RNG 和 episode counter；该方法不调用构造器／reset，也不声称来源是新独立数据。

`restore(state, binding)` 校验完整 binding（含验收收据 SHA）、身份 SHA、状态摘要、
物理配置、frame、terminal、RNG 快照及由快照重新计算的公开行动历史。冻结 NPZ
经内容寻址 resolver 装入，检查文件 SHA 与两个实际角色权重 SHA；每次复用缓存前
仍检查资产完整性。NPZ 中的 NumPy 权重数组标为不可写。

`decide` 运行两套冻结 NN 的完整前向及环境物理掩码，使用已绑定的动作随机流。
`step` 只把 robot_1 的 NN 建议替换为用户的明确命令。AI 的 robot_2 建议、送入环境
的动作及环境记录的控制器动作必须相同。环境碰撞、墙、计分与电量规则继续处理
物理结果。环境将真人替换记为 robot_1 override；适配层只允许这一项，并要求
AI override 严格为零。

## 演示、回放与反事实

- `demonstration(binding, state=recorded_state, steps=N)` 两个角色均使用 NN。
  未来实际 runtime 也可显式传 `seed`，但不可同时传 `state`；本轮没有消费新 seed。
- `replay(bundle, binding)` 重新执行每帧 NN 和物理步骤，完整比较决定、动作、奖励、
  公开状态、下一环境与 RNG 状态；不匹配即拒绝。
- `counterfactual(state, human_commands, binding)` 从同一个已记录快照复制，最多执行
  12 步明确的真人命令；AI 每帧重新调用同一个冻结 NN。返回可独立回放的 bundle。
  这些是假设条件下的模拟，不是模型动机推断或新的任务质量证据。

`public_state` 不输出私有环境快照、RNG 或动作随机种子。

## 解释

`explanation(state, bound_domain)` 必须使用同一个 registry／adapter／catalog 的
`BoundDomain`。只有 A 组 Task 2 能读取程序；B／C、缺独立审计、其他任务均不返回
解释。pilot handle 还必须实际通过三域正式准入。通过 gate 后仅调用 portable reader
的 `evidence`，将当前 NN 实际采样动作、173 维公开特征、物理掩码及当前角色 SHA
同时传入。程序首选动作与当前 NN 动作不一致或当前叶未获审计资格时不返回证据。
没有程序选动作接口，也不从 NN 输出推断唯一的决策动机。

## 本轮软件证据

测试读取已消费的 calibration-02、017824 批次原始物理记录和各版本真实 NPZ，
每版重放同一已消费 episode 的 65 帧，共 130 帧，包含碰撞。逐帧 NN 双角色动作、
掩码、环境结果、奖励及公开历史与原记录完全相同。额外覆盖真人接管、撞墙物理
处理、NN–NN 演示、恢复、回放、反事实、状态／决定／文件篡改、丢失模型、实际
角色 SHA 不符、A/B/C 解释隔离与缺审计拒绝解释。

已有科学数据缺失时测试跳过并明确标注，不生成替代场景。这些记录属于已消费
开发数据，不是新独立验收，也没有重跑封存集或留出集。
