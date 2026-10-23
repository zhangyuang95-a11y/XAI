# v642 Pong／Kitchen 既有模型迁移

## 运行接口

新增 `runtime_v642/legacy_domain.py` 的 `PongAdapter(catalog)`、
`KitchenAdapter(catalog)`，与 Warehouse 使用相同的显式 DI 方式：

```python
state = adapter.recorded_state(native_snapshot, binding, provenance=source)
decision = adapter.decide(state, binding)
next_state = adapter.step(state, human_command, binding, decision=decision)
adapter.restore(next_state, binding)
demo = adapter.demonstration(binding, state=state, steps=3)
adapter.replay(demo, binding)
adapter.counterfactual(state, human_commands, binding)
adapter.explanation(state, bound_domain)
```

`initial_state(seed, task, binding)` 是后续正式入口的显式场景入口；软件回归测试使用
已归档快照。Pong 场景参数从 catalog 中固定 SHA 的配置加载，Kitchen 只接受固定
配置内已保存的完整菜单，不回退旧全局配置或临时随机菜单。

Pong 直接读指定 JSON 的两套实际层权重，执行原 `logits_from_payload`／
`choose_from_payload`，然后调用接受外部双动作的原 `step_pair` 物理入口。
Kitchen 显式检查 NPZ 文件、双角色权重、版本、每层形状与有限性，然后复用原
`FrozenActors.logits/choose`。不会调用原构造器内用于维数探测的隐式
`initial_state(1000, 1)`。物理步骤只接收当前 NN 明确选择的 AI action／slot／order。

状态封装完整原物理快照、快照 SHA、frame、完整不可变 PolicyBinding（含验收收据）、
来源与状态 SHA。所有入口重查 catalog／模型／底层资产；恢复、回放、反事实保持
同一绑定。两名 Actor 每帧均前向，真人仅替换 human 角色。没有任务分配、避让或
救场控制器，没有旧 module registry 的 monkeypatch。

## 历史验收范围明确保留

原有 Pong／Kitchen 已验收权重和程序保持原字节，原收据 SHA 固定为：

| 域 | 原收据 SHA |
|---|---|
| Pong | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` |
| Kitchen | `012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a` |

新 `ProgramBinding.audit_scope` 区分 `independent_final_program_audit` 与
`carried_selected_actor_development_audit`。Warehouse 只接受前者；Pong／Kitchen
历史程序只能使用后者。后者的 audit SHA 指向固定原验收收据，**不是新独立审计**。
现有资产继续按各域原协议核验：Pong AI 一致率≥95%、KL≤0.10，human 一致率≥75%、
KL≤0.60；Kitchen 两角色一致率≥80%、KL≤0.35。没有修改原阈值或原收据含义。

解释仅限 A 组 Task 2，且程序动作必须等于当前绑定 NN 动作。证据显式标明
`existing_v5_selected_actor_development_audit`，不推断唯一动机。B／C 不开放解释。
Kitchen 中立教学练习只有独立标记接口 `neutral_practice_metadata()`，明确它是
教学例外、不是双 Actor 模型评估；本层没有创建假练习完成或报名成功流程。

## 新的真实迁移准入 reader

`runtime_v642/legacy_acceptance.py` 内置两个明确的 schema：

- `runtime-v642-carried-pong-revalidation.v1`
- `runtime-v642-carried-kitchen-revalidation.v1`

这些分支不强制 Pong／Kitchen补造 Warehouse 的新 BC／三开发流／预算锁阶段。
它们读取并重新核对原收据中三配对训练、相同初始化及场景流、200k 步、raw 零反馈、
regularized 双角色正反馈、原留出数字与逐 task 门槛、原程序资格及所有内容哈希。
Kitchen 的原一次性程序与留出收据、原训练报告也必须完整存在且哈希一致。
重新核对数字不执行原留出集。

新的 revalidation JSON 至少包含：

```text
version / stage=carried_v5_release_revalidation / passed=true
release_id / domain / original_acceptance_sha256 / runtime_config_sha256
policy_identity_sha256={raw: ..., regularized: ...}
source_files=runtime_source_files(domain)
heldout_rerun=false / new_training=false / new_human_results=false
execution_cases_sha256=[...]
```

`passed=true` 只是必要字段。reader 不相信这个标志本身，会实际加载对应 NN 并重放
case。必须覆盖 raw／regularized × 三任务 × 双 NN／真人接管，共至少 12 项，每项
1–12 步。case 包含完整原快照、明确真人命令（NN 演示为 null）、逐帧期望原状态和
来源。每项重新核对恢复、双 NN 选择、物理步骤、回放，以及相应演示或反事实。
任何状态、动作、资产、源码或结果不一致都拒绝。

允许来源只有固定 SHA 的旧 QA 完整快照、Kitchen 已保存开发菜单（1000–1023），
或确实列在 Pong 原收据开发记录中的历史场景重建。Pong 890000–890099、Kitchen
50000–50099 留出场景明确禁止用于迁移重放。不接受新随机场景冒充已花费来源。
历史场景重建使用原确定性规则，没有新增训练或性能成绩。

未知 schema、缺少来源／哈希、失败原收据、缺失 case 或只有占位 passed 的收据均
保持关闭。三域全部真实准入仍是 pilot 的必要条件。本层不发布或启用任何 release。

## 软件验证

回归测试使用两域各 4 个既有 QA 完整快照及各自 raw／regularized 真实模型，比较
双角色 NN 动作与原纯物理结果；额外覆盖 demo、真人接管、恢复、回放、反事实、
程序一致性门、状态和资产篡改、A/B/C 隔离。完整 reader 正向测试仅使用上述已保存
菜单或原收据明确列出的开发场景，重放 12 条路径；Warehouse 仍未通过的组合不能
开放 pilot。测试不执行任何原留出集，也不输出真人协作效果。
