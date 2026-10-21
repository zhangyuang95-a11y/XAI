# v6.29 每次PPO更新的当前NN证据与失败关闭状态

## Material Passport

- Mode: run / validate；Date: 2026-10-21。
- Scope: dormant regularized PPO evidence provider and recovery integration。
- Verification: same-agent synthetic tests；不构成科学候选、独立模型验收或训练授权。

## 接通的行为

`backend/training/warehouse_v629_evidence_provider.py` 接入现有PPO core的
`evidence_provider(state, epoch_id)`回调：

1. 创建前先过真实A/B训练前置检查；raw不构建本provider，窗口前不允许调用。
   当前没有锁定赢家，真实入口在创建目录或导出NN前拒绝。
2. 每个反馈更新前从真实TrainingState导出当前完整NN，基于已审计的固定源状态重新生成
   明确标记的反事实目标，重新判定两个角色的叶资格和独立保真门槛。
   验证源状态、NN角色哈希和优化epoch；旧NN/旧epoch证据不得复用。
3. 任一角色保真失败即对两个角色持续关闭程序反馈。即使后续NN恢复一致，也不能自行重新开启。
   只有下一个登记的75%／84%／93%程序重拟合完成、来源绑定有效且双角色当前验证通过，才能重置关闭状态。
   不改写真实验证指标，只通过独立、显式的provider开关使可反馈掩码为空。
4. **结束在**84%／93%边界的PPO批次使用上一程序；该批更新完成后、下一批采集前安装新程序，
   与原driver时序一致。越过边界仍未安装下一程序时拒绝更新。
5. 每次安装/验证写入顺序、前项SHA、NN、程序、源包及验证报告摘要；实际接受的梯度/更新由PPO core另记。
   不把反事实可反馈率或梯度可用性写成实际已接受更新。
6. 恢复绑定完整TrainingState、源码、owner和原始事件/资产。重新推导关闭状态、程序、epoch消费和计数，
   不能通过改snapshot布尔值恢复反馈，也不能从旧snapshot静默重做之后已经执行的provider工作。

新`warehouse-v629-provider-controlled-evidence.v3`继承只提供loss targets的接口；
仍没有环境动作选择器。provider源码属于后续训练闭包，不修改原v6.28/v6.29科学运行源码。

## 验证范围

10项测试全部通过，229.94秒。

- 从存储的合成源轨迹构建固定状态池，非实际v6.20或新科学场景。
- 实际连接PPOUpdateCore完成四次合成Actor及四次critic优化更新；每次独立生成新证据，两角色均有被接受的正程序梯度。
- uniform合成NN造成真实保真失败，恢复原NN后反馈仍关闭；下一次登记重拟合才重新开启。
- 精确边界、窗口前、重复epoch、NN变化、未登记程序替换及错误历史均拒绝。
- checkpoint恢复保留失败关闭状态；改关闭布尔值、篡改NN导出或使用有后续工作的新目录/旧快照均拒绝。

这些数值更新只发生在测试临时模型；不是科学候选的PPO步数，也不证明实际实验已获得程序梯度。
完整命令、测试输出和源码SHA见`experiments/warehouse/evidence/warehouse-v629-evidence-provider-implementation-01.json`。

## 当前科学实验

原v6.28已有12/18项结算，最近已完成外层观察为第1折角色2：F43.29%、E39.41%，
同场景父E33.59%，改善5.82pp，F/E仍差1.71/5.59pp。原作业预算耗尽，准入仍失败。
第2折角色1仍在搜索；内部评分已出现F45.83%、E42.90%、预期F43.26%，E还差2.10pp。
内部评分经过搜索选择，不能当成外层验证通过，也不能证明完整六单元通过。

v6.29改进对照仍排队。PPO、独立确认/压力集、最终审计/留出和pilot没有提前启动。
仍须完成真正的有限Stage C owner/采集器及源数据隔离、20k配对校准、三个开发流预算选择、
正式质量选择、一次独立程序审计及各task100新种子留出、三域模型/运行一致性与工程发布核验。
