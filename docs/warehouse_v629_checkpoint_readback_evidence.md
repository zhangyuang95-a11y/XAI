# v6.29 检查点回读与数值重放

2026-10-22，same-agent 工程验证。真实候选仍在程序搜索；没有启动实际配对训练。

## 已验证的范围

新增 `experiments/warehouse/v629_checkpoint_readback.py`，在真实 A/B 准入之后，
只读重建既有安全批次检查点。核对 owner、单次 claim、注册源码合同、种子计划、
参考 NN 和缓冲、已消费场景前缀及实际更新收据；不发放执行 owner 或继续训练权限。
重建完整双 Actor、Adam、critic、参考 KL 的 β、风险和本地随机状态，
恢复8个环境及未完成 episode 的 cost 历史。

测试使用模拟 A/B、临时非饱和 WAIT NN、临时场景和真实环境/PPO计算。
连续执行到5024步，保存检查点；再完成1024步至6048步。
另行回读5024步检查点，仅重放已保存的下一批物理轨迹和优化更新：

| 数值检查 | 实测 |
|---|---:|
| Actor 优化器步 | 16 → 32 |
| critic 优化器步 | 80 → 96 |
| 双 Actor 权重变化检查 | 检测到变化 |
| 更新报告、完整模型/Adam/β/风险/RNG及物理状态与连续执行 | 精确相同 |
| 回读时全局 Python/NumPy/Torch RNG | 不变 |
| 下一批重放新增环境 reset / 改写原训练文件 | 0 / 0 |

同时测试场景游标、环境状态、步数、owner、更新收据、程序来源游标和下一批前缀篡改拒绝。
regularized 的测试只核验保存的事件前缀：资格失败后持续关闭反馈，只有规定的下一次
程序重拟合可以重新开放；不声称完成了 regularized 完整数值恢复。

13项测试全部通过，143.60秒。命令、JUnit、源码及收据SHA见
`experiments/warehouse/evidence/warehouse-v629-checkpoint-readback-implementation-01.json`。
临时大轨迹不作为可恢复科学包交付；测试收据不能替代真实候选实验。

## 真实准入和限制

对真实候选调用回读入口，在任何检查点读取、模型或 RNG 创建之前拒绝：
`v629 has no locked audited winner; confirmation remains closed`。
v6.28/v6.29 科学源码闭包与已封存计划仍一致。

本接口允许只读取证既有失败单次 claim 的检查点，不会复活该候选，
不改变原 regularized provider 的事件记录。没有实现 owner 恢复授权或自动重试。
真实配对校准、任务保留评估、完整预算训练及独立封存验收仍待前置通过后执行。
程序门槛和运行时 NN 动作来源保持原约束，pilot 关闭。
