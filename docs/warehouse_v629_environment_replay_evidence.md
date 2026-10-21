# v6.29 完整环境与原始训练奖励重放

## Material Passport

- Mode: run / validate；Date: 2026-10-21。
- Scope: dormant Stage C collector/recovery integration。
- Verification: same-agent tests on explicit synthetic states；不是科学候选、独立最终审计或留出评估。

## 解决的缺口

原公开轨迹reader可以重算NN输入、合法掩码、采样动作、critic与完整停机标签，
但公开快照刻意不包含奖励账本、任务历史、内部目标及环境RNG，不能据其证明原环境奖励。

新增 `backend/training/warehouse_v629_environment_replay.py`：

1. 显式、白名单、数据类型化保存完整环境config、state、RNG和episode counter。
   保留tuple/list及所有dataclass字段；不使用pickle、动态导入或对象执行钩子。
2. 用`__new__`和已记录的RNG状态建立重放对象，不调用环境构造器、reset、seed或set_state。
   后者会重新生成目标账本，故不能用于精确恢复。
3. 按实际八环境向量顺序，从记录的初始/跨批状态重放已发生的动作；逐帧比较
   原环境训练奖励、动作提议、无覆盖列表、公开前后状态、停机和终止标志。
   对每个slot比较完整末态和RNG，包含正好结束的episode。
4. 核对固定训练reward/物理配置、finite inventory成员资格、role、slot和episode消费。
   存储的数据不会进入Actor：仍由原PublicRolloutReplay从公开快照重建NN输入/动作。
   两条核对都通过后才交付PPO batch；期间源码、文件或模型变化也会拒绝。
5. checkpoint同时携带完整环境和公开pending成本历史；恢复绑定完整TrainingState、
   owner、finite inventory、原始资产和源码。恢复后的私有末态必须与原末态资产一致，
   不能只对上公开位置而丢掉RNG或改写隐藏账本。

该模块重放**同一环境实现**，证明记录与源绑定实现的一致性；不声称提供独立物理引擎证明。
已有RNG状态中的随机变化是原记录事件的重放，不创建新场景流。

## 验证结果

20项测试通过，50.29秒：

- 明确构造的合成状态，全程不reset；零参考Actor只输出WAIT，非实际v6.20候选。
- 两批各1024联合帧；完整120步episode及跨批pending历史正确连接。
- 奖励篡改、post-state变化、动作覆盖、未知case、缺失初态、slot错序、reward配置漂移、
  末态RNG变化均拒绝；失败后reader永久关闭，没有PPO更新。
- 完整状态/RNG保存恢复后，下一批返回的训练数组和审计结果逐位一致。
- 单独覆盖swap碰撞、真实电量停机、配送后随机补充订单及其RNG消耗。
- 已重算外层checksum仍不能掩盖恢复私有RNG与原末态资产不一致。
- 未知类型、不完整dataclass、NaN/非法RNG拒绝；NN重放期间替换资产也拒绝交付batch。

证据：`experiments/warehouse/evidence/warehouse-v629-environment-replay-implementation-01.json`。

## 准入与当前效果

finite inventory只能证明已登记成员资格，不能证明reset授权、新鲜性或初态确由声明seed产生。
未来训练owner必须先通过完整A/B，完成自身源码封存、四历史根精确种子审计、单次授权，
并核对实际reset日志与env初态/RNG的来源。本模块没有这些权限，返回training_authorized=false。
它也没有实现完整owner、在线采集/程序拟合调度或发布流程。

本轮没有新的候选环境轨迹、程序拟合/证明LP、Actor更新或PPO联合步。
原v6.28三作业仍在运行，v6.29继续排队；两者科学源码闭包保持原值。
最近完成的外层开发观察仍是第1折角色1解释覆盖40.77%，距45%还差4.23pp；
原作业科学预算停止，未获完整准入。新增工程核对不计作模型效果提升。

完整目标仍包括程序对照/新确认通过、20k配对校准、三个开发随机流预算选择、正式质量选择、
一次程序审计及各task100新种子留出、Pong/Kitchen已有证据及三域执行/绑定/工程发布核验。
NN动作来源与门槛不变；pilot关闭。
