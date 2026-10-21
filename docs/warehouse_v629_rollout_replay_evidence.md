# v6.29 公开轨迹到配对PPO的独立核对接口

## Material Passport

- Mode: run / validate；Type: training interface preparation。
- Date: 2026-10-21。
- Scope: public-frame assembly and pending full-episode cost history。
- Status: synthetic integration verified; no scientific collection/training authority。

## 实现

`backend/training/warehouse_v629_rollout_replay.py` 独立读取owner绑定的训练轨迹：

1. 检查固定8环境向量顺序、128向量步和4000暖身／提取／最终预算边界。
   case、slot和完整公开状态跨帧、跨批连接；新episode从0开始，ID不复用。
2. 使用当前完整冻结Actor与公开状态重建观察、173维特征、物理掩码、五动作分布。
   按既有episode/frame/role动作流重新采样，逐角色核对提议和post-state的提交动作。
   这只是独立重放，不向环境输出动作，也不让程序产生执行动作。
3. 采样时Actor和critic哈希、更新计数绑定原轨迹；critic基线和next value由实际
   当前critic重算。GAE按时间×8环境×2角色组装，不跨episode递推。
4. 未完成episode不作安全负例。跨PPO批次保留第0帧以来的动作、行为log probability、
   原cost baseline和终止标志；真实停机或完整120步结束时才交给已有完整cost目标函数。
5. trace、Actor、owner资产在整批前后复核SHA。失败永久关闭该次reader，不把部分结果
   交给数值更新核心，不静默重试。

## 恢复检查与复核修复

只比较joint_steps会允许接入同counter的其他训练流历史。已改为同时绑定：

- 独立owner文件指针，trace与恢复调用必须显式匹配；
- 全部model tensor和buffer、trainability、两个optimizer状态；
- reference NN SHA、初始化seed、β、joint/Actor/critic计数；
- reader源码、原始轨迹SHA及完整pending/episode消费记录。

状态fingerprint的字典有长度及结束符，tensor保留真实shape、dtype和逐位数据。
只有已加载完整TrainingState checkpoint且fingerprint完全相同才可恢复。
owner仍需另外绑定环境状态、程序、当前证据、独立RNG、capability及安全暂停账本。
本reader快照不授予中断作业重试或新采集权限。

## 已验证的具体范围

合成测试不创建环境、不reset、不更新候选Actor。

- 完整1024帧向量批次：8个120步完整episode、8个各8帧pending。
- 第二批完成原pending后，其前8帧仍使用上批critic baseline，后续帧来自更新后的critic。
  短episode真实合成停机得到正成本；部分episode始终不被记作零停机。
- 动作改选、override、post-state动作不符、断链、slot错序、非训练分区、伪终止、
  Actor/critic版本不符及任意缩短批次均拒绝。
- fresh state、同counter混Actor／critic／β／optimizer／初始化seed／owner拒绝恢复；
  并发替换Actor或owner文件也会在交付批次前拒绝。
- 接入实际PPOUpdateCore，两批分别完成8局，仅累计16局风险事件；合成critic接受32次
  Adam，暖身Actor及程序更新均0。第一批Adam后保存完整checkpoint，恢复后下一批组装逐位一致。

## 尚未授权与完整目标

这不是完整Stage C发行器或采集器。公开快照不能证明scenario reset的历史新鲜性、
训练capability，也不能独立推导所有原环境奖励；未来owner必须核验有限case/reset账本、
源闭包、实际step返回奖励和物理配置，不能通过填写partition字段代替这些证据。

v6.29必须先完成24项程序对照并通过固定新确认和压力集，才能启动20k配对校准。
之后仍须三个继续训练流的200k／300k／400k开发预算选择、正式质量选择、一次程序审计、
各task100新种子留出及三域执行／绑定／发布核验。没有提前消费新确认或封存种子。
原v6.28/v6.29科学源码闭包不变，模型效果与工程接口测试分别报告，pilot继续关闭。
