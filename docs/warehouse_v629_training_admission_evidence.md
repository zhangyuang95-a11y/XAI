# v6.29 程序确认到配对训练的前置检查

## Material Passport

- Mode: run / validate。
- Date: 2026-10-21。
- Scope: read-only prerequisites, fixed program language and existing reference-buffer readback。
- Verification: same-agent synthetic tests and deterministic readback；不是封存审计或模型通过证据。

## 已实现的实际边界

`experiments/warehouse/v629_training_admission.py` 和只读命令
`scripts/check_warehouse_v629_training_admission.py` 不分配种子、不构建环境或优化器、不更新Actor。

1. 首先读取真实A阶段锁定赢家。必须24项结算、按六单元全部通过选最小G容量且算术审计通过；
   缺失时在读取参考缓冲、分配数据或初始化训练之前拒绝。
2. 读取唯一成功B确认，核对固定六程序、完整NN、原始数组/轨迹及采集审计；
   对全部绝对门槛和四个场景级bootstrap重新计算，不把passed布尔值当作证据。
3. 重放已有历史种子审计的绑定证据。该步骤不重新扫描或授予未来训练种子新鲜性；
   Stage C仍须自己的有限计划、完整源码、精确历史审计和单次授权。
4. 从已审计的共同fit-only参考缓冲逐块核对原300局来源、顺序、掩码和episode ID。
   raw和regularized未来必须使用同一SHA；当前没有创建实际配对训练绑定。
5. 固定使用第0折的两个角色程序，规则先于B结果；不根据确认分数挑折。
   显式绑定拓扑、容量、稀疏支持、合法witness、父保护合同及参考Actor角色哈希。
   结构哈希支持64/128/256叶、深9，旧32叶读者及旧资产语义保持原状。

## 补齐传递依赖

只有父JSON的SHA不能证明它引用的文件保持不变。因此前置清单还绑定：

- A阶段全部24个result和completed seal，以及已完成单元的search report；
- A/B计划封条、源码快照、原始共同数据和声明的固定依赖；
- B allocation及其seal，exact plan、bounded/triaged inventory、可选context annotations和历史trace registry；
- 原v6.28计划/采集完成封条、全部父程序、参考缓冲及其源代码快照和300局源文件。

`TrainingReadiness.verify()` 在交付前及后续调用时检查清单SHA、所有文件SHA、
源码闭包和配方；读取过程中发生替换也会拒绝。该对象始终没有环境、优化或种子分配权限。

## 本轮验证

- 32项测试通过，16.51秒。包括64/128/256边界、B指标重算失败、伪通过/错误赢家、
  前置文件变动、24项选择证据遗漏、缓冲来源改写及长读取过程中源码/资产变化。
- 对真实已有参考包只读检查：300局、35,931联合帧、1,120个依赖指针通过；
  此结果只证明参考缓冲组件，不构成真实训练准入。
- 真实CLI返回exit 2：`v629 has no locked audited winner; confirmation remains closed`。
  相关确认/锁定标记前后一致；没有新环境reset、拟合、LP或Actor/PPO更新。
- 原科学源码闭包不变：v6.28 `1fd77d5d…e823721b`，v6.29 `4cd4de16…d04723f`。

完整数值与文件SHA见
`experiments/warehouse/evidence/warehouse-v629-training-admission-implementation-01.json`。

## 同时核实的模型效果

以下是固定检查点的外层开发观察，每行64/128/256三组相同；原作业预算停止，
跳过了最终重拟合/校准，原准入仍失败。不能把本表当成完成实验或留出结论。

| 折/角色 | 反馈覆盖 | 解释覆盖 | 预期反馈 | 解释一致 | 解释KL | 同场景父解释覆盖 | 改善 | 解释距45% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/robot_1 | 44.93% | 37.87% | 42.55% | 98.38% | 0.0538 | 36.83% | +1.04pp | 7.13pp |
| 0/robot_2 | 43.47% | 39.15% | 41.55% | 98.81% | 0.0413 | 37.18% | +1.98pp | 5.85pp |
| 1/robot_1 | 43.81% | 40.77% | 41.82% | 98.45% | 0.0471 | 37.23% | +3.54pp | 4.23pp |

上述整树原始非法动作及未证明路径均0。第1折角色1用54叶，反馈覆盖还差1.19pp。
第1折角色2的三个原作业已核实真实进程存活；v6.29继续等待原18项结算，未重启旧作业。
新增的准入实现不会改变这些程序指标，也不构成v6.29改进方法的实验结果。

## 剩余完整任务

仍须A阶段全过及一次新确认/压力集通过；之后完成Stage C有限授权、真实采集、
在线程序重拟合和安全环境/RNG恢复，再运行20k校准配对及三个200k/300k/400k开发流。
正式阶段按质量选择，固定资产后依次做一次程序审计和各task100个新种子的留出。
Pong/Kitchen旧证据、三域执行动作一致性、A/B/C哈希绑定及工程发布门槛也须通过。
当前完整目标未完成，pilot关闭；不降低门槛或通过手写动作规则补救。
