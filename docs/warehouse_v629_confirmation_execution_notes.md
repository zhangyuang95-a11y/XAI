# v6.29 条件确认流程：实现与当前证据

## Material Passport

- Mode: run / validate; Type: code experiment preparation.
- Date: 2026-10-21.
- Status: implemented, prerequisite winner absent, no new confirmation allocation or collection.
- Goal: v6.28基础上继续提升并通过全部门槛；当前未完成。

## 本次完成

独立后续模块实现：真实六程序winner准入、单次分配与源码快照、四历史根种子审计、
clearance与exact证据绑定、有限516次采集能力、冻结完整NN逐帧检查、公开状态独立
动作及计数重放、固定程序评分及最终原始数组重算。详情见确认plan。

修复已发现的实际风险：审计inventory和最终selection脱节、固定trace/扫描范围/排除
集合不匹配、历史seed_ledger.py模板未进入stat差异、跨24h暂停后继续reset、
后生成claim/clearance/exact证据缺少贯穿绑定，以及重复pipeline误写旧成功终态。
评分时仍检查这些后生成资产哈希；部分失败保留claim和实际日志，拒绝自动重采。

42项相关合成测试通过，4.31秒；包括清单/压力复用、错范围和模板、逐局有效期、
失败claim、重复终态、后生成证据篡改、固定程序指标与场景bootstrap。独立code_runner
复核39项通过4.34秒，未发现新的实际阻断。新增最后3项是评分时证据篡改回归。

已花费v6.28三局公开轨迹的兼容检查：360物理帧、720次双角色动作，状态重建、
173维特征、物理掩码、完整NN分布、采样动作、post-state last_action和公共任务计数
全部一致。它不是新确认数据、效果改善或验收证明。原执行时audit源码文件SHA和
verify_frame函数源码/SHA保留在原收据；随后仅改变外围准入审计，函数本身未改。

证据版本01为39项测试时的实现快照；02追加最后3项后以新SHA绑定，不覆盖01。
零新确认场景、零reset、零拟合/LP、零Actor更新、零PPO。本次实现不修改冻结的
v6.28/v6.29科学源码，两个原源码闭包仍经真实checked_plan验证。

## 模型进度与尚缺证据

原v6.28三项第0折角色1已预算停止，外层观察解释覆盖37.86964%，仍差7.13036pp；
不能用内部44.67097%组合诊断替代。角色2搜索实测内部解释覆盖42.47%，同分区父程序
38.91%，提高3.56pp，仍非独立验证。三个容量仍需全部18项结算。

监督器41972及角色2工人44651/44652/44653已通过实时ps验证；v6.29监督器43591
保持依赖等待。原v6.29登记24项累积修复/合法位组合条件对照，不因这些观察改参数。

确认入口在真实all-six winner缺失时实际拒绝。后续仍需通过固定程序新确认与压力、
20k训练校准、三开发配对预算、正式质量选择、一次封存审计和留出、三域绑定执行
及工程发布门槛。当前没有开放PPO、封存集或pilot，不声称已完成目标。

## 证据位置

- `experiments/warehouse/evidence/warehouse-v629-confirmation-implementation-02.json`
- `output/warehouse_v629/diagnostics/interface_replay_20261021_01.json`
- `docs/warehouse_v628_terminal_observations.md`
- `docs/warehouse_v629_confirmation_plan.md`
