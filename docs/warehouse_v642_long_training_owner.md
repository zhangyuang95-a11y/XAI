# v6.42 有限长预算 development 执行入口

本模块只实现后续开发配对训练的生产者。当前代码与合成测试本身不是模型验收证据，不消费真实场景。20k 校准必须先由现有独立审计器完整重放，通过任务保持和双角色程序预检，才能读取调用方的长预算计划。手写 `passed` JSON 不能授权。

## 入口与前置条件

`experiments/warehouse/v642_long_training_owner.py` 是独立版本入口；原407训练源码和六份恢复源码均不修改。新 seed-plan 模块负责有限库存与四工作树历史种子查重，本入口另行注册完整执行源码快照，再将精确查重结果绑定该快照。普通构造函数、导入、声明验证器均无执行能力。

顺序为：真实20k独立预检 → 显式新库存计划 → `--prepare-sources` → `--audit-seeds` → `--supervise`。三个阶段均要求 `--reference-directory PATH --calibration-precheck PATH --plan PATH`。CLI 在读取长计划之前重放校准准入；完整入口在单次 claim 前重验源码和24小时精确查重启动许可。24小时仅限制启动；200k、300k、400k分别使用源代码预注册的7天、10天、14天独立有限execution lease。claim固定started/expires、预算天数、原启动截止时间及automatic_renewal=false；执行中按该expiry校验，不能自动续期、延长或重签已消费claim。

程序不自动分配库存或选择新种子。库存必须按已有 seed-plan API 显式准备。`development-0/1/2` 的200k、300k、400k每个预算均从完整v6.20双Actor共同起点重新开始；不能把20k owner改预算，也不能把小预算末端当大预算起点。每个 `(phase, stream_id, budget)` 使用跨 run ID 的独占 claim，中断保留 claim，没有自动恢复或重试。

目前仅允许 development。formal 明确拒绝：必须先实现并通过独立开发预算锁 reader，再增加来源绑定的正式入口；现有符号计划或用户授权均不能跳过该顺序。现代码不会选择模型、签发预算锁、消费封存集或开放 pilot。

## 数值与证据约束

- raw/reg 从完全相同的双Actor、critic初始化种子、空优化器、优化随机流、场景序列开始；记录初始完整训练状态及两优化器哈希。跨预算还必须逐项等于旧预算独立审计restart proof中的实测完整起点；相同seed声明不能替代该检查。每个预算强制 joint step 0、非恢复状态，角色权重必须等于冻结参考。
- 保留原完整NN前向与物理掩码。两个执行动作均由NN产生；训练轨迹逐帧交给原独立物理/NN replay reader后才进入PPO。执行后动作改选、教师标签、替代停机标签均无接口。
- 两版本共用PPO、4000联合步critic warmup、真实完整episode停机成本与v6.20参考KL。程序深12、最多256叶，每节点和叶最多3个公开源，锁定已确认结构和支持。75%/84%/93%重新拟合；λ在80%—85%升至0.01。原provider每个PPO epoch重验当前NN，证据失效即停止反馈。
- 1650个program-source episode（900拟合、150校准、300资格、300验证）仅在75%采集一次，后两次在相同已观察物理状态上重标当前NN，不声称是当前策略新轨迹。它们不计联合训练步，不进入奖励或cost梯度缓冲。raw始终无程序provider和梯度；reg结束时必须有双角色实际接受的正程序梯度更新。
- 固定8槽有限场景序列，耗尽直接失败，无自动扩容。checkpoint在4000、75%、84%、93%、100%已完成优化边界保存：完整双Actor/critic/优化器/β/全局随机态、core风险与shuffle随机态、reader的完整episode cost历史、环境与行动随机态、场景游标、refit、provider、更新指针、源与claim。checkpoint不构成恢复授权。journal独立记录checkpoint前后边界及实际wall秒数，供后继工程开销门核验；不以该工程门阻断模型效果报告。
- 完成收据仅证明该次有限数值训练结束，明确 `publishable=false`、`model_acceptance_passed=false`、`budget_lock_issued=false`。任务质量、最终程序、开发预算选择、正式质量选择、一次性程序审计、留出测试、三域绑定/发布审计仍须独立实现并通过。

监督器用当前已注册CLI的确切 `--run` argv启动子进程，保存启动承诺和真实 `Popen.wait()` 退出码。只有实际exit0且完整结果绑定成功才返回完成；result存在、PID消失或日志停止均不能替代退出证据。exit0缺结果仍记录真实0和 `result_binding_error`，审核必须拒绝。启动/退出收据和日志独占且只读。

## 测试范围

新增测试仅使用临时合成计划、模型/环境/优化器替身与四空历史树。检查完整状态工厂、实际claimed context、共同初始化、预算和单次claim、完整配对驱动时序、raw零/reg正反馈、1650证据排除、完整checkpoint及真实wait语义。测试通过不表示真实候选指标通过。

## New source-bound storage callback

Before any long plan can receive a source contract, it now includes
`warehouse_v642_long_targets`, `warehouse_v642_long_evidence`, rows CAS, APFS
clone helper and their tests/documentation. Scheduled raw/regularized refits
both use the new producer. Only regularized updates install the separately
bound `LongEvidenceCallback`; the owner still holds and verifies the actual
native `CurrentNNEvidenceProvider`. The core receives the explicit composed
callable. Native NN numerics, qualification, latch and accepted-gradient
accounting are unchanged. Checkpoints retain both native and callback state.
The owner only seals its own new storage tree at completion; historical trees
and current 20k/recovery files are never retrofitted. Formal execution remains
closed until its independent authorization chain is implemented.
