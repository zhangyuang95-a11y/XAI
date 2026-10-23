# 长预算训练完成审计与冻结资产

本层仅处理新schema的development长预算结果，不能读取20k校准／恢复结果来替代200k、300k或400k。
没有运行真实训练或审计；单测仅使用临时合成收据、纯张量和模拟权限，不能称为模型通过。

## 接口

- `run(reference_directory=...,calibration_precheck_ref=...,result_ref=...,supervisor_exit_ref=...,output=...)`
  先完整审计再创建独立导出目录；输出不可发布的report指针。
- `read_assets(report_ref)`返回`report/actors/actor_role_sha256/inputs`，三种策略是reference、raw、regularized。
- `finalization_context(report_ref,reference_directory)`返回`plan/events/owners/program_sources/checkpoints/result/assets`。
  `plan`携带phase、stream和预算，供新的长预算最终程序与质量阶段绑定。

顶层report明确包含`seed_plan`，监督器可定位既定输出；实际子进程仍须完整校验准入和输入。
冷启动读者重放审计；同进程缓存只复用已经完成的审计事实，并重新核对全部输入哈希、计划与源码契约。

## 拒绝条件

真实20k校准独立重放必须先通过。长训练须由绑定源码的监督器真实`Popen.wait()`观察到exit0，
并匹配命令、PID、起止时间、单次claim、完成文件和日志。PID消失、`training_finished`标志或任意工具成功消息不能替代它。
所有训练输入只读、内容哈希一致，模型与数据缺失或变动即拒绝。

审计核对新owner源码快照、历史精确clearance、claim时限、每条journal、实际reset及有限库存顺序。
历史clearance的24小时约束启动时间；200k／300k／400k执行claim分别为固定7／10／14天，
所有训练journal事件必须处于对应claim执行期内，禁止自动续期或篡改到期时间。
raw／regularized必须从同一完整Actor、critic、两种空优化器及零步状态出发；
4k预热的完整状态必须相同，不能只凭相同初始化seed推断。

全部五个检查点须包含真实训练张量、两种优化器、RNG、beta、参数可训练状态、读者／环境／provider状态。
直接从`torch.load(weights_only=True)`重算原生全状态指纹及掩码后Actor参数哈希，不初始化训练模型或恢复RNG。
逐笔PPO收据的步数、四epoch、minibatch身份、全部物理帧归一化、预扣／接受日志和NN哈希链必须一致。
raw外部程序梯度严格为零；regularized双角色均须有实际接受的正梯度和权重变化。

实际训练轨迹用原冻结Actor与原物理环境实现重放，比较五动作掩码、实际动作、奖励、终止、计分和私有环境／RNG末态。
这不构造新环境或产生新的科学场景。程序1650源episode另行重放，核对分区、三个提取里程碑和当前NN资格。
provider每次已提交证据按原完整NN targets与原资格算法重算，验证失败锁和对应优化前NN哈希。
这可能消耗大量只读计算，不能计为新样本或新训练步。

本层不重执行优化器梯度，也不独立重建每条rollout当时的critic基线梯度；
这些数值来自来源绑定的原生PPO核心，并与真实全状态检查点和完整优化收据链核对。
不能把“训练完成审计”写成一次从头重新训练复现。

导出采用既有无RNG张量导出器：逐推理张量及hard-support掩码处理与原Actor相同，
冻结NPZ角色哈希须等于原生检查点，warmup保留原角色哈希，两个最终版本与原起点及彼此均不同。

## 仍然关闭的阶段

导出仅证明数值训练完成和资产可追溯。最终NN的独立任务保留、最终程序保真、三流预算锁定、
正式三配对、封存审计、留出测试以及三域发布核验仍各有独立门槛。
本report的budget-lock、正式训练、封存、模型验收、发布和pilot字段始终为假。
# 新长训练存储来源的强制复核

后继读者对每次75%／84%／93%重拟合以及每次实际反馈验证，都必须独立读取
`long_materialization_provenance`：核对新owner命名空间、永久CAS／APFS基准、真实硬链接和共享区间、
精确的临时文件生命周期及全部持久输入。只符合旧目标数据schema不能通过此项。

regularized完整检查点必须同时含原native provider和新显式storage callback的来源绑定快照。
两个事件链逐项对应；每个中间检查点必须是最终独立重放历史在该步的完整前缀。
raw及critic warmup不能安装反馈callback。存储证明始终不能替代完整NN目标、采样和资格重放。

上述为软件验收边界。合成链测试不表示实际长训练已经运行，也不授予模型、封存或发布通过。

