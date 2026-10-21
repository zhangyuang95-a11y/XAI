# v6.29 固定程序新数据确认（条件阶段）

## Material Passport

- Mode: run; Type: code experiment; Status: implemented, not allocated or executed.
- Source: v6.28 common development pool → v6.29 cumulative repair.
- Scope: new development confirmation, not sealed audit, holdout, PPO or human study.
- Actor: original complete frozen v6.20 dual Actor; physical action masks only.

## 前置条件

原v6.29的24项全部结算，六个角色／折在同一G容量全部通过、最小合格容量锁定、
独立算术和证书审计通过后，才允许分配新种子。缺少真实winner时，allocate在任何
环境或RNG调用之前拒绝。当前没有winner，不预分配或消费新场景。

本流程是独立后续源码闭包，不修改正在运行的v6.28／排队v6.29科学源码。
源码、配方、输入Actor、六个程序、父程序、选择／审计收据全部写入SHA绑定及只读快照。

## 数据与唯一采集

- 确认：300个新场景，每task100；全部按NN分布采样。
- 压力：72个另外的新场景，每task24，每场景3个动作随机轨迹，共216局。
- 共516次显式环境reset。环境构造前，四历史仓元数据／精确参数及上下文审计通过；
  固定注册trace表、扫描根、排除集合、模板SHA与最终stat差异全部复核。
- 全部场景和动作流先查重；压力三轨迹的同场景复用显式登记。派生逐帧动作种子
  采用原冻结Actor的source-pinned SHA函数，不改NN行为，不把数值控制0声称为新随机流。
- clearance最长24小时；每局构造RNG前检查有效期、绑定资产和源码，写入fsync预授权。
  历史delta在首次claim前检查，不能表述为整个采集期间持续扫描。
- 单次candidate claim在Actor加载和环境构造前写入，绑定plan、clearance、exact证据SHA。
  中断／部分失败保留claim、实际reset日志和失败证据，不自动重采、恢复或续期。

## 动作与独立重放

每帧两个动作均由完整冻结NN的物理掩码后分布决定；独立重算抽样、比对Actor输出、
环境收到的动作、无覆盖列表及post-state last_action。程序完全不参与执行。
保存动作前后公开状态、173维公开特征、物理掩码、完整五动作分布与采样动作。
独立审计从公开状态重建特征和NN分布，不创建环境，不消耗新的场景随机流；
逐局重算配送、碰撞、真实停机及完整episode，检查516次实际reset与物理配置。

## 固定程序评分和停止条件

六程序的结构、叶系数、温度、反馈／解释资格集合保持锁定；确认不拟合、不校准、
不重新资格筛选。父程序使用同一轨迹、相同固定资格语义。
确认和压力分开，六单元各自满足F/E覆盖≥45%、预期F≥40%、一致≥95%/97%、
合格区域KL≤0.20、原始非法／未证0。任务质量两类分别要求配送≥5.5、碰撞≤6、停机≤5%。

相对父程序的两角色E及预期F四端点：先按完整episode求均值，再同场景平均三折；
压力再平均三动作轨迹。按task分层的场景配对bootstrap20000次，数值控制0；
主确认四个单侧98.75%下界均>0，含保守浮点舍入界。压力bootstrap作为次级报告，
压力的绝对门槛依然必须通过。不能把连续帧／三折／三动作轨迹当独立场景。

结果先写原始指标，再从审计过的原始数组独立重算，符合才写confirmation.passed。
失败关闭候选，不换容量复用确认集。通过仅允许下一步20k配对训练校准；
正式PPO、最终程序审计、每task100种子留出、Pong/Kitchen绑定复核、工程门槛及pilot发布
仍须后续分别完成。确认通过不等于整个研究发布通过。

## 命令顺序

入口 `scripts/run_warehouse_v629_confirmation.py`：

1. `--allocate --run-id <new-id> --base-start <new-proposal> --action-start <new-proposal>`
2. `--prepare --allocation <recorded-allocation.json>`
3. `--pipeline --plan <plan.json>`（依次查重、采集、独立采集审计、固定评分、最终重算）

各步骤也可用 `--audit-seeds`、`--collect`、`--audit-collection`、`--score`、`--finalize`
单独执行。单步执行不绕过准入、哈希或单次claim。任何实际启动必须先通过相关测试。
