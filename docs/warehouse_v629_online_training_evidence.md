# v6.29 训练前的在线保真证据与程序反馈

## Material Passport

- Mode: run / validate; Type: code experiment preparation.
- Date: 2026-10-21.
- Status: pure APIs implemented and synthetic tests passed; no candidate PPO started.
- Goal: v6.28基础上改进并通过全部模型、程序、执行和发布指标；当前未完成。

## 解决的绑定问题

原v6.28/v6.29 Program的actor_role_sha256同时参与保护父程序读取。把它直接改成
训练后NN哈希，会失去原父程序绑定；保留旧字段却继续声称对当前NN保真，也不成立。

新增独立、显式版本化online evidence：原Program及其父保护契约保持原字节；
另存当前完整NN资产／角色哈希、当前optimizer epoch、训练程序证据池及重新计算的
叶资格和独立验证指标。读取时重新验证原NN目标分布、动作随机流、资格与门槛，
不信任存储的passed布尔值。原Program reader不解释或接受这个新格式。

四个分区按episode与场景互斥；只用eligibility判定叶资格，validation仅决定反馈是否启用。
所有五动作目标都来自当前完整冻结NN经物理掩码后的分布；程序不选择执行动作。
外部接口只生成loss targets，不能调用环境。当前NN权重或epoch变化，旧证据立即拒绝。

## 实际发现并修复的数值错误

独立复核旧已花费dev-0300发现批量NN矩阵前向与原runtime逐帧向量前向存在float32
舍入差异：角色1有10/120行不同、最大2.086162567138672e-7；角色2有14/120行、
最大1.1920928955078125e-7。不能用这些差异误拒合法runtime记录，也不降低原始逐位判定。

因此新runtime_probabilities保持原1D逐帧前向形状。独立重放同一两角色240行后，
与原runtime概率逐位一致、差异0。收据绑定执行时函数原文/SHA、源码、Actor、旧
plan、collection、数组与公开轨迹，不消耗新场景，不作为效果或确认通过证据。
非零随机完整NN及混合物理输入的回归保证这条路径不再偷偷改为batch forward。

## 程序反馈的纯tensor接入

- λ按80%–85%从0升到0.01；raw始终0。raw与窗口前regularized不读取程序证据。
- regularized按两个角色分别核对真实当前model角色哈希和epoch，检查不可变资产。
- q完全detach；只有当前资格合格、验证通过、程序首选＝当前NN首选＝实际采样动作
  的帧提供KL(q||NN)梯度，按全部物理帧归一。
- 记录两角色程序梯度的真实范数，位于共享clip之前；它们不被冒充成已接受优化更新。
- 共享Adam之前必须再次传入真实当前epoch与model，防止在loss生成后NN或epoch改变。
- 合成NN的真实Adam更新证明：两角色均收到正程序梯度，更新后旧证据拒绝；
  raw程序梯度严格0。该更新只发生在单元测试的临时模型，未触及科学候选。

## 测试与明确剩余工作

25项相关合成测试通过，26.64秒。独立code_runner分别复核14项online evidence
（16.62秒）和11项反馈（11.34秒），未发现新的实际阻断。这些API测试及只读回放
不启动新环境场景或科学候选拟合/LP，不更新候选Actor或执行PPO。原v6.28搜索继续
执行其已登记的拟合工作；v6.28/v6.29源码闭包保持冻结。

原搜索当前第0折角色2的内部F/E覆盖均为44.48%，同分区父程序38.91%，提升5.57pp；
预期反馈42.46%，一致98.22%，KL0.0523，非法/未证0。F/E仍各差约0.52pp。
这是fit内部180场景评分，不能替代外层验证或全六单元通过；角色1外层解释仍为37.87%。

这些API尚不构成可启动的PPO driver，也不授予采集或训练权限。后续必须先得到
六程序真实合格winner及一次新确认通过，再绑定完整episode公共审计、种子新鲜台账、
训练专用pool成员身份、PPO/critic/真实停机成本、50/50参考KL、共同初始化与配对流、
检查点和接受更新账本。pool的scope标签和本模块的行级校验不能替代这些能力。

程序更新／每epoch重新验证需要新source-bound训练流程；不得把旧确认、开发验证、
最终审计或留出数据改标成训练数据。最终准入和pilot仍关闭。

## 源码与证据

- `backend/training/warehouse_v629_online_evidence.py`
- `backend/training/warehouse_v629_training_feedback.py`
- `tests/test_warehouse_v629_online_evidence.py`
- `tests/test_warehouse_v629_training_feedback.py`
- `output/warehouse_v629/diagnostics/online_target_replay_20261021_01.json`
- `experiments/warehouse/evidence/warehouse-v629-online-training-implementation-01.json`
