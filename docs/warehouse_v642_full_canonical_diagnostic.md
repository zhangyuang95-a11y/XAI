# v6.42 完整旧目标池数值核验

此工程任务只读取原 regularized program-source `265fd2ba…3453f`，及已完整提交的 native call31、47、58。没有环境/reset、NN 更新、拟合、新场景、独立审计、留出或新科学样本，也不改原407、恢复6或当前训练 callback。

## 固定范围

每个 NN/角色的旧四分区：fit107,327、calibration17,892、eligibility35,606、validation35,595，共196,420帧。3个NN×2角色合计 **1,178,520 个旧 role-frame**。同一帧会在不同旧NN权重下做纯数值前向，不能把这些重复读计作新 episode。

每8,192帧保存一个只读完成片段；参考块256帧，原 scalar 每个物理帧分别计算。每层转写还再次调用原 frozen runtime 核对完整 logits 和五动作概率。候选保留原 gate M=1、残差 M=5 的矩阵形状，并逐帧保留 sigmoid/tanh 与 masked softmax 的 ufunc 形状；同时检查候选 chunk64和256。

逐位比较所有层，包括 signed zero，不接受容差或仅 argmax 一致。所有五动作概率还与原 NPZ target 字节、完整 `float.hex()` 和零符号计数比较；按原 row 的 action_seed/episode_id/frame/role 重放原 counterfactual draw，要求与原 sampled 完全相同。这些动作不送入环境，不是新轨迹。

W-left 向量候选已在有限旧数据上被拒绝，本完整池任务不再尝试它。

## 执行与记录

新目录为 `output/warehouse_v642/engineering/<run-id>`；程序强制 nice≥10，BLAS/OMP/VECLIB/MKL 线程变量均为1。先清点已有 NPY headers，再保存完整输入指针、Actor双角色SHA、source closure及源码快照。当前机器/NumPy/线程信息也保存。

按此前有限纯 scalar 计时，原 scalar 一遍约507秒，逐层转写加独立原 runtime 约1,014秒；加 chunk64/256、完整哈希/hex/采样，预估20–30分钟。nice优先级可能延长；真实运行耗时以完成片段为准。

每片段立即保存层哈希、不同元素数、最大差、完整概率/采样哈希及实际耗时，追加 fsync 进度。全部已验证的原 scalar logits、五动作概率和原 sampled 另存只读 NPZ（全池约57MB），避免只剩布尔结果。中间层保存完整逐位比较计数与双方字节 SHA，不复制约数十GB的全部激活。若中断，只能以相同源码/旧输入显式恢复；已完成片段不重复运行，只有未完成片段重做。完成/失败目录不允许重试或覆盖。

任一点不一致，保存第一处数值或采样反例及其旧源行/NPZ/float.hex/符号信息，关闭候选。优化方法不变、不降低精度、不尝试把错误结果转成正式通过。

最终 `full_pool_numerical_verification=true` 仅表示这一个固定完整旧池在这些三个旧NN和该已登记软件/机器环境中实测逐位相同。它**不证明所有未来NN/机器逐位相同，不授权未来long接入，更不表示模型效果提升或独立验收**。未来接入需单独评估新source closure和新NN的数值边界；本任务不自动修改 pipeline。
