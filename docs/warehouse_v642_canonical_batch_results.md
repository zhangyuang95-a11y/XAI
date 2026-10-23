# v6.42 冻结 NN 数值诊断：有限旧数据结果

代码提交：`18c6423`；17 项纯数值单元测试通过。此次没有新环境、reset、训练、研究种子或科学样本。

原始工程报告：`output/warehouse_v642/engineering/canonical-batch-diagnostic-01/result.json`，SHA256 `780e19d1f318f21510166c8b422ec6ee34f60fe1680f1af1cb3807459b413bae`。报告固定 `full_pool_numerical_verification=false` 和 `optimized_runtime_installed=false`。

## 逐层数值

旧完整 native call 31、47、58，分别对应 16,800、17,824、18,600 联合步；每个 NN 的两个角色各检查128个缓存 validation 帧和512个公开输入域数值探针。

保守的 `frame_matrices_scalar_ufunc` 候选在上述有限数据中，树路由、每层中间量、完整 logits 和全部五动作概率全部逐位相同。探针仅用于数值工程，不是新 episode 或独立验收数据。

`W[None] @ x[...,None]` 候选在每个旧 NN/角色上都出现浮点不等，已排除。它把原残差分支每帧五行动作矩阵改为分开的矩阵向量乘法，改变了累加路径。

## 纯 runtime 时间

每行另用同一旧 validation 缓存的前1,024帧；先实际重算原逐帧 runtime 并逐位核对存储 target，再计候选时间。BLAS/OMP 单线程，进程 nice=10。

|旧 call|角色|原 runtime 秒|chunk64 加速|chunk256 加速|
|---:|---|---:|---:|---:|
|31|robot_1|0.422|9.87×|12.61×|
|31|robot_2|0.447|10.16×|12.94×|
|47|robot_1|0.407|9.22×|11.84×|
|47|robot_2|0.433|9.72×|13.01×|
|58|robot_1|0.452|9.43×|12.26×|
|58|robot_2|0.425|9.49×|12.98×|

这些是小块 NN 前向计时，不是端到端训练提速承诺。完整池逐层等价检查尚未执行，恢复代码保持原样。

## 来源哈希时间

同一既有 program-source refs 的原重复读循环：6,603 次 pointer、1,658,515,799 字节，实际0.884秒。按唯一文件诊断读集：4,954次、72,553,771字节，实际0.295秒；没有安装缓存豁免。这表明该次 source 文件哈希不是230秒级既有 call 重建的主要耗时。

资格计算的完整实测暂缓，优先保护正在执行的真实20k恢复。静态代码显示 `current_report` 每个叶/用途都由 `_metrics` 重新调用整树 `certificates()`，256叶每个角色重复512次；是否为主要瓶颈仍需独立 profile，不把静态推测写成实测结论。

## 后续工程门槛

恢复完成后，先在固定旧 NN 和完整缓存池上调用 `verify_all_layers`，跨 chunk 检查全部层/完整分布逐位相同，再测完整资格报告与整段更新链。任何浮点不等均拒绝，不放宽门槛或以 argmax 一致替代。未来接入需单独新 source closure；本结果不授权长预算训练、独立审计、留出或 pilot。
