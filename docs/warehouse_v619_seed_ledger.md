# Warehouse v6.19 种子与固定 NN 诊断边界

本次仅准备脚本并读取历史元数据；没有启动环境、训练、程序科学拟合或封存评估。6150 的固定 BC Actor 与旧缓存仅用于探索程序方法，不能充当新配对 PPO 检查点或模型验收证据。

## 台账核查

核查 XAI、XAI-study-v3、XAI-v6-integration 和 XAI-v6-recovery-20261015T165508Z 四个根目录，读取 4,135 个按名称筛选的收据、配置、授权、claim、reset 和 seed 元数据文件，共约 318.7 MB。没有读取模型权重格式文件。逐文件路径、实际 SHA 和分类保存在 `output/warehouse_v619_ledger/seed-ledger-preflight-02.json`，SHA 为 `1fc8121eee5445f448d83d97c642cd203d2d6250e88df4ba8baf79f0e660ed61`。

| 提案 | 已解析历史元数据中的结果 | 决定 |
| --- | --- | --- |
| 原 265xxx / 266xxx base 与 task 展开场景 | 展开场景落入有效旧保留区间 500004–10500203；部分 base 也落入更早区间 | 拒绝复用 |
| 新 100265xxx / 100266xxx base 与 task 展开场景 | 没有发现显式引用或区间冲突 | 仍受下述未完成核查限制 |
| 固定 NN 独立验证原 hash RNG | 3854737、7269167、8059535 命中旧保留区间 | 拒绝；已改为预登记连续区块 |
| 固定 NN 独立验证 RNG 1700000000–1700000215 | 216 个值没有发现引用或保留冲突 | 锁入当前 seed plan |
| 未来 stress 的原 hash RNG | 13 个不同值命中旧保留区间 | 正式执行前另配 RNG 区块 |
| 配对初始化 seed 6153–6155 | 没有显式已用引用，但数值与旧环境保留区间重叠 | 必须区分初始化 RNG 与环境 seed 命名空间 |
| 原 critic probe seed = pair + 6200 | 与旧环境保留区间重叠 | 真训练前改用单独登记环境场景 |

当前固定 NN seed plan SHA 为 `0e9284843be55345f1b970305e8d1afe0a653a0c9f88a8a4815d0273de9a45e3`：fit 50 局（25 sampled、25 argmax）、calibration 25、eligibility 25、validation 24，base 从 100266000 分段分配；独立验证为 100266200–100266223 × 三 task × 三 action RNG stream，共 216 条轨迹。

**首轮收据的机器准入字段为 `approved_for_fixed_validation=false`，该收据原样保留。** 当时还有 23 个超过 12 MiB 的旧 native JSON 和 83 个解析失败文件没有完成检查，采集器必须拒绝首轮收据。已扫描的默认 v6 sealed claim 目录没有发现 claim，也不构成外部系统或整仓历史未使用封存集的证明。任何以后发现的封存 claim，不论成功、失败或中断，一律按已消耗处理。

紧凑、可追溯的报告为 `experiments/warehouse/evidence/warehouse-v619-seed-ledger-preflight-01.json`。它仅涉及固定旧 Actor 的开发数据，不能授权正式训练、封存审计或发布。

## 缓存比较准备

`scripts/run_warehouse_v619_cached_comparison.py` 已准备完成，默认只写 plan，科学拟合需显式 `--execute` 与冻结源码 SHA，并使用新建且互相隔离的输出目录。

- P0：读取原程序，完整复算验证帧，必须重现两角色 736/2880、756/2880 的合格覆盖以及 538、194 次整树原始非法输出。
- P1：固定 P0 拓扑，只重新拟合有路径合法性约束的叶分布。旧叶 logits 不作为目标。
- P2：使用预先锁定的新树搜索和约束叶拟合；完整 NN 的物理掩码后五动作分布是拟合目标。

每臂前后核对源码、脚本、Actor 文件与双角色、数据文件、全部物理帧及完整 NN 分布 SHA，逐角色记录合格覆盖、实际/预期可用覆盖、一致率、KL、整树与合格区域的非法计数、五动作分层、耗时和求解器错误。程序不能改变 Actor 或选择环境执行动作。

输入清单为 `experiments/warehouse/evidence/warehouse-v619-cached-comparison-manifest.json`。合成测试 17 项通过（0.72 秒），覆盖帧完整性、物理掩码、跨 split 隔离、拓扑身份和错误报告；未运行真实科学拟合或新环境。这些测试不能作为方法效果或模型验收结果。

## 补充核查与限定开发准入

106 个首轮未解决文件已全部完成检查：70 个 YAML、12 个扩展名为 `.json` 的 gzip pickle 轨迹、1 个零字节旧报告和 23 个大 JSON。大 JSON 合计 1,674,434,060 字节，使用 `jq --stream` 完整解析；gzip pickle 仅解码为本地惰性字段容器，没有导入或执行原环境类、函数或 reducer。最终共检查 4,158 个所选元数据文件，剩余未解析文件为零；原 4,135 个已读取文件的 SHA 再次核对全部一致。

保守数值检索在 8 个文件中发现了 10 处巧合值：100266059、1002660631、1002662122。逐项 JSON 路径均是 `candidate_batches[…].snapshot.rng[1][136/172/144]`，其父值是 version=3、内部数组长度=625 的 Python MT19937 状态元组。这些值是内部 32 位随机状态字，不是场景 seed、action seed 或 RNG 初始化参数。原始保守 false 结果保留，另写精确路径、文件 SHA、状态结构和判定理由；没有因巧合数值更改科学种子。

新的独立准入报告为 `experiments/warehouse/evidence/warehouse-v619-seed-ledger-clearance-01.json`，SHA 为 `17357639ff250a9483d5df062409b5fa5bccdcfc1ba56fdd1d12ee51d5da1a06`。它把 `approved_for_fixed_validation` 设为 **true**，仍绑定相同 seed plan SHA `0e9284843be55345f1b970305e8d1afe0a653a0c9f88a8a4815d0273de9a45e3`，并引用原清单、106 项完整解析、精确数值判定和原文件 SHA 复核。最终 triage 脚本起止 SHA 均为 `97b8f03bfc95aa0618b0a8f3e43aab144ff737a5b980e4ef03abc6149cde8b84`；之前两个中断尝试及已完成、被复核后复用的文件收据也予以保留。

此批准只针对四个声明根目录中记录的历史元数据，以及固定旧 6150 Actor 的新开发数据采集。它不声称能排除未记录运行或外部系统使用，也不授权正式配对训练、封存审计、留出测试或 pilot。补充核查本身的环境 reset、训练步和科学拟合次数均为零。新增 3 项解析器合成测试通过（0.04 秒）。
