# v6.42：15k快照的程序数据描述成绩

2026-10-22，实际raw/regularized配对20k校准仍在运行。raw已到15,000联合训练步，暂停权重更新以收集登记的程序专用1650个完整episode。
以下为其中已完成的684个fit episode，模型是15k步冻结双Actor快照。regularized尚未进入相应收集阶段。

| 范围 | 完整episode | 平均配送 | 平均机器人碰撞 | 停机episode | 停机率 |
|---|---:|---:|---:|---:|---:|
| raw／fit合计 | 684 | 15.13304 | 2.74123 | 15 | 2.19298% |
| task 1 | 228 | 15.22368 | 2.53070 | 6 | 2.63158% |
| task 2 | 228 | 15.08772 | 2.90789 | 5 | 2.19298% |
| task 3 | 228 | 15.08772 | 2.78509 | 4 | 1.75439% |

共81,498个额外程序数据物理联合帧，单独记账；NN训练预算在这一阶段仍为15k步。
从每帧公开任务替换、承运位置、碰撞事件和机器人失活重建计数，并与完整最终环境快照的计数、帧数及终止标志逐局核对。
此次报告不执行环境、NN采样、拟合或优化器更新。全部NN动作与物理状态的独立重放由后续完整程序源池审计负责。

这些是程序提取阶段的数据描述，不能用于最终质量准入、模型改选或与不同场景的v6.20均值比较训练收益。
程序覆盖率/一致率/KL仍等待1650个episode收集完成及实际提取验证；当前尚未到80%反馈窗口。
最终20k冻结模型还须在全新75场景完成三政策任务保留与四个角色程序的联合校准预检。
随后三个开发流、正式三配对、一次性程序审计及留出、三域模型绑定与发布仍待完成；pilot关闭。

原始只读收据：[program-source-outcomes-01.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/program-source-outcomes-01.json)
SHA-256：`0cf43405bdea15e072f6e01ca7ea4110d85b83d21dc2c41f7ab85a88e4896ece`。

## 更新：900个fit episode完整收集

同一15k冻结快照，完整fit分区的公开计数复核结果如下；每个task恰300个episode。

| 完整episode | 物理联合帧 | 平均配送 | 平均机器人碰撞 | 停机episode | 停机率 |
|---:|---:|---:|---:|---:|---:|
| 900 | 107,327 | 15.15333 | 2.71667 | 18 | 2.00000% |

以上仍为程序拟合分区描述，NN训练仍暂停于15k步。收据还观察到18个已完成calibration episode，未混入此900局统计。
程序资格与验证分区尚待收集，不能由此宣布程序或最终模型通过。

第二份只读收据：[program-source-outcomes-02.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/program-source-outcomes-02.json)
SHA-256：`17d2d193fc7d6854d7af12f1cfbbf7995597bf3bdab6d7427bb7a5eb329e4410`。

Git另经实时只读核对：PR [#5](https://github.com/zhangyuang95-a11y/XAI/pull/5)已合入main；2026-10-22 fetch后main仍为`f7a489305a668c28df4165815ee24d399b9e7166`，包含远端Task 2 `4eecd7d`及研究v3 `b7ccfcf`。
模型发布的第二次PR尚未启动。现有部署分支、旧脏工作树及恢复快照保持原状，当前NN训练源码契约保持不变。

## 更新：raw完整程序源池重放通过

实际owner于2026-10-22 14:45:31 UTC记录`program_source_replayed`（journal sequence 3929）。
登记的1650个完整episode已全部采集并通过NN动作、环境物理、奖励、终止及最终私有状态/RNG的重放核对，合计196,420个额外物理联合帧。
这些帧不增加PPO联合训练步数，也不进入NN梯度缓冲；raw训练前缀仍为15,000步。

源清单SHA-256：`6ca91fc6d75dfa051e38f6d108c40eb0c9959497576ce600fb473994d408a1d9`；
完整源池审计内容摘要：`a42e16224b45c0e00f09c1d55032699d25210310785ac32e2ea5bf34e9d22e12`。
证据为实际`calibration/owner-journal.jsonl`中上述事件；此摘要不是额外审计文件的SHA。
程序重拟合、资格及分布保真尚待实际结果；该重放通过不等于新的程序或20k模型验收通过。
