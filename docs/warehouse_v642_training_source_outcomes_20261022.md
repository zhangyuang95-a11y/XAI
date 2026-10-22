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

## 更新：raw 15k程序重拟合两角色均通过

实际`step-15000/result.json`及观察源绑定完成标记均已生成，两个角色的当前NN程序严格门槛全部通过。
以下为独立validation分区；覆盖率分母为该角色的全部物理帧，一致率和KL只针对对应合格区域。

| 角色 | 反馈覆盖 | 解释覆盖 | 预期反馈覆盖 | 反馈一致率 | 解释一致率 | 反馈KL | 解释KL | 非法/未证明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| robot_1 | 59.78649% | 56.97429% | 57.05339% | 98.34594% | 98.48126% | 0.059870 | 0.057294 | 0/0 |
| robot_2 | 50.59138% | 49.41143% | 48.34017% | 98.70058% | 98.79463% | 0.053085 | 0.051553 | 0/0 |

登记门槛保持反馈/解释覆盖各45%、预期反馈40%、反馈/解释一致率95%/97%、KL≤0.20、非法及未证明路径0。
本次仅重拟合固定语言的非保护输出：role1叶优化185次/证明222次，role2叶优化128次/证明154次，节点搜索均0。
全树一致率分别85.55134%、82.08737%，全树KL分别0.238574、0.286359；不能将合格区域的保真度声称为全树保真度。

这是raw训练15k快照在登记程序数据上的结果，非最终20k模型或全新任务保留评估；raw仍不安装反馈provider。
与原冻结模型的确认成绩来自不同轨迹，不能直接用差值声称训练改善。regularized程序、双角色实际程序梯度及最终任务质量仍待完成。

原始结果：[step-15000/result.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-extraction/3f86b1e7d98ed3b833a62b45bfe529bbb9d8f730b18da6f9696463a359ce1d96/step-15000/result.json)。
SHA-256：`8a7b2f11a301bd6408201fdc0679eef75ea0ab75c23d95a4c4cb902b3d334330`。

## 更新：程序反馈前的实际配对轨迹一致性

两版本15k源池Actor的两个角色权重哈希分别相同。对已完成的前100个登记程序episode做只读比较，
每版11,913个物理联合帧的完整事件（动作、公开前后状态、奖励、终止和停机）完全相同；
每局初始及最终私有环境/RNG快照也逐字节相同。所有输入只读并在比较前后核对SHA，未重新执行环境、NN或优化器。
这核对反馈前的配对过程；只覆盖前100局，不代替regularized完整源池的独立物理/NN重放，也不构成模型质量证据。

只读输入清单及结果：[paired-program-source-prefix-100-01.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/paired-program-source-prefix-100-01.json)。
SHA-256：`344b5cabae1ccc8dc634a5d9b31caff5d2a96c3c984290ee22673c7b1b4b375c`。

## 2026-10-23更新：两版本900局fit分区均完成

两版本15k快照在各自完整900局拟合源数据上均为：配送15.15333、机器人碰撞2.71667、停机18/900＝2.00%，物理联合帧各107,327。
此时尚未进入80%的外部程序反馈窗口；两版本相同成绩不能解释为程序正则化带来的收益。
这些仍是程序拟合源数据描述，后续独立任务质量和最终模型验收保持待验证。

只读收据：[program-source-outcomes-03.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/program-source-outcomes-03.json)。
SHA-256：`e237c9538e813d86732fb0a68eacd88736198a1b6932a921b30c6f992a5c0e26`。

## 用户确认的本轮收尾边界

2026-10-23，用户明确选择“当前20k训练跑完后结束”。因此当前`warehouse-v642-calibration-02`配对运行结束后，
核对真实终止状态、完成收据、检查点及实际梯度，保存模型和指标并结束本轮工作。
之后的最终程序拟合、新75场景评估、三个开发流、正式配对及发布等仍是未完成工作，本轮不继续启动。
这项收尾安排不改变科学门槛，不将20k训练完成表述为完整模型验收通过。

## 2026-10-23更新：regularized 15k程序通过，首批双角色实际梯度已接受

regularized的15k程序提取和观察源绑定完成标记均已生成，两个角色的全部严格门槛通过。
其validation指标与上文raw 15k表逐项相同；此时两版本尚未接受外部程序梯度，不能解释为正则化收益。
原始结果：[regularized step-15000/result.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-extraction/097bca0716954649b5f213456c2c0ee32efd13eedd267d8cd8746876760c4581/step-15000/result.json)。
SHA-256：`57015c9a85f6d350dff3635f167fb0854a256725648b9aac7502974ae63f2b48`。

两版本15,000→16,024联合步的实际PPO更新收据均已完成。各执行4个epoch、合计16次优化器更新；
以下梯度为程序损失对对应Actor的梯度范数，记录于共享梯度裁剪前。

| 版本/角色 | 实际接受的正程序梯度更新 | 程序梯度范数范围 | 选中/全部优化器帧暴露 |
|---|---:|---:|---:|
| raw / robot_1 | 0 | 0–0 | 0 / 4096 |
| raw / robot_2 | 0 | 0–0 | 0 / 4096 |
| regularized / robot_1 | 16 | 0.000125613–0.000287727 | 2412 / 4096 |
| regularized / robot_2 | 16 | 0.000091172–0.000187038 | 1975 / 4096 |

帧暴露数包含4个epoch的重复使用，不是4096个独立物理帧，也不是独立验证覆盖率。
两个角色在两版本中均发生实际权重变化，且两版本在本批结束后的角色哈希分别不同。
这些权重变化来自PPO、成本、参考KL及适用的程序项的合并更新，不能全部归因于程序项。
本表确认raw零外部反馈和regularized双角色实际反馈；尚未证明最终任务保留、20k最终程序保真或完整模型验收通过。

原始收据：

- [raw update.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/raw/rollouts/015000/update.json)，SHA-256：`841e83ead584badf7389f56b807c7e27d48af590b4ad1475c4848b8af0ecc96f`。
- [regularized update.json](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/regularized/rollouts/015000/update.json)，SHA-256：`5ad11a53a3734bb959475dbaee83ab551a444987bdc1692d3833de4d0ad08955`。

记录时训练仍在继续，未观察到终止状态；后续按上述用户确认边界完成20k并核对最终收据。

## 2026-10-23更新：16,800步两版本程序均通过，检查点已保存

两版本的16,800步程序重拟合、观察源绑定及安全批次边界检查点均已实际生成。
每份检查点均记录`processed_joint_steps=16800`、`refits=[15000,16800]`、`safe_batch_boundary=true`及`failed=false`。
检查点内累计接受的正程序梯度更新与全部已完成PPO收据逐项求和一致：raw两个角色各0次，regularized两个角色各32次。

以下是各自当前NN在登记源池validation分区的程序指标，四个角色/版本组合全部通过既定严格开发门槛。
全部原始非法动作及未证明路径计数均为0；一致率与KL仍只针对对应合格区域。

| 版本/角色 | 反馈覆盖 | 解释覆盖 | 预期反馈覆盖 | 反馈一致率 | 解释一致率 | 反馈KL | 解释KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw / robot_1 | 59.78649% | 57.79463% | 56.94284% | 98.27076% | 98.42504% | 0.061681 | 0.058938 |
| raw / robot_2 | 50.59138% | 49.54628% | 48.49315% | 98.73945% | 98.77523% | 0.051023 | 0.050590 |
| regularized / robot_1 | 59.78649% | 57.79463% | 56.94057% | 98.25666% | 98.41532% | 0.061657 | 0.058913 |
| regularized / robot_2 | 50.59138% | 49.54628% | 48.50118% | 98.76721% | 98.80358% | 0.050864 | 0.050438 |

两版本角色哈希均已与各自16,024步快照不同，且在相同角色之间raw/regularized哈希不同。
本轮没有新增独立任务质量评估；上文900局fit描述不属于16,800步模型的新质量成绩。
这些程序验证结果也不替代最终20k模型的独立保真验证或最终准入，不据两版本的微小差异声称正则化收益。
当前20k训练仍在运行，继续遵守用户确认的收尾边界。

原始结果及检查点：

- [raw 16,800步程序](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-extraction/3f86b1e7d98ed3b833a62b45bfe529bbb9d8f730b18da6f9696463a359ce1d96/step-16800/result.json)，SHA-256：`3fafea85a12ce21bd527497adb70151c8c07db5d30729ced4f7bf4188aaf1cce`。
- [regularized 16,800步程序](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-extraction/097bca0716954649b5f213456c2c0ee32efd13eedd267d8cd8746876760c4581/step-16800/result.json)，SHA-256：`aa6bbbeb59dd0941b147ec535cfda7fc9e0b2e27a8e14d1ad14bb45eda554100`。
- [raw检查点](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/raw/checkpoints/016800/owner-state.json)，SHA-256：`4019ffb17b612be0c95979ddf0086b79b60b3d5eec5560d234fb903091236755`。
- [regularized检查点](/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/calibration/regularized/checkpoints/016800/owner-state.json)，SHA-256：`1c6d1fde78693b7fecb03ca11ae298108f238ca36047108ee818ee4e7a565553`。
