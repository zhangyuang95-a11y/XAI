# v6.29 双版本共用的初始参考KL缓冲

## Material Passport

- Mode: run / validate；Type: read-only training-data preparation。
- Source: existing v6.28 complete collection, original frozen v6.20 dual Actor。
- Scope: prepare initial reference states; no collection or training authority。

## 固定选择与用途

两版本共同使用冻结v6.20参考KL，50%状态来自共同初始缓冲，50%来自当前训练状态。
为补齐原数值核心要求的实际参考文件，固定使用已采集的全部300个argmax episode，
每task100个；不依据配送、碰撞、停机或程序指标选择／丢弃其中的局。

这300局在v6.28三折的inner与outer中始终仅属于fit。生成器逐折检查，不允许任何
sampled、calibration、eligibility、validation或后续confirmation帧进入该文件。
它是已花费的冻结NN拟合状态，不是原BC训练历史的恢复，也不是一次新的BC复现。
原来的三折数据独立性陈述不因此改变，未来训练也不能把这些旧状态称为新独立证据。

raw／regularized将显式绑定同一个文件SHA；共同参考目标仍是完整v6.20 NN的合法
动作分布。参考buffer只提供状态，外部程序不选择环境动作、不产生参考行为标签。
该文件不授权普通调用、训练随机流、环境构造或pilot。

## 实现与核验

入口：`scripts/prepare_warehouse_v629_reference_buffer.py`。

- 原plan、完整1650局收据、冻结双Actor及源代码闭包先复核。
- 每个原始只读NPZ及gzip公开轨迹均检查SHA；公开状态重建观察、173维输入、物理掩码、
  完整五动作NN分布与argmax动作，与原数组、提议和post-state提交动作逐位比较。
- 从原公开事件独立重算完整配送、机器人碰撞与真实停机，比较原采集器收据；
  短局必须有真实公开停机事件，不能记为零停机。
- 保存共同NPZ、每局来源／offset／帧数、SHA、原始代码只读快照及完成封条。
  provenance明确标注旧fit状态、非BC复现、零新reset及无训练authority。
- 审计再次从原公开轨迹核对输出buffer每个block和episode ID，不信任存储的通过布尔值；
  检查输出位置、只读权限、完整metadata、源快照及全部case来源。
- 失败保留新目录与证据，不覆盖原始文件，不静默重试。

以上是原始轨迹到输出缓冲的确定性读取复核，不能替代最后的独立程序审计或模型留出。
环境奖励和新训练场景授权不在此文件中；完整Stage C owner仍须自行绑定。

## 准入边界

此步骤不使用新随机种子、不更新任何Actor或critic、不执行PPO、BC、程序拟合或LP。
当前仍须等v6.29全部24项完成并通过六单元程序门槛，再通过固定新确认与压力集，
才能发行独立的20k配对校准能力。缓冲存在不意味着这些前置条件已通过。

后续仍为三个继续训练流的200k／300k／400k最小全通过预算、正式质量选择、
一次独立程序审计与每task100新种子留出、Pong/Kitchen既有收据核验、三域执行绑定、
工程门槛与最终发布。原实验闭包、预算、门槛及失败收据保持不变，pilot继续关闭。
