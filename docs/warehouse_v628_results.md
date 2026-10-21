# Warehouse v6.28 容量对照实测

| 容量 | 折 | 角色 | 实际叶/深 | F覆盖 | E覆盖 | 预期F | F/E一致 | KL(F/E) | 原E覆盖 | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 0 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 0 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 64 | 0 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 0 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 0 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 64 | 1 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 1 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 1 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 64 | 1 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 1 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 1 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 64 | 2 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 2 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 2 | robot_1 | None/None | — | — | — | —/— | None/None | — | 否 |
| 64 | 2 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 128 | 2 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |
| 256 | 2 | robot_2 | None/None | — | — | — | —/— | None/None | — | 否 |

选中容量：None。

F/E覆盖≥45%、预期F≥40%、F/E一致≥95%/97%、KL≤.20、整树非法及未证路径均为0，并完成全部可编辑区域检查。
三折是数据分区，非三个独立NN；本轮冻结同一个v6.20双Actor，容量变化不会改变NN配送能力。
原v6.26深7和本轮深9的比较包含深度与数据差异；三种新容量之间深度、数据、搜索预算相同。
预算耗尽不能证明容量无效；失败/未完成单元的指标留空，不能视为零。
程序只用于训练反馈或解释。独立确认、PPO、封存审计、留出与pilot均未执行。
工程检查点开销单独验收，尚未独立测量。
