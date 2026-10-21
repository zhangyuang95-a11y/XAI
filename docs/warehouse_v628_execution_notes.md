# Warehouse v6.28 执行记录

## 2026-10-21 启动

- 分支：`codex/warehouse-v628-v626-capacity`，从干净的 v6.27 封存提交开始。
- 原 v6.26/v6.27 科学源码与资产均未修改。
- 新旧回归：49 passed（含本轮10项合成契约检查）。
- 计划：`output/warehouse_v628/warehouse-v628-v626-capacity-01/plan.json`。
- 科学源码闭包：`1fd77d5d7de38a99260e1ee06dddb6104bf0d99f11a3b22027539508e823721b`。
- 驱动：`driver-started.json`、`driver.log`；初始进程 PID40970。
- 驱动顺序：历史种子审计 → 有限1650场景采集 → 18项对照 → 独立算术审计和全表。
- 同一角色/折的64、128、256容量并行；各臂独立缓存及同等预算。
- 这是启动记录，不是效果或验收通过收据。当前没有本轮新覆盖率或任务质量结果。
- Actor更新、PPO、最终审计、留出、发布和pilot均未启动。

## 读取进度

`driver.log`记录采集进度；`execution/events.jsonl`记录真实子进程启动和回收。
`jobs/<job_id>/result.json`与`completed.json`构成单元封印。
全18项结算并算术复核后才生成`metrics-18.csv`、`results-18.md`和`summary.json`。

不重启已有collection或job。若驱动失败，先读取失败证据并保留现场。
尚在运行的子进程可观察/接管监督，不能把未知退出状态写成waitpid成功。

## 条件阶段

当前有限驱动截至容量对照；若六个角色/折全部通过，写入方法和六程序锁定。
之后必须另行登记新的固定资产确认场景并运行一次确认；通过前不进入PPO。
若无容量全过，则由驱动关闭候选、保存失败结果，pilot继续关闭。

## 启动后兼容性复核

旧275场景缓存上的全部18个初始程序已逐帧复核：硬路由、未掩码logits、
掩码概率均与对应v6.26最终父程序完全相同。
收据：`output/warehouse_v628/warehouse-v628-v626-capacity-01/parent-compatibility.json`。
这只是兼容性证据，不是新开发效果或验收通过证据。

## 2026-10-21 指标交付补充

新增只读检查点指标读取器 `scripts/measure_warehouse_v628_intermediate.py`。
使用预登记的fit内部180场景评分集和对应内部资格叶集合，报告F/E覆盖、预期F、
一致率、KL，以及同一评分集上重新判定资格的原父程序差值。
不得将这些搜索中间指标写成外层验证、独立确认或完整验收。
合成测试验证读取前后检查点目录所有文件字节不变，并拒绝身份不符的检查点。

最终单元实测仍读取 `jobs/*/result.json` 和完成封印；18项结算前不选择容量。
实时最终表：`progress/latest.md`；搜索中间表：`progress/inner-score/latest.md`。

审计第二次工程继续使用完整有界JSON解析，最大768 MiB，之后保持原精确区间分类。
49,859,266字节历史报告完整语义复核一致，耗时2.739秒；未减少审计范围。
原bounded inventory、两次未完成流解析记录及进程终止观察均保留。
新增来源在 `seed-ledger/bounded-admission.json` 单独绑定，原科学计划和源码不变。
当前科学实验未重试，前置工程继续发生在首次reset之前。
