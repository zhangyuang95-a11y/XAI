# v6.28 前置种子审计工程修复

2026-10-21。原驱动运行约52分钟仍未开始场景采集，容量实验0/18。
定位：jq流解析器在每个标量处遍历约1650个离散区间，且对大型历史元数据重复执行。

原进程组在任何实验reset之前停止，保留原bounded inventory和partial triage。
`audit-interruption.json`明确记录只能观察原进程终止，无法获得waitpid返回码。
没有重跑科学实验、改变场景、选择阈值或改变拟合预算。

新增独立只读解析器`experiments/warehouse/v628_seed_triage.py`：仅把jq的数值预筛选
改为每族一个超集包络；后续仍按原始精确离散区间分类，因此间隙值不授予资格。
原科学源码、计划和旧解析器都未修改。新增文件哈希和语义登记在
`seed-ledger/fast-admission.json`，由显式继续命令消费完整只读的旧bounded inventory。

3项等价性测试通过（JSON/JSONL、间隙排除、完整精确值覆盖）。
已完成的330份历史元数据与新解析结果逐项对比，语义证据完全一致。
这些是审计正确性证据，不是模型效果或模型验收证据。

继续驱动：`scripts/continue_warehouse_v628_seed_audit.py`。
日志：`output/warehouse_v628/warehouse-v628-v626-capacity-01/driver-fast-01.log`。
仍需完整历史审计及最终差异检查通过后才采集1650场景，再运行18项容量实验。
