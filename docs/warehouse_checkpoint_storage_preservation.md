# Warehouse 已封存检查点的无损存储维护

2026-10-21。v6.28 搜索继续运行，v6.29 已封存并等待前者的全部作业结算。
本机可用磁盘约54 GiB，后续42项中的剩余检查点可能超过余量。
本维护不删除历史对象，不修改程序、Actor、JSON、调用账本或实验源码。

## 实施边界

- 只处理两个登记批次中，由原监督器真实 waitpid 回收且结果已封存的作业。
- 仅压缩其 `checkpoint.json.objects` 内名称与 SHA256 相符的常规、单链接 `.npy` 字节包。
- 使用 macOS `ditto --noclone --hfsCompression` 生成私有副本，校验逻辑字节、SHA、NumPy读取、
  权限、mtime、uid/gid 后原子替换；保持原路径。解压仍可恢复完全相同的字节。
- 写入逐文件 prepared/verified/committed 事务日志，目录 fsync；失败保留日志及临时副本，不自动重试。
- 拒绝符号链接、硬链接、未知文件标志、低临时空间、并发源文件变化及活跃作业。
- 异常终态缺少 plan hash 时，使用原 started 收据与检查点 identity 的双重绑定；不补写旧收据。
- watcher 只跟随启动时实际确认的两个监督器身份，最多处理原18项与新24项，二者退出即结束。

## 预检证据

真实旧包的副本探针：2,824,025逻辑字节，分配空间从2,826,240降至475,136字节，
解压 SHA、NumPy数组、权限和mtime均相同；原包未修改。
见 `output/warehouse_v629/diagnostics/storage-native-probe-01/result.json`。

`python -m pytest -q tests/test_warehouse_checkpoint_compression.py -x --tb=short`：
15通过，1.17秒。含真实 macOS 压缩／解压、异常副本不替换原件、并发变化、
inventory符号链接逃逸回归、原始异常收据绑定及仍存活的作业拒绝。

维护脚本在两批科学源码闭包之外。应用后另存逐包核验、原读取器导入及实际磁盘节约收据。
磁盘空间改善不属于程序覆盖率或模型效果提升，不改变科学预算、失败状态或准入结论。

## 实际执行与读取复核

- 已处理第0折两个角色的三个容量，共6个已退出作业、5,859个物理包。
- 实际节约9,673,342,976字节（9.009 GiB）；删除逻辑文件0，可用空间约63 GiB。
- 第二次只读逐包核验全部SHA和逻辑metadata一致。
- 未修改的原v6.29 `initial_program` 已成功导入六个实际源；角色1仍为51叶、角色2为49叶，
  原有结构与证明均通过。没有新增拟合、LP、环境reset或Actor更新。
- v6.28源码哈希仍为 `1fd77d5d7de38a99260e1ee06dddb6104bf0d99f11a3b22027539508e823721b`，
  v6.29仍为 `4cd4de16e45d76026ee76dc22ef2c3d42c8f54c8652ad17dce5f757a7d04723f`。
- 有限维护进程54635跟随实际监督器41972与43591，在后续作业真实退出且封存后处理其包。
  启动脚本SHA已固定，不维护活跃作业；不安排周期自动任务。

逐项收据见 `experiments/warehouse/evidence/warehouse-checkpoint-storage-preservation-01.json`；
独立读取复核在 `output/warehouse_v629/diagnostics/storage-reader-verification-01/`。
启动记录在 `output/warehouse_storage/native-compression-watch-20261021-01/launch.json`。

此时三个第1折角色1搜索进程47879—47881仍实际存活，v6.29监督器43591继续等待旧批次。
科学目标尚未通过；新的程序确认、PPO和pilot继续关闭。
