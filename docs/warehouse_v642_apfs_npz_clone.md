# APFS NPZ 字节等价存储实验

## 结论与边界

本机 APFS 内核能够共享旧 NPZ 中不变的压缩前缀。新文件与目标原 NPZ 的完整 SHA
相同，`np.load(allow_pickle=False)` 后每个数组的 dtype、shape 和 C-order 字节全部
相同。该实验只改变新文件的物理存储布局，不改变 NN、程序、随机流或验收门槛。

目前只新增实验 helper、测试和此文档，**没有接入训练管线**，没有替换、删除、重写
任何既有 20k／恢复训练的科学资产，也没有执行 NN、环境 reset 或新场景。

## 实测

在新隔离目录 `output/warehouse_v642/storage-engineering/apfs-prefix-bench-20261023-01/`
中，先用 kernel clone 创建旧 `call-000000` 的 12 个只读 NPZ 基准，再分别生成与旧
`call-000001`／`call-000058` 字节完全相同的新 NPZ。

| 12 个 NPZ 合计 | call 1 | call 58 |
|---|---:|---:|
| 目标逻辑字节 | 33,222,728 | 33,225,796 |
| inode `st_blocks×512` | 34,295,808 | 35,344,384 |
| 实测共享物理 extent 字节 | 26,929,292 | 26,929,292 |
| 实际改写字节 | 6,293,436 | 6,296,504 |
| 独占数据分配保守上估 | 7,366,516 | 8,415,092 |
| 全部验证耗时（秒） | 0.4355 | 0.4391 |

约 81.1% 的逻辑 NPZ 字节与基准共享物理块。包含创建只读基准在内总计 1.45 秒。
耗时包含源／目标 SHA、clone、差块写、最终 SHA、所有数组逐位比较、物理 extent
查询及源字节／权限重验。原科学输入字节和 mode 全部保持不变。

机器可读收据：

```text
output/warehouse_v642/storage-engineering/apfs-prefix-bench-20261023-01/benchmark.json
SHA256 059ca87d305fbac921cb47385dad22123d6d0d4d746c86a32a9644a08a30027e
```

APFS 的 `st_blocks` 包含共享 extent，不能把 inode allocated 字节直接相加当作新增
磁盘占用。这里通过 `F_LOG2PHYS_EXT` 比较两个文件同一逻辑偏移的物理 device offset，
确认它们实际引用同一数据区域；独占数据上估为 `st_blocks×512−已证明共享字节`。
该数值不包括文件系统目录、B-tree、快照等外部元数据。volume-free 差值也有记录，
但后台任务与 APFS 事务可能干扰，因此没有将它当作隔离的物理测量。

每个完整旧 call 还包含约 58.4 MB JSON，其中大多数是完全重复的 rows。此 helper
只处理 NPZ；结合另行实现的只读 rows CAS，才可将完整包的新增占用降到上述 NPZ
变化量附近。容量计划仍应为检查点、其他 metadata 和文件系统开销保留余量。

## Helper

```python
from experiments.warehouse.v642_apfs_npz_clone import clone_patch_npz

report = clone_patch_npz(
    {"path": immutable_base_npz, "sha256": base_sha},
    {"path": immutable_target_npz, "sha256": target_sha},
    new_output_path,
    block_bytes=65536,
)
```

输入必须是只读普通文件。输出必须是新路径，父目录必须已存在。helper 检查：

- Darwin 的真实 libSystem `clonefile` 存在，输入与输出目录都在同一 APFS 文件系统。
- 所有路径组件均非 symlink，目标原文件与基准 SHA 匹配，输出不存在。
- `clonefile(..., CLONE_NOFOLLOW_ANY)` 成功，且结果是独立 inode。
- 按 64 KiB 比较目标与基准，只有不相同的块才 `pwrite`，最后按目标长度 truncate。
- 新文件完整 SHA、所有 NPZ 数组逐位相同，源 byte／mode／inode／mtime／ctime 未变。
- 用真实物理 extent 地址记录共享量，封为原目标 mode，再无覆盖地原子发布新路径。

clone 不可用、非 APFS、不同文件系统、symlink、可写输入、哈希或数组不符立即拒绝。
**没有普通 copy 降级**，也不会把普通 copy 声称为 reflink。失败只删除本次新创建的
暂存文件；已存在的目标和所有科学输入均保留。

13 项测试覆盖实际 kernel 共享、零改写 clone、目标增长／缩短、输入／路径拒绝、
clone 失败不降级，以及差块写错误时 SHA 验证拒绝。合成随机数组只是字节测试数据，
不是 Warehouse 随机场景。

## 后续显式集成接口

未来只允许新 long-training producer 的**未封存、独占授权 staging 包**：

1. 先照原数值方法生成当前 NN 的临时 NPZ，得到原始完整 SHA；暂不生成最终 manifest
   或 seal。用严格绑定的 observed-data、角色、分区、序列化格式选择一个只读基准。
2. 调此 helper 创建另一个新 inode，证明它与当前临时 NPZ 字节完全相同，并记录真实
   共享／变化量。低共享量不能被隐瞒；容量准入可按实测上界拒绝后续启动。
3. 只有持有该新 staging token 的 producer 才可将验证后的新 inode 换到它自己尚未
   封存的逻辑 NPZ 路径，删除自己创建的冗余临时 inode。
4. 与 rows CAS 组合后，再按原 schema、原逻辑路径和原 SHA 生成 manifest／收据，
   封存整个新包。记录存储实现、baseline SHA 和物理共享收据作为独立工程证据。

该交换和 stage-token 控制**尚未实现／接入**。既有 20k、恢复02和任何已经绑定或
封存的包不能事后去重。新的存储实现也必须在后续训练授权前锁定源码并测试安全恢复。
