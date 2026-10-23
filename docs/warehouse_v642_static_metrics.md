# v642 精确静态统计缓存：已验证的组件实验

新增 `backend/training/warehouse_v642_static_metrics.py`，尚未接入任何训练、恢复或
证据发放流程。它只计算统计，不授权训练，不替代当前 NN 概率／采样／来源的独立核验。

## 缓存内容

缓存绑定完整 program JSON SHA、递归 parent 资产、原语义完整源码、NumPy 版本，
以及 fit／calibration／eligibility／validation 四分区的完整 x、m、rows 哈希。
program／parent／源码或任意静态输入变化立即拒绝复用。

编译一次的内容为 program 物理路径证明、路由、稀疏叶输出 q、raw／masked 首选动作、
合法及路径掩码、固定单叶区域、episode／scene 频次与公开状态分层。p、sampled、
当前 NN argmax、一致率、KL、当前合格叶集合与最终门不进入缓存，每次重新计算。

数值表达式、float32 运算、布尔选取顺序与 NumPy 求和／均值顺序保持 native 原样。
没有以逐叶聚合替代原始逐行 reduction，也没有修改阈值或浮点容差。

```python
cache = StaticMetricsCache(program_ref, already_independently_checked_parts)
report = cache.current_report(new_independently_checked_current_NN_parts)
```

输入必须仍由原证据流程独立核验。此 API 不检查 NN 权重与 p 的推导关系，也不验证
sampled 的随机流，因此它自身不能作为反馈许可或独立审计证据。

## 实测结果

仅使用已消费的 robot_1／step16800 程序（102 叶）与 call34／call35 的原目标数组。
分区帧数：fit 107327、calibration 17892、eligibility 35606、validation 35595。
Nice=10，BLAS／OMP 单线程，无 NN forward、reset、新场景或科学文件写入。

| 操作 | 耗时 |
|---|---:|
| 编译，包括完整程序验证 | 5.171 秒 |
| 一次 native 完整 current_report | 4.472 秒 |
| 缓存后 call34 | 1.596 秒 |
| 缓存后 call35 | 1.607 秒 |

该组件约快 2.8 倍。native 和缓存计时均启用 cProfile，不能作为正式生产开销门。
call34 的 native 完整报告与原保存报告逐字段／类型／`float.hex()` 相同；两次缓存
报告也分别与各自原保存报告完全相同。

profile 中 native 的 206 次 `_metrics` 累计 3.366 秒，206 次 certificates 累计
1.307 秒；缓存暖调用主要还包括完整静态绑定复核。**该样本 current_report 只有
4.47 秒，不能解释整个恢复 call 的约 230 秒耗时**。因此暂停此缓存的扩展／集成，
转而测量真实 readiness／ObservedPool／证据 verify 外层调用，避免围绕错误瓶颈
扩大优化范围。

只读工程收据：

```text
output/warehouse_v642/statistics-engineering/static-metrics-bench-20261023-01/benchmark.json
SHA256 a0228d937b36b8390e344792912e469327111ce324a77d7c1599fbde4708ba3c
```

同目录保存 native／两次缓存的 pstats，以及与收据 SHA 一致的 helper 源码快照。
这里只验证一个真实角色／程序和两个已消费目标包，不能声称覆盖全部原程序、独立
证据审计或完整训练恢复流程。所有既有科学门与原 407／恢复执行源码保持不变。

## 测试

19 项纯合成统计测试覆盖完整报告和每个字段的 float.hex、空／单叶／叶集合 union、
当前 p／sampled／资格变化、四分区静态输入变化、源／程序／parent 篡改、数组布局，
以及暖调用不再重复证书／路由／程序前向。合成数据不涉及 NN 或环境执行。

## 外层只读诊断

随后直接调用原 `admission.load_readiness(reference-buffer-01)`，没有替造
`TrainingReadiness` 或 `ObservedPool`。NN forward、环境构造和 reset 均设置拒绝调用
哨兵。首次冷加载在 cProfile 下用时 517.390 秒，但其后同一个真实 readiness 的
`verify()` 仅 0.505 秒，绑定 3458 个文件、合计 174484377 字节。冷加载的完整 B
重核不应算作每次 provider call 的成本；cProfile 对十亿量级 Python 小函数调用的
开销也使该冷计时不能当普通运行速度。

原 refit15000 的目标 NN 与所有来源行为 NN 哈希相同。仅从其已封存 x／m／p、
来源实际动作及 rows 组装普通字典，原生 `data_sha` 与原封存
`observed_data_sha256` 完全相同，再计时得到 0.243 秒；两角色 `parts_sha` 为
0.363 秒。没有创建带准入权力的对象或私有目标证书，也没有重建 1650 条物理轨迹。

诊断收据分别为：

```text
output/warehouse_v642/statistics-engineering/readiness-profile-20261023-01/benchmark.json
SHA256 e6ceb12b21e10cff7c85d082e888e59d8f9bbad4832c3e25175ff69edb164336
output/warehouse_v642/statistics-engineering/observed-static-profile-20261023-01/benchmark.json
SHA256 48c4f5d6d653966a47be6994372b53f30f4e9df2f7b25b26458f58999b5afaf4
```

随后使用 step16800 的两个已消费程序（分别为 102／77 叶），只读测量原生程序
验证链；序列化结果与原 program JSON 全等。未构造 ObservedEvidence。

| 操作（cProfile） | robot_1 | robot_2 |
|---|---:|---:|
| Program.from_dict | 6.645 秒 | 5.481 秒 |
| Program.validate | 3.748 秒 | 3.101 秒 |
| Program.to_dict | 6.724 秒 | 5.566 秒 |
| OnlineEvidence.payload | 0.021 秒 | 0.016 秒 |
| digest(payload) | 0.025 秒 | 0.020 秒 |

一次 `to_dict` 中，protection 被完整调用两次，累计分别为 5.793／4.874 秒；
v622 `digest` 累计为 6.215／5.162 秒，其递归 `plain` 累计为 5.483／4.605 秒。
这些累计值相互包含，不能求和。仅 `domain_sha` 就重复 1581／1395 次，累计
2.066／1.855 秒。保护父链的重复纯验证是明确成本，普通 payload 不是主要热点。
本诊断仍没有完整 provider call 的调用次数和端到端 profile，不能宣称已解释其全部
约 230 秒，更不能把 cProfile 放大后的小函数成本当作生产耗时。

```text
output/warehouse_v642/statistics-engineering/program-verify-profile-20261023-01/benchmark.json
SHA256 16b5db1b350e3e1bcbabec7f2dce0506c0f442bae291f81accd21c94e77299b2
```

静态统计组件继续保持未集成。当前 407／恢复执行源码与已有科学资产不变。
