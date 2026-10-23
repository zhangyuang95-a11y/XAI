# v642 原程序静态验证封装（未接入训练）

新增 `backend/training/warehouse_v642_prepared_program.py`，只复用不可变原程序的完整
静态验证及序列化结果。它是原 `warehouse_v642_program.Program` 的显式子类，
满足原 `isinstance(program, Program)`，没有 monkeypatch 或全局 reader 替换。
当前 407 个科学源、6 个恢复源、当前训练和所有既有资产均不变。

## 严格绑定

```python
program = PreparedProgram.from_reference(
    original_program_ref, target_actor_role_sha256=expected_target_role_sha256)
program.validate()
original_native_dictionary = program.to_dict()
```

冷准备调用原 `Program.from_dict`、`validate`、`to_dict`，完全验证物理路径、叶证书、
保护 contract、父资产、复杂度、域和原资格内容。原序列化与输入程序逐类型及
`float.hex()` 等价才允许准备；未知字典、目标角色哈希不符、重复 JSON key、
非有限数字和非规范／符号链接路径均拒绝。

每次 `validate()`／`to_dict()` 复用前仍重新检查：

- 原 dataclass **全部字段**的完整内容 SHA，区分 bool／int／float、正负零、
  list／tuple、set／frozenset；不使用会合并这些值的 Python `==`。
- 原 program 文件字节 SHA，以及所有递归 protection parent 的真实文件 SHA。
- 原完整语义 source closure 和新封装模块 SHA。
- 不可变缓存序列化字节自身 SHA，以及未出现额外实例字段。

返回的字典由原序列化字节重新解码，不与内部缓存或可变原字段共享对象。
类型／值／源／父链任一变化即拒绝，不回落为重新接纳修改后的程序。
anchor NN 与 target NN 仍保留各自独立哈希。

`distributions`、`leaf_indices`、`certificates` **直接继承原实现**，每次计算，没有缓存
输出。当前 NN p、采样、argmax、KL、current_report、当前反馈／解释资格与门均不
缓存。直接数值计算本身不发放证据；需要认可程序身份时仍须经过上述绑定检查。
避免在逐叶 `_metrics` 内重复执行完整 source hash，也没有改变逐帧计算和阈值。

## 软件测试

33 项纯软件测试覆盖：

- 原生子类关系、全部四分区路由／raw／q 精确字节、完整 current_report 及每个 float。
- 当前 NN p／sampled 改变时，原生资格和统计重新计算。
- structure、leaf、report、protection、两类资格集合及其余全部字段变更拒绝。
- signed zero、数值类型、容器类型、NumPy scalar、重复 key、非有限值拒绝。
- 原程序、父资产、递归祖先、源码、缓存字节缺损／篡改／symlink 拒绝。
- 返回值不别名；暖复用不再调用原静态证明入口。

测试中的合成数据不代表科学验收。

## 已消费数据实测

Nice10、BLAS／OMP 单线程；**不启用 cProfile**。只读取 step16800 两角色程序与
call34 的原目标包，禁止 NN forward、环境构造及 reset。每角色四分区共 196420
旧帧，两个角色共 **392840 个角色帧**（共享物理轨迹，不是新增独立样本）。

| 操作 | robot_1（102 叶） | robot_2（77 叶） |
|---|---:|---:|
| 原 Program.from_dict | 2.583 秒 | 2.131 秒 |
| 原 Program.to_dict | 2.584 秒 | 2.094 秒 |
| Prepared 冷准备 | 8.373 秒 | 6.098 秒 |
| Prepared 暖 to_dict（三次） | 0.227／0.188／0.243 秒 | 0.179／0.174／0.223 秒 |
| 原完整 current_report | 3.123 秒 | 2.336 秒 |
| Prepared 完整 current_report | 3.141 秒 | 2.306 秒 |

序列化、四分区 route／raw logits／q 与原程序完全一致，完整 current_report 与
原程序及原保存报告逐类型／float.hex 一致。暖序列化约快 **9.4–13.7 倍**，但冷准备
更贵，且 current_report 不变，符合只缓存静态证明的边界。

只读工程收据及源码快照：

```text
output/warehouse_v642/statistics-engineering/prepared-program-bench-20261023-01/benchmark.json
SHA256 4a948e4cb550b9582551987dfd45b732a103484030c73c8d3d3e1f4e34006c76
```

这轮真实验证仅覆盖上述两个程序及完整旧池，尚未遍历全部历史程序，也未接入
ProviderEvidence／PPO。**它不足以直接授权生产集成**：未来独立版本需登记源码、
程序准备时点与无复用失效的接口，在完整旧调用上重放全部证据 payload／更新收据，
并确认重复 `to_dict` 在完整耗时中的真实比例。不得用本组件微基准宣称完整训练
加速或模型验收通过。
