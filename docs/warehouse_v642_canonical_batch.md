# Warehouse v6.42：冻结前向的纯数值工程诊断

此模块只读取已消费的旧 NN、已完成的原生 target/event 和缓存帧，不创建环境、不 reset、不分配研究场景或动作种子、不训练。输出是工程诊断，不是新的质量、程序、独立审计或留出证据。

## 计算路径

原冻结 Actor 的 gate 每个物理帧使用 `(1,D) @ (D,64)`，残差分支每帧使用 `(5,D) @ (D,256)`，其中五行是五个动作。将残差改成五次 `W @ x` 会改变 float32 的累加顺序。

登记三种仅用于诊断的候选：

- `w_left_vectors`：显式尝试 `W[None] @ x[...,None]`；发现任何不等即判弃。
- `frame_matrices`：gate 保留每帧 M=1、残差保留每帧 M=5 的矩阵形状，向 NumPy 暴露批次维。
- `frame_matrices_scalar_ufunc`：再逐帧保持 gate 的 sigmoid/tanh 和末端掩码 softmax 的原始 ufunc 形状。这是保守候选；尚未接入训练或运行时。

原树路由、位置几何、动作特征、五动作、掩码和 NN 参数全部复用原定义。没有改动原 407 源码或恢复 6 文件。

## 核验

`canonical_trace(role,x)` 逐帧复现每层，并另外调用原 `role.logits_encoded(row)` 验证日志值和全部动作概率严格相同。

`compare` 返回每层的逐位相等、不同元素数、最大绝对差和 SHA。**不接受容差、仅 argmax 相同或 masked action 相同**；signed zero 也逐位比较。

`verify_all_layers` 分块遍历调用者提供的全部旧帧，对所有层、日志值和概率累计逐位比较，也可同时核对原 target 缓存。其参考计时包括“转写 trace＋独立原 runtime”，不能把该计时当纯 runtime 基线来夸大速度。

`benchmark_probabilities` 独立测量原逐帧 runtime 和多个有限 chunk 的候选，最终全部五动作概率逐位比较。真实速度只采用这里的纯原 runtime 时间。

`numerical_probes` 从旧行保留可通行坐标、地图、分类状态和 active 位，扰动公开时间、电量、历史标量，并重建物理 extras 与 context。控制数只用于数值测试，非环境或动作随机流；它们不是新 episode，也不声称每个组合都是可达完整环境状态。

`profile_qualification` 用完整旧四分区重算原 `current_report` 并要求其 digest 与原 event 完全相同；`profile_source_hashes` 测量原 source refs 的重复读哈希循环与 distinct 文件读集。后者不等价于完整 physics replay，也没有安装缓存豁免。

## 保护当前真实恢复

`run_finite_diagnostic` 仅进行有限旧 validation 帧和数值探针核对，逐角色覆盖多个旧 NN。报告固定 `full_pool_numerical_verification=false`，不能将局部通过写成完整池等价。

执行使用 BLAS/OMP 单线程、`nice -n 10`；真实 20k 恢复优先。完整旧池核对和完整资格 profile 等恢复结束后再运行。

调用示例（指针必须指向原先完整提交的 call/event，而非 partial 或新调用）：

```python
run_finite_diagnostic([
    {'result': pointer(old_result), 'event': pointer(old_native_event)},
], output=new_engineering_directory,
   trace_rows=128, numeric_rows=512, benchmark_rows=1024,
   profile_hashes=True)
```

输出目录不能覆盖；输入和当前源码闭包在前后校验。没有训练权限、封存权限、效果通过声明或 pilot 开关。即使完整旧数据逐位一致，接入未来训练仍需新版本工程闭包、端到端更新等价与正式的预检授权；本诊断不修改在运行的恢复代码。
