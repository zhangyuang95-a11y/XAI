# 三域统一方法 v7：实现与执行状态

候选：`three-domain-unified-01`  
目标发布：`policylens-three-domain-20261024.v7.0`

## 科学状态

**本文件不是模型通过收据。旧 Pong／Kitchen 的通过收据不授予 v7 准入。**
旧 Warehouse v6.42 的20k校准失败、旧模型、收据及447份绑定源码全部保留。
新模型的正式预算开发、封存审计、留出测试及发布尚未完成，pilot关闭。

## 已实现的共同核心

`training_v7/` 独立于历史版本，三个域共用以下实现：

- `contracts.py`：唯一不可变 MethodSpec、域/角色公开特征契约、ExperimentSpec和共同程序门槛。
- `model.py`：双独立Actor、256宽两层动作评分网络、64宽门控、16叶内部稀疏树、双角色reward critic及共同cost critic。包含独立冻结NumPy前向。
- `losses.py`：等权BC、内部树CE/分布蒸馏/0.03对齐项、参考KL、全物理帧归一的外部程序KL和raw零程序梯度。
- `bc.py`：1270局配额，原分组累计的BC训练日程，公开教师与NN访问分开，NN访问全程执行NN动作。
- `ppo.py`、`collector.py`：共同PPO数值核心、8环境固定场景块、4000步critic预热、完整episode失败成本、参考KL与实际程序梯度计账。
- `program.py`、`bridge.py`：四分区隔离、深12/256叶外部程序、角色专属稀疏线性叶、合法性证明及独立有理数证书复核、当前NN资格失效处理。
- `storage.py`：不可覆盖claim、reset及联合步预扣账、原子检查点、角色张量哈希、静态证据的开始/结束完整校验。
- `quality_run.py`：训练后的真实模型新场景评估和独立NN/物理回放；reference、raw、regularized分别记录，不相互替代成绩。
- `runtime.py`：同一绑定下的演示、接管、恢复、回放和反事实。当前仅允许preview/test；缺少新独立程序审计时明确返回无法推断动机。

各域适配器仅提供公开特征、物理步骤、原生奖励、教师标签及完整失败事件；没有优化器或NN动作选择器。
教师仅在BC标签及明确标注的离线人类代理中使用；正式NN–NN步骤不调用教师。
Kitchen保留6个人类角色动作、13个AI角色动作，槽位和订单由AI的NN输出。

### 新程序的初始拓扑

新BC模型没有可直接继承的历史S1证书。共同抽取器在fit内部80%数据上，
从完整NN动作分布生成至多16叶的种子模板，再编译计数的显式合法性分支。
支持及结构比较只使用fit内部20%场景；可靠输出在fit内部判定并保护。
扩展按解释覆盖、预期反馈覆盖、全树KL、叶数排序；最终使用完整fit重拟合。
校准只选温度，资格和验证不选择结构或支持。在线阶段锁定拓扑、支持、witness及原保护输出，
只重拟合未保护叶输出，然后重新证明与判资格。没有继承旧Actor的合格叶。

每叶支持最多3个公开源，系数最多`动作数×4`；内部16叶树和外部256叶程序是两个不同资产。
WAIT的合法性来自各域非终止物理状态的不变量，并在公开特征域显式固定为1。
外部程序没有环境执行权限。

### 完整episode与采样口径

固定场景块保证配对的每个联合步具有相同场景暴露。提前结束后在同一块内重启相同场景。
块末未完成的轨迹明确记录为训练截断，不能作为零失败完整局。reward GAE在真正终止、
原生时限终止、训练截断间保持区别；成本仅来自真实完整局。

共同风险目标2%，窗口256局，至少128局，90%单侧二项上界，释放线1%。
模型初始化、critic初始化、BC动作采样、程序源采样、PPO动作采样和优化顺序使用分别登记的随机流。
同一配对内raw/regularized复制相同Actor、critic和初始优化器，并使用相同采样及场景流。

参考缓冲来自300个完整BC NN访问episode；证据四分区不进入该缓冲。
新的最终Actor绑定自身角色哈希；程序同时保存BC锚点和当前拟合目标Actor哈希。

## 已实现但尚未构成正式准入的接口

`evaluation.common_budget_lock`及`select_quality_only`是预算/排序计算函数，
**不是可凭passed布尔值授予训练或发布权限的reader**。

当前生产命令覆盖注册、历史种子查重、BC及单域20k校准。
完整九开发配对有限owner、正式三配对owner、一次性封存执行器、新研究报名/分组接入、
独立发布reader及发布打包器仍需后续实现和实际验证；不能因本次单元测试通过而标记完成。
PPO数值检查点包含模型、参考NN、双优化器、beta、风险状态、RNG、程序及采集游标，
支持数值恢复。阶段级自动续跑命令尚未开放；重复claim会被拒绝，不能从零重跑已消费的随机流。

## 执行命令

数值库必须单线程；入口在导入训练库前检查并设置线程环境。

```sh
python3 scripts/run_three_domain_unified.py register
python3 scripts/run_three_domain_unified.py seed-audit
python3 scripts/run_three_domain_unified.py calibration --domain warehouse
python3 scripts/run_three_domain_unified.py calibration --domain pong
python3 scripts/run_three_domain_unified.py calibration --domain kitchen
python3 scripts/run_three_domain_unified.py status
```

每条命令实际退出状态单独记录。seed-audit使用4个既有历史根目录和原参数化惰性审计器；
冲突、缺失历史根或未解析元数据均禁止首次reset。该审计只证明声明元数据范围内的查重，
不声称对未记录外部实验提供全局新颖性证明。

任一域失败会写入不可覆盖的`CLOSED.json`，其他域在后续reset/优化边界停止。
BC程序预检失败不得启动PPO。最终质量只使用本次20k后的真实导出模型和新75场景；
代理结果单独标注，真人样本数为0。

## 软件验证与旧版本边界

2026-10-24回归：**166项通过，0失败/错误/跳过**；其中48项为新共同核心测试。
测试前后835份执行源码未变，源码闭包SHA为
`066b493a4b4cab661c7a4375b03e0126b9609efd92bab2233a82d0932f5051e1`。
实际子进程退出0的记录位于`output/three_domain_v7/engineering-20261024-03/result.json`。
该记录明确标记`scientific_model_evaluation=false`、`model_acceptance_passed=false`。

新测试覆盖三域公开观察、私有未来/路线历史排除、动作维度1/3/5/6/13、
Torch/冻结前向、真实梯度、raw零反馈、接管/恢复/回放/反事实、
证书篡改、显式移动合法性、全部物理帧归一、未完成成本排除、
质量选择不得读程序分数及检查点后下一次更新等价。
这些测试中的人工分布和旧开发快照明确属于软件夹具，不是模型通过证据。

原v6打包器的一个旧测试夹具缺少当前已要求的Actor版本和拓扑训练字段，
导致在检查源码哈希之前就被正确拒绝。仅补齐该合成夹具的结构字段，
并增加禁止提前复制资产的断言；没有改变旧打包器或放宽准入。

检查点开销≤10%是另行测量的发布门槛，尚未以本次软件测试证明。
新方法所有科学及工程门槛通过之前，不生成可用发布清单，不开放pilot。

v7种子选择器扩大为四个历史仓库中的白名单元数据、启动guard和已登记完整轨迹，
覆盖Pong、Kitchen及混合域记录。它在独立命名空间调用原审计数值代码，
同时记录原engine和新selection adapter的SHA；没有修改或monkeypatch旧模块。
不能把旧Warehouse路径范围的查重结论当作三域查重通过。
