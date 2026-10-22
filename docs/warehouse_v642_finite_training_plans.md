# v6.42 有限配对训练分区与原始随机参数计划

真实开发与新数据确认通过后，新的`v642_training_partitions.py`和`v642_training_seed_plan.py`
将同一raw/regularized的完整Actor起点、新critic初始化、优化器随机流及训练/程序证据场景表固定。
普通调用先检查真实准入，未通过时不读取新清单、不分配真实随机流或创建输出。

## 隔离和有界调用

训练case按8个slot固定次序，20k联合步由显式有限库存覆盖；库存耗尽必须失败，不能临时补seed。
程序fit/calibration/qualification/validation按完整场景与训练隔离；开发旧1650场景与一次新确认场景
也被明确排除。全历史的新鲜性仍须独立元数据/精确参数/上下文查重，不由本地集合交集代替。

所有原始constructor、显式reset、逐帧动作基础参数、critic初始化和优化器采样参数均明确登记。
raw/reg使用相同case和数值流的配对复用有显式tag；不把两个版本或压力多轨迹当独立场景。
未消费的有限条目只是保留，不算已执行reset。真实reset数只由owner的原生审计日志证明。

训练case库存继续使用原通用`warehouse-v629-finite-training-case-inventory.v1`格式；
原始物理/公开轨迹读者和语义没有变化。新分区计划、新程序证据库存和历史clearance独立使用v6.42版本，
不会将旧计划或旧准入收据重新解释为新候选许可。

进入梯度缓冲前，wrapper要求真实环境/RNG、NN动作及reward重放的同一个训练库存成员。
任何程序证据case即使改名为train也不能进入NN梯度；参考KL的初始半份必须来自原已审计缓冲原字节。
当前半份参考状态只能来自这个通过重放的训练batch。

## 未开放范围

本接口只准备最初20k配对校准，不允许直接提出200k/300k/400k正式开发预算。
来源、只读源码快照、完成封条、原随机参数表和24小时历史clearance均须重验。
clearance本身不授权环境或优化器；后续仍需有源码约束和全局单次claim的实际owner。
程序证据各分区在真实readiness中固定为fit900/calibration150/eligibility300/validation300，每版本1650个完整episode；准备阶段拒绝更小或不同清单。它们均不进入NN梯度缓冲。

测试中的有限清单与历史库均为临时合成夹具，没有分配新的研究场景或启动PPO。
当前真实准入未通过，pilot继续关闭。
