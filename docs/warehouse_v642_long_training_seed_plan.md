# v6.42 长预算单配对种子与有限库存层

本层新增prepare／check／prepare_formal／check_formal／audit／checked_clearance接口，不修改旧20k claim、原407源码、恢复源码或协议注册层。
只在真实20k独立预检完整重放通过之后读取调用者proposal、检查原始整数参数或创建输出。
目前没有真实通过收据，因此此次只运行模拟权限与临时数据的测试，没有实际prepare、扫描工作树或分配研究随机流。

## 调用与提案

`prepare(calibration_precheck_ref=...,reference_directory=...,proposal_ref=...,run_id=...)`先调用
`protocol.register_preparation`，再载入原确认过的v6.20 readiness，最后读取不可写的显式proposal。
proposal版本为`warehouse-v642-long-pair-seed-proposal.v1`，字段严格限定为：

- `phase`：仅development；`stream_id`：development-0、development-1或development-2。
- `joint_step_budget`：200000、300000或400000。
- `initialization_seed`、`optimizer_seed`：调用者显式提供的原始uint63参数。
- `train_case_inventory`、`program_evidence_inventory`：内容哈希指针。

本层不会搜索“新种子”。训练库存沿用八slot schema；每slot至少预留
`ceil((budget/8)/120)`个episode，最多`budget/8`个。若实际停机导致库存提前用尽，owner必须关闭，不能自行补种子。
程序库存严格为fit900／校准150／资格300／验证300，共1650个完整episode；与训练按完整场景隔离。
所有base、task展开scenario、action、初始化及优化参数互不重叠，封存参数与其task展开形式全部拒绝。
显式排除旧开发集、旧确认集以及已经用过的20k独立质量评估场景；其他历史使用由四根精确审计识别。

返回`LongTrainingSeedPlan`有`.ref/.value/.readiness/.registration/.config/.verify()`。
兼容属性`.proposal`返回同一个对象，因此owner可读取`.proposal.value`和`.proposal.readiness`。
`check(plan_ref,calibration_precheck_ref=...,reference_directory=...)`同样先重放真实预检，不能依据未读取的计划自行推断权限。
活动owner持有的对象调用`.verify()`复核已绑定输入与源码，不重复生成科学数据。

## 不可变声明和复用

raw与regularized共享同一完整v6.20 Actor起点、critic初始化参数、空优化器状态和优化随机流。
准备阶段只登记这些要求，真实critic／优化器哈希必须由owner初始化后记录并核对，不能预先伪造。
三个流不是独立BC复现；不同总预算均从共同起点重启，提取与λ窗口由各自PPOConfig计算。

原始调用表明确记录task base、构造环境的`random.Random(config.seed)`、显式`reset(seed)`、
以及每episode action base。raw／regularized的相同参数有唯一phase／stream／budget／episode复用标签；
相同数字不能借标签混入不同用途。帧级动作种子仍由原公开派生函数确定，不声称新增独立样本。
一个实际使用的episode对应一次新环境构造和一次显式reset，有限上限等于两个版本全部库存之和。
代码核对表明环境构造只初始化`random.Random(config.seed)`，不调用reset；
因此`maximum_original_reset_invocations`等于库存预留数。
构造的Random初始化和显式reset的重新seed作为两个原始参数操作，
另记`maximum_original_environment_seed_operations`为两倍库存预留数，二者不能混称两次episode reset。
未使用库存仅是预留，不是已完成episode或reset。

输出位于独立`long-training-seed-plans/<run_id>`，包含不可写源码快照、原始参数表、案例预留、plan及完成封印。
初始train各slot cursor和program cursor均为零。运行输出固定为该目录下`training`；
全局claim为`claims/<phase>-<stream_id>-<budget>.json`，更换run ID不能绕过同一配对的单次执行。

## 历史查重

`audit(plan)`使用版本绑定的indexed metadata、完整triage与exact/context投影。
必须包含既定四个历史工作树和现有显式trace registry；不读取82GB模型权重进行搜索。
只排除当前配对的隔离输出、三个明确输入文件及其唯一global claim路径，不排除整个claims目录。
未解析元数据、历史引用或reservation冲突均拒绝。源码模板预留按原source context识别并重验。

`checked_clearance(plan,execution=True)`重算exact投影、绑定registry、四个根、indexed/triage/interval源码，
核对24小时有效期和最后文件名／大小／mtime增量，历史变化即拒绝。
clearance始终是零实际reset、无训练／封存权限的证据；owner还须自己的source-contract、独占claim、
真实预算阶段准入及每次reset预扣。正式阶段必须先通过真实budget-lock独立读者，再走显式正式入口；普通prepare/check不能推断formal权限。

长预算的24小时clearance只约束启动时的历史新鲜度；200k／300k／400k的实际执行期分别固定7／10／14天。
plan绑定`execution_lease_days`和`seed_clearance_start_window_hours=24`；owner的独占claim记录确切开始与到期，禁止自动续期。
这项新工程注册不改变旧20k或恢复版本的时限。

## 同流跨预算的共同起点重启

300k／400k开发提案必须额外绑定`restart_declaration`：
`warehouse-v642-long-common-start-restart.v1`，包括同一phase／stream、相邻from/to预算、
原initialization／optimizer参数，以及前预算真实`long_training_assets`指针。
新的读者独立验证前预算真实退出、完整更新与初始身份，保存`expected_initial_identity`；
新owner必须把实际新初始化的Actor／critic／两optimizer完整身份与其逐项比较。
这不是从前预算的最终模型续训。

只有这两个已注册的共同起点参数有跨预算复用标签；训练／程序源的场景、episode和动作参数继续使用全新且互斥的库存。
历史原始exact结果完整保存。新的`restart-context.json`仅对通过前预算资产闭包验证的明确初始化／优化参数位置，
及已链接的只读历史审计引用，记录有来源证明的有意复用，不冒称fresh。
其他场景参数、未知文件、同值但不同用途、未解析元数据和区间冲突仍然拒绝。
该读者不修改旧exact/context全局规则，也不凭重启声明颁发预算递进或正式实验权限。

`assemble_training_batch`仅让有限训练成员经原环境／NN重放读者进入PPO；程序证据不能进入reward或cost缓冲。
`check_reference_buffer`只接受原已审计common-fit buffer。二者不执行训练更新。

测试的历史审计只扫描四个空临时目录，不能表述为真实仓库查重或新流已经获准。

## 正式训练的显式预算锁入口

`prepare_formal(budget_lock_ref=...,calibration_precheck_ref=...,reference_directory=...,proposal_ref=...,run_id=...)`
和`check_formal(plan_ref,budget_lock_ref=...,calibration_precheck_ref=...,reference_directory=...)`
首先调用`v642_formal_training_admission.read_admission`，在任何其他调用方文件读取之前由真实
`development.read_budget_lock`独立重放三条开发流。缺锁、passed字典或没有真实锁时直接拒绝。

正式提案使用独立版本`warehouse-v642-budget-locked-formal-seed-proposal.v1`，新增显式
`development_budget_lock`内容哈希指针。只接受锁定预算和formal-0／formal-1／formal-2；
不允许restart_declaration、自选另一预算或复用开发初始化／optimizer／场景参数。
初始化、optimizer、八槽训练及1650程序源库存继续全部通过原四工作树indexed/triage/exact审计，
没有正式阶段的历史复用例外。更换run ID不能绕过已消费的formal stream claim。

Plan对象持有从真实reader得到的不可直接构造token，记录formal_admission绑定并将完整预算锁输入纳入refs。
允许缓存真实token，但每次verify仍重验完整源码与所有原输入SHA。源码闭包以平铺文件SHA加入新正式模块，
避免development→quality→owner→seed→formal的递归调用。所有准备收据仍不授予执行、质量、封存或发布权限。
正向测试明确使用fake低层开发auditor；历史审计测试只扫描四个临时树，均不属于真实科学证据。
