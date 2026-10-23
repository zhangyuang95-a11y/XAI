# v6.42 长预算单配对种子与有限库存层

本层新增prepare／check／audit／checked_clearance接口，不修改旧20k claim、原407源码、恢复源码或协议注册层。
只在真实20k独立预检完整重放通过之后读取调用者proposal、检查原始整数参数或创建输出。
目前没有真实通过收据，因此此次只运行模拟权限与临时数据的测试，没有实际prepare、扫描工作树或分配研究随机流。

## 调用与提案

`prepare(calibration_precheck_ref=...,reference_directory=...,proposal_ref=...,run_id=...)`先调用
`protocol.register_preparation`，再载入原确认过的v6.20 readiness，最后读取不可写的显式proposal。
proposal版本为`warehouse-v642-long-pair-seed-proposal.v1`，字段严格限定为：

- `phase`：development或formal；`stream_id`：协议中的相应三个符号流之一。
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
真实预算阶段准入及每次reset预扣。正式阶段没有真实budget-lock独立读者时仍不得运行。
此层可保存formal符号库存，但不因此授权formal训练。

`assemble_training_batch`仅让有限训练成员经原环境／NN重放读者进入PPO；程序证据不能进入reward或cost缓冲。
`check_reference_buffer`只接受原已审计common-fit buffer。二者不执行训练更新。

测试的历史审计只扫描四个空临时目录，不能表述为真实仓库查重或新流已经获准。
