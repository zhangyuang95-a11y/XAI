# v6.42 可移植程序证据边界

`warehouse_v642_portable_program.build_archive`只封装已有程序。原v6.42 JSON及递归parent JSON的字节与SHA保持不变；归档以`assets/<sha>/asset.json`保存，原绝对路径只作为溯源标签。
读取使用显式内容解析器，绝不从原训练目录补读或猴补旧程序模块。原合法性证书、保护前缀和输出、深度／叶数／公开特征源、线性叶及特征域验证均保留。

原`actor_role_sha256`继续表示v6.20保护锚点；`target_actor_role_sha256`才是该程序实际拟合和解释的当前NN。两者不得互换。

归档不包含事后生成的独立审计，因此其SHA可在正式封存前锁定，不产生循环哈希。正式独立审计应同时绑定归档SHA、原程序SHA、当前角色SHA、保护锚点、固定合格叶集合及一次性claim／独立数据分区SHA。
`PortableProgram.load`接收独立审计收据及来自已准入发布绑定的`expected_independent_audit_sha256`。
缺少审计时只允许数值比较，`evidence`返回空；这不构成模型验收或发布许可。

独立审计封装schema为`warehouse-v642-portable-independent-program-audit.v1`，phase固定为`final_independent_program_audit`。
它要求`independent=true`、`passed=true`、`program_archive_sha256`、`source_program_sha256`、`role`、`anchor_actor_role_sha256`、`target_actor_role_sha256`、`explanation_eligible_leaves`、`sample_count`、`one_use_claim_sha256`、`audit_partition_sha256`及双用途`metrics`。
指标沿用正式覆盖／动作可用覆盖≥20%、反馈／解释一致≥85%／90%、KL≤0.35、原始非法和路径未证明均零；叶集合不得超出该程序原先合格解释叶。
上游独立验收器负责重放审计数据、验证一次性与分区独立性；本层检查已经准入的内容哈希及数值、模型和程序绑定，不能从布尔值自行证明研究独立性。

`evidence(x,m,executed_NN_action,current_actor_role_sha256=...)`仅对当前目标NN、独立审计合格叶、程序首选与实际NN动作一致且合法的状态返回路径与保真证据。
它不提供`act`、`decide`、环境接口、教师动作或救场动作。没有证据时不推断动机；返回证据也不声称唯一因果动机。

新模块未接入旧runtime。旧版本继续使用旧读者和旧门槛。旧开发数组上的等价性测试只证明实现一致，不是新独立验收结果。

## 本次实现核验

23项测试通过。已花费开发数据的fold0／robot_2使用原77叶程序，在300个既有episode的35,440个验证帧上，原读者与可移植读者的路由、原始logits及掩码后五动作分布逐项完全相同。
合成测试另覆盖归档搬迁与删除原绝对路径、全部16种移动合法掩码、可变线性叶、依赖缺失／篡改、目标NN哈希错误、证书／保护／容量／特征域拒绝，以及独立收据缺失、错绑、失败和当前动作不一致时拒绝解释。
这些测试没有创建环境、运行NN训练、消费新场景或构成正式独立验收；旧runtime接入及完整release验收仍待后续版本化实现。
