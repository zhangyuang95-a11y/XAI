# v642 runtime 依赖注入基础层

本包只提供冻结绑定、只读内容解析和显式adapter注册，不启动环境、NN、训练、服务或报名流程。旧`study_v3`及其全局常量保持不变。

`PolicyCatalog.load(path, expected_sha256=..., verifiers=...)`读取只读发布清单；所有NN、程序、独立审计和正式验收资产均通过SHA索引及显式递归依赖闭包读取。没有训练目录或网络fallback。
`PolicyBinding`及双角色`ProgramBinding`是不可变值对象，绑定release/policy ID、完整Actor、两角色权重、观察／物理schema、训练方法、程序archive/source及独立审计、共同正式验收收据。
A/B共用同一regularized绑定，C使用独立raw；组别不进入传给Actor adapter的绑定。

`RuntimeRegistry(catalog, adapters={...}, legacy_reader=...)`接收三个显式adapter对象；`bind(domain,group,mode=...)`只产生绑定句柄，不伪造运行结果。
未来Store/Server须显式传入依赖并在每次操作前调用句柄`verify()`；当前没有接入旧服务器，也没有实现或模拟成功的报名流程。
旧实例接口只提供读取与导出，不将旧状态交给新模型继续执行。

Warehouse程序通过可移植读者按当前目标NN role SHA及发布绑定中的可信audit SHA加载；保护锚点SHA不当作当前NN。
A/Task2才可请求程序解释；缺独立审计时不给解释。Pong/Kitchen程序语言须由各自DI reader显式支持，不能误当Warehouse程序解析。

**默认pilot始终关闭**：当前没有注册真正的v642正式验收verifier。未知schema拒绝；20k校准、部分训练、缺正式阶段和占位passed收据即使传入宽松回调也被通用门拒绝。
正式阶段必须有三开发流、三正式配对、200k/300k/400k锁定预算、质量选择、程序预审、一次独立审计、留出、执行一致性和工程证据，并绑定两版本完整身份。
只有按schema显式注入的只读验收器独立验证全部原始证据后，才能返回匹配schema/收据SHA的`FormalVerdict`。基础层不根据布尔字段自行证明研究有效。
测试中的fake verifier和synthetic资产仅核验DI边界，不是实际模型验收或pilot开放证据。
