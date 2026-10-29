# v6.42 三域核心研究 API

本实现新增 `runtime_v642/store.py`、`runtime_v642/server.py`，通过 `RuntimeRegistry` 显式注入 Warehouse、Pong、Kitchen 冻结适配器。不导入旧 `study_v3.Store`、全局 `engine()` 或旧服务器，不修改原407源码或恢复6代码；原冻结 Actor/物理函数通过显式适配器正常复用。没有自动启动服务、迁移 Render/live、写发布清单或打开 pilot。

**这是完整核心 API 的软件接入，不是完整旧 UI 研究流程已迁完，也不是模型验收通过。** 默认没有 Warehouse 新正式验收 reader，所有 `/v642/` pilot 入口均先调用 `catalog.require_pilot()`；未知 schema、缺正式阶段、模型或依赖哈希不符均返回503。真实校准、三流开发、正式训练、独立审计、留出、工程及三域执行验收不能被软件测试代替。

## 组合与持久化

`compose_store(database=..., catalog_path=..., catalog_sha256=..., scenario_pools=..., demonstration_seeds=..., mode='pilot', legacy_database=...)` 显式构造三个实际适配器和 Store。只读 catalog 由上游独立流程提供，此入口不生成或发布它，也不暴露假正式验证器参数。

场景池和三域演示种子均由调用者显式提供，服务无训练目录 fallback，不自行搜索或分配研究训练种子。Kitchen 必须能在其已绑定配置中找到指定场景，否则拒绝；不能悄悄调用旧全局场景生成器。

新 Store 只使用独立 SQLite 数据库；发现 `pl3_` 历史表即拒绝初始化，不能误把旧库当新库迁移。表为 `v642_*`，单事务保存报名、分配、run、frame 和命令结果。线程锁及 SQLite 写事务使分配和动作提交原子化。重开同一 DB 保留同一绑定和进度。

报名保存完整 `PolicyBinding` JSON、其 SHA、release/policy ID、完整 Actor、双角色 SHA、观察/物理/方法 schema、程序/审计及接受收据哈希。SQL trigger 禁止修改分组、场景、绑定及 run 起点；帧、演示、问答和命令为 append-only。会话密钥只保存哈希，导出不包含密钥。

## 分配与三任务

- A/B 绑定同一个 regularized 双 Actor；C 绑定独立 raw 双 Actor。group 保留在研究元数据和解释门，传入 Actor 的参数只有显式 `PolicyBinding` 和状态；状态不加入实验组别。
- pilot 禁止指定 group 或场景。分配先在 domain 内选择当前人数最少的已登记场景，再在该 domain×场景内从人数最少的 A/B/C 随机选组；平局使用系统随机源。preview/test 的组别或场景覆盖只给独立 admin 能力。
- 流程是报名→对应绑定的双 NN 演示→确认演示→Task1→Task2→Task3。只有前一任务物理终止才能进入下一任务，不依据客户端声称的完成状态推进。
- 人类动作仅替换 human 角色，AI 动作仍由绑定冻结 NN 和物理掩码决定。每次动作同时核对 `step`、同绑定单步 counterfactual 和原生 replay 的一致性。
- 命令请求身份包含 run ID、task、原 turn 和 human action。同 run 的同命令重试幂等，跨 task/run 复用同 ID 返回409，不能取到上一任务的旧响应。

## 恢复、回放和问答

每帧保存完整绑定、state、public state、当时 NN decision、上一步原生 transition bundle 和连续 frame SHA。恢复/回放/导出重新验证所有相关帧及 NN decision、合法物理转移和初始场景身份，不调用 reset 或重新抽场景。缺资产和错误绑定没有规则控制 fallback。

只有 A 的**正在执行的 Task2**可问答；可选本实例已执行的 Task1/Task2 决策，禁止询问当前未执行的下一步动作。实际 adapter 用 `BoundDomain.explanation_program` 的审计门加载程序，并要求程序证据匹配该帧当前 NN 动作和角色 SHA；Store 再核对 source-program、audit、当前目标 role SHA 和 action。证据不足时明确返回“无法推断动机”，不虚构目标或救场理由。

当前问答是透明的动作/程序证据模板，提供结构化路径，**不声称已经迁移旧自由语义 LLM 问答**。有证据也只描述模型行为模式，不声称证明唯一动机。可附1–12个明确人类动作的反事实，固定同一绑定和初始完整快照，实际 NN/物理 replay 后保存；导出时再重放原生反事实和审计证据。

Warehouse 使用其绑定的独立审计程序；Pong/Kitchen reader 保留 carried v5 的原始 development audit 范围，不把它们改写成新的独立程序审计。

## HTTP 接口与权限

`make_server(store, host='127.0.0.1', port=0, admin_token=...)` 返回**未运行**的 HTTPServer，由调用者明确 `serve_forever()`。没有自动 CLI 启动或部署修改。请求为有限 JSON body，拒绝重复 JSON key、未知字段及非有限数。响应 `no-store`，不记录 token/问题/URL 日志。

参与者用 `Authorization: Bearer <session_token>`；admin 使用独立 `X-Admin-Token`。没有客户端 `admin=true`、`mode=preview` 或 `policy_id=...` 绕过入口。

|方法/路径|作用|
|---|---|
|GET `/health`|纯存活状态，不表示验收通过|
|GET `/v642/status`|按当前模式检查准入；pilot 未完成返回503|
|POST `/v642/instances`|报名：domain、participant_reference；preview/test 的覆盖仅 admin|
|GET `/v642/instances/{id}`|认证后的恢复视图|
|POST `/v642/instances/{id}/demonstration`|保存/读取并重放绑定 NN–NN 演示|
|POST `/v642/instances/{id}/demonstration/complete`|确认演示|
|POST `/v642/instances/{id}/tasks`|开始下一项合法任务|
|POST `/v642/instances/{id}/actions`|command_id、expected_turn、human_action|
|GET `/v642/instances/{id}/frame?task=2&turn=1`|从起点重放后的指定帧|
|GET `/v642/instances/{id}/replay?task=2`|重算的任务帧序列|
|POST `/v642/instances/{id}/questions`|question；可选 target_task、decision_turn（旧 turn 为别名）、language/human_actions；仅 A 活跃 Task2|
|GET `/v642/instances/{id}/kitchen-practice`|独立中立练习例外标签，不代表 NN 评估|
|GET `/v642/instances/{id}/answers`|仅 A 活跃 Task2 的已核验答案历史；完成后撤权|
|GET `/v642/research/instances`|admin 研究索引，不需要/返回参与者密钥|
|GET `/v642/research/instances/{id}/export`|admin 完整实例导出，并重新验证 NN/物理/QA|
|GET `/v642/instances/{id}/export`|同时持有实例 session 和 admin 的完整导出|
|GET `/archive/{old_id}?export=1`|admin 只读旧数据；独立于新 pilot 的模型准入|

参与者 HTTP 视图只返回公开状态、已执行 AI 动作和绑定身份；当前 head 的未执行 decision 不公开，last_executed_action 绑定上一动作及结果回合；双 NN 演示显示两个 NN 动作。不会发送完整物理/RNG快照、任务转移私有 bundle、B/C 的程序解释或原始 NN 分布。完整状态留在 DB 和 admin 研究导出中。HTTP 公开演示/反事实标记为 public view，不能当作原生可恢复快照传回 adapter。

## 历史数据

`ReadOnlyLegacyReader` 只读已封存的 SQLite `pl3_*` snapshot：`mode=ro&immutable=1`、`query_only=ON`，拒绝未完成 WAL/journal，并在读前后校验原 DB SHA。它不实例化旧 Store、不建表、不更新、不恢复旧任务到新 NN。导出保留原 release、绑定、帧和已识别旧研究表原值；密钥/recovery hash 不导出。若历史库仍在变动或为 PostgreSQL，须另提供注册的只读 reader，不能冒充本 snapshot reader 已兼容。

## 明确剩余兼容缺口

- 旧网页、Prolific/同意书/招募、报酬、问卷、理解评分、自动解释提示和旧完整导出格式尚未迁移。
- Kitchen 中立教学练习已单独标注 `model_evaluation=false`，交互教学 UI 尚未接入；没有把规则练习当 NN 演示。
- 旧 Pong 协调控制页面不挂载；未知 `/pong/` 页面返回404。
- 当前服务存储为 SQLite；生产 PostgreSQL 存储和旧 PostgreSQL 历史 reader 不在本提交内。
- 当前问答为程序证据模板，旧自由语义提供器未接入。
- 仍需真实正式科学验收 reader、实际三域端到端验收和完整参与者 UI/运营流程核验；本代码不会替它们开放 pilot。

测试只用临时 DB 和显式 toy adapter；正向仅 preview/test，pilot 仅验证拒绝路径，未注入 fake verifier 伪装实际准入。测试通过不是模型、人类协作或发布效果。

最新历史坐标、反事实窗口及防提前暴露动作约定见 [committed history API](warehouse_v642_committed_history_API.md)。此更新使用新的 core Store/HTTP/export schema，旧 core DB 不按新语义续写；不改变任何模型准入结论。
