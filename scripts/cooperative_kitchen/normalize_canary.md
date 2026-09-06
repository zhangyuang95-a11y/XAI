# 去敏 canary 证据整理

`normalize_canary.py` 不联网、不读取环境变量或密钥配置，不接收 cookie、完整 API 响应、原始数据库导出、问题文本或答案正文。输入必须是受控采集流程**事先生成的去敏 JSON**；任意未知字段、重复 JSON 键、非有限数、符号链接均拒绝。错误只打印固定代码，不回显输入内容。

它验证测量记录的内部一致性，不能独立证明人工提供的数据真实发生。原始测量必须保留在受控位置，供研究者核对；不要用单元测试中的合成数据作为真实证据。

## 命令与顺序

在 XAI 根目录，先检查关闭入口时采集的证据：

```bash
python scripts/cooperative_kitchen/normalize_canary.py \
  --release output/cooperative_kitchen/v3-id-pilot \
  --evidence output/cooperative_kitchen/private/closed-canary-sanitized.json \
  --expected-url https://YOUR-KITCHEN-SERVICE.onrender.com
```

默认 dry-run，不创建锁、备份或报告文件。确认预览后，**同一命令加 `--write`** 才写入。再执行 `build_candidate.py --output output/cooperative_kitchen/v3-id-pilot`，用新文件名运行 `package_secret_release.py`，验证并部署。不要在两次写入之间手改报告哈希。

随后开启内部入口，完成四个测试会话，使用 `kind=internal_four` 的新去敏证据重复以上步骤。新证据绑定当次实际部署的 manifest；脚本保留之前重启证据中的旧 manifest 身份，要求其 runtime、Actor、程序、问答配置和 URL 与当前一致。未先验证并冻结真实 closed restart，四会话输入会被拒绝。脚本不更改 protocol、manifest、descriptor、模型或发布状态；重新冻结后的新 manifest 会使旧测试会话不能继续使用。

## 输入契约：`kitchen_canary_evidence_v1`

所有对象的字段集合必须与下表完全一致。SHA256 使用 64 位小写十六进制；ID 只含字母、数字、下划线和连字符，最长 128 位。时间戳保留实际证据中的 UTC Unix 秒；HTTP 记录没有逐请求绝对时间时，只记录真实归档捕获时间与记录顺序，不反推或伪造请求时刻。耗时使用单调时钟测量且包括网络重试。JSON 摘要按 UTF-8、`ensure_ascii=False`、键排序、紧凑分隔符和禁止 NaN 的规范化 JSON 计算。

顶层字段：

| 字段 | 内容 |
|---|---|
| `schema` | 固定 `kitchen_canary_evidence_v1` |
| `kind` | `closed_restart` 或 `internal_four` |
| `endpoint` | 与 `--expected-url` 相同的 HTTPS 应用源站；不含身份、路径、查询或片段 |
| `binding` | 下述版本对象 |
| `started_at/finished_at` | 本次实际测量的起止 Unix 秒 |
| `enrollment_mode` | closed restart 为 `closed`；四内部会话为 `internal_pilot` |
| `freeplay_qa_enabled` | closed restart 为 true；四内部会话为 false |
| `sessions` | closed restart 恰好一个独立试玩会话；internal four 恰好四个独立研究会话 |
| `questions` | 下述已完成真实问答记录，至少两项，覆盖 zh 和 en，最多 24 项 |
| `restart` | 仅 closed restart 需要，internal four 不得包含 |

`binding` 恰好包含 `runtime_sha256/manifest_sha256/actor_sha256/program_sha256/qa_configuration_sha256`。来源为测试时 `/api/status.versions`；把其中 `manifest` 对应到 `manifest_sha256`。必须与当前待更新的本机 candidate manifest 一致，且当前源码 runtime hash 相同。不要把事后新 manifest 的哈希填回旧测量。

每项 `sessions` 恰好包含：

- `run_id/episode_id`：真实运行及回合 ID，各会话互不相同。
- `namespace/mode`：closed 为 `development/freeplay`，internal 为 `pilot/pilot`。
- `condition/task_order`：closed 均为 null；internal 为实际 A/B 和 XY/YX，四会话必须覆盖 A、B。脚本不假定登记从随机区组边界开始。
- `before/after/replay`：下述三个检查点；分别表示重启或刷新前、恢复后、重复原动作请求后。
- `operation`：下述原动作及重放记录。

每个检查点恰好包含 `endpoint/binding/run_id/episode_id/phase/version/turn/state_sha256/captured_at/capture_sha256/record_index`。`version` 是 `view.run.version`；`turn` 是 `view.state.turn`，至少已确认一步；状态摘要覆盖完整 `view.state`。phase 为 `freeplay` 或 `task1`。三个恢复检查点的身份、版本、步数和状态摘要必须一致。`capture_sha256` 是所在原始证据文件的字节摘要，`record_index` 是该文件 HTTP transcript 的零基索引。after 与 replay 如果在同一份证据中，必须按索引顺序发生；两者可共享同一个真实捕获时间。

`operation` 恰好包含：

- `operation_id_sha256/replayed_operation_id_sha256`：同一操作 ID 字符串的 UTF-8 SHA256，必须相等。
- `request_sha256/replayed_request_sha256`：原请求与重放请求的规范化 JSON 摘要，必须相等。
- `first_response_sha256/replayed_response_sha256`：原确认响应及重放响应的规范化 JSON 摘要，仅作追溯，**不要求相等**。接口按幂等契约返回当前 view，期间问答、语言切换可以合法改变 run.version 和回答列表。
- `request_version/first_response_version/replayed_response_version`：动作提交版本、首次确认版本、重放返回版本。首次确认必须为请求版本加 1，且不大于 before 检查点版本；重放版本等于 replay 检查点版本。例如动作首次返回 2，之后问答与语言切换到 5，恢复与重放都返回 5，是有效记录。
- `first_state_sha256/replayed_state_sha256`：两次动作响应中的完整游戏状态摘要，均必须等于 before、after、replay 的状态摘要；版本变化不能掩盖新增游戏步骤。
- `first_http_status/replayed_http_status`：均为整数 200。
- `first_ack_seconds/replay_ack_seconds`：两次请求实际耗时，包含网络重试。
- `database`：恰好含 `evidence_source/record_sha256/operation_receipt_count/joint_step_event_count/request_sha256/recorded_response_version`。来源为 postgresql 或 authenticated_admin_export，摘要绑定受控数据库证据。针对该同一操作的回执数、joint_step 事件数均须为整数 1，数据库请求摘要与原 HTTP 请求一致，回执中的原确认版本与 first_response_version 一致。

`restart` 恰好包含 `provider/operation/dashboard_event/operator_confirmed/evidence_sha256`。前两项固定为 `render/restart`；操作者须确认确实按先捕获、再重启、后恢复的顺序完成，摘要绑定保存的 dashboard 证据。`dashboard_event` 恰好含 `service_id/dashboard_url/event_text/displayed_timezone/displayed_event_minute/time_precision/source`。URL 必须为对应服务的 `https://dashboard.render.com/web/<service_id>/events`，文案保留实际的 “Service restarted by you …”，来源为 authenticated_render_dashboard_accessibility_tree 或 authenticated_render_dashboard_screenshot。

Render 只显示到分钟时，`time_precision=minute`，`displayed_event_minute` 使用含时区的真实显示分钟，例如 `2026-09-06T17:05:00+08:00`。校验使用该分钟区间与前后捕获顺序的一致性；不会把 17:05:00 当成精确点击时刻。未提供事件 ID 的 UI 不需要也不能虚构 event_id。不能用本地进程重启、页面刷新或后续另一条服务重启事件替代。

每项 `questions` 恰好包含：

- `question_id/run_id/episode_id/frame`：真实问答 ID、已列出的运行和同一回合、实际绑定帧。帧不能晚于 before 检查点。
- `language/provider/status`：zh 或 en，deepseek，complete。
- `llm_success/verified`：均为布尔 true，来自持久化诊断；不能把公开回答的 verified 单独当作真实 API 成功。
- `evidence_source/record_sha256`：postgresql 或 authenticated_admin_export，以及预先计算的实际私有问答记录摘要。脚本只收摘要，不读取那份私有记录。
- `request_version/completed_version`：提问提交版本与完成轮询返回的当前版本；后者至少为前者加 1，且不大于 before 检查点版本。问答可发生在原动作之后，最后一条问答的完成版本可以等于 before。不同问答不能复用同一运行的提交版本。
- `game_before_sha256/game_after_sha256`：提问前后游戏状态摘要，必须相等。
- `elapsed_seconds`：提问含排队、重试至实际完成的总耗时。

四会话中每个 A 会话都必须提供真实问答，B 会话不得出现问答记录；整个 canary 必须覆盖中英文。这不是完整 A/B 六回合协议验收的替代。

## 输出、备份和限制

写入前校验现有两报告的 schema、runtime 和 manifest 文件哈希，保留所有已有历史字段。`remote_load_report.passed` **始终 false**，历史二十会话 p95 **6.76／37.97 秒**及原失败标记必须保持；缺失或不匹配则拒绝。只有真实云端证据经检查后才把报告 mode 标为 real_remote。关闭入口与四会话测量分别写入 `closed_canary/four_session_canary`。

各 nested canary 保存实测 p95，动作超过 1 秒或问答超过 30 秒会标记该 canary 未通过，不隐藏慢请求。状态恢复检查独立于延迟：实际重启与重放一致时，可以将 recovery 的恢复结果标为通过，但不能因此宣称四人或二十人性能通过。

`--write` 创建 owner-only provenance 备份目录，保存原 manifest、两报告和事务日志，然后以临时文件、fsync、原子 replace 更新每份报告。普通写入异常会恢复两份原报告。文件系统不提供跨两文件的整体原子事务；若进程被强杀在两次替换之间，旧 manifest 哈希会使服务校验失败，完整备份可用于恢复。必须随后重新冻结，不能在中间状态部署。原始去敏输入自行保留；汇总同时记录规范化证据摘要和原输入文件字节摘要。
