# 协作厨房研究数据与离线分析

本说明对应 `KitchenStore` 的 JSONL 导出，以及 `scripts/cooperative_kitchen/analyze_events.py`。分析脚本不调用模型、不推进游戏，也不修改输入日志。它生成逐运行的参与者 CSV、逐回合 CSV 和验证报告。

## 命令

从项目根目录执行：

```bash
python scripts/cooperative_kitchen/study_admin.py export --format jsonl --output events.jsonl
python scripts/cooperative_kitchen/analyze_events.py events.jsonl --output-dir analysis/run-2026-09-06
```

输出目录内生成 `participants.csv`、`episodes.csv`、`analysis_report.json`。已有同名结果会拒绝覆盖；文件权限为仅当前用户读写。可输入多个完整导出文件，但同一次运行不能重复出现，避免把不同时点的导出拼成一条记录。

默认只纳入 **namespace 为 pilot 或 confirmatory，且 mode 为 pilot** 的运行。`--include-test` 显式包含开发、测试和自由试玩数据，供流程验收使用；这些行保留命名空间和模式，不能作为真人结果。当前合成验收样例使用 `namespace=test`，参与者编号以 `SYNTHETIC` 标注。

## JSONL 原始记录

导出以 `type=run` 开始一个运行数据块，随后包含该运行的下列记录。连接多个导出时须保持每个数据块完整；不能仅拼接事件行。

| type | 身份与主要内容 | 用途 |
|---|---|---|
| run | `namespace`；`document.id/participant_id/retry_id/mode/condition/task_order/phase/language/versions` | 参与者、运行、重试、分组、任务顺序和冻结版本 |
| episode | `id/run_id/episode_index/phase`；`document.attempt_id/scenario/snapshot/done/summary/versions` | 练习、Task 1、Task 2 或自由试玩的小回合 |
| frame | `episode_id/turn/snapshot/public` | 从第 0 步开始的每个已确认状态；snapshot 含内部物品记录，public 为参与者视图 |
| event | `id/run_id/episode_id/operation_id/kind/created/document` | `joint_step` 保存前后状态、动作及环境事件；`answer_exposure` 保存回答展示/关闭 |
| question | `id/run_id/episode_id/frame/status/document` | 所选帧快照、问题、回答、版本及执行状态 |
| survey | `run_id/submitted/document` | 问卷草稿或提交答案；提交时保存冻结题库的评分 |

状态的 `turn` 是联合步数；同一输入使双方各执行一个动作。`created` 和 `submitted` 是服务端 Unix 时间，单位为秒。`versions` 是完整冻结版本映射，分析时要求一致。运行的 `version` 是请求状态版本，包含问答等非移动操作，因此不作为游戏步数。

问答入队采用固定服务端额度：每回合 8 次、每运行 24 次、每数据库命名空间累计 500 次，同一运行两次已接受提问至少间隔 2 秒。统计所有已入队 `question`，包括失败、取消和旧版本任务；重连、重启、换回合不会清空运行或命名空间的累计额度。重复请求使用原操作 ID，不重复计数。额度按问答任务计数，不能直接解释为云端请求数或货币费用，因为解析、核验重试和任务恢复可能产生多次 API 调用。

额度或频率不足时，服务在同一数据库事务中保存一条 `kind=question_rejected` 事件与幂等错误回执，然后返回 HTTP 429；不创建 `question`，不推进状态版本或游戏步数。事件 `document` 只包含 `code/scope/frame/version/usage/limits`。其中 `usage` 为 `per_episode/per_run/per_namespace/pending/last_accepted_at`，最后一项是同一运行最近一次接受提问的 Unix 时间或 null。拒绝事件和回执不保存问题文本或状态快照，回执仅保存固定错误信息与请求摘要哈希；同一操作 ID 重发仍返回原错误且只保留一条事件。错误码分别为 `question_episode_limit`、`question_run_limit`、`question_budget_exhausted`、`question_rate_limit`；原有至多两条待处理问答的限制使用 `question_limit`。这些拒绝记录表示服务访问限制，不是参与者理解错误。

内部 `_items`、`_counter_item_ids`、Actor 的 `_held_id` 和 `_batches` 记录物品/菜品批次来源。物品 ID 只在**同一回合**内连接；不同回合即使出现相同 ID，也绝不视为同一个物品。

## 输入验证与纳入规则

分析前校验每回合第 `0…T` 帧完整且唯一，必须存在恰好 `T` 个联合步。每步前后快照必须分别等于对应的相邻帧，最终快照必须等于回合记录。公共状态、固定场景、原始事件、动作和身份归属也须一致。

所有帧及已结束回合的分数重新按 `100 × 出餐数 − 已用步数` 检查；出餐事件数、首次出餐步数和汇总必须一致。分析不会用已存汇总覆盖矛盾的原始数据。缺帧、漏步、重复记录、物品交接 ID 不匹配、跨运行引用，以及任一版本混用均会报错，不输出部分 CSV。多个版本应分别分析。

正式任务的三局 Task 1 和三局 Task 2 全部结束、轨迹均验证通过后，才填写主要指标 `task2_mean_score`。问卷尚未提交不会抹去已经完成的表现指标；问卷分数和量表在提交前保持空值。技术重试关闭的旧运行保留一行，但主要指标为空，标为 `technical_retry_closed`。新运行与旧运行不会合并。

研究命名空间中的正式模式另核验 180 步、两份汤和固定左侧玩家分工。开发测试可以使用独立夹具，但只能通过 `--include-test` 纳入。

## 参与者 CSV

每行身份键为 `(namespace, mode, participant_id, run_id, retry_id)`；同一个人的不同运行或技术重试分别保留。脚本不计算跨参与者平均值、组间检验或干预因果效果。

| 字段 | 定义 |
|---|---|
| `namespace/mode/participant_id/run_id/retry_id` | 数据空间、模式、匿名参与者、独立运行、技术重试序号 |
| `condition/task_order/language/phase` | A/B 条件、XY/YX 场景顺序、当前语言和运行阶段 |
| `versions_sha256` | 对完整 versions 映射计算的规范化 JSON SHA-256 |
| `previous_run_id` | 技术重试之前的运行，仅作关联，不能合并轨迹 |
| `research_data` | 是否来自研究命名空间的正式模式；测试样例为 false |
| `task1_rounds_complete/task2_rounds_complete` | 已完成且验证通过的对应阶段回合数 |
| `six_rounds_complete/primary_eligible/primary_exclusion_reason` | 六局是否完成，主要指标是否可填写，以及缺失原因 |
| `task1_mean_score` | 三局 Task 1 均完成后，三个分数的算术平均 |
| `task2_mean_score` | **主要表现指标**：符合上述六局规则时的 Task 2 三局平均分 |
| `task2_minus_task1_descriptive` | Task 2 均分减 Task 1 均分，只作描述，不解释为干预前后因果效果 |
| `task1_orders/task2_orders` | 各阶段已完成回合的出餐总数；同时查看完成回合数 |
| `task2_completion_rate` | 三局 Task 2 均完成后，达到两汤目标的回合比例 |
| `task2_steps` | Task 2 已完成回合的总联合步数 |
| `task2_steps_per_delivered_soup` | Task 2 总步数除以出餐总数；没有出餐时为空，不记作 0 |
| `task2_mean_first_delivery_step` | 在有出餐的 Task 2 已完成回合中，首次出餐步数的平均 |
| `task2_rounds_with_delivery` | 上一指标的实际回合分母，避免遗漏无出餐回合的含义 |
| `prediction_accuracy` | 四道下一动作题的正确率，来自已提交问卷的 `prediction_item_accuracy` |
| `counterfactual_accuracy` | 四道三步反事实题的正确率，来自 `counterfactual_item_accuracy` |
| `all_prediction_items_accuracy` | 八题综合正确率；要求等于两个四题子集正确率的平均 |
| `cooperation_understanding/predictability/difficulty` | 三项已提交量表，整数 1–7；理解题兼容开发夹具旧键 `understanding` |
| `survey_submitted` | 是否存在问卷提交时间；草稿不输出为正式量表答案 |
| `task_handoff_count/task_handoff_latency_mean_steps` | 两阶段实际交接次数与逐次交接延迟的平均，不包含练习 |

问卷正确率使用服务端对冻结题库的持久化评分。JSONL 不包含完整题库答案键；脚本校验范围及两个子集与综合分的一致性，不从自由文本推断正确答案。

## 回合 CSV 与行为指标

`episodes.csv` 包含所纳入运行的全部回合，使用 `phase` 区分练习、Task 1、Task 2 和自由试玩。`episode_id/attempt_id/episode_index` 保留原始身份；`scenario_id/scenario_seed` 标识场景。未结束回合也保留当前步数和当前分数，但 `done=false`，不进入主要表现指标。

`orders/steps/score/completed/reason` 分别表示出餐数、联合步、分数、是否完成两汤和结束原因。`first_delivery_step` 取第一条 serve 事件的确认步数，没有出餐则为空。`steps_per_delivered_soup` 为总步数除以出餐数，没有出餐同样为空。

下列行为列按玩家 `human_` 和队友 `ai_` 分开。回合行统计本回合；参与者行只累计 Task 1 和 Task 2，不混入共同练习。

| 后缀 | 解释 |
|---|---|
| `move_count` | 位置实际改变 |
| `wait_count` | 明确执行等待；不自动视为错误 |
| `blocked_count` | 未移动且朝向也未改变的受阻方向动作 |
| `turn_in_place_count` | 未移动但朝向改变，包括面向设施的必要转向；与受阻分开 |
| `invalid_interaction_count` | 交互条件不满足，消耗一个联合步 |

`handoff_count` 仅计算同一工作台上：Actor A 放下某个稳定 item_id，之后 Actor B 取走同一物品。延迟是 `取走确认步 − 放下确认步`，至少为 1。自己放下又自己取回不计合作交接。`onion_` 与 `soup_` 前缀分别给出相应次数和平均延迟；总平均按每次交接加权，不取各回合均值的均值。`initial_counter_pickups` 是拿走开局已在工作台上的物品，没有先前放下时间，故不虚构延迟。

## 问答暴露与缺失值

参与者行统计整个运行；回合行统计该回合。

| 字段 | 定义 |
|---|---|
| `qa_requested_count` | 持久化问题数，包含失败/取消的请求；不等于被阅读的回答数 |
| `qa_exposure_count` | 服务端记录的 shown 次数，同一答案可被再次展示 |
| `qa_exposure_closed_intervals` | 能配对的 shown→closed 区间数 |
| `qa_exposure_seconds` | 完整区间的秒数之和；没有完整区间则为空 |
| `qa_exposure_unclosed` | 展示后没有对应关闭的区间数；重新展示前的未关闭区间也计入 |
| `qa_exposure_unmatched_closes` | 没有对应展示的关闭事件数 |

暴露时长是服务端接收展示/关闭请求的间隔，不等同于注视或实际阅读时间。刷新、断线或阶段切换可能缺少关闭事件；脚本保留删失计数，不用回合结束或下次操作补齐时间。

CSV 空单元格表示缺失或不适用，数值 0 表示确实观察到零。`analysis_report.json` 保存输入文件哈希、过滤记录、版本指纹、纳入数量和指标定义，便于复核分析来源。
