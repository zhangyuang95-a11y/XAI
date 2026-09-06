# PolicyLens · 协作厨房研究版本

这是独立于仓库、糖豆迷宫和厨房 Demo 的研究运行版本。网页由服务端推进，使用冻结的六动作神经 Actor；未通过本环境验收的产物保持候选状态。**网页能打开、流程测试通过、训练已运行，都不等于已经可以招募正式参与者。** 当前内部预实验通过 `KITCHEN_ENROLLMENT_MODE=internal_pilot` 显式开启，以 `GET /api/status.enrollment.enabled` 判断该入口；`study_ready` 与 `enrollment.formal_ready` 单独表示完整正式门槛。当前 DeepSeek 模型名是滚动别名，服务将其标为 `qa_model_snapshot_unpinned`；该缺口允许内部预实验，但始终阻断正式入口，直到换用可冻结模型快照并重新验收。

厨房 Demo 仍在原目录 `ui/cooperative_kitchen_demo/`，使用程序队友；研究前端在 `ui/cooperative_kitchen_web/`。本次提供独立 Render 配置，尚未替换或部署现有仓库服务。

当前主库部署采用 `output/cooperative_kitchen/v3-id-pilot` 与私有 Render Secret File，具体步骤见 [XAI 主库部署说明](cooperative_kitchen_monorepo_deployment.md)。v1 的训练产物、Qwen 清单和 v2-deepseek 独立私有部署保留为历史版本；下文对应旧版本的复现命令不应覆盖新发布、线上数据库或历史日志。

## 玩法与实验流程

- 方向键或 WASD 移动并转向，面向设施按 E 交互，空格等待。撞墙和无效交互也消耗一个联合步；询问、阅读和回放不推进游戏时间。
- 三份洋葱入锅，再经过四个联合步煮熟，用盘子装汤，经共享工作台交回出餐。初始配置为每回合 180 步、两份汤；需完成预实验难度校准后冻结，不能在正式收集中途改动。
- 正式任务固定人类在左侧供料和出餐，AI 在右侧烹饪和装盘。自由试玩可以交换分工；正式任务不可重打覆盖成绩，不提供自动操作或随意换角色。
- 自由试玩可打开“自动演示”：服务端程序代玩家操作，队友仍使用同一个神经 Actor。每个动作确认后才继续；人工输入、解释、回放、页面隐藏或断线均暂停，刷新不会自动继续。该按钮不出现在参与者实验中。
- 用户 ID → 参与说明与同意 → 操作说明和共同练习 → Task 1 三回合 → Task 2 三回合 → 问卷 → 完成。ID 为 3–32 位 ASCII 字母、数字、下划线或连字符，以字母开头。分组由服务端随机四人区组持久保存，ID 不决定条件；重复 ID 仅可由原浏览器会话恢复，不能从另一个浏览器接管。
- A 组只在 Task 1 进行中及其回顾时问答；B 组有同等回放但无问答。Task 2 两组均关闭问答与旧答案展示。分组不由参与者选择。
- Task 2 平均分是主要迁移指标；分数为 `100 × 出餐数 − 已用步数`。同时保留出餐数、完成率、出餐时间和行为预测；两阶段分差仅作描述。等待可能是合理配合，不能直接计作错误。

## 本地启动与依赖

从仓库根目录执行，Python 版本为 3.13。网页运行依赖与 PyTorch 训练依赖分开：

```bash
python3 -m venv output/cooperative_kitchen/.venv
output/cooperative_kitchen/.venv/bin/python -m pip install -r requirements-kitchen.txt
source output/cooperative_kitchen/.venv/bin/activate
```

纯本地开发可显式使用独立 SQLite；该模式始终不能打开研究参与者入口：

```bash
KITCHEN_NAMESPACE=development KITCHEN_ALLOW_SQLITE=1 KITCHEN_SECURE_COOKIE=0 KITCHEN_FREEPLAY_QA=0 \
KITCHEN_OUTPUT=output/cooperative_kitchen/v3-id-pilot \
output/cooperative_kitchen/.venv/bin/python -m ui.cooperative_kitchen_server --port 8003
```

如当前 shell 已配置 `DATABASE_URL`，服务会优先使用它；不要把生产数据库连接带入本地调试。研究数据使用 PostgreSQL：内部预实验只写入 `pilot`，未来冻结后的正式收集只写入 `confirmatory`。服务端会同时核验入口模式与命名空间，因此不能只把 `KITCHEN_ENROLLMENT_MODE` 改为 `formal`。`development`、`test` 不可混入研究分析。

浏览器地址为 `http://127.0.0.1:8003/`。刷新后以服务端会话 cookie 恢复已确认进度。浏览器只保存尚未确认的请求和问卷草稿；重复网络请求复用操作 ID，不能推进两次。版本更新后的自由试玩需要明确开始新回合，旧记录保留；研究会话不能静默改用新版本。

本次验收已在 Mac 建立独立 PostgreSQL 17 本地集群，位于 `output/cooperative_kitchen/postgres`，仅使用 `/tmp/policylens-kitchen-pg` Unix socket 和端口 55432，不监听网络。重启电脑后可从项目根目录恢复此集群，再启动研究服务：

```bash
mkdir -p /tmp/policylens-kitchen-pg
chmod 700 /tmp/policylens-kitchen-pg
/opt/homebrew/opt/postgresql@17/bin/pg_ctl -D output/cooperative_kitchen/postgres \
  -l output/cooperative_kitchen/postgres.log \
  -o "-k /tmp/policylens-kitchen-pg -p 55432 -c listen_addresses=''" start
DATABASE_URL='postgresql+psycopg://zhangyuang@/kitchen_development?host=/tmp/policylens-kitchen-pg&port=55432' \
KITCHEN_NAMESPACE=development KITCHEN_SECURE_COOKIE=0 KITCHEN_QA_WORKERS=8 KITCHEN_FREEPLAY_QA=1 \
output/cooperative_kitchen/.venv/bin/python -m ui.cooperative_kitchen_server --port 8003
```

这些命令针对本机已初始化的开发数据库，不应对生产数据库重新运行 `initdb`。数据库已启动时无须再次 `pg_ctl start`。

## 神经策略、程序与问答

训练在 Mac 本机执行。环境只对当前 Actor 实际采样的动作计算 PPO 策略损失；程序伙伴的动作不伪装成 on-policy 数据。正式执行是六动作 Actor 的直接 argmax，没有脚本接管或动作重选。厨房的撞墙方向动作仍可用于转向，因此不能把所有不可通行方向屏蔽。

安装训练依赖后，三个种子分别运行，输出目录分开：

```bash
output/cooperative_kitchen/.venv/bin/python -m pip install torch==2.13.0 pytest==9.1.1
output/cooperative_kitchen/.venv/bin/python -m backend.training.cooperative_kitchen \
  --seed 0 --steps 2000000 --device mps --output output/cooperative_kitchen/v1/seed_0
```

种子 1、2 同样执行并改变输出目录。续训需要给出原 checkpoint，保持原种子与环境/训练配置；预算可增加，改变环境后必须新开版本：

```bash
output/cooperative_kitchen/.venv/bin/python -m backend.training.cooperative_kitchen \
  --seed 0 --steps 4000000 --device mps \
  --output output/cooperative_kitchen/v1/seed_0 \
  --resume output/cooperative_kitchen/v1/seed_0/checkpoint_002000000.pt
```

实际文件名以对应目录的训练清单为准。云端只需要导出的 `.npz` Actor、抽取程序、冻结场景/问卷及清单和验收报告，不安装 torch、MPS 或本地语言模型权重。

策略抽取、独立行为验证及场景校准的 CLI：

```bash
python -m backend.training.cooperative_kitchen_validation ACTOR.npz \
  --split validation --episodes 60 --output validation.json
python -m backend.training.cooperative_kitchen_extract ACTOR.npz \
  --output output/cooperative_kitchen/v1/extraction
python -m backend.cooperative_kitchen.calibration ACTOR.npz \
  --output output/cooperative_kitchen/v1/calibration
```

`ACTOR.npz` 替换为已经选定的产物。最终测试地图不得用于选择 checkpoint、程序或调整规则。抽取使用真实 Actor 概率，分开抽取训练、选择和留出审计场景；程序与 Actor 不一致时不能把该分支说成神经动作的原因。

三个快捷问答和自由输入都绑定 `episode_id + frame`。反事实只在复制状态中最多模拟三步，并明确假设。云端语言模型 解析问题、选择已有证据；可展示事实来自实际状态、Actor、程序和隔离模拟的核验。失败返回已确认事实或澄清，不编造意图。

DeepSeek 配置使用 `DEEPSEEK_API_KEY`、`KITCHEN_LLM_PROVIDER=deepseek`、`KITCHEN_LLM_BASE_URL=https://api.deepseek.com` 和 `KITCHEN_LLM_MODEL=deepseek-v4-flash`，请求关闭 thinking。**`deepseek-v4-flash` 是提供方会更新的模型别名，不是冻结快照。** 官方文档明确说明该名称指向最新版本；研究清单记录请求模型名、可接受的返回模型身份、配置与提示词哈希，实际响应模型名写入审计。不能把仅固定请求字符串写成模型权重已冻结，提供方升级后仍需重新审计。参见 [DeepSeek 官方首次调用说明](https://api-docs.deepseek.com/)。没有成功的实际 API 审计时不能宣称真实问答验收通过。密钥只放在服务端环境变量，不进入浏览器、源码、日志或导出。旧 Qwen 产物保留在 v1，当前 ID 预实验发布使用 `output/cooperative_kitchen/v3-id-pilot`。

长问答作为数据库中的持久作业排队，worker 获取租约后在事务外推理，再单独提交答案；不会持有整个游戏数据库锁等待云 API。完成时重新检查阶段权限，已进入 Task 2 的迟到答案不显示。解释暴露事件单独记录，不增加游戏步数或命令状态版本。

## 历史 v2 独立 Render Free + Neon 部署

以下记录旧独立私有仓库方案，供已冻结 v2 复现。当前 XAI 主库 + Secret File 方案按 [主库部署说明](cooperative_kitchen_monorepo_deployment.md) 执行；不要将旧独立目录的忽略规则或目录审计覆盖主库，也不要向公开库提交旧私有产物。

当前的 `render-kitchen.yaml` 已用于 XAI 主库中的 v3 迁移，不能拿来重建这一历史 v2 部署。旧 v2 的下面几条命令只用于复现当时的私有最小源码目录；它们不会修改当前 Render 服务，也不是本轮发布入口。

原 `zhangyuang95-a11y/XAI` 是公开仓库，不放入含题库答案的厨房发布产物。厨房使用独立私有仓库 `zhangyuang95-a11y/policylens-kitchen-study`，从经过审计的最小目录构建，不复制原 Git 历史。代码和清单稳定后生成新目录：

```bash
python scripts/cooperative_kitchen/build_deployment.py --allow-candidate \
  --release output/cooperative_kitchen/v2-deepseek \
  --destination ../policylens-kitchen-study-deploy
```

该目录只包含厨房运行代码、保持原路径的共享依赖、部署配置以及清单选定的十二个产物和 manifest；不会递归复制整个 `output`、原仓库/糖豆/Demo、数据库、日志、checkpoint、参与者数据或真实环境文件。`DEPLOYMENT_AUDIT.json` 记录所有选中文件的哈希；`.gitignore` 默认拒绝未知文件，只允许本次白名单。目录可由研究者初始化到独立私有仓库的 main 分支，不能推到原公开仓库的另一个分支。打包命令本身不执行 Git、上传或部署。

Render 构建末尾执行 `verify_deployment.py`，同时检查源码运行时哈希、manifest 和全部产物哈希。仅校验通过不会解除候选门槛。Web 运行只安装 NumPy、scikit-learn、FastAPI、SQLAlchemy、psycopg 和 HTTP 客户端等 CPU 依赖；scikit-learn 是现有共享 core 初始化的实际依赖，不能删掉。PyTorch 文件为保持运行时哈希原样随源码保留，但线上导入与 Actor 推理不加载 PyTorch。

在新厨房服务中配置 Neon PostgreSQL `DATABASE_URL` 和已有 DeepSeek key；管理员密钥由 Blueprint 生成，仅由研究者持有。`.env.kitchen.example` 列出了名称，不包含真实凭据。Neon 项目应为厨房使用独立数据库；备份和恢复使用直接连接，Web 服务可以使用提供方连接池地址。

**Free 方案不能把本地 SQLite 或运行时文件当永久存储。** Render 的免费服务会在空闲 15 分钟后休眠，文件系统在重启/部署/休眠后重置；Free 实例小时数也与同一 workspace 的仓库服务共享。外部 PostgreSQL 保存状态，Actor 等静态产物必须包含在不可变的部署内容中。Free 上的可用性和并发性能必须实测，不能由本地 20 会话测试代替。参见 [Render Free 限制](https://render.com/docs/free)。

仓库的 `.gitignore` 默认忽略 `output/*`，因此仅提交源代码不会把训练产物部署上去。发布清单冻结后，显式运行私有打包命令：

```bash
python scripts/cooperative_kitchen/package_release.py --release output/cooperative_kitchen/v2-deepseek --output output/cooperative_kitchen/v2-deepseek/release-bundle.zip --allow-candidate
```

它在 `output/cooperative_kitchen/v2-deepseek/release-bundle.zip` 中只放入清单、选定 Actor、程序、场景、问卷与验收报告，校验路径和每个文件哈希，不打包 checkpoint、数据库、原始训练轨迹或日志。候选产物必须显式使用 `--allow-candidate`，并打印候选提示；不会因此解锁实验。

**该历史压缩包包含题目答案和抽取程序，只能通过受控渠道保存，不能放进公开 GitHub。** 旧方案将它解压到私有部署目录的 `output/cooperative_kitchen/v2-deepseek`；当前 v3 改用 XAI 主库源码与 Render Secret File，具体以 [主库部署说明](cooperative_kitchen_monorepo_deployment.md) 为准。旧邀请码、密钥和参与者备份均不属于任何发布包。

正式招募前在部署地址验证：冷启动后恢复、断线重复请求、20 个并发会话的动作/问答响应、A/B 六回合、真实 DeepSeek、服务重启及数据库恢复。通不过的服务保持入口锁定；不能通过手改清单的 `passed` 字段解除。

## 用户 ID、导出与数据库恢复

服务端固定限制每回合 8 次、每运行 24 次、同一数据库命名空间累计 500 次问答，同一运行接受两次提问至少间隔 2 秒；当前值通过 `GET /api/status` 的 `qa_limits` 返回。限制不可通过环境变量、HTTP 参数或发布清单调高，只有 `test_mode=True` 且使用 `test` 命名空间的测试构造允许显式覆盖。全流程浏览器夹具仅将最小间隔设为 0，生产默认值不变。

数据库命名空间锁保证多个进程同时入队也不能突破额度。额度用尽返回 HTTP 429，拒绝记录与幂等回执提交后才回复；重复操作 ID 不重复计费任务或拒绝事件。全局错误 `question_budget_exhausted` 表示该命名空间已达累计上限，重启服务不会恢复额度。失败、取消的任务仍占额度；不要为绕过上限删除历史问答或切换研究命名空间。上线前应按预计参与者数量核对总预算；上述上限限制入队任务数，云端解析、核验和恢复重试的实际调用次数及 token 费用仍须通过已记录用量核对。这不等于 API 提供商账户的货币消费上限。

研究者本机从环境读取 `KITCHEN_URL` 和 `KITCHEN_ADMIN_KEY`，使用管理 CLI：

```bash
python scripts/cooperative_kitchen/study_admin.py status
python scripts/cooperative_kitchen/study_admin.py export --format jsonl --output events.jsonl
python scripts/cooperative_kitchen/study_admin.py export --format csv --output participants.csv
python scripts/cooperative_kitchen/study_admin.py retry --run-id EXISTING_RUN_ID \
  --reason '技术中断的实际原因' --output technical-retry.json
```

参与者自行输入研究用 ID，管理 CLI 不注册参与者或发放邀请码；旧 `/api/admin/invitations` 返回 410。导出与重试回执保存为 0600 的新文件。技术重试请求如果网络中断，用终端已打印的同一 `--operation-id`、原 run ID 和原因重试。JSONL/CSV 是分析导出，不替代包含会话、问答作业和幂等回执的数据库备份。研究重试由受保护的管理接口另建尝试，保留原结果与分组。也可在命令前加 `--env-file <私有0600配置文件>`，避免在 shell 命令中写入管理员密钥。

更完整的离线分析使用 `python scripts/cooperative_kitchen/analyze_events.py events.jsonl --output-dir analysis/new-export`。它验证帧链、分数、版本和物品交接，并输出参与者运行级及回合级 CSV；默认排除开发、测试和试玩。指标与缺失值规则见 [数据字典](cooperative_kitchen_data_dictionary.md)。

在研究者本机安装与数据库主版本兼容的 `pg_dump`、`pg_restore`，配置 `KITCHEN_DIRECT_DATABASE_URL`。工具使用一致性快照同时记录数据与表行数，不在命令参数中暴露密码，拒绝覆盖已有归档：

```bash
python scripts/cooperative_kitchen/postgres_backup.py backups/kitchen-2026-09-06.dump
```

归档旁生成 `.manifest.json`，包含校验和、表行数与来源指纹。它是整个源数据库的备份，包含所有命名空间与会话凭据，须与公开代码和截图分开保存。此工具是手动备份/恢复演练工具，不代替数据库服务商的持续备份策略。`pg_dump` 的自定义归档和快照语义见 [PostgreSQL 文档](https://www.postgresql.org/docs/current/app-pgdump.html)。

恢复时配置另一个空数据库或隔离 Neon 分支的 `KITCHEN_RESTORE_DATABASE_URL`，先校验，再恢复：

```bash
python scripts/cooperative_kitchen/postgres_restore.py backups/kitchen-2026-09-06.dump --check-only
python scripts/cooperative_kitchen/postgres_restore.py backups/kitchen-2026-09-06.dump \
  --report backups/recovery-check.json
```

工具拒绝源数据库与非空目标，不删除现有表；恢复使用单事务，之后验证表行数。只恢复本项目可信归档。完成数据库验证后，还需用同一发布版本启动隔离服务，验证 cookie 会话、尚未完成回合、问答作业与幂等回执恢复。单事务恢复语义见 [pg_restore 文档](https://www.postgresql.org/docs/current/app-pgrestore.html)。

## 验收记录的解释

- `tests/cooperative_kitchen_study_browser.cjs` 默认运行隔离 HTTP 协议夹具，缩短小回合以验证页面全流程、权限、幂等和问卷。报告明确标记夹具，不能作为神经策略表现或真实云端问答的证据。
- 额外传入 `--real http://127.0.0.1:8003` 会通过实际浏览器操作本地真实服务的自由试玩和恢复；不会生成正式参与者数据。
- 完整物理回归使用独立测试服务：先把 `KITCHEN_TEST_DATABASE_URL` 设为本机 PostgreSQL；新版本的 fixture 显式指定 `--release output/cooperative_kitchen/v3-id-pilot --metadata output/cooperative_kitchen/v3-id-pilot/browser-full/fixture.json`，并启用 `--isolated-test-only`。浏览器脚本的 `--full` 指向该 fixture 文件，`KITCHEN_STUDY_BROWSER_OUTPUT` 指向对应新版本报告目录。它使用清单选中的真实 Actor、解释证据引擎、六个场景和八道预测/反事实题，对 A/B 各操作共同练习和六个完整 180 步回合，共 2520 次网页动作，再填写三项量表并验证问卷恢复。测试服务固定仅监听本机 8006，在新建的 PostgreSQL schema 中显式绕过发布门槛，不改环境动力学或终止条件；完成后 Ctrl-C 退出并清理其测试 schema。该结果仍不能替代真实 DeepSeek、模型质量或远程负载验收。
- 界面检查包括 1365×900 / 1280×800，中英文、完整厨房和主要操作首屏可见、长答案、回放绑定和键盘焦点。输出位于 `output/cooperative_kitchen/v1/browser/`。
- 正式发布还需分别通过训练、程序抽取、真实问答、校准、冻结问卷、协议、远程负载与恢复门槛，并补齐研究者联系、数据保留和适用审查信息。内部预实验通过单独模式记录，不会把候选报告改写为正式通过。

本文件记录的是运行方式和验收边界；最新完成情况以发布清单和每项实际验收报告为准。
