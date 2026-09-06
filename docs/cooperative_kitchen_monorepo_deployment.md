# XAI 主库中的协作厨房部署

厨房源码随 XAI 主库管理，厨房和物流仓库仍是两个独立 Render Web Service。厨房使用 `render-kitchen.yaml`；原仓库的 `render.yaml`、依赖和启动入口保留。

公开仓库只提交厨房源码、部署脚本，以及 `deployment/cooperative_kitchen/release.json` 中的路径、字节数和 SHA256。Actor、抽取程序、场景、问卷答案及验收报告通过私有 Render Secret File 交付，不提交 Git，不放入 `ui/` 或其他静态目录。Base64 只是文本编码，不提供加密。

## 私有产物与公开校验描述

先完成并冻结 `output/cooperative_kitchen/v3-id-pilot/manifest.json`。不得为了发布而改写未通过的门槛。公开描述文件在冻结前为 `awaiting_freeze`，构建会明确失败；它不是可部署的占位模型。

在研究者本机、XAI 根目录运行：

```bash
python scripts/cooperative_kitchen/package_secret_release.py \
  --release output/cooperative_kitchen/v3-id-pilot \
  --archive output/cooperative_kitchen/private/v3-id-pilot-release.zip \
  --secret-file output/cooperative_kitchen/private/v3-id-pilot-kitchen_release.b64 \
  --descriptor deployment/cooperative_kitchen/release.json \
  --replace-descriptor --allow-candidate
```

`--allow-candidate` 只允许私有打包候选，不改变 `formal_ready`。本轮通过显式 `internal_pilot` 模式允许内部预实验，服务端仍核验实际研究组件；正式招募必须另过完整门槛。ZIP 和 `.b64` 只可生成到被忽略的 `output/cooperative_kitchen/` 下，权限为 0600；已存在时拒绝覆盖，下一次使用新文件名。`--replace-descriptor` 显式允许原子更新公开描述文件，不替换任何旧私有包或研究数据。

工具要求清单恰好选择十二个产物：Actor、程序、场景、问卷，以及训练、抽取、问答、校准、问卷、协议、远程负载、恢复八份报告。另加 manifest，共十三个 ZIP 成员。公开描述记录这些成员的精确路径、大小和 SHA256，并绑定 ZIP、manifest 与源码 runtime hash。它不包含答案正文、实际状态、API key 或数据库地址。

## Render 设置

将 `.b64` 文件的完整内容复制到厨房服务 Environment → Secret Files，文件名设为 `kitchen_release.b64`。不要复制到源码、终端命令、环境变量或聊天消息。Render 原生 Python 服务会将文件放到 `/etc/secrets/kitchen_release.b64`。所有 Secret Files 的合计上限为 1 MB；打包工具按 1,000,000 字节保守限制单个文件，仍需计入服务已有的其他 Secret Files。[Render Secret Files 文档](https://render.com/docs/configure-environment-variables#secret-files)

厨房服务配置：

| 项目 | 值 |
|---|---|
| Git 仓库 | XAI 主库 |
| Root Directory | 留空，使用仓库根目录 |
| Python | 3.13.2 |
| Build | 下方构建命令 |
| Start | `python -m ui.cooperative_kitchen_server --host 0.0.0.0 --port $PORT` |
| Health Check | `/api/status` |
| `KITCHEN_OUTPUT` | `output/cooperative_kitchen/v3-id-pilot` |
| `KITCHEN_NAMESPACE` | `pilot` |
| `KITCHEN_ENROLLMENT_MODE` | 首次部署设为 `closed` |
| `KITCHEN_ALLOW_SQLITE` | `0` |
| `KITCHEN_SECURE_COOKIE` | `1` |
| `KITCHEN_FREEPLAY_QA` | `0`，匿名试玩不调用付费云端问答 |
| Auto Deploy | Off |

```bash
python -m pip install -r requirements-kitchen.txt && python scripts/cooperative_kitchen/materialize_release.py && python scripts/cooperative_kitchen/verify_deployment.py
```

保留厨房原有 `DATABASE_URL`、`DEEPSEEK_API_KEY`、`KITCHEN_ADMIN_KEY` 和 LLM 配置。密钥只保存在厨房服务，不与物流仓库共享。DeepSeek `deepseek-v4-flash` 仍是滚动模型别名；固定请求字符串不等于固定提供方权重。服务会把这一点记为 `qa_model_snapshot_unpinned`：它是当前内部预实验明确允许的候选缺口，但会阻断 `formal_ready`，更换到可冻结快照后必须重新执行问答与远程验收。

先以 `closed` 部署并完成哈希、数据库、试玩、幂等和会话恢复检查。为了形成首次远程问答证据，可在仍为 `closed` 的短暂维护窗口把 `KITCHEN_FREEPLAY_QA` 临时设为 `1`，只运行受控 canary，完成后立即恢复为 `0` 并重新部署冻结包；不能在公开内部预实验期间保持开启。随后改为 `internal_pilot`，用专门测试用户 ID 复核 A 组真实问答，确认后才向内部参与者开放。Blueprint 默认保持 `closed`；后续手动 Blueprint 同步前注意它会重新应用该默认值，不要在有参与者时误改配置。当前内部预实验必须绑定 `KITCHEN_NAMESPACE=pilot`。未来转为正式收集时，必须冻结一个新的正式发布，并同时把命名空间改为 `confirmatory`、把入口模式改为 `formal`；只切换入口模式会被服务端拒绝。

开启后，参与者输入用户 ID（3–32 位 ASCII 字母、数字、下划线或连字符，以字母开头），无需邀请码。相同 ID 不能从陌生浏览器接管已有会话；原浏览器刷新通过 HttpOnly cookie 恢复，有效研究 cookie 也不会被另一个标签页的自由试玩请求覆盖。A/B 和地图顺序仍由服务端数据库事务内的随机四人区组分配，ID 内容不决定条件。`/api/status.enrollment` 区分 `closed`、`internal_pilot` 与 `formal`，不要将内部入口开启解读为正式验收通过。旧 `/api/admin/invitations` 已返回 410。线上匿名自由试玩不开放 DeepSeek 问答；A 组 Task 1 的研究问答权限不受影响。

物化器先严格解码 Base64，校验整个 ZIP 的 SHA256、大小、精确成员白名单、成员类型、每个文件的 SHA256 及 manifest 关联，再写入权限为 0700 的临时目录，验证后原子改名到目标发布目录。它不调用 ZIP 的通用 `extractall`。绝对路径、路径穿越、重复成员、符号链接、隐藏路径、超限文件、未知成员、额外目录和错误版本均会失败。

若目标发布目录已存在，只有十三个文件全部匹配才视为幂等成功；已有目录不匹配时不会删除或修补，部署失败并保留旧内容。源码与描述、Secret File 不匹配时也失败。构建缓存中有发布目录，仍须提供匹配的 Secret File，不能绕过校验。

本地验证可通过 `--secret-file <本地私有路径>` 显式指定测试文件。应在干净构建目录中测试：训练目录可能含额外验收日志，不能视为已经物化的部署目录。最终 `verify_deployment.py` 会再次验证公开描述、运行时哈希及十三个文件；服务器本身还会检查研究门槛。

## 防止影响物流仓库

两个服务都需要从仓库根目录导入共享 Python 模块，不应将厨房 Root Directory 缩小到 `ui/` 或 `backend/`。厨房 Blueprint 已提供只覆盖厨房及共享依赖的 `buildFilter.paths`，同时关闭自动部署。路径过滤控制自动部署触发，不是文件保密边界，也不会阻止手动部署。[Render monorepo 文档](https://render.com/docs/monorepo-support)

推送共用分支前，核实物流仓库服务的自动部署设置。现有仓库会话在进程内存中，意外重启会清掉活动进度。迁移期间可在 Render 暂停仓库自动部署；若日后使用路径过滤，应只包含仓库前端、环境、adapter、共享 core、`requirements-render.txt`、仓库 Actor 发布目录与 `render.yaml`。本次厨房脚本不修改仓库服务配置。

不要复制独立私有部署目录的全局 `.gitignore`、`.gitattributes`、README 或 `DEPLOYMENT_AUDIT.json` 覆盖 XAI 主库。本方案用公开厨房描述和源码哈希完成构建校验，不依赖旧目录的 README 等文件哈希。

## 发布核验

1. 提交范围只包含公开代码与描述文件；私有 ZIP、Base64、模型题库、真实 `.env`、数据库和日志都保持在 Git 外。
2. 厨房构建应显示物化与校验通过，`/api/status` 的 runtime、Actor、program、manifest、LLM 配置与本机冻结版本一致。
3. 自由试玩验证操作、回放、刷新和幂等；真实问答只由受控的 A 组 canary 用户在 Task 1 验证。在同一发布版本下重启 Render 后确认 PostgreSQL 会话恢复。
4. 原物流仓库没有触发意外部署。若新建厨房服务改变域名，旧域名的浏览器 cookie 不会自动转移，数据库历史仍保留。
5. 正式模式的远程负载、模型抽取等门槛未通过时仍不能称为正式验收通过；内部预实验的单独开启由 `KITCHEN_ENROLLMENT_MODE` 明确记录。正式发布还必须使用新的冻结版本与 `confirmatory` 命名空间，不能在 `pilot` 中原地切换。Secret File 交付成功不等于研究验收通过。
