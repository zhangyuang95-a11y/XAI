# 厨房研究运维工具

这些命令只在显式执行时操作本机文件或指定的服务，不自动发布、上传、发消息或创建定时任务。

- `package_release.py` 默认按 `output/cooperative_kitchen/v3-id-pilot/manifest.json` 的白名单与哈希生成权限为 0600 的私有发布 ZIP。候选包需要 `--allow-candidate`，不会解锁研究入口。包内含问卷答案与抽取程序，**不得放进公开 GitHub、公开下载链接或前端静态目录**。历史 v1/v2 复现必须显式指定 `--release` 与新输出文件名，不覆盖旧包。
- 新 ID 预实验发布使用 `--release output/cooperative_kitchen/v3-id-pilot`，保留 v1/v2 产物。历史 `build_deployment.py` 仍支持旧独立私有部署目录；当前公开主库改用下述 Secret File。`verify_deployment.py` 是部署构建阶段的只读源码与产物哈希校验。
- `postgres_backup.py` 使用一致性 PostgreSQL 快照生成归档和校验清单。整个数据库备份含会话凭据，必须私有保存。
- `postgres_restore.py` 校验归档，只恢复到另一个空数据库；拒绝覆盖源库或非空目标。
- `study_admin.py` 从环境或私有配置文件读取管理员密钥，查看入口状态、导出分析文件或提交有审计原因的技术重试。参与者直接输入用户 ID，旧邀请码命令已移除，不代注册或发送消息。
- `browser_fixture_server.py --isolated-test-only` 仅在本机 PostgreSQL 新建临时测试 schema，以真实环境、Actor、场景、问卷运行完整浏览器回归。只在 `namespace=test` 中绕过发布门槛并固定第一块 A/B 顺序；不预注册用户 ID，浏览器通过真实入口首次登记。不会改正式发布清单、生产随机分组或简化物理。退出后移除自己创建的 schema。
- `normalize_canary.py` 默认只检查事先去敏的远程 canary JSON；显式 `--write` 才备份并更新恢复/远程报告，不读取密钥、不联网、不改变二十人门槛。字段契约和两阶段更新顺序见 [normalize_canary.md](normalize_canary.md)。

完整命令和验收边界见 `docs/cooperative_kitchen_research.md`。DeepSeek 的 `deepseek-v4-flash` 是滚动别名，不声称权重已冻结。不把本地测试、候选打包或协议夹具结果称为远程部署、模型效果或真实云端语言模型验收。

当前主库部署采用 `docs/cooperative_kitchen_monorepo_deployment.md`：`package_secret_release.py` 生成私有 Base64 ZIP 与不含正文的公开 SHA256 描述；`materialize_release.py` 从 Render Secret File `/etc/secrets/kitchen_release.b64` 严格校验并原子物化到 `output/cooperative_kitchen/v3-id-pilot`，随后运行 `verify_deployment.py`。公开 XAI 仓库不提交私有包、模型题库或数据库。旧独立私有目录工具保留，仅用于历史版本复现。
