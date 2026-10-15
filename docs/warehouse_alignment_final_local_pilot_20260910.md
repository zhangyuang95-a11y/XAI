# 物流仓库最终本地预实验版本（2026-09-10）

当前入口为 **[http://127.0.0.1:8012/](http://127.0.0.1:8012/)**。这是已经通过模型、解释、题库和保存证据技术验收的本地 A/B 预实验版本；`formal_ready=false`，因为解释对真人合作表现的效果仍须通过预实验和正式随机实验验证。

## 最终冻结组件

- NN Actor：累计 395 万联合训练步，两层 128 单元 MLP，输出 `UP / DOWN / LEFT / RIGHT / WAIT`。运行时动作直接来自冻结 NN；无动作掩码、输出后覆盖、动作重选或程序接管。墙、碰撞和电量只由公开环境动力学解析。
- Actor SHA-256：`309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b`。
- 能力验收：熟练／主动／扰动搭档下，NN 平均配送 `8.72 / 8.40 / 8.80`，团队平均配送 `13.38 / 14.84 / 12.50`；17,856 个保存的神经动作全部原样提交，覆盖次数为 0。
- RCPD 反馈确实参与过 NN 更新：390 万步分支以 `lambda=0.2` 运行 2 万步，保存非零 KL、反馈梯度和 PPO 更新证据。后续保护门在能力下降时关闭约束；最终新增 5 万步没有伪称仍在施加反馈。
- 最终 RCPD 程序：深度 12、最多 256 叶。独立验收的普通动作一致率为 `92.371%`，非等待动作 `91.870%`；窄通道、共享取货、共享充电分别为 `95.929% / 89.601% / 91.041%`，两个角色分别为 `91.970% / 92.772%`。
- 反事实证据由同一冻结 NN 的隔离重放负责：3,556 个动作案例与 215 个有效方向变化案例均与独立实现一致，真实回合状态、随机状态和动作提交轨迹保持不变。
- 旧的“只靠同一棵树预测反事实方向”诊断仍为 `57.209%`，明确保留为非门控失败。系统不会把树分支冒充神经网络的确定因果原因；树负责普通行为近似，反事实结论来自冻结 NN 重放。
- 自由问答审计覆盖 40 道中英文问题 × 12 个独立场景，共 480 个案例；原因、替代动作、反事实、失败、规则和澄清六类均经独立生成／验证双阶段逐条通过。

## 实验流程

页面按用户 ID 登记并以四人区组覆盖 `A/XY、A/YX、B/XY、B/YX`。流程为共同练习、Task 1 三局、Task 2 三局、4 道下一动作题、4 道三步预测题和 3 项量表。A 组只在 Task 1 进行中及回顾时可解释；B 组只看相同回放；Task 2 两组均不能读取新解释、旧答案或缓存响应。两组使用同一个冻结 Actor。

主要指标预先固定为 Task 2 三局配送数的参与者级算术平均；同时保留原始得分、碰撞、断电和效率。Task 1/Task 2 分差只作描述，不把技术测试或小样本预试称为解释效果。

## 最终技术回归

- 共 161 项相关测试通过，覆盖冻结 Actor 与纯 NN 动作权限、环境动力学、解释与问答、版本绑定、存储、A/B 阶段权限、同意先于 ID 登记和分析导出。
- 四名独立技术参与者完成真实 HTTP A/B 全流程，恰好覆盖 `A/XY、A/YX、B/XY、B/YX`：共完成 24 个正式任务局、2,900 个已确认游戏步和 3,136 个 HTTP 请求；A 组在许可阶段完成并展示了 4 个中英文问答，Task 2 权限限制保持生效。该流程只用于技术验收，不属于正式样本。
- 流程中发生过一次客户端未收到响应，但对应操作已经提交。使用相同 `operation_id` 和请求摘要恢复后，服务端返回既有结果，确认状态只前进一次，没有重复推进。此前各次失败运行及其 `failure.json` 均原样保留，没有被成功续跑覆盖。
- 操作员确实停止并重新启动了服务，随后以同一发布清单和数据库恢复四个会话；重启核对期间新增游戏动作和问题均为 0。进程检查器本身没有独立观察停止与启动的完整生命周期，因此该项准确表述为“操作员执行的真实进程重启后恢复验证”，不作为独立生命周期监控证据。
- HTTP 续跑报告见 [report.json](../output/warehouse_native/alignment_local_pilot_http_resume_395m_20260910/report.json)，重启恢复报告见 [post_restart/report.json](../output/warehouse_native/alignment_local_pilot_http_resume_395m_20260910/post_restart/report.json)。以上结果不改变发布状态：`formal_ready=false`，真人解释效果仍待实验验证。
- 首轮浏览器验收发现完整参与说明和同意框在用户 ID 登记后才出现。r2 已将记录范围、数据保留、去标识提醒和同意框提前到登记页，并增加服务端显式同意门控。未勾选时浏览器不发送命令，直接调用服务端也会在绑定 ID 前被拒绝且状态版本不变。1365×900 中文和 1280×800 英文复测均完整落在首屏、无横向溢出；原 F-001 已关闭。见 [浏览器 delta 报告](../output/warehouse_native/alignment_local_pilot_browser_qa_r2_395m_20260910/delta_qa_report.json)和[服务端 delta 报告](../output/warehouse_native/alignment_local_pilot_consent_qa_r2_395m_20260910/consent_delta_report.json)。完整 HTTP 流程绑定 r1；r2 仅改变 `ui/warehouse_alignment_server.py`，其余冻结组件与证据哈希逐项不变。

## 发布与复现

不可变发布清单：

- [r2 manifest.json](../output/warehouse_native/alignment_local_pilot_release_r2_395m_20260910/manifest.json)，SHA-256 `a2d85304982cae72e2d33f80fb21a555f7db4f410914270d19139d0bb857eb7d`
- [r1 manifest.json](../output/warehouse_native/alignment_local_pilot_release_395m_20260910/manifest.json)，SHA-256 `921a3b6de163cf7175a579038d289e59ce865797151b735c6266d71eed0d2277`（保留用于对应完整 HTTP 验收）
- [evidence bundle](../output/warehouse_native/alignment_local_pilot_evidence_bundle_395m_20260910.json)，SHA-256 `70840c2846c969f66000589c8d7441a544e834dcf569fce55c53e378490ed335`
- [系统解释验收](../output/warehouse_native/alignment_explanation_system_acceptance_v3_395m_20260910/report.json)
- [自由问答验收](../output/warehouse_native/alignment_diverse_answer_audit_395m_20260910/answer_receipt.json)
- [能力验收](../output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/validation/step_0050000/report.json)

启动命令：

```sh
python -m ui.warehouse_alignment_server \
  --release-root output/warehouse_native/alignment_local_pilot_release_r2_395m_20260910 \
  --expected-manifest-sha a2d85304982cae72e2d33f80fb21a555f7db4f410914270d19139d0bb857eb7d \
  --db output/warehouse_native/alignment_local_pilot_runtime_final_r2_395m_20260910/study.sqlite3 \
  --port 8012
```

没有推送 GitHub 或部署线上；旧权重、旧失败、8011 盲测入口和历史数据均保留。
