# v6.42 去重加速与实测状态

2026-10-23。科学阈值未改；本报告不授予发布。

## 已测加速

| 相同工作 | 原执行 | 去重执行 | 校验结果 |
|---|---:|---:|---|
| 完整历史事件控制及文件SHA | 66.912秒 | 5.427秒 | 全部控制字段、类型、float.hex一致 |
| 文件请求 | 407,360次 | 6,489个不同文件各完整读取 | 冲突SHA拒绝；不使用stat缓存 |

这部分为12.33倍；不是端到端训练或验收速度。原流程每次重复读取约118.3GB，而不同文件实际约5.73GB。新程序仍检查全部不同文件。

原407+6源码未改。加速恢复v3另立源快照、claim和执行记录；checkpoint/event实际调用原v2载体，执行器绑定明确为v3。45项相关软件测试已通过，另补macOS解释器实际exec路径回归通过（共11项cutover测试）。这不是模型验收证据。

## 已有科学指标：冻结v6.20 NN＋v6.42抽取程序

| 完整episode评估 | 配送/局 | 碰撞/局 | 停机episode |
|---|---:|---:|---:|
| 全新确认300局 | 14.89667 | 2.95000 | 3/300＝1.00000% |
| 压力216局，72独立场景 | 14.82407 | 3.02315 | 8/216＝3.70370% |

确认集六个角色/折：解释覆盖48.5290%—53.9837%；反馈覆盖49.1676%—58.7663%；预期反馈47.0096%—55.7420%；反馈一致率最低97.8557%，解释一致率最低98.1057%；KL最大0.086198，非法动作和未证明路径均0。全部达到登记的汇总开发门槛。

压力task3停机5/72＝6.94444%，保留此分层结果；不能说每个task均低于5%。[完整表](warehouse_v642_confirmation_results.md)及原始summary/metrics/audit链接见该表。

## 当前20k训练尚待完成

旧v2任务只有原17824更新的精确重放提交，pending776帧无新环境样本或新提交；独立cutover核验后原监督器真实wait(-15)。旧文件保留。加速v3于08:46UTC另立监督任务，仍受原11:53UTC截止约束。

- 当前新运行：`output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/recovery/warehouse-v642-calibration-02-recovery-04`
- cutover收据：`output/warehouse_v642/checkpoint-cutovers/accelerated-recovery-cutover-05/cutover-completed.json`
- source contract：`output/warehouse_v642/engineering/accelerated-recovery-source-05/source-contract.json`
- 原v2控制台文件仅在真实退出后chmod444，内容SHA未变；封存说明在source目录的`console-permission-seal.json`。
- 第一次prepare因该控制台权限拒绝，尚未发claim/构造模型/运行训练，残留recovery-03保留；实际v3使用新的recovery-04目录。没有覆盖失败证据或重启科学流。

上述确认指标不是20k新权重结果。完成后还需独立更新/权重链审计、最终双角色程序预检和新质量评估，才能填写raw/regularized的训练后指标。pilot保持关闭。
