# 长预算开发质量：共享75场景，完整675局，独立重放

这是新版本入口；不修改已花费20k数据、源码或其门槛含义。当前仅 development。全部代码和合成测试均不构成真实模型通过证据。

## 顺序与库存

同一预算三个stream的真实长训练必须先通过独立`long_training_assets` reader，十二个最终程序必须经独立`long_training_final_programs.read_completed`通过，才允许`prepare_cohort`进行场景算术或创建输出。缺任何收据/真实退出/完整性/资格均停止；不能补写passed JSON，也不能为失败流切模型或另开质量集。

一个预先封存cohort含75个独立场景，各task25；三个stream都使用完全相同case列表与原始scenario/action参数。每stream跑reference/raw/regularized共225局，全cohort675次有限reset。reference的三次重复明确为配对重复，独立场景数始终75。各stream独占claim，cohort同预算独占claim，均中断不自动重试。

CLI `run_warehouse_v642_long_quality.py --prepare-cohort`显式接收三组`--stream COMPLETED_ASSETS FINAL_PROGRAMS`、`--run-id`、`--base-start`、`--action-start`。生成三个`plan.json`；任选一个调用`--audit-seeds --plan`为整个cohort做一次四根indexed→triage→exact审计。声明绑定全部675槽位，reuse_tag绑定cohort内容哈希，24小时clearance严格绑定源码和实际历史原参数。后续`--collect --plan`必须由新的long-stage supervisor实际wait监督。

历史只排除该cohort准确注册输出、唯一cohort claim和三个stream claim；不豁免其他历史delta。排除目录内不接受未知文件、symlink或其他run。seed审计自身输出在clearance中逐文件提交，之后只读验证。每个stream开头和结尾复查外部历史，逐reset写只读预授权和原reset hook日志；独立审计重建每个有限slot。

## 执行、指标与审计

两个动作只来自绑定完整NN及物理掩码。每帧保存公开观察、完整掩码后分布、采样动作及物理前后状态；额外保存私有初始/结束环境与RNG，仅用于物理replay，绝不作为NN输入。独立reader重新计算两个NN动作和分布，并从已有初始快照重放相同物理动作/奖励/终止，不构造新环境、不reset、不分配新场景。

raw/reg分别在全部75场景判质量：配送≥5.5且≥同场景reference的95%，碰撞≤min(6,reference+1)，停机≤5%；每task只分层报告，不新增每task阈值。最终四程序固定资格、不refit，逐角色F/E覆盖≥45%、预期反馈≥40%、一致率≥95%/97%、KL≤.20、非法/未证明路径0。程序不选执行动作。

`audit(result_ref,supervisor_exit_ref)`独立重放，输出`warehouse-v642-independent-long-development-pair-gate.v1`：phase/stream/budget/seed_plan/cohort、quality_cases_sha256、task_metrics、program_metrics和完整input_asset_refs。`development_pair_gate_passed`由实测重算；失败也保留完整有效receipt，不能伪造成功。再调用audit必须得到相同对象。脚本`audit_warehouse_v642_long_quality.py`独占写只读审计收据。

单流通过仍不签发budget lock、不授权formal/封存集、不开放pilot。后继预算锁reader须验证同budget三流同cases_sha、全三流结果及更小预算完整失败历史；原正式准入、一次性程序审计、留出和三域发布门继续独立执行。
