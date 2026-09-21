> Current three-domain release: [Kitchen v3.8.1 rules, tutorial, persistence and deployment status](docs/three_domain_deployment_v3_8.md). The sections below describe an earlier study version; use the versioned release documents for current behavior.

# Warehouse 与 Pong 三任务试玩

这个应用包含协作配送（Warehouse）和合作接球（Pong）。参与者控制机器人 1，机器人 2 使用各环境现有的冻结 NN 与规则辅助控制器。两组在同一任务中使用相同种子、机器人控制器和计分规则；解释仅改变 A 组在 Task 2 能看到和询问的信息。

| 任务 | A 组 | B 组 | Warehouse 种子 | Pong 种子 |
| --- | --- | --- | ---: | ---: |
| Task 1：无解释基线 | 无解释 | 无解释 | 52000 | 260920 |
| Task 2：解释对比 | 点击“为什么？”及提问；Pong 可暂停并回放提问 | 无解释 | 51000（原 Task 1） | 260918（原 Task 1） |
| Task 3：无解释迁移 | 无解释 | 无解释 | 51500（原 Task 2） | 260919（原 Task 2） |

每个任务只有一局。Warehouse 每局最多 120 步；Pong 每局 90 秒有效游戏时间，暂停不计时。每局结束后点击按钮进入下一任务；最终 1–5 分问卷在 Task 3 后出现。Pong 用 `S` 暂停或继续；只有 A 组在 Task 2 暂停或局后能选择帧、时间段提问。Warehouse 默认中文，可点击右上角切换英文。

Warehouse 的 AI–AI 演示仍使用固定种子 40786 的真实 120 步轨迹。网页一次载入后，每步播放 0.6 秒，完整播放约 72 秒；可暂停、继续、重播或提前进入 Task 1。切到后台会暂停，刷新后从上一已完成的演示步骤继续。

Warehouse 参与者计分 `warehouse-participant-score.v2`：每件配送 +10、每次机器人碰撞 −10、每台断电机器人 −5、每次违规占用共享充电桩 −5、每步 −1、每单位参与者绕路 −2。提前断电结束仍会补计剩余步骤的时间成本。旧记录保留原分值；没有 `score_version` 字段的记录按旧版处理，不能把新旧总分直接合并比较。

Pong 每局结束显示该局漏球成绩，Task 3 问卷前及提交后的完成页显示三局表格。小球每漏接一次计 1，大球每漏接一次计 3；“大球漏接次数”是事件次数，“加权漏球数”是小球次数加大球次数的三倍。

## 本地启动

在检出本分支的仓库根目录执行（当前本地工作目录为 `/private/tmp/xai-render-pong-hub`）：

```bash
python3 -m pip install -r requirements-render.txt
python3 -m ui.domain_hub_server --host 127.0.0.1 --port 18765
```

打开 [环境选择页](http://127.0.0.1:18765/)，选择 [Warehouse](http://127.0.0.1:18765/warehouse/) 或 [Pong](http://127.0.0.1:18765/pong/)。Pong 不能直接双击 HTML 文件试玩，因为浏览器无法从 `file://` 加载模型。本次演示、成绩和计分修改只在本地分支，尚未部署。

## 记录与限制

三任务流程版本为 `three-task-explanation.v1`，种子与解释权限定义在 [配置文件](configs/three_task_study.json)。Warehouse 提交最终问卷后，会将三局成绩、计分版本、提问记录和问卷原子保存到 `output/study_records/warehouse/three-task-explanation.v1/`；进行中的会话只在服务器内存，刷新同一标签页可继续，服务器重启后无法恢复。Render 本地磁盘不是持久存储，正式收集数据前需配置持久化导出。Pong 完成三局并提交问卷后，将记录保存在本机浏览器的 `localStorage`，完成页的恢复指针保存在本标签页的 `sessionStorage`；刷新完成页仍可看三局成绩，另开新标签页不能恢复该完成页。进行中的 Pong 局会用同一种子从头开始，精确物理帧和局后回放不能跨刷新恢复。旧两任务记录不改名、不与新记录合并。

本次没有重新训练模型。三局种子不同，不能把同一个人跨任务的成绩差异直接当作解释效果。历史 Warehouse 说明保留在 [旧版说明](docs/warehouse_legacy_readme_20260919.md)，Pong 训练说明见 [Pong 训练 README](domains/pong/training/README.md)。
