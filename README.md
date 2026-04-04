# Pac-Man XAI Demo

一个带实时问答解释的 Pac-Man 风格迷宫演示。  
Agent 先收集所有豆子，再冲向出口；过程中会持续规避怪物危险区，并支持中英文自由提问。

## What Changed

- 迷宫目标从“直接到出口”改成了“两阶段任务”
  - 阶段 1：优先收集全部豆子
  - 阶段 2：出口解锁后再冲向终点
- Agent 现在使用多目标 A*
  - 同时考虑最近可达豆子/出口
  - 同时考虑怪物危险区与即时碰撞风险
- 解释系统升级为更自然的中英双语输出
  - 支持动作原因、替代动作、怪物影响、安全评估、目标进度、吃豆策略、局面总结
- UI 改成 Pac-Man 视觉
  - 豆子、锁定/开启出口、鬼怪造型、危险区高亮、右侧状态卡片

## Requirements

- Python 3.11+
- Tkinter

可选但推荐：

- `sentence-transformers`
- `scikit-learn`

说明：

- `QuestionParser` 会优先使用 `sentence-transformers`
- 如果缺少该依赖，会自动降级到 TF-IDF 或规则匹配，不会阻止程序启动

## Run

```bash
py -3 run.py
```

## Controls

- `Start`: 自动运行
- `Pause`: 暂停自动运行并允许提问
- `Resume`: 继续自动运行
- `Step`: 单步执行
- `Reset`: 重新生成迷宫并重开

## Ask Questions

暂停后可以输入中文或英文问题。

示例：

- `Why not go right?`
- `为什么去吃那个豆子？`
- `怪物#2影响了这次决策吗？`
- `Is it safe here?`
- `出口什么时候打开？`

## Explanation Output

每次回答都会输出 3 层内容：

1. `All Evidence (S_t)`
   - 当前时刻可用的结构化证据
2. `Evidence Used (E)`
   - 真正用于生成回答的最小证据子集
3. `Natural-Language Explanation (x)`
   - 面向用户的自然语言解释

同时还会附带 `validation`，用于显示 explanation 数学框架下的检查结果。

## Core Modules

- `environment.py`
  - 生成迷宫、维护豆子/出口/怪物状态
- `agent.py`
  - 多目标 A* 决策器
- `evidence_recorder.py`
  - 记录每一步的结构化证据
- `question_parser.py`
  - 中英双语意图解析与语义匹配
- `explanation_engine.py`
  - 证据选择、最小化、自然语言生成
- `ui.py`
  - Pac-Man 风格 Tkinter 界面
- `run.py`
  - 程序入口

## Typical Flow

1. 启动程序
2. Pac-Man 在迷宫中自动收豆并避开怪物
3. 暂停后输入自然语言问题
4. 系统解析意图
5. 从最近一步证据中筛选 explanation basis
6. 生成中英双语自然语言解释并显示在右侧面板

## Notes

- 当前解释基于最近一步记录的证据，而不是离线日志回放
- 当语义模型依赖缺失时，问句解析仍可运行，但泛化能力会弱一些
