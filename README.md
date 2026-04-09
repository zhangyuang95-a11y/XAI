# Pac-Man RL + XAI Demo

This project is a small Pac-Man demo for learning and experimentation. You can train an RL agent, watch it play in a live UI, and ask why it made a decision. The question-and-answer part supports both Chinese and English.

- [中文](#chinese)
- [English](#english)

<a id="chinese"></a>
## 中文

### 项目简介

这是一个简单易懂的 Pac-Man 演示项目。

你可以先训练一个 RL 智能体，再打开图形界面看它自动玩游戏。运行过程中，你可以直接提问，比如“为什么不往右走？”系统会结合当前局面给出解释。提问支持中文和英文。

### 这个项目能做什么

- 训练一个会玩 Pac-Man 迷宫的 RL 智能体
- 在图形界面里观看智能体自动行动
- 在游戏进行时随时暂停并提问
- 根据当前状态生成通俗的解释
- 记录真实用户提问和解释结果

### 安装

建议使用 Python 3。

安装依赖：

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

说明：

- `torch` 用来训练和加载 RL 模型
- `matplotlib` 用来保存训练曲线
- `sentence-transformers` 和 `scikit-learn` 用来理解用户问题
- 图形界面使用 `tkinter`，一般 Python 自带

### 快速开始

第一次使用时，按下面顺序操作：

1. 安装依赖
2. 训练模型：`py -3 train_rl.py`
3. 运行演示：`py -3 run.py`
4. 点击 `Start / 开始` 看智能体自动运行
5. 需要提问时，点击 `Pause / 暂停`，或者直接输入问题

如果你还没有训练好的模型，程序会自动回退到启发式 agent，所以界面仍然可以打开。

### 训练模型

最常用的命令：

```bash
py -3 train_rl.py
```

这条命令会训练模型，并默认输出这些文件：

- `models/dqn_pacman.pt`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

常用示例：

```bash
py -3 train_rl.py
```

简单说明：

- 训练配置已经固定为 `11x11 + 2 个怪物 + 1500 回合`
- 脚本不再给用户暴露运行时选择项
- 训练指标现在会同时记录 `eval_stage` 和 `eval_final`
- 当 `eval_final` 连续两次达到收敛门槛时，训练会自动提前停止
- 现在推荐流程就是训练这一个固定模型，然后运行这一个固定演示

现在脚本不再提供面向用户的训练参数选项。

### 运行演示

启动界面：

```bash
py -3 run.py
```

常用运行方式：

```bash
py -3 run.py
```

现在演示只保留一种固定模式：训练好的 RL 模型、`11x11` 地图、`2` 个怪物，以及人工输入问题。程序会从 checkpoint 里读取匹配的 reward preset 和 step limit。

运行时你可以做这些事：

- 点击 `Start / 开始` 进入自动模式
- 点击 `Pause / 暂停` 暂停游戏
- 点击 `Resume / 继续` 继续游戏
- 直接输入中文或英文问题
- 查看系统给出的解释和验证结果

真实用户问题会自动保存到：

- `artifacts/user_question_log.jsonl`

### 可解释性问答是怎么工作的

这个项目的思路并不复杂，可以简单理解为三步：

1. 系统先记录当前局面，比如玩家位置、出口方向、怪物距离、豆子数量和可选动作
2. 系统读取你的问题，判断你是在问“为什么这样走”“为什么不那样走”还是“现在安全吗”
3. 系统从当前证据里挑出最相关的信息，生成一段自然语言解释

所以这里的解释，不是随便编出来的，而是尽量基于当前游戏状态来回答。

### 可选：验证 explanation 的脚本

这部分更偏开发和测试，不是主流程必须要跑。

它会让 agent 连续玩多局游戏，并检查生成的解释是否满足当前项目里的 explanation 规则。

命令：

```bash
py -3 validate_explanations.py --agent rl --model-path models/dqn_pacman.pt --episodes 20
```

输出文件：

- `artifacts/explanation_validation.json`

如果你只是想体验项目，可以先不运行这一步。

### 主要文件说明

- `train_rl.py`：训练 RL 模型
- `run.py`：启动图形界面和问答演示
- `validate_explanations.py`：批量检查解释质量
- `agent.py`：定义 RL agent 和启发式 agent
- `environment.py`：游戏环境和移动规则
- `question_parser.py`：解析用户问题
- `explanation_engine.py`：根据证据生成解释
- `evidence_recorder.py`：记录当前局面的关键证据
- `ui.py`：界面逻辑

### 常见提醒

- RL 模型最好和训练时的地图设置保持一致
- `models/` 和 `artifacts/` 默认不提交到 Git
- 如果没有训练好的模型，`run.py` 会自动使用启发式 agent
- 训练太慢时，可以减少 `--episodes`，或者关闭实时窗口

<a id="english"></a>
## English

### Project Overview

This is a beginner-friendly Pac-Man demo.

You can train an RL agent, open a live UI, and watch it play automatically. While the game is running, you can ask questions such as "Why not go right?" and the system will answer based on the current game state. Questions can be entered in either Chinese or English.

### What This Project Can Do

- Train an RL agent to play the Pac-Man maze
- Show the agent in a live desktop UI
- Let you pause the game and ask questions at any time
- Generate simple explanations from the current state
- Save real user questions and explanation results

### Install

Python 3 is recommended.

Install dependencies:

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

Notes:

- `torch` is used for training and loading the RL model
- `matplotlib` is used for training charts
- `sentence-transformers` and `scikit-learn` help the system understand user questions
- The UI uses `tkinter`, which usually comes with Python

### Quick Start

For a first run, follow this order:

1. Install dependencies
2. Train a model: `py -3 train_rl.py`
3. Start the demo: `py -3 run.py`
4. Click `Start / 开始` to watch the agent play
5. Click `Pause / 暂停`, or just type a question when you want an explanation

If you do not have a trained model yet, the app will fall back to the heuristic agent, so the UI can still run.

### Train The Model

Most common command:

```bash
py -3 train_rl.py
```

This command trains a model and saves these files by default:

- `models/dqn_pacman.pt`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

Common examples:

```bash
py -3 train_rl.py
```

Simple notes:

- The training setup is fixed at `11x11 + 2 monsters + 1500 episodes`
- The script no longer exposes runtime configuration choices
- Training metrics now include both `eval_stage` and `eval_final`
- The trainer can stop early once `eval_final` reaches the convergence target twice in a row
- The intended workflow is to train this one fixed model and then run the fixed demo

There are no user-facing training options in the script now.

### Run The Demo

Start the UI:

```bash
py -3 run.py
```

Common run modes:

```bash
py -3 run.py
```

The demo now runs in one fixed mode only: trained RL model, `11x11` map, `2` monsters, and manual question input. The app reads the saved checkpoint metadata for the matching reward preset and step limit.

While the app is running, you can:

- Click `Start / 开始` to enter auto mode
- Click `Pause / 暂停` to pause the game
- Click `Resume / 继续` to continue
- Type a question in Chinese or English
- Read the generated explanation and validation result

Real user questions are saved to:

- `artifacts/user_question_log.jsonl`

### How The Explanation QA Works

The idea is simple and can be understood in three steps:

1. The system records the current state, such as player position, exit direction, monster distance, remaining dots, and available actions
2. It reads your question and decides whether you are asking about the chosen move, an alternative move, safety, or something else
3. It selects the most relevant evidence from the current state and turns it into a natural-language explanation

So the explanation is meant to come from the current game evidence, not from a random answer.

### Optional: Validate The Explanation Script

This part is mainly for development and testing. It is not required for the main demo flow.

It lets the agent play multiple episodes and checks whether the generated explanations satisfy the explanation rules used in this project.

Command:

```bash
py -3 validate_explanations.py --agent rl --model-path models/dqn_pacman.pt --episodes 20
```

Output file:

- `artifacts/explanation_validation.json`

If you only want to try the project, you can skip this step at first.

### Main Files

- `train_rl.py`: train the RL model
- `run.py`: start the UI and QA demo
- `validate_explanations.py`: batch-check explanation quality
- `agent.py`: RL agent and heuristic agent
- `environment.py`: game environment and movement rules
- `question_parser.py`: parse user questions
- `explanation_engine.py`: generate explanations from evidence
- `evidence_recorder.py`: store key evidence from the current state
- `ui.py`: desktop UI logic

### Common Reminders

- The RL checkpoint should stay consistent with the map setup used for training
- `models/` and `artifacts/` are ignored by Git by default
- If no trained model is found, `run.py` falls back to the heuristic agent
- If training feels slow, lower `--episodes` or disable the live window
