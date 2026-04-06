# Pac-Man RL + XAI Demo

- [中文](#中文)
- [English](#english)

## 中文

### 项目

- DQN 强化学习 agent：`RLAgent`
- A* 基线 agent：`HeuristicAgent`
- explanation 定义验证脚本：`validate_explanations.py`
- 双语问答界面：`run.py`

### 安装

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

### 训练

默认训练并保存最佳模型：

```bash
py -3 train_rl.py
```

常用示例：

```bash
py -3 train_rl.py --episodes 3000 --grid-size 21 --num-monsters 8
py -3 train_rl.py --episodes 500 --grid-size 15 --num-monsters 4
py -3 train_rl.py --no-show-plot --no-show-game
```

输出文件：

- `models/dqn_pacman.pt`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

### 运行

推荐流程：

1. 先完成 RL 训练，得到 `models/dqn_pacman.pt`
2. 运行 `py -3 run.py`
3. 点击 `Start / 开始` 进入自动模式
4. 运行过程中可以随时点击 `Pause / 暂停` 提问
5. 也可以直接输入问题，系统会自动暂停并回答
6. 每次真实提问都会自动保存到 `artifacts/user_question_log.jsonl`

自动模式：

```bash
py -3 run.py
```

说明：

- 有 `models/dqn_pacman.pt` 时优先加载 RL
- 没有模型时回退到启发式 agent
- 自动模式下用户可以随时暂停并提问
- 用户真实提问的回答和 validation 结果会自动记录

显式指定 agent：

```bash
py -3 run.py --agent rl --model-path models/dqn_pacman.pt
py -3 run.py --agent heuristic
```

### 验证 Explanation 定义

主流程是: 训练 RL 后，用户在自动模式里真实提问，系统基于当前状态回答，并记录 validation 结果。

真实提问日志：

- `artifacts/user_question_log.jsonl`

下面这个脚本只是开发测试工具，不是主要交互流程。

让 RL 智能体连续跑多局游戏，自动抽取一些决策时刻，检查系统给出的回答是否满足 explanation 定义：

```bash
py -3 validate_explanations.py --agent rl --model-path models/dqn_pacman.pt --episodes 20
```

输出文件：

- `artifacts/explanation_validation.json`

验证项：

- `E ⊆ S_t`
- `Basis_{u,t}(E, Q)`
- `Minimal(E)`
- `x = R_u(E, Q)`
- `Readable_u(x)`
- `Explain_u(Q, t, x)`

### 常见问题

- `Why not go right?`
- `为什么去吃那个豆子？`
- `怪物#2影响了这次决策吗？`
- `Is it safe here?`

### 主要文件

- `agent.py`
- `environment.py`
- `train_rl.py`
- `training_game_viewer.py`
- `run.py`
- `validate_explanations.py`
- `explanation.py`
- `explanation_engine.py`
- `question_parser.py`
- `ui.py`

### 备注

- RL 模型应和运行时 `grid_size` 匹配
- 长时间训练建议使用 `--no-show-game`
- `models/`、`artifacts/` 默认不提交

## English

### Project

- DQN RL agent: `RLAgent`
- A* baseline agent: `HeuristicAgent`
- explanation-definition validator: `validate_explanations.py`
- bilingual QA demo UI: `run.py`

### Install

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

### Train

Train and save the best checkpoint:

```bash
py -3 train_rl.py
```

Common examples:

```bash
py -3 train_rl.py --episodes 3000 --grid-size 21 --num-monsters 8
py -3 train_rl.py --episodes 500 --grid-size 15 --num-monsters 4
py -3 train_rl.py --no-show-plot --no-show-game
```

Output files:

- `models/dqn_pacman.pt`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

### Run

Recommended flow:

1. Train the RL model first and get `models/dqn_pacman.pt`
2. Run `py -3 run.py`
3. Click `Start / 开始` to enter auto mode
4. Pause at any time to ask a question
5. You can also type a question directly and the UI will pause automatically before answering
6. Each real user question is saved to `artifacts/user_question_log.jsonl`

Auto mode:

```bash
py -3 run.py
```

Behavior:

- load RL first if `models/dqn_pacman.pt` exists
- otherwise fall back to the heuristic agent
- in auto mode, the user can pause and ask questions at any time
- answers and validation results for real user questions are logged automatically

Explicit agent selection:

```bash
py -3 run.py --agent rl --model-path models/dqn_pacman.pt
py -3 run.py --agent heuristic
```

### Validate The Explanation Definition

The main workflow is: train the RL model, let the user ask real questions in auto mode, and record the validation result for each answer.

Real question log:

- `artifacts/user_question_log.jsonl`

The script below is a developer test tool, not the main interaction flow.

Let the RL agent play multiple episodes, sample decision points, and check whether the generated answers satisfy the explanation definition:

```bash
py -3 validate_explanations.py --agent rl --model-path models/dqn_pacman.pt --episodes 20
```

Output file:

- `artifacts/explanation_validation.json`

Checks:

- `E ⊆ S_t`
- `Basis_{u,t}(E, Q)`
- `Minimal(E)`
- `x = R_u(E, Q)`
- `Readable_u(x)`
- `Explain_u(Q, t, x)`

### Example Questions

- `Why not go right?`
- `为什么去吃那个豆子？`
- `怪物#2影响了这次决策吗？`
- `Is it safe here?`

### Main Files

- `agent.py`
- `environment.py`
- `train_rl.py`
- `training_game_viewer.py`
- `run.py`
- `validate_explanations.py`
- `explanation.py`
- `explanation_engine.py`
- `question_parser.py`
- `ui.py`

### Notes

- The RL checkpoint should match the runtime `grid_size`
- Use `--no-show-game` for long training runs
- `models/` and `artifacts/` are ignored by default
