# Pac-Man RL + XAI Demo

Choose your language:

- [中文](#中文说明)
- [English](#english)

---

## 中文说明

这是一个 Pac-Man 风格的强化学习与可解释 AI 演示项目。项目同时提供：

- 一个可训练的 DQN 强化学习 agent
- 一个作为对照基线的启发式 A* agent
- 一个可以回答“为什么这样走”的解释系统
- 一个可视化训练界面和游戏演示界面

### 项目特点

- `RLAgent`
  - 使用 `train_rl.py` 训练出的 DQN 模型进行推理
- `HeuristicAgent`
  - 使用多目标 A* 作为无训练基线
- 双语问答
  - 支持中英文问题解析与解释生成
- 可视化训练
  - 支持训练曲线图
  - 支持训练中的实时 Pac-Man 游戏窗口
- XAI 输出
  - 输出证据集合、使用证据和自然语言解释

### 强化学习部分包含什么

- 环境接口
  - `reset_rl()`
  - `step_rl(action) -> observation, reward, done, info`
- observation 编码
  - 墙体
  - 可通行路径
  - Pac-Man 位置
  - 出口位置与开启状态
  - 豆子分布
  - 怪物分布
  - 危险区域
  - 进度与距离等标量特征
- reward shaping
  - 每步惩罚
  - 非法动作惩罚
  - 原地停留惩罚
  - 无效拖延惩罚
  - 随步数增加的时间压力惩罚
  - 吃豆奖励
  - 解锁出口奖励
  - 靠近当前目标奖励
  - 远离当前目标惩罚
  - 靠近怪物惩罚
  - 胜利奖励
  - 快速通关额外奖励
  - 失败惩罚
  - 超时惩罚
- 训练流程
  - experience replay
  - epsilon-greedy exploration
  - target network
  - checkpoint 保存
  - 周期性 evaluation

### 主要文件

- `environment.py`
  - 游戏环境
  - RL 状态编码与 reward 逻辑
- `agent.py`
  - `HeuristicAgent`
  - `RLAgent`
  - `DQNNetwork`
- `train_rl.py`
  - DQN 训练入口
- `training_game_viewer.py`
  - 训练时的实时游戏窗口
- `run.py`
  - 游戏演示入口
- `ui.py`
  - Pac-Man 风格图形界面
- `question_parser.py`
  - 中英双语问题解析
- `explanation_engine.py`
  - 自然语言解释生成
- `evidence_recorder.py`
  - 证据记录与回放

### 安装依赖

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

### 训练模型

默认训练 `3000` 个 episode，并保存最佳模型到 `models/dqn_pacman.pt`：

```bash
py -3 train_rl.py
```

训练时默认会同时显示：

- 训练曲线
- 训练过程中的实时游戏窗口

同时还会持续保存：

- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

常用参数示例：

```bash
py -3 train_rl.py --episodes 3000 --grid-size 21 --num-monsters 8
py -3 train_rl.py --episodes 500 --grid-size 15 --num-monsters 4
py -3 train_rl.py --no-show-plot
py -3 train_rl.py --no-show-game
```

训练日志会输出：

- 当前 episode reward
- 滚动平均 reward
- 胜率
- 平均步数
- epsilon
- loss
- evaluation 结果

### 运行演示

自动模式：

```bash
py -3 run.py
```

规则如下：

- 如果 `models/dqn_pacman.pt` 存在，默认优先加载 RL 模型
- 如果模型不存在，则回退到启发式 agent

显式运行 RL 模型：

```bash
py -3 run.py --agent rl --model-path models/dqn_pacman.pt
```

显式运行启发式基线：

```bash
py -3 run.py --agent heuristic
```

### 可以问的问题

暂停后可以继续提问，例如：

- `Why not go right?`
- `为什么去吃那个豆子？`
- `怪物#2影响了这次决策吗？`
- `Is it safe here?`
- `出口什么时候打开？`

解释系统会输出三层内容：

1. `All Evidence (S_t)`
2. `Evidence Used (E)`
3. `Natural-Language Explanation (x)`

### 使用说明

- RL 模型训练时的 `grid_size` 应与运行时环境匹配
- `run.py` 会优先读取 checkpoint 中记录的网格大小和怪物数量
- `models/` 已加入 `.gitignore`
- 开启实时游戏窗口会拖慢训练速度；长时间训练时可以使用 `--no-show-game`
- 是否真正收敛应看评估 reward、胜率和平均步数是否稳定，而不是只看脚本是否跑完

---

## English

This is a Pac-Man-style Reinforcement Learning and Explainable AI demo project. It includes:

- a trainable DQN-based RL agent
- a heuristic A* baseline agent
- a natural-language explanation system
- a visual training dashboard and live game viewer

### Highlights

- `RLAgent`
  - runs inference with a DQN model trained by `train_rl.py`
- `HeuristicAgent`
  - serves as a no-training multi-objective A* baseline
- Bilingual QA
  - supports both Chinese and English question parsing and explanations
- Visual training
  - live metric plots
  - live Pac-Man game window during training
- XAI output
  - evidence set, evidence used, and natural-language explanation

### What The RL Pipeline Includes

- environment API
  - `reset_rl()`
  - `step_rl(action) -> observation, reward, done, info`
- observation encoding
  - walls
  - walkable paths
  - Pac-Man position
  - exit position and exit-open status
  - dot distribution
  - monster distribution
  - danger zones
  - scalar progress and distance features
- reward shaping
  - step penalty
  - invalid-action penalty
  - stay penalty
  - stall penalty
  - increasing time-pressure penalty
  - dot reward
  - exit unlock reward
  - reward for moving toward the current objective
  - penalty for moving away from the current objective
  - monster proximity penalty
  - win reward
  - fast-finish bonus
  - lose penalty
  - timeout penalty
- training loop
  - experience replay
  - epsilon-greedy exploration
  - target network
  - checkpoint saving
  - periodic evaluation

### Main Files

- `environment.py`
  - game environment
  - RL state encoding and reward logic
- `agent.py`
  - `HeuristicAgent`
  - `RLAgent`
  - `DQNNetwork`
- `train_rl.py`
  - DQN training entry point
- `training_game_viewer.py`
  - live training game window
- `run.py`
  - demo entry point
- `ui.py`
  - Pac-Man-style GUI
- `question_parser.py`
  - bilingual question parsing
- `explanation_engine.py`
  - natural-language explanation generation
- `evidence_recorder.py`
  - evidence tracking and replay

### Install

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn
```

### Train The Model

By default, training runs for `3000` episodes and saves the best checkpoint to `models/dqn_pacman.pt`:

```bash
py -3 train_rl.py
```

Training shows both of these by default:

- the metric dashboard
- the live Pac-Man training window

It also keeps saving:

- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`

Common examples:

```bash
py -3 train_rl.py --episodes 3000 --grid-size 21 --num-monsters 8
py -3 train_rl.py --episodes 500 --grid-size 15 --num-monsters 4
py -3 train_rl.py --no-show-plot
py -3 train_rl.py --no-show-game
```

Training logs report:

- current episode reward
- rolling average reward
- win rate
- average steps
- epsilon
- loss
- evaluation results

### Run The Demo

Auto mode:

```bash
py -3 run.py
```

Behavior:

- if `models/dqn_pacman.pt` exists, the RL model is loaded first
- otherwise the app falls back to the heuristic agent

Run the RL model explicitly:

```bash
py -3 run.py --agent rl --model-path models/dqn_pacman.pt
```

Run the heuristic baseline explicitly:

```bash
py -3 run.py --agent heuristic
```

### Example Questions

After pausing, you can ask questions such as:

- `Why not go right?`
- `为什么去吃那个豆子？`
- `怪物#2影响了这次决策吗？`
- `Is it safe here?`
- `出口什么时候打开？`

The explanation system still outputs three layers:

1. `All Evidence (S_t)`
2. `Evidence Used (E)`
3. `Natural-Language Explanation (x)`

### Notes

- The RL checkpoint should match the runtime `grid_size`
- `run.py` prefers the grid size and monster count stored in the checkpoint metadata
- `models/` is already ignored by `.gitignore`
- The live game window slows training down, so use `--no-show-game` for long experiments
- Real convergence should be judged by evaluation reward, win rate, and average steps, not just by whether the script finished
