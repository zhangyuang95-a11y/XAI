# Pac-Man RL + XAI Demo

This project trains a Pac-Man reinforcement-learning agent and explains its decisions through a question-aware, evidence-grounded explanation pipeline. The system supports Chinese and English questions such as "Why did you go up?", "Why not go left?", "Is it safe?", and "Did the monster affect the decision?"

- [English](#english)
- [中文](#中文)

<a id="english"></a>
## English

### What This Project Does

- Trains a fixed Pac-Man RL agent for an `11x11` maze with `2` monsters.
- Runs a Tkinter UI where the agent plays and the user can ask questions.
- Parses user questions into structured intents such as `why_this_action`, `why_not_other`, `safety_reason`, `monster_influence`, `dot_collection`, `policy_summary`, and `irrelevant`.
- Generates natural-language explanations from current decision evidence instead of only dumping state variables.
- Shows audit information, including all evidence, selected evidence, validation checks, symbolic match status, and risk metrics.
- Uses a symbolic policy only as optional support, not as a mandatory explanation source.

### Current Method

The current explanation method is:

```text
user question
  -> semantic frame / intent
  -> current decision evidence
  -> question-specific evidence selection
  -> natural-language answer
  -> evidence and validation diagnostics
```

The system records evidence at each timestep, including the chosen action, available actions, collision risks, nearest target, nearest monster, dot progress, exit status, and agent reasoning. The explanation engine then chooses the evidence relevant to the user's question.

For example:

- If the user asks "Why go up?", the answer explains the actual action, target, monster distance, and risk.
- If the user asks "Why not go left?", the answer compares the chosen action with the mentioned alternative.
- If the user asks an unrelated question such as "What is the weather today?", the system returns an irrelevant-question response instead of forcing a Pac-Man explanation.
- If the user asks about an action that did not happen, the system first corrects the premise and then explains the real action.

### Symbolic Policy Role

The symbolic policy is a distilled decision-tree surrogate for the neural RL policy. It is optional.

```text
if symbolic policy exists and matches the neural action:
    use symbolic rule / trace as extra explanation evidence
else:
    use evidence-only explanation
```

This avoids treating an unfaithful surrogate as the explanation. New environments do not need to implement a symbolic policy first; they can start with an evidence recorder and add symbolic support later.

### Project Layout

```text
XAI/
  core/
    explanation.py
    explanation_strict.py
    symbolic_policy.py
    explanation_engine.py
    evidence.py

  envs/
    pacman/
      environment.py
      agent.py
      train_rl.py
      run.py
      ui.py
      evidence_recorder.py
      explanation_engine.py
      question_parser.py
      symbolic_policy_adapter.py
      validate_explanations.py

  run.py
  train_rl.py
  validate_explanations.py
```

`core/` contains environment-agnostic explanation concepts and interfaces. `envs/pacman/` contains Pac-Man-specific environment logic, evidence extraction, question parsing, and explanation rendering. The top-level `run.py`, `train_rl.py`, and `validate_explanations.py` are compatibility wrappers.

### Installation

Python 3 is recommended.

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn joblib
```

Notes:

- `torch` is used for the RL model.
- `sentence-transformers` and `scikit-learn` are used for question understanding and symbolic policy distillation.
- `matplotlib` is used for training plots.
- `tkinter` is used for the desktop UI and usually comes with Python.

### Train

Train the fixed Pac-Man model:

```bash
py -3 train_rl.py
```

Equivalent package command:

```bash
py -3 -m envs.pacman.train_rl
```

Default outputs:

- `models/dqn_pacman.pt`
- `models/dqn_pacman_symbolic.joblib`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`
- `artifacts/policy_code.py`
- `artifacts/policy_summary.json`

### Run The UI

Start the demo:

```bash
py -3 run.py
```

Equivalent package command:

```bash
py -3 -m envs.pacman.run
```

Run without symbolic policy support:

```bash
py -3 -m envs.pacman.run --no-symbolic-policy
```

Require a symbolic policy artifact and fail if it is unavailable:

```bash
py -3 -m envs.pacman.run --require-symbolic-policy
```

User questions are logged to:

- `artifacts/user_question_log.jsonl`

### Validate Explanations

Run a quick validation pass:

```bash
py -3 -m envs.pacman.validate_explanations --agent rl --model-path models/dqn_pacman.pt --episodes 1 --max-steps 40
```

Validate evidence-only explanations:

```bash
py -3 -m envs.pacman.validate_explanations --agent rl --model-path models/dqn_pacman.pt --episodes 1 --max-steps 40 --no-symbolic-policy
```

Validation output:

- `artifacts/explanation_validation.json`

### Development Checks

Check that `core/` does not depend on Pac-Man-specific classes:

```bash
rg "MazeEnvironment|monster|dot|exit|grid|RLAgent|ACTION_NAMES" core
```

Compile the package:

```bash
py -3 -m compileall -q core envs run.py train_rl.py validate_explanations.py
```

### Adding A New Environment

Add a new folder under `envs/`, for example:

```text
envs/
  pacman/
  traffic/
  gridworld/
  robotics/
```

Each environment should implement its own environment, agent, evidence recorder, explanation engine, question parser, runner, and validation script. The shared `core/` package should remain environment-agnostic.

<a id="中文"></a>
## 中文

### 项目简介

这个项目训练一个 Pac-Man 强化学习 agent，并用一个面向用户问题、基于证据的 explanation pipeline 来解释它的决策。系统支持中文和英文问题，例如“为什么向上走”“为什么不往左走”“安全吗”“怪物有没有影响决策”等。

### 当前功能

- 训练固定配置的 Pac-Man RL agent：`11x11` 地图，`2` 个怪物。
- 打开 Tkinter UI，让 agent 自动运行，并允许用户随时提问。
- 将用户问题解析成结构化 intent，例如 `why_this_action`、`why_not_other`、`safety_reason`、`monster_influence`、`dot_collection`、`policy_summary`、`irrelevant`。
- 根据当前决策 evidence 生成自然语言解释，而不是简单堆砌状态变量。
- 展示审计信息，包括全部证据、实际用到的证据、validation checks、symbolic match 状态和风险指标。
- symbolic policy 是可选证据来源，不是系统必须依赖的解释来源。

### 当前 Explanation 方法

当前 explanation 流程是：

```text
用户问题
  -> semantic frame / intent
  -> 当前决策 evidence
  -> 根据问题类型选择 evidence
  -> 生成自然语言回答
  -> 展示 evidence 和 validation diagnostics
```

系统会在每一步记录当前决策相关证据，例如实际动作、可选动作、碰撞风险、最近目标、最近怪物、豆子进度、出口状态和 agent reasoning。然后 explanation engine 会根据用户问题选择相关证据。

例如：

- 用户问“为什么向上走”，系统会解释真实动作、目标、怪物距离和风险。
- 用户问“为什么不往左走”，系统会比较真实动作和用户提到的替代动作。
- 用户问“今天天气如何”，系统会识别为无关问题，不会强行生成 Pac-Man 决策解释。
- 如果用户问的动作和真实动作不一致，系统会先纠正问题前提，再解释真实发生的动作。

### Symbolic Policy 的角色

当前 symbolic policy 是一个从神经网络 RL policy 蒸馏出来的 decision-tree surrogate。它是可选的。

```text
如果 symbolic policy 存在，并且它和神经网络当前动作一致：
    使用 symbolic rule / trace 作为额外解释证据
否则：
    回退到 evidence-only explanation
```

这样可以避免把不忠实的 surrogate 当成真正解释。未来新增环境时，不需要一开始就实现 symbolic policy；只要能提供 evidence recorder，就可以进入 explanation pipeline。

### 项目结构

```text
XAI/
  core/
    explanation.py
    explanation_strict.py
    symbolic_policy.py
    explanation_engine.py
    evidence.py

  envs/
    pacman/
      environment.py
      agent.py
      train_rl.py
      run.py
      ui.py
      evidence_recorder.py
      explanation_engine.py
      question_parser.py
      symbolic_policy_adapter.py
      validate_explanations.py

  run.py
  train_rl.py
  validate_explanations.py
```

`core/` 放通用 explanation 概念和接口。`envs/pacman/` 放 Pac-Man 专用环境、证据提取、问题解析和解释生成逻辑。顶层 `run.py`、`train_rl.py`、`validate_explanations.py` 是兼容入口。

### 安装依赖

推荐使用 Python 3。

```bash
py -3 -m pip install torch numpy matplotlib sentence-transformers scikit-learn joblib
```

说明：

- `torch` 用于 RL 模型。
- `sentence-transformers` 和 `scikit-learn` 用于问题理解和 symbolic policy distillation。
- `matplotlib` 用于训练曲线。
- `tkinter` 用于桌面 UI，通常随 Python 自带。

### 训练

训练固定 Pac-Man 模型：

```bash
py -3 train_rl.py
```

等价 package 命令：

```bash
py -3 -m envs.pacman.train_rl
```

默认输出：

- `models/dqn_pacman.pt`
- `models/dqn_pacman_symbolic.joblib`
- `artifacts/training_progress.png`
- `artifacts/training_metrics.csv`
- `artifacts/policy_code.py`
- `artifacts/policy_summary.json`

### 运行 UI

启动 demo：

```bash
py -3 run.py
```

等价 package 命令：

```bash
py -3 -m envs.pacman.run
```

禁用 symbolic policy：

```bash
py -3 -m envs.pacman.run --no-symbolic-policy
```

要求必须加载 symbolic policy，否则直接报错：

```bash
py -3 -m envs.pacman.run --require-symbolic-policy
```

用户问题日志保存到：

- `artifacts/user_question_log.jsonl`

### 验证 Explanation

快速验证：

```bash
py -3 -m envs.pacman.validate_explanations --agent rl --model-path models/dqn_pacman.pt --episodes 1 --max-steps 40
```

验证 evidence-only explanation：

```bash
py -3 -m envs.pacman.validate_explanations --agent rl --model-path models/dqn_pacman.pt --episodes 1 --max-steps 40 --no-symbolic-policy
```

验证结果输出：

- `artifacts/explanation_validation.json`

### 开发检查

检查 `core/` 是否误引入 Pac-Man 专用内容：

```bash
rg "MazeEnvironment|monster|dot|exit|grid|RLAgent|ACTION_NAMES" core
```

编译检查：

```bash
py -3 -m compileall -q core envs run.py train_rl.py validate_explanations.py
```

### 新增环境

未来新增环境时，在 `envs/` 下加新文件夹，例如：

```text
envs/
  pacman/
  traffic/
  gridworld/
  robotics/
```

每个环境实现自己的 environment、agent、evidence recorder、explanation engine、question parser、runner 和 validation script。共享的 `core/` 应保持与具体环境无关。
