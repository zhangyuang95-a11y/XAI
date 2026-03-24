# Explanation 统一形式定义

## 版本：v1.0 (基础定义)

这是explanation的全球统一基础定义，所有后续的explanation研究都必须基于此定义或其扩展版本。

---

## 一、基本符号系统

### 1. 形式语言 L
设 L 为表示 agent 可用知识的形式语言（formal language）。
- L 中的元素是命题、公式或语句
- L 具有明确的语法结构和语义解释

### 2. 证据空间 S
设 S = {e₁, e₂, ..., eₙ} 表示 agent 在某一推理时刻可访问的全部证据集合。
- S ⊆ P(L) 或 S ⊆ L（根据具体实现）
- 每个 eᵢ ∈ L
- **关键区分**：S 是 agent 的 evidence space（可访问证据空间），不是客观世界本身

### 3. Why-question q
设 q ∈ L 表示需要被解释的目标命题。
- q 是需要回答"为什么成立"的对象
- q 是待解释的结论或现象

### 4. Explanation Candidate E
设 E ⊆ S。
- E 是一个候选 explanation（解释候选）
- E 是由证据组成的集合/结构

### 5. 语义蕴含 ⊨
设 M 为 L 的模型类（class of models）。

**定义**：若对任意模型 m ∈ M，只要 m ⊨ E（m 满足所有证据），就必然有 m ⊨ q，则称 E ⊨ q。

记法：E |= q

### 6. Explanation Function f
定义函数 f : (E, q) ↦ text，将形式 explanation 与问题转化为自然语言解释文本。

**重要区分**：
- E 是 explanation 的数学本体（mathematical object）
- f(E, q) 是其自然语言表达（natural language rendering）

---

## 二、核心定义

### Definition 1 (Explanation)
**解释**：若 E ⊆ S 且 E |= q，则 E 是 q 的一个 explanation。

形式化：
```
Explanation(E, q, S) ≡ (E ⊆ S) ∧ (E |= q)
```

### Definition 2 (Minimal Explanation)
**最小解释**：若 E |= q，并且对任意 e ∈ E，都有 E \ {e} ⊭ q，则 E 是 minimal explanation。

形式化：
```
MinimalExplanation(E, q, S) ≡ Explanation(E, q, S) ∧ (∀e∈E, ¬Explanation(E\{e}, q, S))
```

### Definition 3 (Proper Explanation)
**真解释**：若 E 是 explanation 且 E 是 minimal 的，则 E 是 proper explanation。

**说明**：某些文献中 proper explanation 等同于 minimal explanation。

---

## 三、关键性质

### 3.1 非唯一性
对于给定的 (q, S)，可能存在多个不同的 minimal explanations：
- E₁, E₂, ..., Eₖ 都是 minimal explanations of q
- 它们互不包含（minimality 保证了相互独立性）

### 3.2 闭合性
若 E 是 explanation 且 E' ⊇ E，则 E' 也是 explanation（但非 minimal）。

### 3.3 反事实敏感性
Explanation 的定义依赖于语义蕴含 ⊨，该关系捕获了逻辑/理论上的必然性，而非因果或反事实关系（后者需要扩展定义）。

---

## 四、扩展方向（不破坏基础定义）

基础定义保持上述简单形式，后续研究可通过以下方式扩展：

### 4.1 加入规则集合 R
某些定义使用三元组 (E, R, q) 而非仅 E：
- E：证据
- R：规则集合（inference rules）
- q：结论

扩展定义：
```
Explanation_R(E, R, q) ≡ (E ⊆ S) ∧ (R ⊢ q from E)
```

**注意**：基础定义中 ⊨ 已经隐含了逻辑系统的规则，显式引入 R 是为了强调推理机制。

### 4.2 支持度度量
定义 explanation quality 函数：
```
quality(E, q) ∈ ℝ
```

可能的度量：
- 简洁度：-|E|（越小越好）
- 闭合度：closure_depth(E, q, S) ∈ ℕ
- 概率度：P(q|E)（概率框架下）

### 4.3 Why-question Chain
定义 follow-up question 的序列：
```
q₀ → q₁ → q₂ → ... → qₙ
```

其中 qᵢ₊₁ 是对 qᵢ 的进一步追问，形成解释链。

### 4.4 交互式 Closure
在对话场景中，解释的"完成度"可通过追问轮次衡量：
- 若经过 k 轮追问后无新有效问题，则 closure depth = k

---

## 五、与其他领域的关系

### 5.1 与 abduction（溯因推理）的关系
- Abduction：从现象 q 寻找最佳解释 E
- 本定义提供了 E 的形式标准：E 必须是满足 E |= q 的子集
- Abduction 的任务：在满足条件的 E 中选择最优者

### 5.2 与 deduction（演绎推理）的关系
- Deduction：从 E 推出 q（即验证 E |= q）
- 本定义将 deduction 作为验证解释有效性的工具

### 5.3 与 induction（归纳推理）的关系
- Induction：从多个实例 E 推广到 q
- 本定义中 E |= q 可以是归纳蕴含（如概率大于阈值）

---

## 六、使用规范

### 6.1 何时使用基础定义
- 理论讨论的起点
- 不同定义方案的比较基准
- 形式化证明的基础

### 6.2 何时需要扩展
- 具体应用场景（如因果解释、概率解释）
- 需要度量解释质量时
- 研究解释的交互过程时

### 6.3 命名约定
- 基础版本：`Explanation_v1`（本定义）
- 扩展版本：`Explanation_vX_Y`，其中 X 是主版本（重大变更），Y 是次版本（修正/细化）

---

## 七、版本历史

### v1.0 (2026-03-24)
- 确立基础符号系统
- 定义 Explanation 和 Minimal Explanation
- 明确数学本体与自然语言表达的分离
- 提出扩展框架

---

## 八、正式声明

**任何关于 explanation 的学术工作，若引用或使用此定义，必须声明版本号。**

建议引用格式：
```
Explanation Definition v1.0. (2026). XAI Project, Global Unified Definition of Explanation.
```

---

## 九、约束检查清单

本定义满足所有形式约束：

- ✅ Explanation 被定义为数学对象（集合 E 及其关系）
- ✅ 不直接定义为自然语言文本（通过函数 f 分离）
- ✅ 不依赖心理学/直觉/叙事定义
- ✅ 包含明确的形式对象、符号、条件与性质
- ✅ 不同定义版本可直接比较（通过扩展/特化关系）

---

## 十、未解决问题（转移到研究任务）

- 语义蕴含 ⊨ 的具体逻辑系统选择（一阶逻辑？模态逻辑？）
- S 的边界如何确定（agent 的认知边界问题）
- f(E, q) 的形式化（自然语言生成的可比性问题）
- 如何处理不一致的证据集 S

**这些问题属于开放研究任务，不在基础定义中解决。**
