# XAI Explanation Framework

**基于严格演绎定义的explanation框架**

```
WExpl_{B,S}(E,φ) ⇔ E⊆S ∧ |E|<∞ ∧ Cons(B∪E) ∧ B⊭φ ∧ B∪E⊨φ ∧ E∩Eq_B(φ)=∅
Expl_{B,S}(E,φ) ⇔ WExpl_{B,S}(E,φ) ∧ ∀E'⊊E, ¬WExpl_{B,S}(E',φ)
```

---

## 📁 项目文件

```
XAI/
├── instructions/
│   └── unified_definition_of_explanation.md  # 理论基础 v1.0
├── explanation_strict.py                      # 核心实现 v2.0 ⭐
├── USAGE.md                                   # 详细使用指南
├── quick_test.py                              # 快速测试脚本
└── README.md                                  # 本文件
```

---

## 🚀 快速开始

### 1. 运行演示（推荐先看）

```bash
.venv\Scripts\python.exe explanation_strict.py
```

输出包括：
- 10步完整演示
- 苏格拉底推理示例
- WExpl 6个条件检查
- 严格最小性验证
- 自然语言渲染
- 4个边界测试（全部通过）
- 数学定义展示

### 2. 快速测试

```bash
.venv\Scripts\python.exe quick_test.py
```

验证5个核心功能：
1. 创建弱解释 ✅
2. 验证 WExpl 条件 ✅
3. 创建严格解释 ✅
4. 搜索最小解释 ✅
5. 自然语言渲染 ✅

### 3. 阅读文档

```bash
# 理论基础（必读）
type instructions\unified_definition_of_explanation.md

# 详细使用指南
type USAGE.md
```

---

## 🎯 核心特性

### ✅ 严格数学定义

基于您提供的演绎定义：
- **弱解释 WExpl**：6个明确条件
- **严格解释 Expl**：在 WExpl 基础上加上严格最小性
- **背景理论 B**：显式支持
- **等价类 Eq_B(φ)**：避免平凡解释

### ✅ 完整验证

`Explanation.__post_init__` 自动验证所有 WExpl 条件：
1. E ⊆ S
2. |E| < ∞
3. Cons(B∪E) - 一致性
4. B ⊭ φ - 背景不单独蕴含
5. B∪E ⊨ φ - 联合蕴含
6. E ∩ Eq_B(φ) = ∅ - 无等价平凡性

### ✅ 严格最小性检查

`_check_strict_minimality()` 遍历 E 的所有真子集，确保不存在更小的有效解释。

### ✅ 丰富 API

- `create_weak(E, φ, B, S)` - 创建弱解释
- `create_strict(E, φ, B, S)` - 创建严格解释
- `validate_weak(...)` - 验证 WExpl
- `validate_strict(...)` - 验证 Expl
- `find_minimal_explanations(...)` - 搜索所有最小解释
- `render(explanation, type)` - 自然语言渲染

### ✅ 可插拔设计

通过 `SemanticEntailment` 接口支持任意逻辑系统：
- 命题逻辑（已提供示例）
- 一阶逻辑（需实现）
- 模态逻辑（需实现）
- 概率逻辑（需实现）

---

## 💻 基本用法

```python
from explanation_strict import *

# 1. 创建系统
system = ExplanationSystem(SimplePropositionalEntailment())

# 2. 定义 B, S, φ
B = {PropositionalFormula("所有人都是会死的")}
S = {PropositionalFormula("苏格拉底是人")}
φ = PropositionalFormula("苏格拉底是会死的")
E = {PropositionalFormula("苏格拉底是人")}

# 3. 创建严格解释
expl = system.create_strict(E, φ, B, S)

# 4. 使用
print(expl)                      # 数学本体
print(system.validate_strict(E, φ, B, S))  # 验证
print(system.render(expl, "strict"))      # 渲染
```

---

## 🔨 自定义开发

### 实现自己的 SemanticEntailment

```python
class MyLogic(SemanticEntailment):
    def entails(self, evidence, conclusion, background_theory=None, model=None):
        # 实现 B∪E ⊨ φ
        # 建议使用 SAT solver（如 python-sat、z3）
        pass

    def is_consistent(self, sentences, background_theory=None):
        # 检查一致性
        pass

    def equivalent_under_background(self, a, b, background_theory=None):
        # 检查 B⊨(a↔b)
        pass

    def get_all_sentences(self):
        # 返回 Sen(L)
        pass

system = ExplanationSystem(MyLogic())
```

---

## 📊 框架对比

| 特性 | v1.0 (explanation.py) | v2.0 (explanation_strict.py) |
|------|----------------------|----------------------------|
| 定义基础 | 简单蕴含 E⊆S ∧ E⊨q | 严格演绎 WExpl/Expl |
| 背景理论 | ❌ 不支持 | ✅ 支持 B |
| 弱解释 | ❌ | ✅ WExpl（6条件） |
| 严格解释 | ❌ | ✅ Expl（+最小性） |
| 验证机制 | 基础检查 | 完整 WExpl 验证 |
| Minimality | 子集移除检查 | 真子集 WExpl 检查 |
| Eq_B(φ) | ❌ | ✅ |
| 语义蕴含接口 | `entails(E, q)` | `entails(E, φ, B)` |
| 代码行数 | ~300 | ~1000 |
| 状态 | 废弃 | 当前版本 ✅ |

**建议**: 所有新项目使用 **v2.0 (explanation_strict.py)**

---

## 📖 数学定义详解

### WExpl_{B,S}(E,φ) 的6个条件

| # | 条件 | 说明 | 代码检查 |
|---|------|------|---------|
| 1 | E ⊆ S | E 是 S 的子集 | `evidence.issubset(candidate_space)` |
| 2 | |E| < ∞ | E 有限（frozenset 自动满足） | `len(evidence) > 0` |
| 3 | Cons(B∪E) | B∪E 一致 | `entailment.is_consistent(E, B)` |
| 4 | B ⊭ φ | B 单独不蕴含 φ | `not entails(set(), φ, B)` |
| 5 | B∪E ⊨ φ | 联合蕴含 | `entailment.entails(E, φ, B)` |
| 6 | E ∩ Eq_B(φ) = ∅ | E 不含等价句子 | `_compute_equivalence_class()` |

### Expl_{B,S}(E,φ) 额外条件

- **条件7**: ∀E'⊊E, ¬WExpl_{B,S}(E',φ)
- 实现：`_check_strict_minimality()` 遍历所有真子集

---

## 🧪 测试状态

| 测试项 | 状态 |
|-------|------|
| 创建弱解释 | ✅ |
| 创建严格解释 | ✅ |
| WExpl 验证 | ✅ |
| Expl 验证 | ✅ |
| 严格最小性检查 | ✅ |
| 搜索最小解释 | ✅ |
| 自然语言渲染 | ✅ |
| 边界条件测试 | ✅ 4/4 |

**运行**: `python explanation_strict.py` (第8部分)

---

## 📂 文件说明

| 文件 | 用途 | 必读? |
|------|------|-------|
| `explanation_strict.py` | 核心实现（1000+行） | ✅ 必读 |
| `instructions/unified_definition_of_explanation.md` | 理论基础 v1.0 | ✅ 必读 |
| `USAGE.md` | 详细使用指南 | ✅ 推荐 |
| `quick_test.py` | 快速验证脚本 | ✅ 推荐 |
| `README.md` | 本文件 | ✅ |

---

## 🔗 引用

```bibtex
@misc{explanation_v2,
  title={Explanation Framework v2.0: Strict Deductive Definition},
  author={XAI Project},
  year={2026},
  url={https://github.com/zhangyuang95-a11y/XAI}
}
```

---

## 📞 支持

- 问题反馈: GitHub Issues
- 讨论: GitHub Discussions

---

**版本**: 2.0 (strict deductive definition)
**更新**: 2026-03-24
**状态**: ✅ 生产就绪

---

## ⚡ 下一步

1. **运行演示**: `python explanation_strict.py`
2. **阅读 USAGE.md**: 了解详细用法
3. **实现自己的 SemanticEntailment**: 使用 SAT solver 或定理证明器
4. **扩展功能**: 添加更多逻辑系统、优化算法、完善渲染器

**开始使用**: `python explanation_strict.py` 🚀
