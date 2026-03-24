# Explanation 框架使用指南

## 📖 快速开始

### 1. 查看核心定义（理论基础）

```bash
# 用任何编辑器打开
instructions/unified_definition_of_explanation.md
```

这是explanation的数学基础定义v1.0，理解它才能正确使用框架。

### 2. 运行完整演示

```bash
.venv\Scripts\python.exe explanation_strict.py
```

演示包含10个步骤，展示所有核心功能。

---

## 🎯 核心概念

### 数学定义（v2.0）

```
WExpl_{B,S}(E,φ) ⇔
  E⊆S ∧ |E|<∞ ∧ Cons(B∪E) ∧ B⊭φ ∧ B∪E⊨φ ∧ E∩Eq_B(φ)=∅

Expl_{B,S}(E,φ) ⇔
  WExpl_{B,S}(E,φ) ∧ ∀E'⊊E, ¬WExpl_{B,S}(E',φ)
```

**符号说明**：
- **L**: 形式语言
- **A**: 基本解释单元集合
- **Sen(L)**: L 中的句子集合
- **B ⊆ Sen(L)**: 背景理论（已知知识）
- **S ⊆ A**: 候选证据空间（可用的证据单元）
- **φ ∈ Sen(L)**: 待解释命题（explanandum）
- **E ⊆ S**: 证据集合
- **Cons(X)**: X 是一致的（∃M(M⊨X)）
- **Eq_B(φ)**: 在 B 下与 φ 逻辑等价的所有句子
- **⊨**: 语义蕴含关系

---

## 💻 基本使用

### 示例1：苏格拉底推理

```python
from explanation_strict import (
    ExplanationSystem,
    SimplePropositionalEntailment,
    PropositionalFormula
)

# 1. 创建系统
system = ExplanationSystem(SimplePropositionalEntailment())

# 2. 定义背景理论 B、证据空间 S、待解释命题 φ
B = {PropositionalFormula("所有人都是会死的")}
S = {
    PropositionalFormula("所有人都是会死的"),
    PropositionalFormula("苏格拉底是人"),
    PropositionalFormula("苏格拉底是哲学家")
}
φ = PropositionalFormula("苏格拉底是会死的")

# 3. 证据 E（从 S 中选择）
E = {PropositionalFormula("苏格拉底是人")}

# 4. 创建严格解释
expl = system.create_strict(E, φ, B, S)
print(expl)
# Output: Explanation(E={苏格拉底是人}, φ=苏格拉底是会死的, B={所有人都是会死的}, type=strict)

# 5. 验证是否为弱解释
print(system.validate_weak(E, φ, B, S))  # True

# 6. 搜索所有最小解释
minimals = system.find_minimal_explanations(φ, S, B, strict=True)
for m in minimals:
    print(f"最小解释: {m.evidence}")
    # 输出:
    # 最小解释: {苏格拉底是人}
    # 最小解释: {苏格拉底是哲学家}

# 7. 自然语言渲染
if system.renderer:
    print(system.render(expl, "strict"))
    # 输出: 严格解释: 基于证据 {苏格拉底是人}，在背景理论 所有人都是会死的 下，
    #       严格证明了 苏格拉底是会死的，且不存在更小的证据集也能证明。
```

### 示例2：自定义语义蕴含

```python
from explanation_strict import (
    SemanticEntailment,
    ExplanationSystem,
    PropositionalFormula,
    Set
)

class MyLogic(SemanticEntailment):
    """自定义逻辑系统（使用 SAT solver 或定理证明器）"""

    def entails(self, evidence, conclusion, background_theory=None, model=None):
        """实现 B∪E ⊨ φ 的检查"""
        # TODO: 调用 SAT solver 检查 (B∪E ∧ ¬φ) 是否不可满足
        pass

    def is_consistent(self, sentences, background_theory=None):
        """检查 Cons(B∪sentences)"""
        # TODO: 检查是否可满足
        pass

    def equivalent_under_background(self, a, b, background_theory=None):
        """检查 B⊨(a↔b)"""
        # TODO: 检查 B⊨(a→b) 且 B⊨(b→a)
        pass

    def get_all_sentences(self):
        """返回语言 L 的所有公式"""
        # TODO: 生成或返回公式集合
        pass

# 使用自定义逻辑
system = ExplanationSystem(MyLogic())
```

---

## 🔧 API 参考

### `Explanation[T, M]`

explanation的数学本体。

**字段**：
- `evidence: FrozenSet[T]` - 证据集合 E
- `explanandum: T` - 待解释命题 φ
- `background_theory: Optional[FrozenSet[T]]` - 背景理论 B
- `candidate_space: Optional[FrozenSet[T]]` - 候选空间 S
- `is_strict: bool` - 是否为严格解释
- `entailment: SemanticEntailment[T, M]` - 语义蕴含实现
- `metadata: Optional[Dict[str, Any]]` - 元数据

**方法**：
- `is_weakly_explanatory() -> bool` - 检查是否为弱解释
- `is_strictly_explanatory() -> bool` - 检查是否为严格解释
- `to_dict() -> Dict[str, Any]` - 序列化为字典
- `__len__() -> int` - 返回证据数量

---

### `ExplanationSystem[T, M]`

系统主接口。

**方法**：
- `create_weak(E, φ, B=None, S=None) -> Explanation` - 创建弱解释
- `create_strict(E, φ, B=None, S=None) -> Explanation` - 创建严格解释
- `validate_weak(E, φ, B=None, S=None) -> bool` - 验证 WExpl 条件
- `validate_strict(E, φ, B=None, S=None) -> bool` - 验证 Expl 条件
- `find_minimal_explanations(φ, S, B=None, strict=True, max_results=100) -> List[Explanation]` - 搜索最小解释
- `render(explanation, type="standard") -> str` - 渲染自然语言
- `filter_by_size(explanations, max_size=None, min_size=None) -> List[Explanation]` - 按大小过滤

---

### `SemanticEntailment[T, M]`（抽象接口）

必须实现以下4个方法：

```python
class SemanticEntailment(ABC, Generic[T, M]):
    @abstractmethod
    def entails(self, evidence, conclusion, background_theory=None, model=None) -> bool:
        """检查 B∪E ⊨ φ"""
        pass

    @abstractmethod
    def is_consistent(self, sentences, background_theory=None) -> bool:
        """检查 Cons(B∪sentences)"""
        pass

    @abstractmethod
    def equivalent_under_background(self, a, b, background_theory=None) -> bool:
        """检查 B⊨(a↔b)"""
        pass

    @abstractmethod
    def get_all_sentences(self) -> Set[T]:
        """返回 Sen(L) 或 A"""
        pass
```

---

### `NaturalLanguageRenderer[T]`（抽象接口）

```python
class NaturalLanguageRenderer(ABC, Generic[T]):
    @abstractmethod
    def render(self, evidence, explanandum, background_theory=None, explanation_type="standard") -> str:
        """渲染为自然语言"""
        pass
```

---

## 📝 详细示例

### 示例3：验证 WExpl 的6个条件

```python
from explanation_strict import ExplanationSystem, SimplePropositionalEntailment, PropositionalFormula

system = ExplanationSystem(SimplePropositionalEntailment())

B = {PropositionalFormula("所有人都是会死的")}
S = {PropositionalFormula("所有人都是会死的"), PropositionalFormula("苏格拉底是人")}
φ = PropositionalFormula("苏格拉底是会死的")
E = {PropositionalFormula("苏格拉底是人")}

# 手动验证每个条件
print("1) E ⊆ S:", E.issubset(S))
# True

print("2) |E| < ∞:", len(E) > 0)
# True

print("3) Cons(B∪E):", system.entailment.is_consistent(E, B))
# True

print("4) B⊭φ:", not system.entailment.entails(set(), φ, B))
# True

print("5) B∪E⊨φ:", system.entailment.entails(E, φ, B))
# True

# 计算 Eq_B(φ)
# 简化实现中，Eq_B(φ) 至少包含 φ 自身
Eq_B_φ = {φ}  # 实际需要通过 _compute_equivalence_class 计算
print("6) E∩Eq_B(φ)=∅:", E.intersection(Eq_B_φ) == set())
# True

# 或者直接使用 validate_weak
is_valid = system.validate_weak(E, φ, B, S)
print("WExpl 条件全部满足?", is_valid)
# True
```

### 示例4：搜索所有最小解释

```python
# 候选空间 S 中有多个可能的证据
S = {
    PropositionalFormula("所有人都是会死的"),
    PropositionalFormula("苏格拉底是人"),
    PropositionalFormula("苏格拉底是哲学家")
}

# 搜索严格最小解释
minimals = system.find_minimal_explanations(
    explanandum=φ,
    candidate_space=S,
    background_theory=B,
    strict=True,  # 找严格解释
    max_results=10
)

print(f"找到 {len(minimals)} 个严格最小解释:")
for expl in minimals:
    print(f"  - E = {expl.evidence}, 大小 = {len(expl)}")

# 输出:
# 找到 2 个严格最小解释:
#   - E = {苏格拉底是人}, 大小 = 1
#   - E = {苏格拉底是哲学家}, 大小 = 1
```

### 示例5：检查严格最小性

```python
E = {PropositionalFormula("苏格拉底是人")}
expl = system.create_strict(E, φ, B, S)

# 方法1: 使用 is_strictly_explanatory（内部已检查）
print(expl.is_strictly_explanatory())  # True

# 方法2: 使用系统方法
is_strict = system.validate_strict(E, φ, B, S)
print(is_strict)  # True
```

---

## 🧪 测试

代码中包含一个测试用例集合（main函数的第8部分）：

```python
test_cases = [
    {"name": "空证据集", "E": set(), "φ": φ, "B": B, "should_pass": False},
    {"name": "E 包含 φ 本身", "E": {φ}, "φ": φ, "B": B, "should_pass": False},
    {"name": "E 蕴含 φ（有效）", "E": {...}, "φ": φ, "B": B, "should_pass": True},
    {"name": "B 已蕴含 φ", "E": set(), "φ": φ, "B": {φ}, "should_pass": False},
]

for tc in test_cases:
    result = system.validate_weak(tc["E"], tc["φ"], tc["B"])
    status = "PASS" if result == tc["should_pass"] else "FAIL"
    print(f"[{status}] {tc['name']}")
```

运行演示时会自动执行这些测试（当前全部通过 ✅）。

---

## 🔨 开发自定义实现

### 实现真正的 SAT-based 语义蕴含

```python
from explanation_strict import SemanticEntailment, PropositionalFormula
from typing import Set, Optional

class SATBasedEntailment(SemanticEntailment):
    """基于 SAT solver 的语义蕴含实现"""

    def __init__(self, solver=None):
        self.solver = solver  # 例如: python-sat

    def entails(self, evidence, conclusion, background_theory=None, model=None):
        # 1. 构造 CNF: (B ∪ E) ∧ ¬φ
        clauses = self._to_cnf(background_theory or set())
        clauses.extend(self._to_cnf(evidence))
        clauses.extend(self._to_cnf({conclusion}, negate=True))  # ¬φ

        # 2. 调用 SAT solver
        result = self.solver.solve(clauses)

        # 3. 如果不可满足，则 B∪E ⊨ φ
        return result == "UNSAT"

    def is_consistent(self, sentences, background_theory=None):
        # 检查 B∪sentences 是否可满足
        clauses = self._to_cnf(background_theory or set())
        clauses.extend(self._to_cnf(sentences))
        result = self.solver.solve(clauses)
        return result == "SAT"

    def equivalent_under_background(self, a, b, background_theory=None):
        # 检查 B⊨(a↔b)
        # 即: B∪{a}⊨b 且 B∪{b}⊨a
        return (
            self.entails({a}, b, background_theory) and
            self.entails({b}, a, background_theory)
        )

    def get_all_sentences(self):
        # 返回语言中所有可能的句子（对于无限语言，返回空或子集）
        return set()

    def _to_cnf(self, formulas, negate=False):
        # 将公式转换为 CNF
        pass
```

---

## 📁 项目结构

```
XAI/
├── instructions/
│   └── unified_definition_of_explanation.md  # 理论基础（必须阅读）
├── explanation_strict.py                     # 核心实现（唯一代码文件）
├── .venv/                                    # 虚拟环境
└── README.md                                 # 本文档
```

---

## ⚠️ 注意事项

1. **演示实现**：`SimplePropositionalEntailment` 只是演示，内置了固定推理规则，不能处理复杂逻辑。实际研究需要自己实现 `SemanticEntailment` 接口。

2. **Eq_B(φ) 计算**：在演示实现中，`equivalent_under_background` 只检查字符串相等或背景中的显式等价。实际系统需要完整的逻辑等价性证明。

3. **Consistency 检查**：演示实现只检查明显的矛盾（p 和 ¬p）。实际系统需要 SAT solver 或模型检查。

4. **get_all_sentences()**：对于无限语言（如命题逻辑），这个函数无法返回所有句子。实际实现可以返回空集或受限的子集。

5. **性能**：
   - `find_minimal_explanations()` 是指数级复杂度 O(2^|S|)
   - 仅适用于小规模 S（|S| < 20）
   - 大规模需要使用 SAT solver 优化或启发式算法

6. **Unicode**：代码中已将数学符号替换为ASCII，避免Windows控制台编码问题。

---

## 🐛 故障排除

### 问题：ImportError: No module named explanation_strict

**解决**：确保在项目根目录运行，或使用完整路径：
```bash
cd /path/to/XAI
python explanation_strict.py
```

### 问题：验证失败，不满足 WExpl 条件

**原因**：检查 `__post_init__` 抛出的异常信息：
- "E ⊈ S" - 证据不在候选空间中
- "Cons(B∪E) 失败" - 不一致
- "B⊨φ" - 背景已蕴含，不需要解释
- "B∪E⊭φ" - 证据+背景不蕴含
- "E ∩ Eq_B(φ) ≠ ∅" - 证据包含等价句子

**解决**：调整 B、E、φ 使得满足所有6个条件。

### 问题：找不到严格最小解释

**原因**：严格解释要求不存在任何真子集也满足 WExpl。如果 E 包含多余证据，就不是严格的。

**解决**：
1. 使用 `find_minimal_explanations(strict=True)` 自动搜索
2. 手动检查：移除每个元素，看是否仍然满足 `validate_weak`

---

## 📚 数学定义详解

### WExpl_{B,S}(E,φ) 的6个条件

| 条件 | 数学符号 | 含义 | 检查位置 |
|------|---------|------|---------|
| 1 | E ⊆ S | E 是 S 的子集 | `__post_init__` |
| 2 | |E| < ∞ | E 有限（代码自动满足） | `__post_init__` |
| 3 | Cons(B∪E) | B∪E 一致 | `is_consistent()` |
| 4 | B ⊭ φ | B 单独不蕴含 φ | `entails(set(), φ, B)` |
| 5 | B∪E ⊨ φ | B∪E 蕴含 φ | `entails(E, φ, B)` |
| 6 | E ∩ Eq_B(φ) = ∅ | E 不包含与 φ 等价的句子 | `_compute_equivalence_class()` |

### Expl_{B,S}(E,φ) 的额外条件

- **条件7**: ∀E'⊊E, ¬WExpl_{B,S}(E',φ)
- 检查：`_check_strict_minimality()` 遍历所有真子集

---

## 🎓 设计哲学

1. **严格形式化**：所有定义都有明确的数学表达式
2. **可验证性**：`__post_init__` 自动验证所有条件
3. **分离关注点**：
   - `Explanation` - 数学本体
   - `SemanticEntailment` - 逻辑语义
   - `Renderer` - 自然语言表达
4. **可扩展**：通过接口支持不同逻辑系统
5. **实用主义**：提供 `create_weak/strict`、`validate`、`find_minimal` 等实用方法

---

## 📖 引用

如果使用本框架，请引用：

```
Explanation Definition v2.0. (2026). XAI Project, Strict Deductive Definition.
https://github.com/zhangyuang95-a11y/XAI
```

---

**开始使用最简单的方式**：
```bash
.venv\Scripts\python.exe explanation_strict.py
```

先看演示，理解所有输出，然后修改 `main()` 函数尝试自己的例子！
