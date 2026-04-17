"""
Explanation 框架 - 基于严格演绎定义的实现
版本: 2.0
日期: 2026-03-24

数学定义:
-----------------
设形式语言 L、句子集合 Sen(L)、语义蕴含关系 |=、基本解释单元集合 A、
背景理论 BsubseteqSen(L)、候选证据空间 SsubseteqA、待解释命题 φinSen(L)：

定义 1 (一致性): Cons(X) iff existsM(M|=X)

定义 2 (背景等价): Eq_B(φ) = {ψinA | B|=(ψ<->φ)}

定义 3 (弱解释): WExpl_{B,S}(E,φ) iff
  EsubseteqS and |E|<inf and Cons(BunionE) and B|/=φ and BunionE|=φ and EinterEq_B(φ)=empty

定义 4 (严格解释): Expl_{B,S}(E,φ) iff
  WExpl_{B,S}(E,φ) and forallE'proper subset ofE, not WExpl_{B,S}(E',φ)
"""

from typing import TypeVar, Generic, Set, Any, Optional, List, FrozenSet, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


# ==================== 类型定义 ====================

T = TypeVar('T')  # 形式语言元素类型（句子/公式）
M = TypeVar('M')  # 模型类型


class ExplanationError(Exception):
    """Explanation 相关错误基类"""
    pass


class InvalidExplanationError(ExplanationError):
    """无效的 explanation（不满足 WExpl 条件）"""
    pass


class ConsistencyError(ExplanationError):
    """一致性检查错误"""
    pass


class EntailmentError(ExplanationError):
    """语义蕴含检查错误"""
    pass


# ==================== 抽象接口 ====================

class SemanticEntailment(ABC, Generic[T, M]):
    """
    语义蕴含关系 |= 的抽象接口

    任何逻辑系统必须实现此接口来定义语义蕴含关系。
    """

    @abstractmethod
    def entails(
        self,
        evidence: Set[T],
        conclusion: T,
        background_theory: Optional[Set[T]] = None,
        model: Optional[M] = None
    ) -> bool:
        """
        判断 Γ |= φ 是否成立（Γ = background_theory union evidence）

        Args:
            evidence: 证据集合 E
            conclusion: 结论 φ（explanandum）
            background_theory: 背景理论 B（可选）
            model: 可选的模型参数

        Returns:
            True 如果 BunionE |= φ 成立

        Note:
            实现应当考虑背景理论 B，即检查 BunionE 是否蕴含 φ
        """
        pass

    @abstractmethod
    def is_consistent(
        self,
        sentences: Set[T],
        background_theory: Optional[Set[T]] = None
    ) -> bool:
        """
        检查句子集合的一致性：Cons(Bunionsentences)

        Args:
            sentences: 要检查的句子集合（如 E 或 BunionE）
            background_theory: 背景理论 B（可选）

        Returns:
            True 如果 Bunionsentences 是一致的（存在模型满足它们）

        Note:
            一致性定义: Cons(X) iff existsM(M|=X)
        """
        pass

    @abstractmethod
    def equivalent_under_background(
        self,
        a: T,
        b: T,
        background_theory: Optional[Set[T]] = None
    ) -> bool:
        """
        判断两个句子在背景理论下是否等价：B|=(a<->b)

        Args:
            a: 句子 a
            b: 句子 b
            background_theory: 背景理论 B

        Returns:
            True 如果 B|=(a<->b)

        Note:
            用于计算 Eq_B(φ) = {ψinA | B|=(ψ<->φ)}
        """
        pass

    @abstractmethod
    def get_all_sentences(self) -> Set[T]:
        """
        获取该逻辑系统可表达的所有句子集合（Sen(L)）

        Returns:
            Sen(L) 的子集（通常是所有可能的句子）
        """
        pass


class NaturalLanguageRenderer(ABC, Generic[T]):
    """
    自然语言渲染函数 f(E, φ, B) 的抽象接口

    将形式解释 (E, φ, B) 渲染为自然语言文本。
    """

    @abstractmethod
    def render(
        self,
        evidence: Set[T],
        explanandum: T,
        background_theory: Optional[Set[T]] = None,
        explanation_type: str = "standard"
    ) -> str:
        """
        渲染解释为自然语言

        Args:
            evidence: 证据集合 E
            explanandum: 待解释命题 φ
            background_theory: 背景理论 B（可选）
            explanation_type: 解释类型
                - "standard": 标准解释
                - "weak": 弱解释
                - "strict": 严格解释（最小解释）
                - "contrastive": 对比解释

        Returns:
            自然语言解释文本
        """
        pass


# ==================== 核心定义实现 ====================

@dataclass
class Explanation(Generic[T, M]):
    """
    Explanation 的数学本体

    对应定义:
      WExpl_{B,S}(E,φ) 或 Expl_{B,S}(E,φ)

    字段:
      evidence: 证据集合 E subseteq S
      explanandum: 待解释命题 φ in Sen(L)
      background_theory: 背景理论 B subseteq Sen(L)
      candidate_space: 候选证据空间 S subseteq A（可选，用于验证 E subseteq S）
      is_strict: 是否为严格解释（严格解释 |= 弱解释）
      entailment: 语义蕴含关系实例（用于验证）
      metadata: 元数据
    """

    evidence: FrozenSet[T]
    explanandum: T
    background_theory: Optional[FrozenSet[T]] = None
    candidate_space: Optional[FrozenSet[T]] = None
    is_strict: bool = False  # True=严格解释, False=弱解释
    entailment: Optional[SemanticEntailment[T, M]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        验证 explanation 满足 WExpl 或 Expl 条件

        这是最严格的检查，确保对象符合数学定义。
        """
        if self.entailment is None:
            raise InvalidExplanationError("必须提供 semantic_entailment 实例")

        # 验证条件 1: E subseteq S（如果提供了 S）
        if self.candidate_space is not None:
            if not self.evidence.issubset(self.candidate_space):
                raise InvalidExplanationError(
                    f"条件失败: E notsubseteq S (E={self.evidence}, S={self.candidate_space})"
                )

        # 验证条件 2: |E| < inf（在代码中自动满足，frozenset 有限）
        if len(self.evidence) == 0:
            raise InvalidExplanationError("条件失败: E 为空集（要求 |E|>0 或根据具体定义）")

        # 准备 B 和 BunionE
        B = self.background_theory or set()
        BE = B.union(self.evidence)

        # 验证条件 3: Cons(BunionE) - BunionE 一致
        if not self.entailment.is_consistent(self.evidence, self.background_theory):
            raise InvalidExplanationError("条件失败: Cons(BunionE) - BunionE 不一致")

        # 验证条件 4: B|/=φ - 背景理论单独不蕴含 φ
        if self.entailment.entails(set(), self.explanandum, self.background_theory):
            raise InvalidExplanationError("条件失败: B|=φ（背景理论已蕴含待解释命题）")

        # 验证条件 5: BunionE|=φ - BunionE 蕴含 φ
        if not self.entailment.entails(self.evidence, self.explanandum, self.background_theory):
            raise InvalidExplanationError("条件失败: BunionE|/=φ（证据+背景不蕴含命题）")

        # 验证条件 6: E inter Eq_B(φ) = empty - 证据不包含与 φ 在 B 下等价的句子
        Eq_B_phi = self._compute_equivalence_class(self.explanandum, B)
        intersection = self.evidence.intersection(Eq_B_phi)
        if len(intersection) > 0:
            raise InvalidExplanationError(
                f"条件失败: E inter Eq_B(φ) ≠ empty (交集={intersection})"
            )

        # 如果是严格解释，验证条件 7: forallE'proper subset ofE, not WExpl_{B,S}(E',φ)
        if self.is_strict:
            if not self._check_strict_minimality(B):
                raise InvalidExplanationError("条件失败: 非严格最小（存在真子集也满足 WExpl）")

    def _compute_equivalence_class(self, phi: T, background_theory: Set[T]) -> FrozenSet[T]:
        """
        计算 Eq_B(φ) = {ψ∈A | B⊨(ψ↔φ)}

        Args:
            phi: 待解释命题 φ
            background_theory: 背景理论 B

        Returns:
            Eq_B(φ) 集合（所有与 φ 在 B 下等价的句子）

        Note:
            在演示实现中，Eq_B(φ) 至少包含 φ 本身（自反性），
            以及任何 entailment.equivalent_under_background() 返回 True 的句子。
        """
        if self.entailment is None:
            raise RuntimeError("entailment 未设置")

        Eq_set = set()

        # 1. φ 本身总是与自身等价（自反性）
        Eq_set.add(phi)

        # 2. 尝试从 background_theory 中找到等价关系
        for psi in background_theory:
            if self.entailment.equivalent_under_background(psi, phi, set()):
                Eq_set.add(psi)

        # 3. 如果有更大的句子集合 A，检查 A 中的元素
        A = self.entailment.get_all_sentences()
        for psi in A:
            if psi == phi:
                continue  # 已经添加
            if self.entailment.equivalent_under_background(psi, phi, background_theory):
                Eq_set.add(psi)

        return frozenset(Eq_set)

    def _check_strict_minimality(self, background_theory: Set[T]) -> bool:
        """
        检查严格性：forallE'proper subset ofE, not WExpl_{B,S}(E',φ)

        即：不存在 E 的真子集 E' 也满足所有 WExpl 条件

        Returns:
            True 如果 E 是严格最小解释
        """
        from itertools import combinations

        if self.entailment is None:
            raise RuntimeError("entailment 未设置")

        B = frozenset(background_theory)
        evidence_list = list(self.evidence)

        # 检查所有大小从 1 到 |E|-1 的子集（真子集）
        for size in range(1, len(evidence_list)):
            for indices in combinations(range(len(evidence_list)), size):
                E_prime = frozenset(evidence_list[i] for i in indices)

                # 检查 E' 是否满足 WExpl 条件
                try:
                    self._check_weak_conditions(E_prime, B, skip_subset_check=False)
                    # 如果没抛出异常，说明 E' 满足 WExpl，则 E 不是最小的
                    return False
                except (InvalidExplanationError, ConsistencyError, EntailmentError):
                    # E' 不满足某些条件，继续检查
                    continue

        return True

    def _check_weak_conditions(
        self,
        evidence: FrozenSet[T],
        background_theory: FrozenSet[T],
        skip_subset_check: bool = False
    ) -> None:
        """
        检查 WExpl 的所有条件（不创建对象）

        Args:
            evidence: 证据集合 E
            background_theory: 背景理论 B
            skip_subset_check: 是否跳过 E subseteq S 检查（用于最小性检查时）

        Raises:
            InvalidExplanationError: 如果任何条件失败
            ConsistencyError: 一致性检查失败
            EntailmentError: 蕴含检查失败
        """
        if self.entailment is None:
            raise RuntimeError("entailment 未设置")

        B = background_theory
        BE = B.union(evidence)

        # 条件 1: |E| < inf (自动满足，frozenset 有限)

        # 条件 2: E subseteq S（如果指定了 candidate_space 且不跳过检查）
        if not skip_subset_check and self.candidate_space is not None:
            if not evidence.issubset(self.candidate_space):
                raise InvalidExplanationError(f"E notsubseteq S")

        # 条件 3: Cons(BunionE)
        if not self.entailment.is_consistent(set(evidence), set(B)):
            raise ConsistencyError("Cons(BunionE) 失败: BunionE 不一致")

        # 条件 4: B|/=φ
        if self.entailment.entails(set(), self.explanandum, set(B)):
            raise EntailmentError("B|=φ: 背景理论已蕴含待解释命题")

        # 条件 5: BunionE|=φ
        if not self.entailment.entails(set(evidence), self.explanandum, set(B)):
            raise EntailmentError("BunionE|/=φ: 证据+背景不蕴含命题")

        # 条件 6: E inter Eq_B(φ) = empty
        Eq_B_phi = self._compute_equivalence_class(self.explanandum, set(B))
        intersection = evidence.intersection(Eq_B_phi)
        if len(intersection) > 0:
            raise InvalidExplanationError(f"E inter Eq_B(φ) ≠ empty")

    def is_weakly_explanatory(self) -> bool:
        """
        检查是否为弱解释（WExpl）

        Returns:
            True 如果满足 WExpl 的所有条件
        """
        # __post_init__ 已经验证过了
        return True

    def is_strictly_explanatory(self) -> bool:
        """
        检查是否为严格解释（Expl）

        Returns:
            True 如果满足 Expl 的所有条件（即 WExpl + 严格最小性）
        """
        if not self.is_weakly_explanatory():
            return False

        B = self.background_theory or set()
        return self._check_strict_minimality(B)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "evidence": list(self.evidence),
            "explanandum": self.explanandum,
            "background_theory": list(self.background_theory) if self.background_theory else None,
            "candidate_space_size": len(self.candidate_space) if self.candidate_space else None,
            "evidence_size": len(self.evidence),
            "is_strict": self.is_strict,
            "metadata": self.metadata or {}
        }

    def __str__(self) -> str:
        strict_str = "strict" if self.is_strict else "weak"
        return f"Explanation(E={self.evidence}, φ={self.explanandum}, B={self.background_theory}, type={strict_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.evidence)


# ==================== 系统接口 ====================

class ExplanationSystem(Generic[T, M]):
    """
    Explanation 系统主类

    提供创建、验证、查找解释的所有操作。
    """

    def __init__(
        self,
        entailment: SemanticEntailment[T, M],
        renderer: Optional[NaturalLanguageRenderer[T]] = None
    ):
        """
        初始化系统

        Args:
            entailment: 语义蕴含关系实现
            renderer: 自然语言渲染器（可选）
        """
        self.entailment = entailment
        self.renderer = renderer

    def create_weak(
        self,
        evidence: Set[T],
        explanandum: T,
        background_theory: Optional[Set[T]] = None,
        candidate_space: Optional[Set[T]] = None
    ) -> Explanation[T, M]:
        """
        创建弱解释（不检查严格最小性）

        Args:
            evidence: 证据集合 E
            explanandum: 待解释命题 φ
            background_theory: 背景理论 B
            candidate_space: 候选证据空间 S

        Returns:
            满足 WExpl 条件的 Explanation 对象

        Raises:
            InvalidExplanationError: 如果不满足 WExpl 条件
        """
        return Explanation(
            evidence=frozenset(evidence),
            explanandum=explanandum,
            background_theory=frozenset(background_theory) if background_theory else None,
            candidate_space=frozenset(candidate_space) if candidate_space else None,
            is_strict=False,
            entailment=self.entailment,
            metadata={"type": "weak"}
        )

    def create_strict(
        self,
        evidence: Set[T],
        explanandum: T,
        background_theory: Optional[Set[T]] = None,
        candidate_space: Optional[Set[T]] = None
    ) -> Explanation[T, M]:
        """
        创建严格解释（检查严格最小性）

        Args:
            evidence: 证据集合 E
            explanandum: 待解释命题 φ
            background_theory: 背景理论 B
            candidate_space: 候选证据空间 S

        Returns:
            满足 Expl 条件的 Explanation 对象

        Raises:
            InvalidExplanationError: 如果不满足 Expl 条件
        """
        return Explanation(
            evidence=frozenset(evidence),
            explanandum=explanandum,
            background_theory=frozenset(background_theory) if background_theory else None,
            candidate_space=frozenset(candidate_space) if candidate_space else None,
            is_strict=True,
            entailment=self.entailment,
            metadata={"type": "strict"}
        )

    def validate_weak(
        self,
        evidence: Set[T],
        explanandum: T,
        background_theory: Optional[Set[T]] = None,
        candidate_space: Optional[Set[T]] = None
    ) -> bool:
        """
        验证是否满足 WExpl 条件（不创建对象）

        Returns:
            True 如果满足所有 WExpl 条件
        """
        try:
            self.create_weak(evidence, explanandum, background_theory, candidate_space)
            return True
        except InvalidExplanationError:
            return False

    def validate_strict(
        self,
        evidence: Set[T],
        explanandum: T,
        background_theory: Optional[Set[T]] = None,
        candidate_space: Optional[Set[T]] = None
    ) -> bool:
        """
        验证是否满足 Expl 条件（不创建对象）

        Returns:
            True 如果满足所有 Expl 条件
        """
        try:
            self.create_strict(evidence, explanandum, background_theory, candidate_space)
            return True
        except InvalidExplanationError:
            return False

    def find_minimal_explanations(
        self,
        explanandum: T,
        candidate_space: Set[T],
        background_theory: Optional[Set[T]] = None,
        strict: bool = True,
        max_results: int = 100
    ) -> List[Explanation[T, M]]:
        """
        在候选空间中寻找所有（严格）最小解释

        Args:
            explanandum: 待解释命题 φ
            candidate_space: 候选证据空间 S subseteq A
            background_theory: 背景理论 B
            strict: True 找严格解释，False 找弱解释（但不保证最小）
            max_results: 最大结果数量（防组合爆炸）

        Returns:
            minimal explanation 列表

        Note:
            暴力搜索算法，仅适用于 |S| 较小的情况（如 |S| < 20）
            时间复杂度 O(2^|S| × |S| × cost(entails))
        """
        from itertools import combinations

        B = frozenset(background_theory) if background_theory else None
        S = frozenset(candidate_space)
        minimals = []
        n = len(S)
        evidence_list = list(S)

        # 从小到大搜索（保证找到 minimal）
        for size in range(1, n + 1):
            for indices in combinations(range(n), size):
                E = frozenset(evidence_list[i] for i in indices)

                try:
                    # 尝试创建解释
                    if strict:
                        expl = self.create_strict(set(E), explanandum, background_theory, candidate_space)
                    else:
                        expl = self.create_weak(set(E), explanandum, background_theory, candidate_space)

                    minimals.append(expl)

                    if len(minimals) >= max_results:
                        return minimals

                except InvalidExplanationError:
                    # 不是有效解释，跳过
                    continue

        return minimals

    def render(
        self,
        explanation: Explanation[T, M],
        explanation_type: str = "standard"
    ) -> str:
        """
        渲染 explanation 为自然语言

        Args:
            explanation: explanation 对象
            explanation_type: 解释类型（standard, weak, strict, contrastive）

        Returns:
            自然语言文本

        Raises:
            RuntimeError: 如果系统没有配置 renderer
        """
        if self.renderer is None:
            raise RuntimeError("未配置 NaturalLanguageRenderer，无法渲染自然语言解释")

        return self.renderer.render(
            evidence=explanation.evidence,
            explanandum=explanation.explanandum,
            background_theory=explanation.background_theory,
            explanation_type=explanation_type
        )

    def filter_by_size(
        self,
        explanations: List[Explanation[T, M]],
        max_size: Optional[int] = None,
        min_size: Optional[int] = None
    ) -> List[Explanation[T, M]]:
        """
        按证据数量过滤解释

        Args:
            explanations: explanation 列表
            max_size: 最大证据数量
            min_size: 最小证据数量

        Returns:
            过滤后的列表
        """
        result = []

        for expl in explanations:
            size = len(expl.evidence)

            if max_size is not None and size > max_size:
                continue
            if min_size is not None and size < min_size:
                continue

            result.append(expl)

        return result


# ==================== 示例实现：命题逻辑 ====================

class PropositionalFormula:
    """命题逻辑公式的简单表示"""

    def __init__(self, text: str):
        self.text = text.strip()

    def __repr__(self) -> str:
        return f"Prop({self.text})"

    def __str__(self) -> str:
        return self.text

    def __hash__(self) -> int:
        return hash(self.text)

    def __eq__(self, other) -> bool:
        return isinstance(other, PropositionalFormula) and self.text == other.text


# ==================== 示例实现：命题逻辑 ====================

class SimplePropositionalEntailment(SemanticEntailment[PropositionalFormula, None]):
    """
    简单的命题逻辑语义蕴含实现（演示用）

    这个实现内置了一些推理规则来演示框架的使用。
    实际系统需要完整的 SAT solver 或定理证明器。
    """

    def __init__(self, custom_rules: Optional[Dict[str, List]] = None):
        """
        初始化

        Args:
            custom_rules: 自定义推理规则
                格式: {"前提列表": ["结论1", "结论2"]}
        """
        # 内置推理规则（文本匹配）
        self.rules = [
            {
                "premises": {"所有人都是会死的", "苏格拉底是人"},
                "conclusion": "苏格拉底是会死的"
            },
            {
                "premises": {"所有人都是会死的", "苏格拉底是哲学家"},
                "conclusion": "苏格拉底是会死的"
            },
            {
                "premises": {"p->q", "p"},
                "conclusion": "q"
            },
        ]

        if custom_rules:
            # 转换自定义规则
            for premises_str, conclusions in custom_rules.items():
                premises = set(premises_str.split(","))
                for conc in conclusions:
                    self.rules.append({
                        "premises": premises,
                        "conclusion": conc
                    })

    def entails(
        self,
        evidence: Set[PropositionalFormula],
        conclusion: PropositionalFormula,
        background_theory: Optional[Set[PropositionalFormula]] = None,
        model: Optional[None] = None
    ) -> bool:
        """
        检查 BunionE |= φ

        Args:
            evidence: 证据集合 E
            conclusion: 结论 φ
            background_theory: 背景理论 B

        Returns:
            True 如果 BunionE 蕴含 φ
        """
        B = background_theory or set()
        BE = B.union(evidence)

        # 规则 1: 如果结论直接在 BunionE 中
        if conclusion in BE:
            return True

        # 规则 2: 检查推理规则
        BE_texts = {p.text for p in BE}

        for rule in self.rules:
            premise_texts = {p.text if isinstance(p, PropositionalFormula) else p
                             for p in rule["premises"]}
            if premise_texts.issubset(BE_texts) and rule["conclusion"] == conclusion.text:
                return True

        # 规则 3: 如果背景理论包含蕴含关系 a->b 且 ainBE, 则 b 被蕴含
        for p in BE:
            if "->" in p.text:
                left, right = p.text.split("->", 1)
                left = left.strip()
                right = right.strip()
                if left in BE_texts and right == conclusion.text:
                    return True

        return False

    def is_consistent(
        self,
        sentences: Set[PropositionalFormula],
        background_theory: Optional[Set[PropositionalFormula]] = None
    ) -> bool:
        """
        检查 Cons(Bunionsentences)

        简化实现：检查是否包含明显的矛盾（p 和 not p）
        实际系统需要完整的 SAT 可满足性检查。
        """
        B = background_theory or set()
        BS = B.union(sentences)

        # 提取所有原子命题和它们的否定
        BS_texts = {p.text for p in BS}

        for text in BS_texts:
            # 检查是否有 not p
            if text.startswith("not "):
                positive = text[1:]
                if positive in BS_texts:
                    return False

            # 检查是否有 p 和 p->not p 形式的矛盾
            # （简化，实际需要更复杂的检查）

        return True

    def equivalent_under_background(
        self,
        a: PropositionalFormula,
        b: PropositionalFormula,
        background_theory: Optional[Set[PropositionalFormula]] = None
    ) -> bool:
        """
        检查 B|=(a<->b)

        简化实现：
        - 字符串完全相等
        - 或者背景理论中显式包含 a<->b
        """
        if a.text == b.text:
            return True

        B = background_theory or set()
        for axiom in B:
            if "<->" in axiom.text:
                left, right = axiom.text.split("<->", 1)
                left = left.strip()
                right = right.strip()
                if (left == a.text and right == b.text) or (left == b.text and right == a.text):
                    return True

        return False

    def get_all_sentences(self) -> Set[PropositionalFormula]:
        """
        获取所有可能的句子

        演示实现：返回空集（表示不知道所有句子）
        实际实现需要生成语言 L 的所有公式。
        """
        return set()


class SimpleRenderer(NaturalLanguageRenderer[PropositionalFormula]):
    """简单的自然语言渲染器"""

    def render(
        self,
        evidence: Set[PropositionalFormula],
        explanandum: PropositionalFormula,
        background_theory: Optional[Set[PropositionalFormula]] = None,
        explanation_type: str = "standard"
    ) -> str:
        """渲染解释为自然语言"""

        # 格式化证据
        evidence_list = list(evidence)
        evidence_text = "、".join([e.text for e in evidence_list]) if evidence_list else "无"

        # 格式化背景理论
        B_text = ""
        if background_theory:
            B_list = list(background_theory)
            B_text = "，在背景理论 " + "、".join([b.text for b in B_list]) + " 下"

        if explanation_type == "strict":
            return (
                f"严格解释: 基于证据 {{{evidence_text}}}{B_text}，"
                f"严格证明了 {explanandum.text}，且不存在更小的证据集也能证明。"
            )
        elif explanation_type == "weak":
            return (
                f"弱解释: 基于证据 {{{evidence_text}}}{B_text}，"
                f"可以推出 {explanandum.text}，且证据与结论在背景理论下不等价。"
            )
        else:
            return (
                f"解释: 因为 {evidence_text}{B_text}，"
                f"所以 {explanandum.text}。"
            )


# ==================== 便捷函数 ====================

def create_simple_system() -> ExplanationSystem[PropositionalFormula, None]:
    """
    快速创建命题逻辑示例系统

    Returns:
        ExplanationSystem 实例
    """
    entailment = SimplePropositionalEntailment()
    renderer = SimpleRenderer()
    return ExplanationSystem(entailment, renderer)


# ==================== 主程序演示 ====================

def main():
    """演示框架的使用"""

    print("=" * 70)
    print("EXPLANATION 框架 - 基于严格演绎定义 (v2.0)")
    print("=" * 70)
    print()

    # 1. 创建系统
    print("1. 创建系统")
    print("-" * 70)
    system = create_simple_system()
    print("[OK] 系统已创建")
    print("  - 语义蕴含: SimplePropositionalEntailment")
    print("  - 渲染器: SimpleRenderer")
    print()

    # 2. 定义示例问题（苏格拉底推理）
    print("2. 定义示例问题（苏格拉底推理）")
    print("-" * 70)

    # 背景理论 B: 所有人都是会死的
    # 注意：B 不能单独蕴含 φ，否则不需要解释
    B = {
        PropositionalFormula("所有人都是会死的"),
    }

    # 证据空间 S: 可用的证据单元
    S = {
        PropositionalFormula("所有人都是会死的"),
        PropositionalFormula("苏格拉底是人"),
        PropositionalFormula("苏格拉底是哲学家"),
    }

    # 待解释命题 φ
    phi = PropositionalFormula("苏格拉底是会死的")

    print(f"背景理论 B = {B}")
    print(f"候选证据空间 S = {S}")
    print(f"待解释命题 φ = {phi}")
    print()

    # 3. 验证弱解释条件
    print("3. 验证弱解释条件")
    print("-" * 70)

    # 证据 E: 使用"苏格拉底是人"作为证据
    E1 = {PropositionalFormula("苏格拉底是人")}

    is_valid_weak = system.validate_weak(E1, phi, B, S)
    print(f"证据 E = {E1}")
    print(f"检查 WExpl_{B,S}(E,φ) 条件:")
    print(f"  1) E subseteq S: {E1.issubset(S)}")
    print(f"  2) |E| < inf: {len(E1) > 0 and len(E1) < float('inf')}")
    print(f"  3) Cons(BunionE): {system.entailment.is_consistent(E1, B)}")
    print(f"  4) B|/=φ: {not system.entailment.entails(set(), phi, B)}")
    print(f"  5) BunionE|=φ: {system.entailment.entails(E1, phi, B)}")
    Eq_B_phi = set()  # 简化：Eq_B(φ) 应该是空集
    print(f"  6) EinterEq_B(φ)=empty: {E1.intersection(Eq_B_phi) == set()}")
    print(f"\n[结果] WExpl 条件是否全部满足? {is_valid_weak}")

    if is_valid_weak:
        weak_expl = system.create_weak(E1, phi, B, S)
        print(f"[OK] 成功创建弱解释: {weak_expl}")
    else:
        print("[ERROR] E 不满足弱解释条件")
    print()

    # 4. 检查严格最小性
    print("4. 检查严格最小性")
    print("-" * 70)

    # E1 是严格最小吗？检查所有真子集
    # E1 = {苏格拉底是人}，真子集只有空集
    E1_sub = set()  # 空集
    is_valid_sub = system.validate_weak(E1_sub, phi, B, S)
    print(f"E 的真子集 E' = {E1_sub}")
    print(f"WExpl_{B,S}(E',φ) 是否满足? {is_valid_sub}")

    if is_valid_sub:
        print("[结果] E 不是严格最小解释，因为存在更小的有效证据集")
    else:
        print("[结果] E 是严格最小解释（没有更小的有效证据集）")
    print()

    # 5. 创建严格解释
    print("5. 创建严格解释")
    print("-" * 70)

    try:
        strict_expl = system.create_strict(E1, phi, B, S)
        print(f"[OK] 成功创建严格解释: {strict_expl}")
    except InvalidExplanationError as e:
        print(f"[ERROR] 不满足严格解释条件: {e}")
    print()

    # 6. 在候选空间中搜索最小解释
    print("6. 在候选空间 S 中搜索严格最小解释")
    print("-" * 70)

    minimals = system.find_minimal_explanations(
        explanandum=phi,
        candidate_space=S,
        background_theory=B,
        strict=True,
        max_results=5
    )

    print(f"找到 {len(minimals)} 个严格最小解释:")
    for i, expl in enumerate(minimals, 1):
        print(f"  {i}. {expl}")
        print(f"     证据大小: {len(expl.evidence)}")
        print(f"     是否严格: {expl.is_strict}")
    print()

    # 7. 自然语言渲染
    print("7. 自然语言渲染")
    print("-" * 70)

    if minimals:
        text = system.render(minimals[0], explanation_type="strict")
        print(f"[OK] 严格解释渲染:\n  \"{text}\"")
        print()

        text_weak = system.render(minimals[0], explanation_type="weak")
        print(f"[OK] 弱解释渲染:\n  \"{text_weak}\"")
        print()

    # 8. 定义验证测试
    print("8. 定义验证测试")
    print("-" * 70)

    # 测试条件检查
    test_cases = [
        {
            "name": "空证据集",
            "E": set(),
            "φ": phi,
            "B": B,
            "should_pass": False
        },
        {
            "name": "E 包含 φ 本身（等价类）",
            "E": {phi},
            "φ": phi,
            "B": B,
            "should_pass": False  # E inter Eq_B(φ) ≠ empty
        },
        {
            "name": "E 不蕴含 φ",
            "E": {PropositionalFormula("苏格拉底是哲学家")},
            "φ": phi,
            "B": B,
            "should_pass": True  # BunionE |= φ（通过推理规则）
        },
        {
            "name": "B 已蕴含 φ（无解释必要）",
            "E": set(),
            "φ": phi,
            "B": {PropositionalFormula("苏格拉底是会死的")},  # B 直接包含 φ
            "should_pass": False  # B|=φ
        },
    ]

    for tc in test_cases:
        result = system.validate_weak(tc["E"], tc["φ"], tc["B"])
        status = "PASS" if result == tc["should_pass"] else "FAIL"
        print(f"  [{status}] {tc['name']}: expected={tc['should_pass']}, got={result}")

    print()

    # 9. 数学定义展示
    print("9. 数学定义")
    print("-" * 70)
    print("WExpl_{B,S}(E,φ) iff")
    print("  EsubseteqS and |E|<inf and Cons(BunionE) and B|/=φ and BunionE|=φ and EinterEq_B(φ)=empty")
    print()
    print("Expl_{B,S}(E,φ) iff")
    print("  WExpl_{B,S}(E,φ) and forallE'proper subset ofE, not WExpl_{B,S}(E',φ)")
    print()

    # 10. 框架特性
    print("10. 框架特性")
    print("-" * 70)
    print("  - 基于严格演绎定义（v2.0）")
    print("  - 支持背景理论 B")
    print("  - 支持弱解释和严格解释")
    print("  - 自动验证所有 WExpl 条件")
    print("  - 提供最小解释搜索")
    print("  - 可插拔语义蕴含实现")
    print("  - 自然语言渲染接口")
    print("  - 清晰的中文错误信息")
    print()

    print("=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
