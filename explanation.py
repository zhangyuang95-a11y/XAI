"""
explanation.py — Explanation 数学定义的严格 Python 实现

严格对应以下三条形式化定义：

  (1) 内部解释基础 (Basis):
      Basis_{u,t}(E, Q) ⇔ E ⊆ S_t
                          ∧ True_t(E)
                          ∧ Faithful_π(E, a_t)
                          ∧ Contrastive_π(E, a_t, Δ_t)

  (2) 最小性 (Minimality):
      Minimal(E) ⇔ ∀E' ⊊ E, ¬Basis_{u,t}(E', Q)

  (3) 最终 explanation:
      Explain_u(Q, t, x) ⇔ ∃E ⊆ S_t (
          Basis_{u,t}(E, Q)
        ∧ Minimal(E)
        ∧ x = R_u(E, Q)
        ∧ Readable_u(x)
      )

符号映射：
  Q   — WhyQuestion（why-question）
  u   — UserModel（用户模型）
  t   — 时刻（int / float）
  a_t — 当前动作（str）
  Δ_t — 对比动作集合（Set[str]）
  S_t — 候选解释因素空间（Set[ExplanationFactor]）
  E   — 内部解释因素集合（FrozenSet[ExplanationFactor]）
  x   — 最终自然语言 explanation（NaturalLanguageExplanation）
  R_u — 面向用户的渲染函数（UserModel.render）
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional, Set


# ═══════════════════════════════════════════════════════════════
#  基本数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WhyQuestion:
    """Q — why-question，表示用户提出的"为什么"问题。

    Attributes:
        text:    问题的自然语言文本
        topic:   问题所针对的主题/对象
        context: 附加的语境信息
    """
    text: str
    topic: str
    context: str = ""


@dataclass(frozen=True)
class ExplanationFactor:
    """解释因素 — S_t 空间中的单个元素。

    每个因素自身携带三个谓词的求值结果：
      - is_true:        True_t  — 在时刻 t 该因素是否为真
      - is_faithful:    Faithful_π — 是否忠实反映策略 π 对 a_t 的依赖
      - is_contrastive: Contrastive_π — 是否能区分 a_t 与 Δ_t

    Attributes:
        name:            因素名称（唯一标识）
        description:     自然语言描述
        is_true:         True_t(factor)
        is_faithful:     Faithful_π(factor, a_t)
        is_contrastive:  Contrastive_π(factor, a_t, Δ_t)
    """
    name: str
    description: str
    is_true: bool = True
    is_faithful: bool = True
    is_contrastive: bool = True

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExplanationFactor):
            return NotImplemented
        return self.name == other.name


@dataclass
class UserModel:
    """u — 用户模型，封装面向特定用户的渲染函数 R_u 和可读性判定 Readable_u。

    Attributes:
        user_id:       用户标识
        language:      用户语言
        detail_level:  详细程度 ("low" / "medium" / "high")
        render_fn:     R_u(E, Q) → str  可选的自定义渲染函数
        readable_fn:   Readable_u(x) → bool  可选的自定义可读性判定
    """
    user_id: str
    language: str = "zh"
    detail_level: str = "medium"
    render_fn: Optional[Callable[[FrozenSet[ExplanationFactor], WhyQuestion], str]] = None
    readable_fn: Optional[Callable[[str], bool]] = None

    # ── R_u: 渲染函数 ──────────────────────────────────────────
    def render(self, factors: FrozenSet[ExplanationFactor], question: WhyQuestion) -> str:
        """R_u(E, Q) — 将内部因素集合渲染为面向用户的自然语言文本。"""
        if self.render_fn is not None:
            return self.render_fn(factors, question)
        # 默认渲染：按因素的 description 连接
        parts = [f.description for f in sorted(factors, key=lambda f: f.name)]
        return f"关于「{question.topic}」：" + "；".join(parts) + "。"

    # ── Readable_u: 可读性判定 ─────────────────────────────────
    def is_readable(self, text: str) -> bool:
        """Readable_u(x) — 判断渲染结果对该用户是否可读。"""
        if self.readable_fn is not None:
            return self.readable_fn(text)
        # 默认：非空且长度合理即为可读
        return 0 < len(text) <= 2000


# ═══════════════════════════════════════════════════════════════
#  ExplanationBasis — 核心：定义 (1) Basis 与 (2) Minimal
# ═══════════════════════════════════════════════════════════════

# 集合级谓词类型：接收因素集合，返回 bool
SetPredicate = Callable[[FrozenSet[ExplanationFactor]], bool]


def _default_true_t(factors: FrozenSet[ExplanationFactor]) -> bool:
    """默认 True_t(E)：逐元素合取 — 每个因素的 is_true 均为真。"""
    return all(f.is_true for f in factors)


def _default_faithful(factors: FrozenSet[ExplanationFactor]) -> bool:
    """默认 Faithful_π(E, a_t)：逐元素合取。"""
    return all(f.is_faithful for f in factors)


def _default_contrastive(factors: FrozenSet[ExplanationFactor]) -> bool:
    """默认 Contrastive_π(E, a_t, Δ_t)：逐元素合取。"""
    return all(f.is_contrastive for f in factors)


@dataclass
class ExplanationBasis:
    """对应定义 (1)(2)，封装内部解释因素集合 E 及其判定逻辑。

    数学对应：
      E   — factors（内部解释因素集合）
      S_t — candidate_space（候选因素空间）
      Q   — question
      u   — user_model
      t   — timestep
      a_t — current_action
      Δ_t — contrastive_actions

    集合级谓词：
      True_t(E)、Faithful_π(E, a_t)、Contrastive_π(E, a_t, Δ_t) 在数学定义中
      是作用于整个集合 E 的谓词，而非简单的逐元素合取。通过 true_t_fn / faithful_fn /
      contrastive_fn 可注入自定义集合级判定逻辑；若未提供则回退到逐元素合取。
    """
    factors: FrozenSet[ExplanationFactor]          # E
    candidate_space: FrozenSet[ExplanationFactor]   # S_t
    question: WhyQuestion                           # Q
    user_model: UserModel                           # u
    timestep: int                                   # t
    current_action: str                             # a_t
    contrastive_actions: Set[str]                   # Δ_t

    # 可选的集合级谓词（覆盖默认的逐元素合取）
    true_t_fn: SetPredicate = field(default=_default_true_t, repr=False)
    faithful_fn: SetPredicate = field(default=_default_faithful, repr=False)
    contrastive_fn: SetPredicate = field(default=_default_contrastive, repr=False)

    # ── 定义 (1): Basis_{u,t}(E, Q) ───────────────────────────
    def is_basis(self) -> bool:
        """Basis_{u,t}(E, Q) ⇔ E ⊆ S_t
                               ∧ True_t(E)
                               ∧ Faithful_π(E, a_t)
                               ∧ Contrastive_π(E, a_t, Δ_t)

        Returns:
            True 当且仅当 E 满足上述四个联合条件。
        """
        return (
            self._subset_of_space()
            and self._eval_true_t()
            and self._eval_faithful()
            and self._eval_contrastive()
        )

    # ── 定义 (2): Minimal(E) ──────────────────────────────────
    def is_minimal(self) -> bool:
        """Minimal(E) ⇔ ∀E' ⊊ E, ¬Basis_{u,t}(E', Q)

        遍历 E 的所有真子集 E'（非空），验证没有任何一个也满足 Basis。

        Returns:
            True 当且仅当 E 自身满足 Basis 且其任何真子集都不满足 Basis。
        """
        if not self.is_basis():
            return False

        factors_list = list(self.factors)
        n = len(factors_list)
        for size in range(1, n):  # 大小 1 .. |E|-1
            for subset in itertools.combinations(factors_list, size):
                sub_basis = ExplanationBasis(
                    factors=frozenset(subset),
                    candidate_space=self.candidate_space,
                    question=self.question,
                    user_model=self.user_model,
                    timestep=self.timestep,
                    current_action=self.current_action,
                    contrastive_actions=self.contrastive_actions,
                    true_t_fn=self.true_t_fn,
                    faithful_fn=self.faithful_fn,
                    contrastive_fn=self.contrastive_fn,
                )
                if sub_basis.is_basis():
                    return False  # 找到一个满足 Basis 的真子集 → 非最小
        return True

    # ── 内部谓词求值 ─────────────────────────────────────────
    def _subset_of_space(self) -> bool:
        """E ⊆ S_t"""
        return self.factors.issubset(self.candidate_space)

    def _eval_true_t(self) -> bool:
        """True_t(E) — 集合级谓词。"""
        return self.true_t_fn(self.factors)

    def _eval_faithful(self) -> bool:
        """Faithful_π(E, a_t) — 集合级谓词。"""
        return self.faithful_fn(self.factors)

    def _eval_contrastive(self) -> bool:
        """Contrastive_π(E, a_t, Δ_t) — 集合级谓词。"""
        return self.contrastive_fn(self.factors)


# ═══════════════════════════════════════════════════════════════
#  NaturalLanguageExplanation — x
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NaturalLanguageExplanation:
    """x — 最终面向用户的自然语言 explanation。

    Attributes:
        text:       渲染后的自然语言文本 (x = R_u(E, Q))
        factors:    生成该文本所依据的因素集合 E
        question:   对应的 why-question Q
        user_model: 渲染所面向的用户模型 u
    """
    text: str
    factors: FrozenSet[ExplanationFactor]
    question: WhyQuestion
    user_model: UserModel


# ═══════════════════════════════════════════════════════════════
#  Explanation — 定义 (3): Explain_u(Q, t, x)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Explanation:
    """对应定义 (3)，将 Basis、Minimal、R_u、Readable_u 组合为完整的 explanation 判定。

    Explain_u(Q, t, x) ⇔ ∃E ⊆ S_t (
        Basis_{u,t}(E, Q)
      ∧ Minimal(E)
      ∧ x = R_u(E, Q)
      ∧ Readable_u(x)
    )
    """
    basis: ExplanationBasis                        # 封装 E, S_t, Q, u, t, a_t, Δ_t
    nl_explanation: NaturalLanguageExplanation      # x

    def is_valid_explanation(self) -> bool:
        """Explain_u(Q, t, x) — 验证完整 explanation 是否成立。

        四个联合条件：
          1. Basis_{u,t}(E, Q)
          2. Minimal(E)
          3. x = R_u(E, Q)         （渲染一致性）
          4. Readable_u(x)         （可读性）
        """
        # 条件 1: Basis
        if not self.basis.is_basis():
            return False

        # 条件 2: Minimal
        if not self.basis.is_minimal():
            return False

        # 条件 3: x = R_u(E, Q) — 渲染一致性
        expected_text = self.basis.user_model.render(
            self.basis.factors, self.basis.question
        )
        if self.nl_explanation.text != expected_text:
            return False

        # 条件 4: Readable_u(x)
        if not self.basis.user_model.is_readable(self.nl_explanation.text):
            return False

        return True


# ═══════════════════════════════════════════════════════════════
#  ExplanationSystem — 系统级入口：构造、渲染、验证
# ═══════════════════════════════════════════════════════════════

class ExplanationSystem:
    """explanation 系统，负责构造、渲染、验证完整 explanation 流程。"""

    @staticmethod
    def create_basis(
        factors: FrozenSet[ExplanationFactor],
        candidate_space: FrozenSet[ExplanationFactor],
        question: WhyQuestion,
        user_model: UserModel,
        timestep: int,
        current_action: str,
        contrastive_actions: Set[str],
        true_t_fn: SetPredicate = _default_true_t,
        faithful_fn: SetPredicate = _default_faithful,
        contrastive_fn: SetPredicate = _default_contrastive,
    ) -> ExplanationBasis:
        """构造 ExplanationBasis 并做基本合法性检查。"""
        if not factors:
            raise ValueError("因素集合 E 不能为空")
        if not candidate_space:
            raise ValueError("候选空间 S_t 不能为空")
        if not factors.issubset(candidate_space):
            extra = factors - candidate_space
            raise ValueError(f"因素 {{{', '.join(f.name for f in extra)}}} 不在候选空间 S_t 中")

        return ExplanationBasis(
            factors=factors,
            candidate_space=candidate_space,
            question=question,
            user_model=user_model,
            timestep=timestep,
            current_action=current_action,
            contrastive_actions=contrastive_actions,
            true_t_fn=true_t_fn,
            faithful_fn=faithful_fn,
            contrastive_fn=contrastive_fn,
        )

    @staticmethod
    def render_explanation(
        basis: ExplanationBasis,
    ) -> NaturalLanguageExplanation:
        """x = R_u(E, Q) — 使用用户模型的渲染函数生成自然语言 explanation。"""
        text = basis.user_model.render(basis.factors, basis.question)
        return NaturalLanguageExplanation(
            text=text,
            factors=basis.factors,
            question=basis.question,
            user_model=basis.user_model,
        )

    @staticmethod
    def create_explanation(
        basis: ExplanationBasis,
    ) -> Explanation:
        """一步完成：渲染 + 组装 Explanation 对象。"""
        nl = ExplanationSystem.render_explanation(basis)
        return Explanation(basis=basis, nl_explanation=nl)

    @staticmethod
    def validate_explanation(explanation: Explanation) -> dict[str, bool]:
        """逐条验证 Explain_u(Q, t, x) 的四个子条件，返回详细报告。

        Returns:
            {
              "E ⊆ S_t":                        bool,
              "True_t(E)":                       bool,
              "Faithful_π(E, a_t)":              bool,
              "Contrastive_π(E, a_t, Δ_t)":      bool,
              "Basis_{u,t}(E, Q)":               bool,
              "Minimal(E)":                      bool,
              "x = R_u(E, Q)":                   bool,
              "Readable_u(x)":                   bool,
              "Explain_u(Q, t, x)":              bool,
            }
        """
        b = explanation.basis

        subset = b._subset_of_space()
        true_t = b._eval_true_t()
        faithful = b._eval_faithful()
        contrastive = b._eval_contrastive()
        basis_ok = subset and true_t and faithful and contrastive
        minimal = b.is_minimal()

        expected_text = b.user_model.render(b.factors, b.question)
        render_match = (explanation.nl_explanation.text == expected_text)
        readable = b.user_model.is_readable(explanation.nl_explanation.text)

        valid = basis_ok and minimal and render_match and readable

        return {
            "E ⊆ S_t":                   subset,
            "True_t(E)":                  true_t,
            "Faithful_π(E, a_t)":         faithful,
            "Contrastive_π(E, a_t, Δ_t)": contrastive,
            "Basis_{u,t}(E, Q)":          basis_ok,
            "Minimal(E)":                 minimal,
            "x = R_u(E, Q)":             render_match,
            "Readable_u(x)":             readable,
            "Explain_u(Q, t, x)":        valid,
        }


# ═══════════════════════════════════════════════════════════════
#  __main__ 示例：为什么这个时候左转？
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io
    # 确保 Windows 控制台以 UTF-8 输出中文
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 64)
    print("  Explanation 数学定义 -- 验证示例")
    print("  场景: 自动驾驶, 为什么这个时候左转?")
    print("=" * 64)

    # -- 1. 构造 Q (why-question) --------------------------
    Q = WhyQuestion(
        text="为什么这个时候左转?",
        topic="左转决策",
        context="城市十字路口, 自动驾驶场景",
    )

    # -- 2. 构造 u (用户模型) ------------------------------
    u = UserModel(user_id="driver_01", language="zh", detail_level="medium")

    # -- 3. 定义 t, a_t, Delta_t ---------------------------
    t = 42                                         # 时刻
    a_t = "左转"                                   # 当前动作
    delta_t: Set[str] = {"直行", "右转", "停车"}   # 对比动作集合 Delta_t

    # -- 4. 构造 S_t (候选因素空间) ------------------------
    f1 = ExplanationFactor(
        name="navigation_route",
        description="导航路线要求在此路口左转",
        is_true=True, is_faithful=True, is_contrastive=True,
    )
    f2 = ExplanationFactor(
        name="traffic_signal",
        description="当前左转信号灯为绿色",
        is_true=True, is_faithful=True, is_contrastive=True,
    )
    f3 = ExplanationFactor(
        name="oncoming_gap",
        description="对向来车安全间隙足够(>4秒)",
        is_true=True, is_faithful=True, is_contrastive=True,
    )
    # 额外因素: 真实但不具备对比性 -> 不应出现在最小 Basis 中
    f4 = ExplanationFactor(
        name="weather_clear",
        description="天气晴朗, 路面干燥",
        is_true=True, is_faithful=True, is_contrastive=False,  # 天气对"左转 vs 直行"无区分力
    )

    S_t: FrozenSet[ExplanationFactor] = frozenset({f1, f2, f3, f4})

    # -- 5. 选取 E (子集 of S_t) ---------------------------
    E: FrozenSet[ExplanationFactor] = frozenset({f1, f2, f3})

    # -- 定义集合级谓词 ------------------------------------
    # 关键: 这三个谓词是集合级的, 而不是简单的逐元素合取。
    # 场景语义:
    #   - True_t(E):        逐元素 — 每个因素在时刻 t 必须为真
    #   - Faithful_pi(E):   整体 — 策略对 a_t 的依赖需要同时考虑路线+信号+间隙
    #   - Contrastive_pi(E): 整体 — 区分"左转"与"直行/右转/停车"需要三因素联合:
    #                         仅知"导航要求左转"不够(可能信号不允许),
    #                         仅知"信号绿"不够(可能不需要左转),
    #                         仅知"间隙够"不够(可能没左转需求),
    #                         三者联合才能充分区分。
    REQUIRED_FACTORS = {"navigation_route", "traffic_signal", "oncoming_gap"}

    def true_t_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
        """True_t(E): 每个因素在时刻 t 必须为真 (逐元素合取)。"""
        return all(f.is_true for f in factors)

    def faithful_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
        """Faithful_pi(E, a_t): 策略对左转的依赖需要路线+信号+间隙三者共同呈现。
        任意真子集不足以忠实表达策略的依赖关系。"""
        names = {f.name for f in factors}
        has_all_required = REQUIRED_FACTORS.issubset(names)
        all_individually_faithful = all(f.is_faithful for f in factors)
        return has_all_required and all_individually_faithful

    def contrastive_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
        """Contrastive_pi(E, a_t, Delta_t): 区分左转与{直行,右转,停车}需要三因素联合。
        仅有部分因素无法与所有对比动作形成区分。"""
        names = {f.name for f in factors}
        has_all_required = REQUIRED_FACTORS.issubset(names)
        no_non_contrastive = all(f.is_contrastive for f in factors)
        return has_all_required and no_non_contrastive

    print(f"\nQ  = \"{Q.text}\"")
    print(f"u  = {u.user_id} (language={u.language})")
    print(f"t  = {t}")
    print(f"a_t = {a_t}")
    print(f"Delta_t = {delta_t}")
    print(f"|S_t| = {len(S_t)} 个候选因素")
    print(f"|E|   = {len(E)} 个选中因素: {{{', '.join(f.name for f in E)}}}")

    # -- 6. 通过 ExplanationSystem 构造并验证 ---------------
    system = ExplanationSystem()

    basis = system.create_basis(
        factors=E,
        candidate_space=S_t,
        question=Q,
        user_model=u,
        timestep=t,
        current_action=a_t,
        contrastive_actions=delta_t,
        true_t_fn=true_t_fn,
        faithful_fn=faithful_fn,
        contrastive_fn=contrastive_fn,
    )

    print("\n-- 定义 (1) Basis_{{u,t}}(E, Q) --")
    print(f"  E <= S_t (子集)              = {basis._subset_of_space()}")
    print(f"  True_t(E)                    = {basis._eval_true_t()}")
    print(f"  Faithful_pi(E, a_t)          = {basis._eval_faithful()}")
    print(f"  Contrastive_pi(E, a_t, D_t)  = {basis._eval_contrastive()}")
    print(f"  => Basis_{{u,t}}(E, Q)        = {basis.is_basis()}")

    print("\n-- 定义 (2) Minimal(E) --")
    print(f"  Minimal(E)                   = {basis.is_minimal()}")

    explanation = system.create_explanation(basis)

    print("\n-- 定义 (3) Explain_u(Q, t, x) --")
    report = system.validate_explanation(explanation)
    for condition, result in report.items():
        symbol = "OK" if result else "FAIL"
        print(f"  [{symbol:4s}] {condition:30s} = {result}")

    print(f"\n-- 自然语言 explanation (x) --")
    print(f"  {explanation.nl_explanation.text}")

    # -- 7. 反例验证: 含 f4 的 E' 不满足 Basis -------------
    print("\n-- 反例: 加入不具对比性的因素 f4 --")
    E_bad: FrozenSet[ExplanationFactor] = frozenset({f1, f2, f3, f4})
    basis_bad = system.create_basis(
        factors=E_bad, candidate_space=S_t, question=Q,
        user_model=u, timestep=t, current_action=a_t,
        contrastive_actions=delta_t,
        true_t_fn=true_t_fn,
        faithful_fn=faithful_fn,
        contrastive_fn=contrastive_fn,
    )
    print(f"  Basis(E + f4)                = {basis_bad.is_basis()}  (预期 False)")
    print(f"  原因: f4.is_contrastive      = {f4.is_contrastive}")

    print("\n" + "=" * 64)
    print("  验证完毕")
    print("=" * 64)
