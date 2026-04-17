"""
explanation_engine.py -- Generate symbolic-first Pac-Man explanations.
"""

from __future__ import annotations

from typing import FrozenSet, Optional

from core.explanation import ExplanationFactor, ExplanationSystem, UserModel, WhyQuestion
from core.symbolic_policy import SymbolicAnalysis, SymbolicPolicy

from .evidence_recorder import EvidenceRecord
from .question_parser import ParsedQuestion, QuestionIntent

SYMBOLIC_SUPPORT_VALIDATION_KEY = "SymbolicSupport(σ, a_t)"
SYMBOLIC_MATCH_VALIDATION_KEY = "SymbolicMatch(σ, π)"


TOPIC_LABELS = {
    QuestionIntent.WHY_THIS_ACTION: {"zh": "动作选择", "en": "action choice"},
    QuestionIntent.WHY_NOT_OTHER: {"zh": "替代动作比较", "en": "alternative action comparison"},
    QuestionIntent.MONSTER_INFLUENCE: {"zh": "怪物影响", "en": "monster influence"},
    QuestionIntent.PATH_REASON: {"zh": "路径规划", "en": "path planning"},
    QuestionIntent.SAFETY_REASON: {"zh": "安全评估", "en": "safety assessment"},
    QuestionIntent.GOAL_REASON: {"zh": "目标进度", "en": "goal progress"},
    QuestionIntent.DOT_COLLECTION: {"zh": "吃豆策略", "en": "dot collection"},
    QuestionIntent.POLICY_SUMMARY: {"zh": "整体策略", "en": "overall policy"},
    QuestionIntent.GENERAL: {"zh": "局面总结", "en": "situation summary"},
    QuestionIntent.IRRELEVANT: {"zh": "问题相关性", "en": "question relevance"},
}

ACTION_LABELS = {
    "UP": {"zh": "上", "en": "up"},
    "DOWN": {"zh": "下", "en": "down"},
    "LEFT": {"zh": "左", "en": "left"},
    "RIGHT": {"zh": "右", "en": "right"},
    "STAY": {"zh": "停留", "en": "stay"},
}


# Clean user-facing labels. These overwrite older mojibake strings left from
# previous encoding conversions.
TOPIC_LABELS = {
    QuestionIntent.WHY_THIS_ACTION: {"zh": "动作选择", "en": "action choice"},
    QuestionIntent.WHY_NOT_OTHER: {"zh": "替代动作比较", "en": "alternative action comparison"},
    QuestionIntent.MONSTER_INFLUENCE: {"zh": "怪物影响", "en": "monster influence"},
    QuestionIntent.PATH_REASON: {"zh": "路线原因", "en": "path planning"},
    QuestionIntent.SAFETY_REASON: {"zh": "安全判断", "en": "safety assessment"},
    QuestionIntent.GOAL_REASON: {"zh": "目标推进", "en": "goal progress"},
    QuestionIntent.DOT_COLLECTION: {"zh": "吃豆原因", "en": "dot collection"},
    QuestionIntent.POLICY_SUMMARY: {"zh": "整体策略", "en": "overall policy"},
    QuestionIntent.GENERAL: {"zh": "当前情况", "en": "situation summary"},
    QuestionIntent.IRRELEVANT: {"zh": "问题相关性", "en": "question relevance"},
}

ACTION_LABELS = {
    "UP": {"zh": "向上走", "en": "move up"},
    "DOWN": {"zh": "向下走", "en": "move down"},
    "LEFT": {"zh": "向左走", "en": "move left"},
    "RIGHT": {"zh": "向右走", "en": "move right"},
    "STAY": {"zh": "原地不动", "en": "stay still"},
}

DIRECTION_LABELS = {
    "same": {"zh": "脚下", "en": "right here"},
    "none": {"zh": "不明显", "en": "not clear"},
    "north": {"zh": "上方", "en": "north"},
    "south": {"zh": "下方", "en": "south"},
    "east": {"zh": "右边", "en": "east"},
    "west": {"zh": "左边", "en": "west"},
    "north-east": {"zh": "右上方", "en": "north-east"},
    "north-west": {"zh": "左上方", "en": "north-west"},
    "south-east": {"zh": "右下方", "en": "south-east"},
    "south-west": {"zh": "左下方", "en": "south-west"},
}


def _pick(zh: str, en: str, lang: str) -> str:
    return zh if lang == "zh" else en


def _action_label(action: str, lang: str) -> str:
    return ACTION_LABELS.get(action, {"zh": action, "en": action})[lang]


def _direction_label(direction: str, lang: str) -> str:
    return DIRECTION_LABELS.get(direction, {"zh": direction, "en": direction})[lang]


def _target_name(evidence: EvidenceRecord, lang: str) -> str:
    if evidence.dots_remaining > 0:
        return _pick("最近的豆子", "nearest dot", lang)
    if evidence.exit_open:
        return _pick("出口", "exit", lang)
    return _pick("当前目标", "current target", lang)


def _distance_text(distance: int, lang: str) -> str:
    if distance < 0 or distance >= 999999:
        return _pick("还不确定有多远", "at an unknown distance", lang)
    if distance == 0:
        return _pick("就在当前位置", "right here", lang)
    return _pick(f"大约 {distance} 步远", f"about {distance} step{'s' if distance != 1 else ''} away", lang)


def _risk_level_text(risk: float, lang: str) -> str:
    if risk >= 0.65:
        return _pick("很危险", "very risky", lang)
    if risk >= 0.35:
        return _pick("有明显风险", "noticeably risky", lang)
    if risk > 0.05:
        return _pick("有一点风险", "a little risky", lang)
    return _pick("基本安全", "mostly safe", lang)


def _safer_alternative(evidence: EvidenceRecord) -> tuple[str, float] | None:
    risks = dict(evidence.collision_risks)
    chosen_risk = risks.get(evidence.chosen_action, 0.0)
    alternatives = [
        (action, risk)
        for action, risk in risks.items()
        if action != evidence.chosen_action and action in evidence.available_actions
    ]
    if not alternatives:
        return None
    best_action, best_risk = min(alternatives, key=lambda item: item[1])
    if best_risk + 0.05 < chosen_risk:
        return best_action, best_risk
    return None


def _mentioned_action_mismatch(evidence: EvidenceRecord, question: ParsedQuestion) -> bool:
    if not question.mentioned_action:
        return False
    if question.intent == QuestionIntent.WHY_NOT_OTHER:
        return question.mentioned_action == evidence.chosen_action
    return question.mentioned_action != evidence.chosen_action


def _render_action_mismatch_text(evidence: EvidenceRecord, question: ParsedQuestion) -> str:
    lang = question.language
    asked_action = question.mentioned_action or ""
    chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
    asked_risk = dict(evidence.collision_risks).get(asked_action)
    target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
    target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
    target = _target_name(evidence, lang)
    direction = _direction_label(target_direction, lang)
    distance = _distance_text(target_distance, lang)
    monster_distance = _distance_text(evidence.nearest_monster_distance, lang)
    monster_direction = _direction_label(evidence.nearest_monster_direction, lang)

    if question.intent == QuestionIntent.WHY_NOT_OTHER:
        return _pick(
            f"你问的是为什么不{_action_label(asked_action, lang)}，但这一帧 Pac-Man 实际上已经{_action_label(evidence.chosen_action, lang)}了。最近的目标在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。所以这个问题和这一帧的真实动作有点反过来了。",
            f"You asked why Pac-Man did not {_action_label(asked_action, lang)}, but in this frame it actually did {_action_label(evidence.chosen_action, lang)}. The nearest target is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}. So the question is reversed relative to what happened in this frame.",
            lang,
        )

    opening = _pick(
        f"你问的是为什么{_action_label(asked_action, lang)}，但这一帧 Pac-Man 实际选择的是{_action_label(evidence.chosen_action, lang)}。",
        f"You asked why Pac-Man chose to {_action_label(asked_action, lang)}, but in this frame it actually chose to {_action_label(evidence.chosen_action, lang)}.",
        lang,
    )
    situation = _pick(
        f"所以我先解释真实发生的动作：{target}在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。",
        f"So I will explain the action that actually happened: the {target} is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}.",
        lang,
    )
    chosen = _pick(
        f"实际选择的风险约 {chosen_risk:.0%}。",
        f"The actual chosen move has about {chosen_risk:.0%} risk.",
        lang,
    )
    if asked_risk is None:
        unavailable = _pick(
            f"你问的{_action_label(asked_action, lang)}在这一步可能不是可执行方向，或者系统没有记录到它的风险。",
            f"The asked-about move, {_action_label(asked_action, lang)}, may not be available here, or its risk was not recorded.",
            lang,
        )
        return " ".join([opening, situation, chosen, unavailable])
    asked = _pick(
        f"你问的方向风险约 {asked_risk:.0%}。",
        f"The move you asked about has about {asked_risk:.0%} risk.",
        lang,
    )
    return " ".join([opening, situation, chosen, asked])


class ExplanationEngine:
    """Create symbolic-first explanations with a raw-evidence fallback."""

    def __init__(self, symbolic_policy: Optional[SymbolicPolicy] = None):
        self.symbolic_policy = symbolic_policy

    def generate_explanation(self, evidence: EvidenceRecord, question: ParsedQuestion) -> dict:
        if question.intent == QuestionIntent.IRRELEVANT:
            return self._generate_irrelevant_explanation(evidence, question)

        if self.symbolic_policy is None:
            return self._generate_legacy_explanation(evidence, question, symbolic_analysis=None)

        analysis = self.symbolic_policy.analyze_state(
            evidence.state_snapshot,
            evidence.chosen_action,
            question.language,
            requested_alternative=question.mentioned_action,
        )
        if not analysis.symbolic_match:
            return self._generate_legacy_explanation(evidence, question, symbolic_analysis=analysis)
        return self._generate_symbolic_explanation(evidence, question, analysis)

    def _generate_irrelevant_explanation(self, evidence: EvidenceRecord, question: ParsedQuestion) -> dict:
        lang = question.language
        rendered_text = _pick(
            "这个问题和当前 Pac-Man 决策没有直接关系。我现在能解释的是：为什么这样走、为什么不走某个方向、是否安全、怪物有没有影响、或者整体策略是什么。",
            "This question is not directly about the current Pac-Man decision. I can explain things like why it moved this way, why it did not choose another direction, whether the move is safe, how monsters affected it, or what the overall policy is.",
            lang,
        )
        factor_payloads = {
            "question_relevance": self._factor_payload(
                "question_relevance",
                rendered_text,
                ["question_text"],
            )
        }
        return self._package_explanation(
            evidence=evidence,
            question=question,
            factor_payloads=factor_payloads,
            selected_names=["question_relevance"],
            rendered_text=rendered_text,
            symbolic_analysis=None,
            policy_summary={"bullets": [], "python_snippet": ""},
            fallback_used=True,
            lang=lang,
        )

    def _generate_symbolic_explanation(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        analysis: SymbolicAnalysis,
    ) -> dict:
        lang = question.language
        policy_summary = self.symbolic_policy.get_policy_summary(lang) if self.symbolic_policy else {"bullets": [], "python_snippet": ""}
        rendered_text = self._render_symbolic_text(evidence, question, analysis, policy_summary)
        factor_payloads = self._build_symbolic_factor_payloads(evidence, question, analysis, policy_summary)
        selected_names = self._select_symbolic_factor_names(question)
        return self._package_explanation(
            evidence=evidence,
            question=question,
            factor_payloads=factor_payloads,
            selected_names=selected_names,
            rendered_text=rendered_text,
            symbolic_analysis=analysis,
            policy_summary=policy_summary,
            fallback_used=False,
            lang=lang,
        )

    def _generate_legacy_explanation(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        symbolic_analysis: SymbolicAnalysis | None,
    ) -> dict:
        lang = question.language
        policy_summary = self.symbolic_policy.get_policy_summary(lang) if self.symbolic_policy else {"bullets": [], "python_snippet": ""}
        rendered_text = self._render_legacy_text(evidence, question, symbolic_analysis)
        factor_payloads = self._build_legacy_factor_payloads(evidence, question, symbolic_analysis)
        selected_names = self._select_legacy_factor_names(question)
        return self._package_explanation(
            evidence=evidence,
            question=question,
            factor_payloads=factor_payloads,
            selected_names=selected_names,
            rendered_text=rendered_text,
            symbolic_analysis=symbolic_analysis,
            policy_summary=policy_summary,
            fallback_used=True,
            lang=lang,
        )

    def _package_explanation(
        self,
        *,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        factor_payloads: dict[str, dict],
        selected_names: list[str],
        rendered_text: str,
        symbolic_analysis: SymbolicAnalysis | None,
        policy_summary: dict[str, object],
        fallback_used: bool,
        lang: str,
    ) -> dict:
        candidate_space = frozenset(payload["factor"] for payload in factor_payloads.values())
        selected_factors = frozenset(
            factor_payloads[name]["factor"] for name in selected_names if name in factor_payloads
        )
        groups = self._required_groups(question)

        def render_fn(_: FrozenSet[ExplanationFactor], __: WhyQuestion) -> str:
            return rendered_text

        user_model = UserModel(
            user_id="player",
            language=lang,
            detail_level="medium",
            render_fn=render_fn,
        )
        why_question = WhyQuestion(
            text=question.original_text,
            topic=TOPIC_LABELS[question.intent][lang],
            context=f"step={evidence.step}, position={evidence.player_pos}",
        )

        def true_t_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
            return all(factor.is_true for factor in factors)

        def faithful_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
            return all(factor.is_faithful for factor in factors) and self._satisfies_groups(factors, groups)

        def contrastive_fn(factors: FrozenSet[ExplanationFactor]) -> bool:
            return all(factor.is_contrastive for factor in factors) and self._satisfies_groups(factors, groups)

        minimized = set(selected_factors)
        for factor in list(selected_factors):
            trial = frozenset(minimized - {factor})
            if not trial:
                continue
            if true_t_fn(trial) and faithful_fn(trial) and contrastive_fn(trial):
                minimized = set(trial)
        evidence_used = frozenset(minimized)

        basis = ExplanationSystem.create_basis(
            factors=evidence_used,
            candidate_space=candidate_space,
            question=why_question,
            user_model=user_model,
            timestep=evidence.step,
            current_action=evidence.chosen_action,
            contrastive_actions=set(evidence.available_actions) - {evidence.chosen_action},
            true_t_fn=true_t_fn,
            faithful_fn=faithful_fn,
            contrastive_fn=contrastive_fn,
        )
        explanation = ExplanationSystem.create_explanation(basis)
        validation = dict(ExplanationSystem.validate_explanation(explanation))
        validation[SYMBOLIC_SUPPORT_VALIDATION_KEY] = bool(symbolic_analysis.symbolic_support) if symbolic_analysis else False
        validation[SYMBOLIC_MATCH_VALIDATION_KEY] = bool(symbolic_analysis.symbolic_match) if symbolic_analysis else False

        return {
            "all_evidence": {
                "label": _pick("全部证据 (S_t)", "All Evidence (S_t)", lang),
                "factors": [
                    {
                        "name": payload["factor"].name,
                        "description": payload["factor"].description,
                        "sources": payload["sources"],
                        "is_true": payload["factor"].is_true,
                        "is_faithful": payload["factor"].is_faithful,
                        "is_contrastive": payload["factor"].is_contrastive,
                    }
                    for payload in sorted(factor_payloads.values(), key=lambda item: item["factor"].name)
                ],
            },
            "evidence_used": {
                "label": _pick("实际使用证据 (E)", "Evidence Used (E)", lang),
                "factors": [
                    {
                        "name": factor_payloads[factor.name]["factor"].name,
                        "description": factor_payloads[factor.name]["factor"].description,
                        "sources": factor_payloads[factor.name]["sources"],
                    }
                    for factor in sorted(evidence_used, key=lambda item: item.name)
                ],
            },
            "explanation_text": {
                "label": _pick("自然语言解释 (x)", "Natural-Language Explanation (x)", lang),
                "text": explanation.nl_explanation.text,
            },
            "symbolic_rule": {
                "text": symbolic_analysis.chosen_rule if symbolic_analysis else "",
                "python": symbolic_analysis.chosen_rule_python if symbolic_analysis else "",
                "fallback_used": fallback_used,
            },
            "symbolic_trace": {
                "chosen_action": symbolic_analysis.chosen_action if symbolic_analysis else evidence.chosen_action,
                "predicted_action": symbolic_analysis.predicted_action if symbolic_analysis else None,
                "alternative_action": symbolic_analysis.alternative_action if symbolic_analysis else question.mentioned_action,
                "trace": symbolic_analysis.chosen_trace if symbolic_analysis else [],
                "approximate_trace": symbolic_analysis.predicted_trace if symbolic_analysis and not symbolic_analysis.symbolic_match else [],
            },
            "policy_summary": policy_summary,
            "symbolic_match": bool(symbolic_analysis.symbolic_match) if symbolic_analysis else False,
            "distillation_metrics": self.symbolic_policy.metrics if self.symbolic_policy else {},
            "validation": validation,
            "language": lang,
        }

    def _build_symbolic_factor_payloads(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        analysis: SymbolicAnalysis,
        policy_summary: dict[str, object],
    ) -> dict[str, dict]:
        lang = question.language
        objective_text = _pick(
            f"当前还剩 {evidence.dots_remaining} 颗豆子，出口{'已打开' if evidence.exit_open else '未打开'}，所以策略先围绕当前目标行动。",
            f"There are {evidence.dots_remaining} dots left and the exit is {'open' if evidence.exit_open else 'locked'}, so the policy still acts around the current objective.",
            lang,
        )
        summary_focus = _pick(
            "这次回答直接引用蒸馏后的符号策略摘要。",
            "This answer directly references the distilled symbolic policy summary.",
            lang,
        )
        support_text = _pick(
            f"符号代理也支持当前动作 {_action_label(evidence.chosen_action, lang)}。",
            f"The symbolic surrogate also supports the chosen action {_action_label(evidence.chosen_action, lang)}.",
            lang,
        )
        chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
        target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
        target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
        target = _target_name(evidence, lang)
        direction = _direction_label(target_direction, lang)
        distance = _distance_text(target_distance, lang)
        monster_distance = _distance_text(evidence.nearest_monster_distance, lang)
        monster_direction = _direction_label(evidence.nearest_monster_direction, lang)
        risk_level = _risk_level_text(chosen_risk, lang)
        objective_text = _pick(
            f"当前目标是{target}，它在{direction}，{distance}。",
            f"The current target is {target}, {distance} toward {direction}.",
            lang,
        )
        summary_focus = _pick(
            "这次回答会先解释当前这一步，再把它和整体策略联系起来。",
            "This answer explains the current move first, then connects it to the overall policy.",
            lang,
        )
        support_text = _pick(
            f"规则检查也支持{_action_label(evidence.chosen_action, lang)}这个选择。",
            f"The rule check also supports choosing to {_action_label(evidence.chosen_action, lang)}.",
            lang,
        )
        risk_comparison_text = _pick(
            f"这一步{risk_level}，撞到怪物的可能性约 {chosen_risk:.0%}。",
            f"This move looks {risk_level}, with about a {chosen_risk:.0%} chance of hitting a monster.",
            lang,
        )
        target_comparison_text = _pick(
            f"这一步主要和{target}有关：它在{direction}，{distance}。",
            f"This move is mainly about the {target}, which is {distance} toward {direction}.",
            lang,
        )
        monster_comparison_text = _pick(
            f"最近的怪物在{monster_direction}，离我{monster_distance}。",
            f"The nearest monster is {monster_distance} toward {monster_direction}.",
            lang,
        )
        decision_text = _pick(
            f"综合目标和风险后，当前选择是{_action_label(evidence.chosen_action, lang)}。",
            f"Balancing target progress and risk, the chosen move is to {_action_label(evidence.chosen_action, lang)}.",
            lang,
        )

        payloads = {
            "objective_context": self._factor_payload("objective_context", objective_text, ["dots_remaining", "exit_open", "nearest_dot_distance", "exit_distance"]),
            "risk_comparison": self._factor_payload("risk_comparison", risk_comparison_text, ["collision_risks"]),
            "target_comparison": self._factor_payload("target_comparison", target_comparison_text, ["nearest_dot_distance", "exit_distance"]),
            "monster_comparison": self._factor_payload("monster_comparison", monster_comparison_text, ["monster_positions"]),
            "decision_clause": self._factor_payload("decision_clause", decision_text, list(analysis.comparison["sources"])),
            "symbolic_support": self._factor_payload(
                "symbolic_support",
                support_text,
                ["symbolic_rule", "collision_risks", "nearest_dot_distance", "monster_positions"],
                faithful=analysis.symbolic_support,
                contrastive=analysis.symbolic_support,
            ),
            "policy_summary_focus": self._factor_payload("policy_summary_focus", summary_focus, ["policy_summary"]),
            "policy_summary_bullet": self._factor_payload(
                "policy_summary_bullet",
                " | ".join(str(item) for item in policy_summary.get("bullets", [])[:2]) or summary_focus,
                ["policy_summary"],
                contrastive=False,
            ),
        }
        return payloads

    def _build_legacy_factor_payloads(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        symbolic_analysis: SymbolicAnalysis | None,
    ) -> dict[str, dict]:
        lang = question.language
        chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
        fallback_text = _pick(
            "由于符号代理和神经网络当前不一致，这里回退到原始证据解释。",
            "Because the symbolic surrogate disagrees with the neural policy here, this answer falls back to raw evidence.",
            lang,
        )
        objective_text = _pick(
            f"当前还剩 {evidence.dots_remaining} 颗豆子，最近目标方向是 {evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction}。",
            f"There are {evidence.dots_remaining} dots left, and the active target is toward {evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction}.",
            lang,
        )
        monster_text = _pick(
            f"最近怪物 #{evidence.nearest_monster_id} 距离 {evidence.nearest_monster_distance} 步。",
            f"Nearest monster #{evidence.nearest_monster_id} is {evidence.nearest_monster_distance} steps away.",
            lang,
        )
        risk_text = _pick(
            f"当前动作 {_action_label(evidence.chosen_action, lang)} 的即时风险约为 {chosen_risk:.0%}。",
            f"The chosen action {_action_label(evidence.chosen_action, lang)} has about {chosen_risk:.0%} immediate risk.",
            lang,
        )
        support_text = _pick(
            f"符号代理更偏好 {_action_label(symbolic_analysis.predicted_action, lang)}，所以这里只把符号规则当作近似参考。",
            f"The symbolic surrogate prefers {_action_label(symbolic_analysis.predicted_action, lang)}, so its rule is treated as an approximation here.",
            lang,
        ) if symbolic_analysis else _pick("当前没有可用的符号代理。", "No symbolic surrogate is available for this step.", lang)
        target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
        target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
        target = _target_name(evidence, lang)
        direction = _direction_label(target_direction, lang)
        distance = _distance_text(target_distance, lang)
        risk_level = _risk_level_text(chosen_risk, lang)
        monster_distance = _distance_text(evidence.nearest_monster_distance, lang)
        fallback_text = _pick(
            "这次回答主要根据画面里的目标、怪物和风险来解释。",
            "This answer uses the visible target, monster position, and immediate risk.",
            lang,
        )
        objective_text = _pick(
            f"当前最重要的是靠近{target}：它在{direction}，{distance}。",
            f"The immediate goal is to get closer to {target}: it is {distance} toward {direction}.",
            lang,
        )
        monster_text = _pick(
            f"最近的怪物在{_direction_label(evidence.nearest_monster_direction, lang)}，离 Pac-Man {monster_distance}。",
            f"The nearest monster is {monster_distance} toward {_direction_label(evidence.nearest_monster_direction, lang)}.",
            lang,
        )
        risk_text = _pick(
            f"选择{_action_label(evidence.chosen_action, lang)}的风险判断是：{risk_level}，估计撞到怪物的可能性约 {chosen_risk:.0%}。",
            f"Choosing to {_action_label(evidence.chosen_action, lang)} looks {risk_level}; the estimated chance of hitting a monster is about {chosen_risk:.0%}.",
            lang,
        )
        if symbolic_analysis:
            support_text = _pick(
                "补充：规则模型对这一步没有足够把握，所以用户正文不把规则当成主要理由。",
                "Note: the rule model is not confident enough here, so the user-facing answer does not treat it as the main reason.",
                lang,
            )
        else:
            support_text = _pick(
                "补充：当前没有加载规则模型，所以只显示基于画面证据的解释。",
                "Note: no rule model is loaded, so this is an evidence-only explanation.",
                lang,
            )

        return {
            "fallback_notice": self._factor_payload("fallback_notice", fallback_text, ["symbolic_rule"], contrastive=False),
            "objective_context": self._factor_payload("objective_context", objective_text, ["dots_remaining", "nearest_dot_distance", "exit_distance"]),
            "monster_context": self._factor_payload("monster_context", monster_text, ["monster_positions"]),
            "risk_context": self._factor_payload("risk_context", risk_text, ["collision_risks"]),
            "symbolic_gap": self._factor_payload("symbolic_gap", support_text, ["symbolic_rule"], contrastive=False),
        }

    @staticmethod
    def _factor_payload(
        name: str,
        description: str,
        sources: list[str],
        *,
        faithful: bool = True,
        contrastive: bool = True,
    ) -> dict:
        return {
            "factor": ExplanationFactor(
                name=name,
                description=description,
                is_true=True,
                is_faithful=faithful,
                is_contrastive=contrastive,
            ),
            "sources": sources,
        }

    @staticmethod
    def _select_symbolic_factor_names(question: ParsedQuestion) -> list[str]:
        if question.intent == QuestionIntent.POLICY_SUMMARY:
            return ["policy_summary_focus", "policy_summary_bullet", "decision_clause"]
        if question.intent == QuestionIntent.SAFETY_REASON:
            return ["objective_context", "risk_comparison", "monster_comparison", "decision_clause", "symbolic_support"]
        return ["objective_context", "risk_comparison", "target_comparison", "monster_comparison", "decision_clause", "symbolic_support"]

    @staticmethod
    def _select_legacy_factor_names(question: ParsedQuestion) -> list[str]:
        del question
        # Fallback notices are provenance metadata, not causal/contrastive
        # evidence. Evidence-only explanations should stand on environment
        # factors instead of requiring a symbolic-policy artifact.
        return ["objective_context", "monster_context", "risk_context"]

    @staticmethod
    def _required_groups(question: ParsedQuestion) -> list[set[str]]:
        if question.intent == QuestionIntent.IRRELEVANT:
            return [{"question_relevance"}]
        if question.intent == QuestionIntent.POLICY_SUMMARY:
            return [
                {"policy_summary_focus", "objective_context"},
                {"policy_summary_bullet", "decision_clause", "risk_context", "monster_context"},
            ]
        if question.intent == QuestionIntent.SAFETY_REASON:
            return [{"objective_context"}, {"risk_comparison", "risk_context"}, {"monster_comparison", "monster_context"}]
        return [
            {"objective_context"},
            {"target_comparison", "risk_comparison", "risk_context"},
            {"decision_clause", "monster_context"},
        ]

    @staticmethod
    def _satisfies_groups(factors: FrozenSet[ExplanationFactor], groups: list[set[str]]) -> bool:
        names = {factor.name for factor in factors}
        return all(bool(names & group) for group in groups)

    def _render_symbolic_text(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        analysis: SymbolicAnalysis,
        policy_summary: dict[str, object],
    ) -> str:
        lang = question.language
        comparison = analysis.comparison
        chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
        target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
        target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
        target = _target_name(evidence, lang)
        direction = _direction_label(target_direction, lang)
        distance = _distance_text(target_distance, lang)
        monster_distance = _distance_text(evidence.nearest_monster_distance, lang)
        monster_direction = _direction_label(evidence.nearest_monster_direction, lang)
        risk_level = _risk_level_text(chosen_risk, lang)

        if _mentioned_action_mismatch(evidence, question):
            return _render_action_mismatch_text(evidence, question)

        if question.intent == QuestionIntent.POLICY_SUMMARY:
            return _pick(
                "整体策略可以理解成三句话：先靠近最近的豆子；豆子吃完后去出口；如果怪物靠近，就优先避开危险。",
                "The overall strategy is simple: move toward the nearest dot first, head for the exit once the dots are gone, and avoid danger when a monster gets close.",
                lang,
            )

        if question.intent == QuestionIntent.WHY_NOT_OTHER and question.mentioned_action:
            alt_risk = dict(evidence.collision_risks).get(question.mentioned_action)
            risk_part = _pick(
                "这个方向这一步可能不可执行，或者没有风险记录。",
                "That action may be unavailable here, or its risk was not recorded.",
                lang,
            )
            if alt_risk is not None:
                risk_part = _pick(
                    f"实际选择的风险约 {chosen_risk:.0%}，你问的方向风险约 {alt_risk:.0%}。",
                    f"The chosen move has about {chosen_risk:.0%} risk, while the move you asked about has about {alt_risk:.0%} risk.",
                    lang,
                )
            return _pick(
                f"我没有{_action_label(question.mentioned_action, lang)}，而是{_action_label(evidence.chosen_action, lang)}。当前目标在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。{risk_part} 规则检查也支持当前选择。",
                f"I did not {_action_label(question.mentioned_action, lang)}; I chose to {_action_label(evidence.chosen_action, lang)} instead. The current target is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}. {risk_part} The rule check also supports the chosen move.",
                lang,
            )

        if analysis.alternative_action:
            return _pick(
                f"我选择{_action_label(evidence.chosen_action, lang)}，而不是{_action_label(analysis.alternative_action, lang)}。现在{target}在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。规则判断也支持当前选择：它在推进目标和控制风险之间更合适。",
                f"I chose to {_action_label(evidence.chosen_action, lang)} instead of {_action_label(analysis.alternative_action, lang)}. The {target} is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}. The rule check also supports the chosen move as the better balance between progress and risk.",
                lang,
            )

        return _pick(
            f"我选择{_action_label(evidence.chosen_action, lang)}。现在{target}在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。这一步{risk_level}，规则判断也支持这个选择。",
            f"I chose to {_action_label(evidence.chosen_action, lang)}. The {target} is {distance} toward {direction}; the nearest monster is {monster_distance} toward {monster_direction}. This move looks {risk_level}, and the rule check supports it.",
            lang,
        )

        if question.intent == QuestionIntent.POLICY_SUMMARY:
            bullets = policy_summary.get("bullets", [])
            summary_sentence = _pick(
                "整体上，这个策略会优先考虑当前目标推进，再在风险接近时用安全性和收益做 tie-break。",
                "Overall, this policy prioritizes progress on the active target, then uses safety and reward as tie-breakers when risks are close.",
                lang,
            )
            bullet_sentence = " ".join(f"- {item}" for item in bullets[:3])
            return f"{summary_sentence} {bullet_sentence}".strip()

        opening = _pick(
            f"这一步选择{_action_label(evidence.chosen_action, lang)}，主要依据是符号代理对当前动作和替代动作的比较。",
            f"This step chooses {_action_label(evidence.chosen_action, lang)} mainly because the symbolic surrogate compares it favorably against the alternatives.",
            lang,
        )
        sentences = [opening]
        for key in ("risk_clause", "target_clause", "monster_clause", "decision_clause"):
            clause = str(comparison.get(key, "")).strip()
            if clause:
                sentences.append(clause)
        return " ".join(sentences)

    def _render_legacy_text(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        symbolic_analysis: SymbolicAnalysis | None,
    ) -> str:
        lang = question.language
        chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
        target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
        target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
        target = _target_name(evidence, lang)
        direction = _direction_label(target_direction, lang)
        distance = _distance_text(target_distance, lang)
        monster_distance = _distance_text(evidence.nearest_monster_distance, lang)
        monster_direction = _direction_label(evidence.nearest_monster_direction, lang)
        risk_level = _risk_level_text(chosen_risk, lang)
        safer = _safer_alternative(evidence)

        if _mentioned_action_mismatch(evidence, question):
            return _render_action_mismatch_text(evidence, question)

        if question.intent == QuestionIntent.POLICY_SUMMARY:
            return _pick(
                "整体上，我会先找最近的豆子；豆子吃完后再去出口。只要怪物离得近，我就会把安全放在前面，宁愿绕一下也不要直接撞上去。",
                "Overall, I first try to reach the nearest dot; once the dots are gone, I head for the exit. If a monster is close, safety comes first, even if that means taking a detour.",
                lang,
            )

        if question.intent == QuestionIntent.SAFETY_REASON:
            base = _pick(
                f"这一步{risk_level}：最近的怪物在{monster_direction}，离我{monster_distance}；选择{_action_label(evidence.chosen_action, lang)}时，撞到怪物的可能性约 {chosen_risk:.0%}。",
                f"This move looks {risk_level}: the nearest monster is {monster_distance} toward {monster_direction}, and choosing to {_action_label(evidence.chosen_action, lang)} has about a {chosen_risk:.0%} chance of hitting it.",
                lang,
            )
            if safer:
                alt_action, alt_risk = safer
                return base + " " + _pick(
                    f"如果只想更安全，{_action_label(alt_action, lang)}看起来更稳，风险约 {alt_risk:.0%}。",
                    f"If we only care about safety, {_action_label(alt_action, lang)} looks safer at about {alt_risk:.0%} risk.",
                    lang,
                )
            return base

        if question.intent == QuestionIntent.WHY_NOT_OTHER and question.mentioned_action:
            alt_risk = dict(evidence.collision_risks).get(question.mentioned_action)
            if alt_risk is None:
                return _pick(
                    f"我没有{_action_label(question.mentioned_action, lang)}，而是{_action_label(evidence.chosen_action, lang)}。你问的方向这一步可能被墙挡住、不可执行，或者系统没有记录到它的风险。当前目标在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。",
                    f"I did not {_action_label(question.mentioned_action, lang)}; I chose to {_action_label(evidence.chosen_action, lang)} instead. The action you asked about may be blocked, unavailable, or missing a risk estimate in this frame. The current target is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}.",
                    lang,
                )
            return _pick(
                f"我没有{_action_label(question.mentioned_action, lang)}，而是{_action_label(evidence.chosen_action, lang)}。当前目标在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。这两个方向的风险大约是：实际选择 {chosen_risk:.0%}，你问的方向 {alt_risk:.0%}。",
                f"I did not {_action_label(question.mentioned_action, lang)}; I chose to {_action_label(evidence.chosen_action, lang)} instead. The current target is {distance} toward {direction}, and the nearest monster is {monster_distance} toward {monster_direction}. The risks are about {chosen_risk:.0%} for the chosen move and {alt_risk:.0%} for the move you asked about.",
                lang,
            )

        if question.intent == QuestionIntent.WHY_NOT_OTHER and question.mentioned_action:
            alt_risk = dict(evidence.collision_risks).get(question.mentioned_action)
            if alt_risk is not None:
                return _pick(
                    f"我没有选{_action_label(question.mentioned_action, lang)}，而是选了{_action_label(evidence.chosen_action, lang)}。现在{target}在{direction}，{distance}；同时怪物离我{monster_distance}。这两个方向的风险大约是：当前选择 {chosen_risk:.0%}，你问的方向 {alt_risk:.0%}。",
                    f"I chose to {_action_label(evidence.chosen_action, lang)} instead of {_action_label(question.mentioned_action, lang)}. The {target} is {distance} toward {direction}, and the nearest monster is {monster_distance}. The risks are about {chosen_risk:.0%} for the chosen move and {alt_risk:.0%} for the alternative.",
                    lang,
                )

        main = _pick(
            f"我选择{_action_label(evidence.chosen_action, lang)}。现在{target}在{direction}，{distance}；最近的怪物在{monster_direction}，离我{monster_distance}。这一步{risk_level}，撞到怪物的可能性约 {chosen_risk:.0%}。",
            f"I chose to {_action_label(evidence.chosen_action, lang)}. The {target} is {distance} toward {direction}; the nearest monster is {monster_distance} toward {monster_direction}. This move looks {risk_level}, with about a {chosen_risk:.0%} chance of hitting a monster.",
            lang,
        )
        if safer:
            alt_action, alt_risk = safer
            return main + " " + _pick(
                f"所以这不是最保守的一步；如果只看安全，{_action_label(alt_action, lang)}可能更稳，风险约 {alt_risk:.0%}。",
                f"So this is not the most conservative move; if we only care about safety, {_action_label(alt_action, lang)} may be safer at about {alt_risk:.0%} risk.",
                lang,
            )
        return main

        chosen_risk = dict(evidence.collision_risks).get(evidence.chosen_action, 0.0)
        target_direction = evidence.nearest_dot_direction if evidence.dots_remaining > 0 else evidence.exit_direction
        target_distance = evidence.nearest_dot_distance if evidence.dots_remaining > 0 else evidence.exit_distance
        opening = _pick(
            f"当前回退到原始证据解释，因为符号代理没有和神经网络在这一帧达成一致。",
            f"This answer falls back to raw evidence because the symbolic surrogate does not match the neural policy on this step.",
            lang,
        )
        main = _pick(
            f"Pac-Man 当前选择{_action_label(evidence.chosen_action, lang)}；目标方向是 {target_direction}，距离约 {target_distance} 步；最近怪物距离 {evidence.nearest_monster_distance} 步；即时风险约为 {chosen_risk:.0%}。",
            f"Pac-Man currently chooses {_action_label(evidence.chosen_action, lang)}; the active target is {target_distance} steps away toward {target_direction}; the nearest monster is {evidence.nearest_monster_distance} steps away; the immediate risk is about {chosen_risk:.0%}.",
            lang,
        )
        if symbolic_analysis is None:
            return f"{opening} {main}"
        note = _pick(
            f"符号代理更像是在推荐{_action_label(symbolic_analysis.predicted_action, lang)}，所以这里只把它的规则作为近似参考。",
            f"The symbolic surrogate would rather recommend {_action_label(symbolic_analysis.predicted_action, lang)}, so its rule is shown only as an approximation.",
            lang,
        )
        return f"{opening} {main} {note}"
