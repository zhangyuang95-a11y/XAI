"""
explanation_engine.py -- Translate structured evidence into bilingual Pac-Man explanations.
"""

from __future__ import annotations

from typing import FrozenSet

from explanation import ExplanationFactor, ExplanationSystem, UserModel, WhyQuestion
from evidence_recorder import EvidenceRecord
from environment import get_relative_direction, manhattan_distance
from question_parser import ParsedQuestion, QuestionIntent


_TOPIC_LABELS = {
    QuestionIntent.WHY_THIS_ACTION: {"zh": "动作选择", "en": "action choice"},
    QuestionIntent.WHY_NOT_OTHER: {"zh": "替代动作比较", "en": "alternative action comparison"},
    QuestionIntent.MONSTER_INFLUENCE: {"zh": "怪物影响", "en": "monster influence"},
    QuestionIntent.PATH_REASON: {"zh": "路径规划", "en": "path planning"},
    QuestionIntent.SAFETY_REASON: {"zh": "安全评估", "en": "safety assessment"},
    QuestionIntent.GOAL_REASON: {"zh": "目标进度", "en": "goal progress"},
    QuestionIntent.DOT_COLLECTION: {"zh": "吃豆策略", "en": "dot collection"},
    QuestionIntent.GENERAL: {"zh": "局面总结", "en": "situation summary"},
    QuestionIntent.IRRELEVANT: {"zh": "问题相关性", "en": "question relevance"},
}

_ACTION_LABELS = {
    "UP": {"zh": "向上", "en": "move up"},
    "DOWN": {"zh": "向下", "en": "move down"},
    "LEFT": {"zh": "向左", "en": "move left"},
    "RIGHT": {"zh": "向右", "en": "move right"},
    "STAY": {"zh": "原地等待", "en": "stay put"},
}

_ACTION_SHORT = {
    "UP": {"zh": "上", "en": "UP"},
    "DOWN": {"zh": "下", "en": "DOWN"},
    "LEFT": {"zh": "左", "en": "LEFT"},
    "RIGHT": {"zh": "右", "en": "RIGHT"},
    "STAY": {"zh": "停", "en": "STAY"},
}

_DIRECTION_LABELS = {
    "north": {"zh": "北侧", "en": "north"},
    "south": {"zh": "南侧", "en": "south"},
    "west": {"zh": "西侧", "en": "west"},
    "east": {"zh": "东侧", "en": "east"},
    "north-east": {"zh": "东北方向", "en": "north-east"},
    "north-west": {"zh": "西北方向", "en": "north-west"},
    "south-east": {"zh": "东南方向", "en": "south-east"},
    "south-west": {"zh": "西南方向", "en": "south-west"},
    "same": {"zh": "当前位置", "en": "the current tile"},
    "none": {"zh": "未知方向", "en": "an unknown direction"},
}

_DIRECTION_TO_ACTIONS = {
    "north": {"UP"},
    "south": {"DOWN"},
    "west": {"LEFT"},
    "east": {"RIGHT"},
}


def _pick(zh: str, en: str, lang: str) -> str:
    return zh if lang == "zh" else en


def _action_label(action: str, lang: str) -> str:
    return _ACTION_LABELS.get(action, {"zh": action, "en": action})[lang]


def _action_short(action: str, lang: str) -> str:
    return _ACTION_SHORT.get(action, {"zh": action, "en": action})[lang]


def _direction_label(direction: str, lang: str) -> str:
    return _DIRECTION_LABELS.get(direction, {"zh": direction, "en": direction})[lang]


def _risk_band(risk: float, lang: str) -> str:
    if risk >= 0.75:
        return _pick("极高风险", "very high risk", lang)
    if risk >= 0.4:
        return _pick("高风险", "high risk", lang)
    if risk >= 0.15:
        return _pick("中等风险", "moderate risk", lang)
    return _pick("低风险", "low risk", lang)


def _pressure_band(distance: int, lang: str) -> str:
    if distance <= 1:
        return _pick("贴身威胁", "an immediate threat", lang)
    if distance <= 3:
        return _pick("高压区", "heavy pressure", lang)
    if distance <= 5:
        return _pick("中压区", "moderate pressure", lang)
    return _pick("低压区", "light pressure", lang)


def _join_sentences(sentences: list[str], lang: str) -> str:
    parts = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    return "".join(parts) if lang == "zh" else " ".join(parts)


def _action_aligns(action: str, direction: str) -> bool:
    if direction == "same":
        return action == "STAY"
    desired: set[str] = set()
    for token in direction.split("-"):
        desired.update(_DIRECTION_TO_ACTIONS.get(token, set()))
    return action in desired if desired else False


class ExplanationEngine:
    """Create 3-layer explanations from recorded game evidence."""

    def build_factors(self, evidence: EvidenceRecord, question: ParsedQuestion) -> list[ExplanationFactor]:
        lang = question.language
        chosen = evidence.chosen_action
        risk_map = dict(evidence.collision_risks)
        chosen_risk = risk_map.get(chosen, 0.0)
        dots_remaining = evidence.dots_remaining
        dots_collected = evidence.dots_collected
        total_dots = evidence.total_dots

        target_dir = evidence.nearest_dot_direction if dots_remaining > 0 else evidence.exit_direction
        target_dist = evidence.nearest_dot_distance if dots_remaining > 0 else evidence.exit_distance
        target_label = _pick("最近的豆子", "the nearest dot", lang) if dots_remaining > 0 else _pick("出口", "the exit", lang)
        aligned = _action_aligns(chosen, target_dir)

        factors: dict[str, ExplanationFactor] = {}

        def add_factor(name: str, description: str, *, faithful: bool = True, contrastive: bool = True) -> None:
            factors[name] = ExplanationFactor(
                name=name,
                description=description,
                is_true=True,
                is_faithful=faithful,
                is_contrastive=contrastive,
            )

        if dots_remaining > 0:
            add_factor(
                "phase_objective",
                _pick(
                    f"出口还未开启，因为还有 {dots_remaining} 颗豆子没吃完",
                    f"The exit is still locked because {dots_remaining} dots remain",
                    lang,
                ),
            )
            add_factor(
                "dot_progress",
                _pick(
                    f"已经吃掉 {dots_collected}/{total_dots} 颗豆子，剩余 {dots_remaining} 颗",
                    f"Pac-Man has cleared {dots_collected}/{total_dots} dots, with {dots_remaining} left",
                    lang,
                ),
            )
            add_factor(
                "nearest_dot",
                _pick(
                    f"最近的豆子在{_direction_label(evidence.nearest_dot_direction, lang)}，距离 {evidence.nearest_dot_distance} 步",
                    f"The nearest dot is {evidence.nearest_dot_distance} steps away to the {_direction_label(evidence.nearest_dot_direction, lang)}",
                    lang,
                ),
                contrastive=evidence.nearest_dot_distance >= 0,
            )
        else:
            add_factor(
                "phase_objective",
                _pick(
                    "所有豆子已经吃完，当前目标改为尽快冲向出口",
                    "All dots are cleared, so the current objective is to sprint for the exit",
                    lang,
                ),
            )

        add_factor(
            "exit_status",
            _pick(
                f"出口在{_direction_label(evidence.exit_direction, lang)}，距离 {evidence.exit_distance} 步，目前{'已开启' if evidence.exit_open else '未开启'}",
                f"The exit is {evidence.exit_distance} steps away to the {_direction_label(evidence.exit_direction, lang)} and is currently {'open' if evidence.exit_open else 'locked'}",
                lang,
            ),
        )

        add_factor(
            "monster_pressure",
            _pick(
                f"最近的怪物 #{evidence.nearest_monster_id} 在{_direction_label(evidence.nearest_monster_direction, lang)}，距离 {evidence.nearest_monster_distance} 步，属于{_pressure_band(evidence.nearest_monster_distance, lang)}",
                f"Nearest monster #{evidence.nearest_monster_id} is {evidence.nearest_monster_distance} steps away to the {_direction_label(evidence.nearest_monster_direction, lang)}, creating {_pressure_band(evidence.nearest_monster_distance, lang)}",
                lang,
            ),
            contrastive=True,
        )

        add_factor(
            "chosen_action_risk",
            _pick(
                f"当前动作 {_action_short(chosen, lang)} 的即时碰撞风险约为 {chosen_risk:.0%}，属于{_risk_band(chosen_risk, lang)}",
                f"The chosen action {chosen} carries about {chosen_risk:.0%} immediate collision risk, which is {_risk_band(chosen_risk, lang)}",
                lang,
            ),
            contrastive=True,
        )

        add_factor(
            "target_alignment",
            _pick(
                f"动作 {_action_short(chosen, lang)} {'与' if aligned else '没有直接对准'}当前目标 {target_label} 的方向",
                f"The chosen action {chosen} {'stays aligned with' if aligned else 'does not point directly toward'} {target_label}",
                lang,
            ),
        )

        if evidence.has_safer_alternative:
            add_factor(
                "safety_tradeoff",
                _pick(
                    "存在更安全的替代动作，说明这一步在安全和推进目标之间做了权衡",
                    "A safer alternative exists, so this move reflects a trade-off between safety and progress",
                    lang,
                ),
            )
        else:
            add_factor(
                "no_safer_alternative",
                _pick(
                    "在当前可行动作里，这一步已经接近最安全的选择",
                    "Among the available actions, this move is already close to the safest option",
                    lang,
                ),
                contrastive=False,
            )

        if question.intent == QuestionIntent.WHY_NOT_OTHER and question.mentioned_action:
            alt = question.mentioned_action
            alt_risk = risk_map.get(alt)
            if alt_risk is None:
                add_factor(
                    "alternative_action_blocked",
                    _pick(
                        f"备选动作 {_action_short(alt, lang)} 当前不可用，通常意味着前方被墙挡住",
                        f"The alternative action {alt} is currently unavailable, which usually means a wall blocks it",
                        lang,
                    ),
                )
            else:
                add_factor(
                    "alternative_action_risk",
                    _pick(
                        f"备选动作 {_action_short(alt, lang)} 的碰撞风险约为 {alt_risk:.0%}",
                        f"The alternative action {alt} carries about {alt_risk:.0%} collision risk",
                        lang,
                    ),
                )

        if question.intent == QuestionIntent.MONSTER_INFLUENCE:
            target_monster_id = (
                question.mentioned_monster_id
                if question.mentioned_monster_id is not None
                else evidence.nearest_monster_id
            )
            for monster_id, row, col in evidence.monster_positions:
                if monster_id != target_monster_id:
                    continue
                distance = manhattan_distance(evidence.player_pos, (row, col))
                direction = get_relative_direction(evidence.player_pos, (row, col))
                add_factor(
                    f"monster_{target_monster_id}_detail",
                    _pick(
                        f"指定怪物 #{target_monster_id} 在{_direction_label(direction, lang)}，距离 {distance} 步",
                        f"Monster #{target_monster_id} is {distance} steps away to the {_direction_label(direction, lang)}",
                        lang,
                    ),
                    faithful=True,
                    contrastive=True,
                )
                break

        if question.intent == QuestionIntent.IRRELEVANT:
            add_factor(
                "question_scope",
                _pick(
                    "这个问题不属于当前 Pac-Man 对局，我会改用眼前局面来回答",
                    "This question is outside the current Pac-Man scene, so I will answer with the game context instead",
                    lang,
                ),
            )

        return list(factors.values())

    def generate_explanation(self, evidence: EvidenceRecord, question: ParsedQuestion) -> dict:
        lang = question.language
        all_factors = self.build_factors(evidence, question)
        factor_map = {factor.name: factor for factor in all_factors}
        candidate_space = frozenset(all_factors)

        selected_names = self._select_factor_names(question, evidence, factor_map)
        selected_factors = frozenset(
            factor_map[name] for name in selected_names if name in factor_map
        )
        if not selected_factors:
            selected_factors = frozenset(list(candidate_space)[: min(3, len(candidate_space))])

        groups = self._required_groups(question, factor_map)

        render_fn = lambda factors, why_question: self._render_explanation(  # noqa: E731
            evidence, question, factors, why_question
        )
        user_model = UserModel(
            user_id="player",
            language=lang,
            detail_level="medium",
            render_fn=render_fn,
        )
        why_question = WhyQuestion(
            text=question.original_text,
            topic=_TOPIC_LABELS[question.intent][lang],
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

        try:
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
            validation = ExplanationSystem.validate_explanation(explanation)
            explanation_text = explanation.nl_explanation.text
        except Exception:
            explanation_text = self._render_explanation(evidence, question, evidence_used, why_question)
            validation = {"fallback": True}

        return {
            "all_evidence": {
                "label": _pick("全部证据 (S_t)", "All Evidence (S_t)", lang),
                "factors": [
                    {
                        "name": factor.name,
                        "description": factor.description,
                        "is_true": factor.is_true,
                        "is_faithful": factor.is_faithful,
                        "is_contrastive": factor.is_contrastive,
                    }
                    for factor in sorted(candidate_space, key=lambda item: item.name)
                ],
            },
            "evidence_used": {
                "label": _pick("实际使用的证据 (E)", "Evidence Used (E)", lang),
                "factors": [
                    {"name": factor.name, "description": factor.description}
                    for factor in sorted(evidence_used, key=lambda item: item.name)
                ],
            },
            "explanation_text": {
                "label": _pick("自然语言解释 (x)", "Natural-Language Explanation (x)", lang),
                "text": explanation_text,
            },
            "validation": validation,
            "language": lang,
        }

    def _select_factor_names(
        self,
        question: ParsedQuestion,
        evidence: EvidenceRecord,
        factor_map: dict[str, ExplanationFactor],
    ) -> list[str]:
        dots_phase = evidence.dots_remaining > 0
        alt_name = (
            "alternative_action_risk"
            if "alternative_action_risk" in factor_map
            else "alternative_action_blocked"
        )
        monster_detail = next(
            (name for name in factor_map if name.startswith("monster_") and name.endswith("_detail")),
            None,
        )

        base_target = "nearest_dot" if dots_phase else "exit_status"
        selections = {
            QuestionIntent.WHY_THIS_ACTION: [
                "phase_objective",
                base_target,
                "monster_pressure",
                "chosen_action_risk",
                "target_alignment",
                "safety_tradeoff",
            ],
            QuestionIntent.WHY_NOT_OTHER: [
                "phase_objective",
                base_target,
                "monster_pressure",
                "chosen_action_risk",
                alt_name,
                "target_alignment",
                "safety_tradeoff",
            ],
            QuestionIntent.MONSTER_INFLUENCE: [
                "phase_objective",
                base_target,
                "chosen_action_risk",
                monster_detail or "monster_pressure",
                "monster_pressure",
                "target_alignment",
            ],
            QuestionIntent.PATH_REASON: [
                "phase_objective",
                base_target,
                "monster_pressure",
                "chosen_action_risk",
                "target_alignment",
                "safety_tradeoff",
            ],
            QuestionIntent.SAFETY_REASON: [
                "phase_objective",
                "monster_pressure",
                monster_detail or "monster_pressure",
                "chosen_action_risk",
                "safety_tradeoff",
            ],
            QuestionIntent.GOAL_REASON: [
                "phase_objective",
                "dot_progress" if dots_phase else "exit_status",
                base_target,
                "monster_pressure",
                "chosen_action_risk",
                "target_alignment",
            ],
            QuestionIntent.DOT_COLLECTION: [
                "phase_objective",
                "dot_progress",
                "nearest_dot",
                "monster_pressure",
                "chosen_action_risk",
                "target_alignment",
            ],
            QuestionIntent.GENERAL: [
                "phase_objective",
                "exit_status",
                "monster_pressure",
                "chosen_action_risk",
                "dot_progress" if dots_phase else "target_alignment",
                "target_alignment",
            ],
            QuestionIntent.IRRELEVANT: [
                "question_scope",
                "phase_objective",
                "monster_pressure",
                "chosen_action_risk",
            ],
        }

        chosen_names = []
        for name in selections.get(question.intent, selections[QuestionIntent.GENERAL]):
            if name in factor_map and name not in chosen_names:
                chosen_names.append(name)
        return chosen_names

    def _required_groups(
        self,
        question: ParsedQuestion,
        factor_map: dict[str, ExplanationFactor],
    ) -> list[set[str]]:
        available = set(factor_map)
        monster_detail = next(
            (name for name in factor_map if name.startswith("monster_") and name.endswith("_detail")),
            None,
        )
        alt_group = {"alternative_action_risk", "alternative_action_blocked"}

        groups_by_intent = {
            QuestionIntent.WHY_THIS_ACTION: [
                {"phase_objective", "dot_progress", "exit_status", "nearest_dot"},
                {"monster_pressure"},
                {"chosen_action_risk", "target_alignment", "safety_tradeoff"},
            ],
            QuestionIntent.WHY_NOT_OTHER: [
                {"phase_objective", "dot_progress", "exit_status", "nearest_dot"},
                {"monster_pressure", "chosen_action_risk"},
                alt_group | {"target_alignment", "safety_tradeoff"},
            ],
            QuestionIntent.MONSTER_INFLUENCE: [
                {monster_detail} if monster_detail else {"monster_pressure"},
                {"phase_objective", "dot_progress", "exit_status", "nearest_dot"},
                {"chosen_action_risk", "target_alignment", "safety_tradeoff"},
            ],
            QuestionIntent.PATH_REASON: [
                {"phase_objective", "dot_progress", "exit_status", "nearest_dot"},
                {"monster_pressure"},
                {"chosen_action_risk", "target_alignment", "safety_tradeoff"},
            ],
            QuestionIntent.SAFETY_REASON: [
                {monster_detail} if monster_detail else {"monster_pressure"},
                {"monster_pressure", "chosen_action_risk"},
                {"phase_objective", "exit_status", "safety_tradeoff"},
            ],
            QuestionIntent.GOAL_REASON: [
                {"phase_objective", "dot_progress", "exit_status"},
                {"nearest_dot", "exit_status", "target_alignment"},
                {"monster_pressure", "chosen_action_risk"},
            ],
            QuestionIntent.DOT_COLLECTION: [
                {"phase_objective", "dot_progress"},
                {"nearest_dot", "target_alignment"},
                {"monster_pressure", "chosen_action_risk"},
            ],
            QuestionIntent.GENERAL: [
                {"phase_objective", "dot_progress", "exit_status"},
                {"monster_pressure"},
                {"chosen_action_risk", "target_alignment"},
            ],
            QuestionIntent.IRRELEVANT: [
                {"question_scope"},
            ],
        }

        groups: list[set[str]] = []
        for group in groups_by_intent.get(question.intent, groups_by_intent[QuestionIntent.GENERAL]):
            actual_group = {name for name in group if name and name in available}
            if actual_group:
                groups.append(actual_group)
        return groups

    @staticmethod
    def _satisfies_groups(factors: FrozenSet[ExplanationFactor], groups: list[set[str]]) -> bool:
        names = {factor.name for factor in factors}
        return all(names & group for group in groups)

    def _render_explanation(
        self,
        evidence: EvidenceRecord,
        question: ParsedQuestion,
        factors: FrozenSet[ExplanationFactor],
        why_question: WhyQuestion,
    ) -> str:
        del factors, why_question

        lang = question.language
        chosen = evidence.chosen_action
        chosen_label = _action_label(chosen, lang)
        chosen_risk = dict(evidence.collision_risks).get(chosen, 0.0)
        risk_text = _risk_band(chosen_risk, lang)
        pressure_text = _pressure_band(evidence.nearest_monster_distance, lang)

        dots_phase = evidence.dots_remaining > 0
        target_dir = evidence.nearest_dot_direction if dots_phase else evidence.exit_direction
        target_dist = evidence.nearest_dot_distance if dots_phase else evidence.exit_distance
        target_label = _pick("最近的豆子", "the nearest dot", lang) if dots_phase else _pick("出口", "the exit", lang)
        aligned = _action_aligns(chosen, target_dir)

        alt = question.mentioned_action
        alt_risk = dict(evidence.collision_risks).get(alt) if alt else None
        alt_aligned = _action_aligns(alt, target_dir) if alt else False

        phase_sentence = (
            _pick(
                f"当前还处在吃豆阶段，剩余 {evidence.dots_remaining} 颗豆子，所以出口还没有真正成为第一目标。",
                f"Pac-Man is still in the dot-clearing phase with {evidence.dots_remaining} dots left, so the exit is not the immediate priority yet.",
                lang,
            )
            if dots_phase
            else _pick(
                "所有豆子都已经吃完，出口现在已开启，当前目标就是尽快冲向出口。",
                "All dots are cleared and the exit is open, so the policy is now focused on finishing quickly.",
                lang,
            )
        )

        target_sentence = _pick(
            f"{target_label}在{_direction_label(target_dir, lang)}，距离 {target_dist} 步，而这一步{('继续沿着目标方向推进' if aligned else '并没有直接对准目标方向')}。",
            f"{target_label.capitalize()} is {target_dist} steps away to the {_direction_label(target_dir, lang)}, and this move {('keeps Pac-Man aligned with that target' if aligned else 'does not point straight at it')}.",
            lang,
        )

        monster_sentence = _pick(
            f"最近的怪物 #{evidence.nearest_monster_id} 在{_direction_label(evidence.nearest_monster_direction, lang)}，距离 {evidence.nearest_monster_distance} 步，这一带现在属于{pressure_text}。",
            f"Nearest monster #{evidence.nearest_monster_id} is {evidence.nearest_monster_distance} steps away to the {_direction_label(evidence.nearest_monster_direction, lang)}, so this side of the maze is under {pressure_text}.",
            lang,
        )

        risk_sentence = _pick(
            f"所选动作 {chosen_label} 的即时碰撞风险约为 {chosen_risk:.0%}，属于{risk_text}。",
            f"The chosen action {chosen} carries about {chosen_risk:.0%} immediate collision risk, which is {risk_text}.",
            lang,
        )

        if question.intent == QuestionIntent.IRRELEVANT:
            return _join_sentences(
                [
                    _pick(
                        "这个问题和当前 Pac-Man 对局不直接相关，所以我改为总结眼前局面。",
                        "That question is not directly about the current Pac-Man scene, so I am summarizing the visible game state instead.",
                        lang,
                    ),
                    phase_sentence,
                    monster_sentence,
                    risk_sentence,
                ],
                lang,
            )

        if question.intent == QuestionIntent.WHY_NOT_OTHER and alt:
            if alt_risk is None:
                comparison = _pick(
                    f"没有选择 {_action_label(alt, lang)}，首先因为那个方向当前不可走，通常是被墙体挡住了。",
                    f"{alt} was not selected because that action is currently unavailable, usually due to a wall.",
                    lang,
                )
            elif alt_risk > chosen_risk:
                comparison = _pick(
                    f"没有选择 {_action_label(alt, lang)}，主要因为它的碰撞风险约为 {alt_risk:.0%}，高于当前动作的 {chosen_risk:.0%}。",
                    f"{alt} was not preferred because its collision risk is about {alt_risk:.0%}, higher than the chosen move at {chosen_risk:.0%}.",
                    lang,
                )
            elif alt_risk < chosen_risk:
                comparison = _pick(
                    f"{_action_label(alt, lang)} 看起来更安全一些，风险约为 {alt_risk:.0%}；但当前动作更贴合当前目标方向，所以策略在安全和推进之间做了取舍。",
                    f"{alt} appears safer at roughly {alt_risk:.0%} risk, but the chosen move fits the active target better, so the policy is trading a bit of safety for progress.",
                    lang,
                )
            else:
                comparison = _pick(
                    f"{_action_label(alt, lang)} 和当前动作的风险接近，但它{('同样' if alt_aligned else '没有')}更贴合当前目标方向，所以没有明显优势。",
                    f"{alt} has roughly the same risk as the chosen move, but it {('matches' if alt_aligned else 'does not match')} the active target direction better, so it offers no clear advantage.",
                    lang,
                )

            return _join_sentences([comparison, phase_sentence, monster_sentence], lang)

        if question.intent == QuestionIntent.MONSTER_INFLUENCE:
            focus_monster_id = (
                question.mentioned_monster_id
                if question.mentioned_monster_id is not None
                else evidence.nearest_monster_id
            )
            focus_monster_sentence = monster_sentence
            for monster_id, row, col in evidence.monster_positions:
                if monster_id != focus_monster_id:
                    continue
                focus_distance = manhattan_distance(evidence.player_pos, (row, col))
                focus_direction = get_relative_direction(evidence.player_pos, (row, col))
                focus_pressure = _pressure_band(focus_distance, lang)
                focus_monster_sentence = _pick(
                    f"你问到的怪物 #{focus_monster_id} 在{_direction_label(focus_direction, lang)}，距离 {focus_distance} 步，对当前路线形成了{focus_pressure}。",
                    f"The monster you asked about, #{focus_monster_id}, is {focus_distance} steps away to the {_direction_label(focus_direction, lang)}, creating {focus_pressure} for the current route.",
                    lang,
                )
                break
            return _join_sentences(
                [
                    _pick(
                        "是的，怪物压力确实参与了这次决策。",
                        "Yes, monster pressure materially influenced this decision.",
                        lang,
                    ),
                    focus_monster_sentence,
                    risk_sentence,
                    _pick(
                        "因此这一步不仅是在推进目标，也是在避免把 Pac-Man 送进更危险的走廊。",
                        "So the move is not only about progress, but also about avoiding a corridor that has become more dangerous.",
                        lang,
                    ),
                ],
                lang,
            )

        if question.intent == QuestionIntent.SAFETY_REASON:
            safety_opening = _pick(
                f"这一格目前{'并不安全' if chosen_risk >= 0.4 or evidence.nearest_monster_distance <= 3 else '相对安全'}。",
                f"This position is {'not very safe' if chosen_risk >= 0.4 or evidence.nearest_monster_distance <= 3 else 'relatively safe'} right now.",
                lang,
            )
            tradeoff_sentence = (
                _pick(
                    "另外还存在更安全的备选动作，所以当前动作更像是在冒一点风险来换取推进速度。",
                    "There is also a safer alternative available, so the chosen move is accepting some danger in exchange for progress.",
                    lang,
                )
                if evidence.has_safer_alternative
                else _pick(
                    "从当前可行动作看，这一步已经接近最稳妥的选择。",
                    "Among the available moves, this is already close to the safest choice.",
                    lang,
                )
            )
            return _join_sentences([safety_opening, monster_sentence, risk_sentence, tradeoff_sentence], lang)

        if question.intent == QuestionIntent.GOAL_REASON:
            progress_sentence = (
                _pick(
                    f"要赢下这一局，眼下真正的进度是先把剩下的 {evidence.dots_remaining} 颗豆子清掉，而不是立刻冲出口。",
                    f"To win from here, the real notion of progress is clearing the remaining {evidence.dots_remaining} dots before rushing the exit.",
                    lang,
                )
                if dots_phase
                else _pick(
                    f"出口已经打开，而且就在{_direction_label(evidence.exit_direction, lang)}，距离 {evidence.exit_distance} 步，当前策略已经进入最后冲线阶段。",
                    f"The exit is open, {evidence.exit_distance} steps away to the {_direction_label(evidence.exit_direction, lang)}, so the policy is in its final sprint.",
                    lang,
                )
            )
            return _join_sentences([progress_sentence, target_sentence, monster_sentence], lang)

        if question.intent == QuestionIntent.DOT_COLLECTION:
            return _join_sentences(
                [
                    _pick(
                        f"吃豆是当前的主目标，已经吃掉 {evidence.dots_collected}/{evidence.total_dots} 颗，还剩 {evidence.dots_remaining} 颗。",
                        f"Dot collection is the active sub-goal: {evidence.dots_collected}/{evidence.total_dots} dots are cleared and {evidence.dots_remaining} remain.",
                        lang,
                    ),
                    target_sentence,
                    monster_sentence,
                ],
                lang,
            )

        if question.intent == QuestionIntent.PATH_REASON:
            return _join_sentences(
                [
                    _pick(
                        "这条路线不是只看最短路，而是在“追目标”和“避怪物危险区”两个目标之间平衡出来的。",
                        "This route is not chosen by shortest distance alone; it balances target progress against monster danger zones.",
                        lang,
                    ),
                    target_sentence,
                    monster_sentence,
                    risk_sentence,
                ],
                lang,
            )

        if question.intent == QuestionIntent.GENERAL:
            return _join_sentences(
                [
                    phase_sentence,
                    target_sentence,
                    monster_sentence,
                    risk_sentence,
                ],
                lang,
            )

        return _join_sentences(
            [
                _pick(
                    f"Pac-Man 选择{chosen_label}，核心是同时兼顾当前目标和怪物压力。",
                    f"Pac-Man chose to {chosen} in order to balance the active target with monster pressure.",
                    lang,
                ),
                phase_sentence,
                target_sentence,
                risk_sentence,
            ],
            lang,
        )
