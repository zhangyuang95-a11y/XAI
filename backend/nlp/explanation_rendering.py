"""Message planning and conversational rendering for explanation IR."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from .explanation_model import (
    DiscourseRelation,
    ExplanationDocumentV3,
    ExplanationIR,
    ExplanationLayer,
    ExplanationMessagePlanV1,
    ExplanationUnit,
    PlannedSentence,
    RenderedSentence,
    SentenceRole,
)


class ExplanationMessagePlanner:
    """Select a compact, condition-preserving conversational story."""

    def plan(self, explanation_ir: ExplanationIR) -> ExplanationMessagePlanV1:
        by_layer = {
            layer: tuple(
                unit for unit in explanation_ir.units if unit.layer == layer
            )
            for layer in ExplanationLayer
        }
        query_answers = by_layer[ExplanationLayer.QUERY_ANSWER]
        final = by_layer[ExplanationLayer.FINAL_ACTION]
        proposal = by_layer[ExplanationLayer.POLICY_PROPOSAL]
        coordination = by_layer[ExplanationLayer.COORDINATION]
        effects = by_layer[ExplanationLayer.ACTION_EFFECT]
        goals = by_layer[ExplanationLayer.TASK_GOAL]
        trace = by_layer[ExplanationLayer.PROPOSAL_RATIONALE][:2]
        counterfactual = by_layer[ExplanationLayer.COUNTERFACTUAL]
        changed = _proposal_differs_from_final(proposal, final)
        planned: list[PlannedSentence] = []
        charger_queue_effects = tuple(
            unit
            for unit in effects
            if unit.predicate == "charger_queue_context"
        )
        if charger_queue_effects:
            return ExplanationMessagePlanV1(
                sentences=(
                    PlannedSentence(
                        role=SentenceRole.ANSWER,
                        relation=DiscourseRelation.STANDALONE,
                        unit_ids=_ids(charger_queue_effects[:1]),
                    ),
                ),
                ir_hash=explanation_ir.ir_hash,
                language=explanation_ir.language,
            )

        focused_effects = tuple(
            unit
            for unit in effects
            if isinstance(unit.value, Mapping)
            and str(unit.value.get("study_focus", ""))
            in {
                "energy", "charge_threshold", "task", "collaboration",
                "allocation", "collision",
            }
        )

        # Study questions name one explanatory dimension.  A focused answer
        # should contain only that observed effect; unrelated policy-trace,
        # task-allocation, energy, or route details remain available for their
        # own questions instead of being appended to every response.
        if focused_effects:
            return ExplanationMessagePlanV1(
                sentences=(
                    PlannedSentence(
                        role=SentenceRole.ANSWER,
                        relation=DiscourseRelation.STANDALONE,
                        unit_ids=_ids(focused_effects[:1]),
                    ),
                ),
                ir_hash=explanation_ir.ir_hash,
                language=explanation_ir.language,
            )

        if explanation_ir.query_kind == "state_query":
            if query_answers:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.ANSWER,
                        relation=DiscourseRelation.STANDALONE,
                        unit_ids=_ids(query_answers[:1]),
                    )
                )
        elif explanation_ir.query_kind == "action_why_not":
            if query_answers:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.ANSWER,
                        relation=DiscourseRelation.BECAUSE,
                        unit_ids=_ids(query_answers[:1]),
                    )
                )
            process = (
                *goals[:1],
                *proposal[:1],
                *coordination[:1],
                *final[:1],
            )
            if process:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.PUBLIC_PROCESS,
                        relation=(
                            DiscourseRelation.BUT
                            if changed
                            else DiscourseRelation.BECAUSE
                        ),
                        unit_ids=_ids(process),
                    )
                )
        elif explanation_ir.query_kind in {
            "objective_query",
            "objective_why_not",
        }:
            contrast = tuple(
                unit
                for unit in goals
                if unit.predicate == "objective_not_selected"
            )
            selected = contrast or goals[:1]
            if selected:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.ANSWER,
                        relation=(
                            DiscourseRelation.BECAUSE
                            if contrast
                            else DiscourseRelation.STANDALONE
                        ),
                        unit_ids=_ids(selected),
                    )
                )
        elif explanation_ir.query_kind == "counterfactual" and counterfactual:
            planned.append(
                PlannedSentence(
                    role=SentenceRole.COUNTERFACTUAL_RESULT,
                    relation=DiscourseRelation.IF_THEN,
                    unit_ids=_ids((*counterfactual, *final)),
                )
            )
            public = (
                *goals,
                *(proposal[:1] if changed else ()),
                *(coordination[:1] if changed else ()),
            )
            if public:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.PUBLIC_PROCESS,
                        relation=DiscourseRelation.BECAUSE,
                        unit_ids=_ids(public),
                    )
                )
        elif changed and coordination:
            planned.append(
                PlannedSentence(
                    role=SentenceRole.ANSWER,
                    relation=DiscourseRelation.BECAUSE,
                    unit_ids=_ids((*final, *coordination[:1])),
                )
            )
            public = (*goals[:1], *proposal[:1])
            if public:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.PUBLIC_PROCESS,
                        relation=DiscourseRelation.BUT,
                        unit_ids=_ids(public),
                    )
                )
        else:
            # When proposal and final action coincide, one sentence can cite
            # both facts without repeating the action in participant-facing text.
            planned.append(
                PlannedSentence(
                    role=SentenceRole.ANSWER,
                    relation=DiscourseRelation.STANDALONE,
                    unit_ids=_ids((*final, *proposal[:1], *coordination[:1])),
                )
            )
            # Coordination prose often repeats the final action.  When the
            # proposal survived unchanged, the direct answer already conveys
            # that fact, so retain only the task context here.
            public = goals[:1]
            if public:
                planned.append(
                    PlannedSentence(
                        role=SentenceRole.PUBLIC_PROCESS,
                        relation=DiscourseRelation.BECAUSE,
                        unit_ids=_ids(public),
                    )
                )
        if trace:
            planned.append(
                PlannedSentence(
                    role=SentenceRole.TRACE_RATIONALE,
                    relation=DiscourseRelation.CONTRAST,
                    unit_ids=_ids(trace),
                )
            )
        if effects:
            planned.append(
                PlannedSentence(
                    role=SentenceRole.PUBLIC_PROCESS,
                    relation=DiscourseRelation.BECAUSE,
                    unit_ids=_ids(effects[:1]),
                )
            )
        return ExplanationMessagePlanV1(
            sentences=tuple(planned[:3]),
            ir_hash=explanation_ir.ir_hash,
            language=explanation_ir.language,
        )


class ConversationalIRRenderer:
    """Render a message plan with general discourse rules, never task rules."""

    def render(
        self,
        explanation_ir: ExplanationIR,
        plan: ExplanationMessagePlanV1 | None = None,
        *,
        issues: Sequence[str] = (),
    ) -> ExplanationDocumentV3:
        message_plan = plan or ExplanationMessagePlanner().plan(explanation_ir)
        units = {item.unit_id: item for item in explanation_ir.units}
        rendered: list[RenderedSentence] = []
        for sentence in message_plan.sentences:
            selected = tuple(
                units[unit_id]
                for unit_id in sentence.unit_ids
                if unit_id in units
            )
            text = self._render_sentence(
                sentence,
                selected,
                explanation_ir=explanation_ir,
            )
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for unit in selected
                    for evidence_id in unit.evidence_ids
                )
            )
            rendered.append(
                RenderedSentence(
                    role=sentence.role,
                    text=_ensure_terminal(text, explanation_ir.language),
                    unit_ids=sentence.unit_ids,
                    evidence_ids=evidence_ids,
                )
            )
        return ExplanationDocumentV3(
            sentences=tuple(rendered),
            ir_hash=explanation_ir.ir_hash,
            message_plan=message_plan,
            validation_issues=tuple(issues),
        )

    def _render_sentence(
        self,
        planned: PlannedSentence,
        units: Sequence[ExplanationUnit],
        *,
        explanation_ir: ExplanationIR,
    ) -> str:
        chinese = explanation_ir.language == "zh-CN"
        by_layer = {
            layer: tuple(item for item in units if item.layer == layer)
            for layer in ExplanationLayer
        }
        if planned.role == SentenceRole.COUNTERFACTUAL_RESULT:
            counterfactual = by_layer[ExplanationLayer.COUNTERFACTUAL]
            return _first_phrase(counterfactual) or _first_phrase(units)

        if planned.role == SentenceRole.ANSWER:
            if explanation_ir.query_kind == "action_why_not":
                answers = by_layer[ExplanationLayer.QUERY_ANSWER]
                if answers:
                    return _first_phrase(answers)
            coordination = by_layer[ExplanationLayer.COORDINATION]
            if coordination:
                return _first_phrase(coordination)
            return (
                _first_phrase(by_layer[ExplanationLayer.FINAL_ACTION])
                or _first_phrase(by_layer[ExplanationLayer.QUERY_ANSWER])
                or _first_phrase(by_layer[ExplanationLayer.TASK_GOAL])
                or _first_phrase(units)
            )

        if planned.role == SentenceRole.PUBLIC_PROCESS:
            effects = by_layer[ExplanationLayer.ACTION_EFFECT]
            if effects:
                return _first_phrase(effects)
            goals = by_layer[ExplanationLayer.TASK_GOAL]
            proposals = by_layer[ExplanationLayer.POLICY_PROPOSAL]
            coordination = by_layer[ExplanationLayer.COORDINATION]
            if explanation_ir.query_kind == "action_why_not":
                proposal_text = _first_phrase(proposals)
                goal_text = _first_phrase(goals)
                coordination_text = _first_phrase(coordination)
                final_text = _first_phrase(
                    by_layer[ExplanationLayer.FINAL_ACTION]
                )
                if chinese:
                    first = proposal_text
                    if first and goal_text:
                        goal = _drop_repeated_subject(
                            goal_text,
                            explanation_ir.target,
                            units=goals,
                        )
                        goal = re.sub(r"^(?:正在|正)", "", goal)
                        first = f"{first}，继续{goal}"
                    if coordination_text:
                        return (
                            f"{first}；不过，{coordination_text}"
                            if first
                            else coordination_text
                        )
                    return first or final_text or goal_text
                first = proposal_text
                if first and goal_text:
                    goal = _drop_repeated_subject(
                        goal_text,
                        explanation_ir.target,
                        units=goals,
                    )
                    goal = re.sub(
                        r"^(?:is currently trying to|is trying to|is working toward|is assigned to)\s+",
                        "",
                        goal,
                    )
                    first = f"{first}, continuing to {goal}"
                if coordination_text:
                    return (
                        f"{first}; however, {coordination_text}"
                        if first
                        else coordination_text
                    )
                return first or final_text or goal_text
            if explanation_ir.query_kind == "counterfactual" and goals:
                parts = [_first_phrase((goal,)) for goal in goals]
                if proposals:
                    parts.append(_first_phrase(proposals))
                if coordination:
                    parts.append(_first_phrase(coordination))
                separator = "；" if chinese else "; "
                return separator.join(part for part in parts if part)
            if proposals:
                proposal = _replace_subject_with_pronoun(
                    _first_phrase(proposals),
                    explanation_ir.target,
                    units=proposals,
                    language=explanation_ir.language,
                )
                if goals:
                    goal = _drop_repeated_subject(
                        _first_phrase(goals),
                        explanation_ir.target,
                        units=goals,
                    )
                    if chinese:
                        goal = re.sub(r"^(?:正在|正)", "", goal)
                        return f"{proposal}，继续{goal}"
                    goal = re.sub(
                        r"^(?:is currently trying to|is trying to|is working toward|is assigned to)\s+",
                        "",
                        goal,
                    )
                    return f"{proposal}, continuing to {goal}"
                return proposal
            if goals:
                return _replace_subject_with_pronoun(
                    _first_phrase(goals),
                    explanation_ir.target,
                    units=goals,
                    language=explanation_ir.language,
                )
            if coordination:
                return _first_phrase(coordination)
            return ""

        if planned.role == SentenceRole.TRACE_RATIONALE:
            proposal = next(
                (
                    unit
                    for unit in explanation_ir.units
                    if unit.layer == ExplanationLayer.POLICY_PROPOSAL
                ),
                None,
            )
            selected_action = _action_label(proposal)
            reasons = [
                _strip_terminal_punctuation(item.reference_text)
                for item in by_layer[ExplanationLayer.PROPOSAL_RATIONALE]
                if item.reference_text.strip()
            ]
            if chinese:
                prefix = (
                    f"选择{selected_action}的原因之一是："
                    if selected_action
                    else "做出这一选择的原因之一是："
                )
                if len(reasons) > 1:
                    return prefix + reasons[0] + "；此外，" + reasons[1]
                return prefix + (reasons[0] if reasons else "当前条件更合适")
            prefix = (
                f"One reason supporting {selected_action} was that "
                if selected_action
                else "One reason supporting this choice was that "
            )
            if len(reasons) > 1:
                return prefix + reasons[0] + "; additionally, " + reasons[1]
            return prefix + (reasons[0] if reasons else "it fit the current conditions better")
        return _first_phrase(units)


def _ids(units: Sequence[ExplanationUnit]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(unit.unit_id for unit in units))


def _first_phrase(units: Sequence[ExplanationUnit]) -> str:
    return next(
        (
            _strip_terminal_punctuation(unit.reference_text)
            for unit in units
            if unit.reference_text.strip()
        ),
        "",
    )


def _action_label(unit: ExplanationUnit | None) -> str:
    if unit is None or not isinstance(unit.value, Mapping):
        return ""
    return str(unit.value.get("action_label", unit.value.get("action", "")))


def _proposal_differs_from_final(
    proposals: Sequence[ExplanationUnit],
    finals: Sequence[ExplanationUnit],
) -> bool:
    if not proposals or not finals:
        return False
    proposal = proposals[0].value
    final = finals[0].value
    if not isinstance(proposal, Mapping) or not isinstance(final, Mapping):
        return False
    return str(proposal.get("action", "")) != str(final.get("action", ""))


def _drop_repeated_subject(
    phrase: str,
    target_id: str,
    *,
    units: Sequence[ExplanationUnit],
) -> str:
    candidates = [str(target_id).replace("_", " ")]
    for unit in units:
        candidates.extend(unit.allowed_entities)
    result = phrase.strip()
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if candidate and result.casefold().startswith(candidate.casefold()):
            return result[len(candidate) :].lstrip(" ,，")
    return result


def _replace_subject_with_pronoun(
    phrase: str,
    target_id: str,
    *,
    units: Sequence[ExplanationUnit],
    language: str,
) -> str:
    tail = _drop_repeated_subject(phrase, target_id, units=units)
    if tail == phrase.strip():
        return phrase.strip()
    return ("它" + tail) if language == "zh-CN" else ("It " + tail)


def _ensure_terminal(text: str, language: str) -> str:
    value = _strip_terminal_punctuation(text)
    return value + ("。" if language == "zh-CN" else ".")

def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip("。！？.!?;； ")
