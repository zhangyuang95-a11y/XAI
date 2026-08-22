"""Environment-neutral IR, message planning, and conversational rendering.

The online path compiles typed execution evidence into this IR and realizes the
selected units deterministically.  Environment adapters own vocabulary and
short predicate phrases; this module deliberately contains no environment
action or task rule.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Mapping, Sequence

from .language_policy import normalize_requested_language
from .schemas import EvidenceBundle


from .explanation_model import (
    DiscourseRelation,
    EvidenceVocabulary,
    ExplanationDocumentV2,
    ExplanationDocumentV3,
    ExplanationIR,
    ExplanationLayer,
    ExplanationMessagePlanV1,
    ExplanationSemantics,
    ExplanationSemanticsAdapter,
    ExplanationUnit,
    PlannedSentence,
    RenderedSentence,
    SentenceRole,
)


class ExplanationIRCompiler:
    """Compile live execution evidence into compact, provenance-bound units."""

    def compile_common(
        self,
        evidence: EvidenceBundle,
        *,
        requested_language: str,
        semantics: ExplanationSemantics | None = None,
    ) -> ExplanationIR:
        """Compile the condition-invariant part exactly once."""

        return self.compile(
            evidence,
            include_program_trace=False,
            requested_language=requested_language,
            semantics=semantics,
        )

    def add_trace(
        self,
        common_ir: ExplanationIR,
        evidence: EvidenceBundle,
        *,
        semantics: ExplanationSemantics | None = None,
    ) -> ExplanationIR:
        """Derive the treatment IR without rebuilding any public unit."""

        if (
            not evidence.program_trace
            or common_ir.query_kind
            not in {"action_explanation", "action_why_not", "counterfactual"}
        ):
            return common_ir
        vocabulary: ExplanationSemantics = semantics or EvidenceVocabulary(evidence)
        proposal = next(
            (
                str(unit.value.get("action", ""))
                for unit in common_ir.units
                if unit.layer == ExplanationLayer.POLICY_PROPOSAL
                and isinstance(unit.value, Mapping)
            ),
            "",
        )
        trace_units = self._trace_units(
            evidence,
            target=common_ir.target,
            proposed=proposal,
            desired_action=(
                _desired_action_query(evidence.query_plan, common_ir.target)
                if common_ir.query_kind == "action_why_not"
                else ""
            ),
            vocabulary=vocabulary,
            language=common_ir.language,
        )
        if not trace_units:
            return common_ir
        return ExplanationIR(
            target=common_ir.target,
            language=common_ir.language,
            query_kind=common_ir.query_kind,
            units=(*common_ir.units, *trace_units),
            condition="rcpd_trace",
        )

    def compile(
        self,
        evidence: EvidenceBundle,
        *,
        include_program_trace: bool,
        requested_language: str,
        semantics: ExplanationSemantics | None = None,
    ) -> ExplanationIR:
        # Output locale is persisted experiment state.  It must not be
        # overwritten by the language of the once-parsed participant question.
        language = normalize_requested_language(requested_language)
        vocabulary: ExplanationSemantics = semantics or EvidenceVocabulary(evidence)
        target = str(
            evidence.policy_results.get(
                "target",
                evidence.direct_result.get("target", "target"),
            )
        )
        contrast = evidence.direct_result.get(
            "counterfactual_contrast",
            evidence.policy_results.get("counterfactual_contrast", {}),
        )
        contrast = contrast if isinstance(contrast, Mapping) else {}
        edited_contrast = contrast.get("edited", {})
        edited_contrast = edited_contrast if isinstance(edited_contrast, Mapping) else {}
        proposed = str(
            evidence.policy_results.get(
                "proposed_action",
                evidence.direct_result.get(
                    "recorded_proposed_action",
                    evidence.direct_result.get("proposed_action", ""),
                ),
            )
            or edited_contrast.get("proposed_action")
            or edited_contrast.get("executed_action")
            or ""
        )
        final = str(
            evidence.policy_results.get(
                "executed_action",
                evidence.direct_result.get(
                    "recorded_executed_action",
                    evidence.direct_result.get(
                        "executed_action",
                        evidence.direct_result.get("argmax_action", proposed),
                    ),
                ),
            )
            or edited_contrast.get("executed_action")
            or proposed
        )
        desired_objective = _desired_objective_query(
            evidence.query_plan,
            target,
        )
        desired_action = _desired_action_query(
            evidence.query_plan,
            target,
        )
        objective_focused = _question_targets_objective(
            evidence.query_plan,
            target,
        )
        state_query = _state_query_spec(evidence, target)
        query_kind = (
            "counterfactual"
            if evidence.query_plan.requires_scene_edit
            else "objective_why_not"
            if objective_focused and desired_objective
            else "action_why_not"
            if desired_action
            else "objective_query"
            if objective_focused
            else "state_query"
            if state_query is not None
            else "action_explanation"
        )
        units: list[ExplanationUnit] = []

        if state_query is not None and not evidence.query_plan.requires_scene_edit:
            query_variable, predicate = state_query
            fact = _state_query_fact(
                evidence,
                target=target,
                predicate=predicate,
            )
            if fact is not None:
                fact_value = fact.get("value")
                units.append(
                    self._unit(
                        vocabulary,
                        language,
                        unit_id="query_answer",
                        layer=ExplanationLayer.QUERY_ANSWER,
                        predicate="queried_state",
                        arguments=(target,),
                        value={
                            "query_variable": query_variable,
                            "source_predicate": predicate,
                            "value": fact_value,
                        },
                        evidence_ids=(
                            f"state::{fact.get('fact_id', predicate)}",
                        ),
                        provenance="state",
                        salience=1.0,
                        mandatory=True,
                    )
                )
            return ExplanationIR(
                target=target,
                language=language,
                query_kind="state_query",
                units=tuple(units),
                condition="no_trace",
            )

        objectives = _selected_objectives(evidence)
        objective_context = _shared_objective_context(evidence)
        for objective_index, objective in enumerate(objectives):
            objective_id = str(objective.get("id", ""))
            objective_label = vocabulary.explanation_objective_label(
                objective_id,
                language,
            )
            value = {
                **objective,
                "label": objective_label,
                "context": objective_context,
                "show_position": _question_requests_position(
                    evidence.query_plan
                ),
            }
            objective_evidence_ids = tuple(
                str(item)
                for item in objective.get("evidence_ids", ())
                if str(item)
            ) or (
                str(
                    objective.get("evidence_id")
                    or _objective_evidence_id(evidence)
                ),
            )
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id=(
                        "goal"
                        if len(objectives) == 1
                        else f"goal_{objective.get('phase', objective_index)}"
                    ),
                    layer=ExplanationLayer.TASK_GOAL,
                    predicate="current_objective",
                    arguments=(target,),
                    value=value,
                    evidence_ids=objective_evidence_ids,
                    provenance="state",
                    salience=0.82,
                    mandatory=(
                        query_kind == "objective_query"
                        or evidence.query_plan.intent.value in {
                        "explanatory",
                        "why_not",
                        "counterfactual",
                        "mixed",
                        }
                        and query_kind != "objective_why_not"
                    ),
                    required_literals=tuple(
                        value
                        for value in (
                            vocabulary.explanation_entity_label(target, language),
                            objective_label,
                            _compact_value(objective.get("target_position")),
                        )
                        if value
                    ),
                )
            )

        if desired_objective and objectives:
            selected_objective = str(objectives[0].get("id", ""))
            contrast_value = {
                "selected_objective": selected_objective,
                "selected_label": vocabulary.explanation_objective_label(
                    selected_objective,
                    language,
                ),
                "requested_objective": desired_objective,
                "requested_label": vocabulary.explanation_objective_label(
                    desired_objective,
                    language,
                ),
                "context": objective_context,
            }
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="objective_contrast",
                    layer=ExplanationLayer.TASK_GOAL,
                    predicate="objective_not_selected",
                    arguments=(target,),
                    value=contrast_value,
                    evidence_ids=(
                        str(
                            objectives[0].get("evidence_id")
                            or _objective_evidence_id(evidence)
                        ),
                    ),
                    provenance="state",
                    salience=1.0,
                    mandatory=True,
                    required_literals=(
                        vocabulary.explanation_entity_label(target, language),
                        vocabulary.explanation_objective_label(
                            desired_objective,
                            language,
                        ),
                        vocabulary.explanation_objective_label(
                            selected_objective,
                            language,
                        ),
                    ),
                )
            )

        if objective_focused and not evidence.query_plan.requires_scene_edit:
            if units:
                return ExplanationIR(
                    target=target,
                    language=language,
                    query_kind=query_kind,
                    units=tuple(units),
                    condition="no_trace",
                )
            # Private objectives may be deliberately absent from participant
            # evidence.  Fall back to the recorded action instead of emitting
            # an empty explanation document.
            query_kind = "action_explanation"

        if desired_action:
            why_not_value, why_not_evidence_ids = _action_why_not_value(
                evidence,
                target=target,
                desired_action=desired_action,
                proposed_action=proposed,
                final_action=final,
                vocabulary=vocabulary,
                language=language,
            )
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="action_contrast",
                    layer=ExplanationLayer.QUERY_ANSWER,
                    predicate="action_not_selected",
                    arguments=(target,),
                    value=why_not_value,
                    evidence_ids=why_not_evidence_ids,
                    provenance=str(
                        why_not_value.get("reason_provenance", "state")
                    ),
                    salience=1.0,
                    mandatory=True,
                    required_literals=(
                        vocabulary.explanation_entity_label(target, language),
                        vocabulary.explanation_action_label(
                            desired_action,
                            language,
                        ),
                    ),
                )
            )

        if proposed:
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="proposal",
                    layer=ExplanationLayer.POLICY_PROPOSAL,
                    predicate="action_proposed",
                    arguments=(target,),
                    value={
                        "action": proposed,
                        "action_label": vocabulary.explanation_action_label(
                            proposed,
                            language,
                        ),
                    },
                    evidence_ids=("neural_policy",),
                    provenance="neural_policy",
                    salience=0.95,
                    mandatory=True,
                    required_literals=(
                        vocabulary.explanation_entity_label(target, language),
                        vocabulary.explanation_action_label(proposed, language),
                    ),
                )
            )

        resolution = evidence.policy_results.get(
            "action_resolution",
            evidence.direct_result.get("action_resolution", {}),
        )
        resolution = dict(resolution) if isinstance(resolution, Mapping) else {}
        coordination_fact = _coordination_fact(evidence, target)
        meaningful_resolution = bool(
            resolution.get("environment_changed_action")
            or resolution.get("blocked_reason")
            or resolution.get("relation")
        )
        if meaningful_resolution or coordination_fact or (proposed and final != proposed):
            coordination_value = {
                "proposed_action": proposed,
                "proposed_action_label": vocabulary.explanation_action_label(
                    proposed,
                    language,
                ),
                "final_action": final,
                "final_action_label": vocabulary.explanation_action_label(
                    final,
                    language,
                ),
                "resolution": resolution,
                "typed_reason": coordination_fact,
            }
            related = _related_entities(coordination_value)
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="coordination",
                    layer=ExplanationLayer.COORDINATION,
                    predicate="coordination_resolution",
                    arguments=(target, *related),
                    value=coordination_value,
                    evidence_ids=(_coordination_evidence_id(evidence, target),),
                    provenance="joint_execution",
                    salience=0.98,
                    mandatory=bool(proposed and final != proposed),
                    required_literals=tuple(
                        dict.fromkeys(
                            value
                            for value in (
                                vocabulary.explanation_entity_label(target, language),
                                vocabulary.explanation_action_label(final, language),
                                *(
                                    vocabulary.explanation_entity_label(item, language)
                                    for item in related
                                ),
                                _coordination_position(coordination_value),
                            )
                            if value
                        )
                    ),
                )
            )

        if final:
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="final",
                    layer=ExplanationLayer.FINAL_ACTION,
                    predicate="final_action",
                    arguments=(target,),
                    value={
                        "action": final,
                        "action_label": vocabulary.explanation_action_label(
                            final,
                            language,
                        ),
                    },
                    evidence_ids=("neural_policy",),
                    provenance="joint_execution",
                    salience=1.0,
                    mandatory=True,
                    required_literals=(
                        vocabulary.explanation_entity_label(target, language),
                        vocabulary.explanation_action_label(final, language),
                    ),
                )
            )

        action_effect = _action_effect_fact(evidence, target)
        if action_effect:
            effect_value = action_effect.get("value", {})
            units.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id="action_effect",
                    layer=ExplanationLayer.ACTION_EFFECT,
                    predicate=str(
                        action_effect.get("predicate", "observed_action_effect")
                    ),
                    arguments=(target,),
                    value=(
                        dict(effect_value)
                        if isinstance(effect_value, Mapping)
                        else effect_value
                    ),
                    evidence_ids=(
                        f"state::{action_effect.get('fact_id', 'action_effect')}",
                    ),
                    provenance="recorded_transition",
                    salience=0.97,
                    mandatory=False,
                )
            )

        if query_kind == "counterfactual":
            units.extend(
                _counterfactual_units(
                    evidence,
                    vocabulary=vocabulary,
                    language=language,
                    target=target,
                    unit_factory=self._unit,
                )
            )

        common_ir = ExplanationIR(
            target=target,
            language=language,
            query_kind=query_kind,
            units=tuple(units),
            condition="no_trace",
        )
        if include_program_trace:
            return self.add_trace(
                common_ir,
                evidence,
                semantics=vocabulary,
            )
        return common_ir

    def _trace_units(
        self,
        evidence: EvidenceBundle,
        *,
        target: str,
        proposed: str,
        desired_action: str = "",
        vocabulary: ExplanationSemantics,
        language: str,
    ) -> tuple[ExplanationUnit, ...]:
        candidates: list[dict[str, Any]] = []
        for index, raw in enumerate(evidence.program_trace):
            if not isinstance(raw, Mapping):
                continue
            selected = str(
                raw.get("program_selected_action", raw.get("action", proposed))
                or proposed
            )
            if proposed and selected and selected != proposed:
                continue
            trace_type = str(raw.get("trace_type", "tree_branch"))
            meaning = raw.get("observed_meaning", {})
            meaning = meaning if isinstance(meaning, Mapping) else {}
            constrained = str(raw.get("constrained_action", ""))
            path_action = str(
                meaning.get("action", constrained or selected)
                or constrained
                or selected
            )
            category = "positive"
            if trace_type == "action_constraint":
                if not raw.get("active_reason_features"):
                    continue
                priority = 0
                category = "alternative"
                path_action = constrained or path_action
            else:
                role = str(meaning.get("explanation_role", ""))
                if not role:
                    continue
                if path_action and selected and path_action != selected:
                    numeric = _trace_numeric_value(raw, meaning)
                    # A rejected alternative is explanatory only when the
                    # executed condition records a disadvantage.  Positive
                    # progress or zero detour for an unselected action cannot
                    # truthfully be presented as a reason for another action.
                    if role == "action_objective_effect" and numeric > 0.0:
                        continue
                    if role == "action_route_efficiency" and numeric <= 0.0:
                        continue
                    category = "alternative"
                elif role == "action_objective_effect" and _trace_numeric_value(
                    raw,
                    meaning,
                ) <= 0.0:
                    continue
                elif role == "action_route_efficiency" and _trace_numeric_value(
                    raw,
                    meaning,
                ) > 0.0:
                    continue
                priority = 0 if category == "positive" and role in {
                    "action_objective_effect",
                    "action_route_efficiency",
                } else 1 if role in {
                    "action_objective_effect",
                    "action_feasibility",
                    "action_safety_effect",
                } else 2
            candidates.append(
                {
                    "priority": priority,
                    "index": index,
                    "raw": raw,
                    "action_group": path_action or f"path_{index}",
                    "category": category,
                }
            )

        # Group conditions about the same action.  The selected proposal gets
        # one positive group when available; one different action supplies the
        # most salient exclusion.  This is independent of environment action
        # names and preserves the original execution order in the final IR.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate["action_group"]), []).append(candidate)

        def group_rank(items: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
            return min(
                (int(item["priority"]), int(item["index"]))
                for item in items
            )

        selected_groups: list[list[dict[str, Any]]] = []
        positive = [
            items
            for action, items in grouped.items()
            if action == proposed
            and any(item["category"] == "positive" for item in items)
        ]
        alternatives = [
            items
            for action, items in grouped.items()
            if action != proposed
            or any(item["category"] == "alternative" for item in items)
        ]
        requested_alternative = (
            grouped.get(desired_action)
            if desired_action and desired_action != proposed
            else None
        )
        # For action Why-not, an unrelated rejected action is not responsive.
        # If the program path contains no condition about the named action,
        # expose no trace rather than implying that another alternative answers
        # the question.  The shared action-contrast unit still provides the
        # observation/simulation-grounded answer in both conditions.
        if desired_action and desired_action != proposed and not requested_alternative:
            return ()
        if positive:
            selected_groups.append(min(positive, key=group_rank))
        if requested_alternative:
            selected_groups.append(requested_alternative)
        elif alternatives:
            selected_groups.append(min(alternatives, key=group_rank))
        if not selected_groups:
            selected_groups = sorted(grouped.values(), key=group_rank)[:2]
        elif len(selected_groups) < 2:
            remaining = [
                items
                for items in grouped.values()
                if items is not selected_groups[0]
            ]
            if remaining:
                selected_groups.append(min(remaining, key=group_rank))
        selected_groups = selected_groups[:2]
        selected_groups.sort(
            key=lambda items: min(int(item["index"]) for item in items)
        )

        result: list[ExplanationUnit] = []
        for ordinal, group in enumerate(selected_groups, start=1):
            # One representative condition per action group keeps the reason
            # compact and prevents repeating the proposal action in prose.
            chosen: list[dict[str, Any]] = []
            roles: set[str] = set()
            for candidate in sorted(
                group,
                key=lambda item: (int(item["priority"]), int(item["index"])),
            ):
                raw = candidate["raw"]
                meaning = raw.get("observed_meaning", {})
                meaning = meaning if isinstance(meaning, Mapping) else {}
                role = str(
                    meaning.get(
                        "explanation_role",
                        raw.get("trace_type", "path_condition"),
                    )
                )
                if role in roles:
                    continue
                chosen.append(candidate)
                roles.add(role)
                break
            chosen.sort(key=lambda item: int(item["index"]))
            primary = chosen[0]
            index = int(primary["index"])
            raw = primary["raw"]
            meaning = raw.get("observed_meaning", {})
            meaning = dict(meaning) if isinstance(meaning, Mapping) else {}
            constrained_actions = tuple(
                dict.fromkeys(
                    _trace_alternative_action(
                        item["raw"],
                        selected_action=str(
                            item["raw"].get(
                                "program_selected_action",
                                item["raw"].get("action", proposed),
                            )
                            or proposed
                        ),
                    )
                    for item in chosen
                    if _trace_alternative_action(
                        item["raw"],
                        selected_action=str(
                            item["raw"].get(
                                "program_selected_action",
                                item["raw"].get("action", proposed),
                            )
                            or proposed
                        ),
                    )
                )
            )
            constrained = constrained_actions[0] if constrained_actions else ""
            selected_action = str(
                raw.get("program_selected_action", raw.get("action", proposed))
                or proposed
            )
            conditions = tuple(
                _trace_condition_value(
                    item["raw"],
                    index=int(item["index"]),
                    selected_action=selected_action,
                    vocabulary=vocabulary,
                    language=language,
                )
                for item in chosen
            )
            value = {**conditions[0], "conditions": conditions}
            required = [vocabulary.explanation_action_label(selected_action, language)]
            required.extend(
                vocabulary.explanation_action_label(action, language)
                for action in constrained_actions
            )
            bound_entities = tuple(
                dict.fromkeys(
                    str(bound)
                    for item in chosen
                    for bound in dict(
                        item["raw"].get("bound_entities", {})
                    ).values()
                    if str(bound)
                )
            )
            result.append(
                self._unit(
                    vocabulary,
                    language,
                    unit_id=f"trace_{ordinal}",
                    layer=ExplanationLayer.PROPOSAL_RATIONALE,
                    predicate=(
                        "action_constraint"
                        if str(raw.get("trace_type")) == "action_constraint"
                        else "executed_path_condition"
                    ),
                    arguments=(
                        target,
                        *bound_entities,
                    ),
                    value=value,
                    evidence_ids=tuple(
                        f"program_{int(item['index'])}"
                        for item in chosen
                    ),
                    provenance="program_trace",
                    salience=0.9 - ordinal * 0.01,
                    mandatory=True,
                    required_literals=tuple(dict.fromkeys(value for value in required if value)),
                    trace_derived=True,
                )
            )
        return tuple(result)

    @staticmethod
    def _unit(
        vocabulary: ExplanationSemantics,
        language: str,
        **values: Any,
    ) -> ExplanationUnit:
        arguments = tuple(str(item) for item in values.get("arguments", ()))
        raw_value = values.get("value", {})
        allowed_entities = tuple(
            dict.fromkeys(
                (
                    *arguments,
                    *(
                        vocabulary.explanation_entity_label(item, language)
                        for item in arguments
                    ),
                )
            )
        )
        allowed_actions = _action_literals(
            raw_value,
            vocabulary=vocabulary,
            language=language,
        )
        allowed_numbers = _literal_numbers(raw_value)
        normalized_values = {
            **values,
            "allowed_entities": tuple(
                values.get("allowed_entities", allowed_entities)
            ),
            "allowed_actions": tuple(
                values.get("allowed_actions", allowed_actions)
            ),
            "allowed_numbers": tuple(
                values.get("allowed_numbers", allowed_numbers)
            ),
        }
        provisional = ExplanationUnit(reference_text="", **normalized_values)
        text = vocabulary.explanation_verbalize_unit(provisional.to_dict(), language).strip()
        required = tuple(
            dict.fromkeys(
                (
                    *tuple(str(item) for item in values.get("required_literals", ())),
                    *_numeric_literals(text),
                )
            )
        )
        return ExplanationUnit(
            **{
                **normalized_values,
                "required_literals": required,
                "reference_text": text,
            }
        )


def _trace_numeric_value(
    raw: Mapping[str, Any],
    meaning: Mapping[str, Any],
) -> float:
    observed = meaning.get("value", raw.get("observed_value", 0.0))
    try:
        return float(observed)
    except (TypeError, ValueError):
        return 0.0


def _trace_condition_value(
    raw: Mapping[str, Any],
    *,
    index: int,
    selected_action: str,
    vocabulary: ExplanationSemantics,
    language: str,
) -> dict[str, Any]:
    meaning = raw.get("observed_meaning", {})
    meaning = dict(meaning) if isinstance(meaning, Mapping) else {}
    constrained = _trace_alternative_action(
        raw,
        selected_action=selected_action,
    )
    return {
        "trace_type": str(raw.get("trace_type", "tree_branch")),
        "selected_action": selected_action,
        "selected_action_label": vocabulary.explanation_action_label(
            selected_action,
            language,
        ),
        "constrained_action": constrained,
        "constrained_action_label": (
            vocabulary.explanation_action_label(constrained, language)
            if constrained
            else ""
        ),
        "feature": raw.get("feature"),
        "observed_value": raw.get("observed_value"),
        "threshold": raw.get("threshold"),
        "result": raw.get("result"),
        "geometric_goal_progress": raw.get("geometric_goal_progress"),
        "active_reason_features": tuple(
            raw.get("active_reason_features", ())
        ),
        "bound_entities": dict(raw.get("bound_entities", {})),
        "observed_meaning": _typed_payload(meaning),
        "path_index": index,
    }


def _trace_alternative_action(
    raw: Mapping[str, Any],
    *,
    selected_action: str,
) -> str:
    explicit = str(raw.get("constrained_action", ""))
    if explicit:
        return explicit
    meaning = raw.get("observed_meaning", {})
    meaning = meaning if isinstance(meaning, Mapping) else {}
    observed_action = str(meaning.get("action", ""))
    return observed_action if observed_action != selected_action else ""


from .explanation_rendering import (
    ConversationalIRRenderer,
    ExplanationMessagePlanner,
    _proposal_differs_from_final,
)


def validate_document(
    document: ExplanationDocumentV3,
    explanation_ir: ExplanationIR,
) -> tuple[str, ...]:
    """Validate provenance and communication obligations, not wording."""

    units = {item.unit_id: item for item in explanation_ir.units}
    issues: list[str] = []
    if document.ir_hash != explanation_ir.ir_hash:
        issues.append("document IR hash does not match the explanation IR")
    if not document.sentences:
        issues.append("document contains no sentences")
        return tuple(issues)
    if document.sentences[0].role not in {
        SentenceRole.ANSWER,
        SentenceRole.COUNTERFACTUAL_RESULT,
    }:
        issues.append("the first sentence must directly answer the question")
    if len(document.sentences) > 3:
        issues.append("document exceeds three sentences")

    cited: list[str] = []
    for sentence in document.sentences:
        if not sentence.text.strip():
            issues.append(f"sentence {sentence.role.value} is empty")
        unknown = tuple(unit_id for unit_id in sentence.unit_ids if unit_id not in units)
        if unknown:
            issues.append(f"sentence cites unknown units {list(unknown)}")
        selected = tuple(units[unit_id] for unit_id in sentence.unit_ids if unit_id in units)
        cited.extend(unit.unit_id for unit in selected)
        expected_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for unit in selected
                for evidence_id in unit.evidence_ids
            )
        )
        if sentence.evidence_ids != expected_evidence:
            issues.append(
                f"sentence {sentence.role.value} evidence references do not match its units"
            )

    final_ids = {
        unit.unit_id
        for unit in explanation_ir.units
        if unit.layer == ExplanationLayer.FINAL_ACTION
    }
    if final_ids and not final_ids.intersection(cited):
        issues.append("final action was not displayed")
    proposal = tuple(
        unit for unit in explanation_ir.units if unit.layer == ExplanationLayer.POLICY_PROPOSAL
    )
    final = tuple(
        unit for unit in explanation_ir.units if unit.layer == ExplanationLayer.FINAL_ACTION
    )
    if _proposal_differs_from_final(proposal, final):
        required = {
            unit.unit_id
            for unit in explanation_ir.units
            if unit.layer in {
                ExplanationLayer.POLICY_PROPOSAL,
                ExplanationLayer.COORDINATION,
            }
        }
        missing = required - set(cited)
        if missing:
            issues.append(
                "changed proposal requires proposal and coordination units: "
                + ", ".join(sorted(missing))
            )
    trace_ids = {unit.unit_id for unit in explanation_ir.trace_units}
    if trace_ids and not trace_ids.intersection(cited):
        issues.append("trace condition did not display a proposal rationale")
    if not trace_ids and any(
        units[unit_id].trace_derived for unit_id in cited if unit_id in units
    ):
        issues.append("no-trace document cited program evidence")
    if any(
        unit.trace_derived and unit.unit_id in cited and not unit.evidence_ids
        for unit in explanation_ir.units
    ):
        issues.append("displayed trace unit has no program evidence")
    mandatory_ids = {
        unit.unit_id for unit in explanation_ir.units if unit.mandatory
    }
    missing_mandatory = mandatory_ids - set(cited)
    if missing_mandatory:
        issues.append(
            "mandatory units were not displayed: "
            + ", ".join(sorted(missing_mandatory))
        )
    if explanation_ir.query_kind == "action_why_not":
        contrasts = tuple(
            unit
            for unit in explanation_ir.units
            if unit.predicate == "action_not_selected"
        )
        if not contrasts:
            issues.append("action Why-not has no requested-action contrast unit")
        else:
            contrast = contrasts[0]
            desired_label = (
                str(contrast.value.get("desired_action_label", ""))
                if isinstance(contrast.value, Mapping)
                else ""
            )
            if contrast.unit_id not in cited:
                issues.append("action Why-not did not display its contrast unit")
            if desired_label and desired_label.casefold() not in document.text.casefold():
                issues.append(
                    "action Why-not did not name the requested alternative action"
                )

    lowered = document.text.casefold()
    banned = (
        "策略提议",
        "联合动作处理",
        "仲裁",
        "执行路径",
        "静态障碍属性",
        "证据 id",
        "execution trace",
        "program trace",
        "arbitration",
        "evidence id",
    )
    for term in banned:
        if term.casefold() in lowered:
            issues.append(f"display text contains implementation term {term!r}")
    position_requested = any(
        unit.layer == ExplanationLayer.TASK_GOAL
        and isinstance(unit.value, Mapping)
        and bool(unit.value.get("show_position", False))
        for unit in explanation_ir.units
    )
    if (
        explanation_ir.query_kind in {"action_explanation", "action_why_not"}
        and not position_requested
        and re.search(
        r"[（(]\s*-?\d+\s*[,，]\s*-?\d+\s*[)）]",
        document.text,
        )
    ):
        issues.append("ordinary action explanation exposes an unnecessary coordinate")
    return tuple(dict.fromkeys(issues))


def render_ir(
    explanation_ir: ExplanationIR,
    *,
    issues: Sequence[str] = (),
) -> ExplanationDocumentV3:
    """Compatibility entry point for the deterministic conversational renderer."""

    return ConversationalIRRenderer().render(explanation_ir, issues=issues)




def _selected_objectives(evidence: EvidenceBundle) -> tuple[dict[str, Any], ...]:
    contrast = evidence.direct_result.get(
        "counterfactual_contrast",
        evidence.policy_results.get("counterfactual_contrast", {}),
    )
    if isinstance(contrast, Mapping) and contrast:
        phased: list[dict[str, Any]] = []
        for phase in ("original", "edited"):
            side = contrast.get(phase, {})
            side = side if isinstance(side, Mapping) else {}
            context = side.get("objective_context", {})
            context = context if isinstance(context, Mapping) else {}
            raw = context.get("value", context)
            raw = raw if isinstance(raw, Mapping) else {}
            selected = raw.get("selected_objective", raw.get("objective"))
            if isinstance(selected, Mapping) and selected.get("id"):
                phased.append(
                    {
                        "id": str(selected.get("id", "")),
                        "target_position": selected.get("target_position"),
                        "phase": phase,
                        "evidence_id": context.get(
                            "evidence_id",
                            f"state::objective.{phase}",
                        ),
                    }
                )
            elif selected:
                phased.append(
                    {
                        "id": str(selected),
                        "target_position": raw.get("target_position"),
                        "phase": phase,
                        "evidence_id": context.get(
                            "evidence_id",
                            f"state::objective.{phase}",
                        ),
                    }
                )
        if phased:
            if (
                len(phased) == 2
                and phased[0].get("id") == phased[1].get("id")
                and phased[0].get("target_position")
                == phased[1].get("target_position")
            ):
                return (
                    {
                        "id": phased[0]["id"],
                        "target_position": phased[0].get(
                            "target_position"
                        ),
                        "phase": "unchanged",
                        "evidence_ids": tuple(
                            dict.fromkeys(
                                str(item.get("evidence_id", ""))
                                for item in phased
                                if str(item.get("evidence_id", ""))
                            )
                        ),
                    },
                )
            return tuple(phased)
    shared = evidence.policy_results.get(
        "shared_task_context",
        evidence.direct_result.get("shared_task_context", {}),
    )
    if isinstance(shared, Mapping):
        value = shared.get("value", shared)
        if isinstance(value, Mapping):
            selected = value.get("selected_objective", value.get("objective"))
            if isinstance(selected, Mapping):
                return ({
                    "id": str(selected.get("id", selected.get("objective", ""))),
                    "target_position": selected.get("target_position"),
                },)
            if selected:
                return ({
                    "id": str(selected),
                    "target_position": value.get("target_position"),
                },)
    for fact in evidence.state_facts:
        if str(fact.get("predicate", "")) not in {
            "objective_selection_reason",
            "selected_objective",
        }:
            continue
        value = fact.get("value", {})
        if isinstance(value, Mapping):
            selected = value.get("selected_objective", value.get("objective"))
            if isinstance(selected, Mapping):
                return ({
                    "id": str(selected.get("id", "")),
                    "target_position": selected.get("target_position"),
                },)
            if selected:
                return ({"id": str(selected), "target_position": value.get("target_position")},)
    return ()


def _question_targets_objective(plan: Any, target: str) -> bool:
    paths = (
        *tuple(getattr(plan, "target_variables", ())),
        *tuple(getattr(plan, "desired_outcomes", {}).keys()),
    )
    for raw in paths:
        path = str(raw)
        owner = path.split(".", 1)[0] if "." in path else target
        leaf = path.rsplit(".", 1)[-1].casefold()
        if owner == target and leaf in {
            "objective",
            "goal",
            "goal_kind",
            "task",
        }:
            return True
    return False


def _state_query_spec(
    evidence: EvidenceBundle,
    target: str,
) -> tuple[str, str] | None:
    vocabulary = evidence.policy_results.get(
        "question_vocabulary",
        evidence.direct_result.get("question_vocabulary", {}),
    )
    vocabulary = vocabulary if isinstance(vocabulary, Mapping) else {}
    variables = vocabulary.get("query_variables", {})
    variables = variables if isinstance(variables, Mapping) else {}
    for raw in evidence.query_plan.target_variables:
        path = str(raw)
        owner = path.split(".", 1)[0] if "." in path else target
        leaf = path.rsplit(".", 1)[-1]
        if owner != target:
            continue
        description = variables.get(leaf, {})
        description = description if isinstance(description, Mapping) else {}
        if str(description.get("kind", "")) != "state":
            continue
        return leaf, str(description.get("predicate", leaf))
    return None


def _state_query_fact(
    evidence: EvidenceBundle,
    *,
    target: str,
    predicate: str,
) -> Mapping[str, Any] | None:
    for fact in evidence.state_facts:
        if str(fact.get("predicate", "")) != predicate:
            continue
        arguments = tuple(str(value) for value in fact.get("arguments", ()))
        if arguments and target not in arguments:
            continue
        return fact
    return None


def _action_effect_fact(
    evidence: EvidenceBundle,
    target: str,
) -> Mapping[str, Any] | None:
    """Return a directly observed effect of the selected recorded action."""

    focus = next(
        (
            str(item).split(":", 1)[1]
            for item in evidence.query_plan.evidence_requirements
            if str(item).startswith("study_focus:")
        ),
        "action",
    )
    if focus in {"collaboration", "allocation"}:
        preferred = (
            "collaboration_context",
            "charging_outcome",
            "movement_outcome",
        )
    elif focus in {"energy", "charge_threshold"}:
        preferred = (
            "energy_decision_context",
            "charging_outcome",
            "movement_outcome",
        )
    elif focus == "collision":
        preferred = (
            "action_resolution_reason",
            "collaboration_context",
            "movement_outcome",
        )
    elif focus == "task":
        preferred = (
            "movement_outcome",
            "charging_outcome",
            "collaboration_context",
        )
    else:
        preferred = (
            "charger_queue_context",
            "charging_outcome",
            "movement_outcome",
            "observed_action_effect",
        )
    facts = tuple(evidence.state_facts)
    for predicate in preferred:
        for fact in facts:
            if str(fact.get("predicate", "")) != predicate:
                continue
            arguments = tuple(str(value) for value in fact.get("arguments", ()))
            if arguments and target not in arguments:
                continue
            if focus == "action":
                return fact
            focused_fact = dict(fact)
            raw_value = fact.get("value")
            if isinstance(raw_value, Mapping):
                focused_fact["value"] = {
                    **dict(raw_value),
                    "study_focus": focus,
                }
            return focused_fact
    fallback_predicates = (
        {
            "charger_queue_context",
            "charging_outcome",
            "movement_outcome",
            "observed_action_effect",
        }
        if focus == "action"
        else {
            "charger_queue_context",
            "charging_outcome",
            "movement_outcome",
            "energy_decision_context",
            "collaboration_context",
            "observed_action_effect",
        }
    )
    for fact in evidence.state_facts:
        if str(fact.get("predicate", "")) not in fallback_predicates:
            continue
        arguments = tuple(str(value) for value in fact.get("arguments", ()))
        if arguments and target not in arguments:
            continue
        if focus == "action":
            return fact
        focused_fact = dict(fact)
        raw_value = fact.get("value")
        if isinstance(raw_value, Mapping):
            focused_fact["value"] = {
                **dict(raw_value),
                "study_focus": focus,
            }
        return focused_fact
    return None


def _desired_objective_query(plan: Any, target: str) -> str:
    desired = getattr(plan, "desired_outcomes", {})
    if not isinstance(desired, Mapping):
        return ""
    for raw_key, raw_value in desired.items():
        path = str(raw_key)
        owner = path.split(".", 1)[0] if "." in path else target
        leaf = path.rsplit(".", 1)[-1].casefold()
        if owner == target and leaf in {
            "objective",
            "goal",
            "goal_kind",
            "task",
        }:
            return str(raw_value)
    return ""


def _desired_action_query(plan: Any, target: str) -> str:
    """Return the explicitly requested alternative action, if any.

    Question parsing owns synonym resolution.  The explanation layer only
    follows the structured desired outcome and never guesses an action from
    the raw question text.
    """

    desired = getattr(plan, "desired_outcomes", {})
    if not isinstance(desired, Mapping):
        return ""
    candidates: list[Any] = []
    for raw_key, raw_value in desired.items():
        path = str(raw_key)
        leaf = path.rsplit(".", 1)[-1].casefold()
        owner = path.split(".", 1)[0] if "." in path else target
        if owner == target and "action" in leaf:
            candidates.append(raw_value)
        elif path.casefold() == "action":
            candidates.append(raw_value)
    return next(
        (str(value).upper() for value in candidates if str(value).strip()),
        "",
    )


def _action_why_not_value(
    evidence: EvidenceBundle,
    *,
    target: str,
    desired_action: str,
    proposed_action: str,
    final_action: str,
    vocabulary: ExplanationSemantics,
    language: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Compile one mandatory, evidence-bound answer to an action Why-not.

    Priority follows the actual decision hierarchy: correct a false premise,
    then use execution coordination when the requested action was proposed,
    then use Actor-visible constraints/progress, and finally report the result
    of the paired recourse search.  An evidence gap remains explicit instead
    of being filled with a reason about a different action.
    """

    value: dict[str, Any] = {
        "desired_action": desired_action,
        "desired_action_label": vocabulary.explanation_action_label(
            desired_action,
            language,
        ),
        "proposed_action": proposed_action,
        "proposed_action_label": vocabulary.explanation_action_label(
            proposed_action,
            language,
        ),
        "final_action": final_action,
        "final_action_label": vocabulary.explanation_action_label(
            final_action,
            language,
        ),
    }
    if final_action == desired_action:
        return (
            {
                **value,
                "reason_kind": "already_executed",
                "reason_provenance": "joint_execution",
            },
            ("neural_policy",),
        )

    coordination = _coordination_fact(evidence, target)
    resolution = evidence.policy_results.get(
        "action_resolution",
        evidence.direct_result.get("action_resolution", {}),
    )
    resolution = dict(resolution) if isinstance(resolution, Mapping) else {}
    if proposed_action == desired_action and final_action != desired_action:
        return (
            {
                **value,
                "reason_kind": "coordination_changed_action",
                "reason_provenance": "joint_execution",
                "coordination": {
                    "proposed_action": proposed_action,
                    "proposed_action_label": value["proposed_action_label"],
                    "final_action": final_action,
                    "final_action_label": value["final_action_label"],
                    "resolution": resolution,
                    "typed_reason": coordination,
                },
            },
            (_coordination_evidence_id(evidence, target),),
        )

    reason = _select_desired_action_reason(
        evidence,
        desired_action=desired_action,
    )
    if reason is not None:
        evidence_id = str(
            reason.get("evidence_id")
            or f"actor_feature::{reason.get('feature', desired_action)}"
        )
        return (
            {
                **value,
                "reason_kind": "observable_action_condition",
                "reason_provenance": str(
                    reason.get("provenance", "actor_observation")
                ),
                "reason": dict(reason),
            },
            (evidence_id,),
        )

    recourse = evidence.why_not_recourse
    recourse = recourse if isinstance(recourse, Mapping) else {}
    selected = recourse.get("selected")
    selected = selected if isinstance(selected, Mapping) else None
    if bool(recourse.get("achieved", False)) and selected is not None:
        candidate_id = str(
            selected.get("candidate_id")
            or recourse.get("selected_candidate_id")
            or "selected"
        )
        return (
            {
                **value,
                "reason_kind": "counterfactual_recourse",
                "reason_provenance": "paired_simulation",
                "recourse": dict(selected),
            },
            (f"simulation::why_not::{candidate_id}",),
        )

    return (
        {
            **value,
            "reason_kind": "insufficient_evidence",
            "reason_provenance": "paired_simulation",
            "refusal_reason": str(
                recourse.get("refusal_reason", "")
                or "No verified evidence explained the requested alternative."
            ),
        },
        ("simulation::why_not_search",),
    )


def _select_desired_action_reason(
    evidence: EvidenceBundle,
    *,
    desired_action: str,
) -> Mapping[str, Any] | None:
    context = evidence.policy_results.get(
        "why_not_action_context",
        evidence.direct_result.get("why_not_action_context", ()),
    )
    if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
        return None
    ranked: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, raw in enumerate(context):
        if not isinstance(raw, Mapping):
            continue
        meaning = raw.get("observed_meaning", {})
        meaning = meaning if isinstance(meaning, Mapping) else {}
        feature = str(raw.get("feature", ""))
        action = str(
            meaning.get("action", raw.get("queried_action", ""))
        ).upper()
        if action != desired_action:
            continue
        role = str(meaning.get("explanation_role", ""))
        numeric = _as_float(meaning.get("value", raw.get("value")))
        if bool(meaning.get("relation_active", False)):
            priority = 0
        elif role == "action_feasibility" and not bool(
            meaning.get("feasible", numeric > 0.5)
        ):
            priority = 0
        elif feature.endswith(".legal") and numeric <= 0.5:
            priority = 0
        elif role == "action_objective_effect" and numeric <= 0.0:
            priority = 1
        elif role == "action_route_efficiency" and numeric > 0.0:
            priority = 2
        else:
            continue
        ranked.append((priority, index, raw))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _shared_objective_context(evidence: EvidenceBundle) -> dict[str, Any]:
    shared = evidence.policy_results.get(
        "shared_task_context",
        evidence.direct_result.get("shared_task_context", {}),
    )
    if isinstance(shared, Mapping):
        raw = shared.get("value", shared)
        if isinstance(raw, Mapping):
            return dict(raw)
    for fact in evidence.state_facts:
        if str(fact.get("predicate", "")) != "objective_selection_reason":
            continue
        raw = fact.get("value", {})
        if isinstance(raw, Mapping):
            return dict(raw)
    return {}


def _question_requests_position(plan: Any) -> bool:
    target_variables = tuple(
        str(value).casefold()
        for value in getattr(plan, "target_variables", ())
    )
    if any(
        token in variable
        for variable in target_variables
        for token in ("position", "location", "coordinate")
    ):
        return True
    text = str(getattr(plan, "raw_text", "")).casefold()
    return any(
        token in text
        for token in ("where", "position", "location", "coordinate", "哪里", "哪儿", "位置", "坐标")
    )


def _coordination_fact(evidence: EvidenceBundle, target: str) -> Mapping[str, Any]:
    for fact in evidence.state_facts:
        raw_value = fact.get("value", {})
        role = str(raw_value.get("explanation_role", "")) if isinstance(raw_value, Mapping) else ""
        predicate = str(fact.get("predicate", ""))
        if (
            predicate
            not in {
                "coordination_resolution",
                "coordination_yield",
                "action_resolution",
            }
            and role not in {"action_coordination_effect", "action_resolution_effect"}
        ):
            continue
        arguments = {str(value) for value in fact.get("arguments", ())}
        if arguments and target not in arguments:
            continue
        value = raw_value
        return dict(value) if isinstance(value, Mapping) else {"value": value}
    return {}


def _counterfactual_units(
    evidence: EvidenceBundle,
    *,
    vocabulary: ExplanationSemantics,
    language: str,
    target: str,
    unit_factory: Any,
) -> tuple[ExplanationUnit, ...]:
    contrast = evidence.direct_result.get(
        "counterfactual_contrast",
        evidence.policy_results.get("counterfactual_contrast", {}),
    )
    if not isinstance(contrast, Mapping) or not contrast:
        return ()
    original = contrast.get("original", {})
    edited = contrast.get("edited", {})
    original = original if isinstance(original, Mapping) else {}
    edited = edited if isinstance(edited, Mapping) else {}
    original_action = str(original.get("executed_action", ""))
    edited_action = str(edited.get("executed_action", ""))
    value = {
        "original_action": original_action,
        "original_action_label": vocabulary.explanation_action_label(original_action, language),
        "edited_action": edited_action,
        "edited_action_label": vocabulary.explanation_action_label(edited_action, language),
        "interventions": tuple(evidence.interventions),
        "constraint_changes": tuple(contrast.get("constraint_changes", ())),
        "objective_changed": bool(contrast.get("objective_changed", False)),
    }
    intervention_entities = tuple(
        dict.fromkeys(
            str(item.get("entity_id", ""))
            for item in evidence.interventions
            if isinstance(item, Mapping) and str(item.get("entity_id", ""))
        )
    )
    return (
        unit_factory(
            vocabulary,
            language,
            unit_id="counterfactual",
            layer=ExplanationLayer.COUNTERFACTUAL,
            predicate="counterfactual_action_change",
            arguments=(target, *intervention_entities),
            value=value,
            evidence_ids=("simulation::paired_contrast",),
            provenance="paired_simulation",
            salience=0.99,
            mandatory=True,
            required_literals=tuple(
                value
                for value in (
                    vocabulary.explanation_entity_label(target, language),
                    *(
                        vocabulary.explanation_entity_label(entity, language)
                        for entity in intervention_entities
                    ),
                    vocabulary.explanation_action_label(original_action, language),
                    vocabulary.explanation_action_label(edited_action, language),
                    *tuple(_literal_values(evidence.interventions)),
                )
                if value
            ),
        ),
    )


def _label_table(evidence: EvidenceBundle, key: str) -> Mapping[str, Any]:
    value = evidence.policy_results.get(key, evidence.direct_result.get(key, {}))
    return value if isinstance(value, Mapping) else {}


def _localized_label(table: Mapping[str, Any], key: str, language: str) -> str:
    raw = table.get(str(key), {})
    if isinstance(raw, Mapping):
        candidate = raw.get("zh" if language == "zh-CN" else "en")
        if candidate:
            return str(candidate)
    fallback = str(key).replace("_", " ")
    return fallback.casefold() if language != "zh-CN" and fallback.isupper() else fallback


def _objective_evidence_id(evidence: EvidenceBundle) -> str:
    shared = evidence.policy_results.get(
        "shared_task_context",
        evidence.direct_result.get("shared_task_context", {}),
    )
    if isinstance(shared, Mapping) and shared.get("evidence_id"):
        return str(shared["evidence_id"])
    return "state::objective"


def _coordination_evidence_id(evidence: EvidenceBundle, target: str) -> str:
    for index, fact in enumerate(evidence.state_facts):
        groups = {str(value) for value in fact.get("factor_groups", ())}
        value = fact.get("value", {})
        role = str(value.get("explanation_role", "")) if isinstance(value, Mapping) else ""
        if str(fact.get("predicate", "")) in {
            "coordination_resolution",
        } or "coordination" in groups or role in {
            "action_coordination_effect",
            "action_resolution_effect",
        }:
            return f"state::{fact.get('fact_id', f'coordination_{index}')}"
    return f"state::{target}.coordination"


def _related_entities(value: Mapping[str, Any]) -> tuple[str, ...]:
    serialized = value.get("typed_reason", {})
    relation = value.get("resolution", {})
    candidates: list[str] = []
    for source in (serialized, relation):
        if not isinstance(source, Mapping):
            continue
        nested = source.get("relation", {})
        mappings = (source, nested if isinstance(nested, Mapping) else {})
        for mapping in mappings:
            for key in (
                "passing_agent",
                "passing_agent_id",
                "beneficiary_agent",
                "winner_agent_id",
            ):
                if mapping.get(key):
                    candidates.append(str(mapping[key]))
            related = mapping.get("related_agents", ())
            if isinstance(related, Sequence) and not isinstance(related, (str, bytes)):
                candidates.extend(str(item) for item in related if str(item))
    return tuple(dict.fromkeys(candidates))


def _coordination_position(value: Mapping[str, Any]) -> str:
    for source in (value.get("typed_reason", {}), value.get("resolution", {})):
        if not isinstance(source, Mapping):
            continue
        nested = source.get("relation", {})
        for mapping in (source, nested if isinstance(nested, Mapping) else {}):
            for key in ("shared_progress_position", "conflict_position"):
                if mapping.get(key) is not None:
                    return _compact_value(mapping[key])
    return ""


def _literal_values(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            result.extend(_literal_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if value and all(isinstance(item, (int, float)) for item in value):
            result.append(_compact_value(value))
        else:
            for nested in value:
                result.extend(_literal_values(nested))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result.append(_compact_value(value))
    return tuple(dict.fromkeys(result))


def _literal_numbers(value: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            number
            for literal in _literal_values(value)
            for number in _numeric_literals(literal)
        )
    )


def _action_literals(
    value: Any,
    *,
    vocabulary: ExplanationSemantics,
    language: str,
) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if normalized.endswith("action_label") and isinstance(nested, str):
                result.append(nested)
            elif normalized.endswith("action") and isinstance(nested, str):
                result.append(
                    vocabulary.explanation_action_label(nested, language)
                )
            elif isinstance(nested, (Mapping, tuple, list)):
                result.extend(
                    _action_literals(
                        nested,
                        vocabulary=vocabulary,
                        language=language,
                    )
                )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            result.extend(
                _action_literals(
                    nested,
                    vocabulary=vocabulary,
                    language=language,
                )
            )
    return tuple(dict.fromkeys(item for item in result if item))


def _typed_payload(value: Any) -> Any:
    """Remove authored prose while retaining typed execution semantics."""

    if isinstance(value, Mapping):
        return {
            str(key): _typed_payload(nested)
            for key, nested in value.items()
            if str(key).lower() not in {
                "zh",
                "en",
                "fact_verbalizations",
                "value_verbalizations",
                "semantic_verbalizations",
                "effect_verbalizations",
            }
            and not str(key).lower().endswith("_text")
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_typed_payload(item) for item in value)
    return value


def _compact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "(" + ", ".join(_compact_value(item) for item in value) + ")"
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _numeric_literals(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", str(text)))


def _sentence_count(text: str) -> int:
    return len([item for item in re.split(r"[。！？.!?]+", text) if item.strip()])


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip("。！？.!?;； ")
