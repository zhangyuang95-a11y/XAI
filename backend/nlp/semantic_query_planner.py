"""One-call question understanding followed by deterministic QueryPlan compilation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import re
import time
from typing import Any, Mapping, Sequence

from .language_policy import normalize_supported_language
from .question_binding import QuestionBindingContext
from .schemas import (
    EntityQueryRole,
    PrimitiveOperation,
    QueryIntent,
    QueryPlan,
    QuestionIRV2,
    RelationalConstraint,
    SceneEditPlan,
    SemanticQueryIR,
)
from .tokenizer import StructuredTransformerBackend


_POLICY_INTENTS = {
    QueryIntent.EXPLANATORY,
    QueryIntent.WHY_NOT,
    QueryIntent.PREDICTIVE,
    QueryIntent.COUNTERFACTUAL,
    QueryIntent.COMPARATIVE,
    QueryIntent.DIAGNOSTIC,
    QueryIntent.MIXED,
}

class SemanticQueryCompiler:
    """Compile a valid semantic request into execution requirements."""

    def compile(self, ir: SemanticQueryIR, *, raw_text: str) -> QueryPlan:
        intent = ir.intent
        prediction_targets = ir.prediction_targets
        desired_outcomes = dict(ir.desired_outcomes)
        # A non-null desired outcome is the semantic distinction between
        # "why X" and "why not Y".  Small instruction models occasionally
        # label the former field as explanatory even after correctly extracting
        # Y; preserve the extracted alternative and enforce the IR invariant.
        if (
            intent == QueryIntent.EXPLANATORY
            and desired_outcomes
            and not ir.has_scene_edit
        ):
            intent = QueryIntent.WHY_NOT
        target_names = tuple(
            str(value).rsplit(".", 1)[-1].casefold()
            for value in ir.target_variables
        )
        desired_names = tuple(
            str(value).rsplit(".", 1)[-1].casefold()
            for value in desired_outcomes
        )
        action_focused = any(
            "action" in name for name in (*target_names, *desired_names)
        )
        objective_focused = any(
            name in {"objective", "goal", "goal_kind", "task"}
            for name in (*target_names, *desired_names)
        )
        direct_action_evidence = any(
            name in {"actor_observation", "neural_policy", "program_trace"}
            for name in target_names
        )
        # Older callers sometimes omitted the target variable for a plain
        # action explanation.  Preserve that compatibility only when the
        # request did not explicitly identify another semantic domain.
        if (
            not ir.target_variables
            and not desired_outcomes
            and intent in {
                QueryIntent.EXPLANATORY,
                QueryIntent.PREDICTIVE,
                QueryIntent.COMPARATIVE,
            }
        ):
            action_focused = True
        elif (
            intent == QueryIntent.EXPLANATORY
            and direct_action_evidence
            and not objective_focused
        ):
            action_focused = True

        entity_references = _ordered_unique(
            (
                *ir.referenced_entities,
                *(item.entity_id for item in ir.entity_roles),
                *(item.entity_id for item in ir.primitive_interventions),
                *(
                    entity
                    for relation in ir.relational_constraints
                    for entity in relation.subjects
                ),
            )
        )
        scene_edit_plan = (
            SceneEditPlan(
                source_frame=ir.frame_reference,
                entity_references=entity_references,
                primitive_interventions=ir.primitive_interventions,
                relational_constraints=ir.relational_constraints,
                preserved_variables=ir.preserved_variables,
                simulation_horizon=ir.horizon,
                confidence=0.0,
            )
            if ir.has_scene_edit
            else None
        )
        external_references = set(entity_references) - set(prediction_targets)
        explicit_causal_probe = bool(
            intent == QueryIntent.EXPLANATORY
            and action_focused
            and (external_references or ir.causal_variables)
        )
        requires_policy_query = bool(
            action_focused
            or ir.has_scene_edit
            or intent
            in {
                QueryIntent.PREDICTIVE,
                QueryIntent.COUNTERFACTUAL,
                QueryIntent.COMPARATIVE,
                QueryIntent.DIAGNOSTIC,
                QueryIntent.MIXED,
            }
        )
        requires_program_trace = action_focused
        requires_simulation = bool(
            ir.has_scene_edit
            or (intent == QueryIntent.WHY_NOT and action_focused)
            or intent == QueryIntent.COMPARATIVE
            or explicit_causal_probe
            or (intent in {QueryIntent.PREDICTIVE, QueryIntent.DIAGNOSTIC} and ir.horizon > 1)
        )
        requires_baseline_comparison = bool(
            requires_simulation
            and (
                ir.has_scene_edit
                or (intent == QueryIntent.WHY_NOT and action_focused)
                or intent == QueryIntent.COMPARATIVE
                or explicit_causal_probe
            )
        )
        evidence_requirements = ["state"]
        if requires_policy_query:
            evidence_requirements.extend(("actor_observation", "neural_policy"))
        if requires_program_trace:
            evidence_requirements.append("program_trace")
        if intent == QueryIntent.EXPLANATORY and action_focused:
            evidence_requirements.append("causal_analysis")
        if requires_simulation:
            evidence_requirements.append("simulation")

        plan = QueryPlan(
            raw_text=raw_text,
            intent=intent,
            frame_reference=ir.frame_reference,
            subjects=prediction_targets,
            referenced_entities=entity_references,
            entity_roles=ir.entity_roles,
            requires_scene_edit=ir.has_scene_edit,
            requires_policy_query=requires_policy_query,
            requires_program_trace=requires_program_trace,
            requires_simulation=requires_simulation,
            requires_baseline_comparison=requires_baseline_comparison,
            scene_edit_plan=scene_edit_plan,
            target_variables=ir.target_variables,
            horizon=ir.horizon,
            rollout_count=(max(2, ir.rollout_count) if action_focused or requires_simulation else ir.rollout_count),
            evidence_requirements=tuple(dict.fromkeys(evidence_requirements)),
            desired_outcomes=desired_outcomes,
            assumed_outcomes=dict(ir.assumed_outcomes),
            confidence=0.0,
            clarification_required=bool(ir.ambiguities),
            clarification_reason="; ".join(ir.ambiguities) if ir.ambiguities else None,
            unsupported_components=ir.unsupported_components,
            response_language=normalize_supported_language(ir.response_language, text=raw_text),
        )
        errors = plan.validate()
        if errors:
            raise ValueError(
                "Semantic query compiler produced an invalid QueryPlan: "
                + "; ".join(errors)
            )
        return plan


class SemanticTransformerQueryPlanner:
    """Decode QuestionIRV2 in exactly one model call.

    Language detection, environment binding checks, execution requirements, and
    clarification behavior are deterministic.  Invalid model output is never
    repaired by a second generation.
    """

    def __init__(
        self,
        backend: StructuredTransformerBackend,
        *,
        verify_response_language: bool = True,
        semantic_attempts: int = 1,
        cache_size: int = 512,
    ) -> None:
        del verify_response_language, semantic_attempts
        if backend is None:
            raise ValueError(
                "A configured Transformer backend is required for open-ended planning."
            )
        self.backend = backend
        self.compiler = SemanticQueryCompiler()
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, QueryPlan] = OrderedDict()
        self.last_question_ir: QuestionIRV2 | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def parse(
        self,
        text: str,
        *,
        selected_frame: int | None = None,
        environment_schema: Mapping[str, Any] | None = None,
        cache_context: Mapping[str, Any] | None = None,
    ) -> QueryPlan:
        question = text.strip()
        if not question:
            raise ValueError("The user question is empty.")
        schema = _compact_environment_schema(environment_schema or {})
        binding_context = QuestionBindingContext.from_environment(
            question,
            schema,
        )
        key = _question_cache_key(
            question,
            selected_frame,
            schema,
            model_id=_backend_identity(self.backend),
            cache_context=cache_context or {},
        )
        started = time.perf_counter()
        if key in self._cache:
            plan = self._cache.pop(key)
            self._cache[key] = plan
            self.last_diagnostics = {
                "schema_version": "question-ir.v2",
                "model_call_count": 0,
                "cache_hit": True,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "binding": {
                    "target_origin": "cached_bound_question_ir",
                    "model_target": plan.primary_prediction_target,
                    "resolved_target": plan.primary_prediction_target,
                    "focus_entity": binding_context.focus_entity,
                    "mentions": [
                        item.to_dict() for item in binding_context.mentions
                    ],
                },
            }
            return plan

        language = normalize_supported_language("auto", text=question)
        binding_audit: dict[str, Any] = {
            "focus_entity": binding_context.focus_entity,
            "mentions": [
                item.to_dict() for item in binding_context.mentions
            ],
        }
        generation_metrics: dict[str, Any] = {}
        payload: Mapping[str, Any] = {}
        try:
            payload = self.backend.generate_json(
                _question_ir_prompt(
                    question,
                    selected_frame=selected_frame,
                    environment_schema=schema,
                    binding_context=binding_context,
                ),
                schema_name="QuestionIRV2",
                max_new_tokens=192,
            )
            generation_metrics = dict(
                getattr(self.backend, "last_generation_metrics", {})
            )
            payload = _repair_literal_relation_payload(
                payload,
                question=question,
                environment_schema=schema,
                binding_context=binding_context,
            )
            payload = _repair_literal_property_edit_payload(
                payload,
                question=question,
                environment_schema=schema,
                binding_context=binding_context,
            )
            question_ir = QuestionIRV2.from_dict(
                payload,
                selected_frame=selected_frame,
            )
            question_ir, binding_audit = binding_context.bind(
                question_ir
            )
            question_ir = _normalize_question_ir(
                question_ir,
                question=question,
                environment_schema=schema,
            )
            validation = _validate_question_ir(
                question_ir,
                question=question,
                environment_schema=schema,
                strict_source_spans=(
                    "target_entity" in payload or "t" in payload
                ),
            )
            if validation:
                raise ValueError("; ".join(validation))
            semantic = question_ir.to_semantic_ir(response_language=language)
            self.last_question_ir = question_ir
            plan = self.compiler.compile(semantic, raw_text=question)
            error: str | None = None
        except Exception as exc:
            generation_metrics = dict(
                getattr(self.backend, "last_generation_metrics", {})
            )
            error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:320]}"
            fallback = _safe_default_question_ir(
                question,
                selected_frame=selected_frame,
                environment_schema=schema,
                binding_context=binding_context,
            )
            if fallback is not None:
                try:
                    fallback, binding_audit = binding_context.bind(fallback)
                    fallback = _normalize_question_ir(
                        fallback,
                        question=question,
                        environment_schema=schema,
                    )
                    fallback_issues = _validate_question_ir(
                        fallback,
                        question=question,
                        environment_schema=schema,
                        strict_source_spans=False,
                    )
                    if fallback_issues:
                        raise ValueError("; ".join(fallback_issues))
                    self.last_question_ir = fallback
                    semantic = fallback.to_semantic_ir(
                        response_language=language
                    )
                    plan = self.compiler.compile(semantic, raw_text=question)
                    binding_audit = {
                        **binding_audit,
                        "degraded_to_source_grounded_default": True,
                        "model_error": error,
                    }
                except Exception:
                    fallback = None
            if fallback is None:
                self.last_question_ir = None
                semantic = SemanticQueryIR(
                    intent=QueryIntent.MIXED,
                    frame_reference=selected_frame,
                    response_language=language,
                    ambiguities=(
                        _clarification_message(language),
                    ),
                )
                plan = self.compiler.compile(semantic, raw_text=question)

        self.last_diagnostics = {
            "schema_version": "question-ir.v2",
            "model_call_count": 1,
            "cache_hit": False,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": error,
            "question_ir": self.last_question_ir.to_dict() if self.last_question_ir else None,
            "binding": binding_audit,
            "input_tokens": int(generation_metrics.get("input_tokens", 0)),
            "output_tokens": int(generation_metrics.get("output_tokens", 0)),
        }
        if not plan.clarification_required and self.cache_size:
            self._cache[key] = plan
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return plan


# Public compatibility name used throughout the project.
TransformerQueryPlanner = SemanticTransformerQueryPlanner


def _question_ir_prompt(
    question: str,
    *,
    selected_frame: int | None,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> str:
    return (
        "Bind the user's meaning to the environment ontology in one pass. "
        "t is the entity whose answer is requested; an entity changed inside "
        "an explicit hypothetical belongs in x/r, not t. Do not emit roles, "
        "language, confidence, or execution flags. Use move_entity, "
        "set_battery, or set_direction only for explicit scene changes. "
        "A named alternative in a why-not question belongs in w, never x/r.\n"
        "A deictic action such as 'this direction' / ‘这个方向’ is read from "
        "the selected snapshot and has w=null.\n"
        "Example shape for an explicit edit: if entity_b's battery becomes "
        "10 and the user asks what entity_a will do, emit counterfactual with "
        "t=entity_a,q=next_action,w=null and one set_battery edit for entity_b.\n"
        f"Selected frame: {selected_frame!r}\n"
        "Deterministic source bindings: "
        + json.dumps(
            binding_context.prompt_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        "Literal ontology value anchors: "
        + json.dumps(
            _ontology_value_mentions(question, environment_schema),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + ". A query_variable anchor identifies q; objective/action anchors "
        "may identify w. Anchors are not scene edits.\n"
        "Environment schema: "
        + json.dumps(environment_schema, ensure_ascii=False, separators=(",", ":"))
        + "\n<user_question>"
        + question
        + "</user_question>\n"
        "Final check: preserve every explicit 'not X' / ‘不去X’ in w. If X "
        "matches an objective alias, emit i=why_not,q=objective,w=<objective "
        "ID>; if X matches an action alias, emit i=why_not,q=observed_action,"
        "w=<action ID>."
    )


def _safe_default_question_ir(
    question: str,
    *,
    selected_frame: int | None,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> QuestionIRV2 | None:
    """Recover an ordinary focused query after structurally invalid output.

    This is not a second language-model pass and it never invents a scene
    modification.  It only uses the selected entity plus literal ontology
    anchors already present in the source.  A complete scalar edit can be
    recovered when the entity, adapter-declared property, assignment marker,
    and typed value are all written literally in the question.
    """

    if _contains_explicit_scene_change(question):
        edit = _literal_typed_property_edit(
            question,
            environment_schema=environment_schema,
            binding_context=binding_context,
        )
        if edit is None:
            return None
        return QuestionIRV2.from_dict(
            {
                "intent": QueryIntent.COUNTERFACTUAL.value,
                "target_entity": edit["entity_id"],
                "primitive_interventions": [edit],
                "relational_constraints": [],
                "target_variables": [
                    f"{edit['entity_id']}.next_action"
                ],
                "desired_outcomes": {},
                "ambiguities": [],
            },
            selected_frame=selected_frame,
        )
    target = binding_context.focus_entity
    explicit = binding_context.explicitly_mentioned_entities
    if len(explicit) == 1:
        target = explicit[0]
    if not target:
        return None

    anchors = _ontology_value_mentions(question, environment_schema)
    variable_anchors = [
        item for item in anchors if item.get("kind") == "query_variable"
    ]
    variable = (
        str(variable_anchors[0]["id"])
        if len(variable_anchors) == 1
        else "observed_action"
    )
    desired_anchor = (
        _select_negated_value_anchor(question, anchors)
        if _is_why_not_question(question)
        else None
    )
    desired_value = (
        str(desired_anchor["id"])
        if desired_anchor is not None
        else None
    )
    if desired_anchor is not None:
        variable = (
            "objective"
            if desired_anchor.get("kind") == "objective"
            else "observed_action"
        )
        intent = QueryIntent.WHY_NOT.value
    elif _looks_predictive(question):
        variable = "next_action" if not variable_anchors else variable
        intent = QueryIntent.PREDICTIVE.value
    elif variable not in {"observed_action", "last_action", "next_action"}:
        intent = QueryIntent.FACTUAL.value
    else:
        intent = QueryIntent.EXPLANATORY.value

    return QuestionIRV2.from_dict(
        {
            "i": intent,
            "t": target,
            "q": variable,
            "w": desired_value,
            "x": [],
            "r": [],
            "a": [],
        },
        selected_frame=selected_frame,
    )


def _repair_literal_relation_payload(
    payload: Mapping[str, Any],
    *,
    question: str,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> Mapping[str, Any]:
    """Canonicalize a relation that the model placed in q/w or fake edits.

    Relation aliases and entity identities come entirely from the environment
    ontology and the literal question.  This lets us discard invented
    coordinates or unsupported operations such as ``set_position`` without
    changing the user's requested relation.
    """

    anchors = [
        item
        for item in _ontology_value_mentions(question, environment_schema)
        if item.get("kind") == "relation"
    ]
    subjects = binding_context.explicitly_mentioned_entities
    if (
        len(anchors) != 1
        or len(subjects) < 2
        or not _contains_explicit_scene_change(question)
    ):
        return payload
    anchor = anchors[0]
    relation = {
        "rel": str(anchor["id"]),
        "s": list(subjects),
        "val": None,
        "span": str(anchor["span"]),
    }
    repaired = dict(payload)
    if "i" in repaired:
        repaired.update(
            {
                "i": QueryIntent.COUNTERFACTUAL.value,
                "q": "next_action",
                "w": None,
                "x": [],
                "r": [relation],
            }
        )
    else:
        repaired.update(
            {
                "intent": QueryIntent.COUNTERFACTUAL.value,
                "primitive_interventions": [],
                "relational_constraints": [
                    {
                        "relation": relation["rel"],
                        "subjects": relation["s"],
                        "value": None,
                        "source_span": relation["span"],
                    }
                ],
                "target_variables": [
                    f"{str(repaired.get('target_entity') or '')}.next_action"
                ],
                "desired_outcomes": {},
            }
        )
    return repaired


def _repair_literal_property_edit_payload(
    payload: Mapping[str, Any],
    *,
    question: str,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> Mapping[str, Any]:
    """Ground one explicit scalar assignment directly in the source text.

    Small instruction models sometimes normalize ``100%`` to ``1.0`` or omit
    the edit object even though they understood the hypothetical.  The value
    that reaches simulation must instead follow the environment property's
    declared type and the user's literal.  Recovery is deliberately narrow:
    one bound entity, one declared scalar property, one value, and an explicit
    assignment are all required.
    """

    edit = _literal_typed_property_edit(
        question,
        environment_schema=environment_schema,
        binding_context=binding_context,
    )
    if edit is None:
        return payload

    repaired = dict(payload)
    if "i" in repaired or "t" in repaired or "x" in repaired:
        repaired.update(
            {
                "i": QueryIntent.COUNTERFACTUAL.value,
                "t": edit["entity_id"],
                "q": "next_action",
                "w": None,
                "x": [
                    {
                        "op": edit["operation"],
                        "e": edit["entity_id"],
                        "p": edit["property_name"],
                        "val": edit["value"],
                        "span": edit["source_span"],
                        "metadata": edit["metadata"],
                    }
                ],
                "r": [],
                "a": [],
                "b": [],
            }
        )
    else:
        repaired.update(
            {
                "intent": QueryIntent.COUNTERFACTUAL.value,
                "target_entity": edit["entity_id"],
                "referenced_entities": [edit["entity_id"]],
                "entity_roles": [],
                "primitive_interventions": [edit],
                "relational_constraints": [],
                "target_variables": [
                    f"{edit['entity_id']}.next_action"
                ],
                "desired_outcomes": {},
                "ambiguities": [],
            }
        )
    return repaired


def _literal_typed_property_edit(
    question: str,
    *,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> dict[str, Any] | None:
    """Recover one complete typed edit declared by the environment adapter."""

    scalar = _literal_scalar_property_edit(
        question,
        environment_schema=environment_schema,
        binding_context=binding_context,
    )
    if scalar is not None:
        return scalar
    return _literal_coordinate_property_edit(
        question,
        environment_schema=environment_schema,
        binding_context=binding_context,
    )


def _literal_scalar_property_edit(
    question: str,
    *,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> dict[str, Any] | None:
    """Extract a fully literal, adapter-declared scalar scene edit.

    The function is domain-neutral: property names, aliases, types, and bounds
    come from ``EnvironmentAdapter.entity_schema``.  It does not infer an edit
    from a property mention alone and therefore cannot turn an observational
    question into a counterfactual.
    """

    if not _contains_explicit_scene_change(question):
        return None
    explicit = binding_context.explicitly_mentioned_entities
    if len(explicit) == 1:
        entity_id = explicit[0]
    elif not explicit and binding_context.focus_entity:
        entity_id = binding_context.focus_entity
    else:
        return None

    property_matches = _literal_property_mentions(
        question,
        environment_schema=environment_schema,
        entity_id=entity_id,
    )
    property_ids = tuple(dict.fromkeys(item[0] for item in property_matches))
    if len(property_ids) != 1:
        return None
    property_name = property_ids[0]
    property_match = max(
        (item for item in property_matches if item[0] == property_name),
        key=lambda item: len(item[2]),
    )
    _, specification, property_span, property_start, property_end = property_match
    property_type = str(specification.get("type", "")).casefold()
    if property_type not in {"percentage", "number", "float", "integer"}:
        return None

    clause_start, clause_end = _clause_bounds(
        question,
        property_start,
        property_end,
    )
    numeric_matches = []
    for match in re.finditer(
        r"(?<![0-9.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:[%％])?",
        question[clause_start:clause_end],
    ):
        start = clause_start + match.start()
        end = clause_start + match.end()
        if any(
            start < mention.end and end > mention.start
            for mention in binding_context.mentions
        ):
            continue
        numeric_matches.append((match.group(0), start, end))
    if len(numeric_matches) != 1:
        return None
    raw_value, value_start, value_end = numeric_matches[0]

    clause = question[clause_start:clause_end].casefold()
    if any(
        marker in clause
        for marker in (
            "低于",
            "高于",
            "少于",
            "多于",
            "超过",
            "不到",
            "至少",
            "至多",
            "below",
            "above",
            "less than",
            "more than",
            "at least",
            "at most",
        )
    ):
        return None
    if not any(
        marker in clause
        for marker in (
            "设为",
            "变成",
            "改成",
            "改为",
            "调到",
            "升到",
            "降到",
            "达到",
            "保持在",
            "有",
            "为",
            "是",
            "set to",
            "change to",
            "changed to",
            "become",
            "becomes",
            "became",
            " had ",
            " has ",
            " is ",
            " was ",
            " were ",
        )
    ):
        return None

    numeric_text = raw_value.rstrip("%％").strip()
    try:
        value = float(numeric_text)
    except ValueError:
        return None
    minimum = specification.get("minimum", specification.get("min"))
    maximum = specification.get("maximum", specification.get("max"))
    if property_type == "percentage":
        minimum = 0.0 if minimum is None else minimum
        maximum = 100.0 if maximum is None else maximum
    try:
        if minimum is not None and value < float(minimum):
            return None
        if maximum is not None and value > float(maximum):
            return None
    except (TypeError, ValueError):
        return None
    if property_type == "integer":
        if not value.is_integer():
            return None
        typed_value: int | float = int(value)
    else:
        typed_value = value

    mention_spans = [
        item
        for item in binding_context.mentions
        if item.entity_id == entity_id
    ]
    source_start = min(
        (
            mention_spans[0].start if mention_spans else property_start,
            property_start,
            value_start,
        )
    )
    source_end = max(
        (
            mention_spans[0].end if mention_spans else property_end,
            property_end,
            value_end,
        )
    )
    return {
        "operation": PrimitiveOperation.SET_ATTRIBUTE.value,
        "entity_id": entity_id,
        "property_name": property_name,
        "value": typed_value,
        "source_span": question[source_start:source_end],
        "metadata": {
            "binding": "literal_typed_property",
            "property_alias": property_span,
            "property_type": property_type,
        },
    }


def _literal_coordinate_property_edit(
    question: str,
    *,
    environment_schema: Mapping[str, Any],
    binding_context: QuestionBindingContext,
) -> dict[str, Any] | None:
    """Bind an explicit entity relocation to an adapter-declared position.

    Coordinates alone are not enough.  The question must also be hypothetical,
    identify exactly one entity (or use the selected entity), and contain a
    locative assignment.  The adapter marks which coordinate property denotes
    the entity's actual location, avoiding warehouse-specific property names in
    the core planner.
    """

    if not _contains_explicit_scene_change(question):
        return None
    explicit = binding_context.explicitly_mentioned_entities
    if len(explicit) == 1:
        entity_id = explicit[0]
    elif not explicit and binding_context.focus_entity:
        entity_id = binding_context.focus_entity
    else:
        return None

    coordinate_matches = tuple(
        re.finditer(
            r"[\(（]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*[,，]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*[\)）]",
            question,
        )
    )
    if len(coordinate_matches) != 1:
        return None
    coordinate_match = coordinate_matches[0]

    specifications = _entity_property_specifications(
        environment_schema,
        entity_id=entity_id,
    )
    mentioned = [
        item
        for item in _literal_property_mentions(
            question,
            environment_schema=environment_schema,
            entity_id=entity_id,
        )
        if str(item[1].get("type", "")).casefold() == "coordinate"
        and item[1].get("editable", True) is not False
    ]
    mentioned_ids = tuple(dict.fromkeys(item[0] for item in mentioned))
    if len(mentioned_ids) == 1:
        property_name = mentioned_ids[0]
        specification = specifications[property_name]
    else:
        location_properties = [
            (name, specification)
            for name, specification in specifications.items()
            if str(specification.get("type", "")).casefold() == "coordinate"
            and specification.get("editable", True) is not False
            and str(specification.get("semantic_role", "")).casefold()
            == "entity_position"
        ]
        if len(location_properties) != 1:
            return None
        property_name, specification = location_properties[0]

    entity_mentions = [
        item
        for item in binding_context.mentions
        if item.entity_id == entity_id
    ]
    entity_end = entity_mentions[0].end if entity_mentions else 0
    before_coordinate = question[entity_end : coordinate_match.start()].casefold()
    full_clause_start, full_clause_end = _clause_bounds(
        question,
        coordinate_match.start(),
        coordinate_match.end(),
    )
    clause = question[full_clause_start:full_clause_end].casefold()
    direct_location = re.search(
        r"(?:在|再|位于|移到|移动到|放到|改到|设在|at|to|in|on)\s*$",
        before_coordinate,
    )
    property_assignment = any(
        marker in clause
        for marker in (
            "位置为",
            "位置是",
            "坐标为",
            "坐标是",
            "position is",
            "location is",
            "set to",
            "move to",
        )
    )
    if direct_location is None and not property_assignment:
        return None

    values: list[int | float] = []
    for raw in coordinate_match.groups():
        value = float(raw)
        values.append(int(value) if value.is_integer() else value)
    source_start = min(
        entity_mentions[0].start if entity_mentions else coordinate_match.start(),
        coordinate_match.start(),
    )
    operation = str(
        specification.get(
            "edit_operation",
            PrimitiveOperation.SET_ATTRIBUTE.value,
        )
    ).casefold()
    try:
        operation = PrimitiveOperation(operation).value
    except ValueError:
        return None
    return {
        "operation": operation,
        "entity_id": entity_id,
        "property_name": property_name,
        "value": values,
        "source_span": question[source_start : coordinate_match.end()],
        "metadata": {
            "binding": "literal_typed_property",
            "property_type": "coordinate",
        },
    }


def _literal_property_mentions(
    question: str,
    *,
    environment_schema: Mapping[str, Any],
    entity_id: str,
) -> tuple[tuple[str, Mapping[str, Any], str, int, int], ...]:
    specifications = _entity_property_specifications(
        environment_schema,
        entity_id=entity_id,
    )
    folded = question.casefold()
    matches: list[tuple[str, Mapping[str, Any], str, int, int]] = []
    for property_name, specification in specifications.items():
        aliases: list[str] = [str(property_name)]
        for key in ("texts", "aliases"):
            values = specification.get(key, ())
            if isinstance(values, Sequence) and not isinstance(
                values,
                (str, bytes),
            ):
                aliases.extend(str(item) for item in values)
        aliases.extend(
            str(specification.get(key, ""))
            for key in ("zh", "en")
        )
        for alias in tuple(dict.fromkeys(item for item in aliases if item)):
            start = folded.find(alias.casefold())
            if start >= 0:
                matches.append(
                    (
                        str(property_name),
                        specification,
                        question[start : start + len(alias)],
                        start,
                        start + len(alias),
                    )
                )
    return tuple(matches)


def _entity_property_specifications(
    environment_schema: Mapping[str, Any],
    *,
    entity_id: str,
) -> dict[str, Mapping[str, Any]]:
    entities = environment_schema.get("entities", {})
    result: dict[str, Mapping[str, Any]] = {}

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        ids = value.get("ids", ())
        raw_id = str(value.get("id", "")).strip()
        belongs = (
            isinstance(ids, Sequence)
            and not isinstance(ids, (str, bytes))
            and entity_id in {str(item) for item in ids}
        ) or raw_id == entity_id
        properties = value.get("properties", {})
        if belongs and isinstance(properties, Mapping):
            for property_name, raw_specification in properties.items():
                result[str(property_name)] = (
                    raw_specification
                    if isinstance(raw_specification, Mapping)
                    else {"type": raw_specification}
                )
        for nested in value.values():
            if isinstance(nested, Mapping):
                visit(nested)

    visit(entities)
    return result


def _clause_bounds(question: str, start: int, end: int) -> tuple[int, int]:
    separators = ",，。.!！?？;；\n"
    clause_start = max((question.rfind(char, 0, start) for char in separators), default=-1) + 1
    endings = [
        index
        for char in separators
        if (index := question.find(char, end)) >= 0
    ]
    clause_end = min(endings) if endings else len(question)
    return clause_start, clause_end


def _contains_explicit_scene_change(question: str) -> bool:
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "if ",
            "suppose",
            "assuming",
            "what if",
            "set ",
            "change ",
            "relocate ",
            "如果",
            "假如",
            "假设",
            "要是",
            "设为",
            "变成",
            "改成",
            "移到",
            "移动到",
        )
    )


def _looks_predictive(question: str) -> bool:
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "next",
            "will ",
            "would ",
            "接下来",
            "下一步",
            "之后会",
            "然后会",
        )
    )


def _clarification_message(language: str) -> str:
    if language == "zh-CN":
        return (
            "我能识别当前选中的对象，但这个问题包含无法安全还原的场景修改。"
            "请明确说明要改变哪个对象、改变什么状态。"
        )
    return (
        "I can identify the selected entity, but the requested scene change "
        "cannot be reconstructed safely. Please state which entity and state "
        "should change."
    )


def _compact_environment_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only identifiers and type constraints needed for safe binding."""

    actions = schema.get("actions", ())
    entities = schema.get("entities", {})
    return {
        "actions": tuple(str(value) for value in actions) if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) else (),
        "entities": _compact_entity_schema(entities),
        "query_variables": _compact_query_vocabulary(
            schema.get("query_variables", {})
        ),
        "objectives": _compact_query_vocabulary(
            schema.get("objectives", {})
        ),
        "action_values": _compact_query_vocabulary(
            schema.get("action_values", {})
        ),
        "relations": _compact_query_vocabulary(
            schema.get("relations", {})
        ),
        "focus_entity": str(schema.get("focus_entity", "")).strip() or None,
    }


def _compact_query_vocabulary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact_query_vocabulary(nested)
            for key, nested in value.items()
            if str(key) in {"kind", "predicate", "aliases", "zh", "en"}
            or isinstance(nested, (Mapping, Sequence))
            and not isinstance(nested, (str, bytes))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return value


def _ontology_value_mentions(
    question: str,
    schema: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Link literal value aliases before generation without inferring intent."""

    result: list[dict[str, str]] = []
    for kind, vocabulary_key in (
        ("query_variable", "query_variables"),
        ("objective", "objectives"),
        ("action", "action_values"),
        ("relation", "relations"),
    ):
        vocabulary = schema.get(vocabulary_key, {})
        if not isinstance(vocabulary, Mapping):
            continue
        for canonical_id, description in vocabulary.items():
            description = description if isinstance(description, Mapping) else {}
            aliases = (
                str(canonical_id),
                str(description.get("zh", "")),
                str(description.get("en", "")),
                *tuple(description.get("aliases", ())),
            )
            folded_question = question.casefold()
            matches = [
                alias
                for alias in aliases
                if alias and alias.casefold() in folded_question
            ]
            if not matches:
                continue
            span = max(matches, key=len)
            start = folded_question.find(span.casefold())
            result.append(
                {
                    "kind": kind,
                    "id": str(canonical_id),
                    "span": span,
                    "start": start,
                    "end": start + len(span),
                }
            )
    return tuple(
        dict(item)
        for item in {
            (
                item["kind"],
                item["id"],
                item["span"],
                item["start"],
            ): item
            for item in result
        }.values()
    )


def _compact_entity_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if str(key) == "references" and isinstance(nested, Mapping):
                result["references"] = {
                    str(entity_id): tuple(str(alias) for alias in aliases)
                    for entity_id, aliases in nested.items()
                    if isinstance(aliases, Sequence)
                    and not isinstance(aliases, (str, bytes))
                }
            elif str(key) in {
                "ids",
                "properties",
                "type",
                "aliases",
                "texts",
                "zh",
                "en",
                "minimum",
                "maximum",
                "min",
                "max",
                "editable",
                "semantic_role",
                "edit_operation",
            }:
                result[str(key)] = _compact_entity_schema(nested)
            elif isinstance(nested, Mapping):
                compact = _compact_entity_schema(nested)
                if compact:
                    result[str(key)] = compact
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_compact_entity_schema(item) for item in value)
    return value


def _normalize_question_ir(
    question_ir: QuestionIRV2,
    *,
    question: str,
    environment_schema: Mapping[str, Any],
) -> QuestionIRV2:
    """Enforce ontology and source-grounding invariants after one model call.

    The Transformer decides the semantic domain.  This pass only canonicalizes
    adapter aliases, removes unmentioned desired values, and restores a
    why-not intent when the model extracted both a negated alternative and its
    ontology value.  It never invents an environment value or scene edit.
    """

    variables = environment_schema.get("query_variables", {})
    variables = variables if isinstance(variables, Mapping) else {}
    canonical_variables = {
        _semantic_token(str(canonical)): str(canonical)
        for canonical in variables
    }
    for canonical, description in variables.items():
        description = description if isinstance(description, Mapping) else {}
        for alias in description.get("aliases", ()):
            canonical_variables[_semantic_token(str(alias))] = str(canonical)

    normalized_paths: list[str] = []
    for raw in question_ir.target_variables:
        path = str(raw)
        owner = path.split(".", 1)[0] if "." in path else question_ir.target_entity
        leaf = path.rsplit(".", 1)[-1]
        canonical = canonical_variables.get(_semantic_token(leaf), leaf)
        normalized_paths.append(f"{owner}.{canonical}")

    anchors = _ontology_value_mentions(question, environment_schema)
    query_variable_anchors = [
        item for item in anchors if item.get("kind") == "query_variable"
    ]
    model_action_variable = any(
        "action" in str(path).rsplit(".", 1)[-1].casefold()
        for path in normalized_paths
    )
    if (
        not query_variable_anchors
        and not model_action_variable
        and not question_ir.desired_outcomes
        and not question_ir.primitive_interventions
        and not question_ir.relational_constraints
        and question_ir.intent
        in {
            QueryIntent.EXPLANATORY,
            QueryIntent.WHY_NOT,
            QueryIntent.PREDICTIVE,
            QueryIntent.COMPARATIVE,
            QueryIntent.DIAGNOSTIC,
            QueryIntent.MIXED,
        }
    ):
        # In this interface an unqualified "why?" asks about the selected
        # entity's visible decision.  A model-selected objective is not
        # source-grounded merely because objective IDs appeared in the prompt.
        # Explicit state/goal wording still wins through the anchors above.
        default_leaf = (
            "next_action"
            if question_ir.intent == QueryIntent.PREDICTIVE
            else "observed_action"
        )
        normalized_paths = [
            f"{question_ir.target_entity}.{default_leaf}"
        ]
    anchored_values = {
        (str(item["kind"]), str(item["id"]))
        for item in anchors
        if item.get("kind") in {"objective", "action"}
    }
    relation_constraints = question_ir.relational_constraints
    relation_anchors = [
        item for item in anchors if item.get("kind") == "relation"
    ]
    recovered_relation = False
    if (
        question_ir.intent == QueryIntent.COUNTERFACTUAL
        and not question_ir.primitive_interventions
        and not relation_constraints
        and len(relation_anchors) == 1
        and len(question_ir.referenced_entities) >= 2
    ):
        relation_anchor = relation_anchors[0]
        relation_constraints = (
            RelationalConstraint.from_dict(
                {
                    "relation": str(relation_anchor["id"]),
                    "subjects": question_ir.referenced_entities,
                    "source_span": str(relation_anchor["span"]),
                }
            ),
        )
        normalized_paths = [
            f"{question_ir.target_entity}.next_action"
        ]
        recovered_relation = True
    value_aliases: dict[tuple[str, str], str] = {}
    for kind, vocabulary_key in (
        ("objective", "objectives"),
        ("action", "action_values"),
    ):
        vocabulary = environment_schema.get(vocabulary_key, {})
        vocabulary = vocabulary if isinstance(vocabulary, Mapping) else {}
        for canonical, description in vocabulary.items():
            description = description if isinstance(description, Mapping) else {}
            values = (
                str(canonical),
                str(description.get("zh", "")),
                str(description.get("en", "")),
                *tuple(description.get("aliases", ())),
            )
            for alias in values:
                if alias:
                    value_aliases[(kind, _semantic_token(str(alias)))] = str(canonical)

    asks_why_not = _is_why_not_question(question)
    desired: dict[str, Any] = {}
    for raw_key, raw_value in question_ir.desired_outcomes.items():
        if recovered_relation:
            break
        path = str(raw_key)
        owner = path.split(".", 1)[0] if "." in path else question_ir.target_entity
        leaf = path.rsplit(".", 1)[-1]
        canonical_leaf = canonical_variables.get(_semantic_token(leaf), leaf)
        description = variables.get(canonical_leaf, {})
        description = description if isinstance(description, Mapping) else {}
        kind = str(description.get("kind", ""))
        canonical_value: Any = (
            value_aliases.get(
                (kind, _semantic_token(str(raw_value))),
                str(raw_value),
            )
            if kind in {"objective", "action"}
            else raw_value
        )
        # Desired alternatives require a literal source anchor. This prevents
        # the model from answering a different goal merely because that goal
        # appeared in the environment ontology.
        if kind in {"objective", "action"}:
            if (kind, canonical_value) not in anchored_values:
                continue
            # An action named in a positive "why X" question describes the
            # observed behavior, not a missing alternative.  Treating every
            # model-populated w value as why-not changed "why move up?" into
            # "why not move up?".  Literal negation/contrast in the original
            # question is authoritative here.
            if (
                not asks_why_not
                and not question_ir.primitive_interventions
                and not question_ir.relational_constraints
            ):
                continue
        desired[f"{owner}.{canonical_leaf}"] = canonical_value

    if asks_why_not:
        anchor = _select_negated_value_anchor(question, anchors)
        if anchor is not None:
            kind = str(anchor["kind"])
            compatible = [
                str(path)
                for path in normalized_paths
                if str(
                    (
                        variables.get(str(path).rsplit(".", 1)[-1], {})
                        if isinstance(
                            variables.get(str(path).rsplit(".", 1)[-1], {}),
                            Mapping,
                        )
                        else {}
                    ).get("kind", "")
                )
                == kind
            ]
            leaf = "objective" if kind == "objective" else "observed_action"
            target_path = (
                compatible[0]
                if compatible
                else f"{question_ir.target_entity}.{leaf}"
            )
            normalized_paths = [target_path]
            desired = {target_path: str(anchor["id"])}
        elif not desired:
            # "Why doesn't it move?" has no single ontology action to test.
            # It is still a valid request for the current action's reasons;
            # only a concrete alternative should trigger counterfactual
            # recourse simulation.
            fallback_path = f"{question_ir.target_entity}.observed_action"
            normalized_paths = [fallback_path]

    intent = question_ir.intent
    if (
        desired
        and asks_why_not
        and not question_ir.primitive_interventions
        and not question_ir.relational_constraints
    ):
        intent = QueryIntent.WHY_NOT
    elif (
        intent == QueryIntent.WHY_NOT
        and not desired
        and not question_ir.primitive_interventions
        and not question_ir.relational_constraints
    ):
        intent = QueryIntent.EXPLANATORY
    return replace(
        question_ir,
        intent=intent,
        relational_constraints=relation_constraints,
        target_variables=tuple(dict.fromkeys(normalized_paths)),
        desired_outcomes=desired,
    )


def _semantic_token(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if character.isalnum()
    )


def _is_why_not_question(question: str) -> bool:
    lowered = question.casefold()
    asks_why = any(
        marker in lowered
        for marker in ("why", "为什么", "为何", "怎么不", "怎会不")
    )
    negated_or_contrasted = any(
        marker in lowered
        for marker in (
            " not ",
            "n't",
            "instead of",
            "rather than",
            "不",
            "没",
            "没有",
            "而非",
            "而不是",
        )
    )
    return asks_why and negated_or_contrasted


def _select_negated_value_anchor(
    question: str,
    anchors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Select the explicitly rejected ontology value from source text.

    The Transformer identifies the broad intent, while this deterministic
    pass grounds *which* named action/objective is negated.  That distinction
    matters for natural contrasts such as "not down, but wait": both actions
    are mentioned, yet only DOWN is the requested alternative.
    """

    candidates = [
        item
        for item in anchors
        if item.get("kind") in {"objective", "action"}
        and int(item.get("start", -1)) >= 0
    ]
    if not candidates:
        return None
    folded = question.casefold()
    negation_markers = (
        "rather than",
        "instead of",
        "而不是",
        "不是",
        "没有",
        "而非",
        " not ",
        "n't",
        "不",
        "没",
    )
    positive_contrasts = ("而是", "反而", " but ", " instead ")
    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for item in candidates:
        start = int(item.get("start", -1))
        prefix = folded[:start]
        negative_at = max(prefix.rfind(marker) for marker in negation_markers)
        contrast_at = max(prefix.rfind(marker) for marker in positive_contrasts)
        clause_at = max(
            prefix.rfind(marker)
            for marker in ("，", ",", "；", ";", "。", ".", "？", "?", "！", "!")
        )
        if negative_at < 0 or negative_at < clause_at or contrast_at > negative_at:
            continue
        scored.append((negative_at, -start, item))
    if scored:
        return max(scored, key=lambda value: (value[0], value[1]))[2]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _validate_question_ir(
    question_ir: QuestionIRV2,
    *,
    question: str,
    environment_schema: Mapping[str, Any],
    strict_source_spans: bool = True,
) -> tuple[str, ...]:
    issues: list[str] = []
    known_entities = _known_entities(environment_schema.get("entities", {}))
    if known_entities and question_ir.target_entity not in known_entities:
        issues.append(f"unknown target entity {question_ir.target_entity!r}")
    for binding in question_ir.entity_roles:
        if known_entities and binding.entity_id not in known_entities:
            issues.append(f"unknown bound entity {binding.entity_id!r}")
        focus_entity = str(environment_schema.get("focus_entity", "")).strip()
        focus_binding = bool(
            not binding.source_span
            and focus_entity
            and binding.entity_id == focus_entity
            and EntityQueryRole.PREDICTION_TARGET in binding.roles
        )
        if strict_source_spans and not binding.source_span and not focus_binding:
            issues.append(
                f"entity {binding.entity_id!r} is missing an exact source_span"
            )
        elif binding.source_span not in question:
            issues.append(f"entity source_span is not literal: {binding.source_span!r}")
    for item in _flatten_interventions(question_ir.primitive_interventions):
        if (
            known_entities
            and item.entity_id != "batch"
            and item.entity_id not in known_entities
        ):
            issues.append(f"unknown intervention entity {item.entity_id!r}")
        if strict_source_spans and (
            not item.source_span or item.source_span not in question
        ):
            issues.append(f"intervention source_span is not literal: {item.source_span!r}")
        if strict_source_spans:
            for literal in _numeric_values(item.value):
                if (
                    literal not in question
                    and literal.rstrip("0").rstrip(".") not in question
                ):
                    issues.append(
                        f"intervention value {literal!r} is absent from the question"
                    )
    for relation in question_ir.relational_constraints:
        for entity_id in relation.subjects:
            if known_entities and entity_id not in known_entities:
                issues.append(f"unknown relation entity {entity_id!r}")
        if strict_source_spans and (
            not relation.source_span or relation.source_span not in question
        ):
            issues.append(
                f"relation source_span is not literal: {relation.source_span!r}"
            )
    target_roles = question_ir.entity_roles
    if target_roles and not any(
        binding.entity_id == question_ir.target_entity
        and EntityQueryRole.PREDICTION_TARGET in binding.roles
        for binding in target_roles
    ):
        issues.append("target_entity lacks the prediction_target role")
    query_variables = environment_schema.get("query_variables", {})
    known_variables = (
        {str(key) for key in query_variables}
        if isinstance(query_variables, Mapping)
        else set()
    )
    for variable in question_ir.target_variables:
        leaf = str(variable).rsplit(".", 1)[-1]
        if known_variables and leaf not in known_variables:
            issues.append(f"unknown query variable {leaf!r}")
    objectives = environment_schema.get("objectives", {})
    known_objectives = (
        {str(key) for key in objectives}
        if isinstance(objectives, Mapping)
        else set()
    )
    for key, value in question_ir.desired_outcomes.items():
        leaf = str(key).rsplit(".", 1)[-1]
        if leaf in {"objective", "goal", "goal_kind", "task"}:
            if known_objectives and str(value) not in known_objectives:
                issues.append(f"unknown desired objective {value!r}")
    if (
        question_ir.intent == QueryIntent.WHY_NOT
        and not question_ir.desired_outcomes
    ):
        issues.append("why_not question is missing its explicitly requested alternative")
    return tuple(dict.fromkeys(issues))


def _known_entities(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) == "ids" and isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                result.update(str(item) for item in nested)
            else:
                result.update(_known_entities(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            result.update(_known_entities(nested))
    return result


def _numeric_values(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            result.extend(_numeric_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            result.extend(_numeric_values(nested))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result.append(f"{value:g}" if isinstance(value, float) else str(value))
    return tuple(result)


def _flatten_interventions(values: Sequence[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for item in values:
        nested = getattr(item, "value", None)
        operation = str(getattr(getattr(item, "operation", ""), "value", ""))
        if (
            operation == "batch_intervention"
            and isinstance(nested, Sequence)
            and not isinstance(nested, (str, bytes))
        ):
            result.extend(_flatten_interventions(nested))
        else:
            result.append(item)
    return tuple(result)


def _question_cache_key(
    question: str,
    selected_frame: int | None,
    schema: Mapping[str, Any],
    *,
    model_id: str,
    cache_context: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        {
            "question": " ".join(question.split()).casefold(),
            "selected_frame": selected_frame,
            "schema": schema,
            "model_id": model_id,
            "cache_context": cache_context,
            "version": "question-ir.v2.literal-property-v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _backend_identity(backend: Any) -> str:
    return str(
        getattr(backend, "model_name_or_path", None)
        or getattr(backend, "model_name", None)
        or f"{type(backend).__module__}.{type(backend).__qualname__}"
    )


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "SemanticQueryCompiler",
    "SemanticTransformerQueryPlanner",
    "TransformerQueryPlanner",
]
