"""Typed environment-neutral explanation documents and vocabulary.

The online path compiles typed execution evidence into this IR and realizes the
selected units deterministically.  Environment adapters own vocabulary and
short predicate phrases; this module deliberately contains no environment
action or task rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from .schemas import EvidenceBundle


class ExplanationLayer(str, Enum):
    QUERY_ANSWER = "query_answer"
    TASK_GOAL = "task_goal"
    POLICY_PROPOSAL = "policy_proposal"
    PROPOSAL_RATIONALE = "proposal_rationale"
    COORDINATION = "coordination"
    FINAL_ACTION = "final_action"
    ACTION_EFFECT = "action_effect"
    COUNTERFACTUAL = "counterfactual"


class SentenceRole(str, Enum):
    ANSWER = "answer"
    PUBLIC_PROCESS = "public_process"
    TRACE_RATIONALE = "trace_rationale"
    COUNTERFACTUAL_RESULT = "counterfactual_result"


class DiscourseRelation(str, Enum):
    STANDALONE = "standalone"
    BECAUSE = "because"
    BUT = "but"
    CONTRAST = "contrast"
    IF_THEN = "if_then"


@dataclass(frozen=True)
class PlannedSentence:
    role: SentenceRole
    relation: DiscourseRelation
    unit_ids: tuple[str, ...]
    mandatory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "relation": self.relation.value,
            "unit_ids": list(self.unit_ids),
            "mandatory": self.mandatory,
        }


@dataclass(frozen=True)
class ExplanationMessagePlanV1:
    sentences: tuple[PlannedSentence, ...]
    ir_hash: str
    language: str
    schema_version: str = "explanation-message-plan.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ir_hash": self.ir_hash,
            "language": self.language,
            "sentences": [item.to_dict() for item in self.sentences],
        }


class ExplanationSemanticsAdapter(Protocol):
    """Small vocabulary boundary implemented by an environment adapter."""

    def explanation_entity_label(self, entity_id: str, language: str) -> str: ...

    def explanation_action_label(self, action: str, language: str) -> str: ...

    def explanation_objective_label(self, objective: str, language: str) -> str: ...

    def explanation_predicate_schema(self) -> Mapping[str, Any]: ...

    def explanation_verbalize_unit(
        self,
        unit: Mapping[str, Any],
        language: str,
    ) -> str: ...


# Short compatibility alias used by the generator type annotations.
ExplanationSemantics = ExplanationSemanticsAdapter


@dataclass(frozen=True)
class ExplanationUnit:
    unit_id: str
    layer: ExplanationLayer
    predicate: str
    arguments: tuple[str, ...]
    value: Any
    evidence_ids: tuple[str, ...]
    provenance: str
    salience: float = 0.0
    mandatory: bool = False
    allowed_entities: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    allowed_numbers: tuple[str, ...] = ()
    required_literals: tuple[str, ...] = ()
    reference_text: str = ""
    trace_derived: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer"] = self.layer.value
        return value


@dataclass(frozen=True)
class ExplanationIR:
    target: str
    language: str
    query_kind: str
    units: tuple[ExplanationUnit, ...]
    condition: str
    schema_version: str = "explanation-ir.v2"

    @property
    def ir_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def common_units(self) -> tuple[ExplanationUnit, ...]:
        return tuple(item for item in self.units if not item.trace_derived)

    @property
    def trace_units(self) -> tuple[ExplanationUnit, ...]:
        return tuple(item for item in self.units if item.trace_derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "language": self.language,
            "query_kind": self.query_kind,
            "condition": self.condition,
            "units": [item.to_dict() for item in self.units],
        }


@dataclass(frozen=True)
class RenderedSentence:
    role: SentenceRole
    text: str
    unit_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "text": self.text,
            "unit_ids": list(self.unit_ids),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ExplanationDocumentV3:
    sentences: tuple[RenderedSentence, ...]
    ir_hash: str
    message_plan: ExplanationMessagePlanV1
    renderer: str = "conversational_ir_renderer"
    validation_issues: tuple[str, ...] = ()
    schema_version: str = "explanation-document.v3"

    @property
    def text(self) -> str:
        separator = "" if self.message_plan.language == "zh-CN" else " "
        return separator.join(
            sentence.text.strip()
            for sentence in self.sentences
            if sentence.text.strip()
        )

    @property
    def sections(self) -> tuple[RenderedSentence, ...]:
        """Read-only compatibility view for v2 callers."""

        return self.sentences

    @property
    def used_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                unit_id
                for sentence in self.sentences
                for unit_id in sentence.unit_ids
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ir_hash": self.ir_hash,
            "renderer": self.renderer,
            "validation_issues": list(self.validation_issues),
            "message_plan": self.message_plan.to_dict(),
            "sentences": [item.to_dict() for item in self.sentences],
            "text": self.text,
            "used_unit_ids": list(self.used_unit_ids),
        }


# Import compatibility for analysis code that only needs the document protocol.
ExplanationDocumentV2 = ExplanationDocumentV3
RenderedSection = RenderedSentence


class EvidenceVocabulary:
    """Vocabulary fallback built only from labels shipped with evidence."""

    def __init__(self, evidence: EvidenceBundle) -> None:
        self.evidence = evidence
        self._actions = _label_table(evidence, "action_descriptions")
        self._objectives = _label_table(evidence, "objective_descriptions")

    def explanation_entity_label(self, entity_id: str, language: str) -> str:
        label = str(entity_id).replace("_", " ")
        return label if language == "zh-CN" else label[:1].upper() + label[1:]

    def explanation_action_label(self, action: str, language: str) -> str:
        return _localized_label(self._actions, action, language)

    def explanation_objective_label(self, objective: str, language: str) -> str:
        return _localized_label(self._objectives, objective, language)

    def explanation_predicate_schema(self) -> Mapping[str, Any]:
        return {}

    def explanation_verbalize_unit(
        self,
        unit: Mapping[str, Any],
        language: str,
    ) -> str:
        # A deliberately plain, domain-neutral fallback over semantic roles.
        # It knows no environment action, objective, or transition rule.
        arguments = tuple(str(value) for value in unit.get("arguments", ()))
        subject = self.explanation_entity_label(arguments[0], language) if arguments else "agent"
        predicate = str(unit.get("predicate", "fact"))
        raw_value = unit.get("value", {})
        value = raw_value if isinstance(raw_value, Mapping) else {"value": raw_value}
        chinese = language == "zh-CN"
        if predicate == "current_objective":
            label = str(value.get("label", value.get("id", "")))
            location = _compact_value(value.get("target_position"))
            show_position = bool(value.get("show_position", False)) and bool(location)
            phase = str(value.get("phase", ""))
            if chinese:
                prefix = "原场景中" if phase == "original" else "修改后" if phase == "edited" else ""
                suffix = f"，目标位置是{location}" if show_position else ""
                return f"{prefix}{subject}正在完成{label}{suffix}"
            prefix = "Originally, " if phase == "original" else "After the edit, " if phase == "edited" else ""
            suffix = f" at {location}" if show_position else ""
            return f"{prefix}{subject} is working toward {label}{suffix}"
        if predicate == "action_proposed":
            action = str(value.get("action_label", value.get("action", "")))
            return f"{subject}原本想{action}" if chinese else f"{subject} originally wanted to {action}"
        if predicate == "action_not_selected":
            desired = str(
                value.get("desired_action_label", value.get("desired_action", ""))
            )
            selected = str(
                value.get("proposed_action_label", value.get("proposed_action", ""))
            )
            reason_kind = str(value.get("reason_kind", ""))
            if reason_kind == "already_executed":
                return (
                    f"{subject}实际上已经执行了{desired}"
                    if chinese
                    else f"{subject} actually did choose to {desired}"
                )
            if reason_kind == "insufficient_evidence":
                return (
                    f"现有证据还不足以可靠说明{subject}为什么没有执行{desired}"
                    if chinese
                    else f"The available evidence is not sufficient to explain reliably why {subject} did not choose to {desired}"
                )
            return (
                f"{subject}没有执行{desired}，因为当前条件更支持{selected}"
                if chinese
                else f"{subject} did not choose to {desired} because the current conditions favored {selected}"
            )
        if predicate == "final_action":
            action = str(value.get("action_label", value.get("action", "")))
            return f"{subject}这一步{action}" if chinese else f"{subject} {action} on this step"
        if predicate == "coordination_resolution":
            final = str(value.get("final_action_label", value.get("final_action", "")))
            peers = [self.explanation_entity_label(item, language) for item in arguments[1:]]
            typed = value.get("typed_reason", {})
            resolution = value.get("resolution", {})
            typed = typed if isinstance(typed, Mapping) else {}
            resolution = resolution if isinstance(resolution, Mapping) else {}
            winner = next(
                (
                    str(source.get(key))
                    for source in (typed, resolution)
                    for key in (
                        "passing_agent",
                        "passing_agent_id",
                        "winner_agent_id",
                    )
                    if source.get(key)
                ),
                "",
            )
            winner_label = self.explanation_entity_label(winner, language) if winner else ""
            if chinese:
                return (
                    f"{subject}需要让{winner_label or peers[0]}先通过，"
                    f"避免发生碰撞，所以{subject}{final}"
                    if peers
                    else f"为了避免相互干扰，{subject}{final}"
                )
            return (
                f"{subject} let {winner_label or peers[0]} go first to avoid a collision, "
                f"so {subject} chose to {final}"
                if peers
                else f"To avoid interference, {subject} chose to {final}"
            )
        if predicate in {"action_constraint", "executed_path_condition"}:
            meaning = value.get("observed_meaning", {})
            if isinstance(meaning, Mapping):
                localized = meaning.get("zh" if chinese else "en")
                if localized:
                    return str(localized)
            selected = str(value.get("selected_action_label", value.get("selected_action", "")))
            constrained = str(value.get("constrained_action_label", value.get("constrained_action", "")))
            if constrained:
                return f"{constrained}不适合当前情况" if chinese else f"{constrained} did not fit the current situation"
            return f"{selected}更符合当前情况" if chinese else f"{selected} fit the current situation better"
        if predicate == "counterfactual_action_change":
            original = str(value.get("original_action_label", value.get("original_action", "")))
            edited = str(value.get("edited_action_label", value.get("edited_action", "")))
            edited_entities = [
                self.explanation_entity_label(item, language)
                for item in arguments[1:]
            ]
            if chinese:
                edit = f"修改{edited_entities[0]}后" if edited_entities else "修改场景后"
                return f"原场景中{subject}{original}；{edit}，{subject}{edited}"
            edit = f"after changing {edited_entities[0]}" if edited_entities else "after the scene edit"
            return f"{subject} originally chose {original}; {edit}, {subject} chose {edited}"
        compact = _compact_value(raw_value)
        if language == "zh-CN":
            return f"{subject}的{predicate.replace('_', ' ')}为{compact}"
        return f"{subject} has {predicate.replace('_', ' ')}: {compact}"


def _label_table(evidence: EvidenceBundle, key: str) -> Mapping[str, Any]:
    value = evidence.policy_results.get(
        key,
        evidence.direct_result.get(key, {}),
    )
    return value if isinstance(value, Mapping) else {}


def _localized_label(
    table: Mapping[str, Any],
    key: str,
    language: str,
) -> str:
    raw = table.get(str(key), {})
    if isinstance(raw, Mapping):
        candidate = raw.get("zh" if language == "zh-CN" else "en")
        if candidate:
            return str(candidate)
    fallback = str(key).replace("_", " ")
    return (
        fallback.casefold()
        if language != "zh-CN" and fallback.isupper()
        else fallback
    )


def _compact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "(" + ", ".join(_compact_value(item) for item in value) + ")"
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return str(value)
