"""Deterministic binding of free-form entity mentions to environment IDs.

The language model describes semantic intent, but it is not authoritative for
environment identity.  This module resolves adapter-declared aliases against
the original question, preserves the original source span, and supplies the
UI's focused entity when a question uses an implicit subject.  Matching is
Unicode-aware and insensitive to presentation punctuation and whitespace, so
``unit_A``, ``unit A``, ``实体A`` and ``实体 A`` can share one canonical
environment identity without adding question-specific answers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import unicodedata
from typing import Any, Mapping, Sequence

from .schemas import (
    EntityQueryRole,
    EntityRoleBinding,
    PrimitiveIntervention,
    PrimitiveOperation,
    QuestionIRV2,
)


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    source_span: str
    start: int
    end: int
    matched_alias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source_span": self.source_span,
            "start": self.start,
            "end": self.end,
            "matched_alias": self.matched_alias,
        }


@dataclass(frozen=True)
class QuestionBindingContext:
    question: str
    known_entities: tuple[str, ...]
    mentions: tuple[EntityMention, ...]
    focus_entity: str | None = None

    @classmethod
    def from_environment(
        cls,
        question: str,
        environment_schema: Mapping[str, Any],
    ) -> "QuestionBindingContext":
        entities = environment_schema.get("entities", {})
        known = tuple(sorted(_known_entities(entities)))
        references = _entity_references(entities)
        mentions = _resolve_mentions(question, references)
        raw_focus = str(environment_schema.get("focus_entity", "")).strip()
        focus = raw_focus if raw_focus and raw_focus in set(known) else None
        return cls(
            question=question,
            known_entities=known,
            mentions=mentions,
            focus_entity=focus,
        )

    @property
    def explicitly_mentioned_entities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.entity_id for item in self.mentions))

    def source_span(self, entity_id: str) -> str:
        candidates = [
            item
            for item in self.mentions
            if item.entity_id == str(entity_id)
        ]
        if not candidates:
            return ""
        selected = max(
            candidates,
            key=lambda item: (item.end - item.start, -item.start),
        )
        return selected.source_span

    def resolve_target(self, proposed: str) -> tuple[str, str]:
        proposed = str(proposed).strip()
        explicit = self.explicitly_mentioned_entities
        if proposed in explicit:
            return proposed, "model_target_confirmed_by_source"
        if len(explicit) == 1:
            return explicit[0], "single_explicit_entity"
        if not explicit and self.focus_entity:
            return self.focus_entity, "ui_focus_entity"
        if proposed in self.known_entities:
            return proposed, "model_target_in_explicit_candidate_set"
        if len(self.known_entities) == 1:
            return self.known_entities[0], "single_environment_entity"
        return proposed, "unresolved"

    def bind(self, question_ir: QuestionIRV2) -> tuple[QuestionIRV2, dict[str, Any]]:
        original_target = question_ir.target_entity
        target, target_origin = self.resolve_target(original_target)
        primitives = tuple(
            self._bind_intervention(item)
            for item in question_ir.primitive_interventions
        )
        relations = tuple(
            replace(
                item,
                source_span=self.canonical_source_span(item.source_span),
            )
            for item in question_ir.relational_constraints
        )

        roles_by_entity: dict[str, list[EntityQueryRole]] = {}
        model_spans = {
            binding.entity_id: self.canonical_source_span(binding.source_span)
            for binding in question_ir.entity_roles
            if binding.source_span
        }
        for binding in question_ir.entity_roles:
            roles_by_entity.setdefault(binding.entity_id, []).extend(
                role
                for role in binding.roles
                if role != EntityQueryRole.PREDICTION_TARGET
            )
        if target:
            roles_by_entity.setdefault(target, []).insert(
                0,
                EntityQueryRole.PREDICTION_TARGET,
            )
        for item in _flatten_interventions(primitives):
            if item.entity_id and item.entity_id != "batch":
                roles_by_entity.setdefault(item.entity_id, []).append(
                    EntityQueryRole.INTERVENTION_SUBJECT
                )
        for relation in relations:
            for entity_id in relation.subjects:
                if entity_id != target:
                    roles_by_entity.setdefault(entity_id, []).append(
                        EntityQueryRole.CONTEXT_ENTITY
                    )
        for entity_id in self.explicitly_mentioned_entities:
            if entity_id != target:
                roles_by_entity.setdefault(entity_id, []).append(
                    EntityQueryRole.CONTEXT_ENTITY
                )

        ordered_entities = tuple(
            dict.fromkeys(
                (
                    target,
                    *self.explicitly_mentioned_entities,
                    *roles_by_entity,
                )
            )
        )
        bindings = tuple(
            EntityRoleBinding(
                entity_id=entity_id,
                roles=tuple(dict.fromkeys(roles_by_entity[entity_id])),
                source_span=(
                    self.source_span(entity_id)
                    or model_spans.get(entity_id, "")
                ),
            )
            for entity_id in ordered_entities
            if entity_id and roles_by_entity.get(entity_id)
        )
        referenced = tuple(
            entity_id
            for entity_id in ordered_entities
            if entity_id
        )
        bound = replace(
            question_ir,
            target_entity=target,
            referenced_entities=referenced,
            entity_roles=bindings,
            primitive_interventions=primitives,
            relational_constraints=relations,
            target_variables=_retarget_paths(
                question_ir.target_variables,
                original_target,
                target,
            ),
            desired_outcomes=_retarget_mapping_keys(
                question_ir.desired_outcomes,
                original_target,
                target,
            ),
            assumed_outcomes=_retarget_mapping_keys(
                question_ir.assumed_outcomes,
                original_target,
                target,
            ),
        )
        audit = {
            "target_origin": target_origin,
            "model_target": original_target,
            "resolved_target": target,
            "focus_entity": self.focus_entity,
            "mentions": [item.to_dict() for item in self.mentions],
        }
        return bound, audit

    def canonical_source_span(self, proposed_span: str) -> str:
        proposed = str(proposed_span).strip()
        if not proposed:
            return ""
        if proposed in self.question:
            return proposed
        match = _find_normalized_span(self.question, proposed)
        return match or proposed

    def _bind_intervention(
        self,
        item: PrimitiveIntervention,
    ) -> PrimitiveIntervention:
        value = item.value
        if (
            item.operation == PrimitiveOperation.BATCH_INTERVENTION
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            value = tuple(
                self._bind_intervention(nested)
                if isinstance(nested, PrimitiveIntervention)
                else nested
                for nested in value
            )
        proposed_span = self.canonical_source_span(item.source_span)
        source_span = (
            proposed_span
            if proposed_span in self.question
            else self._intervention_source_span(item)
            or proposed_span
        )
        return replace(
            item,
            value=value,
            source_span=source_span,
        )

    def _intervention_source_span(
        self,
        item: PrimitiveIntervention,
    ) -> str:
        """Recover a literal edit span from bound entities and value anchors.

        Small instruction models often paraphrase only the connective word in
        a source span (for example, replacing Chinese ``把`` with ``将``) even
        while the typed entity and coordinate are correct.  The environment
        binder therefore reconstructs the literal span from the original text
        instead of accepting model-authored prose as provenance.
        """

        mentions = [
            mention
            for mention in self.mentions
            if mention.entity_id == item.entity_id
        ]
        if not mentions:
            return ""
        mention = mentions[0]
        cursor = mention.end
        last_end = cursor
        anchors = _literal_value_anchors(item.value)
        if not anchors:
            return ""
        for variants in anchors:
            found: tuple[int, int] | None = None
            for candidate in variants:
                index = self.question.find(candidate, cursor)
                if index >= 0 and (
                    found is None or index < found[0]
                ):
                    found = (index, index + len(candidate))
            if found is None:
                return ""
            cursor = found[1]
            last_end = found[1]
        while (
            last_end < len(self.question)
            and self.question[last_end] in " \t)]）}%％"
        ):
            last_end += 1
        return self.question[mention.start:last_end]

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "explicit_mentions": [
                {
                    "entity_id": item.entity_id,
                    "source_span": item.source_span,
                }
                for item in self.mentions
            ],
            "focus_entity": self.focus_entity,
            "rule": (
                "Use an explicit mentioned entity as t. When no entity is "
                "written, t is focus_entity."
            ),
        }


def _retarget_paths(
    values: Sequence[str],
    original_target: str,
    resolved_target: str,
) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if original_target and resolved_target and value.startswith(
            original_target + "."
        ):
            value = resolved_target + value[len(original_target) :]
        elif resolved_target and "." not in value:
            value = f"{resolved_target}.{value}"
        elif resolved_target and value.startswith("."):
            value = resolved_target + value
        result.append(value)
    return tuple(result)


def _retarget_mapping_keys(
    values: Mapping[str, Any],
    original_target: str,
    resolved_target: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if original_target and resolved_target and key.startswith(
            original_target + "."
        ):
            key = resolved_target + key[len(original_target) :]
        elif resolved_target and "." not in key:
            key = f"{resolved_target}.{key}"
        elif resolved_target and key.startswith("."):
            key = resolved_target + key
        result[key] = value
    return result


def _resolve_mentions(
    question: str,
    references: Mapping[str, Sequence[str]],
) -> tuple[EntityMention, ...]:
    normalized_question, offsets = _normalized_with_offsets(question)
    candidates: list[EntityMention] = []
    seen: set[tuple[str, int, int]] = set()
    for entity_id, aliases in references.items():
        for alias in aliases:
            normalized_alias, _ = _normalized_with_offsets(str(alias))
            if not normalized_alias:
                continue
            start = 0
            while True:
                index = normalized_question.find(normalized_alias, start)
                if index < 0:
                    break
                end_index = index + len(normalized_alias)
                start = index + 1
                if not _is_entity_boundary(
                    question,
                    offsets,
                    index,
                    end_index,
                    normalized_alias,
                ):
                    continue
                source_start = offsets[index]
                source_end = offsets[end_index - 1] + 1
                key = (str(entity_id), source_start, source_end)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    EntityMention(
                        entity_id=str(entity_id),
                        source_span=question[source_start:source_end],
                        start=source_start,
                        end=source_end,
                        matched_alias=str(alias),
                    )
                )
    candidates.sort(
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            item.entity_id,
        )
    )
    selected: list[EntityMention] = []
    for candidate in candidates:
        overlapping = next(
            (
                item
                for item in selected
                if item.entity_id == candidate.entity_id
                and not (
                    candidate.end <= item.start
                    or candidate.start >= item.end
                )
            ),
            None,
        )
        if overlapping is None:
            selected.append(candidate)
    return tuple(selected)


def _find_normalized_span(text: str, fragment: str) -> str:
    normalized_text, offsets = _normalized_with_offsets(text)
    normalized_fragment, _ = _normalized_with_offsets(fragment)
    if not normalized_fragment:
        return ""
    index = normalized_text.find(normalized_fragment)
    if index < 0:
        return ""
    end = index + len(normalized_fragment)
    return text[offsets[index] : offsets[end - 1] + 1]


def _literal_value_anchors(value: Any) -> tuple[tuple[str, ...], ...]:
    anchors: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            anchors.extend(_literal_value_anchors(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            anchors.extend(_literal_value_anchors(nested))
    elif isinstance(value, bool) or value is None:
        return ()
    elif isinstance(value, int):
        anchors.append((str(value),))
    elif isinstance(value, float):
        compact = f"{value:g}"
        anchors.append(tuple(dict.fromkeys((compact, str(value)))))
    elif isinstance(value, str) and value.strip():
        anchors.append((value.strip(),))
    return tuple(anchors)


def _normalized_with_offsets(text: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(str(text)):
        folded = unicodedata.normalize("NFKC", character).casefold()
        for value in folded:
            category = unicodedata.category(value)
            if value.isalnum() or category.startswith(("L", "N")):
                characters.append(value)
                offsets.append(index)
    return "".join(characters), tuple(offsets)


def _is_entity_boundary(
    original_text: str,
    offsets: Sequence[int],
    start: int,
    end: int,
    alias: str,
) -> bool:
    source_start = offsets[start]
    source_end = offsets[end - 1] + 1
    if (
        alias
        and alias[-1].isdigit()
        and source_end < len(original_text)
        and original_text[source_end].isdigit()
    ):
        return False
    if alias and alias[0].isascii() and alias[0].isalpha():
        if (
            source_start > 0
            and original_text[source_start - 1].isascii()
            and original_text[source_start - 1].isalnum()
        ):
            return False
    if alias and alias[-1].isascii() and alias[-1].isalpha():
        if (
            source_end < len(original_text)
            and original_text[source_end].isascii()
            and original_text[source_end].isalnum()
        ):
            return False
    return True


def _entity_references(value: Any) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    if isinstance(value, Mapping):
        raw = value.get("references")
        if isinstance(raw, Mapping):
            for entity_id, aliases in raw.items():
                if isinstance(aliases, Sequence) and not isinstance(
                    aliases,
                    (str, bytes),
                ):
                    result[str(entity_id)] = tuple(
                        str(alias) for alias in aliases
                    )
        for nested in value.values():
            for entity_id, aliases in _entity_references(nested).items():
                result.setdefault(entity_id, aliases)
    return result


def _known_entities(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                str(key) == "ids"
                and isinstance(nested, Sequence)
                and not isinstance(nested, (str, bytes))
            ):
                result.update(str(item) for item in nested)
            else:
                result.update(_known_entities(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            result.update(_known_entities(nested))
    return result


def _flatten_interventions(
    values: Sequence[PrimitiveIntervention],
) -> tuple[PrimitiveIntervention, ...]:
    result: list[PrimitiveIntervention] = []
    for item in values:
        if (
            item.operation == PrimitiveOperation.BATCH_INTERVENTION
            and isinstance(item.value, Sequence)
            and not isinstance(item.value, (str, bytes))
        ):
            result.extend(
                _flatten_interventions(
                    tuple(
                        nested
                        for nested in item.value
                        if isinstance(nested, PrimitiveIntervention)
                    )
                )
            )
        else:
            result.append(item)
    return tuple(result)


__all__ = [
    "EntityMention",
    "QuestionBindingContext",
]
