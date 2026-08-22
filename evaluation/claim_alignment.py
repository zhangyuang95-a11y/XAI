"""Transformer semantic alignment between extracted claims and simulator facts."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel

from backend.nlp.schemas import (
    AtomicClaim,
    ClaimVerdict,
    ClaimVerdictStatus,
    EvidenceBundle,
    EvidenceRecord,
)
from backend.nlp.tokenizer import DEFAULT_TOKENIZER, MultilingualQueryTokenizer


class TransformerFactMatcher:
    """A pretrained semantic matcher; it contains no environment keyword rules."""

    def __init__(
        self,
        model_name: str = DEFAULT_TOKENIZER,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = MultilingualQueryTokenizer(model_name, local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.encoder.eval()

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, int(self.encoder.config.hidden_size)), dtype=np.float32)
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer.backend(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=192,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                hidden = self.encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                batches.append(pooled.cpu().numpy())
        return np.concatenate(batches, axis=0)

def _numeric_literals(text: str) -> tuple[float, ...]:
    values: list[float] = []
    current = ""
    for character in text:
        if character.isdigit() or (character == "." and current and "." not in current):
            current += character
        elif current:
            try:
                values.append(float(current))
            except ValueError:
                pass
            current = ""
    if current:
        try:
            values.append(float(current))
        except ValueError:
            pass
    return tuple(values)


def _complete_alignment(
    claim: AtomicClaim,
    selected: tuple[EvidenceRecord, ...],
    records: Sequence[EvidenceRecord],
) -> tuple[EvidenceRecord, ...]:
    """Add only type-mandated evidence when semantic alignment is empty.

    The Transformer still determines the natural-language claim type. This
    deterministic completion prevents a malformed or empty list of IDs from
    hiding directly recorded policy/state evidence; causal claims deliberately
    receive no fallback because their factor-specific pairing must be aligned.
    """

    if claim.claim_type.lower() == "causal":
        return selected
    source_types = {
        "action": ("neural_policy",),
        "state": ("state",),
        "relation": ("state", "neural_policy"),
        "environment_constraint": ("neural_policy",),
        "program": ("program_trace",),
        "future": ("counterfactual_rollout",),
        "trajectory": ("counterfactual_rollout",),
        "counterfactual": ("counterfactual_rollout",),
        "comparison": (
            "baseline_rollout",
            "counterfactual_rollout",
        ),
        "uncertainty": ("direct_result",),
    }.get(claim.claim_type.lower(), ())
    mandated = tuple(
        record
        for record in records
        if record.source_type in source_types
    )
    if not mandated:
        return selected
    selected_ids = {record.evidence_id for record in selected}
    selected_has_required_type = any(
        record.source_type in source_types for record in selected
    )
    matching = tuple(
        record
        for record in mandated
        if claim.expected_outcome is not None
        and _structured_value_matches(
            claim.expected_outcome,
            record.observed_value,
        )
    )
    direct_singleton_types = {
        "action",
        "environment_constraint",
        "uncertainty",
    }
    additions = matching
    if (
        not additions
        and not selected_has_required_type
        and claim.claim_type.lower() in direct_singleton_types
        and len(mandated) == 1
    ):
        additions = mandated
    return selected + tuple(
        record
        for record in additions
        if record.evidence_id not in selected_ids
    )


def _short_error(error: BaseException, *, limit: int = 280) -> str:
    return " ".join(str(error).split())[:limit]


def _alignment_catalog(
    records: Sequence[EvidenceRecord],
) -> list[dict[str, Any]]:
    """Compact paired causal branches for Transformer ID selection."""

    catalog: list[dict[str, Any]] = []
    causal_pairs: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if (
            record.evidence_id.startswith("cause_")
            and record.rollout_id is not None
        ):
            causal_pairs.setdefault(record.rollout_id, []).append(
                record
            )
            continue
        catalog.append(
            {
                "evidence_ids": [record.evidence_id],
                "source_type": record.source_type,
                "frame_id": record.frame_id,
                "observed_value": record.observed_value,
                "provenance": dict(record.provenance),
            }
        )
    for rollout_id, pair in causal_pairs.items():
        counterfactual = next(
            (
                item
                for item in pair
                if item.source_type == "counterfactual_rollout"
            ),
            None,
        )
        provenance = next(
            (
                dict(item.provenance)
                for item in pair
                if item.provenance
            ),
            {},
        )
        catalog.append(
            {
                "evidence_ids": [
                    item.evidence_id for item in pair
                ],
                "source_type": "paired_causal_rollout",
                "rollout_id": rollout_id,
                "factor": provenance.get("factor"),
                "description": provenance.get("description"),
                "observable_by_actor": provenance.get(
                    "observable_by_actor"
                ),
                "supported": provenance.get("supported"),
                "intervention": (
                    counterfactual.intervention
                    if counterfactual is not None
                    else None
                ),
                "observed_effect": (
                    counterfactual.observed_value
                    if counterfactual is not None
                    else None
                ),
            }
        )
    return catalog


def _claim_alignment_prompt(
    claims: Sequence[AtomicClaim],
    records: Sequence[EvidenceRecord],
) -> str:
    """Build a domain-neutral semantic-normalization request."""

    skeleton = {
        "alignments": [
            {
                "claim_id": claim.claim_id,
                "evidence_assertions": [],
            }
            for claim in claims
        ]
    }
    return (
        "Translate every independently extracted natural-language claim into "
        "the canonical fields of the smallest relevant evidence records. "
        "Do not decide truth and do not emit evaluation labels. Return an "
        "`alignments` array with exactly one item for every supplied claim. "
        "Each item has claim_id and an `evidence_assertions` array. Each "
        "assertion has an evidence_id copied exactly from Evidence and "
        "`claim_value`: the value asserted by the claim, converted to the "
        "canonical type shown by that record. A negative natural-language "
        "status can map to false and a translated action can map to its "
        "canonical action name. Never copy an observed value merely because "
        "it appears in Evidence; claim_value must come from the claim. Use "
        "multiple assertions when one claim compares or connects several "
        "facts. An assertion about a program-derived action effect must select "
        "that exact program feature, not a different branch or a static "
        "coordinate offset. Fill every evidence_assertions array in this exact "
        "output skeleton; do not change, omit, or invent claim_id values: "
        f"{json.dumps(skeleton, ensure_ascii=False)}\n"
        f"Claims: {json.dumps([claim.to_dict() for claim in claims], ensure_ascii=False, default=str)}\n"
        f"Evidence: {json.dumps(_alignment_catalog(records), ensure_ascii=False, default=str)}"
    )


def _semantic_record_alignments(
    claims: Sequence[AtomicClaim],
    records: Sequence[EvidenceRecord],
    matcher: TransformerFactMatcher,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, dict[str, float]],
]:
    """Align final-answer claims with a multilingual encoder.

    This path is fast, bounded, and independent of environment vocabulary.
    The encoder selects candidate records; deterministic checks below still
    decide whether their observed values support the assertion.
    """

    descriptions: list[str] = []
    description_records: list[EvidenceRecord] = []
    for record in records:
        for text in _record_semantic_texts(record):
            if text.strip():
                descriptions.append(text)
                description_records.append(record)
    if not descriptions:
        return {}, {}
    claim_vectors = matcher.encode(
        [claim.text for claim in claims]
    )
    description_vectors = matcher.encode(descriptions)
    alignments: dict[str, tuple[str, ...]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    for claim_index, claim in enumerate(claims):
        claim_numbers = _claim_numeric_literals(
            claim.text,
            claim.entities,
        )
        compatible_sources = _compatible_semantic_sources(
            claim.claim_type
        )
        scores: dict[str, float] = {}
        for description_index, record in enumerate(
            description_records
        ):
            if (
                compatible_sources
                and record.source_type not in compatible_sources
            ):
                continue
            if not _semantic_record_compatible(
                claim.claim_type,
                record,
            ):
                continue
            if (
                claim.claim_type.lower() == "state"
                and record.source_type == "program_trace"
                and len(claim_numbers) > 1
                and not all(
                    _record_contains_number(record, number)
                    for number in claim_numbers
                )
            ):
                continue
            similarity = float(
                description_vectors[description_index]
                @ claim_vectors[claim_index]
            )
            scores[record.evidence_id] = max(
                scores.get(record.evidence_id, -1.0),
                similarity,
            )
        if not scores:
            continue
        if claim.claim_type.lower() == "state":
            program_ids = [
                record.evidence_id
                for record in records
                if record.source_type == "program_trace"
                and record.evidence_id in scores
            ]
            direct_ids = [
                record.evidence_id
                for record in records
                if record.source_type != "program_trace"
                and record.evidence_id in scores
            ]
            if (
                program_ids
                and direct_ids
                and max(scores[evidence_id] for evidence_id in program_ids)
                < max(scores[evidence_id] for evidence_id in direct_ids)
            ):
                for evidence_id in program_ids:
                    scores.pop(evidence_id, None)
        ordered = sorted(
            scores,
            key=scores.__getitem__,
            reverse=True,
        )
        selected: list[str] = []
        covered_numbers: list[float] = []
        for number in claim_numbers:
            matching = [
                record
                for record in records
                if record.evidence_id in scores
                and _record_contains_number(record, number)
            ]
            if matching:
                best = max(
                    matching,
                    key=lambda record: scores[record.evidence_id],
                )
                selected.append(best.evidence_id)
                covered_numbers.append(number)
        if not claim_numbers or len(covered_numbers) != len(
            claim_numbers
        ):
            selected.insert(0, ordered[0])
        best_score = scores[ordered[0]]
        if not claim_numbers:
            selected.extend(
                evidence_id
                for evidence_id in ordered[1:3]
                if scores[evidence_id]
                >= max(0.18, best_score - 0.06)
            )
        alignments[claim.claim_id] = tuple(
            dict.fromkeys(selected)
        )
        score_maps[claim.claim_id] = {
            evidence_id: scores[evidence_id]
            for evidence_id in alignments[claim.claim_id]
        }
    return alignments, score_maps


def _compatible_semantic_sources(
    claim_type: str,
) -> tuple[str, ...]:
    return {
        "causal": (
            "baseline_rollout",
            "counterfactual_rollout",
        ),
        "counterfactual": ("counterfactual_rollout",),
        "future": ("counterfactual_rollout",),
        "trajectory": ("counterfactual_rollout",),
        "comparison": (
            "baseline_rollout",
            "counterfactual_rollout",
        ),
        "uncertainty": ("direct_result",),
    }.get(
        claim_type.lower(),
        ("state", "neural_policy", "program_trace"),
    )


def _semantic_record_compatible(
    claim_type: str,
    record: EvidenceRecord,
) -> bool:
    """Use program records exposed explicitly for human explanation."""

    if record.source_type != "program_trace":
        return True
    if claim_type.lower() in {"program", "relation"}:
        return True
    observed = record.observed_value
    if not isinstance(observed, Mapping):
        return False
    meaning = observed.get("observed_meaning")
    return bool(
        isinstance(meaning, Mapping)
        and str(meaning.get("explanation_role", "")).strip()
    )


def _record_semantic_texts(
    record: EvidenceRecord,
) -> tuple[str, ...]:
    observed = record.observed_value
    if not isinstance(observed, Mapping):
        return (str(observed),)
    texts: list[str] = []
    for key in ("verbalizations", "value_verbalizations"):
        values = observed.get(key, ())
        if isinstance(values, Sequence) and not isinstance(
            values,
            (str, bytes),
        ):
            texts.extend(str(value) for value in values)
    for key in ("feature_description", "observed_meaning"):
        value = observed.get(key)
        if isinstance(value, Mapping):
            texts.extend(
                str(item)
                for item in value.values()
                if not isinstance(item, (Mapping, Sequence))
                or isinstance(item, str)
            )
            requirements = value.get("explanation_requirements")
            if isinstance(requirements, Sequence) and not isinstance(
                requirements,
                (str, bytes),
            ):
                texts.extend(
                    json.dumps(
                        requirement,
                        ensure_ascii=False,
                        default=str,
                    )
                    for requirement in requirements
                    if isinstance(requirement, Mapping)
                )
    semantic_fields = {
        key: observed[key]
        for key in (
            "subject",
            "predicate",
            "arguments",
            "semantic_name",
            "semantic_verbalizations",
            "role",
            "group",
            "value",
            "value_verbalizations",
            "unit",
            "relation",
            "effect",
            "action",
            "action_verbalizations",
            "feature",
            "observed_value",
            "result",
        )
        if key in observed
    }
    if semantic_fields:
        texts.append(
            json.dumps(
                semantic_fields,
                ensure_ascii=False,
                default=str,
            )
        )
    return tuple(dict.fromkeys(texts))


def _claim_numeric_literals(
    text: str,
    entity_references: Sequence[str] = (),
) -> tuple[float, ...]:
    sanitized = text
    for entity in sorted(
        (
            value
            for value in entity_references
            if value and any(character.isalpha() for character in value)
        ),
        key=len,
        reverse=True,
    ):
        sanitized = sanitized.replace(entity, " ")
    return tuple(
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?",
            sanitized,
        )
    )


def _record_contains_number(
    record: EvidenceRecord,
    number: float,
) -> bool:
    canonical_numbers = _all_numeric_values(
        _canonical_assertion_value(record.observed_value)
    )
    # Generic semantic facts keep participating entities in ``arguments``.
    # Their identifiers can contain numbers (for example an entity translated
    # as "robot 2") even when the canonical value slot is a relation schema.
    # Include only these declared arguments, not arbitrary provenance strings
    # or schema-version labels, in the numeric-evidence audit.
    argument_numbers: tuple[float, ...] = ()
    if isinstance(record.observed_value, Mapping):
        arguments = record.observed_value.get("arguments", ())
        if isinstance(arguments, Sequence) and not isinstance(
            arguments,
            (str, bytes),
        ):
            argument_numbers = tuple(
                candidate
                for argument in arguments
                for candidate in _numeric_literals(str(argument))
            )
    return any(
        math.isclose(
            number,
            candidate,
            rel_tol=1e-4,
            abs_tol=1e-4,
        )
        for candidate in (*canonical_numbers, *argument_numbers)
    )


def _semantic_evidence_supports_claim(
    claim: AtomicClaim,
    records: Sequence[EvidenceRecord],
    scores: Mapping[str, float],
    *,
    threshold: float,
) -> bool:
    matched = tuple(
        record
        for record in records
        if scores.get(record.evidence_id, -1.0) >= threshold
    )
    if not matched:
        return False
    claim_numbers = _claim_numeric_literals(
        claim.text,
        claim.entities,
    )
    if claim_numbers and not all(
        any(_record_contains_number(record, number) for record in matched)
        for number in claim_numbers
    ):
        return False
    asserted_boolean = _expected_boolean(
        claim.expected_outcome
    )
    if asserted_boolean is not None:
        observed_booleans = {
            value
            for record in matched
            for value in _all_boolean_values(
                _canonical_assertion_value(
                    record.observed_value
                )
            )
        }
        if asserted_boolean not in observed_booleans:
            return False
    return True


def _expected_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return None
    if isinstance(value, Mapping):
        if "asserted_value" in value:
            return _expected_boolean(value["asserted_value"])
        # Arbitrary Transformer-produced keys describe whole propositions
        # (for example ``{"carries_nothing": true}``). Their Boolean says
        # that the proposition is asserted, not necessarily that the
        # environment's canonical property is true. Semantic alignment below
        # resolves that proposition; do not reverse a negated property here.
        return None
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        values = {
            candidate
            for item in value
            for candidate in (_expected_boolean(item),)
            if candidate is not None
        }
        return next(iter(values)) if len(values) == 1 else None
    return None


def _all_boolean_values(value: Any) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            result
            for item in value.values()
            for result in _all_boolean_values(item)
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        return tuple(
            result
            for item in value
            for result in _all_boolean_values(item)
        )
    return ()


def _alignment_candidates(
    records: Sequence[EvidenceRecord],
    claims: Sequence[AtomicClaim],
    *,
    query_entities: Sequence[str] = (),
) -> tuple[EvidenceRecord, ...]:
    """Discard state records owned only by unrelated named entities."""

    claim_entities = {
        _normalized_reference(entity)
        for claim in claims
        for entity in claim.entities
        if _normalized_reference(entity)
    }
    canonical_query_entities = {
        _normalized_reference(entity)
        for entity in query_entities
        if _normalized_reference(entity)
    }
    entities = canonical_query_entities or claim_entities
    if not entities:
        return tuple(records)
    selected: list[EvidenceRecord] = []
    for record in records:
        if record.source_type != "state":
            selected.append(record)
            continue
        observed_entities = {
            _normalized_reference(value)
            for value in _record_entity_references(record.observed_value)
            if _normalized_reference(value)
        }
        if not observed_entities or observed_entities.intersection(entities):
            selected.append(record)
    return tuple(selected)


def _record_entity_references(observed: Any) -> tuple[str, ...]:
    if not isinstance(observed, Mapping):
        return ()
    values: list[str] = []
    subject = observed.get("subject")
    if subject is not None:
        values.append(str(subject))
    arguments = observed.get("arguments", ())
    if isinstance(arguments, Sequence) and not isinstance(
        arguments,
        (str, bytes),
    ):
        values.extend(str(value) for value in arguments[:1])
    for key in ("target", "entity", "subject"):
        value = observed.get(key)
        if value is not None:
            values.append(str(value))
    return tuple(dict.fromkeys(values))


def _normalized_reference(value: Any) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def _bundle_records(bundle: EvidenceBundle) -> tuple[EvidenceRecord, ...]:
    direct_keys = (
        "execution_refused",
        "refusal_reasons",
        "executed_frame",
        "decision_outcome_frame",
        "target",
        "baseline_action_distribution",
        "edited_action_distribution",
        "edited_argmax_action",
        "recorded_proposed_action",
        "recorded_executed_action",
        "counterfactual_contrast",
    )
    records: list[EvidenceRecord] = [
        EvidenceRecord(
            evidence_id="direct_result",
            source_type="direct_result",
            frame_id=bundle.query_plan.frame_reference,
            observed_value={
                key: bundle.direct_result[key]
                for key in direct_keys
                if key in bundle.direct_result
            },
        )
    ]
    for index, fact in enumerate(bundle.state_facts):
        observed = {
            key: fact[key]
            for key in (
                "fact_id",
                "predicate",
                "arguments",
                "value",
                "factor_groups",
                "verbalizations",
                "value_verbalizations",
            )
            if key in fact
        }
        requirements = _embedded_explanation_requirements(
            observed.get("value")
        )
        if requirements:
            # The individual requirements below are the canonical semantic
            # leaves used for alignment. Keeping the complete environment
            # object here would make a small Transformer search an unrelated
            # domain payload and would duplicate the same values many times.
            value = observed.get("value")
            observed["value"] = {
                "schema": value.get("schema"),
                "requirement_count": len(requirements),
            }
        records.append(
            EvidenceRecord(
                evidence_id=f"state_{index}",
                source_type="state",
                frame_id=bundle.query_plan.frame_reference,
                observed_value=observed,
            )
        )
        for requirement_index, requirement in enumerate(requirements):
            records.append(
                EvidenceRecord(
                    evidence_id=(
                        f"state_{index}_requirement_"
                        f"{requirement_index}"
                    ),
                    source_type="state",
                    frame_id=bundle.query_plan.frame_reference,
                    observed_value={
                        "subject": (
                            fact.get("arguments", (None,))[0]
                            if fact.get("arguments")
                            else None
                        ),
                        "source_predicate": fact.get("predicate"),
                        **{
                            key: requirement[key]
                            for key in (
                                "key",
                                "semantic_name",
                                "semantic_verbalizations",
                                "role",
                                "group",
                                "value",
                                "value_verbalizations",
                                "unit",
                                "relation",
                                "effect",
                                "action",
                                "action_verbalizations",
                            )
                            if key in requirement
                        },
                        "value_verbalizations": list(
                            requirement.get(
                                "value_verbalizations",
                                fact.get(
                                    "value_verbalizations",
                                    (),
                                ),
                            )
                        )
                        if (
                            isinstance(
                                requirement.get("value"),
                                (str, bool),
                            )
                            or requirement.get(
                                "value_verbalizations"
                            )
                        )
                        else [],
                    },
                    provenance={
                        "semantic_leaf": True,
                        "parent_fact_id": fact.get("fact_id"),
                    },
                )
            )
    records.extend(
        EvidenceRecord(
            evidence_id=f"baseline_{index}",
            source_type="baseline_rollout",
            frame_id=bundle.query_plan.frame_reference,
            rollout_id=str(
                value.get("rollout_id", f"baseline_{index}")
            ),
            observed_value=dict(value),
            uncertainty={
                "paired": True,
                "seed": value.get("seed"),
            },
            provenance={"policy": "original_neural_policy"},
        )
        for index, value in enumerate(bundle.baseline_results[:8])
    )
    records.extend(
        EvidenceRecord(
            evidence_id=f"counterfactual_{index}",
            source_type="counterfactual_rollout",
            frame_id=bundle.query_plan.frame_reference,
            rollout_id=str(
                value.get("rollout_id", f"counterfactual_{index}")
            ),
            observed_value=dict(value),
            intervention={"items": list(bundle.interventions)},
            uncertainty={
                "paired": True,
                "seed": value.get("seed"),
            },
            provenance={"policy": "original_neural_policy"},
        )
        for index, value in enumerate(bundle.counterfactual_results[:8])
    )
    causal_candidates = bundle.causal_analysis.get("candidates", ())
    if isinstance(causal_candidates, Sequence) and not isinstance(
        causal_candidates, (str, bytes)
    ):
        usable_candidates = [
            value
            for value in causal_candidates
            if isinstance(value, Mapping)
        ]
        usable_candidates.sort(
            key=lambda value: (
                bool(value.get("supported", False)),
                float(value.get("l1_policy_effect", 0.0)),
                float(value.get("action_change_rate", 0.0)),
            ),
            reverse=True,
        )
        for index, value in enumerate(usable_candidates):
            if not isinstance(value, Mapping):
                continue
            candidate_id = str(
                value.get("candidate_id", f"candidate_{index}")
            )
            provenance = {
                "factor": value.get("factor"),
                "description": value.get("description"),
                "observable_by_actor": value.get(
                    "observable_by_actor", False
                ),
                "supported": value.get("supported", False),
            }
            records.append(
                EvidenceRecord(
                    evidence_id=f"cause_{candidate_id}_baseline",
                    source_type="baseline_rollout",
                    frame_id=bundle.query_plan.frame_reference,
                    rollout_id=candidate_id,
                    observed_value={
                        "distribution": value.get(
                            "baseline_first_action_distribution", {}
                        ),
                        "paired_seeds": value.get("paired_seeds", ()),
                    },
                    provenance=provenance,
                )
            )
            records.append(
                EvidenceRecord(
                    evidence_id=f"cause_{candidate_id}_counterfactual",
                    source_type="counterfactual_rollout",
                    frame_id=bundle.query_plan.frame_reference,
                    rollout_id=candidate_id,
                    intervention={
                        "items": value.get("interventions", ())
                    },
                    observed_value={
                        "distribution": value.get(
                            "counterfactual_first_action_distribution", {}
                        ),
                        "action_probability_delta": value.get(
                            "action_probability_delta", {}
                        ),
                        "l1_policy_effect": value.get(
                            "l1_policy_effect", 0.0
                        ),
                        "action_change_rate": value.get(
                            "action_change_rate", 0.0
                        ),
                    },
                    provenance=provenance,
                )
            )
    if bundle.why_not_recourse:
        selected_recourse = bundle.why_not_recourse.get("selected")
        records.append(
            EvidenceRecord(
                evidence_id="why_not_recourse",
                source_type="why_not_recourse",
                frame_id=bundle.query_plan.frame_reference,
                intervention=(
                    {
                        "items": (
                            selected_recourse or {}
                        ).get("interventions", ())
                    }
                    if isinstance(
                        selected_recourse, Mapping
                    )
                    else None
                ),
                observed_value={
                    key: bundle.why_not_recourse[key]
                    for key in (
                        "target_entity",
                        "desired_action",
                        "baseline_probability",
                        "achieved",
                        "selected_candidate_id",
                        "refusal_reason",
                    )
                    if key in bundle.why_not_recourse
                }
                | (
                    {"selected": dict(selected_recourse)}
                    if isinstance(selected_recourse, Mapping)
                    else {}
                ),
            )
        )
    if bundle.policy_results:
        policy_keys = (
            "target",
            "baseline_distribution",
            "edited_distribution",
            "causal_observable",
            "action_mask",
            "argmax_action",
            "proposed_action",
            "executed_action",
            "proposed_action_probability",
            "selection_mode",
            "environment_enforced_action_change",
            "environment_action_resolution",
        )
        records.append(
            EvidenceRecord(
                evidence_id="neural_policy",
                source_type="neural_policy",
                frame_id=bundle.query_plan.frame_reference,
                observed_value={
                    key: bundle.policy_results[key]
                    for key in policy_keys
                    if key in bundle.policy_results
                },
            )
        )
    records.extend(
        EvidenceRecord(
            evidence_id=f"program_{index}",
            source_type="program_trace",
            frame_id=bundle.query_plan.frame_reference,
            program_branch_id=str(value.get("branch_id", index)),
            observed_value=dict(value),
        )
        for index, value in enumerate(bundle.program_trace)
    )
    if bundle.disagreement.get("program_available"):
        records.append(
            EvidenceRecord(
                evidence_id="program_output",
                source_type="program_trace",
                frame_id=bundle.query_plan.frame_reference,
                observed_value=dict(bundle.disagreement),
                provenance={
                    "surrogate_only": True,
                    "program_reliable": bundle.disagreement.get(
                        "program_reliable"
                    ),
                },
            )
        )
    return tuple(records)


def _embedded_explanation_requirements(
    value: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Read an environment's generic semantic-fact interface.

    Environments decide which facts are relevant and describe them with
    semantic roles. This evaluator never knows domain-specific predicates,
    thresholds, objects, or natural-language templates.
    """

    if not isinstance(value, Mapping):
        return ()
    raw = value.get("explanation_requirements", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _canonical_assertion_value(observed: Any) -> Any:
    """Return the canonical value slot of an aligned evidence record."""

    if not isinstance(observed, Mapping):
        return observed
    if "value" in observed:
        aliases = observed.get("value_verbalizations", ())
        semantic_alias = (
            observed.get("relation", {}).get("semantic_value")
            if isinstance(observed.get("relation"), Mapping)
            else None
        )
        if aliases or semantic_alias is not None:
            return {
                "value": observed["value"],
                "aliases": [
                    *(
                        list(aliases)
                        if isinstance(aliases, Sequence)
                        and not isinstance(aliases, (str, bytes))
                        else ()
                    ),
                    *(
                        (semantic_alias,)
                        if semantic_alias is not None
                        else ()
                    ),
                ],
            }
        return observed["value"]
    for key in (
        "executed_action",
        "recorded_executed_action",
        "proposed_action",
        "recorded_proposed_action",
        "argmax_action",
        "observed_value",
        "result",
    ):
        if key in observed:
            return observed[key]
    return observed


def _numeric_values(value: Any) -> tuple[float, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Mapping):
        values: list[float] = []
        for key, item in value.items():
            if "delta" in str(key).lower() or "effect" in str(key).lower():
                values.extend(_all_numeric_values(item))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(number for item in value for number in _numeric_values(item))
    return ()


def _all_numeric_values(value: Any) -> tuple[float, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Mapping):
        return tuple(
            number for item in value.values() for number in _all_numeric_values(item)
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(number for item in value for number in _all_numeric_values(item))
    return ()


def _structured_value_matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, Mapping):
        if (
            "asserted_value" in expected
            and set(expected).issubset(
                {
                    "asserted_value",
                    "asserted_numeric_literals",
                }
            )
        ):
            asserted = expected["asserted_value"]
            canonical = _canonical_assertion_value(observed)
            if _structured_value_matches(asserted, canonical):
                return True
            asserted_numbers = _numeric_literals(str(asserted))
            observed_numbers = _all_numeric_values(canonical)
            return bool(
                asserted_numbers
                and len(asserted_numbers) == len(observed_numbers)
                and all(
                    any(
                        math.isclose(
                            value,
                            candidate,
                            rel_tol=1e-4,
                            abs_tol=1e-4,
                        )
                        for candidate in observed_numbers
                    )
                    for value in asserted_numbers
                )
            )
        if len(expected) == 1:
            asserted = next(iter(expected.values()))
            canonical = _canonical_assertion_value(observed)
            if _structured_value_matches(asserted, canonical):
                return True
        if (
            set(expected) == {"action"}
            and isinstance(observed, Mapping)
        ):
            return any(
                key in observed
                and _structured_value_matches(
                    expected["action"],
                    observed[key],
                )
                for key in (
                    "executed_action",
                    "recorded_executed_action",
                    "proposed_action",
                    "recorded_proposed_action",
                    "argmax_action",
                )
            )
        if (
            "predicate" in expected
            and "arguments" in expected
            and isinstance(observed, Mapping)
        ):
            if not _predicate_reference_matches(
                {
                    "predicate": expected["predicate"],
                    "arguments": expected["arguments"],
                },
                observed,
            ):
                return False
            remaining = {
                key: value
                for key, value in expected.items()
                if key not in {"predicate", "arguments"}
            }
            if not remaining:
                return True
            return all(
                key in observed
                and _structured_value_matches(
                    value,
                    observed[key],
                )
                for key, value in remaining.items()
            )
        if (
            "operator" in expected
            and "value" in expected
        ):
            return _numeric_relation_matches(
                expected,
                observed,
            )
        if "predicate" in expected and set(expected).issubset(
            {"predicate", "arguments"}
        ):
            return _predicate_reference_matches(
                expected,
                observed,
            )
        if isinstance(observed, Mapping):
            if all(
                key in observed
                and _structured_value_matches(value, observed[key])
                for key, value in expected.items()
            ):
                return True
            return any(
                _structured_value_matches(expected, value)
                for value in observed.values()
            )
        if isinstance(observed, Sequence) and not isinstance(
            observed, (str, bytes)
        ):
            return any(
                _structured_value_matches(expected, value)
                for value in observed
            )
        return False
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes)
    ):
        if not isinstance(observed, Sequence) or isinstance(
            observed, (str, bytes)
        ):
            return False
        return len(expected) == len(observed) and all(
            _structured_value_matches(left, right)
            for left, right in zip(expected, observed)
        )
    if isinstance(observed, Mapping):
        return any(_structured_value_matches(expected, value) for value in observed.values())
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        return any(_structured_value_matches(expected, value) for value in observed)
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(expected), float(observed), rel_tol=1e-4, abs_tol=1e-4)
    expected_number = _scalar_number(expected)
    observed_number = _scalar_number(observed)
    if expected_number is not None and observed_number is not None:
        return math.isclose(
            expected_number,
            observed_number,
            rel_tol=1e-4,
            abs_tol=1e-4,
        )
    return str(expected).strip().casefold() == str(observed).strip().casefold()


def _scalar_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*[^\d]*",
        value,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _assertions_cover_claim_literals(
    claim: AtomicClaim,
    asserted_values: Sequence[Any],
) -> bool:
    """Reject alignments that silently drop a literal from the claim."""

    claim_numbers = _claim_numeric_literals(
        claim.text,
        claim.entities,
    )
    if not claim_numbers:
        return True
    asserted_numbers = _asserted_numeric_values(asserted_values)
    return all(
        any(
            math.isclose(
                expected,
                observed,
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
            for observed in asserted_numbers
        )
        for expected in claim_numbers
    )


def _asserted_numeric_values(value: Any) -> tuple[float, ...]:
    scalar = _scalar_number(value)
    if scalar is not None:
        return (scalar,)
    if isinstance(value, Mapping):
        return tuple(
            number
            for item in value.values()
            for number in _asserted_numeric_values(item)
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        return tuple(
            number
            for item in value
            for number in _asserted_numeric_values(item)
        )
    return ()


def _predicate_reference_matches(
    expected: Mapping[str, Any],
    observed: Any,
) -> bool:
    """Match a predicate and multilingual entity/value references."""

    if not isinstance(observed, Mapping):
        return False
    if str(observed.get("predicate", "")) != str(
        expected.get("predicate", "")
    ):
        return any(
            _predicate_reference_matches(expected, value)
            for value in observed.values()
            if isinstance(value, Mapping)
        )
    expected_arguments = expected.get("arguments", ())
    if not isinstance(expected_arguments, Sequence) or isinstance(
        expected_arguments,
        (str, bytes),
    ):
        return False
    observed_arguments = observed.get("arguments", ())
    if not isinstance(observed_arguments, Sequence) or isinstance(
        observed_arguments,
        (str, bytes),
    ):
        return not expected_arguments
    if len(expected_arguments) != len(observed_arguments):
        return False
    verbalizations = tuple(
        str(value).casefold()
        for key in ("verbalizations", "value_verbalizations")
        for value in observed.get(key, ())
    )
    return all(
        _reference_matches(
            str(expected_value),
            str(observed_value),
            verbalizations,
        )
        for expected_value, observed_value in zip(
            expected_arguments,
            observed_arguments,
        )
    )


def _reference_matches(
    expected: str,
    observed: str,
    verbalizations: Sequence[str],
) -> bool:
    expected_normalized = expected.strip().casefold()
    observed_normalized = observed.strip().casefold()
    if expected_normalized == observed_normalized:
        return True
    expected_number = _last_integer(expected_normalized)
    observed_number = _last_integer(observed_normalized)
    if (
        expected_number is not None
        and observed_number is not None
        and expected_number == observed_number
    ):
        return True
    return bool(
        expected_normalized
        and any(
            expected_normalized in text
            for text in verbalizations
        )
    )


def _last_integer(text: str) -> int | None:
    digits = ""
    values: list[int] = []
    for character in text:
        if character.isdigit():
            digits += character
        elif digits:
            values.append(int(digits))
            digits = ""
    if digits:
        values.append(int(digits))
    return values[-1] if values else None


def _numeric_relation_matches(
    expected: Mapping[str, Any],
    observed: Any,
) -> bool:
    """Evaluate a structured numeric claim against an execution record."""

    operator = str(expected.get("operator", "")).strip()
    try:
        target = float(expected["value"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_feature = (
        str(expected["feature"])
        if expected.get("feature") is not None
        else None
    )
    exact_value = expected.get("exact_value")
    try:
        exact_numeric = (
            float(exact_value)
            if exact_value is not None
            else None
        )
    except (TypeError, ValueError):
        return False
    candidates = _observed_numeric_feature_values(observed)
    for feature, value in candidates:
        if (
            expected_feature is not None
            and feature != expected_feature
        ):
            continue
        if exact_numeric is not None and not math.isclose(
            value,
            exact_numeric,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            continue
        if operator == ">" and value > target:
            return True
        if operator == ">=" and value >= target:
            return True
        if operator == "<" and value < target:
            return True
        if operator == "<=" and value <= target:
            return True
        if operator in {"=", "=="} and math.isclose(
            value,
            target,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            return True
        if operator == "!=" and not math.isclose(
            value,
            target,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            return True
    return False


def _observed_numeric_feature_values(
    observed: Any,
) -> tuple[tuple[str | None, float], ...]:
    values: list[tuple[str | None, float]] = []
    if isinstance(observed, Mapping):
        raw_value = observed.get("observed_value")
        if (
            isinstance(raw_value, (int, float))
            and not isinstance(raw_value, bool)
        ):
            values.append(
                (
                    (
                        str(observed["feature"])
                        if observed.get("feature") is not None
                        else None
                    ),
                    float(raw_value),
                )
            )
        for value in observed.values():
            if isinstance(value, (Mapping, list, tuple)):
                values.extend(
                    _observed_numeric_feature_values(value)
                )
    elif isinstance(observed, Sequence) and not isinstance(
        observed,
        (str, bytes),
    ):
        for value in observed:
            values.extend(_observed_numeric_feature_values(value))
    return tuple(values)


def _verdict(
    claim: AtomicClaim,
    status: ClaimVerdictStatus,
    evidence: Sequence[EvidenceRecord],
    reason: str,
) -> ClaimVerdict:
    confidence = (
        min(1.0, max(0.0, claim.confidence))
        if status != ClaimVerdictStatus.UNVERIFIABLE
        else 0.0
    )
    return ClaimVerdict(
        claim=claim,
        status=status,
        evidence=tuple(evidence),
        confidence=confidence,
        verifier_reason=reason,
    )
