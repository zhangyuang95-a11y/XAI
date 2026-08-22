"""Independent Transformer claim extraction from explanation text only."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any, Mapping, Sequence

from .schemas import AtomicClaim
from .tokenizer import StructuredTransformerBackend

_CLAIM_TYPE_ALIASES = {
    "environment_override": "environment_constraint",
    "environmental_constraint": "environment_constraint",
    "environmental_factor": "state",
    "policy_action": "action",
    "action_selection": "action",
}

class TransformerAtomicClaimExtractor:
    """Extract claims after generation from final answer text only."""

    def __init__(
        self,
        backend: StructuredTransformerBackend,
        *,
        minimum_confidence: float = 0.0,
    ) -> None:
        if backend is None:
            raise ValueError("A Transformer backend is required for claim extraction.")
        self.backend = backend
        self.minimum_confidence = float(minimum_confidence)
        self.last_coverage_warning: str | None = None

    def extract(self, explanation_text: str) -> tuple[AtomicClaim, ...]:
        self.last_coverage_warning = None
        if not explanation_text.strip():
            return ()
        prompt = _atomic_claim_prompt(explanation_text)
        previous_payload: Mapping[str, Any] | None = None
        coverage_issues: tuple[str, ...] = ()
        claims: tuple[AtomicClaim, ...] = ()
        accumulated: dict[str, AtomicClaim] = {}
        for _attempt in range(2):
            repair_context = (
                ""
                if previous_payload is None
                else (
                    "\nYour previous extraction was incomplete. Return a "
                    "replacement claims array covering the entire final "
                    "answer, not a patch.\nCoverage issues: "
                    f"{list(coverage_issues)}\nPrevious extraction: "
                    f"{previous_payload}"
                )
            )
            payload = self.backend.generate_json(
                prompt + repair_context,
                schema_name="AtomicClaimList",
                max_new_tokens=512,
            )
            extracted = _decode_atomic_claim_payload(
                payload,
                minimum_confidence=self.minimum_confidence,
            )
            for claim in extracted:
                accumulated.setdefault(claim.text, claim)
            used_ids: set[str] = set()
            merged_claims: list[AtomicClaim] = []
            for index, claim in enumerate(
                accumulated.values(),
                start=1,
            ):
                claim_id = claim.claim_id
                if not claim_id or claim_id in used_ids:
                    claim_id = f"claim_{index}"
                used_ids.add(claim_id)
                merged_claims.append(
                    (
                        claim
                        if claim_id == claim.claim_id
                        else replace(claim, claim_id=claim_id)
                    )
                )
            claims = tuple(merged_claims)
            coverage_issues = _claim_extraction_coverage_issues(
                explanation_text,
                claims,
            )
            if not coverage_issues:
                return claims
            previous_payload = payload
        self.last_coverage_warning = (
            "Transformer claim extraction did not cover the complete final "
            f"answer: {list(coverage_issues)}"
        )
        missing_segments = _uncovered_explanation_segments(
            explanation_text,
            claims,
        )
        for segment in missing_segments:
            try:
                payload = self.backend.generate_json(
                    _atomic_claim_prompt(
                        segment,
                        focused=True,
                    ),
                    schema_name="AtomicClaimList",
                    max_new_tokens=384,
                )
                for claim in _decode_atomic_claim_payload(
                    payload,
                    minimum_confidence=self.minimum_confidence,
                ):
                    accumulated.setdefault(claim.text, claim)
            except (ValueError, RuntimeError, TypeError):
                continue
        claims = _deduplicate_claim_ids(
            tuple(accumulated.values())
        )
        final_issues = _claim_extraction_coverage_issues(
            explanation_text,
            claims,
        )
        self.last_coverage_warning = (
            None
            if not final_issues
            else (
                "Transformer claim extraction did not cover the complete "
                f"final answer: {list(final_issues)}"
            )
        )
        return claims

def _atomic_claim_prompt(
    explanation_text: str,
    *,
    focused: bool = False,
) -> str:
    """Build a compact, domain-neutral post-hoc extraction instruction."""

    scope = (
        "This passage was missed earlier. Extract all of it."
        if focused
        else "Process the answer from beginning to end; do not stop early."
    )
    return (
        "Read only the displayed answer passage and return JSON with a claims "
        "array. Split every independently checkable proposition into a minimal "
        "claim. Claim text must be an exact contiguous span copied from the "
        "passage. Do not add an entity, number, action or language absent from "
        f"the passage. {scope}\n"
        "Every item needs claim_id, text, claim_type, entities, frame_scope, "
        "time_scope, predicate, non-empty expected_outcome, modality and "
        "confidence. Every entity string must be an exact contiguous span "
        "copied from the claim text; do not translate or canonicalize entity "
        "mentions because a later stage resolves them. Preserve canonical "
        "action identifiers only when the passage itself contains them.\n"
        "Use claim_type causal only when the passage explicitly describes an "
        "intervention, hypothetical edit, counterfactual comparison or paired "
        "cause-effect test. Words meaning 'because' or 'therefore' in an "
        "explanation of the recorded current state are NOT enough for causal; "
        "use state instead. A statement that the recorded action changed a "
        "measured distance or state is also state/relation, not causal, unless "
        "the passage explicitly compares an alternative intervention. "
        "Do not label current-frame facts as future or "
        "trajectory. Use action for a directly observed or proposed action. "
        "Use program only when the passage asserts how a particular action "
        "changes progress toward an objective according to the extracted "
        "policy program; use state for the objective and its current factual "
        "context.\n"
        "Use a domain-neutral expected_outcome that records only what the "
        "passage asserts. A scalar fact may use "
        "{\"asserted_value\":...}; a comparison may use "
        "{\"left\":...,\"operator\":\"<\", \"right\":...}; a coordinate or "
        "other ordered value may use an array. The later evidence-alignment "
        "stage will map these assertions to environment-defined predicates. "
        "For a negated property statement, asserted_value must represent the "
        "negated property value (normally false), not true merely because the "
        "sentence itself is being asserted. "
        "Do not invent an environment predicate, field name, threshold, or "
        "causal relation. Preserve every number in some claim text and in its "
        "expected_outcome.\n"
        f"Displayed answer passage:\n{explanation_text}"
    )


def _deduplicate_claim_ids(
    claims: Sequence[AtomicClaim],
) -> tuple[AtomicClaim, ...]:
    """Keep model IDs when unique and repair only collisions."""

    used_ids: set[str] = set()
    result: list[AtomicClaim] = []
    for index, claim in enumerate(claims, start=1):
        claim_id = claim.claim_id
        if not claim_id or claim_id in used_ids:
            claim_id = f"claim_{index}"
        used_ids.add(claim_id)
        result.append(
            (
                claim
                if claim_id == claim.claim_id
                else replace(claim, claim_id=claim_id)
            )
        )
    return tuple(result)


def _decode_atomic_claim_payload(
    payload: Mapping[str, Any],
    *,
    minimum_confidence: float,
) -> tuple[AtomicClaim, ...]:
    """Validate and decode one post-hoc Transformer claim payload."""

    raw_claims = payload.get("claims", ())
    if not isinstance(raw_claims, Sequence) or isinstance(
        raw_claims, (str, bytes)
    ):
        raise ValueError("Transformer claim output must contain a claims array.")
    claims: list[AtomicClaim] = []
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, Mapping):
            raise ValueError("Each extracted claim must be a JSON object.")
        raw_confidence = raw.get("confidence", 0.0)
        try:
            confidence = (
                0.0
                if raw_confidence is None
                else float(raw_confidence)
            )
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < minimum_confidence:
            continue
        claim_type = str(
            raw.get("claim_type", "other")
        ).lower()
        claim_text = str(raw.get("text", "")).strip()
        expected_outcome = _preserve_asserted_numbers(
            claim_text,
            _normalize_atomic_expected_outcome(
                claim_type,
                raw.get("expected_outcome"),
            ),
            entity_references=tuple(
                str(item) for item in raw.get("entities", ())
            ),
        )
        claim = AtomicClaim(
            claim_id=str(raw.get("claim_id", f"claim_{index + 1}")),
            text=claim_text,
            claim_type=_CLAIM_TYPE_ALIASES.get(
                claim_type,
                claim_type,
            ),
            entities=tuple(str(item) for item in raw.get("entities", ())),
            frame_scope=tuple(int(item) for item in raw.get("frame_scope", ())),
            time_scope=(
                str(raw["time_scope"])
                if raw.get("time_scope") is not None
                else None
            ),
            predicate=(
                str(raw["predicate"])
                if raw.get("predicate") is not None
                else None
            ),
            expected_outcome=expected_outcome,
            modality=(
                str(raw["modality"])
                if raw.get("modality") is not None
                else None
            ),
            confidence=confidence,
        )
        if claim.text:
            claims.append(claim)
    return tuple(claims)


def _normalize_atomic_expected_outcome(
    claim_type: str,
    expected_outcome: Any,
) -> Any:
    """Normalize domain-neutral action aliases emitted by small LMs."""

    if (
        claim_type == "action"
        and isinstance(expected_outcome, Mapping)
    ):
        normalized = dict(expected_outcome)
        if (
            "executed_action" not in normalized
            and "direction" in normalized
        ):
            normalized["executed_action"] = normalized.pop(
                "direction"
            )
        elif (
            "executed_action" not in normalized
            and set(normalized) == {"action"}
        ):
            normalized["executed_action"] = normalized.pop("action")
        return normalized
    return expected_outcome


def _preserve_asserted_numbers(
    claim_text: str,
    expected_outcome: Any,
    *,
    entity_references: Sequence[str] = (),
) -> Any:
    """Keep literal coverage without assigning environment semantics."""

    numbers = tuple(
        dict.fromkeys(
            _numeric_literal_strings(
                claim_text,
                entity_references=entity_references,
            )
        )
    )
    if not numbers:
        return expected_outcome
    serialized = json.dumps(
        expected_outcome,
        ensure_ascii=False,
        default=str,
    )
    # Compare complete numeric tokens, not substrings.  Otherwise a displayed
    # magnitude such as ``1`` is incorrectly treated as represented by a
    # model-emitted signed value ``-1``.  The later coverage audit correctly
    # distinguishes those values, so preservation must use the same rule.
    structured_numbers = set(_numeric_literal_strings(serialized))
    missing = tuple(
        number for number in numbers if number not in structured_numbers
    )
    if not missing:
        return expected_outcome
    normalized = (
        dict(expected_outcome)
        if isinstance(expected_outcome, Mapping)
        else {"asserted_value": expected_outcome}
    )
    normalized["asserted_numeric_literals"] = list(missing)
    return normalized


def _numeric_literal_strings(
    text: str,
    *,
    entity_references: Sequence[str] = (),
) -> tuple[str, ...]:
    """Extract values next to any language while ignoring entity IDs."""

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
        re.findall(
            r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?",
            sanitized,
        )
    )


def _claim_extraction_coverage_issues(
    explanation_text: str,
    claims: Sequence[AtomicClaim],
) -> tuple[str, ...]:
    """Require post-hoc claims to cover the complete displayed answer."""

    issues: list[str] = []
    claim_texts = tuple(claim.text.strip() for claim in claims if claim.text.strip())
    if not claim_texts:
        return ("no claims were extracted",)
    segments = tuple(
        value.strip()
        for value in re.split(r"(?<=[。！？.!?；;])\s*", explanation_text)
        if value.strip()
    )
    normalized_claims = tuple(
        re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()
        for value in claim_texts
    )
    for index, segment in enumerate(segments, start=1):
        normalized_segment = re.sub(
            r"[\W_]+",
            "",
            segment,
            flags=re.UNICODE,
        ).casefold()
        if not any(
            claim_text
            and (
                claim_text in normalized_segment
                or normalized_segment in claim_text
            )
            for claim_text in normalized_claims
        ):
            issues.append(
                f"sentence {index} has no extracted claim: {segment[:120]}"
            )
    answer_entity_references = tuple(
        dict.fromkeys(
            entity
            for claim in claims
            for entity in claim.entities
            if entity
        )
    )
    answer_numbers = set(
        _numeric_literal_strings(
            explanation_text,
            entity_references=answer_entity_references,
        )
    )
    claim_numbers = {
        number
        for claim in claims
        for number in _numeric_literal_strings(
            claim.text,
            # Entity aliases are sometimes attached to only one of several
            # extracted claims.  Reuse the union for every claim so the digit
            # in a repeated label such as “机器人1” is never mistaken for a
            # factual numeric assertion in another sentence.
            entity_references=tuple(
                dict.fromkeys(
                    (*answer_entity_references, *claim.entities)
                )
            ),
        )
    }
    missing_numbers = sorted(answer_numbers - claim_numbers)
    if missing_numbers:
        issues.append(
            "numeric literals absent from extracted claim text: "
            + ", ".join(missing_numbers)
        )
    for claim in claims:
        claim_entity_references = tuple(
            dict.fromkeys(
                (*answer_entity_references, *claim.entities)
            )
        )
        text_numbers = set(
            _numeric_literal_strings(
                claim.text,
                entity_references=claim_entity_references,
            )
        )
        structured_numbers = set(
            _numeric_literal_strings(
                json.dumps(
                    claim.expected_outcome,
                    ensure_ascii=False,
                    default=str,
                ),
                entity_references=claim_entity_references,
            )
        )
        omitted = sorted(text_numbers - structured_numbers)
        if omitted:
            issues.append(
                f"{claim.claim_id} combines numeric assertions not encoded "
                "in expected_outcome; split it into smaller claims or encode "
                "each asserted number: "
                + ", ".join(omitted)
            )
    return tuple(issues)


def _uncovered_explanation_segments(
    explanation_text: str,
    claims: Sequence[AtomicClaim],
) -> tuple[str, ...]:
    """Find only displayed sentences that need another Transformer pass."""

    segments = tuple(
        value.strip()
        for value in re.split(
            r"(?<=[。！？.!?；;])\s*",
            explanation_text,
        )
        if value.strip()
    )
    missing: list[str] = []
    for segment in segments:
        normalized_segment = re.sub(
            r"[\W_]+",
            "",
            segment,
            flags=re.UNICODE,
        ).casefold()
        aligned_claims = [
            claim
            for claim in claims
            if (
                re.sub(
                    r"[\W_]+",
                    "",
                    claim.text,
                    flags=re.UNICODE,
                ).casefold()
                in normalized_segment
            )
        ]
        segment_entity_references = tuple(
            dict.fromkeys(
                entity
                for claim in aligned_claims
                for entity in claim.entities
                if entity
            )
        )
        segment_numbers = set(
            _numeric_literal_strings(
                segment,
                entity_references=segment_entity_references,
            )
        )
        covered_numbers = {
            number
            for claim in aligned_claims
            for number in _numeric_literal_strings(
                claim.text,
                entity_references=claim.entities,
            )
        }
        if not aligned_claims or not segment_numbers.issubset(
            covered_numbers
        ):
            missing.append(segment)
    return tuple(missing)

# End of claim extraction helpers.
