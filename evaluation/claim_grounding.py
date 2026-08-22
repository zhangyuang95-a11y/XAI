"""Transformer semantic alignment between extracted claims and simulator facts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from backend.nlp.schemas import (
    AtomicClaim,
    ClaimVerdict,
    ClaimVerdictStatus,
    EvidenceBundle,
    EvidenceRecord,
)
from backend.nlp.tokenizer import StructuredTransformerBackend


from .claim_alignment import (
    TransformerFactMatcher,
    _alignment_candidates,
    _assertions_cover_claim_literals,
    _bundle_records,
    _canonical_assertion_value,
    _claim_alignment_prompt,
    _claim_numeric_literals,
    _complete_alignment,
    _numeric_values,
    _semantic_evidence_supports_claim,
    _semantic_record_alignments,
    _short_error,
    _structured_value_matches,
    _verdict,
)


class ClaimGroundingEvaluator:
    """Independently align final-answer claims to execution evidence.

    The Transformer performs semantic alignment only. Status is assigned after
    alignment by deterministic evidence rules, and causal claims require paired
    neural baseline/counterfactual evidence.
    """

    def __init__(
        self,
        backend: StructuredTransformerBackend,
        *,
        causal_effect_threshold: float = 1e-6,
        semantic_matcher: TransformerFactMatcher | None = None,
        semantic_support_threshold: float = 0.22,
    ) -> None:
        if backend is None:
            raise ValueError("A Transformer backend is required for claim grounding.")
        self.backend = backend
        self.causal_effect_threshold = float(causal_effect_threshold)
        self.semantic_matcher = semantic_matcher
        self.semantic_support_threshold = float(
            semantic_support_threshold
        )
        self.last_alignment_warning: str | None = None

    def evaluate(
        self,
        claims: Sequence[AtomicClaim],
        evidence: EvidenceBundle,
    ) -> tuple[ClaimVerdict, ...]:
        records = _bundle_records(evidence)
        self.last_alignment_warning = None
        if not claims:
            return ()
        query_entities = (
            *evidence.query_plan.subjects,
            *evidence.query_plan.referenced_entities,
        )
        alignment_errors: list[str] = []
        raw_alignments: list[Any] = []
        semantic_scores: dict[str, dict[str, float]] = {}
        semantic_alignment: dict[str, tuple[str, ...]] = {}
        if self.semantic_matcher is not None:
            semantic_alignment, semantic_scores = (
                _semantic_record_alignments(
                    claims,
                    _alignment_candidates(
                        records,
                        claims,
                        query_entities=query_entities,
                    ),
                    self.semantic_matcher,
                )
            )
        else:
            # Small instruction-tuned models frequently stop after four or
            # five array items. Bounded batches make completeness independent
            # of that model-specific output-length behavior.
            for start in range(0, len(claims), 4):
                claim_batch = tuple(claims[start : start + 4])
                prompt = _claim_alignment_prompt(
                    claim_batch,
                    _alignment_candidates(
                        records,
                        claim_batch,
                        query_entities=query_entities,
                    ),
                )
                try:
                    payload = self.backend.generate_json(
                        prompt,
                        schema_name="ClaimEvidenceAlignment",
                        max_new_tokens=448,
                    )
                except (ValueError, RuntimeError, TypeError) as exc:
                    alignment_errors.append(
                        f"{type(exc).__name__}: {_short_error(exc)}"
                    )
                    continue
                values = payload.get("alignments", ())
                if isinstance(values, Sequence) and not isinstance(
                    values,
                    (str, bytes),
                ):
                    raw_alignments.extend(values)
        alignment_error = (
            "Transformer claim-evidence alignment was unavailable for one "
            f"or more batches ({'; '.join(alignment_errors)}). Only "
            "conservative type-mandated evidence was used for those claims."
            if alignment_errors
            else None
        )
        if alignment_error is not None:
            self.last_alignment_warning = alignment_error
        alignment_map: dict[str, tuple[str, ...]] = {}
        assertion_map: dict[str, dict[str, Any]] = {}
        for item in raw_alignments:
            if isinstance(item, Mapping):
                claim_id = str(item.get("claim_id", ""))
                assertions = item.get("evidence_assertions", ())
                assertion_values: dict[str, Any] = {}
                if isinstance(assertions, Sequence) and not isinstance(
                    assertions,
                    (str, bytes),
                ):
                    for assertion in assertions:
                        if not isinstance(assertion, Mapping):
                            continue
                        evidence_id = str(
                            assertion.get("evidence_id", "")
                        )
                        if evidence_id and "claim_value" in assertion:
                            assertion_values[evidence_id] = assertion[
                                "claim_value"
                            ]
                listed_ids = item.get("evidence_ids", ())
                if not isinstance(listed_ids, Sequence) or isinstance(
                    listed_ids,
                    (str, bytes),
                ):
                    listed_ids = ()
                alignment_map[claim_id] = tuple(
                    dict.fromkeys(
                        (
                            *assertion_values,
                            *(str(value) for value in listed_ids),
                        )
                    )
                )
                assertion_map[claim_id] = assertion_values
        by_id = {record.evidence_id: record for record in records}
        verdicts = tuple(
            self._verify(
                claim,
                _complete_alignment(
                    claim,
                    tuple(
                        by_id[evidence_id]
                        for evidence_id in (
                            semantic_alignment.get(
                                claim.claim_id,
                                (),
                            )
                            if self.semantic_matcher is not None
                            else alignment_map.get(
                                claim.claim_id,
                                (),
                            )
                        )
                        if evidence_id in by_id
                    ),
                    records,
                ),
                evidence,
                assertion_map.get(claim.claim_id, {}),
                semantic_scores.get(claim.claim_id, {}),
            )
            for claim in claims
        )
        if alignment_error is None:
            return verdicts
        return tuple(
            replace(
                verdict,
                verifier_reason=(
                    f"{verdict.verifier_reason} Semantic alignment was "
                    "unavailable; conservative type-mandated evidence "
                    "was used."
                ).strip(),
            )
            for verdict in verdicts
        )

    def _verify(
        self,
        claim: AtomicClaim,
        selected: tuple[EvidenceRecord, ...],
        bundle: EvidenceBundle,
        asserted_values: Mapping[str, Any],
        semantic_scores: Mapping[str, float],
    ) -> ClaimVerdict:
        claim_type = claim.claim_type.lower()
        if claim_type == "causal":
            observable = bundle.policy_results.get("causal_observable", True)
            baseline = [item for item in selected if item.source_type == "baseline_rollout"]
            counterfactual = [
                item for item in selected if item.source_type == "counterfactual_rollout"
            ]
            if counterfactual:
                observable = bool(observable) and all(
                    bool(
                        item.provenance.get(
                            "observable_by_actor", observable
                        )
                    )
                    for item in counterfactual
                )
            deltas = [
                abs(float(value))
                for item in counterfactual
                for value in _numeric_values(item.observed_value)
            ]
            if not observable or not baseline or not counterfactual:
                return _verdict(
                    claim,
                    ClaimVerdictStatus.UNVERIFIABLE,
                    selected,
                    "Causal claims require an observable cause and paired neural-policy branches.",
                )
            if max(deltas, default=0.0) <= self.causal_effect_threshold:
                return _verdict(
                    claim,
                    ClaimVerdictStatus.CONTRADICTED,
                    selected,
                    "The paired intervention produced no measurable policy effect.",
                )
            return _verdict(
                claim,
                ClaimVerdictStatus.SUPPORTED,
                selected,
                "A paired observable intervention changed the neural-policy result.",
            )

        required_sources = {
            "counterfactual": ("counterfactual_rollout",),
            "future": ("counterfactual_rollout",),
            "trajectory": ("counterfactual_rollout",),
            "program": ("program_trace",),
            "action": ("neural_policy",),
            # A factual sentence may describe state, an observed action, or a
            # value computed by the extracted program. The semantic matcher,
            # literal checks, and exact record provenance determine which.
            "state": ("state", "neural_policy", "program_trace"),
            "relation": (
                "state",
                "neural_policy",
                "program_trace",
            ),
            "environment_constraint": ("neural_policy",),
        }.get(claim_type)
        if claim_type == "program" and bundle.disagreement.get("program_reliable") is False:
            return _verdict(
                claim,
                ClaimVerdictStatus.UNVERIFIABLE,
                selected,
                "The extracted program exceeds the configured neural-policy disagreement limit.",
            )
        relevant = (
            tuple(
                item
                for item in selected
                if item.source_type in required_sources
            )
            if required_sources
            else selected
        )
        if not relevant:
            return _verdict(
                claim,
                ClaimVerdictStatus.UNVERIFIABLE,
                selected,
                "No independently aligned execution evidence of the required type is available.",
            )
        if (
            claim.expected_outcome is None
            or (
                isinstance(claim.expected_outcome, Mapping)
                and not claim.expected_outcome
            )
        ):
            if claim_type not in {
                "action",
                "program",
                "uncertainty",
            }:
                return _verdict(
                    claim,
                    ClaimVerdictStatus.UNVERIFIABLE,
                    relevant,
                    "This factual claim has no structured expected outcome to compare with execution evidence.",
                )
            return _verdict(
                claim,
                ClaimVerdictStatus.SUPPORTED,
                relevant,
                "The aligned execution record contains the asserted evidence type.",
            )
        consistent = any(
            _structured_value_matches(claim.expected_outcome, item.observed_value)
            for item in relevant
        )
        normalized_assertions = tuple(
            (
                item,
                asserted_values[item.evidence_id],
            )
            for item in relevant
            if item.evidence_id in asserted_values
        )
        assertions_consistent = (
            bool(normalized_assertions)
            and _assertions_cover_claim_literals(
                claim,
                tuple(
                    asserted_value
                    for _item, asserted_value in normalized_assertions
                ),
            )
            and all(
            _structured_value_matches(
                asserted_value,
                _canonical_assertion_value(item.observed_value),
            )
            for item, asserted_value in normalized_assertions
            )
        )
        semantic_consistent = (
            bool(semantic_scores)
            and _semantic_evidence_supports_claim(
                claim,
                relevant,
                semantic_scores,
                threshold=self.semantic_support_threshold,
            )
        )
        consistent = (
            consistent
            or assertions_consistent
            or semantic_consistent
        )
        return _verdict(
            claim,
            (
                ClaimVerdictStatus.SUPPORTED
                if consistent
                else ClaimVerdictStatus.CONTRADICTED
            ),
            relevant,
            (
                "The structured expected outcome matches execution evidence."
                if consistent
                else "The structured expected outcome conflicts with execution evidence."
            ),
        )


@dataclass(frozen=True)
class ClaimEvaluationSummary:
    total: int
    support_rate: float
    contradiction_rate: float
    unverifiable_rate: float
    evidence_coverage: float
    unsupported_statement_rate: float
    mean_confidence: float
    counterfactual_correctness: float | None = None
    program_actor_disagreement_detection_rate: float | None = None
    abstention_accuracy: float | None = None
    calibration_error: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total": self.total,
            "support_rate": self.support_rate,
            "contradiction_rate": self.contradiction_rate,
            "unverifiable_rate": self.unverifiable_rate,
            "evidence_coverage": self.evidence_coverage,
            "unsupported_statement_rate": self.unsupported_statement_rate,
            "mean_confidence": self.mean_confidence,
            "counterfactual_correctness": self.counterfactual_correctness,
            "program_actor_disagreement_detection_rate": (
                self.program_actor_disagreement_detection_rate
            ),
            "abstention_accuracy": self.abstention_accuracy,
            "calibration_error": self.calibration_error,
        }


def summarize_claim_verdicts(
    verdicts: Sequence[ClaimVerdict],
    *,
    reference_statuses: Mapping[
        str, ClaimVerdictStatus | str
    ] | None = None,
    program_disagreement_present: bool | None = None,
    program_disagreement_detected: bool | None = None,
) -> ClaimEvaluationSummary:
    """Aggregate post-generation verdicts.

    Metrics that genuinely require reference outcomes remain ``None`` unless
    callers provide independently collected references. This avoids presenting
    self-consistency as correctness.
    """

    total = len(verdicts)
    denominator = max(1, total)
    supported = sum(
        item.status == ClaimVerdictStatus.SUPPORTED for item in verdicts
    )
    contradicted = sum(
        item.status == ClaimVerdictStatus.CONTRADICTED for item in verdicts
    )
    unverifiable = sum(
        item.status == ClaimVerdictStatus.UNVERIFIABLE for item in verdicts
    )
    covered = sum(bool(item.evidence) for item in verdicts)
    references = {
        str(claim_id): (
            status
            if isinstance(status, ClaimVerdictStatus)
            else ClaimVerdictStatus(str(status))
        )
        for claim_id, status in (reference_statuses or {}).items()
    }
    referenced = [
        item
        for item in verdicts
        if item.claim.claim_id in references
    ]
    counterfactual_referenced = [
        item
        for item in referenced
        if item.claim.claim_type.lower()
        in {"counterfactual", "causal", "future", "trajectory"}
    ]
    abstention_referenced = [
        item
        for item in referenced
        if references[item.claim.claim_id]
        == ClaimVerdictStatus.UNVERIFIABLE
    ]
    calibration_values = [
        abs(
            float(item.confidence)
            - float(
                item.status
                == references[item.claim.claim_id]
            )
        )
        for item in referenced
    ]
    return ClaimEvaluationSummary(
        total=total,
        support_rate=supported / denominator,
        contradiction_rate=contradicted / denominator,
        unverifiable_rate=unverifiable / denominator,
        evidence_coverage=covered / denominator,
        unsupported_statement_rate=(contradicted + unverifiable) / denominator,
        mean_confidence=(
            sum(float(item.confidence) for item in verdicts) / denominator
        ),
        counterfactual_correctness=(
            sum(
                item.status
                == references[item.claim.claim_id]
                for item in counterfactual_referenced
            )
            / len(counterfactual_referenced)
            if counterfactual_referenced
            else None
        ),
        program_actor_disagreement_detection_rate=(
            float(
                bool(program_disagreement_detected)
                == bool(program_disagreement_present)
            )
            if program_disagreement_present is not None
            and program_disagreement_detected is not None
            else None
        ),
        abstention_accuracy=(
            sum(
                item.status
                == ClaimVerdictStatus.UNVERIFIABLE
                for item in abstention_referenced
            )
            / len(abstention_referenced)
            if abstention_referenced
            else None
        ),
        calibration_error=(
            sum(calibration_values) / len(calibration_values)
            if calibration_values
            else None
        ),
    )

# End of claim-grounding helpers.
