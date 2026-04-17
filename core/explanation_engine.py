"""Generic explanation-engine helpers.

The current Pac-Man demo keeps its domain-specific rendering in
``envs.pacman.explanation_engine``. This module stays environment-neutral and
models policy interpretation as an optional "lens". A new environment can start
with evidence-only explanations, then add a symbolic, score-based, logical, or
LLM-readable policy lens later without changing the core explanation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class HumanFactorProfile:
    """Human-factor profile for future adaptive explanations."""

    age_group: str = "general"
    education_level: str = "general"
    occupation_type: str = "general"
    domain_expertise: str = "novice"
    game_experience: str = "unknown"
    ai_literacy: str = "novice"
    language: str = "en"
    preferred_detail: str = "medium"
    preferred_modality: str = "text"


@dataclass(frozen=True)
class ExplanationEnvelope:
    """Environment-neutral container for rendered explanations."""

    explanation_text: str
    evidence_used: list[dict[str, Any]]
    validation: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyLensResult:
    """Optional policy-level support for an explanation.

    ``available=False`` means the explanation should be treated as evidence-only.
    This lets environments be useful before they have a symbolic policy.
    """

    kind: str = "none"
    available: bool = False
    supports_action: bool | None = None
    predicted_action: str | None = None
    alternative_action: str | None = None
    rule_text: str = ""
    rule_code: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    approximate_trace: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyLens(Protocol):
    """Protocol for optional policy interpreters.

    Examples include symbolic decision trees, action-score inspectors, policy
    graphs, logical policies, program/code policies, or LLM-readable summaries.
    """

    kind: str

    def analyze(
        self,
        *,
        state: dict[str, Any],
        chosen_action: str,
        available_actions: tuple[str, ...],
        question: Any,
    ) -> PolicyLensResult:
        ...


class EvidenceOnlyPolicyLens:
    """Default no-op lens used when no policy interpreter is available."""

    kind = "none"

    def analyze(
        self,
        *,
        state: dict[str, Any],
        chosen_action: str,
        available_actions: tuple[str, ...],
        question: Any,
    ) -> PolicyLensResult:
        del state, chosen_action, available_actions, question
        return PolicyLensResult()
