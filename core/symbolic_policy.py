"""Generic symbolic-policy interfaces.

Environment adapters own feature extraction, rollout collection, and
domain-specific language. The core only defines small reusable data shapes and
protocols that future environments can implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: tuple[str, ...]
    label_en: str
    label_zh: str = ""


@dataclass
class SymbolicAnalysis:
    chosen_action: str
    predicted_action: str
    alternative_action: str | None
    symbolic_match: bool
    symbolic_support: bool
    chosen_score: float
    alternative_score: float | None
    scores: dict[str, float]
    chosen_trace: list[dict[str, Any]]
    predicted_trace: list[dict[str, Any]]
    chosen_rule: str
    chosen_rule_python: str
    comparison: dict[str, Any]


class SymbolicPolicy(Protocol):
    """Protocol implemented by concrete symbolic-policy adapters."""

    metrics: dict[str, float]
    metadata: dict[str, object]

    def validate_compatibility(self, checkpoint_metadata: dict[str, object]) -> None:
        ...

    def predict_action(self, state: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
        ...

    def get_policy_summary(self, lang: str) -> dict[str, object]:
        ...

    def analyze_state(
        self,
        state: dict[str, Any],
        chosen_action: str,
        lang: str,
        requested_alternative: str | None = None,
    ) -> SymbolicAnalysis:
        ...

    def save(self, path: Path) -> None:
        ...

