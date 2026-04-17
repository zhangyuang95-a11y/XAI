"""Generic evidence primitives shared by environment adapters.

Concrete environments should define their own recorder and rich record type,
then expose evidence as plain dictionaries/factors for the core explanation
pipeline. This module intentionally avoids importing any environment package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceSource:
    """A small, environment-neutral evidence item."""

    name: str
    value: Any
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GenericEvidenceRecord:
    """Minimal shape for environment adapters that want a core-level record."""

    state_snapshot: dict[str, Any]
    step: int
    chosen_action: str
    available_actions: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)

