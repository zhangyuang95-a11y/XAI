"""Environment-neutral policy output contracts.

This module is deliberately below adapters, environments, training and UI.
Runtime policies can therefore expose auditable action distributions without
depending on any of those higher-level packages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionDistribution:
    """A policy's complete output for one agent at one decision frame."""

    agent_id: str
    actions: tuple[str, ...]
    probabilities: tuple[float, ...]
    logits: tuple[float, ...] = ()
    action_mask: tuple[float, ...] = ()
    proposed_action: str | None = None

    @property
    def argmax_action(self) -> str:
        if not self.actions:
            raise ValueError("Action distribution is empty.")
        return self.actions[
            max(range(len(self.actions)), key=self.probabilities.__getitem__)
        ]
