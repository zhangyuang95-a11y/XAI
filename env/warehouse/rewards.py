"""Training-only reward settings for collaborative delivery."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import REWARD_VERSION


@dataclass(frozen=True)
class RewardConfig:
    """Training-only safe-mission shaping contract."""

    progress_scale: float = 2.0
    coordination_clearance_cost: float = 16.0
    task_age_priority_scale: float = 0.50
    task_age_priority_horizon: int = 40
    avoidable_wait_cost: float = 0.01
    mission_regression_scale: float = 1.0
    version: str = REWARD_VERSION

    def __post_init__(self) -> None:
        if self.progress_scale < 0:
            raise ValueError("progress_scale cannot be negative.")
        if self.coordination_clearance_cost < 0:
            raise ValueError("coordination_clearance_cost cannot be negative.")
        if self.task_age_priority_scale < 0:
            raise ValueError("task_age_priority_scale cannot be negative.")
        if self.task_age_priority_horizon < 1:
            raise ValueError("task_age_priority_horizon must be positive.")
        if self.avoidable_wait_cost < 0:
            raise ValueError("avoidable_wait_cost cannot be negative.")
        if self.mission_regression_scale < 0:
            raise ValueError("mission_regression_scale cannot be negative.")
        if self.version != REWARD_VERSION:
            raise ValueError(
                f"Unsupported reward version {self.version!r}; "
                f"expected {REWARD_VERSION!r}. Retrain from a new checkpoint."
            )
