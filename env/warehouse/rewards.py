"""Training-only reward settings for collaborative delivery."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import REWARD_VERSION


@dataclass(frozen=True)
class RewardConfig:
    """Training-only individual-credit safe-mission reward contract."""

    # Explicitly false only in the isolated v19 ablation runner.  Production
    # defaults and newly saved checkpoints always use individual credit.
    individual_credit_enabled: bool = True
    # One safe mission action removed is worth +0.01.  The old scale of two
    # made the dense potential larger than the delivered-task objective and
    # obscured which robot actually made the progress.
    progress_scale: float = 1.0
    # This is now an upper bound on a measured clearance manoeuvre, not a
    # fixed sixteen-step bonus.  See ``_coordination_delay_cost``.
    coordination_clearance_cost: float = 4.0
    coordination_progress_cap: float = 0.04
    task_age_priority_scale: float = 0.50
    task_age_priority_horizon: int = 40
    counterfactual_regret_cost: float = 0.02
    avoidable_wait_streak_cost: float = 0.01
    avoidable_wait_streak_cap: int = 4
    # Retained for explicit ablations and old command-line compatibility.
    # First-step WAIT regret is represented by counterfactual regret; applying
    # another flat cost here recreated the unsafe ``WAIT cost=0.03`` ablation.
    avoidable_wait_cost: float = 0.0
    # Individual frozen-goal progress and regret replace the old duplicated
    # global regression penalty.
    mission_regression_scale: float = 0.0
    version: str = REWARD_VERSION

    def __post_init__(self) -> None:
        if self.progress_scale < 0:
            raise ValueError("progress_scale cannot be negative.")
        if self.coordination_clearance_cost < 0:
            raise ValueError("coordination_clearance_cost cannot be negative.")
        if self.coordination_progress_cap < 0:
            raise ValueError("coordination_progress_cap cannot be negative.")
        if self.task_age_priority_scale < 0:
            raise ValueError("task_age_priority_scale cannot be negative.")
        if self.task_age_priority_horizon < 1:
            raise ValueError("task_age_priority_horizon must be positive.")
        if self.avoidable_wait_cost < 0:
            raise ValueError("avoidable_wait_cost cannot be negative.")
        if self.counterfactual_regret_cost < 0:
            raise ValueError("counterfactual_regret_cost cannot be negative.")
        if self.avoidable_wait_streak_cost < 0:
            raise ValueError("avoidable_wait_streak_cost cannot be negative.")
        if self.avoidable_wait_streak_cap < 0:
            raise ValueError("avoidable_wait_streak_cap cannot be negative.")
        if self.mission_regression_scale < 0:
            raise ValueError("mission_regression_scale cannot be negative.")
        if self.version != REWARD_VERSION:
            raise ValueError(
                f"Unsupported reward version {self.version!r}; "
                f"expected {REWARD_VERSION!r}. Retrain from a new checkpoint."
            )
