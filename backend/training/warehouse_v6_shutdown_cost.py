"""Complete-episode, real-event Warehouse shutdown risk PPO.

This module never selects an environment action. The only observed cost is
whether a real environment episode ended in shutdown.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import comb, isfinite
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


def shutdown_cost_gae(
    costs: Sequence[float], values: Sequence[float], *, complete: bool,
    bootstrap: float = 0.0, gamma: float = 1.0, gae_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true episode-event cost-to-go minus the critic baseline.

    Costs are zero except for an optional one at the terminal transition.
    An incomplete fragment has no true 0/1 outcome and is rejected. Gamma
    and GAE lambda default to one because delaying failure must not make an
    episode shutdown cheaper than an early one.
    """
    if not costs or len(costs) != len(values):
        raise ValueError("costs and values must be nonempty and equal length")
    if gamma != 1.0 or gae_lambda != 1.0:
        raise ValueError("invalid discount or GAE lambda")
    if any((not isfinite(c)) or c not in (0.0, 1.0) for c in costs):
        raise ValueError("shutdown cost must be a real binary event")
    if any(not isfinite(v) for v in values) or not isfinite(bootstrap):
        raise ValueError("nonfinite cost value")
    if any(costs[:-1]):
        raise ValueError("shutdown can only occur at a completed episode end")
    if not complete or bootstrap != 0.0:
        raise ValueError("shutdown cost requires a complete, unbootstrapped episode")

    advantage = [0.0] * len(costs)
    future_value = 0.0
    future_advantage = 0.0
    for t in range(len(costs) - 1, -1, -1):
        delta = costs[t] + gamma * future_value - values[t]
        future_advantage = delta + gamma * gae_lambda * future_advantage
        advantage[t] = future_advantage
        future_value = values[t]

    returns = [0.0] * len(costs)
    future_return = 0.0
    for t in range(len(costs) - 1, -1, -1):
        future_return = costs[t] + future_return
        returns[t] = future_return
    return (torch.tensor(advantage, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32))


def joint_shutdown_surrogate(
    new_logp: torch.Tensor, old_logp: torch.Tensor,
    cost_advantage: torch.Tensor, episode_id: torch.Tensor,
    complete_episode_ids: Sequence[int], *, clip: float = 0.2,
    reference_horizon: int = 120,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO upper surrogate for a shared event under a product of two Actors.

    The joint ratio is exp(sum_r (new_logp_r - old_logp_r)); reward PPO may
    retain its separate per-role ratios.  Complete episodes are averaged as
    trajectories, so a short shutdown episode is not diluted by long safe
    episodes in a flattened transition mean.  Dividing by the fixed horizon
    keeps the numerical loss scale comparable across run lengths.
    """
    if (new_logp.ndim != 2 or new_logp.shape[-1] != 2
            or old_logp.shape != new_logp.shape
            or cost_advantage.shape != new_logp.shape[:1]
            or episode_id.shape != new_logp.shape[:1]):
        raise ValueError("expected [T,2] log probabilities and [T] cost/ids")
    if not (0.0 < clip < 1.0) or reference_horizon < 1:
        raise ValueError("invalid PPO clip or reference horizon")
    ids = tuple(int(item) for item in complete_episode_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate completed episode id")
    joint_log_ratio = (new_logp - old_logp).sum(dim=-1)
    ratio = joint_log_ratio.exp()
    per_step = torch.maximum(
        ratio * cost_advantage,
        ratio.clamp(1.0 - clip, 1.0 + clip) * cost_advantage,
    )
    episode_terms = []
    for identifier in ids:
        selected = episode_id == identifier
        if not bool(selected.any()):
            raise ValueError("completed episode id has no transitions")
        episode_terms.append(per_step[selected].sum() / reference_horizon)
    if episode_terms:
        surrogate = torch.stack(episode_terms).mean()
        selected_ratios = torch.cat([
            ratio[episode_id == identifier] for identifier in ids])
        joint_ratio_mean = float(selected_ratios.detach().mean())
    else:
        surrogate = new_logp.sum() * 0.0
        joint_ratio_mean = 0.0
    return surrogate, {
        "completed_episodes": len(episode_terms),
        "joint_ratio_mean": joint_ratio_mean,
        "cost_actor_surrogate": float(surrogate.detach()),
    }


def episode_balanced_cost_critic_loss(
    cost_logits: torch.Tensor, episode_id: torch.Tensor,
    complete_episode_outcomes: Mapping[int, bool],
) -> torch.Tensor:
    """Calibrate P(shutdown before episode end) from actual full outcomes.

    Each complete episode contributes one mean binary cross entropy, so an
    early shutdown is not underweighted simply because its trajectory ended
    sooner.  Incomplete fragments have no binary label and are excluded.
    The critic output used in GAE is sigmoid(cost_logits), not its raw logit.
    """
    if cost_logits.ndim != 1 or episode_id.shape != cost_logits.shape:
        raise ValueError("expected [T] cost logits and episode ids")
    losses = []
    for identifier, shutdown in complete_episode_outcomes.items():
        if not isinstance(identifier, int) or not isinstance(shutdown, bool):
            raise TypeError("episode outcome must map integer id to bool")
        selected = episode_id == identifier
        if not bool(selected.any()):
            raise ValueError("complete episode outcome has no transitions")
        labels = torch.full_like(cost_logits[selected], float(shutdown))
        losses.append(F.binary_cross_entropy_with_logits(
            cost_logits[selected], labels))
    return (torch.stack(losses).mean() if losses else
            cost_logits.sum() * 0.0)


def one_sided_binomial_upper(events: int, episodes: int,
                             confidence: float = 0.9) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a Bernoulli rate."""
    if episodes <= 0 or not 0 <= events <= episodes or not 0.0 < confidence < 1.0:
        raise ValueError("invalid binomial count or confidence")
    if events == episodes:
        return 1.0
    alpha = 1.0 - confidence
    if events == 0:
        return 1.0 - alpha ** (1.0 / episodes)
    lo, hi = 0.0, 1.0
    for _ in range(72):
        mid = (lo + hi) * 0.5
        cdf = sum(comb(episodes, j) * mid ** j * (1.0 - mid) ** (episodes - j)
                  for j in range(events + 1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@dataclass(frozen=True)
class RiskDualConfig:
    """Predeclare this config before any registered development rollout."""

    target: float = 0.04
    window_episodes: int = 128
    minimum_episodes: int = 64
    confidence: float = 0.90
    release_upper: float = 0.025
    ascent_gain: float = 2.0
    descent_gain: float = 1.0
    max_ascent_per_update: float = 0.10
    max_descent_per_update: float = 0.05
    max_multiplier: float = 20.0
    initial_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not (0 < self.release_upper < self.target < 1
                and 1 <= self.minimum_episodes <= self.window_episodes
                and 0 < self.confidence < 1
                and self.ascent_gain > 0 and self.descent_gain > 0
                and self.max_ascent_per_update > 0
                and self.max_descent_per_update > 0
                and self.max_multiplier > 0
                and 0 <= self.initial_multiplier <= self.max_multiplier):
            raise ValueError("invalid predeclared risk dual configuration")


class ShutdownRiskDual:
    """Update only from complete episode outcomes with upper-bound hysteresis."""

    def __init__(self, config: RiskDualConfig = RiskDualConfig()):
        self.config = config
        self.multiplier = config.initial_multiplier
        self.completed = deque(maxlen=config.window_episodes)

    def update(self, outcomes: Sequence[tuple[bool, bool]]) -> dict:
        """Each input is (complete, shutdown); incomplete runs are ignored."""
        accepted = 0
        for complete, shutdown in outcomes:
            if not isinstance(complete, bool) or not isinstance(shutdown, bool):
                raise TypeError("episode completion and shutdown must be bool")
            if shutdown and not complete:
                raise ValueError("shutdown event in an incomplete episode")
            if complete:
                self.completed.append(shutdown)
                accepted += 1
        n = len(self.completed)
        k = sum(self.completed)
        upper = one_sided_binomial_upper(k, n, self.config.confidence) if n else None
        previous = self.multiplier
        phase = "insufficient_complete_episodes"
        if accepted and n >= self.config.minimum_episodes:
            if upper > self.config.target:
                step = min(self.config.max_ascent_per_update,
                           self.config.ascent_gain * (upper - self.config.target))
                self.multiplier = min(self.config.max_multiplier, previous + step)
                phase = "increase"
            elif upper < self.config.release_upper:
                step = min(self.config.max_descent_per_update,
                           self.config.descent_gain * (self.config.release_upper - upper))
                self.multiplier = max(0.0, previous - step)
                phase = "release"
            else:
                phase = "hold_hysteresis"
        elif not accepted:
            phase = "no_new_complete_episode"
        return {
            "accepted_complete_episodes": accepted,
            "window_episodes": n,
            "window_shutdowns": k,
            "window_shutdown_rate": k / n if n else None,
            "one_sided_upper": upper,
            "phase": phase,
            "previous_multiplier": previous,
            "next_multiplier": self.multiplier,
        }

    def snapshot(self) -> dict:
        return {"config": asdict(self.config), "multiplier": self.multiplier,
                "completed": list(self.completed)}

    @classmethod
    def restore(cls, snapshot: Mapping, config: RiskDualConfig) -> "ShutdownRiskDual":
        if snapshot.get("config") != asdict(config):
            raise ValueError("risk dual checkpoint config mismatch")
        value = float(snapshot["multiplier"])
        completed = snapshot["completed"]
        if (not isfinite(value) or not 0 <= value <= config.max_multiplier
                or len(completed) > config.window_episodes
                or any(not isinstance(item, bool) for item in completed)):
            raise ValueError("invalid risk dual checkpoint state")
        result = cls(config)
        result.multiplier = value
        result.completed.extend(completed)
        return result
