"""Reusable per-agent efficiency metric accumulators for MAPPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


AGENT_IDS = ("robot_1", "robot_2")


def _float_map() -> dict[str, float]:
    return {agent_id: 0.0 for agent_id in AGENT_IDS}


def _int_map() -> dict[str, int]:
    return {agent_id: 0 for agent_id in AGENT_IDS}


@dataclass
class EfficiencyMetrics:
    """Episode/evaluation totals for individual reward attribution."""

    individual_training_rewards: dict[str, float] = field(default_factory=_float_map)
    progress_rewards: dict[str, float] = field(default_factory=_float_map)
    regret_units: dict[str, float] = field(default_factory=_float_map)
    regret_penalties: dict[str, float] = field(default_factory=_float_map)
    repeated_wait_penalties: dict[str, float] = field(default_factory=_float_map)
    avoidable_wait_counts: dict[str, int] = field(default_factory=_int_map)
    maximum_wait_streaks: dict[str, int] = field(default_factory=_int_map)
    detour_counts: dict[str, int] = field(default_factory=_int_map)
    loaded_detour_counts: dict[str, int] = field(default_factory=_int_map)
    coordination_progress_reward: float = 0.0
    path_actual_steps: float = 0.0
    path_shortest_safe_steps: float = 0.0

    def update_step(
        self,
        info: Mapping[str, Any],
        rewards: Mapping[str, float] | None = None,
    ) -> None:
        if rewards is not None:
            for agent_id in AGENT_IDS:
                self.individual_training_rewards[agent_id] += float(
                    rewards[agent_id]
                )
        self.coordination_progress_reward += float(
            info.get("coordination_progress_reward", 0.0)
        )
        mappings = (
            (self.progress_rewards, "individual_progress_rewards"),
            (self.regret_units, "counterfactual_regret_units"),
            (self.regret_penalties, "counterfactual_regret_penalty_rewards"),
            (
                self.repeated_wait_penalties,
                "repeated_avoidable_wait_penalty_rewards",
            ),
        )
        for target, key in mappings:
            values = info.get(key, {})
            for agent_id in AGENT_IDS:
                target[agent_id] += float(values.get(agent_id, 0.0))
        wait_agents = info.get("avoidable_wait_agents", ())
        detour_agents = info.get("avoidable_detour_agents", ())
        loaded_agents = info.get(
            "avoidable_loaded_delivery_detour_agents",
            (),
        )
        streaks = info.get("avoidable_wait_streaks", {})
        for agent_id in AGENT_IDS:
            self.avoidable_wait_counts[agent_id] += int(agent_id in wait_agents)
            self.detour_counts[agent_id] += int(agent_id in detour_agents)
            self.loaded_detour_counts[agent_id] += int(agent_id in loaded_agents)
            self.maximum_wait_streaks[agent_id] = max(
                self.maximum_wait_streaks[agent_id],
                int(streaks.get(agent_id, 0)),
            )

    def update_completed_tasks(self, state: Any) -> None:
        for task in state.completed_tasks:
            if (
                task.delivered_frame is None
                or task.claimed_frame is None
                or task.shortest_safe_delivery_steps is None
            ):
                continue
            self.path_actual_steps += float(
                task.delivered_frame - task.claimed_frame
            )
            self.path_shortest_safe_steps += float(
                task.shortest_safe_delivery_steps
            )

    @property
    def path_efficiency(self) -> float:
        return self.path_actual_steps / max(1.0, self.path_shortest_safe_steps)

    def evaluation_kwargs(self) -> dict[str, Any]:
        return {
            "per_agent_progress_rewards": self.progress_rewards,
            "per_agent_counterfactual_regret_units": self.regret_units,
            "per_agent_counterfactual_regret_penalties": self.regret_penalties,
            "per_agent_repeated_wait_penalties": self.repeated_wait_penalties,
            "per_agent_avoidable_wait_counts": self.avoidable_wait_counts,
            "per_agent_maximum_avoidable_wait_streaks": self.maximum_wait_streaks,
            "per_agent_detour_counts": self.detour_counts,
            "per_agent_loaded_detour_counts": self.loaded_detour_counts,
            "coordination_progress_reward": self.coordination_progress_reward,
            "path_actual_steps": self.path_actual_steps,
            "path_shortest_safe_steps": self.path_shortest_safe_steps,
        }


def mean_agent_attribute(items: Iterable[Any], attribute: str) -> dict[str, float]:
    rows = tuple(items)
    return {
        agent_id: (
            sum(float(getattr(item, attribute).get(agent_id, 0.0)) for item in rows)
            / max(1, len(rows))
        )
        for agent_id in AGENT_IDS
    }


def sum_agent_attribute(items: Iterable[Any], attribute: str) -> dict[str, int]:
    rows = tuple(items)
    return {
        agent_id: sum(
            int(getattr(item, attribute).get(agent_id, 0)) for item in rows
        )
        for agent_id in AGENT_IDS
    }


def max_agent_attribute(items: Iterable[Any], attribute: str) -> dict[str, int]:
    rows = tuple(items)
    return {
        agent_id: max(
            (int(getattr(item, attribute).get(agent_id, 0)) for item in rows),
            default=0,
        )
        for agent_id in AGENT_IDS
    }


def batch_efficiency_log_fields(batch: Any) -> dict[str, float | int]:
    """Flatten one EpisodeBatch for CSV/JSON training metrics."""

    output: dict[str, float | int] = {
        "coordination_progress_reward": batch.coordination_progress_reward,
        "path_efficiency_actual_over_shortest_safe": (
            batch.path_efficiency_actual_over_shortest_safe
        ),
    }
    attributes = (
        ("training_reward", "individual_training_rewards"),
        ("progress_reward", "individual_progress_rewards"),
        ("counterfactual_regret_units", "counterfactual_regret_units"),
        (
            "counterfactual_regret_penalty_reward",
            "counterfactual_regret_penalty_rewards",
        ),
        (
            "repeated_wait_penalty_reward",
            "repeated_avoidable_wait_penalty_rewards",
        ),
        ("avoidable_waits", "avoidable_wait_counts"),
        ("max_avoidable_wait_streak", "maximum_avoidable_wait_streaks"),
        ("detours", "detour_counts"),
        ("loaded_detours", "loaded_detour_counts"),
    )
    for label, attribute in attributes:
        values = getattr(batch, attribute)
        for agent_id in AGENT_IDS:
            output[f"{agent_id}_{label}"] = values.get(agent_id, 0)
    return output
