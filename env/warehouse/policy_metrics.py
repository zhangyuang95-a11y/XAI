"""Reusable per-agent efficiency metric accumulators for MAPPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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
        *,
        excluded_agent_ids: Iterable[str] = (),
    ) -> None:
        excluded = set(excluded_agent_ids)
        if rewards is not None:
            for agent_id in AGENT_IDS:
                if agent_id in excluded:
                    continue
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
                if agent_id in excluded:
                    continue
                target[agent_id] += float(values.get(agent_id, 0.0))
        wait_agents = info.get("avoidable_wait_agents", ())
        detour_agents = info.get("avoidable_detour_agents", ())
        loaded_agents = info.get(
            "avoidable_loaded_delivery_detour_agents",
            (),
        )
        streaks = info.get("avoidable_wait_streaks", {})
        for agent_id in AGENT_IDS:
            if agent_id in excluded:
                continue
            self.avoidable_wait_counts[agent_id] += int(agent_id in wait_agents)
            self.detour_counts[agent_id] += int(agent_id in detour_agents)
            self.loaded_detour_counts[agent_id] += int(agent_id in loaded_agents)
            self.maximum_wait_streaks[agent_id] = max(
                self.maximum_wait_streaks[agent_id],
                int(streaks.get(agent_id, 0)),
            )

    def update_completed_tasks(
        self,
        state: Any,
        *,
        excluded_agent_ids: Iterable[str] = (),
    ) -> None:
        excluded = set(excluded_agent_ids)
        for task in state.completed_tasks:
            if (
                task.delivered_frame is None
                or task.claimed_frame is None
                or task.shortest_safe_delivery_steps is None
                or task.carrier_agent_id in excluded
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


def evaluation_summary(
    *,
    training_rewards: Sequence[float],
    base_training_rewards: Sequence[float],
    potential_shaping_rewards: Sequence[float],
    user_scores: Sequence[float],
    deliveries: Sequence[int],
    steps: Sequence[int],
    collision_episodes: int,
    shutdown_episodes: int,
    collision_counts: Sequence[int],
    shutdown_counts: Sequence[int],
    charger_use_steps: int,
    detour_units: float,
    delivery_durations: Sequence[int],
    minimum_batteries: Sequence[float],
    terminal_reasons: Mapping[str, int],
    proxy_human_overrides: int = 0,
    deadlock_episodes: int = 0,
    yield_events: int = 0,
    head_on_risk_events: int = 0,
    post_policy_action_interventions: int = 0,
    avoidable_mission_detour_steps: int = 0,
    avoidable_loaded_delivery_detour_steps: int = 0,
    charger_return_cycle_episodes: int = 0,
    charger_return_cycles: int = 0,
    task_starvation_episodes: int = 0,
    unexplained_reversal_steps: int = 0,
    short_cycle_steps: int = 0,
    invalid_goal_switch_steps: int = 0,
    explanation_fact_failures: int = 0,
    simultaneous_semantics_violations: int = 0,
    per_agent_progress_rewards: Mapping[str, float] | None = None,
    per_agent_counterfactual_regret_units: Mapping[str, float] | None = None,
    per_agent_counterfactual_regret_penalties: Mapping[str, float] | None = None,
    per_agent_repeated_wait_penalties: Mapping[str, float] | None = None,
    per_agent_avoidable_wait_counts: Mapping[str, int] | None = None,
    per_agent_maximum_avoidable_wait_streaks: Mapping[str, int] | None = None,
    per_agent_detour_counts: Mapping[str, int] | None = None,
    per_agent_loaded_detour_counts: Mapping[str, int] | None = None,
    coordination_progress_reward: float = 0.0,
    path_actual_steps: float = 0.0,
    path_shortest_safe_steps: float = 0.0,
    efficiency_agent_ids: Sequence[str] | None = None,
    failure_seeds: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    episode_count = max(1, len(training_rewards))
    total_steps = max(1, sum(steps))
    agent_ids = tuple(efficiency_agent_ids or ("robot_1", "robot_2"))
    progress = dict(per_agent_progress_rewards or {})
    regret = dict(per_agent_counterfactual_regret_units or {})
    regret_penalties = dict(per_agent_counterfactual_regret_penalties or {})
    wait_penalties = dict(per_agent_repeated_wait_penalties or {})
    wait_counts = dict(per_agent_avoidable_wait_counts or {})
    max_wait_streaks = dict(per_agent_maximum_avoidable_wait_streaks or {})
    detours = dict(per_agent_detour_counts or {})
    loaded_detours = dict(per_agent_loaded_detour_counts or {})
    return {
        "episodes": float(len(training_rewards)),
        "mean_training_reward": float(np.mean(training_rewards)),
        "training_reward_std": float(np.std(training_rewards)),
        "mean_base_training_reward": float(np.mean(base_training_rewards)),
        "mean_potential_shaping_reward": float(
            np.mean(potential_shaping_rewards)
        ),
        "mean_user_score": float(np.mean(user_scores)),
        "user_score_std": float(np.std(user_scores)),
        "mean_deliveries": float(np.mean(deliveries)),
        "delivery_std": float(np.std(deliveries)),
        "mean_episode_steps": float(np.mean(steps)),
        "deliveries_per_100_steps": 100.0 * sum(deliveries) / total_steps,
        "collision_episode_rate": collision_episodes / episode_count,
        "shutdown_episode_rate": shutdown_episodes / episode_count,
        "mean_robot_collision_events": float(np.mean(collision_counts)),
        "maximum_robot_collision_events": int(max(collision_counts, default=0)),
        "repeated_collision_episode_rate": (
            sum(value > 1 for value in collision_counts) / episode_count
        ),
        "mean_shutdown_events": float(np.mean(shutdown_counts)),
        "mean_charger_use_steps": charger_use_steps / episode_count,
        "charger_utilization_rate": charger_use_steps / total_steps,
        "mean_human_detour_units": detour_units / episode_count,
        "mean_claim_to_delivery_steps": (
            float(np.mean(delivery_durations)) if delivery_durations else 0.0
        ),
        "mean_minimum_battery": float(np.mean(minimum_batteries)),
        "mean_proxy_human_overrides": proxy_human_overrides / episode_count,
        "efficiency_metric_agent_ids": list(agent_ids),
        "delivery_episode_rate": sum(value > 0 for value in deliveries) / episode_count,
        "deadlock_episode_rate": deadlock_episodes / episode_count,
        "mean_coordination_yield_events": yield_events / episode_count,
        "mean_head_on_risk_events": head_on_risk_events / episode_count,
        "mean_post_policy_action_interventions": (
            post_policy_action_interventions / episode_count
        ),
        "avoidable_mission_detour_steps": int(avoidable_mission_detour_steps),
        "avoidable_mission_detours_per_1000_steps": (
            1000.0 * avoidable_mission_detour_steps / total_steps
        ),
        "avoidable_loaded_delivery_detour_steps": int(
            avoidable_loaded_delivery_detour_steps
        ),
        "avoidable_loaded_delivery_detours_per_1000_steps": (
            1000.0 * avoidable_loaded_delivery_detour_steps / total_steps
        ),
        "charger_departure_return_cycle_episode_rate": (
            charger_return_cycle_episodes / episode_count
        ),
        "mean_charger_departure_return_cycles": (
            charger_return_cycles / episode_count
        ),
        "task_starvation_episode_rate": task_starvation_episodes / episode_count,
        "unexplained_reversals": int(unexplained_reversal_steps),
        "short_cycles": int(short_cycle_steps),
        "invalid_goal_switches": int(invalid_goal_switch_steps),
        "explanation_fact_failures": int(explanation_fact_failures),
        "simultaneous_semantics_violations": int(
            simultaneous_semantics_violations
        ),
        "unnecessary_waits": int(
            sum(int(wait_counts.get(agent_id, 0)) for agent_id in agent_ids)
        ),
        "avoidable_wait_rate": (
            sum(int(wait_counts.get(agent_id, 0)) for agent_id in agent_ids)
            / max(1, total_steps * len(agent_ids))
        ),
        "mean_coordination_progress_reward": (
            coordination_progress_reward / episode_count
        ),
        "per_agent_efficiency": {
            agent_id: {
                "mean_progress_reward": progress.get(agent_id, 0.0)
                / episode_count,
                "mean_counterfactual_regret_units": regret.get(agent_id, 0.0)
                / episode_count,
                "mean_counterfactual_regret_penalty_reward": (
                    regret_penalties.get(agent_id, 0.0) / episode_count
                ),
                "mean_repeated_wait_penalty_reward": (
                    wait_penalties.get(agent_id, 0.0) / episode_count
                ),
                "avoidable_wait_count": int(wait_counts.get(agent_id, 0)),
                "avoidable_waits_per_1000_steps": (
                    1000.0 * wait_counts.get(agent_id, 0) / total_steps
                ),
                "maximum_avoidable_wait_streak": int(
                    max_wait_streaks.get(agent_id, 0)
                ),
                "detour_count": int(detours.get(agent_id, 0)),
                "loaded_detour_count": int(
                    loaded_detours.get(agent_id, 0)
                ),
            }
            for agent_id in agent_ids
        },
        "path_actual_steps": float(path_actual_steps),
        "path_shortest_safe_steps": float(path_shortest_safe_steps),
        "path_efficiency_actual_over_shortest_safe": (
            float(path_actual_steps) / max(1.0, float(path_shortest_safe_steps))
        ),
        "user_score_samples": [float(value) for value in user_scores],
        "delivery_samples": [int(value) for value in deliveries],
        "collision_event_samples": [int(value) for value in collision_counts],
        "failure_seeds": {
            str(kind): [int(value) for value in values]
            for kind, values in sorted((failure_seeds or {}).items())
        },
        **{
            f"terminal_{reason}_rate": count / episode_count
            for reason, count in sorted(terminal_reasons.items())
        },
    }
