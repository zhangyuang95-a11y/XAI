"""Data-only rollout batch contract shared by MAPPO training helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class EpisodeBatch:
    observations: np.ndarray
    global_states: np.ndarray
    agent_indices: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    trainable_mask: np.ndarray
    episode_reward: float
    episode_steps: int
    pickups: int
    deliveries: int
    collisions: int
    shutdowns: int
    terminal_reason: str | None
    charger_uses: int = 0
    avoidable_detours: int = 0
    route_regret: float = 0.0
    minimum_battery: float = 0.0
    proxy_human_overrides: int = 0
    base_training_reward: float = 0.0
    potential_shaping_reward: float = 0.0
    avoidable_wait_penalty_reward: float = 0.0
    mission_regression_penalty_reward: float = 0.0
    individual_training_rewards: dict[str, float] = field(default_factory=dict)
    individual_progress_rewards: dict[str, float] = field(default_factory=dict)
    coordination_progress_reward: float = 0.0
    counterfactual_regret_units: dict[str, float] = field(default_factory=dict)
    counterfactual_regret_penalty_rewards: dict[str, float] = field(
        default_factory=dict
    )
    repeated_avoidable_wait_penalty_rewards: dict[str, float] = field(
        default_factory=dict
    )
    avoidable_wait_counts: dict[str, int] = field(default_factory=dict)
    maximum_avoidable_wait_streaks: dict[str, int] = field(default_factory=dict)
    detour_counts: dict[str, int] = field(default_factory=dict)
    loaded_detour_counts: dict[str, int] = field(default_factory=dict)
    path_efficiency_actual_over_shortest_safe: float = 0.0
    energy_curriculum_applied: bool = False
    coordination_curriculum_kind: str | None = None
    initial_minimum_battery: float = 100.0
    semantic_features: tuple[Mapping[str, float], ...] = ()
    regularization_observations: np.ndarray | None = None
    regularization_targets: np.ndarray | None = None
    regularization_weights: np.ndarray | None = None
