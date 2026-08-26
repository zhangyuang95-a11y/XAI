"""Reproducible interaction-density calibration for warehouse layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable

import numpy as np

from .coordination import stable_coordination_actions
from .domain import WarehouseConfig
from .environment import WarehouseMultiAgentEnv
from .navigation import ACTIONS


@dataclass(frozen=True)
class InteractionEpisode:
    seed: int
    deliveries: int
    collision_opportunity_frames: int
    robot_collisions: int
    charger_use_steps: int
    coordination_event_frames: int


def _has_collision_opportunity(environment: WarehouseMultiAgentEnv) -> bool:
    state = environment.get_state()
    masks = environment.action_masks()
    legal = {
        agent_id: tuple(
            action
            for action, allowed in zip(ACTIONS, masks[agent_id])
            if allowed > 0.5
        )
        for agent_id in environment.agent_ids
    }
    return any(
        environment._resolve_motion(
            state,
            {"robot_1": robot_one, "robot_2": robot_two},
        )[3]
        for robot_one in legal["robot_1"]
        for robot_two in legal["robot_2"]
    )


def calibrate_interactions(
    config: WarehouseConfig,
    *,
    seeds: Iterable[int],
    participant_noise_probability: float = 0.35,
) -> dict[str, object]:
    """Run independent-AI/noisy-participant episodes over fixed seeds.

    The development teacher chooses robot 2 from S_t before the simulated
    participant command replaces robot 1. This mirrors the simultaneous UI
    contract and deliberately allows real conflicts.
    """

    episodes: list[InteractionEpisode] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        environment = WarehouseMultiAgentEnv(config)
        environment.reset(seed=seed)
        rng = np.random.default_rng(seed + 999)
        opportunities = 0
        collisions = 0
        charger_steps = 0
        coordination_frames = 0
        while True:
            opportunities += int(_has_collision_opportunity(environment))
            actions = stable_coordination_actions(environment)
            if rng.random() < float(participant_noise_probability):
                mask = environment.action_masks()["robot_1"]
                legal = [
                    action
                    for action, allowed in zip(ACTIONS, mask)
                    if allowed > 0.5
                ]
                actions["robot_1"] = str(rng.choice(legal))
            _, _, terminated, truncated, info = environment.step(actions)
            collisions += int(bool(info.get("robot_collision_event", False)))
            charger_steps += int(bool(info.get("charger_used", False)))
            coordination_frames += int(bool(info.get("coordination_events", ())))
            if terminated or truncated:
                break
        state = environment.get_state()
        episodes.append(
            InteractionEpisode(
                seed=seed,
                deliveries=int(state.total_deliveries),
                collision_opportunity_frames=opportunities,
                robot_collisions=collisions,
                charger_use_steps=charger_steps,
                coordination_event_frames=coordination_frames,
            )
        )
    if not episodes:
        raise ValueError("At least one calibration seed is required.")
    fields = (
        "deliveries",
        "collision_opportunity_frames",
        "robot_collisions",
        "charger_use_steps",
        "coordination_event_frames",
    )
    return {
        "layout_id": config.map_layout_id,
        "horizon": config.horizon,
        "participant_noise_probability": float(participant_noise_probability),
        "episodes": [asdict(item) for item in episodes],
        "mean": {
            field: mean(float(getattr(item, field)) for item in episodes)
            for field in fields
        },
        "minimum": {
            field: min(int(getattr(item, field)) for item in episodes)
            for field in fields
        },
    }
