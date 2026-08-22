"""Deterministic regression probes for archived collaborative-policy failures.

These probes are production-side acceptance evidence rather than test-only
fixtures.  Keeping the exact transition-start snapshots here lets the formal
evaluator prove that a candidate still fixes the v20 seed-42027 detour bug.
"""

from __future__ import annotations

from typing import Any

from .domain import AgentState, DeliveryTask, WarehouseConfig, WarehouseState
from .environment import WarehouseMultiAgentEnv
from .layouts import get_map_layout


SEED_42027_DETOUR_ACTIONS: dict[int, dict[str, str]] = {
    25: {"robot_1": "RIGHT", "robot_2": "UP"},
    69: {"robot_1": "DOWN", "robot_2": "UP"},
    99: {"robot_1": "DOWN", "robot_2": "UP"},
    119: {"robot_1": "DOWN", "robot_2": "UP"},
    120: {"robot_1": "DOWN", "robot_2": "WAIT"},
}


def seed_42027_regression_state(
    frame: int,
    config: WarehouseConfig | None = None,
) -> WarehouseState:
    """Return the exact transition-start state for one archived failure."""

    if frame not in SEED_42027_DETOUR_ACTIONS:
        raise ValueError(f"Unsupported seed-42027 regression frame: {frame}.")
    active_config = config or WarehouseConfig()
    charger = get_map_layout(active_config.map_layout_id).charger_position
    if frame == 25:
        agents = [
            AgentState("robot_1", (3, 4), 56.0, "task_4", heading="LEFT"),
            AgentState("robot_2", (6, 5), 35.0, heading="UP"),
        ]
        tasks = [
            DeliveryTask("task_3", (2, 5), (5, 4), created_frame=15),
            DeliveryTask(
                "task_4",
                (3, 4),
                (5, 2),
                status="carried",
                carrier_agent_id="robot_1",
                created_frame=16,
                claimed_frame=24,
            ),
        ]
    elif frame in {69, 99}:
        agents = [
            AgentState(
                "robot_1",
                (5, 5),
                18.0 if frame == 69 else 34.0,
                heading="DOWN",
            ),
            AgentState(
                "robot_2",
                charger,
                53.0 if frame == 69 else 45.0,
                heading="UP",
            ),
        ]
        second = (
            DeliveryTask("task_8", (3, 5), (1, 2), created_frame=65)
            if frame == 69
            else DeliveryTask("task_10", (6, 8), (3, 4), created_frame=95)
        )
        tasks = [
            DeliveryTask("task_7", (2, 9), (7, 3), created_frame=57),
            second,
        ]
    else:
        agents = [
            AgentState(
                "robot_1",
                (3, 5) if frame == 119 else (4, 5),
                18.0 if frame == 119 else 16.0,
                heading="RIGHT" if frame == 119 else "DOWN",
            ),
            AgentState(
                "robot_2",
                (7, 5) if frame == 119 else (6, 5),
                51.0 if frame == 119 else 49.0,
                "task_11",
                heading="DOWN" if frame == 119 else "UP",
            ),
        ]
        tasks = [
            DeliveryTask("task_7", (2, 9), (7, 3), created_frame=57),
            DeliveryTask(
                "task_11",
                (7, 5),
                (3, 3),
                status="carried",
                carrier_agent_id="robot_2",
                created_frame=117,
                claimed_frame=118,
            ),
        ]
    return WarehouseState(
        episode_id=42027,
        frame=frame - 1,
        agents=agents,
        tasks=tasks,
        next_task_index=12,
    )


def evaluate_seed_42027_detour_regressions(
    config: WarehouseConfig,
) -> dict[str, Any]:
    """Execute every archived transition and report corrected detour units."""

    frames: dict[str, dict[str, Any]] = {}
    passed = True
    for frame, actions in SEED_42027_DETOUR_ACTIONS.items():
        environment = WarehouseMultiAgentEnv(config)
        environment.reset(seed=42027)
        environment.set_state(seed_42027_regression_state(frame, config))
        _observations, rewards, _terminated, _truncated, info = environment.step(
            actions
        )
        regret = float(info["route_regret"][config.human_agent_id])
        frame_passed = regret == 0.0
        passed = passed and frame_passed
        frames[str(frame)] = {
            "actions": dict(actions),
            "route_regret_units": regret,
            "human_detour_score_delta": float(
                info["reward_breakdown"]["human_detour"]
            ),
            "team_rewards_equal": (
                float(rewards["robot_1"]) == float(rewards["robot_2"])
            ),
            "passed": frame_passed,
        }
    return {"seed": 42027, "frames": frames, "passed": passed}
