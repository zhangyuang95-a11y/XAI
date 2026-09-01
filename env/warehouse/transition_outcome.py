"""Terminal-state and participant-score finalization for one transition."""

from __future__ import annotations

from typing import Any


def finalize_transition_outcome(
    config: Any,
    layout: Any,
    next_state: Any,
    *,
    delivered_count: int,
    robot_collision: bool,
    route_regret: float,
) -> tuple[tuple[str, ...], dict[str, float], float, bool, bool, str | None]:
    """Apply shutdown, score, and terminal facts after motion/task effects."""

    shutdown_agents = tuple(
        agent.agent_id
        for agent in next_state.agents
        if (
            agent.active
            and agent.battery <= 0.0
            # Equality is enough to reach the station.  Docking on exactly
            # 0% is a successful arrival; the following WAIT begins charge.
            and agent.position != layout.charger_position
        )
    )
    for agent_id in shutdown_agents:
        next_state.by_id(agent_id).active = False
    next_state.shutdown_count += len(shutdown_agents)

    score_components = {
        "delivery": config.delivery_points * max(0, int(delivered_count)),
        "robot_collision": config.robot_collision_points if robot_collision else 0.0,
        "shutdown": config.shutdown_points * len(shutdown_agents),
        "time": config.step_points,
        "human_detour": config.human_detour_points_per_unit * route_regret,
    }
    if shutdown_agents and next_state.frame < config.horizon:
        score_components["time"] += config.step_points * (
            config.horizon - next_state.frame
        )
    score_delta = float(sum(score_components.values()))
    terminated = bool(shutdown_agents)
    truncated = bool(next_state.frame >= config.horizon and not terminated)
    reason = "battery_shutdown" if terminated else "horizon" if truncated else None
    return (
        shutdown_agents,
        score_components,
        score_delta,
        terminated,
        truncated,
        reason,
    )
