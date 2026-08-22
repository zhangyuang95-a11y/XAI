"""Pure JSON presentation for warehouse map and state snapshots."""

from __future__ import annotations

from typing import Any, Mapping

from env.warehouse.domain import WarehouseState
from env.warehouse.layouts import DEFAULT_MAP_LAYOUT, MapLayout


AI_AI_AGENT_CONTROL = {"robot_1": "ai", "robot_2": "ai"}
HUMAN_AI_AGENT_CONTROL = {"robot_1": "human", "robot_2": "ai"}


def _point(value: tuple[int, int]) -> list[int]:
    return [int(value[0]), int(value[1])]


def _study_question_focus(question: str) -> str:
    """Preserve the participant's requested explanatory dimension."""

    normalized = " ".join(str(question).casefold().split())
    if any(
        token in normalized
        for token in (
            "碰撞", "冲突", "撞", "collision", "crash", "conflict",
        )
    ):
        return "collision"
    if any(
        token in normalized
        for token in (
            "电量", "电池", "充电", "battery", "energy", "charge", "charging",
        )
    ):
        return "charge_threshold" if any(
            token in normalized
            for token in ("多少", "阈值", "再走", "离开", "how much", "before", "hold")
        ) else "energy"
    if any(
        token in normalized
        for token in (
            "a1", "a2", "谁去取", "由谁", "分配", "认领", "没有去取",
            "不去拿", "去拿", "task allocation", "assigned", "assignment",
            "claim", "pickup allocation",
        )
    ):
        return "allocation"
    if any(
        token in normalized
        for token in (
            "队友", "合作", "协作", "另一个机器人", "teammate", "coordinate",
            "coordination", "collaborate", "other robot",
        )
    ):
        return "collaboration"
    if any(
        token in normalized
        for token in (
            "配送任务", "任务进度", "交付", "取货", "delivery task",
            "task progress", "delivery", "drop-off",
        )
    ):
        return "task"
    return "action"


def warehouse_map_payload(
    layout: MapLayout = DEFAULT_MAP_LAYOUT,
) -> dict[str, Any]:
    """Return immutable map geometry used by the browser renderer."""

    return {
        "layout_id": layout.layout_id,
        "rows": layout.rows,
        "cols": layout.cols,
        "shelves": [_point(position) for position in layout.blocked_positions],
        "charger_position": _point(layout.charger_position),
        "waiting_zone": [
            _point(position)
            for position in layout.passable_positions
            if position[0] == layout.rows - 1
            and position != layout.charger_position
        ],
        "robot_start_positions": [
            _point(position) for position in layout.robot_start_positions
        ],
        "shared_delivery_tasks": True,
    }


def serialize_warehouse_state(
    state: WarehouseState,
    *,
    selected_agent: str,
    actions: Mapping[str, str] | None = None,
    distributions: Mapping[str, Any] | None = None,
    rewards: Mapping[str, float] | None = None,
    events: Any = None,
    reveal_policy: bool = True,
) -> dict[str, Any]:
    """Serialize one exact state without leaking test answers."""

    action_map = dict(actions or {}) if reveal_policy else {}
    reward_map = dict(rewards or {}) if reveal_policy else {}
    ordered_tasks = sorted(state.tasks, key=lambda item: item.task_id)
    task_slots = {
        task.task_id: index
        for index, task in enumerate(ordered_tasks, start=1)
    }
    agents = []
    for agent in state.agents:
        agents.append(
            {
                "id": agent.agent_id,
                "position": _point(agent.position),
                "battery": float(agent.battery),
                "carrying_task_id": agent.carrying_task_id,
                "carrying_label": (
                    f"A{task_slots[agent.carrying_task_id]}"
                    if agent.carrying_task_id in task_slots
                    else None
                ),
                "deliveries_completed": int(agent.deliveries_completed),
                "active": bool(agent.active),
                "heading": str(agent.heading),
                "last_action": str(agent.last_action),
                "last_executed_action": str(
                    agent.last_executed_action
                ),
                "proposed_action": action_map.get(agent.agent_id),
                "reward": (
                    float(reward_map[agent.agent_id])
                    if agent.agent_id in reward_map
                    else None
                ),
                "selected": agent.agent_id == selected_agent,
            }
        )
    return {
        "episode_id": int(state.episode_id),
        "frame": int(state.frame),
        "total_deliveries": int(state.total_deliveries),
        "active_count": sum(agent.active for agent in state.agents),
        "collision_count": int(state.collision_count),
        "shutdown_count": int(state.shutdown_count),
        "terminated": bool(state.terminated),
        "truncated": bool(state.truncated),
        "terminal_reason": state.terminal_reason,
        "selected_agent": selected_agent,
        "agents": agents,
        "tasks": [
            {
                "task_id": task.task_id,
                "pickup_position": _point(task.pickup_position),
                "delivery_position": _point(task.delivery_position),
                "status": task.status,
                "carrier_agent_id": task.carrier_agent_id,
                "created_frame": int(task.created_frame),
                "claimed_frame": (
                    int(task.claimed_frame)
                    if task.claimed_frame is not None
                    else None
                ),
            }
            for task in ordered_tasks
        ],
        "user_score": float(state.user_score),
        "score_breakdown": {
            key: float(value)
            for key, value in state.score_breakdown.items()
        },
        "human_route_regret_units": float(
            state.human_route_regret_units
        ),
        "robot_collision_events": int(state.robot_collision_events),
        "invalid_move_count": int(state.invalid_move_count),
        "events": events if reveal_policy else None,
        "policy_hidden": not reveal_policy,
    }
