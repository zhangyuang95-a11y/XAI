"""Versioned, pre-action Warehouse observations for the v6 dual Actor.

Every value is either in the public state or is a deterministic calculation
from the public map, robot positions, batteries, orders, and action history.
In particular, environment goals, route commitments, coordination plans,
and teacher decisions are never inputs to this encoder.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json

import numpy as np

from .layouts import STUDY_MAP_LAYOUT
from .navigation import ACTIONS, shortest_path_distance
from .pure_nn import AGENTS, action_masks, observation as legacy_observation
from .pure_nn import observation_feature_names as legacy_feature_names


OBSERVATION_VERSION = "warehouse-public-observation.v6.1"
MOVE_BATTERY_COST = 2.0
_DISTANCE_SCALE = float(STUDY_MAP_LAYOUT.rows * STUDY_MAP_LAYOUT.cols)
_CAPACITY_SCALE = 50.0
_HISTORY_LENGTH = 3
_COLLISION_STREAK_CAP = 8

_AGENT_FIELDS = (
    "charger_shortest_distance", "feasible_move_steps", "charger_return_margin",
    "last_battery_delta", "steps_since_charging", "charger_wait_streak",
    *(f"last_submitted_{action.lower()}" for action in ACTIONS),
    *(f"recent_position_{index}_{axis}"
      for index in range(_HISTORY_LENGTH) for axis in ("row", "column")),
)
_ORDER_FIELDS = tuple(
    f"order_{order}_{role}_{metric}"
    for order in (0, 1)
    for role in ("self", "partner")
    for metric in ("remaining_mission_distance", "charger_return_margin")
)
OBSERVATION_FEATURE_NAMES = (
    *legacy_feature_names(),
    *(f"{role}_{field}" for role in ("self", "partner") for field in _AGENT_FIELDS),
    *_ORDER_FIELDS,
    "robots_shortest_distance",
    "same_target_collision_streak_capped",
)
OBSERVATION_DIM = len(OBSERVATION_FEATURE_NAMES)
FEATURE_NAMES = (*OBSERVATION_FEATURE_NAMES,
                 *(f"legal_{action.lower()}" for action in ACTIONS))
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_SCHEMA_SHA256 = sha256(json.dumps(
    FEATURE_NAMES, separators=(",", ":")).encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return shortest_path_distance(start, end, STUDY_MAP_LAYOUT.layout_id)


def _mission_distance(agent, task) -> float:
    """Distance to delivery and then charger, including pickup if needed.

    This is a measurement of a possible route, never a selected goal or action.
    A task owned by the other robot has no available mission for this robot.
    """
    if task.carrier_agent_id not in (None, agent.agent_id):
        return 0.0
    if task.status == "carried" and task.carrier_agent_id == agent.agent_id:
        return float(_distance(agent.position, task.delivery_position)
                     + _distance(task.delivery_position, STUDY_MAP_LAYOUT.charger_position))
    if task.status == "available":
        return float(_distance(agent.position, task.pickup_position)
                     + _distance(task.pickup_position, task.delivery_position)
                     + _distance(task.delivery_position, STUDY_MAP_LAYOUT.charger_position))
    return 0.0


def encode_role_observation(state, agent_id: str) -> np.ndarray:
    if agent_id not in AGENTS:
        raise ValueError("unknown Warehouse role")
    ordered = (state.by_id(agent_id), state.by_id(AGENTS[1 - AGENTS.index(agent_id)]))
    tasks = sorted(state.tasks, key=lambda task: task.task_id)
    if len(tasks) != 2:
        raise ValueError("Warehouse v6 requires exactly two public active orders")
    values = legacy_observation(state, agent_id).tolist()
    for agent in ordered:
        charger_distance = _distance(agent.position, STUDY_MAP_LAYOUT.charger_position)
        capacity = max(0.0, float(agent.battery) / MOVE_BATTERY_COST)
        history = tuple(agent.recent_positions)[-_HISTORY_LENGTH:]
        # The reset position is public; repeat it if fewer changes have occurred.
        history = (agent.position,) * (_HISTORY_LENGTH - len(history)) + history
        values.extend((
            charger_distance / _DISTANCE_SCALE,
            capacity / _CAPACITY_SCALE,
            (capacity - charger_distance) / _CAPACITY_SCALE,
            float(agent.last_battery_delta) / 10.0,
            min(1.0, max(0.0, float(agent.steps_since_charging) / 120.0)),
            min(1.0, max(0.0, float(agent.charger_wait_streak) / 12.0)),
        ))
        # At reset there was no preceding decision; the legacy AgentState
        # default WAIT must not masquerade as a submitted command.
        values.extend(float(state.frame > 0 and agent.last_action == action)
                      for action in ACTIONS)
        for position in history:
            values.extend((position[0] / max(1, STUDY_MAP_LAYOUT.rows - 1),
                           position[1] / max(1, STUDY_MAP_LAYOUT.cols - 1)))
    for task in tasks:
        for agent in ordered:
            mission_distance = _mission_distance(agent, task)
            available = task.status == "available" or task.carrier_agent_id == agent.agent_id
            capacity = max(0.0, float(agent.battery) / MOVE_BATTERY_COST)
            values.extend((mission_distance / _DISTANCE_SCALE if available else 0.0,
                           (capacity - mission_distance) / _CAPACITY_SCALE
                           if available else 0.0))
    values.append(_distance(ordered[0].position, ordered[1].position) / _DISTANCE_SCALE)
    values.append(min(_COLLISION_STREAK_CAP,
                      max(0, state.same_target_collision_streak)) / _COLLISION_STREAK_CAP)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("Warehouse v6 observation schema drifted")
    return result


def encode_public_observation(state) -> np.ndarray:
    """Return the two ego/partner role rows for the current pre-action state."""
    return np.stack([encode_role_observation(state, role) for role in AGENTS])


def encode_policy_features(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    data, legal = np.asarray(obs, dtype=np.float32), np.asarray(masks, dtype=np.float32)
    if (data.shape[-1:] != (OBSERVATION_DIM,)
            or legal.shape != data.shape[:-1] + (len(ACTIONS),)
            or not np.isfinite(data).all() or not np.isfinite(legal).all()
            or not np.all((legal == 0.0) | (legal == 1.0))
            or np.any(legal.sum(axis=-1) < 1.0)):
        raise ValueError("invalid Warehouse v6 pre-action observation or physical mask")
    return np.concatenate((data, legal), axis=-1)


def public_pre_action_snapshot(state) -> dict:
    """Serialize exactly the public fields needed to rebuild both NN inputs.

    This intentionally omits private goals, route commitments, coordination
    plans, reward bookkeeping, and any current or future action.
    """
    return {
        "schema": OBSERVATION_VERSION,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
        "episode_id": int(state.episode_id),
        "frame": int(state.frame),
        "agents": [{
            "agent_id": agent.agent_id,
            "position": list(agent.position),
            "battery": float(agent.battery),
            "active": bool(agent.active),
            "carrying_task_id": agent.carrying_task_id,
            "heading": agent.heading,
            "last_action": agent.last_action,
            "last_executed_action": agent.last_executed_action,
            "last_battery_delta": float(agent.last_battery_delta),
            "steps_since_charging": int(agent.steps_since_charging),
            "charger_wait_streak": int(agent.charger_wait_streak),
            "recent_positions": [list(position) for position in agent.recent_positions],
        } for agent in (state.by_id(role) for role in AGENTS)],
        "tasks": [{
            "task_id": task.task_id,
            "pickup_position": list(task.pickup_position),
            "delivery_position": list(task.delivery_position),
            "status": task.status,
            "carrier_agent_id": task.carrier_agent_id,
        } for task in sorted(state.tasks, key=lambda item: item.task_id)],
        "last_robot_collision_event": bool(state.last_robot_collision_event),
        "last_robot_collision_kind": state.last_robot_collision_kind,
        "same_target_collision_streak": int(state.same_target_collision_streak),
    }


def rebuild_public_pre_action_state(snapshot: dict):
    """Rebuild the observation/mask state from an independent trace row."""
    from .domain import AgentState, DeliveryTask, WarehouseState
    if (snapshot.get("schema") != OBSERVATION_VERSION
            or snapshot.get("map_layout_id") != STUDY_MAP_LAYOUT.layout_id):
        raise ValueError("Warehouse public snapshot schema or map mismatch")
    frame = snapshot.get("frame")
    streak = snapshot.get("same_target_collision_streak")
    collision = snapshot.get("last_robot_collision_event")
    kind = snapshot.get("last_robot_collision_kind")
    if (type(frame) is not int or frame < 0
            or type(streak) is not int or not 0 <= streak <= frame
            or type(collision) is not bool
            or kind not in (None, "same_target", "swap", "occupied_stationary")
            or collision != (kind is not None)
            or (streak > 0) != (kind == "same_target")
            or (frame == 0 and (collision or streak))):
        raise ValueError("Warehouse public collision history is invalid")
    if any((entry.get("last_action") not in ACTIONS
            or entry.get("last_executed_action") not in ACTIONS
            or (frame == 0 and (entry["last_action"] != "WAIT"
                               or entry["last_executed_action"] != "WAIT")))
           for entry in snapshot["agents"]):
        raise ValueError("Warehouse public action history is invalid")
    agents = [AgentState(**{
        **entry,
        "position": tuple(entry["position"]),
        "recent_positions": tuple(tuple(value) for value in entry["recent_positions"]),
    }) for entry in snapshot["agents"]]
    tasks = [DeliveryTask(**{
        **entry,
        "pickup_position": tuple(entry["pickup_position"]),
        "delivery_position": tuple(entry["delivery_position"]),
    }) for entry in snapshot["tasks"]]
    if [agent.agent_id for agent in agents] != list(AGENTS) or len(tasks) != 2:
        raise ValueError("Warehouse public snapshot roles or orders mismatch")
    return WarehouseState(episode_id=int(snapshot["episode_id"]),
                          frame=frame, agents=agents, tasks=tasks,
                          last_robot_collision_event=collision,
                          last_robot_collision_kind=kind,
                          same_target_collision_streak=streak)


def reconstruct_public_history(frames: list[dict] | tuple[dict, ...],
                               actor: str) -> dict:
    """Rebuild action and outcome history from consecutive exported v6 frames.

    The encoder reads the same environment-maintained summaries for speed.
    This calculation consumes only public frames and verifies the advertised
    streak independently; it never reads an environment snapshot or goal.
    """
    if actor not in ("human", "ai") or not frames:
        raise ValueError("public history requires a role and consecutive frames")
    positions: tuple[tuple[int, int], ...] = ()
    prior_battery = None
    steps_since_charging = 0
    charger_wait_streak = 0
    last_delta = 0.0
    last_submitted = None
    last_executed = None
    collision_kind = None
    same_target_streak = 0
    episode_id = frames[0].get("episode_id")
    for index, frame in enumerate(frames):
        if (frame.get("turn") != index or frame.get("episode_id") != episode_id
                or frame.get("map_id") != STUDY_MAP_LAYOUT.layout_id
                or frame.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256):
            raise ValueError("public Warehouse frames must start at turn zero and be consecutive")
        previous = frame.get("previous_joint_actions")
        kind = frame.get("previous_collision_kind")
        if index == 0:
            if previous is not None or kind is not None:
                raise ValueError("Warehouse reset frame cannot contain earlier actions or collisions")
        else:
            if (not isinstance(previous, dict)
                    or set(previous) != {"human", "ai"}
                    or any(not isinstance(value, dict)
                           or set(value) != {"proposed", "executed"}
                           or any(action not in {item.lower() for item in ACTIONS}
                                  for action in value.values())
                           for value in previous.values())
                    or kind not in (None, "same_target", "swap", "occupied_stationary")):
                raise ValueError("Warehouse public frame action or collision history is invalid")
            last_submitted = previous[actor]["proposed"].upper()
            last_executed = previous[actor]["executed"].upper()
            collision_kind = kind
            same_target_streak = (same_target_streak + 1
                                  if kind == "same_target" else 0)
        if frame.get("same_target_collision_streak") != same_target_streak:
            raise ValueError("Warehouse public frame collision streak was altered")
        view = frame[actor]
        position = (int(view["y"]), int(view["x"]))
        battery = float(view["battery"])
        if not positions or positions[-1] != position:
            positions = (positions + (position,))[-6:]
        if prior_battery is not None:
            last_delta = battery - prior_battery
            if last_delta > 0:
                steps_since_charging = 0
                charger_wait_streak += 1
            else:
                steps_since_charging = min(120, steps_since_charging + 1)
                charger_wait_streak = 0
        prior_battery = battery
    return {"recent_positions": positions,
            "last_battery_delta": last_delta,
            "steps_since_charging": steps_since_charging,
            "charger_wait_streak": charger_wait_streak,
            "last_submitted_action": last_submitted,
            "last_executed_action": last_executed,
            "last_robot_collision_kind": collision_kind,
            "same_target_collision_streak": same_target_streak}


def rebuild_public_pre_action_state_from_frames(
        frames: list[dict] | tuple[dict, ...]):
    """Independently rebuild both Actor inputs from exported v6 public frames.

    The caller supplies the complete prefix starting at turn zero.  Only the
    last frame's public geometry and the independently folded action history
    contribute to the rebuilt state; private persisted snapshots are unused.
    """
    from .domain import AgentState, DeliveryTask, WarehouseState
    histories = {role: reconstruct_public_history(frames, role)
                 for role in ("human", "ai")}
    latest = frames[-1]
    agents = []
    for role, agent_id in (("human", "robot_1"), ("ai", "robot_2")):
        view, history = latest[role], histories[role]
        agents.append(AgentState(
            agent_id=agent_id,
            position=(int(view["y"]), int(view["x"])),
            battery=float(view["battery"]),
            carrying_task_id=view["carrying"],
            active=bool(view["active"]),
            heading=str(view["heading"]).upper(),
            last_action=history["last_submitted_action"] or "WAIT",
            last_executed_action=history["last_executed_action"] or "WAIT",
            last_battery_delta=history["last_battery_delta"],
            steps_since_charging=history["steps_since_charging"],
            charger_wait_streak=history["charger_wait_streak"],
            recent_positions=history["recent_positions"],
        ))
    carrier_ids = {"human": "robot_1", "ai": "robot_2", None: None}
    tasks = []
    for order in latest["orders"]:
        if order["carrier"] not in carrier_ids:
            raise ValueError("Warehouse public frame has an unknown task carrier")
        tasks.append(DeliveryTask(
            task_id=order["id"],
            pickup_position=(int(order["pickup"][1]), int(order["pickup"][0])),
            delivery_position=(int(order["dropoff"][1]), int(order["dropoff"][0])),
            status=order["status"],
            carrier_agent_id=carrier_ids[order["carrier"]],
        ))
    if len(tasks) != 2:
        raise ValueError("Warehouse public frame must contain two orders")
    history = histories["human"]
    return WarehouseState(
        episode_id=int(latest["episode_id"]), frame=int(latest["turn"]),
        agents=agents, tasks=tasks,
        last_robot_collision_event=history["last_robot_collision_kind"] is not None,
        last_robot_collision_kind=history["last_robot_collision_kind"],
        same_target_collision_streak=history["same_target_collision_streak"],
    )


__all__ = [
    "OBSERVATION_VERSION", "OBSERVATION_DIM", "OBSERVATION_FEATURE_NAMES",
    "FEATURE_NAMES", "FEATURE_DIM", "FEATURE_SCHEMA_SHA256", "action_masks",
    "encode_role_observation", "encode_public_observation", "encode_policy_features",
    "reconstruct_public_history", "public_pre_action_snapshot",
    "rebuild_public_pre_action_state", "rebuild_public_pre_action_state_from_frames",
]
