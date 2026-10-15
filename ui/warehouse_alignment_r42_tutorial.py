"""Build and replay the complete 120-step neutral r4.3 teaching demo."""
from __future__ import annotations

from copy import deepcopy
from collections import deque
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path
from typing import Any, Mapping

from backend.training.warehouse_native_common import digest, file_hash
from backend.warehouse_r43_runtime import R43WarehouseEnv
from backend.warehouse_r44_runtime import R44WarehouseEnv
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse_native.environment import collaborative_study_config
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from ui import warehouse_alignment_online_server as projection
from ui import warehouse_alignment_r41_tutorial as base


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-alignment-r42-neutral-tutorial.v2"
R43_VERSION = "warehouse-alignment-r43-neutral-tutorial.v3"
R44_VERSION = "warehouse-alignment-r44-neutral-tutorial.v4"
SOURCE = "independent_neutral_ai_ai"
DURATION_MS = 380
FRAME_COUNT = 121
_COVERAGE_FIELDS = frozenset((
    "pickup_frames", "delivery_frames", "simultaneous_movement_frames",
    "collision_frames", "wait_frames", "charge_frames",
))


def _environment(scene):
    snapshot = scene.get("snapshot", {})
    if "r44_shared_charger" in snapshot:
        return R44WarehouseEnv(
            collaborative_study_config(move_battery_cost=3.0))
    if "r43_shared_charger" in snapshot:
        return R43WarehouseEnv(
            collaborative_study_config(move_battery_cost=3.0))
    return R41DiagnosticConflictWarehouseEnv()


def _source_sha256() -> str:
    paths = (
        Path(__file__), ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "ui/warehouse_alignment_r41_tutorial.py",
        ROOT / "backend/warehouse_r43_runtime.py",
        ROOT / "env/warehouse_native/r43_charger.py",
        ROOT / "backend/warehouse_r44_runtime.py",
        ROOT / "env/warehouse_native/r44_charger.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
        ROOT / "env/warehouse_native/environment.py",
    )
    return digest({str(path.relative_to(ROOT)): file_hash(path) for path in paths})


def _append(env, frames, coverage, metrics, actions, *, final=False):
    _, _, terminated, truncated, info = env.step(dict(actions))
    metrics = base._metrics(env, metrics, info)
    frames.append(base._public_frame(env, metrics, actions))
    base._coverage_update(coverage, info, int(env.state.frame))
    if (terminated or truncated) != bool(final):
        raise ValueError("r4.2 tutorial terminal boundary differs")
    return metrics, info


_DELTAS = {"UP": (-1, 0), "LEFT": (0, -1), "RIGHT": (0, 1),
           "DOWN": (1, 0), "WAIT": (0, 0)}


@lru_cache(maxsize=4096)
def _distance(env, start, goal):
    return len(base._path_actions(env, start, goal))


def _teaching_goals(env, charging):
    """Assign current public jobs; the independent teaching controller only.

    A job's owner remains its real carrier. Empty robots split the remaining
    pickups by public route length. Charging is a persistent, energy-triggered
    task, rather than a filler motion or a fixed turn-taking schedule.
    """
    agents = env.state.agents
    available = sorted((t for t in env.state.tasks if t.status == "available"),
                       key=lambda task: task.task_id)
    empty = [a for a in agents if not a.carrying_task_id]
    allocations = []
    for ordering in permutations(available, len(empty)):
        cost = sum(_distance(env, a.position, t.pickup_position)
                   + _distance(env, t.pickup_position, t.delivery_position)
                   for a, t in zip(empty, ordering))
        allocations.append((cost, tuple(t.task_id for t in ordering), ordering))
    tasks = {a.agent_id: env.state.task_by_id(a.carrying_task_id)
             for a in agents if a.carrying_task_id}
    if allocations:
        for agent, task in zip(empty, min(allocations, key=lambda x: x[:2])[2]):
            tasks[agent.agent_id] = task
    charger = tuple(env.layout.charger_position)
    goals = {}
    for agent in agents:
        task = tasks[agent.agent_id]
        target = task.delivery_position if agent.carrying_task_id else task.pickup_position
        route = _distance(env, agent.position, target)
        if not agent.carrying_task_id:
            route += _distance(env, task.pickup_position, task.delivery_position)
        route += _distance(env, task.delivery_position, charger)
        if agent.battery < env.config.move_battery_cost * (route + 3):
            charging.add(agent.agent_id)
        if agent.battery >= 90:
            charging.discard(agent.agent_id)
        goals[agent.agent_id] = tuple(target)
    if charging:
        # Only one robot may occupy the shared charger. The robot already
        # there, otherwise the one with least return reserve, gets service.
        owner = min((a for a in agents if a.agent_id in charging), key=lambda a: (
            a.position != charger,
            a.battery - env.config.move_battery_cost * _distance(env, a.position, charger),
            a.agent_id))
        goals[owner.agent_id] = charger
        for agent in agents:
            if agent.agent_id in charging and agent.agent_id != owner.agent_id:
                holding = ((5, 2), (5, 4), (4, 4))
                goals[agent.agent_id] = min(holding, key=lambda position: (
                    _distance(env, agent.position, position), position))
    return goals


def _joint_teaching_action(env, goals):
    """Shortest joint route respecting same-cell/swap collision physics.

    Both commands are planned from the same public state. Replanning after
    every authoritative joint step responds to pickup, delivery and new jobs.
    This planner is never installed as the experimental robot's controller.
    """
    agents = env.state.agents
    start = tuple(tuple(a.position) for a in agents)
    goal = tuple(goals[a.agent_id] for a in agents)
    if start == goal:
        return {a.agent_id: "WAIT" for a in agents}
    queue, seen = deque([(start, None)]), {start}
    while queue:
        positions, first = queue.popleft()
        choices = []
        for index, position in enumerate(positions):
            # A robot that reached its goal can still step aside if needed;
            # distance ordering avoids gratuitous detours among shortest paths.
            rows = []
            for action, delta in _DELTAS.items():
                target = position[0] + delta[0], position[1] + delta[1]
                if env.layout.is_passable(target):
                    rows.append((_distance(env, target, goal[index]),
                                 action == "WAIT", action, target))
            choices.append(sorted(rows))
        for left, right in product(*choices):
            destinations = left[3], right[3]
            if (destinations[0] == destinations[1]
                    or destinations == positions[::-1]):
                continue
            if destinations in seen:
                continue
            actions = first or (left[2], right[2])
            if destinations == goal:
                return {a.agent_id: action for a, action in zip(agents, actions)}
            seen.add(destinations)
            queue.append((destinations, actions))
    raise ValueError("Teaching delivery targets have no joint route")


def build_tutorial(scenarios: Mapping[str, Any]) -> dict[str, Any]:
    play = scenarios.get("splits", {}).get("play")
    if not isinstance(play, list) or len(play) < 7:
        raise ValueError("r4.2 requires tutorial plus six frozen scenes")
    scene = deepcopy(play[0])
    env = _environment(scene)
    reset_diagnostic_scenario(env, scene)
    if int(env.config.horizon) != 120:
        raise ValueError("r4.2 tutorial requires a 120-step environment")

    metrics = base._metrics(env)
    frames = [base._public_frame(env, metrics, {})]
    coverage = base._empty_coverage()
    prelude = (
        {"robot_1": "RIGHT", "robot_2": "LEFT"},
        {"robot_1": "WAIT", "robot_2": "WAIT"},
        {"robot_1": "RIGHT", "robot_2": "WAIT"},
        {"robot_1": "WAIT", "robot_2": "UP"},
        {"robot_1": "UP", "robot_2": "DOWN"},
    )
    for actions in prelude:
        metrics, _ = _append(env, frames, coverage, metrics, actions)
    charging = set()
    while env.state.frame < 120:
        actions = _joint_teaching_action(env, _teaching_goals(env, charging))
        metrics, _ = _append(env, frames, coverage, metrics, actions,
                             final=env.state.frame == 119)
    payload = {
        "version": (R44_VERSION if "r44_shared_charger" in scene["snapshot"]
                    else R43_VERSION if "r43_shared_charger" in scene["snapshot"]
                    else VERSION),
        "source": SOURCE,
        "uses_final_actor": False,
        "scene_id": scene["id"],
        "duration_ms": DURATION_MS,
        "map_sha256": projection._digest(projection._public_map(env)),
        "bindings": {
            "parent_scene_fingerprint": str(scene["fingerprint"]),
            "parent_snapshot_sha256": digest(scene["snapshot"]),
            "producer_sources_sha256": _source_sha256(),
        },
        "coverage": coverage,
        "frames": frames,
    }
    validate_tutorial(payload, scene)
    return payload


def validate_tutorial(payload: Mapping[str, Any], scene: Mapping[str, Any]):
    if (not isinstance(payload, Mapping)
            or payload.get("version") != (
                R44_VERSION if "r44_shared_charger" in scene.get("snapshot", {})
                else R43_VERSION if "r43_shared_charger" in scene.get("snapshot", {})
                else VERSION)
            or payload.get("source") != SOURCE
            or payload.get("uses_final_actor") is not False
            or payload.get("duration_ms") != DURATION_MS
            or payload.get("scene_id") != scene.get("id")
            or len(payload.get("frames", ())) != FRAME_COUNT):
        raise ValueError("exact complete r4.2 tutorial required")
    expected_bindings = {
        "parent_scene_fingerprint": str(scene["fingerprint"]),
        "parent_snapshot_sha256": digest(scene["snapshot"]),
        "producer_sources_sha256": _source_sha256(),
    }
    if payload.get("bindings") != expected_bindings:
        raise ValueError("r4.2 tutorial binding differs")
    env = _environment(scene)
    reset_diagnostic_scenario(env, deepcopy(scene))
    metrics = base._metrics(env)
    if payload["frames"][0] != base._public_frame(env, metrics, {}):
        raise ValueError("r4.2 tutorial initial frame differs")
    recovered = base._empty_coverage()
    for index, saved in enumerate(payload["frames"][1:], 1):
        actions = saved.get("actions")
        if (not isinstance(actions, Mapping)
                or set(actions) != {"robot_1", "robot_2"}
                or any(action not in base.ACTIONS for action in actions.values())):
            raise ValueError("r4.2 tutorial action differs")
        _, _, terminated, truncated, info = env.step(dict(actions))
        metrics = base._metrics(env, metrics, info)
        base._coverage_update(recovered, info, int(env.state.frame))
        if saved != base._public_frame(env, metrics, actions):
            raise ValueError("r4.2 tutorial contains a fabricated frame")
        if bool(terminated or truncated) != (index == 120):
            raise ValueError("r4.2 tutorial termination differs")
    if payload.get("coverage") != recovered or any(
            not recovered[name] for name in _COVERAGE_FIELDS):
        raise ValueError("r4.2 tutorial physical coverage differs")
    deliveries = [agent.deliveries_completed for agent in env.state.agents]
    if min(deliveries) < 3:
        raise ValueError("both r4.2 tutorial robots must sustain deliveries")
    delivery_windows = {
        f"{start + 1}-{start + 40}": (
            payload["frames"][start + 40]["metrics"]["deliveries"]
            - payload["frames"][start]["metrics"]["deliveries"])
        for start in (0, 40, 80)
    }
    if min(delivery_windows.values()) < 2:
        raise ValueError("tutorial must sustain deliveries in every 40-step window")
    if len(recovered["simultaneous_movement_frames"]) < 60:
        raise ValueError("tutorial must demonstrate ongoing joint movement")
    progress_frames = sorted(set(recovered["pickup_frames"]
                                 + recovered["delivery_frames"]))
    longest_no_progress = max(right - left - 1 for left, right in zip(
        [0, *progress_frames], [*progress_frames, 121]))
    if longest_no_progress > 20:
        raise ValueError("tutorial contains a prolonged no-task-progress segment")
    if env.state.shutdown_count:
        raise ValueError("tutorial must complete without a battery shutdown")
    for agent_id in ("robot_1", "robot_2"):
        actions = [frame["actions"].get(agent_id)
                   for frame in payload["frames"][1:]]
        longest_wait = current = 0
        for action in actions:
            current = current + 1 if action == "WAIT" else 0
            longest_wait = max(longest_wait, current)
        # Charging and a short shared-charger queue can require waiting; a
        # sustained inactive teaching robot is still rejected.
        if longest_wait > 16:
            raise ValueError("r4.2 tutorial robot remains stationary too long")
    return {"passed": True, "frame_count": FRAME_COUNT,
            "tutorial_signature": digest(dict(payload)),
            "individual_deliveries": deliveries,
            "delivery_windows": delivery_windows,
            "simultaneous_movement_count": len(recovered["simultaneous_movement_frames"]),
            "longest_no_task_progress": longest_no_progress,
            "maximum_consecutive_waits": {
                agent_id: max(
                    len(run) for run in "".join(
                        "1" if frame["actions"].get(agent_id) == "WAIT" else "0"
                        for frame in payload["frames"][1:]
                    ).split("0")
                ) for agent_id in ("robot_1", "robot_2")
            },
            "coverage": deepcopy(recovered)}


__all__ = ["VERSION", "DURATION_MS", "FRAME_COUNT", "build_tutorial",
           "validate_tutorial"]
