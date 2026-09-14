"""Build and replay the complete 120-step neutral r4.2 teaching demo."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from backend.training.warehouse_native_common import digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from ui import warehouse_alignment_online_server as projection
from ui import warehouse_alignment_r41_tutorial as base


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-alignment-r42-neutral-tutorial.v1"
SOURCE = "independent_neutral_ai_ai"
DURATION_MS = 380
FRAME_COUNT = 121
_COVERAGE_FIELDS = frozenset((
    "pickup_frames", "delivery_frames", "simultaneous_movement_frames",
    "collision_frames", "wait_frames", "charge_frames",
))


def _source_sha256() -> str:
    paths = (
        Path(__file__), ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "ui/warehouse_alignment_r41_tutorial.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
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


def _deliver(env, frames, coverage, metrics, worker_id, other_id):
    worker, other = env.state.by_id(worker_id), env.state.by_id(other_id)
    candidates = []
    for task in env.state.tasks:
        if task.status != "available":
            continue
        try:
            first = base._path_actions(
                env, worker.position, task.pickup_position,
                blocked=(other.position,),
            )
            second = base._path_actions(
                env, task.pickup_position, task.delivery_position,
                blocked=(other.position,),
            )
        except ValueError:
            continue
        candidates.append((len(first) + len(second), task.task_id, first))
    if not candidates:
        raise ValueError("r4.2 neutral worker has no delivery route")
    _, task_id, first = min(candidates)
    for action in first:
        metrics, _ = _append(
            env, frames, coverage, metrics,
            {worker_id: action, other_id: "WAIT"},
        )
    if env.state.by_id(worker_id).carrying_task_id != task_id:
        raise ValueError("r4.2 neutral worker failed to collect")
    task = next(item for item in env.state.tasks if item.task_id == task_id)
    second = base._path_actions(
        env, env.state.by_id(worker_id).position, task.delivery_position,
        blocked=(env.state.by_id(other_id).position,),
    )
    delivered = False
    for action in second:
        metrics, info = _append(
            env, frames, coverage, metrics,
            {worker_id: action, other_id: "WAIT"},
        )
        delivered |= any(event.get("event") == "delivery"
                         and event.get("task_id") == task_id
                         for event in info["events"])
    if not delivered:
        raise ValueError("r4.2 neutral worker failed to deliver")
    return metrics


def _move_to(env, frames, coverage, metrics, worker_id, other_id, target):
    """Move one teaching robot to a public waypoint while the other waits."""
    worker = env.state.by_id(worker_id)
    other = env.state.by_id(other_id)
    for action in base._path_actions(
            env, worker.position, target, blocked=(other.position,)):
        metrics, _ = _append(
            env, frames, coverage, metrics,
            {worker_id: action, other_id: "WAIT"},
        )
    if env.state.by_id(worker_id).position != target:
        raise ValueError("r4.2 neutral worker failed to reach waypoint")
    return metrics


def _active_tail(env, frames, coverage, metrics):
    """Keep both robots visibly active and energy-safe through step 119.

    The eight-step choreography uses only the three-cell exit beside the
    charger.  Each robot moves on six of every eight steps and receives two
    charging turns, so the full tutorial remains physically valid without
    exposing any preference of the deployed Actor.
    """
    metrics = _move_to(
        env, frames, coverage, metrics, "robot_2", "robot_1", (5, 4))
    metrics = _move_to(
        env, frames, coverage, metrics, "robot_1", "robot_2", (5, 2))
    cycle = (
        {"robot_1": "RIGHT", "robot_2": "UP"},
        {"robot_1": "WAIT", "robot_2": "DOWN"},
        {"robot_1": "WAIT", "robot_2": "UP"},
        {"robot_1": "LEFT", "robot_2": "DOWN"},
        {"robot_1": "UP", "robot_2": "LEFT"},
        {"robot_1": "DOWN", "robot_2": "WAIT"},
        {"robot_1": "UP", "robot_2": "WAIT"},
        {"robot_1": "DOWN", "robot_2": "RIGHT"},
    )
    offset = 0
    while env.state.frame < 119:
        actions = cycle[offset % len(cycle)]
        metrics, _ = _append(env, frames, coverage, metrics, actions)
        offset += 1
    return metrics


def build_tutorial(scenarios: Mapping[str, Any]) -> dict[str, Any]:
    play = scenarios.get("splits", {}).get("play")
    if not isinstance(play, list) or len(play) < 7:
        raise ValueError("r4.2 requires tutorial plus six frozen scenes")
    scene = deepcopy(play[0])
    env = R41DiagnosticConflictWarehouseEnv()
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
    metrics = _deliver(env, frames, coverage, metrics, "robot_1", "robot_2")
    metrics = _deliver(env, frames, coverage, metrics, "robot_2", "robot_1")
    metrics = _active_tail(env, frames, coverage, metrics)
    metrics, _ = _append(
        env, frames, coverage, metrics,
        {"robot_1": "WAIT", "robot_2": "WAIT"}, final=True,
    )
    payload = {
        "version": VERSION,
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
            or payload.get("version") != VERSION
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
    env = R41DiagnosticConflictWarehouseEnv()
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
    if min(deliveries) < 1:
        raise ValueError("both r4.2 tutorial robots must contribute a delivery")
    for agent_id in ("robot_1", "robot_2"):
        actions = [frame["actions"].get(agent_id)
                   for frame in payload["frames"][1:]]
        longest_wait = current = 0
        for action in actions:
            current = current + 1 if action == "WAIT" else 0
            longest_wait = max(longest_wait, current)
        # A robot may wait while its partner demonstrates one complete
        # pickup-to-delivery route, but the long inactive tail from r4.1 is
        # forbidden.
        if longest_wait > 16:
            raise ValueError("r4.2 tutorial robot remains stationary too long")
    return {"passed": True, "frame_count": FRAME_COUNT,
            "tutorial_signature": digest(dict(payload)),
            "individual_deliveries": deliveries,
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
