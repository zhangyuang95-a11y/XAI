"""Build and replay-verify the fixed neutral r4.1 warehouse tutorial.

The tutorial is a sequence of real joint transitions in the dedicated
``tutorial`` split.  It never asks the frozen study Actor for an action.  The
saved actions are the two neutral demonstrators' requested actions, so a
loader can replay every physical frame and independently recover pickup,
delivery, movement, collision, waiting, and charging evidence.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.warehouse_alignment_online_runtime import ACTIONS, digest, file_hash
from backend.warehouse_r41_online_runtime import (
    R41ConflictWarehouseEnv,
    R41OnlineAlignmentRuntime,
)
from env.warehouse.navigation import MOVE_DELTAS
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH_SHA256,
    CONTRACT_SHA256,
    reset_r41_scenario,
)
from ui import warehouse_alignment_online_server as server


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-alignment-neutral-tutorial.v1"
SOURCE = "independent_neutral_ai_ai"
DURATION_MS = 380
_COVERAGE_FIELDS = frozenset((
    "pickup_frames", "delivery_frames", "simultaneous_movement_frames",
    "collision_frames", "wait_frames", "charge_frames",
))
_BINDING_FIELDS = frozenset((
    "scene_manifest_version", "scene_manifest_content_sha256",
    "tutorial_scene_fingerprint", "tutorial_successor_state_sha256",
    "tutorial_snapshot_sha256", "conflict_contract_sha256",
    "conflict_graph_sha256", "producer_sources_sha256",
))
_PAYLOAD_FIELDS = frozenset((
    "version", "source", "uses_final_actor", "scene_id", "duration_ms",
    "map_sha256", "bindings", "coverage", "frames",
))
_FORBIDDEN_PUBLIC_FIELDS = frozenset((
    "probabilities", "logits", "decision", "policy_actions",
    "proposed_actions", "observation_hashes", "program", "tree",
    "target_goal",
))


def producer_sources() -> dict[str, str]:
    """Current code bytes that define tutorial physics and projection."""
    paths = (
        Path(__file__),
        ROOT / "ui/warehouse_alignment_online_server.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "env/warehouse_native/environment.py",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Neutral tutorial source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return dict(sorted(result.items()))


def _bindings(manifest: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, str]:
    successor = scene.get("snapshot", {}).get("r41_conflict", {}).get(
        "successor_state_sha256")
    return {
        "scene_manifest_version": str(manifest.get("version", "")),
        "scene_manifest_content_sha256": str(manifest.get("content_sha256", "")),
        "tutorial_scene_fingerprint": str(scene.get("fingerprint", "")),
        "tutorial_successor_state_sha256": str(successor or ""),
        "tutorial_snapshot_sha256": digest(scene.get("snapshot")),
        "conflict_contract_sha256": str(scene.get("contract_sha256", "")),
        "conflict_graph_sha256": str(scene.get("conflict_graph_sha256", "")),
        "producer_sources_sha256": digest(producer_sources()),
    }


def _metrics(env, previous: Mapping[str, Any] | None = None,
             info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = env.state
    values = dict(previous or {"ai_waits": 0, "ai_blocked": 0, "overrides": 0})
    values.update(
        steps=int(state.frame), deliveries=int(state.total_deliveries),
        score=float(state.user_score), native_score=float(state.user_score),
        legacy_score=None, collisions=int(state.robot_collision_events),
        shutdowns=int(state.shutdown_count),
        individual_deliveries=[int(agent.deliveries_completed)
                               for agent in state.agents],
    )
    if info is not None:
        requested = info["requested_actions"]
        executed = info["executed_actions"]
        values["ai_waits"] += int(requested["robot_2"] == "WAIT")
        values["ai_blocked"] += int(
            requested["robot_2"] != "WAIT" and executed["robot_2"] == "WAIT")
    return values


def _public_frame(env, metrics: Mapping[str, Any],
                  actions: Mapping[str, str]) -> dict[str, Any]:
    state = server._public_state(env)
    state["public_feedback"] = server._public_history(env)
    return {
        "state": state,
        "metrics": deepcopy(dict(metrics)),
        # Requested joint actions are required for independent physics replay.
        "actions": deepcopy(dict(actions)),
    }


def _path_actions(env, start: tuple[int, int], goal: tuple[int, int],
                  *, blocked: Sequence[tuple[int, int]] = ()) -> list[str]:
    if start == goal:
        return []
    blocked_set = set(blocked) - {start, goal}
    order = ("UP", "LEFT", "RIGHT", "DOWN")
    queue = deque([start])
    previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    seen = {start}
    while queue:
        position = queue.popleft()
        for action in order:
            delta = MOVE_DELTAS[action]
            target = (position[0] + delta[0], position[1] + delta[1])
            if target in seen or target in blocked_set or not env.layout.is_passable(target):
                continue
            seen.add(target)
            previous[target] = (position, action)
            if target == goal:
                actions = []
                cursor = goal
                while cursor != start:
                    cursor, step = previous[cursor]
                    actions.append(step)
                return list(reversed(actions))
            queue.append(target)
    raise ValueError("Neutral tutorial target has no collision-free path")


def _coverage_update(result: dict[str, list[int]], info: Mapping[str, Any],
                     frame: int) -> None:
    requested = info["requested_actions"]
    executed = info["executed_actions"]
    events = {event.get("event") for event in info["events"]}
    if "pickup" in events:
        result["pickup_frames"].append(frame)
    if "delivery" in events:
        result["delivery_frames"].append(frame)
    if all(executed[agent] in MOVE_DELTAS for agent in ("robot_1", "robot_2")):
        result["simultaneous_movement_frames"].append(frame)
    if info["robot_collision"]:
        result["collision_frames"].append(frame)
    if all(requested[agent] == "WAIT" for agent in ("robot_1", "robot_2")):
        result["wait_frames"].append(frame)
    if "charge" in events:
        result["charge_frames"].append(frame)


def _empty_coverage() -> dict[str, list[int]]:
    return {name: [] for name in sorted(_COVERAGE_FIELDS)}


def _append_step(env, frames: list[dict[str, Any]], coverage: dict[str, list[int]],
                 metrics: Mapping[str, Any], actions: Mapping[str, str]):
    if set(actions) != {"robot_1", "robot_2"} or any(
            value not in ACTIONS for value in actions.values()):
        raise ValueError("Neutral tutorial requires two valid requested actions")
    _, _, terminated, truncated, info = env.step(dict(actions))
    next_metrics = _metrics(env, metrics, info)
    frames.append(_public_frame(env, next_metrics, actions))
    _coverage_update(coverage, info, int(env.state.frame))
    if terminated or truncated:
        raise ValueError("Neutral tutorial ended before all teaching events")
    return next_metrics, info


def build_neutral_tutorial(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the fixed tutorial without invoking any Actor.

    The short prelude demonstrates a real same-target collision, an explicit
    joint wait, charging, and simultaneous movement.  One demonstrator then
    completes the shortest available delivery while the other stays clear.
    """
    from backend.training.warehouse_r41_conflict_scenarios import (
        validate_conflict_manifest,
    )

    validate_conflict_manifest(manifest, replay=True)
    rows = manifest.get("splits", {}).get("tutorial")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("Exactly one registered r4.1 tutorial scene is required")
    scene = deepcopy(rows[0])
    env = R41ConflictWarehouseEnv()
    reset_r41_scenario(env, scene)
    if (tuple(env.layout.robot_start_positions) != ((5, 2), (5, 4))
            or tuple(env.layout.charger_position) != (5, 3)):
        raise ValueError("Neutral tutorial choreography requires the frozen 6x7 layout")

    metrics = _metrics(env)
    frames = [_public_frame(env, metrics, {})]
    coverage = _empty_coverage()
    # Every item is a real joint environment step.  No final Actor is queried.
    prelude = (
        {"robot_1": "RIGHT", "robot_2": "LEFT"},  # same-target collision
        {"robot_1": "WAIT", "robot_2": "WAIT"},   # explicit wait
        {"robot_1": "RIGHT", "robot_2": "WAIT"},  # enter charger
        {"robot_1": "WAIT", "robot_2": "UP"},     # actual charge event
        {"robot_1": "UP", "robot_2": "DOWN"},     # both execute movement
    )
    for joint in prelude:
        metrics, _ = _append_step(env, frames, coverage, metrics, joint)

    worker = env.state.by_id("robot_1")
    partner = env.state.by_id("robot_2")
    if worker.carrying_task_id is not None:
        raise ValueError("Neutral tutorial prelude unexpectedly claimed a task")
    candidates = []
    for task in env.state.tasks:
        to_pickup = _path_actions(env, worker.position, task.pickup_position,
                                  blocked=(partner.position,))
        to_delivery = _path_actions(env, task.pickup_position,
                                    task.delivery_position,
                                    blocked=(partner.position,))
        candidates.append((len(to_pickup) + len(to_delivery), task.task_id,
                           task, to_pickup, to_delivery))
    _, task_id, task, to_pickup, to_delivery = min(candidates, key=lambda row: row[:2])
    for action in to_pickup:
        metrics, _ = _append_step(
            env, frames, coverage, metrics,
            {"robot_1": action, "robot_2": "WAIT"},
        )
    if env.state.by_id("robot_1").carrying_task_id != task_id:
        raise ValueError("Neutral tutorial did not physically pick up its selected task")
    # Recompute from the confirmed pickup position to keep saved actions bound
    # to physical state rather than to a separately assumed route.
    to_delivery = _path_actions(
        env, env.state.by_id("robot_1").position, task.delivery_position,
        blocked=(env.state.by_id("robot_2").position,),
    )
    delivery_seen = False
    for action in to_delivery:
        metrics, info = _append_step(
            env, frames, coverage, metrics,
            {"robot_1": action, "robot_2": "WAIT"},
        )
        delivery_seen |= any(event.get("event") == "delivery"
                             and event.get("task_id") == task_id
                             for event in info["events"])
    if not delivery_seen:
        raise ValueError("Neutral tutorial did not physically deliver its selected task")

    if any(not coverage[name] for name in _COVERAGE_FIELDS):
        raise ValueError("Neutral tutorial is missing a required physical event")
    payload = {
        "version": VERSION,
        "source": SOURCE,
        "uses_final_actor": False,
        "scene_id": scene["id"],
        "duration_ms": DURATION_MS,
        "map_sha256": server._digest(server._public_map(env)),
        "bindings": _bindings(manifest, scene),
        "coverage": coverage,
        "frames": frames,
    }
    # The same validator used by the package loader performs a full replay.
    validate_neutral_tutorial(payload, tutorial_scene=scene)
    return payload


def _check_public(value: Any) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_PUBLIC_FIELDS.intersection(value):
            raise ValueError("Neutral tutorial exposes policy-internal evidence")
        for child in value.values():
            _check_public(child)
    elif isinstance(value, list):
        for child in value:
            _check_public(child)


def validate_neutral_tutorial(
    payload: Mapping[str, Any], *, tutorial_scene: Mapping[str, Any],
    runtime: Any | None = None, expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Replay every requested joint action and compare every public frame."""
    if (not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS
            or payload.get("version") != VERSION or payload.get("source") != SOURCE
            or payload.get("uses_final_actor") is not False
            or payload.get("scene_id") != tutorial_scene.get("id")
            or payload.get("duration_ms") != DURATION_MS):
        raise ValueError("Exact neutral tutorial payload required")
    _check_public(payload)
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _BINDING_FIELDS:
        raise ValueError("Neutral tutorial binding schema differs")
    if expected_bindings is not None and dict(bindings) != dict(expected_bindings):
        raise ValueError("Neutral tutorial release binding differs")
    for name, value in bindings.items():
        if name == "scene_manifest_version":
            if value != "warehouse-r41-conflict-scene-manifest.v1":
                raise ValueError("Neutral tutorial scene manifest version differs")
        elif not isinstance(value, str) or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Neutral tutorial binding is not a lowercase SHA-256")
    scene_metadata = tutorial_scene.get("snapshot", {}).get("r41_conflict", {})
    direct = {
        "tutorial_scene_fingerprint": tutorial_scene.get("fingerprint"),
        "tutorial_successor_state_sha256": scene_metadata.get(
            "successor_state_sha256"),
        "tutorial_snapshot_sha256": digest(tutorial_scene.get("snapshot")),
        "conflict_contract_sha256": tutorial_scene.get("contract_sha256"),
        "conflict_graph_sha256": tutorial_scene.get("conflict_graph_sha256"),
        "producer_sources_sha256": digest(producer_sources()),
    }
    if any(bindings.get(name) != value for name, value in direct.items()):
        raise ValueError("Neutral tutorial scene or source binding changed")
    if (bindings["conflict_contract_sha256"] != CONTRACT_SHA256
            or bindings["conflict_graph_sha256"] != CONFLICT_GRAPH_SHA256):
        raise ValueError("Neutral tutorial uses another conflict contract")

    coverage = payload.get("coverage")
    if (not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_FIELDS
            or any(not isinstance(values, list) or not values
                   or any(type(frame) is not int or frame < 1 for frame in values)
                   or len(values) != len(set(values))
                   for values in coverage.values())):
        raise ValueError("Neutral tutorial coverage declaration differs")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 121:
        raise ValueError("Neutral tutorial frame count is invalid")

    if runtime is None:
        env = R41ConflictWarehouseEnv()
        reset_r41_scenario(env, deepcopy(tutorial_scene))
    else:
        if type(runtime) is not R41OnlineAlignmentRuntime:
            raise ValueError("Neutral tutorial replay requires the r4.1 runtime")
        runtime.verify_binding()
        env = runtime.environment(deepcopy(tutorial_scene))
    if payload.get("map_sha256") != server._digest(server._public_map(env)):
        raise ValueError("Neutral tutorial map binding differs")

    metrics = _metrics(env)
    expected_initial = _public_frame(env, metrics, {})
    if frames[0] != expected_initial:
        raise ValueError("Neutral tutorial initial public frame differs")
    recovered = _empty_coverage()
    for index, saved in enumerate(frames[1:], 1):
        if not isinstance(saved, Mapping) or set(saved) != {"state", "metrics", "actions"}:
            raise ValueError("Neutral tutorial public frame schema differs")
        actions = saved.get("actions")
        if (not isinstance(actions, Mapping)
                or set(actions) != {"robot_1", "robot_2"}
                or any(action not in ACTIONS for action in actions.values())):
            raise ValueError("Neutral tutorial replay actions differ")
        if env.done:
            raise ValueError("Neutral tutorial contains transitions after termination")
        _, _, terminated, truncated, info = env.step(dict(actions))
        metrics = _metrics(env, metrics, info)
        expected = _public_frame(env, metrics, actions)
        if saved != expected or saved["state"].get("frame") != index:
            raise ValueError("Neutral tutorial contains a fabricated or spliced frame")
        _coverage_update(recovered, info, int(env.state.frame))
        if terminated or truncated:
            raise ValueError("Neutral tutorial terminates during teaching playback")
    if dict(coverage) != recovered or any(not recovered[name] for name in _COVERAGE_FIELDS):
        raise ValueError("Neutral tutorial physical coverage differs after replay")
    return {
        "version": "warehouse-alignment-neutral-tutorial-replay.v1",
        "passed": True,
        "uses_final_actor": False,
        "tutorial_signature": digest(dict(payload)),
        "scene_id": tutorial_scene["id"],
        "frame_count": len(frames),
        "coverage": deepcopy(recovered),
        "bindings": deepcopy(dict(bindings)),
    }


__all__ = [
    "VERSION", "SOURCE", "DURATION_MS", "build_neutral_tutorial",
    "validate_neutral_tutorial", "producer_sources",
]
