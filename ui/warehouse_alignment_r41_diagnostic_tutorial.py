"""Build and physically replay the neutral r4.1 diagnostic tutorial."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from backend.training.warehouse_native_common import digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
    R41DiagnosticOnlineAlignmentRuntime,
)
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
    reset_diagnostic_scenario,
)
from ui import warehouse_alignment_online_server as server
from ui import warehouse_alignment_r41_tutorial as base


ROOT = Path(__file__).resolve().parents[1]
VERSION = "warehouse-alignment-diagnostic-neutral-tutorial.v3"
REPLAY_VERSION = "warehouse-alignment-diagnostic-neutral-tutorial-replay.v3"
SCENE_MANIFEST_VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
SOURCE = "independent_neutral_ai_ai"
DURATION_MS = 380
_COVERAGE_FIELDS = base._COVERAGE_FIELDS
_PAYLOAD_FIELDS = frozenset((
    "version", "source", "uses_final_actor", "scene_id", "duration_ms",
    "map_sha256", "bindings", "coverage", "frames",
))
_BINDING_FIELDS = frozenset((
    "scene_manifest_version", "scene_manifest_file_sha256",
    "scene_manifest_content_sha256", "scene_manifest_semantic_sha256",
    "tutorial_scene_fingerprint", "tutorial_successor_state_sha256",
    "tutorial_snapshot_sha256", "diagnostic_contract_sha256",
    "diagnostic_contract_version", "diagnostic_conflict_graph_sha256",
    "conflict_families_sha256", "producer_sources_sha256",
))


def producer_sources() -> dict[str, str]:
    paths = (
        Path(__file__), Path(base.__file__),
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_alignment_online_runtime.py",
        ROOT / "env/warehouse_native/r41_diagnostic_conflict.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
        ROOT / "env/warehouse_native/environment.py",
        ROOT / "ui/warehouse_alignment_online_server.py",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Diagnostic tutorial source is missing: " + str(path))
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return dict(sorted(result.items()))


def bindings(manifest: Mapping[str, Any], scene: Mapping[str, Any], *,
             manifest_file_sha256: str) -> dict[str, str]:
    metadata = scene.get("snapshot", {}).get("r41_diagnostic_conflict", {})
    return {
        "scene_manifest_version": str(manifest.get("version", "")),
        "scene_manifest_file_sha256": str(manifest_file_sha256),
        "scene_manifest_content_sha256": str(manifest.get("content_sha256", "")),
        "scene_manifest_semantic_sha256": digest(manifest),
        "tutorial_scene_fingerprint": str(scene.get("fingerprint", "")),
        "tutorial_successor_state_sha256": str(metadata.get("binding_sha256", "")),
        "tutorial_snapshot_sha256": digest(scene.get("snapshot")),
        "diagnostic_contract_sha256": str(scene.get("diagnostic_contract_sha256", "")),
        "diagnostic_contract_version": str(
            manifest.get("diagnostic_contract_version", "")),
        "diagnostic_conflict_graph_sha256": str(
            scene.get("diagnostic_conflict_graph_sha256", "")),
        "conflict_families_sha256": str(scene.get("conflict_families_sha256", "")),
        "producer_sources_sha256": digest(producer_sources()),
    }


def build_neutral_tutorial(manifest: Mapping[str, Any], *,
                           manifest_file_sha256: str) -> dict[str, Any]:
    from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
        validate_diagnostic_manifest,
    )

    validate_diagnostic_manifest(manifest, replay=True)
    rows = manifest.get("splits", {}).get("tutorial")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("Exactly one diagnostic tutorial scene is required")
    scene = deepcopy(rows[0])
    env = R41DiagnosticConflictWarehouseEnv()
    reset_diagnostic_scenario(env, scene)
    if (tuple(env.layout.robot_start_positions) != ((5, 2), (5, 4))
            or tuple(env.layout.charger_position) != (5, 3)):
        raise ValueError("Diagnostic tutorial choreography requires the frozen layout")

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
    for joint in prelude:
        metrics, _ = base._append_step(env, frames, coverage, metrics, joint)

    worker = env.state.by_id("robot_1")
    partner = env.state.by_id("robot_2")
    candidates = []
    for task in env.state.tasks:
        to_pickup = base._path_actions(
            env, worker.position, task.pickup_position,
            blocked=(partner.position,),
        )
        to_delivery = base._path_actions(
            env, task.pickup_position, task.delivery_position,
            blocked=(partner.position,),
        )
        candidates.append((len(to_pickup) + len(to_delivery), task.task_id,
                           task, to_pickup, to_delivery))
    _, task_id, task, to_pickup, _ = min(candidates, key=lambda row: row[:2])
    for action in to_pickup:
        metrics, _ = base._append_step(
            env, frames, coverage, metrics,
            {"robot_1": action, "robot_2": "WAIT"},
        )
    if env.state.by_id("robot_1").carrying_task_id != task_id:
        raise ValueError("Diagnostic tutorial did not physically pick up its task")
    to_delivery = base._path_actions(
        env, env.state.by_id("robot_1").position, task.delivery_position,
        blocked=(env.state.by_id("robot_2").position,),
    )
    delivery_seen = False
    for action in to_delivery:
        metrics, info = base._append_step(
            env, frames, coverage, metrics,
            {"robot_1": action, "robot_2": "WAIT"},
        )
        delivery_seen |= any(
            event.get("event") == "delivery" and event.get("task_id") == task_id
            for event in info["events"]
        )
    if not delivery_seen or any(not coverage[name] for name in _COVERAGE_FIELDS):
        raise ValueError("Diagnostic tutorial is missing a required physical event")
    payload = {
        "version": VERSION,
        "source": SOURCE,
        "uses_final_actor": False,
        "scene_id": scene["id"],
        "duration_ms": DURATION_MS,
        "map_sha256": server._digest(server._public_map(env)),
        "bindings": bindings(
            manifest, scene, manifest_file_sha256=manifest_file_sha256,
        ),
        "coverage": coverage,
        "frames": frames,
    }
    validate_neutral_tutorial(payload, tutorial_scene=scene)
    return payload


def validate_neutral_tutorial(
    payload: Mapping[str, Any], *, tutorial_scene: Mapping[str, Any],
    runtime: Any | None = None,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if (not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS
            or payload.get("version") != VERSION or payload.get("source") != SOURCE
            or payload.get("uses_final_actor") is not False
            or payload.get("scene_id") != tutorial_scene.get("id")
            or payload.get("duration_ms") != DURATION_MS):
        raise ValueError("Exact diagnostic neutral tutorial payload required")
    base._check_public(payload)
    actual_bindings = payload.get("bindings")
    if (not isinstance(actual_bindings, Mapping)
            or set(actual_bindings) != _BINDING_FIELDS):
        raise ValueError("Diagnostic tutorial binding schema differs")
    if expected_bindings is not None and dict(actual_bindings) != dict(expected_bindings):
        raise ValueError("Diagnostic tutorial release binding differs")
    for name, value in actual_bindings.items():
        if name == "scene_manifest_version":
            if value != SCENE_MANIFEST_VERSION:
                raise ValueError("Diagnostic tutorial manifest version differs")
        elif name == "diagnostic_contract_version":
            if value != DIAGNOSTIC_CONTRACT_VERSION:
                raise ValueError("Diagnostic tutorial contract version differs")
        elif (type(value) is not str or len(value) != 64
              or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("Diagnostic tutorial binding is not a lowercase SHA-256")
    metadata = tutorial_scene.get("snapshot", {}).get(
        "r41_diagnostic_conflict", {})
    direct = {
        "tutorial_scene_fingerprint": tutorial_scene.get("fingerprint"),
        "tutorial_successor_state_sha256": metadata.get("binding_sha256"),
        "tutorial_snapshot_sha256": digest(tutorial_scene.get("snapshot")),
        "diagnostic_contract_sha256": tutorial_scene.get(
            "diagnostic_contract_sha256"),
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_conflict_graph_sha256": tutorial_scene.get(
            "diagnostic_conflict_graph_sha256"),
        "conflict_families_sha256": tutorial_scene.get(
            "conflict_families_sha256"),
        "producer_sources_sha256": digest(producer_sources()),
    }
    if any(actual_bindings.get(name) != value for name, value in direct.items()):
        raise ValueError("Diagnostic tutorial scene or source binding changed")
    if (actual_bindings["diagnostic_contract_sha256"]
            != DIAGNOSTIC_CONTRACT_SHA256
            or actual_bindings["diagnostic_contract_version"]
                != DIAGNOSTIC_CONTRACT_VERSION
            or actual_bindings["diagnostic_conflict_graph_sha256"]
                != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
            or actual_bindings["conflict_families_sha256"]
                != CONFLICT_FAMILIES_SHA256):
        raise ValueError("Diagnostic tutorial uses another conflict contract")

    coverage = payload.get("coverage")
    if (not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_FIELDS
            or any(not isinstance(values, list) or not values
                   or any(type(frame) is not int or frame < 1 for frame in values)
                   or len(values) != len(set(values))
                   for values in coverage.values())):
        raise ValueError("Diagnostic tutorial coverage declaration differs")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 121:
        raise ValueError("Diagnostic tutorial frame count is invalid")
    if runtime is None:
        env = R41DiagnosticConflictWarehouseEnv()
        reset_diagnostic_scenario(env, deepcopy(tutorial_scene))
    else:
        if type(runtime) is not R41DiagnosticOnlineAlignmentRuntime:
            raise ValueError("Diagnostic tutorial requires the diagnostic runtime")
        runtime.verify_binding()
        env = runtime.environment(deepcopy(tutorial_scene))
    if payload.get("map_sha256") != server._digest(server._public_map(env)):
        raise ValueError("Diagnostic tutorial map binding differs")

    metrics = base._metrics(env)
    if frames[0] != base._public_frame(env, metrics, {}):
        raise ValueError("Diagnostic tutorial initial public frame differs")
    recovered = base._empty_coverage()
    for index, saved in enumerate(frames[1:], 1):
        if not isinstance(saved, Mapping) or set(saved) != {"state", "metrics", "actions"}:
            raise ValueError("Diagnostic tutorial frame schema differs")
        actions = saved.get("actions")
        if (not isinstance(actions, Mapping)
                or set(actions) != {"robot_1", "robot_2"}
                or any(action not in base.ACTIONS for action in actions.values())):
            raise ValueError("Diagnostic tutorial replay actions differ")
        if env.done:
            raise ValueError("Diagnostic tutorial contains post-terminal frames")
        _, _, terminated, truncated, info = env.step(dict(actions))
        metrics = base._metrics(env, metrics, info)
        expected = base._public_frame(env, metrics, actions)
        if saved != expected or saved["state"].get("frame") != index:
            raise ValueError("Diagnostic tutorial contains a fabricated frame")
        base._coverage_update(recovered, info, int(env.state.frame))
        if terminated or truncated:
            raise ValueError("Diagnostic tutorial terminates during playback")
    if dict(coverage) != recovered or any(not recovered[name] for name in _COVERAGE_FIELDS):
        raise ValueError("Diagnostic tutorial physical coverage differs")
    return {
        "version": REPLAY_VERSION,
        "passed": True,
        "uses_final_actor": False,
        "tutorial_signature": digest(dict(payload)),
        "scene_id": tutorial_scene["id"],
        "frame_count": len(frames),
        "coverage": deepcopy(recovered),
        "bindings": deepcopy(dict(actual_bindings)),
    }


__all__ = [
    "VERSION", "REPLAY_VERSION", "SCENE_MANIFEST_VERSION", "SOURCE", "DURATION_MS", "bindings",
    "build_neutral_tutorial", "validate_neutral_tutorial", "producer_sources",
]
