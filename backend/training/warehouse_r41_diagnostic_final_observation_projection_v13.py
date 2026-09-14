"""Observation-only projection for warehouse protected-final screening.

This module mirrors the exact public observations produced by the frozen v7
collector without returning Actor actions, probabilities, program labels, or
raw observations.  A post-claim materializer may therefore reject a candidate
scene whose eventual audit observations overlap an exposed split while keeping
all target values unavailable to selection.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import digest


VERSION = "warehouse-r41-diagnostic-final-observation-projection.v13"


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "partners": list(rows_api.PARTNERS),
        "actions": list(rows_api.ACTIONS),
        "partner_rng_scheme": (
            "41900000 + (scene_offset + local_scene_index) * 101 + partner_index"
        ),
        "ordinary_observation_before_each_joint_step": True,
        "critical_anchor_period_when_not_dense": 5,
        "all_player_action_intervention_endpoints": True,
        "terminal_intervention_endpoints_omitted": True,
        "actor_inference_only_inside_environment_step": True,
        "runtime_decision_output_read": False,
        "raw_observations_returned": False,
        "actor_actions_returned": False,
        "actor_probabilities_returned": False,
        "program_access": False,
        "labels_returned": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _observation_hash(value: np.ndarray) -> str:
    # Use the exact collector hash implementation.  The returned digest is a
    # one-way public projection; the observation itself never leaves this frame.
    return rows_api.legacy._obs_hash(value)


def project_observation_hashes(
    runtime: Any, scenes: Sequence[Mapping[str, Any]], *, scene_offset: int,
    dense_critical: bool = False,
) -> dict[str, Any]:
    """Project the exact v7 audit input hashes, in collector row order."""
    if (type(scene_offset) is not int or isinstance(scene_offset, bool)
            or scene_offset < 0 or type(dense_critical) is not bool
            or not isinstance(scenes, Sequence) or not scenes):
        raise ValueError("Observation projection schedule differs")
    ordered: list[str] = []
    scene_summaries: list[dict[str, Any]] = []
    environment_steps = 0
    for local_index, raw_scene in enumerate(scenes):
        if (not isinstance(raw_scene, Mapping)
                or type(raw_scene.get("fingerprint")) is not str
                or not raw_scene["fingerprint"]):
            raise ValueError("Observation projection scene identity differs")
        scene = deepcopy(dict(raw_scene))
        scene_index = scene_offset + local_index
        start = len(ordered)
        for partner_index, partner in enumerate(rows_api.PARTNERS):
            env = runtime.environment(scene)
            rng = np.random.default_rng(
                41_900_000 + scene_index * 101 + partner_index)
            while not env.done:
                source = env.snapshot()
                source_sha = digest(source)
                observation = env.observations()["robot_2"]
                ordered.append(_observation_hash(observation))
                groups = tuple(rows_api.critical_groups(env, "robot_2"))
                anchor_enabled = bool(groups) and (
                    dense_critical or env.state.frame % 5 == 0)
                if anchor_enabled:
                    for player_action in rows_api.ACTIONS:
                        branch = runtime.from_snapshot(source)
                        transition = runtime.step(branch, player_action)
                        environment_steps += 1
                        if (transition["submitted_actions"]["robot_2"]
                                != transition["policy_actions"]["robot_2"]):
                            raise RuntimeError(
                                "Observation projection detected an Actor override")
                        if not branch.done:
                            ordered.append(_observation_hash(
                                branch.observations()["robot_2"]))
                player_action = rows_api.partner_action(
                    env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_sha:
                    raise RuntimeError(
                        "Observation projection changed the source state")
                transition = runtime.step(env, player_action)
                environment_steps += 1
                if (transition["submitted_actions"]["robot_2"]
                        != transition["policy_actions"]["robot_2"]):
                    raise RuntimeError(
                        "Observation projection detected an Actor override")
        scene_hashes = ordered[start:]
        scene_summaries.append({
            "local_scene_index": local_index,
            "scene_index": scene_index,
            "fingerprint": scene["fingerprint"],
            "row_count": len(scene_hashes),
            "ordered_observation_hashes_sha256": digest(scene_hashes),
            "unique_observation_count": len(set(scene_hashes)),
            "unique_observation_hashes_sha256": digest(sorted(set(scene_hashes))),
        })
    unique = sorted(set(ordered))
    sources = producer_sources()
    result: dict[str, Any] = {
        "version": VERSION,
        "status": "complete_observation_only_projection",
        "schedule": {
            "scene_offset": scene_offset,
            "scene_count": len(scenes),
            "partners": list(rows_api.PARTNERS),
            "critical_anchor_period": 1 if dense_critical else 5,
            "dense_critical": dense_critical,
        },
        "ordered_observation_hashes": ordered,
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_hashes": unique,
        "unique_observation_hashes_sha256": digest(unique),
        "row_count": len(ordered),
        "unique_observation_count": len(unique),
        "environment_steps": environment_steps,
        "scenes": scene_summaries,
        "scene_summaries_sha256": digest(scene_summaries),
        "information_boundary": {
            "raw_observations_included": False,
            "actor_actions_included": False,
            "actor_probabilities_included": False,
            "program_predictions_included": False,
            "labels_included": False,
            "actor_inference_used_only_to_advance_fixed_replay": True,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "contract": contract(),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    result["content_sha256"] = digest(result)
    return result


def validate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Observation-only projection must be a mapping")
    ordered = value.get("ordered_observation_hashes")
    unique = value.get("unique_observation_hashes")
    scenes = value.get("scenes")
    boundary = value.get("information_boundary")
    if (value.get("version") != VERSION
            or value.get("status") != "complete_observation_only_projection"
            or value.get("contract") != contract()
            or value.get("formal_ready") is not False
            or value.get("content_sha256") != digest({
                key: child for key, child in value.items()
                if key != "content_sha256"})
            or not isinstance(ordered, list) or not isinstance(unique, list)
            or any(type(item) is not str or len(item) != 64
                   or any(char not in "0123456789abcdef" for char in item)
                   for item in ordered)
            or unique != sorted(set(ordered))
            or value.get("row_count") != len(ordered)
            or value.get("unique_observation_count") != len(unique)
            or value.get("ordered_observation_hashes_sha256") != digest(ordered)
            or value.get("unique_observation_hashes_sha256") != digest(unique)
            or not isinstance(scenes, list)
            or value.get("scene_summaries_sha256") != digest(scenes)
            or not isinstance(boundary, Mapping)
            or any(boundary.get(name) is not False for name in (
                "raw_observations_included", "actor_actions_included",
                "actor_probabilities_included", "program_predictions_included",
                "labels_included", "runtime_action_override", "formal_ready"))
            or boundary.get(
                "actor_inference_used_only_to_advance_fixed_replay") is not True
            or value.get("producer_sources_sha256")
                != digest(dict(value.get("producer_sources", {})))):
        raise ValueError("Observation-only projection semantics differ")
    return deepcopy(dict(value))


__all__ = [
    "VERSION", "contract", "producer_sources", "project_observation_hashes",
    "validate_projection",
]
