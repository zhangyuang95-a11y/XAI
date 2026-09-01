"""Pure validation helpers for deployable collaborative-study artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    REFERENCE_TRAJECTORY_FORMAT,
    RCPD_PROGRAM_VERSION,
    RUNTIME_ACTION_SOURCE,
    RUNTIME_CONTROLLER,
)


def validate_posthoc_rcpd_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject programs that were not distilled one-way from Actor execution."""

    if (
        metadata.get("warehouse_program_version") != RCPD_PROGRAM_VERSION
        or metadata.get("distilled_component") != "neural_actor"
        or metadata.get("training_data_source")
        != "executed_neural_rollout_only"
        or metadata.get("action_execution_version")
        != ACTION_EXECUTION_VERSION
        or metadata.get("runtime_controller") != RUNTIME_CONTROLLER
        or bool(metadata.get("runtime_control_allowed", True))
        or bool(metadata.get("feedback_allowed", True))
        or metadata.get("training_role") != "posthoc_explanation_only"
        or tuple(metadata.get("program_roles", ()))
        != ("local_explanation_audit",)
        or bool(metadata.get("regularization_version", True))
    ):
        raise ValueError(
            "The RCPD program is not a post-hoc explanation-only distillation "
            "of directly executed MAPPO Actor trajectories."
        )


def validate_reference_trajectory_manifest(
    payload: Mapping[str, Any],
    *,
    model_version: str,
    environment_version: str,
    map_layout_id: str,
    frame_count: int = 121,
) -> None:
    """Reject reference trajectories with any non-neural action source."""

    identity = dict(payload)
    claimed_hash = str(identity.pop("trajectory_manifest_hash", ""))
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_hash_valid = bool(
        claimed_hash and claimed_hash == sha256(encoded).hexdigest()
    )
    if (
        payload.get("format") != REFERENCE_TRAJECTORY_FORMAT
        or payload.get("model_version") != model_version
        or payload.get("environment_version") != environment_version
        or payload.get("warehouse_program_version") != RCPD_PROGRAM_VERSION
        or payload.get("map_layout_id") != map_layout_id
        or payload.get("action_execution_version") != ACTION_EXECUTION_VERSION
        or payload.get("runtime_controller") != RUNTIME_CONTROLLER
        or payload.get("rollout_action_source") != RUNTIME_ACTION_SOURCE
        or int(payload.get("post_policy_action_interventions", -1)) < 0
        or int(payload.get("frame_count", 0)) != int(frame_count)
        or not bool(payload.get("frozen", False))
        or bool(payload.get("battery_shutdown", True))
        or dict(payload.get("agent_control", {}))
        != {"robot_1": "ai", "robot_2": "ai"}
        or not manifest_hash_valid
    ):
        raise ValueError(
            "The reference trajectory is not a frozen causal-runtime AI-AI "
            "trajectory for the current artifact contract."
        )
