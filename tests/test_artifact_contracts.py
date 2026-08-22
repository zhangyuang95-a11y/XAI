from __future__ import annotations

from hashlib import sha256
import json

import pytest

from backend.artifact_contracts import (
    validate_posthoc_rcpd_metadata,
    validate_reference_trajectory_manifest,
)
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    REFERENCE_TRAJECTORY_FORMAT,
    RCPD_PROGRAM_VERSION,
    RUNTIME_CONTROLLER,
)


def _program_metadata() -> dict[str, object]:
    return {
        "warehouse_program_version": RCPD_PROGRAM_VERSION,
        "distilled_component": "neural_actor",
        "training_data_source": "executed_neural_rollout_only",
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "runtime_controller": RUNTIME_CONTROLLER,
        "runtime_control_allowed": False,
        "feedback_allowed": False,
        "training_role": "posthoc_explanation_only",
        "program_roles": ["local_explanation_audit"],
        "regularization_version": False,
    }


def _reference_manifest() -> dict[str, object]:
    identity: dict[str, object] = {
        "format": REFERENCE_TRAJECTORY_FORMAT,
        "model_version": "test-model",
        "environment_version": "test-environment",
        "warehouse_program_version": RCPD_PROGRAM_VERSION,
        "map_layout_id": "test-map",
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "runtime_controller": RUNTIME_CONTROLLER,
        "rollout_action_source": "mappo_actor",
        "post_policy_action_interventions": 0,
        "frame_count": 121,
        "frozen": True,
        "battery_shutdown": False,
        "agent_control": {"robot_1": "ai", "robot_2": "ai"},
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**identity, "trajectory_manifest_hash": sha256(encoded).hexdigest()}


def test_posthoc_program_contract_accepts_explanation_only_actor_distillation() -> None:
    validate_posthoc_rcpd_metadata(_program_metadata())


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("runtime_control_allowed", True),
        ("feedback_allowed", True),
        ("training_role", "bounded_program_actor_regularizer"),
        ("program_roles", ["training_regularity_signal"]),
        ("regularization_version", True),
        ("training_data_source", "teacher_rollout"),
    ),
)
def test_posthoc_program_contract_rejects_control_or_feedback_roles(
    field: str,
    unsafe_value: object,
) -> None:
    metadata = _program_metadata()
    metadata[field] = unsafe_value
    with pytest.raises(ValueError, match="post-hoc explanation-only"):
        validate_posthoc_rcpd_metadata(metadata)


def test_reference_contract_accepts_frozen_pure_actor_ai_ai_trajectory() -> None:
    validate_reference_trajectory_manifest(
        _reference_manifest(),
        model_version="test-model",
        environment_version="test-environment",
        map_layout_id="test-map",
    )


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("rollout_action_source", "coordination_teacher"),
        ("post_policy_action_interventions", 1),
        ("runtime_controller", "coordination_shield"),
        ("agent_control", {"robot_1": "human", "robot_2": "ai"}),
        ("frozen", False),
        ("battery_shutdown", True),
        ("trajectory_manifest_hash", "tampered"),
    ),
)
def test_reference_contract_rejects_non_neural_or_mutable_trajectory(
    field: str,
    unsafe_value: object,
) -> None:
    payload = _reference_manifest()
    payload[field] = unsafe_value
    with pytest.raises(ValueError, match="pure-neural AI-AI"):
        validate_reference_trajectory_manifest(
            payload,
            model_version="test-model",
            environment_version="test-environment",
            map_layout_id="test-map",
        )
