from __future__ import annotations

from hashlib import sha256
import json

import pytest

from backend.artifact_contracts import (
    validate_posthoc_rcpd_metadata,
    validate_reference_trajectory_manifest,
)
from backend.training.release_staggered_policy import (
    _release_acceptance_gates,
    _validated_formal_evaluation,
)
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    REFERENCE_TRAJECTORY_FORMAT,
    RCPD_PROGRAM_VERSION,
    RUNTIME_CONTROLLER,
    MODEL_VERSION,
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


def test_release_packaging_requires_matching_strict_formal_gate(tmp_path) -> None:
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"candidate checkpoint")
    formal = tmp_path / "formal.json"
    formal.write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "episodes_per_condition": 1_000,
                "multi_partner_episodes": 1_000,
                "formal_candidate": True,
                "acceptance_checks": {"strict": True},
                "artifact_hashes": {
                    "model": sha256(source.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _validated_formal_evaluation(formal, source)
    assert payload["formal_candidate"] is True

    payload["multi_partner_episodes"] = 999
    formal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Strict formal evaluation"):
        _validated_formal_evaluation(formal, source)


def test_release_packaging_rejects_a_failed_fresh_seed_smoke_gate() -> None:
    tutorial = {
        "steps": 120,
        "deliveries": 1,
        "charging_steps": 1,
        "shutdowns": 0,
    }
    passing = _release_acceptance_gates(
        tutorial,
        {"collision_rate": True, "avoidable_wait_rate": True},
    )
    failing = _release_acceptance_gates(
        tutorial,
        {"collision_rate": True, "avoidable_wait_rate": False},
    )

    assert all(passing.values())
    assert failing["supplemental_seed_ranges_pass"] is False
    assert not all(failing.values())
