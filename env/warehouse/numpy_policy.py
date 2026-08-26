"""Dependency-light execution of a trained warehouse Actor.

Render's free service does not need the PyTorch training runtime.  The export
contains the exact Actor tensors from a validated MAPPO checkpoint, and this
module evaluates the same network with NumPy.  It does not add a rule layer or
post-process actions: both robots receive independent observations from the
same pre-move state and are sampled independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.policy_contracts import ActionDistribution

from .contracts import (
    ACTION_EXECUTION_VERSION,
    ENVIRONMENT_VERSION,
    MODEL_VERSION,
    RUNTIME_CONTROLLER,
)
from .navigation import ACTIONS
from .rewards import REWARD_VERSION


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


@dataclass(frozen=True)
class NumpyPolicyMetadata:
    model_version: str
    environment_version: str
    reward_version: str
    action_execution_version: str
    runtime_controller: str
    map_layout_id: str
    observation_dim: int
    action_names: tuple[str, ...]
    action_dim: int
    per_action_feature_dim: int
    own_action_features_start: int
    teammate_action_features_start: int
    joint_collision_matrix_start: int
    checkpoint_sha256: str


class NumpyWarehousePolicy:
    """Immutable Actor weights with caller-owned random-number generation."""

    def __init__(
        self,
        *,
        metadata: NumpyPolicyMetadata,
        weights: Mapping[str, np.ndarray],
        artifact_sha256: str,
    ) -> None:
        self.metadata = metadata
        self.weights = {
            # Float64 accumulation avoids platform-specific Accelerate/BLAS
            # overflow warnings seen for otherwise finite float32 matrices.
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in weights.items()
        }
        self.artifact_sha256 = str(artifact_sha256)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        expected = {
            "model_version": MODEL_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "reward_version": REWARD_VERSION,
            "action_execution_version": ACTION_EXECUTION_VERSION,
            "runtime_controller": RUNTIME_CONTROLLER,
            "action_names": tuple(ACTIONS),
            "action_dim": len(ACTIONS),
        }
        for field, value in expected.items():
            actual = getattr(self.metadata, field)
            if actual != value:
                raise ValueError(
                    f"Incompatible deployed Actor {field}: {actual!r}; "
                    f"expected {value!r}."
                )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyWarehousePolicy":
        source = Path(path)
        payload = source.read_bytes()
        with np.load(source, allow_pickle=False) as archive:
            metadata_payload = json.loads(str(archive["metadata_json"].item()))
            metadata_payload["action_names"] = tuple(
                metadata_payload["action_names"]
            )
            metadata = NumpyPolicyMetadata(**metadata_payload)
            weights = {
                name: archive[name].copy()
                for name in archive.files
                if name != "metadata_json"
            }
        return cls(
            metadata=metadata,
            weights=weights,
            artifact_sha256=sha256(payload).hexdigest(),
        )

    def _linear(self, values: np.ndarray, prefix: str) -> np.ndarray:
        # ``einsum`` is used instead of ``@`` because some macOS Accelerate
        # builds emit spurious floating-point warnings for finite float32/64
        # matrix-vector products. Render and local inference remain identical.
        return np.einsum(
            "...j,ij->...i",
            values,
            self.weights[f"{prefix}.weight"],
        ) + self.weights[f"{prefix}.bias"]

    def logits(self, observation: Any) -> np.ndarray:
        """Evaluate the exact exported Actor for one local observation."""

        local = np.asarray(observation, dtype=np.float64)
        expected = (self.metadata.observation_dim,)
        if local.shape != expected:
            raise ValueError(
                f"Expected one local observation with shape {expected}, "
                f"received {local.shape}."
            )

        latent = np.tanh(self._linear(local, "intent_encoder.0"))
        latent = np.tanh(self._linear(latent, "intent_encoder.2"))
        mission_hidden = np.tanh(self._linear(latent, "mission_head.0"))
        mission_logits = self._linear(mission_hidden, "mission_head.2")
        intent = np.concatenate((latent, _softmax(mission_logits)))

        base = np.concatenate((local, intent))
        base = np.tanh(self._linear(base, "actor.0"))
        base = np.tanh(self._linear(base, "actor.2"))
        base = np.tanh(self._linear(base, "actor.4"))
        base_logits = self._linear(base, "actor.6")

        action_dim = self.metadata.action_dim
        feature_dim = self.metadata.per_action_feature_dim
        own_start = self.metadata.own_action_features_start
        teammate_start = self.metadata.teammate_action_features_start
        collision_start = self.metadata.joint_collision_matrix_start
        own_features = local[
            own_start : own_start + feature_dim * action_dim
        ].reshape(action_dim, feature_dim)
        teammate_features = local[
            teammate_start : teammate_start + feature_dim * action_dim
        ].reshape(action_dim, feature_dim)
        collision_matrix = local[
            collision_start : collision_start + action_dim**2
        ].reshape(action_dim, action_dim)
        action_identity = np.eye(action_dim, dtype=np.float64)
        action_intent = np.repeat(intent[None, :], action_dim, axis=0)

        teammate_input = np.concatenate(
            (teammate_features, action_identity, action_intent), axis=-1
        )
        teammate_hidden = np.maximum(
            self._linear(teammate_input, "teammate_action_predictor.0"),
            0.0,
        )
        teammate_logits = self._linear(
            teammate_hidden, "teammate_action_predictor.2"
        ).reshape(action_dim)
        predicted_teammate = _softmax(teammate_logits)
        selected_collision = np.einsum(
            "ij,j->i", collision_matrix, predicted_teammate
        )[:, None]
        legal = local[-action_dim:, None]

        structured = np.concatenate(
            (
                own_features,
                selected_collision,
                legal,
                action_identity,
                action_intent,
            ),
            axis=-1,
        )
        structured_hidden = np.maximum(
            self._linear(structured, "action_scorer.0"), 0.0
        )
        structured_hidden = np.maximum(
            self._linear(structured_hidden, "action_scorer.2"), 0.0
        )
        structured_logits = self._linear(
            structured_hidden, "action_scorer.4"
        ).reshape(action_dim)
        return np.asarray(base_logits + structured_logits, dtype=np.float32)

    def act(
        self,
        observations: Mapping[str, Any],
        *,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> tuple[dict[str, str], dict[str, ActionDistribution]]:
        """Sample each Actor independently from the frozen pre-move state."""

        if sorted(observations) != ["robot_1", "robot_2"]:
            raise ValueError("The deployed Actor requires robot_1 and robot_2.")
        actions: dict[str, str] = {}
        distributions: dict[str, ActionDistribution] = {}
        for agent_id in ("robot_1", "robot_2"):
            local = np.asarray(observations[agent_id], dtype=np.float32)
            raw_logits = self.logits(local)
            mask = local[-len(ACTIONS) :] > 0.5
            if not bool(mask.any()):
                raise ValueError(f"{agent_id} has no legal action.")
            masked_logits = np.where(mask, raw_logits, np.float32(-1.0e9))
            probabilities = _softmax(masked_logits)
            index = (
                int(np.argmax(probabilities))
                if deterministic
                else int(rng.choice(len(ACTIONS), p=probabilities))
            )
            action = ACTIONS[index]
            actions[agent_id] = action
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id,
                actions=ACTIONS,
                probabilities=tuple(float(item) for item in probabilities),
                logits=tuple(float(item) for item in masked_logits),
                action_mask=tuple(float(item) for item in mask),
                proposed_action=action,
            )
        return actions, distributions
