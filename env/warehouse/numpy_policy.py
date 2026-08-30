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
from .decision_protocol import independent_agent_seed
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
    horizon: int
    move_battery_cost: float
    battery_safety_margin: float
    map_rows: int
    map_cols: int
    observation_dim: int
    action_names: tuple[str, ...]
    action_dim: int
    per_action_feature_dim: int
    teammate_charge_goal_index: int
    teammate_previous_action_start: int
    own_frames_since_charger_departure_index: int
    teammate_steps_since_charging_index: int
    teammate_legal_action_mask_start: int
    coordination_start: int
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
        """Evaluate one or more local observations in a single Actor pass."""

        local = np.asarray(observation, dtype=np.float64)
        if local.ndim < 1 or local.shape[-1] != self.metadata.observation_dim:
            raise ValueError(
                "Expected local observations ending in dimension "
                f"{self.metadata.observation_dim}, "
                f"received {local.shape}."
            )

        latent = np.tanh(self._linear(local, "intent_encoder.0"))
        latent = np.tanh(self._linear(latent, "intent_encoder.2"))
        mission_hidden = np.tanh(self._linear(latent, "mission_head.0"))
        mission_logits = self._linear(mission_hidden, "mission_head.2")
        intent = np.concatenate((latent, _softmax(mission_logits)), axis=-1)

        base = np.concatenate((local, intent), axis=-1)
        base = np.tanh(self._linear(base, "actor.0"))
        base = np.tanh(self._linear(base, "actor.2"))
        base = np.tanh(self._linear(base, "actor.4"))
        learned_logit_limit = 4.0
        raw_base_logits = self._linear(base, "actor.6")
        base_logits = learned_logit_limit * np.tanh(
            raw_base_logits / learned_logit_limit
        )

        action_dim = self.metadata.action_dim
        feature_dim = self.metadata.per_action_feature_dim
        own_start = self.metadata.own_action_features_start
        teammate_start = self.metadata.teammate_action_features_start
        collision_start = self.metadata.joint_collision_matrix_start
        leading = local.shape[:-1]
        own_features = local[
            ..., own_start : own_start + feature_dim * action_dim
        ].reshape(*leading, action_dim, feature_dim)
        teammate_features = local[
            ..., teammate_start : teammate_start + feature_dim * action_dim
        ].reshape(*leading, action_dim, feature_dim)
        collision_matrix = local[
            ..., collision_start : collision_start + action_dim**2
        ].reshape(*leading, action_dim, action_dim)
        action_identity = np.broadcast_to(
            np.eye(action_dim, dtype=np.float64),
            (*leading, action_dim, action_dim),
        )
        action_intent = np.broadcast_to(
            intent[..., None, :],
            (*leading, action_dim, intent.shape[-1]),
        )

        teammate_input = np.concatenate(
            (teammate_features, action_identity, action_intent), axis=-1
        )
        teammate_hidden = np.maximum(
            self._linear(teammate_input, "teammate_action_predictor.0"),
            0.0,
        )
        teammate_logits = self._linear(
            teammate_hidden, "teammate_action_predictor.2"
        ).reshape(*leading, action_dim)
        teammate_context = np.maximum(
            self._linear(local, "teammate_context_predictor.0"), 0.0
        )
        teammate_context = np.maximum(
            self._linear(teammate_context, "teammate_context_predictor.2"),
            0.0,
        )
        actor_teammate_logits = teammate_logits + self._linear(
            teammate_context,
            "teammate_context_predictor.4",
        )
        participant_context = np.maximum(
            self._linear(local, "participant_context_predictor.0"), 0.0
        )
        participant_context = np.maximum(
            self._linear(participant_context, "participant_context_predictor.2"),
            0.0,
        )
        participant_teammate_logits = self._linear(
            participant_context,
            "participant_context_predictor.4",
        )
        participant_flag = local[..., 23, None]
        teammate_logits = (
            actor_teammate_logits * (1.0 - participant_flag)
            + participant_teammate_logits * participant_flag
        )
        predicted_teammate = _softmax(teammate_logits)
        selected_collision = np.einsum(
            "...ij,...j->...i", collision_matrix, predicted_teammate
        )[..., None]
        legal = local[..., -action_dim:, None]

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
        raw_structured_logits = self._linear(
            structured_hidden, "action_scorer.4"
        ).reshape(*leading, action_dim)
        structured_logits = learned_logit_limit * np.tanh(
            raw_structured_logits / learned_logit_limit
        )
        risk_log_scale = self.weights["collision_risk_log_scale"]
        collision_scale = np.logaddexp(0.0, risk_log_scale)
        collision_penalty = collision_scale * selected_collision[..., 0]
        teammate_has_priority = local[..., self.metadata.coordination_start + 7]
        self_has_priority = local[..., self.metadata.coordination_start + 6]
        teammate_legal_actions = local[
            ...,
            self.metadata.teammate_legal_action_mask_start : (
                self.metadata.teammate_legal_action_mask_start + action_dim
            ),
        ]
        yielding_collision = np.max(
            collision_matrix * teammate_legal_actions[..., None, :],
            axis=-1,
        )
        non_wait_action = 1.0 - action_identity[..., -1]
        yield_scale = np.logaddexp(0.0, self.weights["yield_risk_log_scale"])
        yield_penalty = (
            yield_scale
            * teammate_has_priority[..., None]
            * yielding_collision
            * non_wait_action
        )
        occupied_scale = np.maximum(
            np.logaddexp(
                0.0, self.weights["occupied_cell_risk_log_scale"]
            ),
            50.0,
        )
        occupied_cell_penalty = occupied_scale * own_features[..., :, 3]
        self_has_charge_goal = local[..., 15]
        teammate_has_charge_goal = local[
            ...,
            self.metadata.teammate_charge_goal_index,
        ]
        teammate_at_charger = teammate_features[..., 0, 6]
        self_at_charger = own_features[..., 0, 6]
        public_dual_charger_priority = (
            self_has_charge_goal
            * teammate_has_charge_goal
            * (1.0 - self_at_charger)
            * (1.0 - teammate_at_charger)
        )
        teammate_previous_horizontal_move = np.clip(
            local[
                ...,
                self.metadata.teammate_previous_action_start
                + ACTIONS.index("LEFT"),
            ]
            + local[
                ...,
                self.metadata.teammate_previous_action_start
                + ACTIONS.index("RIGHT"),
            ],
            0.0,
            1.0,
        )
        teammate_previous_wait_for_clearance = local[
            ...,
            self.metadata.teammate_previous_action_start
            + ACTIONS.index("WAIT"),
        ]
        own_previous_horizontal_departure = (
            local[..., 24 + ACTIONS.index("LEFT")]
            + local[..., 24 + ACTIONS.index("RIGHT")]
        )
        self_adjacent_to_charger = np.max(
            own_features[..., :, 5], axis=-1
        ) * (1.0 - self_at_charger)
        teammate_current_charger_distance = (
            teammate_features[..., -1, 1]
            * float(self.metadata.map_rows * self.metadata.map_cols)
        )
        teammate_currently_adjacent_to_charger = (
            np.abs(teammate_current_charger_distance - 1.0) <= 1e-6
        ).astype(np.float32) * (1.0 - teammate_at_charger)
        participant_handoff_followthrough = (
            participant_flag[..., 0]
            * public_dual_charger_priority
            * self_has_priority
            * self_adjacent_to_charger
            * teammate_currently_adjacent_to_charger
            * teammate_previous_horizontal_move
        )
        current_charger_distance = (
            own_features[..., -1, 1]
            * float(self.metadata.map_rows * self.metadata.map_cols)
        )
        current_charger_slack = (
            local[..., 2]
            - current_charger_distance
            * float(self.metadata.move_battery_cost)
            / 100.0
        )
        public_critical_charger_priority = (
            self_has_charge_goal
            * (
                current_charger_slack
                <= float(self.metadata.battery_safety_margin)
                * float(self.metadata.move_battery_cost)
                / 100.0
                + 1e-8
            ).astype(np.float64)
        )
        charger_clearance_gain = np.clip(
            (
                own_features[..., :, 1]
                - own_features[..., -1:, 1]
            )
            * float(self.metadata.map_rows * self.metadata.map_cols),
            0.0,
            1.0,
        )
        charger_progress_gain = np.clip(
            (
                own_features[..., -1:, 1]
                - own_features[..., :, 1]
            )
            * float(self.metadata.map_rows * self.metadata.map_cols),
            0.0,
            1.0,
        )
        teammate_charger_progress_gain = np.clip(
            (
                teammate_features[..., -1:, 1]
                - teammate_features[..., :, 1]
            )
            * float(self.metadata.map_rows * self.metadata.map_cols),
            0.0,
            1.0,
        )
        teammate_charger_clearance_gain = np.clip(
            (
                teammate_features[..., :, 1]
                - teammate_features[..., -1:, 1]
            )
            * float(self.metadata.map_rows * self.metadata.map_cols),
            0.0,
            1.0,
        )
        teammate_battery = local[
            ...,
            self.metadata.teammate_charge_goal_index - 10,
        ]
        teammate_clearance_remaining_battery = (
            teammate_battery[..., None]
            - non_wait_action
            * float(self.metadata.move_battery_cost)
            / 100.0
        )
        teammate_clearance_required_battery = (
            teammate_features[..., :, 1]
            * float(self.metadata.map_rows * self.metadata.map_cols)
            + float(self.metadata.battery_safety_margin)
        ) * float(self.metadata.move_battery_cost) / 100.0
        teammate_energy_safe_charger_clearance = (
            teammate_clearance_remaining_battery + 1e-8
            >= teammate_clearance_required_battery
        ).astype(np.float64)
        teammate_safe_charger_clearance_exists = np.max(
            teammate_has_charge_goal[..., None]
            * teammate_charger_clearance_gain
            * teammate_energy_safe_charger_clearance
            * (1.0 - collision_matrix[..., -1, :])
            * teammate_legal_actions,
            axis=-1,
        )
        public_critical_charger_advance = (
            public_critical_charger_priority
            * self_has_priority
            * (1.0 - teammate_safe_charger_clearance_exists)
        )
        teammate_progress_worst_collision = np.max(
            collision_matrix * legal[..., 0, None],
            axis=-2,
        )
        teammate_robust_charger_progress_exists = np.max(
            teammate_charger_progress_gain
            * (1.0 - teammate_progress_worst_collision)
            * teammate_legal_actions,
            axis=-1,
        )
        dual_charger_clearance_required = (
            1.0 - teammate_robust_charger_progress_exists
        )
        dual_charger_clearance_for_actor = (
            1.0
            - participant_flag[..., 0]
            + participant_flag[..., 0] * dual_charger_clearance_required
            * teammate_previous_wait_for_clearance
            * (1.0 - own_previous_horizontal_departure)
        )
        public_priority_action_commitment = np.clip(
            public_dual_charger_priority[..., None]
            * (
                self_has_priority[..., None]
                * charger_progress_gain
                * (
                    1.0
                    - participant_flag
                    * public_dual_charger_priority[..., None]
                    * yielding_collision
                    * (1.0 - participant_handoff_followthrough)[..., None]
                )
                + teammate_has_priority[..., None]
                * dual_charger_clearance_for_actor[..., None]
                * charger_clearance_gain
            )
            + public_critical_charger_advance[..., None]
            * (1.0 - teammate_at_charger)[..., None]
            * charger_progress_gain,
            0.0,
            1.0,
        )
        teammate_goal_arrival_actions = (
            (teammate_features[..., :, 2] > 0.0).astype(np.float64)
            * (teammate_features[..., :, 0] <= 1e-8).astype(np.float64)
            * teammate_legal_actions
        )
        participant_goal_arrival_collision = np.max(
            collision_matrix
            * teammate_goal_arrival_actions[..., None, :],
            axis=-1,
        )
        public_priority_action_commitment = (
            public_priority_action_commitment
            * (
                1.0
                - participant_flag
                * participant_goal_arrival_collision
                * (1.0 - participant_handoff_followthrough)[..., None]
            )
        )
        public_participant_action_commitment = (
            participant_flag * public_priority_action_commitment
        )
        public_participant_progress_bonus = (
            occupied_scale
            * public_participant_action_commitment
            * (1.0 - own_features[..., :, 3])
        )
        collision_penalty = collision_penalty * (
            1.0 - public_participant_action_commitment
        )
        yield_penalty = yield_penalty * (
            1.0 - public_participant_action_commitment
        )
        priority_progress_blocked_by_teammate = np.max(
            charger_progress_gain * own_features[..., :, 3],
            axis=-1,
        )
        blocked_priority_departure_penalty = (
            occupied_scale
            * participant_flag
            * public_dual_charger_priority[..., None]
            * self_has_priority[..., None]
            * priority_progress_blocked_by_teammate[..., None]
            * non_wait_action
        )
        dual_charger_yield_wait_penalty = (
            occupied_scale
            * public_dual_charger_priority[..., None]
            * teammate_has_priority[..., None]
            * (1.0 - dual_charger_clearance_required)[..., None]
            * non_wait_action
        )
        participant_dual_unrobust_progress_penalty = (
            occupied_scale
            * participant_flag
            * public_dual_charger_priority[..., None]
            * self_has_priority[..., None]
            * (1.0 - public_critical_charger_advance)[..., None]
            * (1.0 - participant_handoff_followthrough)[..., None]
            * charger_progress_gain
            * yielding_collision
            * non_wait_action
        )
        participant_dual_charger_clearance_bonus = (
            np.logaddexp(
                0.0,
                self.weights["participant_standoff_progress_log_scale"],
            )
            * participant_flag
            * self_has_charge_goal[..., None]
            * teammate_has_charge_goal[..., None]
            * teammate_has_priority[..., None]
            * (1.0 - self_at_charger)[..., None]
            * (1.0 - teammate_at_charger)[..., None]
            * dual_charger_clearance_required[..., None]
            * teammate_previous_wait_for_clearance[..., None]
            * (1.0 - own_previous_horizontal_departure)[..., None]
            * charger_clearance_gain
            * (1.0 - selected_collision[..., 0])
            * non_wait_action
        )
        public_ai_ai_parallel_charger_progress_bonus = (
            occupied_scale
            * (1.0 - participant_flag)
            * public_dual_charger_priority[..., None]
            * teammate_has_priority[..., None]
            * teammate_robust_charger_progress_exists[..., None]
            * charger_progress_gain
            * (1.0 - yielding_collision)
            * non_wait_action
        )
        participant_collision_scale = np.logaddexp(
            0.0,
            self.weights["participant_collision_risk_log_scale"],
        )
        participant_worst_collision_penalty = (
            participant_collision_scale
            * participant_flag
            * (1.0 - public_priority_action_commitment)
            * yielding_collision
            * non_wait_action
        )
        participant_expected_collision_penalty = (
            np.logaddexp(
                0.0,
                self.weights["participant_expected_collision_risk_log_scale"],
            )
            * participant_flag
            * selected_collision[..., 0]
            * non_wait_action
        )
        participant_expected_collision_penalty = (
            participant_expected_collision_penalty
            * (1.0 - public_participant_action_commitment)
        )
        energy_exhaustion_scale = np.logaddexp(
            0.0,
            self.weights["energy_exhaustion_risk_log_scale"],
        )
        energy_exhaustion_penalty = (
            energy_exhaustion_scale
            * (
                local[..., 2]
                <= float(self.metadata.move_battery_cost) / 100.0 + 1e-8
            )[..., None]
            * non_wait_action
        )
        own_departure_age = local[
            ...,
            self.metadata.own_frames_since_charger_departure_index,
        ]
        teammate_charge_age = local[
            ...,
            self.metadata.teammate_steps_since_charging_index,
        ]
        teammate_adjacent_to_charger = np.max(
            teammate_features[..., :, 5],
            axis=-1,
        ) * (1.0 - teammate_at_charger)
        adjacent_to_charger = self_adjacent_to_charger
        recent_departure = (
            own_departure_age <= 6.0 / float(self.metadata.horizon)
        ).astype(np.float32)
        coordinated_return = (
            (teammate_charge_age > 0.0)
            & (teammate_charge_age < own_departure_age)
        ).astype(np.float32)
        recent_unproductive_return = (
            recent_departure
            * (1.0 - coordinated_return)
            * (1.0 - public_critical_charger_priority)
        )
        handoff_reentry = (
            recent_departure
            * self_has_charge_goal
            * teammate_has_charge_goal
            * teammate_has_priority
            * (1.0 - teammate_at_charger)
        )
        recent_unproductive_return = np.clip(
            recent_unproductive_return + handoff_reentry,
            0.0,
            1.0,
        )
        entry_followthrough_allowed = 1.0 - recent_departure * (
            1.0 - coordinated_return
        )
        queue_scale = np.logaddexp(
            0.0, self.weights["charger_queue_wait_log_scale"]
        )
        charger_queue_penalty = (
            queue_scale
            * self_has_charge_goal[..., None]
            * teammate_at_charger[..., None]
            * adjacent_to_charger[..., None]
            * teammate_has_priority[..., None]
            * non_wait_action
        )
        charger_entry_delay_penalty = (
            queue_scale
            * self_has_charge_goal[..., None]
            * (1.0 - teammate_at_charger)[..., None]
            * adjacent_to_charger[..., None]
            * self_has_priority[..., None]
            * entry_followthrough_allowed[..., None]
            * (1.0 - own_features[..., :, 5])
        )
        charger_reentry_cycle_penalty = (
            np.maximum(queue_scale, 200.0)
            * recent_unproductive_return[..., None]
            * (1.0 - self_at_charger)[..., None]
            * own_features[..., :, 5]
        )
        occupant_scale = np.maximum(
            np.logaddexp(
                0.0,
                self.weights["charger_occupant_wait_log_scale"],
            ),
            50.0,
        )
        charger_occupant_penalty = (
            occupant_scale
            * self_at_charger[..., None]
            * (
                self_has_charge_goal[..., None]
                * (
                    1.0
                    - teammate_has_priority[..., None]
                    * teammate_adjacent_to_charger[..., None]
                    * teammate_has_charge_goal[..., None]
                )
                * non_wait_action
                + teammate_has_priority[..., None]
                * teammate_adjacent_to_charger[..., None]
                * teammate_has_charge_goal[..., None]
                * action_identity[..., -1]
            )
        )
        self_has_work_goal = np.clip(
            local[..., 13] + local[..., 14],
            0.0,
            1.0,
        )
        self_has_route_goal = np.clip(
            self_has_work_goal + self_has_charge_goal,
            0.0,
            1.0,
        )
        robust_progress_exit_exists = np.max(
            (1.0 - yielding_collision)
            * np.maximum(own_features[..., :, 2], 0.0)
            * non_wait_action,
            axis=-1,
        )
        completed_charge_wait_penalty = (
            occupant_scale
            * self_at_charger[..., None]
            * (1.0 - self_has_charge_goal)[..., None]
            * self_has_work_goal[..., None]
            * robust_progress_exit_exists[..., None]
            * action_identity[..., -1]
        )
        progress_scale = np.logaddexp(
            0.0, self.weights["priority_progress_log_scale"]
        )
        priority_progress_bonus = (
            progress_scale
            * self_has_priority[..., None]
            * np.maximum(own_features[..., :, 2], 0.0)
        )
        detour_scale = np.logaddexp(
            0.0, self.weights["delivery_detour_log_scale"]
        )
        detour_scale = np.maximum(detour_scale, 50.0)
        self_has_delivery_goal = local[..., 14]
        robust_wait_safe = 1.0 - np.max(collision_matrix[..., -1, :], axis=-1)
        robust_action_safe = 1.0 - np.max(collision_matrix, axis=-1)
        teammate_progress_actions = (
            (teammate_features[..., :, 2] > 0.0).astype(np.float64)
            * teammate_legal_actions
        )
        ai_ai_progress_wait_safe = 1.0 - np.max(
            collision_matrix[..., -1, :] * teammate_progress_actions,
            axis=-1,
        )
        public_yield_wait_available = (
            (1.0 - participant_flag[..., 0])
            * teammate_has_priority
            * ai_ai_progress_wait_safe
        )
        robust_nonregression_exit_exists = np.max(
            robust_action_safe
            * (own_features[..., :, 2] >= 0.0).astype(np.float64),
            axis=-1,
        )
        robust_nonregression_exit_exists = np.maximum(
            robust_nonregression_exit_exists,
            public_yield_wait_available,
        )
        mission_detour_penalty = (
            detour_scale
            * self_has_route_goal[..., None]
            * robust_nonregression_exit_exists[..., None]
            * np.maximum(-own_features[..., :, 2], 0.0)
        )
        robust_progress_scale = np.logaddexp(
            0.0, self.weights["robust_progress_log_scale"]
        )
        robust_progress_bonus = (
            robust_progress_scale
            * robust_action_safe
            * np.maximum(own_features[..., :, 2], 0.0)
        )
        participant_robust_progress_bonus = (
            np.logaddexp(
                0.0,
                self.weights["participant_robust_progress_log_scale"],
            )
            * participant_flag
            * robust_action_safe
            * np.maximum(own_features[..., :, 2], 0.0)
        )
        teammate_previous_wait = local[
            ...,
            self.metadata.teammate_previous_action_start + action_dim - 1,
        ]
        teammate_fields_start = (
            self.metadata.teammate_previous_action_start - 5
        )
        relative_row = local[..., teammate_fields_start] * max(
            1,
            self.metadata.map_rows - 1,
        )
        relative_col = local[..., teammate_fields_start + 1] * max(
            1,
            self.metadata.map_cols - 1,
        )
        row_deltas = np.asarray((-1.0, 1.0, 0.0, 0.0, 0.0))
        col_deltas = np.asarray((0.0, 0.0, -1.0, 1.0, 0.0))
        current_manhattan = np.abs(relative_row) + np.abs(relative_col)
        next_manhattan = np.abs(
            relative_row[..., None] - row_deltas
        ) + np.abs(relative_col[..., None] - col_deltas)
        separation_gain = np.clip(
            next_manhattan - current_manhattan[..., None],
            0.0,
            1.0,
        )
        horizontal_clearance_action = (
            action_identity[..., ACTIONS.index("LEFT")]
            + action_identity[..., ACTIONS.index("RIGHT")]
        )
        participant_station_handoff_direction_bonus = (
            100.0
            * participant_flag
            * self_at_charger[..., None]
            * self_has_charge_goal[..., None]
            * teammate_has_charge_goal[..., None]
            * teammate_has_priority[..., None]
            * teammate_adjacent_to_charger[..., None]
            * horizontal_clearance_action
            * separation_gain
            * robust_action_safe
            * non_wait_action
        )
        participant_approach_clearance_direction_bonus = (
            100.0
            * participant_flag
            * (1.0 - self_at_charger)[..., None]
            * (1.0 - teammate_at_charger)[..., None]
            * self_has_charge_goal[..., None]
            * teammate_has_charge_goal[..., None]
            * teammate_has_priority[..., None]
            * dual_charger_clearance_required[..., None]
            * teammate_previous_wait[..., None]
            * (1.0 - own_previous_horizontal_departure)[..., None]
            * charger_clearance_gain
            * separation_gain
            * robust_action_safe
            * non_wait_action
        )
        own_goal_blocked_by_teammate = np.max(
            np.maximum(own_features[..., :, 2], 0.0)
            * own_features[..., :, 3],
            axis=-1,
        )
        observed_goal_block_escape_bonus = (
            50.0
            * local[..., 11, None]
            * own_goal_blocked_by_teammate[..., None]
            * ((1.0 - local[..., 22]) * (1.0 - local[..., 23]))[..., None]
            * robust_action_safe
            * separation_gain
            * non_wait_action
        )
        move_energy_viable = (
            local[..., 2]
            > float(self.metadata.move_battery_cost) / 100.0 + 1e-8
        )
        remaining_battery = own_features[..., :, 8]
        charger_route_energy = (
            own_features[..., :, 1]
            * float(self.metadata.map_rows * self.metadata.map_cols)
            * float(self.metadata.move_battery_cost)
            / 100.0
            + float(self.metadata.move_battery_cost) / 100.0
        )
        charger_route_viable = (
            remaining_battery + 1e-8 >= charger_route_energy
        )
        energy_route_deficit_penalty = (
            np.logaddexp(
                0.0,
                self.weights["energy_route_deficit_log_scale"],
            )
            * self_has_charge_goal[..., None]
            * (1.0 - charger_route_viable)
            * non_wait_action
        )
        participant_standoff_progress_bonus = (
            np.logaddexp(
                0.0,
                self.weights["participant_standoff_progress_log_scale"],
            )
            * participant_flag
            * local[..., 11, None]
            * teammate_previous_wait[..., None]
            * move_energy_viable[..., None]
            * charger_route_viable
            * robust_action_safe
            * separation_gain
            * non_wait_action
        )
        participant_delivery_detour_penalty = (
            np.logaddexp(
                0.0,
                self.weights["participant_delivery_detour_log_scale"],
            )
            * participant_flag
            * self_has_delivery_goal[..., None]
            * robust_wait_safe[..., None]
            * np.maximum(-own_features[..., :, 2], 0.0)
        )
        participant_partner = np.maximum(
            self._linear(local, "participant_partner_action_head.0"), 0.0
        )
        participant_partner = np.maximum(
            self._linear(
                participant_partner,
                "participant_partner_action_head.2",
            ),
            0.0,
        )
        raw_participant_partner_residual = self._linear(
            participant_partner,
            "participant_partner_action_head.4",
        )
        participant_partner_logit_limit = 12.0
        participant_partner_residual = (
            participant_partner_logit_limit
            * np.tanh(
                raw_participant_partner_residual
                / participant_partner_logit_limit
            )
            * participant_flag
        )
        teammate_carrying_index = (
            self.metadata.teammate_previous_action_start - 1
        )
        teammate_fields_start = (
            self.metadata.teammate_previous_action_start - 5
        )
        exactly_one_carrying = np.abs(
            local[..., 4] - local[..., teammate_carrying_index]
        )
        own_row = local[..., 0] * float(max(1, self.metadata.map_rows - 1))
        same_horizontal_corridor = (
            (np.abs(local[..., teammate_fields_start]) <= 1e-6)
            & (own_row >= 1.0 - 1e-6)
            & (own_row <= 5.0 + 1e-6)
        ).astype(np.float64)
        ai_ai_mode = (1.0 - local[..., 22]) * (1.0 - local[..., 23])
        deadlock_escape_gate = (
            local[..., 12]
            * exactly_one_carrying
            * same_horizontal_corridor
            * ai_ai_mode
        )[..., None]
        deadlock_escape = np.maximum(
            self._linear(local, "deadlock_escape_action_head.0"), 0.0
        )
        raw_deadlock_escape_residual = self._linear(
            deadlock_escape,
            "deadlock_escape_action_head.2",
        )
        deadlock_escape_logit_limit = 12.0
        deadlock_escape_residual = (
            deadlock_escape_logit_limit
            * np.tanh(
                raw_deadlock_escape_residual
                / deadlock_escape_logit_limit
            )
            * deadlock_escape_gate
        )
        return np.asarray(
            base_logits
            + structured_logits
            - collision_penalty
            - yield_penalty
            - occupied_cell_penalty
            - participant_worst_collision_penalty
            - participant_expected_collision_penalty
            - blocked_priority_departure_penalty
            - dual_charger_yield_wait_penalty
            - participant_dual_unrobust_progress_penalty
            - energy_exhaustion_penalty
            - energy_route_deficit_penalty
            - charger_queue_penalty
            - charger_entry_delay_penalty
            - charger_reentry_cycle_penalty
            - charger_occupant_penalty
            - completed_charge_wait_penalty
            + priority_progress_bonus
            - mission_detour_penalty
            - participant_delivery_detour_penalty
            + robust_progress_bonus
            + participant_robust_progress_bonus
            + participant_standoff_progress_bonus
            + participant_dual_charger_clearance_bonus
            + participant_station_handoff_direction_bonus
            + participant_approach_clearance_direction_bonus
            + observed_goal_block_escape_bonus
            + public_ai_ai_parallel_charger_progress_bonus
            + public_participant_progress_bonus
            + participant_partner_residual
            + deadlock_escape_residual,
            dtype=np.float32,
        )

    def act(
        self,
        observations: Mapping[str, Any],
        *,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
        base_seed: int | None = None,
        decision_key: tuple[int, int] | None = None,
    ) -> tuple[dict[str, str], dict[str, ActionDistribution]]:
        """Sample each Actor independently from the frozen pre-move state."""

        if sorted(observations) != ["robot_1", "robot_2"]:
            raise ValueError("The deployed Actor requires robot_1 and robot_2.")
        agent_ids = ("robot_1", "robot_2")
        local = np.stack(
            [np.asarray(observations[agent_id], dtype=np.float32) for agent_id in agent_ids]
        )
        raw_logits = self.logits(local)
        masks = local[..., -len(ACTIONS) :] > 0.5
        if not bool(masks.any(axis=-1).all()):
            raise ValueError("Every deployed Actor must have a legal action.")
        masked_logits = np.where(masks, raw_logits, np.float32(-1.0e9))
        probabilities = _softmax(masked_logits)
        actions: dict[str, str] = {}
        distributions: dict[str, ActionDistribution] = {}
        for row, agent_id in enumerate(agent_ids):
            mask = masks[row]
            if not bool(mask.any()):
                raise ValueError(f"{agent_id} has no legal action.")
            if deterministic:
                index = int(np.argmax(probabilities[row]))
            elif decision_key is not None:
                if base_seed is None:
                    raise ValueError("base_seed is required with decision_key.")
                episode_id, frame = decision_key
                agent_rng = np.random.default_rng(
                    independent_agent_seed(
                        base_seed=base_seed,
                        episode_id=episode_id,
                        frame=frame,
                        agent_id=agent_id,
                    )
                )
                index = int(
                    agent_rng.choice(len(ACTIONS), p=probabilities[row])
                )
            else:
                if rng is None:
                    raise ValueError("rng is required for unkeyed stochastic inference.")
                index = int(rng.choice(len(ACTIONS), p=probabilities[row]))
            action = ACTIONS[index]
            actions[agent_id] = action
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id,
                actions=ACTIONS,
                probabilities=tuple(float(item) for item in probabilities[row]),
                logits=tuple(float(item) for item in masked_logits[row]),
                action_mask=tuple(float(item) for item in mask),
                proposed_action=action,
            )
        return actions, distributions
