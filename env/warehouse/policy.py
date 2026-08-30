"""Inference-only shared MAPPO policy for the warehouse domain.

Keeping checkpoint validation and action selection here lets Web, tutorial and
evaluation code load a policy without importing optimizers or rollout
collection.  Training-specific state lives in :mod:`env.warehouse.mappo` for
backwards compatibility while callers migrate to the narrower modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from itertools import chain

import numpy as np
import torch
from torch import nn

from core.policy_contracts import ActionDistribution

from .contracts import (
    ACTION_EXECUTION_VERSION,
    ENVIRONMENT_VERSION,
    MODEL_VERSION,
    RUNTIME_CONTROLLER,
)
from .decision_protocol import independent_agent_seed
from .domain import WarehouseConfig
from .navigation import ACTIONS
from .observations import (
    NAVIGATION_GOAL_KINDS,
    global_state_dim,
    observation_dim,
    own_frames_since_charger_departure_index,
    teammate_goal_kind_start,
    teammate_legal_action_mask_start,
    teammate_steps_since_charging_index,
)
from .rewards import REWARD_VERSION, RewardConfig


MISSION_INTENT_NAMES = (
    "task_slot_1",
    "task_slot_2",
    "delivery",
    "charge",
    "wait",
)


def independent_actor_input(observation: Any) -> np.ndarray:
    """Return one actor input containing pre-move observable state only."""

    local = np.asarray(observation, dtype=np.float32)
    if local.ndim != 1:
        raise ValueError("An Actor input must be one flat local observation.")
    return local


@dataclass(frozen=True)
class MAPPOConfig:
    hidden_dim: int = 256
    intent_dim: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coef: float = 0.02
    entropy_coef_final: float = 0.001
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    minibatch_size: int = 256
    seed: int = 2026


class SharedActorCentralCritic(nn.Module):
    """One shared actor and one critic conditioned on the acting agent ID."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        max_agents: int,
        action_dim: int,
        hidden_dim: int,
        intent_dim: int,
        local_patch_size: int,
        active_task_count: int,
        horizon: int,
        move_battery_cost: float,
        mission_reserve_steps: float,
        map_rows: int,
        map_cols: int,
    ) -> None:
        super().__init__()
        # Keep unconstrained learned residuals below the observable safety
        # terms.  Earlier Actor fits produced logits above 30, silently
        # overwhelming an otherwise-correct lower-priority yield penalty and
        # causing deterministic collision loops.  This smooth neural bound is
        # part of the Actor forward pass, not a mask or post-action shield.
        self.learned_logit_limit = 4.0
        # Participant-mode calibration sometimes has to override a generic
        # mission preference that is individually efficient but unsafe for a
        # different partner distribution.  This isolated residual needs
        # enough range to defeat two saturated generic heads; the observable
        # episode-provenance bit gates it to zero throughout AI-AI execution.
        self.participant_partner_logit_limit = 12.0
        # A separately trained residual can break an observed AI-AI shelf-arm
        # stand-off without changing ordinary navigation.  Its exact S_t gate
        # is evaluated inside the Actor forward pass and the final layer is
        # zero-initialised, so legacy checkpoints retain bit-identical logits.
        self.deadlock_escape_logit_limit = 12.0
        # A robot whose public goal is still charging must not stochastically
        # leave an uncontested station. Two bounded learned heads can differ
        # by as much as 32 logits, so retain a larger monotone floor for this
        # observable energy-safety term. The handoff branch below disables
        # the move penalty only when S_t proves a lower-energy peer owns the
        # station.
        self.minimum_charger_occupant_logit_scale = 50.0
        # Re-entry competes with as much as 100 logits of public-priority and
        # participant-robust progress evidence. A floor of 50 could therefore
        # still make a three-step unproductive return deterministic. Keep the
        # cycle veto above every bounded positive Actor component; productive
        # mission/coordination returns are excluded by the state predicate.
        self.minimum_charger_reentry_logit_scale = 200.0
        self.minimum_occupied_cell_logit_scale = 50.0
        # If a collision-robust non-regressing alternative exists, pickup,
        # delivery, and charge regressions are not useful exploration. Keep
        # their probability below the combined bounded learned heads. The
        # gate stays zero when every safe action must temporarily detour, so
        # public clearance manoeuvres remain available.
        self.minimum_mission_detour_logit_scale = 250.0
        # Before an empty robot commits to a shared task, positive pickup
        # progress is admissible only for a task whose complete A->B->charger
        # route remains energy-safe.  This prevents the first half of the
        # observed UP->DOWN oscillation: the unsafe UP action is rejected in
        # S_t instead of being regretted only after its battery cost is paid.
        self.minimum_unsafe_task_progress_logit_scale = 250.0
        self.active_task_count = int(active_task_count)
        self.action_dim = int(action_dim)
        self.base_local_dim = int(local_dim)
        self.horizon = int(horizon)
        self.move_battery_cost_fraction = float(move_battery_cost) / 100.0
        self.mission_reserve_steps = float(mission_reserve_steps)
        self.map_rows = int(map_rows)
        self.map_cols = int(map_cols)
        coordination_dim = (
            8
            + 2 * self.action_dim
            + 2
            * (9 + 6 * int(active_task_count))
            * self.action_dim
            + self.action_dim**2
        )
        self.per_action_feature_dim = 9 + 6 * int(active_task_count)
        self.teammate_charge_goal_index = (
            teammate_goal_kind_start(
                max_agents=max_agents,
                active_task_count=active_task_count,
                action_dim=self.action_dim,
            )
            + NAVIGATION_GOAL_KINDS.index("charge")
        )
        self.teammate_previous_action_start = (
            teammate_goal_kind_start(
                max_agents=max_agents,
                active_task_count=active_task_count,
                action_dim=self.action_dim,
            )
            - self.action_dim
        )
        self.own_frames_since_charger_departure_index = (
            own_frames_since_charger_departure_index(action_dim=self.action_dim)
        )
        self.teammate_steps_since_charging_index = (
            teammate_steps_since_charging_index(
                max_agents=max_agents,
                active_task_count=active_task_count,
                action_dim=self.action_dim,
            )
        )
        self.teammate_legal_action_mask_start = (
            teammate_legal_action_mask_start(
                max_agents=max_agents,
                active_task_count=active_task_count,
                action_dim=self.action_dim,
            )
        )
        self.coordination_start = (
            self.base_local_dim
            - self.action_dim
            - int(local_patch_size)
            - coordination_dim
        )
        if self.coordination_start < 0:
            raise ValueError("Actor dimensions do not contain coordination features.")
        self.own_action_features_start = (
            self.coordination_start + 8 + 2 * self.action_dim
        )
        self.teammate_action_features_start = (
            self.own_action_features_start
            + self.per_action_feature_dim * self.action_dim
        )
        self.joint_collision_matrix_start = (
            self.coordination_start
            + 8
            + 2 * self.action_dim
            + 2 * self.per_action_feature_dim * self.action_dim
        )
        self.intent_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, intent_dim),
            nn.Tanh(),
        )
        # The mission head is a neural latent decision, not an environment
        # assignment.  Its probabilities condition the final action logits
        # and are trained jointly with the Actor.  No mission prediction is
        # used to mask, replace, or post-process an executed action.
        self.mission_head = nn.Sequential(
            nn.Linear(intent_dim, intent_dim),
            nn.Tanh(),
            nn.Linear(intent_dim, len(MISSION_INTENT_NAMES)),
        )
        action_intent_dim = int(intent_dim) + len(MISSION_INTENT_NAMES)
        self.actor = nn.Sequential(
            nn.Linear(local_dim + action_intent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # A shared per-action neural scorer reasons about collision risk under
        # a learned teammate-action forecast.  Both robots make that forecast
        # from the same pre-move state; neither receives the teammate's action
        # for the current frame.
        structured_feature_dim = (
            self.per_action_feature_dim
            + 1
            + 1
            + self.action_dim
            + action_intent_dim
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(structured_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.teammate_action_predictor = nn.Sequential(
            nn.Linear(
                self.per_action_feature_dim + self.action_dim + action_intent_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # A full-observation residual forecast supplies the task, energy,
        # history, and geometry context that per-action route features alone
        # cannot represent.  It still observes only the frozen local S_t and
        # never receives the teammate's sampled action for the current frame.
        self.teammate_context_predictor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        # Participant-controlled Robot 1 follows a different distribution
        # from the shared Actor.  A separate S_t-only predictor prevents the
        # AI teammate forecast from being reused for humans while preserving
        # the same causal information boundary.
        self.participant_context_predictor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        # Robot 2 needs a policy response that is robust to a participant,
        # whose action distribution is intentionally different from the
        # shared Actor.  Keep that response in an isolated residual head so
        # partner calibration cannot damage ordinary AI-AI collaboration.
        # The gate is episode provenance already present in frozen S_t; this
        # head never receives the participant's sampled current-frame action.
        self.participant_partner_action_head = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.deadlock_escape_action_head = nn.Sequential(
            nn.Linear(local_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        nn.init.zeros_(self.deadlock_escape_action_head[2].weight)
        nn.init.zeros_(self.deadlock_escape_action_head[2].bias)
        # A learned monotone coefficient makes the explicit joint-collision
        # feature impossible for the generic scorer to silently ignore.  It
        # remains part of the neural logits (and is exported as a weight), not
        # an action mask, shield, or post-policy rewrite.
        self.collision_risk_log_scale = nn.Parameter(
            torch.tensor(15.0, dtype=torch.float32)
        )
        # Only the observable lower-priority robot receives this conservative
        # penalty.  The higher-priority peer can keep moving, preventing the
        # symmetric "both wait" failure of a global worst-case penalty.
        self.yield_risk_log_scale = nn.Parameter(
            torch.tensor(30.0, dtype=torch.float32)
        )
        self.charger_queue_wait_log_scale = nn.Parameter(
            torch.tensor(20.0, dtype=torch.float32)
        )
        self.charger_occupant_wait_log_scale = nn.Parameter(
            torch.tensor(30.0, dtype=torch.float32)
        )
        self.priority_progress_log_scale = nn.Parameter(
            torch.tensor(15.0, dtype=torch.float32)
        )
        self.delivery_detour_log_scale = nn.Parameter(
            torch.tensor(20.0, dtype=torch.float32)
        )
        self.occupied_cell_risk_log_scale = nn.Parameter(
            torch.tensor(30.0, dtype=torch.float32)
        )
        # A participant is not sampled from the shared Actor distribution.
        # Penalize actions that can collide with any legal participant move,
        # using only the frozen collision matrix and episode provenance.  It
        # remains a differentiable Actor-logit term—not a mask or action
        # replacement—and is identically zero in ordinary AI-AI episodes.
        self.participant_collision_risk_log_scale = nn.Parameter(
            torch.tensor(80.0, dtype=torch.float32)
        )
        # A move at or below its energy cost deterministically shuts the robot
        # down.  Keep that observable terminal risk inside the Actor logits so
        # a low-energy policy cannot overwhelm it with a mission preference.
        # This is not an action mask or a post-policy rewrite.
        self.energy_exhaustion_risk_log_scale = nn.Parameter(
            torch.tensor(200.0, dtype=torch.float32)
        )
        self.participant_standoff_progress_log_scale = nn.Parameter(
            torch.tensor(100.0, dtype=torch.float32)
        )
        self.participant_delivery_detour_log_scale = nn.Parameter(
            torch.tensor(40.0, dtype=torch.float32)
        )
        self.participant_expected_collision_risk_log_scale = nn.Parameter(
            torch.tensor(80.0, dtype=torch.float32)
        )
        self.participant_robust_progress_log_scale = nn.Parameter(
            torch.tensor(80.0, dtype=torch.float32)
        )
        self.energy_route_deficit_log_scale = nn.Parameter(
            torch.tensor(80.0, dtype=torch.float32)
        )
        self.robust_progress_log_scale = nn.Parameter(
            torch.tensor(2.0, dtype=torch.float32)
        )
        self.critic = nn.Sequential(
            nn.Linear(global_dim + max_agents, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_outputs(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final action logits and the Actor's neural mission logits."""

        latent_intent = self.intent_encoder(observations)
        mission_logits = self.mission_head(latent_intent)
        mission_probabilities = torch.softmax(mission_logits, dim=-1)
        intent = torch.cat((latent_intent, mission_probabilities), dim=-1)
        raw_base_logits = self.actor(
            torch.cat((observations, intent), dim=-1)
        )
        base_logits = self.learned_logit_limit * torch.tanh(
            raw_base_logits / self.learned_logit_limit
        )
        local = observations
        own_action_features = local[
            ...,
            self.own_action_features_start : (
                self.own_action_features_start
                + self.per_action_feature_dim * self.action_dim
            ),
        ].reshape(
            *local.shape[:-1],
            self.action_dim,
            self.per_action_feature_dim,
        )
        teammate_action_features = local[
            ...,
            self.teammate_action_features_start : (
                self.teammate_action_features_start
                + self.per_action_feature_dim * self.action_dim
            ),
        ].reshape(
            *local.shape[:-1],
            self.action_dim,
            self.per_action_feature_dim,
        )
        collision_matrix = local[
            ...,
            self.joint_collision_matrix_start : (
                self.joint_collision_matrix_start + self.action_dim**2
            ),
        ].reshape(*local.shape[:-1], self.action_dim, self.action_dim)
        action_identity = torch.eye(
            self.action_dim,
            dtype=observations.dtype,
            device=observations.device,
        ).expand(*local.shape[:-1], self.action_dim, self.action_dim)
        action_intent = intent.unsqueeze(-2).expand(
            *local.shape[:-1],
            self.action_dim,
            intent.shape[-1],
        )
        actor_teammate_logits = self.teammate_action_predictor(
            torch.cat(
                (teammate_action_features, action_identity, action_intent),
                dim=-1,
            )
        ).squeeze(-1) + self.teammate_context_predictor(observations)
        participant_teammate = local[..., 23].unsqueeze(-1)
        participant_teammate_logits = self.participant_context_predictor(
            observations
        )
        predicted_teammate_logits = (
            actor_teammate_logits * (1.0 - participant_teammate)
            + participant_teammate_logits * participant_teammate
        )
        predicted_teammate = torch.softmax(
            predicted_teammate_logits,
            dim=-1,
        )
        selected_collision = torch.einsum(
            "...ij,...j->...i",
            collision_matrix,
            predicted_teammate,
        ).unsqueeze(-1)
        legal = local[..., -self.action_dim :].unsqueeze(-1)
        structured = torch.cat(
            (
                own_action_features,
                selected_collision,
                legal,
                action_identity,
                action_intent,
            ),
            dim=-1,
        )
        raw_structured_logits = self.action_scorer(structured).squeeze(-1)
        structured_logits = self.learned_logit_limit * torch.tanh(
            raw_structured_logits / self.learned_logit_limit
        )
        collision_penalty = torch.nn.functional.softplus(
            self.collision_risk_log_scale
        ) * selected_collision.squeeze(-1)
        teammate_has_priority = local[..., self.coordination_start + 7]
        self_has_priority = local[..., self.coordination_start + 6]
        teammate_legal_actions = local[
            ...,
            self.teammate_legal_action_mask_start : (
                self.teammate_legal_action_mask_start + self.action_dim
            ),
        ]
        yielding_collision = (
            collision_matrix * teammate_legal_actions.unsqueeze(-2)
        ).amax(dim=-1)
        non_wait_action = 1.0 - action_identity[..., -1]
        yield_penalty = (
            torch.nn.functional.softplus(self.yield_risk_log_scale)
            * teammate_has_priority.unsqueeze(-1)
            * yielding_collision
            * non_wait_action
        )
        # Entering the teammate's currently occupied cell is unsafe under a
        # simultaneous protocol because the teammate may independently wait.
        # Both priority roles apply this same Actor-logit penalty; paired with
        # an unpenalized yielding WAIT, it creates a common observable rule
        # without reading the teammate's sampled current-frame action.
        occupied_cell_scale = torch.clamp(
            torch.nn.functional.softplus(self.occupied_cell_risk_log_scale),
            min=self.minimum_occupied_cell_logit_scale,
        )
        occupied_cell_penalty = (
            occupied_cell_scale * own_action_features[..., :, 3]
        )
        # Local observation indices 13:17 are pickup/delivery/charge/wait.
        self_has_charge_goal = local[..., 15]
        teammate_has_charge_goal = local[..., self.teammate_charge_goal_index]
        # Per-action feature 6 is the acting robot's current charger occupancy;
        # in the teammate block it therefore means the peer occupies it.
        teammate_at_charger = teammate_action_features[..., 0, 6]
        self_at_charger = own_action_features[..., 0, 6]
        public_dual_charger_priority = (
            self_has_charge_goal
            * teammate_has_charge_goal
            * (1.0 - self_at_charger)
            * (1.0 - teammate_at_charger)
        )
        teammate_previous_horizontal_move = torch.clamp(
            local[..., self.teammate_previous_action_start + ACTIONS.index("LEFT")]
            + local[..., self.teammate_previous_action_start + ACTIONS.index("RIGHT")],
            min=0.0,
            max=1.0,
        )
        teammate_previous_wait_for_clearance = local[
            ..., self.teammate_previous_action_start + ACTIONS.index("WAIT")
        ]
        own_previous_horizontal_departure = (
            local[..., 24 + ACTIONS.index("LEFT")]
            + local[..., 24 + ACTIONS.index("RIGHT")]
        )
        self_adjacent_to_charger = (
            own_action_features[..., :, 5].amax(dim=-1)
            * (1.0 - self_at_charger)
        )
        teammate_current_charger_distance = (
            teammate_action_features[..., -1, 1]
            * float(self.map_rows * self.map_cols)
        )
        teammate_currently_adjacent_to_charger = (
            (
                torch.abs(teammate_current_charger_distance - 1.0)
                <= 1e-6
            ).to(local.dtype)
            * (1.0 - teammate_at_charger)
        )
        participant_handoff_followthrough = (
            participant_teammate.squeeze(-1)
            * public_dual_charger_priority
            * self_has_priority
            * self_adjacent_to_charger
            * teammate_currently_adjacent_to_charger
            * teammate_previous_horizontal_move
        )
        current_charger_distance = (
            own_action_features[..., -1, 1]
            * float(self.map_rows * self.map_cols)
        )
        current_charger_slack = (
            local[..., 2]
            - current_charger_distance * self.move_battery_cost_fraction
        )
        public_critical_charger_priority = (
            self_has_charge_goal
            * (
                current_charger_slack
                <= self.mission_reserve_steps
                * self.move_battery_cost_fraction
                + 1e-8
            ).to(local.dtype)
        )
        charger_clearance_gain = torch.clamp(
            (
                own_action_features[..., :, 1]
                - own_action_features[..., -1:, 1]
            )
            * float(self.map_rows * self.map_cols),
            min=0.0,
            max=1.0,
        )
        charger_progress_gain = torch.clamp(
            (
                own_action_features[..., -1:, 1]
                - own_action_features[..., :, 1]
            )
            * float(self.map_rows * self.map_cols),
            min=0.0,
            max=1.0,
        )
        teammate_charger_progress_gain = torch.clamp(
            (
                teammate_action_features[..., -1:, 1]
                - teammate_action_features[..., :, 1]
            )
            * float(self.map_rows * self.map_cols),
            min=0.0,
            max=1.0,
        )
        teammate_charger_clearance_gain = torch.clamp(
            (
                teammate_action_features[..., :, 1]
                - teammate_action_features[..., -1:, 1]
            )
            * float(self.map_rows * self.map_cols),
            min=0.0,
            max=1.0,
        )
        teammate_battery = local[
            ...,
            self.teammate_charge_goal_index - 10,
        ]
        teammate_clearance_remaining_battery = (
            teammate_battery.unsqueeze(-1)
            - non_wait_action * self.move_battery_cost_fraction
        )
        teammate_clearance_required_battery = (
            teammate_action_features[..., :, 1]
            * float(self.map_rows * self.map_cols)
            + self.mission_reserve_steps
        ) * self.move_battery_cost_fraction
        teammate_energy_safe_charger_clearance = (
            teammate_clearance_remaining_battery + 1e-8
            >= teammate_clearance_required_battery
        ).to(local.dtype)
        # A public emergency priority does not imply that the priority robot
        # may advance in this frame.  If the participant has a legal charger-
        # clearance move that is safe while this Actor waits, the causal
        # protocol is two-phase: participant clears at S_t, then the priority
        # Actor advances from S_{t+1}.  This distinction prevents the priority
        # bit from authorising a same-target collision at an aisle approach.
        teammate_safe_charger_clearance_exists = (
            teammate_has_charge_goal.unsqueeze(-1)
            * teammate_charger_clearance_gain
            * teammate_energy_safe_charger_clearance
            * (1.0 - collision_matrix[..., -1, :])
            * teammate_legal_actions
        ).amax(dim=-1)
        public_critical_charger_advance = (
            public_critical_charger_priority
            * self_has_priority
            * (1.0 - teammate_safe_charger_clearance_exists)
        )
        teammate_progress_worst_collision = (
            collision_matrix * legal.squeeze(-1).unsqueeze(-1)
        ).amax(dim=-2)
        teammate_robust_charger_progress_exists = (
            teammate_charger_progress_gain
            * (1.0 - teammate_progress_worst_collision)
            * teammate_legal_actions
        ).amax(dim=-1)
        dual_charger_clearance_required = (
            1.0 - teammate_robust_charger_progress_exists
        )
        dual_charger_clearance_for_actor = (
            1.0
            - participant_teammate.squeeze(-1)
            + participant_teammate.squeeze(-1)
            * dual_charger_clearance_required
            * teammate_previous_wait_for_clearance
            * (1.0 - own_previous_horizontal_departure)
        )
        # A critical charger route is a public reservation derived from S_t.
        # Every supported participant profile observes the same bit and
        # yields to it. Once the participant has vacated the target cell,
        # retaining the generic worst-legal-action veto makes both robots
        # WAIT forever even though neither reads the other's current action.
        # Occupied-cell safety below still prevents entry before clearance.
        public_priority_action_commitment = torch.clamp(
            public_dual_charger_priority.unsqueeze(-1)
            * (
                self_has_priority.unsqueeze(-1)
                * charger_progress_gain
                * (
                    1.0
                    - participant_teammate
                    * public_dual_charger_priority.unsqueeze(-1)
                    * yielding_collision
                    * (1.0 - participant_handoff_followthrough).unsqueeze(-1)
                )
                + teammate_has_priority.unsqueeze(-1)
                * dual_charger_clearance_for_actor.unsqueeze(-1)
                * charger_clearance_gain
            )
            + public_critical_charger_advance.unsqueeze(-1)
            * (1.0 - teammate_at_charger).unsqueeze(-1)
            * charger_progress_gain,
            min=0.0,
            max=1.0,
        )
        teammate_goal_arrival_actions = (
            (teammate_action_features[..., :, 2] > 0.0).to(local.dtype)
            * (teammate_action_features[..., :, 0] <= 1e-8).to(local.dtype)
            * teammate_legal_actions
        )
        participant_goal_arrival_collision = (
            collision_matrix
            * teammate_goal_arrival_actions.unsqueeze(-2)
        ).amax(dim=-1)
        # Critical battery priority is not permission to enter a cell that a
        # participant can reach as its visible goal in the same
        # joint step.  Keep the collision term active for that exact action;
        # after the participant has passed, the same frozen-state test clears
        # and the charger advance becomes available.
        public_priority_action_commitment = (
            public_priority_action_commitment
            * (
                1.0
                - participant_teammate
                * participant_goal_arrival_collision
                * (1.0 - participant_handoff_followthrough).unsqueeze(-1)
            )
        )
        public_participant_action_commitment = (
            participant_teammate * public_priority_action_commitment
        )
        public_participant_progress_bonus = (
            occupied_cell_scale
            * public_participant_action_commitment
            * (1.0 - own_action_features[..., :, 3])
        )
        collision_penalty = collision_penalty * (
            1.0 - public_participant_action_commitment
        )
        yield_penalty = yield_penalty * (
            1.0 - public_participant_action_commitment
        )
        priority_progress_blocked_by_teammate = (
            charger_progress_gain
            * own_action_features[..., :, 3]
        ).amax(dim=-1)
        blocked_priority_departure_penalty = (
            occupied_cell_scale
            * participant_teammate
            * public_dual_charger_priority.unsqueeze(-1)
            * self_has_priority.unsqueeze(-1)
            * priority_progress_blocked_by_teammate.unsqueeze(-1)
            * non_wait_action
        )
        dual_charger_yield_wait_penalty = (
            occupied_cell_scale
            * public_dual_charger_priority.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * (1.0 - dual_charger_clearance_required).unsqueeze(-1)
            * non_wait_action
        )
        participant_dual_unrobust_progress_penalty = (
            occupied_cell_scale
            * participant_teammate
            * public_dual_charger_priority.unsqueeze(-1)
            * self_has_priority.unsqueeze(-1)
            * (1.0 - public_critical_charger_advance).unsqueeze(-1)
            * (1.0 - participant_handoff_followthrough).unsqueeze(-1)
            * charger_progress_gain
            * yielding_collision
            * non_wait_action
        )
        participant_dual_charger_clearance_bonus = (
            torch.nn.functional.softplus(
                self.participant_standoff_progress_log_scale
            )
            * participant_teammate
            * self_has_charge_goal.unsqueeze(-1)
            * teammate_has_charge_goal.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * (1.0 - self_at_charger).unsqueeze(-1)
            * (1.0 - teammate_at_charger).unsqueeze(-1)
            * dual_charger_clearance_required.unsqueeze(-1)
            * teammate_previous_wait_for_clearance.unsqueeze(-1)
            * (1.0 - own_previous_horizontal_departure).unsqueeze(-1)
            * charger_clearance_gain
            * (1.0 - selected_collision.squeeze(-1))
            * non_wait_action
        )
        public_ai_ai_parallel_charger_progress_bonus = (
            occupied_cell_scale
            * (1.0 - participant_teammate)
            * public_dual_charger_priority.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * teammate_robust_charger_progress_exists.unsqueeze(-1)
            * charger_progress_gain
            * (1.0 - yielding_collision)
            * non_wait_action
        )
        participant_worst_collision_penalty = (
            torch.nn.functional.softplus(
                self.participant_collision_risk_log_scale
            )
            * participant_teammate
            * (1.0 - public_priority_action_commitment)
            * yielding_collision
            * non_wait_action
        )
        participant_expected_collision_penalty = (
            torch.nn.functional.softplus(
                self.participant_expected_collision_risk_log_scale
            )
            * participant_teammate
            * selected_collision.squeeze(-1)
            * non_wait_action
        )
        participant_expected_collision_penalty = (
            participant_expected_collision_penalty
            * (1.0 - public_participant_action_commitment)
        )
        energy_exhaustion_penalty = (
            torch.nn.functional.softplus(
                self.energy_exhaustion_risk_log_scale
            )
            * (
                local[..., 2] <= self.move_battery_cost_fraction + 1e-8
            ).to(local.dtype).unsqueeze(-1)
            * non_wait_action
        )
        teammate_adjacent_to_charger = (
            teammate_action_features[..., :, 5].amax(dim=-1)
            * (1.0 - teammate_at_charger)
        )
        # Per-action feature 5 identifies a target at the charger.  If any
        # movement target has that value while the robot is not already on the
        # charger, the robot is exactly at an entrance cell.  Restricting the
        # queue hold to that apron prevents a distant low-energy robot from
        # waiting in an aisle merely because its teammate is charging.
        adjacent_to_charger = self_adjacent_to_charger
        own_departure_age = local[
            ..., self.own_frames_since_charger_departure_index
        ]
        teammate_charge_age = local[
            ..., self.teammate_steps_since_charging_index
        ]
        recent_departure = (
            own_departure_age <= 6.0 / float(self.horizon)
        ).to(local.dtype)
        coordinated_return = (
            (teammate_charge_age > 0.0)
            & (teammate_charge_age < own_departure_age)
        ).to(local.dtype)
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
        recent_unproductive_return = torch.clamp(
            recent_unproductive_return + handoff_reentry,
            min=0.0,
            max=1.0,
        )
        entry_followthrough_allowed = 1.0 - recent_departure * (
            1.0 - coordinated_return
        )
        charger_queue_penalty = (
            torch.nn.functional.softplus(self.charger_queue_wait_log_scale)
            * self_has_charge_goal.unsqueeze(-1)
            * teammate_at_charger.unsqueeze(-1)
            * adjacent_to_charger.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * non_wait_action
        )
        # Once the station is empty, the observable priority robot must
        # complete the second half of a causal handoff.  Without this
        # symmetric follow-through term both independently acting robots can
        # WAIT forever: the winner delays entry while the loser correctly
        # yields.  This remains a logit feature of S_t, not an action rewrite.
        charger_entry_delay_penalty = (
            torch.nn.functional.softplus(self.charger_queue_wait_log_scale)
            * self_has_charge_goal.unsqueeze(-1)
            * (1.0 - teammate_at_charger).unsqueeze(-1)
            * adjacent_to_charger.unsqueeze(-1)
            * self_has_priority.unsqueeze(-1)
            * entry_followthrough_allowed.unsqueeze(-1)
            * (1.0 - own_action_features[..., :, 5])
        )
        # The converse is an observed charger loop: an empty robot that left
        # within six frames, made no teammate handoff progress, and turns
        # straight back.  Penalize that re-entry distributionally so the
        # Actor continues into the aisle. A genuinely critical return remains
        # admissible, as does a state-proved productive coordination return.
        # No sampled action is inspected or changed.
        charger_reentry_scale = torch.clamp(
            torch.nn.functional.softplus(
                self.charger_queue_wait_log_scale
            ),
            min=self.minimum_charger_reentry_logit_scale,
        )
        charger_reentry_cycle_penalty = (
            charger_reentry_scale
            * recent_unproductive_return.unsqueeze(-1)
            * (1.0 - self_at_charger).unsqueeze(-1)
            * own_action_features[..., :, 5]
        )
        charger_occupant_scale = torch.clamp(
            torch.nn.functional.softplus(
                self.charger_occupant_wait_log_scale
            ),
            min=self.minimum_charger_occupant_logit_scale,
        )
        charger_occupant_penalty = (
            charger_occupant_scale
            * self_at_charger.unsqueeze(-1)
            * (
                self_has_charge_goal.unsqueeze(-1)
                * (
                    1.0
                    - teammate_has_priority.unsqueeze(-1)
                    * teammate_adjacent_to_charger.unsqueeze(-1)
                    * teammate_has_charge_goal.unsqueeze(-1)
                )
                * non_wait_action
                + teammate_has_priority.unsqueeze(-1)
                * teammate_adjacent_to_charger.unsqueeze(-1)
                * teammate_has_charge_goal.unsqueeze(-1)
                * action_identity[..., -1]
            )
        )
        # Charging WAIT is productive only while the frozen public goal is
        # still charge.  Once hysteresis releases that mode, an Actor with an
        # active pickup/delivery mission must take a robustly safe progress
        # exit instead of stochastically filling the battery to 100%.  Requiring
        # safety against every legal peer action keeps this causal: it depends
        # only on S_t and never assumes which action the peer just sampled.
        self_has_work_goal = torch.clamp(
            local[..., 13] + local[..., 14],
            min=0.0,
            max=1.0,
        )
        self_has_route_goal = torch.clamp(
            self_has_work_goal + self_has_charge_goal,
            min=0.0,
            max=1.0,
        )
        robust_progress_exit_exists = (
            (1.0 - yielding_collision)
            * torch.clamp(own_action_features[..., :, 2], min=0.0)
            * non_wait_action
        ).amax(dim=-1)
        completed_charge_wait_penalty = (
            charger_occupant_scale
            * self_at_charger.unsqueeze(-1)
            * (1.0 - self_has_charge_goal).unsqueeze(-1)
            * self_has_work_goal.unsqueeze(-1)
            * robust_progress_exit_exists.unsqueeze(-1)
            * action_identity[..., -1]
        )
        priority_progress_bonus = (
            torch.nn.functional.softplus(self.priority_progress_log_scale)
            * self_has_priority.unsqueeze(-1)
            * torch.clamp(own_action_features[..., :, 2], min=0.0)
        )
        self_has_delivery_goal = local[..., 14]
        robust_wait_safe = 1.0 - collision_matrix[..., -1, :].amax(dim=-1)
        robust_action_safe = 1.0 - collision_matrix.amax(dim=-1)
        teammate_progress_actions = (
            (teammate_action_features[..., :, 2] > 0.0).to(local.dtype)
            * teammate_legal_actions
        )
        # Under the shared frozen-state priority contract the priority Actor
        # must choose one collision-free progress route when one exists.  The
        # yielding robot therefore need not retreat merely because a
        # different, avoidable peer action could enter its cell.  Requiring
        # WAIT to survive *all* legal peer actions created a LEFT->RIGHT
        # cul-de-sac cycle even though the priority robot had a safe DOWN
        # route.  This remains independent of the peer's sampled action.
        ai_ai_progress_wait_safe = (
            (1.0 - collision_matrix[..., -1, :])
            * teammate_progress_actions
        ).amax(dim=-1)
        public_yield_wait_available = (
            (1.0 - participant_teammate.squeeze(-1))
            * teammate_has_priority
            * ai_ai_progress_wait_safe
        )
        robust_nonregression_exit_exists = (
            robust_action_safe
            * (own_action_features[..., :, 2] >= 0.0).to(local.dtype)
        ).amax(dim=-1)
        robust_nonregression_exit_exists = torch.maximum(
            robust_nonregression_exit_exists,
            public_yield_wait_available,
        )
        mission_detour_penalty = (
            torch.clamp(
                torch.nn.functional.softplus(self.delivery_detour_log_scale),
                min=self.minimum_mission_detour_logit_scale,
            )
            * self_has_route_goal.unsqueeze(-1)
            * robust_nonregression_exit_exists.unsqueeze(-1)
            * torch.clamp(-own_action_features[..., :, 2], min=0.0)
        )
        task_action_features = own_action_features[..., 9:].reshape(
            *local.shape[:-1],
            self.action_dim,
            self.active_task_count,
            6,
        )
        task_progress = torch.clamp(
            task_action_features[..., 1],
            min=0.0,
        )
        task_energy_safe = (
            task_action_features[..., 4] >= -1e-8
        ).to(local.dtype)
        safe_task_progress = task_progress * task_energy_safe
        unsafe_task_progress = task_progress * (1.0 - task_energy_safe)
        safe_task_progress_exists = (
            safe_task_progress.amax(dim=(-2, -1)) > 0.0
        ).to(local.dtype)
        action_has_safe_task_progress = (
            safe_task_progress.amax(dim=-1) > 0.0
        ).to(local.dtype)
        action_has_unsafe_task_progress = (
            unsafe_task_progress.amax(dim=-1) > 0.0
        ).to(local.dtype)
        uncommitted_task_selection = (
            (1.0 - self_has_route_goal) * (1.0 - local[..., 4])
        )
        unsafe_task_progress_penalty = (
            torch.clamp(
                torch.nn.functional.softplus(
                    self.energy_route_deficit_log_scale
                ),
                min=self.minimum_unsafe_task_progress_logit_scale,
            )
            * uncommitted_task_selection.unsqueeze(-1)
            * safe_task_progress_exists.unsqueeze(-1)
            * action_has_unsafe_task_progress
            * (1.0 - action_has_safe_task_progress)
        )
        robust_progress_bonus = (
            torch.nn.functional.softplus(self.robust_progress_log_scale)
            * robust_action_safe
            * torch.clamp(own_action_features[..., :, 2], min=0.0)
        )
        participant_robust_progress_bonus = (
            torch.nn.functional.softplus(
                self.participant_robust_progress_log_scale
            )
            * participant_teammate
            * robust_action_safe
            * torch.clamp(own_action_features[..., :, 2], min=0.0)
        )
        teammate_previous_wait = local[
            ...,
            self.teammate_previous_action_start + self.action_dim - 1,
        ]
        teammate_fields_start = self.teammate_previous_action_start - 5
        relative_row = (
            local[..., teammate_fields_start] * max(1, self.map_rows - 1)
        )
        relative_col = (
            local[..., teammate_fields_start + 1] * max(1, self.map_cols - 1)
        )
        row_deltas = torch.tensor(
            (-1.0, 1.0, 0.0, 0.0, 0.0),
            dtype=local.dtype,
            device=local.device,
        )
        col_deltas = torch.tensor(
            (0.0, 0.0, -1.0, 1.0, 0.0),
            dtype=local.dtype,
            device=local.device,
        )
        current_manhattan = relative_row.abs() + relative_col.abs()
        next_manhattan = (
            relative_row.unsqueeze(-1) - row_deltas
        ).abs() + (
            relative_col.unsqueeze(-1) - col_deltas
        ).abs()
        separation_gain = torch.clamp(
            next_manhattan - current_manhattan.unsqueeze(-1),
            min=0.0,
            max=1.0,
        )
        horizontal_clearance_action = (
            action_identity[..., ACTIONS.index("LEFT")]
            + action_identity[..., ACTIONS.index("RIGHT")]
        )
        # A participant waiting beside the single charger is an observed
        # two-phase handoff request.  From the same S_t, choose the side exit
        # away from that participant when leaving the occupied station, then
        # continue away from both the station and participant on the apron.
        # This removes the UP->RIGHT->LEFT loop that used to arise because all
        # charger-clearance moves received the same structural score.
        participant_station_handoff_direction_bonus = (
            self.minimum_occupied_cell_logit_scale
            * 2.0
            * participant_teammate
            * self_at_charger.unsqueeze(-1)
            * self_has_charge_goal.unsqueeze(-1)
            * teammate_has_charge_goal.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * teammate_adjacent_to_charger.unsqueeze(-1)
            * horizontal_clearance_action
            * separation_gain
            * robust_action_safe
            * non_wait_action
        )
        participant_approach_clearance_direction_bonus = (
            self.minimum_occupied_cell_logit_scale
            * 2.0
            * participant_teammate
            * (1.0 - self_at_charger).unsqueeze(-1)
            * (1.0 - teammate_at_charger).unsqueeze(-1)
            * self_has_charge_goal.unsqueeze(-1)
            * teammate_has_charge_goal.unsqueeze(-1)
            * teammate_has_priority.unsqueeze(-1)
            * dual_charger_clearance_required.unsqueeze(-1)
            * teammate_previous_wait.unsqueeze(-1)
            * (1.0 - own_previous_horizontal_departure).unsqueeze(-1)
            * charger_clearance_gain
            * separation_gain
            * robust_action_safe
            * non_wait_action
        )
        own_goal_blocked_by_teammate = (
            torch.clamp(own_action_features[..., :, 2], min=0.0)
            * own_action_features[..., :, 3]
        ).amax(dim=-1)
        unoccupied_robust_goal_progress = (
            torch.clamp(own_action_features[..., :, 2], min=0.0)
            * (1.0 - own_action_features[..., :, 3])
            * robust_action_safe
        )
        unoccupied_goal_progress_exists = (
            unoccupied_robust_goal_progress.amax(dim=-1) > 0.0
        ).to(local.dtype)
        blocked_priority_action_penalty = (
            self.minimum_occupied_cell_logit_scale
            * ((1.0 - local[..., 22]) * (1.0 - local[..., 23])).unsqueeze(-1)
            * self_has_priority.unsqueeze(-1)
            * own_goal_blocked_by_teammate.unsqueeze(-1)
            * (
                unoccupied_goal_progress_exists.unsqueeze(-1)
                * (
                    unoccupied_robust_goal_progress <= 0.0
                ).to(local.dtype)
                + (1.0 - unoccupied_goal_progress_exists).unsqueeze(-1)
                * non_wait_action
            )
        )
        # If both AIs have already produced an ineffective joint wait while
        # one robot physically occupies the other's next goal-progress cell,
        # the blocked robot must take a collision-robust separating step.
        # The gate uses only the recorded S_t wait streak and geometry; no
        # current-frame peer action is observed.
        observed_goal_block_escape_bonus = (
            self.minimum_occupied_cell_logit_scale
            * local[..., 11].unsqueeze(-1)
            * own_goal_blocked_by_teammate.unsqueeze(-1)
            * ((1.0 - local[..., 22]) * (1.0 - local[..., 23])).unsqueeze(-1)
            * robust_action_safe
            * separation_gain
            * non_wait_action
        )
        move_energy_viable = (
            local[..., 2] > self.move_battery_cost_fraction + 1e-8
        ).to(local.dtype)
        remaining_battery = own_action_features[..., :, 8]
        charger_route_energy = (
            own_action_features[..., :, 1]
            * float(self.map_rows * self.map_cols)
            * self.move_battery_cost_fraction
            + self.move_battery_cost_fraction
        )
        charger_route_viable = (
            remaining_battery + 1e-8 >= charger_route_energy
        ).to(local.dtype)
        energy_route_deficit_penalty = (
            torch.nn.functional.softplus(
                self.energy_route_deficit_log_scale
            )
            * self_has_charge_goal.unsqueeze(-1)
            * (1.0 - charger_route_viable)
            * non_wait_action
        )
        participant_standoff_progress_bonus = (
            torch.nn.functional.softplus(
                self.participant_standoff_progress_log_scale
            )
            * participant_teammate
            * local[..., 11].unsqueeze(-1)
            * teammate_previous_wait.unsqueeze(-1)
            * move_energy_viable.unsqueeze(-1)
            * charger_route_viable
            * robust_action_safe
            * separation_gain
            * non_wait_action
        )
        participant_delivery_detour_penalty = (
            torch.nn.functional.softplus(
                self.participant_delivery_detour_log_scale
            )
            * participant_teammate
            * self_has_delivery_goal.unsqueeze(-1)
            * robust_wait_safe.unsqueeze(-1)
            * torch.clamp(-own_action_features[..., :, 2], min=0.0)
        )
        raw_participant_partner_residual = (
            self.participant_partner_action_head(observations)
        )
        participant_partner_residual = (
            self.participant_partner_logit_limit
            * torch.tanh(
                raw_participant_partner_residual
                / self.participant_partner_logit_limit
            )
            * participant_teammate
        )
        teammate_carrying_index = self.teammate_previous_action_start - 1
        teammate_fields_start = self.teammate_previous_action_start - 5
        own_carrying = local[..., 4]
        teammate_carrying = local[..., teammate_carrying_index]
        exactly_one_carrying = torch.abs(own_carrying - teammate_carrying)
        own_row = local[..., 0] * float(max(1, self.map_rows - 1))
        same_horizontal_corridor = (
            (torch.abs(local[..., teammate_fields_start]) <= 1e-6)
            & (own_row >= 1.0 - 1e-6)
            & (own_row <= 5.0 + 1e-6)
        ).to(local.dtype)
        ai_ai_mode = (1.0 - local[..., 22]) * (1.0 - local[..., 23])
        deadlock_escape_gate = (
            local[..., 12]
            * exactly_one_carrying
            * same_horizontal_corridor
            * ai_ai_mode
        ).unsqueeze(-1)
        raw_deadlock_escape_residual = self.deadlock_escape_action_head(
            observations
        )
        deadlock_escape_residual = (
            self.deadlock_escape_logit_limit
            * torch.tanh(
                raw_deadlock_escape_residual
                / self.deadlock_escape_logit_limit
            )
            * deadlock_escape_gate
        )
        return (
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
            - blocked_priority_action_penalty
            + priority_progress_bonus
            - mission_detour_penalty
            - unsafe_task_progress_penalty
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
            mission_logits,
        )

    def teammate_action_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict the teammate action using pre-move observation only."""

        latent_intent = self.intent_encoder(observations)
        mission_probabilities = torch.softmax(
            self.mission_head(latent_intent), dim=-1
        )
        intent = torch.cat((latent_intent, mission_probabilities), dim=-1)
        local = observations
        teammate_action_features = local[
            ...,
            self.teammate_action_features_start : (
                self.teammate_action_features_start
                + self.per_action_feature_dim * self.action_dim
            ),
        ].reshape(
            *local.shape[:-1],
            self.action_dim,
            self.per_action_feature_dim,
        )
        action_identity = torch.eye(
            self.action_dim,
            dtype=observations.dtype,
            device=observations.device,
        ).expand(*local.shape[:-1], self.action_dim, self.action_dim)
        action_intent = intent.unsqueeze(-2).expand(
            *local.shape[:-1], self.action_dim, intent.shape[-1]
        )
        actor_logits = self.teammate_action_predictor(
            torch.cat(
                (teammate_action_features, action_identity, action_intent),
                dim=-1,
            )
        ).squeeze(-1) + self.teammate_context_predictor(observations)
        participant_flag = local[..., 23].unsqueeze(-1)
        participant_logits = self.participant_context_predictor(observations)
        return (
            actor_logits * (1.0 - participant_flag)
            + participant_logits * participant_flag
        )

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_outputs(observations)[0]

    def mission_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_outputs(observations)[1]

    def actor_parameters(self):
        """All trainable decentralized-Actor parameters, including intent."""

        return chain(
            (self.collision_risk_log_scale,),
            (self.yield_risk_log_scale,),
            (self.charger_queue_wait_log_scale,),
            (self.charger_occupant_wait_log_scale,),
            (self.priority_progress_log_scale,),
            (self.delivery_detour_log_scale,),
            (self.occupied_cell_risk_log_scale,),
            (self.participant_collision_risk_log_scale,),
            (self.energy_exhaustion_risk_log_scale,),
            (self.participant_standoff_progress_log_scale,),
            (self.participant_delivery_detour_log_scale,),
            (self.participant_expected_collision_risk_log_scale,),
            (self.participant_robust_progress_log_scale,),
            (self.energy_route_deficit_log_scale,),
            (self.robust_progress_log_scale,),
            self.intent_encoder.parameters(),
            self.mission_head.parameters(),
            self.actor.parameters(),
            self.action_scorer.parameters(),
            self.teammate_action_predictor.parameters(),
            self.teammate_context_predictor.parameters(),
            self.participant_context_predictor.parameters(),
            self.participant_partner_action_head.parameters(),
            self.deadlock_escape_action_head.parameters(),
        )

    def ppo_actor_parameters(self):
        """Action-policy parameters, excluding S_t-only peer forecasts.

        Peer forecast modules have their own supervised objective. Allowing
        PPO or action imitation to repurpose them as arbitrary latent features
        destroyed forecast calibration and produced inconsistent yielding
        beliefs. The response policy and all neural safety scales remain
        trainable here.
        """

        return chain(
            (self.collision_risk_log_scale,),
            (self.yield_risk_log_scale,),
            (self.charger_queue_wait_log_scale,),
            (self.charger_occupant_wait_log_scale,),
            (self.priority_progress_log_scale,),
            (self.delivery_detour_log_scale,),
            (self.occupied_cell_risk_log_scale,),
            (self.participant_collision_risk_log_scale,),
            (self.energy_exhaustion_risk_log_scale,),
            (self.participant_standoff_progress_log_scale,),
            (self.participant_delivery_detour_log_scale,),
            (self.participant_expected_collision_risk_log_scale,),
            (self.participant_robust_progress_log_scale,),
            (self.energy_route_deficit_log_scale,),
            (self.robust_progress_log_scale,),
            self.intent_encoder.parameters(),
            self.mission_head.parameters(),
            self.actor.parameters(),
            self.action_scorer.parameters(),
            self.participant_partner_action_head.parameters(),
            self.deadlock_escape_action_head.parameters(),
        )

    def values(
        self,
        global_states: torch.Tensor,
        agent_indices: torch.Tensor,
        max_agents: int,
    ) -> torch.Tensor:
        agent_one_hot = torch.nn.functional.one_hot(
            agent_indices,
            num_classes=max_agents,
        ).float()
        return self.critic(
            torch.cat((global_states, agent_one_hot), dim=-1)
        ).squeeze(-1)


class MAPPOPolicy:
    """Neural policy used by every warehouse robot at execution time."""

    model_version = MODEL_VERSION

    def __init__(
        self,
        environment_config: WarehouseConfig,
        algorithm_config: MAPPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.environment_config = environment_config
        self.algorithm_config = algorithm_config or MAPPOConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.algorithm_config.seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.algorithm_config.seed)
        self._base_inference_seed = int(self.algorithm_config.seed)
        self.network = SharedActorCentralCritic(
            observation_dim(environment_config),
            global_state_dim(environment_config),
            environment_config.max_agents,
            len(ACTIONS),
            self.algorithm_config.hidden_dim,
            self.algorithm_config.intent_dim,
            (2 * environment_config.local_patch_radius + 1) ** 2,
            environment_config.active_task_count,
            environment_config.horizon,
            environment_config.move_battery_cost,
            environment_config.mission_reserve_steps,
            environment_config.rows,
            environment_config.cols,
        ).to(self.device)

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTIONS

    def actor_input(
        self,
        observation: Any,
    ) -> np.ndarray:
        local = np.asarray(observation, dtype=np.float32)
        expected = observation_dim(self.environment_config)
        if local.shape != (expected,):
            raise ValueError(
                f"Expected one local observation with shape {(expected,)}, "
                f"received {local.shape}."
            )
        return independent_actor_input(local)

    def masked_actor_logits(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        local_dim = observation_dim(self.environment_config)
        if observations.shape[-1] != local_dim:
            raise ValueError(
                "Actor inputs must be independent local observations from the "
                "shared pre-move state."
            )
        logits = self.network.actor_logits(observations)
        mask = observations[..., -len(ACTIONS) :] > 0.5
        # MPS materialises a Python bool by synchronising the whole command
        # stream.  Runtime observations are validated below while they are
        # still NumPy arrays, and every offline label is checked against its
        # frozen-state mask before fitting.  Retain the defensive tensor-side
        # assertion for CPU/CUDA callers without imposing one synchronisation
        # per MPS minibatch.
        if self.device.type != "mps" and not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every agent observation must expose at least one legal action."
            )
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def act(
        self,
        observations: Mapping[str, Any],
        global_state: Any,
        *,
        deterministic: bool = False,
        decision_key: tuple[int, int] | None = None,
    ) -> tuple[dict[str, str], dict[str, ActionDistribution]]:
        del global_state  # The decentralized actor consumes local observations.
        agent_ids = sorted(observations, key=agent_index)
        if agent_ids != ["robot_1", "robot_2"]:
            raise ValueError("The shared Actor requires robot_1 and robot_2.")
        actor_inputs = np.stack(
            [self.actor_input(observations[agent_id]) for agent_id in agent_ids]
        )
        action_masks = actor_inputs[:, -len(ACTIONS) :] > 0.5
        if not bool(np.all(action_masks.any(axis=-1))):
            raise ValueError(
                "Every agent observation must expose at least one legal action."
            )
        tensor = torch.as_tensor(
            actor_inputs,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.masked_actor_logits(tensor)
            probabilities = torch.softmax(logits, dim=-1)
        cpu_logits = logits.detach().cpu().numpy()
        cpu_probabilities = probabilities.detach().cpu().numpy()
        actions: dict[str, str] = {}
        distributions: dict[str, ActionDistribution] = {}
        for row, agent_id in enumerate(agent_ids):
            if deterministic:
                index = int(np.argmax(cpu_probabilities[row]))
            elif decision_key is not None:
                episode_id, frame = decision_key
                rng = np.random.default_rng(
                    independent_agent_seed(
                        base_seed=self._base_inference_seed,
                        episode_id=episode_id,
                        frame=frame,
                        agent_id=agent_id,
                    )
                )
                index = int(
                    rng.choice(len(ACTIONS), p=cpu_probabilities[row])
                )
            else:
                index = int(
                    torch.multinomial(
                        probabilities[row].detach().cpu(),
                        1,
                        generator=self._generator,
                    ).item()
                )
            actions[agent_id] = ACTIONS[index]
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id,
                actions=ACTIONS,
                probabilities=tuple(
                    float(value)
                    for value in cpu_probabilities[row].tolist()
                ),
                logits=tuple(
                    float(value)
                    for value in cpu_logits[row].tolist()
                ),
                action_mask=tuple(
                    float(value)
                    for value in np.asarray(
                        observations[agent_id],
                        dtype=np.float32,
                    )[-len(ACTIONS) :].tolist()
                ),
                proposed_action=ACTIONS[index],
            )
        return actions, distributions

    def values(
        self,
        global_state: Any,
        agent_ids: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        global_array = np.asarray(global_state, dtype=np.float32)
        repeated = np.repeat(global_array[None, :], len(agent_ids), axis=0)
        indices = np.asarray(
            [agent_index(agent_id) for agent_id in agent_ids],
            dtype=np.int64,
        )
        with torch.no_grad():
            values = self.network.values(
                torch.as_tensor(
                    repeated,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=self.device,
                ),
                self.environment_config.max_agents,
            )
        return values.detach().cpu().numpy()

    def get_rng_state(self) -> torch.Tensor:
        return self._generator.get_state().clone()

    def set_rng_state(self, state: Any) -> None:
        if state is not None:
            self._generator.set_state(
                torch.as_tensor(state, dtype=torch.uint8, device="cpu")
            )

    def seed_rng(self, seed: int) -> None:
        self._generator.manual_seed(seed)
        self._base_inference_seed = int(seed)

    def fork_for_inference(self, *, seed: int) -> "MAPPOPolicy":
        """Share immutable network weights while isolating session RNG state."""

        policy = object.__new__(type(self))
        policy.environment_config = self.environment_config
        policy.algorithm_config = self.algorithm_config
        policy.device = self.device
        policy.network = self.network
        policy._generator = torch.Generator(device="cpu")
        policy._generator.manual_seed(int(seed))
        policy._base_inference_seed = int(seed)
        return policy

    def save(
        self,
        path: str | Path,
        *,
        training_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": self.model_version,
            "environment_version": ENVIRONMENT_VERSION,
            "reward_version": REWARD_VERSION,
            "network_state_dict": self.network.state_dict(),
            "environment_config": asdict(self.environment_config),
            "algorithm_config": asdict(self.algorithm_config),
            "training_metadata": dict(training_metadata or {}),
            "action_execution_version": ACTION_EXECUTION_VERSION,
            "runtime_controller": RUNTIME_CONTROLLER,
        }
        torch.save(payload, target)
        return target

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> "MAPPOPolicy":
        if payload.get("model_version") != cls.model_version:
            raise ValueError(
                f"Unsupported MAPPO checkpoint version "
                f"{payload.get('model_version')!r}; expected {cls.model_version!r}. "
                "Retrain the policy for the current task."
            )
        if (
            payload.get("environment_version") != ENVIRONMENT_VERSION
        ):
            raise ValueError(
                "The MAPPO checkpoint uses an incompatible environment version. "
                "Retrain the policy for safe-mission energy shaping."
            )
        if payload.get("reward_version") != REWARD_VERSION:
            raise ValueError(
                "The MAPPO checkpoint uses an incompatible reward version. "
                "Retrain the policy for the current reward contract."
            )
        if payload.get("action_execution_version") != ACTION_EXECUTION_VERSION:
            raise ValueError(
                "The MAPPO checkpoint does not use direct neural action "
                "execution. Retrain it for the current execution contract."
            )
        if payload.get("runtime_controller") != RUNTIME_CONTROLLER:
            raise ValueError(
                "The MAPPO checkpoint declares an incompatible runtime "
                "controller. Only direct MAPPO actor execution is accepted."
            )
        environment_payload = dict(payload["environment_config"])
        environment_payload["reward"] = RewardConfig(
            **environment_payload.get("reward", {})
        )
        policy = cls(
            WarehouseConfig(**environment_payload),
            MAPPOConfig(**payload["algorithm_config"]),
            device=device,
        )
        try:
            incompatible = policy.network.load_state_dict(
                payload["network_state_dict"], strict=False
            )
            allowed_missing = {
                "collision_risk_log_scale",
                "yield_risk_log_scale",
                "charger_queue_wait_log_scale",
                "charger_occupant_wait_log_scale",
                "priority_progress_log_scale",
                "delivery_detour_log_scale",
                "occupied_cell_risk_log_scale",
                "participant_collision_risk_log_scale",
                "energy_exhaustion_risk_log_scale",
                "participant_standoff_progress_log_scale",
                "participant_delivery_detour_log_scale",
                "participant_expected_collision_risk_log_scale",
                "participant_robust_progress_log_scale",
                "energy_route_deficit_log_scale",
                "robust_progress_log_scale",
                *{
                    f"teammate_context_predictor.{index}.{suffix}"
                    for index in (0, 2, 4)
                    for suffix in ("weight", "bias")
                },
                *{
                    f"participant_context_predictor.{index}.{suffix}"
                    for index in (0, 2, 4)
                    for suffix in ("weight", "bias")
                },
                *{
                    f"deadlock_escape_action_head.{index}.{suffix}"
                    for index in (0, 2)
                    for suffix in ("weight", "bias")
                },
            }
            if (
                set(incompatible.missing_keys) - allowed_missing
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "checkpoint tensors do not match the current Actor"
                )
        except RuntimeError as exc:
            raise ValueError(
                "The MAPPO checkpoint is incompatible with the current "
                "observation schema. Retrain it with `python train_rl.py`."
            ) from exc
        if payload.get("policy_rng_state") is not None:
            policy.set_rng_state(payload["policy_rng_state"])
        return policy

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "MAPPOPolicy":
        payload = torch.load(
            Path(path),
            map_location=device,
            weights_only=False,
        )
        policy = cls.from_payload(payload, device=device)
        policy.network.eval()
        return policy

    def metadata_json(self) -> str:
        return json.dumps(
            {
                "model_version": self.model_version,
                "environment_config": asdict(self.environment_config),
                "algorithm_config": asdict(self.algorithm_config),
                "action_names": ACTIONS,
            },
            ensure_ascii=False,
            indent=2,
        )


def agent_index(agent_id: str) -> int:
    """Return the zero-based critic identity encoded by a public agent ID."""

    try:
        return int(agent_id.rsplit("_", 1)[1]) - 1
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Agent IDs must end in a one-based integer: {agent_id}"
        ) from exc
