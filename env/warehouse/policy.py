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
from .domain import WarehouseConfig
from .navigation import ACTIONS
from .observations import global_state_dim, observation_dim
from .rewards import REWARD_VERSION, RewardConfig


AUTOREGRESSIVE_CONTEXT_DIM = len(ACTIONS) + 1
MISSION_INTENT_NAMES = (
    "task_slot_1",
    "task_slot_2",
    "delivery",
    "charge",
    "wait",
)


def autoregressive_actor_input(
    observation: Any,
    *,
    preceding_action: str | None,
) -> np.ndarray:
    """Append the start token or robot-1 action used by the shared Actor."""

    local = np.asarray(observation, dtype=np.float32)
    if local.ndim != 1:
        raise ValueError("An Actor input must start with one flat observation.")
    context = np.zeros(AUTOREGRESSIVE_CONTEXT_DIM, dtype=np.float32)
    if preceding_action is None:
        context[0] = 1.0
    else:
        if preceding_action not in ACTIONS:
            raise ValueError(f"Unknown preceding action {preceding_action!r}.")
        context[1 + ACTIONS.index(preceding_action)] = 1.0
    return np.concatenate((local, context)).astype(np.float32, copy=False)


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
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.base_local_dim = int(local_dim) - AUTOREGRESSIVE_CONTEXT_DIM
        coordination_dim = (
            8
            + 2 * self.action_dim
            + 2
            * (9 + 6 * int(active_task_count))
            * self.action_dim
            + self.action_dim**2
        )
        self.per_action_feature_dim = 9 + 6 * int(active_task_count)
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
        # A shared per-action neural scorer exposes the algebra that a plain
        # MLP struggled to learn: robot 2 must select the collision-matrix
        # column identified by robot 1's already sampled action.  This branch
        # is fully trainable and only contributes logits; it never masks,
        # replaces, or post-processes the Actor's selected action.
        structured_feature_dim = (
            self.per_action_feature_dim
            + 1
            + 1
            + self.action_dim
            + 1
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
        base_logits = self.actor(torch.cat((observations, intent), dim=-1))
        local = observations[..., : self.base_local_dim]
        context = observations[..., -AUTOREGRESSIVE_CONTEXT_DIM:]
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
        predicted_teammate_logits = self.teammate_action_predictor(
            torch.cat(
                (teammate_action_features, action_identity, action_intent),
                dim=-1,
            )
        ).squeeze(-1)
        predicted_teammate = torch.softmax(
            predicted_teammate_logits,
            dim=-1,
        )
        # Robot 1 uses a learned teammate forecast. Robot 2 receives robot 1's
        # sampled action explicitly and therefore selects that exact matrix
        # column. Both paths remain differentiable Actor computations.
        preceding_action = (
            context[..., :1] * predicted_teammate
            + context[..., 1:]
        )
        selected_collision = torch.einsum(
            "...ij,...j->...i",
            collision_matrix,
            preceding_action,
        ).unsqueeze(-1)
        legal = local[..., -self.action_dim :].unsqueeze(-1)
        start_token = context[..., :1].unsqueeze(-2).expand(
            *local.shape[:-1], self.action_dim, 1
        )
        structured = torch.cat(
            (
                own_action_features,
                selected_collision,
                legal,
                action_identity,
                start_token,
                action_intent,
            ),
            dim=-1,
        )
        structured_logits = self.action_scorer(structured).squeeze(-1)
        return base_logits + structured_logits, mission_logits

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_outputs(observations)[0]

    def mission_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_outputs(observations)[1]

    def actor_parameters(self):
        """All trainable decentralized-Actor parameters, including intent."""

        return chain(
            self.intent_encoder.parameters(),
            self.mission_head.parameters(),
            self.actor.parameters(),
            self.action_scorer.parameters(),
            self.teammate_action_predictor.parameters(),
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
        self.network = SharedActorCentralCritic(
            observation_dim(environment_config) + AUTOREGRESSIVE_CONTEXT_DIM,
            global_state_dim(environment_config),
            environment_config.max_agents,
            len(ACTIONS),
            self.algorithm_config.hidden_dim,
            self.algorithm_config.intent_dim,
            (2 * environment_config.local_patch_radius + 1) ** 2,
            environment_config.active_task_count,
        ).to(self.device)

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTIONS

    def actor_input(
        self,
        observation: Any,
        *,
        preceding_action: str | None,
    ) -> np.ndarray:
        local = np.asarray(observation, dtype=np.float32)
        expected = observation_dim(self.environment_config)
        if local.shape != (expected,):
            raise ValueError(
                f"Expected one local observation with shape {(expected,)}, "
                f"received {local.shape}."
            )
        return autoregressive_actor_input(
            local,
            preceding_action=preceding_action,
        )

    def masked_actor_logits(
        self,
        observations: torch.Tensor,
        action_contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_dim = observation_dim(self.environment_config)
        if observations.shape[-1] == local_dim:
            mask_source = observations
            if action_contexts is None:
                action_contexts = torch.zeros(
                    (*observations.shape[:-1], AUTOREGRESSIVE_CONTEXT_DIM),
                    dtype=observations.dtype,
                    device=observations.device,
                )
                action_contexts[..., 0] = 1.0
            if action_contexts.shape != (
                *observations.shape[:-1],
                AUTOREGRESSIVE_CONTEXT_DIM,
            ):
                raise ValueError("Autoregressive action contexts do not align.")
            actor_inputs = torch.cat((observations, action_contexts), dim=-1)
        elif observations.shape[-1] == local_dim + AUTOREGRESSIVE_CONTEXT_DIM:
            if action_contexts is not None:
                raise ValueError(
                    "Do not provide a separate context for augmented Actor inputs."
                )
            mask_source = observations[..., :local_dim]
            actor_inputs = observations
        else:
            raise ValueError(
                "Actor inputs must be local observations or local observations "
                "augmented with one autoregressive action context."
            )
        logits = self.network.actor_logits(actor_inputs)
        mask = mask_source[..., -len(ACTIONS) :] > 0.5
        if not torch.all(mask.any(dim=-1)):
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
        fixed_actions: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, ActionDistribution]]:
        del global_state  # The decentralized actor consumes local observations.
        agent_ids = sorted(observations, key=agent_index)
        if agent_ids != ["robot_1", "robot_2"]:
            raise ValueError("The autoregressive Actor requires robot_1 then robot_2.")
        fixed = {str(key): str(value) for key, value in (fixed_actions or {}).items()}
        if any(agent_id not in agent_ids for agent_id in fixed):
            raise ValueError("fixed_actions contains an unknown robot.")
        if any(agent_id != self.environment_config.human_agent_id for agent_id in fixed):
            raise ValueError(
                "Only the configured participant/proxy-human robot may replace "
                "an Actor action; AI robot actions are immutable."
            )
        if any(action not in ACTIONS for action in fixed.values()):
            raise ValueError("fixed_actions contains an unknown action.")
        actions: dict[str, str] = {}
        distributions: dict[str, ActionDistribution] = {}
        preceding_action: str | None = None
        for agent_id in agent_ids:
            actor_input = self.actor_input(
                observations[agent_id],
                preceding_action=preceding_action,
            )
            tensor = torch.as_tensor(
                actor_input[None, :],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                logits = self.masked_actor_logits(tensor)
                probabilities = torch.softmax(logits, dim=-1)
            if agent_id in fixed:
                index = ACTIONS.index(fixed[agent_id])
            elif deterministic:
                index = int(probabilities[0].argmax(dim=-1).item())
            else:
                index = int(
                    torch.multinomial(
                        probabilities[0].detach().cpu(),
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
                    for value in probabilities[0].detach().cpu().tolist()
                ),
                logits=tuple(
                    float(value)
                    for value in logits[0].detach().cpu().tolist()
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
            preceding_action = actions[agent_id]
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

    def fork_for_inference(self, *, seed: int) -> "MAPPOPolicy":
        """Share immutable network weights while isolating session RNG state."""

        policy = object.__new__(type(self))
        policy.environment_config = self.environment_config
        policy.algorithm_config = self.algorithm_config
        policy.device = self.device
        policy.network = self.network
        policy._generator = torch.Generator(device="cpu")
        policy._generator.manual_seed(int(seed))
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
            policy.network.load_state_dict(payload["network_state_dict"])
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
