"""Parameter-sharing MAPPO with a centralized critic and decentralized actors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from core.policy_program_regularizer import (
    PolicyProgramRegularizer,
    RegularizationStateBatch,
)

from .contracts import (
    ACTION_EXECUTION_VERSION,
    RUNTIME_CONTROLLER,
    TRAINING_CHECKPOINT_VERSION,
)
from .environment import (
    ACTIONS,
    WarehouseConfig,
    WarehouseMultiAgentEnv,
    shortest_path_distance,
)
from .coordination import is_necessary_urgent_charger_clearance
from .policy import (
    MAPPOConfig,
    MAPPOPolicy,
    SharedActorCentralCritic,
    agent_index as _agent_index,
)
from .policy_metrics import (
    EfficiencyMetrics,
    max_agent_attribute,
    mean_agent_attribute,
    sum_agent_attribute,
)
from .rewards import REWARD_VERSION
from .scenarios import (
    apply_charger_commitment_scenario,
    apply_charger_handoff_scenario,
    apply_delivery_goal_clearance_scenario,
    apply_head_on_scenario,
    apply_task_commitment_scenario,
)


MAPPO_TRAINING_CHECKPOINT_VERSION = TRAINING_CHECKPOINT_VERSION


def _guard_program_gradients(
    task_gradients: Sequence[torch.Tensor | None],
    program_gradients: Sequence[torch.Tensor | None],
    *,
    regularization_weight: float,
    maximum_gradient_ratio: float,
    task_gradient_floor: float,
    project_conflicts: bool,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Combine task and program gradients without sacrificing the task step.

    The extracted program is an intentionally lossy teacher.  Its gradient is
    first projected away from the task gradient when the two conflict, then
    capped relative to the task-gradient norm.  Consequently, a large
    configured lambda can strengthen compatible regularity updates without
    allowing the auxiliary objective to reverse the local PPO improvement.
    """

    if len(task_gradients) != len(program_gradients):
        raise ValueError("Task and program gradient lists must have equal length.")
    if regularization_weight < 0.0:
        raise ValueError("regularization_weight must be non-negative.")
    if maximum_gradient_ratio < 0.0:
        raise ValueError("maximum_gradient_ratio must be non-negative.")
    if task_gradient_floor < 0.0:
        raise ValueError("task_gradient_floor must be non-negative.")

    reference = next(
        (
            gradient
            for gradient in (*task_gradients, *program_gradients)
            if gradient is not None
        ),
        None,
    )
    if reference is None:
        return [None for _ in task_gradients], {
            "task_gradient_norm": 0.0,
            "program_gradient_norm": 0.0,
            "applied_program_gradient_norm": 0.0,
            "applied_program_lambda": 0.0,
            "program_gradient_ratio": 0.0,
            "gradient_conflict": 0.0,
            "gradient_guard_saturated": 0.0,
        }

    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    task_squared_norm = zero
    program_squared_norm = zero
    task_program_dot = zero
    for task_gradient, program_gradient in zip(
        task_gradients,
        program_gradients,
    ):
        if task_gradient is not None:
            task_squared_norm = (
                task_squared_norm + task_gradient.detach().pow(2).sum()
            )
        if program_gradient is not None:
            program_squared_norm = (
                program_squared_norm + program_gradient.detach().pow(2).sum()
            )
        if task_gradient is not None and program_gradient is not None:
            task_program_dot = task_program_dot + (
                task_gradient.detach() * program_gradient.detach()
            ).sum()

    epsilon = torch.finfo(reference.dtype).eps
    gradient_conflict = bool(task_program_dot.item() < 0.0)
    projection_coefficient = zero
    if project_conflicts and gradient_conflict:
        projection_coefficient = task_program_dot / (
            task_squared_norm + epsilon
        )

    projected_program_gradients: list[torch.Tensor | None] = []
    projected_squared_norm = zero
    for task_gradient, program_gradient in zip(
        task_gradients,
        program_gradients,
    ):
        if program_gradient is None:
            projected_program_gradients.append(None)
            continue
        projected = program_gradient.detach()
        if project_conflicts and gradient_conflict and task_gradient is not None:
            projected = projected - (
                projection_coefficient * task_gradient.detach()
            )
        projected_program_gradients.append(projected)
        projected_squared_norm = projected_squared_norm + projected.pow(2).sum()

    task_norm = torch.sqrt(task_squared_norm)
    projected_program_norm = torch.sqrt(projected_squared_norm)
    protected_task_norm = torch.maximum(
        task_norm,
        torch.as_tensor(
            task_gradient_floor,
            dtype=reference.dtype,
            device=reference.device,
        ),
    )
    requested_lambda = torch.as_tensor(
        regularization_weight,
        dtype=reference.dtype,
        device=reference.device,
    )
    maximum_program_norm = maximum_gradient_ratio * protected_task_norm
    applied_lambda = requested_lambda
    if projected_program_norm.item() > 0.0:
        applied_lambda = torch.minimum(
            requested_lambda,
            maximum_program_norm / (projected_program_norm + epsilon),
        )
    else:
        applied_lambda = zero

    combined_gradients: list[torch.Tensor | None] = []
    for task_gradient, projected_program_gradient in zip(
        task_gradients,
        projected_program_gradients,
    ):
        if task_gradient is None and projected_program_gradient is None:
            combined_gradients.append(None)
            continue
        if task_gradient is None:
            assert projected_program_gradient is not None
            combined_gradients.append(
                applied_lambda * projected_program_gradient
            )
            continue
        combined = task_gradient.detach()
        if projected_program_gradient is not None:
            combined = combined + applied_lambda * projected_program_gradient
        combined_gradients.append(combined)

    applied_program_norm = applied_lambda * projected_program_norm
    return combined_gradients, {
        "task_gradient_norm": float(task_norm.detach().cpu()),
        "program_gradient_norm": float(
            projected_program_norm.detach().cpu()
        ),
        "applied_program_gradient_norm": float(
            applied_program_norm.detach().cpu()
        ),
        "applied_program_lambda": float(applied_lambda.detach().cpu()),
        "program_gradient_ratio": float(
            (
                applied_program_norm
                / torch.clamp(protected_task_norm, min=epsilon)
            )
            .detach()
            .cpu()
        ),
        "gradient_conflict": float(gradient_conflict),
        "gradient_guard_saturated": float(
            applied_lambda.item() + float(epsilon) < regularization_weight
        ),
    }


@dataclass
class EpisodeBatch:
    observations: np.ndarray
    global_states: np.ndarray
    agent_indices: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    trainable_mask: np.ndarray
    episode_reward: float
    episode_steps: int
    pickups: int
    deliveries: int
    collisions: int
    shutdowns: int
    terminal_reason: str | None
    charger_uses: int = 0
    avoidable_detours: int = 0
    route_regret: float = 0.0
    minimum_battery: float = 0.0
    proxy_human_overrides: int = 0
    base_training_reward: float = 0.0
    potential_shaping_reward: float = 0.0
    avoidable_wait_penalty_reward: float = 0.0
    mission_regression_penalty_reward: float = 0.0
    individual_training_rewards: dict[str, float] = field(default_factory=dict)
    individual_progress_rewards: dict[str, float] = field(default_factory=dict)
    coordination_progress_reward: float = 0.0
    counterfactual_regret_units: dict[str, float] = field(default_factory=dict)
    counterfactual_regret_penalty_rewards: dict[str, float] = field(
        default_factory=dict
    )
    repeated_avoidable_wait_penalty_rewards: dict[str, float] = field(
        default_factory=dict
    )
    avoidable_wait_counts: dict[str, int] = field(default_factory=dict)
    maximum_avoidable_wait_streaks: dict[str, int] = field(default_factory=dict)
    detour_counts: dict[str, int] = field(default_factory=dict)
    loaded_detour_counts: dict[str, int] = field(default_factory=dict)
    path_efficiency_actual_over_shortest_safe: float = 0.0
    energy_curriculum_applied: bool = False
    coordination_curriculum_kind: str | None = None
    initial_minimum_battery: float = 100.0
    semantic_features: tuple[Mapping[str, float], ...] = ()
    regularization_observations: np.ndarray | None = None
    regularization_targets: np.ndarray | None = None
    regularization_weights: np.ndarray | None = None


class MAPPOTrainer:
    """On-policy trainer with GAE and clipped PPO updates."""

    def __init__(self, policy: MAPPOPolicy) -> None:
        self.policy = policy
        cfg = policy.algorithm_config
        self.actor_optimizer = torch.optim.Adam(
            policy.network.actor_parameters(),
            lr=cfg.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(policy.network.critic.parameters(), lr=cfg.critic_lr)
        self._rng = np.random.default_rng(cfg.seed)
        self._rollout_rng = np.random.default_rng(cfg.seed + 1)
        # Program-target sampling must not advance PPO's minibatch RNG.
        # Otherwise merely enabling the regularizer changes all later
        # minibatch permutations, confounding paired lambda comparisons.
        self._regularizer_rng = np.random.default_rng(cfg.seed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "numpy_rng_state": self._rng.bit_generator.state,
            "rollout_numpy_rng_state": self._rollout_rng.bit_generator.state,
            "regularizer_numpy_rng_state": (
                self._regularizer_rng.bit_generator.state
            ),
            "policy_rng_state": self.policy.get_rng_state(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self._rng.bit_generator.state = payload["numpy_rng_state"]
        rollout_state = payload.get("rollout_numpy_rng_state")
        if rollout_state is not None:
            self._rollout_rng.bit_generator.state = rollout_state
        regularizer_state = payload.get("regularizer_numpy_rng_state")
        if regularizer_state is not None:
            self._regularizer_rng.bit_generator.state = regularizer_state
        self.policy.set_rng_state(payload.get("policy_rng_state"))

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        episode: int,
        metrics: Sequence[Mapping[str, Any]],
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_kind": "warehouse_mappo_training",
                "checkpoint_version": MAPPO_TRAINING_CHECKPOINT_VERSION,
                "model_version": self.policy.model_version,
                "environment_version": WarehouseMultiAgentEnv.environment_name,
                "reward_version": REWARD_VERSION,
                "network_state_dict": self.policy.network.state_dict(),
                "environment_config": asdict(self.policy.environment_config),
                "algorithm_config": asdict(self.policy.algorithm_config),
                "policy_rng_state": self.policy.get_rng_state(),
                "action_execution_version": ACTION_EXECUTION_VERSION,
                "runtime_controller": RUNTIME_CONTROLLER,
                "trainer_state": self.state_dict(),
                "episode": int(episode),
                "metrics": [dict(item) for item in metrics],
                "extra_state": dict(extra_state or {}),
            },
            target,
        )
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["MAPPOTrainer", int, list[dict[str, Any]]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("checkpoint_kind") != "warehouse_mappo_training":
            raise ValueError("The requested file is not a MAPPO training checkpoint.")
        if payload.get("checkpoint_version") != MAPPO_TRAINING_CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported MAPPO training checkpoint version; start safe-mission "
                "training from scratch instead of resuming the v8 reward run."
            )
        policy = MAPPOPolicy.from_payload(payload, device=device)
        policy.network.train()
        trainer = cls(policy)
        trainer.load_state_dict(payload["trainer_state"])
        return trainer, int(payload["episode"]), [dict(item) for item in payload.get("metrics", ())]

    def collect_episode(
        self,
        env: WarehouseMultiAgentEnv,
        *,
        seed: int | None = None,
        semantic_feature_provider: Callable[[Any, str], Mapping[str, float]] | None = None,
        energy_curriculum_probability: float = 0.0,
        energy_curriculum_min_battery: float = 15.0,
        energy_curriculum_max_battery: float = 35.0,
        coordination_curriculum_probability: float = 0.0,
    ) -> EpisodeBatch:
        if not 0.0 <= energy_curriculum_probability <= 1.0:
            raise ValueError("energy_curriculum_probability must be between zero and one.")
        if not 0.0 < energy_curriculum_min_battery <= energy_curriculum_max_battery:
            raise ValueError("Invalid energy curriculum battery range.")
        if not 0.0 <= coordination_curriculum_probability <= 1.0:
            raise ValueError(
                "coordination_curriculum_probability must be between zero and one."
            )
        observations, _ = env.reset(seed=seed)
        coordination_curriculum_kind: str | None = None
        if self._rollout_rng.random() < coordination_curriculum_probability:
            curriculum_choice = int(self._rollout_rng.integers(0, 5))
            if curriculum_choice == 0:
                coordination_curriculum_kind = "head_on"
                apply_head_on_scenario(
                    env,
                    reverse=bool(self._rollout_rng.integers(0, 2)),
                )
            elif curriculum_choice == 1:
                coordination_curriculum_kind = "charger_handoff"
                occupant_index = int(self._rollout_rng.integers(0, 2))
                apply_charger_handoff_scenario(
                    env,
                    occupant_agent_id=env.agent_ids[occupant_index],
                    queued_battery=float(self._rollout_rng.uniform(8.0, 28.0)),
                )
            elif curriculum_choice == 2:
                coordination_curriculum_kind = "delivery_goal_clearance"
                apply_delivery_goal_clearance_scenario(
                    env,
                    variant=int(self._rollout_rng.integers(0, 10_000)),
                )
            elif curriculum_choice == 3:
                coordination_curriculum_kind = "charger_commitment"
                apply_charger_commitment_scenario(
                    env,
                    agent_id=env.agent_ids[int(self._rollout_rng.integers(0, 2))],
                    variant=int(self._rollout_rng.integers(0, 10_000)),
                )
            else:
                coordination_curriculum_kind = "task_commitment"
                apply_task_commitment_scenario(
                    env,
                    variant=int(self._rollout_rng.integers(0, 10_000)),
                )
            observations = env.observations()
        energy_curriculum_applied = bool(
            coordination_curriculum_kind is None
            and self._rollout_rng.random() < energy_curriculum_probability
        )
        if energy_curriculum_applied:
            curriculum_state = env.get_state()
            selected_index = int(
                self._rollout_rng.integers(0, len(curriculum_state.agents))
            )
            curriculum_state.agents[selected_index].battery = float(
                self._rollout_rng.uniform(
                    energy_curriculum_min_battery,
                    energy_curriculum_max_battery,
                )
            )
            env.set_state(curriculum_state)
            observations = env.observations()
        initial_minimum_battery = min(
            (agent.battery for agent in env.state.agents),
            default=0.0,
        ) if env.state else 0.0
        agent_ids = list(env.agent_ids)
        rows: dict[str, list[Any]] = {
            "observations": [],
            "global_states": [],
            "agent_indices": [],
            "actions": [],
            "old_log_probs": [],
            "old_values": [],
            "rewards": [],
            "dones": [],
            "trainable_mask": [],
            "semantic_features": [],
        }
        episode_reward = 0.0
        base_training_reward = 0.0
        potential_shaping_reward = 0.0
        avoidable_wait_penalty_reward = 0.0
        mission_regression_penalty_reward = 0.0
        efficiency = EfficiencyMetrics()
        pickup_count = 0
        charger_uses = 0
        avoidable_detour_count = 0
        route_regret_total = 0.0
        terminal_reason: str | None = None
        proxy_human_episode = bool(self._rollout_rng.random() < 0.30)
        proxy_human_overrides = 0
        minimum_battery_seen = initial_minimum_battery
        while True:
            state_before_step = deepcopy(env.state)
            global_state = env.global_state()
            overridden_agents: set[str] = set()
            fixed_actions: dict[str, str] = {}
            if proxy_human_episode and self._rollout_rng.random() < 0.20:
                human_id = env.config.human_agent_id
                mask = env.action_masks()[human_id]
                if self._rollout_rng.random() < 0.50:
                    replacement = "WAIT"
                else:
                    legal_moves = [
                        action
                        for action, allowed in zip(ACTIONS, mask)
                        if allowed > 0.5 and action != "WAIT"
                    ]
                    replacement = (
                        str(self._rollout_rng.choice(legal_moves))
                        if legal_moves
                        else "WAIT"
                    )
                fixed_actions[human_id] = replacement
                overridden_agents.add(human_id)
                proxy_human_overrides += 1
            actions, distributions = self.policy.act(
                observations,
                global_state,
                deterministic=False,
                fixed_actions=fixed_actions,
            )
            values = self.policy.values(global_state, agent_ids)
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            for index, agent_id in enumerate(agent_ids):
                action_index = ACTIONS.index(actions[agent_id])
                probability = max(1e-8, distributions[agent_id].probabilities[action_index])
                preceding_action = (
                    None if index == 0 else actions[agent_ids[index - 1]]
                )
                rows["observations"].append(
                    self.policy.actor_input(
                        observations[agent_id],
                        preceding_action=preceding_action,
                    )
                )
                rows["global_states"].append(np.asarray(global_state, dtype=np.float32))
                rows["agent_indices"].append(index)
                rows["actions"].append(action_index)
                rows["old_log_probs"].append(float(np.log(probability)))
                rows["old_values"].append(float(values[index]))
                rows["rewards"].append(float(rewards[agent_id]))
                # A time-limit truncation is not an absorbing MDP state. Only
                # true physical termination (collision/shutdown) zeros the
                # bootstrap value used by GAE.
                rows["dones"].append(float(terminated))
                rows["trainable_mask"].append(
                    float(
                        agent_id not in overridden_agents
                        and
                        np.count_nonzero(
                            np.asarray(
                                observations[agent_id],
                                dtype=np.float32,
                            )[-len(ACTIONS) :]
                            > 0.5
                        )
                        > 1
                    )
                )
                if semantic_feature_provider is not None and state_before_step is not None:
                    rows["semantic_features"].append(
                        dict(semantic_feature_provider(state_before_step, agent_id))
                    )
            episode_reward += float(np.mean(tuple(rewards.values())))
            base_training_reward += float(info.get("base_training_reward", 0.0))
            potential_shaping_reward += float(
                info.get("potential_shaping_reward", 0.0)
            )
            avoidable_wait_penalty_reward += float(
                info.get("avoidable_wait_penalty_reward", 0.0)
            )
            mission_regression_penalty_reward += float(
                info.get("mission_regression_penalty_reward", 0.0)
            )
            efficiency.update_step(info, rewards)
            pickup_count += len(info.get("pickup_agents", ()))
            charger_uses += int(bool(info.get("charger_used", False)))
            avoidable_detour_count += len(
                info.get("avoidable_detour_agents", ())
            )
            route_regret_total += sum(
                float(value)
                for value in info.get("route_regret", {}).values()
            )
            if env.state is not None:
                minimum_battery_seen = min(
                    minimum_battery_seen,
                    *(agent.battery for agent in env.state.agents),
                )
            observations = next_observations
            if done:
                terminal_reason = info.get("terminal_reason")
                break

        steps = len(rows["actions"]) // len(agent_ids)
        rewards_matrix = np.asarray(rows["rewards"], dtype=np.float32).reshape(steps, len(agent_ids))
        values_matrix = np.asarray(rows["old_values"], dtype=np.float32).reshape(steps, len(agent_ids))
        dones_matrix = np.asarray(rows["dones"], dtype=np.float32).reshape(steps, len(agent_ids))
        advantages = np.zeros_like(rewards_matrix)
        last_advantage = np.zeros(len(agent_ids), dtype=np.float32)
        next_values = (
            self.policy.values(env.global_state(), agent_ids).astype(np.float32)
            if env.state is not None and env.state.truncated
            else np.zeros(len(agent_ids), dtype=np.float32)
        )
        cfg = self.policy.algorithm_config
        for step in reversed(range(steps)):
            nonterminal = 1.0 - dones_matrix[step]
            delta = rewards_matrix[step] + cfg.gamma * next_values * nonterminal - values_matrix[step]
            last_advantage = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_advantage
            advantages[step] = last_advantage
            next_values = values_matrix[step]
        returns = advantages + values_matrix
        return EpisodeBatch(
            observations=np.stack(rows["observations"]),
            global_states=np.stack(rows["global_states"]),
            agent_indices=np.asarray(rows["agent_indices"], dtype=np.int64),
            actions=np.asarray(rows["actions"], dtype=np.int64),
            old_log_probs=np.asarray(rows["old_log_probs"], dtype=np.float32),
            old_values=np.asarray(rows["old_values"], dtype=np.float32),
            rewards=np.asarray(rows["rewards"], dtype=np.float32),
            dones=np.asarray(rows["dones"], dtype=np.float32),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            trainable_mask=np.asarray(
                rows["trainable_mask"],
                dtype=np.float32,
            ),
            episode_reward=episode_reward,
            episode_steps=steps,
            pickups=pickup_count,
            deliveries=env.state.total_deliveries if env.state else 0,
            collisions=env.state.collision_count if env.state else 0,
            shutdowns=env.state.shutdown_count if env.state else 0,
            terminal_reason=terminal_reason,
            charger_uses=charger_uses,
            avoidable_detours=avoidable_detour_count,
            route_regret=route_regret_total,
            minimum_battery=minimum_battery_seen,
            proxy_human_overrides=proxy_human_overrides,
            base_training_reward=base_training_reward,
            potential_shaping_reward=potential_shaping_reward,
            avoidable_wait_penalty_reward=avoidable_wait_penalty_reward,
            mission_regression_penalty_reward=(
                mission_regression_penalty_reward
            ),
            individual_training_rewards=efficiency.individual_training_rewards,
            individual_progress_rewards=efficiency.progress_rewards,
            coordination_progress_reward=efficiency.coordination_progress_reward,
            counterfactual_regret_units=efficiency.regret_units,
            counterfactual_regret_penalty_rewards=efficiency.regret_penalties,
            repeated_avoidable_wait_penalty_rewards=(
                efficiency.repeated_wait_penalties
            ),
            avoidable_wait_counts=efficiency.avoidable_wait_counts,
            maximum_avoidable_wait_streaks=efficiency.maximum_wait_streaks,
            detour_counts=efficiency.detour_counts,
            loaded_detour_counts=efficiency.loaded_detour_counts,
            path_efficiency_actual_over_shortest_safe=float(
                info.get("path_efficiency_actual_over_shortest_safe", 0.0)
            ),
            energy_curriculum_applied=energy_curriculum_applied,
            coordination_curriculum_kind=coordination_curriculum_kind,
            initial_minimum_battery=initial_minimum_battery,
            semantic_features=tuple(rows["semantic_features"]),
        )

    def update(
        self,
        batch: EpisodeBatch,
        *,
        entropy_coef: float | None = None,
        regularization_weight: float = 0.0,
        program_regularizer: PolicyProgramRegularizer | None = None,
        regularization_gradient_guard_ratio: float | None = None,
        regularization_task_gradient_floor: float = 0.0,
        project_conflicting_regularization_gradients: bool = True,
        skill_anchor_observations: np.ndarray | None = None,
        skill_anchor_labels: np.ndarray | None = None,
        skill_anchor_weight: float = 0.0,
    ) -> dict[str, float]:
        trainable_indices = np.flatnonzero(
            batch.trainable_mask > 0.5
        )
        if len(trainable_indices) == 0:
            raise ValueError(
                "The episode batch contains no policy-controlled decisions."
            )
        advantages = batch.advantages.copy()
        trainable_advantages = advantages[trainable_indices]
        advantages[trainable_indices] = (
            trainable_advantages - trainable_advantages.mean()
        ) / max(1e-8, trainable_advantages.std())
        size = len(trainable_indices)
        cfg = self.policy.algorithm_config
        effective_entropy_coef = cfg.entropy_coef if entropy_coef is None else float(entropy_coef)
        has_regularizer = (
            regularization_weight > 0.0
            and batch.regularization_observations is not None
            and batch.regularization_targets is not None
            and len(batch.regularization_observations) > 0
        )
        has_skill_anchor = bool(
            skill_anchor_weight > 0.0
            and skill_anchor_observations is not None
            and skill_anchor_labels is not None
            and len(skill_anchor_observations) > 0
        )
        if skill_anchor_weight < 0.0:
            raise ValueError("skill_anchor_weight must be non-negative.")
        if has_skill_anchor:
            assert skill_anchor_observations is not None
            assert skill_anchor_labels is not None
            if len(skill_anchor_observations) != len(skill_anchor_labels):
                raise ValueError(
                    "Skill-anchor observations and labels must align."
                )
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "critic_masked_sample_loss": 0.0,
            "critic_masked_sample_updates": 0.0,
            "entropy": 0.0,
            "program_regularity_loss": 0.0,
            "program_complexity_loss": 0.0,
            "program_regularization_total_loss": 0.0,
            "mean_program_target_weight": 0.0,
            "program_feedback_coverage": 0.0,
            "program_lambda_active_fraction": 0.0,
            "task_actor_gradient_norm": 0.0,
            "program_gradient_norm": 0.0,
            "applied_program_gradient_norm": 0.0,
            "applied_program_gradient_lambda": 0.0,
            "program_to_task_gradient_ratio": 0.0,
            "program_gradient_conflict_fraction": 0.0,
            "program_gradient_guard_saturation_fraction": 0.0,
            "skill_anchor_loss": 0.0,
            "skill_anchor_accuracy": 0.0,
            "skill_anchor_weight": float(skill_anchor_weight),
            "updates": 0.0,
        }
        if (
            regularization_gradient_guard_ratio is not None
            and regularization_gradient_guard_ratio < 0.0
        ):
            raise ValueError(
                "regularization_gradient_guard_ratio must be non-negative."
            )
        if regularization_task_gradient_floor < 0.0:
            raise ValueError(
                "regularization_task_gradient_floor must be non-negative."
            )
        for _ in range(cfg.update_epochs):
            indices = trainable_indices[
                self._rng.permutation(size)
            ]
            for start in range(0, size, cfg.minibatch_size):
                selected = indices[start : start + cfg.minibatch_size]
                obs = torch.as_tensor(batch.observations[selected], dtype=torch.float32, device=self.policy.device)
                global_states = torch.as_tensor(batch.global_states[selected], dtype=torch.float32, device=self.policy.device)
                agent_indices = torch.as_tensor(batch.agent_indices[selected], dtype=torch.long, device=self.policy.device)
                actions = torch.as_tensor(batch.actions[selected], dtype=torch.long, device=self.policy.device)
                old_log_probs = torch.as_tensor(batch.old_log_probs[selected], dtype=torch.float32, device=self.policy.device)
                selected_advantages = torch.as_tensor(advantages[selected], dtype=torch.float32, device=self.policy.device)
                returns = torch.as_tensor(batch.returns[selected], dtype=torch.float32, device=self.policy.device)

                logits = self.policy.masked_actor_logits(obs)
                distribution = Categorical(logits=logits)
                log_probs = distribution.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * selected_advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * selected_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                entropy = distribution.entropy().mean()
                program_regularity_loss = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                program_regularization_objective = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                program_complexity_loss = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                mean_program_target_weight = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                program_feedback_coverage = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                program_lambda_active = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                skill_anchor_loss = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                skill_anchor_accuracy = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.policy.device,
                )
                if has_skill_anchor:
                    assert skill_anchor_observations is not None
                    assert skill_anchor_labels is not None
                    anchor_size = len(skill_anchor_observations)
                    anchor_sample_size = min(len(selected), anchor_size)
                    anchor_indices = self._regularizer_rng.choice(
                        anchor_size,
                        size=anchor_sample_size,
                        replace=False,
                    )
                    anchor_obs = torch.as_tensor(
                        skill_anchor_observations[anchor_indices],
                        dtype=torch.float32,
                        device=self.policy.device,
                    )
                    anchor_labels = torch.as_tensor(
                        skill_anchor_labels[anchor_indices],
                        dtype=torch.long,
                        device=self.policy.device,
                    )
                    anchor_logits = self.policy.masked_actor_logits(anchor_obs)
                    skill_anchor_loss = torch.nn.functional.cross_entropy(
                        anchor_logits,
                        anchor_labels,
                    )
                    skill_anchor_accuracy = (
                        anchor_logits.argmax(dim=-1) == anchor_labels
                    ).to(torch.float32).mean()
                if has_regularizer:
                    assert batch.regularization_observations is not None
                    assert batch.regularization_targets is not None
                    regularizer_size = len(batch.regularization_observations)
                    sample_size = min(len(selected), regularizer_size)
                    regularizer_indices = self._regularizer_rng.choice(
                        regularizer_size,
                        size=sample_size,
                        replace=False,
                    )
                    regularizer_obs = torch.as_tensor(
                        batch.regularization_observations[regularizer_indices],
                        dtype=torch.float32,
                        device=self.policy.device,
                    )
                    target = torch.as_tensor(
                        batch.regularization_targets[regularizer_indices],
                        dtype=torch.float32,
                        device=self.policy.device,
                    ).clamp_min(0.0)
                    target = target / target.sum(dim=-1, keepdim=True)
                    regularizer_logits = self.policy.masked_actor_logits(
                        regularizer_obs
                    )
                    if batch.regularization_weights is not None:
                        weights = torch.as_tensor(
                            batch.regularization_weights[regularizer_indices],
                            dtype=torch.float32,
                            device=self.policy.device,
                        )
                        # Average over the whole sampled regularizer batch,
                        # including gated-out examples.  Dividing by the
                        # positive weight sum would concentrate the full
                        # configured lambda on a small set of surviving
                        # examples, so a low-coverage safety gate could
                        # paradoxically produce a large Actor update.
                        mean_program_target_weight = weights.mean()
                        program_feedback_coverage = (
                            weights > 0.0
                        ).to(torch.float32).mean()
                        program_lambda_active = (
                            weights.sum() > 0.0
                        ).to(torch.float32)
                    else:
                        weights = None
                        mean_program_target_weight = torch.ones(
                            (),
                            dtype=torch.float32,
                            device=self.policy.device,
                        )
                        program_feedback_coverage = torch.ones(
                            (),
                            dtype=torch.float32,
                            device=self.policy.device,
                        )
                        program_lambda_active = torch.ones(
                            (),
                            dtype=torch.float32,
                            device=self.policy.device,
                        )
                    if program_regularizer is not None:
                        structural_program = (
                            program_regularizer.program
                            if program_regularizer.program is not None
                            else target
                        )
                        program_regularization_objective = (
                            program_regularizer.regularization_loss(
                                regularizer_logits,
                                structural_program,
                                RegularizationStateBatch(
                                    program_probabilities=target,
                                    sample_weights=weights,
                                ),
                            )
                        )
                        program_regularity_loss = (
                            program_regularization_objective.new_tensor(
                                program_regularizer.last_fidelity_loss or 0.0
                            )
                        )
                        program_complexity_loss = (
                            program_regularization_objective.new_tensor(
                                (
                                    program_regularizer.last_complexity.loss
                                    if program_regularizer.last_complexity
                                    is not None
                                    else 0.0
                                )
                            )
                        )
                    else:
                        # Direct-target path used by low-level regularizer
                        # tests. Production training supplies the structured
                        # PolicyProgramRegularizer above.
                        program_log_probs = torch.log_softmax(
                            regularizer_logits,
                            dim=-1,
                        )
                        target_safe = target.clamp_min(1e-8)
                        target_safe = target_safe / target_safe.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        actor_probabilities = torch.exp(program_log_probs)
                        per_sample_kl = torch.sum(
                            actor_probabilities
                            * (
                                program_log_probs
                                - torch.log(target_safe)
                            ),
                            dim=-1,
                        )
                        program_regularity_loss = (
                            per_sample_kl
                            if weights is None
                            else per_sample_kl * weights
                        ).mean()
                        program_regularization_objective = (
                            program_regularity_loss
                        )
                self.actor_optimizer.zero_grad(set_to_none=True)
                task_actor_loss = (
                    actor_loss - effective_entropy_coef * entropy
                    + float(skill_anchor_weight) * skill_anchor_loss
                )
                gradient_metrics = {
                    "task_gradient_norm": 0.0,
                    "program_gradient_norm": 0.0,
                    "applied_program_gradient_norm": 0.0,
                    "applied_program_lambda": 0.0,
                    "program_gradient_ratio": 0.0,
                    "gradient_conflict": 0.0,
                    "gradient_guard_saturated": 0.0,
                }
                if (
                    has_regularizer
                    and regularization_gradient_guard_ratio is not None
                ):
                    actor_parameters = tuple(
                        self.policy.network.actor_parameters()
                    )
                    task_gradients = torch.autograd.grad(
                        task_actor_loss,
                        actor_parameters,
                        allow_unused=True,
                    )
                    program_gradients = torch.autograd.grad(
                        program_regularization_objective,
                        actor_parameters,
                        allow_unused=True,
                    )
                    guarded_gradients, gradient_metrics = (
                        _guard_program_gradients(
                            task_gradients,
                            program_gradients,
                            regularization_weight=(
                                1.0
                                if program_regularizer is not None
                                else float(regularization_weight)
                            ),
                            maximum_gradient_ratio=float(
                                regularization_gradient_guard_ratio
                            ),
                            task_gradient_floor=float(
                                regularization_task_gradient_floor
                            ),
                            project_conflicts=bool(
                                project_conflicting_regularization_gradients
                            ),
                        )
                    )
                    if program_regularizer is not None:
                        # The new objective already contains lambda_extract
                        # before gradient guarding.  Report the realized scale
                        # relative to the unweighted KL for comparable logs.
                        gradient_metrics["applied_program_lambda"] *= float(
                            program_regularizer.lambda_extract
                        )
                    for parameter, gradient in zip(
                        actor_parameters,
                        guarded_gradients,
                    ):
                        parameter.grad = gradient
                else:
                    (
                        task_actor_loss
                        + (
                            program_regularization_objective
                            if program_regularizer is not None
                            else float(regularization_weight)
                            * program_regularization_objective
                        )
                    ).backward()
                nn.utils.clip_grad_norm_(
                    self.policy.network.actor_parameters(),
                    cfg.max_grad_norm,
                )
                self.actor_optimizer.step()

                values = self.policy.network.values(global_states, agent_indices, self.policy.environment_config.max_agents)
                critic_loss = torch.nn.functional.mse_loss(values, returns)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.network.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                metrics["actor_loss"] += float(actor_loss.detach().cpu())
                metrics["critic_loss"] += float(critic_loss.detach().cpu())
                metrics["entropy"] += float(entropy.detach().cpu())
                metrics["skill_anchor_loss"] += float(
                    skill_anchor_loss.detach().cpu()
                )
                metrics["skill_anchor_accuracy"] += float(
                    skill_anchor_accuracy.detach().cpu()
                )
                metrics["program_regularity_loss"] += float(
                    program_regularity_loss.detach().cpu()
                )
                metrics["program_complexity_loss"] += float(
                    program_complexity_loss.detach().cpu()
                )
                metrics["program_regularization_total_loss"] += float(
                    program_regularization_objective.detach().cpu()
                )
                metrics["mean_program_target_weight"] += float(
                    mean_program_target_weight.detach().cpu()
                )
                metrics["program_feedback_coverage"] += float(
                    program_feedback_coverage.detach().cpu()
                )
                metrics["program_lambda_active_fraction"] += float(
                    program_lambda_active.detach().cpu()
                )
                metrics["task_actor_gradient_norm"] += gradient_metrics[
                    "task_gradient_norm"
                ]
                metrics["program_gradient_norm"] += gradient_metrics[
                    "program_gradient_norm"
                ]
                metrics[
                    "applied_program_gradient_norm"
                ] += gradient_metrics["applied_program_gradient_norm"]
                metrics[
                    "applied_program_gradient_lambda"
                ] += gradient_metrics["applied_program_lambda"]
                metrics[
                    "program_to_task_gradient_ratio"
                ] += gradient_metrics["program_gradient_ratio"]
                metrics[
                    "program_gradient_conflict_fraction"
                ] += gradient_metrics["gradient_conflict"]
                metrics[
                    "program_gradient_guard_saturation_fraction"
                ] += gradient_metrics["gradient_guard_saturated"]
                metrics["updates"] += 1.0
        # Samples replaced by the proxy-human process are deliberately absent
        # from PPO's Actor objective, but they remain valid centralized-Critic
        # transitions.  Give those masked rows their own value-only pass.
        critic_only_indices = np.flatnonzero(batch.trainable_mask <= 0.5)
        if len(critic_only_indices) > 0:
            for _ in range(cfg.update_epochs):
                shuffled = critic_only_indices[
                    self._rng.permutation(len(critic_only_indices))
                ]
                for start in range(0, len(shuffled), cfg.minibatch_size):
                    selected = shuffled[start : start + cfg.minibatch_size]
                    global_states = torch.as_tensor(
                        batch.global_states[selected],
                        dtype=torch.float32,
                        device=self.policy.device,
                    )
                    agent_indices = torch.as_tensor(
                        batch.agent_indices[selected],
                        dtype=torch.long,
                        device=self.policy.device,
                    )
                    returns = torch.as_tensor(
                        batch.returns[selected],
                        dtype=torch.float32,
                        device=self.policy.device,
                    )
                    values = self.policy.network.values(
                        global_states,
                        agent_indices,
                        self.policy.environment_config.max_agents,
                    )
                    critic_only_loss = torch.nn.functional.mse_loss(
                        values,
                        returns,
                    )
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (cfg.value_coef * critic_only_loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.network.critic.parameters(),
                        cfg.max_grad_norm,
                    )
                    self.critic_optimizer.step()
                    metrics["critic_masked_sample_loss"] += float(
                        critic_only_loss.detach().cpu()
                    )
                    metrics["critic_masked_sample_updates"] += 1.0
        masked_divisor = max(1.0, metrics["critic_masked_sample_updates"])
        metrics["critic_masked_sample_loss"] /= masked_divisor
        divisor = max(1.0, metrics["updates"])
        for key in (
            "actor_loss",
            "critic_loss",
            "entropy",
            "program_regularity_loss",
            "program_complexity_loss",
            "program_regularization_total_loss",
            "mean_program_target_weight",
            "program_feedback_coverage",
            "program_lambda_active_fraction",
            "task_actor_gradient_norm",
            "program_gradient_norm",
            "applied_program_gradient_norm",
            "applied_program_gradient_lambda",
            "program_to_task_gradient_ratio",
            "program_gradient_conflict_fraction",
            "program_gradient_guard_saturation_fraction",
            "skill_anchor_loss",
            "skill_anchor_accuracy",
        ):
            metrics[key] /= divisor
        metrics["entropy_coef"] = effective_entropy_coef
        metrics["program_regularity_weight"] = float(
            regularization_weight
        )
        metrics["mean_effective_program_lambda"] = (
            float(regularization_weight)
            * metrics["mean_program_target_weight"]
        )
        return metrics

    def update_many(
        self,
        batches: Sequence[EpisodeBatch],
        *,
        entropy_coef: float | None = None,
        regularization_weight: float = 0.0,
        program_regularizer: PolicyProgramRegularizer | None = None,
        regularization_gradient_guard_ratio: float | None = None,
        regularization_task_gradient_floor: float = 0.0,
        project_conflicting_regularization_gradients: bool = True,
        skill_anchor_observations: np.ndarray | None = None,
        skill_anchor_labels: np.ndarray | None = None,
        skill_anchor_weight: float = 0.0,
    ) -> dict[str, float]:
        """Run one PPO update over several complete on-policy trajectories."""

        if not batches:
            raise ValueError("At least one episode batch is required.")
        merged = EpisodeBatch(
            observations=np.concatenate([item.observations for item in batches], axis=0),
            global_states=np.concatenate([item.global_states for item in batches], axis=0),
            agent_indices=np.concatenate([item.agent_indices for item in batches], axis=0),
            actions=np.concatenate([item.actions for item in batches], axis=0),
            old_log_probs=np.concatenate([item.old_log_probs for item in batches], axis=0),
            old_values=np.concatenate([item.old_values for item in batches], axis=0),
            rewards=np.concatenate([item.rewards for item in batches], axis=0),
            dones=np.concatenate([item.dones for item in batches], axis=0),
            advantages=np.concatenate([item.advantages for item in batches], axis=0),
            returns=np.concatenate([item.returns for item in batches], axis=0),
            trainable_mask=np.concatenate(
                [item.trainable_mask for item in batches],
                axis=0,
            ),
            episode_reward=float(np.mean([item.episode_reward for item in batches])),
            base_training_reward=float(
                np.mean([item.base_training_reward for item in batches])
            ),
            potential_shaping_reward=float(
                np.mean([item.potential_shaping_reward for item in batches])
            ),
            avoidable_wait_penalty_reward=float(
                np.mean(
                    [item.avoidable_wait_penalty_reward for item in batches]
                )
            ),
            mission_regression_penalty_reward=float(
                np.mean(
                    [item.mission_regression_penalty_reward for item in batches]
                )
            ),
            individual_training_rewards=mean_agent_attribute(
                batches, "individual_training_rewards"
            ),
            individual_progress_rewards=mean_agent_attribute(
                batches, "individual_progress_rewards"
            ),
            coordination_progress_reward=float(
                np.mean([item.coordination_progress_reward for item in batches])
            ),
            counterfactual_regret_units=mean_agent_attribute(
                batches, "counterfactual_regret_units"
            ),
            counterfactual_regret_penalty_rewards=mean_agent_attribute(
                batches, "counterfactual_regret_penalty_rewards"
            ),
            repeated_avoidable_wait_penalty_rewards=mean_agent_attribute(
                batches, "repeated_avoidable_wait_penalty_rewards"
            ),
            avoidable_wait_counts=sum_agent_attribute(
                batches, "avoidable_wait_counts"
            ),
            maximum_avoidable_wait_streaks=max_agent_attribute(
                batches, "maximum_avoidable_wait_streaks"
            ),
            detour_counts=sum_agent_attribute(batches, "detour_counts"),
            loaded_detour_counts=sum_agent_attribute(
                batches, "loaded_detour_counts"
            ),
            path_efficiency_actual_over_shortest_safe=float(
                np.mean(
                    [
                        item.path_efficiency_actual_over_shortest_safe
                        for item in batches
                    ]
                )
            ),
            energy_curriculum_applied=any(
                item.energy_curriculum_applied for item in batches
            ),
            initial_minimum_battery=min(
                item.initial_minimum_battery for item in batches
            ),
            episode_steps=sum(item.episode_steps for item in batches),
            pickups=sum(item.pickups for item in batches),
            deliveries=sum(item.deliveries for item in batches),
            collisions=sum(item.collisions for item in batches),
            shutdowns=sum(item.shutdowns for item in batches),
            terminal_reason=None,
            charger_uses=sum(item.charger_uses for item in batches),
            avoidable_detours=sum(
                item.avoidable_detours for item in batches
            ),
            route_regret=sum(item.route_regret for item in batches),
            minimum_battery=min(
                item.minimum_battery for item in batches
            ),
            semantic_features=tuple(
                feature for item in batches for feature in item.semantic_features
            ),
            regularization_observations=_concatenate_optional(
                [item.regularization_observations for item in batches]
            ),
            regularization_targets=_concatenate_optional(
                [item.regularization_targets for item in batches]
            ),
            regularization_weights=_concatenate_optional(
                [item.regularization_weights for item in batches]
            ),
        )
        return self.update(
            merged,
            entropy_coef=entropy_coef,
            regularization_weight=regularization_weight,
            program_regularizer=program_regularizer,
            regularization_gradient_guard_ratio=(
                regularization_gradient_guard_ratio
            ),
            regularization_task_gradient_floor=(
                regularization_task_gradient_floor
            ),
            project_conflicting_regularization_gradients=(
                project_conflicting_regularization_gradients
            ),
            skill_anchor_observations=skill_anchor_observations,
            skill_anchor_labels=skill_anchor_labels,
            skill_anchor_weight=skill_anchor_weight,
        )


def _concatenate_optional(values: Sequence[np.ndarray | None]) -> np.ndarray | None:
    present = [value for value in values if value is not None and len(value) > 0]
    return np.concatenate(present, axis=0) if present else None


def _evaluation_summary(
    *,
    training_rewards: Sequence[float],
    base_training_rewards: Sequence[float],
    potential_shaping_rewards: Sequence[float],
    user_scores: Sequence[float],
    deliveries: Sequence[int],
    steps: Sequence[int],
    collision_episodes: int,
    shutdown_episodes: int,
    collision_counts: Sequence[int],
    shutdown_counts: Sequence[int],
    charger_use_steps: int,
    detour_units: float,
    delivery_durations: Sequence[int],
    minimum_batteries: Sequence[float],
    terminal_reasons: Mapping[str, int],
    proxy_human_overrides: int = 0,
    deadlock_episodes: int = 0,
    yield_events: int = 0,
    head_on_risk_events: int = 0,
    post_policy_action_interventions: int = 0,
    avoidable_loaded_delivery_detour_steps: int = 0,
    charger_return_cycle_episodes: int = 0,
    charger_return_cycles: int = 0,
    task_starvation_episodes: int = 0,
    per_agent_progress_rewards: Mapping[str, float] | None = None,
    per_agent_counterfactual_regret_units: Mapping[str, float] | None = None,
    per_agent_counterfactual_regret_penalties: Mapping[str, float] | None = None,
    per_agent_repeated_wait_penalties: Mapping[str, float] | None = None,
    per_agent_avoidable_wait_counts: Mapping[str, int] | None = None,
    per_agent_maximum_avoidable_wait_streaks: Mapping[str, int] | None = None,
    per_agent_detour_counts: Mapping[str, int] | None = None,
    per_agent_loaded_detour_counts: Mapping[str, int] | None = None,
    coordination_progress_reward: float = 0.0,
    path_actual_steps: float = 0.0,
    path_shortest_safe_steps: float = 0.0,
) -> dict[str, Any]:
    episode_count = max(1, len(training_rewards))
    total_steps = max(1, sum(steps))
    agent_ids = ("robot_1", "robot_2")
    progress = dict(per_agent_progress_rewards or {})
    regret = dict(per_agent_counterfactual_regret_units or {})
    regret_penalties = dict(per_agent_counterfactual_regret_penalties or {})
    wait_penalties = dict(per_agent_repeated_wait_penalties or {})
    wait_counts = dict(per_agent_avoidable_wait_counts or {})
    max_wait_streaks = dict(per_agent_maximum_avoidable_wait_streaks or {})
    detours = dict(per_agent_detour_counts or {})
    loaded_detours = dict(per_agent_loaded_detour_counts or {})
    return {
        "episodes": float(len(training_rewards)),
        "mean_training_reward": float(np.mean(training_rewards)),
        "training_reward_std": float(np.std(training_rewards)),
        "mean_base_training_reward": float(np.mean(base_training_rewards)),
        "mean_potential_shaping_reward": float(
            np.mean(potential_shaping_rewards)
        ),
        "mean_user_score": float(np.mean(user_scores)),
        "user_score_std": float(np.std(user_scores)),
        "mean_deliveries": float(np.mean(deliveries)),
        "delivery_std": float(np.std(deliveries)),
        "mean_episode_steps": float(np.mean(steps)),
        "deliveries_per_100_steps": 100.0 * sum(deliveries) / total_steps,
        "collision_episode_rate": collision_episodes / episode_count,
        "shutdown_episode_rate": shutdown_episodes / episode_count,
        "mean_robot_collision_events": float(np.mean(collision_counts)),
        "maximum_robot_collision_events": int(max(collision_counts, default=0)),
        "repeated_collision_episode_rate": (
            sum(value > 1 for value in collision_counts) / episode_count
        ),
        "mean_shutdown_events": float(np.mean(shutdown_counts)),
        "mean_charger_use_steps": charger_use_steps / episode_count,
        "charger_utilization_rate": charger_use_steps / total_steps,
        "mean_human_detour_units": detour_units / episode_count,
        "mean_claim_to_delivery_steps": (
            float(np.mean(delivery_durations)) if delivery_durations else 0.0
        ),
        "mean_minimum_battery": float(np.mean(minimum_batteries)),
        "mean_proxy_human_overrides": proxy_human_overrides / episode_count,
        "delivery_episode_rate": sum(value > 0 for value in deliveries) / episode_count,
        "deadlock_episode_rate": deadlock_episodes / episode_count,
        "mean_coordination_yield_events": yield_events / episode_count,
        "mean_head_on_risk_events": head_on_risk_events / episode_count,
        "mean_post_policy_action_interventions": (
            post_policy_action_interventions / episode_count
        ),
        "avoidable_loaded_delivery_detour_steps": int(
            avoidable_loaded_delivery_detour_steps
        ),
        "avoidable_loaded_delivery_detours_per_1000_steps": (
            1000.0 * avoidable_loaded_delivery_detour_steps / total_steps
        ),
        "charger_departure_return_cycle_episode_rate": (
            charger_return_cycle_episodes / episode_count
        ),
        "mean_charger_departure_return_cycles": (
            charger_return_cycles / episode_count
        ),
        "task_starvation_episode_rate": task_starvation_episodes / episode_count,
        "mean_coordination_progress_reward": (
            coordination_progress_reward / episode_count
        ),
        "per_agent_efficiency": {
            agent_id: {
                "mean_progress_reward": progress.get(agent_id, 0.0)
                / episode_count,
                "mean_counterfactual_regret_units": regret.get(agent_id, 0.0)
                / episode_count,
                "mean_counterfactual_regret_penalty_reward": (
                    regret_penalties.get(agent_id, 0.0) / episode_count
                ),
                "mean_repeated_wait_penalty_reward": (
                    wait_penalties.get(agent_id, 0.0) / episode_count
                ),
                "avoidable_wait_count": int(wait_counts.get(agent_id, 0)),
                "avoidable_waits_per_1000_steps": (
                    1000.0 * wait_counts.get(agent_id, 0) / total_steps
                ),
                "maximum_avoidable_wait_streak": int(
                    max_wait_streaks.get(agent_id, 0)
                ),
                "detour_count": int(detours.get(agent_id, 0)),
                "loaded_detour_count": int(
                    loaded_detours.get(agent_id, 0)
                ),
            }
            for agent_id in agent_ids
        },
        "path_actual_steps": float(path_actual_steps),
        "path_shortest_safe_steps": float(path_shortest_safe_steps),
        "path_efficiency_actual_over_shortest_safe": (
            float(path_actual_steps) / max(1.0, float(path_shortest_safe_steps))
        ),
        "user_score_samples": [float(value) for value in user_scores],
        "delivery_samples": [int(value) for value in deliveries],
        "collision_event_samples": [int(value) for value in collision_counts],
        **{
            f"terminal_{reason}_rate": count / episode_count
            for reason, count in sorted(terminal_reasons.items())
        },
    }


def evaluate_policy(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
    noisy_teammate_probability: float = 0.0,
) -> dict[str, float]:
    """Evaluate deterministic robot 2 with an optional noisy robot 1 teammate."""

    if not 0.0 <= noisy_teammate_probability <= 1.0:
        raise ValueError("noisy_teammate_probability must be between zero and one.")

    training_rewards: list[float] = []
    base_training_rewards: list[float] = []
    potential_shaping_rewards: list[float] = []
    user_scores: list[float] = []
    deliveries: list[int] = []
    steps: list[int] = []
    delivery_durations: list[int] = []
    minimum_batteries: list[float] = []
    collision_episodes = 0
    shutdown_episodes = 0
    charger_use_steps = 0
    detour_units = 0.0
    terminal_reasons: dict[str, int] = {}
    collision_counts: list[int] = []
    shutdown_counts: list[int] = []
    proxy_human_overrides = 0
    deadlock_episodes = 0
    yield_events = 0
    head_on_risk_events = 0
    post_policy_action_interventions = 0
    avoidable_loaded_delivery_detour_steps = 0
    charger_return_cycle_episodes = 0
    charger_return_cycles = 0
    task_starvation_episodes = 0
    efficiency = EfficiencyMetrics()
    rng = np.random.default_rng(seed + 17_000_000)
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        observations, _ = environment.reset(seed=seed + episode)
        total_training_reward = 0.0
        total_base_training_reward = 0.0
        total_potential_shaping_reward = 0.0
        collided = False
        shutdown = False
        episode_steps = 0
        ineffective_wait_streak = 0
        deadlocked = False
        episode_return_cycles = 0
        episode_starvation = False
        minimum_battery_seen = min(
            agent.battery for agent in environment.get_state().agents
        )
        while True:
            fixed_agent_ids: tuple[str, ...] = ()
            fixed_actions: dict[str, str] = {}
            if rng.random() < noisy_teammate_probability:
                human_id = environment.config.human_agent_id
                mask = environment.action_masks()[human_id]
                if rng.random() < 0.50:
                    fixed_actions[human_id] = "WAIT"
                else:
                    legal_moves = [
                        action
                        for action, allowed in zip(ACTIONS, mask)
                        if allowed > 0.5 and action != "WAIT"
                    ]
                    fixed_actions[human_id] = (
                        str(rng.choice(legal_moves)) if legal_moves else "WAIT"
                    )
                proxy_human_overrides += 1
                fixed_agent_ids = (human_id,)
            actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
                fixed_actions=fixed_actions,
            )
            # The actor output is submitted directly.  The only intentional
            # replacement is the configured proxy-human action for robot 1;
            # robot 2 is never rewritten by a controller or program.
            state_before = environment.get_state()
            final_targets = environment._resolve_motion(state_before, actions)[0]
            for agent in state_before.agents:
                if (
                    agent.agent_id in fixed_agent_ids
                    or agent.carrying_task_id is None
                    or agent.navigation_goal_kind != "delivery"
                    or environment._requires_charge(state_before, agent)
                ):
                    continue
                current_distance = shortest_path_distance(
                    agent.position,
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                final_distance = shortest_path_distance(
                    final_targets[agent.agent_id],
                    agent.navigation_goal_position,
                    environment.config.map_layout_id,
                )
                if final_distance <= current_distance:
                    continue
                if is_necessary_urgent_charger_clearance(
                    environment,
                    state_before,
                    agent,
                ):
                    continue
                held_actions = dict(actions)
                held_actions[agent.agent_id] = "WAIT"
                if not environment._resolve_motion(
                    state_before,
                    held_actions,
                )[3]:
                    avoidable_loaded_delivery_detour_steps += 1
            observations, reward, terminated, truncated, info = environment.step(
                actions
            )
            total_training_reward += float(np.mean(tuple(reward.values())))
            total_base_training_reward += float(
                info.get("base_training_reward", 0.0)
            )
            total_potential_shaping_reward += float(
                info.get("potential_shaping_reward", 0.0)
            )
            efficiency.update_step(info)
            episode_steps += 1
            minimum_battery_seen = min(
                minimum_battery_seen,
                *(agent.battery for agent in environment.get_state().agents),
            )
            collided = collided or bool(info["collisions"])
            shutdown = shutdown or bool(info["shutdowns"])
            charger_use_steps += int(bool(info.get("charger_used", False)))
            coordination_events = tuple(info.get("coordination_events", ()))
            yield_events += sum(
                str(item.get("event", "")) == "coordination_yield"
                for item in coordination_events
                if isinstance(item, Mapping)
            )
            head_on_risk_events += sum(
                str(item.get("event", "")) == "head_on_conflict_risk"
                for item in coordination_events
                if isinstance(item, Mapping)
            )
            episode_return_cycles += sum(
                str(item.get("event", "")) == "charger_return_cycle"
                for item in info.get("energy_events", ())
                if isinstance(item, Mapping)
            )
            episode_starvation = episode_starvation or bool(
                info.get("starving_task_ids", ())
            )
            ineffective_joint_wait = bool(
                all(
                    str(value) == "WAIT"
                    for value in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            ineffective_wait_streak = (
                ineffective_wait_streak + 1 if ineffective_joint_wait else 0
            )
            deadlocked = deadlocked or ineffective_wait_streak >= 8
            if terminated or truncated:
                reason = str(info.get("terminal_reason") or "unknown")
                terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
                break
        state = environment.get_state()
        training_rewards.append(total_training_reward)
        base_training_rewards.append(total_base_training_reward)
        potential_shaping_rewards.append(total_potential_shaping_reward)
        user_scores.append(state.user_score)
        deliveries.append(state.total_deliveries)
        steps.append(episode_steps)
        collision_episodes += int(collided)
        shutdown_episodes += int(shutdown)
        deadlock_episodes += int(deadlocked)
        charger_return_cycle_episodes += int(episode_return_cycles > 0)
        charger_return_cycles += episode_return_cycles
        task_starvation_episodes += int(episode_starvation)
        collision_counts.append(state.robot_collision_events)
        shutdown_counts.append(state.shutdown_count)
        detour_units += state.human_route_regret_units
        minimum_batteries.append(minimum_battery_seen)
        delivery_durations.extend(
            task.delivered_frame - task.claimed_frame
            for task in state.completed_tasks
            if task.delivered_frame is not None and task.claimed_frame is not None
        )
        efficiency.update_completed_tasks(state)
    return _evaluation_summary(
        training_rewards=training_rewards,
        base_training_rewards=base_training_rewards,
        potential_shaping_rewards=potential_shaping_rewards,
        user_scores=user_scores,
        deliveries=deliveries,
        steps=steps,
        collision_episodes=collision_episodes,
        shutdown_episodes=shutdown_episodes,
        collision_counts=collision_counts,
        shutdown_counts=shutdown_counts,
        charger_use_steps=charger_use_steps,
        detour_units=detour_units,
        delivery_durations=delivery_durations,
        minimum_batteries=minimum_batteries,
        terminal_reasons=terminal_reasons,
        proxy_human_overrides=proxy_human_overrides,
        deadlock_episodes=deadlock_episodes,
        yield_events=yield_events,
        head_on_risk_events=head_on_risk_events,
        post_policy_action_interventions=post_policy_action_interventions,
        avoidable_loaded_delivery_detour_steps=(
            avoidable_loaded_delivery_detour_steps
        ),
        charger_return_cycle_episodes=charger_return_cycle_episodes,
        charger_return_cycles=charger_return_cycles,
        task_starvation_episodes=task_starvation_episodes,
        **efficiency.evaluation_kwargs(),
    )


def evaluate_random_policy(
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate uniformly sampled legal joint actions as a reference."""

    rng = np.random.default_rng(seed)
    training_rewards: list[float] = []
    base_training_rewards: list[float] = []
    potential_shaping_rewards: list[float] = []
    user_scores: list[float] = []
    deliveries: list[int] = []
    steps: list[int] = []
    delivery_durations: list[int] = []
    minimum_batteries: list[float] = []
    collision_episodes = 0
    shutdown_episodes = 0
    charger_use_steps = 0
    detour_units = 0.0
    terminal_reasons: dict[str, int] = {}
    collision_counts: list[int] = []
    shutdown_counts: list[int] = []
    deadlock_episodes = 0
    yield_events = 0
    head_on_risk_events = 0
    charger_return_cycle_episodes = 0
    charger_return_cycles = 0
    task_starvation_episodes = 0
    efficiency = EfficiencyMetrics()
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        total_training_reward = 0.0
        total_base_training_reward = 0.0
        total_potential_shaping_reward = 0.0
        collided = False
        shutdown = False
        episode_steps = 0
        ineffective_wait_streak = 0
        deadlocked = False
        episode_return_cycles = 0
        episode_starvation = False
        minimum_battery_seen = min(
            agent.battery for agent in environment.get_state().agents
        )
        while True:
            actions: dict[str, str] = {}
            for agent_id, mask in environment.action_masks().items():
                legal = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 0.5)
                actions[agent_id] = ACTIONS[int(rng.choice(legal))]
            _, reward, terminated, truncated, info = environment.step(actions)
            total_training_reward += float(np.mean(tuple(reward.values())))
            total_base_training_reward += float(
                info.get("base_training_reward", 0.0)
            )
            total_potential_shaping_reward += float(
                info.get("potential_shaping_reward", 0.0)
            )
            efficiency.update_step(info)
            episode_steps += 1
            minimum_battery_seen = min(
                minimum_battery_seen,
                *(agent.battery for agent in environment.get_state().agents),
            )
            collided = collided or bool(info["collisions"])
            shutdown = shutdown or bool(info["shutdowns"])
            charger_use_steps += int(bool(info.get("charger_used", False)))
            coordination_events = tuple(info.get("coordination_events", ()))
            yield_events += sum(
                str(item.get("event", "")) == "coordination_yield"
                for item in coordination_events
                if isinstance(item, Mapping)
            )
            head_on_risk_events += sum(
                str(item.get("event", "")) == "head_on_conflict_risk"
                for item in coordination_events
                if isinstance(item, Mapping)
            )
            episode_return_cycles += sum(
                str(item.get("event", "")) == "charger_return_cycle"
                for item in info.get("energy_events", ())
                if isinstance(item, Mapping)
            )
            episode_starvation = episode_starvation or bool(
                info.get("starving_task_ids", ())
            )
            ineffective_joint_wait = bool(
                all(
                    str(value) == "WAIT"
                    for value in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            ineffective_wait_streak = (
                ineffective_wait_streak + 1 if ineffective_joint_wait else 0
            )
            deadlocked = deadlocked or ineffective_wait_streak >= 8
            if terminated or truncated:
                reason = str(info.get("terminal_reason") or "unknown")
                terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
                break
        state = environment.get_state()
        training_rewards.append(total_training_reward)
        base_training_rewards.append(total_base_training_reward)
        potential_shaping_rewards.append(total_potential_shaping_reward)
        user_scores.append(state.user_score)
        deliveries.append(state.total_deliveries)
        steps.append(episode_steps)
        collision_episodes += int(collided)
        shutdown_episodes += int(shutdown)
        deadlock_episodes += int(deadlocked)
        charger_return_cycle_episodes += int(episode_return_cycles > 0)
        charger_return_cycles += episode_return_cycles
        task_starvation_episodes += int(episode_starvation)
        collision_counts.append(state.robot_collision_events)
        shutdown_counts.append(state.shutdown_count)
        detour_units += state.human_route_regret_units
        minimum_batteries.append(minimum_battery_seen)
        delivery_durations.extend(
            task.delivered_frame - task.claimed_frame
            for task in state.completed_tasks
            if task.delivered_frame is not None and task.claimed_frame is not None
        )
        efficiency.update_completed_tasks(state)
    return _evaluation_summary(
        training_rewards=training_rewards,
        base_training_rewards=base_training_rewards,
        potential_shaping_rewards=potential_shaping_rewards,
        user_scores=user_scores,
        deliveries=deliveries,
        steps=steps,
        collision_episodes=collision_episodes,
        shutdown_episodes=shutdown_episodes,
        collision_counts=collision_counts,
        shutdown_counts=shutdown_counts,
        charger_use_steps=charger_use_steps,
        detour_units=detour_units,
        delivery_durations=delivery_durations,
        minimum_batteries=minimum_batteries,
        terminal_reasons=terminal_reasons,
        deadlock_episodes=deadlock_episodes,
        yield_events=yield_events,
        head_on_risk_events=head_on_risk_events,
        charger_return_cycle_episodes=charger_return_cycle_episodes,
        charger_return_cycles=charger_return_cycles,
        task_starvation_episodes=task_starvation_episodes,
        **efficiency.evaluation_kwargs(),
    )


def evaluate_head_on_yield_scenarios(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate whether a head-on encounter is safely cleared by right-of-way.

    The scenario ends successfully once the priority robot enters the other
    robot's initial corridor cell and the yielding robot has vacated it.  A
    later delivery is deliberately not required: after the robots have passed,
    their independent mission routes can meet again and would measure routing,
    not whether the original right-of-way decision succeeded.
    """

    successes = 0
    collisions = 0
    deadlocks = 0
    right_of_way_yields = 0
    for episode in range(episodes):
        environment = WarehouseMultiAgentEnv(environment_config)
        environment.reset(seed=seed + episode)
        reverse = bool(episode % 2)
        apply_head_on_scenario(environment, reverse=reverse)
        scenario_state = environment.get_state()
        robot_one = scenario_state.by_id("robot_1")
        robot_two = scenario_state.by_id("robot_2")
        scenario_agents = tuple(
            scenario_state.by_id(agent_id)
            for agent_id in environment.agent_ids
        )
        priority_agent = min(
            scenario_agents,
            key=lambda agent: (
                -int(agent.carrying_task_id is not None),
                shortest_path_distance(
                    agent.position,
                    agent.navigation_goal_position,
                    environment_config.map_layout_id,
                ),
                agent.agent_id,
            ),
        )
        yielding_agent_id = (
            robot_two.agent_id
            if priority_agent.agent_id == robot_one.agent_id
            else robot_one.agent_id
        )
        initial_yielding_position = environment.get_state().by_id(
            yielding_agent_id
        ).position
        wait_streak = 0
        yielded = False
        collided = False
        passage_completed = False
        observations = environment.observations()
        for _ in range(min(32, environment_config.horizon)):
            actions, _ = policy.act(
                observations,
                environment.global_state(),
                deterministic=True,
            )
            observations, _, terminated, truncated, info = environment.step(actions)
            collided = collided or bool(info.get("robot_collision_event", False))
            coordination = tuple(info.get("coordination_events", ()))
            for item in coordination:
                if not isinstance(item, Mapping):
                    continue
                if (
                    str(item.get("event", "")) == "coordination_yield"
                    and str(item.get("yielding_agent_id", "")) == yielding_agent_id
                ):
                    yielded = True
                    right_of_way_yields += 1
            ineffective_wait = bool(
                all(
                    str(value) == "WAIT"
                    for value in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            wait_streak = wait_streak + 1 if ineffective_wait else 0
            current_state = environment.get_state()
            passage_completed = (
                current_state.by_id(priority_agent.agent_id).position
                == initial_yielding_position
                and current_state.by_id(yielding_agent_id).position
                != initial_yielding_position
            )
            if (
                passage_completed
                or collided
                or wait_streak >= 8
                or terminated
                or truncated
            ):
                break
        success = passage_completed and yielded and not collided and wait_streak < 8
        successes += int(success)
        collisions += int(collided)
        deadlocks += int(wait_streak >= 8)
    count = max(1, episodes)
    return {
        "episodes": float(episodes),
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "deadlock_rate": deadlocks / count,
        "mean_right_of_way_yields": right_of_way_yields / count,
    }
