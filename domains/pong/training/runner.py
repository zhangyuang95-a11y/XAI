"""PPO/RCPD training loop for Cooperative Pong v2–v2.2.

The environment is authoritative. Program extraction labels come only from the
current frozen Actor; programs never choose an in-game action.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import signal
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.distributions import Categorical
import yaml

from core.rcpd import RCPD
from core.rcpd_config import OracleOutput, RCPDConfig
from domains.pong.adapters.core_adapter import (
    feature_mapping_from_vector, feature_names, feature_vector, observation_signature,
)
from domains.pong.config import PongConfig
from domains.pong.environment.engine import PongEnvironment
from .curriculum import PongCurriculum
from .corrections import CorrectionCollector
from .networks import PongActorCritic
from .partners import ACTIONS, PartnerManager
from .reward import RewardLedger
from ..policies.hybrid import VERSION as HYBRID_RULE_VERSION


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"device '{name}' is unavailable; choose cpu, mps, or cuda")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _signature(config: PongConfig, names: tuple[str, ...]) -> dict[str, Any]:
    environment = PongEnvironment(config)
    return observation_signature(environment) | {"actions": list(ACTIONS), "feature_names": list(names)}


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """Configuration identity excluding an allowed maximum-step extension."""
    values = json.loads(json.dumps(config))
    values.get("training", {}).pop("max_joint_steps", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _actor_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    relative = ("domains/pong/training/runner.py", "domains/pong/training/corrections.py",
                "domains/pong/training/curriculum.py", "domains/pong/training/evaluation.py",
                "domains/pong/training/partners.py", "domains/pong/training/reward.py",
                "domains/pong/training/networks.py", "domains/pong/training/browser_compat.py",
                "domains/pong/config.py", "domains/pong/environment/engine.py",
                "domains/pong/environment/model.py", "domains/pong/adapters/core_adapter.py",
                "domains/pong/policies/hybrid.py", "domains/pong/policies/rule_demo.py",
                "domains/pong/web/app.js",
                "scripts/train_pong_nn.py", "scripts/evaluate_pong_nn.py", "scripts/export_pong_nn.py",
                "core/rcpd.py", "core/rcpd_config.py", "core/rcpd_tree.py", "core/program.py",
                "core/policy_program_regularizer.py")
    return {name: _checkpoint_hash(root / name) for name in relative}


def _environment_content_signature(environment: PongEnvironment) -> str:
    frame = environment.frame()
    body = {
        "paddles": [round(frame.player_x, 6), round(frame.ai_x, 6)],
        "balls": [(ball.ball_id, ball.active, round(ball.x, 6), round(ball.y, 6),
                   round(ball.vx, 6), round(ball.vy, 6)) for ball in frame.balls],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class PongPPOTrainer:
    """Vectorized PPO with episode-stable partners and curriculum state."""

    def __init__(self, config: Mapping[str, Any], *, device: torch.device) -> None:
        self.config = json.loads(json.dumps(config))
        self.device = device
        environment_values = dict(self.config.get("environment", {}))
        environment_values.pop("decision_interval_frames", None)
        environment_values.pop("parallel_environments", None)
        environment_values.pop("episode_seconds", None)
        environment_values["duration_seconds"] = float(self.config["environment"].get("episode_seconds", 90))
        environment_values["control_mode"] = "frozen_nn"
        self.env_config = PongConfig.from_mapping(environment_values)
        self.interval = int(self.config["environment"].get("decision_interval_frames", 6))
        count = int(self.config["environment"].get("parallel_environments", 16))
        seed = int(self.config["seed"])
        self.envs = [PongEnvironment(self.env_config, seed=seed + index) for index in range(count)]
        self.names = feature_names(self.envs[0])
        network = self.config["network"]
        self.model = PongActorCritic(len(self.names), len(ACTIONS), int(network.get("hidden_size", 128))).to(device)
        ppo = self.config["ppo"]
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=float(ppo["actor_learning_rate"]))
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=float(ppo["critic_learning_rate"]))
        rcpd_values = dict(self.config.get("rcpd", {}))
        allowed = set(RCPDConfig.__dataclass_fields__)
        self.rcpd = RCPD(RCPDConfig(**{key: value for key, value in rcpd_values.items() if key in allowed}))
        training = self.config["training"]
        curriculum_values = dict(self.config.get("curriculum", {}))
        curriculum_values.setdefault("max_joint_steps", int(training["max_joint_steps"]))
        self.curriculum = PongCurriculum(self.env_config, curriculum_values, seed=seed + 101)
        self.partners = PartnerManager(self.env_config, training, count, seed=seed + 211, model=self.model)
        self.corrections = (CorrectionCollector(self.env_config,
                            {**training, **self.config.get("hybrid", {})}, interval=self.interval, seed=seed + 100_000)
                            if bool(self.config.get("hybrid", {}).get("enabled", False)) else None)
        timing = str(self.config.get("reward", {}).get("timing", "score_at_bottom"))
        self.reward_ledgers = [RewardLedger(timing) for _ in self.envs]
        self.joint_steps = 0
        self.neural_action_samples = 0
        self.actor_updates = 0
        self.critic_updates = 0
        self.feedback_updates = 0
        self.episode_ids = [index for index in range(count)]
        self.episode_metadata: list[dict[str, str]] = [{} for _ in self.envs]
        self.episode_decisions = [0 for _ in self.envs]
        self.short_episode_count = 0
        # Keep observations by *complete* environment episode for extraction.
        # A 90-second game spans several PPO rollouts, so using only the
        # current rollout would make an episode fragment look like a trajectory.
        self.active_trajectory_observations: list[list[np.ndarray]] = [[] for _ in self.envs]
        self.completed_trajectories: list[dict[str, Any]] = []
        self.trajectory_cache_size = int(training.get("extraction_trajectory_cache", 8))
        self.stage_counts: dict[str, int] = {"foundation": 0, "coordination": 0, "full": 0}
        self.feedback_gate_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.best_selection: dict[str, Any] | None = None
        self.best_pure_selection: dict[str, Any] | None = None
        self.best_hybrid_selection: dict[str, Any] | None = None
        self.parent_provenance: dict[str, Any] | None = None
        self.training_initial_signatures: set[str] = set()
        self.last_evaluation_step = 0
        self.feedback_started_at: int | None = None
        self._stop = False
        for index in range(count):
            self._reset_episode(index, initial=True)

    def request_stop(self, *_args: Any) -> None:
        self._stop = True

    def _progress(self) -> float:
        return min(1.0, self.joint_steps / max(1, int(self.config["training"]["max_joint_steps"])))

    def _stage(self) -> str:
        return self.curriculum.stage_for_steps(self.joint_steps)

    def _reset_episode(self, index: int, *, initial: bool = False) -> None:
        episode = self.curriculum.sample(self.joint_steps)
        seed = int(self.config["seed"]) + self.episode_ids[index]
        self.envs[index].reset_training_episode(
            seed=seed, scenario_name=episode.scenario, active_ball_ids=set(episode.active_ball_ids),
        )
        signature = _environment_content_signature(self.envs[index])
        self.training_initial_signatures.add(signature)
        self.episode_metadata[index] = {"stage": episode.stage, "scenario": episode.scenario}
        self.episode_decisions[index] = 0
        self.stage_counts[episode.stage] += 1
        if not initial:
            self.partners.reset_episode(index)
            self.reward_ledgers[index] = RewardLedger(str(self.config.get("reward", {}).get("timing", "score_at_bottom")))

    def _complete_trajectory(self, index: int, episode_id: int, metadata: Mapping[str, str]) -> None:
        """Freeze one full episode before its environment is reset."""
        rows = self.active_trajectory_observations[index]
        if rows:
            self.completed_trajectories.append({
                "episode_id": int(episode_id), "stage": str(metadata["stage"]),
                "scenario": str(metadata["scenario"]), "observations": np.stack(rows).astype(np.float32),
            })
            self.completed_trajectories = self.completed_trajectories[-self.trajectory_cache_size:]
        self.active_trajectory_observations[index] = []

    def _obs(self) -> np.ndarray:
        rows: list[np.ndarray] = []
        for environment in self.envs:
            rows.extend((feature_vector(environment, "player"), feature_vector(environment, "ai")))
        return np.stack(rows)

    def _joint(self, observations: np.ndarray) -> np.ndarray:
        return observations.reshape(len(self.envs), 2, -1).reshape(len(self.envs), -1)

    def collect(self, horizon: int | None = None) -> dict[str, Any]:
        horizon = int(horizon or self.config["ppo"]["rollout_decisions"])
        keys = ("obs", "joint", "actions", "old_logp", "values", "rewards", "dones", "boundaries", "next_values", "neural", "features",
                "episodes", "stages", "scenarios", "reward_components")
        buffer: dict[str, list[Any]] = {key: [] for key in keys}
        for _ in range(horizon):
            observations = self._obs()
            joint_observations = self._joint(observations)
            features = [feature_mapping_from_vector(self.names, row) for row in observations]
            pre_episode_ids = list(self.episode_ids)
            pre_metadata = [dict(item) for item in self.episode_metadata]
            for index in range(len(self.envs)):
                self.active_trajectory_observations[index].extend((
                    observations[2 * index].copy(), observations[2 * index + 1].copy(),
                ))
            with torch.no_grad():
                tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                logits = self.model.actor_logits(tensor)
                distribution = Categorical(logits=logits)
                sampled = distribution.sample()
                old_logp = distribution.log_prob(sampled)
                values = self.model.values(torch.as_tensor(joint_observations, dtype=torch.float32, device=self.device))
            actions = sampled.cpu().numpy()
            neural = np.ones(len(actions), dtype=bool)
            rewards = np.zeros(len(actions), dtype=np.float32)
            dones = np.zeros(len(actions), dtype=bool)
            boundaries = np.zeros(len(actions), dtype=bool)
            components = {"small_miss": 0.0, "large_miss": 0.0, "total": 0.0}
            finished: list[int] = []
            for index, environment in enumerate(self.envs):
                player, ai, player_neural, ai_neural = self.partners.action_pair(
                    environment, index, int(actions[2 * index]), int(actions[2 * index + 1]),
                )
                actions[2 * index], actions[2 * index + 1] = player, ai
                neural[2 * index], neural[2 * index + 1] = player_neural, ai_neural
                total_reward = 0.0
                for _physics in range(self.interval):
                    transition = environment.step(ACTIONS[player], ACTIONS[ai])
                    reward, reward_components = self.reward_ledgers[index].reward(environment, transition)
                    total_reward += reward
                    for key, value in reward_components.items():
                        components[key] += value
                    if environment.terminal:
                        break
                rewards[2 * index:2 * index + 2] = total_reward
                dones[2 * index:2 * index + 2] = environment.terminal
                self.episode_decisions[index] += 1
                short_limit = round(float(self.config["training"].get("short_episode_seconds", 0)) /
                                    (self.interval * self.env_config.fixed_dt))
                safe_to_truncate = not any(ball.active and ball.pending_miss for ball in environment.balls)
                truncated = (bool(self.config.get("curriculum", {}).get("v22"))
                             and pre_metadata[index]["scenario"] != "full"
                             and self.episode_decisions[index] >= short_limit > 0
                             and safe_to_truncate and not environment.terminal)
                boundaries[2 * index:2 * index + 2] = environment.terminal or truncated
                self.curriculum.note_steps(pre_metadata[index]["stage"], pre_metadata[index]["scenario"])
                if environment.terminal or truncated:
                    finished.append(index)
                    self.short_episode_count += int(truncated)
            with torch.no_grad():
                post_values = self.model.values(torch.as_tensor(self._joint(self._obs()), dtype=torch.float32,
                                                                 device=self.device)).cpu().numpy()
            buffer["obs"].append(observations)
            buffer["joint"].append(joint_observations)
            buffer["actions"].append(actions)
            buffer["old_logp"].append(old_logp.cpu().numpy())
            buffer["values"].append(np.repeat(values.cpu().numpy(), 2))
            buffer["rewards"].append(rewards)
            buffer["dones"].append(dones)
            buffer["boundaries"].append(boundaries)
            buffer["next_values"].append(np.repeat(post_values, 2))
            buffer["neural"].append(neural)
            buffer["features"].extend(features)
            buffer["episodes"].extend([pre_episode_ids[index // 2] for index in range(len(observations))])
            buffer["stages"].extend([pre_metadata[index // 2]["stage"] for index in range(len(observations))])
            buffer["scenarios"].extend([pre_metadata[index // 2]["scenario"] for index in range(len(observations))])
            buffer["reward_components"].append(components)
            self.joint_steps += len(self.envs)
            self.neural_action_samples += int(neural.sum())
            for index in finished:
                self._complete_trajectory(index, pre_episode_ids[index], pre_metadata[index])
                self.episode_ids[index] += len(self.envs)
                self._reset_episode(index)
            if self._stop:
                break
        for key in ("obs", "joint", "actions", "old_logp", "values", "rewards", "dones", "boundaries", "next_values", "neural"):
            buffer[key] = np.asarray(buffer[key])
        return buffer

    def _advantages(self, batch: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rewards, values, dones = batch["rewards"], batch["values"], batch["dones"]
        steps, rows = rewards.shape
        if "next_values" in batch:
            next_values = np.asarray(batch["next_values"], dtype=np.float32)
            boundaries = np.asarray(batch["boundaries"], dtype=bool)
        else:
            with torch.no_grad():
                last_values = self.model.values(
                    torch.as_tensor(self._joint(self._obs()), dtype=torch.float32, device=self.device),
                ).cpu().numpy()
            last_values = np.repeat(last_values, 2)
            next_values = np.concatenate((values[1:], last_values[None, :]), axis=0)
            boundaries = dones
        advantages = np.zeros_like(rewards)
        gae = np.zeros(rows, dtype=np.float32)
        gamma = float(self.config["ppo"]["gamma"])
        lam = float(self.config["ppo"]["gae_lambda"])
        for step in reversed(range(steps)):
            bootstrap_mask = 1.0 - dones[step].astype(np.float32)
            continuation_mask = 1.0 - boundaries[step].astype(np.float32)
            delta = rewards[step] + gamma * next_values[step] * bootstrap_mask - values[step]
            gae = delta + gamma * lam * continuation_mask * gae
            advantages[step] = gae
        return advantages, advantages + values

    def _feedback_gate(self) -> tuple[bool, str]:
        requirements = self.config.get("feedback_gate", {})
        minimum_steps = int(requirements.get("minimum_joint_steps", 125_000))
        if getattr(self.curriculum, "v21", False) and self.joint_steps < minimum_steps:
            return False, "minimum_pure_ppo_steps_not_reached"
        if self._stage() != "full":
            return False, "not_in_full_curriculum_stage"
        required = int(requirements.get("consecutive_validations", 3))
        if len(self.validation_history) < required:
            return False, "insufficient_full_validation_history"
        recent = self.validation_history[-required:]
        reduction = float(requirements.get("idle_baseline_reduction", .10))
        large_success = float(requirements.get("minimum_large_success_rate", .50))
        for report in recent:
            if report.get("stage") != "full":
                return False, "validation_precedes_full_stage"
            nn_nn = report.get("conditions", {}).get("nn_nn", {})
            idle = report.get("conditions", {}).get("idle_idle", {})
            nn_misses = float(nn_nn.get("mean_weighted_misses", float("inf")))
            idle_misses = float(idle.get("mean_weighted_misses", 0.0))
            actual_reduction = ((idle_misses - nn_misses) / idle_misses) if idle_misses > 0 else -float("inf")
            if actual_reduction < reduction:
                return False, "idle_baseline_reduction_not_met"
            direct_large = nn_nn.get("aggregate_large_success_rate")
            if direct_large is None or float(direct_large) < large_success:
                return False, "large_ball_success_not_met"
            if not bool(report.get("action_chain_valid", False)):
                return False, "action_chain_not_valid"
        if self.feedback_started_at is None:
            self.feedback_started_at = self.joint_steps
        return True, "passed"

    def _feedback_weight(self) -> float:
        if self.feedback_started_at is None or self.rcpd.regularization_weight <= 0.0:
            return 0.0
        values = self.config.get("feedback_gate", {})
        initial = float(values.get("initial_feedback_lambda", .002))
        maximum = min(float(values.get("maximum_feedback_lambda", .005)), self.rcpd.regularization_weight)
        ramp_steps = max(1, int(values.get("feedback_ramp_joint_steps", 25_000)))
        fraction = min(1.0, max(0.0, (self.joint_steps - self.feedback_started_at) / ramp_steps))
        return initial + (maximum - initial) * fraction

    def _program_slice_quality(self, program: Any, states: list[dict[str, float]], oracle: Any) -> dict[str, Any]:
        """Report program fidelity on decision situations that matter in Pong."""
        buckets: dict[str, list[dict[str, float]]] = {"large_near_contact": [], "large_hold": [],
                                                      "large_partner_fill": [], "dual_ball_conflict": [],
                                                      "small_large_tradeoff": []}
        large_ids = tuple(self.env_config.large_ball_ids)
        ball_ids = tuple(self.env_config.ball_ids)
        for state in states:
            near_large = any(
                state[f"ball.{ball_id}.opportunity_valid"] > .5
                and state[f"ball.{ball_id}.time_to_contact"] <= .15
                for ball_id in large_ids
            )
            if near_large:
                buckets["large_near_contact"].append(state)
            if any(
                state[f"ball.{ball_id}.large"] > .5 and state[f"ball.{ball_id}.opportunity_valid"] > .5
                and state[f"ball.{ball_id}.self_covers"] > .5
                for ball_id in large_ids
            ):
                buckets["large_hold"].append(state)
            if any(state[f"ball.{ball_id}.opportunity_valid"] > .5 and
                   state[f"ball.{ball_id}.other_covers"] > .5 and
                   state[f"ball.{ball_id}.self_covers"] < .5 for ball_id in large_ids):
                buckets["large_partner_fill"].append(state)
            if sum(state[f"ball.{ball_id}.opportunity_valid"] > .5 for ball_id in ball_ids) >= 2:
                buckets["dual_ball_conflict"].append(state)
            if (any(state[f"ball.{ball_id}.opportunity_valid"] > .5 for ball_id in large_ids)
                    and any(state[f"ball.{ball_id}.opportunity_valid"] > .5 for ball_id in self.env_config.small_ball_ids)):
                buckets["small_large_tradeoff"].append(state)
        quality: dict[str, Any] = {}
        for name, rows in buckets.items():
            if not rows:
                quality[name] = {"samples": 0, "argmax_fidelity": None, "mean_kl": None}
                continue
            agreements: list[float] = []
            kls: list[float] = []
            for state in rows:
                actor = oracle(state).normalized(ACTIONS)
                target = np.asarray([program.predict_proba(state)[action] for action in ACTIONS], dtype=np.float64)
                target = np.clip(target / max(1e-12, float(target.sum())), 1e-12, 1.0)
                actor = np.clip(actor, 1e-12, 1.0)
                agreements.append(float(np.argmax(actor) == np.argmax(target)))
                kls.append(float(np.sum(actor * np.log(actor / target))))
            quality[name] = {"samples": len(rows), "argmax_fidelity": float(np.mean(agreements)),
                             "mean_kl": float(np.mean(kls))}
        return quality

    def _maybe_extract(self) -> dict[str, Any] | None:
        gate, reason = self._feedback_gate()
        self.feedback_gate_history.append({"joint_steps": self.joint_steps, "enabled": gate, "reason": reason})
        if not gate:
            self.rcpd.regularization_weight = 0.0
            return {"feedback_gate": reason, "feedback_enabled": False}
        grouped = {
            int(item["episode_id"]): np.asarray(item["observations"], dtype=np.float32)
            for item in self.completed_trajectories if item["stage"] == "full"
        }
        if not grouped:
            self.rcpd.regularization_weight = 0.0
            return {"feedback_gate": "no_complete_full_trajectory", "feedback_enabled": False}
        train: list[dict[str, float]] = []
        validation: list[dict[str, float]] = []
        for episode, rows in grouped.items():
            target = validation if episode % 5 == 0 else train
            target.extend(feature_mapping_from_vector(self.names, row) for row in rows)
        if not validation:
            self.rcpd.regularization_weight = 0.0
            return {"feedback_gate": "no_complete_validation_trajectory", "feedback_enabled": False}

        def oracle(state: Mapping[str, float]) -> OracleOutput:
            vector = np.asarray([[state[name] for name in self.names]], dtype=np.float32)
            with torch.no_grad():
                probabilities = torch.softmax(
                    self.model.actor_logits(torch.as_tensor(vector, dtype=torch.float32, device=self.device)), -1,
                )[0].cpu().numpy()
            return OracleOutput(dict(zip(ACTIONS, map(float, probabilities))))

        result = self.rcpd.maybe_extract(
            self.joint_steps, train, oracle, lambda state: state, validation_states=validation,
            program_metadata={"domain": "pong", "controller": "frozen_nn", "trajectory_split": "episode_modulo_5",
                              "actor_parameters_sha256": _actor_hash(self.model)},
        )
        if result is None:
            return {"feedback_gate": reason, "feedback_enabled": bool(self.rcpd.regularization_weight),
                    "reason": self.rcpd.last_error or "not_due"}
        program_hash = hashlib.sha256(
            json.dumps(result.program.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
        return {"feedback_gate": reason, "feedback_enabled": bool(self.rcpd.regularization_weight),
                "program_sha256": program_hash,
                "actor_parameters_sha256": _actor_hash(self.model),
                "trajectory_count": len(grouped), "trajectory_split": "complete_episode_id_modulo_5",
                "slice_quality": self._program_slice_quality(result.program, validation, oracle),
                **result.metrics.to_dict()}

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        advantages, returns = self._advantages(batch)
        steps, rows = batch["actions"].shape
        total = steps * rows
        flat = {key: np.asarray(value).reshape((total,) + np.asarray(value).shape[2:]) for key, value in batch.items()
                if key in {"obs", "actions", "old_logp", "values", "neural"}}
        flat_joint = np.asarray(batch["joint"]).reshape(steps * (rows // 2), -1)
        flat["advantages"], flat["returns"] = advantages.reshape(total), returns.reshape(total)
        valid = np.flatnonzero(flat["neural"])
        if not len(valid):
            raise RuntimeError("rollout contains no neural actions")
        normalized = flat["advantages"].copy()
        normalized[valid] = (normalized[valid] - normalized[valid].mean()) / (normalized[valid].std() + 1e-8)
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ppo_kl": 0.0,
                   "clip_fraction": 0.0, "feedback_kl": 0.0, "feedback_loss": 0.0,
                   "correction_loss": 0.0, "weighted_correction_loss": 0.0,
                   "correction_gradient_norm": 0.0}
        first_gradient_norms = {"ppo_gradient_norm_first_minibatch": 0.0,
                                "feedback_gradient_norm_first_minibatch": 0.0}
        updates = 0
        ppo = self.config["ppo"]
        size = int(ppo["minibatch_size"])
        for _epoch in range(int(ppo["update_epochs"])):
            order = np.random.permutation(valid)  # one full permutation each PPO epoch
            for start in range(0, len(order), size):
                index = order[start:start + size]
                observation = torch.as_tensor(flat["obs"][index], dtype=torch.float32, device=self.device)
                joint = torch.as_tensor(flat_joint[index // 2], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(flat["actions"][index], device=self.device)
                old_logp = torch.as_tensor(flat["old_logp"][index], dtype=torch.float32, device=self.device)
                advantage = torch.as_tensor(normalized[index], dtype=torch.float32, device=self.device)
                returns_t = torch.as_tensor(flat["returns"][index], dtype=torch.float32, device=self.device)
                logits = self.model.actor_logits(observation)
                distribution = Categorical(logits=logits)
                log_ratio = distribution.log_prob(actions) - old_logp
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - float(ppo["clip"]), 1 + float(ppo["clip"]))
                actor_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                correction_loss = torch.zeros((), device=self.device)
                weighted_correction = torch.zeros((), device=self.device)
                correction_gradient_norm = 0.0
                beta = self._correction_weight()
                correction_batch = (self.corrections.training_batch(size, current_ppo_joint_steps=self.joint_steps)
                                    if self.corrections is not None and beta > 0 else None)
                if correction_batch is not None:
                    correction_obs, correction_actions = correction_batch
                    correction_logits = self.model.actor_logits(torch.as_tensor(correction_obs, dtype=torch.float32, device=self.device))
                    correction_loss = torch.nn.functional.cross_entropy(
                        correction_logits, torch.as_tensor(correction_actions, dtype=torch.int64, device=self.device))
                    weighted_correction = beta * correction_loss
                    gradients = torch.autograd.grad(weighted_correction, tuple(self.model.actor.parameters()),
                                                    retain_graph=True)
                    correction_gradient_norm = float(torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in gradients])))
                    self.corrections.supervised_updates += 1
                raw_feedback = torch.zeros((), device=self.device)
                weighted_feedback = torch.zeros((), device=self.device)
                if self.rcpd.program is not None and self.rcpd.regularization_weight > 0:
                    feature_rows = [feature_mapping_from_vector(self.names, row) for row in flat["obs"][index]]
                    actor_probabilities = torch.softmax(logits.detach(), -1).cpu().numpy()
                    targets = self.rcpd.program_feedback_targets(feature_rows, actor_probabilities=actor_probabilities)
                    weights = self.rcpd.program_target_weights(feature_rows, actor_probabilities=actor_probabilities)
                    weight = self._feedback_weight()
                    if targets is not None and weights is not None and weight > 0:
                        raw_feedback = RCPD.regularization_loss(logits, targets, weights)
                        weighted_feedback = weight * raw_feedback
                        self.feedback_updates += 1
                if updates == 0:
                    parameters = tuple(self.model.actor.parameters())
                    ppo_gradients = torch.autograd.grad(actor_loss, parameters, retain_graph=True)
                    first_gradient_norms["ppo_gradient_norm_first_minibatch"] = float(torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in ppo_gradients])))
                    if weighted_feedback.requires_grad:
                        feedback_gradients = torch.autograd.grad(weighted_feedback, parameters, retain_graph=True)
                        first_gradient_norms["feedback_gradient_norm_first_minibatch"] = float(torch.linalg.vector_norm(
                            torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in feedback_gradients])))
                self.actor_optimizer.zero_grad()
                (actor_loss + weighted_correction - float(ppo["entropy_coefficient"]) * distribution.entropy().mean()
                 + weighted_feedback).backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), float(ppo["max_grad_norm"]))
                self.actor_optimizer.step()
                values = self.model.values(joint)
                critic_loss = .5 * (values - returns_t).square().mean()
                self.critic_optimizer.zero_grad()
                (float(ppo["value_coefficient"]) * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), float(ppo["max_grad_norm"]))
                self.critic_optimizer.step()
                self.actor_updates += 1
                self.critic_updates += 1
                metrics["actor_loss"] += float(actor_loss.detach())
                metrics["critic_loss"] += float(critic_loss.detach())
                metrics["entropy"] += float(distribution.entropy().mean().detach())
                metrics["ppo_kl"] += float((-log_ratio).mean().detach())
                metrics["clip_fraction"] += float((torch.abs(ratio - 1) > float(ppo["clip"])).float().mean().detach())
                metrics["feedback_kl"] += float(raw_feedback.detach())
                metrics["feedback_loss"] += float(weighted_feedback.detach())
                metrics["correction_loss"] += float(correction_loss.detach())
                metrics["weighted_correction_loss"] += float(weighted_correction.detach())
                metrics["correction_gradient_norm"] += correction_gradient_norm
                updates += 1
        return {**{key: value / max(1, updates) for key, value in metrics.items()}, **first_gradient_norms}

    def _correction_weight(self) -> float:
        if self.corrections is None:
            return 0.0
        settings = self.config.get("hybrid", {})
        steps = self.joint_steps
        start = float(settings.get("correction_beta_initial", .10))
        middle = float(settings.get("correction_beta_at_250k", .02))
        if steps < 100_000:
            return start
        if steps < 250_000:
            return start + (middle - start) * (steps - 100_000) / 150_000
        zero = int(settings.get("correction_beta_zero_from", 400_000))
        return max(0.0, middle * (zero - steps) / max(1, zero - 250_000))

    def state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(), "joint_steps": self.joint_steps,
            "neural_action_samples": self.neural_action_samples, "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates, "feedback_updates": self.feedback_updates,
            "envs": [environment.snapshot() for environment in self.envs], "episode_ids": list(self.episode_ids),
            "episode_decisions": list(self.episode_decisions), "short_episode_count": self.short_episode_count,
            "corrections": self.corrections.state_dict() if self.corrections else None,
            "episode_metadata": list(self.episode_metadata), "partners": self.partners.state_dict(),
            "active_trajectory_observations": [np.stack(rows).astype(np.float32) if rows else np.empty((0, len(self.names)), np.float32)
                                               for rows in self.active_trajectory_observations],
            "completed_trajectories": list(self.completed_trajectories),
            "reward_ledgers": [ledger.state_dict() for ledger in self.reward_ledgers],
            "curriculum": self.curriculum.state_dict(), "stage_counts": dict(self.stage_counts),
            "rcpd": self.rcpd.training_state(), "feedback_gate_history": list(self.feedback_gate_history),
            "validation_history": list(self.validation_history), "best_selection": self.best_selection,
            "best_pure_selection": self.best_pure_selection,
            "best_hybrid_selection": self.best_hybrid_selection,
            "parent_provenance": self.parent_provenance,
            "training_initial_signatures": sorted(self.training_initial_signatures),
            "last_evaluation_step": self.last_evaluation_step,
            "feedback_started_at": self.feedback_started_at,
            "signature": _signature(self.env_config, self.names), "config_fingerprint": _config_fingerprint(self.config),
            "physics_config": self.env_config.to_dict,
            "hybrid_rule_version": HYBRID_RULE_VERSION if self.corrections else None,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            result["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            try:
                result["torch_mps_rng"] = torch.mps.get_rng_state()
            except RuntimeError:
                pass
        return result

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("signature") != _signature(self.env_config, self.names):
            raise ValueError("checkpoint observation/action signature mismatch")
        if payload.get("config_fingerprint") != _config_fingerprint(self.config):
            raise ValueError("checkpoint training configuration mismatch; create a new versioned run")
        if payload.get("physics_config", self.env_config.to_dict) != self.env_config.to_dict:
            raise ValueError("checkpoint Pong physics configuration mismatch")
        if payload.get("hybrid_rule_version") != (HYBRID_RULE_VERSION if self.corrections else None):
            raise ValueError("checkpoint hybrid rule version mismatch")
        self.model.load_state_dict(payload["model"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.joint_steps = int(payload["joint_steps"])
        self.neural_action_samples = int(payload["neural_action_samples"])
        self.actor_updates = int(payload["actor_updates"])
        self.critic_updates = int(payload["critic_updates"])
        self.feedback_updates = int(payload.get("feedback_updates", 0))
        for environment, snapshot in zip(self.envs, payload["envs"]):
            environment.restore(snapshot)
        self.episode_ids = [int(value) for value in payload["episode_ids"]]
        self.episode_decisions = [int(value) for value in payload.get("episode_decisions", [0] * len(self.envs))]
        self.short_episode_count = int(payload.get("short_episode_count", 0))
        if (payload.get("corrections") is None) != (self.corrections is None):
            raise ValueError("checkpoint hybrid correction configuration mismatch")
        if self.corrections is not None:
            self.corrections.restore_state(payload["corrections"])
        self.episode_metadata = [dict(value) for value in payload["episode_metadata"]]
        active = payload.get("active_trajectory_observations", ())
        if active and len(active) != len(self.envs):
            raise ValueError("checkpoint trajectory-environment count mismatch")
        self.active_trajectory_observations = [
            [np.asarray(row, dtype=np.float32) for row in np.asarray(rows, dtype=np.float32)]
            for rows in active
        ] if active else [[] for _ in self.envs]
        self.completed_trajectories = [
            {**dict(item), "observations": np.asarray(item["observations"], dtype=np.float32)}
            for item in payload.get("completed_trajectories", ())
        ][-self.trajectory_cache_size:]
        self.partners.restore_state(payload["partners"])
        for ledger, state in zip(self.reward_ledgers, payload["reward_ledgers"]):
            ledger.restore_state(state)
        self.curriculum.restore_state(payload["curriculum"])
        self.stage_counts = {key: int(value) for key, value in payload["stage_counts"].items()}
        self.rcpd.restore_training_state(payload.get("rcpd", {}))
        self.feedback_gate_history = [dict(item) for item in payload.get("feedback_gate_history", ())]
        self.validation_history = [dict(item) for item in payload.get("validation_history", ())]
        self.best_selection = dict(payload["best_selection"]) if payload.get("best_selection") else None
        self.best_pure_selection = dict(payload["best_pure_selection"]) if payload.get("best_pure_selection") else None
        self.best_hybrid_selection = dict(payload["best_hybrid_selection"]) if payload.get("best_hybrid_selection") else None
        self.parent_provenance = dict(payload["parent_provenance"]) if payload.get("parent_provenance") else None
        self.training_initial_signatures = set(map(str, payload.get("training_initial_signatures", ())))
        self.last_evaluation_step = int(payload.get("last_evaluation_step", 0))
        self.feedback_started_at = (None if payload.get("feedback_started_at") is None
                                    else int(payload["feedback_started_at"]))
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("torch_cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
        if payload.get("torch_mps_rng") is not None and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
            try:
                torch.mps.set_rng_state(payload["torch_mps_rng"])
            except RuntimeError:
                pass

    def initialize_from_parent(self, payload: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
        """Start a new experiment from network weights only.

        Optimizer moments, environments, curriculum counts and feedback state
        deliberately stay fresh. This is distinct from ``load_state_dict``.
        """
        signature = payload.get("signature")
        if signature != _signature(self.env_config, self.names):
            raise ValueError("parent checkpoint observation/action signature mismatch")
        self.model.load_state_dict(payload["model"])
        self.partners.set_initial_actor(self.model)
        self.parent_provenance = dict(provenance)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    base = config.pop("base_config", None)
    if base is None:
        return config
    parent = load_config(config_path.parent / str(base))
    def merge(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in right.items():
            left[key] = merge(dict(left.get(key, {})), value) if isinstance(value, dict) else value
        return left
    return merge(parent, config)


def _selection_is_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_score = float(candidate["selection"]["mean_weighted_misses"])
    best_score = float(best["selection"]["mean_weighted_misses"])
    if candidate_score < best_score - 1e-9:
        return True
    if abs(candidate_score - best_score) <= 1e-9:
        candidate_large = candidate["selection"].get("mean_large_success_rate")
        best_large = best["selection"].get("mean_large_success_rate")
        candidate_large = -1.0 if candidate_large is None else float(candidate_large)
        best_large = -1.0 if best_large is None else float(best_large)
        if candidate_large > best_large + 1e-9:
            return True
        # Existing best wins exact ties, yielding the earlier checkpoint.
        return False
    return False


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")


def _parent_paths(config: Mapping[str, Any]) -> list[Path]:
    initialization = config.get("initialization", {})
    values = initialization.get("parent_candidates", ())
    root = Path(__file__).resolve().parents[3]
    paths = [(Path(str(value)).expanduser() if Path(str(value)).is_absolute()
              else root / str(value)).resolve() for value in values]
    if not paths:
        raise ValueError("v2.1 initialization requires explicit parent_candidates")
    return paths


def _choose_parent(trainer: PongPPOTrainer, config: Mapping[str, Any], destination: Path,
                   progress: Any | None = None) -> dict[str, Any]:
    """Evaluate explicit v2 parents under one corrected, fixed protocol."""
    from .evaluation import evaluate_trainer
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for path in _parent_paths(config):
        if not path.is_file():
            raise FileNotFoundError(f"configured parent checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("signature") != _signature(trainer.env_config, trainer.names):
            raise ValueError(f"parent checkpoint signature mismatch: {path}")
        trainer.model.load_state_dict(payload["model"])
        if progress:
            progress({"event": "parent_model_evaluation_started", "checkpoint": str(path)})
        report = evaluate_trainer(trainer, split="validation", partner="all", progress=progress,
                                  controller_mode="pure_nn")
        candidate = {
            "path": str(path.resolve()), "checkpoint_sha256": _checkpoint_hash(path),
            "checkpoint_joint_steps": int(payload.get("joint_steps", 0)), "actor_parameters_sha256": _actor_hash(trainer.model),
            "evaluation": report,
        }
        candidates.append(candidate)
        compared = {"selection": report["selection"], "joint_steps": candidate["checkpoint_joint_steps"]}
        if selected is None or _selection_is_better(compared, {"selection": selected["evaluation"]["selection"]}):
            selected = candidate
    if selected is None:
        raise RuntimeError("no parent was evaluated")
    report = {
        "comparison_version": "pong-nn-v2.2-parent-pure-validation" if config.get("curriculum", {}).get("v22") else "pong-nn-v2.1-fixed-encounter-validation",
        "candidates": candidates, "selected_path": selected["path"],
        "selected_reason": "lowest corrected three-NN-condition weighted misses; then higher direct large success; then earlier checkpoint",
    }
    _atomic_json(report, destination / "parent_comparison.json")
    payload = torch.load(Path(selected["path"]), map_location="cpu", weights_only=False)
    provenance = {
        "mode": "v2_parent_initialization", "checkpoint": selected["path"],
        "checkpoint_sha256": selected["checkpoint_sha256"], "checkpoint_joint_steps": selected["checkpoint_joint_steps"],
        "actor_parameters_sha256": selected["actor_parameters_sha256"],
        "parent_comparison_version": report["comparison_version"],
        "optimizer_moments_imported": False, "parent_training_counts_imported": False,
    }
    trainer.initialize_from_parent(payload, provenance)
    _atomic_json(provenance, destination / "parent_provenance.json")
    return report


def _should_stop_for_plateau(trainer: PongPPOTrainer, config: Mapping[str, Any]) -> bool:
    settings = config.get("stopping", {})
    minimum_full = int(settings.get("minimum_full_stage_steps", 50_000))
    start = int(config.get("curriculum", {}).get("coordination_until_joint_steps", 125_000))
    if trainer.joint_steps < start + minimum_full or len(trainer.validation_history) < 3:
        return False
    window = trainer.validation_history[-3:]
    current = min(float(item["selection"]["mean_weighted_misses"]) for item in window)
    previous = trainer.validation_history[:-3]
    if not previous:
        return False
    reference = min(float(item["selection"]["mean_weighted_misses"]) for item in previous)
    improvement = (reference - current) / max(1e-8, abs(reference))
    return improvement < float(settings.get("minimum_improvement_fraction", .01))


def _integrity_error(trainer: PongPPOTrainer, batch: Mapping[str, Any] | None = None,
                     metrics: Mapping[str, Any] | None = None) -> str | None:
    if batch is not None:
        for key in ("obs", "joint", "rewards", "values", "old_logp"):
            if key in batch and not np.isfinite(np.asarray(batch[key], dtype=np.float64)).all():
                return f"non_finite_{key}"
        actions = np.asarray(batch.get("actions", ()), dtype=np.int64)
        if actions.size and (actions.min() < 0 or actions.max() >= len(ACTIONS)):
            return "illegal_action_index"
    if metrics is not None:
        for key, value in metrics.items():
            if not np.isfinite(float(value)):
                return f"non_finite_metric_{key}"
    for ledger in trainer.reward_ledgers:
        entry_ids = [str(item.get("encounter_id")) for item in ledger.entries]
        if len(entry_ids) != len(set(entry_ids)) or set(entry_ids) != ledger.rewarded_ids:
            return "duplicate_or_missing_reward_event"
    return None


def run(config_path: str | Path, output: str | Path, *, device: str = "cpu", resume: bool = False,
        max_joint_steps: int | None = None, max_runtime_minutes: float | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if max_joint_steps is not None:
        config["training"]["max_joint_steps"] = int(max_joint_steps)
        config.setdefault("training", {})["skip_full_evaluation_for_testing"] = True
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if not resume and any(destination.iterdir()):
        raise FileExistsError("Fresh training requires an empty output directory; use --resume")
    _seed(int(config["seed"]))
    trainer = PongPPOTrainer(config, device=_device(device))
    checkpoint = destination / "last.pt"
    source_manifest_path = destination / "source_manifest.json"
    if config.get("curriculum", {}).get("v22"):
        current_sources = _source_manifest()
        if resume:
            if not source_manifest_path.is_file() or json.loads(source_manifest_path.read_text(encoding="utf-8")) != current_sources:
                raise ValueError("Pong v2.2 training sources changed; initialize a new experiment instead of resume")
        else:
            _atomic_json(current_sources, source_manifest_path)
    parent_comparison: dict[str, Any] | None = None
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume requires {checkpoint}")
        trainer.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    else:
        print(json.dumps({"event": "parent_model_comparison_started", "candidates": [str(path) for path in _parent_paths(config)]}, ensure_ascii=False), flush=True)
        parent_comparison = _choose_parent(
            trainer, config, destination,
            progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
        )
    target = int(config["training"]["max_joint_steps"])
    _atomic_json(config, destination / "config.yaml")
    _atomic_json(_signature(trainer.env_config, trainer.names), destination / "signature.json")
    signal.signal(signal.SIGINT, trainer.request_stop)
    signal.signal(signal.SIGTERM, trainer.request_stop)
    started = time.monotonic()
    starting_joint_steps = trainer.joint_steps
    last_save = trainer.joint_steps
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    print(json.dumps({"event": "training_ready", "device": str(trainer.device), "joint_steps": trainer.joint_steps,
                      "target_joint_steps": target, "curriculum_stage": trainer._stage(),
                      "initialization": "exact_resume" if resume else "parent_checkpoint_continuation"}, ensure_ascii=False), flush=True)
    from .evaluation import evaluate_trainer
    last_evaluation: dict[str, Any] | None = None
    stop_reason: str | None = None
    interval = int(config["training"]["checkpoint_every_joint_steps"])
    skip_evaluation = bool(config["training"].get("skip_full_evaluation_for_testing", False))
    v22 = bool(config.get("curriculum", {}).get("v22"))
    while trainer.joint_steps < target and not trainer._stop:
        if max_runtime_minutes is not None and (time.monotonic() - started) >= max_runtime_minutes * 60:
            stop_reason = "runtime_limit"
            trainer.request_stop()
            break
        remaining = (target - trainer.joint_steps) // len(trainer.envs)
        if remaining <= 0:
            break
        batch = trainer.collect(min(int(config["ppo"]["rollout_decisions"]), remaining))
        integrity = _integrity_error(trainer, batch)
        if integrity:
            stop_reason = integrity
            _atomic_json({"joint_steps": trainer.joint_steps, "reason": integrity}, destination / "integrity_failure.json")
            trainer.request_stop()
            break
        extraction = None if v22 else trainer._maybe_extract()
        assisted_records: list[dict[str, Any]] = []
        if trainer.corrections is not None and trainer.joint_steps < int(config.get("hybrid", {}).get("correction_beta_zero_from", 400_000)):
            remaining_assisted = max(0, int(config["training"].get("assisted_joint_steps_budget", 0))
                                     - trainer.corrections.assisted_steps)
            requested_assisted = round(batch["actions"].shape[0] * len(trainer.envs)
                                       * float(config["training"].get("assisted_fraction", 0.0)))
            if remaining_assisted and requested_assisted:
                assisted_records = trainer.corrections.collect(trainer.model, trainer.device,
                                                               min(remaining_assisted, requested_assisted),
                                                               ppo_joint_steps=trainer.joint_steps)
                for sample in assisted_records:
                    _append_jsonl(destination / "correction_audit.jsonl", sample)
                for update in trainer.corrections.settled_updates:
                    _append_jsonl(destination / "correction_outcomes.jsonl", update)
                trainer.corrections.settled_updates.clear()
        metrics = trainer.update(batch)
        integrity = _integrity_error(trainer, metrics=metrics)
        if integrity:
            stop_reason = integrity
            _atomic_json({"joint_steps": trainer.joint_steps, "reason": integrity}, destination / "integrity_failure.json")
            trainer.request_stop()
        elapsed = max(.001, time.monotonic() - started)
        reward_components = {key: sum(item[key] for item in batch["reward_components"])
                             for key in ("small_miss", "large_miss", "total")}
        report = {"event": "ppo_update", "joint_steps": trainer.joint_steps, "target_joint_steps": target,
                  "joint_steps_per_second": (trainer.joint_steps - starting_joint_steps) / elapsed,
                  "curriculum_stage": trainer._stage(),
                  "curriculum_actual_proportions": trainer.curriculum.actual_proportions(),
                  "partner_joint_steps": trainer.partners.joint_steps, "reward_components": reward_components,
                  "assisted_joint_steps": trainer.corrections.assisted_steps if trainer.corrections else 0,
                  "comparison_replay_steps": trainer.corrections.comparison_steps if trainer.corrections else 0,
                  "correction_accepted": len(trainer.corrections.accepted) if trainer.corrections else 0,
                  "correction_rejected": len(trainer.corrections.rejected) if trainer.corrections else 0,
                  "correction_supervised_updates": trainer.corrections.supervised_updates if trainer.corrections else 0,
                  "rule_interventions": trainer.corrections.interventions if trainer.corrections else 0,
                  "rule_intervention_reasons": dict(trainer.corrections.reason_counts) if trainer.corrections else {},
                  "correction_beta": trainer._correction_weight(),
                  **metrics, "feedback_enabled": trainer.rcpd.regularization_weight > 0,
                  "feedback_weight": trainer._feedback_weight(), "extraction": extraction}
        _append_jsonl(destination / "training_metrics.jsonl", report)
        print(json.dumps(report, ensure_ascii=False, default=_json_default), flush=True)
        due = trainer.joint_steps - trainer.last_evaluation_step >= interval
        final = trainer.joint_steps >= target
        if (due or final) and not skip_evaluation:
            if v22:
                extraction = trainer._maybe_extract()
                _append_jsonl(destination / "training_metrics.jsonl",
                              {"event": "rcpd_extraction_check", "joint_steps": trainer.joint_steps,
                               "result": extraction, "actor_parameters_sha256": _actor_hash(trainer.model)})
            last_evaluation = evaluate_trainer(
                trainer, split="validation", partner="all",
                progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True),
                controller_mode="all" if v22 else "pure_nn",
            )
            last_evaluation.update({"joint_steps": trainer.joint_steps, "stage": trainer._stage()})
            trainer.validation_history.append(last_evaluation)
            trainer.last_evaluation_step = trainer.joint_steps
            _atomic_json(last_evaluation, destination / "evaluations" / f"validation_{trainer.joint_steps:07d}.json")
            candidate = {"joint_steps": trainer.joint_steps, **last_evaluation}
            if v22:
                pure_report = {"joint_steps": trainer.joint_steps, **last_evaluation["modes"]["pure_nn"]}
                hybrid_report = {"joint_steps": trainer.joint_steps, **last_evaluation["modes"]["hybrid"]}
                if _selection_is_better(pure_report, trainer.best_pure_selection):
                    trainer.best_pure_selection = pure_report
                    _atomic_torch_save(trainer.state_dict(), destination / "best_pure_candidate.pt")
                    _atomic_json(pure_report, destination / "best_pure_candidate.json")
                if _selection_is_better(hybrid_report, trainer.best_hybrid_selection):
                    trainer.best_hybrid_selection = hybrid_report
                    trainer.best_selection = candidate
                    state = trainer.state_dict()
                    _atomic_torch_save(state, destination / "best_hybrid_candidate.pt")
                    _atomic_torch_save(state, destination / "best_candidate.pt")
                    _atomic_json(hybrid_report, destination / "best_hybrid_candidate.json")
                    _atomic_json({"alias_of": "best_hybrid_candidate.pt", **candidate}, destination / "best_candidate.json")
                trainer.partners.add_actor_snapshot(trainer.model)
            elif _selection_is_better(candidate, trainer.best_selection):
                trainer.best_selection = candidate
                _atomic_torch_save(trainer.state_dict(), destination / "best_candidate.pt")
                _atomic_json(candidate, destination / "best_candidate.json")
            if _should_stop_for_plateau(trainer, config):
                stop_reason = "full_stage_validation_plateau"
                trainer.request_stop()
        due_save = trainer.joint_steps - last_save >= interval
        if due_save or trainer._stop or final:
            state = trainer.state_dict()
            _atomic_torch_save(state, checkpoint)
            _atomic_torch_save(state, checkpoints / f"step_{trainer.joint_steps:07d}.pt")
            last_save = trainer.joint_steps
    state = trainer.state_dict()
    _atomic_torch_save(state, checkpoint)
    summary = {"status": "early_stopped" if stop_reason else ("interrupted" if trainer._stop else "budget_complete"),
               "stop_reason": stop_reason, "joint_steps": trainer.joint_steps, "target_joint_steps": target,
               "actor_updates": trainer.actor_updates, "critic_updates": trainer.critic_updates,
               "neural_action_samples": trainer.neural_action_samples, "feedback_updates": trainer.feedback_updates,
               "assisted_joint_steps": trainer.corrections.assisted_steps if trainer.corrections else 0,
               "comparison_replay_steps": trainer.corrections.comparison_steps if trainer.corrections else 0,
               "correction_supervised_updates": trainer.corrections.supervised_updates if trainer.corrections else 0,
               "correction_accepted": len(trainer.corrections.accepted) if trainer.corrections else 0,
               "correction_rejected": len(trainer.corrections.rejected) if trainer.corrections else 0,
               "rule_interventions": trainer.corrections.interventions if trainer.corrections else 0,
               "rule_intervention_reasons": dict(trainer.corrections.reason_counts) if trainer.corrections else {},
               "partner_episodes": trainer.partners.episode_counts, "partner_joint_steps": trainer.partners.joint_steps,
               "partner_neural_action_samples": trainer.partners.neural_samples,
               "curriculum_episode_counts": trainer.stage_counts,
               "curriculum_step_counts": trainer.curriculum.step_counts,
               "curriculum_actual_proportions": trainer.curriculum.actual_proportions(),
               "parent_provenance": trainer.parent_provenance, "parent_comparison": parent_comparison,
               "best": trainer.best_selection,
               "qualification": {"performed": False, "status": "formal_not_run",
                                 "training_progress_checks": last_evaluation.get("progress_checks") if last_evaluation else None},
               "best_pure_candidate": trainer.best_pure_selection,
               "best_hybrid_candidate": trainer.best_hybrid_selection,
               "last_evaluation": last_evaluation}
    _atomic_json(summary, destination / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return summary
