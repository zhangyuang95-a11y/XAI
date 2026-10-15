"""Centered delivery credit: a new PPO continuation, never a runtime controller.

Only returned per-role training rewards change. The public environment state,
including the observer's state hash and original shared last_rewards, remains
unchanged. Actual credit values are retained in transition info, not observations.
This module grants no sampling budget and does not load checkpoint files.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import numpy as np
import torch

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback import PublicFeedbackEnvironment, VERSION as OBSERVER_VERSION
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import (
    PublicFeedbackTrainer, COUNTERS, EPISODE_FIELDS, _cpu, _integer, _finite,
    _validate_learning_state, _preserve_global_rng, _set_global_rng,
)
from .warehouse_native_partner_mix_trainer import (
    PartnerMixTrainer, VERSION as SOURCE_VERSION, PARTNER_MIXES, _sha, _prefix,
    execution_sources as parent_sources,
)
from .warehouse_native_revision_reward import REWARD_REVISION
from .warehouse_native_v2 import NativeV2Trainer
from env.warehouse_native.scenarios import reset_scenario

VERSION = "warehouse-native-delivery-credit-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-delivery-credit-protocol.v1"
CREDIT_REWARD_VERSION = "warehouse-native-centered-delivery-credit.v1"
BRANCHES = ("team_credit", "own_credit")
ALPHAS = {"team_credit": 0., "own_credit": .5}
_CREDIT_KEYS = {"credit_reward_version", "delivery_credit_alpha"}


def execution_sources():
    result = parent_sources()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


class CreditRewardEnvironment(PublicFeedbackEnvironment):
    """Identical public physics/history; two-role mean reward stays at r1."""

    def __init__(self, config=None, reward_config=None, collision_cost=.05, *, alpha=0., mode="observed"):
        if type(alpha) not in (int, float) or alpha not in ALPHAS.values() or mode != "observed":
            raise ValueError("Only registered delivery credit and observed history are supported")
        self.alpha = float(alpha)
        super().__init__(config, reward_config, collision_cost=collision_cost, mode=mode)

    def step(self, actions, *, decision_metadata=None):
        observations, shared, terminated, truncated, info = super().step(actions, decision_metadata=decision_metadata)
        deliveries = {key: 0 for key in self.agent_ids}
        for event in info["events"]:
            if event["event"] == "delivery": deliveries[event["agent_id"]] += 1
        team = sum(deliveries.values())
        adjustments = {key: self.alpha * (2 * deliveries[key] - team) for key in self.agent_ids}
        rewards = {key: shared[key] + adjustments[key] for key in self.agent_ids}
        info["delivery_credit"] = {"version": CREDIT_REWARD_VERSION, "alpha": self.alpha,
            "by_role_deliveries": deliveries, "team_deliveries": team,
            "adjustments": adjustments, "shared_rewards": dict(shared), "rewards": dict(rewards)}
        return observations, rewards, terminated, truncated, info

    def snapshot(self):
        result = super().snapshot()
        result.update(credit_reward_version=CREDIT_REWARD_VERSION, delivery_credit_alpha=self.alpha)
        return result

    def restore(self, payload, *, require_feedback=False, require_credit=False):
        if not isinstance(payload, dict) or type(require_credit) is not bool:
            raise ValueError("Invalid credit snapshot")
        keys = {key for key in payload if key.startswith("credit_") or key.startswith("delivery_credit_")}
        if (require_credit and not keys) or (keys and (keys != _CREDIT_KEYS
                or payload.get("credit_reward_version") != CREDIT_REWARD_VERSION
                or type(payload.get("delivery_credit_alpha")) not in (int, float)
                or payload["delivery_credit_alpha"] != self.alpha)):
            raise ValueError("Credit snapshot version/alpha is missing or different")
        candidate = PublicFeedbackEnvironment(self.config, copy.deepcopy(self.reward_config),
            collision_cost=self.collision_cost, mode="observed")
        candidate.restore({k: copy.deepcopy(v) for k, v in payload.items() if k not in _CREDIT_KEYS},
                          require_feedback=require_feedback)
        self.state = candidate.state
        self.set_rng_state(candidate.get_rng_state())
        self._episode_counter, self._history = candidate._episode_counter, copy.deepcopy(candidate._history)

    def branch(self):
        result = type(self)(self.config, copy.deepcopy(self.reward_config), self.collision_cost, alpha=self.alpha)
        result.restore(self.snapshot(), require_feedback=True, require_credit=True)
        return result


def make_protocol(source_protocol, *, source_checkpoint_sha256, source_state_sha256, test_fixture=False):
    if not isinstance(source_protocol, dict) or type(test_fixture) is not bool:
        raise ValueError("Source protocol and explicit fixture mode are required")
    return {"version": PROTOCOL_VERSION, "test_fixture": test_fixture, "arms": list(BRANCHES),
        "public_feedback_version": OBSERVER_VERSION, "public_feedback_mode": "observed",
        "credit_reward_version": CREDIT_REWARD_VERSION, "alphas_by_branch": dict(ALPHAS),
        "partners_by_branch": {key: copy.deepcopy(PARTNER_MIXES["baseline_mix"]) for key in BRANCHES},
        "source_protocol": copy.deepcopy(source_protocol),
        **{key: copy.deepcopy(source_protocol[key]) for key in ("training", "reward", "collision_training_cost", "seed")},
        "feedback_training_enabled": False,
        "source": {"checkpoint_sha256": _sha(source_checkpoint_sha256, "source checkpoint"),
            "state_sha256": _sha(source_state_sha256, "source state"), "protocol_sha256": digest(source_protocol),
            "trainer_version": SOURCE_VERSION, "branch": "baseline_mix", "mode": "observed",
            "joint_steps": 100000, "cumulative_joint_steps": 480000},
        "budget": {"maximum_ppo_joint_steps_per_arm": 250000,
            "requested_total_ppo_joint_steps": 500000, "curriculum_generation_steps": 0},
        "evaluation": {"checkpoints_ppo_steps": [50000, 250000], "partners": ["skilled", "assertive", "noisy"],
            "scenarios_per_partner": 50, "deterministic": True, "horizon": 120,
            "maximum_environment_steps": 72000, "reuse_source_zero_step_validation": True,
            "read_final_test": False, "training_credit_in_evaluation": False},
        "continuation": {"inflight_episodes": "preserve_exactly", "rng": "resume_without_reseeding",
            "reward_change": "centered_delivery_credit_only", "critic_structure": "unchanged"}}


def _validate_protocol(protocol, source, context, fixture, checkpoint_sha, state_sha):
    if (type(fixture) is not bool or not isinstance(protocol, dict)
            or protocol.get("version") != PROTOCOL_VERSION or protocol.get("test_fixture") is not fixture
            or protocol.get("arms") != list(BRANCHES) or protocol.get("alphas_by_branch") != ALPHAS
            or protocol.get("public_feedback_version") != OBSERVER_VERSION
            or protocol.get("public_feedback_mode") != "observed"
            or protocol.get("credit_reward_version") != CREDIT_REWARD_VERSION
            or protocol.get("feedback_training_enabled") is not False
            or protocol.get("partners_by_branch") != {key: PARTNER_MIXES["baseline_mix"] for key in BRANCHES}):
        raise ValueError("Delivery-credit-only experiment contract differs")
    if not isinstance(context, dict) or context.get("protocol") != source.get("protocol"):
        raise ValueError("The genuine source protocol context is required")
    original = context["protocol"]
    if protocol.get("source_protocol") != original:
        raise ValueError("Archived source protocol differs")
    for key in ("training", "reward", "collision_training_cost", "seed"):
        if protocol.get(key) != original[key]: raise ValueError("Credit must not change " + key)
    expected = {"checkpoint_sha256": checkpoint_sha, "state_sha256": state_sha,
        "protocol_sha256": digest(original), "trainer_version": SOURCE_VERSION,
        "branch": "baseline_mix", "mode": "observed", "joint_steps": source.get("joint_steps"),
        "cumulative_joint_steps": source.get("joint_steps", 0) + source.get("source_counters", {}).get("joint_steps", 0)}
    if protocol.get("source") != expected or (not fixture and
            (expected["joint_steps"] != 100000 or expected["cumulative_joint_steps"] != 480000)):
        raise ValueError("Credit source is not the fixed baseline endpoint")
    if protocol.get("continuation") != {"inflight_episodes": "preserve_exactly", "rng": "resume_without_reseeding",
            "reward_change": "centered_delivery_credit_only", "critic_structure": "unchanged"}:
        raise ValueError("Continuation contract differs")
    budget = protocol.get("budget", {}); maximum = budget.get("maximum_ppo_joint_steps_per_arm")
    _integer(maximum, "Per-arm budget", 1)
    if (maximum % protocol["training"]["environments"] or budget.get("requested_total_ppo_joint_steps") != 2 * maximum
            or budget.get("curriculum_generation_steps") != 0 or (not fixture and maximum != 250000)):
        raise ValueError("Credit continuation requires its own fixed round budget")


class CreditTrainer(PartnerMixTrainer):
    def __init__(self, protocol, scenarios, source_state, *, source_context, branch,
                 expected_source_state_sha256, source_checkpoint_sha256, device="cpu", test_fixture=False):
        if branch not in BRANCHES or type(branch) is not str: raise ValueError("Unknown credit branch")
        _sha(expected_source_state_sha256, "source state"); _sha(source_checkpoint_sha256, "source checkpoint")
        if not isinstance(source_state, dict) or initialization_sha256(source_state) != expected_source_state_sha256:
            raise ValueError("Source trainer state differs from its external anchor")
        _validate_protocol(protocol, source_state, source_context, test_fixture,
                           source_checkpoint_sha256, expected_source_state_sha256)
        context = source_context
        # The actual PartnerMixTrainer validates its original observed initializer
        # and then its own untouched baseline endpoint schema, including both Adam states.
        original = PartnerMixTrainer(context["protocol"], scenarios, context["original_source_state"],
            branch="baseline_mix", expected_source_state_sha256=context["original_source_state_sha256"],
            source_checkpoint_sha256=context["original_source_checkpoint_sha256"], device=device, test_fixture=test_fixture)
        original.load_state_dict(source_state)
        self.device, self.mode, self.branch, self.test_fixture = original.device, "observed", branch, test_fixture
        self.protocol, self.scenarios = copy.deepcopy(protocol), copy.deepcopy(scenarios)
        self.cfg, self.seed, self.alpha = self.protocol["training"], self.protocol["seed"], ALPHAS[branch]
        self.sources, self.initialization_sha256 = execution_sources(), expected_source_state_sha256
        self.source_checkpoint_sha256 = source_checkpoint_sha256
        self.source_partner_mix_counters = {key: source_state[key] for key in COUNTERS}
        self.source_ancestry_counters = {"source_counters": copy.deepcopy(source_state["source_counters"]),
            "source_public_history_counters": copy.deepcopy(source_state["source_public_history_counters"]),
            "source_r1_counters": copy.deepcopy(source_state["source_r1_counters"])}
        self.source_context_sha256 = initialization_sha256(source_context)
        self.source_counters = {key: original.source_counters[key] + source_state[key]
                               for key in ("joint_steps", "optimizer_updates", "minibatch_updates")}
        self.source_counters.update({role + "_optimizer_steps": original.source_counters[role + "_optimizer_steps"]
            + source_state[role + "_optimizer_steps"] for role in ("actor", "critic")})
        self.source_frames = [env.state.frame for env in original.envs]
        self.source_completed_episodes_sha256 = initialization_sha256(source_state["completed_episodes"])
        self.source_round_episode_prefixes_sha256 = initialization_sha256(source_state["round_episode_prefixes"])
        self.model, self.optimizers, self.rng = original.model, original.optimizers, original.rng
        self._owned_rng_state, self._train_ids = _cpu(original._owned_rng_state), original._train_ids.copy()
        self.envs = []
        for saved in source_state["envs"]:
            env = self._environment(); env.restore(saved, require_feedback=True)
            self.envs.append(env)
        for key in EPISODE_FIELDS: setattr(self, key, copy.deepcopy(getattr(original, key)))
        self.round_episode_prefixes = [_prefix(env, self.episode_returns[i], self.episode_reward_components[i])
                                      for i, env in enumerate(self.envs)]
        self.completed_episodes = []
        for key in COUNTERS: setattr(self, key, 0)
        self.elapsed_seconds, self.feedback_enabled, self._collecting = 0., False, False

    def _environment(self):
        return CreditRewardEnvironment(reward_config=self.protocol["reward"],
            collision_cost=self.protocol["collision_training_cost"], alpha=self.alpha)

    def collect(self, time_steps):
        _integer(time_steps, "Rollout length", 1)
        if self.joint_steps + time_steps * len(self.envs) > self.protocol["budget"]["maximum_ppo_joint_steps_per_arm"]:
            raise ValueError("Collect exceeds the credit round limit")
        self._collecting = True
        try: batch = PublicFeedbackTrainer.collect(self, time_steps)
        finally: self._collecting = False
        for record in batch["transition_records"]:
            record.update(version=VERSION, branch=self.branch, delivery_credit_alpha=self.alpha,
                source_cumulative_joint_steps=self.source_counters["joint_steps"],
                cumulative_joint_step=self.source_counters["joint_steps"] + record["joint_step"])
        return batch

    def _bindings(self):
        return {"version": VERSION, "branch": self.branch, "credit_reward_version": CREDIT_REWARD_VERSION,
            "delivery_credit_alpha": self.alpha, "public_feedback_version": OBSERVER_VERSION,
            "public_feedback_mode": "observed", "training_device": self.device.type,
            "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios), "sources": copy.deepcopy(self.sources),
            "source_sha256": digest(self.sources), "initialization_sha256": self.initialization_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256, "source_context_sha256": self.source_context_sha256,
            "source_counters": copy.deepcopy(self.source_counters),
            "source_partner_mix_counters": copy.deepcopy(self.source_partner_mix_counters),
            "source_ancestry_counters": copy.deepcopy(self.source_ancestry_counters),
            "source_completed_episodes_sha256": self.source_completed_episodes_sha256,
            "source_round_episode_prefixes_sha256": self.source_round_episode_prefixes_sha256,
            "source_frames": self.source_frames.copy(), "obs_dim": 197, "state_dim": 354,
            "feature_names": list(self.envs[0].feature_names), "reward_revision": REWARD_REVISION,
            "test_fixture": self.test_fixture, "curriculum_generation_steps": 0, "feedback_enabled": False}

    def state_dict(self):
        if execution_sources() != self.sources: raise ValueError("Credit execution sources changed")
        result = {**self._bindings(), "model": _cpu(self.model.state_dict()), "optimizers": _cpu(self.optimizers.state_dict()),
            "rng": copy.deepcopy(self.rng.bit_generator.state), **_cpu(self._owned_rng_state),
            "envs": [env.snapshot() for env in self.envs], "completed_episodes": _cpu(self.completed_episodes),
            "elapsed_seconds": self.elapsed_seconds, "round_episode_prefixes": copy.deepcopy(self.round_episode_prefixes)}
        for key in (*COUNTERS, *EPISODE_FIELDS): result[key] = _cpu(getattr(self, key))
        result["episode_reward_components"] = [dict(x) for x in self.episode_reward_components]
        return result

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid checkpoint or changed credit sources")
        if any(payload.get(k) != v for k, v in self._bindings().items()):
            raise ValueError("Credit checkpoint identity/alpha/device/source differs")
        p, count = copy.deepcopy(payload), len(self.envs)
        for key in COUNTERS: _integer(p.get(key), key)
        if (p["joint_steps"] % count or p["joint_steps"] > self.protocol["budget"]["maximum_ppo_joint_steps_per_arm"]
                or p["optimizer_updates"] > p["minibatch_updates"] or p["actor_optimizer_steps"] > p["minibatch_updates"]
                or p["critic_optimizer_steps"] != p["minibatch_updates"] or p["last_evaluated_joint_steps"] > p["joint_steps"]):
            raise ValueError("Invalid credit round counters")
        _finite(p.get("elapsed_seconds"), "elapsed time")
        if p["elapsed_seconds"] < 0: raise ValueError("Negative elapsed time")
        _validate_learning_state(p.get("model"), p.get("optimizers"), self.cfg,
            {role: self.source_counters[role + "_optimizer_steps"] + p[role + "_optimizer_steps"] for role in ("actor", "critic")})
        for key in (*EPISODE_FIELDS, "envs", "round_episode_prefixes"):
            if not isinstance(p.get(key), (list, np.ndarray)) or len(p[key]) != count:
                raise ValueError("Invalid episode vector: " + key)
        returns = np.asarray(p["episode_returns"])
        if returns.shape != (count,) or returns.dtype != np.float64 or not np.isfinite(returns).all():
            raise ValueError("Invalid episode returns")
        envs = []
        for i, saved in enumerate(p["envs"]):
            kind, role = p["partner_kinds"][i], p["program_roles"][i]
            if (kind not in PARTNER_MIXES["baseline_mix"] or type(role) is not int
                    or role not in ((-1,) if kind == "selfplay" else (0, 1)) or p["scenario_ids"][i] not in self._train_ids):
                raise ValueError("Invalid episode partner/scenario")
            entry = next(x for x in self.scenarios["splits"]["train"] if x["id"] == p["scenario_ids"][i])
            baseline = self._environment(); reset_scenario(baseline, entry)
            expected = {"source": "original_train", "curriculum_id": None, "category": None, "start_frame": 0,
                "prefix": {"team_deliveries": baseline.state.total_deliveries,
                    "individual_deliveries": [a.deliveries_completed for a in baseline.state.agents],
                    "shutdowns": baseline.state.shutdown_count, "collisions": baseline.state.robot_collision_events}}
            if p["episode_context"][i] != expected: raise ValueError("Source episode context differs")
            env = self._environment(); env.restore(saved, require_feedback=True, require_credit=True)
            if (env.done or env.state.frame > self.source_frames[i] + p["joint_steps"] // count
                    or (env.state.frame and not env.public_history()["valid"])):
                raise ValueError("Invalid confirmed in-flight state")
            components = p["episode_reward_components"][i]
            if not isinstance(components, dict): raise ValueError("Invalid episode reward accounting")
            for k, v in components.items():
                if type(k) is not str: raise ValueError("Invalid component name")
                _finite(v, k)
            prefix = p["round_episode_prefixes"][i]
            if not isinstance(prefix, dict) or set(prefix) != set(_prefix(env, 0., {})):
                raise ValueError("Invalid inherited episode prefix")
            for key in ("frame", "team_deliveries", "shutdowns", "collisions"): _integer(prefix[key], key)
            if prefix["frame"] > env.state.frame or not isinstance(prefix["individual_deliveries"], list) or len(prefix["individual_deliveries"]) != 2:
                raise ValueError("Invalid prefix frame/deliveries")
            for value in prefix["individual_deliveries"]: _integer(value, "prefix delivery")
            _finite(prefix["return"], "prefix return")
            if not isinstance(prefix["reward_components"], dict): raise ValueError("Invalid prefix rewards")
            for value in prefix["reward_components"].values(): _finite(value, "prefix reward")
            envs.append(env)
        if not isinstance(p.get("completed_episodes"), list): raise ValueError("Invalid episode journal")
        initialization_sha256(p["completed_episodes"])
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(p["rng"])
        owned = {k: p[k] for k in ("python_rng", "numpy_rng", "torch_rng")}
        if self.device.type == "mps": owned["mps_rng"] = p["mps_rng"]
        elif "mps_rng" in p: raise ValueError("CPU cannot restore MPS RNG")
        for key in ("torch_rng", *(("mps_rng",) if self.device.type == "mps" else ())):
            value = owned[key]
            if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.device.type != "cpu" or value.ndim != 1:
                raise ValueError("Invalid RNG tensor")
        with _preserve_global_rng(self.device):
            _set_global_rng(owned, self.device)
            model, optimizers = self._new_learning_state(p["model"], p["optimizers"])
        self.model, self.optimizers, self.envs, self.rng = model, optimizers, envs, rng
        self._owned_rng_state = _cpu(owned)
        for key in (*COUNTERS, *EPISODE_FIELDS, "elapsed_seconds", "completed_episodes", "round_episode_prefixes"):
            setattr(self, key, copy.deepcopy(p[key]))
        self.episode_returns = returns.copy()
        self.episode_reward_components = [Counter(x) for x in p["episode_reward_components"]]

    def export(self, path):
        if execution_sources() != self.sources: raise ValueError("Credit execution sources changed")
        metadata = self._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, joint_steps=self.joint_steps,
            optimizer_updates=self.optimizer_updates, minibatch_updates=self.minibatch_updates,
            initialization="exact_baseline_mix_endpoint_with_inflight_episodes", candidate=True)
        return self.model.export_npz(path, metadata)
