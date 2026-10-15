"""Exact observed-history continuation with one changed episode mixture.

This component neither authorizes sampling nor selects a model. The caller must
bind the original checkpoint bytes and reserve the new round's budget. All
in-flight episodes, optimizer moments and random streams survive the fork.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import numpy as np
import torch

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback import VERSION as OBSERVER_VERSION, OBSERVATION_SIZE, GLOBAL_STATE_SIZE
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import (
    PublicFeedbackTrainer, VERSION as SOURCE_VERSION, COUNTERS, EPISODE_FIELDS,
    execution_sources as original_sources, _validate_protocol, _validate_learning_state,
    _integer, _finite, _cpu, _preserve_global_rng, _set_global_rng,
)
from .warehouse_native_revision_reward import REWARD_REVISION
from .warehouse_native_v2 import NativeV2Trainer
from env.warehouse_native.scenarios import reset_scenario

VERSION = "warehouse-native-partner-mix-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-partner-mix-protocol.v1"
BRANCHES = ("baseline_mix", "active_mix")
PARTNER_MIXES = {
    "baseline_mix": {"selfplay": .5, "skilled": .2, "assertive": .2, "noisy": .1},
    "active_mix": {"selfplay": .5, "skilled": .05, "assertive": .35, "noisy": .1},
}


def execution_sources():
    result = original_sources()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _sha(value, name):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Invalid SHA-256: " + name)
    return value


def make_protocol(source_protocol, *, source_checkpoint_sha256, source_state_sha256,
                  test_fixture=False):
    """New experiment identity; retains the old protocol verbatim as evidence."""
    _validate_protocol(source_protocol, test_fixture)
    return {
        "version": PROTOCOL_VERSION, "test_fixture": test_fixture,
        "public_feedback_version": OBSERVER_VERSION, "arms": list(BRANCHES),
        "public_feedback_mode": "observed", "partners_by_branch": copy.deepcopy(PARTNER_MIXES),
        "seed": source_protocol["seed"], "training": copy.deepcopy(source_protocol["training"]),
        "reward": copy.deepcopy(source_protocol["reward"]),
        "collision_training_cost": source_protocol["collision_training_cost"],
        "feedback_training_enabled": False, "source_protocol": copy.deepcopy(source_protocol),
        "source": {"checkpoint_sha256": _sha(source_checkpoint_sha256, "source checkpoint"),
            "state_sha256": _sha(source_state_sha256, "source state"),
            "protocol_sha256": digest(source_protocol), "trainer_version": SOURCE_VERSION,
            "mode": "observed", "joint_steps": 100000},
        "budget": {"maximum_ppo_joint_steps_per_arm": 100000,
            "requested_total_ppo_joint_steps": 200000, "curriculum_generation_steps": 0},
        "evaluation": {"checkpoints_ppo_steps": [50000, 100000],
            "partners": ["skilled", "assertive", "noisy"], "scenarios_per_partner": 50,
            "deterministic": True, "horizon": 120, "maximum_environment_steps": 72000,
            "reuse_source_zero_step_validation": True, "read_final_test": False},
        "continuation": {"inflight_episodes": "preserve_exactly",
            "partner_change": "next_normal_episode_reset_only", "rng": "resume_without_reseeding",
            "source_selection": "fixed_observed_failure_endpoint_not_best_model"},
    }


def _protocol(protocol, source, fixture):
    if type(fixture) is not bool or not isinstance(protocol, dict):
        raise ValueError("Explicit fixture mode and protocol are required")
    original = protocol.get("source_protocol")
    _validate_protocol(original, fixture)
    if (protocol.get("version") != PROTOCOL_VERSION or protocol.get("test_fixture") is not fixture
            or protocol.get("public_feedback_version") != OBSERVER_VERSION
            or protocol.get("public_feedback_mode") != "observed"
            or protocol.get("arms") != list(BRANCHES)
            or protocol.get("partners_by_branch") != PARTNER_MIXES
            or protocol.get("feedback_training_enabled") is not False):
        raise ValueError("Partner-mixture-only protocol differs")
    for name in ("training", "reward", "collision_training_cost", "seed"):
        if protocol.get(name) != original[name]:
            raise ValueError("The mixture experiment cannot alter " + name)
    expected_source = {"checkpoint_sha256": source["checkpoint_sha256"],
        "state_sha256": source["state_sha256"], "protocol_sha256": digest(original),
        "trainer_version": SOURCE_VERSION, "mode": "observed", "joint_steps": source["joint_steps"]}
    if protocol.get("source") != expected_source or (not fixture and source["joint_steps"] != 100000):
        raise ValueError("Source must be the fixed observed-history endpoint")
    if original["partners"] != PARTNER_MIXES["baseline_mix"]:
        raise ValueError("The source episode mixture differs from baseline_mix")
    if protocol.get("continuation") != {
            "inflight_episodes": "preserve_exactly", "partner_change": "next_normal_episode_reset_only",
            "rng": "resume_without_reseeding", "source_selection": "fixed_observed_failure_endpoint_not_best_model"}:
        raise ValueError("In-flight and RNG preservation contract differs")
    budget = protocol.get("budget", {})
    maximum = budget.get("maximum_ppo_joint_steps_per_arm")
    _integer(maximum, "Per-arm budget", 1)
    if (maximum % protocol["training"]["environments"]
            or budget.get("requested_total_ppo_joint_steps") != 2 * maximum
            or budget.get("curriculum_generation_steps") != 0
            or (not fixture and maximum != 100000)):
        raise ValueError("Invalid separately bounded continuation budget")


def _load_original(payload, scenarios, device, fixture):
    """Use the authentic old loader, with its authentic version and protocol.

    No old validator or checkpoint field is overridden. The outer semantic SHA
    is checked by our constructor before this complete detached staging load.
    """
    if (payload.get("version") != SOURCE_VERSION or payload.get("public_feedback_mode") != "observed"
            or payload.get("training_device") != device.type
            or payload.get("test_fixture") is not fixture or payload.get("sources") != original_sources()):
        raise ValueError("Original observed checkpoint identity/device/source differs")
    _validate_protocol(payload["protocol"], fixture)
    old = PublicFeedbackTrainer.__new__(PublicFeedbackTrainer)
    old.device, old.mode, old.test_fixture = device, "observed", fixture
    old.protocol, old.scenarios = copy.deepcopy(payload["protocol"]), copy.deepcopy(scenarios)
    old.cfg, old.seed, old.sources = old.protocol["training"], old.protocol["seed"], original_sources()
    for key in ("initialization_sha256", "initialization_source", "source_counters"):
        setattr(old, key, copy.deepcopy(payload[key]))
    old.envs = [old._environment() for _ in range(old.cfg["environments"])]
    old._validate_scenarios()
    PublicFeedbackTrainer.load_state_dict(old, payload)
    return old


def _prefix(env, episode_return, components):
    state = env.state
    return {"frame": state.frame, "team_deliveries": state.total_deliveries,
        "individual_deliveries": [a.deliveries_completed for a in state.agents],
        "shutdowns": state.shutdown_count, "collisions": state.robot_collision_events,
        "return": float(episode_return), "reward_components": dict(components)}


class PartnerMixTrainer(PublicFeedbackTrainer):
    """One observed arm, exact in-flight fork, new counters and new schema."""

    def __init__(self, protocol, scenarios, source_state, *, branch,
                 expected_source_state_sha256, source_checkpoint_sha256,
                 device="cpu", test_fixture=False):
        if branch not in BRANCHES or type(branch) is not str:
            raise ValueError("Unknown partner-mixture branch")
        _sha(expected_source_state_sha256, "source state")
        _sha(source_checkpoint_sha256, "source checkpoint")
        if not isinstance(source_state, dict) or initialization_sha256(source_state) != expected_source_state_sha256:
            raise ValueError("Original checkpoint trainer state differs from external digest")
        binding = {"checkpoint_sha256": source_checkpoint_sha256,
            "state_sha256": expected_source_state_sha256, "joint_steps": source_state.get("joint_steps")}
        _protocol(protocol, binding, test_fixture)
        if source_state.get("protocol") != protocol["source_protocol"]:
            raise ValueError("The original trainer protocol differs from the archived source protocol")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "mps") or self.device.index is not None:
            raise ValueError("A fixed CPU or MPS device is required")
        old = _load_original(source_state, scenarios, self.device, test_fixture)
        self.protocol, self.scenarios = copy.deepcopy(protocol), copy.deepcopy(scenarios)
        self.cfg, self.seed = self.protocol["training"], self.protocol["seed"]
        self.branch, self.mode, self.test_fixture = branch, "observed", test_fixture
        self.sources, self.initialization_sha256 = execution_sources(), expected_source_state_sha256
        self.source_checkpoint_sha256 = source_checkpoint_sha256
        self.source_public_history_counters = {key: source_state[key] for key in COUNTERS}
        self.source_r1_counters = copy.deepcopy(old.source_counters)
        self.source_counters = {key: old.source_counters[key] + source_state[key]
                               for key in ("joint_steps", "optimizer_updates", "minibatch_updates")}
        self.source_counters.update({role + "_optimizer_steps": old.source_counters["minibatch_updates"]
            + source_state[role + "_optimizer_steps"] for role in ("actor", "critic")})
        self.source_initialization_sha256 = old.initialization_sha256
        self.source_completed_episodes_sha256 = initialization_sha256(source_state["completed_episodes"])
        self.source_frames = [env.state.frame for env in old.envs]
        self.model, self.optimizers, self.envs, self.rng = old.model, old.optimizers, old.envs, old.rng
        self._owned_rng_state, self._train_ids = _cpu(old._owned_rng_state), old._train_ids.copy()
        for key in EPISODE_FIELDS: setattr(self, key, copy.deepcopy(getattr(old, key)))
        self.round_episode_prefixes = [_prefix(env, self.episode_returns[i], self.episode_reward_components[i])
                                      for i, env in enumerate(self.envs)]
        self.completed_episodes = []
        for key in COUNTERS: setattr(self, key, 0)
        self.elapsed_seconds, self.feedback_enabled, self._collecting = 0., False, False

    def reset_one(self, i):
        # This is reached only by the frozen collector's normal terminal path.
        if self._collecting and self.completed_episodes:
            record, prefix, state = self.completed_episodes[-1], self.round_episode_prefixes[i], self.envs[i].state
            record["metric_scope"] = "whole_episode_including_any_pre_fork_prefix"
            record["round_metrics"] = {
                "start_frame": prefix["frame"], "length": state.frame - prefix["frame"],
                "team_deliveries": state.total_deliveries - prefix["team_deliveries"],
                "individual_deliveries": [a.deliveries_completed - prefix["individual_deliveries"][j]
                    for j, a in enumerate(state.agents)],
                "shutdowns": state.shutdown_count - prefix["shutdowns"],
                "collisions": state.robot_collision_events - prefix["collisions"],
                "return": self.episode_returns[i] - prefix["return"],
                "reward_components": {key: self.episode_reward_components[i].get(key, 0.)
                    - prefix["reward_components"].get(key, 0.)
                    for key in self.episode_reward_components[i].keys() | prefix["reward_components"].keys()},
            }
        entries = self.scenarios["splits"]["train"]
        entry = entries[int(self.rng.integers(len(entries)))]
        reset_scenario(self.envs[i], entry)
        self.scenario_ids[i] = entry["id"]
        mixture = self.protocol["partners_by_branch"][self.branch]
        self.partner_kinds[i] = str(self.rng.choice(list(mixture), p=list(mixture.values())))
        self.program_roles[i] = -1 if self.partner_kinds[i] == "selfplay" else int(self.rng.integers(2))
        self.episode_returns[i], self.episode_reward_components[i] = 0., Counter()
        self.episode_context[i] = {"source": "original_train", "curriculum_id": None, "category": None,
            "start_frame": 0, "prefix": {"team_deliveries": self.envs[i].state.total_deliveries,
                "individual_deliveries": [a.deliveries_completed for a in self.envs[i].state.agents],
                "shutdowns": self.envs[i].state.shutdown_count, "collisions": self.envs[i].state.robot_collision_events}}
        self.round_episode_prefixes[i] = _prefix(self.envs[i], 0., {})
        self.episode_count += 1

    def collect(self, time_steps):
        _integer(time_steps, "Rollout length", 1)
        if self.joint_steps + time_steps * len(self.envs) > self.protocol["budget"]["maximum_ppo_joint_steps_per_arm"]:
            raise ValueError("Collect exceeds the new per-arm round limit")
        self._collecting = True
        try: batch = PublicFeedbackTrainer.collect(self, time_steps)
        finally: self._collecting = False
        for record in batch["transition_records"]:
            record.update(version=VERSION, branch=self.branch,
                source_cumulative_joint_steps=self.source_counters["joint_steps"],
                cumulative_joint_step=self.source_counters["joint_steps"] + record["joint_step"])
        return batch

    def update(self, batch):
        with self._rng_context(): result = NativeV2Trainer.update(self, batch)
        for role in ("actor", "critic"):
            current = int(getattr(self.optimizers, role).state_dict()["state"][0]["step"].item())
            setattr(self, role + "_optimizer_steps", current - self.source_counters[role + "_optimizer_steps"])
        return result

    def train_chunk(self, time_steps):
        """Caller reserves first and persists batch/checkpoint before acknowledging."""
        batch = self.collect(time_steps)
        return batch, self.update(batch)

    def _bindings(self):
        return {"version": VERSION, "branch": self.branch, "public_feedback_version": OBSERVER_VERSION,
            "public_feedback_mode": "observed", "training_device": self.device.type,
            "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios), "sources": copy.deepcopy(self.sources),
            "source_sha256": digest(self.sources), "initialization_sha256": self.initialization_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_counters": copy.deepcopy(self.source_counters),
            "source_public_history_counters": copy.deepcopy(self.source_public_history_counters),
            "source_r1_counters": copy.deepcopy(self.source_r1_counters),
            "source_initialization_sha256": self.source_initialization_sha256,
            "source_completed_episodes_sha256": self.source_completed_episodes_sha256,
            "source_frames": self.source_frames.copy(), "obs_dim": OBSERVATION_SIZE,
            "state_dim": GLOBAL_STATE_SIZE, "feature_names": list(self.envs[0].feature_names),
            "reward_revision": REWARD_REVISION, "test_fixture": self.test_fixture,
            "curriculum_generation_steps": 0, "feedback_enabled": False}

    def state_dict(self):
        if execution_sources() != self.sources: raise ValueError("Execution sources changed")
        result = {**self._bindings(), "model": _cpu(self.model.state_dict()),
            "optimizers": _cpu(self.optimizers.state_dict()), "rng": copy.deepcopy(self.rng.bit_generator.state),
            **_cpu(self._owned_rng_state), "envs": [env.snapshot() for env in self.envs],
            "completed_episodes": _cpu(self.completed_episodes), "elapsed_seconds": self.elapsed_seconds,
            "round_episode_prefixes": copy.deepcopy(self.round_episode_prefixes)}
        for key in (*COUNTERS, *EPISODE_FIELDS): result[key] = _cpu(getattr(self, key))
        result["episode_reward_components"] = [dict(item) for item in self.episode_reward_components]
        return result

    def load_state_dict(self, payload):
        """Stage all validation before touching this arm's live model or RNG."""
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid checkpoint or changed execution sources")
        if any(payload.get(k) != v for k, v in self._bindings().items()):
            raise ValueError("Partner-mixture checkpoint identity/device/source differs")
        p, count = copy.deepcopy(payload), len(self.envs)
        for key in COUNTERS: _integer(p.get(key), key)
        if (p["joint_steps"] % count or p["joint_steps"] > self.protocol["budget"]["maximum_ppo_joint_steps_per_arm"]
                or p["optimizer_updates"] > p["minibatch_updates"]
                or p["actor_optimizer_steps"] > p["minibatch_updates"]
                or p["critic_optimizer_steps"] != p["minibatch_updates"]
                or p["last_evaluated_joint_steps"] > p["joint_steps"]):
            raise ValueError("Inconsistent continuation counters")
        _finite(p.get("elapsed_seconds"), "elapsed seconds")
        if p["elapsed_seconds"] < 0: raise ValueError("Negative elapsed time")
        _validate_learning_state(p.get("model"), p.get("optimizers"), self.cfg,
            {role: self.source_counters[role + "_optimizer_steps"] + p[role + "_optimizer_steps"]
             for role in ("actor", "critic")})
        for key in (*EPISODE_FIELDS, "envs", "round_episode_prefixes"):
            if not isinstance(p.get(key), (list, np.ndarray)) or len(p[key]) != count:
                raise ValueError("Invalid continuation vector: " + key)
        returns = np.asarray(p["episode_returns"])
        if returns.shape != (count,) or returns.dtype != np.float64 or not np.isfinite(returns).all():
            raise ValueError("Invalid episode returns")
        envs = []
        for i, snapshot in enumerate(p["envs"]):
            kind, role = p["partner_kinds"][i], p["program_roles"][i]
            if (kind not in PARTNER_MIXES[self.branch] or type(role) is not int
                    or role not in ((-1,) if kind == "selfplay" else (0, 1))
                    or p["scenario_ids"][i] not in self._train_ids):
                raise ValueError("Invalid partner or scenario identity")
            entry = next(x for x in self.scenarios["splits"]["train"] if x["id"] == p["scenario_ids"][i])
            baseline = self._environment(); reset_scenario(baseline, entry)
            expected = {"source": "original_train", "curriculum_id": None, "category": None, "start_frame": 0,
                "prefix": {"team_deliveries": baseline.state.total_deliveries,
                    "individual_deliveries": [a.deliveries_completed for a in baseline.state.agents],
                    "shutdowns": baseline.state.shutdown_count, "collisions": baseline.state.robot_collision_events}}
            if p["episode_context"][i] != expected: raise ValueError("Original episode context differs")
            components = p["episode_reward_components"][i]
            if not isinstance(components, dict): raise ValueError("Invalid reward accounting")
            for k, v in components.items():
                if type(k) is not str: raise ValueError("Invalid reward component name")
                _finite(v, k)
            env = self._environment()
            if snapshot.get("training_reward_revision") != REWARD_REVISION:
                raise ValueError("Missing revised reward metadata")
            env.restore(snapshot, require_feedback=True)
            if env.done or env.state.frame > self.source_frames[i] + p["joint_steps"] // count:
                raise ValueError("Invalid in-flight frame")
            if env.state.frame and not env.public_history()["valid"]:
                raise ValueError("Missing confirmed public history")
            prefix = p["round_episode_prefixes"][i]
            if not isinstance(prefix, dict) or set(prefix) != set(_prefix(env, 0., {})):
                raise ValueError("Invalid source episode prefix")
            for key in ("frame", "team_deliveries", "collisions", "shutdowns"):
                _integer(prefix[key], key)
            if (prefix["frame"] > env.state.frame
                    or not isinstance(prefix["individual_deliveries"], list)
                    or len(prefix["individual_deliveries"]) != 2):
                raise ValueError("Source prefix exceeds current state")
            for number in prefix["individual_deliveries"]: _integer(number, "prefix delivery")
            _finite(prefix["return"], "prefix return")
            if not isinstance(prefix["reward_components"], dict): raise ValueError("Invalid prefix rewards")
            for value in prefix["reward_components"].values(): _finite(value, "prefix component")
            envs.append(env)
        if not isinstance(p.get("completed_episodes"), list): raise ValueError("Invalid episode journal")
        initialization_sha256(p["completed_episodes"])
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(p["rng"])
        owned = {key: p[key] for key in ("python_rng", "numpy_rng", "torch_rng")}
        if self.device.type == "mps": owned["mps_rng"] = p["mps_rng"]
        elif "mps_rng" in p: raise ValueError("CPU checkpoint cannot carry MPS RNG")
        for key in ("torch_rng", *(("mps_rng",) if self.device.type == "mps" else ())):
            v = owned[key]
            if not torch.is_tensor(v) or v.dtype != torch.uint8 or v.device.type != "cpu" or v.ndim != 1:
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
        if execution_sources() != self.sources: raise ValueError("Execution sources changed")
        metadata = self._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, joint_steps=self.joint_steps,
            optimizer_updates=self.optimizer_updates, minibatch_updates=self.minibatch_updates,
            initialization="exact_observed_endpoint_with_inflight_episodes", candidate=True)
        return self.model.export_npz(path, metadata)
