"""Matched public-history PPO arm; no CLI, sampling authorization or qualification.

The frozen v2 collect/update implementations are reused literally. This module
owns fresh episode starts and its distinct checkpoint schema; an r1 in-flight
snapshot is evidence in the audited initialization, never a new training start.
Callers must reserve a separately authorized budget before collect. Instances
run sequentially: their private RNG contexts preserve the caller's globals,
not a promise that Python/PyTorch global generators are thread-safe.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch

from env.warehouse_native.policy import NativeActorCritic, ACTIONS
from env.warehouse_native.scenarios import SCENARIO_VERSION, reset_scenario
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_v2 import NativeV2Trainer, IndependentOptimizers
from .warehouse_native_revision import revision_source_hashes
from .warehouse_native_revision_reward import REWARD_REVISION
from .warehouse_native_public_feedback import (
    PublicFeedbackEnvironment, VERSION as PUBLIC_FEEDBACK_VERSION, MODES,
    OBSERVATION_SIZE, GLOBAL_STATE_SIZE, BASE_OBSERVATION_SIZE,
)
from .warehouse_native_public_feedback_initialization import (
    VERSION as INITIALIZATION_VERSION, initialization_sha256,
    migrate_public_feedback_initialization,
)

VERSION = "warehouse-native-public-feedback-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-public-history-pair.proposal.v1"
RNG_STREAM = 601
COUNTERS = ("joint_steps", "optimizer_updates", "minibatch_updates", "actor_optimizer_steps",
            "critic_optimizer_steps", "episode_count", "last_evaluated_joint_steps")
EPISODE_FIELDS = ("partner_kinds", "program_roles", "scenario_ids", "episode_context",
                  "episode_returns", "episode_reward_components")


def execution_sources():
    """The original 59-file closure plus this trainer and its two new contracts."""
    result = revision_source_hashes()
    for name in ("warehouse_native_public_feedback.py", "warehouse_native_public_feedback_initialization.py",
                 "warehouse_native_public_feedback_trainer.py"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list): return [_cpu(item) for item in value]
    if isinstance(value, tuple): return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(name + " must be finite")


def _global_rng(device):
    result = {"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
              "torch_rng": torch.get_rng_state().clone()}
    if device.type == "mps": result["mps_rng"] = torch.mps.get_rng_state().cpu().clone()
    return result


def _set_global_rng(state, device):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "mps": torch.mps.set_rng_state(state["mps_rng"])


@contextmanager
def _preserve_global_rng(device):
    previous = _global_rng(device)
    try: yield
    finally: _set_global_rng(previous, device)


def _validate_tensor(value, shape, name):
    if (not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.layout != torch.strided or tuple(value.shape) != tuple(shape)
            or not torch.isfinite(value).all().item()):
        raise ValueError("Invalid checkpoint float32 tensor: " + name)


def _shapes():
    actor = {"0.weight": (128, 197), "0.bias": (128,), "2.weight": (128, 128),
             "2.bias": (128,), "4.weight": (5, 128), "4.bias": (5,)}
    critic = {**actor, "0.weight": (128, 356), "4.weight": (1, 128), "4.bias": (1,)}
    return {"actor": actor, "critic": critic}


def _validate_learning_state(model, optimizers, cfg, steps):
    shapes = _shapes()
    expected = {f"{role}.{name}": shape for role, values in shapes.items() for name, shape in values.items()}
    if not isinstance(model, dict) or set(model) != set(expected):
        raise ValueError("Checkpoint requires exactly the two native MLPs")
    for name, shape in expected.items(): _validate_tensor(model[name], shape, name)
    if not isinstance(optimizers, dict) or set(optimizers) != set(shapes):
        raise ValueError("Checkpoint requires both independent Adam optimizers")
    for role, role_shapes in shapes.items():
        opt = optimizers[role]
        if not isinstance(opt, dict) or set(opt) != {"state", "param_groups"}:
            raise ValueError("Incomplete optimizer state: " + role)
        groups = opt["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
            raise ValueError("Each independent Adam requires one parameter group")
        group = groups[0]
        if (group.get("params") != list(range(6)) or any(type(x) is not int for x in group["params"])
                or not isinstance(opt["state"], dict) or set(opt["state"]) != set(range(6))
                or any(type(x) is not int for x in opt["state"])
                or group.get("lr") != cfg[role + "_learning_rate"]
                or group.get("eps") != 1e-5 or tuple(group.get("betas", ())) != (.9, .999)
                or group.get("weight_decay") != 0
                or any(group.get(key, False) is not False for key in
                       ("amsgrad", "maximize", "differentiable", "decoupled_weight_decay", "capturable"))
                or group.get("foreach") is not None or group.get("fused") is not None):
            raise ValueError("Frozen Adam configuration differs: " + role)
        for index, (name, shape) in enumerate(role_shapes.items()):
            state = opt["state"][index]
            if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Incomplete Adam moments: " + role + "." + name)
            _validate_tensor(state["step"], (), role + ".step")
            if state["step"].item() != steps[role]:
                raise ValueError("Adam step differs from persisted applied-update count: " + role)
            for field in ("exp_avg", "exp_avg_sq"):
                _validate_tensor(state[field], shape, role + "." + field)
            if (state["exp_avg_sq"] < 0).any().item():
                raise ValueError("Adam variance cannot be negative")


def _validate_protocol(protocol, test_fixture):
    if type(test_fixture) is not bool or not isinstance(protocol, dict):
        raise ValueError("Protocol and explicit fixture mode are required")
    if protocol.get("version") != PROTOCOL_VERSION or protocol.get("test_fixture", False) is not test_fixture:
        raise ValueError("Public-history protocol identity or fixture mode differs")
    if (protocol.get("public_feedback_version") != PUBLIC_FEEDBACK_VERSION
            or protocol.get("arms") != ["control", "observed"]
            or protocol.get("seed") != 260908 or type(protocol.get("seed")) is not int
            or protocol.get("feedback_training_enabled") is not False
            or protocol.get("collision_training_cost") != .05
            or protocol.get("budget", {}).get("curriculum_generation_steps") != 0):
        raise ValueError("Public-history observation-only experiment contract differs")
    cfg = protocol["training"]
    for key in ("environments", "rollout_steps", "epochs", "minibatch", "checkpoint_interval"):
        _integer(cfg.get(key), "training." + key, 1)
    fixed = {"learning_rate": 3e-4, "actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4,
             "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy": .01,
             "actor_gradient_norm": .5, "critic_gradient_norm": .5}
    if any(cfg.get(key) != value or type(cfg.get(key)) not in (float, int) for key, value in fixed.items()):
        raise ValueError("Frozen PPO parameters differ")
    if not test_fixture and any(cfg[key] != value for key, value in
                               {"environments": 16, "rollout_steps": 128, "epochs": 4,
                                "minibatch": 512, "checkpoint_interval": 50000}.items()):
        raise ValueError("Production PPO dimensions differ")
    partners = protocol.get("partners", {})
    if (not isinstance(partners, dict) or not partners or set(partners) - {"selfplay", "skilled", "assertive", "noisy"}
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in partners.values())
            or not math.isclose(sum(partners.values()), 1., abs_tol=1e-12, rel_tol=0)):
        raise ValueError("Invalid partner mixture")
    if not test_fixture and partners != {"selfplay": .5, "skilled": .2, "assertive": .2, "noisy": .1}:
        raise ValueError("Production partner mixture differs")
    expected_rng = {"python": 260908, "numpy_global": 260908, "torch_cpu": 260908,
                    "torch_mps": 260908, "sampler_seed_sequence": [260908, RNG_STREAM],
                    "after_weight_loading": True, "independent_mutable_state_per_arm": True,
                    "diverging_trajectories_expected": True}
    if protocol.get("rng_initialization") != expected_rng:
        raise ValueError("Matched-arm RNG initialization differs")
    initial = protocol.get("initialization", {})
    if (initial.get("initialization_migration_version") != INITIALIZATION_VERSION
            or initial.get("old_actor_input_size") != 177 or initial.get("new_actor_input_size") != 197
            or initial.get("critic_public_input_size") != 354
            or initial.get("new_episode_starts") != "train_original_snapshots_only_no_curriculum"):
        raise ValueError("Fresh-episode initialization contract differs")


class PublicFeedbackTrainer(NativeV2Trainer):
    """One fully independent arm. Construction/restoration collect zero steps."""

    def __init__(self, protocol, scenarios, initialization, *, mode,
                 expected_initialization_sha256, device="cpu", test_fixture=False):
        _validate_protocol(protocol, test_fixture)
        if mode not in MODES or type(mode) is not str:
            raise ValueError("Unknown public-feedback arm")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "mps") or self.device.index is not None:
            raise ValueError("Only a fixed CPU or MPS device is supported")
        self.protocol, self.scenarios = copy.deepcopy(protocol), copy.deepcopy(scenarios)
        self.mode, self.test_fixture = mode, test_fixture
        self.cfg, self.seed = self.protocol["training"], self.protocol["seed"]
        self.sources = execution_sources()
        self.initialization_sha256 = initialization_sha256(initialization)
        if self.initialization_sha256 != expected_initialization_sha256:
            raise ValueError("Audited migration digest differs")
        initial = copy.deepcopy(initialization)
        if (initial.get("version") != INITIALIZATION_VERSION
                or initial.get("observer_version") != PUBLIC_FEEDBACK_VERSION
                or initial.get("test_fixture") is not test_fixture
                or initial.get("obs_dim") != OBSERVATION_SIZE or initial.get("state_dim") != GLOBAL_STATE_SIZE):
            raise ValueError("An audited public-history initialization is required")
        source = initial["source"]
        if (source["checkpoint_sha256"] != protocol["initialization"]["checkpoint_sha256"]
                or source["joint_steps"] != protocol["initialization"]["source_joint_steps"]):
            raise ValueError("Initialization differs from the prescribed source endpoint")
        # Reconstruct and validate the exact old semantic payload using the pure
        # helper. The independently supplied outer digest is the trust anchor;
        # self-reported hashes inside the initialization cannot substitute it.
        original = copy.deepcopy(initial["source_resume_evidence"])
        original["model"] = copy.deepcopy(initial["model"])
        original["model"]["actor.0.weight"] = original["model"]["actor.0.weight"][:, :177].clone()
        original["optimizers"] = copy.deepcopy(initial["optimizers"])
        for name in ("exp_avg", "exp_avg_sq"):
            original["optimizers"]["actor"]["state"][0][name] = original["optimizers"]["actor"]["state"][0][name][:, :177].clone()
        if original["revision_sources"] != revision_source_hashes():
            raise ValueError("Initialization r1 execution sources differ from the frozen closure")
        checked = migrate_public_feedback_initialization(original,
            source_checkpoint_sha256=source["checkpoint_sha256"],
            expected_source_checkpoint_sha256=protocol["initialization"]["checkpoint_sha256"],
            expected_initialization_sha256=source["semantic_payload_sha256"],
            expected_protocol=original["protocol"], expected_sources=revision_source_hashes(),
            source_feature_names=initial["source_feature_names"], expected_feature_names=initial["feature_names"],
            expected_joint_steps=protocol["initialization"]["source_joint_steps"], allow_test_fixture=test_fixture)
        if initialization_sha256(checked) != self.initialization_sha256:
            raise ValueError("Migration contents do not preserve the exact audited source")
        if (self.protocol["reward"] != original["protocol"]["reward"]
                or self.protocol["collision_training_cost"] != original["protocol"]["experiment_revision"]["collision_training_cost"]):
            raise ValueError("The public-history arms cannot change the r1 training reward")
        self.source_counters = {key: source[key] for key in ("joint_steps", "optimizer_updates", "minibatch_updates")}
        self.initialization_source = copy.deepcopy(source)
        self.envs = [self._environment() for _ in range(self.cfg["environments"])]
        if list(self.envs[0].feature_names) != initial["feature_names"]:
            raise ValueError("Migrated observation order differs from the actual observer")
        self._validate_scenarios()
        _validate_learning_state(initial["model"], initial["optimizers"], self.cfg,
                                 {role: source["minibatch_updates"] for role in ("actor", "critic")})
        with _preserve_global_rng(self.device):
            self.model, self.optimizers = self._new_learning_state(initial["model"], initial["optimizers"])
            # The old checkpoint's RNG is deliberately not installed.
            random.seed(self.seed); np.random.seed(self.seed)
            torch.random.default_generator.manual_seed(self.seed)
            if self.device.type == "mps": torch.mps.manual_seed(self.seed)
            self._owned_rng_state = _global_rng(self.device)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, RNG_STREAM]))
        count = len(self.envs)
        self.partner_kinds, self.program_roles, self.scenario_ids = [""] * count, [-1] * count, [""] * count
        self.episode_context = [None] * count
        self.episode_returns = np.zeros(count, dtype=np.float64)
        self.episode_reward_components = [Counter() for _ in range(count)]
        self.completed_episodes = []
        for key in COUNTERS: setattr(self, key, 0)
        self.elapsed_seconds = 0.
        self.feedback_enabled = False
        for i in range(count): self.reset_one(i)

    def _environment(self):
        return PublicFeedbackEnvironment(reward_config=self.protocol["reward"],
            collision_cost=self.protocol["collision_training_cost"], mode=self.mode)

    def _validate_scenarios(self):
        manifest = self.scenarios
        if (not isinstance(manifest, dict) or manifest.get("version") != SCENARIO_VERSION
                or manifest.get("configuration") != asdict(self.envs[0].config)):
            raise ValueError("Scenario manifest differs from the original warehouse configuration")
        splits = manifest.get("splits", {})
        if not isinstance(splits, dict) or not splits.get("train"):
            raise ValueError("Original training scenarios are required")
        if "split" in manifest and manifest["split"] != splits:
            raise ValueError("Scenario manifest split aliases disagree")
        seen, ids = set(), set()
        for split, entries in splits.items():
            if not isinstance(entries, list) or manifest.get("counts", {}).get(split) != len(entries):
                raise ValueError("Scenario split counts differ")
            for entry in entries:
                if entry["fingerprint"] in seen or entry["id"] in ids:
                    raise ValueError("Scenario identities or physical initial states overlap")
                seen.add(entry["fingerprint"]); ids.add(entry["id"])
        if not self.test_fixture and len(splits["train"]) != 512:
            raise ValueError("Production arms require the original 512 training scenarios")
        validator = self._environment()
        for entry in splits["train"]:
            if entry["snapshot"].get("state", {}).get("frame") != 0:
                raise ValueError("Only original frame-zero training starts are allowed")
            reset_scenario(validator, entry)
            if validator.done or validator.public_history()["valid"]:
                raise ValueError("A new training start cannot carry prior public history")
        self._train_ids = {entry["id"] for entry in splits["train"]}

    def _new_learning_state(self, model, optimizers):
        network = NativeActorCritic(OBSERVATION_SIZE, GLOBAL_STATE_SIZE).to(self.device)
        network.load_state_dict(model, strict=True)
        optimizer = IndependentOptimizers(network, self.cfg)
        optimizer.load_state_dict(copy.deepcopy(optimizers))
        return network, optimizer

    def reset_one(self, i):
        entries = self.scenarios["splits"]["train"]
        entry = entries[int(self.rng.integers(len(entries)))]
        env = self.envs[i]
        reset_scenario(env, entry)
        self.scenario_ids[i] = entry["id"]
        mix = self.protocol["partners"]
        self.partner_kinds[i] = str(self.rng.choice(list(mix), p=list(mix.values())))
        self.program_roles[i] = -1 if self.partner_kinds[i] == "selfplay" else int(self.rng.integers(2))
        self.episode_returns[i] = 0.
        self.episode_reward_components[i] = Counter()
        self.episode_context[i] = {"source": "original_train", "curriculum_id": None, "category": None,
            "start_frame": 0, "prefix": {"team_deliveries": env.state.total_deliveries,
                "individual_deliveries": [a.deliveries_completed for a in env.state.agents],
                "shutdowns": env.state.shutdown_count, "collisions": env.state.robot_collision_events}}
        self.episode_count += 1

    @contextmanager
    def _rng_context(self):
        with _preserve_global_rng(self.device):
            _set_global_rng(self._owned_rng_state, self.device)
            try: yield
            finally: self._owned_rng_state = _global_rng(self.device)

    def collect(self, time_steps):
        _integer(time_steps, "Rollout length", 1)
        if time_steps > self.cfg["rollout_steps"]:
            raise ValueError("Collect cannot exceed one configured rollout")
        records, latest = [], {}
        original_infer = self.infer
        prior_infer = self.__dict__.get("infer")
        step_bindings = [(env, env.__dict__.get("step"), env.step) for env in self.envs]
        def infer(observations, states):
            probabilities, values = original_infer(observations, states)
            latest["probabilities"] = probabilities.copy()
            return probabilities, values
        def record_step(index, env, original):
            def step(actions, *, decision_metadata=None):
                before = env.snapshot()
                result = original(actions, decision_metadata=decision_metadata)
                _, rewards, terminated, truncated, info = result
                records.append({"version": VERSION, "public_feedback_mode": self.mode,
                    "environment_index": index, "rollout_index": len(records) // len(self.envs),
                    "joint_step": self.joint_steps + 1, "scenario_id": self.scenario_ids[index],
                    "partner": self.partner_kinds[index], "program_role": self.program_roles[index],
                    "before": before, "after": env.snapshot(),
                    "neural_probabilities": latest["probabilities"][index].tolist(),
                    "requested_actions": copy.deepcopy(info["requested_actions"]),
                    "executed_actions": copy.deepcopy(info["executed_actions"]),
                    "rewards": copy.deepcopy(rewards), "terminated": terminated, "truncated": truncated,
                    "events": copy.deepcopy(info["events"]), "info": copy.deepcopy(info)})
                return result
            return step
        self.infer = infer
        for i, (env, _, original) in enumerate(step_bindings): env.step = record_step(i, env, original)
        try:
            with self._rng_context(): batch = NativeV2Trainer.collect(self, time_steps)
        finally:
            if prior_infer is None: self.__dict__.pop("infer", None)
            else: self.infer = prior_infer
            for env, previous, _ in step_bindings:
                if previous is None: env.__dict__.pop("step", None)
                else: env.step = previous
        for record in records:
            t, i = record["rollout_index"], record["environment_index"]
            record["neural_sampled_actions"] = [ACTIONS[int(x)] for x in batch["actions"][t, i]]
            record["neural_old_log_probs"] = batch["old_log_probs"][t, i].tolist()
            record["trainable"] = batch["trainable"][t, i].tolist()
            for j, key in enumerate(self.envs[i].agent_ids):
                if record["trainable"][j] and record["requested_actions"][key] != record["neural_sampled_actions"][j]:
                    raise RuntimeError("A trainable NN command was changed before physics")
        batch["transition_records"] = records
        return batch

    def update(self, batch):
        with self._rng_context():
            result = NativeV2Trainer.update(self, batch)
        source_steps = self.source_counters["minibatch_updates"]
        self.actor_optimizer_steps = int(self.optimizers.actor.state_dict()["state"][0]["step"].item()) - source_steps
        self.critic_optimizer_steps = int(self.optimizers.critic.state_dict()["state"][0]["step"].item()) - source_steps
        return result

    def _bindings(self):
        return {"version": VERSION, "public_feedback_version": PUBLIC_FEEDBACK_VERSION,
            "public_feedback_mode": self.mode, "training_device": self.device.type,
            "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios), "sources": copy.deepcopy(self.sources),
            "source_sha256": digest(self.sources), "initialization_sha256": self.initialization_sha256,
            "initialization_source": copy.deepcopy(self.initialization_source),
            "source_counters": copy.deepcopy(self.source_counters), "obs_dim": OBSERVATION_SIZE,
            "state_dim": GLOBAL_STATE_SIZE, "feature_names": list(self.envs[0].feature_names),
            "reward_revision": REWARD_REVISION, "test_fixture": self.test_fixture,
            "curriculum_generation_steps": 0, "feedback_enabled": False}

    def state_dict(self):
        if execution_sources() != self.sources:
            raise ValueError("Execution sources changed during this arm")
        result = {**self._bindings(), "model": _cpu(self.model.state_dict()),
            "optimizers": _cpu(self.optimizers.state_dict()), "rng": copy.deepcopy(self.rng.bit_generator.state),
            **_cpu(self._owned_rng_state), "envs": [env.snapshot() for env in self.envs],
            "completed_episodes": _cpu(self.completed_episodes), "elapsed_seconds": self.elapsed_seconds}
        for key in (*COUNTERS, *EPISODE_FIELDS): result[key] = _cpu(getattr(self, key))
        result["episode_reward_components"] = [dict(item) for item in self.episode_reward_components]
        return result

    def load_state_dict(self, payload):
        """Validate a detached candidate completely before replacing any live state."""
        if not isinstance(payload, dict): raise ValueError("Checkpoint must be a dictionary")
        if execution_sources() != self.sources: raise ValueError("Execution sources changed")
        bindings = self._bindings()
        if any(payload.get(key) != value for key, value in bindings.items()):
            raise ValueError("Public-history checkpoint identity, protocol, device or source binding differs")
        p = copy.deepcopy(payload)
        for key in COUNTERS: _integer(p.get(key), key)
        count = len(self.envs)
        if (p["joint_steps"] % count or p["optimizer_updates"] > p["minibatch_updates"]
                or p["actor_optimizer_steps"] > p["minibatch_updates"]
                or p["critic_optimizer_steps"] != p["minibatch_updates"]
                or p["last_evaluated_joint_steps"] > p["joint_steps"] or p["episode_count"] < count):
            raise ValueError("Saved training counters are inconsistent")
        _finite(p.get("elapsed_seconds"), "elapsed_seconds")
        if p["elapsed_seconds"] < 0: raise ValueError("Elapsed seconds cannot be negative")
        source_steps = self.source_counters["minibatch_updates"]
        _validate_learning_state(p.get("model"), p.get("optimizers"), self.cfg,
            {role: source_steps + p[role + "_optimizer_steps"] for role in ("actor", "critic")})
        for key in (*EPISODE_FIELDS, "envs"):
            if not isinstance(p.get(key), (list, np.ndarray)) or len(p[key]) != count:
                raise ValueError("Saved episode vector length differs: " + key)
        returns = np.asarray(p["episode_returns"])
        if returns.shape != (count,) or returns.dtype != np.float64 or not np.isfinite(returns).all():
            raise ValueError("Invalid episode returns")
        envs = []
        for i, snapshot in enumerate(p["envs"]):
            kind, role = p["partner_kinds"][i], p["program_roles"][i]
            if (kind not in self.protocol["partners"] or type(role) is not int
                    or role not in ((-1,) if kind == "selfplay" else (0, 1))
                    or p["scenario_ids"][i] not in self._train_ids):
                raise ValueError("Invalid persisted episode partner or training scenario")
            context = p["episode_context"][i]
            entry = next(x for x in self.scenarios["splits"]["train"] if x["id"] == p["scenario_ids"][i])
            baseline = self._environment(); reset_scenario(baseline, entry)
            expected = {"source": "original_train", "curriculum_id": None, "category": None, "start_frame": 0,
                "prefix": {"team_deliveries": baseline.state.total_deliveries,
                    "individual_deliveries": [a.deliveries_completed for a in baseline.state.agents],
                    "shutdowns": baseline.state.shutdown_count, "collisions": baseline.state.robot_collision_events}}
            if context != expected: raise ValueError("Persisted episode context is not an original training start")
            components = p["episode_reward_components"][i]
            if not isinstance(components, dict) or any(not isinstance(k, str) for k in components):
                raise ValueError("Invalid reward-component accounting")
            for key, value in components.items(): _finite(value, "reward component " + key)
            env = self._environment()
            # Stripped metadata cannot silently erase feedback in a saved run.
            if snapshot.get("training_reward_revision") != REWARD_REVISION:
                raise ValueError("Persisted reward revision is missing")
            env.restore(snapshot, require_feedback=True)
            if env.done or env.state.frame > p["joint_steps"] // count:
                raise ValueError("Invalid persisted in-flight frame")
            if env.state.frame and not env.public_history()["valid"]:
                raise ValueError("In-flight training requires confirmed public history")
            envs.append(env)
        if not isinstance(p.get("completed_episodes"), list):
            raise ValueError("Completed-episode journal must be a list")
        # Typed finite hashing validates detached journal data without trusting it
        # as qualification or a replacement for the caller's raw event log.
        initialization_sha256(p["completed_episodes"])
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(p["rng"])
        owned = {key: p[key] for key in ("python_rng", "numpy_rng", "torch_rng")}
        if self.device.type == "mps": owned["mps_rng"] = p["mps_rng"]
        elif "mps_rng" in p: raise ValueError("CPU checkpoint cannot carry an MPS resume stream")
        for key in ("torch_rng", *(('mps_rng',) if self.device.type == "mps" else ())):
            value = owned[key]
            if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.device.type != "cpu" or value.ndim != 1:
                raise ValueError("Invalid persisted RNG tensor")
        with _preserve_global_rng(self.device):
            _set_global_rng(owned, self.device)
            model, optimizers = self._new_learning_state(p["model"], p["optimizers"])
        # All validation and staging succeeded. Assignment below cannot invoke
        # another restore, consume randomness, load weights or advance physics.
        self.model, self.optimizers, self.envs, self.rng = model, optimizers, envs, rng
        self._owned_rng_state = _cpu(owned)
        for key in (*COUNTERS, *EPISODE_FIELDS, "elapsed_seconds", "completed_episodes"):
            setattr(self, key, copy.deepcopy(p[key]))
        self.episode_returns = returns.copy()
        self.episode_reward_components = [Counter(x) for x in p["episode_reward_components"]]

    def export(self, path):
        if execution_sources() != self.sources: raise ValueError("Execution sources changed")
        return self.model.export_npz(path, {"experiment_version": VERSION,
            "public_feedback_version": PUBLIC_FEEDBACK_VERSION, "public_feedback_mode": self.mode,
            "obs_dim": OBSERVATION_SIZE, "state_dim": GLOBAL_STATE_SIZE,
            "feature_names": list(self.envs[0].feature_names), "joint_steps": self.joint_steps,
            "optimizer_updates": self.optimizer_updates, "minibatch_updates": self.minibatch_updates,
            "protocol_sha256": digest(self.protocol), "scenario_manifest_sha256": digest(self.scenarios),
            "initialization_sha256": self.initialization_sha256, "source_sha256": digest(self.sources),
            "source_checkpoint_sha256": self.initialization_source["checkpoint_sha256"],
            "source_counters": copy.deepcopy(self.source_counters), "training_device": self.device.type,
            "initialization": "exact_r1_migration_new_original_train_episodes",
            "reward_revision": REWARD_REVISION, "curriculum_generation_steps": 0,
            "feedback_enabled": False, "candidate": True, "test_fixture": self.test_fixture})
