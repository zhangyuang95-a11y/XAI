"""Dormant, capability-gated continuation PPO for the complete v6.20 NN.

Importing this file grants no constructor, reset, sample, or training authority.
Stage C must bind this *additional* source closure and passing Stage A/B
receipts.  The environment adapter and extraction/validation callbacks must be
included in that closure.  Extraction callbacks supply evidence distributions;
both environment actions are sampled exclusively from the learned Actors.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from .warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES, FrozenWarehouseV6Actor
from .warehouse_v6_actor_training import role_hashes
from .warehouse_v6_shutdown_cost import RiskDualConfig, ShutdownRiskDual
from .warehouse_v622_actor_training import (
    CRITIC_WARMUP_STEPS, ReferenceRow, TrainingState, masked_log_probabilities,
    masked_reference_kl, reference_logits_encoded, reference_kl_loss,
    verify_initial_equivalence,
)
from .warehouse_v622_continuation_safety import program_kl_loss
from .warehouse_v622_program import program_structure_sha256

SCHEMA = "warehouse-v625-ppo-driver.v1"
CAPABILITY_SCHEMA = "warehouse-v625-ppo-capability.v1"
BUFFER_SCHEMA = "warehouse-v625-initial-reference-training-buffer.v1"
ROLES = tuple(AGENTS)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def critic_initialization_sha256(model):
    digest = sha256()
    for name in ("reward_critic", "cost_critic"):
        for key, value in getattr(model, name).state_dict().items():
            array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
            digest.update((name + "." + key).encode())
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def validate_program_lock(lock):
    """Typed B-locked topology and shared source support; coefficients may vary."""
    if not isinstance(lock, dict) or set(lock) != set(ROLES):
        raise ValueError("both roles require a Stage B program language lock")
    for role, item in lock.items():
        if (not isinstance(item, dict) or set(item) != {"structure_sha256", "supports", "witnesses"}
                or not _is_sha(item["structure_sha256"]) or not isinstance(item["supports"], list)
                or not isinstance(item["witnesses"], list) or not 1 <= len(item["supports"]) <= 32
                or len(item["supports"]) != len(item["witnesses"])):
            raise ValueError("malformed Stage B program topology/support/witness lock")
        for support, witness in zip(item["supports"], item["witnesses"], strict=True):
            if (not isinstance(support, list) or any(type(i) is not int or not 0 <= i < FEATURE_DIM for i in support)
                    or support != sorted(set(support)) or len(support) > 3
                    or sum(2 if FEATURE_NAMES[i] == "public_hard_risk_context" else 1 for i in support) > 3
                    or type(witness) is not int or not 0 <= witness < 5):
                raise ValueError("invalid B-locked three-source shared support or fixed witness")
    return deepcopy(lock)


def validate_locked_programs(programs, lock, actor_role_sha256):
    """An extraction may refit outputs, never choose a new language on PPO data."""
    lock = validate_program_lock(lock)
    if not isinstance(programs, Mapping) or set(programs) != set(ROLES):
        raise RuntimeError("program extraction omitted a locked Actor role")
    for role, program in programs.items():
        program.validate()
        item = lock[role]
        if program.role != role or program.actor_role_sha256 != actor_role_sha256[role]:
            raise RuntimeError("online program is not bound to the current role Actor")
        if (program_structure_sha256(program.structure) != item["structure_sha256"]
                or [leaf["support"] for leaf in program.leaves] != item["supports"]
                or [leaf["witness"] for leaf in program.leaves] != item["witnesses"]):
            raise RuntimeError("online extraction changed the B-locked topology, shared support or witness")


@dataclass(frozen=True)
class PPOConfig:
    joint_step_budget: int = 20000
    parallel_environments: int = 8
    rollout_vector_steps: int = 128
    epochs: int = 4
    minibatch_joint_frames: int = 256
    actor_learning_rate: float = 3.e-5
    critic_learning_rate: float = 3.e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy_coefficient: float = .005
    critic_coefficient: float = .5
    maximum_gradient_norm: float = .5
    actor_warmup_joint_steps: int = 4000
    program_maximum_coefficient: float = .01
    episode_horizon: int = 120

    def __post_init__(self):
        if self.joint_step_budget not in (20000, 200000, 300000, 400000):
            raise ValueError("continuation budget must be a registered 20k/200k/300k/400k value")
        # Avoid recursively constructing a default config during its own
        # validation; all non-budget settings have one registered value.
        registered = {"parallel_environments": 8, "rollout_vector_steps": 128, "epochs": 4,
            "minibatch_joint_frames": 256, "actor_learning_rate": 3.e-5, "critic_learning_rate": 3.e-4,
            "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coefficient": .005,
            "critic_coefficient": .5, "maximum_gradient_norm": .5, "actor_warmup_joint_steps": 4000,
            "program_maximum_coefficient": .01, "episode_horizon": 120}
        if any(getattr(self, key) != value for key, value in registered.items()):
            raise ValueError("unregistered PPO hyperparameter change")

    @property
    def extraction_steps(self):
        return tuple(self.joint_step_budget * percentage // 100 for percentage in (75, 84, 93))

    def program_coefficient(self, joint_steps):
        fraction = joint_steps / self.joint_step_budget
        return self.program_maximum_coefficient * min(1., max(0., (fraction - .80) / .05))

    def next_collection_length(self, joint_steps):
        boundaries = [self.joint_step_budget, self.actor_warmup_joint_steps, *self.extraction_steps]
        future = [step for step in boundaries if step > joint_steps]
        if not future:
            return 0
        frames = min(future) - joint_steps
        if frames % self.parallel_environments:
            raise ValueError("registered boundary does not align with eight joint frames")
        return min(self.rollout_vector_steps, frames // self.parallel_environments)


class HashBoundTrainingCapability:
    """Finite, externally hash-bound Stage C authority with a predebit journal.

    Required receipt statuses and file hashes are declarations from the new
    Stage C issuer, not inferred from a filename.  Both A and B must explicitly
    be present and passed.  This loader does not issue that capability.
    """
    def __init__(self, path, expected_sha256, *, root, state, variant, config, journal_path):
        self.path, self.root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
        if not _is_sha(expected_sha256) or _sha(self.path) != expected_sha256:
            raise ValueError("training capability is missing or has a mismatched external hash")
        self.sha256 = expected_sha256
        self.value = json.loads(self.path.read_text())
        value = self.value
        if (value.get("schema") != CAPABILITY_SCHEMA or value.get("stage") != "C"
                or value.get("training_partition") != "train" or value.get("variant") != variant
                or variant not in ("raw", "regularized") or value.get("config") != asdict(config)
                or not isinstance(value.get("candidate_id"), str) or not value["candidate_id"].startswith("warehouse-v625-")
                or value.get("reference_sha256") != state.reference_sha256
                or value.get("initialization_seed") != state.initialization_seed
                or value.get("initial_role_sha256") != dict(state.reference.role_sha256)
                or value.get("initial_critic_sha256") != critic_initialization_sha256(state.model)
                or role_hashes(state.model) != dict(state.reference.role_sha256)
                or state.joint_steps != 0 or state.actor_updates != 0 or state.critic_updates != 0
                or state.actor_optimizer.state or state.critic_optimizer.state
                or state.beta != {r: 1. for r in ROLES}):
            raise ValueError("capability does not authorize this new paired reference initialization")
        self.config = config
        sources = value.get("source_files")
        if not isinstance(sources, dict) or not sources or value.get("source_sha256") != _digest(sources):
            raise ValueError("training source closure is absent or malformed")
        required = {"backend/training/warehouse_v625_actor_training.py", "backend/training/warehouse_v622_actor_training.py",
            "backend/training/warehouse_v622_continuation_safety.py", "backend/training/warehouse_v6_shutdown_cost.py"}
        if not required <= sources.keys():
            raise ValueError("new Stage C source closure must include its driver and safety helpers")
        if (self.root / "backend/training/warehouse_v625_actor_training.py").resolve() != Path(__file__).resolve():
            raise ValueError("training closure root does not contain the actually loaded driver")
        for name, sha in sources.items():
            file = (self.root / name).resolve(strict=True)
            if not file.is_relative_to(self.root) or not _is_sha(sha) or _sha(file) != sha:
                raise ValueError("training source hash mismatch")
        receipts = value.get("prerequisites")
        if not isinstance(receipts, list) or {r.get("stage") for r in receipts} != {"A", "B"}:
            raise ValueError("both passing Stage A and Stage B receipts are required")
        program_lock = validate_program_lock(value.get("program_lock"))
        for item in receipts:
            file = (self.root / item["path"]).resolve(strict=True)
            if not file.is_relative_to(self.root) or not _is_sha(item.get("sha256")) or _sha(file) != item["sha256"]:
                raise ValueError("prerequisite receipt hash mismatch")
            receipt = json.loads(file.read_text())
            if receipt.get("passed") is not True or receipt.get("candidate_id") != value["candidate_id"]:
                raise ValueError("a training prerequisite did not pass for this candidate")
            if item["stage"] == "B" and receipt.get("program_lock_sha256") != _digest(program_lock):
                raise ValueError("training program language differs from the passed Stage B lock")
        risk = value.get("risk_config")
        if risk != asdict(RiskDualConfig()):
            raise ValueError("risk dual settings differ from the registered complete-episode contract")
        cases = value.get("scene_schedule")
        if not isinstance(cases, list) or not cases:
            raise ValueError("training needs an immutable finite train scene schedule")
        by_slot, all_ids, all_seeds = defaultdict(list), set(), set()
        for case in cases:
            if (not isinstance(case, dict) or set(case) != {"id", "seed", "task", "slot", "ordinal", "partition"}
                    or case["partition"] != "train" or type(case["seed"]) is not int or case["seed"] < 0
                    or type(case["slot"]) is not int or not 0 <= case["slot"] < 8
                    or type(case["ordinal"]) is not int or case["ordinal"] != len(by_slot[case["slot"]])
                    or type(case["task"]) is not int or case["task"] not in (1, 2, 3) or not isinstance(case["id"], str) or not case["id"]
                    or case["id"] in all_ids or case["seed"] in all_seeds):
                raise ValueError("training scene list has ambiguous, reused, non-train, or reordered cases")
            by_slot[case["slot"]].append(deepcopy(case))
            all_ids.add(case["id"]); all_seeds.add(case["seed"])
        if set(by_slot) != set(range(8)):
            raise ValueError("every one of eight environment slots needs a train stream")
        self.cases = dict(by_slot)
        self.constructors, self.resets, self.reserved_steps = set(), [0] * 8, 0
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.journal_path.open("x")
        self._append({"kind": "authorize", "capability_sha256": self.sha256,
            "source_sha256": value["source_sha256"], "config": asdict(config)})

    def _append(self, item):
        self.journal.write(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.journal.flush(); os.fsync(self.journal.fileno())

    def assert_live(self):
        if _sha(self.path) != self.sha256:
            raise RuntimeError("training capability changed after binding")

    def before_constructor(self, slot):
        self.assert_live()
        if type(slot) is not int or not 0 <= slot < 8 or slot in self.constructors:
            raise RuntimeError("environment constructor is outside the finite capability")
        self._append({"kind": "construct", "slot": slot})
        self.constructors.add(slot)

    def before_reset(self, slot):
        self.assert_live()
        ordinal = self.resets[slot]
        if slot not in self.constructors or ordinal >= len(self.cases[slot]):
            raise RuntimeError("environment reset is outside the finite train case schedule")
        case = deepcopy(self.cases[slot][ordinal])
        self._append({"kind": "reset", "case": case})
        self.resets[slot] += 1
        return case

    def reserve_rollout(self, joint_frames):
        self.assert_live()
        if type(joint_frames) is not int or joint_frames <= 0 or self.reserved_steps + joint_frames > self.config.joint_step_budget:
            raise RuntimeError("training rollout exceeds the finite joint-step authority")
        self._append({"kind": "reserve_joint_frames", "frames": joint_frames, "start": self.reserved_steps})
        self.reserved_steps += joint_frames

    def event(self, item):
        self._append(item)

    def close(self):
        self.journal.close()


@dataclass
class ReferenceTrainingBuffer:
    encoded: np.ndarray
    masks: np.ndarray
    episode_ids: tuple
    source_sha256: str
    source_path: str | None = None

    @classmethod
    def load(cls, path, expected_sha256, reference_sha256):
        if not _is_sha(expected_sha256) or _sha(path) != expected_sha256:
            raise ValueError("initial reference train buffer hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"encoded", "masks", "episode_ids", "metadata_json"}:
                raise ValueError("unknown initial training buffer schema")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("schema") != BUFFER_SCHEMA or metadata.get("split") != "train"
                    or metadata.get("reference_sha256") != reference_sha256):
                raise ValueError("reference buffer is not frozen-initialization training data")
            encoded, masks = archive["encoded"].astype(np.float32), archive["masks"].astype(np.float32)
            episodes = tuple(map(str, archive["episode_ids"].tolist()))
        result = cls(encoded, masks, episodes, expected_sha256, str(Path(path).resolve()))
        result.validate()
        result.encoded.flags.writeable = False; result.masks.flags.writeable = False
        return result

    def validate(self):
        if (self.encoded.ndim != 3 or self.encoded.shape[1:] != (2, FEATURE_DIM) or not len(self.encoded)
                or self.masks.shape != (len(self.encoded), 2, 5) or len(self.episode_ids) != len(self.encoded)
                or not np.isfinite(self.encoded).all() or not np.isfinite(self.masks).all()
                or np.any((self.masks != 0) & (self.masks != 1)) or np.any(self.masks[..., 4] != 1)
                or any(not isinstance(s, str) or not s for s in self.episode_ids) or not _is_sha(self.source_sha256)):
            raise ValueError("invalid bound initial reference training buffer")
        return self


def reward_gae(rewards, values, next_values, terminated, truncated, config=PPOConfig()):
    """Role-wise reward GAE; horizon bootstraps, but recursion stops at reset."""
    r, v, nv = (np.asarray(a, np.float64) for a in (rewards, values, next_values))
    term, trunc = np.asarray(terminated, bool), np.asarray(truncated, bool)
    if (r.ndim != 3 or r.shape[-1] != 2 or v.shape != r.shape or nv.shape != r.shape
            or term.shape != r.shape[:2] or trunc.shape != term.shape
            or not all(np.isfinite(a).all() for a in (r, v, nv)) or np.any(term & trunc)):
        raise ValueError("reward GAE needs finite vector rollout values and exclusive end flags")
    advantage = np.zeros_like(r)
    following = np.zeros_like(r[0])
    for index in range(len(r) - 1, -1, -1):
        delta = r[index] + config.gamma * nv[index] * (~term[index])[:, None] - v[index]
        following = delta + config.gamma * config.gae_lambda * following * (~(term[index] | trunc[index]))[:, None]
        advantage[index] = following
    return advantage.astype(np.float32), (advantage + v).astype(np.float32)


def complete_episode_cost(trajectory, *, terminated, truncated, horizon=120):
    """Label only completed real outcomes; never a partial-episode negative."""
    if (not trajectory or type(terminated) is not bool or type(truncated) is not bool
            or terminated == truncated or len(trajectory) > horizon):
        raise ValueError("one completed physical episode is required")
    events = [row["shutdown"] for row in trajectory]
    if (any(type(value) is not bool for value in events) or any(events[:-1])
            or terminated != events[-1] or truncated and len(trajectory) != horizon):
        raise ValueError("cost target must be the true terminal shutdown or complete physical horizon")
    outcome = bool(events[-1])
    records = []
    for row in trajectory:
        record = dict(row)
        record["cost_return"] = float(outcome)
        record["cost_advantage"] = float(outcome) - float(row["cost_value"])
        record["episode_length"] = len(trajectory)
        records.append(record)
    return records, outcome


def _tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _program_gradient_audit(loss, model):
    result = {}
    for index, role in enumerate(ROLES):
        gradients = torch.autograd.grad(loss.role_losses[index], tuple(model.actors[role].parameters()),
            retain_graph=True, allow_unused=True)
        squared = sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)
        result[role] = {"l2": squared ** .5, "selected_frames": int(loss.selected_counts[index]),
            "physical_frames": loss.physical_frames, "coefficient": loss.coefficient}
    return result


class PPOContinuationDriver:
    """A new run, using eight owned environments and externally bound callbacks.

    ``observe(env)`` returns public ``obs[2,O], encoded[2,F], masks[2,5]``.
    ``extractor(step,state,capability)`` performs separately authorized 275-ep
    extraction and returns programs by role. ``validator(programs,state,epoch)``
    must provide independently validated current actor hashes and gate results.
    No callback can supply an environment action.
    """
    def __init__(self, state: TrainingState, capability: HashBoundTrainingCapability,
                 reference_buffer: ReferenceTrainingBuffer, *, make_environment, reset_environment,
                 observe, stream_seed, variant, extractor=None, validator=None, config=PPOConfig()):
        if (not isinstance(capability, HashBoundTrainingCapability) or capability.config != config
                or capability.value["variant"] != variant or variant not in ("raw", "regularized")
                or type(stream_seed) is not int or stream_seed != capability.value.get("stream_seed")
                or state.reference_sha256 != capability.value["reference_sha256"]
                or state.joint_steps or state.actor_updates or state.critic_updates):
            raise ValueError("PPO constructor requires the exact verified new-run capability")
        reference_buffer.validate()
        binding = capability.value.get("initial_buffer")
        if not isinstance(binding, dict) or binding.get("sha256") != reference_buffer.source_sha256:
            raise ValueError("reference buffer is not the capability-bound initial train buffer")
        if any(group["lr"] != config.actor_learning_rate for group in state.actor_optimizer.param_groups):
            raise ValueError("Actor optimizer rate differs from the registered value")
        if any(group["lr"] != config.critic_learning_rate for group in state.critic_optimizer.param_groups):
            raise ValueError("critic optimizer rate differs from the registered value")
        if not callable(extractor) or variant == "regularized" and not callable(validator):
            raise ValueError("both arms require extraction; regularized additionally requires current-NN validation")
        if reference_buffer.source_path is None or _sha(reference_buffer.source_path) != reference_buffer.source_sha256:
            raise ValueError("initial buffer file is absent or differs from its bound hash")
        bound_path = (capability.root / binding.get("path", "")).resolve(strict=True)
        if bound_path != Path(reference_buffer.source_path).resolve():
            raise ValueError("initial buffer path differs from its capability binding")
        reloaded = ReferenceTrainingBuffer.load(bound_path, reference_buffer.source_sha256, state.reference_sha256)
        if (not np.array_equal(reloaded.encoded, reference_buffer.encoded)
                or not np.array_equal(reloaded.masks, reference_buffer.masks)
                or reloaded.episode_ids != reference_buffer.episode_ids):
            raise ValueError("initial buffer arrays differ from the bound immutable file")
        capability.assert_live()
        self.state, self.capability, self.config = state, capability, config
        self.reference_buffer, self.variant = reference_buffer, variant
        self.make_environment, self.reset_environment, self.observe = make_environment, reset_environment, observe
        self.extractor, self.validator = extractor, validator
        self.rng = np.random.default_rng(stream_seed)
        self.action_generators = [torch.Generator(device="cpu").manual_seed(stream_seed + 131 * slot + 17 * role)
            for slot in range(8) for role in range(2)]
        self.risk = ShutdownRiskDual(RiskDualConfig())
        self.environments, self.cases, self.pending = [], [None] * 8, [[] for _ in range(8)]
        self.programs, self.extractions, self.updates, self.completed_episodes = None, [], [], []
        self.program_lock = validate_program_lock(capability.value.get("program_lock"))
        self.extraction_attempts = []
        self.program_feedback_disabled = False
        self.program_feedback_disabled_since = None
        self.last_fidelity_failure = None
        self.initial_hashes = role_hashes(state.model)
        verify_initial_equivalence(state, _tensor(reference_buffer.encoded[:32]), _tensor(reference_buffer.masks[:32]))
        # No environment constructor can occur above this point.
        for slot in range(8):
            capability.before_constructor(slot)
            self.environments.append(make_environment(slot))
            self._reset(slot)

    def _reset(self, slot):
        case = self.capability.before_reset(slot)
        if self.pending[slot]:
            raise RuntimeError("cannot reset an incomplete cost trajectory")
        self.reset_environment(self.environments[slot], case)
        self.cases[slot] = case

    def _observation(self, slot):
        obs, encoded, mask = self.observe(self.environments[slot])
        obs, encoded, mask = map(lambda x: np.asarray(x, np.float32), (obs, encoded, mask))
        if (obs.shape != (2, OBSERVATION_DIM) or encoded.shape != (2, FEATURE_DIM) or mask.shape != (2, 5)
                or not all(np.isfinite(a).all() for a in (obs, encoded, mask))
                or np.any((mask != 0) & (mask != 1)) or np.any(mask[:, 4] != 1)):
            raise ValueError("environment adapter did not provide public pre-action arrays")
        return obs, encoded, mask

    def _extract_if_due(self):
        step = self.state.joint_steps
        if step not in self.config.extraction_steps or step in self.extractions:
            return
        if not callable(self.extractor):
            raise RuntimeError("registered extraction boundary has no source-bound extraction callback")
        if step in self.extraction_attempts:
            raise RuntimeError("a consumed extraction attempt cannot be retried or used for rescue")
        self.extraction_attempts.append(step)
        self.capability.event({"kind": "program_extraction_started", "step": step,
            "program_lock_sha256": _digest(self.program_lock), "retry_authorized": False})
        before = role_hashes(self.state.model)
        programs = self.extractor(step, self.state, self.capability)
        if role_hashes(self.state.model) != before or set(programs) != set(ROLES):
            raise RuntimeError("program extraction changed the NN or omitted a role")
        validate_locked_programs(programs, self.program_lock, before)
        previously_disabled = self.program_feedback_disabled
        self.programs = programs
        self.extractions.append(step)
        # Only this predeclared extraction can reopen a disabled feedback
        # window. The next epoch must still obtain new current-NN evidence.
        self.program_feedback_disabled = False
        self.program_feedback_disabled_since = None
        self.last_fidelity_failure = None
        self.capability.event({"kind": "program_extraction", "step": step,
            "program_sha256": {role: program.sha256() for role, program in programs.items()},
            "actor_role_sha256": before, "actor_updates_by_extractor": 0,
            "feedback_reenabled_by_scheduled_extraction": previously_disabled})

    def collect(self):
        self._extract_if_due()
        length = self.config.next_collection_length(self.state.joint_steps)
        if not length:
            return None
        self.capability.reserve_rollout(length * 8)
        start, rows, finished = self.state.joint_steps, [], []
        for _ in range(length):
            vector_rows = []
            for slot, environment in enumerate(self.environments):
                obs, encoded, mask = self._observation(slot)
                with torch.no_grad():
                    logp = masked_log_probabilities(self.state.model.outputs_encoded(_tensor(encoded[None]), _tensor(mask[None])), _tensor(mask[None]))[0]
                    probabilities = logp.exp()
                    actions = torch.stack([torch.multinomial(probabilities[role], 1,
                        generator=self.action_generators[2 * slot + role]).squeeze(0) for role in range(2)])
                    chosen_logp = logp.gather(-1, actions[:, None]).squeeze(-1)
                    critic_obs = _tensor(obs.reshape(1, -1))
                    value = self.state.model.reward_critic(critic_obs)[0]
                    cost_value = self.state.model.cost_critic(critic_obs).sigmoid()[0, 0]
                if np.any(mask[np.arange(2), actions.numpy()] != 1):
                    raise RuntimeError("sampled Actor action violates its physical mask")
                requested = {role: ACTIONS[int(actions[i])] for i, role in enumerate(ROLES)}
                _, rewards, terminated, truncated, info = environment.step(requested)
                if type(terminated) is not bool or type(truncated) is not bool or terminated and truncated:
                    raise ValueError("environment terminal flags are malformed")
                shutdown = bool(info.get("shutdowns", ()))
                if shutdown != terminated:
                    raise ValueError("true shutdown event and physical termination disagree")
                next_obs, _, _ = self._observation(slot)
                with torch.no_grad():
                    next_value = self.state.model.reward_critic(_tensor(next_obs.reshape(1, -1)))[0]
                row = {"obs": obs.copy(), "encoded": encoded.copy(), "mask": mask.copy(),
                    "actions": actions.numpy().copy(), "old_logp": chosen_logp.numpy().copy(),
                    "rewards": np.asarray([rewards[role] for role in ROLES], np.float32),
                    "values": value.numpy().copy(), "next_values": next_value.numpy().copy(),
                    "cost_value": float(cost_value), "shutdown": shutdown, "terminated": terminated,
                    "truncated": truncated, "episode_id": self.cases[slot]["id"], "slot": slot}
                self.pending[slot].append(row)
                vector_rows.append(row)
                self.state.advance_joint_steps(1)
                if terminated or truncated:
                    complete, outcome = complete_episode_cost(self.pending[slot], terminated=terminated,
                        truncated=truncated, horizon=self.config.episode_horizon)
                    finished.extend(complete)
                    self.completed_episodes.append({"episode_id": row["episode_id"], "shutdown": outcome,
                        "length": len(complete), "scene": deepcopy(self.cases[slot])})
                    self.pending[slot] = []
                    # No extra reset is spent once the complete joint budget is consumed.
                    if self.state.joint_steps + (7 - slot) < self.config.joint_step_budget:
                        self._reset(slot)
            rows.append(vector_rows)
        rewards = np.asarray([[row["rewards"] for row in vector] for vector in rows])
        values = np.asarray([[row["values"] for row in vector] for vector in rows])
        next_values = np.asarray([[row["next_values"] for row in vector] for vector in rows])
        terminal = np.asarray([[row["terminated"] for row in vector] for vector in rows])
        truncated = np.asarray([[row["truncated"] for row in vector] for vector in rows])
        advantage, returns = reward_gae(rewards, values, next_values, terminal, truncated, self.config)
        flattened = [row for vector in rows for row in vector]
        for row, a, r in zip(flattened, advantage.reshape(-1, 2), returns.reshape(-1, 2), strict=True):
            row["reward_advantage"], row["reward_return"] = a, r
        return {"start_joint_step": start, "rows": flattened, "complete_cost_rows": finished,
            "end_joint_step": self.state.joint_steps}

    def _reference_mix(self, rows):
        count = len(rows)
        if count % 2:
            raise ValueError("joint minibatches must support exact half reference / half current mixing")
        half = count // 2
        old = self.rng.integers(0, len(self.reference_buffer.encoded), size=half)
        current = self.rng.choice(count, size=half, replace=False)
        encoded = np.concatenate((self.reference_buffer.encoded[old], np.asarray([rows[i]["encoded"] for i in current])), axis=0)
        masks = np.concatenate((self.reference_buffer.masks[old], np.asarray([rows[i]["mask"] for i in current])), axis=0)
        provenance = [ReferenceRow("reference_train", "train", self.reference_buffer.episode_ids[i], self.reference_buffer.source_sha256) for i in old]
        provenance += [ReferenceRow("current_train", "train", rows[i]["episode_id"], self.capability.sha256) for i in current]
        return _tensor(encoded), _tensor(masks), provenance

    def _validated_programs(self, epoch):
        if self.variant == "raw" or self.config.program_coefficient(self.state.joint_steps) == 0:
            return None
        if self.program_feedback_disabled:
            return None
        if self.programs is None:
            raise RuntimeError("feedback window reached before a certified program extraction")
        hashes = role_hashes(self.state.model)
        result = self.validator(self.programs, self.state, epoch)
        if role_hashes(self.state.model) != hashes:
            raise RuntimeError("independent program validation changed the current NN")
        if (not isinstance(result, dict) or result.get("actor_role_sha256") != hashes
                or result.get("program_sha256") != {r: self.programs[r].sha256() for r in ROLES}
                or result.get("independent_validation") is not True
                or result.get("no_train_scene_overlap") is not True
                or not isinstance(result.get("passed"), dict) or set(result["passed"]) != set(ROLES)
                or any(type(v) is not bool for v in result["passed"].values())):
            raise RuntimeError("current NN has no independent per-epoch program qualification")
        if not all(result["passed"].values()):
            self.program_feedback_disabled = True
            self.program_feedback_disabled_since = self.state.joint_steps
            self.last_fidelity_failure = {"step": self.state.joint_steps, "epoch": epoch,
                "actor_role_sha256": hashes, "passed": deepcopy(result["passed"]),
                "program_sha256": deepcopy(result["program_sha256"])}
            self.capability.event({"kind": "program_feedback_disabled", "reason": "current_NN_fidelity_failed",
                "until_next_registered_extraction": True, "evidence": self.last_fidelity_failure,
                "rescue_extraction_authorized": False})
            return None
        return result

    def _program_loss(self, logits, masks, actions, encoded, coefficient, validated):
        if self.variant == "raw":
            return program_kl_loss(logits, None, masks, None, None,
                raw_program_top=None, variant="raw", coefficient=coefficient)
        q, qualified, raw_top = np.zeros(logits.shape, np.float32), np.zeros(logits.shape[:-1], bool), np.zeros(logits.shape[:-1], np.int64)
        if validated is not None:
            values, physical = encoded.detach().numpy(), masks.detach().numpy()
            for i, role in enumerate(ROLES):
                program = self.programs[role]
                routes, raw, probabilities = program.distributions(values[:, i], physical[:, i])
                q[:, i], raw_top[:, i] = probabilities, raw.argmax(-1)
                qualified[:, i] = np.isin(routes, tuple(program.feedback_leaves))
        else:
            # No program evidence has been accepted; a normalized dummy target
            # is never selected and has zero external feedback gradient.
            q = masks.detach().numpy() / masks.detach().numpy().sum(-1, keepdims=True)
        return program_kl_loss(logits, _tensor(q), masks, actions, torch.as_tensor(qualified),
            raw_program_top=torch.as_tensor(raw_top), variant="regularized", coefficient=coefficient)

    def update(self, batch):
        if batch["end_joint_step"] != self.state.joint_steps:
            raise ValueError("rollout is not the most recently collected fixed behavior batch")
        rows, costs = batch["rows"], batch["complete_cost_rows"]
        actor_enabled = batch["start_joint_step"] >= CRITIC_WARMUP_STEPS
        complete_outcomes = {}
        for row in costs:
            complete_outcomes[row["episode_id"]] = bool(row["cost_return"])
        risk = self.risk.update([(True, outcome) for outcome in complete_outcomes.values()])
        normalized = np.asarray([row["reward_advantage"] for row in rows], np.float32)
        normalized = (normalized - normalized.mean(0)) / (normalized.std(0) + 1.e-8)
        for row, advantage in zip(rows, normalized, strict=True):
            row["normalized_advantage"] = advantage
        report = {"start_joint_step": batch["start_joint_step"], "end_joint_step": self.state.joint_steps,
            "actor_updates_enabled": actor_enabled, "complete_episodes": len(complete_outcomes),
            "shutdown_episodes": sum(complete_outcomes.values()), "partial_episodes_labeled_negative": 0,
            "risk": risk, "epochs": [], "program_gradient_updates": {r: 0 for r in ROLES}}
        for epoch in range(self.config.epochs):
            validated = self._validated_programs(epoch) if actor_enabled else None
            order = self.rng.permutation(len(rows))
            epoch_report = {"epoch": epoch, "updates": []}
            for start in range(0, len(order), self.config.minibatch_joint_frames):
                selected = [rows[i] for i in order[start:start + self.config.minibatch_joint_frames]]
                obs = _tensor(np.asarray([r["obs"] for r in selected])).reshape(len(selected), -1)
                encoded, masks = _tensor([r["encoded"] for r in selected]), _tensor([r["mask"] for r in selected])
                actions = torch.as_tensor(np.asarray([r["actions"] for r in selected]), dtype=torch.long)
                logits = self.state.model.outputs_encoded(encoded, masks)
                logp = masked_log_probabilities(logits, masks)
                chosen_logp = logp.gather(-1, actions[..., None]).squeeze(-1)
                old = _tensor([r["old_logp"] for r in selected])
                ratio, advantage = (chosen_logp - old).exp(), _tensor([r["normalized_advantage"] for r in selected])
                reward_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - self.config.clip, 1 + self.config.clip) * advantage).mean(0).sum()
                entropy = -(logp.exp() * logp.masked_fill(masks <= .5, 0.)).sum(-1).mean(0).sum()
                reward_prediction = self.state.model.reward_critic(obs)
                critic_loss = self.config.critic_coefficient * F.mse_loss(reward_prediction, _tensor([r["reward_return"] for r in selected]).detach())
                cost_loss = (logits[..., 0] * 0.).sum()
                if costs:
                    indices = self.rng.choice(len(costs), size=min(len(costs), self.config.minibatch_joint_frames), replace=False)
                    cost_rows = [costs[i] for i in indices]
                    cost_obs = _tensor([r["obs"] for r in cost_rows]).reshape(len(cost_rows), -1)
                    cost_logits = self.state.model.cost_critic(cost_obs).squeeze(-1)
                    targets = _tensor([r["cost_return"] for r in cost_rows]).detach()
                    weights = _tensor([len(costs) / (len(complete_outcomes) * r["episode_length"]) for r in cost_rows])
                    critic_loss = critic_loss + self.config.critic_coefficient * (weights * F.binary_cross_entropy_with_logits(cost_logits, targets, reduction="none")).mean()
                    current_cost_logp = masked_log_probabilities(self.state.model.outputs_encoded(
                        _tensor([r["encoded"] for r in cost_rows]), _tensor([r["mask"] for r in cost_rows])), _tensor([r["mask"] for r in cost_rows]))
                    cost_actions = torch.as_tensor(np.asarray([r["actions"] for r in cost_rows]), dtype=torch.long)
                    new_chosen = current_cost_logp.gather(-1, cost_actions[..., None]).squeeze(-1)
                    joint_ratio = (new_chosen - _tensor([r["old_logp"] for r in cost_rows])).sum(-1).exp()
                    cost_advantage = _tensor([r["cost_advantage"] for r in cost_rows]).detach()
                    # Uniform transition minibatch, exactly weighted to the
                    # complete-episode mean of sums / fixed physical horizon.
                    cost_loss = (len(costs) / (len(complete_outcomes) * self.config.episode_horizon)) * torch.maximum(
                        joint_ratio * cost_advantage, joint_ratio.clamp(1 - self.config.clip, 1 + self.config.clip) * cost_advantage).mean()
                self.state.critic_step(critic_loss)
                update = {"critic_loss": float(critic_loss.detach()), "actor_updated": False}
                if actor_enabled:
                    ref_encoded, ref_masks, provenance = self._reference_mix(selected)
                    current = self.state.model.outputs_encoded(ref_encoded, ref_masks)
                    reference = reference_logits_encoded(self.state, ref_encoded)
                    reference_loss = reference_kl_loss(current, reference, ref_masks, provenance=provenance, beta=self.state.beta)
                    coefficient = self.config.program_coefficient(self.state.joint_steps)
                    program_loss = self._program_loss(logits, masks, actions, encoded, coefficient, validated)
                    program_audit = _program_gradient_audit(program_loss, self.state.model)
                    if self.variant == "raw" and any(v["l2"] or v["selected_frames"] for v in program_audit.values()):
                        raise RuntimeError("raw arm received an external program gradient")
                    loss = reward_loss + self.risk.multiplier * cost_loss - self.config.entropy_coefficient * entropy + reference_loss + program_loss.loss
                    def observed():
                        with torch.no_grad():
                            values = masked_reference_kl(self.state.model.outputs_encoded(ref_encoded, ref_masks), reference, ref_masks).mean(0)
                        return {role: float(values[i]) for i, role in enumerate(ROLES)}
                    parameters_before = {role: [parameter.detach().clone() for parameter in self.state.model.actors[role].parameters()]
                        for role in ROLES}
                    updates_before = self.state.actor_updates
                    result = self.state.actor_step(loss, observed_reference_kl=observed, max_grad_norm=self.config.maximum_gradient_norm)
                    if self.state.actor_updates != updates_before + 1:
                        raise RuntimeError("Actor optimizer did not record one accepted update")
                    common_clip = min(1., self.config.maximum_gradient_norm / (float(result["gradient_norm"]) + 1.e-6))
                    for role, audit in program_audit.items():
                        delta_squared = sum(float((parameter.detach() - before).double().square().sum())
                            for parameter, before in zip(self.state.model.actors[role].parameters(), parameters_before[role], strict=True))
                        delta = delta_squared ** .5
                        accepted = bool(audit["l2"] > 0 and common_clip > 0 and delta > 0)
                        audit.update({"accepted_optimizer_step": True, "common_gradient_clip_factor": common_clip,
                            "program_gradient_l2_after_common_clip": audit["l2"] * common_clip,
                            "role_weight_delta_l2": delta, "role_weights_changed": delta > 0,
                            "accepted_positive_program_gradient": accepted,
                            "weight_delta_attribution": "combined_PPO_risk_reference_and_program_update_not_causal_program_attribution"})
                        report["program_gradient_updates"][role] += int(accepted)
                    update.update({"actor_updated": True, "actor": result, "program_gradient": program_audit,
                        "reference_loss": float(reference_loss.detach()), "program_loss": float(program_loss.loss.detach()),
                        "physical_frames": len(selected)})
                epoch_report["updates"].append(update)
            report["epochs"].append(epoch_report)
        self.updates.append(report)
        self.capability.event({"kind": "ppo_update", "report": report})
        return report

    def run(self):
        try:
            while self.state.joint_steps < self.config.joint_step_budget:
                batch = self.collect()
                self.update(batch)
            self._extract_if_due()
            if tuple(self.extractions) != self.config.extraction_steps:
                raise RuntimeError("not all registered extraction callbacks completed")
            final = role_hashes(self.state.model)
            result = {"schema": SCHEMA, "variant": self.variant, "joint_steps": self.state.joint_steps,
                "reference_sha256": self.state.reference_sha256, "initial_role_sha256": self.initial_hashes,
                "final_role_sha256": final, "both_actor_weights_changed": all(final[r] != self.initial_hashes[r] for r in ROLES),
                "actor_updates": self.state.actor_updates, "critic_updates": self.state.critic_updates,
                "program_gradient_updates": {r: sum(row["program_gradient_updates"][r] for row in self.updates) for r in ROLES},
                "program_extraction_steps": list(self.extractions), "beta": dict(self.state.beta),
                "risk_dual": self.risk.snapshot(), "complete_episode_count": len(self.completed_episodes),
                "shutdown_episode_count": sum(r["shutdown"] for r in self.completed_episodes),
                "pending_episode_lengths": [len(rows) for rows in self.pending],
                "partial_episodes_labeled_negative": 0, "action_override_count": 0,
                "physical_mask_illegal_sample_count": 0, "publishable": False, "pilot_enabled": False}
            result["program_feedback_disabled_at_end"] = self.program_feedback_disabled
            self.capability.event({"kind": "training_complete", "report": result})
            return result
        except BaseException as error:
            self.capability.event({"kind": "training_failed", "exception": type(error).__name__,
                "message": str(error), "completed_joint_steps": self.state.joint_steps,
                "reserved_joint_steps": self.capability.reserved_steps, "candidate_retry_authorized": False})
            raise
        finally:
            self.capability.close()

    def snapshot(self, *, environment_snapshot):
        """Primitive full driver state for a separately authorized checkpoint.

        The caller stores TrainingState with its existing exact checkpoint API,
        and must include this dictionary in ``extra_rng_states``. Restoring an
        interrupted scientific attempt requires a new bound continuation
        capability; this snapshot does not grant it.
        """
        result = {"schema": SCHEMA, "capability_sha256": self.capability.sha256, "config": asdict(self.config),
            "variant": self.variant, "state_joint_steps": self.state.joint_steps,
            "reference_buffer_sha256": self.reference_buffer.source_sha256,
            "capability_state": {"constructor_slots": sorted(self.capability.constructors),
                "reset_ordinals": list(self.capability.resets), "reserved_joint_steps": self.capability.reserved_steps},
            "rng": deepcopy(self.rng.bit_generator.state),
            "action_rng": [generator.get_state() for generator in self.action_generators],
            "cases": deepcopy(self.cases), "pending_complete_episode_context": deepcopy(self.pending),
            "environments": [environment_snapshot(env) for env in self.environments],
            "risk": self.risk.snapshot(), "extractions": list(self.extractions),
            "extraction_attempts": list(self.extraction_attempts), "program_lock": deepcopy(self.program_lock),
            "program_feedback_disabled": self.program_feedback_disabled,
            "program_feedback_disabled_since": self.program_feedback_disabled_since,
            "last_fidelity_failure": deepcopy(self.last_fidelity_failure),
            "programs": None if self.programs is None else {r: p.to_dict() for r, p in self.programs.items()},
            "updates": deepcopy(self.updates), "completed_episodes": deepcopy(self.completed_episodes),
            "initial_role_sha256": dict(self.initial_hashes)}
        return _checkpoint_primitives(result)


def _checkpoint_primitives(value):
    """Keep the existing weights_only checkpoint loader's safe primitive set."""
    if isinstance(value, np.ndarray):
        return torch.as_tensor(np.array(value, copy=True))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _checkpoint_primitives(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_primitives(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("environment snapshot must expose only explicit safe primitives")
