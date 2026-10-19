"""Inactive v6.22 tensor training adapter for the unchanged v6.20 Actor.

This module neither constructs an environment nor authorizes training.  A new
run imports *only* the two exported Actor parameter sets; its critics and Adam
states are new.  Only a checkpoint written here is a training-state resume.
External programs supply loss targets, never runtime actions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from .warehouse_v6_actor import (
    MODEL_VERSION, FrozenWarehouseV6Actor, _role_arrays, _tensor_names,
)
from .warehouse_v6_actor_training import WarehouseV6Actor, role_hashes

ROLES = tuple(AGENTS)
CHECKPOINT_SCHEMA = "warehouse-v622-full-nn-training-checkpoint.v1"
REFERENCE_TARGET = .01
BETA_RATE = .05
BETA_MIN, BETA_MAX = .001, 100.
CRITIC_WARMUP_STEPS = 4000


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _finite_number(value, name, *, minimum=0.):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def _physical_mask(logits, masks):
    if (logits.ndim < 2 or logits.shape[-1] != 5 or masks.shape != logits.shape
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or not torch.isfinite(masks).all()
            or not torch.all((masks == 0) | (masks == 1))
            or not torch.all(masks[..., 4] == 1)):
        raise ValueError("expected finite five-action logits and binary physical masks with WAIT legal")
    return masks.detach().to(device=logits.device, dtype=torch.bool)


def masked_log_probabilities(logits, masks):
    legal = _physical_mask(logits, masks)
    return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)


def masked_reference_kl(current_logits, reference_logits, masks):
    """Per-physical-frame KL(reference || current), with no teacher gradient."""
    if reference_logits.shape != current_logits.shape:
        raise ValueError("reference/current shapes differ")
    legal = _physical_mask(current_logits, masks)
    # Float64 accumulation avoids a slightly negative reported KL near the
    # function-preserving starting point; casting keeps NN gradients live.
    current = masked_log_probabilities(current_logits.to(torch.float64), masks)
    reference = masked_log_probabilities(reference_logits.detach().to(
        device=current_logits.device, dtype=torch.float64), masks)
    # Avoid 0*(-inf - -inf); illegal actions have exactly zero contribution.
    safe_ref = reference.masked_fill(~legal, 0.)
    safe_cur = current.masked_fill(~legal, 0.)
    return (reference.exp() * (safe_ref-safe_cur)).sum(-1).clamp_min(0.)


@dataclass(frozen=True)
class ReferenceRow:
    source: str  # reference_train or current_train, never validation/audit
    split: str
    episode_id: str
    source_sha256: str


def validate_reference_mix(rows: Sequence[ReferenceRow], count: int):
    if count <= 0 or count % 2 or len(rows) != count:
        raise ValueError("reference KL requires a nonempty even physical-row count")
    sources = {"reference_train": 0, "current_train": 0}
    for row in rows:
        if (not isinstance(row, ReferenceRow) or row.split != "train"
                or row.source not in sources or not isinstance(row.episode_id, str)
                or not row.episode_id or not isinstance(row.source_sha256, str)
                or len(row.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in row.source_sha256)):
            raise ValueError("reference rows require bound train-only provenance")
        sources[row.source] += 1
    if sources["reference_train"] != sources["current_train"]:
        raise ValueError("reference/current training rows must each be exactly 50 percent")
    return sources


def reference_kl_loss(current_logits, reference_logits, masks, *, provenance, beta):
    """Mean reference loss per role; provenance is checked before computation.

    This validates typed provenance, not its external source file.  The stage
    runner must bind those files and episode membership to its capability.
    """
    validate_reference_mix(provenance, len(current_logits))
    rows = masked_reference_kl(current_logits, reference_logits, masks)
    if rows.ndim == 1:
        coefficient = _finite_number(beta, "beta")
        return coefficient*rows.mean()
    if rows.ndim != 2 or rows.shape[1] != 2 or set(beta) != set(ROLES):
        raise ValueError("joint reference loss requires both role beta values")
    coefficients = rows.new_tensor([_finite_number(beta[r], "beta") for r in ROLES])
    return (rows.mean(0)*coefficients).sum()


def updated_reference_beta(beta, observed_kl):
    """Pure dual-controller arithmetic; TrainingState calls it after Adam.step."""
    if set(beta) != set(ROLES) or set(observed_kl) != set(ROLES):
        raise ValueError("beta update requires both roles")
    result = {}
    for role in ROLES:
        old = _finite_number(beta[role], "beta", minimum=BETA_MIN)
        value = _finite_number(observed_kl[role], "observed KL")
        if old > BETA_MAX:
            raise ValueError("beta exceeds registered bound")
        # Clip in log space before exp, including for large but finite KL.
        log_value = math.log(old)+BETA_RATE*(value/REFERENCE_TARGET-1.)
        result[role] = min(BETA_MAX, max(BETA_MIN, math.exp(
            min(math.log(BETA_MAX), max(math.log(BETA_MIN), log_value)))))
    return result


@dataclass
class ProgramLoss:
    loss: torch.Tensor
    role_losses: torch.Tensor
    selected: torch.Tensor
    selected_counts: torch.Tensor
    physical_frames: int
    coefficient: float


def program_kl_loss(current_logits, program_probabilities, masks, sampled_indices,
                    qualified, *, raw_program_top, variant, coefficient=.01):
    """Qualified KL(program || NN), divided by *all* physical frames per role.

    raw_program_top is the program's pre-mask top action. Qualification must
    already incorporate independent current-NN validation and path proof.
    Neither q, masks, qualification, nor sampled-action selection gets a
    gradient.  Current NN logits stay live, even when the top actions agree.
    """
    if variant not in ("raw", "regularized"):
        raise ValueError("unknown training variant")
    coefficient = _finite_number(coefficient, "program coefficient")
    legal = _physical_mask(current_logits, masks)
    if len(current_logits) == 0 or current_logits.ndim not in (2, 3):
        raise ValueError("program feedback requires physical rows, optionally two roles")
    if current_logits.ndim == 3 and current_logits.shape[1] != 2:
        raise ValueError("joint program feedback requires two roles")
    shape = current_logits.shape[:-1]
    if variant == "raw":
        # The raw arm need not load, validate, or differentiate a program.
        # Keep a zero derivative wrt NN so a combined loss remains usable.
        selected = torch.zeros(shape, dtype=torch.bool, device=current_logits.device)
        role_losses = current_logits.sum(-1).sum(0)*0.
        return ProgramLoss(role_losses.sum(), role_losses, selected,
                           selected.sum(0), len(current_logits), 0.)
    if (program_probabilities.shape != current_logits.shape
            or sampled_indices.shape != shape or qualified.shape != shape
            or raw_program_top.shape != shape):
        raise ValueError("program feedback tensor shapes differ")
    for indices in (sampled_indices, raw_program_top):
        if (indices.dtype not in (torch.int32, torch.int64)
                or torch.any(indices < 0) or torch.any(indices >= 5)):
            raise ValueError("action indices must be integers in [0,5)")
    if qualified.dtype != torch.bool:
        raise ValueError("qualified must be a boolean evidence mask")
    q = program_probabilities.detach().to(current_logits)
    if (not torch.isfinite(q).all() or torch.any(q < 0)
            or torch.any(q.masked_select(~legal) != 0)
            or not torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-6, rtol=0)):
        raise ValueError("program must provide a normalized, physically masked distribution")
    sampled = sampled_indices.detach().to(current_logits.device, dtype=torch.long)
    top = raw_program_top.detach().to(current_logits.device, dtype=torch.long)
    if not legal.gather(-1, sampled[..., None]).all():
        raise ValueError("sampled NN proposal is physically illegal")
    logp = masked_log_probabilities(current_logits, masks)
    nn_top = logp.detach().argmax(-1)
    selected = (qualified.detach().to(current_logits.device)
                & legal.gather(-1, top[..., None]).squeeze(-1)
                & (top == nn_top) & (sampled == nn_top))
    safe_logp = logp.masked_fill(~legal, 0.)
    logq = torch.where(q > 0, torch.log(q.clamp_min(torch.finfo(q.dtype).tiny)), torch.zeros_like(q))
    row_kl = (q*(logq-safe_logp)).sum(-1)
    role_losses = coefficient*(row_kl*selected).sum(0)/len(current_logits)
    return ProgramLoss(role_losses.sum(), role_losses, selected,
                       selected.sum(0), len(current_logits), coefficient)


def episode_shutdown_returns(events, *, terminated, truncated, horizon=120):
    """Undiscounted, true once-per-complete-episode shutdown Monte Carlo target."""
    values = np.asarray(events)
    if (values.ndim != 1 or not 0 < len(values) <= horizon or values.dtype != np.bool_
            or type(terminated) is not bool or type(truncated) is not bool
            or terminated == truncated):
        raise ValueError("shutdown targets require one complete terminated or truncated episode")
    indices = np.flatnonzero(values)
    if len(indices):
        if len(indices) != 1 or indices[0] != len(values)-1 or not terminated:
            raise ValueError("real shutdown must occur once at the terminal frame")
    elif not truncated or len(values) != horizon:
        raise ValueError("a non-shutdown episode must complete the physical horizon")
    # No gamma discount: its episode return equals the acceptance event rate.
    return torch.from_numpy(np.cumsum(values[::-1], dtype=np.float32)[::-1].copy())


def complete_shutdown_batch(episodes):
    """Warmup targets and honest observation counts, not critic validation.

    A rollout cut by the 4000-step warmup boundary is not a complete negative
    example: keep it pending until completion, or omit it from this batch.
    """
    if not episodes:
        raise ValueError("at least one complete episode is required")
    ids, targets, lengths, costs = set(), [], [], []
    for episode in episodes:
        key = episode["episode_id"]
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("complete episode IDs must be nonempty and unique")
        ids.add(key)
        result = episode_shutdown_returns(episode["shutdown_events"],
            terminated=episode["terminated"], truncated=episode["truncated"])
        targets.append(result); lengths.append(len(result)); costs.append(int(result[0]))
    return {"targets": torch.cat(targets), "episode_lengths": lengths,
        "episode_costs": costs, "complete_episode_count": len(costs),
        "shutdown_episode_count": sum(costs), "observed_shutdown_episode_rate": sum(costs)/len(costs),
        "no_positive_shutdown_examples_observed": not any(costs),
        "cost_critic_validated": False, "partial_episodes_labeled_negative": 0,
        "discount_factor": 1.}


def shutdown_critic_loss(predictions, targets):
    if (predictions.shape != targets.shape or not torch.isfinite(predictions).all()
            or not torch.isfinite(targets).all()
            or not torch.all((targets == 0) | (targets == 1))):
        raise ValueError("cost critic needs finite complete-episode binary return targets")
    return nn.functional.mse_loss(predictions, targets.detach().to(predictions))


def warmup_allows_actor_update(joint_steps):
    if type(joint_steps) is not int or joint_steps < 0:
        raise ValueError("joint steps must be nonnegative integer")
    return joint_steps >= CRITIC_WARMUP_STEPS


def _rng_snapshot():
    name, words, position, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": name,
        "words": torch.from_numpy(words.astype(np.int64)), "position": position,
        "gaussian": gaussian, "cached": cached}, "torch_cpu": torch.get_rng_state()}


def _restore_rng(value):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n["name"], n["words"].numpy().astype(np.uint32),
                         n["position"], n["gaussian"], n["cached"]))
    torch.set_rng_state(value["torch_cpu"].cpu())


@dataclass
class TrainingState:
    model: WarehouseV6Actor
    reference: FrozenWarehouseV6Actor
    reference_path: str
    reference_sha256: str
    initialization_seed: int
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    reference_metadata: dict
    beta: dict = field(default_factory=lambda: {r: 1. for r in ROLES})
    joint_steps: int = 0
    actor_updates: int = 0
    critic_updates: int = 0
    extra_rng_states: dict = field(default_factory=dict)
    resumed_training_state: bool = False

    @property
    def initialization_report(self):
        return {"reference_sha256": self.reference_sha256,
            "reference_role_sha256": dict(self.reference.role_sha256),
            "imported_actor_tensor_names": list(_tensor_names(MODEL_VERSION)),
            "initialization_seed": self.initialization_seed,
            "resumed_training_state": self.resumed_training_state,
            "critics_and_optimizers_fresh_at_reference_import": True,
            "old_critic_or_optimizer_imported": False}

    def advance_joint_steps(self, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("step increment must be nonnegative integer")
        self.joint_steps += amount

    def actor_step(self, loss, *, observed_reference_kl, max_grad_norm=.5):
        if not warmup_allows_actor_update(self.joint_steps):
            raise RuntimeError("first 4000 joint steps permit critic updates only")
        if not callable(observed_reference_kl):
            raise ValueError("reference KL observer must run after the Actor update")
        if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:
            raise ValueError("Actor loss must be a finite differentiable scalar")
        _finite_number(max_grad_norm, "maximum gradient norm", minimum=1e-30)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.model.actors.parameters(), max_grad_norm,
                                      error_if_nonfinite=True)
        self.actor_optimizer.step()
        # No caller can update beta through this state before the Actor step.
        new_beta = updated_reference_beta(self.beta, observed_reference_kl())
        self.actor_updates += 1
        self.beta = new_beta
        return {"actor_updates": self.actor_updates, "gradient_norm": float(norm),
                "beta": dict(self.beta)}

    def critic_step(self, loss):
        if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:
            raise ValueError("critic loss must be a finite differentiable scalar")
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.model.reward_critic.parameters())+list(self.model.cost_critic.parameters())
        nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_updates += 1


def initialize_from_reference(path, expected_sha256, expected_role_sha256, *,
                              initialization_seed, actor_lr=3e-4,
                              critic_lr=3e-4, device="cpu"):
    if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
        raise ValueError("initialization seed must be explicitly registered")
    if torch.device(device).type != "cpu" or torch.get_default_dtype() != torch.float32:
        raise ValueError("v6.22 adapter registers CPU/float32 RNG and inference only")
    actor_lr = _finite_number(actor_lr, "Actor learning rate", minimum=1e-30)
    critic_lr = _finite_number(critic_lr, "critic learning rate", minimum=1e-30)
    reference = FrozenWarehouseV6Actor.load(path, expected_sha256=expected_sha256,
        expected_role_sha256=expected_role_sha256)
    if reference.model_version != MODEL_VERSION:
        raise ValueError("v6.22 requires the unchanged complete v6.20/v6.18 Actor formula")
    # CPU-only construction avoids mutating accelerator or caller RNG states.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(initialization_seed)
        model = WarehouseV6Actor(model_version=MODEL_VERSION).float()
    for role in ROLES:
        actor = model.actors[role]
        targets = actor.state_dict()
        with torch.no_grad():
            for name, values in reference.roles[role].weights.items():
                target = targets[name]
                value = torch.as_tensor(np.array(values, copy=True), dtype=target.dtype)
                if value.shape != target.shape:
                    raise ValueError("reference Actor tensor shape differs")
                target.copy_(value)
            # Imported NPZ holds already sparse effective inference weights.
            actor.tree.support_mask.copy_(actor.tree.split.weight != 0)
            actor.tree.support_fixed[0] = True
            actor.tree.guard_node_mask.zero_()
    if role_hashes(model) != reference.role_sha256:
        raise RuntimeError("complete Actor NPZ import did not preserve both role hashes")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    if _hash(path) != expected_sha256:
        raise RuntimeError("reference changed during explicit Actor import")
    model = model.to(device)
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(list(model.reward_critic.parameters())+
                                      list(model.cost_critic.parameters()), lr=critic_lr, eps=1e-5)
    return TrainingState(model, reference, str(Path(path).resolve()), expected_sha256,
        initialization_seed, actor_optimizer, critic_optimizer, metadata)


def reference_logits_encoded(state, encoded):
    values = encoded.detach().cpu().numpy()
    result = np.stack([state.reference.roles[r].logits_encoded(values[..., i, :])
                       for i, r in enumerate(ROLES)], -2)
    return torch.as_tensor(result, dtype=encoded.dtype, device=encoded.device)


def verify_initial_equivalence(state, encoded, masks):
    frozen = FrozenWarehouseV6Actor.from_model(state.model)
    if frozen.role_sha256 != state.reference.role_sha256:
        raise RuntimeError("Actor is no longer the exact reference initialization")
    with torch.no_grad():
        current = state.model.outputs_encoded(encoded, masks)
        ref = reference_logits_encoded(state, encoded)
        p = masked_log_probabilities(current, masks).exp()
        q = masked_log_probabilities(ref, masks).exp()
    maximum = float((p-q).abs().max())
    if maximum > 2e-5 or not torch.equal(p.argmax(-1), q.argmax(-1)):
        raise RuntimeError("Torch/frozen reference probability or action equivalence failed")
    return {"role_sha256": dict(frozen.role_sha256), "maximum_probability_difference": maximum,
            "masked_argmax_mismatches": 0, "physical_frames": len(encoded)}


def export_current_actor(state, path, *, pair_seed, variant, candidate_id):
    if variant not in ("raw", "regularized") or type(pair_seed) is not int or pair_seed < 0 or not candidate_id:
        raise ValueError("export requires explicit pair, variant and candidate")
    target = Path(path)
    values = {r: _role_arrays(state.model.actors[r]) for r in ROLES}
    hashes = role_hashes(state.model)
    metadata = {**deepcopy(state.reference_metadata), "pair_seed": pair_seed,
        "variant": variant, "candidate_id": candidate_id, "role_sha256": hashes,
        "training_recipe": "warehouse-v622-behavior-preserving", "publishable": False,
        "reference_sha256": state.reference_sha256}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{f"{r}.{name}": value for r, weights in values.items() for name, value in weights.items()})
    raw = buffer.getvalue()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(raw)
    digest = sha256(raw).hexdigest()
    FrozenWarehouseV6Actor.load(target, expected_sha256=digest, expected_role_sha256=hashes)
    return {"path": str(target.resolve()), "sha256": digest, "role_sha256": hashes,
            "model_version": MODEL_VERSION, "publishable": False}


def save_checkpoint(state, path):
    value = {"schema": CHECKPOINT_SCHEMA, "model_version": MODEL_VERSION,
        "reference_path": state.reference_path, "reference_sha256": state.reference_sha256,
        "reference_role_sha256": dict(state.reference.role_sha256),
        "initialization_seed": state.initialization_seed,
        "model": state.model.state_dict(), "actor_optimizer": state.actor_optimizer.state_dict(),
        "critic_optimizer": state.critic_optimizer.state_dict(), "beta": dict(state.beta),
        "joint_steps": state.joint_steps, "actor_updates": state.actor_updates,
        "critic_updates": state.critic_updates, "rng": _rng_snapshot(),
        "extra_rng_states": deepcopy(state.extra_rng_states),
        "requires_grad": {k: v.requires_grad for k, v in state.model.named_parameters()}}
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        torch.save(value, handle)
    return {"path": str(target.resolve()), "sha256": _hash(target),
            "schema": CHECKPOINT_SCHEMA, "resumable_training_state": True}


def load_checkpoint(path, expected_sha256, *, reference_path=None, device="cpu", restore_rng=True):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("training checkpoint hash mismatch")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("model_version") != MODEL_VERSION:
        raise ValueError("not a v6.22 full training-state checkpoint")
    state = initialize_from_reference(reference_path or value["reference_path"], value["reference_sha256"],
        value["reference_role_sha256"], initialization_seed=value["initialization_seed"], device=device)
    state.model.load_state_dict(value["model"], strict=True)
    parameters = dict(state.model.named_parameters())
    if set(parameters) != set(value["requires_grad"]):
        raise ValueError("checkpoint trainable-parameter map differs")
    for name, enabled in value["requires_grad"].items():
        if type(enabled) is not bool:
            raise ValueError("checkpoint trainability must be boolean")
        parameters[name].requires_grad_(enabled)
    state.actor_optimizer.load_state_dict(value["actor_optimizer"])
    state.critic_optimizer.load_state_dict(value["critic_optimizer"])
    for name in ("joint_steps", "actor_updates", "critic_updates"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("checkpoint counters invalid")
        setattr(state, name, value[name])
    if set(value["beta"]) != set(ROLES):
        raise ValueError("checkpoint beta roles differ")
    for beta in value["beta"].values():
        if not BETA_MIN <= _finite_number(beta, "checkpoint beta") <= BETA_MAX:
            raise ValueError("checkpoint beta out of bounds")
    state.beta = dict(value["beta"])
    state.extra_rng_states = deepcopy(value["extra_rng_states"])
    state.resumed_training_state = True
    if restore_rng:
        _restore_rng(value["rng"])
    return state
