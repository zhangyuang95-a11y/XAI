"""One versioned PPO update with a separate two-endpoint soft-tree objective.

This is an update component, not a trainer, sampler, checkpoint reader or gate.
The caller owns legal two-role samples, the frozen tree/source binding, budgets,
refresh/capability guards, and persistence of native and auxiliary RNG state.
No auxiliary row supplies a PPO action, advantage, return or critic target.
"""
from numbers import Real
import math

import numpy as np
import torch

from .warehouse_native import ppo_actor_objective
from .warehouse_native_continuation_feedback_trainer import ContinuationFeedbackTrainer
from .warehouse_family_branch_feedback_loss import branch_feedback_loss, MAX_LAMBDA

VERSION = "warehouse-family-branch-ppo-update.v1"


class _OrdinaryView:
    """Explicit field adapter for the unchanged ordinary feedback algorithm."""
    def __init__(self, native, feedback):
        self.native, self.feedback = native, feedback

    def __getattr__(self, name):
        return getattr(self.native, name)


def _scalar(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be an explicit real scalar")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(name + " is outside its finite allowed range")
    return value


def _auxiliary(observations, weights):
    if not isinstance(observations, np.ndarray) or observations.dtype != np.float32:
        raise ValueError("auxiliary_observations must be NumPy float32 [pairs,2,197]")
    if observations.ndim != 3 or observations.shape[1:] != (2, 197):
        raise ValueError("auxiliary_observations must have shape [pairs,2,197]")
    weights = np.asarray(weights)
    if weights.dtype not in (np.dtype("float32"), np.dtype("float64")) or weights.shape != (len(observations),):
        raise ValueError("auxiliary_weights must be floating [pairs]")
    if not np.isfinite(observations).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Auxiliary observations/weights must be finite with nonnegative weights")
    with np.errstate(over="ignore"):
        total = weights.astype(np.float64).sum()
    if not np.isfinite(total):
        raise ValueError("Total auxiliary weight must be finite")
    return weights, np.flatnonzero(weights > 0)


def _adam_steps(optimizer):
    values = [int(s["step"].item()) for s in optimizer.state.values() if "step" in s]
    if values and len(set(values)) != 1:
        raise ValueError("Adam parameter step counters disagree")
    return values[0] if values else 0


def _sync_steps(native):
    for role in ("actor", "critic"):
        count = _adam_steps(getattr(native.optimizers, role))
        source = native.source_counters[role + "_optimizer_steps"]
        if count < source:
            raise ValueError("Adam counter precedes its native source")
        setattr(native, role + "_optimizer_steps", count - source)


def _norm(gradients, reference):
    value = torch.sqrt(sum((g.detach().square().sum() for g in gradients if g is not None),
                           reference.new_zeros(())))
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite diagnostic gradient")
    return float(value)


def _grad(loss, parameters):
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _finish(native, result, audit, before):
    audit["actor_adam_steps"] = _adam_steps(native.optimizers.actor) - before[0]
    audit["critic_adam_steps"] = _adam_steps(native.optimizers.critic) - before[1]
    audit["minibatch_updates"] = native.minibatch_updates - before[2]
    audit["optimizer_updates"] = native.optimizer_updates - before[3]
    return {**result, "branch_update": audit}


def branch_ppo_update(native, batch, feedback, auxiliary_observations,
                      auxiliary_weights, auxiliary_rng, branch_fraction=.5,
                      auxiliary_pairs_per_minibatch=64):
    """Update once, preserving native PPO ordering and one shared Actor Adam step.

    Inputs are legal frozen auxiliary observations [P,2,197], nonnegative weights
    [P], and an independent np.random.Generator. Endpoint order is WAIT, changed.
    Each NN-containing minibatch samples uniformly without replacement from
    positive-weight pairs; pairs may recur across minibatches. Weights affect KL,
    not the sampling probability. All five actions remain in both objectives.

    Positive feedback with effective auxiliary data uses (1-f)*ordinary_KL plus
    f*branch_KL at the SAME total lambda. None/zero lambda delegates to native
    update. Empty/zero-weight/f=0 auxiliary input delegates to the exact ordinary
    feedback loop at its FULL lambda. Neither fallback consumes auxiliary RNG.
    Statistics are detached; gradient norms are before the existing shared clip.
    There is no transaction rollback: a failure after an optimizer step requires
    caller-owned checkpoint/operation recovery, not automatic repeat invocation.
    """
    fraction = _scalar(branch_fraction, "branch_fraction", 1.)
    if type(auxiliary_pairs_per_minibatch) is not int or auxiliary_pairs_per_minibatch < 1:
        raise ValueError("auxiliary_pairs_per_minibatch must be a positive integer")
    if native.model.obs_dim != 197 or native.model.state_dim != 354:
        raise ValueError("The native contract requires observed197/state354")
    weights, eligible = _auxiliary(auxiliary_observations, auxiliary_weights)
    lam = 0. if feedback is None else _scalar(feedback.current_lambda, "feedback lambda", MAX_LAMBDA)
    if lam > 0 and (feedback.reliable is not True or feedback.program is None):
        raise ValueError("Positive lambda requires a reliable, present frozen tree")
    before = (_adam_steps(native.optimizers.actor), _adam_steps(native.optimizers.critic),
              native.minibatch_updates, native.optimizer_updates)
    audit = dict(version=VERSION, lambda_value=lam, branch_fraction=fraction,
        auxiliary_pool_pairs=len(weights), positive_weight_pool_pairs=len(eligible),
        auxiliary_rng_draws=0, auxiliary_pairs_used=0, auxiliary_endpoints_used=0,
        auxiliary_actor_forward_calls=0, auxiliary_tree_rows=0, ppo_actor_forward_calls=0,
        ppo_nn_row_visits=0, ordinary_tree_row_visits=0, minibatches=[],
        environment_steps=0, generated_ppo_rows=0, generated_actions=0,
        qualification_evaluated=False, persistence_owned_by_caller=True)
    if feedback is None or lam == 0:
        audit["path"] = "native_exact"
        result = native.update(batch)
        audit.update(ppo_actor_forward_calls=native.minibatch_updates-before[2],
            ppo_nn_row_visits=int((batch["trainable"] > 0).sum())*native.cfg["epochs"],
            instrumentation_scope="PPO counts derived from completed native minibatches and original trainable rows; auxiliary not called",
            gradient_scope="unchanged native metrics; no extra diagnostics")
        return _finish(native, result, audit, before)
    if not len(eligible) or fraction == 0:
        audit["path"] = "ordinary_feedback_exact"
        with native._rng_context():
            result = ContinuationFeedbackTrainer._feedback_update(_OrdinaryView(native, feedback), batch)
        _sync_steps(native)
        audit.update(ppo_actor_forward_calls=native.minibatch_updates-before[2],
            ppo_nn_row_visits=int((batch["trainable"] > 0).sum())*native.cfg["epochs"],
            ordinary_tree_row_visits=int((batch["trainable"] > 0).sum())*native.cfg["epochs"],
            instrumentation_scope="PPO/tree counts derived from completed frozen ordinary loop and original trainable rows; auxiliary not called",
            gradient_scope="unchanged ordinary feedback metrics; no extra diagnostics")
        return _finish(native, result, audit, before)
    if (not isinstance(auxiliary_rng, np.random.Generator) or auxiliary_rng is native.rng
            or auxiliary_rng.bit_generator is native.rng.bit_generator):
        raise ValueError("A distinct, caller-persisted auxiliary NumPy Generator is required")

    # Preserve the original batch preparation, including trainable-only GAE scale.
    obs = batch["observations"].reshape(-1, native.model.obs_dim)
    states = np.repeat(batch["states"][:, :, None, :], 2, axis=2).reshape(-1, native.model.state_dim)
    roles = np.tile([0, 1], len(obs) // 2); trainable = batch["trainable"].reshape(-1)
    advantages = batch["advantages"].reshape(-1).copy(); selected = trainable > 0
    if selected.any():
        advantages[selected] = (advantages[selected] - advantages[selected].mean()) / (advantages[selected].std() + 1e-8)
    data = {"obs": obs, "states": states, "roles": roles, "trainable": trainable,
        "actions": batch["actions"].reshape(-1), "old": batch["old_log_probs"].reshape(-1),
        "advantages": advantages, "returns": batch["returns"].reshape(-1)}
    data = {key: torch.as_tensor(value, device=native.device) for key, value in data.items()}
    # Freeze targets for this call before any parameter update. No NN query here.
    targets = None
    if selected.any():
        targets = np.asarray(feedback.targets(auxiliary_observations.reshape(-1, 197)))
        if (targets.shape != (2*len(weights), 5) or not np.issubdtype(targets.dtype, np.floating)
                or not np.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any()
                or not np.allclose(targets.sum(-1), 1., atol=1e-6, rtol=0.)):
            raise ValueError("Frozen auxiliary tree targets must be valid five-action distributions")
        targets = targets.reshape(-1, 2, 5).copy()
        audit["auxiliary_tree_rows"] = 2*len(weights)
    audit["path"] = "mixed_feedback"
    audit["gradient_scope"] = "separate actual PPO/ordinary/branch Actor gradients before shared clipping"
    parameters = tuple(native.model.actor.parameters()); metrics = []
    with native._rng_context():
        for epoch in range(native.cfg["epochs"]):
            permutation = native.rng.permutation(len(obs))
            for start in range(0, len(obs), native.cfg["minibatch"]):
                index = torch.as_tensor(permutation[start:start + native.cfg["minibatch"]], device=native.device)
                logits = native.model.actor_logits(data["obs"][index])
                audit["ppo_actor_forward_calls"] += 1
                actor_loss, part = ppo_actor_objective(logits, data["actions"][index], data["old"][index],
                    data["advantages"][index], data["trainable"][index], clip=native.cfg["clip"], entropy=native.cfg["entropy"])
                critic_loss = .5*(native.model.values(data["states"][index], data["roles"][index]) - data["returns"][index]).square().mean()
                mask = data["trainable"][index].bool(); count = int(mask.sum())
                audit["ppo_nn_row_visits"] += count
                ordinary_loss = branch_loss = logits.reshape(-1)[:0].sum()
                extra = dict(ordinary_feedback_loss=0., branch_feedback_loss=0., feedback_loss=0.,
                    ordinary_feedback_kl=0., branch_feedback_kl=0., ppo_gradient_norm=0.,
                    ordinary_feedback_gradient_norm=0., branch_feedback_gradient_norm=0.,
                    feedback_gradient_norm=0., feedback_rows=count)
                record = dict(epoch=epoch, start=start, ppo_rows=len(index), ppo_nn_rows=count,
                    auxiliary_pair_indices=[], branch_endpoint_logit_gradient_norms=[0., 0.])
                if count:
                    ordinary, evidence = feedback.loss(logits[mask], data["obs"][index][mask].detach().cpu().numpy())
                    ordinary_loss = (1.-fraction)*ordinary
                    audit["ordinary_tree_row_visits"] += count
                    chosen = auxiliary_rng.choice(eligible, size=min(len(eligible), auxiliary_pairs_per_minibatch), replace=False)
                    audit["auxiliary_rng_draws"] += 1
                    record["auxiliary_pair_indices"] = chosen.tolist()
                    aux_obs = torch.as_tensor(auxiliary_observations[chosen].reshape(-1, 197), device=native.device)
                    auxiliary_logits = native.model.actor_logits(aux_obs).reshape(-1, 2, 5)
                    audit["auxiliary_actor_forward_calls"] += 1
                    audit["auxiliary_pairs_used"] += len(chosen)
                    audit["auxiliary_endpoints_used"] += 2*len(chosen)
                    # Scale weights on CPU before conversion, keeping float32/MPS finite.
                    w = weights[chosen].astype(np.float64); w = w/w.max()
                    branch_loss, branch_stats = branch_feedback_loss(auxiliary_logits,
                        torch.as_tensor(targets[chosen], device=native.device, dtype=auxiliary_logits.dtype),
                        pair_weights=torch.as_tensor(w, device=native.device, dtype=auxiliary_logits.dtype),
                        lambda_value=lam*fraction)
                    ppo_grad = _grad(actor_loss, parameters)
                    ordinary_grad = _grad(ordinary_loss, parameters)
                    branch_all = _grad(branch_loss, (*parameters, auxiliary_logits))
                    branch_grad, endpoint_grad = branch_all[:-1], branch_all[-1]
                    combined = tuple((a if a is not None else torch.zeros_like(p)) +
                                     (b if b is not None else torch.zeros_like(p))
                                     for a, b, p in zip(ordinary_grad, branch_grad, parameters))
                    record["branch_endpoint_logit_gradient_norms"] = [
                        _norm((endpoint_grad[:, i],), logits) for i in range(2)]
                    record["branch_statistics"] = branch_stats
                    extra.update(ordinary_feedback_loss=float(ordinary_loss.detach()),
                        branch_feedback_loss=float(branch_loss.detach()),
                        feedback_loss=float((ordinary_loss+branch_loss).detach()),
                        ordinary_feedback_kl=evidence.get("kl", 0.),
                        branch_feedback_kl=branch_stats["weighted_mean_kl"],
                        ppo_gradient_norm=_norm(ppo_grad, logits),
                        ordinary_feedback_gradient_norm=_norm(ordinary_grad, logits),
                        branch_feedback_gradient_norm=_norm(branch_grad, logits),
                        feedback_gradient_norm=_norm(combined, logits))
                result = native.optimizers.step(actor_loss+ordinary_loss+branch_loss, critic_loss, train_actor=bool(count))
                metrics.append({**part, **result, **extra, "critic_loss": float(critic_loss.detach()),
                    "feedback_lambda": lam, "ordinary_feedback_lambda": lam*(1-fraction),
                    "branch_feedback_lambda": lam*fraction})
                record.update(extra); audit["minibatches"].append(record)
                native.minibatch_updates += 1
        native.optimizer_updates += 1
    _sync_steps(native)
    result = {key: float(np.mean([row.get(key, 0.) for row in metrics]))
              for key in set().union(*(row.keys() for row in metrics))}
    return _finish(native, result, audit, before)
