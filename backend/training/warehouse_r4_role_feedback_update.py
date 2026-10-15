"""Versioned native PPO plus an explicitly bounded direct soft-tree KL update.

No manager identity substitution, policy action, sampling, or capability gate is
provided here. The owning trainer binds the Actor/tree/data, admits feedback,
sets learning rates, and persists the native and independent auxiliary RNGs.
"""
import numpy as np
import torch

from .warehouse_native import ppo_actor_objective
from . import warehouse_native_program_batch as prediction
from .warehouse_family_branch_ppo_update import _scalar, _adam_steps, _sync_steps, _grad, _norm

VERSION = "warehouse-r4-role-feedback-update.v1"
MAX_LAMBDA = .2
AUXILIARY_PAIRS_PER_MINIBATCH = 64
TARGET_FLOOR = 1e-8


def _kl(logits, probabilities):
    """Original all-action KL(NN||tree), with detached normalized soft targets."""
    targets = torch.as_tensor(probabilities, device=logits.device, dtype=logits.dtype).detach()
    if (logits.ndim < 2 or logits.shape[-1] != 5 or targets.shape != logits.shape
            or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all())
            or bool((targets < 0).any()) or bool((targets > 1).any())
            or not bool(torch.isclose(targets.sum(-1), torch.ones_like(targets[..., 0]),
                                      atol=1e-6, rtol=0.).all())):
        raise ValueError("Finite aligned logits and normalized five-action tree targets required")
    if not logits.numel(): return logits.reshape(-1)[:0].sum()
    targets = targets.clamp_min(TARGET_FLOOR)
    targets = targets / targets.sum(-1, keepdim=True)
    log_pi = torch.log_softmax(logits, dim=-1)
    result = (log_pi.exp() * (log_pi - targets.log())).sum(-1).mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("Nonfinite KL from supplied tensor magnitudes")
    return result


def _auxiliary(observations):
    if observations is None: return np.empty((0, 2, 197), np.float32)
    if (not isinstance(observations, np.ndarray) or observations.dtype != np.float32
            or observations.ndim != 3 or observations.shape[1:] != (2, 197)
            or not np.isfinite(observations).all()):
        raise ValueError("Auxiliary observations must be finite float32 [pairs,2,197]")
    return observations


def _finish(native, result, audit, before):
    audit.update(actor_adam_steps=_adam_steps(native.optimizers.actor)-before[0],
        critic_adam_steps=_adam_steps(native.optimizers.critic)-before[1],
        minibatch_updates=native.minibatch_updates-before[2],
        optimizer_updates=native.optimizer_updates-before[3])
    return {**result, "stable_update": audit}


def stable_ppo_update(native, batch, program=None, *, lambda_value=0,
                      auxiliary_observations=None, auxiliary_rng=None,
                      feedback_role=None, feedback_transform=None,
                      feedback_feature_names=None):
    """One native update; positive lambda <=.2 shares the SAME Actor Adam step.

    With nonempty auxiliary pairs, total lambda is split 50/50 between ordinary
    trainable PPO observations and uniformly sampled, equally weighted paired
    endpoints (at most64 pairs per NN-containing minibatch). Empty/None auxiliary
    input assigns full lambda to ordinary KL. No auxiliary row enters PPO/critic.
    Zero lambda delegates to native.update without inspecting tree/auxiliary or
    consuming auxiliary RNG. Positive updates keep native PPO/critic/shuffle and
    source-relative Adam counters. Diagnostic gradients are before shared clip.
    Failures are not rolled back; caller-owned durable recovery must prevent replay.
    """
    lam = _scalar(lambda_value, "lambda_value", MAX_LAMBDA)
    if feedback_role not in (None, 0, 1):
        raise ValueError("feedback_role must be None,0,or1")
    before = (_adam_steps(native.optimizers.actor), _adam_steps(native.optimizers.critic),
              native.minibatch_updates, native.optimizer_updates)
    audit = dict(version=VERSION, lambda_value=lam, maximum_lambda=MAX_LAMBDA,
        auxiliary_rng_draws=0, auxiliary_pool_pairs=0, auxiliary_pairs_used=0,
        auxiliary_endpoints_used=0, auxiliary_actor_forward_calls=0,
        auxiliary_tree_rows=0, ordinary_tree_rows=0, ordinary_kl_row_visits=0,
        ppo_actor_forward_calls=0, critic_forward_calls=0, ppo_nn_row_visits=0,
        minibatches=[], generated_ppo_rows=0, generated_actions=0, environment_steps=0,
        qualification_evaluated=False, capability_admission_owned_by_caller=True,
        feedback_role=feedback_role,
        complexity_has_actor_gradient=False, tree_targets_detached=True,
        persistence_owned_by_caller=True)
    if lam == 0:
        native_update = getattr(native, "native_ppo_update", native.update)
        result = native_update(batch)
        completed = native.minibatch_updates-before[2]
        audit.update(path="native_exact", ppo_actor_forward_calls=completed,
            critic_forward_calls=completed,
            ppo_nn_row_visits=int((batch["trainable"] > 0).sum())*native.cfg["epochs"],
            instrumentation_scope="fallback counts derived from completed native minibatches and input rows; auxiliary not inspected",
            gradient_scope="original native metrics; no extra diagnostic backward")
        return _finish(native, result, audit, before)
    if native.model.obs_dim != 197 or native.model.state_dim != 354:
        raise ValueError("Positive feedback requires observed197/state354")
    aux = _auxiliary(auxiliary_observations)
    if feedback_role is not None and len(aux):
        raise ValueError("Role-scoped feedback does not accept paired auxiliary rows")
    if len(aux) and (not isinstance(auxiliary_rng, np.random.Generator)
            or auxiliary_rng is native.rng or auxiliary_rng.bit_generator is native.rng.bit_generator):
        raise ValueError("Independent caller-persisted auxiliary NumPy Generator required")
    actor_names = tuple(native.envs[0].feature_names)
    names = tuple(feedback_feature_names or actor_names)
    obs = batch["observations"].reshape(-1, native.model.obs_dim)
    states = np.repeat(batch["states"][:, :, None, :], 2, axis=2).reshape(-1, native.model.state_dim)
    roles = np.tile([0, 1], len(obs)//2); trainable = batch["trainable"].reshape(-1)
    if not np.isin(trainable, (0, 1)).all():
        raise ValueError("PPO row ownership must be explicit zero/one trainable flags")
    advantages = batch["advantages"].reshape(-1).copy(); selected = trainable > 0
    feedback_selected = selected if feedback_role is None else (selected & (roles == feedback_role))
    if selected.any():
        advantages[selected] = (advantages[selected]-advantages[selected].mean())/(advantages[selected].std()+1e-8)
    # The actual pure predictor enforces feature/action order and unmasked program
    # identity, even for an empty selected subset. No Actor forward is used here.
    ordinary_targets = np.zeros((len(obs), 5), np.float32)
    selected_observations = obs[feedback_selected]
    if feedback_transform is not None:
        transformed = feedback_transform(selected_observations, actor_names)
        if (not isinstance(transformed, np.ndarray) or transformed.dtype != np.float32
                or transformed.ndim != 2 or transformed.shape != (len(selected_observations), len(names))
                or not np.isfinite(transformed).all()):
            raise ValueError("Feedback transform must deterministically return finite float32 rows")
        selected_observations = transformed
    elif names != actor_names:
        raise ValueError("Non-Actor feedback schema requires an explicit public transform")
    ordinary_targets[feedback_selected] = prediction.predict(program, selected_observations, names)
    aux_targets = None
    if selected.any() and len(aux):
        aux_targets = prediction.predict(program, aux.reshape(-1, 197), names).reshape(-1, 2, 5)
        audit["auxiliary_tree_rows"] = 2*len(aux)
    audit.update(path="mixed_feedback" if len(aux) else "ordinary_feedback",
        auxiliary_pool_pairs=len(aux), ordinary_tree_rows=int(feedback_selected.sum()),
        gradient_scope="actual separate PPO/ordinary/auxiliary and combined feedback Actor gradients before shared clip",
        instrumentation_scope="direct counters inside this versioned positive-lambda loop")
    data = dict(obs=obs, states=states, roles=roles, trainable=trainable,
        actions=batch["actions"].reshape(-1), old=batch["old_log_probs"].reshape(-1),
        advantages=advantages, returns=batch["returns"].reshape(-1), targets=ordinary_targets)
    data = {k: torch.as_tensor(v, device=native.device) for k, v in data.items()}
    fraction = .5 if len(aux) else 0.
    parameters = tuple(native.model.actor.parameters()); metrics = []
    with native._rng_context():
        for epoch in range(native.cfg["epochs"]):
            permutation = native.rng.permutation(len(obs))
            for start in range(0, len(obs), native.cfg["minibatch"]):
                index = torch.as_tensor(permutation[start:start+native.cfg["minibatch"]], device=native.device)
                logits = native.model.actor_logits(data["obs"][index])
                audit["ppo_actor_forward_calls"] += 1
                actor_loss, part = ppo_actor_objective(logits, data["actions"][index], data["old"][index],
                    data["advantages"][index], data["trainable"][index],
                    clip=native.cfg["clip"], entropy=native.cfg["entropy"])
                critic_loss = .5*(native.model.values(data["states"][index], data["roles"][index])-data["returns"][index]).square().mean()
                audit["critic_forward_calls"] += 1
                mask = data["trainable"][index].bool(); count = int(mask.sum())
                feedback_mask = mask if feedback_role is None else (
                    mask & (data["roles"][index] == int(feedback_role)))
                feedback_count = int(feedback_mask.sum())
                audit["ppo_nn_row_visits"] += count
                ordinary_loss = auxiliary_loss = logits.reshape(-1)[:0].sum()
                extra = dict(ordinary_feedback_loss=0., auxiliary_feedback_loss=0., feedback_loss=0.,
                    ordinary_feedback_kl=0., auxiliary_feedback_kl=0., ppo_gradient_norm=0.,
                    ordinary_feedback_gradient_norm=0., auxiliary_feedback_gradient_norm=0.,
                    feedback_gradient_norm=0., feedback_rows=feedback_count)
                record = dict(epoch=epoch, start=start, ppo_rows=len(index), ppo_nn_rows=count,
                    feedback_rows=feedback_count,
                    auxiliary_pair_indices=[], auxiliary_endpoint_logit_gradient_norms=[0., 0.])
                if feedback_count:
                    ordinary_kl = _kl(logits[feedback_mask], data["targets"][index][feedback_mask])
                    ordinary_loss = lam*(1.-fraction)*ordinary_kl
                    audit["ordinary_kl_row_visits"] += feedback_count
                    auxiliary_grad = tuple(None for _ in parameters)
                    if len(aux):
                        chosen = auxiliary_rng.choice(len(aux), size=min(len(aux), AUXILIARY_PAIRS_PER_MINIBATCH), replace=False)
                        audit["auxiliary_rng_draws"] += 1
                        record["auxiliary_pair_indices"] = chosen.tolist()
                        auxiliary_logits = native.model.actor_logits(torch.as_tensor(aux[chosen].reshape(-1, 197),
                            device=native.device)).reshape(-1, 2, 5)
                        audit["auxiliary_actor_forward_calls"] += 1
                        audit["auxiliary_pairs_used"] += len(chosen)
                        audit["auxiliary_endpoints_used"] += 2*len(chosen)
                        auxiliary_kl = _kl(auxiliary_logits, aux_targets[chosen])
                        auxiliary_loss = lam*fraction*auxiliary_kl
                        grad_all = _grad(auxiliary_loss, (*parameters, auxiliary_logits))
                        auxiliary_grad, endpoint_grad = grad_all[:-1], grad_all[-1]
                        record["auxiliary_endpoint_logit_gradient_norms"] = [_norm((endpoint_grad[:, i],), logits) for i in range(2)]
                        extra["auxiliary_feedback_kl"] = float(auxiliary_kl.detach())
                    ppo_grad = _grad(actor_loss, parameters)
                    ordinary_grad = _grad(ordinary_loss, parameters)
                    combined = tuple((a if a is not None else torch.zeros_like(p)) +
                        (b if b is not None else torch.zeros_like(p)) for a, b, p in zip(ordinary_grad, auxiliary_grad, parameters))
                    extra.update(ordinary_feedback_kl=float(ordinary_kl.detach()),
                        ordinary_feedback_loss=float(ordinary_loss.detach()),
                        auxiliary_feedback_loss=float(auxiliary_loss.detach()),
                        feedback_loss=float((ordinary_loss+auxiliary_loss).detach()),
                        ppo_gradient_norm=_norm(ppo_grad, logits),
                        ordinary_feedback_gradient_norm=_norm(ordinary_grad, logits),
                        auxiliary_feedback_gradient_norm=_norm(auxiliary_grad, logits),
                        feedback_gradient_norm=_norm(combined, logits))
                result = native.optimizers.step(actor_loss+ordinary_loss+auxiliary_loss, critic_loss, train_actor=bool(count))
                metrics.append({**part, **result, **extra, "critic_loss": float(critic_loss.detach()),
                    "feedback_lambda": lam, "ordinary_feedback_lambda": lam*(1.-fraction),
                    "auxiliary_feedback_lambda": lam*fraction})
                record.update(extra); audit["minibatches"].append(record)
                native.minibatch_updates += 1
        native.optimizer_updates += 1
    _sync_steps(native)
    result = {k: float(np.mean([row.get(k, 0.) for row in metrics])) for k in set().union(*(m.keys() for m in metrics))}
    return _finish(native, result, audit, before)
