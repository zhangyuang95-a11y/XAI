"""Independent soft-tree loss on paired branch observations; no policy or gate.

The caller owns source/role admission, pair selection, capability guards and
optimizer scheduling. This component neither chooses actions nor creates PPO
samples. Its zero loss means zero contribution from this component, not a promise
that an optimizer with momentum or another loss cannot change parameters.
"""
from numbers import Real
import math

import torch

VERSION = 'warehouse-family-branch-soft-tree-feedback-loss.v1'
MAX_LAMBDA = .01
TARGET_FLOOR = 1e-8
PROBABILITY_ATOL = 1e-6


def branch_feedback_loss(actor_logits, tree_probabilities, *, pair_weights, lambda_value):
    """Return (loss, detached statistics) for tensors [pairs, 2, 5].

    Loss = lambda * sum(w[p] * mean_endpoint KL(pi_NN || pi_tree)) / sum(w).
    All five actions are retained. As in the existing on-policy loss, validated
    tree probabilities are floored at 1e-8 and renormalized before KL. Weights
    and tree targets are detached. Logits/targets/weights must be float32 or
    float64 tensors on one device; weights have shape [pairs]. Empty or inactive
    inputs are still validated, then return a differentiable zero without KL.
    """
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, Real):
        raise ValueError('lambda_value must be an explicit finite real scalar')
    lam = float(lambda_value)
    if not math.isfinite(lam) or not 0 <= lam <= MAX_LAMBDA:
        raise ValueError('lambda_value must be in [0, .01]')
    for name, value in (('actor_logits', actor_logits), ('tree_probabilities', tree_probabilities),
                        ('pair_weights', pair_weights)):
        if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
            raise ValueError(name + ' must be a float32 or float64 tensor')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + ' must be finite')
        if value.device != actor_logits.device:
            raise ValueError('All branch tensors must be on the same device')
    if actor_logits.ndim != 3 or actor_logits.shape[1:] != (2, 5):
        raise ValueError('actor_logits must have shape [pairs, 2, 5]')
    if tree_probabilities.shape != actor_logits.shape:
        raise ValueError('Tree and NN endpoints must have identical shapes')
    pairs = actor_logits.shape[0]
    if pair_weights.shape != (pairs,):
        raise ValueError('pair_weights must have shape [pairs]')
    targets = tree_probabilities.detach()
    weights = pair_weights.detach()
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        raise ValueError('Tree probabilities must be in [0, 1]')
    if not bool(torch.isclose(targets.sum(-1), torch.ones_like(targets[..., 0]),
                              atol=PROBABILITY_ATOL, rtol=0.).all()):
        raise ValueError('Each tree endpoint must sum to one')
    if bool((weights < 0).any()):
        raise ValueError('Pair weights must be nonnegative')
    weight_sum = float(weights.detach().cpu().to(torch.float64).sum())
    if not math.isfinite(weight_sum):
        raise ValueError('Total pair weight must be finite')
    positive = int((weights > 0).sum().item())
    stats = {'version': VERSION, 'lambda': lam, 'pairs': pairs, 'endpoints': 2*pairs,
        'positive_weight_pairs': positive, 'positive_weight_endpoints': 2*positive,
        'zero_weight_pairs': pairs-positive, 'pair_weight_sum': weight_sum,
        'effective_pair_weight_sum': weight_sum if lam > 0 else 0.,
        'active': bool(lam > 0 and positive), 'kl_computed': False,
        'weighted_mean_kl': 0., 'weighted_endpoint_mean_kl': [0., 0.], 'loss': 0.,
        'target_probability_floor': TARGET_FLOOR,
        'target_components_below_floor': int((targets < TARGET_FLOOR).sum().item()),
        'tree_target_has_gradient': False, 'pair_weights_have_gradient': False,
        'complexity_has_actor_gradient': False, 'qualification_evaluated': False,
        'actions_generated': 0, 'ppo_samples_generated': 0}
    if not stats['active']:
        # An empty slice avoids sum-overflow for large but finite inactive logits.
        return actor_logits.reshape(-1)[:0].sum(), stats
    dtype = torch.promote_types(actor_logits.dtype, tree_probabilities.dtype)
    dtype = torch.promote_types(dtype, pair_weights.dtype)
    targets = targets.to(dtype).clamp_min(TARGET_FLOOR)
    targets = targets / targets.sum(-1, keepdim=True)
    log_pi = torch.log_softmax(actor_logits.to(dtype), dim=-1)
    endpoint_kl = (log_pi.exp() * (log_pi-targets.log())).sum(-1)
    # Normalize after scaling, avoiding a float32 overflow in a large weight sum.
    normalized = weights.to(dtype) / weights.max().to(dtype)
    normalized = normalized / normalized.sum()
    endpoint_mean = (endpoint_kl * normalized[:, None]).sum(0)
    kl = endpoint_mean.mean()
    loss = lam * kl
    if not bool(torch.isfinite(endpoint_kl).all()) or not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite KL from the supplied finite tensor magnitudes')
    stats.update(kl_computed=True, weighted_mean_kl=float(kl.detach().cpu()),
        weighted_endpoint_mean_kl=endpoint_mean.detach().cpu().tolist(),
        loss=float(loss.detach().cpu()))
    return loss, stats
