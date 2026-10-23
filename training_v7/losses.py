"""Action-count independent objectives, used identically in all three domains."""
import math
import torch
from torch.nn import functional as F
from .contracts import MethodSpec
from .model import mask_logits


def masked_kl(target_logits, logits, masks):
    target = mask_logits(target_logits.detach().double(), masks).softmax(-1)
    logq = mask_logits(logits.double(), masks).log_softmax(-1).masked_fill(~masks.bool(), 0.)
    return (torch.xlogy(target, target) - target * logq).sum(-1).clamp_min(0.)


def bc_loss(full, tree, primary, masks, labels):
    method = MethodSpec()
    if labels.dtype != torch.long or not masks.bool().gather(-1, labels[..., None]).all():
        raise ValueError('public teacher labels must be legal')
    f, t = mask_logits(full, masks), mask_logits(tree, masks)
    full_ce, tree_ce = F.cross_entropy(f, labels), F.cross_entropy(t, labels)
    distill = masked_kl(full, tree, masks).mean()
    mean = primary.masked_fill(~masks.bool(), 0.).sum(-1) / masks.sum(-1)
    centered = (primary - mean[..., None]).masked_fill(~masks.bool(), 0.)
    correct = t.detach().argmax(-1) == labels
    alignment = (centered.square().sum(-1) / masks.sum(-1) * correct).mean()
    return full_ce + method.bc_tree_ce * tree_ce + method.bc_distillation * distill + method.bc_alignment * alignment


def reference_loss(current, reference, masks, sources, beta):
    if (len(sources) != len(current) or len(current) == 0 or len(current) % 2
            or sources.count('reference_train') != len(sources) // 2
            or sources.count('current_train') != len(sources) // 2):
        raise ValueError('reference KL requires a 50:50 train-only provenance mix')
    return float(beta) * masked_kl(reference, current, masks).mean()


def next_beta(beta, observed):
    s = MethodSpec()
    if not s.beta_min <= beta <= s.beta_max or not math.isfinite(observed) or observed < 0:
        raise ValueError('invalid reference dual state')
    return math.exp(min(math.log(s.beta_max), max(math.log(s.beta_min),
                    math.log(beta) + s.beta_rate * (observed / s.reference_kl_target - 1))))


def program_loss(logits, q, masks, sampled, qualified, raw_top, *, variant, coefficient):
    if variant not in ('raw', 'regularized') or not 0 <= coefficient <= MethodSpec().program_lambda:
        raise ValueError('invalid treatment or coefficient')
    if variant == 'raw':
        return logits.masked_fill(~masks.bool(), 0.).sum() * 0., torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
    if (q.shape != logits.shape or not torch.isfinite(q).all() or (q < 0).any()
            or not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-6)
            or (q.masked_select(~masks.bool()) != 0).any()):
        raise ValueError('program targets must be complete masked distributions')
    p = mask_logits(logits, masks)
    allowed = masks.bool().gather(-1, raw_top[..., None]).squeeze(-1)
    selected = qualified.detach().bool() & allowed & (raw_top == sampled) & (sampled == p.detach().argmax(-1))
    logp = p.log_softmax(-1).masked_fill(~masks.bool(), 0.)
    target = q.detach().to(logits)
    kl = (torch.xlogy(target, target) - target * logp).sum(-1)
    return coefficient * (kl * selected).mean(), selected


def risk_config():
    from backend.training.warehouse_v6_shutdown_cost import RiskDualConfig
    s = MethodSpec()
    return RiskDualConfig(target=s.risk_target, release_upper=s.risk_release,
                          window_episodes=s.risk_window, minimum_episodes=s.risk_minimum,
                          confidence=s.risk_confidence)
