"""Training-only tree/full-Actor alignment; no runtime action rule."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES


_STREAK = OBSERVATION_FEATURE_NAMES.index("same_target_collision_streak_capped")
_DISTANCE = OBSERVATION_FEATURE_NAMES.index("robots_shortest_distance")
_RETURN_MARGIN = OBSERVATION_FEATURE_NAMES.index("self_charger_return_margin")
_WEIGHTS = {"tree_ce", "full_to_tree_kl", "tree_to_full_kl", "gate", "cert_margin"}
_MARGIN_DELTA = 0.26
_MARGIN_TEMPERATURE = 0.25


def core_intrinsic_loss(model, encoded: torch.Tensor, masks: torch.Tensor,
                        *, labels: torch.Tensor | None,
                        weights: Mapping[str, float],
                        full_logits: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, dict]:
    """Recruit stable training states even when the old core mask is empty.

    Public labels supervise the tree on *all* BC rows.  Detached complete-NN
    probabilities teach the tree on stable pre-action rows, including mixed
    leaves.  Reverse KL and a tree-margin/gate objective act only where the
    tree's current action is compatible with the public BC label (BC) or the
    full Actor's current top action (PPO).  PPO supplies no teacher label.
    Independent eligibility/validation/program rows never enter this loss.
    """
    if (set(weights) != _WEIGHTS or
            any(not 0 <= float(value) <= 10 for value in weights.values())):
        raise ValueError("invalid Warehouse intrinsic loss weights")
    if (encoded.ndim != 3 or encoded.shape[1] != len(AGENTS)
            or masks.shape != encoded.shape[:2] + (len(ACTIONS),)
            or not torch.isfinite(encoded).all()
            or not torch.isfinite(masks).all()
            or torch.any((masks != 0) & (masks != 1))
            or torch.any(masks.sum(-1) < 1)):
        raise ValueError("invalid Warehouse intrinsic observation/mask shape")
    if labels is not None and (labels.shape != encoded.shape[:2]
                               or torch.any(labels < 0)
                               or torch.any(labels >= len(ACTIONS))
                               or torch.any(masks.gather(-1, labels[..., None]) <= 0.5)):
        raise ValueError("invalid Warehouse intrinsic public labels")
    if full_logits is None:
        full_logits = model.outputs_encoded(encoded, masks)
    if (full_logits.shape != masks.shape or torch.isnan(full_logits).any()
            or torch.isposinf(full_logits).any()):
        raise ValueError("invalid Warehouse intrinsic full logits")

    total = encoded.sum() * 0
    fields = ("bc_tree_rows", "candidate_rows", "candidate_fraction", "aligned_rows",
              "raw_illegal_rows", "tree_ce", "full_to_tree_kl", "tree_to_full_kl",
              "cert_margin_loss", "gate_mean", "mean_tree_margin",
              "expected_sampled_actionable")
    summary = {name: {} for name in fields}
    for index, role in enumerate(AGENTS):
        actor = model.actors[role]
        row, physical = encoded[:, index], masks[:, index]
        tree_logits, _ = actor.tree(row)
        tree = tree_logits.masked_fill(physical <= 0.5, -1e9)
        full = full_logits[:, index]
        tree_top, full_top = tree.argmax(-1), full.argmax(-1)
        raw_top = tree_logits.argmax(-1)
        raw_legal = physical.gather(-1, raw_top[:, None]).squeeze(-1) > 0.5
        summary["raw_illegal_rows"][role] = int((~raw_legal).sum())
        role_loss = tree_logits.sum() * 0
        if labels is not None:
            ce = nn.functional.cross_entropy(tree, labels[:, index])
            role_loss = role_loss + float(weights["tree_ce"]) * ce
            summary["tree_ce"][role] = float(ce.detach())
            summary["bc_tree_rows"][role] = len(row)
        else:
            summary["tree_ce"][role] = 0.0
            summary["bc_tree_rows"][role] = 0

        # Only pre-action public measurements gate this *training loss*.
        # Core qualification is intentionally absent: a zero-core tree still
        # receives gradients.  Close/conflict/low-return rows keep residual
        # flexibility for the reward and true-event cost objectives.
        candidate = (raw_legal & (row[:, _STREAK] <= 0)
                     & (row[:, _DISTANCE] * 42 > 2)
                     & (row[:, _RETURN_MARGIN] * 50 > 4))
        summary["candidate_rows"][role] = int(candidate.sum())
        summary["candidate_fraction"][role] = float(candidate.float().mean())
        if bool(candidate.any()):
            log_tree = nn.functional.log_softmax(tree[candidate], -1)
            log_full = nn.functional.log_softmax(full[candidate], -1)
            p_full = log_full.detach().exp()
            f2t = (p_full * (log_full.detach() - log_tree)).sum(-1).mean()
            role_loss = role_loss + float(weights["full_to_tree_kl"]) * f2t
            summary["full_to_tree_kl"][role] = float(f2t.detach())
        else:
            summary["full_to_tree_kl"][role] = 0.0

        aligned = candidate & (tree_top == (labels[:, index] if labels is not None
                                            else full_top))
        summary["aligned_rows"][role] = int(aligned.sum())
        if bool(aligned.any()):
            t, f = tree[aligned], full[aligned]
            log_t = nn.functional.log_softmax(t, -1)
            log_f = nn.functional.log_softmax(f, -1)
            q_tree = log_t.detach().exp()
            t2f = (q_tree * (log_t.detach() - log_f)).sum(-1).mean()
            gate = actor.gate_values(row[aligned])
            _, _, bound = actor.components(row[aligned])
            chosen = tree_top[aligned]
            others = t.masked_fill(physical[aligned] <= 0.5, -1e9).scatter(
                -1, chosen[:, None], -1e9)
            margin = t.gather(-1, chosen[:, None]).squeeze(-1) - others.max(-1).values
            margin = torch.where((physical[aligned] > 0.5).sum(-1) > 1,
                                 margin, torch.full_like(margin, 1e9))
            margin_loss = (_MARGIN_TEMPERATURE * nn.functional.softplus(
                (2 * gate * bound + _MARGIN_DELTA - margin) /
                _MARGIN_TEMPERATURE)).mean()
            role_loss = (role_loss + float(weights["tree_to_full_kl"]) * t2f
                         + float(weights["gate"]) * gate.mean()
                         + float(weights["cert_margin"]) * margin_loss)
            summary["tree_to_full_kl"][role] = float(t2f.detach())
            summary["cert_margin_loss"][role] = float(margin_loss.detach())
            summary["gate_mean"][role] = float(gate.detach().mean())
            summary["mean_tree_margin"][role] = float(margin.detach().mean())
            summary["expected_sampled_actionable"][role] = float(
                nn.functional.softmax(f.detach(), -1).gather(-1, chosen[:, None]).mean())
        else:
            for key in ("tree_to_full_kl", "cert_margin_loss", "gate_mean",
                        "mean_tree_margin", "expected_sampled_actionable"):
                summary[key][role] = 0.0
        total = total + role_loss / len(AGENTS)
    return total, summary
