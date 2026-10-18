"""Public-record-only shared BC alignment for the optional v6.20 route.

This module computes training losses. It neither collects environment rows,
changes Actor inference, nor consumes an extracted explanation program. The
caller must separately hold a source/recipe/seed-bound BC capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS

if TYPE_CHECKING:
    from .warehouse_v6_actor_training import WarehouseV6Actor


ALIGNMENT_RECIPE_VERSION = "warehouse-public-shared-bc-alignment.v6.20.r1"
ALLOWED_GAMMAS = (0.01, 0.03)
SHARED_BC_PHASE = "shared_bc"


@dataclass(frozen=True)
class BCAlignmentRecipe:
    """The two preregistered BC recipes vary only in residual soft penalty."""

    gamma: float

    def __post_init__(self) -> None:
        if (isinstance(self.gamma, bool) or not isinstance(self.gamma, Real)
                or float(self.gamma) not in ALLOWED_GAMMAS):
            raise ValueError("Warehouse BC alignment gamma must be 0.01 or 0.03")

    def as_dict(self) -> dict:
        return {
            "version": ALIGNMENT_RECIPE_VERSION,
            "phase": SHARED_BC_PHASE,
            "full_task_weight": 1.0,
            "internal_tree_task_weight": 0.2,
            "full_to_internal_tree_kl_weight": 0.2,
            "full_to_tree_target": "stop_gradient_complete_masked_nn_distribution",
            "residual_penalty_weight": float(self.gamma),
            "residual_penalty": "mean_squared_legal_centered_primary_logits",
            "penalty_region": "masked_internal_tree_argmax_equals_public_bc_teacher",
            "penalty_denominator": "all_joint_role_training_rows",
            "sample_weight_normalization": "mean_over_joint_role_training_rows",
            "teacher_source": "pre_action_public_training_records_only",
            "external_program_targets_or_gradients": False,
            "ppo_auxiliary_loss": 0.0,
            "inference_postprocessing": False,
        }


@dataclass(frozen=True)
class BCAlignmentLoss:
    total: torch.Tensor
    full_task: torch.Tensor
    internal_tree_task: torch.Tensor
    full_to_internal_tree_kl: torch.Tensor
    residual_penalty: torch.Tensor
    correct_tree_fraction: torch.Tensor
    # [role, component], unweighted coefficients but weighted training rows.
    # Columns are full CE, tree CE, detached-full->tree KL, residual penalty.
    per_role_components: torch.Tensor

    def metrics(self) -> dict:
        return {
            "loss": float(self.total.detach()),
            "full_task_ce": float(self.full_task.detach()),
            "internal_tree_task_ce": float(self.internal_tree_task.detach()),
            "full_to_internal_tree_kl": float(self.full_to_internal_tree_kl.detach()),
            "residual_penalty": float(self.residual_penalty.detach()),
            "correct_tree_fraction": float(self.correct_tree_fraction.detach()),
            "per_role_components": {
                role: self.per_role_components[index].detach().cpu().tolist()
                for index, role in enumerate(AGENTS)
            },
        }


def shared_bc_alignment_loss(
        full_logits: torch.Tensor, internal_tree_logits: torch.Tensor,
        primary_logits: torch.Tensor, masks: torch.Tensor,
        public_teacher_actions: torch.Tensor, *, recipe: BCAlignmentRecipe,
        phase: str, sample_weights: torch.Tensor | None = None) -> BCAlignmentLoss:
    """Compute a joint [physical frame, independent role, five actions] loss.

    The internal tree is the Actor's raw trainable branch before its learned
    additive gate. Primary logits are the unchanged 256-wide NN contribution.
    The gate and primary remain trainable through the full public task loss.
    Only the KL target and the correct-tree region decision are detached.
    """
    if phase != SHARED_BC_PHASE:
        raise ValueError("Warehouse alignment is restricted to shared BC, never PPO")
    if not isinstance(recipe, BCAlignmentRecipe):
        raise TypeError("Warehouse BC alignment requires an immutable recipe")
    shape = tuple(full_logits.shape)
    if (full_logits.ndim != 3 or shape[0] == 0
            or shape[1:] != (len(AGENTS), len(ACTIONS))):
        raise ValueError("Warehouse BC alignment requires nonempty [frame, 2, 5] logits")
    for value in (full_logits, internal_tree_logits, primary_logits):
        if (tuple(value.shape) != shape or not value.is_floating_point()
                or value.device != full_logits.device
                or value.dtype != full_logits.dtype
                or not torch.isfinite(value).all()):
            raise ValueError("Warehouse BC alignment logit shape, dtype or values invalid")
    if (tuple(masks.shape) != shape or masks.device != full_logits.device
            or not torch.isfinite(masks).all()
            or not torch.all((masks == 0) | (masks == 1))):
        raise ValueError("Warehouse BC alignment requires binary physical masks")
    legal = masks.bool()
    if not legal.any(-1).all():
        raise ValueError("Warehouse BC alignment encountered an empty physical mask")
    labels = public_teacher_actions
    if (tuple(labels.shape) != shape[:-1] or labels.dtype != torch.long
            or labels.device != full_logits.device
            or torch.any(labels < 0) or torch.any(labels >= len(ACTIONS))):
        raise ValueError("Warehouse BC alignment public teacher labels invalid")
    if not legal.gather(-1, labels.unsqueeze(-1)).all():
        raise ValueError("Warehouse BC alignment public teacher chose an illegal action")
    if sample_weights is None:
        weights = torch.ones_like(full_logits[..., 0])
    else:
        if (tuple(sample_weights.shape) != shape[:-1]
                or sample_weights.device != full_logits.device
                or not sample_weights.is_floating_point()
                or not torch.isfinite(sample_weights).all()
                or torch.any(sample_weights < 0) or not sample_weights.sum() > 0):
            raise ValueError("Warehouse BC alignment sample weights invalid")
        # Episode/public risk weights are data, not a trainable routing gate.
        weights = sample_weights.detach().to(full_logits.dtype)
        weights = weights / weights.mean()

    # Finite fill avoids 0 * -inf during complete-distribution KL. Physical
    # illegal actions have no gradient and zero probability after softmax.
    fill = torch.finfo(full_logits.dtype).min
    full = full_logits.masked_fill(~legal, fill)
    tree = internal_tree_logits.masked_fill(~legal, fill)
    full_ce = F.cross_entropy(full.reshape(-1, len(ACTIONS)), labels.reshape(-1),
                              reduction="none").reshape(shape[:-1])
    tree_ce = F.cross_entropy(tree.reshape(-1, len(ACTIONS)), labels.reshape(-1),
                              reduction="none").reshape(shape[:-1])
    target = F.softmax(full.detach(), -1)
    # torch.xlogy defines 0*log(0) as zero without adding probability mass.
    kl = (torch.xlogy(target, target)
          - target * F.log_softmax(tree, -1)).sum(-1)
    correct = (tree.detach().argmax(-1) == labels).to(full_logits.dtype)
    legal_count = legal.sum(-1).to(full_logits.dtype)
    legal_primary = primary_logits.masked_fill(~legal, 0.0)
    legal_mean = legal_primary.sum(-1) / legal_count
    centered = (primary_logits - legal_mean.unsqueeze(-1)).masked_fill(~legal, 0.0)
    # A common shift across legal logits does not change any policy. Penalize
    # only policy-relevant residual variation, never an irrelevant intercept.
    residual = centered.square().sum(-1) / legal_count * correct
    # Keep scalar graphs separate: asking only for the KL must not attach a
    # zero-valued full-policy CE graph to its detached target.
    weighted = tuple(value * weights for value in (full_ce, tree_ce, kl, residual))
    means = tuple(value.mean() for value in weighted)
    components = torch.stack(weighted, dim=-1)
    total = means[0] + 0.2 * means[1] + 0.2 * means[2] + float(recipe.gamma) * means[3]
    return BCAlignmentLoss(total, means[0], means[1], means[2], means[3],
                           correct.mean(), components.mean(0))


def actor_shared_bc_alignment_loss(
        model: WarehouseV6Actor, encoded: torch.Tensor, masks: torch.Tensor,
        public_teacher_actions: torch.Tensor, *, recipe: BCAlignmentRecipe,
        phase: str, sample_weights: torch.Tensor | None = None) -> BCAlignmentLoss:
    """Adapter for the unchanged additive Actor; no state or mode mutation.

    Call after selecting a BC minibatch, rather than on an entire dataset.
    No rollout, teacher, extractor, optimizer, checkpoint or authorization is
    reachable from this adapter. It deliberately rejects historical bounded
    Actor variants because their component semantics are different.
    """
    if phase != SHARED_BC_PHASE:
        raise ValueError("Warehouse alignment is restricted to shared BC, never PPO")
    from .warehouse_v6_actor import FEATURE_DIM, MODEL_VERSION

    if (encoded.ndim != 3 or tuple(encoded.shape[1:]) != (len(AGENTS), FEATURE_DIM)
            or tuple(masks.shape) != tuple(encoded.shape[:-1]) + (len(ACTIONS),)):
        raise ValueError("Warehouse shared BC adapter encoded shape mismatch")
    if any(model.actors[role].model_version != MODEL_VERSION for role in AGENTS):
        raise ValueError("Warehouse shared BC requires current additive Actor semantics")
    full, trees, primary = [], [], []
    for index, role in enumerate(AGENTS):
        actor = model.actors[role]
        branch, residual, _ = actor.components(encoded[:, index])
        raw_tree, _ = actor.tree(encoded[:, index])
        full.append(branch + residual)
        trees.append(raw_tree)
        primary.append(residual)
    return shared_bc_alignment_loss(
        torch.stack(full, 1), torch.stack(trees, 1), torch.stack(primary, 1),
        masks, public_teacher_actions, recipe=recipe, phase=phase,
        sample_weights=sample_weights)
