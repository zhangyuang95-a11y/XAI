"""Inactive Stage C safety API; this module is not a PPO driver.

The frozen capacity-experiment implementation is unchanged.  Future training
must explicitly bind this additional source before using it.  Regularized
feedback delegates to that implementation; raw feedback preserves its input
contract while avoiding an overflowing reduction before multiplication by zero.
No program supplies execution actions, and no training is authorized here.
"""
from __future__ import annotations

import torch

from . import warehouse_v622_actor_training as _frozen

ProgramLoss = _frozen.ProgramLoss
__all__ = ["ProgramLoss", "program_kl_loss"]


def program_kl_loss(current_logits, program_probabilities, masks, sampled_indices,
                    qualified, *, raw_program_top, variant, coefficient=.01):
    """Keep the frozen API, with a finite connected zero for the raw arm.

    Raw feedback never reads program targets, selected actions or qualification.
    NN and physical-mask validation remains identical to the frozen helper.
    """
    if variant != "raw":
        return _frozen.program_kl_loss(current_logits, program_probabilities,
            masks, sampled_indices, qualified, raw_program_top=raw_program_top,
            variant=variant, coefficient=coefficient)

    _frozen._finite_number(coefficient, "program coefficient")
    _frozen._physical_mask(current_logits, masks)
    if len(current_logits) == 0 or current_logits.ndim not in (2, 3):
        raise ValueError("program feedback requires physical rows, optionally two roles")
    if current_logits.ndim == 3 and current_logits.shape[1] != 2:
        raise ValueError("joint program feedback requires two roles")

    selected = torch.zeros(current_logits.shape[:-1], dtype=torch.bool,
                           device=current_logits.device)
    # Every finite value becomes zero before reduction.  Reducing finite
    # float32 logits first can overflow and turn the later '* 0' into NaN.
    role_losses = (current_logits[..., 0] * 0.).sum(0)
    return ProgramLoss(role_losses.sum(), role_losses, selected,
                       selected.sum(0), len(current_logits), 0.)
