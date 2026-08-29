"""Gradient arbitration for the optional post-hoc program objective."""

from __future__ import annotations

from typing import Sequence

import torch


def guard_program_gradients(
    task_gradients: Sequence[torch.Tensor | None],
    program_gradients: Sequence[torch.Tensor | None],
    *,
    regularization_weight: float,
    maximum_gradient_ratio: float,
    task_gradient_floor: float,
    project_conflicts: bool,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Project and cap an auxiliary gradient without reversing PPO progress."""

    if len(task_gradients) != len(program_gradients):
        raise ValueError("Task and program gradient lists must have equal length.")
    if regularization_weight < 0.0:
        raise ValueError("regularization_weight must be non-negative.")
    if maximum_gradient_ratio < 0.0:
        raise ValueError("maximum_gradient_ratio must be non-negative.")
    if task_gradient_floor < 0.0:
        raise ValueError("task_gradient_floor must be non-negative.")
    reference = next(
        (g for g in (*task_gradients, *program_gradients) if g is not None),
        None,
    )
    if reference is None:
        return [None for _ in task_gradients], {
            "task_gradient_norm": 0.0,
            "program_gradient_norm": 0.0,
            "applied_program_gradient_norm": 0.0,
            "applied_program_lambda": 0.0,
            "program_gradient_ratio": 0.0,
            "gradient_conflict": 0.0,
            "gradient_guard_saturated": 0.0,
        }
    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    task_squared_norm = zero
    task_program_dot = zero
    for task_gradient, program_gradient in zip(task_gradients, program_gradients):
        if task_gradient is not None:
            task_squared_norm = (
                task_squared_norm + task_gradient.detach().pow(2).sum()
            )
        if task_gradient is not None and program_gradient is not None:
            task_program_dot = task_program_dot + (
                task_gradient.detach() * program_gradient.detach()
            ).sum()
    epsilon = torch.finfo(reference.dtype).eps
    conflict = bool(task_program_dot.item() < 0.0)
    projection = (
        task_program_dot / (task_squared_norm + epsilon)
        if project_conflicts and conflict
        else zero
    )
    projected: list[torch.Tensor | None] = []
    projected_squared_norm = zero
    for task_gradient, program_gradient in zip(task_gradients, program_gradients):
        if program_gradient is None:
            projected.append(None)
            continue
        value = program_gradient.detach()
        if project_conflicts and conflict and task_gradient is not None:
            value = value - projection * task_gradient.detach()
        projected.append(value)
        projected_squared_norm = projected_squared_norm + value.pow(2).sum()
    task_norm = torch.sqrt(task_squared_norm)
    program_norm = torch.sqrt(projected_squared_norm)
    protected_task_norm = torch.maximum(
        task_norm,
        torch.as_tensor(task_gradient_floor, dtype=reference.dtype, device=reference.device),
    )
    requested = torch.as_tensor(
        regularization_weight, dtype=reference.dtype, device=reference.device
    )
    applied = (
        torch.minimum(
            requested,
            maximum_gradient_ratio * protected_task_norm / (program_norm + epsilon),
        )
        if program_norm.item() > 0.0
        else zero
    )
    combined: list[torch.Tensor | None] = []
    for task_gradient, program_gradient in zip(task_gradients, projected):
        if task_gradient is None and program_gradient is None:
            combined.append(None)
        elif task_gradient is None:
            combined.append(applied * program_gradient)
        elif program_gradient is None:
            combined.append(task_gradient.detach())
        else:
            combined.append(task_gradient.detach() + applied * program_gradient)
    applied_norm = applied * program_norm
    return combined, {
        "task_gradient_norm": float(task_norm.detach().cpu()),
        "program_gradient_norm": float(program_norm.detach().cpu()),
        "applied_program_gradient_norm": float(applied_norm.detach().cpu()),
        "applied_program_lambda": float(applied.detach().cpu()),
        "program_gradient_ratio": float(
            (applied_norm / torch.clamp(protected_task_norm, min=epsilon)).detach().cpu()
        ),
        "gradient_conflict": float(conflict),
        "gradient_guard_saturated": float(
            applied.item() + float(epsilon) < regularization_weight
        ),
    }
