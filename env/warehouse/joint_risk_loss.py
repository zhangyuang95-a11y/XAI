"""Paired same-state collision-risk objective for decentralized Actors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .navigation import ACTIONS


def trainable_joint_pairs(batch: Any) -> np.ndarray:
    """Return complete AI-AI frame indices and reject broken row ordering."""

    if len(batch.observations) % 2:
        raise ValueError("Joint Actor rows must contain complete pairs.")
    steps = np.arange(len(batch.observations) // 2)
    first = steps * 2
    second = first + 1
    if not (
        np.all(batch.agent_indices[first] == 0)
        and np.all(batch.agent_indices[second] == 1)
    ):
        raise ValueError(
            "Joint Actor rows must be ordered robot_1 then robot_2."
        )
    # A proxy participant is not sampled from the Actor distribution.  Those
    # rows belong to the separate profile-robust objective rather than this
    # AI-AI expectation.
    return steps[
        (batch.trainable_mask[first] > 0.5)
        & (batch.trainable_mask[second] > 0.5)
    ]


def expected_collision_loss(
    policy: Any,
    batch: Any,
    pairs: np.ndarray,
    *,
    selected_row_count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute ``p1.T @ C(S_t) @ p2`` from one frozen observation pair."""

    if len(pairs) == 0:
        return torch.zeros((), dtype=torch.float32, device=policy.device)
    sample_size = min(max(1, selected_row_count // 2), len(pairs))
    chosen = rng.choice(pairs, size=sample_size, replace=False)
    first_rows = chosen * 2
    second_rows = first_rows + 1
    observations = torch.as_tensor(
        np.concatenate(
            (batch.observations[first_rows], batch.observations[second_rows]),
            axis=0,
        ),
        dtype=torch.float32,
        device=policy.device,
    )
    logits = policy.masked_actor_logits(observations)
    first_probabilities, second_probabilities = torch.softmax(
        logits,
        dim=-1,
    ).split(sample_size)
    start = policy.network.joint_collision_matrix_start
    action_dim = len(ACTIONS)
    matrices = observations[
        :sample_size,
        start : start + action_dim**2,
    ].reshape(sample_size, action_dim, action_dim)
    return torch.einsum(
        "bi,bij,bj->b",
        first_probabilities,
        matrices,
        second_probabilities,
    ).mean()
