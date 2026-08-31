"""Balanced offline replay utilities for the warehouse neural Actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env.warehouse.mappo import MAPPOPolicy
from env.warehouse.navigation import ACTIONS


STRONG_ACTOR_CORRECTION_CATEGORIES = frozenset(
    {
        "collision",
        "junction_conflict",
        "loaded_detour",
        "charger_cycle",
        "task_starvation",
    }
)


def strong_actor_correction_mask(categories: np.ndarray) -> np.ndarray:
    """Rows whose offline label must beat every competing Actor action."""

    return np.isin(categories, tuple(STRONG_ACTOR_CORRECTION_CATEGORIES))


REPLAY_CATEGORIES = (
    "collision",
    "joint_wait",
    "loaded_detour",
    "charger_cycle",
    "task_starvation",
    "charger_queue",
    "critical_energy",
    "junction_conflict",
    "teammate_response",
    "energy",
    "head_on",
    "ordinary",
)
MAXIMUM_OVERSAMPLE_FACTOR = 4


@dataclass
class CategoryBalancedReplay:
    """Retain categories independently with bounded minority oversampling."""

    capacity: int
    rows: np.ndarray | None = None
    labels: np.ndarray | None = None
    categories: np.ndarray | None = None

    def append(
        self,
        rows: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
    ) -> None:
        if self.rows is None:
            self.rows = rows
            self.labels = labels
            self.categories = categories
        else:
            assert self.labels is not None
            if self.categories is None:
                self.categories = np.full(
                    len(self.rows), "ordinary", dtype="<U32"
                )
            self.rows = np.concatenate((self.rows, rows), axis=0)
            self.labels = np.concatenate((self.labels, labels), axis=0)
            self.categories = np.concatenate(
                (self.categories, categories), axis=0
            )
        self._cap()

    def _cap(self) -> None:
        assert self.rows is not None
        assert self.labels is not None
        assert self.categories is not None
        active_categories = [
            category
            for category in REPLAY_CATEGORIES
            if bool(np.any(self.categories == category))
        ]
        # Allocate capacity across categories that actually exist. Reserving
        # empty buckets silently discarded useful history whenever a new
        # failure category was added to the contract.
        per_category = max(
            1,
            self.capacity // max(1, len(active_categories)),
        )
        retained: list[np.ndarray] = []
        for category in active_categories:
            indices = np.flatnonzero(self.categories == category)
            if len(indices) > per_category:
                # Preserve representative old and new states. Keeping only
                # the newest rows caused the Actor to forget earlier map/task
                # geometries after every dataset-aggregation round.
                positions = np.linspace(
                    0,
                    len(indices) - 1,
                    num=per_category,
                    dtype=np.int64,
                )
                indices = indices[positions]
            retained.append(indices)
        selected = np.concatenate(retained)
        selected.sort()
        self.rows = self.rows[selected]
        self.labels = self.labels[selected]
        self.categories = self.categories[selected]

    def balanced(
        self,
        *,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.rows is not None
        assert self.labels is not None
        assert self.categories is not None
        nonempty = [
            category
            for category in REPLAY_CATEGORIES
            if np.any(self.categories == category)
        ]
        largest_bucket = max(
            int(np.sum(self.categories == category))
            for category in nonempty
        )
        target = max(
            1,
            min(
                self.capacity // max(1, len(nonempty)),
                largest_bucket,
            ),
        )
        rng = np.random.default_rng(seed + 73)
        groups: list[np.ndarray] = []
        for category in nonempty:
            indices = np.flatnonzero(self.categories == category)
            category_target = min(
                target,
                len(indices) * MAXIMUM_OVERSAMPLE_FACTOR,
            )
            groups.append(
                rng.choice(
                    indices,
                    size=category_target,
                    replace=len(indices) < category_target,
                )
            )
        selected = np.concatenate(groups)
        selected = selected[rng.permutation(len(selected))]
        return (
            self.rows[selected],
            self.labels[selected],
            self.categories[selected],
        )

    def category_counts(self) -> dict[str, int]:
        assert self.categories is not None
        return {
            category: int(np.sum(self.categories == category))
            for category in REPLAY_CATEGORIES
        }


def fit_actor_supervised(
    policy: MAPPOPolicy,
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    non_wait_margin: float,
    non_wait_weight: float,
    escape_wait_margin: float,
    escape_wait_weight: float,
    escape_wait_mask: np.ndarray,
    correction_margin: float,
    correction_weight: float,
    correction_mask: np.ndarray,
    wait_margin: float,
    wait_weight: float,
    wait_margin_mask: np.ndarray,
    seed: int,
    parameter_scope: str = "all",
    teammate_labels: np.ndarray | None = None,
    teammate_loss_weight: float = 0.0,
    action_loss_weight: float = 1.0,
) -> dict[str, float | str]:
    """Fit Actor weights to offline labels without executing those labels."""

    observations = torch.as_tensor(rows, dtype=torch.float32, device=policy.device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=policy.device)
    teammate_targets = (
        torch.as_tensor(
            teammate_labels,
            dtype=torch.long,
            device=policy.device,
        )
        if teammate_labels is not None
        else None
    )
    if teammate_targets is not None and len(teammate_targets) != len(rows):
        raise ValueError("teammate_labels must align with supervised rows.")
    if len(wait_margin_mask) != len(rows):
        raise ValueError("wait_margin_mask must align with supervised rows.")
    if len(escape_wait_mask) != len(rows):
        raise ValueError("escape_wait_mask must align with supervised rows.")
    if len(correction_mask) != len(rows):
        raise ValueError("correction_mask must align with supervised rows.")
    strong_escape_rows = torch.as_tensor(
        escape_wait_mask,
        dtype=torch.bool,
        device=policy.device,
    )
    strong_correction_rows = torch.as_tensor(
        correction_mask,
        dtype=torch.bool,
        device=policy.device,
    )
    strong_wait_rows = torch.as_tensor(
        wait_margin_mask,
        dtype=torch.bool,
        device=policy.device,
    )
    wait_index = ACTIONS.index("WAIT")
    teammate_loss_active = bool(
        teammate_labels is not None
        and teammate_loss_weight > 0.0
        and np.any(teammate_labels >= 0)
    )
    non_wait_margin_active = bool(
        non_wait_weight > 0.0 and np.any(labels != wait_index)
    )
    escape_wait_margin_active = bool(
        escape_wait_weight > 0.0
        and np.any(escape_wait_mask & (labels != wait_index))
    )
    correction_margin_active = bool(
        correction_weight > 0.0 and np.any(correction_mask)
    )
    wait_margin_active = bool(
        wait_weight > 0.0
        and np.any(wait_margin_mask & (labels == wait_index))
    )

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return a differentiable zero for an empty minibatch mask.

        The replay loop runs on MPS in production.  Calling ``bool(mask.any())``
        for every minibatch synchronizes the accelerator with the CPU and made
        each relabel round needlessly slow.  A clamped tensor denominator keeps
        the exact non-empty mean while an empty mask contributes zero without
        leaving the device.
        """

        weights = mask.to(dtype=values.dtype)
        selected = torch.where(mask, values, torch.zeros_like(values))
        return selected.sum() / weights.sum().clamp_min(1.0)

    def accuracy() -> float:
        with torch.no_grad():
            predicted = policy.masked_actor_logits(observations).argmax(dim=-1)
        return float((predicted == targets).float().mean().detach().cpu())

    accuracy_before = accuracy()

    def teammate_accuracy() -> float | None:
        if teammate_targets is None:
            return None
        valid = teammate_targets >= 0
        if not bool(valid.any()):
            return None
        with torch.no_grad():
            predicted = policy.network.teammate_action_logits(
                observations[valid]
            ).argmax(dim=-1)
        return float(
            (predicted == teammate_targets[valid]).float().mean().cpu()
        )

    teammate_accuracy_before = teammate_accuracy()
    # The decentralized Actor includes both the intent encoder and the action
    # head.  Optimising only ``network.actor`` leaves the intent representation
    # frozen during BC/relabel rounds even though PPO updates it later.  That
    # mismatch made every relabel pass chase a moving latent representation and
    # plateau around 89% accuracy.  Keep one stable parameter tuple for the
    # optimiser and gradient clipping so the complete runtime Actor is fitted.
    if parameter_scope == "all":
        actor_parameters = tuple(policy.network.actor_parameters())
    elif parameter_scope == "actor_without_teammate_predictor":
        actor_parameters = tuple(policy.network.ppo_actor_parameters())
    elif parameter_scope == "action_heads_only":
        actor_parameters = tuple(
            list(policy.network.actor.parameters())
            + list(policy.network.action_scorer.parameters())
        )
    elif parameter_scope == "teammate_predictor_only":
        actor_parameters = tuple(
            policy.network.teammate_action_predictor.parameters()
        )
    elif parameter_scope == "teammate_context_predictor_only":
        actor_parameters = tuple(
            policy.network.teammate_context_predictor.parameters()
        )
    elif parameter_scope == "teammate_forecast_only":
        actor_parameters = tuple(
            list(policy.network.teammate_action_predictor.parameters())
            + list(policy.network.teammate_context_predictor.parameters())
            + list(policy.network.participant_context_predictor.parameters())
        )
    elif parameter_scope == "participant_partner_only":
        actor_parameters = tuple(
            list(policy.network.participant_partner_action_head.parameters())
            + list(policy.network.participant_context_predictor.parameters())
        )
    elif parameter_scope == "deadlock_escape_only":
        actor_parameters = tuple(
            policy.network.deadlock_escape_action_head.parameters()
        )
    elif parameter_scope == "structured":
        actor_parameters = tuple(
            list(policy.network.action_scorer.parameters())
            + list(policy.network.teammate_action_predictor.parameters())
        )
    else:
        raise ValueError(
            "parameter_scope must be 'all', 'structured', or "
            "'actor_without_teammate_predictor', 'action_heads_only', or "
            "'teammate_predictor_only', 'teammate_context_predictor_only', "
            "'teammate_forecast_only', 'participant_partner_only', or "
            "'deadlock_escape_only'."
        )
    optimizer = torch.optim.Adam(
        actor_parameters,
        lr=float(learning_rate),
    )
    rng = np.random.default_rng(seed + 1)
    final_loss = 0.0
    for _ in range(int(epochs)):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), int(batch_size)):
            indices = torch.as_tensor(
                order[start : start + int(batch_size)],
                dtype=torch.long,
                device=policy.device,
            )
            logits = policy.masked_actor_logits(observations[indices])
            selected_targets = targets[indices]
            loss = float(action_loss_weight) * torch.nn.functional.cross_entropy(
                logits,
                selected_targets,
            )
            if teammate_loss_active:
                assert teammate_targets is not None
                selected_teammate_targets = teammate_targets[indices]
                valid_teammate = selected_teammate_targets >= 0
                teammate_logits = policy.network.teammate_action_logits(
                    observations[indices]
                )
                teammate_loss = masked_mean(
                    torch.nn.functional.cross_entropy(
                        teammate_logits,
                        selected_teammate_targets.clamp_min(0),
                        reduction="none",
                    ),
                    valid_teammate,
                )
                loss = loss + float(teammate_loss_weight) * teammate_loss
            non_wait = selected_targets != wait_index
            if non_wait_margin_active:
                row_indices = torch.arange(
                    len(selected_targets), device=policy.device
                )
                target_logits = logits[row_indices, selected_targets]
                wait_logits = logits[:, wait_index]
                non_wait_margin_loss = masked_mean(
                    torch.relu(
                        float(non_wait_margin)
                        - (target_logits - wait_logits)
                    ),
                    non_wait,
                )
                loss = loss + float(non_wait_weight) * non_wait_margin_loss
            escape_wait = non_wait & strong_escape_rows[indices]
            if escape_wait_margin_active:
                row_indices = torch.arange(
                    len(selected_targets), device=policy.device
                )
                target_logits = logits[row_indices, selected_targets]
                wait_logits = logits[:, wait_index]
                escape_wait_margin_loss = masked_mean(
                    torch.relu(
                        float(escape_wait_margin)
                        - (target_logits - wait_logits)
                    ),
                    escape_wait,
                )
                loss = (
                    loss
                    + float(escape_wait_weight) * escape_wait_margin_loss
                )
            correction = strong_correction_rows[indices]
            if correction_margin_active:
                correction_row_indices = torch.arange(
                    len(selected_targets), device=policy.device
                )
                correction_target_logits = logits[
                    correction_row_indices,
                    selected_targets,
                ]
                alternative_logits = logits.clone()
                alternative_logits[
                    correction_row_indices,
                    selected_targets,
                ] = -torch.inf
                strongest_alternative = alternative_logits.max(dim=-1).values
                correction_margin_loss = masked_mean(
                    torch.relu(
                        float(correction_margin)
                        - (correction_target_logits - strongest_alternative)
                    ),
                    correction,
                )
                loss = loss + float(correction_weight) * correction_margin_loss
            wait = (selected_targets == wait_index) & strong_wait_rows[indices]
            if wait_margin_active:
                selected_wait_logits = logits[:, wait_index]
                non_wait_logits = torch.cat(
                    (logits[:, :wait_index], logits[:, wait_index + 1 :]),
                    dim=-1,
                )
                strongest_non_wait_logits = non_wait_logits.max(dim=-1).values
                wait_margin_loss = masked_mean(
                    torch.relu(
                        float(wait_margin)
                        - (selected_wait_logits - strongest_non_wait_logits)
                    ),
                    wait,
                )
                loss = loss + float(wait_weight) * wait_margin_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "Non-finite supervised Actor loss; verify that every "
                    "target is allowed by its frozen S_t action mask."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor_parameters,
                policy.algorithm_config.max_grad_norm,
            )
            optimizer.step()
            final_loss = loss.detach()
    return {
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy(),
        "final_loss": float(final_loss.cpu()),
        "non_wait_margin": float(non_wait_margin),
        "non_wait_weight": float(non_wait_weight),
        "escape_wait_margin": float(escape_wait_margin),
        "escape_wait_weight": float(escape_wait_weight),
        "escape_wait_rows": float(np.sum(escape_wait_mask)),
        "correction_margin": float(correction_margin),
        "correction_weight": float(correction_weight),
        "correction_rows": float(np.sum(correction_mask)),
        "wait_margin": float(wait_margin),
        "wait_weight": float(wait_weight),
        "wait_margin_rows": float(np.sum(wait_margin_mask)),
        "parameter_scope": parameter_scope,
        "teammate_loss_weight": float(teammate_loss_weight),
        "action_loss_weight": float(action_loss_weight),
        "teammate_accuracy_before": teammate_accuracy_before,
        "teammate_accuracy_after": teammate_accuracy(),
    }


def supervised_category_accuracy(
    policy: MAPPOPolicy,
    rows: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
    """Measure each replay skill separately so rare failures cannot hide.

    An aggregate imitation score near one can still contain every premature
    charger departure or every old-task decision.  Reporting the buckets
    independently makes those deployment-critical errors visible to the
    training loop and to the saved audit summary.
    """

    if not (len(rows) == len(labels) == len(categories)):
        raise ValueError("supervised category inputs must have equal length")
    if not len(rows):
        return {}
    observations = torch.as_tensor(
        rows,
        dtype=torch.float32,
        device=policy.device,
    )
    with torch.no_grad():
        predictions = (
            policy.masked_actor_logits(observations)
            .argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
    return {
        category: float(np.mean(predictions[mask] == labels[mask]))
        for category in REPLAY_CATEGORIES
        if bool(np.any(mask := categories == category))
    }


def fit_teammate_forecast_supervised(
    policy: MAPPOPolicy,
    rows: np.ndarray,
    actor_labels: np.ndarray,
    teammate_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float | str]:
    """Fit only S_t-based peer forecasts to paired offline labels."""

    empty_mask = np.zeros(len(rows), dtype=bool)
    return fit_actor_supervised(
        policy,
        rows,
        actor_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=empty_mask,
        correction_margin=0.0,
        correction_weight=0.0,
        correction_mask=empty_mask,
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=empty_mask,
        seed=seed,
        parameter_scope="teammate_forecast_only",
        teammate_labels=teammate_labels,
        teammate_loss_weight=1.0,
        action_loss_weight=0.0,
    )
