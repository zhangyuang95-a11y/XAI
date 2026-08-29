"""Offline robust-risk calibration for Robot 2 across participant profiles.

Every profile distribution and the neural Actor distribution are computed from
the same frozen pre-move state.  The loss never receives the participant's
sampled current action; it minimizes worst-profile expected collision risk.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.partner_policies import (
    PARTNER_PROFILES,
    participant_surrogate_action,
    participant_surrogate_distribution,
    robust_partner_robot_two_action,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.policy import MAPPOPolicy


@dataclass(frozen=True)
class PartnerRiskConfig:
    rows: int = 30_000
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 5e-6
    worst_profile_collision_weight: float = 12.0
    ordinary_kl_weight: float = 1.0
    ai_mode_kl_weight: float = 2.0
    teacher_anchor_weight: float = 0.20
    avoidable_wait_weight: float = 1.0
    unsafe_action_probability_weight: float = 20.0
    unsafe_logit_margin_weight: float = 5.0
    unsafe_logit_margin: float = 10.0
    progress_logit_margin_weight: float = 2.0
    progress_logit_margin: float = 6.0
    participant_forecast_weight: float = 10.0
    optimization_temperature: float = 1.5
    participant_collision_log_scale_floor: float = 80.0
    energy_exhaustion_log_scale_floor: float = 200.0
    parameter_scope: str = "participant"
    failure_repeat: int = 4
    seed: int = 9_500_000


@dataclass(frozen=True)
class PartnerRiskDataset:
    robot_two_observations: np.ndarray
    ai_mode_robot_two_observations: np.ndarray
    collision_matrices: np.ndarray
    profile_probabilities: np.ndarray
    teacher_labels: np.ndarray
    collection: dict[str, Any]


def _robust_robot_two_label(
    environment: WarehouseMultiAgentEnv,
    collision_matrix: np.ndarray,
    profile_probabilities: np.ndarray,
    *,
    preferred_action_index: int,
) -> int:
    """Choose one S_t-only response robust to every participant profile."""

    del collision_matrix
    action = robust_partner_robot_two_action(
        environment,
        preferred_action=ACTIONS[int(preferred_action_index)],
        profile_probabilities=profile_probabilities,
    )
    return ACTIONS.index(action)


def _collision_matrix(policy: MAPPOPolicy, observation: np.ndarray) -> np.ndarray:
    start = policy.network.joint_collision_matrix_start
    size = len(ACTIONS)
    return np.asarray(
        observation[start : start + size**2],
        dtype=np.float32,
    ).reshape(size, size)


def _optimization_parameters(
    policy: MAPPOPolicy,
    parameter_scope: str,
) -> tuple[torch.nn.Parameter, ...]:
    """Resolve the exact trainable modules for offline risk calibration."""

    if parameter_scope == "participant":
        return tuple(
            list(policy.network.participant_partner_action_head.parameters())
            + list(policy.network.participant_context_predictor.parameters())
        )
    if parameter_scope == "structured":
        return tuple(
            list(policy.network.action_scorer.parameters())
            + list(policy.network.teammate_action_predictor.parameters())
        )
    if parameter_scope == "all":
        # ``all`` must include the S_t-only participant forecast that is
        # optimized by ``participant_forecast``. PPO intentionally uses the
        # narrower ``ppo_actor_parameters`` set, but this supervised pass
        # must not report a forecast objective whose module cannot update.
        return tuple(policy.network.actor_parameters())
    raise ValueError(
        "parameter_scope must be 'participant', 'structured', or 'all'."
    )


def collect_partner_risk_dataset(
    policy: MAPPOPolicy,
    config: PartnerRiskConfig,
) -> PartnerRiskDataset:
    rows: list[np.ndarray] = []
    ai_mode_rows: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    profile_rows: list[np.ndarray] = []
    labels: list[int] = []
    profile_episode_counts = {name: 0 for name in PARTNER_PROFILES}
    sampled_collisions = 0
    zero_risk_labels = 0
    zero_legal_action_risk_labels = 0
    precursor_rows = 0

    def append_snapshot(
        snapshot: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
        ],
        *,
        precursor: bool,
    ) -> None:
        nonlocal zero_risk_labels
        nonlocal zero_legal_action_risk_labels
        nonlocal precursor_rows
        if len(rows) >= int(config.rows):
            return
        (
            second,
            ai_mode_second,
            matrix,
            frozen_profiles,
            participant_legal,
            robust_label,
        ) = snapshot
        rows.append(second.copy())
        ai_mode_rows.append(ai_mode_second.copy())
        matrices.append(matrix.copy())
        profile_rows.append(frozen_profiles.copy())
        labels.append(int(robust_label))
        zero_risk_labels += int(
            float(
                np.max(
                    np.einsum(
                        "pi,i->p",
                        frozen_profiles,
                        matrix[:, robust_label],
                    )
                )
            )
            == 0.0
        )
        zero_legal_action_risk_labels += int(
            float(np.max(matrix[:, robust_label] * participant_legal)) == 0.0
        )
        precursor_rows += int(precursor)

    episode = 0
    while len(rows) < int(config.rows):
        seed = int(config.seed) + episode
        environment = WarehouseMultiAgentEnv(policy.environment_config)
        observations, _ = environment.reset(seed=seed)
        participant_state = environment.get_state()
        participant_state.participant_controlled_agent_id = (
            environment.config.human_agent_id
        )
        environment.set_state(participant_state)
        observations = environment.observations()
        profile = PARTNER_PROFILES[episode % len(PARTNER_PROFILES)]
        profile_episode_counts[profile] += 1
        profile_rng = np.random.default_rng(seed + 13_000_000)
        inference = policy.fork_for_inference(seed=seed + 17_000_000)
        repeated_collisions = 0
        precursor_history: deque[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                int,
            ]
        ] = deque(maxlen=8)
        last_precursor_mine_frame = -100
        while len(rows) < int(config.rows):
            state = environment.get_state()
            first = np.asarray(observations["robot_1"], dtype=np.float32).copy()
            second = np.asarray(observations["robot_2"], dtype=np.float32).copy()
            # Pair each participant-mode row with the identical physical
            # state in AI-AI mode.  Only the episode-known control-provenance
            # bit changes, so this anchor protects ordinary collaboration
            # without exposing either agent's current action.
            participant_state = deepcopy(state)
            ai_mode_state = deepcopy(state)
            ai_mode_state.participant_controlled_agent_id = None
            environment.set_state(ai_mode_state)
            ai_mode_second = np.asarray(
                environment.observations()["robot_2"], dtype=np.float32
            ).copy()
            environment.set_state(participant_state)
            observations = environment.observations()
            # Store every profile's distribution from the same S_t.  The
            # optimizer sees no sampled current participant action.
            coordinated_actions = stable_coordination_actions(environment)
            frozen_profiles = np.stack(
                [
                    participant_surrogate_distribution(
                        environment,
                        profile=name,
                        coordinated_actions=coordinated_actions,
                    )
                    for name in PARTNER_PROFILES
                ]
            ).astype(np.float32, copy=False)
            actions, distributions = inference.act(
                observations,
                environment.global_state(),
                deterministic=False,
                decision_key=(state.episode_id, state.frame),
            )
            preferred_action = int(
                np.argmax(distributions["robot_2"].probabilities)
            )
            robust_label = _robust_robot_two_label(
                environment,
                _collision_matrix(policy, first),
                frozen_profiles,
                preferred_action_index=preferred_action,
            )
            snapshot = (
                second,
                ai_mode_second,
                _collision_matrix(policy, first),
                frozen_profiles,
                np.asarray(
                    environment.action_masks()["robot_1"],
                    dtype=np.float32,
                ),
                robust_label,
            )
            append_snapshot(snapshot, precursor=False)
            precursor_history.append(snapshot)

            participant_action = participant_surrogate_action(
                environment,
                profile=profile,
                rng=profile_rng,
                coordinated_actions=coordinated_actions,
            )
            actions[environment.config.human_agent_id] = participant_action
            observations, _, terminated, truncated, info = environment.step(
                actions,
                decision_metadata=distribution_decision_metadata(
                    distributions,
                    decision_source="participant_profile_plus_pytorch_actor",
                    participant_overrides={
                        environment.config.human_agent_id: participant_action,
                    },
                ),
            )
            collided = bool(info.get("robot_collision_event", False))
            sampled_collisions += int(collided)
            repeated_collisions = repeated_collisions + 1 if collided else 0
            charger_cycle = any(
                event.get("event") == "charger_return_cycle"
                for event in info.get("energy_events", ())
                if isinstance(event, dict)
            )
            ineffective_joint_wait = bool(
                all(
                    action == "WAIT"
                    for action in info.get("executed_actions", {}).values()
                )
                and not info.get("charger_used", False)
                and not info.get("task_changes", ())
            )
            missed_robust_label = preferred_action != robust_label
            if (
                (missed_robust_label or charger_cycle or ineffective_joint_wait)
                and state.frame - last_precursor_mine_frame >= 4
            ):
                for _ in range(int(config.failure_repeat)):
                    for historical in precursor_history:
                        append_snapshot(historical, precursor=True)
                last_precursor_mine_frame = state.frame
            if terminated or truncated or repeated_collisions >= 3:
                break
        episode += 1
    return PartnerRiskDataset(
        robot_two_observations=np.stack(rows).astype(np.float32, copy=False),
        ai_mode_robot_two_observations=np.stack(ai_mode_rows).astype(
            np.float32, copy=False
        ),
        collision_matrices=np.stack(matrices).astype(np.float32, copy=False),
        profile_probabilities=np.stack(profile_rows).astype(np.float32, copy=False),
        teacher_labels=np.asarray(labels, dtype=np.int64),
        collection={
            "rows": len(rows),
            "episodes": episode,
            "profile_episode_counts": profile_episode_counts,
            "sampled_collision_steps": sampled_collisions,
            "failure_precursor_rows": precursor_rows,
            "failure_repeat": int(config.failure_repeat),
            "zero_worst_profile_risk_label_fraction": (
                zero_risk_labels / max(1, len(rows))
            ),
            "zero_any_legal_participant_risk_label_fraction": (
                zero_legal_action_risk_labels / max(1, len(rows))
            ),
            "decision_evidence": "all profile distributions and Robot 2 Actor use frozen S_t",
            "participant_current_action_used_by_loss_or_label": False,
            "robot_two_actions_rewritten": 0,
        },
    )


def fine_tune_partner_risk(
    policy: MAPPOPolicy,
    dataset: PartnerRiskDataset,
    config: PartnerRiskConfig,
) -> dict[str, Any]:
    observations = torch.as_tensor(
        dataset.robot_two_observations,
        dtype=torch.float32,
        device=policy.device,
    )
    ai_mode_observations = torch.as_tensor(
        dataset.ai_mode_robot_two_observations,
        dtype=torch.float32,
        device=policy.device,
    )
    matrices = torch.as_tensor(
        dataset.collision_matrices,
        dtype=torch.float32,
        device=policy.device,
    )
    profiles = torch.as_tensor(
        dataset.profile_probabilities,
        dtype=torch.float32,
        device=policy.device,
    )
    labels = torch.as_tensor(
        dataset.teacher_labels,
        dtype=torch.long,
        device=policy.device,
    )
    target_participant_distribution = profiles.mean(dim=1)
    # Hard margins use the most dangerous supported participant profile.  All
    # profile distributions and the collision matrix come from the same S_t;
    # no sampled current action enters the loss.
    worst_action_risk = torch.einsum(
        "bpi,bij->bpj",
        profiles,
        matrices,
    ).amax(dim=1)
    own_legal = observations[:, -len(ACTIONS) :] > 0.5
    enough_move_energy = observations[:, 2:3] > (
        float(policy.environment_config.move_battery_cost) / 100.0
    )
    energy_viable = torch.cat(
        (
            enough_move_energy.expand(-1, len(ACTIONS) - 1),
            torch.ones(
                (len(observations), 1),
                dtype=torch.bool,
                device=policy.device,
            ),
        ),
        dim=1,
    )
    executable_actions = own_legal & energy_viable
    effective_action_risk = worst_action_risk.masked_fill(
        ~executable_actions,
        2.0,
    )
    minimum_action_risk = effective_action_risk.amin(
        dim=-1,
        keepdim=True,
    )
    unsafe_actions = (
        executable_actions
        & (effective_action_risk > minimum_action_risk + 1e-8)
    ) | (own_legal & ~energy_viable)
    risk_opportunities = unsafe_actions.any(dim=-1)
    if bool(
        (
            effective_action_risk.gather(1, labels[:, None]).squeeze(1)
            > minimum_action_risk.squeeze(1) + 1e-8
        ).any()
    ):
        raise ValueError("Partner-risk labels must minimize legal-action risk.")

    def risk_metrics(
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> dict[str, float]:
        selected = logits.argmax(dim=-1)
        unsafe_selected = worst_action_risk.gather(
            1,
            selected[:, None],
        ).squeeze(1) > 1e-8
        unsafe_probability = (
            probabilities * unsafe_actions.to(probabilities.dtype)
        ).sum(dim=-1)
        opportunity_count = int(risk_opportunities.sum().item())
        return {
            "risk_opportunity_rows": float(opportunity_count),
            "risk_opportunity_fraction": float(
                risk_opportunities.to(torch.float32).mean().cpu()
            ),
            "risk_row_label_accuracy": float(
                (
                    logits.argmax(dim=-1)[risk_opportunities]
                    == labels[risk_opportunities]
                )
                .to(torch.float32)
                .mean()
                .cpu()
            )
            if opportunity_count
            else 1.0,
            "unsafe_argmax_fraction_on_risk_rows": float(
                unsafe_selected[risk_opportunities]
                .to(torch.float32)
                .mean()
                .cpu()
            )
            if opportunity_count
            else 0.0,
            "unsafe_probability_on_risk_rows": float(
                unsafe_probability[risk_opportunities].mean().cpu()
            )
            if opportunity_count
            else 0.0,
        }

    with torch.no_grad():
        base_logits = policy.masked_actor_logits(observations)
        base_probabilities = torch.softmax(base_logits, dim=-1)
        base_ai_mode_logits = policy.masked_actor_logits(ai_mode_observations)
        base_ai_mode_probabilities = torch.softmax(base_ai_mode_logits, dim=-1)
        accuracy_before = float(
            (base_logits.argmax(dim=-1) == labels).float().mean().cpu()
        )
        robust_label_probability_before = float(
            base_probabilities.gather(1, labels[:, None]).mean().cpu()
        )
        worst_profile_collision_before = float(
            torch.einsum(
                "bpi,bij,bj->bp",
                profiles,
                matrices,
                base_probabilities,
            )
            .max(dim=-1)
            .values
            .mean()
            .cpu()
        )
        before_risk_metrics = risk_metrics(base_logits, base_probabilities)
        participant_forecast_kl_before = float(
            F.kl_div(
                F.log_softmax(
                    policy.network.teammate_action_logits(observations),
                    dim=-1,
                ),
                target_participant_distribution,
                reduction="batchmean",
            ).cpu()
        )

    parameters = _optimization_parameters(policy, config.parameter_scope)
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    rng = np.random.default_rng(int(config.seed) + 1)
    wait_index = ACTIONS.index("WAIT")
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        order = rng.permutation(len(observations))
        totals = {
            "loss": 0.0,
            "worst_profile_expected_collision": 0.0,
            "ordinary_kl": 0.0,
            "ai_mode_kl": 0.0,
            "teacher_anchor": 0.0,
            "avoidable_wait": 0.0,
            "unsafe_action_probability": 0.0,
            "unsafe_logit_margin": 0.0,
            "progress_logit_margin": 0.0,
            "participant_forecast": 0.0,
        }
        batches = 0
        policy.network.train()
        for start in range(0, len(order), int(config.batch_size)):
            indices = torch.as_tensor(
                order[start : start + int(config.batch_size)],
                dtype=torch.long,
                device=policy.device,
            )
            logits = policy.masked_actor_logits(observations[indices])
            probabilities = torch.softmax(
                logits / float(config.optimization_temperature),
                dim=-1,
            )
            per_profile_collision = torch.einsum(
                "bpi,bij,bj->bp",
                profiles[indices],
                matrices[indices],
                probabilities,
            )
            batch_opportunities = risk_opportunities[indices]
            per_row_collision = per_profile_collision.max(dim=-1).values
            worst_collision = (
                per_row_collision[batch_opportunities].mean()
                if bool(batch_opportunities.any())
                else torch.zeros((), dtype=logits.dtype, device=policy.device)
            )
            ordinary_kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                base_probabilities[indices],
                reduction="batchmean",
            )
            ai_mode_logits = policy.masked_actor_logits(
                ai_mode_observations[indices]
            )
            ai_mode_kl = F.kl_div(
                F.log_softmax(ai_mode_logits, dim=-1),
                base_ai_mode_probabilities[indices],
                reduction="batchmean",
            )
            teacher_anchor = F.cross_entropy(logits, labels[indices])
            batch_unsafe = unsafe_actions[indices]
            unsafe_action_probability = (
                probabilities[batch_opportunities]
                * batch_unsafe[batch_opportunities].to(probabilities.dtype)
            ).sum(dim=-1).mean() if bool(batch_opportunities.any()) else torch.zeros(
                (), dtype=logits.dtype, device=policy.device
            )
            label_logits = logits.gather(
                1,
                labels[indices, None],
            ).squeeze(1)
            unsafe_logits = logits.masked_fill(
                ~batch_unsafe,
                torch.finfo(logits.dtype).min,
            ).amax(dim=-1)
            unsafe_logit_margin = (
                torch.relu(
                    unsafe_logits[batch_opportunities]
                    + float(config.unsafe_logit_margin)
                    - label_logits[batch_opportunities]
                ).mean()
                if bool(batch_opportunities.any())
                else torch.zeros((), dtype=logits.dtype, device=policy.device)
            )
            progress_rows = labels[indices] != wait_index
            avoidable_wait = (
                probabilities[progress_rows, wait_index].mean()
                if bool(progress_rows.any())
                else torch.zeros((), dtype=logits.dtype, device=policy.device)
            )
            progress_logit_margin = (
                torch.relu(
                    logits[progress_rows, wait_index]
                    + float(config.progress_logit_margin)
                    - label_logits[progress_rows]
                ).mean()
                if bool(progress_rows.any())
                else torch.zeros((), dtype=logits.dtype, device=policy.device)
            )
            participant_forecast = F.kl_div(
                F.log_softmax(
                    policy.network.teammate_action_logits(
                        observations[indices]
                    ),
                    dim=-1,
                ),
                target_participant_distribution[indices],
                reduction="batchmean",
            )
            loss = (
                float(config.worst_profile_collision_weight) * worst_collision
                + float(config.ordinary_kl_weight) * ordinary_kl
                + float(config.ai_mode_kl_weight) * ai_mode_kl
                + float(config.teacher_anchor_weight) * teacher_anchor
                + float(config.avoidable_wait_weight) * avoidable_wait
                + float(config.unsafe_action_probability_weight)
                * unsafe_action_probability
                + float(config.unsafe_logit_margin_weight)
                * unsafe_logit_margin
                + float(config.progress_logit_margin_weight)
                * progress_logit_margin
                + float(config.participant_forecast_weight)
                * participant_forecast
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            batches += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["worst_profile_expected_collision"] += float(
                worst_collision.detach().cpu()
            )
            totals["ordinary_kl"] += float(ordinary_kl.detach().cpu())
            totals["ai_mode_kl"] += float(ai_mode_kl.detach().cpu())
            totals["teacher_anchor"] += float(teacher_anchor.detach().cpu())
            totals["avoidable_wait"] += float(avoidable_wait.detach().cpu())
            totals["unsafe_action_probability"] += float(
                unsafe_action_probability.detach().cpu()
            )
            totals["unsafe_logit_margin"] += float(
                unsafe_logit_margin.detach().cpu()
            )
            totals["progress_logit_margin"] += float(
                progress_logit_margin.detach().cpu()
            )
            totals["participant_forecast"] += float(
                participant_forecast.detach().cpu()
            )
        row = {key: value / max(1, batches) for key, value in totals.items()}
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    policy.network.eval()
    with torch.no_grad():
        final_logits = policy.masked_actor_logits(observations)
        final_probabilities = torch.softmax(final_logits, dim=-1)
        accuracy_after = float(
            (final_logits.argmax(dim=-1) == labels)
            .float()
            .mean()
            .cpu()
        )
        robust_label_probability_after = float(
            final_probabilities.gather(1, labels[:, None]).mean().cpu()
        )
        worst_profile_collision_after = float(
            torch.einsum(
                "bpi,bij,bj->bp",
                profiles,
                matrices,
                final_probabilities,
            )
            .max(dim=-1)
            .values
            .mean()
            .cpu()
        )
        after_risk_metrics = risk_metrics(final_logits, final_probabilities)
        participant_forecast_kl_after = float(
            F.kl_div(
                F.log_softmax(
                    policy.network.teammate_action_logits(observations),
                    dim=-1,
                ),
                target_participant_distribution,
                reduction="batchmean",
            ).cpu()
        )
    return {
        "config": asdict(config),
        "collection": dict(dataset.collection),
        "history": history,
        "robust_label_accuracy_before": accuracy_before,
        "robust_label_accuracy_after": accuracy_after,
        "robust_label_probability_before": robust_label_probability_before,
        "robust_label_probability_after": robust_label_probability_after,
        "worst_profile_collision_before": worst_profile_collision_before,
        "worst_profile_collision_after": worst_profile_collision_after,
        "risk_metrics_before": before_risk_metrics,
        "risk_metrics_after": after_risk_metrics,
        "participant_forecast_kl_before": participant_forecast_kl_before,
        "participant_forecast_kl_after": participant_forecast_kl_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=9_500_000)
    parser.add_argument("--failure-repeat", type=int, default=4)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.20)
    parser.add_argument("--ai-mode-kl-weight", type=float, default=2.0)
    parser.add_argument("--ordinary-kl-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=12.0)
    parser.add_argument("--avoidable-wait-weight", type=float, default=1.0)
    parser.add_argument(
        "--unsafe-action-probability-weight",
        type=float,
        default=20.0,
    )
    parser.add_argument("--unsafe-logit-margin-weight", type=float, default=5.0)
    parser.add_argument("--unsafe-logit-margin", type=float, default=10.0)
    parser.add_argument(
        "--progress-logit-margin-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument("--progress-logit-margin", type=float, default=6.0)
    parser.add_argument(
        "--participant-forecast-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--parameter-scope",
        choices=("participant", "structured", "all"),
        default="participant",
    )
    args = parser.parse_args()

    policy = MAPPOPolicy.load(args.checkpoint, device=args.device)
    configuration = PartnerRiskConfig(
        rows=int(args.rows),
        epochs=int(args.epochs),
        seed=int(args.seed),
        teacher_anchor_weight=float(args.teacher_anchor_weight),
        ai_mode_kl_weight=float(args.ai_mode_kl_weight),
        ordinary_kl_weight=float(args.ordinary_kl_weight),
        worst_profile_collision_weight=float(args.collision_weight),
        avoidable_wait_weight=float(args.avoidable_wait_weight),
        unsafe_action_probability_weight=float(
            args.unsafe_action_probability_weight
        ),
        unsafe_logit_margin_weight=float(args.unsafe_logit_margin_weight),
        unsafe_logit_margin=float(args.unsafe_logit_margin),
        progress_logit_margin_weight=float(
            args.progress_logit_margin_weight
        ),
        progress_logit_margin=float(args.progress_logit_margin),
        participant_forecast_weight=float(args.participant_forecast_weight),
        learning_rate=float(args.learning_rate),
        parameter_scope=str(args.parameter_scope),
        failure_repeat=int(args.failure_repeat),
    )
    if configuration.failure_repeat < 0:
        parser.error("failure repeat cannot be negative")
    with torch.no_grad():
        policy.network.participant_collision_risk_log_scale.clamp_(
            min=float(configuration.participant_collision_log_scale_floor)
        )
        policy.network.energy_exhaustion_risk_log_scale.clamp_(
            min=float(configuration.energy_exhaustion_log_scale_floor)
        )
    dataset = collect_partner_risk_dataset(policy, configuration)
    report = fine_tune_partner_risk(policy, dataset, configuration)
    source_metadata = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    ).get("training_metadata", {})
    policy.save(
        args.output,
        training_metadata={
            "base_training": dict(source_metadata),
            "partner_risk_finetuning": report,
        },
    )
    report_path = Path(args.output).with_suffix(".partner_risk.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
