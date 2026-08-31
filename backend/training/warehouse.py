"""Application-layer orchestration for warehouse MAPPO/RCPD training."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from backend.adapters.warehouse import WAREHOUSE_PROGRAM_VERSION, WarehouseAdapter
from backend.artifacts import CollaborativeArtifactPaths
from backend.simulation.trajectory_store import TrajectoryStore
from backend.training.learner_replay import (
    CategoryBalancedReplay,
    STRONG_ACTOR_CORRECTION_CATEGORIES,
    fit_actor_supervised,
    fit_teammate_forecast_supervised,
    strong_actor_correction_mask as _strong_actor_correction_mask,
    supervised_category_accuracy,
)
from backend.training.learner_dataset import (
    actor_supported_teacher_action as _actor_supported_teacher_action,
    collect_actor_collision_correction_dataset as _collect_actor_collision_correction_dataset,
    collect_actor_commitment_failure_dataset as _collect_actor_commitment_failure_dataset,
    collect_commitment_curriculum_dataset as _collect_commitment_curriculum_dataset,
    collect_critical_coordination_dataset as _collect_critical_coordination_dataset,
    collect_learner_state_relabel_dataset as _collect_learner_state_relabel_dataset,
    collect_loaded_detour_correction_dataset as _collect_loaded_detour_correction_dataset,
    configure_learner_state_head_on as _configure_learner_state_head_on,
    safe_navigation_teacher_actions as _safe_navigation_teacher_actions,
    teacher_mission_intent_label as _teacher_mission_intent_label,
)
from backend.training.warehouse_options import (
    add_teacher_balance_options,
    skill_retention_weight as _skill_retention_weight,
)
from backend.training.seed_ledger import training_seed_ledger
from core.rcpd import RCPD, RCPDConfig

from env.warehouse.environment import (
    ACTIONS,
    WarehouseConfig,
    WarehouseMultiAgentEnv,
)
from env.warehouse.domain import collaborative_study_config
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    ARTIFACT_NAMESPACE,
    RUNTIME_CONTROLLER,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.mappo import (
    MAPPOConfig,
    MAPPOPolicy,
    MAPPOTrainer,
    evaluate_head_on_yield_scenarios,
    evaluate_policy,
    evaluate_random_policy,
)
from env.warehouse.policy import independent_actor_input
from env.warehouse.partner_policies import PARTNER_PROFILES
from env.warehouse.policy_metrics import batch_efficiency_log_fields
from env.warehouse.observations import observation_dim
from env.warehouse.rewards import RewardConfig
from env.warehouse.seed_calibration import (
    calibrate_parallel_seed_pairs,
    save_parallel_seed_library,
)
from reference_trajectory import (
    TUTORIAL_SEED,
    calibrate_reference_trajectory,
    save_reference_trajectory_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = CollaborativeArtifactPaths.under(
    PROJECT_ROOT,
    ARTIFACT_NAMESPACE,
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACTS.root
DEFAULT_MODEL = DEFAULT_ARTIFACTS.model
DEFAULT_METRICS = DEFAULT_ARTIFACTS.metrics
DEFAULT_PLOT = DEFAULT_ARTIFACTS.training_plot
DEFAULT_SUMMARY = DEFAULT_ARTIFACTS.training_summary
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACTS.training_checkpoint
DEFAULT_PROGRAM = DEFAULT_ARTIFACTS.rcpd_program
DEFAULT_PROGRAM_PYTHON = DEFAULT_ARTIFACTS.rcpd_python
DEFAULT_TRAJECTORY = DEFAULT_ARTIFACTS.training_trajectory
DEFAULT_SEEDS = DEFAULT_ARTIFACTS.parallel_seed_pairs
DEFAULT_REFERENCE_TRAJECTORY = DEFAULT_ARTIFACTS.reference_trajectory

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the two-robot shared-task Warehouse MAPPO policy."
    )
    parser.add_argument("--episodes", type=int, default=2800)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--agents", type=int, default=None, help="Alias for --num-agents.")
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--intent-dim", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--actor-lr-final", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr-final", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef-start", type=float, default=0.005)
    parser.add_argument("--entropy-coef-final", type=float, default=0.001)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--joint-collision-loss-weight", type=float, default=0.25,
                        help="Weight for paired same-S_t p1^T C(S_t) p2.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--periodic-eval-every", type=int, default=200)
    parser.add_argument("--periodic-eval-episodes", type=int, default=20)
    parser.add_argument("--reward-overrides", default=None)
    parser.add_argument("--output", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS))
    parser.add_argument("--plot-output", default=str(DEFAULT_PLOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--checkpoint-output", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--rcpd-program-output", default=str(DEFAULT_PROGRAM))
    parser.add_argument("--rcpd-python-output", default=str(DEFAULT_PROGRAM_PYTHON))
    parser.add_argument("--trajectory-output", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--parallel-seed-output", default=str(DEFAULT_SEEDS))
    parser.add_argument("--parallel-seed-candidates", type=int, default=256)
    parser.add_argument(
        "--reference-trajectory-output",
        default=str(DEFAULT_REFERENCE_TRAJECTORY),
    )
    parser.add_argument("--reference-seed-candidates", type=int, default=2000)
    parser.add_argument("--skip-reference-calibration", action="store_true")
    parser.add_argument("--skip-seed-calibration", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--energy-curriculum-probability", type=float, default=0.40)
    parser.add_argument("--energy-curriculum-min-battery", type=float, default=15.0)
    parser.add_argument("--energy-curriculum-max-battery", type=float, default=35.0)
    parser.add_argument("--energy-curriculum-fade-start", type=float, default=0.60)
    parser.add_argument("--energy-curriculum-fade-end", type=float, default=0.90)
    parser.add_argument(
        "--coordination-curriculum-probability",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--coordination-curriculum-fade-start",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--coordination-curriculum-fade-end",
        type=float,
        default=0.95,
    )
    add_teacher_balance_options(parser)
    parser.add_argument(
        "--mission-intent-loss-coef",
        type=float,
        default=1.0,
        help=(
            "Auxiliary loss for the Actor's internal neural mission intent. "
            "It trains weights only and never replaces a runtime action."
        ),
    )
    parser.add_argument(
        "--resume-behavior-cloning-epochs",
        type=int,
        default=0,
        help=(
            "Optional offline actor rehearsal after loading a same-version "
            "checkpoint. Labels update network weights only; they are never "
            "submitted as runtime actions."
        ),
    )
    parser.add_argument("--learner-state-relabel-every", type=int, default=0)
    parser.add_argument("--learner-state-relabel-warmup-rounds", type=int, default=0)
    parser.add_argument("--learner-state-relabel-samples", type=int, default=0)
    parser.add_argument("--learner-state-relabel-replay-capacity", type=int, default=65536)
    parser.add_argument("--learner-state-relabel-epochs", type=int, default=50)
    parser.add_argument("--learner-state-relabel-lr", type=float, default=3e-4)
    parser.add_argument(
        "--learner-state-parameter-scope",
        choices=("structured", "all", "actor_without_teammate_predictor"),
        default="actor_without_teammate_predictor",
    )
    parser.add_argument("--learner-state-detour-samples", type=int, default=16)
    parser.add_argument("--learner-state-detour-search-episodes", type=int, default=80)
    parser.add_argument("--learner-state-collision-samples", type=int, default=32)
    parser.add_argument("--learner-state-collision-search-episodes", type=int, default=128)
    parser.add_argument("--learner-state-charger-cycle-samples", type=int, default=64)
    parser.add_argument("--learner-state-task-starvation-samples", type=int, default=64)
    parser.add_argument("--learner-state-commitment-search-episodes", type=int, default=64)
    parser.add_argument("--learner-state-commitment-curriculum-samples", type=int, default=2048)
    # Pure cross-entropy produced materially better exact Actor fidelity than
    # overlapping action margins. Keep the knobs for controlled ablations but
    # make the faithful objective the production default.
    parser.add_argument("--learner-state-non-wait-margin", type=float, default=0.0)
    parser.add_argument("--learner-state-non-wait-weight", type=float, default=0.0)
    parser.add_argument("--learner-state-escape-wait-margin", type=float, default=0.0)
    parser.add_argument("--learner-state-escape-wait-weight", type=float, default=0.0)
    parser.add_argument("--learner-state-correction-margin", type=float, default=0.0)
    parser.add_argument("--learner-state-correction-weight", type=float, default=0.0)
    parser.add_argument("--learner-state-wait-margin", type=float, default=0.0)
    parser.add_argument("--learner-state-wait-weight", type=float, default=0.0)
    parser.add_argument("--use-rcpd", action="store_true")
    parser.add_argument(
        "--rcpd-extract-every", "--extraction-interval",
        dest="rcpd_extract_every", type=int, default=500,
    )
    parser.add_argument("--lambda-complexity", type=float, default=0.001)
    # The final Actor is deliberately confident.  A mildly softened leaf
    # distribution plus a small action-boundary split objective gives the
    # post-hoc tree a faithful probability model without feeding it back into
    # training.  These values were selected on the declared post-hoc interval,
    # never on formal evaluation seeds.
    parser.add_argument("--rcpd-target-temperature", type=float, default=1.5)
    parser.add_argument("--rcpd-action-structure-weight", type=float, default=0.25)
    parser.add_argument("--rcpd-max-depth", type=int, default=10)
    parser.add_argument("--rcpd-max-leaf-nodes", type=int, default=96)
    parser.add_argument("--rcpd-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rcpd-replay-records", type=int, default=8192)
    parser.add_argument(
        "--minimum-counterfactual-pairs", "--rcpd-minimum-counterfactual-pairs",
        dest="rcpd_minimum_counterfactual_pairs", type=int, default=100,
    )
    return parser


def _reward_config_from_json(payload: str | None) -> RewardConfig:
    if payload is None:
        return RewardConfig()
    raw = payload.strip()
    if not raw:
        return RewardConfig()
    if raw.startswith("{"):
        values = json.loads(raw)
    else:
        values = {}
        for item in raw.split(","):
            name, separator, value = item.partition("=")
            if not separator:
                raise ValueError("Reward overrides must use name=value pairs.")
            values[name.strip()] = float(value)
    if not isinstance(values, Mapping):
        raise ValueError("Reward overrides must decode to an object.")
    return RewardConfig(**{str(key): float(value) for key, value in values.items()})


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    requested_agents = args.agents if args.agents is not None else args.num_agents
    if requested_agents != 2 or args.max_agents != 2:
        parser.error("the collaborative environment requires exactly two robots")
    if args.episodes <= 0 or args.horizon <= 0 or args.episodes_per_update <= 0:
        parser.error("episodes, horizon, and episodes-per-update must be positive")
    if args.periodic_eval_every < 0 or args.periodic_eval_episodes <= 0:
        parser.error("periodic evaluation interval must be non-negative and episodes positive")
    if args.joint_collision_loss_weight < 0.0:
        parser.error("--joint-collision-loss-weight must be non-negative")
    if args.reference_seed_candidates <= 0:
        parser.error("reference seed candidates must be positive")
    for name in ("actor_lr", "critic_lr"):
        value = float(getattr(args, name))
        final = getattr(args, f"{name}_final")
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
        if final is not None and (not np.isfinite(float(final)) or float(final) <= 0.0):
            parser.error(f"--{name.replace('_', '-')}-final must be positive")
    if args.rcpd_extract_every <= 0 or args.rcpd_replay_records < 32:
        parser.error("RCPD extraction interval must be positive and replay must hold >=32 rows")
    if args.lambda_complexity < 0.0:
        parser.error("--lambda-complexity must be non-negative")
    if not 0.0 <= args.energy_curriculum_probability <= 1.0:
        parser.error("--energy-curriculum-probability must be between zero and one")
    if not 0.0 < args.energy_curriculum_min_battery <= args.energy_curriculum_max_battery:
        parser.error("invalid energy curriculum battery range")
    if not 0.0 <= args.energy_curriculum_fade_start < args.energy_curriculum_fade_end <= 1.0:
        parser.error("energy curriculum fade fractions must satisfy 0 <= start < end <= 1")
    if not 0.0 <= args.coordination_curriculum_probability <= 1.0:
        parser.error(
            "--coordination-curriculum-probability must be between zero and one"
        )
    if not (
        0.0
        <= args.coordination_curriculum_fade_start
        < args.coordination_curriculum_fade_end
        <= 1.0
    ):
        parser.error(
            "coordination curriculum fade fractions must satisfy 0 <= start < end <= 1"
        )
    if args.behavior_cloning_samples < 0 or args.behavior_cloning_epochs < 0:
        parser.error("behavior cloning samples and epochs cannot be negative")
    if args.resume_behavior_cloning_epochs < 0:
        parser.error("resume behavior-cloning epochs cannot be negative")
    if args.behavior_cloning_batch_size <= 0 or args.behavior_cloning_lr <= 0.0:
        parser.error("behavior cloning batch size and learning rate must be positive")
    if args.mission_intent_loss_coef < 0.0:
        parser.error("mission-intent loss coefficient cannot be negative")
    if args.skill_retention_samples < 0:
        parser.error("skill retention samples cannot be negative")
    if args.skill_retention_weight < 0.0:
        parser.error("skill retention weight cannot be negative")
    if not 0.0 < args.skill_retention_fade_end <= 1.0:
        parser.error("skill retention fade end must be in (0, 1]")
    if (
        args.learner_state_relabel_every < 0
        or args.learner_state_relabel_warmup_rounds < 0
        or args.learner_state_relabel_samples < 0
        or args.learner_state_relabel_replay_capacity < 0
        or args.learner_state_relabel_epochs < 0
    ):
        parser.error(
            "learner-state relabel interval, samples, and epochs cannot be negative"
        )
    if (
        args.learner_state_detour_samples < 0
        or args.learner_state_detour_search_episodes < 0
        or args.learner_state_collision_samples < 0
        or args.learner_state_collision_search_episodes < 0
        or args.learner_state_charger_cycle_samples < 0
        or args.learner_state_task_starvation_samples < 0
        or args.learner_state_commitment_search_episodes < 0
        or args.learner_state_commitment_curriculum_samples < 0
    ):
        parser.error("learner-state failure-mining settings cannot be negative")
    if 0 < args.learner_state_relabel_samples < 4:
        parser.error("learner-state relabeling needs at least four samples")
    if (
        args.learner_state_relabel_samples > 0
        and args.learner_state_relabel_replay_capacity
        < args.learner_state_relabel_samples // 2
    ):
        parser.error(
            "learner-state replay capacity must hold at least one learner batch"
        )
    if args.learner_state_relabel_lr <= 0.0:
        parser.error("learner-state relabel learning rate must be positive")
    if (
        args.learner_state_non_wait_margin < 0.0
        or args.learner_state_non_wait_weight < 0.0
        or args.learner_state_escape_wait_margin < 0.0
        or args.learner_state_escape_wait_weight < 0.0
        or args.learner_state_correction_margin < 0.0
        or args.learner_state_correction_weight < 0.0
        or args.learner_state_wait_margin < 0.0
        or args.learner_state_wait_weight < 0.0
    ):
        parser.error("learner-state action-margin settings cannot be negative")
    if not 1 <= args.rcpd_max_depth <= 10:
        parser.error("--rcpd-max-depth must be in [1, 10]")
    if args.rcpd_max_leaf_nodes < 2 or args.rcpd_min_samples_leaf < 1:
        parser.error("RCPD leaf limits must be positive")
    if args.rcpd_action_structure_weight < 0.0:
        parser.error("RCPD action-structure weight cannot be negative")
    if args.parallel_seed_candidates < 8 and not args.skip_seed_calibration:
        parser.error("at least eight candidate seeds are needed for four disjoint pairs")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")


def _record_scenario_tags(features: Mapping[str, float]) -> tuple[str, ...]:
    tags: set[str] = set()
    for action in policy_actions():
        prefix = f"candidate.{action}"
        if features.get(f"{prefix}.blocked_by_robot", 0.0) > 0.5:
            tags.add("occupied_progress")
        if features.get(f"{prefix}.predicted_same_cell_conflict", 0.0) > 0.5:
            tags.add("same_cell_conflict")
        if features.get(f"{prefix}.predicted_swap_conflict", 0.0) > 0.5:
            tags.add("swap_conflict")
    if features.get("charger.occupied", 0.0) > 0.5:
        tags.add("charger_competition")
    return tuple(sorted(tags or {"ordinary"}))


def policy_actions() -> tuple[str, ...]:
    # Kept as a function so every RCPD record uses the policy checkpoint ABI.
    return ACTIONS


def _fit_rcpd(
    rcpd: RCPD,
    records: Sequence[dict[str, object]],
    adapter: WarehouseAdapter,
    *,
    step: int,
) -> bool:
    result = rcpd.maybe_extract(
        step,
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        force=True,
        action_legality_features=adapter.action_legality_features(),
        action_constraint_reason_features=adapter.action_constraint_reason_features(),
        required_predicate_groups=adapter.required_program_predicate_groups(),
        group_provider=lambda record: tuple(record.get("scenario_tags", ("ordinary",))),
        counterfactual_pair_provider=lambda record: (
            str(record["counterfactual_pair_id"])
            if record.get("counterfactual_pair_id")
            else None
        ),
        split_group_provider=lambda record: str(record.get("split_group", "train")),
        interaction_groups=(
            "occupied_progress",
            "same_cell_conflict",
            "swap_conflict",
            "charger_competition",
        ),
        program_metadata={
            "warehouse_program_version": WAREHOUSE_PROGRAM_VERSION,
            "environment": "two_robot_shared_delivery",
            "distilled_component": "neural_actor",
            "training_data_source": "executed_neural_rollout_only",
            "action_execution_version": ACTION_EXECUTION_VERSION,
            "runtime_controller": RUNTIME_CONTROLLER,
            "runtime_control_allowed": False,
            "feedback_allowed": False,
            "training_role": "posthoc_explanation_only",
            "program_roles": ("local_explanation_audit",),
            "regularization_version": False,
        },
    )
    return result is not None


def _collect_posthoc_rcpd_records(
    policy: MAPPOPolicy,
    adapter: WarehouseAdapter,
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    """Collect evidence only from final deterministic Actor execution.

    This collection happens after every Actor update and learner-state relabel
    pass has finished. Both robots are controlled by the final MAPPO Actor;
    no proxy-human replacement, expert action, shield, or program action is
    submitted during these trajectories.
    """

    records: list[dict[str, object]] = []
    for episode in range(max(1, int(episodes))):
        adapter.environment.reset(seed=seed + episode)
        rollout = adapter.rollout(
            policy,
            horizon=policy.environment_config.horizon,
            deterministic=True,
        )
        for frame in rollout.frames:
            for agent_id in adapter.environment.agent_ids:
                distribution = frame.distributions[agent_id]
                records.append(
                    {
                        "features": dict(
                            adapter.semantic_policy_features(
                                frame.snapshot,
                                agent_id,
                            )
                        ),
                        "observation": np.asarray(
                            frame.observations[agent_id],
                            dtype=np.float32,
                        ),
                        "probabilities": {
                            action: float(probability)
                            for action, probability in zip(
                                policy.action_names,
                                distribution.probabilities,
                            )
                        },
                        "advantage": 0.0,
                        "sampled_action": frame.proposed_actions[agent_id],
                        "scenario_tags": _record_scenario_tags(
                            adapter.semantic_policy_features(
                                frame.snapshot,
                                agent_id,
                            )
                        ),
                        "split_group": f"final_actor:{episode}",
                        "execution_source": "mappo_actor",
                    }
                )
    return records


def _linear_value(initial: float, final: float | None, progress: float) -> float:
    target = initial if final is None else float(final)
    return float(initial + (target - initial) * min(1.0, max(0.0, progress)))


def _next_interval_boundary(completed: int, interval: int) -> int | None:
    """Return the next due episode independently of rollout batch size."""

    if interval <= 0:
        return None
    return (int(completed) // int(interval) + 1) * int(interval)


def _energy_curriculum_probability(
    args: argparse.Namespace,
    episode: int,
) -> float:
    progress = episode / max(1, int(args.episodes))
    if progress <= args.energy_curriculum_fade_start:
        return float(args.energy_curriculum_probability)
    if progress >= args.energy_curriculum_fade_end:
        return 0.0
    fade_progress = (
        progress - args.energy_curriculum_fade_start
    ) / (
        args.energy_curriculum_fade_end - args.energy_curriculum_fade_start
    )
    return float(args.energy_curriculum_probability * (1.0 - fade_progress))


def _coordination_curriculum_probability(
    args: argparse.Namespace,
    episode: int,
) -> float:
    progress = episode / max(1, int(args.episodes))
    if progress <= args.coordination_curriculum_fade_start:
        return float(args.coordination_curriculum_probability)
    if progress >= args.coordination_curriculum_fade_end:
        return 0.0
    fade_progress = (
        progress - args.coordination_curriculum_fade_start
    ) / (
        args.coordination_curriculum_fade_end
        - args.coordination_curriculum_fade_start
    )
    return float(
        args.coordination_curriculum_probability * (1.0 - fade_progress)
    )


def _collect_behavior_cloning_dataset(
    environment_config: WarehouseConfig,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    environment = WarehouseMultiAgentEnv(environment_config)
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    targets: list[int] = []
    intent_targets: list[int] = []
    charge_samples = 0
    collision_recovery_samples = 0
    coordination_samples = 0
    # Forty percent of warm-start rows are stratified critical coordination
    # states.  Labels train weights only; they never replace a rollout action.
    critical_limit = max(4, 2 * int(sample_count) // 5)
    (
        critical_rows,
        critical_labels,
        critical_intent_labels,
        critical_charge_samples,
    ) = (
        _collect_critical_coordination_dataset(
            environment_config,
            sample_count=critical_limit,
            seed=seed + 8_000_000,
        )
    )
    observations.extend(critical_rows)
    targets.extend(int(value) for value in critical_labels)
    intent_targets.extend(int(value) for value in critical_intent_labels)
    coordination_samples += len(critical_rows)
    charge_samples += critical_charge_samples
    episode = 0
    while len(observations) < sample_count:
        environment.reset(seed=seed + episode)
        state = environment.get_state()
        if rng.random() < 0.50:
            selected = int(rng.integers(0, len(state.agents)))
            state.agents[selected].battery = float(rng.uniform(15.0, 55.0))
        environment.set_state(state)
        episode += 1
        while len(observations) < sample_count:
            local = environment.observations()
            state = environment.get_state()
            actions = _safe_navigation_teacher_actions(environment)
            for agent in state.agents:
                teacher_action = _actor_supported_teacher_action(
                    local[agent.agent_id],
                    actions[agent.agent_id],
                )
                actions[agent.agent_id] = teacher_action
                repetitions = (
                    3
                    if agent.navigation_goal_kind == "charge"
                    or state.last_robot_collision_event
                    else 1
                )
                for _ in range(repetitions):
                    if len(observations) >= sample_count:
                        break
                    observations.append(
                        independent_actor_input(local[agent.agent_id])
                    )
                    targets.append(ACTIONS.index(teacher_action))
                    intent_targets.append(
                        _teacher_mission_intent_label(
                            environment,
                            state,
                            agent.agent_id,
                        )
                    )
                    charge_samples += int(agent.navigation_goal_kind == "charge")
                    collision_recovery_samples += int(
                        state.last_robot_collision_event
                    )
            if rng.random() < 0.40 and len(observations) < sample_count:
                original_state = environment.get_state()
                masks = environment.action_masks()
                collision_actions = next(
                    (
                        {"robot_1": first, "robot_2": second}
                        for first, first_allowed in zip(ACTIONS, masks["robot_1"])
                        for second, second_allowed in zip(ACTIONS, masks["robot_2"])
                        if first_allowed > 0.5
                        and second_allowed > 0.5
                        and environment._resolve_motion(
                            original_state,
                            {"robot_1": first, "robot_2": second},
                        )[3]
                    ),
                    None,
                )
                if collision_actions is None:
                    environment.set_state(original_state)
                    collision_actions = {}
                else:
                    environment.step(collision_actions)
                recovery_state = environment.get_state()
                recovery_local = environment.observations()
                recovery_actions = _safe_navigation_teacher_actions(environment)
                for agent in environment.get_state().agents:
                    teacher_action = _actor_supported_teacher_action(
                        recovery_local[agent.agent_id],
                        recovery_actions[agent.agent_id],
                    )
                    recovery_actions[agent.agent_id] = teacher_action
                    for _ in range(8 if collision_actions else 0):
                        if len(observations) >= sample_count:
                            break
                        observations.append(
                            independent_actor_input(recovery_local[agent.agent_id])
                        )
                        targets.append(ACTIONS.index(teacher_action))
                        intent_targets.append(
                            _teacher_mission_intent_label(
                                environment,
                                recovery_state,
                                agent.agent_id,
                            )
                        )
                        charge_samples += int(
                            agent.navigation_goal_kind == "charge"
                        )
                        collision_recovery_samples += 1
                environment.set_state(original_state)
            executed_actions = actions
            if rng.random() < 0.15:
                executed_actions = {
                    agent_id: str(
                        rng.choice(
                            [
                                action
                                for action, allowed in zip(
                                    ACTIONS,
                                    environment.action_masks()[agent_id],
                                )
                                if allowed > 0.5
                            ]
                        )
                    )
                    for agent_id in environment.agent_ids
                }
            _, _, terminated, truncated, _ = environment.step(executed_actions)
            if terminated or truncated:
                break
    return (
        np.stack(observations),
        np.asarray(targets, dtype=np.int64),
        np.asarray(intent_targets, dtype=np.int64),
        charge_samples,
        collision_recovery_samples,
        coordination_samples,
    )


def _behavior_cloning_accuracy(
    policy: MAPPOPolicy,
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    with torch.no_grad():
        predictions = policy.masked_actor_logits(observations).argmax(dim=-1)
    return float((predictions == targets).float().mean().detach().cpu())


def _assert_supervised_action_support(
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    phase: str,
) -> None:
    """Fail closed when a label and its frozen-state mask disagree."""

    if not len(labels):
        return
    masks = observations[:, -len(ACTIONS) :]
    supported = masks[np.arange(len(labels)), labels] > 0.5
    if not bool(np.all(supported)):
        raise ValueError(
            f"{phase} contains {int(np.sum(~supported))} labels outside "
            "their same-state Actor action support."
        )


def _mission_intent_accuracy(
    policy: MAPPOPolicy,
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    with torch.no_grad():
        predictions = policy.network.mission_logits(observations).argmax(dim=-1)
    return float((predictions == targets).float().mean().detach().cpu())


def _pretrain_safe_navigation(
    policy: MAPPOPolicy,
    args: argparse.Namespace,
) -> dict[str, object]:
    if args.behavior_cloning_samples == 0 or args.behavior_cloning_epochs == 0:
        return {"enabled": False, "reason": "disabled_by_configuration"}
    (
        rows,
        labels,
        intent_labels,
        charge_samples,
        collision_recovery_samples,
        coordination_samples,
    ) = (
        _collect_behavior_cloning_dataset(
        policy.environment_config,
        sample_count=int(args.behavior_cloning_samples),
        seed=int(args.seed) + 700_000,
        )
    )
    _assert_supervised_action_support(
        rows,
        labels,
        phase="behavior cloning",
    )
    observations = torch.as_tensor(rows, dtype=torch.float32, device=policy.device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=policy.device)
    intent_targets = torch.as_tensor(
        intent_labels,
        dtype=torch.long,
        device=policy.device,
    )
    accuracy_before = _behavior_cloning_accuracy(policy, observations, targets)
    intent_accuracy_before = _mission_intent_accuracy(
        policy,
        observations,
        intent_targets,
    )
    optimizer = torch.optim.Adam(
        policy.network.actor_parameters(),
        lr=float(args.behavior_cloning_lr),
    )
    rng = np.random.default_rng(int(args.seed) + 710_000)
    final_loss = 0.0
    final_action_loss = 0.0
    final_mission_intent_loss = 0.0
    for _ in range(int(args.behavior_cloning_epochs)):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), int(args.behavior_cloning_batch_size)):
            indices = torch.as_tensor(
                order[start : start + int(args.behavior_cloning_batch_size)],
                dtype=torch.long,
                device=policy.device,
            )
            batch = observations[indices]
            action_logits, mission_logits = policy.network.actor_outputs(batch)
            local_dim = observation_dim(policy.environment_config)
            action_mask = (
                batch[..., local_dim - len(ACTIONS) : local_dim] > 0.5
            )
            masked_action_logits = action_logits.masked_fill(
                ~action_mask,
                torch.finfo(action_logits.dtype).min,
            )
            action_loss = torch.nn.functional.cross_entropy(
                masked_action_logits,
                targets[indices],
            )
            mission_intent_loss = torch.nn.functional.cross_entropy(
                mission_logits,
                intent_targets[indices],
            )
            loss = (
                action_loss
                + float(args.mission_intent_loss_coef) * mission_intent_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.network.actor_parameters(),
                policy.algorithm_config.max_grad_norm,
            )
            optimizer.step()
            final_loss = loss.detach()
            final_action_loss = action_loss.detach()
            final_mission_intent_loss = mission_intent_loss.detach()
    return {
        "enabled": True,
        "samples": len(rows),
        "charge_goal_samples": charge_samples,
        "collision_recovery_samples": collision_recovery_samples,
        "coordination_scenario_samples": coordination_samples,
        "epochs": int(args.behavior_cloning_epochs),
        "learning_rate": float(args.behavior_cloning_lr),
        "mission_intent_loss_coefficient": float(
            args.mission_intent_loss_coef
        ),
        "accuracy_before": accuracy_before,
        "accuracy_after": _behavior_cloning_accuracy(
            policy,
            observations,
            targets,
        ),
        "mission_intent_accuracy_before": intent_accuracy_before,
        "mission_intent_accuracy_after": _mission_intent_accuracy(
            policy,
            observations,
            intent_targets,
        ),
        "final_loss": float(final_loss.cpu()),
        "final_action_loss": float(final_action_loss.cpu()),
        "final_mission_intent_loss": float(final_mission_intent_loss.cpu()),
        "teacher": "collision_free_assigned_safe_navigation_v1",
    }


class _LearnerStateRelabeler:
    """Category-balanced Actor-state aggregation with no action intervention.

    Rare collision, joint-wait, energy, head-on, and teammate-response states
    are retained independently.  The balanced view is used only as an offline
    supervised objective; rollout and deployed actions always come from the
    MAPPO Actor.
    """

    def __init__(
        self,
        policy: MAPPOPolicy,
        environment_config: WarehouseConfig,
        args: argparse.Namespace,
    ) -> None:
        self.policy = policy
        self.environment_config = environment_config
        self.args = args
        self.replay = CategoryBalancedReplay(
            capacity=int(args.learner_state_relabel_replay_capacity)
        )
        self.retention_rows: np.ndarray | None = None
        self.retention_labels: np.ndarray | None = None

    def run_round(self, *, seed: int) -> dict[str, object]:
        if (
            self.args.learner_state_relabel_samples == 0
            or self.args.learner_state_relabel_epochs == 0
        ):
            return {"enabled": False, "reason": "disabled_by_configuration"}
        learner_sample_count = max(
            2,
            int(self.args.learner_state_relabel_samples) // 2,
        )
        rehearsal_sample_count = (
            int(self.args.learner_state_relabel_samples)
            - learner_sample_count
        )
        (learner_rows, learner_labels, learner_teammate_labels,
         learner_categories, coverage) = _collect_learner_state_relabel_dataset(
                self.policy,
                self.environment_config,
                sample_count=learner_sample_count,
                seed=seed,
                include_teammate_labels=True,
        )
        forecast_fit = fit_teammate_forecast_supervised(
            self.policy,
            learner_rows,
            learner_labels,
            learner_teammate_labels,
            epochs=int(self.args.learner_state_relabel_epochs),
            batch_size=int(self.args.behavior_cloning_batch_size),
            learning_rate=float(self.args.learner_state_relabel_lr),
            seed=seed + 1,
        )
        self.replay.append(
            learner_rows,
            learner_labels,
            learner_categories,
        )
        (
            detour_rows,
            detour_labels,
            detour_categories,
            detour_coverage,
        ) = _collect_loaded_detour_correction_dataset(
            self.policy,
            self.environment_config,
            sample_count=int(self.args.learner_state_detour_samples),
            maximum_episodes=int(
                self.args.learner_state_detour_search_episodes
            ),
            seed=seed + 30_000_000,
        )
        if len(detour_rows):
            self.replay.append(
                detour_rows,
                detour_labels,
                detour_categories,
            )
        (
            collision_rows,
            collision_labels,
            collision_categories,
            collision_coverage,
        ) = _collect_actor_collision_correction_dataset(
            self.policy,
            self.environment_config,
            sample_count=int(self.args.learner_state_collision_samples),
            maximum_episodes=int(
                self.args.learner_state_collision_search_episodes
            ),
            seed=seed + 40_000_000,
        )
        if len(collision_rows):
            self.replay.append(
                collision_rows,
                collision_labels,
                collision_categories,
            )
        (
            commitment_rows,
            commitment_labels,
            commitment_categories,
            commitment_coverage,
        ) = _collect_actor_commitment_failure_dataset(
            self.policy,
            self.environment_config,
            charger_cycle_samples=int(
                self.args.learner_state_charger_cycle_samples
            ),
            task_starvation_samples=int(
                self.args.learner_state_task_starvation_samples
            ),
            maximum_episodes=int(
                self.args.learner_state_commitment_search_episodes
            ),
            seed=seed + 50_000_000,
        )
        if len(commitment_rows):
            self.replay.append(
                commitment_rows,
                commitment_labels,
                commitment_categories,
            )
        (
            curriculum_rows,
            curriculum_labels,
            curriculum_categories,
            curriculum_coverage,
        ) = _collect_commitment_curriculum_dataset(
            self.environment_config,
            sample_count=int(
                self.args.learner_state_commitment_curriculum_samples
            ),
            seed=seed + 60_000_000,
        )
        if len(curriculum_rows):
            self.replay.append(
                curriculum_rows,
                curriculum_labels,
                curriculum_categories,
            )

        (
            rehearsal_rows,
            rehearsal_labels,
            _,
            charge_rows,
            recovery_rows,
            coordination_rows,
        ) = _collect_behavior_cloning_dataset(
            self.environment_config,
            sample_count=rehearsal_sample_count,
            seed=seed + 10_000,
        )
        balanced_rows, balanced_labels, balanced_categories = (
            self.replay.balanced(seed=seed)
        )
        rows = np.concatenate((balanced_rows, rehearsal_rows), axis=0)
        labels = np.concatenate((balanced_labels, rehearsal_labels), axis=0)
        _assert_supervised_action_support(
            rows,
            labels,
            phase="learner-state relabeling",
        )
        strong_wait_categories = {
            "collision",
            "charger_queue",
            "critical_energy",
            "junction_conflict",
            "head_on",
        }
        wait_margin_mask = np.concatenate(
            (
                np.isin(
                    balanced_categories,
                    tuple(strong_wait_categories),
                ),
                np.zeros(len(rehearsal_rows), dtype=bool),
            )
        )
        # Rows mined from a true neural WAIT/WAIT stall need a larger
        # separation from WAIT than ordinary navigation rows.  This remains
        # offline supervision only: the expert label is never submitted to
        # the environment or consulted by the runtime Actor.
        escape_wait_mask = np.concatenate(
            (
                balanced_categories == "joint_wait",
                np.zeros(len(rehearsal_rows), dtype=bool),
            )
        )
        # A collision label must dominate the Actor's current colliding
        # argmax, not merely win ordinary cross-entropy on average.  The
        # previous mask applied this strongest margin only to loaded detours,
        # which left rare same-target loops under-corrected even though those
        # states were present in replay.  This remains offline fitting; no
        # label is submitted to the environment.
        correction_mask = np.concatenate(
            (
                _strong_actor_correction_mask(balanced_categories),
                np.zeros(len(rehearsal_rows), dtype=bool),
            )
        )
        fit = fit_actor_supervised(
            self.policy,
            rows,
            labels,
            epochs=int(self.args.learner_state_relabel_epochs),
            batch_size=int(self.args.behavior_cloning_batch_size),
            learning_rate=float(self.args.learner_state_relabel_lr),
            non_wait_margin=float(self.args.learner_state_non_wait_margin),
            non_wait_weight=float(self.args.learner_state_non_wait_weight),
            escape_wait_margin=float(
                self.args.learner_state_escape_wait_margin
            ),
            escape_wait_weight=float(
                self.args.learner_state_escape_wait_weight
            ),
            escape_wait_mask=escape_wait_mask,
            correction_margin=float(
                self.args.learner_state_correction_margin
            ),
            correction_weight=float(
                self.args.learner_state_correction_weight
            ),
            correction_mask=correction_mask,
            wait_margin=float(self.args.learner_state_wait_margin),
            wait_weight=float(self.args.learner_state_wait_weight),
            wait_margin_mask=wait_margin_mask,
            seed=seed,
            parameter_scope=str(self.args.learner_state_parameter_scope),
        )
        category_accuracy = supervised_category_accuracy(
            self.policy,
            balanced_rows,
            balanced_labels,
            balanced_categories,
        )
        critical_category_accuracy = {
            category: accuracy
            for category, accuracy in category_accuracy.items()
            if category
            in {
                "charger_cycle",
                "task_starvation",
                "collision",
                "loaded_detour",
            }
        }
        self.retention_rows = rows
        self.retention_labels = labels
        return {
            "enabled": True,
            "execution_contract": "actor_actions_only",
            "expert_actions_submitted_to_environment": 0,
            "samples": int(len(rows)),
            "epochs": int(self.args.learner_state_relabel_epochs),
            "learning_rate": float(self.args.learner_state_relabel_lr),
            "parameter_scope": str(self.args.learner_state_parameter_scope),
            "learner_visited_rows": int(len(learner_rows)),
            "teammate_forecast_fit": forecast_fit,
            "targeted_detour_rows": int(len(detour_rows)),
            "targeted_collision_rows": int(len(collision_rows)),
            "targeted_commitment_rows": int(len(commitment_rows)),
            "commitment_curriculum_rows": int(len(curriculum_rows)),
            "learner_replay_rows": int(len(self.replay.rows)),
            "balanced_replay_rows": int(len(balanced_rows)),
            "rehearsal_rows": int(len(rehearsal_rows)),
            "charge_rehearsal_rows": int(charge_rows),
            "collision_rehearsal_rows": int(recovery_rows),
            "coordination_rehearsal_rows": int(coordination_rows),
            "replay_category_rows": self.replay.category_counts(),
            "replay_category_accuracy": category_accuracy,
            "minimum_critical_category_accuracy": (
                min(critical_category_accuracy.values())
                if critical_category_accuracy
                else None
            ),
            **fit,
            **coverage,
            **detour_coverage,
            **collision_coverage,
            **commitment_coverage,
            **curriculum_coverage,
        }


def _relabel_actor_visited_states(
    policy: MAPPOPolicy,
    environment_config: WarehouseConfig,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, object]:
    """Compatibility helper for one isolated Actor-state relabel round."""

    return _LearnerStateRelabeler(
        policy,
        environment_config,
        args,
    ).run_round(seed=seed)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _rcpd_from_args(args: argparse.Namespace) -> RCPD | None:
    if not args.use_rcpd:
        return None
    return RCPD(
        RCPDConfig(
            extraction_interval=int(args.rcpd_extract_every),
            minimum_extraction_samples=32,
            # RCPD is fitted only after MAPPO has finished.  Keeping both
            # feedback coefficients structurally zero prevents an ablation
            # preset or resumed command from ever feeding the tree back into
            # the Actor.
            regularization_lambda=0.0,
            maximum_regularization_lambda=0.0,
            regularization_start_fraction=0.80,
            regularization_ramp_fraction=0.05,
            feedback_target_mode="program_distribution",
            program_target_temperature=float(args.rcpd_target_temperature),
            max_depth=int(args.rcpd_max_depth),
            max_leaf_nodes=int(args.rcpd_max_leaf_nodes),
            min_samples_leaf=int(args.rcpd_min_samples_leaf),
            complexity_penalty=float(args.lambda_complexity),
            action_structure_weight=float(args.rcpd_action_structure_weight),
            require_action_agreement_for_feedback=True,
            # Natural-language answers are additionally guarded by the v5
            # per-question semantic validators and deterministic templates.
            # Keep the extracted tree compact and disclose its measured
            # fidelity instead of inflating depth until paths are unreadable.
            minimum_overall_fidelity_for_explanation=0.75,
            maximum_mean_kl_for_explanation=0.35,
            minimum_counterfactual_pairs=0,
            random_seed=int(args.seed),
        )
    )


def train(
    args: argparse.Namespace,
) -> tuple[MAPPOPolicy, list[dict[str, object]], dict[str, object], RCPD | None]:
    """Train MAPPO, then optionally distil its executed trajectories post hoc."""

    requested_agents = args.agents if args.agents is not None else args.num_agents
    environment_config = collaborative_study_config(
        num_agents=requested_agents,
        max_agents=args.max_agents,
        horizon=args.horizon,
        seed=args.seed,
        teacher_efficiency_guard_enabled=(not args.legacy_teacher_supervision),
        reward=_reward_config_from_json(args.reward_overrides),
    )
    # Do not even instantiate RCPD until the final Actor update has completed.
    # This makes the distillation direction structurally one-way: final neural
    # execution traces -> explanatory program.
    rcpd: RCPD | None = None
    relabel_history: list[dict[str, object]] = []
    restored_relabel_rows: np.ndarray | None = None
    restored_relabel_labels: np.ndarray | None = None
    restored_relabel_categories: np.ndarray | None = None
    if args.resume:
        trainer, completed_episodes, metrics = MAPPOTrainer.load_checkpoint(
            args.resume, device=args.device
        )
        policy = trainer.policy
        policy.algorithm_config = replace(
            policy.algorithm_config,
            actor_lr=float(args.actor_lr),
            critic_lr=float(args.critic_lr),
            update_epochs=int(args.update_epochs),
            minibatch_size=int(args.minibatch_size),
            entropy_coef=float(args.entropy_coef_start),
            entropy_coef_final=float(args.entropy_coef_final),
        )
        if policy.environment_config != environment_config:
            raise ValueError("Resume checkpoint does not match the requested environment.")
        payload = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        extra_state = payload.get("extra_state", {})
        if isinstance(extra_state, Mapping):
            restored_relabels = extra_state.get("learner_state_relabel_history", ())
            if isinstance(restored_relabels, list):
                relabel_history = [
                    dict(item)
                    for item in restored_relabels
                    if isinstance(item, Mapping)
                ]
            raw_relabel_rows = extra_state.get("learner_state_relabel_rows")
            raw_relabel_labels = extra_state.get("learner_state_relabel_labels")
            raw_relabel_categories = extra_state.get(
                "learner_state_relabel_categories"
            )
            if isinstance(raw_relabel_rows, np.ndarray) and isinstance(
                raw_relabel_labels,
                np.ndarray,
            ):
                restored_relabel_rows = raw_relabel_rows
                restored_relabel_labels = raw_relabel_labels
                if isinstance(raw_relabel_categories, np.ndarray):
                    restored_relabel_categories = raw_relabel_categories
        if args.resume_behavior_cloning_epochs:
            rehearsal_args = argparse.Namespace(**vars(args))
            rehearsal_args.behavior_cloning_epochs = int(
                args.resume_behavior_cloning_epochs
            )
            behavior_cloning_summary = _pretrain_safe_navigation(
                policy,
                rehearsal_args,
            )
            behavior_cloning_summary["phase"] = "resume_actor_rehearsal"
            behavior_cloning_summary["runtime_action_source"] = "mappo_actor"
        else:
            behavior_cloning_summary = {
                "enabled": False,
                "reason": "resume_preserves_existing_actor",
            }
    else:
        completed_episodes = 0
        metrics = []
        policy = MAPPOPolicy(
            environment_config,
            MAPPOConfig(
                hidden_dim=args.hidden_dim,
                intent_dim=args.intent_dim,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                entropy_coef=args.entropy_coef_start,
                entropy_coef_final=args.entropy_coef_final,
                seed=args.seed,
            ),
            device=args.device,
        )
        behavior_cloning_summary = _pretrain_safe_navigation(policy, args)
        trainer = MAPPOTrainer(policy)
    if completed_episodes >= args.episodes:
        raise ValueError("--episodes must exceed the completed checkpoint episode count.")

    relabeler = _LearnerStateRelabeler(
        policy,
        environment_config,
        args,
    )
    relabeler.replay.rows = restored_relabel_rows
    relabeler.replay.labels = restored_relabel_labels
    relabeler.replay.categories = restored_relabel_categories
    if not args.resume:
        for round_index in range(int(args.learner_state_relabel_warmup_rounds)):
            relabel_result = relabeler.run_round(
                seed=args.seed + 8_000_000 + round_index * 100_000,
            )
            relabel_result["phase"] = "pre_mappo_warmup"
            relabel_result["round"] = round_index + 1
            relabel_history.append(relabel_result)

    if (
        args.skill_retention_samples > 0
        and args.skill_retention_weight > 0.0
    ):
        if (
            relabeler.retention_rows is not None
            and relabeler.retention_labels is not None
        ):
            skill_anchor_observations = relabeler.retention_rows
            skill_anchor_labels = relabeler.retention_labels
        else:
            (
                skill_anchor_observations,
                skill_anchor_labels,
                _,
                _,
                _,
                _,
            ) = _collect_behavior_cloning_dataset(
                environment_config,
                sample_count=int(args.skill_retention_samples),
                seed=int(args.seed) + 8_500_000,
            )
    else:
        skill_anchor_observations = None
        skill_anchor_labels = None

    environment = WarehouseMultiAgentEnv(environment_config)
    adapter = WarehouseAdapter(environment)

    started = time.perf_counter()
    periodic_evaluations: list[dict[str, object]] = []
    next_periodic_evaluation = _next_interval_boundary(
        completed_episodes,
        int(args.periodic_eval_every),
    )
    next_learner_state_relabel = _next_interval_boundary(
        completed_episodes,
        int(args.learner_state_relabel_every),
    )
    next_episode = completed_episodes + 1
    while next_episode <= args.episodes:
        group_end = min(args.episodes, next_episode + args.episodes_per_update - 1)
        batches = [
            trainer.collect_episode(
                environment,
                seed=args.seed + episode,
                energy_curriculum_probability=_energy_curriculum_probability(
                    args,
                    episode,
                ),
                energy_curriculum_min_battery=args.energy_curriculum_min_battery,
                energy_curriculum_max_battery=args.energy_curriculum_max_battery,
                coordination_curriculum_probability=(
                    _coordination_curriculum_probability(args, episode)
                ),
            )
            for episode in range(next_episode, group_end + 1)
        ]
        # RCPD is a one-way, post-hoc description of neural execution traces.
        # It must never provide targets or gradients to the MAPPO actor.
        progress = group_end / max(1, args.episodes)
        actor_lr = _linear_value(args.actor_lr, args.actor_lr_final, progress)
        critic_lr = _linear_value(args.critic_lr, args.critic_lr_final, progress)
        entropy = _linear_value(
            args.entropy_coef_start, args.entropy_coef_final, progress
        )
        _set_optimizer_lr(trainer.actor_optimizer, actor_lr)
        _set_optimizer_lr(trainer.critic_optimizer, critic_lr)
        active_skill_retention_weight = _skill_retention_weight(
            args,
            group_end,
        )
        losses = trainer.update_many(
            batches,
            entropy_coef=entropy,
            skill_anchor_observations=skill_anchor_observations,
            skill_anchor_labels=skill_anchor_labels,
            skill_anchor_weight=active_skill_retention_weight,
            joint_collision_loss_weight=float(args.joint_collision_loss_weight),
        )
        relabel_applied = False
        if (
            next_learner_state_relabel is not None
            and group_end >= next_learner_state_relabel
        ):
            relabel_result = relabeler.run_round(
                seed=args.seed + 9_000_000 + group_end * 100,
            )
            relabel_result["episode"] = group_end
            relabel_history.append(relabel_result)
            relabel_applied = bool(relabel_result.get("enabled", False))
            if (
                relabeler.retention_rows is not None
                and relabeler.retention_labels is not None
            ):
                skill_anchor_observations = relabeler.retention_rows
                skill_anchor_labels = relabeler.retention_labels
            while next_learner_state_relabel <= group_end:
                next_learner_state_relabel += int(
                    args.learner_state_relabel_every
                )
        for episode, batch in zip(range(next_episode, group_end + 1), batches):
            metrics.append(
                {
                    "episode": episode,
                    "training_reward": batch.episode_reward,
                    "base_training_reward": batch.base_training_reward,
                    "potential_shaping_reward": batch.potential_shaping_reward,
                    "avoidable_wait_penalty_reward": (
                        batch.avoidable_wait_penalty_reward
                    ),
                    "mission_regression_penalty_reward": (
                        batch.mission_regression_penalty_reward
                    ),
                    **batch_efficiency_log_fields(batch),
                    "steps": batch.episode_steps,
                    "pickups": batch.pickups,
                    "deliveries": batch.deliveries,
                    "robot_collisions": batch.collisions,
                    "shutdowns": batch.shutdowns,
                    "charger_uses": batch.charger_uses,
                    "human_detour_units": batch.route_regret,
                    "minimum_battery": batch.minimum_battery,
                    "initial_minimum_battery": batch.initial_minimum_battery,
                    "energy_curriculum_applied": int(
                        batch.energy_curriculum_applied
                    ),
                    "coordination_curriculum_kind": (
                        batch.coordination_curriculum_kind or ""
                    ),
                    "proxy_human_overrides": batch.proxy_human_overrides,
                    "terminal_reason": batch.terminal_reason,
                    "actor_loss": losses["actor_loss"],
                    "critic_loss": losses["critic_loss"],
                    "entropy": losses["entropy"],
                    "joint_expected_collision_loss": losses["joint_expected_collision_loss"],
                    "joint_collision_pair_updates": losses["joint_collision_pair_updates"],
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "skill_retention_loss": losses[
                        "skill_anchor_loss"
                    ],
                    "skill_retention_accuracy": losses[
                        "skill_anchor_accuracy"
                    ],
                    "skill_retention_weight": losses[
                        "skill_anchor_weight"
                    ],
                    "configured_skill_retention_weight": float(
                        args.skill_retention_weight
                    ),
                    "skill_retention_fade_end": float(
                        args.skill_retention_fade_end
                    ),
                    "learner_state_relabel_applied": int(relabel_applied),
                    "rcpd_extracted": 0,
                    "rcpd_extraction_error": "",
                }
            )
        if args.log_every and (
            group_end == args.episodes or group_end % args.log_every == 0
        ):
            recent = batches[-1]
            print(
                f"episode={group_end} reward={recent.episode_reward:.3f} "
                f"deliveries={recent.deliveries} collisions={recent.collisions} "
                f"shutdowns={recent.shutdowns} proxy_overrides={recent.proxy_human_overrides}"
            )
        if args.checkpoint_every and (
            group_end % args.checkpoint_every == 0 or group_end == args.episodes
        ):
            trainer.save_checkpoint(
                args.checkpoint_output,
                episode=group_end,
                metrics=metrics,
                extra_state={
                    "rcpd": None,
                    "rcpd_replay": (),
                    "learner_state_relabel_history": relabel_history,
                    "learner_state_relabel_rows": relabeler.replay.rows,
                    "learner_state_relabel_labels": relabeler.replay.labels,
                    "learner_state_relabel_categories": (
                        relabeler.replay.categories
                    ),
                },
            )
        if (
            next_periodic_evaluation is not None
            and group_end >= next_periodic_evaluation
        ):
            evaluation_seed = args.seed + 2_000_000 + next_periodic_evaluation * 10
            periodic_evaluations.append(
                {
                    "episode": next_periodic_evaluation,
                    "ai_ai": evaluate_policy(
                        policy,
                        environment_config,
                        episodes=args.periodic_eval_episodes,
                        seed=evaluation_seed,
                    ),
                    "noisy_teammate_20_percent": evaluate_policy(
                        policy,
                        environment_config,
                        episodes=args.periodic_eval_episodes,
                        seed=evaluation_seed + 100_000,
                        noisy_teammate_probability=0.20,
                    ),
                    "random": evaluate_random_policy(
                        environment_config,
                        episodes=args.periodic_eval_episodes,
                        seed=evaluation_seed + 200_000,
                    ),
                    "standard_head_on": evaluate_head_on_yield_scenarios(
                        policy,
                        environment_config,
                        episodes=max(10, args.periodic_eval_episodes),
                        seed=evaluation_seed + 300_000,
                    ),
                }
            )
            next_periodic_evaluation += int(args.periodic_eval_every)
        next_episode = group_end + 1

    final_rcpd_extraction: dict[str, object] = {
        "enabled": bool(args.use_rcpd),
        "training_data_source": "final_executed_neural_rollout_only",
        "runtime_control_allowed": False,
        "records": 0,
        "extracted": False,
        "error": None,
    }
    if args.use_rcpd:
        rcpd = _rcpd_from_args(args)
        assert rcpd is not None
        extraction_episodes = max(
            40,
            int(
                np.ceil(
                    args.rcpd_replay_records
                    / max(1, 2 * environment_config.horizon)
                )
            )
            * 2,
        )
        replay = _collect_posthoc_rcpd_records(
            policy,
            adapter,
            episodes=extraction_episodes,
            seed=args.seed + 11_000_000,
        )
        replay = replay[-int(args.rcpd_replay_records) :]
        final_rcpd_extraction["records"] = len(replay)
        try:
            final_rcpd_extraction["extracted"] = _fit_rcpd(
                rcpd,
                replay,
                adapter,
                step=int(args.episodes),
            )
            final_rcpd_extraction["error"] = rcpd.last_error
        except (RuntimeError, ValueError) as exc:
            final_rcpd_extraction["error"] = str(exc)

    neural = evaluate_policy(
        policy, environment_config, episodes=args.eval_episodes, seed=args.seed + 100_000
    )
    noisy_teammate = evaluate_policy(
        policy,
        environment_config,
        episodes=args.eval_episodes,
        seed=args.seed + 150_000,
        noisy_teammate_probability=0.20,
    )
    random_baseline = evaluate_random_policy(
        environment_config, episodes=args.eval_episodes, seed=args.seed + 200_000
    )
    evaluation: dict[str, object] = {
        "behavior_cloning_warm_start": behavior_cloning_summary,
        "skill_retention": {
            "enabled": bool(
                skill_anchor_observations is not None
                and skill_anchor_labels is not None
                and args.skill_retention_weight > 0.0
            ),
            "samples": int(
                len(skill_anchor_observations)
                if skill_anchor_observations is not None
                else 0
            ),
            "initial_weight": float(args.skill_retention_weight),
            "fade_end_fraction": float(args.skill_retention_fade_end),
            "final_weight": _skill_retention_weight(args, args.episodes),
            "data_source": "offline_full_navigation_skill_retention_only",
            "runtime_action_source": "mappo_actor",
            "environment_action_interventions": 0,
        },
        "neural_policy": neural,
        "noisy_teammate_policy": noisy_teammate,
        "random_policy": random_baseline,
        "periodic_evaluations": periodic_evaluations,
        "learner_state_relabels": relabel_history,
        "final_posthoc_rcpd_extraction": final_rcpd_extraction,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return policy, metrics, evaluation, rcpd


def write_metrics(path: str | Path, metrics: Sequence[Mapping[str, object]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in metrics for key in row)) or ["episode"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    return target


def _program_regularization_summary(
    args: argparse.Namespace,
    evaluation: Mapping[str, object],
    rcpd: RCPD | None,
) -> dict[str, object]:
    distilled = rcpd.last_result.metrics if rcpd and rcpd.last_result else None
    neural = evaluation.get("neural_policy", {})
    reward = None
    if isinstance(neural, Mapping):
        reward = neural.get("mean_training_reward", neural.get("mean_reward"))

    def metric(name: str, default: object = None) -> object:
        return getattr(distilled, name, default) if distilled is not None else default

    return {
        "mode": "posthoc_extraction" if rcpd is not None else "disabled",
        "runtime_controller": RUNTIME_CONTROLLER,
        "lambda_extract": 0.0,
        "lambda_complexity": float(args.lambda_complexity),
        "complexity_lambda": float(args.lambda_complexity),
        "extraction_interval": int(args.rcpd_extract_every),
        "program_target_temperature": float(args.rcpd_target_temperature),
        "action_structure_weight": float(args.rcpd_action_structure_weight),
        "minimum_counterfactual_pairs": int(args.rcpd_minimum_counterfactual_pairs),
        "feedback_target": "none_posthoc_only",
        "reward": reward,
        "action_fidelity": metric("action_fidelity"),
        "fidelity": metric("action_fidelity"),
        "mean_KL": metric("mean_kl_divergence"),
        "program_depth": metric("program_depth"),
        "program_size": metric("program_size"),
        "program_leaf_count": metric("program_leaf_count"),
        "program_predicate_count": metric("program_predicate_count"),
        "explanation_eligible": bool(metric("explanation_eligible", False)),
    }


def write_plot(path: str | Path, metrics: Sequence[Mapping[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    episodes = [int(item["episode"]) for item in metrics]
    rewards = [float(item["training_reward"]) for item in metrics]
    deliveries = [float(item["deliveries"]) for item in metrics]
    window = min(100, max(1, len(metrics)))
    smooth = [mean(rewards[max(0, i - window + 1) : i + 1]) for i in range(len(rewards))]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.25, label="episode reward")
    axes[0].plot(episodes, smooth, linewidth=2, label=f"moving average ({window})")
    axes[0].set_ylabel("training reward")
    axes[0].legend()
    axes[1].plot(episodes, deliveries, color="#159947")
    axes[1].set_ylabel("shared deliveries")
    axes[1].set_xlabel("episode")
    figure.suptitle("Two-Robot Collaborative Delivery MAPPO")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def write_policy_trajectory(
    policy: MAPPOPolicy, path: str | Path, *, seed: int
) -> Path:
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=seed)
    adapter = WarehouseAdapter(environment)
    store = TrajectoryStore(
        metadata={
            "source": "current_two_robot_training",
            "seed": seed,
            "warehouse_program_version": WAREHOUSE_PROGRAM_VERSION,
        }
    )
    rollout = adapter.rollout(
        policy, horizon=policy.environment_config.horizon, deterministic=True
    )
    for frame in rollout.frames:
        store.append("ai_ai_demo", frame)
    return store.save(path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke_test:
        smoke_root = DEFAULT_OUTPUT_ROOT / "smoke"
        default_smoke_paths = {
            "output": (DEFAULT_MODEL, smoke_root / "warehouse_mappo.pt"),
            "metrics_output": (DEFAULT_METRICS, smoke_root / "training_metrics.csv"),
            "plot_output": (DEFAULT_PLOT, smoke_root / "training_progress.png"),
            "summary_output": (DEFAULT_SUMMARY, smoke_root / "training_summary.json"),
            "checkpoint_output": (
                DEFAULT_CHECKPOINT,
                smoke_root / "training_checkpoint.pt",
            ),
            "rcpd_program_output": (DEFAULT_PROGRAM, smoke_root / "rcpd_program.json"),
            "rcpd_python_output": (
                DEFAULT_PROGRAM_PYTHON,
                smoke_root / "rcpd_program.py",
            ),
            "trajectory_output": (
                DEFAULT_TRAJECTORY,
                smoke_root / "training_trajectory.pkl.gz",
            ),
            "parallel_seed_output": (
                DEFAULT_SEEDS,
                smoke_root / "parallel_seed_pairs.json",
            ),
            "reference_trajectory_output": (
                DEFAULT_REFERENCE_TRAJECTORY,
                smoke_root / "reference_trajectory.json",
            ),
        }
        for name, (formal_default, smoke_default) in default_smoke_paths.items():
            if Path(getattr(args, name)) == formal_default:
                setattr(args, name, str(smoke_default))
        args.episodes = min(args.episodes, 100)
        args.eval_episodes = min(args.eval_episodes, 10)
        args.periodic_eval_every = 0
        args.episodes_per_update = 1
        args.update_epochs = 1
        args.minibatch_size = min(args.minibatch_size, 64)
        args.checkpoint_every = 0
        args.no_plot = True
        args.skip_seed_calibration = True
        args.skip_reference_calibration = True
        args.behavior_cloning_samples = min(args.behavior_cloning_samples, 512)
        args.behavior_cloning_epochs = min(args.behavior_cloning_epochs, 1)
        args.skill_retention_samples = min(args.skill_retention_samples, 256)
        args.learner_state_relabel_warmup_rounds = min(
            args.learner_state_relabel_warmup_rounds,
            1,
        )
        args.learner_state_relabel_samples = min(
            args.learner_state_relabel_samples,
            256,
        )
        args.learner_state_detour_samples = min(
            args.learner_state_detour_samples,
            16,
        )
        args.learner_state_detour_search_episodes = min(
            args.learner_state_detour_search_episodes,
            4,
        )
        args.learner_state_collision_samples = min(
            args.learner_state_collision_samples,
            16,
        )
        args.learner_state_collision_search_episodes = min(
            args.learner_state_collision_search_episodes,
            4,
        )
        args.learner_state_charger_cycle_samples = min(
            args.learner_state_charger_cycle_samples,
            16,
        )
        args.learner_state_task_starvation_samples = min(
            args.learner_state_task_starvation_samples,
            16,
        )
        args.learner_state_commitment_search_episodes = min(
            args.learner_state_commitment_search_episodes,
            8,
        )
        args.learner_state_commitment_curriculum_samples = min(
            args.learner_state_commitment_curriculum_samples,
            128,
        )
        args.learner_state_relabel_epochs = min(
            args.learner_state_relabel_epochs,
            1,
        )
        if args.learner_state_relabel_every:
            args.learner_state_relabel_every = min(
                args.learner_state_relabel_every,
                50,
            )
    _validate_args(args, parser)
    policy, metrics, evaluation, rcpd = train(args)
    regularization = _program_regularization_summary(args, evaluation, rcpd)
    metrics_path = write_metrics(args.metrics_output, metrics)
    trajectory_path = write_policy_trajectory(
        policy, args.trajectory_output, seed=args.seed + 300_000
    )
    program_path: Path | None = None
    if rcpd is not None and rcpd.program is not None:
        program_path = rcpd.program.save_json(args.rcpd_program_output)
        rcpd.program.export_python(args.rcpd_python_output)
    seed_path: Path | None = None
    if not args.skip_seed_calibration:
        candidate_start = args.seed + 400_000
        pairs = calibrate_parallel_seed_pairs(
            policy,
            range(candidate_start, candidate_start + args.parallel_seed_candidates),
        )
        seed_path = save_parallel_seed_library(args.parallel_seed_output, pairs)
    reference_path: Path | None = None
    reference_calibration_error: str | None = None
    if not args.skip_reference_calibration:
        try:
            reference = calibrate_reference_trajectory(
                policy,
                maximum_candidates=args.reference_seed_candidates,
            )
            reference_path = save_reference_trajectory_manifest(
                args.reference_trajectory_output,
                reference,
                policy,
            )
        except RuntimeError as exc:
            reference_calibration_error = str(exc)
    rcpd_explanation_eligible = bool(
        regularization.get("explanation_eligible", False)
    )
    regularization["rcpd_explanation_eligible"] = rcpd_explanation_eligible
    regularization["reference_trajectory_eligible"] = reference_path is not None
    regularization["explanation_eligible"] = bool(
        rcpd_explanation_eligible and reference_path is not None
    )
    # Rewrite only the newly created candidate so its embedded formal-use
    # metadata reflects both RCPD and frozen-reference eligibility.
    model_path = policy.save(
        args.output,
        training_metadata={
            "environment": "two_robot_shared_delivery",
            "episodes": args.episodes,
            "proxy_human_episode_probability": 0.50,
            "proxy_human_override_probability": 1.0,
            "proxy_human_profiles": PARTNER_PROFILES,
            "joint_collision_loss_weight": float(args.joint_collision_loss_weight),
            "energy_curriculum": {
                "initial_probability": args.energy_curriculum_probability,
                "minimum_battery": args.energy_curriculum_min_battery,
                "maximum_battery": args.energy_curriculum_max_battery,
                "fade_start_fraction": args.energy_curriculum_fade_start,
                "fade_end_fraction": args.energy_curriculum_fade_end,
                "formal_evaluation_initial_battery": 100.0,
            },
            "coordination_curriculum": {
                "initial_probability": args.coordination_curriculum_probability,
                "fade_start_fraction": args.coordination_curriculum_fade_start,
                "fade_end_fraction": args.coordination_curriculum_fade_end,
                "scenarios": (
                    "head_on",
                    "charger_handoff",
                    "delivery_goal_clearance",
                    "empty_delivery_clearance",
                    "dual_charger_approach", "outer_exit_charger_approach",
                    "same_target_conflict",
                    "charger_commitment",
                    "task_commitment",
                ),
                "action_source": "mappo_actor",
            },
            "skill_retention": {
                "samples": int(args.skill_retention_samples),
                "weight": float(args.skill_retention_weight),
                "training_only": True,
                "runtime_action_source": "mappo_actor",
                "environment_action_interventions": 0,
            },
            "learner_state_relabeling": {
                "interval": int(args.learner_state_relabel_every),
                "warmup_rounds": int(
                    args.learner_state_relabel_warmup_rounds
                ),
                "samples": int(args.learner_state_relabel_samples),
                "replay_capacity": int(
                    args.learner_state_relabel_replay_capacity
                ),
                "epochs": int(args.learner_state_relabel_epochs),
                "learning_rate": float(args.learner_state_relabel_lr),
                "targeted_detour_samples": int(
                    args.learner_state_detour_samples
                ),
                "targeted_detour_search_episodes": int(
                    args.learner_state_detour_search_episodes
                ),
                "targeted_collision_samples": int(
                    args.learner_state_collision_samples
                ),
                "targeted_collision_search_episodes": int(
                    args.learner_state_collision_search_episodes
                ),
                "targeted_charger_cycle_samples": int(
                    args.learner_state_charger_cycle_samples
                ),
                "targeted_task_starvation_samples": int(
                    args.learner_state_task_starvation_samples
                ),
                "targeted_commitment_search_episodes": int(
                    args.learner_state_commitment_search_episodes
                ),
                "commitment_curriculum_samples": int(
                    args.learner_state_commitment_curriculum_samples
                ),
                "non_wait_margin": float(args.learner_state_non_wait_margin),
                "non_wait_weight": float(args.learner_state_non_wait_weight),
                "escape_wait_margin": float(
                    args.learner_state_escape_wait_margin
                ),
                "escape_wait_weight": float(
                    args.learner_state_escape_wait_weight
                ),
                "correction_margin": float(
                    args.learner_state_correction_margin
                ),
                "correction_weight": float(
                    args.learner_state_correction_weight
                ),
                "wait_margin": float(args.learner_state_wait_margin),
                "wait_weight": float(args.learner_state_wait_weight),
                "rollout_action_source": "mappo_actor",
                "expert_actions_submitted_to_environment": 0,
            },
            "program_regularization": regularization,
        },
    )
    if not args.no_plot:
        write_plot(args.plot_output, metrics)
    summary = {
        "format": "warehouse_collaborative_training_v28_causal_clearance",
        "model_version": policy.model_version,
        "warehouse_program_version": WAREHOUSE_PROGRAM_VERSION,
        "environment_config": asdict(policy.environment_config),
        "algorithm_config": {**asdict(policy.algorithm_config), "joint_collision_loss_weight": float(args.joint_collision_loss_weight)},
        "seed_ledger": training_seed_ledger(args),
        "evaluation": evaluation,
        "program_regularization": regularization,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "trajectory": str(trajectory_path),
            "program": str(program_path) if program_path else None,
            "parallel_seed_pairs": str(seed_path) if seed_path else None,
            "reference_trajectory": str(reference_path) if reference_path else None,
        },
        "reference_trajectory": {
            "eligible": reference_path is not None,
            "calibration_error": reference_calibration_error,
        },
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
