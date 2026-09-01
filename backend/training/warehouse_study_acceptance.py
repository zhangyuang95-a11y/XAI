"""Reproducible production acceptance for the 6x7 Human-AI study map.

This audit exercises the exact NumPy Actor loaded by the public Render
service.  Robot 2 is sampled from one frozen pre-move state before the
simulated participant command replaces Robot 1, and the pair is submitted in
one environment step.  The participant follows the stable teacher most of the
time and chooses a random legal action with calibrated probability, providing
real proximity, yielding, blocking, and collision examples without granting
the AI access to the current human command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.domain import WarehouseConfig, collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.interaction_calibration import calibrate_interactions
from env.warehouse.layouts import COMPACT_STAGGERED_8X9_LAYOUT
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse.numpy_policy import NumpyWarehousePolicy
from env.warehouse.runtime_coordination import (
    causal_participant_actions,
    guard_participant_action,
    select_ai_ai_joint_actions,
    select_human_ai_action,
)
from backend.adapters.warehouse_explanations import WarehouseExplanationMixin


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTOR = (
    ROOT / "output" / "deployment" / "warehouse_mappo_v68_6x7_actor.npz"
)


@dataclass(frozen=True)
class EpisodeAudit:
    seed: int
    deliveries: int
    proximity_frames: int
    potential_conflict_frames: int
    robot_collisions: int
    yield_events: int
    avoidable_waits: int
    avoidable_detours: int
    unexplained_reversals: int
    short_cycles: int
    invalid_goal_switches: int
    longest_ineffective_wait_streak: int
    long_standoff: bool
    deadlocked: bool
    shutdowns: int
    path_actual_steps: int
    path_shortest_safe_steps: int
    simultaneous_violations: int
    explanation_fact_failures: int
    ai_wait_actions: int
    ai_decisions: int
    charger_return_cycles: int
    runtime_selection_overrides: int
    noncausal_participant_requests: int
    dominated_joint_decisions: int
    coordination_plan_churn: int
    contradictory_joint_reason_frames: int
    explanation_internal_diagnostic_leaks: int
    future_state_causality_failures: int

    @property
    def path_efficiency(self) -> float:
        return self.path_actual_steps / max(1, self.path_shortest_safe_steps)


def _has_collision_opportunity(environment: WarehouseMultiAgentEnv) -> bool:
    state = environment.get_state()
    masks = environment.action_masks()
    legal = {
        agent_id: tuple(
            action
            for action, allowed in zip(ACTIONS, masks[agent_id])
            if allowed > 0.5
        )
        for agent_id in environment.agent_ids
    }
    return any(
        environment._resolve_motion(
            state,
            {"robot_1": human_action, "robot_2": ai_action},
        )[3]
        for human_action in legal["robot_1"]
        for ai_action in legal["robot_2"]
    )


def _simultaneous_violation(info: Mapping[str, Any]) -> bool:
    audit = info.get("decision_audit", {})
    trace = info.get("decision_trace", {})
    return bool(
        not audit.get("same_pre_move_state", False)
        or int(audit.get("environment_step_calls", 0)) != 1
        or not trace.get("same_frozen_state_for_all_agents", False)
        or not trace.get("pre_state_hash")
    )


def _pareto_dominating_joint_actions(
    runtime_decision: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return safe records that strictly dominate the selected joint result."""

    selected = runtime_decision.get("selected_joint_action")
    if not isinstance(selected, Mapping):
        return ()
    selected_actions = dict(selected.get("actions", {}))
    selected_breakdown = dict(selected.get("score_breakdown", {}))
    selected_distances = dict(selected.get("distances", {}))
    lower_is_better = (
        "committed_goal_regressions",
        "loaded_delivery_regressions",
        "low_energy_priority_regressions",
        "charger_access_blocking",
        "route_clearance_failures",
        "future_head_on_bottlenecks",
        "anticipated_charger_route_blocking",
        "charging_route_regressions",
        "noncharging_waits",
        "immediate_reversals",
        "short_cycles",
    )
    higher_is_better = (
        "loaded_delivery_progress",
        "low_energy_priority_progress",
        "charging_route_progress",
        "progressing_agents",
    )
    dominating: list[Mapping[str, Any]] = []
    for candidate in runtime_decision.get("safe_joint_actions", ()):
        if not isinstance(candidate, Mapping):
            continue
        if dict(candidate.get("actions", {})) == selected_actions:
            continue
        breakdown = dict(candidate.get("score_breakdown", {}))
        distances = dict(candidate.get("distances", {}))
        if any(
            int(breakdown.get(key, 0))
            > int(selected_breakdown.get(key, 0))
            for key in lower_is_better
        ):
            continue
        if any(
            int(breakdown.get(key, 0))
            < int(selected_breakdown.get(key, 0))
            for key in higher_is_better
        ):
            continue
        if any(
            int(distances.get(agent_id, {}).get("after", 10**6))
            > int(selected_distances.get(agent_id, {}).get("after", 10**6))
            for agent_id in selected_distances
        ):
            continue
        strict = bool(
            any(
                int(breakdown.get(key, 0))
                < int(selected_breakdown.get(key, 0))
                for key in lower_is_better
            )
            or any(
                int(breakdown.get(key, 0))
                > int(selected_breakdown.get(key, 0))
                for key in higher_is_better
            )
            or any(
                int(distances.get(agent_id, {}).get("after", 10**6))
                < int(selected_distances.get(agent_id, {}).get("after", 10**6))
                for agent_id in selected_distances
            )
        )
        if strict:
            dominating.append(candidate)
    return tuple(dominating)


def _plan_signature(plan: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    if not isinstance(plan, Mapping):
        return None
    return (
        str(plan.get("plan_kind", "")),
        str(plan.get("reason_code", "")),
        str(plan.get("priority_agent_id", "")),
        str(plan.get("yielding_agent_id", "")),
        str(plan.get("priority_goal_id", "")),
        tuple(plan.get("occupied_position", ())),
        tuple(plan.get("moving_target", ())),
    )


def _has_contradictory_joint_reasons(trace: Mapping[str, Any]) -> bool:
    reasons = tuple(
        str(decision.get("primary_reason_code", ""))
        for decision in dict(trace.get("agents", {})).values()
        if isinstance(decision, Mapping)
    )
    yielding = {
        "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE",
        "WAIT_FOR_CONFLICTING_TARGET",
        "WAIT_FOR_PRIORITY_PASSAGE",
    }
    return len(reasons) == 2 and all(reason in yielding for reason in reasons)


def _explanation_has_internal_diagnostic(text: str) -> bool:
    normalized = text.casefold()
    forbidden = (
        "decisiontrace",
        "decision trace",
        "reason_code",
        "reason code",
        "internal field",
        "日志缺失",
        "内部字段",
        "原因码",
    )
    return any(token in normalized for token in forbidden)


def _run_episode(
    actor: NumpyWarehousePolicy,
    config: WarehouseConfig,
    *,
    seed: int,
    participant_noise_probability: float,
    participant_controlled: bool,
) -> EpisodeAudit:
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=seed)
    initial = environment.get_state()
    initial.participant_controlled_agent_id = (
        config.human_agent_id if participant_controlled else None
    )
    environment.set_state(initial)
    rng = np.random.default_rng(seed + 999)

    proximity_frames = potential_conflict_frames = collisions = yields = 0
    avoidable_waits = avoidable_detours = reversals = short_cycles = 0
    invalid_switches = shutdowns = simultaneous_violations = fact_failures = 0
    charger_return_cycles = runtime_selection_overrides = 0
    noncausal_participant_requests = 0
    dominated_joint_decisions = coordination_plan_churn = 0
    contradictory_joint_reason_frames = 0
    explanation_internal_diagnostic_leaks = 0
    future_state_causality_failures = 0
    ineffective_wait_streak = 0
    longest_ineffective_wait_streak = 0
    ai_wait_actions = ai_decisions = 0
    previous_plan_id: str | None = None
    previous_plan_signature: tuple[Any, ...] | None = None
    previous_plan_execution_aligned: bool | None = None
    explainer = WarehouseExplanationMixin()
    audited_agent_ids = (
        ("robot_2",) if participant_controlled else environment.agent_ids
    )

    while True:
        before = environment.get_state()
        active_plan = before.active_coordination_plan
        active_signature = _plan_signature(active_plan)
        active_plan_id = (
            str(active_plan.get("plan_id"))
            if isinstance(active_plan, Mapping)
            else None
        )
        if (
            active_signature is not None
            and active_signature == previous_plan_signature
            and active_plan_id != previous_plan_id
            and (
                not participant_controlled
                or previous_plan_execution_aligned is not False
            )
        ):
            coordination_plan_churn += 1
        separation = shortest_path_distance(
            before.by_id("robot_1").position,
            before.by_id("robot_2").position,
            config.map_layout_id,
        )
        proximity_frames += int(separation <= 2)
        potential_conflict_frames += int(_has_collision_opportunity(environment))

        observations = environment.observations()
        policy_actions, distributions = actor.act(
            observations,
            deterministic=False,
            base_seed=seed + 13_000_000,
            decision_key=(before.episode_id, before.frame),
        )
        actions = dict(policy_actions)
        participant_action: str | None = None
        runtime_decision: dict[str, Any]
        if participant_controlled:
            participant_action = stable_coordination_actions(environment)["robot_1"]
            if rng.random() < participant_noise_probability:
                legal = list(causal_participant_actions(environment))
                participant_action = str(rng.choice(legal))
            ai_action, runtime_decision = select_human_ai_action(
                environment,
                policy_actions["robot_2"],
            )
            submitted_participant_action, participant_guard = guard_participant_action(
                environment,
                participant_action,
            )
            noncausal_participant_requests += int(participant_guard["blocked"])
            actions = {
                "robot_1": submitted_participant_action,
                "robot_2": ai_action,
            }
            runtime_decision = {
                **runtime_decision,
                "participant_action_guard": participant_guard,
                "selected_actions": dict(actions),
            }
        else:
            actions, runtime_decision = select_ai_ai_joint_actions(
                environment,
                policy_actions,
            )
            dominated_joint_decisions += int(
                bool(_pareto_dominating_joint_actions(runtime_decision))
            )
        if participant_controlled:
            future_state_causality_failures += int(
                runtime_decision.get(
                    "participant_action_known_at_decision_time"
                )
                is not False
                or not bool(runtime_decision.get("same_frozen_state", False))
            )
        runtime_selection_overrides += sum(
            str(policy_actions.get(agent_id, "WAIT")) != action
            for agent_id, action in actions.items()
            if not (participant_controlled and agent_id == "robot_1")
        )
        ai_wait_actions += sum(actions[agent_id] == "WAIT" for agent_id in audited_agent_ids)
        ai_decisions += len(audited_agent_ids)

        _, _, terminated, truncated, info = environment.step(
            actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source=(
                    "simulated_participant_plus_robust_numpy_actor"
                    if participant_controlled
                    else "numpy_actors_plus_joint_optimizer"
                ),
                participant_overrides=(
                    {"robot_1": participant_action}
                    if participant_action is not None
                    else None
                ),
                policy_actions=policy_actions,
                selected_actions=actions,
                runtime_decision=runtime_decision,
            ),
        )
        collisions += int(bool(info.get("robot_collision_event", False)))
        yields += sum(
            str(event.get("event", "")) == "coordination_yield"
            for event in info.get("coordination_events", ())
            if isinstance(event, Mapping)
        )
        avoidable_waits += sum(
            agent_id in info.get("avoidable_wait_agents", ())
            for agent_id in audited_agent_ids
        )
        avoidable_detours += sum(
            agent_id in info.get("avoidable_detour_agents", ())
            for agent_id in audited_agent_ids
        )
        reversals += sum(
            agent_id in info.get("unexplained_reversal_agents", ())
            for agent_id in audited_agent_ids
        )
        short_cycles += sum(
            agent_id in info.get("short_cycle_agents", ())
            for agent_id in audited_agent_ids
        )
        invalid_switches += sum(
            agent_id in info.get("invalid_goal_switch_agents", ())
            for agent_id in audited_agent_ids
        )
        shutdowns += sum(
            agent_id in info.get("shutdowns", ())
            for agent_id in audited_agent_ids
        )
        simultaneous_violations += int(_simultaneous_violation(info))
        fact_failures += len(
            info.get("decision_trace", {}).get("fact_validation_failures", ())
        )
        trace = info.get("decision_trace", {})
        if isinstance(trace, Mapping):
            contradictory_joint_reason_frames += int(
                _has_contradictory_joint_reasons(trace)
            )
            for agent_id in audited_agent_ids:
                for language in ("en", "zh-CN"):
                    explanation = explainer._decision_trace_explanation(
                        trace,
                        target_agent=agent_id,
                        focus="action",
                        language=language,
                    )
                    explanation_internal_diagnostic_leaks += int(
                        explanation is not None
                        and _explanation_has_internal_diagnostic(explanation)
                    )
        charger_return_cycles += sum(
            str(event.get("event", "")) == "charger_return_cycle"
            and str(event.get("agent_id", "")) in audited_agent_ids
            for event in info.get("energy_events", ())
            if isinstance(event, Mapping)
        )
        ineffective_joint_wait = bool(
            all(
                str(action) == "WAIT"
                for action in info.get("executed_actions", {}).values()
            )
            and not info.get("charger_used", False)
            and not info.get("task_changes", ())
        )
        ineffective_wait_streak = (
            ineffective_wait_streak + 1 if ineffective_joint_wait else 0
        )
        longest_ineffective_wait_streak = max(
            longest_ineffective_wait_streak,
            ineffective_wait_streak,
        )
        plan_event = next(
            (
                event
                for event in info.get("coordination_events", ())
                if isinstance(event, Mapping)
                and str(event.get("event", ""))
                == "joint_coordination_plan"
                and str(event.get("plan_id", "")) == str(active_plan_id or "")
            ),
            None,
        )
        previous_plan_id = active_plan_id
        previous_plan_signature = active_signature
        previous_plan_execution_aligned = (
            bool(plan_event.get("execution_aligned", False))
            if isinstance(plan_event, Mapping)
            else None
        )
        if terminated or truncated:
            break

    final = environment.get_state()
    completed_by_ai = tuple(
        task
        for task in final.completed_tasks
        if task.carrier_agent_id in audited_agent_ids
        and task.claimed_frame is not None
        and task.delivered_frame is not None
        and task.shortest_safe_delivery_steps is not None
    )
    return EpisodeAudit(
        seed=seed,
        deliveries=int(final.total_deliveries),
        proximity_frames=proximity_frames,
        potential_conflict_frames=potential_conflict_frames,
        robot_collisions=collisions,
        yield_events=yields,
        avoidable_waits=avoidable_waits,
        avoidable_detours=avoidable_detours,
        unexplained_reversals=reversals,
        short_cycles=short_cycles,
        invalid_goal_switches=invalid_switches,
        longest_ineffective_wait_streak=longest_ineffective_wait_streak,
        long_standoff=longest_ineffective_wait_streak >= 8,
        # A long standoff that later recovers is poor coordination, but it is
        # not a permanent deadlock.  Permanent deadlock is reserved for an
        # unrecovered ineffective joint-wait suffix at episode termination.
        deadlocked=ineffective_wait_streak >= 8,
        shutdowns=shutdowns,
        path_actual_steps=sum(
            int(task.delivered_frame) - int(task.claimed_frame)
            for task in completed_by_ai
        ),
        path_shortest_safe_steps=sum(
            int(task.shortest_safe_delivery_steps) for task in completed_by_ai
        ),
        simultaneous_violations=simultaneous_violations,
        explanation_fact_failures=fact_failures,
        ai_wait_actions=ai_wait_actions,
        ai_decisions=ai_decisions,
        charger_return_cycles=charger_return_cycles,
        runtime_selection_overrides=runtime_selection_overrides,
        noncausal_participant_requests=noncausal_participant_requests,
        dominated_joint_decisions=dominated_joint_decisions,
        coordination_plan_churn=coordination_plan_churn,
        contradictory_joint_reason_frames=contradictory_joint_reason_frames,
        explanation_internal_diagnostic_leaks=(
            explanation_internal_diagnostic_leaks
        ),
        future_state_causality_failures=future_state_causality_failures,
    )


def _summarize(episodes: Iterable[EpisodeAudit]) -> dict[str, Any]:
    rows = tuple(episodes)
    count = len(rows)
    if not rows:
        raise ValueError("At least one episode is required.")
    integer_fields = (
        "deliveries",
        "proximity_frames",
        "potential_conflict_frames",
        "robot_collisions",
        "yield_events",
        "avoidable_waits",
        "avoidable_detours",
        "unexplained_reversals",
        "short_cycles",
        "invalid_goal_switches",
        "shutdowns",
        "simultaneous_violations",
        "explanation_fact_failures",
        "ai_wait_actions",
        "ai_decisions",
        "charger_return_cycles",
        "runtime_selection_overrides",
        "noncausal_participant_requests",
        "dominated_joint_decisions",
        "coordination_plan_churn",
        "contradictory_joint_reason_frames",
        "explanation_internal_diagnostic_leaks",
        "future_state_causality_failures",
    )
    actual = sum(row.path_actual_steps for row in rows)
    shortest = sum(row.path_shortest_safe_steps for row in rows)
    return {
        "episodes": count,
        "mean": {
            field: mean(float(getattr(row, field)) for row in rows)
            for field in integer_fields
        },
        "total": {
            field: sum(int(getattr(row, field)) for row in rows)
            for field in integer_fields
        },
        "deadlock_episodes": sum(row.deadlocked for row in rows),
        "long_standoff_episodes": sum(row.long_standoff for row in rows),
        "maximum_ineffective_wait_streak": max(
            row.longest_ineffective_wait_streak for row in rows
        ),
        "path_efficiency_actual_over_shortest_safe": actual / max(1, shortest),
        "delivery_episode_rate": sum(row.deliveries > 0 for row in rows) / count,
        "collision_episode_rate": (
            sum(row.robot_collisions > 0 for row in rows) / count
        ),
        "ai_wait_action_rate": (
            sum(row.ai_wait_actions for row in rows)
            / max(1, sum(row.ai_decisions for row in rows))
        ),
        "seeds": [row.seed for row in rows],
        "per_episode": [asdict(row) | {"path_efficiency": row.path_efficiency} for row in rows],
    }


def _conflict_scripts(config: WarehouseConfig) -> dict[str, Any]:
    specifications = {
        "same_target": (
            (1, 1),
            (2, 2),
            {"robot_1": "RIGHT", "robot_2": "UP"},
        ),
        "swap": (
            (1, 2),
            (1, 3),
            {"robot_1": "RIGHT", "robot_2": "LEFT"},
        ),
        "occupied_stationary": (
            (1, 2),
            (1, 3),
            {"robot_1": "RIGHT", "robot_2": "WAIT"},
        ),
    }
    results: dict[str, Any] = {}
    for expected_kind, (human_position, ai_position, actions) in specifications.items():
        environment = WarehouseMultiAgentEnv(replace(config, horizon=1))
        environment.reset(seed=862)
        state = environment.get_state()
        state.by_id("robot_1").position = human_position
        state.by_id("robot_2").position = ai_position
        environment.set_state(state)
        *_, collision, collision_kind, _ = environment._resolve_motion(state, actions)
        _, _, _, _, info = environment.step(actions)
        results[expected_kind] = {
            "collision": bool(collision),
            "resolver_kind": collision_kind,
            "reported_kind": info.get("robot_collision_kind"),
            "same_pre_move_state": bool(
                info.get("decision_audit", {}).get("same_pre_move_state", False)
            ),
            "one_environment_step": (
                int(info.get("decision_audit", {}).get("environment_step_calls", 0))
                == 1
            ),
            "passed": bool(
                collision
                and collision_kind == expected_kind
                and info.get("robot_collision_kind") == expected_kind
                and info.get("decision_audit", {}).get("same_pre_move_state", False)
                and int(
                    info.get("decision_audit", {}).get("environment_step_calls", 0)
                )
                == 1
            ),
        }
    return results


def run_acceptance(
    actor_path: Path,
    *,
    participant_noise_probability: float = 0.35,
) -> dict[str, Any]:
    actor = NumpyWarehousePolicy.load(actor_path)
    config = collaborative_study_config()
    fixed_seeds = (40_786, *range(51_000, 51_249))
    rng = np.random.default_rng(20_260_831)
    random_seeds = tuple(
        int(value)
        for value in rng.choice(
            np.arange(1_000_000, 9_000_000), size=250, replace=False
        )
    )

    human_fixed = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=participant_noise_probability,
            participant_controlled=True,
        )
        for seed in fixed_seeds
    )
    human_random = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=participant_noise_probability,
            participant_controlled=True,
        )
        for seed in random_seeds
    )
    ai_ai_fixed = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=0.0,
            participant_controlled=False,
        )
        for seed in fixed_seeds
    )
    ai_ai_random = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=0.0,
            participant_controlled=False,
        )
        for seed in random_seeds
    )

    old_config = replace(
        config,
        rows=COMPACT_STAGGERED_8X9_LAYOUT.rows,
        cols=COMPACT_STAGGERED_8X9_LAYOUT.cols,
        map_layout_id=COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
    )
    production_calibration = calibrate_interactions(
        config,
        seeds=fixed_seeds,
        participant_noise_probability=participant_noise_probability,
    )
    old_calibration = calibrate_interactions(
        old_config,
        seeds=fixed_seeds,
        participant_noise_probability=participant_noise_probability,
    )
    conflict_scripts = _conflict_scripts(config)
    human_combined = _summarize((*human_fixed, *human_random))
    ai_ai_combined = _summarize((*ai_ai_fixed, *ai_ai_random))
    combined = _summarize(
        (*human_fixed, *human_random, *ai_ai_fixed, *ai_ai_random)
    )
    zero_trace_fields = (
        "avoidable_waits",
        "avoidable_detours",
        "unexplained_reversals",
        "short_cycles",
        "invalid_goal_switches",
        "simultaneous_violations",
        "explanation_fact_failures",
        "coordination_plan_churn",
        "contradictory_joint_reason_frames",
        "explanation_internal_diagnostic_leaks",
        "future_state_causality_failures",
    )
    gates = {
        "map_is_exactly_6x7": (config.rows, config.cols) == (6, 7),
        "map_has_no_four_way_cross": not bool(
            WarehouseMultiAgentEnv(config).layout.four_way_intersections
        ),
        "three_cell_exit": len(
            WarehouseMultiAgentEnv(config).layout.robot_exit_positions
        )
        == 3,
        "human_ai_mean_deliveries_not_below_v67_baseline": (
            human_combined["mean"]["deliveries"] >= 7.345
        ),
        "ai_ai_mean_deliveries_at_least_v67_human_ai_baseline": (
            ai_ai_combined["mean"]["deliveries"] >= 7.345
        ),
        "real_proximity_observed": combined["total"]["proximity_frames"] > 0,
        "real_potential_conflicts_observed": (
            combined["total"]["potential_conflict_frames"] > 0
        ),
        "real_yields_observed": combined["total"]["yield_events"] > 0,
        "ai_ai_no_robot_collision": (
            ai_ai_combined["total"]["robot_collisions"] == 0
        ),
        "human_ai_no_robot_collision": (
            human_combined["total"]["robot_collisions"] == 0
        ),
        "ai_ai_at_least_500_episodes": ai_ai_combined["episodes"] >= 500,
        "human_ai_at_least_500_episodes": human_combined["episodes"] >= 500,
        "no_unproductive_charger_return_cycles": (
            combined["total"]["charger_return_cycles"] == 0
        ),
        "all_ai_ai_runtime_choices_pareto_undominated": (
            ai_ai_combined["total"]["dominated_joint_decisions"] == 0
        ),
        "ai_ai_no_trace_regression": all(
            ai_ai_combined["total"][field] == 0
            for field in zero_trace_fields
        ),
        "human_ai_ai_has_no_trace_regression": all(
            human_combined["total"][field] == 0
            for field in zero_trace_fields
        ),
        "no_permanent_deadlock": (
            human_combined["deadlock_episodes"] == 0
            and ai_ai_combined["deadlock_episodes"] == 0
        ),
        "no_long_standoff": (
            human_combined["long_standoff_episodes"] == 0
            and ai_ai_combined["long_standoff_episodes"] == 0
        ),
        "ai_ai_wait_rate_below_half": (
            ai_ai_combined["ai_wait_action_rate"] < 0.5
        ),
        "six_by_seven_has_more_conflict_opportunities": (
            production_calibration["mean"]["collision_opportunity_frames"]
            > old_calibration["mean"]["collision_opportunity_frames"]
        ),
        "all_explicit_conflict_scripts_pass": all(
            item["passed"] for item in conflict_scripts.values()
        ),
    }
    return {
        "schema_version": "warehouse-study-acceptance.v4",
        "actor": {
            "path": str(actor_path.resolve()),
            "artifact_sha256": actor.artifact_sha256,
            "model_version": actor.metadata.model_version,
            "training_map_layout_id": actor.metadata.map_layout_id,
        },
        "production_map_layout_id": config.map_layout_id,
        "participant_noise_probability": participant_noise_probability,
        "human_ai_fixed_seed_audit": _summarize(human_fixed),
        "human_ai_random_seed_audit": _summarize(human_random),
        "human_ai_combined_audit": human_combined,
        "ai_ai_fixed_seed_audit": _summarize(ai_ai_fixed),
        "ai_ai_random_seed_audit": _summarize(ai_ai_random),
        "ai_ai_combined_audit": ai_ai_combined,
        "combined_audit": combined,
        "interaction_calibration": {
            "production_6x7": production_calibration,
            "archived_8x9": old_calibration,
        },
        "explicit_conflict_scripts": conflict_scripts,
        "acceptance_gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--participant-noise", type=float, default=0.35)
    args = parser.parse_args()
    report = run_acceptance(
        args.actor,
        participant_noise_probability=args.participant_noise,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
