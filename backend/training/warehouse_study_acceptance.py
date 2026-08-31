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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTOR = (
    ROOT / "output" / "deployment" / "warehouse_mappo_v67_6x7_actor.npz"
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


def _run_episode(
    actor: NumpyWarehousePolicy,
    config: WarehouseConfig,
    *,
    seed: int,
    participant_noise_probability: float,
) -> EpisodeAudit:
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=seed)
    initial = environment.get_state()
    initial.participant_controlled_agent_id = config.human_agent_id
    environment.set_state(initial)
    rng = np.random.default_rng(seed + 999)

    proximity_frames = potential_conflict_frames = collisions = yields = 0
    avoidable_waits = avoidable_detours = reversals = short_cycles = 0
    invalid_switches = shutdowns = simultaneous_violations = fact_failures = 0
    ineffective_wait_streak = 0
    longest_ineffective_wait_streak = 0

    while True:
        before = environment.get_state()
        separation = shortest_path_distance(
            before.by_id("robot_1").position,
            before.by_id("robot_2").position,
            config.map_layout_id,
        )
        proximity_frames += int(separation <= 2)
        potential_conflict_frames += int(_has_collision_opportunity(environment))

        observations = environment.observations()
        actions, distributions = actor.act(
            observations,
            deterministic=False,
            base_seed=seed + 13_000_000,
            decision_key=(before.episode_id, before.frame),
        )
        participant_action = stable_coordination_actions(environment)["robot_1"]
        if rng.random() < participant_noise_probability:
            mask = environment.action_masks()["robot_1"]
            legal = [
                action
                for action, allowed in zip(ACTIONS, mask)
                if allowed > 0.5
            ]
            participant_action = str(rng.choice(legal))
        actions["robot_1"] = participant_action

        _, _, terminated, truncated, info = environment.step(
            actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source="simulated_participant_plus_numpy_actor",
                participant_overrides={"robot_1": participant_action},
            ),
        )
        collisions += int(bool(info.get("robot_collision_event", False)))
        yields += sum(
            str(event.get("event", "")) == "coordination_yield"
            for event in info.get("coordination_events", ())
            if isinstance(event, Mapping)
        )
        avoidable_waits += int("robot_2" in info.get("avoidable_wait_agents", ()))
        avoidable_detours += int(
            "robot_2" in info.get("avoidable_detour_agents", ())
        )
        reversals += int("robot_2" in info.get("unexplained_reversal_agents", ()))
        short_cycles += int("robot_2" in info.get("short_cycle_agents", ()))
        invalid_switches += int(
            "robot_2" in info.get("invalid_goal_switch_agents", ())
        )
        shutdowns += int("robot_2" in info.get("shutdowns", ()))
        simultaneous_violations += int(_simultaneous_violation(info))
        fact_failures += len(
            info.get("decision_trace", {}).get("fact_validation_failures", ())
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
        if terminated or truncated:
            break

    final = environment.get_state()
    completed_by_ai = tuple(
        task
        for task in final.completed_tasks
        if task.carrier_agent_id == "robot_2"
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
    fixed_seeds = tuple(range(51_000, 51_100))
    rng = np.random.default_rng(20_260_831)
    random_seeds = tuple(
        int(value)
        for value in rng.choice(
            np.arange(1_000_000, 9_000_000), size=100, replace=False
        )
    )

    fixed = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=participant_noise_probability,
        )
        for seed in fixed_seeds
    )
    random = tuple(
        _run_episode(
            actor,
            config,
            seed=seed,
            participant_noise_probability=participant_noise_probability,
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
    combined = _summarize((*fixed, *random))
    gates = {
        "map_is_exactly_6x7": (config.rows, config.cols) == (6, 7),
        "map_has_no_four_way_cross": not bool(
            WarehouseMultiAgentEnv(config).layout.four_way_intersections
        ),
        "three_cell_exit": len(
            WarehouseMultiAgentEnv(config).layout.robot_exit_positions
        )
        == 3,
        "mean_deliveries_at_least_5": combined["mean"]["deliveries"] >= 5.0,
        "real_proximity_observed": combined["total"]["proximity_frames"] > 0,
        "real_potential_conflicts_observed": (
            combined["total"]["potential_conflict_frames"] > 0
        ),
        "real_collisions_observed": combined["total"]["robot_collisions"] > 0,
        "real_yields_observed": combined["total"]["yield_events"] > 0,
        "no_permanent_deadlock": combined["deadlock_episodes"] == 0,
        "no_simultaneous_semantics_violation": (
            combined["total"]["simultaneous_violations"] == 0
        ),
        "no_explanation_fact_failure": (
            combined["total"]["explanation_fact_failures"] == 0
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
        "schema_version": "warehouse-study-acceptance.v2",
        "actor": {
            "path": str(actor_path.resolve()),
            "artifact_sha256": actor.artifact_sha256,
            "model_version": actor.metadata.model_version,
            "training_map_layout_id": actor.metadata.map_layout_id,
        },
        "production_map_layout_id": config.map_layout_id,
        "participant_noise_probability": participant_noise_probability,
        "fixed_seed_audit": _summarize(fixed),
        "random_seed_audit": _summarize(random),
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
