"""Targeted r4.4 neural adaptation for 3% energy and charger handoff."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

from backend.training import warehouse_r42_delivery_conservative as trainer
from backend.training import warehouse_r42_delivery_dagger as dagger
from backend.training import warehouse_r42_delivery_finetune as base
from backend.training import warehouse_r43_adaptation as r43
from backend.warehouse_r41_diagnostic_online_runtime import DEFAULT_REWARD_CONFIG
from backend.warehouse_r44_runtime import R44WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.r41_diagnostic_conflict import (
    diagnostic_scene_fingerprint,
    reset_diagnostic_scenario,
)
from env.warehouse_native.r44_charger import SNAPSHOT_KEY, VERSION as RULE_VERSION


VERSION = "warehouse-r44-targeted-adaptation.v1"


def migrated_scene(scene):
    value = deepcopy(scene)
    snapshot = value["snapshot"]
    snapshot.pop("r43_shared_charger", None)
    snapshot[SNAPSHOT_KEY] = {
        "version": RULE_VERSION,
        "streaks": {"robot_1": 0, "robot_2": 0},
        "penalized": {"robot_1": False, "robot_2": False},
        "last_event": None,
    }
    snapshot["configuration"]["move_battery_cost"] = 3.0
    return value


def actor_environment(actor, scene):
    env = R44WarehouseEnv(
        collaborative_study_config(move_battery_cost=3.0),
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG),
    )
    migrated = migrated_scene(scene)
    # The fingerprint includes the complete rule snapshot, so bind the
    # migrated r4.4 state before using the standard registry verifier.
    env.restore(deepcopy(migrated["snapshot"]))
    migrated["fingerprint"] = diagnostic_scene_fingerprint(env)
    reset_diagnostic_scenario(env, migrated)
    if list(env.feature_names) != actor.metadata.get("feature_names"):
        raise ValueError("r4.4 Actor feature contract differs")
    return env


def _append(rows, seen, env, action):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = sha256(observation.tobytes()).digest()
    if key not in seen:
        seen.add(key)
        rows.append((observation, base.ACTION_INDEX[action]))


def collect_dagger_rows(model, contract, entries, **kwargs):
    """Add public, recoverable stall states to the ordinary DAgger rollout."""
    original = dagger.collect_dagger_rows
    dagger.collect_dagger_rows = r43._ORIGINAL_COLLECT
    try:
        rows = list(r43.collect_dagger_rows(model, contract, entries, **kwargs))
    finally:
        dagger.collect_dagger_rows = original
    seen = {sha256(row[0].tobytes()).digest() for row in rows}
    for scene in entries[:64]:
        probe = actor_environment(contract, scene)
        excluded = {
            probe.layout.charger_position,
            *(task.pickup_position for task in probe.state.tasks),
            *(task.delivery_position for task in probe.state.tasks),
        }
        positions = [point for point in sorted(probe.layout.passable_positions)
                     if point not in excluded]
        for point in positions[:16]:
            for battery in (34.0, 55.0, 76.0):
                env = actor_environment(contract, scene)
                state = env.get_state()
                learner = state.by_id("robot_2")
                if point == state.by_id("robot_1").position:
                    continue
                learner.position = point
                learner.battery = battery
                learner.active = True
                learner.carrying_task_id = None
                learner.last_action = "WAIT"
                learner.last_executed_action = "WAIT"
                env.set_state(state)
                action = partner_action(
                    env, "robot_2", "skilled", random.Random(44)
                )
                if action != "WAIT":
                    _append(rows, seen, env, action)
        # The previous adaptation barely sampled the complete charger cycle.
        # Add teacher labels both below and above the exact work budget so the
        # network can learn a stable WAIT-until-ready boundary rather than
        # oscillating between the charger and its neighbouring cell.
        for carrying in (False, True):
            for battery in (9.0, 19.0, 29.0, 39.0, 45.0, 49.0,
                            55.0, 61.0, 69.0, 79.0, 89.0, 99.0):
                env = actor_environment(contract, scene)
                state = env.get_state()
                learner = state.by_id("robot_2")
                teammate = state.by_id("robot_1")
                teammate_positions = [
                    point for point in sorted(env.layout.passable_positions)
                    if point != env.layout.charger_position
                ]
                teammate.position = teammate_positions[0]
                teammate.battery = 100.0
                teammate.carrying_task_id = None
                learner.position = env.layout.charger_position
                learner.battery = battery
                learner.active = True
                learner.last_action = "WAIT"
                learner.last_executed_action = "WAIT"
                learner.charge_mode_active = True
                learner.charger_wait_streak = 2
                learner.steps_since_charging = 0
                learner.recent_positions = (
                    env.layout.charger_position,
                    env.layout.charger_position,
                    env.layout.charger_position,
                )
                active = next((task for task in state.tasks if task.active), None)
                if carrying and active is not None:
                    for task in state.tasks:
                        if task.task_id == active.task_id:
                            task.status = "carried"
                            task.carrier_agent_id = "robot_2"
                        elif task.status == "carried" and (
                                task.carrier_agent_id == "robot_2"):
                            task.status = "available"
                            task.carrier_agent_id = None
                    learner.carrying_task_id = active.task_id
                else:
                    learner.carrying_task_id = None
                    for task in state.tasks:
                        if (task.status == "carried"
                                and task.carrier_agent_id == "robot_2"):
                            task.status = "available"
                            task.carrier_agent_id = None
                env.set_state(state)
                action = partner_action(
                    env, "robot_2", "skilled", random.Random(4400)
                )
                _append(rows, seen, env, action)
    return rows


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parent_index = arguments.index("--actor-parent")
        parent_actor = Path(arguments[parent_index + 1]).expanduser().resolve()
        output_index = arguments.index("--output")
        output = Path(arguments[output_index + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        raise ValueError("r4.4 adaptation requires --actor-parent and --output")
    base.actor_environment = actor_environment
    dagger.actor_environment = actor_environment
    trainer.actor_environment = actor_environment
    r43.actor_environment = actor_environment
    dagger.collect_dagger_rows = collect_dagger_rows
    base._load_model = r43._load_expanded_model
    trainer.VERSION = VERSION
    status = trainer.main(arguments)
    report_path = output / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["version"] = VERSION
    report["parent"]["base_actor_path"] = str(parent_actor)
    report["training"].update({
        "energy_budget_source": "environment_config.move_battery_cost",
        "move_battery_cost": 3.0,
        "charger_penalty_in_ppo_reward": True,
        "targeted_recoverable_stall_states": True,
        "runtime_action_override": False,
    })
    base.write_json(report_path, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
