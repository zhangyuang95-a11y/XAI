"""Bounded r4.5 adaptation for complete charge-work-return cycles."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

from backend.training import warehouse_r42_delivery_conservative as trainer
from backend.training import warehouse_r42_delivery_dagger as dagger
from backend.training import warehouse_r42_delivery_finetune as base
from backend.training import warehouse_r43_adaptation as r43
from backend.training.warehouse_r44_adaptation import migrated_scene
from backend.warehouse_r41_diagnostic_online_runtime import DEFAULT_REWARD_CONFIG
from backend.warehouse_r45_runtime import R45WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor
from env.warehouse_native.partners import partner_action
from env.warehouse_native.r41_diagnostic_conflict import (
    diagnostic_scene_fingerprint, reset_diagnostic_scenario,
)
from env.warehouse_native.r45_cycle import R45_OBSERVATION_FEATURE_NAMES
from env.warehouse_native.r45_energy import selected_energy_budget


VERSION = "warehouse-r45-cycle-adaptation.v1"
_ORIGINAL_COLLECT = dagger.collect_dagger_rows
_ORIGINAL_ANCHORS = dagger.collect_anchor_rows


def _expanded_parent(source: Path, destination: Path):
    parent = NumPyNativeActor(source)
    expected_features = list(R45WarehouseEnv(
                collaborative_study_config(move_battery_cost=3.0),
                reward_config=deepcopy(DEFAULT_REWARD_CONFIG),
            ).feature_names)
    parent_features = list(parent.metadata.get("feature_names", []))
    if parent_features == expected_features:
        destination.write_bytes(source.read_bytes())
        return parent.sha256
    if not parent_features or not set(parent_features).issubset(expected_features):
        raise ValueError("r4.5 parent observation features cannot be migrated")
    model = NativeActorCritic(
        len(expected_features),
        parent.state_dim, parent.hidden,
    )
    state = {}
    for name, value in parent.weights.items():
        if name == "0.weight":
            padded = np.zeros((parent.hidden, model.obs_dim), dtype=np.float32)
            target_index = {name: index for index, name in enumerate(expected_features)}
            for old_index, feature_name in enumerate(parent_features):
                padded[:, target_index[feature_name]] = value[:, old_index]
            value = padded
        state[name] = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)
    model.actor.load_state_dict(state)
    metadata = {
        **parent.metadata,
        "obs_dim": model.obs_dim,
        "feature_names": expected_features,
        "r45_observation_features": list(R45_OBSERVATION_FEATURE_NAMES),
        "r45_base_actor_sha256": parent.sha256,
        "r45_parent_expansion": "zero_initialized_then_trained",
        "runtime_action_override": False,
    }
    model.export_npz(destination, metadata)
    return parent.sha256


def actor_environment(actor, scene):
    env = R45WarehouseEnv(
        collaborative_study_config(move_battery_cost=3.0),
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG),
    )
    item = migrated_scene(scene)
    env.restore(deepcopy(item["snapshot"]))
    item["fingerprint"] = diagnostic_scene_fingerprint(env)
    reset_diagnostic_scenario(env, item)
    if list(env.feature_names) != actor.metadata.get("feature_names"):
        raise ValueError("r4.5 Actor feature contract differs")
    return env


def _teacher_action(env, agent_id="robot_2"):
    agent = env.state.by_id(agent_id)
    other_id = next(key for key in env.agent_ids if key != agent_id)
    budget = selected_energy_budget(env, agent_id)
    charger = env.layout.charger_position
    if budget is None:
        return "WAIT"
    urgent_handoff = False
    if agent.position == charger:
        evidence = env._qualification(
            env.state, {agent_id: "WAIT", other_id: "WAIT"}, agent_id
        )
        other = env.state.by_id(other_id)
        urgent_handoff = bool(
            agent.battery + env.config.charge_per_wait > 60.0
            and other.active and other.battery <= 20.0
            and evidence["teammate_distance"] <= 2
            and evidence["safe_departures"]
        )
        if not urgent_handoff and agent.battery < budget["required_battery"]:
            return "WAIT"
    goal = charger if (
        agent.position != charger and agent.battery < budget["required_battery"]
    ) else env.state.task_by_id(budget["task_id"]).delivery_position if (
        agent.carrying_task_id
    ) else env.state.task_by_id(budget["task_id"]).pickup_position
    current = shortest_path_distance(agent.position, goal, env.config.map_layout_id)
    improving = []
    for action, delta in MOVE_DELTAS.items():
        target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
        if not env.layout.is_passable(target):
            continue
        _targets, executed, _invalid, collision, _kind, _intended = env._resolve_motion(
            env.state, {agent_id: action, other_id: "WAIT"}
        )
        if collision or executed[agent_id] != action:
            continue
        distance = shortest_path_distance(target, goal, env.config.map_layout_id)
        if distance < current:
            improving.append((distance, ACTIONS.index(action), action))
    if improving:
        return min(improving)[2]
    # A teammate can occupy every distance-reducing exit in a narrow aisle.
    # One bounded sidestep is then productive clearance rather than a detour;
    # without it both public-state policies can wait forever.
    if shortest_path_distance(
            agent.position, env.state.by_id(other_id).position,
            env.config.map_layout_id) <= 2:
        clearance = []
        for action, delta in MOVE_DELTAS.items():
            target = (agent.position[0] + delta[0],
                      agent.position[1] + delta[1])
            if not env.layout.is_passable(target):
                continue
            _targets, executed, _invalid, collision, _kind, _intended = (
                env._resolve_motion(
                    env.state, {agent_id: action, other_id: "WAIT"}
                )
            )
            if collision or executed[agent_id] != action:
                continue
            clearance.append((
                shortest_path_distance(target, goal,
                                       env.config.map_layout_id),
                -shortest_path_distance(
                    target, env.state.by_id(other_id).position,
                    env.config.map_layout_id),
                ACTIONS.index(action), action,
            ))
        if clearance:
            return min(clearance)[-1]
    return "WAIT"


def _coordination_teacher_action(env, agent_id="robot_2"):
    """Combine initiative with public-history collision recovery."""
    agent = env.state.by_id(agent_id)
    budget = selected_energy_budget(env, agent_id)
    other_id = next(key for key in env.agent_ids if key != agent_id)
    other = env.state.by_id(other_id)
    if env.state.last_robot_collision_event:
        if budget is None:
            return "WAIT"
        task = env.state.task_by_id(budget["task_id"])
        goal = (env.layout.charger_position
                if agent.battery < budget["required_battery"] else
                task.delivery_position if agent.carrying_task_id else
                task.pickup_position)
        predicted = (other.last_action if other.last_action in ACTIONS else
                     "WAIT")
        candidates = []
        for action in ACTIONS:
            _targets, executed, _invalid, collision, _kind, _intended = (
                env._resolve_motion(
                    env.state, {agent_id: action, other_id: predicted}
                )
            )
            if collision or executed[agent_id] != action:
                continue
            position = _targets[agent_id]
            candidates.append((
                shortest_path_distance(position, goal,
                                       env.config.map_layout_id),
                action == "WAIT", ACTIONS.index(action), action,
            ))
        return min(candidates)[-1] if candidates else "WAIT"
    # An energy-deficit robot follows the charger route directly.  This avoids
    # inheriting a partner's unrelated task allocation while it is returning.
    if budget is not None and agent.battery < budget["required_battery"]:
        return _teacher_action(env, agent_id)
    if other.last_executed_action != "WAIT":
        return partner_action(env, agent_id, "skilled", random.Random(450_045))
    return _teacher_action(env, agent_id)


def _append(rows, seen, env, action, weight=1):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = sha256(observation.tobytes()).digest()
    if key in seen:
        return
    seen.add(key)
    rows.extend((observation.copy(), base.ACTION_INDEX[action]) for _ in range(weight))


def collect_dagger_rows(model, contract, entries, **kwargs):
    # r4.4 corrections used a 2%-energy teacher and conflict with the unified
    # r4.5 budget.  Build a fresh deterministic dataset instead of mixing those
    # contradictory labels into the adaptation target.
    maximum = min(96, int(kwargs.get("maximum_scenes", 96)))
    rows = []
    seen = set()
    for scene in entries[:maximum]:
        template = actor_environment(contract, scene)
        charger = template.layout.charger_position
        for battery in range(9, 100, 4):
            for carrying in (False, True):
                env = actor_environment(contract, scene)
                state = env.get_state()
                learner = state.by_id("robot_2")
                other = state.by_id("robot_1")
                learner.position = charger
                learner.battery = float(battery)
                learner.active = True
                other.battery = 100.0
                if carrying:
                    task = next(task for task in state.tasks if task.active)
                    task.status = "carried"
                    task.carrier_agent_id = learner.agent_id
                    learner.carrying_task_id = task.task_id
                else:
                    learner.carrying_task_id = None
                env.set_state(state)
                _append(rows, seen, env, _coordination_teacher_action(env), weight=40)
        # Roll out complete teacher cycles against a stationary and a moving
        # public-state partner.  The learner never sees the partner's current command.
        for partner_kind in ("wait", "skilled", "assertive"):
            env = actor_environment(contract, scene)
            rng = random.Random(450000 + len(rows))
            for _ in range(env.config.horizon):
                action = _coordination_teacher_action(env)
                weight = 16 if env.state.last_robot_collision_event else 3
                _append(rows, seen, env, action, weight=weight)
                # Expert rollouts avoid most collisions, so explicitly expose
                # the NN to the same public post-collision state it must recover
                # from at runtime.  No future partner command is included.
                distance = shortest_path_distance(
                    env.state.by_id("robot_1").position,
                    env.state.by_id("robot_2").position,
                    env.config.map_layout_id,
                )
                if distance <= 3 and not env.state.last_robot_collision_event:
                    saved = env.get_state()
                    collision_state = deepcopy(saved)
                    collision_state.last_robot_collision_event = True
                    collision_state.last_robot_collision_kind = "same_cell"
                    for person in collision_state.agents:
                        person.last_executed_action = "WAIT"
                    env.set_state(collision_state)
                    _append(rows, seen, env, _coordination_teacher_action(env),
                            weight=16)
                    env.set_state(saved)
                human = ("WAIT" if partner_kind == "wait" else
                         partner_action(env, "robot_1", partner_kind, rng))
                env.step({"robot_1": human, "robot_2": action})
                if env.done:
                    break
        # DAgger pass: visit the states produced by the current neural policy,
        # then label those exact states with the public-state teacher.  Pure
        # teacher rollouts miss the rare collision and charger-cycle states in
        # which a small classification error otherwise compounds for many
        # steps.
        device = next(model.parameters()).device
        for partner_kind in ("wait", "skilled", "assertive"):
            env = actor_environment(contract, scene)
            rng = random.Random(451000 + len(rows))
            for _ in range(env.config.horizon):
                observation = env.observations()["robot_2"].astype(
                    np.float32, copy=True
                )
                with torch.no_grad():
                    logits = model.actor_logits(torch.as_tensor(
                        observation, device=device
                    ).unsqueeze(0))
                actor_action = ACTIONS[int(logits.argmax(-1).item())]
                teacher_action = _coordination_teacher_action(env)
                _append(
                    rows, seen, env, teacher_action,
                    weight=24 if actor_action != teacher_action else 2,
                )
                human = ("WAIT" if partner_kind == "wait" else
                         partner_action(env, "robot_1", partner_kind, rng))
                env.step({"robot_1": human, "robot_2": actor_action})
                if env.done:
                    break
        # Reproduce the public sequence around a blocked charger and its
        # release, which is the controlled three-step recovery acceptance case.
        env = actor_environment(contract, scene)
        state = env.get_state()
        learner, other = state.by_id("robot_2"), state.by_id("robot_1")
        charger = env.layout.charger_position
        adjacent = sorted(
            (position for position in env.layout.passable_positions
             if shortest_path_distance(position, charger,
                                       env.config.map_layout_id) == 1),
        )[0]
        learner.position, learner.battery = adjacent, 34.0
        other.position, other.battery = charger, 71.0
        learner.last_executed_action = other.last_executed_action = "WAIT"
        env.set_state(state)
        _append(rows, seen, env, _coordination_teacher_action(env), weight=20)
        departures = env._qualification(
            env.state, {"robot_1": "WAIT", "robot_2": "WAIT"}, "robot_1"
        )["safe_departures"]
        if departures:
            env.step({"robot_1": departures[0], "robot_2": "WAIT"})
            _append(rows, seen, env, _coordination_teacher_action(env), weight=20)
    return rows


def _r45_train(model, corrections, anchors, parent_logits, *, device,
               epochs, seed, callback, retention_loss_weight=0.0):
    """Class-balanced supervised adaptation to the corrected r4.5 teacher."""
    correction_x, correction_y = trainer._arrays(corrections)
    anchor_x = np.stack([row[0] for row in anchors]).astype(np.float32)
    counts = np.bincount(correction_y, minlength=len(ACTIONS)).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    class_weights = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4, eps=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(correction_x))
        total = correct = count = 0
        for start in range(0, len(order), 384):
            indices = order[start:start + 384]
            x = torch.as_tensor(correction_x[indices], device=device)
            y = torch.as_tensor(correction_y[indices], device=device)
            logits = model.actor_logits(x)
            loss = torch.nn.functional.cross_entropy(
                logits, y, weight=class_weights
            )
            if retention_loss_weight > 0 and len(anchor_x):
                anchor_indices = rng.integers(0, len(anchor_x), size=len(indices))
                ax = torch.as_tensor(anchor_x[anchor_indices], device=device)
                target = torch.as_tensor(parent_logits[anchor_indices], device=device)
                loss = loss + retention_loss_weight * torch.nn.functional.mse_loss(
                    model.actor_logits(ax), target
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == y).sum().detach())
            count += len(indices)
        history.append({"epoch": epoch, "loss": total / count,
                        "correction_accuracy": correct / count})
        callback(epoch, deepcopy(model.state_dict()))
    return history


def collect_anchor_rows(contract, entries, **kwargs):
    kwargs["maximum_scenes"] = min(32, int(kwargs.get("maximum_scenes", 32)))
    return _ORIGINAL_ANCHORS(contract, entries, **kwargs)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parent_index = arguments.index("--actor-parent")
    source = Path(arguments[parent_index + 1]).expanduser().resolve()
    output_index = arguments.index("--output")
    output = Path(arguments[output_index + 1]).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="warehouse-r45-expand-") as name:
        expanded = Path(name) / "expanded_parent.npz"
        base_sha = _expanded_parent(source, expanded)
        arguments[parent_index + 1] = str(expanded)
        base.actor_environment = actor_environment
        dagger.actor_environment = actor_environment
        trainer.actor_environment = actor_environment
        dagger.collect_dagger_rows = collect_dagger_rows
        dagger.collect_anchor_rows = collect_anchor_rows
        trainer._conservative_train = _r45_train
        base._load_model = r43._load_expanded_model
        trainer.VERSION = VERSION
        status = trainer.main(arguments)
        report_path = output / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["version"] = VERSION
        report["parent"]["base_actor_path"] = str(source)
        report["parent"]["base_actor_sha256"] = base_sha
        report["training"].update({
            "energy_budget_version": "warehouse-r45-energy-budget.v2",
            "move_battery_cost": 3.0,
            "complete_cycle_rows": True,
            "observation_features_added": list(R45_OBSERVATION_FEATURE_NAMES),
            "runtime_action_override": False,
        })
        base.write_json(report_path, report)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
