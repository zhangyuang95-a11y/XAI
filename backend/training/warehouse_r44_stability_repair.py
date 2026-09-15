"""Small post-PPO neural repair for r4.4 charger stability and recoverable stalls."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from backend.training import warehouse_r42_delivery_conservative as behavior
from backend.training import warehouse_r42_delivery_finetune as training
from backend.training import warehouse_r43_adaptation as r43
from backend.training.warehouse_r44_adaptation import actor_environment
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r44-neural-stability-repair.v1"


def _hash_observation(value):
    return sha256(np.asarray(value, dtype=np.float32).tobytes()).digest()


def _append(rows, seen, env, action):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = _hash_observation(observation)
    if key not in seen:
        seen.add(key)
        rows.append((observation, training.ACTION_INDEX[action]))


def _append_weighted(rows, env, action, weight=12):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    label = training.ACTION_INDEX[action]
    rows.extend((observation.copy(), label) for _ in range(weight))


def _set_carried(state, carrying):
    learner = state.by_id("robot_2")
    active = next((task for task in state.tasks if task.active), None)
    learner.carrying_task_id = None
    for task in state.tasks:
        if task.status == "carried" and task.carrier_agent_id == "robot_2":
            task.status = "available"
            task.carrier_agent_id = None
    if carrying and active is not None:
        active.status = "carried"
        active.carrier_agent_id = "robot_2"
        learner.carrying_task_id = active.task_id


def _active_reference_action(env):
    """Public-state training label that does not assume a human current move."""
    learner = env.state.by_id("robot_2")
    teammate = env.state.by_id("robot_1")
    charger = env.layout.charger_position
    move_cost = float(env.config.move_battery_cost)
    reserve_steps = float(env.config.charge_release_hysteresis_steps) + 2.0

    if learner.carrying_task_id:
        task = env.state.task_by_id(learner.carrying_task_id)
        goal = task.delivery_position
        route_steps = (
            shortest_path_distance(
                learner.position, task.delivery_position,
                env.config.map_layout_id,
            )
            + shortest_path_distance(
                task.delivery_position, charger,
                env.config.map_layout_id,
            )
        )
    else:
        available = [task for task in env.state.tasks
                     if task.status == "available"]
        task = min(available, key=lambda item: (
            shortest_path_distance(
                learner.position, item.pickup_position,
                env.config.map_layout_id,
            )
            + shortest_path_distance(
                item.pickup_position, item.delivery_position,
                env.config.map_layout_id,
            ),
            item.task_id,
        )) if available else None
        goal = task.pickup_position if task is not None else learner.position
        route_steps = (0 if task is None else
            shortest_path_distance(
                learner.position, task.pickup_position,
                env.config.map_layout_id,
            )
            + shortest_path_distance(
                task.pickup_position, task.delivery_position,
                env.config.map_layout_id,
            )
            + shortest_path_distance(
                task.delivery_position, charger,
                env.config.map_layout_id,
            ))
    required = move_cost * (route_steps + reserve_steps)
    urgent_handoff = bool(
        learner.position == charger
        and learner.battery + env.config.charge_per_wait > 60.0
        and teammate.active and teammate.battery <= 20.0
        and shortest_path_distance(
            teammate.position, charger, env.config.map_layout_id
        ) <= 2
    )
    # A two-move release margin makes the charging decision invariant to the
    # first moves away from the charger.  It is still a learned label: the
    # runtime never forces the Actor to remain on or leave the charger.
    if (learner.position == charger and not urgent_handoff
            and task is not None
            and learner.battery < min(100.0, required)):
        return "WAIT"
    if learner.position != charger and learner.battery < min(100.0, required):
        goal = charger
    if learner.position == goal and not urgent_handoff:
        return "WAIT"
    # When a charged occupant must yield, select a task-facing safe departure
    # instead of teaching the obsolete long-occupancy behavior.
    if urgent_handoff:
        if task is not None:
            goal = (task.delivery_position if learner.carrying_task_id
                    else task.pickup_position)
    candidates = []
    current_distance = shortest_path_distance(
        learner.position, goal, env.config.map_layout_id
    )
    for action, delta in MOVE_DELTAS.items():
        target = (learner.position[0] + delta[0],
                  learner.position[1] + delta[1])
        if not env.layout.is_passable(target):
            continue
        _targets, executed, _invalid, collision, _kind, _intended = (
            env._resolve_motion(
                env.state, {"robot_1": "WAIT", "robot_2": action}
            )
        )
        if collision or executed["robot_2"] != action:
            continue
        distance = shortest_path_distance(target, goal, env.config.map_layout_id)
        if goal == charger and distance >= current_distance:
            continue
        candidates.append((
            distance,
            ACTIONS.index(action), action,
        ))
    return min(candidates)[2] if candidates else "WAIT"


def targeted_rows(contract, scenes):
    rows, seen = [], set()
    rng = random.Random(260915944)
    for scene in scenes[:64]:
        template = actor_environment(contract, scene)
        charger = template.layout.charger_position
        far = max(
            (point for point in template.layout.passable_positions
             if point != charger),
            key=lambda point: shortest_path_distance(
                point, charger, template.config.map_layout_id
            ),
        )
        blocker_positions = [far]
        for action in ("UP", "LEFT", "RIGHT"):
            delta = {"UP": (-1, 0), "LEFT": (0, -1),
                     "RIGHT": (0, 1)}[action]
            point = (charger[0] + delta[0], charger[1] + delta[1])
            if template.layout.is_passable(point):
                blocker_positions.append(point)
        for carrying in (False, True):
            for battery in range(9, 100, 5):
                for teammate_position in blocker_positions:
                    teammate_batteries = ((100.0,) if teammate_position == far
                                          else (15.0, 100.0))
                    for teammate_battery in teammate_batteries:
                        env = actor_environment(contract, scene)
                        state = env.get_state()
                        learner = state.by_id("robot_2")
                        teammate = state.by_id("robot_1")
                        learner.position = charger
                        learner.battery = float(battery)
                        learner.active = True
                        learner.charge_mode_active = True
                        learner.charger_wait_streak = 2
                        learner.last_action = learner.last_executed_action = "WAIT"
                        learner.recent_positions = (charger, charger, charger)
                        teammate.position = teammate_position
                        teammate.battery = teammate_battery
                        teammate.active = True
                        _set_carried(state, carrying)
                        env.set_state(state)
                        _append(rows, seen, env, _active_reference_action(env))

        # Teach the complete screenshot-like recovery: approach the occupied
        # charger, wait while it is unavailable, enter within one decision of
        # release, charge to the public work budget, and resume the task.
        env = actor_environment(contract, scene)
        state = env.get_state()
        state.frame = 50
        player = state.by_id("robot_1")
        learner = state.by_id("robot_2")
        player.position = charger
        player.battery = 71.0
        learner.position = (3, 0)
        learner.battery = 34.0
        learner.active = True
        learner.carrying_task_id = None
        learner.last_action = learner.last_executed_action = "WAIT"
        for task, endpoints in zip(
            state.tasks, (((2, 5), (4, 4)), ((1, 1), (4, 2)))
        ):
            task.pickup_position, task.delivery_position = endpoints
            task.status = "available"
            task.carrier_agent_id = None
        env.set_state(state)
        for step in range(35):
            action = _active_reference_action(env)
            _append_weighted(rows, env, action)
            player_action = "UP" if step == 5 else "WAIT"
            env.step({"robot_1": player_action, "robot_2": action},
                     decision_metadata={"policy_action": action,
                                        "submitted_action": action,
                                        "post_policy_overrides": 0})
            if env.done:
                break

        # DAgger on the parent Actor's own training-scene distribution.  This
        # directly labels the wait/leave/return states that static charger
        # samples miss, while retaining the train/play split.
        rollout = actor_environment(contract, scene)
        rollout_rng = random.Random(260915500 + len(rows))
        for _ in range(rollout.config.horizon):
            label = _active_reference_action(rollout)
            actor_actions, _ = contract.act(
                rollout.observations(), deterministic=True
            )
            learner = rollout.state.by_id("robot_2")
            near_charger = shortest_path_distance(
                learner.position, rollout.layout.charger_position,
                rollout.config.map_layout_id,
            ) <= 1
            correction_weight = (
                24 if near_charger
                and actor_actions["robot_2"] != label else 4
            )
            _append_weighted(rows, rollout, label,
                             weight=correction_weight)
            player_action = partner_action(
                rollout, "robot_1", "skilled", rollout_rng
            )
            rollout.step(
                {"robot_1": player_action,
                 "robot_2": actor_actions["robot_2"]},
                decision_metadata={
                    "policy_action": actor_actions["robot_2"],
                    "submitted_action": actor_actions["robot_2"],
                    "post_policy_overrides": 0,
                },
            )
            if rollout.done:
                break
    return rows


def anchor_rows(actor, scenes):
    rows, seen = [], set()
    for index, scene in enumerate(scenes[:48]):
        env = actor_environment(actor, scene)
        rng = random.Random(440000 + index)
        for _ in range(70):
            observations = env.observations()
            for agent_id in env.agent_ids:
                value = observations[agent_id].astype(np.float32, copy=True)
                key = _hash_observation(value)
                if key not in seen:
                    seen.add(key)
                    rows.append(value)
            actions, _ = actor.act(observations, deterministic=True)
            player = partner_action(env, "robot_1", "skilled", rng)
            env.step({"robot_1": player, "robot_2": actions["robot_2"]},
                     decision_metadata={"policy_action": actions["robot_2"],
                                        "submitted_action": actions["robot_2"],
                                        "post_policy_overrides": 0})
            if env.done:
                break
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--retention", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=260915946)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    actor_path = Path(args.actor).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = (list(manifest["splits"]["train"])
              + list(manifest["splits"]["conflict_validation"]))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    training.actor_environment = actor_environment
    behavior.actor_environment = actor_environment
    model, contract, _ = r43._load_expanded_model(
        Path(args.parent).resolve(), device, actor_path
    )
    targets = targeted_rows(contract, scenes)
    anchors = anchor_rows(contract, scenes)
    target_x = np.stack([row[0] for row in targets]).astype(np.float32)
    target_y = np.asarray([row[1] for row in targets], dtype=np.int64)
    anchor_x = np.stack(anchors).astype(np.float32)
    parent_logits = contract.logits(anchor_x)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate,
                                 eps=1e-5)
    rng = np.random.default_rng(args.seed)
    candidates, history = [], []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(target_x))
        total = correct = count = 0
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            anchor_indices = rng.integers(0, len(anchor_x), size=len(indices))
            x = torch.as_tensor(target_x[indices], device=device)
            y = torch.as_tensor(target_y[indices], device=device)
            ax = torch.as_tensor(anchor_x[anchor_indices], device=device)
            expected = torch.as_tensor(parent_logits[anchor_indices], device=device)
            logits = model.actor_logits(x)
            loss = (nn.functional.cross_entropy(logits, y)
                    + args.retention * nn.functional.mse_loss(
                        model.actor_logits(ax), expected
                    ))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .35)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == y).sum().detach())
            count += len(indices)
        path = output / f"actor_epoch_{epoch:02d}.npz"
        training.export_actor(model, contract, path, metadata={
            "finetune_version": VERSION,
            "parent_actor_sha256": contract.sha256,
            "scene_manifest_sha256": training.file_hash(manifest_path),
            "targeted_rows": len(targets),
            "retention_rows": len(anchors),
            "fresh_ppo_joint_steps_before_repair": 32768,
            "runtime_action_override": False,
        })
        carried = training.evaluate_carried(path, manifest, maximum_scenes=32)
        dev = behavior._dev_behavior(path, manifest, maximum_scenes=16)
        row = {"epoch": epoch, "path": path.name,
               "sha256": training.file_hash(path),
               "target_accuracy": correct / count,
               "carried": carried, "development": dev}
        candidates.append(row)
        history.append({"epoch": epoch, "loss": total / count,
                        "target_accuracy": correct / count})
    eligible = [row for row in candidates
                if row["carried"]["rate"] >= .90
                and row["development"]["shutdowns"] == 0]
    pool = eligible or candidates
    selected = max(pool, key=lambda row: (
        row["target_accuracy"], row["development"]["total_deliveries"],
        -row["development"]["waits"], -row["development"]["collisions"],
    ))
    final = output / "actor.npz"
    final.write_bytes((output / selected["path"]).read_bytes())
    report = {
        "version": VERSION, "passed": bool(eligible),
        "actor_parent_sha256": contract.sha256,
        "actor_sha256": training.file_hash(final),
        "selected": selected, "candidates": candidates, "history": history,
        "targeted_rows": len(targets), "retention_rows": len(anchors),
        "fresh_ppo_joint_steps_before_repair": 32768,
        "move_battery_cost": 3.0,
        "runtime_action_override": False,
        "action_authority": {"policy_action_equals_submitted_action": True,
                             "runtime_overrides": 0},
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "training_report.json").write_text(json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
