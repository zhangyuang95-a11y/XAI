"""Bounded r4.2 delivery-first neural fine-tuning.

The teacher is used only to label public observations before PPO.  The exported
artifact is a plain two-layer neural Actor and the runtime never imports this
module or a route planner.  A short actor-specific PPO phase follows imitation
so every selectable checkpoint contains fresh reinforcement-learning updates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    actor_environment,
)
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor


VERSION = "warehouse-r42-delivery-first-finetune.v2"
DEFAULT_PARENT = Path(
    "output/warehouse_native/r41_active_2m_20260911/"
    "boundaries/step_0250000"
)
DEFAULT_SCENES = Path(
    "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/"
    "manifest.json"
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value) -> str:
    return sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_model(parent: Path, device: torch.device, actor_parent: Path | None = None):
    checkpoint = torch.load(
        parent / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    actor = NumPyNativeActor(actor_parent or (parent / "actor.npz"))
    model = NativeActorCritic(actor.obs_dim, actor.state_dim, actor.hidden)
    model.load_state_dict(checkpoint["trainer"]["model"])
    if actor_parent is not None:
        actor_state = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in actor.weights.items()
        }
        model.actor.load_state_dict(actor_state)
    return model.to(device), actor, checkpoint


def _set_carried_probe(env, task_index: int, position=None, battery=100.0):
    state = env.get_state()
    actor = state.by_id("robot_2")
    participant = state.by_id("robot_1")
    task = state.tasks[task_index]
    actor.position = tuple(position or task.pickup_position)
    actor.battery = float(battery)
    actor.active = True
    actor.carrying_task_id = task.task_id
    task.status = "carried"
    task.carrier_agent_id = actor.agent_id
    task.claimed_frame = state.frame
    task.claimed_battery = float(battery)
    if participant.position == actor.position:
        participant.position = next(
            point for point in sorted(env.layout.passable_positions)
            if point not in {actor.position, task.delivery_position,
                             env.layout.charger_position}
        )
    env.set_state(state)


def _append_sample(rows, seen, env, label):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = sha256(observation.tobytes()).digest()
    if key in seen:
        return
    seen.add(key)
    rows.append((observation, ACTION_INDEX[label]))


def build_imitation_rows(scenes, actor, *, split: str, maximum_scenes=None):
    """Generate public-observation labels without reading neural logits."""
    entries = scenes["splits"][split]
    if maximum_scenes is not None:
        entries = entries[:maximum_scenes]
    rows, seen = [], set()
    for entry_index, entry in enumerate(entries):
        # Dense carried-task coverage is the missing behavior in the r4.2
        # failure.  Enumerate all legal positions and two useful energy levels.
        probe = actor_environment(actor, entry)
        for task_index in range(2):
            for position in sorted(probe.layout.passable_positions):
                for battery in (100.0, 64.0, 36.0, 20.0, 12.0, 6.0, 2.0):
                    env = actor_environment(actor, entry)
                    if position == env.state.by_id("robot_1").position:
                        continue
                    _set_carried_probe(env, task_index, position, battery)
                    # Once a package is carried, follow its shortest feasible
                    # route directly.  The public carrying flag separates this
                    # convention from empty-agent task allocation, so it does
                    # not recreate the contradictory empty-state labels.
                    label = partner_action(
                        env, "robot_2", "assertive", random.Random(0)
                    )
                    _append_sample(rows, seen, env, label)

        # The failed online Actor also waited indefinitely before selecting a
        # package in one conflict family.  Cover every legal empty-agent
        # position and the public charging boundary, without using play data.
        for position in sorted(probe.layout.passable_positions):
            for battery in (100.0, 64.0, 36.0, 20.0, 12.0, 6.0, 2.0):
                env = actor_environment(actor, entry)
                state = env.get_state()
                learner = state.by_id("robot_2")
                participant = state.by_id("robot_1")
                learner.position = position
                learner.battery = battery
                learner.active = True
                learner.carrying_task_id = None
                for task in state.tasks:
                    task.status = "available"
                    task.carrier_agent_id = None
                    task.claimed_frame = None
                    task.claimed_battery = None
                if participant.position == learner.position:
                    participant.position = next(
                        point for point in sorted(env.layout.passable_positions)
                        if point not in {learner.position,
                                         env.layout.charger_position}
                    )
                env.set_state(state)
                label = partner_action(
                    env, "robot_2", "skilled", random.Random(0)
                )
                _append_sample(rows, seen, env, label)
                # Cover the history pattern created when a previous policy has
                # already waited.  Without these public-history variants a
                # single borderline WAIT became self-reinforcing forever.
                if battery >= 36.0:
                    # Time remaining and task age are public inputs.  Cover a
                    # meaningful prefix of an otherwise unchanged state so a
                    # single WAIT/collision cannot become a self-reinforcing
                    # out-of-distribution loop later in the round.
                    for _ in range(12):
                        env.step({"robot_1": "WAIT", "robot_2": "WAIT"},
                                 decision_metadata={"policy_action": "WAIT",
                                 "submitted_action": "WAIT",
                                 "post_policy_overrides": 0})
                        if env.done:
                            break
                        recovery_label = partner_action(
                            env, "robot_2", "skilled", random.Random(0)
                        )
                        _append_sample(rows, seen, env, recovery_label)

        # Empty-agent task selection, full delivery cycles, and recovery from
        # partners who wait or rush.  Robot 2 always follows the same public
        # convention: coordinated task allocation while empty, direct routing
        # while carrying.  The player never sees the current neural action.
        # A single coherent public-state convention is essential here.  Mixing
        # the symmetric assignment with "both rush the nearest task" assigns
        # different labels to the same observation and produced the observed
        # waiting/collision failures in whole episodes.
        for player_profile in ("skilled", "assertive", "wait"):
            env = actor_environment(actor, entry)
            rng = random.Random(42_000 + entry_index)
            for _ in range(env.config.horizon):
                label_profile = ("assertive" if
                    env.state.by_id("robot_2").carrying_task_id else "skilled")
                label = partner_action(env, "robot_2", label_profile, rng)
                _append_sample(rows, seen, env, label)
                participant = ("WAIT" if player_profile == "wait" else
                    partner_action(env, "robot_1", player_profile, rng))
                env.step({"robot_1": participant, "robot_2": label})
                if env.done:
                    break
    observations = np.stack([row[0] for row in rows]).astype(np.float32)
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    return observations, labels


def imitation_train(model, observations, labels, *, device, epochs, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4, eps=1e-5)
    class_weights = torch.ones(len(ACTIONS), device=device)
    class_weights[ACTION_INDEX["WAIT"]] = .60
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    batch_size = 512
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(observations))
        total_loss = correct = count = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = torch.as_tensor(observations[indices], device=device)
            y = torch.as_tensor(labels[indices], device=device)
            logits = model.actor_logits(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == y).sum().detach())
            count += len(indices)
        history.append({"epoch": epoch, "loss": total_loss / count,
                        "accuracy": correct / count})
    return history


def _goal_distance(env):
    actor = env.state.by_id("robot_2")
    charger = env.layout.charger_position
    distance = lambda left, right: shortest_path_distance(
        left, right, env.config.map_layout_id
    )
    if actor.carrying_task_id:
        task = env.state.task_by_id(actor.carrying_task_id)
        work = distance(actor.position, task.delivery_position)
        required = env.config.move_battery_cost * (
            work + distance(task.delivery_position, charger) + 1
        )
        goal = charger if actor.battery < required else task.delivery_position
        return float(distance(actor.position, goal))
    available = [task for task in env.state.tasks if task.status == "available"]
    if not available:
        return float(distance(actor.position, charger))
    task = min(available, key=lambda item: (
        distance(actor.position, item.pickup_position), item.task_id
    ))
    work = distance(actor.position, task.pickup_position) + distance(
        task.pickup_position, task.delivery_position
    )
    required = env.config.move_battery_cost * (
        work + distance(task.delivery_position, charger) + 1
    )
    goal = charger if actor.battery < required else task.pickup_position
    return float(distance(actor.position, goal))


def _reset_env(env, entries, rng, actor):
    entry = entries[int(rng.integers(len(entries)))]
    fresh = actor_environment(actor, entry)
    env.__dict__.clear()
    env.__dict__.update(fresh.__dict__)


def ppo_finetune(model, scenes, actor_contract, *, device, joint_steps, seed,
                 checkpoint_callback, reference_observations,
                 reference_labels, actor_learning_rate=8e-5,
                 behavior_coefficient=.20):
    """Small actor-specific PPO update after imitation."""
    rng = np.random.default_rng(seed)
    entries = scenes["splits"]["train"]
    envs = [actor_environment(actor_contract, entries[i]) for i in range(16)]
    partner_profiles = (["skilled"] * 8 + ["assertive"] * 4
                        + ["wait"] * 4)
    partner_rngs = [random.Random(seed + i) for i in range(16)]
    actor_optimizer = torch.optim.Adam(
        model.actor.parameters(), lr=actor_learning_rate, eps=1e-5
    )
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4, eps=1e-5)
    gamma, gae_lambda, clip = .99, .95, .2
    completed_steps = 0
    updates = []
    model.train()
    while completed_steps < joint_steps:
        horizon = min(64, (joint_steps - completed_steps + 15) // 16)
        rollout = {key: [] for key in (
            "obs", "states", "actions", "old_logp", "rewards", "values", "dones"
        )}
        for _ in range(horizon):
            observations = np.stack([
                env.observations()["robot_2"] for env in envs
            ]).astype(np.float32)
            states = np.stack([env.global_state() for env in envs]).astype(np.float32)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observations, device=device)
                state_tensor = torch.as_tensor(states, device=device)
                logits = model.actor_logits(obs_tensor)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = distribution.sample()
                old_logp = distribution.log_prob(actions)
                roles = torch.ones(len(envs), dtype=torch.long, device=device)
                values = model.values(state_tensor, roles)
            rewards = np.zeros(len(envs), dtype=np.float32)
            dones = np.zeros(len(envs), dtype=np.float32)
            for index, env in enumerate(envs):
                action = ACTIONS[int(actions[index])]
                before = env.state.by_id("robot_2")
                before_deliveries = before.deliveries_completed
                before_carrying = before.carrying_task_id
                before_distance = _goal_distance(env)
                participant = ("WAIT" if partner_profiles[index] == "wait" else
                    partner_action(env, "robot_1", partner_profiles[index],
                                   partner_rngs[index]))
                _, _, terminated, truncated, info = env.step({
                    "robot_1": participant, "robot_2": action,
                }, decision_metadata={
                    "policy_action": action, "submitted_action": action,
                    "post_policy_overrides": 0,
                })
                after = env.state.by_id("robot_2")
                reward = -.01
                if after.deliveries_completed > before_deliveries:
                    reward += 12.0
                if before_carrying is None and after.carrying_task_id is not None:
                    reward += 2.0
                after_distance = _goal_distance(env) if not env.done else 0.0
                if not env.done:
                    reward += .25 * np.sign(before_distance - after_distance)
                if action == "WAIT" and before_distance > 0:
                    reward -= .10
                reward -= .08 * int(info["robot_collision"])
                reward -= 6.0 * int(not after.active)
                rewards[index] = reward
                if terminated or truncated:
                    dones[index] = 1.0
                    _reset_env(env, entries, rng, actor_contract)
            rollout["obs"].append(observations)
            rollout["states"].append(states)
            rollout["actions"].append(actions.detach().cpu().numpy())
            rollout["old_logp"].append(old_logp.detach().cpu().numpy())
            rollout["rewards"].append(rewards)
            rollout["values"].append(values.detach().cpu().numpy())
            rollout["dones"].append(dones)
            completed_steps += len(envs)
        with torch.no_grad():
            final_states = torch.as_tensor(
                np.stack([env.global_state() for env in envs]).astype(np.float32),
                device=device,
            )
            final_values = model.values(
                final_states, torch.ones(len(envs), dtype=torch.long, device=device)
            ).cpu().numpy()
        for key in rollout:
            rollout[key] = np.asarray(rollout[key])
        advantages = np.zeros_like(rollout["rewards"])
        last = np.zeros(len(envs), dtype=np.float32)
        for step in range(len(rollout["rewards"]) - 1, -1, -1):
            next_values = final_values if step == len(rollout["rewards"]) - 1 else rollout["values"][step + 1]
            alive = 1.0 - rollout["dones"][step]
            delta = rollout["rewards"][step] + gamma * next_values * alive - rollout["values"][step]
            last = delta + gamma * gae_lambda * alive * last
            advantages[step] = last
        returns = advantages + rollout["values"]
        flat = {
            "obs": rollout["obs"].reshape(-1, model.obs_dim),
            "states": rollout["states"].reshape(-1, model.state_dim),
            "actions": rollout["actions"].reshape(-1),
            "old_logp": rollout["old_logp"].reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        normalized = (flat["advantages"] - flat["advantages"].mean()) / (
            flat["advantages"].std() + 1e-8
        )
        flat["advantages"] = normalized
        losses = []
        for _ in range(4):
            for start in range(0, len(flat["actions"]), 256):
                indices = rng.permutation(len(flat["actions"]))[start:start + 256]
                ix = torch.as_tensor(indices, device=device)
                obs = torch.as_tensor(flat["obs"], device=device)[ix]
                states = torch.as_tensor(flat["states"], device=device)[ix]
                actions = torch.as_tensor(flat["actions"], device=device)[ix]
                old_logp = torch.as_tensor(flat["old_logp"], device=device)[ix]
                advantage = torch.as_tensor(flat["advantages"], device=device)[ix]
                target = torch.as_tensor(flat["returns"], device=device)[ix]
                distribution = torch.distributions.Categorical(
                    logits=model.actor_logits(obs)
                )
                logp = distribution.log_prob(actions)
                ratio = (logp - old_logp).exp()
                ppo_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - clip, 1 + clip) * advantage,
                ).mean() - .001 * distribution.entropy().mean()
                # Retain the coherent delivery convention while still making
                # real on-policy PPO updates.  This is training-only; the
                # exported Actor has no teacher, planner or action override.
                reference_indices = rng.integers(
                    0, len(reference_observations), size=min(256, len(indices))
                )
                reference_obs = torch.as_tensor(
                    reference_observations[reference_indices], device=device
                )
                reference_target = torch.as_tensor(
                    reference_labels[reference_indices], device=device
                )
                behavior_weights = torch.ones(len(ACTIONS), device=device)
                behavior_weights[ACTION_INDEX["WAIT"]] = .60
                behavior_loss = nn.functional.cross_entropy(
                    model.actor_logits(reference_obs), reference_target,
                    weight=behavior_weights,
                )
                actor_loss = ppo_loss + behavior_coefficient * behavior_loss
                value = model.values(
                    states, torch.ones(len(indices), dtype=torch.long, device=device)
                )
                critic_loss = .5 * (value - target).square().mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5)
                actor_optimizer.step()
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
                critic_optimizer.step()
                losses.append((float(actor_loss.detach()), float(critic_loss.detach())))
        updates.append({
            "joint_steps": completed_steps,
            "actor_loss": float(np.mean([item[0] for item in losses])),
            "critic_loss": float(np.mean([item[1] for item in losses])),
            "reward_mean": float(rollout["rewards"].mean()),
        })
        if completed_steps % 5_000 < len(envs) * horizon:
            checkpoint_callback(completed_steps, deepcopy(model.state_dict()))
    return updates


def export_actor(model, parent_actor, destination: Path, *, metadata):
    payload = dict(parent_actor.metadata)
    payload.update(metadata)
    payload["actor_parameters_sha256"] = sha256(b"".join(
        value.detach().cpu().numpy().astype(np.float32).tobytes()
        for value in model.actor.state_dict().values()
    )).hexdigest()
    model.export_npz(destination, payload)
    return file_hash(destination)


def evaluate_carried(actor_path: Path, scenes, *, maximum_scenes=32):
    actor = NumPyNativeActor(actor_path)
    rows = []
    for entry in scenes["splits"]["conflict_validation"][:maximum_scenes]:
        for task_index in range(2):
            env = actor_environment(actor, entry)
            _set_carried_probe(env, task_index)
            agent = env.state.by_id("robot_2")
            task = env.state.task_by_id(agent.carrying_task_id)
            shortest = shortest_path_distance(
                agent.position, task.delivery_position, env.config.map_layout_id
            )
            initial = agent.deliveries_completed
            actions = []
            for _ in range(shortest + 3):
                observations = env.observations()
                selected, _ = actor.act(observations, deterministic=True)
                action = selected["robot_2"]
                actions.append(action)
                env.step({"robot_1": "WAIT", "robot_2": action},
                         decision_metadata={"policy_action": action,
                         "submitted_action": action, "post_policy_overrides": 0})
                if env.state.by_id("robot_2").deliveries_completed > initial:
                    break
            delivered = env.state.by_id("robot_2").deliveries_completed > initial
            rows.append({"delivered": delivered,
                         "within_shortest_plus_two": delivered and len(actions) <= shortest + 2,
                         "shortest": shortest, "steps": len(actions)})
    return {
        "samples": len(rows),
        "delivered": sum(item["delivered"] for item in rows),
        "within_shortest_plus_two": sum(
            item["within_shortest_plus_two"] for item in rows
        ),
        "rate": sum(item["within_shortest_plus_two"] for item in rows) / len(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--actor-parent")
    parser.add_argument("--scenes", default=str(DEFAULT_SCENES))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--ppo-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260_915_042)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = Path(args.parent).resolve()
    scenes_path = Path(args.scenes).resolve()
    scenes = json.loads(scenes_path.read_text())
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    started = time.monotonic()
    actor_parent = (Path(args.actor_parent).resolve()
                    if args.actor_parent else None)
    model, parent_actor, parent_checkpoint = _load_model(
        parent, device, actor_parent
    )
    observations, labels = build_imitation_rows(
        scenes, parent_actor, split="train"
    )
    validation_observations, validation_labels = build_imitation_rows(
        scenes, parent_actor, split="conflict_validation", maximum_scenes=16
    )
    history = imitation_train(
        model, observations, labels, device=device, epochs=args.epochs,
        seed=args.seed,
    )
    with torch.no_grad():
        logits = model.actor_logits(torch.as_tensor(
            validation_observations, device=device
        ))
        validation_accuracy = float((
            logits.argmax(-1).cpu().numpy() == validation_labels
        ).mean())
    common_metadata = {
        "finetune_version": VERSION,
        "parent_actor_sha256": parent_actor.sha256,
        "parent_checkpoint_sha256": file_hash(parent / "checkpoint.pt"),
        "scene_manifest_sha256": file_hash(scenes_path),
        "imitation_rows": len(observations),
        "imitation_validation_rows": len(validation_observations),
        "imitation_validation_accuracy": validation_accuracy,
        "imitation_teacher_runtime_available": False,
        "runtime_action_override": False,
    }
    bc_actor = output / "actor_bc.npz"
    export_actor(model, parent_actor, bc_actor, metadata={
        **common_metadata, "fresh_ppo_joint_steps": 0,
    })
    evaluations = [{"stage": "bc", **evaluate_carried(bc_actor, scenes)}]
    checkpoints = []

    def save(step, state):
        model.load_state_dict(state)
        path = output / f"actor_ppo_{step:07d}.npz"
        actor_sha = export_actor(model, parent_actor, path, metadata={
            **common_metadata, "fresh_ppo_joint_steps": int(step),
            "ppo_actor_specific_reward": True,
        })
        evaluation = evaluate_carried(path, scenes)
        evaluations.append({"stage": f"ppo_{step}", **evaluation})
        checkpoints.append({"step": int(step), "path": path.name,
                            "sha256": actor_sha, "evaluation": evaluation})

    updates = ppo_finetune(
        model, scenes, parent_actor, device=device, joint_steps=args.ppo_steps,
        seed=args.seed + 1, checkpoint_callback=save,
        reference_observations=observations,
        reference_labels=labels,
    )
    passing = [item for item in checkpoints
               if item["evaluation"]["rate"] >= .90]
    selected = min(passing, key=lambda item: item["step"]) if passing else max(
        checkpoints, key=lambda item: item["evaluation"]["rate"]
    )
    selected_path = output / selected["path"]
    final_path = output / "actor.npz"
    final_path.write_bytes(selected_path.read_bytes())
    report = {
        "version": VERSION,
        "passed": bool(passing),
        "selected": {**selected, "path": final_path.name,
                     "sha256": file_hash(final_path)},
        "parent": {"actor_sha256": parent_actor.sha256,
                   "checkpoint_sha256": file_hash(parent / "checkpoint.pt")},
        "training": {
            "imitation_rows": len(observations),
            "imitation_epochs": args.epochs,
            "imitation_final": history[-1],
            "imitation_validation_rows": len(validation_observations),
            "imitation_validation_accuracy": validation_accuracy,
            "fresh_ppo_joint_steps": args.ppo_steps,
            "ppo_updates": updates,
            "runtime_action_override": False,
        },
        "carried_validation": evaluations,
        "action_authority": {
            "policy_action_equals_submitted_action": True,
            "runtime_overrides": 0,
        },
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
