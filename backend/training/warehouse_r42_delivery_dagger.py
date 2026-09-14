"""Focused DAgger recovery for the r4.2 delivery Actor.

The public-state skilled partner supplies labels only during training.  The
deployed artifact remains the same plain two-layer neural Actor and receives
no route planner, action mask, override, or retry at runtime.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from backend.training import warehouse_r42_delivery_finetune as base
from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    actor_environment,
)
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r42-delivery-dagger.v1"


def _teacher_action(env):
    """Direct delivery with public-state yielding only for an occupied cell."""
    learner = env.state.by_id("robot_2")
    if learner.carrying_task_id is None:
        # Give robot 2 a stable task convention while both tasks exist.  If
        # the participant leaves only the other parity unfinished, take the
        # nearest remaining task instead of waiting indefinitely.
        assigned = partner_action(
            env, "robot_2", "fixed_task", random.Random(0)
        )
        available = [task for task in env.state.tasks
                     if task.status == "available"]
        charging = (learner.position == env.layout.charger_position
                    and learner.battery < 100)
        return (partner_action(env, "robot_2", "assertive", random.Random(0))
                if assigned == "WAIT" and available and not charging
                else assigned)
    direct = partner_action(env, "robot_2", "assertive", random.Random(0))
    delta = base.MOVE_DELTAS.get(direct, (0, 0)) if hasattr(base, "MOVE_DELTAS") else {
        "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
        "RIGHT": (0, 1), "WAIT": (0, 0),
    }[direct]
    target = (learner.position[0] + delta[0], learner.position[1] + delta[1])
    other = env.state.by_id("robot_1")
    if direct != "WAIT" and target == other.position:
        return partner_action(env, "robot_2", "skilled", random.Random(0))
    return direct


def _append(rows, seen, env, label):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = sha256(observation.tobytes()).digest()
    if key not in seen:
        seen.add(key)
        rows.append((observation, base.ACTION_INDEX[label]))


def _temporary_actor(model, contract, directory: Path, name: str):
    path = directory / name
    base.export_actor(model, contract, path, metadata={
        "finetune_version": VERSION,
        "runtime_action_override": False,
        "dagger_intermediate": True,
    })
    return NumPyNativeActor(path)


def collect_dagger_rows(model, contract, entries, *, directory, round_index,
                        maximum_scenes=64):
    actor = _temporary_actor(model, contract, directory,
                             f"dagger_round_{round_index}.npz")
    rows, seen = [], set()
    profiles = ("skilled", "assertive", "wait")
    for scene_index, entry in enumerate(entries[:maximum_scenes]):
        for profile_index, profile in enumerate(profiles):
            env = actor_environment(contract, entry)
            rng = random.Random(260_915_800 + 1000 * round_index
                                + 10 * scene_index + profile_index)
            for _ in range(env.config.horizon):
                label = _teacher_action(env)
                # The parent already passes the direct carried-delivery gate.
                # DAgger targets the empty/recovery distribution shift without
                # relabelling that established carrying behavior.
                if env.state.by_id("robot_2").carrying_task_id is None:
                    _append(rows, seen, env, label)
                actions, _ = actor.act(env.observations(), deterministic=True)
                player = ("WAIT" if profile == "wait" else
                          partner_action(env, "robot_1", profile, rng))
                env.step({"robot_1": player, "robot_2": actions["robot_2"]},
                         decision_metadata={
                             "policy_action": actions["robot_2"],
                             "submitted_action": actions["robot_2"],
                             "post_policy_overrides": 0,
                         })
                if env.done:
                    break
    return rows


def collect_anchor_rows(contract, entries, *, maximum_scenes=48):
    """Protect direct carrying, empty allocation, and charging boundaries."""
    rows, seen = [], set()
    for entry in entries[:maximum_scenes]:
        probe = actor_environment(contract, entry)
        positions = sorted(probe.layout.passable_positions)
        for task_index in range(2):
            for position in positions:
                for battery in (100.0, 48.0, 24.0, 12.0, 4.0):
                    env = actor_environment(contract, entry)
                    if position == env.state.by_id("robot_1").position:
                        continue
                    base._set_carried_probe(env, task_index, position, battery)
                    label = partner_action(
                        env, "robot_2", "assertive", random.Random(0)
                    )
                    _append(rows, seen, env, label)
        for position in positions:
            for battery in (100.0, 48.0, 24.0, 12.0, 4.0):
                env = actor_environment(contract, entry)
                state = env.get_state()
                learner = state.by_id("robot_2")
                other = state.by_id("robot_1")
                learner.position = position
                learner.battery = battery
                learner.active = True
                learner.carrying_task_id = None
                for task in state.tasks:
                    task.status = "available"
                    task.carrier_agent_id = None
                    task.claimed_frame = None
                    task.claimed_battery = None
                if other.position == position:
                    other.position = next(
                        point for point in positions
                        if point not in {position, env.layout.charger_position}
                    )
                env.set_state(state)
                _append(rows, seen, env, partner_action(
                    env, "robot_2", "skilled", random.Random(0)
                ))
    return rows


def _arrays(rows):
    return (np.stack([row[0] for row in rows]).astype(np.float32),
            np.asarray([row[1] for row in rows], dtype=np.int64))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default=str(base.DEFAULT_PARENT))
    parser.add_argument("--scenes", default=str(base.DEFAULT_SCENES))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--ppo-steps", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=260_915_842)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    actor_parent = Path(args.actor_parent).resolve()
    parent = Path(args.parent).resolve()
    scene_path = Path(args.scenes).resolve()
    scenes = json.loads(scene_path.read_text(encoding="utf-8"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, contract, _ = base._load_model(parent, device, actor_parent)
    anchors = collect_anchor_rows(contract, scenes["splits"]["train"])
    anchor_repeats = 8
    aggregate = list(anchors)
    seen = {sha256(row[0].tobytes()).digest() for row in aggregate}
    rounds = []
    with tempfile.TemporaryDirectory(prefix="r42-dagger-") as temporary:
        directory = Path(temporary)
        for round_index in range(1, args.rounds + 1):
            discovered = collect_dagger_rows(
                model, contract, scenes["splits"]["train"], directory=directory,
                round_index=round_index,
            )
            added = 0
            for row in discovered:
                key = sha256(row[0].tobytes()).digest()
                if key not in seen:
                    seen.add(key)
                    aggregate.append(row)
                    added += 1
            observations, labels = _arrays(aggregate + anchors * (anchor_repeats - 1))
            history = base.imitation_train(
                model, observations, labels, device=device,
                epochs=5 if round_index == 1 else 3,
                seed=args.seed + round_index,
            )
            rounds.append({"round": round_index, "discovered": len(discovered),
                           "added": added, "aggregate": len(aggregate),
                           "final": history[-1]})
    observations, labels = _arrays(aggregate + anchors * (anchor_repeats - 1))
    bc_path = output / "actor_bc.npz"
    base.export_actor(model, contract, bc_path, metadata={
        "finetune_version": VERSION,
        "parent_actor_sha256": contract.sha256,
        "scene_manifest_sha256": base.file_hash(scene_path),
        "dagger_rows": len(aggregate),
        "fresh_ppo_joint_steps": 0,
        "runtime_action_override": False,
    })
    bc_carried = base.evaluate_carried(bc_path, scenes)
    checkpoints = []

    def save(step, state):
        model.load_state_dict(state)
        path = output / f"actor_ppo_{step:07d}.npz"
        actor_sha = base.export_actor(model, contract, path, metadata={
            "finetune_version": VERSION,
            "parent_actor_sha256": contract.sha256,
            "scene_manifest_sha256": base.file_hash(scene_path),
            "dagger_rows": len(aggregate),
            "fresh_ppo_joint_steps": int(step),
            "ppo_actor_specific_reward": True,
            "runtime_action_override": False,
        })
        checkpoints.append({"step": int(step), "path": path.name,
                            "sha256": actor_sha,
                            "carried": base.evaluate_carried(path, scenes)})

    updates = base.ppo_finetune(
        model, scenes, contract, device=device, joint_steps=args.ppo_steps,
        seed=args.seed + 100, checkpoint_callback=save,
        reference_observations=observations, reference_labels=labels,
    )
    if not checkpoints:
        save(args.ppo_steps, model.state_dict())
    selected = max(checkpoints, key=lambda item: (
        item["carried"]["rate"], -item["step"]
    ))
    final = output / "actor.npz"
    final.write_bytes((output / selected["path"]).read_bytes())
    report = {
        "version": VERSION,
        "passed": selected["carried"]["rate"] >= .90,
        "selected": {**selected, "path": final.name,
                     "sha256": base.file_hash(final)},
        "parent": {"actor_sha256": contract.sha256,
                   "actor_path": str(actor_parent)},
        "training": {
            "teacher_scope": "public_state_training_only",
            "dagger_rounds": rounds,
            "anchor_rows": len(anchors),
            "anchor_repeats": anchor_repeats,
            "aggregate_rows": len(aggregate),
            "weighted_training_rows": len(observations),
            "fresh_ppo_joint_steps": args.ppo_steps,
            "bc_carried": bc_carried,
            "ppo_updates": updates,
            "runtime_action_override": False,
        },
        "action_authority": {
            "policy_action_equals_submitted_action": True,
            "runtime_overrides": 0,
        },
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
    }
    base.write_json(output / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
