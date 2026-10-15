"""Targeted public-state distillation for carried delivery and charger budgets."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from backend.training.warehouse_r45_adaptation import (
    _coordination_teacher_action, actor_environment,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor


VERSION = "warehouse-r45-targeted-public-state-distillation.v1"


def _load(actor, device):
    model = NativeActorCritic(actor.obs_dim, actor.state_dim, actor.hidden).to(device)
    model.actor.load_state_dict({
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in actor.weights.items()
    })
    return model


def _add(rows, seen, env, weight):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    digest = sha256(observation.tobytes()).digest()
    if digest in seen:
        return
    seen.add(digest)
    label = ACTIONS.index(_coordination_teacher_action(env))
    rows.extend((observation.copy(), label) for _ in range(weight))


def _rows(contract, scenes):
    rows = []; seen = set()
    for scene in scenes:
        template = actor_environment(contract, scene)
        passable = sorted(template.layout.passable_positions)
        tasks = [task.task_id for task in template.state.tasks if task.active]
        for task_id in tasks:
            for position in passable:
                for battery in (24.0, 39.0, 48.0, 57.0, 66.0, 78.0, 93.0):
                    env = actor_environment(contract, scene)
                    state = env.get_state()
                    learner = state.by_id("robot_2")
                    other = state.by_id("robot_1")
                    if position == other.position:
                        continue
                    for task in state.tasks:
                        if task.task_id == task_id:
                            task.status = "carried"
                            task.carrier_agent_id = "robot_2"
                        elif task.carrier_agent_id == "robot_2":
                            task.status = "available"; task.carrier_agent_id = None
                    learner.position = position
                    learner.battery = battery
                    learner.active = True
                    learner.carrying_task_id = task_id
                    learner.last_executed_action = "WAIT"
                    learner.last_action = "WAIT"
                    state.last_robot_collision_event = False
                    env.set_state(state)
                    _add(rows, seen, env, weight=6)
        # Empty-agent states teach candidate pickup selection and the charger
        # decision across the whole map, rather than only along one rollout.
        for position in passable:
            for battery in (24.0, 39.0, 48.0, 57.0, 66.0, 78.0, 93.0):
                env = actor_environment(contract, scene)
                state = env.get_state(); learner = state.by_id("robot_2")
                if position == state.by_id("robot_1").position:
                    continue
                learner.position = position; learner.battery = battery
                learner.active = True; learner.carrying_task_id = None
                learner.last_executed_action = "WAIT"; learner.last_action = "WAIT"
                state.last_robot_collision_event = False
                env.set_state(state)
                _add(rows, seen, env, weight=6)
        # Dense charger thresholds for both empty and loaded work cycles.
        for carrying in (None, *tasks):
            for battery in range(1, 101, 2):
                env = actor_environment(contract, scene)
                state = env.get_state(); learner = state.by_id("robot_2")
                learner.position = env.layout.charger_position
                learner.battery = float(battery); learner.active = True
                learner.carrying_task_id = carrying
                if carrying:
                    task = state.task_by_id(carrying)
                    task.status = "carried"; task.carrier_agent_id = "robot_2"
                state.last_robot_collision_event = False
                env.set_state(state)
                _add(rows, seen, env, weight=12)
    return rows, len(seen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-parent", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    source = Path(args.actor_parent).resolve(); actor = NumPyNativeActor(source)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Development training split only; formal play scenes remain untouched.
    scenes = list(manifest["splits"]["train"][:64])
    rows, unique = _rows(actor, scenes)
    x = np.stack([row[0] for row in rows]).astype(np.float32)
    y = np.asarray([row[1] for row in rows], dtype=np.int64)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _load(actor, device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-5, eps=1e-5)
    rng = np.random.default_rng(457500); history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x)); total = 0.0; correct = count = 0
        for start in range(0, len(order), 512):
            ids = order[start:start + 512]
            xb = torch.as_tensor(x[ids], device=device)
            yb = torch.as_tensor(y[ids], device=device)
            logits = model.actor_logits(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            correct += int((logits.argmax(-1) == yb).sum().detach())
            count += len(ids)
        history.append({"epoch": epoch, "loss": total / count,
                        "accuracy": correct / count})
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    metadata = deepcopy(actor.metadata)
    metadata.update({"r45_targeted_distillation": VERSION,
                     "r45_targeted_parent_sha256": actor.sha256,
                     "runtime_action_override": False})
    destination = output / "actor.npz"; model.export_npz(destination, metadata)
    report = {"version": VERSION, "parent": str(source),
              "parent_sha256": actor.sha256, "actor": str(destination),
              "actor_sha256": sha256(destination.read_bytes()).hexdigest(),
              "formal_play_scenes_used": False, "training_scenes": len(scenes),
              "unique_states": unique, "weighted_rows": len(rows),
              "history": history, "runtime_action_override": False}
    (output / "targeted_distillation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
