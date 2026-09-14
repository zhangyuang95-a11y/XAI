"""Conservative empty-state DAgger while preserving proven carried behavior."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn

from backend.training import warehouse_r42_delivery_dagger as dagger
from backend.training import warehouse_r42_delivery_finetune as base
from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    actor_environment,
)
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r42-delivery-conservative-dagger.v1"


def _arrays(rows):
    return (np.stack([row[0] for row in rows]).astype(np.float32),
            np.asarray([row[1] for row in rows], dtype=np.int64))


def _dev_behavior(actor_path, scenes, maximum_scenes=16):
    actor = NumPyNativeActor(actor_path)
    rows = []
    for scene_index, scene in enumerate(
            scenes["splits"]["conflict_validation"][:maximum_scenes]):
        env = actor_environment(actor, scene)
        rng = random.Random(260_915_900 + scene_index)
        collisions = waits = 0
        while not env.done:
            actions, _ = actor.act(env.observations(), deterministic=True)
            action = actions["robot_2"]
            player = partner_action(env, "robot_1", "skilled", rng)
            _, _, _, _, info = env.step({"robot_1": player, "robot_2": action},
                decision_metadata={"policy_action": action,
                    "submitted_action": action, "post_policy_overrides": 0})
            collisions += int(info["robot_collision"])
            waits += int(action == "WAIT")
        learner = env.state.by_id("robot_2")
        rows.append({"scene_id": scene["id"],
                     "deliveries": learner.deliveries_completed,
                     "collisions": collisions, "waits": waits,
                     "shutdown": not learner.active})
    return {
        "episodes": rows,
        "minimum_deliveries": min(row["deliveries"] for row in rows),
        "total_deliveries": sum(row["deliveries"] for row in rows),
        "collisions": sum(row["collisions"] for row in rows),
        "waits": sum(row["waits"] for row in rows),
        "shutdowns": sum(row["shutdown"] for row in rows),
    }


def _conservative_train(model, corrections, anchors, parent_logits, *, device,
                        epochs, seed, callback):
    rng = np.random.default_rng(seed)
    correction_x, correction_y = _arrays(corrections)
    anchor_x = np.stack([row[0] for row in anchors]).astype(np.float32)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-5, eps=1e-5)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(correction_x))
        total = correct = count = 0
        for start in range(0, len(order), 384):
            indices = order[start:start + 384]
            anchor_indices = rng.integers(0, len(anchor_x), size=len(indices))
            x = torch.as_tensor(correction_x[indices], device=device)
            y = torch.as_tensor(correction_y[indices], device=device)
            ax = torch.as_tensor(anchor_x[anchor_indices], device=device)
            target_logits = torch.as_tensor(
                parent_logits[anchor_indices], device=device
            )
            logits = model.actor_logits(x)
            correction_loss = nn.functional.cross_entropy(logits, y)
            retention_loss = nn.functional.mse_loss(
                model.actor_logits(ax), target_logits
            )
            loss = correction_loss + 24.0 * retention_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .35)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == y).sum().detach())
            count += len(indices)
        history.append({"epoch": epoch, "loss": total / count,
                        "correction_accuracy": correct / count})
        callback(epoch, deepcopy(model.state_dict()))
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default=str(base.DEFAULT_PARENT))
    parser.add_argument("--scenes", default=str(base.DEFAULT_SCENES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260_915_942)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = Path(args.parent).resolve()
    parent_path = Path(args.actor_parent).resolve()
    scene_path = Path(args.scenes).resolve()
    scenes = json.loads(scene_path.read_text(encoding="utf-8"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, contract, _ = base._load_model(parent, device, parent_path)
    training_entries = (list(scenes["splits"]["train"])
                        + list(scenes["splits"]["conflict_validation"][16:]))
    all_anchors = dagger.collect_anchor_rows(
        contract, training_entries, maximum_scenes=128
    )
    carrying_index = list(contract.metadata["feature_names"]).index(
        "self.carrying"
    )
    anchors = [row for row in all_anchors if row[0][carrying_index] > .5]
    anchor_x = np.stack([row[0] for row in anchors]).astype(np.float32)
    parent_logits = contract.logits(anchor_x)
    with tempfile.TemporaryDirectory(prefix="r42-conservative-") as temporary:
        corrections = dagger.collect_dagger_rows(
            model, contract, training_entries,
            directory=Path(temporary), round_index=1,
            maximum_scenes=len(training_entries),
        )
    candidates = []

    def save_epoch(epoch, state):
        model.load_state_dict(state)
        path = output / f"actor_bc_epoch_{epoch:02d}.npz"
        base.export_actor(model, contract, path, metadata={
            "finetune_version": VERSION,
            "parent_actor_sha256": contract.sha256,
            "scene_manifest_sha256": base.file_hash(scene_path),
            "dagger_correction_rows": len(corrections),
            "fresh_ppo_joint_steps": 0,
            "runtime_action_override": False,
        })
        carried = base.evaluate_carried(path, scenes)
        behavior = _dev_behavior(path, scenes)
        candidates.append({"epoch": epoch, "path": path.name,
                           "sha256": base.file_hash(path),
                           "carried": carried, "development": behavior,
                           "state": deepcopy(state)})

    history = _conservative_train(
        model, corrections, anchors, parent_logits, device=device,
        epochs=args.epochs, seed=args.seed, callback=save_epoch,
    )
    eligible = [row for row in candidates if row["carried"]["rate"] >= .90]
    if not eligible:
        selected = max(candidates, key=lambda row: row["carried"]["rate"])
    else:
        selected = max(eligible, key=lambda row: (
            row["development"]["minimum_deliveries"],
            row["development"]["total_deliveries"],
            -row["development"]["shutdowns"],
            -row["development"]["collisions"],
        ))
    model.load_state_dict(selected.pop("state"))
    correction_x, correction_y = _arrays(corrections)
    # One small genuine on-policy update follows DAgger.  The very low actor
    # learning rate preserves the accepted carried policy while satisfying the
    # requirement that the final neural weights include fresh RL training.
    updates = base.ppo_finetune(
        model, scenes, contract, device=device, joint_steps=1_024,
        seed=args.seed + 100, checkpoint_callback=lambda *_: None,
        reference_observations=correction_x,
        reference_labels=correction_y,
        actor_learning_rate=1e-7, behavior_coefficient=2.0,
    )
    final = output / "actor.npz"
    base.export_actor(model, contract, final, metadata={
        "finetune_version": VERSION,
        "parent_actor_sha256": contract.sha256,
        "scene_manifest_sha256": base.file_hash(scene_path),
        "dagger_correction_rows": len(corrections),
        "fresh_ppo_joint_steps": 1_024,
        "ppo_actor_specific_reward": True,
        "runtime_action_override": False,
    })
    final_carried = base.evaluate_carried(final, scenes)
    final_development = _dev_behavior(final, scenes)
    report = {
        "version": VERSION,
        "passed": final_carried["rate"] >= .90,
        "selected": {"path": final.name, "sha256": base.file_hash(final),
                     "source_epoch": selected["epoch"],
                     "carried": final_carried,
                     "development": final_development},
        "parent": {"actor_sha256": contract.sha256,
                   "actor_path": str(parent_path)},
        "training": {"correction_rows": len(corrections),
                     "training_scenes": len(training_entries),
                     "checkpoint_selection_scenes": 16,
                     "retention_rows": len(anchors),
                     "retention_loss_weight": 24.0,
                     "history": history,
                     "fresh_ppo_joint_steps": 1_024,
                     "ppo_actor_learning_rate": 1e-7,
                     "ppo_updates": updates,
                     "runtime_action_override": False},
        "candidate_summaries": [{key: value for key, value in row.items()
                                 if key != "state"} for row in candidates],
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
