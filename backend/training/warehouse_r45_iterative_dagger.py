"""Iterative on-policy correction for r4.5 charging and collision states."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from backend.training import warehouse_r42_delivery_evaluation as evaluation
from backend.training.warehouse_r45_adaptation import (
    _coordination_teacher_action, actor_environment,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor


VERSION = "warehouse-r45-iterative-on-policy-dagger.v1"


def _model(actor: NumPyNativeActor, device):
    model = NativeActorCritic(actor.obs_dim, actor.state_dim, actor.hidden).to(device)
    state = {
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in actor.weights.items()
    }
    model.actor.load_state_dict(state)
    return model


def _collect(model, contract, scenes, round_index):
    device = next(model.parameters()).device
    rows = []
    seen = set()
    disagreements = 0
    for scene_index, scene in enumerate(scenes):
        for profile_index, profile in enumerate(evaluation.PROFILES):
            env = actor_environment(contract, scene)
            rng = random.Random(455000 + round_index * 10000
                                + scene_index * 10 + profile_index)
            while not env.done:
                observation = env.observations()["robot_2"].astype(
                    np.float32, copy=True
                )
                with torch.no_grad():
                    logits = model.actor_logits(torch.as_tensor(
                        observation, device=device
                    ).unsqueeze(0))
                action = ACTIONS[int(logits.argmax(-1).item())]
                teacher = _coordination_teacher_action(env)
                digest = sha256(observation.tobytes()).digest()
                if digest not in seen:
                    seen.add(digest)
                    repeat = 32 if action != teacher else 1
                    rows.extend((observation.copy(), ACTIONS.index(teacher))
                                for _ in range(repeat))
                    disagreements += int(action != teacher)
                human = ("WAIT" if profile == "wait" else partner_action(
                    env, "robot_1", profile, rng
                ))
                env.step({"robot_1": human, "robot_2": action})
    return rows, disagreements, len(seen)


def _train(model, rows, device, seed):
    x = np.stack([row[0] for row in rows]).astype(np.float32)
    y = np.asarray([row[1] for row in rows], dtype=np.int64)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-5, eps=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, 9):
        order = rng.permutation(len(x)); correct = count = 0; total = 0.0
        for start in range(0, len(order), 384):
            ids = order[start:start + 384]
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
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-parent", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    source = Path(args.actor_parent).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    actor = NumPyNativeActor(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Training split only. Formal play scenes remain an untouched acceptance set.
    scenes = list(manifest["splits"][args.split])
    if args.split == "play" and len(scenes) == 7:
        scenes = scenes[1:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _model(actor, device)
    rounds = []
    for round_index in range(1, args.rounds + 1):
        rows, disagreements, unique = _collect(
            model, actor, scenes, round_index
        )
        history = _train(model, rows, device, 456000 + round_index)
        candidate = output / f"actor_round_{round_index:02d}.npz"
        metadata = deepcopy(actor.metadata)
        metadata.update({
            "r45_iterative_dagger_version": VERSION,
            "r45_iterative_dagger_parent_sha256": actor.sha256,
            "r45_iterative_dagger_round": round_index,
            "runtime_action_override": False,
        })
        model.export_npz(candidate, metadata)
        actor = NumPyNativeActor(candidate)
        rounds.append({"round": round_index, "unique_states": unique,
                       "disagreements": disagreements, "weighted_rows": len(rows),
                       "history": history, "actor_sha256": actor.sha256})
    final = output / "actor.npz"; final.write_bytes(candidate.read_bytes())
    report = {
        "version": VERSION,
        "parent": str(source),
        "parent_sha256": NumPyNativeActor(source).sha256,
        "actor": str(final), "actor_sha256": sha256(final.read_bytes()).hexdigest(),
        "training_split": args.split,
        "training_split_scenes": len(scenes),
        "formal_play_scenes_used": args.split == "play",
        "rounds": rounds, "runtime_action_override": False,
    }
    (output / "iterative_dagger_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
