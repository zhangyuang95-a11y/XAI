"""High-capacity neural distillation of the full r4.5 public-state policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from backend.training.warehouse_r45_iterative_dagger import _collect
from backend.training.warehouse_r45_targeted_distill import _rows
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor


VERSION = "warehouse-r45-wide-full-policy-distillation.v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-parent", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    source = Path(args.actor_parent).resolve(); parent = NumPyNativeActor(source)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    collector = NativeActorCritic(
        parent.obs_dim, parent.state_dim, parent.hidden
    ).to(device)
    collector.actor.load_state_dict({
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in parent.weights.items()
    })
    development_scenes = (
        list(manifest["splits"]["conflict_validation"])
        + list(manifest["splits"]["question_bank"])
    )
    rollout_rows, disagreements, rollout_unique = _collect(
        collector, parent, development_scenes, 45
    )
    targeted_rows, targeted_unique = _rows(
        parent, development_scenes
    )
    rows = rollout_rows + targeted_rows
    x = np.stack([row[0] for row in rows]).astype(np.float32)
    y = np.asarray([row[1] for row in rows], dtype=np.int64)
    model = NativeActorCritic(
        parent.obs_dim, parent.state_dim, int(args.hidden)
    ).to(device)
    if parent.hidden == int(args.hidden):
        model.actor.load_state_dict({
            key: torch.as_tensor(np.array(value, copy=True), device=device)
            for key, value in parent.weights.items()
        })
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4, eps=1e-5)
    rng = np.random.default_rng(458500); history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x)); total = 0.0; correct = count = 0
        for start in range(0, len(order), 768):
            ids = order[start:start + 768]
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
    metadata = deepcopy(parent.metadata)
    metadata.update({"hidden": int(args.hidden),
                     "r45_wide_distillation": VERSION,
                     "r45_wide_parent_sha256": parent.sha256,
                     "runtime_action_override": False})
    actor_path = output / "actor.npz"; model.export_npz(actor_path, metadata)
    report = {"version": VERSION, "parent": str(source),
              "parent_sha256": parent.sha256, "actor": str(actor_path),
              "actor_sha256": sha256(actor_path.read_bytes()).hexdigest(),
              "hidden": int(args.hidden), "formal_play_scenes_used": False,
              "training_scenes": len(development_scenes),
              "rollout_unique": rollout_unique,
              "rollout_disagreements": disagreements,
              "targeted_unique": targeted_unique, "weighted_rows": len(rows),
              "history": history, "runtime_action_override": False}
    (output / "wide_distillation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
