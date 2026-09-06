"""Auditable CPU execution of the frozen six-action kitchen Actor.

This module deliberately imports neither torch nor a scripted controller.
The policy performs the learned MLP and argmax, with no runtime reranking.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from core.policy_contracts import ActionDistribution

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT")
ACTOR_SCHEMA = "cooperative_kitchen_numpy_actor_v1"
CHECKPOINT_SCHEMA = "cooperative_kitchen_mappo_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward(values, layers):
    z = np.asarray(values, dtype=np.float32)
    for index, (weight, bias) in enumerate(layers):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = z @ weight.T + bias
        if index < len(layers) - 1:
            z = np.tanh(z)
    if not np.isfinite(z).all():
        raise FloatingPointError("Non-finite kitchen Actor output")
    return z


def probabilities(logits):
    values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


class NumpyKitchenPolicy:
    kind = "neural"
    action_names = ACTIONS

    def __init__(self, artifact):
        self.path = Path(artifact)
        with np.load(self.path, allow_pickle=False) as stored:
            self.metadata = json.loads(str(stored["metadata"]))
            if self.metadata.get("schema") != ACTOR_SCHEMA:
                raise ValueError("Not a kitchen NumPy Actor")
            self.layers = [(np.array(stored[f"weight_{i}"], copy=True),
                            np.array(stored[f"bias_{i}"], copy=True)) for i in range(3)]
        self.feature_names = tuple(self.metadata["feature_names"])
        from env.cooperative_kitchen import OBSERVATION_FEATURES
        if self.feature_names != tuple(OBSERVATION_FEATURES):
            raise ValueError("Kitchen Actor observation vocabulary/order mismatch")
        sizes = (len(self.feature_names), 128, 128, len(ACTIONS))
        for i, (weight, bias) in enumerate(self.layers):
            if weight.shape != (sizes[i + 1], sizes[i]) or bias.shape != (sizes[i + 1],):
                raise ValueError("Kitchen Actor architecture mismatch")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("Non-finite kitchen weights")
        if tuple(self.metadata.get("actions", [])) != ACTIONS:
            raise ValueError("Kitchen action ordering mismatch")
        self.checkpoint_id = self.metadata["checkpoint_sha256"]
        self.artifact_sha256 = digest(self.path)
        self.trained = int(self.metadata.get("joint_steps", 0)) > 0

    def logits(self, observations):
        values = np.asarray(observations, dtype=np.float32)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("Kitchen observation width mismatch")
        return forward(values, self.layers)

    def act(self, observations, *, deterministic=True, rng=None):
        if not deterministic and rng is None:
            raise ValueError("Stochastic evaluation requires an explicit isolated RNG")
        ids = [actor for actor in ("human", "ai") if actor in observations]
        if not ids:
            raise ValueError("No kitchen actor observations supplied")
        logits = self.logits(np.asarray([observations[actor] for actor in ids]))
        probs = probabilities(logits)
        actions, distributions = {}, {}
        for i, actor in enumerate(ids):
            index = int(probs[i].argmax()) if deterministic else int(rng.choice(len(ACTIONS), p=probs[i]))
            actions[actor] = ACTIONS[index]
            contract = ActionDistribution(actor, ACTIONS, tuple(map(float, probs[i])),
                                          tuple(map(float, logits[i])), (1.0,) * 6, actions[actor])
            distributions[actor] = {**asdict(contract), "chosen_action": actions[actor],
                                    "checkpoint_id": self.checkpoint_id}
        return actions, distributions


def export_checkpoint(checkpoint, output):
    """Export exact float32 learned parameters; torch is training-only."""
    import torch
    source = Path(checkpoint)
    data = torch.load(source, map_location="cpu", weights_only=False)
    if data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Not a kitchen training checkpoint")
    metadata = {"schema": ACTOR_SCHEMA, "checkpoint_sha256": digest(source),
                "feature_names": data["feature_names"], "actions": ACTIONS,
                "seed": data["seed"], "joint_steps": data["joint_steps"],
                "config": data["config"], "environment_signature": data["environment_signature"],
                "hidden_units": [128, 128], "activation": "tanh", "runtime_controller": "direct_actor_argmax",
                "rcpd_feedback": False}
    arrays = {"metadata": np.asarray(json.dumps(metadata))}
    for index, layer in enumerate((0, 2, 4)):
        arrays[f"weight_{index}"] = data["actor"][f"network.{layer}.weight"].numpy()
        arrays[f"bias_{index}"] = data["actor"][f"network.{layer}.bias"].numpy()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    target.with_suffix(".tmp").replace(target)
    return NumpyKitchenPolicy(target)
