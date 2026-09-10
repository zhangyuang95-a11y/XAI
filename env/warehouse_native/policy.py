"""Native shared neural Actor and centralized Critic, with no rule controller.

All five action logits are unconstrained outputs of a fresh two-layer MLP.
The Actor sees only the observation supplied by the native environment. The
Critic additionally receives the acting role, solely during training.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
NATIVE_ACTOR_FORMAT = "warehouse_native_actor_v1"
NATIVE_POLICY_VERSION = "warehouse_native_plain_mlp_v1"


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, outputs),
    )


class NativeActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        if min(obs_dim, state_dim, hidden) <= 0:
            raise ValueError("Network dimensions must be positive")
        self.obs_dim, self.state_dim, self.hidden = int(obs_dim), int(state_dim), int(hidden)
        self.actor = _mlp(self.obs_dim, self.hidden, len(ACTIONS))
        self.critic = _mlp(self.state_dim + 2, self.hidden, 1)

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 1 or observations.shape[-1] != self.obs_dim:
            raise ValueError("Actor observation width differs from its native contract")
        return self.actor(observations)

    def values(self, states: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        if states.ndim < 1 or states.shape[-1] != self.state_dim:
            raise ValueError("Critic state width differs from its native contract")
        if tuple(role_ids.shape) != tuple(states.shape[:-1]):
            raise ValueError("Each centralized state must have one acting role")
        roles = torch.nn.functional.one_hot(role_ids.to(torch.long), num_classes=2)
        return self.critic(torch.cat((states, roles.to(states)), dim=-1)).squeeze(-1)

    def export_npz(self, path: str | Path, metadata: Mapping[str, Any]) -> Path:
        """Atomically export the exact Actor tensors; never select/rewrite actions."""
        contract = {
            "format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
            "obs_dim": self.obs_dim, "state_dim": self.state_dim,
            "hidden": self.hidden, "actions": list(ACTIONS),
            "architecture": "two_hidden_layer_tanh", "action_masks": False,
            "runtime_action_override": False,
        }
        for key in contract.keys() & metadata.keys():
            if metadata[key] != contract[key]:
                raise ValueError("Cannot override native Actor contract: " + key)
        payload = {**dict(metadata), **contract}
        tensors = {name: value.detach().cpu().numpy().astype(np.float32, copy=True)
                   for name, value in self.actor.state_dict().items()}
        if not all(np.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Cannot export non-finite neural weights")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata_json=json.dumps(payload, sort_keys=True), **tensors)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination


class NumPyNativeActor:
    """Exact exported MLP inference; all five actions remain available."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.artifact_sha256 = sha256(self.path.read_bytes()).hexdigest()
        self.sha256 = self.artifact_sha256
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata_json"].item()))
            self.weights = {name: archive[name].astype(np.float32, copy=True)
                            for name in archive.files if name != "metadata_json"}
        expected = {
            "format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
            "actions": list(ACTIONS), "architecture": "two_hidden_layer_tanh",
            "action_masks": False, "runtime_action_override": False,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError("Incompatible native Actor metadata: " + key)
        self.obs_dim = int(self.metadata["obs_dim"])
        self.state_dim = int(self.metadata["state_dim"])
        self.hidden = int(self.metadata["hidden"])
        if min(self.obs_dim, self.state_dim, self.hidden) <= 0:
            raise ValueError("Invalid native Actor dimensions")
        shapes = {
            "0.weight": (self.hidden, self.obs_dim), "0.bias": (self.hidden,),
            "2.weight": (self.hidden, self.hidden), "2.bias": (self.hidden,),
            "4.weight": (len(ACTIONS), self.hidden), "4.bias": (len(ACTIONS),),
        }
        if set(self.weights) != set(shapes):
            raise ValueError("Native Actor export must contain only its six MLP tensors")
        for name, shape in shapes.items():
            value = self.weights[name]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid native Actor tensor: " + name)
            value.setflags(write=False)

    def logits(self, observations: Any) -> np.ndarray:
        value = np.asarray(observations, dtype=np.float32)
        if value.ndim < 1 or value.shape[-1] != self.obs_dim or not np.isfinite(value).all():
            raise ValueError("Invalid native Actor observations")
        # Apple's Accelerate BLAS may leave stale floating-point status flags
        # on an otherwise finite matmul. Check every actual result instead;
        # genuine numerical overflow must never reach action selection.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for index in (0, 2, 4):
                value = value @ self.weights[f"{index}.weight"].T + self.weights[f"{index}.bias"]
                if not np.isfinite(value).all():
                    raise FloatingPointError("Native neural inference produced non-finite logits")
                if index != 4:
                    value = np.tanh(value)
        return np.asarray(value, dtype=np.float32)

    def act(self, observations: Mapping[str, Any], deterministic: bool = True,
            rng: np.random.Generator | None = None) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        if len(observations) != 2:
            raise ValueError("The shared native Actor requires two agent observations")
        if not deterministic and rng is None:
            raise ValueError("Stochastic inference requires a caller-owned NumPy RNG")
        agents = sorted(observations)
        values = self.logits(np.stack([observations[agent] for agent in agents]))
        probabilities = np.exp(values - values.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        actions, distributions = {}, {}
        for row, agent in enumerate(agents):
            index = (int(np.argmax(probabilities[row])) if deterministic
                     else int(rng.choice(len(ACTIONS), p=probabilities[row].astype(float)
                                         / probabilities[row].sum(dtype=float))))
            actions[agent] = ACTIONS[index]
            distributions[agent] = probabilities[row].copy()
        return actions, distributions


# ASCII spelling is convenient in command-line callers and manifests.
NumpyNativeActor = NumPyNativeActor
