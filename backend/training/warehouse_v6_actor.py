"""Independent Warehouse v6 neural Actors and exact frozen NumPy inference.

The learned sparse tree is a branch of each Actor. The learned, leaf-bounded
256-unit residual is also part of that Actor. Only their combined logits,
after the environment's physical action mask, select actions at runtime.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from core.policy_contracts import ActionDistribution
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    FEATURE_DIM, FEATURE_SCHEMA_SHA256, OBSERVATION_DIM, OBSERVATION_VERSION,
    encode_policy_features, encode_public_observation,
)


MODEL_VERSION = "warehouse-dual-actor.v6.0"
TREE_DEPTH = 4
TREE_LEAVES = 1 << TREE_DEPTH
TREE_NODES = TREE_LEAVES - 1
MAX_SPLIT_FEATURES = 3
RESIDUAL_WIDTH = 256
MIN_RESIDUAL_BOUND = 0.03
TREE_GRADIENT_SCALE = 4.0
MASK_GRADIENT_TEMPERATURE = 0.05
_TENSOR_NAMES = (
    "tree.split.weight", "tree.split.bias", "tree.leaf_logits",
    "residual.0.weight", "residual.0.bias", "residual.2.weight",
    "residual.2.bias", "residual.4.weight", "residual.4.bias",
    "leaf_bound_logits",
)


def policy_features_torch(obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (obs.shape[-1:] != (OBSERVATION_DIM,)
            or mask.shape != obs.shape[:-1] + (len(ACTIONS),)):
        raise ValueError("Warehouse v6 Actor observation or mask dimension mismatch")
    return torch.cat((obs, mask), dim=-1)


class SparseHardTree(nn.Module):
    """Sixteen-leaf trainable tree; hard forward, straight-through gradient."""

    def __init__(self):
        super().__init__()
        self.split = nn.Linear(FEATURE_DIM, TREE_NODES)
        self.leaf_logits = nn.Parameter(torch.zeros(TREE_LEAVES, len(ACTIONS)))
        nn.init.normal_(self.split.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.split.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.025)

    def effective_weights(self) -> torch.Tensor:
        weights = self.split.weight
        selected = torch.argsort(-weights.abs(), dim=-1, stable=True)[
            :, :MAX_SPLIT_FEATURES]
        hard = torch.zeros_like(weights).scatter_(-1, selected, 1.0)
        soft = torch.softmax(weights.abs() / MASK_GRADIENT_TEMPERATURE,
                             dim=-1) * MAX_SPLIT_FEATURES
        return weights * (hard + soft - soft.detach())

    def path(self, encoded: torch.Tensor) -> torch.Tensor:
        values = encoded.reshape(-1, FEATURE_DIM)
        split = nn.functional.linear(values, self.effective_weights(), self.split.bias)
        soft_left = torch.sigmoid(-TREE_GRADIENT_SCALE * split)
        hard_left = (split <= 0).to(split.dtype)
        left = hard_left + soft_left - soft_left.detach()
        path = torch.ones((len(values), 1), dtype=values.dtype, device=values.device)
        for depth in range(TREE_DEPTH):
            begin, end = (1 << depth) - 1, (1 << (depth + 1)) - 1
            gate = left[:, begin:end]
            path = torch.stack((path * gate, path * (1.0 - gate)), dim=-1)
            path = path.reshape(len(values), -1)
        return path.reshape(encoded.shape[:-1] + (TREE_LEAVES,))

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path(encoded)
        return path @ self.leaf_logits, path


class RoleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree = SparseHardTree()
        self.residual = nn.Sequential(
            nn.Linear(FEATURE_DIM, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, len(ACTIONS)),
        )
        self.leaf_bound_logits = nn.Parameter(torch.full((TREE_LEAVES,), -1.0))
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual[-1].bias)

    def bounds(self) -> torch.Tensor:
        return MIN_RESIDUAL_BOUND + (1.0 - MIN_RESIDUAL_BOUND) * torch.sigmoid(
            self.leaf_bound_logits)

    def components(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tree_logits, path = self.tree(encoded)
        bound = path @ self.bounds()
        bounded_residual = bound.unsqueeze(-1) * torch.tanh(self.residual(encoded))
        return tree_logits, bounded_residual, bound

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        tree_logits, residual, _ = self.components(encoded)
        return tree_logits + residual


class WarehouseV6Actor(nn.Module):
    """Two independent Actors with centralized reward and shutdown critics."""

    def __init__(self):
        super().__init__()
        self.actors = nn.ModuleDict({role: RoleActor() for role in AGENTS})
        self.reward_critic = nn.Sequential(
            nn.Linear(2 * OBSERVATION_DIM, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, len(AGENTS)),
        )
        self.cost_critic = nn.Sequential(
            nn.Linear(2 * OBSERVATION_DIM, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, 1),
        )

    def outputs_encoded(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (encoded.shape[-2:] != (len(AGENTS), FEATURE_DIM)
                or mask.shape != encoded.shape[:-1] + (len(ACTIONS),)):
            raise ValueError("Warehouse v6 Actor expects two role rows")
        logits = torch.stack([self.actors[role](encoded[..., index, :])
                              for index, role in enumerate(AGENTS)], dim=-2)
        return logits.masked_fill(mask <= 0.5, -1.0e9)

    def outputs(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.outputs_encoded(policy_features_torch(obs, mask), mask)

    def role_logits(self, role: str, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if role not in AGENTS:
            raise ValueError("unknown Warehouse v6 role")
        encoded = policy_features_torch(obs, mask)
        return self.actors[role](encoded).masked_fill(mask <= 0.5, -1.0e9)


def _role_arrays(actor: RoleActor) -> dict[str, np.ndarray]:
    state = actor.state_dict()
    return {name: state[name].detach().cpu().numpy().astype(np.float32).copy()
            for name in _TENSOR_NAMES}


def role_sha256(weights: Mapping[str, np.ndarray]) -> str:
    if set(weights) != set(_TENSOR_NAMES):
        raise ValueError("Warehouse v6 role tensor keys mismatch")
    digest = sha256()
    for name in _TENSOR_NAMES:
        value = np.ascontiguousarray(weights[name], dtype="<f4")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def role_hashes(model: WarehouseV6Actor) -> dict[str, str]:
    return {role: role_sha256(_role_arrays(model.actors[role])) for role in AGENTS}


class FrozenWarehouseV6Role:
    def __init__(self, weights: Mapping[str, np.ndarray]):
        self.weights = {name: np.asarray(value, dtype=np.float32)
                        for name, value in weights.items()}
        expected = {
            "tree.split.weight": (TREE_NODES, FEATURE_DIM),
            "tree.split.bias": (TREE_NODES,),
            "tree.leaf_logits": (TREE_LEAVES, len(ACTIONS)),
            "residual.0.weight": (RESIDUAL_WIDTH, FEATURE_DIM),
            "residual.0.bias": (RESIDUAL_WIDTH,),
            "residual.2.weight": (RESIDUAL_WIDTH, RESIDUAL_WIDTH),
            "residual.2.bias": (RESIDUAL_WIDTH,),
            "residual.4.weight": (len(ACTIONS), RESIDUAL_WIDTH),
            "residual.4.bias": (len(ACTIONS),),
            "leaf_bound_logits": (TREE_LEAVES,),
        }
        if (set(self.weights) != set(expected)
                or any(self.weights[name].shape != shape
                       or not np.isfinite(self.weights[name]).all()
                       for name, shape in expected.items())):
            raise ValueError("Warehouse v6 role tensor shape or value mismatch")
        self.role_sha256 = role_sha256(self.weights)

    def leaf(self, encoded: np.ndarray) -> np.ndarray:
        data = np.asarray(encoded, dtype=np.float32)
        if data.shape[-1:] != (FEATURE_DIM,):
            raise ValueError("Warehouse v6 role feature dimension mismatch")
        flat = data.reshape(-1, FEATURE_DIM)
        weight = self.weights["tree.split.weight"]
        selected = np.argsort(-np.abs(weight), axis=-1, kind="stable")[:, :MAX_SPLIT_FEATURES]
        sparse = np.zeros_like(weight)
        np.put_along_axis(sparse, selected, 1.0, axis=-1)
        split = flat @ (weight * sparse).T + self.weights["tree.split.bias"]
        node = np.zeros(len(flat), dtype=np.int64)
        for _ in range(TREE_DEPTH):
            node = 2 * node + 1 + (split[np.arange(len(flat)), node] > 0).astype(np.int64)
        return (node - TREE_NODES).reshape(data.shape[:-1])

    def components_encoded(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.asarray(encoded, dtype=np.float32)
        if data.shape[-1:] != (FEATURE_DIM,) or not np.isfinite(data).all():
            raise ValueError("invalid Warehouse v6 role features")
        flat = data.reshape(-1, FEATURE_DIM)
        leaves = self.leaf(data).reshape(-1)
        w = self.weights
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tree = w["tree.leaf_logits"][leaves]
            bound = MIN_RESIDUAL_BOUND + (1.0 - MIN_RESIDUAL_BOUND) / (
                1.0 + np.exp(-w["leaf_bound_logits"][leaves]))
            hidden = np.maximum(flat @ w["residual.0.weight"].T + w["residual.0.bias"], 0)
            hidden = np.maximum(hidden @ w["residual.2.weight"].T + w["residual.2.bias"], 0)
            residual = bound[:, None] * np.tanh(
                hidden @ w["residual.4.weight"].T + w["residual.4.bias"])
        if not np.isfinite(tree).all() or not np.isfinite(residual).all():
            raise RuntimeError("Warehouse v6 role produced nonfinite logits")
        shape = data.shape[:-1]
        return (tree.reshape(shape + (len(ACTIONS),)),
                residual.reshape(shape + (len(ACTIONS),)), bound.reshape(shape))

    def logits_encoded(self, encoded: np.ndarray) -> np.ndarray:
        tree, residual, _ = self.components_encoded(encoded)
        return tree + residual


class FrozenWarehouseV6Actor:
    """Content-addressed, NumPy-only inference for both frozen role NNs."""

    def __init__(self, roles: Mapping[str, FrozenWarehouseV6Role], artifact_sha256: str):
        if set(roles) != set(AGENTS):
            raise ValueError("Warehouse v6 frozen Actor requires both roles")
        self.roles = dict(roles)
        self.artifact_sha256 = artifact_sha256
        self.role_sha256 = {role: item.role_sha256 for role, item in self.roles.items()}

    @classmethod
    def from_model(cls, model: WarehouseV6Actor) -> "FrozenWarehouseV6Actor":
        return cls({role: FrozenWarehouseV6Role(_role_arrays(model.actors[role]))
                    for role in AGENTS}, "in-memory")

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             expected_role_sha256: Mapping[str, str] | None = None) -> "FrozenWarehouseV6Actor":
        raw = Path(path).read_bytes()
        digest = sha256(raw).hexdigest()
        if not expected_sha256 or digest != expected_sha256:
            raise RuntimeError("Warehouse v6 Actor artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("model_version") != MODEL_VERSION
                    or metadata.get("observation_version") != OBSERVATION_VERSION
                    or metadata.get("observation_dim") != OBSERVATION_DIM
                    or metadata.get("feature_dim") != FEATURE_DIM
                    or metadata.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                    or metadata.get("map_layout_id") != STUDY_MAP_LAYOUT.layout_id
                    or metadata.get("agent_ids") != list(AGENTS)
                    or metadata.get("action_names") != list(ACTIONS)
                    or metadata.get("tree_depth") != TREE_DEPTH
                    or metadata.get("tree_leaves") != TREE_LEAVES
                    or metadata.get("max_split_features") != MAX_SPLIT_FEATURES):
                raise RuntimeError("Warehouse v6 Actor artifact contract mismatch")
            roles = {role: FrozenWarehouseV6Role({
                name: np.asarray(archive[f"{role}.{name}"], dtype=np.float32)
                for name in _TENSOR_NAMES}) for role in AGENTS}
        result = cls(roles, digest)
        if metadata.get("role_sha256") != result.role_sha256:
            raise RuntimeError("Warehouse v6 Actor embedded role hashes mismatch")
        if expected_role_sha256 is not None and dict(expected_role_sha256) != result.role_sha256:
            raise RuntimeError("Warehouse v6 Actor expected role hashes mismatch")
        return result

    def logits(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        encoded = encode_policy_features(obs, masks)
        if encoded.shape[-2:] != (len(AGENTS), FEATURE_DIM):
            raise ValueError("Warehouse v6 Actor needs both role rows")
        return np.stack([self.roles[role].logits_encoded(encoded[..., index, :])
                         for index, role in enumerate(AGENTS)], axis=-2)

    def act(self, state, *, base_seed: int, deterministic: bool = False,
            temperature: float = 1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        obs, masks = encode_public_observation(state), action_masks(state)
        logits = np.where(masks > 0.5, self.logits(obs, masks) / temperature, -1.0e9)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        actions, distributions = {}, {}
        for index, role in enumerate(AGENTS):
            rng = np.random.default_rng(independent_agent_seed(
                base_seed=base_seed, episode_id=state.episode_id,
                frame=state.frame, agent_id=role))
            chosen = int(np.argmax(probs[index]) if deterministic else
                         rng.choice(len(ACTIONS), p=probs[index]))
            action = ACTIONS[chosen]
            actions[role] = action
            distributions[role] = ActionDistribution(
                agent_id=role, actions=ACTIONS,
                probabilities=tuple(float(x) for x in probs[index]),
                logits=tuple(float(x) for x in logits[index]),
                action_mask=tuple(float(x) for x in masks[index]),
                proposed_action=action,
            )
        return actions, distributions


def export_actor(model: WarehouseV6Actor, path: str | Path, *, pair_seed: int,
                 variant: str, candidate_id: str) -> dict:
    if variant not in ("raw", "regularized"):
        raise ValueError("invalid Warehouse v6 variant")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = {role: _role_arrays(model.actors[role]) for role in AGENTS}
    role_digests = {role: role_sha256(weights[role]) for role in AGENTS}
    metadata = {
        "model_version": MODEL_VERSION, "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM, "feature_dim": FEATURE_DIM,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
        "agent_ids": list(AGENTS), "action_names": list(ACTIONS),
        "tree_depth": TREE_DEPTH, "tree_leaves": TREE_LEAVES,
        "max_split_features": MAX_SPLIT_FEATURES,
        "minimum_residual_bound": MIN_RESIDUAL_BOUND,
        "role_sha256": role_digests, "pair_seed": pair_seed,
        "variant": variant, "candidate_id": candidate_id,
    }
    np.savez_compressed(target, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                        **{f"{role}.{name}": value
                           for role, role_weights in weights.items()
                           for name, value in role_weights.items()})
    digest = sha256(target.read_bytes()).hexdigest()
    frozen = FrozenWarehouseV6Actor.load(target, expected_sha256=digest,
                                         expected_role_sha256=role_digests)
    return {"path": str(target), "sha256": digest, "role_sha256": frozen.role_sha256,
            "model_version": MODEL_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256}
