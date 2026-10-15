"""Exact NumPy inference for the independent Warehouse v6 neural Actors.

The learned tree and bounded 256-unit residual jointly select an action after
the environment's physical mask. Torch training definitions load lazily from
``warehouse_v6_actor_training`` and never enter the deployed study runtime.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from core.policy_contracts import ActionDistribution
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256, OBSERVATION_DIM, OBSERVATION_VERSION,
    encode_policy_features, encode_public_observation,
)


MODEL_VERSION = "warehouse-dual-actor.v6.1"
TREE_DEPTH = 4
TREE_LEAVES = 1 << TREE_DEPTH
TREE_NODES = TREE_LEAVES - 1
MAX_SPLIT_FEATURES = 3
RESIDUAL_WIDTH = 256
MIN_RESIDUAL_BOUND = 0.05
MAX_RESIDUAL_BOUND = 8.05
GATE_WIDTH = 64
GEOMETRY_DIM = 6
_POSITION_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in (
    "self_row", "self_column", "partner_row", "partner_column"))
_MOVE = np.asarray([MOVE_DELTAS.get(action, (0, 0)) for action in ACTIONS],
                   dtype=np.float32) / np.asarray(
                       [max(1, STUDY_MAP_LAYOUT.rows - 1),
                        max(1, STUDY_MAP_LAYOUT.cols - 1)], dtype=np.float32)
TREE_GRADIENT_SCALE = 4.0
MASK_GRADIENT_TEMPERATURE = 0.05
_TENSOR_NAMES = (
    "tree.split.weight", "tree.split.bias", "tree.leaf_logits",
    "gate.0.weight", "gate.0.bias", "gate.2.weight", "gate.2.bias",
    "action_residual.0.weight", "action_residual.0.bias",
    "action_residual.2.weight", "action_residual.2.bias",
    "action_residual.4.weight", "action_residual.4.bias",
    "leaf_bound_logits",
)


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


class FrozenWarehouseV6Role:
    def __init__(self, weights: Mapping[str, np.ndarray]):
        self.weights = {name: np.asarray(value, dtype=np.float32)
                        for name, value in weights.items()}
        expected = {
            "tree.split.weight": (TREE_NODES, FEATURE_DIM),
            "tree.split.bias": (TREE_NODES,),
            "tree.leaf_logits": (TREE_LEAVES, len(ACTIONS)),
            "gate.0.weight": (GATE_WIDTH, FEATURE_DIM + 4),
            "gate.0.bias": (GATE_WIDTH,),
            "gate.2.weight": (1, GATE_WIDTH),
            "gate.2.bias": (1,),
            "action_residual.0.weight": (
                RESIDUAL_WIDTH, FEATURE_DIM + GEOMETRY_DIM + len(ACTIONS)),
            "action_residual.0.bias": (RESIDUAL_WIDTH,),
            "action_residual.2.weight": (RESIDUAL_WIDTH, RESIDUAL_WIDTH),
            "action_residual.2.bias": (RESIDUAL_WIDTH,),
            "action_residual.4.weight": (1, RESIDUAL_WIDTH),
            "action_residual.4.bias": (1,),
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
            bound = MIN_RESIDUAL_BOUND + (MAX_RESIDUAL_BOUND - MIN_RESIDUAL_BOUND) / (
                1.0 + np.exp(-w["leaf_bound_logits"][leaves]))
            own = flat[:, list(_POSITION_COLUMNS[:2])]
            partner = flat[:, list(_POSITION_COLUMNS[2:])]
            relative = own - partner
            gate_input = np.concatenate((flat, relative, np.abs(relative)), axis=-1)
            gate_hidden = np.maximum(gate_input @ w["gate.0.weight"].T +
                                     w["gate.0.bias"], 0)
            gate = 1.0 / (1.0 + np.exp(-(
                gate_hidden @ w["gate.2.weight"].T + w["gate.2.bias"])))
            target_gap = own[:, None, :] + _MOVE[None] - partner[:, None, :]
            geometry = np.concatenate((
                np.broadcast_to(relative[:, None, :], target_gap.shape),
                target_gap, np.abs(target_gap)), axis=-1)
            repeated = np.broadcast_to(flat[:, None, :],
                                       (len(flat), len(ACTIONS), FEATURE_DIM))
            action = np.broadcast_to(np.eye(len(ACTIONS), dtype=np.float32)[None],
                                     (len(flat), len(ACTIONS), len(ACTIONS)))
            relation = np.concatenate((repeated, geometry, action), axis=-1)
            hidden = np.maximum(relation @ w["action_residual.0.weight"].T +
                                w["action_residual.0.bias"], 0)
            hidden = np.maximum(hidden @ w["action_residual.2.weight"].T +
                                w["action_residual.2.bias"], 0)
            score = (hidden @ w["action_residual.4.weight"].T +
                     w["action_residual.4.bias"]).squeeze(-1)
            residual = (gate * bound[:, None]) * np.tanh(score)
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
                    or metadata.get("max_split_features") != MAX_SPLIT_FEATURES
                    or metadata.get("minimum_residual_bound") != MIN_RESIDUAL_BOUND
                    or metadata.get("maximum_residual_bound") != MAX_RESIDUAL_BOUND
                    or metadata.get("relation_geometry_dim") != GEOMETRY_DIM
                    or metadata.get("gate_width") != GATE_WIDTH):
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




_TRAINING_EXPORTS = frozenset({
    "SparseHardTree", "RoleActor", "WarehouseV6Actor", "policy_features_torch",
    "role_hashes", "export_actor",
})


def __getattr__(name: str):
    """Load Torch training definitions only when a caller requests them."""
    if name not in _TRAINING_EXPORTS:
        raise AttributeError(name)
    from . import warehouse_v6_actor_training as training
    value = getattr(training, name)
    globals()[name] = value
    return value
