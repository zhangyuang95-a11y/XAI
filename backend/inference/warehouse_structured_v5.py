"""NumPy-only reader for archived Warehouse v5.4 frozen dual Actors."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from backend.training.warehouse_selective_v5 import FEATURE_NAMES, observable_features
from core.policy_contracts import ActionDistribution
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import (
    AGENTS, OBSERVATION_DIM, OBSERVATION_VERSION, action_masks,
    observation_feature_names, observations,
)


STRUCTURED_MODEL_VERSION = "warehouse-structured-dual-nn.v5.4.sparse3"
TREE_DEPTH = 4
TREE_LEAVES = 1 << TREE_DEPTH
TREE_INTERNAL_NODES = TREE_LEAVES - 1
TREE_MAX_FEATURES_PER_NODE = 3
TREE_GATE_SCALE = 4.0
TREE_MASK_GRADIENT_TEMPERATURE = 0.05
RESIDUAL_SCALE = 1.0
_RAW_NAMES = observation_feature_names()
_EXTRA_RAW_NAMES = tuple(name for name in _RAW_NAMES
                         if name.startswith("order_") and name not in FEATURE_NAMES)
STRUCTURED_FEATURE_NAMES = FEATURE_NAMES + _EXTRA_RAW_NAMES
STRUCTURED_FEATURE_DIM = len(STRUCTURED_FEATURE_NAMES)
STRUCTURED_FEATURE_SCHEMA_SHA256 = sha256(json.dumps(
    STRUCTURED_FEATURE_NAMES, separators=(",", ":")).encode()).hexdigest()
_RAW_INDICES = tuple(_RAW_NAMES.index(name) for name in _EXTRA_RAW_NAMES)
_TENSOR_NAMES = (
    "tree.split.weight", "tree.split.bias", "tree.leaf_logits",
    "residual.0.weight", "residual.0.bias", "residual.2.weight",
    "residual.2.bias", "residual.4.weight", "residual.4.bias",
)


def structured_features_np(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Encode only public pre-action observation and physical action mask.

    Existing distance/route descriptors are geometric features of the public
    map and orders.  They never select or replace an Actor action.  The extra
    raw order coordinates preserve information available to the neural Actor
    but absent from the earlier selective tree input.
    """
    data = np.asarray(obs, dtype=np.float32)
    legal = np.asarray(masks, dtype=np.float32)
    if (data.shape[-1:] != (OBSERVATION_DIM,)
            or legal.shape != data.shape[:-1] + (len(ACTIONS),)
            or not np.isfinite(data).all() or not np.isfinite(legal).all()):
        raise ValueError("invalid Warehouse structured Actor observation or mask")
    flat_data = data.reshape(-1, OBSERVATION_DIM)
    flat_mask = legal.reshape(-1, len(ACTIONS))
    rows = []
    for row, mask in zip(flat_data, flat_mask, strict=True):
        features = observable_features({"obs": row, "mask": mask})
        semantic = [features[name] for name in FEATURE_NAMES]
        raw_order = row[list(_RAW_INDICES)].tolist()
        rows.append(semantic + raw_order)
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(flat_data), STRUCTURED_FEATURE_DIM):
        raise RuntimeError("Warehouse structured feature shape drifted")
    return result.reshape(data.shape[:-1] + (STRUCTURED_FEATURE_DIM,))


def _role_arrays(actor) -> dict[str, np.ndarray]:
    state = actor.state_dict()
    return {key: state[key].detach().cpu().numpy().astype(np.float32).copy()
            for key in _TENSOR_NAMES}


def structured_role_sha256(weights: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    if set(weights) != set(_TENSOR_NAMES):
        raise ValueError("structured Warehouse role tensor keys mismatch")
    for key in _TENSOR_NAMES:
        tensor = np.ascontiguousarray(weights[key], dtype="<f4")
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


class FrozenStructuredRole:
    def __init__(self, weights: Mapping[str, np.ndarray]):
        self.weights = {key: np.asarray(value, dtype=np.float32)
                        for key, value in weights.items()}
        if set(self.weights) != set(_TENSOR_NAMES):
            raise ValueError("structured Warehouse role tensor keys mismatch")
        feature_dim = STRUCTURED_FEATURE_DIM
        hidden = 256
        expected = {
            "tree.split.weight": (TREE_INTERNAL_NODES, feature_dim),
            "tree.split.bias": (TREE_INTERNAL_NODES,),
            "tree.leaf_logits": (TREE_LEAVES, len(ACTIONS)),
            "residual.0.weight": (hidden, feature_dim),
            "residual.0.bias": (hidden,),
            "residual.2.weight": (hidden, hidden),
            "residual.2.bias": (hidden,),
            "residual.4.weight": (len(ACTIONS), hidden),
            "residual.4.bias": (len(ACTIONS),),
        }
        if any(self.weights[key].shape != shape or
               not np.isfinite(self.weights[key]).all()
               for key, shape in expected.items()):
            raise ValueError("structured Warehouse role tensor shape or value invalid")
        self.role_sha256 = structured_role_sha256(self.weights)

    def logits_encoded(self, encoded: np.ndarray) -> np.ndarray:
        values = np.asarray(encoded, dtype=np.float32)
        if values.shape[-1:] != (STRUCTURED_FEATURE_DIM,) or not np.isfinite(values).all():
            raise ValueError("structured role feature vector invalid")
        flat = values.reshape(-1, STRUCTURED_FEATURE_DIM)
        w = self.weights
        # Some BLAS builds set floating-point warning flags during ordinary
        # matrix multiplication despite returning finite values.  Validate
        # the final result explicitly, as the existing frozen NN does.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tree_weight = w["tree.split.weight"]
            selected = np.argsort(-np.abs(tree_weight), axis=-1, kind="stable")[
                :, :TREE_MAX_FEATURES_PER_NODE]
            hard_mask = np.zeros_like(tree_weight)
            np.put_along_axis(hard_mask, selected, 1.0, axis=-1)
            split = flat @ (tree_weight * hard_mask).T + w["tree.split.bias"]
            left = 1.0 / (1.0 + np.exp(np.clip(TREE_GATE_SCALE * split, -80.0, 80.0)))
            path = np.ones((len(flat), 1), dtype=np.float32)
            for depth in range(TREE_DEPTH):
                begin = (1 << depth) - 1
                end = (1 << (depth + 1)) - 1
                gate = left[:, begin:end]
                path = np.stack((path * gate, path * (1.0 - gate)), axis=-1)
                path = path.reshape(len(flat), -1)
            tree = path @ w["tree.leaf_logits"]
            hidden = np.maximum(flat @ w["residual.0.weight"].T + w["residual.0.bias"], 0.0)
            hidden = np.maximum(hidden @ w["residual.2.weight"].T + w["residual.2.bias"], 0.0)
            residual = hidden @ w["residual.4.weight"].T + w["residual.4.bias"]
            result = tree + RESIDUAL_SCALE * np.tanh(residual)
        if not np.isfinite(result).all():
            raise RuntimeError("structured Warehouse role produced nonfinite logits")
        return result.reshape(values.shape[:-1] + (len(ACTIONS),))


class FrozenStructuredDualNN:
    """NumPy inference of the two frozen NN Actors, with no action controller."""

    def __init__(self, roles: Mapping[str, FrozenStructuredRole], artifact_sha256: str):
        if set(roles) != set(AGENTS):
            raise ValueError("structured dual NN requires both robot roles")
        self.roles = dict(roles)
        self.artifact_sha256 = artifact_sha256
        self.role_sha256 = {role: actor.role_sha256 for role, actor in self.roles.items()}

    @classmethod
    def from_model(cls, model: StructuredDualNN) -> "FrozenStructuredDualNN":
        return cls({role: FrozenStructuredRole(_role_arrays(model.actors[role]))
                    for role in AGENTS}, "in-memory")

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             expected_role_sha256: Mapping[str, str] | None = None) -> "FrozenStructuredDualNN":
        raw = Path(path).read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError("structured Warehouse NN artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("model_version") != STRUCTURED_MODEL_VERSION
                    or metadata.get("observation_version") != OBSERVATION_VERSION
                    or metadata.get("observation_dim") != OBSERVATION_DIM
                    or metadata.get("feature_schema_sha256") != STRUCTURED_FEATURE_SCHEMA_SHA256
                    or metadata.get("feature_dim") != STRUCTURED_FEATURE_DIM
                    or metadata.get("map_layout_id") != STUDY_MAP_LAYOUT.layout_id
                    or metadata.get("agent_ids") != list(AGENTS)
                    or metadata.get("action_names") != list(ACTIONS)
                    or metadata.get("tree_depth") != TREE_DEPTH
                    or metadata.get("tree_leaves") != TREE_LEAVES
                    or metadata.get("tree_max_features_per_node") != TREE_MAX_FEATURES_PER_NODE
                    or metadata.get("tree_mask_gradient_temperature") !=
                       TREE_MASK_GRADIENT_TEMPERATURE
                    or metadata.get("tree_gate_scale") != TREE_GATE_SCALE
                    or metadata.get("residual_scale") != RESIDUAL_SCALE):
                raise RuntimeError("structured Warehouse NN artifact contract mismatch")
            roles = {role: FrozenStructuredRole({
                key: np.asarray(archive[f"{role}.{key}"], dtype=np.float32)
                for key in _TENSOR_NAMES}) for role in AGENTS}
        result = cls(roles, digest)
        if metadata.get("role_sha256") != result.role_sha256:
            raise RuntimeError("structured Warehouse NN metadata role hash mismatch")
        if (expected_role_sha256 is not None and
                dict(expected_role_sha256) != result.role_sha256):
            raise RuntimeError("structured Warehouse NN role hash mismatch")
        return result

    def logits(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if np.asarray(obs).shape[-2:] != (len(AGENTS), OBSERVATION_DIM):
            raise ValueError("structured dual NN needs both role observations")
        encoded = structured_features_np(obs, masks)
        return np.stack([self.roles[role].logits_encoded(encoded[..., index, :])
                         for index, role in enumerate(AGENTS)], axis=-2)

    def act(self, state, *, base_seed: int, deterministic: bool = False,
            temperature: float = 1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        obs, masks = observations(state), action_masks(state)
        raw_logits = self.logits(obs, masks) / temperature
        logits = np.where(masks > 0.5, raw_logits, -1.0e9)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=-1, keepdims=True)
        actions, distributions = {}, {}
        for index, role in enumerate(AGENTS):
            rng = np.random.default_rng(independent_agent_seed(
                base_seed=base_seed, episode_id=state.episode_id,
                frame=state.frame, agent_id=role))
            chosen = int(np.argmax(probabilities[index]) if deterministic else
                         rng.choice(len(ACTIONS), p=probabilities[index]))
            actions[role] = ACTIONS[chosen]
            distributions[role] = ActionDistribution(
                agent_id=role, actions=ACTIONS,
                probabilities=tuple(float(value) for value in probabilities[index]),
                logits=tuple(float(value) for value in logits[index]),
                action_mask=tuple(float(value) for value in masks[index]),
                proposed_action=ACTIONS[chosen],
            )
        return actions, distributions
