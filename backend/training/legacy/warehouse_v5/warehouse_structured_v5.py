"""Development-only structured dual Actor for a future Warehouse v5 candidate.

Both role Actors remain neural policies.  A differentiable 16-leaf tree branch
and a bounded, 256-wide MLP residual jointly produce each Actor's logits.
The tree is a representation constraint, never an environment action source.
This module does not change the accepted v5 trainer, frozen assets, or pilot
gate.  Any release program still needs a separate fit and fidelity audit
against the *whole* frozen Actor distribution.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

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


def structured_features_torch(obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Torch view of a deterministic, non-trainable public-state encoder."""
    if obs.requires_grad or masks.requires_grad:
        raise ValueError("Warehouse state observations must not require gradients")
    encoded = structured_features_np(obs.detach().cpu().numpy(),
                                     masks.detach().cpu().numpy())
    return torch.from_numpy(encoded).to(device=obs.device)


class SoftTreeBranch(nn.Module):
    """Depth-four neural tree with at most three active inputs per split."""

    depth = TREE_DEPTH
    leaf_count = TREE_LEAVES
    max_features_per_node = TREE_MAX_FEATURES_PER_NODE

    def __init__(self, feature_dim: int = STRUCTURED_FEATURE_DIM):
        super().__init__()
        self.split = nn.Linear(feature_dim, TREE_INTERNAL_NODES)
        self.leaf_logits = nn.Parameter(torch.zeros(TREE_LEAVES, len(ACTIONS)))
        nn.init.normal_(self.split.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.split.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.025)

    def split_mask(self) -> torch.Tensor:
        """Stable top-three forward mask with a dense search gradient.

        Hard top-k alone would trap every split at its random initial three
        features. The straight-through soft mask lets other public features
        receive a gradient while the executed forward remains exactly sparse.
        """
        weights = self.split.weight
        selected = torch.argsort(-weights.abs(), dim=-1, stable=True)[
            :, :TREE_MAX_FEATURES_PER_NODE]
        hard = torch.zeros_like(weights).scatter_(-1, selected, 1.0)
        soft = torch.softmax(weights.abs() / TREE_MASK_GRADIENT_TEMPERATURE,
                             dim=-1) * TREE_MAX_FEATURES_PER_NODE
        return hard + soft - soft.detach()

    def effective_split_weights(self) -> torch.Tensor:
        return self.split.weight * self.split_mask()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.split.in_features:
            raise ValueError("structured tree feature dimension mismatch")
        leading = features.shape[:-1]
        values = features.reshape(-1, features.shape[-1])
        split = nn.functional.linear(values, self.effective_split_weights(),
                                     self.split.bias)
        left = torch.sigmoid(-TREE_GATE_SCALE * split)
        path = torch.ones((len(values), 1), dtype=values.dtype, device=values.device)
        for depth in range(TREE_DEPTH):
            begin = (1 << depth) - 1
            end = (1 << (depth + 1)) - 1
            gate = left[:, begin:end]
            path = torch.stack((path * gate, path * (1.0 - gate)), dim=-1)
            path = path.reshape(len(values), -1)
        return (path @ self.leaf_logits).reshape(leading + (len(ACTIONS),))


class StructuredRoleActor(nn.Module):
    """A neural action head with a bounded high-capacity residual."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        if hidden != 256:
            raise ValueError("structured Warehouse residual width must be 256")
        self.tree = SoftTreeBranch()
        self.residual = nn.Sequential(
            nn.Linear(STRUCTURED_FEATURE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.tree(encoded) + RESIDUAL_SCALE * torch.tanh(self.residual(encoded))


class StructuredDualNN(nn.Module):
    """Independent role Actors and a two-value centralized PPO critic."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.actors = nn.ModuleDict({role: StructuredRoleActor(hidden) for role in AGENTS})
        self.critic = nn.Sequential(
            nn.Linear(2 * OBSERVATION_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(AGENTS)),
        )

    def outputs_encoded(self, encoded: torch.Tensor,
                        masks: torch.Tensor) -> torch.Tensor:
        if (encoded.shape[-2:] != (len(AGENTS), STRUCTURED_FEATURE_DIM)
                or masks.shape != encoded.shape[:-1] + (len(ACTIONS),)):
            raise ValueError("structured dual Actor needs both role feature rows and masks")
        logits = torch.stack([self.actors[role](encoded[..., index, :])
                              for index, role in enumerate(AGENTS)], dim=-2)
        return logits.masked_fill(masks <= 0.5, -1.0e9)

    def outputs(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2:] != (len(AGENTS), OBSERVATION_DIM):
            raise ValueError("structured dual Actor needs both raw observations")
        return self.outputs_encoded(structured_features_torch(obs, masks), masks)

    def role_logits(self, role: str, obs: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """Masked role oracle for program extraction and fidelity auditing."""
        if role not in AGENTS:
            raise ValueError("unknown Warehouse role")
        encoded = structured_features_torch(obs, mask)
        return self.actors[role](encoded).masked_fill(mask <= 0.5, -1.0e9)


def _role_arrays(actor: StructuredRoleActor) -> dict[str, np.ndarray]:
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


def structured_weight_hashes(model: StructuredDualNN) -> dict[str, str]:
    """Stable per-role hashes for paired starts, checkpoints, and bindings."""
    return {role: structured_role_sha256(_role_arrays(model.actors[role]))
            for role in AGENTS}


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


def export_structured_actor(model: StructuredDualNN, path: str | Path,
                            *, pair_seed: int, variant: str) -> dict:
    """Export both role NNs with independently verified role and artifact hashes."""
    if variant not in ("raw", "regularized"):
        raise ValueError("unknown Warehouse structured variant")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    role_weights = {role: _role_arrays(model.actors[role]) for role in AGENTS}
    role_hashes = {role: structured_role_sha256(weights)
                   for role, weights in role_weights.items()}
    metadata = {
        "model_version": STRUCTURED_MODEL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
        "feature_dim": STRUCTURED_FEATURE_DIM,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
        "agent_ids": list(AGENTS), "action_names": list(ACTIONS),
        "tree_depth": TREE_DEPTH, "tree_leaves": TREE_LEAVES,
        "tree_max_features_per_node": TREE_MAX_FEATURES_PER_NODE,
        "tree_mask_gradient_temperature": TREE_MASK_GRADIENT_TEMPERATURE,
        "tree_gate_scale": TREE_GATE_SCALE, "residual_scale": RESIDUAL_SCALE,
        "role_sha256": role_hashes, "pair_seed": pair_seed, "variant": variant,
    }
    payload = {f"{role}.{key}": value for role, weights in role_weights.items()
               for key, value in weights.items()}
    np.savez_compressed(target,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                        **payload)
    digest = sha256(target.read_bytes()).hexdigest()
    frozen = FrozenStructuredDualNN.load(target, expected_sha256=digest,
                                         expected_role_sha256=role_hashes)
    # Use several real pre-action states and masks. Synthetic coordinates need
    # not describe passable cells for the semantic distance encoder.
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    differences = []
    mask_patterns = set()
    for seed in (1000, 1001, 1002):
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=seed)
        marked = env.get_state()
        marked.participant_controlled_agent_id = "robot_1"
        env.set_state(marked)
        for _ in range(8):
            if env.state.terminated or env.state.truncated:
                break
            env_obs, env_mask = observations(env.state), action_masks(env.state)
            mask_patterns.add(env_mask.tobytes())
            with torch.no_grad():
                expected = model.outputs(torch.from_numpy(env_obs),
                                         torch.from_numpy(env_mask)).numpy()
            actual = np.where(env_mask > 0.5, frozen.logits(env_obs, env_mask), -1.0e9)
            differences.append(float(np.max(np.abs(expected - actual))))
            actions, _ = frozen.act(env.state, base_seed=seed)
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("structured Warehouse parity probe changed NN action")
    if len(differences) < 3 or len(mask_patterns) < 2:
        raise RuntimeError("structured Warehouse parity probe lacks real state/mask diversity")
    difference = max(differences)
    if difference > 1e-4:
        raise RuntimeError("structured Warehouse Torch/NumPy logit parity failed")
    return {"path": str(target), "sha256": digest,
            "role_sha256": role_hashes,
            "feature_schema_sha256": STRUCTURED_FEATURE_SCHEMA_SHA256,
            "maximum_logit_difference": difference,
            "parity_pre_action_rows": len(differences),
            "parity_mask_patterns": len(mask_patterns)}
