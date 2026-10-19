"""NumPy-only, content-addressed shared-structure Warehouse Actors.

The tree is part of the learned Actor. An external program is never called by
this forward or by the inherited physical-mask/RNG action sampler.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.decision_protocol import independent_agent_seed
from core.policy_contracts import ActionDistribution
from env.warehouse.public_observation_v6 import OBSERVATION_DIM, OBSERVATION_VERSION, encode_public_observation
from env.warehouse.public_action_features_v617 import (
    ACTION_FEATURE_DIM, ACTION_PHYSICS_CONFIG_SHA256, action_features_from_tree_encoded,
    encode_actor_tree_features,
)
from .warehouse_v6_actor import (
    FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256, TREE_DEPTH, TREE_LEAVES,
    TREE_NODES, MAX_SPLIT_FEATURES, RESIDUAL_WIDTH, GEOMETRY_DIM,
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, _POSITION_COLUMNS, _MOVE,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, certified_feature_bounds, certified_leaf_actions,
    leaf_indices, validate_certification_domain,
)

MODEL_VERSION = "warehouse-dual-actor.v6.21-static-shared-structure"
CORE_BOUND_MODE = "static_leaf_anchor_and_box_bounded_residual.v1"
ACTOR_FORMULA = "tree_logits+where(static_core_leaf,b*tanh(nn256_logits),nn256_logits)"
PROGRAM_BOX_EPSILON = .05
CORE_MARGIN_SAFETY = .26
ALLOWED_CORE_BOUNDS = (.35, .50)
MASKED_DISTRIBUTION_VERSION = "legal_max_centered_float64_exp_sum_no_cast.v1"
TREE_MASK_INDICES = tuple(FEATURE_NAMES.index(f"legal_{a.lower()}") for a in ACTIONS)


def required_core_margin(bound):
    if isinstance(bound, bool) or float(bound) not in ALLOWED_CORE_BOUNDS:
        raise ValueError("v6.21 core residual bound must be .35 or .50")
    return 2*(float(bound)+PROGRAM_BOX_EPSILON)+CORE_MARGIN_SAFETY


def masked_policy_logits(logits, masks, *, temperature=1.):
    """Stable float64 physical-mask logits, with a legal maximum of zero.

    Common legal shifts preserve the NN distribution. This is numerical
    softmax stabilization, never a task-dependent action override.
    """
    z, mask = np.asarray(logits, np.float64), np.asarray(masks)
    if (z.ndim < 1 or z.shape[-1] != 5 or mask.shape != z.shape
            or not np.isfinite(z).all() or not np.isfinite(mask).all()
            or not np.all((mask == 0) | (mask == 1)) or not np.all(mask[..., 4] == 1)
            or isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("v6.21 physical mask/logits/temperature contract mismatch")
    legal = mask > .5
    maximum = np.max(np.where(legal, z, -np.inf), axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = (z-maximum)/float(temperature)
    # Only negative differences can overflow for valid finite logits. Values
    # below -1e300 already have exactly zero float64 exponential probability.
    shifted = np.clip(shifted, -1.e300, 0.)
    result = np.where(legal, shifted, -1.e9)
    if not np.isfinite(result).all():
        raise RuntimeError("v6.21 masked policy normalization is nonfinite")
    return result


def masked_probabilities(logits, masks, *, temperature=1.):
    """Float64 exp and sum, no final cast; illegal probability is exactly zero."""
    stabilized = masked_policy_logits(logits, masks, temperature=temperature)
    probabilities = np.where(np.asarray(masks) > .5, np.exp(stabilized), 0.)
    probabilities /= probabilities.sum(axis=-1, keepdims=True, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("v6.21 NN policy probabilities are nonfinite")
    return probabilities


def tensor_shapes():
    return {
        "tree.split.weight": (TREE_NODES, FEATURE_DIM),
        "tree.split.bias": (TREE_NODES,), "tree.leaf_logits": (TREE_LEAVES, 5),
        "tree.left_refs": (TREE_NODES,), "tree.right_refs": (TREE_NODES,),
        "tree.node_count": (1,), "tree.leaf_count": (1,),
        "action_residual.0.weight": (RESIDUAL_WIDTH, FEATURE_DIM+GEOMETRY_DIM+5+ACTION_FEATURE_DIM),
        "action_residual.0.bias": (RESIDUAL_WIDTH,),
        "action_residual.2.weight": (RESIDUAL_WIDTH, RESIDUAL_WIDTH),
        "action_residual.2.bias": (RESIDUAL_WIDTH,),
        "action_residual.4.weight": (1, RESIDUAL_WIDTH),
        "action_residual.4.bias": (1,),
        "core_leaf_mask": (TREE_LEAVES,), "core_actions": (TREE_LEAVES,),
        "core_residual_bound": (1,), "core_margin_floor": (1,),
    }


def role_sha256(weights: Mapping[str, np.ndarray]):
    if set(weights) != set(tensor_shapes()):
        raise ValueError("v6.21 role tensor keys mismatch")
    digest = sha256((MODEL_VERSION+CORE_BOUND_MODE+ACTOR_FORMULA+MASKED_DISTRIBUTION_VERSION).encode())
    for name in sorted(weights):
        value = np.asarray(weights[name])
        if value.dtype != np.float32 or value.shape != tensor_shapes()[name] or not np.isfinite(value).all():
            raise ValueError("v6.21 role tensor shape/dtype/value mismatch")
        value = np.ascontiguousarray(value, dtype="<f4")
        digest.update(name.encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_bound(value):
    for bound in ALLOWED_CORE_BOUNDS:
        if value == float(np.float32(bound)):
            return bound
    raise ValueError("v6.21 asset has an unregistered residual bound")


class FrozenWarehouseV621Role(FrozenWarehouseV6Role):
    """Strict new version; inherited compatibility type never dispatches old math."""
    def __init__(self, weights: Mapping[str, np.ndarray]):
        # Do not call the legacy version decoder or convert invalid dtypes.
        role_sha256(weights)
        values = {name: np.array(value, dtype=np.float32, copy=True)
                  for name, value in weights.items()}
        self.model_version, self.core_bound_mode = MODEL_VERSION, CORE_BOUND_MODE
        self.feature_names, self.feature_dim = FEATURE_NAMES, FEATURE_DIM
        self.weights = values
        self._validate_tree()
        n, count = int(values["tree.node_count"][0]), int(values["tree.leaf_count"][0])
        if np.any(np.count_nonzero(values["tree.split.weight"], axis=-1)
                  +(values["tree.split.weight"][:, FEATURE_NAMES.index("public_hard_risk_context")] != 0) > 3):
            raise ValueError("v6.21 source feature cost exceeds three")
        for name in ("tree.split.weight", "tree.split.bias", "tree.left_refs", "tree.right_refs"):
            if np.any(values[name][n:] != 0):
                raise ValueError("v6.21 unused node tensors must be zero")
        if np.any(values["tree.leaf_logits"][count:] != 0):
            raise ValueError("v6.21 unused leaf logits must be zero")
        core, anchors = values["core_leaf_mask"], values["core_actions"]
        if (np.any((core != 0) & (core != 1)) or np.any(core[count:] != 0)
                or np.any(anchors != np.floor(anchors))
                or np.any(anchors[core == 0] != -1)
                or np.any((anchors[core == 1] < 0) | (anchors[core == 1] >= 5))):
            raise ValueError("v6.21 static core membership/anchors invalid")
        self.core_residual_bound = _canonical_bound(float(values["core_residual_bound"][0]))
        floor = required_core_margin(self.core_residual_bound)
        if values["core_margin_floor"][0] != np.float32(floor):
            raise ValueError("v6.21 core margin metadata mismatch")
        certificates = self.certified_actions()
        for index in np.flatnonzero(core):
            action = int(anchors[index])
            logits = values["tree.leaf_logits"][index]
            gap = float(logits[action])-float(np.delete(logits, action).max())
            if (action not in certificates[index] or int(logits.argmax()) != action
                    or gap < float(values["core_margin_floor"][0])):
                raise ValueError("v6.21 core anchor lacks path or margin proof")
        if not np.allclose(values["tree.leaf_logits"][:count].mean(-1), 0., atol=2.e-5, rtol=0):
            raise ValueError("v6.21 tree logits must use the registered centered gauge")
        self.role_sha256 = role_sha256(values)
        for value in values.values():
            value.flags.writeable = False
        self.weights = MappingProxyType(values)

    def certified_actions(self):
        w = self.weights
        return certified_leaf_actions(w["tree.split.weight"], w["tree.split.bias"],
            w["tree.left_refs"].astype(np.int64), w["tree.right_refs"].astype(np.int64),
            int(w["tree.node_count"][0]), int(w["tree.leaf_count"][0]),
            TREE_MASK_INDICES, explicit=True, wait_action=4)

    def leaf(self, encoded):
        data = np.asarray(encoded, dtype=np.float32)
        if data.shape[-1:] != (FEATURE_DIM,) or not np.isfinite(data).all():
            raise ValueError("v6.21 encoded feature shape/value mismatch")
        flat = data.reshape(-1, FEATURE_DIM)
        validate_certification_domain(flat, TREE_MASK_INDICES,
                                      certified_feature_bounds(FEATURE_NAMES))
        w = self.weights
        route = leaf_indices(flat, w["tree.split.weight"], w["tree.split.bias"],
            w["tree.left_refs"], w["tree.right_refs"], int(w["tree.node_count"][0]), TREE_DEPTH)
        return route.reshape(data.shape[:-1])

    def tree_logits_encoded(self, encoded):
        return self.weights["tree.leaf_logits"][self.leaf(encoded)]

    def primary_logits_encoded(self, encoded):
        data = np.asarray(encoded, np.float32)
        if data.shape[-1:] != (FEATURE_DIM,) or not np.isfinite(data).all():
            raise ValueError("v6.21 encoded feature shape/value mismatch")
        flat = data.reshape(-1, FEATURE_DIM)
        own, partner = flat[:, list(_POSITION_COLUMNS[:2])], flat[:, list(_POSITION_COLUMNS[2:])]
        relative = own-partner
        target_gap = own[:, None, :]+_MOVE[None]-partner[:, None, :]
        geometry = np.concatenate((np.broadcast_to(relative[:, None], target_gap.shape),
                                   target_gap, np.abs(target_gap)), -1)
        relation = np.concatenate((np.broadcast_to(flat[:, None], (len(flat), 5, FEATURE_DIM)),
            geometry, np.broadcast_to(np.eye(5, dtype=np.float32)[None], (len(flat), 5, 5)),
            action_features_from_tree_encoded(flat)), -1)
        w = self.weights
        # BLAS can retain floating exception flags from unrelated Torch/LAPACK
        # calls. Check the actual NN values instead of trusting those flags.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            first = relation@w["action_residual.0.weight"].T+w["action_residual.0.bias"]
            second = np.maximum(first, 0)@w["action_residual.2.weight"].T+w["action_residual.2.bias"]
            score = (np.maximum(second, 0)@w["action_residual.4.weight"].T+w["action_residual.4.bias"]).squeeze(-1)
        if any(not np.isfinite(value).all() for value in (first, second, score)):
            raise RuntimeError("v6.21 residual NN produced nonfinite logits")
        return score.reshape(data.shape[:-1]+(5,))

    def effective_residual_encoded(self, encoded):
        route = self.leaf(encoded)
        score = self.primary_logits_encoded(encoded)
        core = self.weights["core_leaf_mask"][route] > .5
        return np.where(core[..., None], np.float32(self.core_residual_bound)*np.tanh(score), score)

    def components_encoded(self, encoded):
        route = self.leaf(encoded)
        return (self.weights["tree.leaf_logits"][route], self.effective_residual_encoded(encoded),
                np.where(self.weights["core_leaf_mask"][route] > .5,
                         np.float32(self.core_residual_bound), np.inf))

    def logits_encoded(self, encoded):
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.tree_logits_encoded(encoded)+self.effective_residual_encoded(encoded)
        if not np.isfinite(result).all():
            raise RuntimeError("v6.21 full Actor produced nonfinite logits")
        return result


class FrozenWarehouseV621Actor(FrozenWarehouseV6Actor):
    def __init__(self, roles, artifact_sha256):
        if (set(roles) != set(AGENTS)
                or any(not isinstance(role, FrozenWarehouseV621Role) for role in roles.values())):
            raise ValueError("v6.21 frozen Actor requires both typed independent roles")
        if len({item.core_residual_bound for item in roles.values()}) != 1:
            raise ValueError("v6.21 role residual bound recipes disagree")
        super().__init__(roles, artifact_sha256, model_version=MODEL_VERSION)

    @classmethod
    def from_model(cls, model):
        from .warehouse_v621_actor_training import role_arrays
        if (getattr(model, "model_version", None) != MODEL_VERSION
                or any(getattr(model.actors[r], "model_version", None) != MODEL_VERSION for r in AGENTS)):
            raise ValueError("v6.21 frozen factory rejects another Actor version")
        return cls({role: FrozenWarehouseV621Role(role_arrays(model.actors[role]))
                    for role in AGENTS}, "in-memory")

    @classmethod
    def load(cls, path, *, expected_sha256, expected_role_sha256=None):
        raw = Path(path).read_bytes()
        if not expected_sha256 or sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("v6.21 Actor artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            expected_keys = {"metadata_json", *[f"{r}.{name}" for r in AGENTS for name in tensor_shapes()]}
            if len(archive.files) != len(expected_keys) or set(archive.files) != expected_keys:
                raise RuntimeError("v6.21 Actor archive keys mismatch")
            metadata = json.loads(str(archive["metadata_json"].item()))
            for name, value in artifact_metadata().items():
                if metadata.get(name) != value:
                    raise RuntimeError("v6.21 Actor artifact contract mismatch: "+name)
            roles = {r: FrozenWarehouseV621Role({name: archive[f"{r}.{name}"]
                                               for name in tensor_shapes()}) for r in AGENTS}
        result = cls(roles, expected_sha256)
        if metadata.get("role_sha256") != result.role_sha256:
            raise RuntimeError("v6.21 embedded role hashes mismatch")
        if expected_role_sha256 is not None and dict(expected_role_sha256) != result.role_sha256:
            raise RuntimeError("v6.21 expected role hashes mismatch")
        return result

    def logits(self, obs, masks):
        encoded = encode_actor_tree_features(obs, masks)
        if encoded.shape[-2:] != (2, FEATURE_DIM):
            raise ValueError("v6.21 frozen Actor requires both role rows")
        return np.stack([self.roles[r].logits_encoded(encoded[..., i, :])
                         for i, r in enumerate(AGENTS)], axis=-2)


    def act(self, state, *, base_seed, deterministic=False, temperature=1.):
        obs, masks = encode_public_observation(state), action_masks(state)
        raw = self.logits(obs, masks)
        logits = masked_policy_logits(raw, masks, temperature=temperature)
        probabilities = masked_probabilities(raw, masks, temperature=temperature)
        selected, distributions = {}, {}
        for index, role in enumerate(AGENTS):
            rng = np.random.default_rng(independent_agent_seed(base_seed=base_seed,
                episode_id=state.episode_id, frame=state.frame, agent_id=role))
            action_index = int(probabilities[index].argmax() if deterministic else
                               rng.choice(5, p=probabilities[index]))
            if masks[index, action_index] != 1:
                raise RuntimeError("v6.21 physical mask rejected its own NN sample")
            action = ACTIONS[action_index]
            selected[role] = action
            distributions[role] = ActionDistribution(agent_id=role, actions=ACTIONS,
                probabilities=tuple(float(p) for p in probabilities[index]),
                logits=tuple(float(z) for z in logits[index]),
                action_mask=tuple(float(m) for m in masks[index]), proposed_action=action)
        return selected, distributions


def artifact_metadata():
    return {"model_version": MODEL_VERSION, "core_bound_mode": CORE_BOUND_MODE,
        "actor_formula": ACTOR_FORMULA, "observation_version": OBSERVATION_VERSION,
        "observation_dim": OBSERVATION_DIM, "feature_dim": FEATURE_DIM,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id, "agent_ids": list(AGENTS),
        "action_names": list(ACTIONS), "tree_depth": TREE_DEPTH, "tree_leaves": TREE_LEAVES,
        "max_split_features": 3, "residual_width": RESIDUAL_WIDTH,
        "relation_geometry_dim": GEOMETRY_DIM, "action_feature_dim": ACTION_FEATURE_DIM,
        "hard_split_routing_version": HARD_SPLIT_ROUTING_VERSION,
        "program_box_epsilon": PROGRAM_BOX_EPSILON, "core_margin_safety": CORE_MARGIN_SAFETY,
        "static_core_membership": True, "runtime_external_program_calls": 0,
        "masked_distribution_version": MASKED_DISTRIBUTION_VERSION,
        "illegal_probability": 0.0}
