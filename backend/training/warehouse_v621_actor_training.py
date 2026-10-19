"""Torch definitions for the inactive v6.21 shared-structure candidate.

No environment creation, seed authorization or admission is present here.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from .warehouse_v6_actor_training import (
    SparseHardTree as LegacySparseHardTree, RoleActor as LegacyRoleActor,
    policy_features_torch,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, explicit_mask_branches, split_scores,
)
from .warehouse_v621_actor import (
    MODEL_VERSION, CORE_BOUND_MODE, FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256,
    TREE_NODES, TREE_LEAVES, TREE_DEPTH, RESIDUAL_WIDTH, GEOMETRY_DIM,
    ACTION_FEATURE_DIM, TREE_MASK_INDICES, PROGRAM_BOX_EPSILON,
    FrozenWarehouseV621Actor, FrozenWarehouseV621Role,
    required_core_margin, tensor_shapes, role_sha256, artifact_metadata,
)


def masked_policy_logits(logits, masks):
    """Physical-mask stabilization with unchanged legal NN probabilities."""
    if (logits.shape[-1:] != (5,) or masks.shape != logits.shape
            or logits.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(logits).all() or not torch.isfinite(masks).all()
            or not torch.all((masks == 0) | (masks == 1)) or not torch.all(masks[..., 4] == 1)):
        raise ValueError("v6.21 physical-mask NN logits contract mismatch")
    legal = masks > .5
    maximum = logits.masked_fill(~legal, -torch.inf).amax(-1, keepdim=True)
    # Double subtraction avoids float32 overflow between two finite extremes.
    difference = logits.double()-maximum.double()
    if not torch.isfinite(difference).all():
        raise RuntimeError("v6.21 physical-mask centering overflowed")
    difference = difference.clamp(min=torch.finfo(logits.dtype).min, max=0).to(logits.dtype)
    return difference.masked_fill(~legal, -1.e9)


def _static_digest(tree):
    digest = sha256()
    for name in ("core_leaf_mask", "core_actions", "left_refs", "right_refs",
                 "node_count", "leaf_count", "guard_node_mask",
                 "fixed_guard_weights", "fixed_guard_bias",
                 "registered_core_residual_bound", "registered_core_margin_floor"):
        value = getattr(tree, name).detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


class SharedSparseHardTree(LegacySparseHardTree):
    def __init__(self, *, core_residual_bound=.35):
        super().__init__()
        self.model_version = MODEL_VERSION
        margin = required_core_margin(core_residual_bound)
        self.register_buffer("registered_core_residual_bound", torch.tensor([core_residual_bound], dtype=torch.float32))
        self.register_buffer("registered_core_margin_floor", torch.tensor([margin], dtype=torch.float32))
        self.register_buffer("core_leaf_mask", torch.zeros(TREE_LEAVES))
        self.register_buffer("core_actions", torch.full((TREE_LEAVES,), -1, dtype=torch.int64))
        self.register_buffer("fixed_guard_weights", torch.zeros(TREE_NODES, FEATURE_DIM))
        self.register_buffer("fixed_guard_bias", torch.zeros(TREE_NODES))
        self.register_buffer("static_contract_digest", torch.zeros(32, dtype=torch.uint8))
        self.register_buffer("structure_installed", torch.tensor([False]))
        self.register_buffer("structure_frozen", torch.tensor([False]))
        self.register_buffer("topology_frozen_at_step", torch.tensor([-1], dtype=torch.int64))
        self.register_buffer("frozen_core_logits", torch.zeros(TREE_LEAVES, 5))
        self.register_buffer("frozen_split_weights", torch.zeros(TREE_NODES, FEATURE_DIM))
        self.register_buffer("frozen_split_bias", torch.zeros(TREE_NODES))
        self.register_buffer("frozen_contract_digest", torch.zeros(32, dtype=torch.uint8))
        self.extra_margin = nn.Parameter(torch.full((TREE_LEAVES,), -3.))

    @property
    def core_residual_bound(self):
        value = float(self.registered_core_residual_bound[0])
        for candidate in (.35, .50):
            if value == float(np.float32(candidate)):
                return candidate
        raise RuntimeError("v6.21 checkpoint has an unregistered core bound")

    @property
    def core_margin_floor(self):
        expected = required_core_margin(self.core_residual_bound)
        if float(self.registered_core_margin_floor[0]) != float(np.float32(expected)):
            raise RuntimeError("v6.21 checkpoint margin/bound mismatch")
        return expected

    def assert_static_contract(self):
        if not bool(self.structure_installed[0]):
            raise RuntimeError("v6.21 structure must be installed before any forward or export")
        actual = torch.as_tensor(_static_digest(self), device=self.static_contract_digest.device)
        if not torch.equal(actual, self.static_contract_digest):
            raise RuntimeError("v6.21 static core/anchor/guard/topology contract changed")
        if bool(torch.any(self.frozen_contract_digest != 0)) and not bool(self.structure_frozen[0]):
            raise RuntimeError("v6.21 frozen topology flag was cleared")
        if bool(self.structure_frozen[0]):
            actual = torch.as_tensor(self._frozen_digest(), device=self.frozen_contract_digest.device)
            if not torch.equal(actual, self.frozen_contract_digest):
                raise RuntimeError("v6.21 frozen structure/core logits contract changed")

    def _frozen_digest(self):
        digest = sha256()
        for name in ("frozen_split_weights", "frozen_split_bias", "frozen_core_logits",
                     "topology_frozen_at_step", "static_contract_digest"):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(getattr(self, name).detach().cpu().numpy()).tobytes())
        return np.frombuffer(digest.digest(), dtype=np.uint8).copy()

    def effective_weights(self):
        if bool(self.structure_frozen[0]):
            return self.frozen_split_weights
        weights = self.split.weight
        hard = torch.where(self.support_fixed[0], self.support_mask,
                           self.sparse_support(weights)).to(weights.dtype)
        surrogate = .1*weights
        effective = weights*hard+(surrogate-surrogate.detach())
        return torch.where(self.guard_node_mask[:, None], self.fixed_guard_weights, effective)

    def effective_bias(self):
        if bool(self.structure_frozen[0]):
            return self.frozen_split_bias
        return torch.where(self.guard_node_mask, self.fixed_guard_bias, self.split.bias)

    def effective_leaf_logits(self):
        raw = self.leaf_logits
        core = self.core_leaf_mask.bool()
        anchors = self.core_actions.clamp_min(0)
        selected = F.one_hot(anchors, 5).bool()
        others = raw.masked_fill(selected, -torch.inf).max(-1).values
        # Ensure the stored float32 difference does not round below the
        # registered margin when leaf offsets grow during optimization.
        safety = 8*torch.finfo(raw.dtype).eps*(raw.detach().abs().max(-1).values
                                              +self.core_margin_floor+1.)
        anchor_logits = others+self.core_margin_floor+F.softplus(self.extra_margin)+safety
        projected = torch.where(core[:, None] & selected, anchor_logits[:, None], raw)
        projected = projected-projected.mean(-1, keepdim=True)
        if bool(self.structure_frozen[0]):
            projected = torch.where(core[:, None], self.frozen_core_logits, projected)
        return projected

    def path(self, encoded):
        self.assert_static_contract()
        values = encoded.reshape(-1, FEATURE_DIM)
        weights, bias = self.effective_weights(), self.effective_bias()
        soft_scores = F.linear(values, weights, bias)
        hard_scores = torch.as_tensor(split_scores(values.detach().cpu().numpy(),
            weights.detach().cpu().numpy(), bias.detach().cpu().numpy()), device=values.device)
        seen = set()
        def route(ref, depth):
            if depth > TREE_DEPTH:
                raise RuntimeError("v6.21 tree exceeds depth five")
            if ref < 0:
                return F.one_hot(torch.full((len(values),), -1-ref,
                    dtype=torch.long, device=values.device), TREE_LEAVES).to(values.dtype)
            if ref in seen or ref >= int(self.node_count[0]):
                raise RuntimeError("v6.21 invalid hard tree graph")
            seen.add(ref)
            left, right = route(int(self.left_refs[ref]), depth+1), route(int(self.right_refs[ref]), depth+1)
            hard = (hard_scores[:, ref] <= HARD_SPLIT_TIE_EPSILON).to(values.dtype)
            soft = torch.sigmoid(-4.*soft_scores[:, ref])
            gate = hard+(soft-soft.detach())
            return gate[:, None]*left+(1-gate[:, None])*right
        return route(0 if int(self.node_count[0]) else -1, 0).reshape(encoded.shape[:-1]+(TREE_LEAVES,))

    def forward(self, encoded):
        path = self.path(encoded)
        return path@self.effective_leaf_logits(), path

    def install_structure(self, structure, core_mask, core_actions, leaf_logits):
        if bool(self.structure_installed[0]):
            raise RuntimeError("v6.21 static structure/core may only be installed once")
        n, count = int(structure.get("node_count", len(structure["split_bias"]))), int(structure["leaf_count"])
        if not (0 <= n <= TREE_NODES and 1 <= count <= TREE_LEAVES and n == count-1
                and int(structure.get("root_ref", 0 if n else -1)) == (0 if n else -1)):
            raise ValueError("v6.21 invalid installation graph size/root")
        def padded(value, shape, *, fill=0, dtype=np.float32):
            value = np.asarray(value)
            if value.shape != shape and value.shape != ((n, FEATURE_DIM) if len(shape) == 2 and shape[-1] == FEATURE_DIM
                    else (n,) if shape == (TREE_NODES,) else (count, 5) if shape == (TREE_LEAVES, 5) else (count,)):
                raise ValueError("v6.21 installation shape mismatch")
            if not np.isfinite(value).all():
                raise ValueError("v6.21 installation contains nonfinite tensors")
            out = np.full(shape, fill, dtype=dtype)
            out[:len(value)] = value
            return out
        weights = padded(structure["split_weights"], (TREE_NODES, FEATURE_DIM))
        bias = padded(structure["split_bias"], (TREE_NODES,))
        lhs = padded(structure["left_refs"], (TREE_NODES,))
        rhs = padded(structure["right_refs"], (TREE_NODES,))
        logits = padded(leaf_logits, (TREE_LEAVES, 5))
        core = padded(core_mask, (TREE_LEAVES,))
        anchors = padded(core_actions, (TREE_LEAVES,), fill=-1)
        if (np.any((core != 0) & (core != 1)) or np.any(core[count:] != 0)
                or np.any(anchors != np.floor(anchors))
                or np.any((anchors[core == 1] < 0) | (anchors[core == 1] >= 5))
                or np.any(lhs != np.floor(lhs)) or np.any(rhs != np.floor(rhs))):
            raise ValueError("v6.21 invalid installation masks/anchors/references")
        anchors[core == 0] = -1
        guards = np.zeros(TREE_NODES, bool)
        for i in range(n):
            ll, rr = explicit_mask_branches(weights[i], float(bias[i]), TREE_MASK_INDICES)
            guards[i] = bool(ll or rr)
        with torch.no_grad():
            self.split.weight.copy_(torch.as_tensor(weights, device=self.split.weight.device))
            self.split.bias.copy_(torch.as_tensor(bias, device=self.split.bias.device))
            self.leaf_logits.copy_(torch.as_tensor(logits, device=self.leaf_logits.device))
            for name, value in (("left_refs", lhs), ("right_refs", rhs),
                                ("core_leaf_mask", core), ("core_actions", anchors),
                                ("guard_node_mask", guards)):
                getattr(self, name).copy_(torch.as_tensor(value, device=getattr(self, name).device))
            self.node_count[0], self.leaf_count[0] = n, count
            self.fixed_guard_weights.copy_(torch.as_tensor(weights*guards[:, None], device=self.split.weight.device))
            self.fixed_guard_bias.copy_(torch.as_tensor(bias*guards, device=self.split.bias.device))
            self.support_mask.copy_(self.split.weight != 0)
            self.support_fixed[0] = True
            self.structure_installed[0] = True
            self.static_contract_digest.copy_(torch.as_tensor(_static_digest(self), device=self.static_contract_digest.device))
        # The caller's invalid installation must never reach BC: validate the
        # complete exported graph, source cost and projected legal anchors.
        try:
            _validate_training_tree(self)
        except BaseException:
            with torch.no_grad():
                self.structure_installed[0] = False
            raise
        return self.contract_report()

    def refresh_sparse_support(self):
        self.assert_static_contract()
        if bool(self.structure_frozen[0]):
            raise RuntimeError("v6.21 frozen topology cannot refresh support")
        with torch.no_grad():
            selected = self.sparse_support(self.split.weight)
            selected[self.guard_node_mask] = self.fixed_guard_weights[self.guard_node_mask] != 0
            self.support_mask.copy_(selected)
            self.support_fixed[0] = True
        _validate_training_tree(self)
        return self.contract_report()

    def freeze_structure(self, joint_step):
        self.assert_static_contract()
        if type(joint_step) is not int or joint_step < 0:
            raise ValueError("v6.21 topology freeze step must be nonnegative integer")
        if bool(self.structure_frozen[0]):
            if int(self.topology_frozen_at_step[0]) != joint_step:
                raise RuntimeError("v6.21 topology freeze step cannot be changed")
            return self.contract_report()
        with torch.no_grad():
            self.frozen_core_logits.copy_(self.effective_leaf_logits())
            self.frozen_split_weights.copy_(self.effective_weights())
            self.frozen_split_bias.copy_(self.effective_bias())
            self.structure_frozen[0] = True
            self.topology_frozen_at_step[0] = joint_step
            self.frozen_contract_digest.copy_(torch.as_tensor(self._frozen_digest(), device=self.frozen_contract_digest.device))
        self.set_topology_trainable(False)
        return self.contract_report()

    def set_topology_trainable(self, enabled):
        if enabled and hasattr(self, "structure_frozen") and bool(self.structure_frozen[0]):
            raise RuntimeError("v6.21 frozen topology cannot be unfrozen")
        super().set_topology_trainable(enabled)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self.set_topology_trainable(not bool(self.structure_frozen[0]))
        if bool(self.structure_installed[0]):
            self.assert_static_contract()

    def contract_report(self):
        self.assert_static_contract()
        return {"static_contract_sha256": bytes(self.static_contract_digest.cpu().tolist()).hex(),
            "core_leaf_mask": self.core_leaf_mask.detach().cpu().tolist(),
            "core_actions": self.core_actions.detach().cpu().tolist(),
            "node_count": int(self.node_count[0]), "leaf_count": int(self.leaf_count[0]),
            "topology_frozen_at_step": int(self.topology_frozen_at_step[0]),
            "structure_frozen": bool(self.structure_frozen[0]),
            "core_residual_bound": self.core_residual_bound,
            "core_margin_floor": self.core_margin_floor}


def _tree_arrays(tree):
    tree.assert_static_contract()
    n, count = int(tree.node_count[0]), int(tree.leaf_count[0])
    values = {"tree.split.weight": tree.effective_weights().detach().cpu().numpy().astype(np.float32, copy=True),
        "tree.split.bias": tree.effective_bias().detach().cpu().numpy().astype(np.float32, copy=True),
        "tree.leaf_logits": tree.effective_leaf_logits().detach().cpu().numpy().astype(np.float32, copy=True)}
    for name in ("left_refs", "right_refs", "node_count", "leaf_count"):
        values["tree."+name] = getattr(tree, name).detach().cpu().numpy().astype(np.float32, copy=True)
    for name in ("tree.split.weight", "tree.split.bias", "tree.left_refs", "tree.right_refs"):
        values[name][n:] = 0
    values["tree.leaf_logits"][count:] = 0
    values.update(core_leaf_mask=tree.core_leaf_mask.detach().cpu().numpy().astype(np.float32, copy=True),
        core_actions=tree.core_actions.detach().cpu().numpy().astype(np.float32, copy=True),
        core_residual_bound=np.asarray([tree.core_residual_bound], np.float32),
        core_margin_floor=np.asarray([tree.core_margin_floor], np.float32))
    return values


def _validate_training_tree(tree):
    values = _tree_arrays(tree)
    shapes = tensor_shapes()
    for name in shapes:
        if name.startswith("action_residual."):
            values[name] = np.zeros(shapes[name], np.float32)
    FrozenWarehouseV621Role(values)


class RoleActor(nn.Module):
    def __init__(self, *, core_residual_bound=.35):
        super().__init__()
        self.model_version, self.core_bound_mode = MODEL_VERSION, CORE_BOUND_MODE
        self.tree = SharedSparseHardTree(core_residual_bound=core_residual_bound)
        self.action_residual = nn.Sequential(
            nn.Linear(FEATURE_DIM+GEOMETRY_DIM+5+ACTION_FEATURE_DIM, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH), nn.ReLU(), nn.Linear(RESIDUAL_WIDTH, 1))
        nn.init.normal_(self.action_residual[-1].weight, mean=0., std=.01)
        nn.init.zeros_(self.action_residual[-1].bias)

    @property
    def core_leaf_mask(self):
        return self.tree.core_leaf_mask

    @property
    def core_actions(self):
        return self.tree.core_actions

    def install_structure(self, structure, core_mask, core_actions, leaf_logits):
        return self.tree.install_structure(structure, core_mask, core_actions, leaf_logits)

    def load_primary_state(self, state):
        """Explicit paired initialization from an in-memory NN state mapping.

        This copies values into independent parameters; it never aliases a
        comparison Actor or reads a checkpoint/seed on its own.
        """
        expected = self.action_residual.state_dict()
        values = {name.removeprefix("action_residual."): value for name, value in state.items()}
        if set(values) != set(expected):
            raise ValueError("v6.21 paired primary tensor keys mismatch")
        converted = {}
        for name, target in expected.items():
            source = torch.as_tensor(values[name], dtype=target.dtype, device=target.device)
            if source.shape != target.shape or not torch.isfinite(source).all():
                raise ValueError("v6.21 paired primary tensor shape/value mismatch")
            converted[name] = source.detach().clone()
        digest = sha256()
        with torch.no_grad():
            for name, target in expected.items():
                source = converted[name]
                target.copy_(source)
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(source.cpu().numpy(), dtype="<f4").tobytes())
        return {"primary_initialization_sha256": digest.hexdigest(), "independent_parameter_copy": True}

    def tree_outputs_encoded(self, encoded):
        return self.tree(encoded)

    def primary_logits_encoded(self, encoded):
        _, relation = LegacyRoleActor._relation_features(self, encoded)
        first = self.action_residual[0](relation)
        if not torch.isfinite(first).all():
            raise RuntimeError("v6.21 primary NN first layer overflowed")
        second = self.action_residual[2](self.action_residual[1](first))
        if not torch.isfinite(second).all():
            raise RuntimeError("v6.21 primary NN second layer overflowed")
        score = self.action_residual[4](self.action_residual[3](second)).squeeze(-1)
        if not torch.isfinite(score).all():
            raise RuntimeError("v6.21 primary NN output overflowed")
        return score

    def details_encoded(self, encoded):
        tree, path = self.tree_outputs_encoded(encoded)
        score = self.primary_logits_encoded(encoded)
        # Hard membership is static leaf identity. The continuous path carries
        # routing gradients for BC's all-frame occupancy loss only.
        core_path = path@self.tree.core_leaf_mask
        membership = core_path.detach() > .5
        residual = torch.where(membership[..., None],
            self.tree.core_residual_bound*torch.tanh(score), score)
        full = tree+residual
        if not torch.isfinite(tree).all() or not torch.isfinite(full).all():
            raise RuntimeError("v6.21 full Actor produced nonfinite logits")
        return {"tree_logits": tree, "residual": residual, "logits": full,
            "core_membership": core_path, "leaf_indices": path.detach().argmax(-1), "path": path}

    def residual_logits_encoded(self, encoded):
        return self.details_encoded(encoded)["residual"]

    def forward(self, encoded):
        return self.details_encoded(encoded)["logits"]

    def freeze_structure(self, joint_step):
        return self.tree.freeze_structure(joint_step)


class WarehouseV621Actor(nn.Module):
    def __init__(self, *, core_residual_bound=.35):
        super().__init__()
        self.model_version = MODEL_VERSION
        self.actors = nn.ModuleDict({role: RoleActor(core_residual_bound=core_residual_bound) for role in AGENTS})
        self.reward_critic = nn.Sequential(nn.Linear(2*OBSERVATION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        self.cost_critic = nn.Sequential(nn.Linear(2*OBSERVATION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def outputs_encoded(self, encoded, masks):
        if encoded.shape[-2:] != (2, FEATURE_DIM) or masks.shape != encoded.shape[:-1]+(5,):
            raise ValueError("v6.21 joint Actor feature/mask shape mismatch")
        if (not torch.isfinite(encoded).all() or not torch.isfinite(masks).all()
                or not torch.all((masks == 0) | (masks == 1))
                or not torch.all(masks[..., 4] == 1)
                or not torch.equal(encoded[..., list(TREE_MASK_INDICES)], masks.to(encoded))):
            raise ValueError("v6.21 encoded legal features differ from physical mask")
        logits = torch.stack([self.actors[r](encoded[..., i, :]) for i, r in enumerate(AGENTS)], -2)
        return masked_policy_logits(logits, masks)

    def outputs(self, obs, masks):
        return self.outputs_encoded(policy_features_torch(obs, masks), masks)

    def role_logits(self, role, obs, masks):
        if role not in AGENTS:
            raise ValueError("v6.21 unknown role")
        encoded = policy_features_torch(obs, masks)
        if encoded.ndim >= 3:
            index = AGENTS.index(role)
            encoded, masks = encoded[..., index, :], masks[..., index, :]
        return masked_policy_logits(self.actors[role](encoded), masks)

    def install_structure(self, role, structure, core_mask, core_actions, leaf_logits):
        return self.actors[role].install_structure(structure, core_mask, core_actions, leaf_logits)

    def freeze_structure(self, joint_step):
        return {r: self.actors[r].freeze_structure(joint_step) for r in AGENTS}

    def constraint_report(self):
        return {r: self.actors[r].tree.contract_report() for r in AGENTS}


def role_arrays(actor):
    values = _tree_arrays(actor.tree)
    for name, value in actor.action_residual.state_dict().items():
        values["action_residual."+name] = value.detach().cpu().numpy().astype(np.float32, copy=True)
    return values


def role_hashes(model):
    return {r: role_sha256(role_arrays(model.actors[r])) for r in AGENTS}


def export_actor(model, path, *, pair_seed, variant, candidate_id):
    if (variant not in ("raw", "regularized") or type(pair_seed) is not int or pair_seed < 0
            or not isinstance(candidate_id, str) or not candidate_id):
        raise ValueError("v6.21 export requires explicit candidate/pair/variant")
    if model.model_version != MODEL_VERSION:
        raise ValueError("v6.21 exporter rejects another model version")
    target = Path(path)
    if target.exists():
        raise FileExistsError("v6.21 Actor export never overwrites an artifact")
    values = {r: role_arrays(model.actors[r]) for r in AGENTS}
    hashes = {r: role_sha256(values[r]) for r in AGENTS}
    for value in values.values():
        FrozenWarehouseV621Role(value)
    metadata = {**artifact_metadata(), "pair_seed": pair_seed, "variant": variant,
        "candidate_id": candidate_id, "role_sha256": hashes,
        "training_constraint_report": model.constraint_report(), "publishable": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(handle, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{f"{r}.{name}": value for r, weights in values.items() for name, value in weights.items()})
    digest = sha256(target.read_bytes()).hexdigest()
    frozen = FrozenWarehouseV621Actor.load(target, expected_sha256=digest, expected_role_sha256=hashes)
    return {"path": str(target), "sha256": digest, "role_sha256": frozen.role_sha256,
            "model_version": MODEL_VERSION, "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "publishable": False}
