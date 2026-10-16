"""Torch-only training definitions for the Warehouse v6 dual Actor.

The deployed frozen Actor lives in warehouse_v6_actor and imports only NumPy.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import (
    OBSERVATION_DIM, OBSERVATION_VERSION,
)
from env.warehouse.public_action_features_v614 import (
    ACTION_FEATURE_DIM, ACTION_PHYSICS_CONFIG_SHA256,
    action_features_from_tree_encoded,
    encode_actor_tree_features,
)
from .warehouse_v6_actor import (
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, MODEL_VERSION,
    FEATURE_DIM, FEATURE_SCHEMA_SHA256,
    TREE_DEPTH, TREE_LEAVES, TREE_NODES, MAX_SPLIT_FEATURES, RESIDUAL_WIDTH,
    MIN_RESIDUAL_BOUND, MAX_RESIDUAL_BOUND, CORE_LOW_GATE_BOUND,
    CORE_BOUND_MODE,
    GATE_WIDTH, GEOMETRY_DIM,
    _POSITION_COLUMNS, _MOVE, TREE_GRADIENT_SCALE,
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    _TENSOR_NAMES, _role_arrays, role_sha256,
)

def policy_features_torch(obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (obs.shape[-1:] != (OBSERVATION_DIM,)
            or mask.shape != obs.shape[:-1] + (len(ACTIONS),)):
        raise ValueError("Warehouse v6 Actor observation or mask dimension mismatch")
    # Public map distances have no trainable parameters.  Both Torch training
    # and frozen NumPy inference call the same deterministic pre-action
    # encoder, avoiding a second implementation of shortest-path features.
    encoded = encode_actor_tree_features(obs.detach().cpu().numpy(),
                                         mask.detach().cpu().numpy())
    return torch.as_tensor(encoded, dtype=obs.dtype, device=obs.device)


class SparseHardTree(nn.Module):
    """Hard sparse tree with local gradients through visited decisions."""

    def __init__(self):
        super().__init__()
        self.split = nn.Linear(FEATURE_DIM, TREE_NODES)
        self.leaf_logits = nn.Parameter(torch.zeros(TREE_LEAVES, len(ACTIONS)))
        left = torch.zeros(TREE_NODES, dtype=torch.int64)
        right = torch.zeros(TREE_NODES, dtype=torch.int64)
        for node in range(TREE_NODES):
            lhs, rhs = 2 * node + 1, 2 * node + 2
            left[node] = lhs if lhs < TREE_NODES else -1 - (lhs - TREE_NODES)
            right[node] = rhs if rhs < TREE_NODES else -1 - (rhs - TREE_NODES)
        self.register_buffer("left_refs", left)
        self.register_buffer("right_refs", right)
        self.register_buffer("node_count", torch.tensor([TREE_NODES], dtype=torch.int64))
        self.register_buffer("leaf_count", torch.tensor([TREE_LEAVES], dtype=torch.int64))
        nn.init.normal_(self.split.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.split.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.025)
        # Training starts from public BC splits; a local straight-through
        # surrogate below keeps their hard routing trainable.

    def effective_weights(self) -> torch.Tensor:
        weights = self.split.weight
        selected = torch.argsort(-weights.abs(), dim=-1, stable=True)[
            :, :MAX_SPLIT_FEATURES]
        hard = torch.zeros_like(weights).scatter_(-1, selected, 1.0)
        return weights * hard

    def path(self, encoded: torch.Tensor) -> torch.Tensor:
        values = encoded.reshape(-1, FEATURE_DIM)
        weights = self.effective_weights()
        split = nn.functional.linear(values, weights, self.split.bias)
        # The hard route must not depend on float32 GEMM batch shape. Keep
        # the existing float32 surrogate for gradients through the split.
        hard_split = nn.functional.linear(
            values.detach().to(torch.float64),
            weights.detach().to(torch.float64),
            self.split.bias.detach().to(torch.float64))
        node_count = int(self.node_count[0])
        leaf_count = int(self.leaf_count[0])
        visited: set[int] = set()

        def subtree(ref: int, depth: int) -> torch.Tensor:
            if depth > TREE_DEPTH:
                raise RuntimeError("Warehouse v6 irregular tree too deep")
            if ref < 0:
                leaf = -1 - ref
                if leaf >= leaf_count:
                    raise RuntimeError("Warehouse v6 irregular tree leaf invalid")
                leaves = torch.full((len(values),), leaf, dtype=torch.long,
                                    device=values.device)
                return nn.functional.one_hot(leaves, TREE_LEAVES).to(values.dtype)
            if ref >= node_count or ref in visited:
                raise RuntimeError("Warehouse v6 irregular tree node invalid")
            visited.add(ref)
            left = subtree(int(self.left_refs[ref]), depth + 1)
            right = subtree(int(self.right_refs[ref]), depth + 1)
            score = split[:, ref]
            hard_left = (hard_split[:, ref] <= HARD_SPLIT_TIE_EPSILON).to(values.dtype)
            soft_left = torch.sigmoid(-TREE_GRADIENT_SCALE * score)
            gate = hard_left + (soft_left - soft_left.detach())
            return gate[:, None] * left + (1.0 - gate)[:, None] * right

        path = subtree(0 if node_count else -1, 0)
        return path.reshape(encoded.shape[:-1] + (TREE_LEAVES,))

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path(encoded)
        return path @ self.leaf_logits, path

    def initialize_from_classifier(self, classifier) -> dict:
        """Initialize hard splits from a supervised, public-observation tree.

        The classifier is fitted separately for each role before self-play.
        Every deployed split has one observed input and thus stays within the
        three-feature program limit. Early terminal nodes remain leaves.
        """
        tree = classifier.tree_
        weights = np.zeros((TREE_NODES, FEATURE_DIM), dtype=np.float32)
        bias = np.zeros(TREE_NODES, dtype=np.float32)
        left_refs = np.zeros(TREE_NODES, dtype=np.int64)
        right_refs = np.zeros(TREE_NODES, dtype=np.int64)
        leaves = np.zeros((TREE_LEAVES, len(ACTIONS)), dtype=np.float32)
        purity = np.zeros(TREE_LEAVES, dtype=np.float32)
        occupied = set()
        node_count = 0
        leaf_count = 0

        def fill(source: int, depth: int) -> int:
            nonlocal node_count, leaf_count
            lhs, rhs = int(tree.children_left[source]), int(tree.children_right[source])
            if lhs < 0:
                if leaf_count >= TREE_LEAVES:
                    raise RuntimeError("public tree exceeds 16 leaves")
                index = leaf_count
                leaf_count += 1
                counts = np.zeros(len(ACTIONS), dtype=np.float64)
                class_values = np.asarray(tree.value[source, 0], dtype=np.float64)
                if np.isclose(class_values.sum(), 1.0):
                    class_values *= float(tree.weighted_n_node_samples[source])
                counts[np.asarray(classifier.classes_, dtype=np.int64)] = class_values
                probability = (counts + 1.0) / (counts.sum() + len(ACTIONS))
                leaves[index] = np.log(probability).astype(np.float32)
                purity[index] = float(counts.max() / max(1.0, counts.sum()))
                occupied.add(source)
                return -1 - index
            if depth >= TREE_DEPTH or node_count >= TREE_NODES:
                raise RuntimeError("public tree complexity exceeds depth or nodes")
            index = node_count
            node_count += 1
            feature = int(tree.feature[source])
            weights[index, feature] = 4.0
            bias[index] = -4.0 * float(tree.threshold[source])
            left_refs[index] = fill(lhs, depth + 1)
            right_refs[index] = fill(rhs, depth + 1)
            return index

        if fill(0, 0) != (0 if node_count else -1):
            raise RuntimeError("public tree root encoding mismatch")
        if node_count != leaf_count - 1:
            raise RuntimeError("public tree graph size mismatch")
        with torch.no_grad():
            self.split.weight.copy_(torch.from_numpy(weights))
            self.split.bias.copy_(torch.from_numpy(bias))
            self.leaf_logits.copy_(torch.from_numpy(leaves))
            self.left_refs.copy_(torch.from_numpy(left_refs))
            self.right_refs.copy_(torch.from_numpy(right_refs))
            self.node_count[0] = node_count
            self.leaf_count[0] = leaf_count
        self.split.weight.requires_grad_(True)
        self.split.bias.requires_grad_(True)
        return {"visited_source_leaves": len(occupied),
                "node_count": node_count, "leaf_count": leaf_count,
                "leaf_purity": purity.tolist(),
                "nonzero_split_features": int(np.count_nonzero(weights))}


class RoleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree = SparseHardTree()
        self.gate = nn.Sequential(
            nn.Linear(FEATURE_DIM + 4, GATE_WIDTH), nn.ReLU(),
            nn.Linear(GATE_WIDTH, 1),
        )
        self.action_residual = nn.Sequential(
            nn.Linear(FEATURE_DIM + GEOMETRY_DIM + len(ACTIONS)
                      + ACTION_FEATURE_DIM, RESIDUAL_WIDTH),
            nn.ReLU(), nn.Linear(RESIDUAL_WIDTH, RESIDUAL_WIDTH), nn.ReLU(),
            nn.Linear(RESIDUAL_WIDTH, 1),
        )
        self.leaf_bound_logits = nn.Parameter(torch.full((TREE_LEAVES,), -0.9))
        # A public-training-data core is an Actor tensor, not a runtime action
        # rule. Its bound is applied inside the NN forward pass for both arms.
        self.register_buffer("core_leaf_mask", torch.zeros(TREE_LEAVES))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -0.7)
        nn.init.normal_(self.action_residual[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.action_residual[-1].bias)

    def initialize_tree_from_classifier(self, classifier) -> dict:
        report = self.tree.initialize_from_classifier(classifier)
        # Pure public-state leaves need only a small bounded correction;
        # ambiguous leaves retain enough capacity for neural coordination.
        desired = np.where(np.asarray(report["leaf_purity"]) >= 0.9, 4.0, 6.0)
        fraction = np.clip((desired - MIN_RESIDUAL_BOUND) /
                           (MAX_RESIDUAL_BOUND - MIN_RESIDUAL_BOUND), 1e-4, 1-1e-4)
        with torch.no_grad():
            self.core_leaf_mask.zero_()
            self.leaf_bound_logits.copy_(torch.from_numpy(
                np.log(fraction / (1.0 - fraction)).astype(np.float32)))
        return report

    def initialize_tree_from_core_result(self, result) -> dict:
        """Install a fit-only public sparse tree and hashed core-leaf mask."""
        arrays = {
            "split_weights": (TREE_NODES, FEATURE_DIM),
            "split_bias": (TREE_NODES,),
            "left_refs": (TREE_NODES,),
            "right_refs": (TREE_NODES,),
            "leaf_logits": (TREE_LEAVES, len(ACTIONS)),
            "core_mask": (TREE_LEAVES,),
        }
        values = {name: np.asarray(getattr(result, name)) for name in arrays}
        if any(values[name].shape != shape or not np.isfinite(values[name]).all()
               for name, shape in arrays.items()):
            raise ValueError("Warehouse core tree result tensor shape/value mismatch")
        nodes, leaves = int(result.node_count), int(result.leaf_count)
        if not 1 <= leaves <= TREE_LEAVES or nodes != leaves - 1:
            raise ValueError("Warehouse core tree result size mismatch")
        core = values["core_mask"].astype(np.float32)
        if (np.any((core != 0.0) & (core != 1.0))
                or np.any(core[leaves:] != 0.0)):
            raise ValueError("Warehouse core tree result mask invalid")
        with torch.no_grad():
            self.tree.split.weight.copy_(torch.from_numpy(np.array(
                values["split_weights"], dtype=np.float32, copy=True)))
            self.tree.split.bias.copy_(torch.from_numpy(np.array(
                values["split_bias"], dtype=np.float32, copy=True)))
            self.tree.left_refs.copy_(torch.from_numpy(np.array(
                values["left_refs"], dtype=np.int64, copy=True)))
            self.tree.right_refs.copy_(torch.from_numpy(np.array(
                values["right_refs"], dtype=np.int64, copy=True)))
            self.tree.node_count[0] = nodes
            self.tree.leaf_count[0] = leaves
            self.tree.leaf_logits.copy_(torch.from_numpy(np.array(
                values["leaf_logits"], dtype=np.float32, copy=True)))
            self.core_leaf_mask.copy_(torch.from_numpy(core))
            # Every leaf retains a learned residual budget. In core leaves a
            # learned pre-action gate makes the effective budget small in
            # ordinary states and can release it in conflict states.
            desired = np.full(TREE_LEAVES, 6.0, dtype=np.float32)
            upper = np.full(TREE_LEAVES, MAX_RESIDUAL_BOUND, dtype=np.float32)
            fraction = np.clip((desired - MIN_RESIDUAL_BOUND) /
                               (upper - MIN_RESIDUAL_BOUND), 1e-4, 1 - 1e-4)
            self.leaf_bound_logits.copy_(torch.from_numpy(
                np.log(fraction / (1.0 - fraction)).astype(np.float32)))
        # Core legality is certified by path predicates.  Keep those split
        # predicates fixed after public-data fitting; leaf logits, gate,
        # bounds, and the 256-unit residual remain trainable.
        self.tree.split.weight.requires_grad_(False)
        self.tree.split.bias.requires_grad_(False)
        # Exact topology, sparsity, and role hash validation is shared with
        # deployed NumPy inference; no separate train-only interpretation.
        FrozenWarehouseV6Role(_role_arrays(self))
        return {"node_count": nodes, "leaf_count": leaves,
                "core_leaf_count": int(core.sum()),
                "split_frozen_after_public_fit": True,
                "core_result_report": getattr(result, "report", None),
                "core_mask_sha256": sha256(core.tobytes()).hexdigest()}

    def bounds(self) -> torch.Tensor:
        return MIN_RESIDUAL_BOUND + (MAX_RESIDUAL_BOUND - MIN_RESIDUAL_BOUND) * torch.sigmoid(
            self.leaf_bound_logits)

    def _relation_features(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        own = encoded[..., list(_POSITION_COLUMNS[:2])]
        partner = encoded[..., list(_POSITION_COLUMNS[2:])]
        relative = own - partner
        gate_input = torch.cat((encoded, relative, relative.abs()), dim=-1)
        move = torch.as_tensor(_MOVE, dtype=encoded.dtype, device=encoded.device)
        target_gap = own.unsqueeze(-2) + move - partner.unsqueeze(-2)
        geometry = torch.cat((relative.unsqueeze(-2).expand_as(target_gap),
                              target_gap, target_gap.abs()), dim=-1)
        repeated = encoded.unsqueeze(-2).expand(encoded.shape[:-1] +
                                                 (len(ACTIONS), FEATURE_DIM))
        eye = torch.eye(len(ACTIONS), dtype=encoded.dtype, device=encoded.device)
        action = eye.expand(encoded.shape[:-1] + (len(ACTIONS), len(ACTIONS)))
        candidate = action_features_from_tree_encoded(
            encoded.detach().cpu().numpy())
        candidate_torch = torch.as_tensor(candidate, dtype=encoded.dtype,
                                          device=encoded.device)
        return gate_input, torch.cat((repeated, geometry, action,
                                      candidate_torch), dim=-1)

    def components(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tree_logits, path = self.tree(encoded)
        learned_bound = path @ self.bounds()
        gate_input, relation = self._relation_features(encoded)
        gate = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        core = (path.detach() @ self.core_leaf_mask) > 0.5
        bound = torch.where(core,
                            CORE_LOW_GATE_BOUND + gate * (
                                learned_bound - CORE_LOW_GATE_BOUND),
                            learned_bound)
        score = self.action_residual(relation).squeeze(-1)
        bounded_residual = (gate * bound).unsqueeze(-1) * torch.tanh(score)
        return tree_logits, bounded_residual, bound

    def gate_values(self, encoded: torch.Tensor) -> torch.Tensor:
        gate_input, _ = self._relation_features(encoded)
        return torch.sigmoid(self.gate(gate_input).squeeze(-1))

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
        """Return [batch, action] logits.

        Joint inputs have [batch, 2, observation] shape.  An input with
        [batch, observation] shape is already role-specific.  Use a leading
        batch dimension for a single joint frame to avoid ambiguity with a
        two-row batch of role-specific observations.
        """
        if role not in AGENTS:
            raise ValueError("unknown Warehouse v6 role")
        encoded = policy_features_torch(obs, mask)
        # A joint pre-action batch has [rows, role, feature] shape.  Select
        # the requested role for both logits and the physical mask; applying
        # a single RoleActor to the whole joint tensor produces [rows, 2, 5]
        # and falsely treats the partner's observation as this role's own.
        if encoded.ndim >= 3:
            if (encoded.shape[-2] != len(AGENTS)
                    or mask.shape[-2] != len(AGENTS)):
                raise ValueError("Warehouse v6 joint observation requires two roles")
            index = AGENTS.index(role)
            encoded, mask = encoded[..., index, :], mask[..., index, :]
        return self.actors[role](encoded).masked_fill(mask <= 0.5, -1.0e9)


def role_hashes(model: WarehouseV6Actor) -> dict[str, str]:
    return {role: role_sha256(_role_arrays(model.actors[role])) for role in AGENTS}


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
        "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
        "agent_ids": list(AGENTS), "action_names": list(ACTIONS),
        "tree_depth": TREE_DEPTH, "tree_leaves": TREE_LEAVES,
        "max_split_features": MAX_SPLIT_FEATURES,
        "minimum_residual_bound": MIN_RESIDUAL_BOUND,
        "maximum_residual_bound": MAX_RESIDUAL_BOUND,
        "core_low_gate_bound": CORE_LOW_GATE_BOUND,
        "core_bound_mode": CORE_BOUND_MODE,
        "relation_geometry_dim": GEOMETRY_DIM,
        "action_feature_dim": ACTION_FEATURE_DIM,
        "gate_width": GATE_WIDTH,
        "hard_split_routing_version": HARD_SPLIT_ROUTING_VERSION,
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
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256}
