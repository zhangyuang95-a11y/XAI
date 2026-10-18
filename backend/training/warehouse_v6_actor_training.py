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
from env.warehouse.public_action_features_v617 import (
    ACTION_FEATURE_DIM, ACTION_PHYSICS_CONFIG_SHA256,
    action_features_from_tree_encoded,
    encode_actor_tree_features,
)
from .warehouse_v6_actor import (
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, MODEL_VERSION,
    V617_MODEL_VERSION, V617_CORE_BOUND_MODE,
    FEATURE_DIM, FEATURE_SCHEMA_SHA256,
    TREE_DEPTH, TREE_LEAVES, TREE_NODES, MAX_SPLIT_FEATURES, RESIDUAL_WIDTH,
    MIN_RESIDUAL_BOUND, MAX_RESIDUAL_BOUND, CORE_LOW_GATE_BOUND,
    CORE_BOUND_MODE, CORE_MARGIN_FLOOR, PROTECTED_RESIDUAL_CAP,
    GATE_WIDTH, GEOMETRY_DIM,
    _POSITION_COLUMNS, _MOVE, TREE_GRADIENT_SCALE,
    _TENSOR_NAMES, _role_arrays, role_sha256,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, HARD_SPLIT_TIE_EPSILON,
    explicit_mask_branches,
    split_scores as hard_tree_split_scores,
)
from env.warehouse.public_action_features_v614 import TREE_MASK_START
from env.warehouse.public_action_features_v617 import CONTEXT_INDEX

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

    def __init__(self, *, model_version: str = MODEL_VERSION):
        super().__init__()
        self.model_version = model_version
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
        # New checkpoints retain the support selected at each scheduled
        # refresh. Legacy state_dicts are populated on load below.
        self.register_buffer("support_mask", torch.zeros(TREE_NODES, FEATURE_DIM,
                                                         dtype=torch.bool))
        self.register_buffer("support_fixed", torch.tensor([False]))
        self.register_buffer("guard_node_mask", torch.zeros(TREE_NODES,
                                                            dtype=torch.bool))
        nn.init.normal_(self.split.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.split.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.025)
        self.split.weight.register_hook(self._mask_guard_weight_gradient)
        self.split.bias.register_hook(self._mask_guard_bias_gradient)

    def _mask_guard_weight_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient.masked_fill(self.guard_node_mask[:, None], 0.0)

    def _mask_guard_bias_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient.masked_fill(self.guard_node_mask, 0.0)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # v6.14 checkpoints contain the same exported NN tensors but lack
        # the v6.16 training-only support and guard buffers.
        weight = state_dict.get(prefix + "split.weight")
        if weight is not None:
            selected = torch.argsort(-weight.abs(), dim=-1, stable=True)[
                :, :MAX_SPLIT_FEATURES]
            support = torch.zeros_like(weight, dtype=torch.bool).scatter_(
                -1, selected, True)
            state_dict.setdefault(prefix + "support_mask", support)
        state_dict.setdefault(prefix + "support_fixed", torch.tensor([False]))
        state_dict.setdefault(prefix + "guard_node_mask", torch.zeros(
            TREE_NODES, dtype=torch.bool))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @staticmethod
    def sparse_support(weights: torch.Tensor) -> torch.Tensor:
        """A derived risk bit costs its two underlying public features."""
        selected = torch.argsort(-weights.abs(), dim=-1, stable=True)[
            :, :MAX_SPLIT_FEATURES]
        support = torch.zeros_like(weights, dtype=torch.bool).scatter_(
            -1, selected, True)
        costly = support[:, CONTEXT_INDEX] & (weights[:, CONTEXT_INDEX] != 0)
        drop = torch.where(selected[:, -1] == CONTEXT_INDEX,
                           selected[:, -2], selected[:, -1])
        support[torch.arange(len(weights), device=weights.device)[costly], drop[costly]] = False
        return support

    def effective_weights(self) -> torch.Tensor:
        weights = self.split.weight
        dynamic = self.sparse_support(weights)
        hard = torch.where(self.support_fixed[0], self.support_mask,
                           dynamic).to(weights.dtype)
        # Exactly three features affect hard routing. A deterministic local
        # straight-through surrogate reaches unselected weights, allowing a
        # feature to enter at the next scheduled support refresh.
        surrogate = 0.1 * weights
        if self.model_version == MODEL_VERSION:
            # Group the zero-valued surrogate before adding it. Left-to-right
            # w+s-s can round selected coefficients by one ULP and thereby
            # change hard routing near a fitted threshold.
            return weights * hard + (surrogate - surrogate.detach())
        return weights * hard + surrogate - surrogate.detach()

    def refresh_sparse_support(self) -> dict:
        """Lock the current top-three support without discarding alternatives."""
        with torch.no_grad():
            support = self.sparse_support(self.split.weight)
            self.support_mask.copy_(support)
            self.support_fixed[0] = True
            mask_indices = tuple(range(TREE_MASK_START,
                                       TREE_MASK_START + len(ACTIONS)))
            effective = (self.split.weight * support).cpu().numpy()
            bias = self.split.bias.cpu().numpy()
            for node in range(int(self.node_count[0])):
                lhs, rhs = explicit_mask_branches(
                    effective[node], float(bias[node]), mask_indices)
                if lhs or rhs:
                    self.guard_node_mask[node] = True
        selected_rows = [torch.nonzero(row, as_tuple=True)[0].tolist()
                         for row in self.support_mask[:int(self.node_count[0])]]
        support_bytes = self.support_mask.cpu().numpy().astype(np.uint8).tobytes()
        return {"support_sha256": sha256(support_bytes).hexdigest(),
                "selected_features": selected_rows,
                "guard_node_count": int(self.guard_node_mask[:int(
                    self.node_count[0])].sum()),
                "node_count": int(self.node_count[0]),
                "leaf_count": int(self.leaf_count[0])}

    def set_topology_trainable(self, enabled: bool) -> None:
        self.split.weight.requires_grad_(enabled)
        self.split.bias.requires_grad_(enabled)

    def path(self, encoded: torch.Tensor) -> torch.Tensor:
        values = encoded.reshape(-1, FEATURE_DIM)
        weights = self.effective_weights()
        split = nn.functional.linear(values, weights, self.split.bias)
        # Route from the exact same detached split score function used by
        # frozen Actor inference and certificate programs. The float32 score
        # above remains the straight-through surrogate for gradients.
        hard_split = torch.as_tensor(hard_tree_split_scores(
            values.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            self.split.bias.detach().cpu().numpy()), device=values.device)
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
            self.guard_node_mask.zero_()
            self.support_fixed[0] = False
        self.set_topology_trainable(True)
        return {"visited_source_leaves": len(occupied),
                "node_count": node_count, "leaf_count": leaf_count,
                "leaf_purity": purity.tolist(),
                "nonzero_split_features": int(np.count_nonzero(weights))}


class RoleActor(nn.Module):
    def __init__(self, *, model_version: str = MODEL_VERSION):
        super().__init__()
        if model_version not in (MODEL_VERSION, V617_MODEL_VERSION):
            raise ValueError("Unsupported Warehouse training Actor version")
        self.model_version = model_version
        self.core_bound_mode = (CORE_BOUND_MODE if model_version == MODEL_VERSION
                                else V617_CORE_BOUND_MODE)
        self.tree = SparseHardTree(model_version=model_version)
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
        # Retained in training checkpoints for compatibility, never used or
        # exported by the v6.18 Actor.
        self.leaf_bound_logits = nn.Parameter(torch.full((TREE_LEAVES,), -0.9),
                                              requires_grad=model_version != MODEL_VERSION)
        # Qualification is a training-only sidecar for v6.18; historical
        # v6.17 dispatch alone consumes the protected mask in its forward.
        self.register_buffer("core_leaf_mask", torch.zeros(TREE_LEAVES))
        self.register_buffer("protected_leaf_mask", torch.zeros(TREE_LEAVES))
        # Only the certified rows are locked at the 75% topology freeze. Keep
        # the hook attached across checkpoint loads; the runner restores the
        # phase with set_certified_core_logits_trainable().
        self._certified_core_logits_trainable = True
        self.tree.leaf_logits.register_hook(self._core_leaf_logit_gradient)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0 if model_version == MODEL_VERSION else -0.7)
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
            self.protected_leaf_mask.zero_()
            self.leaf_bound_logits.copy_(torch.from_numpy(
                np.log(fraction / (1.0 - fraction)).astype(np.float32)))
        return report

    def initialize_tree_from_core_result(self, result) -> dict:
        """Install a BC tree; v6.18 never inherits its protected-core verdict."""
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
            self.core_leaf_mask.zero_()
            if self.model_version != MODEL_VERSION:
                self.protected_leaf_mask.copy_(torch.from_numpy(core).to(self.protected_leaf_mask.device))
            else:
                self.protected_leaf_mask.zero_()
            # Every leaf retains a learned residual budget. In core leaves a
            # learned pre-action gate makes the effective budget small in
            # ordinary states and can release it in conflict states.
            desired = np.full(TREE_LEAVES, 6.0, dtype=np.float32)
            upper = np.full(TREE_LEAVES, MAX_RESIDUAL_BOUND, dtype=np.float32)
            fraction = np.clip((desired - MIN_RESIDUAL_BOUND) /
                               (upper - MIN_RESIDUAL_BOUND), 1e-4, 1 - 1e-4)
            self.leaf_bound_logits.copy_(torch.from_numpy(
                np.log(fraction / (1.0 - fraction)).astype(np.float32)))
            self.tree.guard_node_mask.zero_()
            mask_indices = tuple(range(TREE_MASK_START,
                                       TREE_MASK_START + len(ACTIONS)))
            for node in range(nodes):
                lhs, rhs = explicit_mask_branches(
                    values["split_weights"][node],
                    float(values["split_bias"][node]), mask_indices)
                if lhs or rhs:
                    self.tree.guard_node_mask[node] = True
        support = self.tree.refresh_sparse_support()
        if self.model_version == MODEL_VERSION:
            # BC may keep learning sparse feature support until its final
            # boundary. The caller locks support once before starting PPO.
            self.tree.support_fixed[0] = False
        for node in getattr(result, "report", {}).get("fixed_node_ids", []):
            self.tree.guard_node_mask[node] = True
        self.install_certified_core_mask(np.zeros_like(core))
        self.tree.set_topology_trainable(True)
        return {"node_count": nodes, "leaf_count": leaves,
                "core_leaf_count": int(core.sum()),
                "split_frozen_after_public_fit": False,
                "explicit_guard_node_count": support["guard_node_count"],
                "support_sha256": support["support_sha256"],
                "core_result_report": getattr(result, "report", None),
                "core_mask_sha256": sha256(core.tobytes()).hexdigest()}

    def set_topology_trainable(self, enabled: bool) -> None:
        """Train only the non-guard decision predicates before 75% freeze."""
        self.tree.set_topology_trainable(enabled)

    def _core_leaf_logit_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.model_version == MODEL_VERSION or self._certified_core_logits_trainable:
            return gradient
        return gradient.masked_fill(self.protected_leaf_mask[:, None] > 0.5, 0.0)

    def set_certified_core_logits_trainable(self, enabled: bool) -> dict:
        """Lock certified leaf outputs after topology freeze, not other leaves.

        Optimizer momentum for the locked rows must also be cleared when this
        switches to False; a zero gradient alone does not stop Adam momentum.
        """
        if self.model_version == MODEL_VERSION:
            self._certified_core_logits_trainable = True
            return {"certified_core_logits_trainable": True,
                    "frozen_leaf_count": 0}
        if not enabled:
            FrozenWarehouseV6Role(_role_arrays(self), core_bound_mode=self.core_bound_mode)
        self._certified_core_logits_trainable = bool(enabled)
        return {"certified_core_logits_trainable": bool(enabled),
                "frozen_leaf_count": (0 if enabled else
                                      int(self.protected_leaf_mask.sum().item()))}

    def refresh_sparse_support(self) -> dict:
        """Select top-three supports and invalidate all prior core claims."""
        report = self.tree.refresh_sparse_support()
        with torch.no_grad():
            invalidated = int(self.core_leaf_mask.sum().item())
            self.core_leaf_mask.zero_()
        return {**report, "invalidated_core_leaves": invalidated}

    def install_certified_core_mask(self, mask: np.ndarray | torch.Tensor) -> dict:
        """Install only leaves whose current raw action has a legal-bit path."""
        core = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor)
                          else mask, dtype=np.float32)
        leaves = int(self.tree.leaf_count[0])
        if (core.shape != (TREE_LEAVES,) or not np.isfinite(core).all()
                or np.any((core != 0.0) & (core != 1.0))
                or np.any(core[leaves:] != 0.0)):
            raise ValueError("Warehouse certified core mask invalid")
        frozen = FrozenWarehouseV6Role(_role_arrays(self), core_bound_mode=self.core_bound_mode)
        if self.model_version != MODEL_VERSION and np.any(core > self.protected_leaf_mask.detach().cpu().numpy()):
            raise ValueError("Warehouse qualified core is not a protected subset")
        with torch.no_grad():
            self.core_leaf_mask.copy_(torch.from_numpy(core).to(
                self.core_leaf_mask.device))
        return {"core_leaf_count": int(core.sum()),
                "core_mask_sha256": sha256(core.tobytes()).hexdigest(),
                "certified_actions": [sorted(actions) for actions in
                                      frozen.certified_actions()]}

    def prune_unproven_core_leaves(self) -> int:
        """Revoke core claims whose *current* raw action lost path legality.

        PPO may move leaf logits or, before 75%, a non-guard split. This
        explicit post-optimizer check changes only the qualification buffer;
        it never replaces the Actor's chosen action. Frozen inference keeps
        rejecting any unpruned invalid certificate.
        """
        if self.model_version == MODEL_VERSION:
            # No certificate participates in the v6.18 action computation.
            return 0
        weights = _role_arrays(self)
        original = weights["protected_leaf_mask"].copy()
        weights["protected_leaf_mask"] = np.zeros_like(original)
        proven = FrozenWarehouseV6Role(weights, core_bound_mode=self.core_bound_mode).certified_actions()
        raw_action = np.argmax(weights["tree.leaf_logits"], axis=-1)
        invalid = [leaf for leaf in range(len(proven))
                   if original[leaf] > 0.5 and int(raw_action[leaf]) not in proven[leaf]]
        if invalid:
            with torch.no_grad():
                self.core_leaf_mask[invalid] = 0.0
                self.protected_leaf_mask[invalid] = 0.0
        # This also checks the unmodified core leaves and the deployed tree.
        FrozenWarehouseV6Role(_role_arrays(self), core_bound_mode=self.core_bound_mode)
        return len(invalid)

    def topology_certificate_report(self,
                                    encoded: torch.Tensor | np.ndarray | None = None) -> dict:
        """Check deployed tree constraints and optional Torch/NumPy parity."""
        frozen = FrozenWarehouseV6Role(_role_arrays(self), core_bound_mode=self.core_bound_mode)
        count = int(self.tree.node_count[0])
        weights = frozen.weights["tree.split.weight"][:count]
        mask_indices = tuple(range(TREE_MASK_START,
                                   TREE_MASK_START + len(ACTIONS)))
        for node in range(count):
            lhs, rhs = explicit_mask_branches(
                weights[node], float(frozen.weights["tree.split.bias"][node]),
                mask_indices)
            if (lhs or rhs) and not bool(self.tree.guard_node_mask[node]):
                raise ValueError("Warehouse legal guard node is not immutable")
        report = {
            "node_count": count,
            "leaf_count": int(self.tree.leaf_count[0]),
            "maximum_features_per_node": int(np.count_nonzero(
                weights, axis=-1).max(initial=0)),
            "explicit_guard_node_count": int(self.tree.guard_node_mask[:count].sum()),
            "core_leaf_count": int(self.core_leaf_mask.sum()),
            "role_sha256": frozen.role_sha256,
        }
        if encoded is not None:
            rows = torch.as_tensor(encoded, dtype=self.tree.split.weight.dtype,
                                   device=self.tree.split.weight.device)
            if rows.shape[-1:] != (FEATURE_DIM,):
                raise ValueError("Warehouse topology audit feature dimension mismatch")
            with torch.no_grad():
                torch_leaves = self.tree.path(rows).argmax(-1).cpu().numpy()
                torch_logits = self(rows).cpu().numpy()
            numpy_rows = rows.detach().cpu().numpy()
            numpy_leaves = frozen.leaf(numpy_rows)
            numpy_logits = frozen.logits_encoded(numpy_rows)
            mismatches = int(np.count_nonzero(torch_leaves != numpy_leaves))
            maximum_delta = float(np.max(np.abs(torch_logits - numpy_logits),
                                         initial=0.0))
            report.update({"routing_mismatches": mismatches,
                           "maximum_logit_abs_difference": maximum_delta,
                           "audited_rows": int(np.prod(rows.shape[:-1]))})
            if mismatches or maximum_delta > 1e-4:
                raise ValueError("Warehouse topology Torch/NumPy parity failed")
        return report

    def bounds(self) -> torch.Tensor:
        if self.model_version == MODEL_VERSION:
            return torch.ones_like(self.leaf_bound_logits)
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
        if self.model_version == MODEL_VERSION:
            gate_input, relation = self._relation_features(encoded)
            gate = torch.tanh(self.gate(gate_input).squeeze(-1))
            branch = gate.unsqueeze(-1) * (tree_logits - tree_logits.mean(-1, keepdim=True))
            primary = self.action_residual(relation).squeeze(-1)
            return branch, primary, torch.ones_like(gate)
        learned_bound = path @ self.bounds()
        gate_input, relation = self._relation_features(encoded)
        gate = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        core = (path.detach() @ self.protected_leaf_mask) > 0.5
        physical = encoded[..., TREE_MASK_START:TREE_MASK_START + len(ACTIONS)]
        masked_tree = tree_logits.masked_fill(physical <= 0.5, -1.0e9)
        top_two = masked_tree.topk(2, dim=-1).values
        margin_cap = ((top_two[..., 0] - top_two[..., 1] -
                       CORE_MARGIN_FLOOR) / 2.0).clamp_min(0.0)
        effective = torch.where(core,
                                torch.minimum(torch.minimum(gate * learned_bound, margin_cap),
                                              torch.full_like(margin_cap, PROTECTED_RESIDUAL_CAP)),
                                gate * learned_bound)
        score = self.action_residual(relation).squeeze(-1)
        bounded_residual = effective.unsqueeze(-1) * torch.tanh(score)
        return tree_logits, bounded_residual, learned_bound

    def gate_values(self, encoded: torch.Tensor) -> torch.Tensor:
        gate_input, _ = self._relation_features(encoded)
        logits = self.gate(gate_input).squeeze(-1)
        return torch.tanh(logits) if self.model_version == MODEL_VERSION else torch.sigmoid(logits)

    def set_tree_branch_enabled(self, enabled: bool) -> None:
        """BC schedule switch implemented entirely by trainable gate weights.

        Disable is allowed only while the final gate layer is exactly zero;
        it never silently changes a trained Actor or adds a runtime flag.
        """
        if self.model_version != MODEL_VERSION:
            raise ValueError("Tree branch scheduling requires the v6.18 Actor")
        if not enabled and (torch.count_nonzero(self.gate[-1].weight).item()
                            or torch.count_nonzero(self.gate[-1].bias).item()):
            raise ValueError("Cannot disable a nonzero learned tree branch")
        for parameter in self.gate.parameters():
            parameter.requires_grad_(bool(enabled))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        tree_logits, residual, _ = self.components(encoded)
        return tree_logits + residual


class WarehouseV6Actor(nn.Module):
    """Two independent Actors with centralized reward and shutdown critics."""

    def __init__(self, *, model_version: str = MODEL_VERSION):
        super().__init__()
        self.model_version = model_version
        self.actors = nn.ModuleDict({role: RoleActor(model_version=model_version)
                                    for role in AGENTS})
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
    return {role: role_sha256(_role_arrays(model.actors[role]),
                              model_version=model.actors[role].model_version)
            for role in AGENTS}


def export_actor(model: WarehouseV6Actor, path: str | Path, *, pair_seed: int,
                 variant: str, candidate_id: str) -> dict:
    if variant not in ("raw", "regularized"):
        raise ValueError("invalid Warehouse v6 variant")
    if any(model.actors[role].model_version != MODEL_VERSION for role in AGENTS):
        raise ValueError("Only the current Warehouse Actor can be newly exported")
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
        "core_bound_mode": CORE_BOUND_MODE,
        "actor_formula": "nn256_logits+tanh(gate)*center(tree_logits)",
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
