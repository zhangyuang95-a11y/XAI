"""Fit a sparse, train-only core for the Warehouse dual neural Actor.

This module constructs Actor weights; it never proposes an environment action.
The input is a pre-action public feature matrix and a *masked* training target
distribution.  Core leaves are selected on those training episodes alone.
Independent program fit, calibration, eligibility, and audit remain separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np
from .warehouse_v6_tree_routing import (
    certified_feature_bounds, certified_mask_branches,
    leaf_indices as hard_tree_leaf_indices, validate_certification_domain,
)

from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v614 import (
    TREE_FEATURE_DIM as FEATURE_DIM, TREE_FEATURE_NAMES as FEATURE_NAMES,
)


MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_NODE_FEATURES = 3
_MASK_INDICES = tuple(FEATURE_NAMES.index(f"legal_{action.lower()}")
                      for action in ACTIONS)
_EPS = 1.0e-9


@dataclass(frozen=True)
class CoreTreeConfig:
    min_leaf_rows: int = 80
    min_leaf_episodes: int = 24
    min_training_agreement: float = 0.96
    max_training_kl: float = 0.20
    min_masked_top_margin: float = 0.26
    max_episode_fraction: float = 0.20
    min_split_rows: tuple[int, ...] = (40, 80, 160)
    tree_shapes: tuple[tuple[int, int], ...] = ((4, 12), (5, 16))
    include_sparse_projections: bool = True

    def __post_init__(self) -> None:
        if (self.min_leaf_rows < 1 or self.min_leaf_episodes < 1
                or not 0.0 < self.min_training_agreement <= 1.0
                or self.max_training_kl <= 0.0
                or self.min_masked_top_margin <= 0.0
                or not 0.0 < self.max_episode_fraction <= 1.0
                or not self.min_split_rows
                or any(value < 1 for value in self.min_split_rows)
                or not self.tree_shapes
                or any(not 1 <= depth <= MAX_DEPTH or not 2 <= leaves <= MAX_LEAVES
                       for depth, leaves in self.tree_shapes)):
            raise ValueError("invalid Warehouse core tree configuration")


@dataclass(frozen=True)
class CoreTreeResult:
    split_weights: np.ndarray
    split_bias: np.ndarray
    left_refs: np.ndarray
    right_refs: np.ndarray
    node_count: int
    leaf_count: int
    leaf_logits: np.ndarray
    core_mask: np.ndarray
    report: dict
    train_episode_ids: frozenset[str]

    def __post_init__(self) -> None:
        if (self.split_weights.shape != (MAX_LEAVES - 1, FEATURE_DIM)
                or self.split_bias.shape != (MAX_LEAVES - 1,)
                or self.left_refs.shape != (MAX_LEAVES - 1,)
                or self.right_refs.shape != (MAX_LEAVES - 1,)
                or self.leaf_logits.shape != (MAX_LEAVES, len(ACTIONS))
                or self.core_mask.shape != (MAX_LEAVES,)
                or self.split_weights.dtype != np.float32
                or self.split_bias.dtype != np.float32
                or self.leaf_logits.dtype != np.float32
                or self.left_refs.dtype != np.int64
                or self.right_refs.dtype != np.int64
                or self.core_mask.dtype != np.bool_
                or not 0 <= self.node_count <= MAX_LEAVES - 1
                or not 1 <= self.leaf_count <= MAX_LEAVES
                or self.node_count != self.leaf_count - 1
                or np.any(np.count_nonzero(self.split_weights, axis=1)
                          > MAX_NODE_FEATURES)
                or np.any(self.core_mask[self.leaf_count:])
                or not np.isfinite(self.split_weights).all()
                or not np.isfinite(self.split_bias).all()
                or not np.isfinite(self.leaf_logits).all()):
            raise ValueError("invalid Warehouse sparse core tree result")
        seen_nodes: set[int] = set()
        seen_leaves: set[int] = set()

        def visit(ref: int, depth: int) -> None:
            if depth > MAX_DEPTH:
                raise ValueError("Warehouse sparse core tree exceeds depth budget")
            if ref < 0:
                leaf = -1 - ref
                if leaf >= self.leaf_count or leaf in seen_leaves:
                    raise ValueError("Warehouse sparse core tree has invalid leaf")
                seen_leaves.add(leaf)
                return
            if ref >= self.node_count or ref in seen_nodes:
                raise ValueError("Warehouse sparse core tree has invalid node")
            seen_nodes.add(ref)
            visit(int(self.left_refs[ref]), depth + 1)
            visit(int(self.right_refs[ref]), depth + 1)

        visit(0 if self.node_count else -1, 0)
        if (seen_nodes != set(range(self.node_count))
                or seen_leaves != set(range(self.leaf_count))):
            raise ValueError("Warehouse sparse core tree contains unreachable branch")
        certificates, _ = _structural_legal_actions(self)
        if any(int(np.argmax(self.leaf_logits[leaf])) not in certificates[leaf]
               for leaf in range(self.leaf_count) if self.core_mask[leaf]):
            raise ValueError("Warehouse core leaf has no structural action legality proof")
        # A frozen dataclass alone does not protect the arrays it contains.
        # Copy them so neither a caller's source array nor a later write can
        # change the selected topology after its content hash is recorded.
        for name in ("split_weights", "split_bias", "left_refs", "right_refs",
                     "leaf_logits", "core_mask"):
            source = np.ascontiguousarray(getattr(self, name))
            sealed = np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)
            object.__setattr__(self, name, sealed)

    def content_sha256(self) -> str:
        digest = sha256()
        for item, dtype in (
            (self.split_weights, "<f4"), (self.split_bias, "<f4"),
            (self.left_refs, "<i8"), (self.right_refs, "<i8"),
            (self.leaf_logits, "<f4"), (self.core_mask, "u1"),
        ):
            digest.update(np.asarray(item.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(item, dtype=dtype).tobytes())
        digest.update(np.asarray((self.node_count, self.leaf_count),
                                 dtype="<i8").tobytes())
        return digest.hexdigest()

    def leaf_indices(self, encoded: np.ndarray) -> np.ndarray:
        x = np.asarray(encoded, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ValueError("invalid Warehouse core tree public features")
        validate_certification_domain(
            x, _MASK_INDICES, certified_feature_bounds(tuple(FEATURE_NAMES)))
        leaves = hard_tree_leaf_indices(
            x, self.split_weights, self.split_bias,
            self.left_refs, self.right_refs, self.node_count, MAX_DEPTH)
        if np.any(leaves >= self.leaf_count):
            raise RuntimeError("Warehouse sparse core tree reached invalid leaf")
        return leaves


def _validated_rows(encoded: np.ndarray, masks: np.ndarray,
                    target_probs: np.ndarray, episode_ids: Sequence[str]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(encoded, dtype=np.float32)
    m = np.asarray(masks, dtype=np.float32)
    p = np.asarray(target_probs, dtype=np.float32)
    episodes = np.asarray([str(value) for value in episode_ids])
    if (x.ndim != 2 or x.shape[1] != FEATURE_DIM or len(x) < 2
            or m.shape != (len(x), len(ACTIONS)) or p.shape != m.shape
            or episodes.shape != (len(x),)
            or any(not value for value in episodes)
            or not np.isfinite(x).all() or not np.isfinite(m).all()
            or not np.isfinite(p).all()
            or np.any((m != 0.0) & (m != 1.0))
            or np.any(m.sum(1) < 1.0)
            or not np.array_equal(x[:, _MASK_INDICES], m)
            or np.any(p < -1e-7) or np.any(p * (1.0 - m) > 1e-6)
            or not np.allclose(p.sum(1), 1.0, rtol=0.0, atol=1e-5)):
        raise ValueError("Warehouse core tree needs valid masked training distributions")
    return x, m, p, episodes


def _masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    selected = np.where(masks > 0.5, logits, -1.0e9)
    exp = np.exp(selected - selected.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _fit_leaf_logits(target: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Fit the complete masked target; invalid actions have zero likelihood."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    valid_count = masks.sum(0)
    prior = np.log((target.sum(0) + 0.2) / (valid_count + 1.0))
    prior -= prior.mean()

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        selected = np.where(masks > 0.5, z[None, :], -1.0e9)
        log_normalizer = logsumexp(selected, axis=1)
        q = np.exp(selected - log_normalizer[:, None])
        offset = z - prior
        loss = float(np.mean(log_normalizer - np.sum(target * z[None], axis=1))
                     + 0.001 * np.mean(offset * offset))
        gradient = (q - target).mean(0) + (0.002 / len(ACTIONS)) * offset
        return loss, gradient.astype(np.float64)

    optimized = minimize(objective, prior.astype(np.float64), jac=True,
                         method="L-BFGS-B", options={"maxiter": 50, "ftol": 1e-10})
    if not optimized.success and not np.isfinite(optimized.fun):
        raise RuntimeError("Warehouse core leaf masked fit failed")
    logits = optimized.x - optimized.x.mean()
    return np.asarray(logits, dtype=np.float32)


def _episode_sha(episodes: Sequence[str]) -> str:
    return sha256(json.dumps(sorted(set(episodes)), separators=(",", ":")).encode()).hexdigest()


def _data_sha(x: np.ndarray, m: np.ndarray, p: np.ndarray,
              episodes: np.ndarray) -> str:
    digest = sha256()
    for values in (x, m, p):
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
    digest.update(json.dumps(episodes.tolist(), separators=(",", ":")).encode())
    return digest.hexdigest()


def _candidate_inputs(x: np.ndarray, target: np.ndarray,
                      config: CoreTreeConfig) -> list[tuple[str, np.ndarray, np.ndarray]]:
    identity = np.eye(FEATURE_DIM, dtype=np.float32)
    candidates = [("axis", x, identity)]
    if not config.include_sparse_projections or len(np.unique(target.argmax(1))) < 2:
        return candidates
    from sklearn.linear_model import LogisticRegression

    labels = target.argmax(1)
    projections = []
    # Every projection is learned from these training rows.  It uses at most
    # three original public features when copied into one Actor tree node.
    for action in range(len(ACTIONS)):
        binary = labels == action
        if binary.sum() < 8 or (~binary).sum() < 8:
            continue
        fitted = LogisticRegression(l1_ratio=1.0, C=0.5, solver="liblinear",
                                    max_iter=80, random_state=0).fit(x, binary)
        dense = np.asarray(fitted.coef_[0], dtype=np.float32)
        selected = np.argsort(-np.abs(dense), kind="stable")[:MAX_NODE_FEATURES]
        vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        vector[selected] = dense[selected]
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
            if not any(np.allclose(vector, earlier) for earlier in projections):
                projections.append(vector)
    if projections:
        basis = np.concatenate((identity, np.stack(projections)), axis=0)
        projected = np.einsum("nf,kf->nk", x, np.stack(projections), optimize=False)
        augmented = np.concatenate((x, projected), axis=1)
        candidates.append(("sparse3", augmented.astype(np.float32), basis))
    return candidates


def _convert_tree(tree, basis: np.ndarray, x: np.ndarray, masks: np.ndarray,
                  target: np.ndarray) -> CoreTreeResult:
    weights = np.zeros((MAX_LEAVES - 1, FEATURE_DIM), dtype=np.float32)
    bias = np.zeros(MAX_LEAVES - 1, dtype=np.float32)
    left = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
    right = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
    logits = np.zeros((MAX_LEAVES, len(ACTIONS)), dtype=np.float32)
    source_leaf: dict[int, int] = {}
    node_count = 0
    leaf_count = 0

    def visit(source: int, depth: int) -> int:
        nonlocal node_count, leaf_count
        lhs, rhs = int(tree.children_left[source]), int(tree.children_right[source])
        if lhs == rhs:
            index = leaf_count
            leaf_count += 1
            source_leaf[source] = index
            return -1 - index
        if depth >= MAX_DEPTH:
            raise RuntimeError("Warehouse candidate tree exceeded depth budget")
        index = node_count
        node_count += 1
        feature = int(tree.feature[source])
        weights[index] = 4.0 * basis[feature]
        bias[index] = np.float32(-4.0 * tree.threshold[source])
        left[index] = visit(lhs, depth + 1)
        right[index] = visit(rhs, depth + 1)
        return index

    root = visit(0, 0)
    if root != (0 if node_count else -1) or leaf_count > MAX_LEAVES:
        raise RuntimeError("Warehouse candidate tree violated topology budget")
    # The source tree is used only during fit; the returned Actor route is
    # checked against it before any leaf is qualified.
    return CoreTreeResult(weights, bias, left, right, node_count, leaf_count,
                          logits, np.zeros(MAX_LEAVES, dtype=np.bool_),
                          {"source_leaf_map": source_leaf}, frozenset())


def _leaf_metrics(result: CoreTreeResult, x: np.ndarray, masks: np.ndarray,
                  target: np.ndarray, episodes: np.ndarray,
                  config: CoreTreeConfig) -> tuple[list[dict], np.ndarray, dict]:
    leaves = result.leaf_indices(x)
    all_q = _masked_probabilities(result.leaf_logits[leaves], masks)
    full_top = target.argmax(1)
    q_top = all_q.argmax(1)
    row_kl = np.sum(target * (np.log(np.maximum(target, _EPS))
                              - np.log(np.maximum(all_q, _EPS))), axis=1)
    top_values = np.sort(np.where(masks > 0.5, result.leaf_logits[leaves], -1e9),
                         axis=1)
    margins = top_values[:, -1] - top_values[:, -2]
    certified_actions, leaf_depths = _structural_legal_actions(result)
    details = []
    core = np.zeros(MAX_LEAVES, dtype=np.bool_)
    for leaf in range(result.leaf_count):
        rows = np.flatnonzero(leaves == leaf)
        if not len(rows):
            details.append({"leaf": leaf, "rows": 0, "episode_count": 0,
                            "max_episode_fraction": 0.0,
                            "target_top1_agreement": 0.0,
                            "target_masked_kl": None,
                            "minimum_masked_top_margin": 0.0,
                            "raw_illegal_actions": 0,
                            "raw_action_legal_by_tree_path": False,
                            "tree_path_depth": leaf_depths[leaf],
                            "raw_top_action": ACTIONS[int(np.argmax(result.leaf_logits[leaf]))],
                            "masked_patterns": 0, "core_eligible": False})
            continue
        by_episode = np.unique(episodes[rows], return_counts=True)[1]
        raw_top = int(np.argmax(result.leaf_logits[leaf]))
        raw_illegal = int(np.sum(masks[rows, raw_top] <= 0.5))
        structurally_legal = raw_top in certified_actions[leaf]
        agreement = float(np.mean(q_top[rows] == full_top[rows]))
        mean_kl = float(np.mean(row_kl[rows]))
        margin = float(np.min(margins[rows]))
        max_fraction = float(by_episode.max() / len(rows))
        eligible = bool(len(rows) >= config.min_leaf_rows
                        and len(by_episode) >= config.min_leaf_episodes
                        and max_fraction <= config.max_episode_fraction
                        and raw_illegal == 0
                        and structurally_legal
                        and agreement >= config.min_training_agreement
                        and mean_kl <= config.max_training_kl
                        and margin >= config.min_masked_top_margin)
        core[leaf] = eligible
        details.append({"leaf": leaf, "rows": len(rows), "episode_count": len(by_episode),
                        "max_episode_fraction": max_fraction,
                        "target_top1_agreement": agreement,
                        "target_masked_kl": mean_kl,
                        "minimum_masked_top_margin": margin,
                        "raw_illegal_actions": raw_illegal,
                        "raw_action_legal_by_tree_path": structurally_legal,
                        "tree_path_depth": leaf_depths[leaf],
                        "raw_top_action": ACTIONS[raw_top],
                        "masked_patterns": int(np.unique(masks[rows], axis=0).shape[0]),
                        "core_eligible": eligible})
    summary = {"core_rows": int(core[leaves].sum()),
               "core_training_coverage": float(core[leaves].mean()),
               "overall_target_masked_kl": float(row_kl.mean()),
               "overall_target_top1_agreement": float(np.mean(q_top == full_top)),
               "core_raw_illegal_actions": int(sum(
                   detail["raw_illegal_actions"] for detail in details
                   if detail["core_eligible"])),
               "core_leaf_count": int(core[:result.leaf_count].sum())}
    return details, core, summary


def _structural_legal_actions(result: CoreTreeResult) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Syntactic proof that a path requires a public legal-action bit to be one.

    Only a direct threshold on one binary mask bit counts.  More general
    oblique/map claims are deliberately not inferred from training examples.
    """
    certificates: dict[int, set[int]] = {}
    depths: dict[int, int] = {}

    def visit(ref: int, depth: int, known_legal: set[int]) -> None:
        if ref < 0:
            leaf = -1 - ref
            certificates[leaf] = known_legal
            depths[leaf] = depth
            return
        weights = result.split_weights[ref]
        left_legal = set(known_legal)
        right_legal = set(known_legal)
        left_cert, right_cert = certified_mask_branches(
            weights, float(result.split_bias[ref]), _MASK_INDICES,
            certified_feature_bounds(tuple(FEATURE_NAMES)))
        left_legal.update(left_cert)
        right_legal.update(right_cert)
        visit(int(result.left_refs[ref]), depth + 1, left_legal)
        visit(int(result.right_refs[ref]), depth + 1, right_legal)

    # The physical Warehouse mask defines WAIT as legal for active and inactive
    # robots alike.  Movement actions require an explicit tree-path mask gate.
    visit(0 if result.node_count else -1, 0, {ACTIONS.index("WAIT")})
    return certificates, depths


def _insert_legality_splits(candidate: CoreTreeResult, x: np.ndarray,
                            masks: np.ndarray, target: np.ndarray,
                            episodes: np.ndarray, config: CoreTreeConfig
                            ) -> CoreTreeResult:
    """Use spare tree capacity to separate legal from illegal raw action rows.

    The legal child retains the fitted leaf index; the other child is an
    ordinary residual-capable Actor leaf.  This is a learned topology change,
    never an execution-time action selection or mask rewrite.
    """
    weights = candidate.split_weights.copy()
    bias = candidate.split_bias.copy()
    left = candidate.left_refs.copy()
    right = candidate.right_refs.copy()
    logits = candidate.leaf_logits.copy()
    node_count = candidate.node_count
    leaf_count = candidate.leaf_count
    current = candidate
    while node_count < MAX_LEAVES - 1 and leaf_count < MAX_LEAVES:
        leaves = current.leaf_indices(x)
        certified, depths = _structural_legal_actions(current)
        options = []
        for leaf in range(leaf_count):
            if depths[leaf] >= MAX_DEPTH:
                continue
            for action in range(len(ACTIONS)):
                if action in certified[leaf]:
                    continue
                rows = np.flatnonzero((leaves == leaf) & (masks[:, action] > 0.5))
                if (len(rows) < config.min_leaf_rows
                        or len(np.unique(episodes[rows])) < config.min_leaf_episodes
                        or np.mean(target[rows].argmax(1) == action)
                        < config.min_training_agreement):
                    continue
                fitted = _fit_leaf_logits(target[rows], masks[rows])
                if int(np.argmax(fitted)) != action:
                    continue
                q = _masked_probabilities(
                    np.broadcast_to(fitted, target[rows].shape), masks[rows])
                agreement = float(np.mean(q.argmax(1) == target[rows].argmax(1)))
                kl = float(np.mean(np.sum(target[rows] * (
                    np.log(np.maximum(target[rows], _EPS))
                    - np.log(np.maximum(q, _EPS))), axis=1)))
                scores = np.sort(np.where(masks[rows] > 0.5, fitted, -1e9), axis=1)
                margin = float(np.min(scores[:, -1] - scores[:, -2]))
                if (agreement < config.min_training_agreement
                        or kl > config.max_training_kl
                        or margin < config.min_masked_top_margin):
                    continue
                options.append((len(rows), leaf, action, fitted))
        if not options:
            break
        _, leaf, action, fitted = max(options,
                                      key=lambda item: (item[0], -item[1], -item[2]))
        old_ref = -1 - leaf
        if node_count == 0:
            parent = None
        else:
            parent = next(((node, side) for node in range(node_count)
                           for side, refs in (("left", left), ("right", right))
                           if refs[node] == old_ref), None)
            if parent is None:
                raise RuntimeError("Warehouse core leaf has no parent")
        new_node = node_count
        new_leaf = leaf_count
        node_count += 1
        leaf_count += 1
        weights[new_node, _MASK_INDICES[action]] = 4.0
        bias[new_node] = -2.0
        left[new_node] = -1 - new_leaf
        right[new_node] = old_ref
        if parent is not None:
            node, side = parent
            (left if side == "left" else right)[node] = new_node
        old_rows = np.flatnonzero(leaves == leaf)
        legal_rows = old_rows[masks[old_rows, action] > 0.5]
        illegal_rows = old_rows[masks[old_rows, action] <= 0.5]
        logits[leaf] = fitted
        logits[new_leaf] = (_fit_leaf_logits(target[illegal_rows], masks[illegal_rows])
                            if len(illegal_rows) else candidate.leaf_logits[leaf])
        current = CoreTreeResult(weights.copy(), bias.copy(), left.copy(), right.copy(),
                                 node_count, leaf_count, logits.copy(),
                                 np.zeros(MAX_LEAVES, dtype=np.bool_), {}, frozenset())
    return current


def fit_core_tree(encoded: np.ndarray, masks: np.ndarray,
                  target_probs: np.ndarray, episode_ids: Sequence[str], *,
                  target_provenance: str,
                  config: CoreTreeConfig = CoreTreeConfig()) -> CoreTreeResult:
    """Select a train-only sparse tree by certifiable core mass, then target KL.

    `target_provenance` is recorded in the result.  Callers must supply only
    BC/DAgger training episodes, never program qualification or heldout rows.
    """
    if target_provenance not in ("public_teacher_one_hot",
                                 "frozen_nn_masked_distribution"):
        raise ValueError("Warehouse core target provenance must be explicit")
    x, masks, target, episodes = _validated_rows(
        encoded, masks, target_probs, episode_ids)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    labels = target.argmax(1)
    seed = int.from_bytes(sha256(_data_sha(x, masks, target, episodes).encode()).digest()[:4],
                          "little")
    fitted = []
    rejected = []
    for input_kind, features, basis in _candidate_inputs(x, target, config):
        for min_split in config.min_split_rows:
            for maximum_depth, maximum_leaves in config.tree_shapes:
                for family in ("classification", "distribution"):
                    if family == "classification":
                        learner = DecisionTreeClassifier(
                            criterion="log_loss", max_depth=maximum_depth,
                            max_leaf_nodes=maximum_leaves, min_samples_leaf=min_split,
                            random_state=seed)
                        learner.fit(features, labels,
                                    sample_weight=np.maximum(target.max(1), 0.05))
                    else:
                        learner = DecisionTreeRegressor(
                            criterion="squared_error", max_depth=maximum_depth,
                            max_leaf_nodes=maximum_leaves, min_samples_leaf=min_split,
                            random_state=seed)
                        learner.fit(features, target)
                    candidate = _convert_tree(learner.tree_, basis, x, masks, target)
                    source_leaves = learner.apply(features)
                    map_to_leaf = candidate.report["source_leaf_map"]
                    fitted_logits = candidate.leaf_logits.copy()
                    for source, leaf in map_to_leaf.items():
                        rows = source_leaves == source
                        if not np.any(rows):
                            raise RuntimeError("Warehouse core candidate contains empty source leaf")
                        fitted_logits[leaf] = _fit_leaf_logits(target[rows], masks[rows])
                    candidate = CoreTreeResult(
                        candidate.split_weights, candidate.split_bias,
                        candidate.left_refs, candidate.right_refs,
                        candidate.node_count, candidate.leaf_count,
                        fitted_logits, candidate.core_mask,
                        candidate.report, frozenset())
                    if not np.array_equal(
                            candidate.leaf_indices(x),
                            np.asarray([map_to_leaf[int(value)] for value in source_leaves])):
                        rejected.append({
                            "input_kind": input_kind,
                            "min_samples_leaf": min_split,
                            "family": family,
                            "fitted_max_depth": maximum_depth,
                            "fitted_max_leaves": maximum_leaves,
                            "reason": "float32_export_route_mismatch",
                        })
                        continue
                    candidate = _insert_legality_splits(
                        candidate, x, masks, target, episodes, config)
                    details, core, summary = _leaf_metrics(
                        candidate, x, masks, target, episodes, config)
                    identity = {"input_kind": input_kind, "min_samples_leaf": min_split,
                                "family": family, "fitted_max_depth": maximum_depth,
                                "fitted_max_leaves": maximum_leaves,
                                "node_count": candidate.node_count,
                                "leaf_count": candidate.leaf_count, **summary}
                    fitted.append((candidate, core, details, identity))
    if not fitted:
        raise RuntimeError("all Warehouse sparse core candidates changed route after export")
    # No qualification set enters this ranking.  The full masked target is
    # used for every leaf, including leaves that never become protected.
    choice = max(fitted, key=lambda item: (
        item[3]["core_training_coverage"],
        -item[3]["overall_target_masked_kl"],
        item[3]["overall_target_top1_agreement"],
        -item[3]["leaf_count"],
        item[3]["input_kind"] == "axis",
        -item[3]["min_samples_leaf"],
        item[3]["family"] == "classification"))
    candidate, core, details, identity = choice
    config_json = {key: getattr(config, key) for key in config.__dataclass_fields__}
    report = {
        "version": "warehouse-v6-core-tree-fit.v1",
        "target_provenance": target_provenance,
        "source": "pre_action_public_features_only",
        "train_episode_count": len(set(episodes)),
        "train_episode_ids_sha256": _episode_sha(episodes),
        "train_data_sha256": _data_sha(x, masks, target, episodes),
        "training_rows": len(x),
        "max_depth": MAX_DEPTH,
        "max_leaves": MAX_LEAVES,
        "max_features_per_split": MAX_NODE_FEATURES,
        "config": config_json,
        "chosen_candidate": identity,
        "candidate_diagnostics": [item[3] for item in fitted],
        "rejected_candidate_diagnostics": rejected,
        "leaf_diagnostics": details,
        "execution_action_selection": False,
        "program_qualification_episodes_used": False,
    }
    result = CoreTreeResult(candidate.split_weights, candidate.split_bias,
                            candidate.left_refs, candidate.right_refs,
                            candidate.node_count, candidate.leaf_count,
                            candidate.leaf_logits, core, report,
                            frozenset(episodes.tolist()))
    report["tree_content_sha256"] = result.content_sha256()
    return result


def evaluate_core_tree(result: CoreTreeResult, encoded: np.ndarray,
                       masks: np.ndarray, target_probs: np.ndarray,
                       episode_ids: Sequence[str]) -> dict:
    """Read-only diagnostic on episodes disjoint from the fitting episodes."""
    x, m, p, episodes = _validated_rows(encoded, masks, target_probs, episode_ids)
    if result.train_episode_ids.intersection(episodes.tolist()):
        raise ValueError("Warehouse core diagnostic overlaps fitting episodes")
    leaves = result.leaf_indices(x)
    selected = result.core_mask[leaves]
    q = _masked_probabilities(result.leaf_logits[leaves], m)
    top = q.argmax(1)
    matched = top == p.argmax(1)
    raw_top = result.leaf_logits[leaves].argmax(1)
    illegal = m[np.arange(len(x)), raw_top] <= 0.5
    kl = np.sum(p * (np.log(np.maximum(p, _EPS))
                     - np.log(np.maximum(q, _EPS))), axis=1)
    return {"episode_count": len(set(episodes)),
            "episode_ids_sha256": _episode_sha(episodes),
            "rows": len(x), "core_rows": int(selected.sum()),
            "core_coverage": float(selected.mean()),
            "core_agreement": float(matched[selected].mean()) if selected.any() else 0.0,
            "core_masked_kl": float(kl[selected].mean()) if selected.any() else None,
            "core_raw_illegal_actions": int(illegal[selected].sum()),
            "all_masked_illegal_actions": int(np.sum(m[np.arange(len(x)), top] <= 0.5)),
            "selection_changed": False}
