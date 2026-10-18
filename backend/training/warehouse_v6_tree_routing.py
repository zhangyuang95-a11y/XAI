"""One numerical contract for Warehouse learned-tree hard routing.

This module has no environment or Torch dependency. Training, frozen Actor
inference, and certificate programs all use these same split scores.
"""
from __future__ import annotations

import numpy as np


HARD_SPLIT_ROUTING_VERSION = "fp64-shared-left-tie-1e-10.v2"
HARD_SPLIT_TIE_EPSILON = 1.0e-10
UNIT_INTERVAL_FEATURE_NAMES = frozenset({
    "self_row", "self_column", "partner_row", "partner_column",
    "self_battery_fraction",
    *(f"{role}_recent_position_{index}_{axis}"
      for role in ("self", "partner") for index in range(3)
      for axis in ("row", "column")),
    *(f"order_{order}_{endpoint}_{axis}"
      for order in (0, 1) for endpoint in ("pickup", "delivery")
      for axis in ("row", "column")),
})


def unit_interval_indices(feature_names: tuple[str, ...]) -> frozenset[int]:
    """Public fractions whose supplied values are checked before proof use."""
    return frozenset(index for index, name in enumerate(feature_names)
                     if name in UNIT_INTERVAL_FEATURE_NAMES)


def certified_feature_bounds(
        feature_names: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """Conservative domains for bounded pre-action physical measurements."""
    bounds = {index: (0.0, 1.0)
              for index in unit_interval_indices(feature_names)}
    for index, name in enumerate(feature_names):
        if (name == "self_charger_return_margin"
                or name.startswith("charger_return_margin_if_")):
            bounds[index] = (-2.0, 1.0)
    return bounds


def validate_certification_domain(
        encoded: np.ndarray, mask_indices: tuple[int, ...],
        feature_bounds: dict[int, tuple[float, float]]) -> None:
    """Reject encoded rows outside the bounds used by legality certificates."""
    rows = np.asarray(encoded, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("invalid Warehouse certification features")
    masks = rows[:, mask_indices]
    if np.any((masks != 0.0) & (masks != 1.0)):
        raise ValueError("Warehouse certification feature outside proven domain")
    for index, (lower, upper) in feature_bounds.items():
        if np.any((rows[:, index] < lower) | (rows[:, index] > upper)):
            raise ValueError("Warehouse certification feature outside proven domain")


def certified_mask_branches(
        weights: np.ndarray, bias: float,
        mask_indices: tuple[int, ...],
        feature_bounds: dict[int, tuple[float, float]],
        ) -> tuple[frozenset[int], frozenset[int]]:
    """Prove a branch implies a legal mask bit by interval arithmetic.

    Every non-mask input used by the split must have a validated interval.
    With mask[a]=0, a right branch is impossible when its maximum
    score is at most the routing tie epsilon. The dual lower bound proves a
    left branch when mask[a]=0 cannot route left.
    """
    row = np.asarray(weights, dtype=np.float32)
    used = frozenset(np.flatnonzero(row).tolist())
    left, right = set(), set()
    for action, mask_index in enumerate(mask_indices):
        others_used = sorted(used - {mask_index})
        if mask_index not in used or not set(others_used) <= feature_bounds.keys():
            continue
        lower = upper = float(bias)
        for index in others_used:
            weight = float(row[index])
            low, high = feature_bounds[index]
            low_score, high_score = weight * low, weight * high
            lower += min(low_score, high_score)
            upper += max(low_score, high_score)
        if upper <= HARD_SPLIT_TIE_EPSILON:
            right.add(action)
        if lower > HARD_SPLIT_TIE_EPSILON:
            left.add(action)
    return frozenset(left), frozenset(right)


def explicit_mask_branches(
        weights: np.ndarray, bias: float,
        mask_indices: tuple[int, ...],
        ) -> tuple[frozenset[int], frozenset[int]]:
    """Certify only a dedicated physical ``legal_ACTION`` split.

    An oblique predicate may have a valid interval proof, but v6.16 requires
    an explicit, immutable legal-bit test on a movement leaf's path.  The
    stored float32 coefficients and the shared left-tie rule determine which
    child implies that the bit equals one.
    """
    row = np.asarray(weights, dtype=np.float32)
    used = np.flatnonzero(row)
    if len(used) != 1 or int(used[0]) not in mask_indices:
        return frozenset(), frozenset()
    action = mask_indices.index(int(used[0]))
    score_zero = float(np.float32(bias))
    score_one = score_zero + float(row[used[0]])
    if score_zero <= HARD_SPLIT_TIE_EPSILON < score_one:
        return frozenset(), frozenset({action})
    if score_one <= HARD_SPLIT_TIE_EPSILON < score_zero:
        return frozenset({action}), frozenset()
    return frozenset(), frozenset()


def certified_leaf_actions(
        split_weights: np.ndarray, split_bias: np.ndarray,
        left_refs: np.ndarray, right_refs: np.ndarray,
        node_count: int, leaf_count: int,
        mask_indices: tuple[int, ...], *, explicit: bool,
        feature_bounds: dict[int, tuple[float, float]] | None = None,
        wait_action: int,
        ) -> tuple[frozenset[int], ...]:
    """Return path-proven legal actions for each leaf of an irregular tree."""
    weights = np.asarray(split_weights, dtype=np.float32)
    bias = np.asarray(split_bias, dtype=np.float32)
    left = np.asarray(left_refs, dtype=np.int64)
    right = np.asarray(right_refs, dtype=np.int64)
    result: list[frozenset[int] | None] = [None] * leaf_count

    def walk(ref: int, proven: frozenset[int]) -> None:
        if ref < 0:
            leaf = -1 - ref
            if leaf < 0 or leaf >= leaf_count or result[leaf] is not None:
                raise ValueError("Warehouse certification tree leaf invalid")
            result[leaf] = proven
            return
        if ref >= node_count:
            raise ValueError("Warehouse certification tree node invalid")
        if explicit:
            left_actions, right_actions = explicit_mask_branches(
                weights[ref], float(bias[ref]), mask_indices)
        else:
            if feature_bounds is None:
                raise ValueError("Warehouse certification bounds missing")
            left_actions, right_actions = certified_mask_branches(
                weights[ref], float(bias[ref]), mask_indices, feature_bounds)
        walk(int(left[ref]), proven | left_actions)
        walk(int(right[ref]), proven | right_actions)

    walk(0 if node_count else -1, frozenset({wait_action}))
    if any(actions is None for actions in result):
        raise ValueError("Warehouse certification tree unreachable leaf")
    return tuple(actions for actions in result if actions is not None)


def split_scores(encoded: np.ndarray, weights: np.ndarray,
                 bias: np.ndarray) -> np.ndarray:
    """Compute every sparse split in float64 from stored float32 tensors."""
    x = np.asarray(encoded, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    if (x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]
            or b.shape != (len(w),) or not np.isfinite(x).all()
            or not np.isfinite(w).all() or not np.isfinite(b).all()):
        raise ValueError("invalid Warehouse hard-tree split inputs")
    scores = np.einsum("nf,kf->nk", x.astype(np.float64),
                       w.astype(np.float64), optimize=False)
    scores += b.astype(np.float64)
    return scores


def leaf_indices(encoded: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                 left_refs: np.ndarray, right_refs: np.ndarray,
                 node_count: int, depth: int) -> np.ndarray:
    """Route rows with a stable left tie, independent of their batch shape."""
    scores = split_scores(encoded, weights, bias)
    left = np.asarray(left_refs, dtype=np.int64)
    right = np.asarray(right_refs, dtype=np.int64)
    if (left.shape != (scores.shape[1],) or right.shape != left.shape
            or not 0 <= node_count <= len(left) or depth < 0):
        raise ValueError("invalid Warehouse hard-tree topology")
    refs = np.full(len(scores), 0 if node_count else -1, dtype=np.int64)
    for _ in range(depth):
        rows = np.flatnonzero(refs >= 0)
        if not len(rows):
            break
        nodes = refs[rows]
        if np.any(nodes >= node_count):
            raise RuntimeError("Warehouse hard-tree node reference is invalid")
        go_right = scores[rows, nodes] > HARD_SPLIT_TIE_EPSILON
        refs[rows] = np.where(go_right, right[nodes], left[nodes])
    if np.any(refs >= 0):
        raise RuntimeError("Warehouse hard-tree route exceeded depth")
    return -1 - refs
