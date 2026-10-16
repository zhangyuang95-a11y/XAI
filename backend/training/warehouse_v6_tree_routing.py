"""One numerical contract for Warehouse learned-tree hard routing.

This module has no environment or Torch dependency. Training, frozen Actor
inference, and certificate programs all use these same split scores.
"""
from __future__ import annotations

import numpy as np


HARD_SPLIT_ROUTING_VERSION = "fp64-shared-left-tie-1e-10.v2"
HARD_SPLIT_TIE_EPSILON = 1.0e-10


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
