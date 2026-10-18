"""Fit-only construction of a depth-five risk-stratified neural tree.

No independent qualification is inferred here. Protected leaves are Actor
architecture parameters, not explanation certificates.
"""
from hashlib import sha256
import numpy as np

from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import (
    TREE_FEATURE_NAMES, TREE_FEATURE_DIM, CONTEXT_INDEX, NEAR_CONTEXT_INDEX,
)
from .warehouse_v6_core_tree import CoreTreeResult
from .warehouse_v614_certificate_tree import (
    CertificateSearchConfig, FeatureSchema, _projection_basis, _fit_leaf_logits,
)
from .warehouse_v6_tree_routing import split_scores, HARD_SPLIT_TIE_EPSILON

PROTECTED_RESIDUAL_CAP = 0.08
PROTECTED_MARGIN_TARGET = 3.0
PROGRAM_LOGIT_BOX = 0.04
METHOD = "actor_risk_stratified_protected_topology_v3"
RISK_TREE_RECIPE = {
    "maximum_depth": 5, "maximum_leaves": 16, "learned_levels_per_context": 2,
    "maximum_movement_guards": 7, "minimum_split_rows": 80,
    "minimum_child_rows": 40, "fit_quantiles": [0.2, 0.4, 0.5, 0.6, 0.8],
    "maximum_axis_features": 18, "protected_minimum_rows": 80,
    "protected_minimum_episodes": 8, "maximum_episode_fraction": 0.20,
    "protected_residual_cap": PROTECTED_RESIDUAL_CAP,
    "protected_margin_target": PROTECTED_MARGIN_TARGET,
    "program_logit_box": PROGRAM_LOGIT_BOX,
}


def fit_boxed_logits(target, masks, center):
    """Convex masked-distribution fit in a small Actor-relative logit box."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    p, m = np.asarray(target, np.float64), np.asarray(masks, bool)
    base = np.asarray(center, np.float64)
    base = base - base.mean()
    def objective(z):
        masked = np.where(m, z[None], -1e9)
        norm = logsumexp(masked, axis=1)
        q = np.exp(masked - norm[:, None])
        return float(np.mean(norm - np.sum(p*z, axis=1))), (q-p).mean(0)
    fitted = minimize(objective, base, jac=True, method="L-BFGS-B",
                      bounds=[(float(v-PROGRAM_LOGIT_BOX), float(v+PROGRAM_LOGIT_BOX))
                              for v in base], options={"maxiter": 60, "ftol": 1e-12})
    if not np.isfinite(fitted.fun):
        raise RuntimeError("Warehouse constrained program fit is nonfinite")
    return fitted.x.astype(np.float32)


def project_logit_box(logits, center):
    base = np.asarray(center, dtype=np.float32)
    base = base - base.mean()
    value = np.asarray(logits, dtype=np.float32)
    value = value - value.mean()
    return np.clip(value, base-PROGRAM_LOGIT_BOX, base+PROGRAM_LOGIT_BOX)


def fit_risk_tree(x, masks, target, episode_ids) -> CoreTreeResult:
    x, masks, target = (np.asarray(v, dtype=np.float32) for v in (x, masks, target))
    episodes = np.asarray(episode_ids)
    if (x.ndim != 2 or x.shape[1] != TREE_FEATURE_DIM or len(x) == 0
            or masks.shape != (len(x), len(ACTIONS)) or target.shape != masks.shape
            or episodes.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isfinite(masks).all() or np.any((masks != 0) & (masks != 1))
            or np.any(masks.sum(1) < 1) or not np.array_equal(x[:, 141:146], masks)
            or not np.isfinite(target).all() or np.any(target < 0)
            or not np.allclose(target.sum(1), 1, atol=1e-5)
            or np.any(target[masks == 0] > 1e-8)):
        raise ValueError("invalid Warehouse risk-tree fit rows")
    schema = FeatureSchema(tuple(TREE_FEATURE_NAMES), tuple(ACTIONS))
    nodes, cells = [], []

    def node(weight, bias, left, right):
        idx = len(nodes)
        nodes.append([weight, bias, left, right])
        return idx

    def cell(rows, protectable):
        idx = len(cells)
        logits = (_fit_leaf_logits(target[rows], masks[rows]) if len(rows)
                  else np.zeros(len(ACTIONS), dtype=np.float32))
        cells.append({"rows": rows, "logits": logits, "protectable": protectable})
        return -1 - idx

    def impurity(rows):
        if not len(rows):
            return 0.0
        p = target[rows].mean(0)
        return float(-len(rows) * np.sum(p * np.log(np.maximum(p, 1e-8))))

    def partition(rows, depth):
        if depth == 0 or len(rows) < 80:
            return cell(rows, True)
        best, best_score = None, impurity(rows)
        for _, weight in _projection_basis(rows, x, target, schema,
                                            CertificateSearchConfig()):
            # Context predicates and legal-bit guards have their own nodes.
            if weight[CONTEXT_INDEX] != 0 or weight[NEAR_CONTEXT_INDEX] != 0:
                continue
            values = split_scores(x[rows], weight[None], np.zeros(1, np.float32))[:, 0]
            for q in (0.2, 0.4, 0.5, 0.6, 0.8):
                threshold = np.float32(np.quantile(values, q))
                right = values - float(threshold) > HARD_SPLIT_TIE_EPSILON
                lhs, rhs = rows[~right], rows[right]
                if min(len(lhs), len(rhs)) < 40:
                    continue
                score = impurity(lhs) + impurity(rhs)
                if score < best_score - 1e-7:
                    best_score, best = score, (weight.copy(), -threshold, lhs, rhs)
        if best is None:
            return cell(rows, True)
        w, bias, lhs, rhs = best
        idx = node(w, bias, None, None)
        nodes[idx][2:] = [partition(lhs, depth - 1), partition(rhs, depth - 1)]
        return idx

    root_weight = np.zeros(TREE_FEATURE_DIM, np.float32)
    root_weight[CONTEXT_INDEX] = 1
    root = node(root_weight, -0.5, None, None)
    risk = x[:, CONTEXT_INDEX] > 0.5
    complex_ref = cell(np.flatnonzero(risk), False)
    near_weight = np.zeros(TREE_FEATURE_DIM, np.float32)
    near_weight[NEAR_CONTEXT_INDEX] = -1
    near_node = node(near_weight, 0.5, None, None)
    safe = np.flatnonzero(~risk)
    far = x[safe, NEAR_CONTEXT_INDEX] < 0.5
    nodes[near_node][2:] = [partition(safe[~far], 2), partition(safe[far], 2)]
    nodes[root][2:] = [near_node, complex_ref]

    # At most seven movement cells can receive a guard: 1 risk leaf +
    # 8 decision cells + 7 illegal-side leaves = 16. WAIT needs no guard.
    movements = [i for i, c in enumerate(cells) if c["protectable"]
                 and int(c["logits"].argmax()) != ACTIONS.index("WAIT")]
    movements.sort(key=lambda i: (-len(cells[i]["rows"]), i))
    guarded = set(movements[:7])
    guard_nodes = {root, near_node}
    protected = np.zeros(16, dtype=bool)
    for leaf in range(len(cells)):
        c = cells[leaf]
        if not c["protectable"]:
            continue
        action = int(c["logits"].argmax())
        rows = c["rows"]
        if leaf in guarded:
            legal_idx = TREE_FEATURE_NAMES.index(f"legal_{ACTIONS[action].lower()}")
            legal = masks[rows, action] > 0.5
            legal_rows, illegal_rows = rows[legal], rows[~legal]
            fallback = cell(illegal_rows, False)
            w = np.zeros(TREE_FEATURE_DIM, np.float32)
            w[legal_idx] = 1
            guard = node(w, -0.5, fallback, -1-leaf)
            guard_nodes.add(guard)
            # Replace the old cell reference, excluding its new guard.
            for n in nodes[:guard]:
                for side in (2, 3):
                    if n[side] == -1-leaf:
                        n[side] = guard
            c["rows"] = rows = legal_rows
        elif action != ACTIONS.index("WAIT"):
            continue
        _, counts = np.unique(episodes[rows], return_counts=True)
        supported = (len(rows) >= 80 and len(counts) >= 8
                     and counts.max(initial=0) / max(1, len(rows)) <= 0.20)
        if supported:
            protected[leaf] = True
            # The designated action is learned from full-NN fit targets.
            # A large initial margin provides stochastic feedback headroom.
            other = np.max(np.delete(c["logits"], action))
            c["logits"][action] = max(c["logits"][action], other + PROTECTED_MARGIN_TARGET)
    weight = np.zeros((15, TREE_FEATURE_DIM), np.float32)
    bias, left, right = np.zeros(15, np.float32), np.zeros(15, np.int64), np.zeros(15, np.int64)
    logits = np.zeros((16, len(ACTIONS)), np.float32)
    for i, (w, b, l, r) in enumerate(nodes):
        weight[i], bias[i], left[i], right[i] = w, b, l, r
    for i, c in enumerate(cells):
        logits[i] = c["logits"]
    report = {"version": "warehouse-risk-stratified-tree.v6.17", "method": METHOD,
              "fit_only_not_independent_certificate": True,
              "fixed_node_ids": sorted(guard_nodes),
              "protected_leaf_ids": np.flatnonzero(protected).tolist(),
              "fit_protected_mass": sum(len(cells[i]["rows"]) for i in np.flatnonzero(protected)) / len(x),
              "risk_rows": int(risk.sum()), "rows": len(x),
              "target_probs_sha256": sha256(target.tobytes()).hexdigest()}
    return CoreTreeResult(weight, bias, left, right, len(nodes), len(cells),
                          logits, protected, report, frozenset(episode_ids))
