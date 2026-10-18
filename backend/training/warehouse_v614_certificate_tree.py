"""Review-only, fit-purpose certificate-first sparse Warehouse tree prototype.

This module has no environment dependency.  The caller supplies pre-action
public features, the physical legal mask, and the *complete masked distribution*
of one already frozen neural Actor.  Calibration, eligibility, and validation
must remain separate from this topology search.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, certified_feature_bounds,
    certified_mask_branches,
    leaf_indices as hard_tree_leaf_indices,
    split_scores as hard_tree_split_scores,
    validate_certification_domain,
)


MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_NODE_FEATURES = 3
_EPS = 1.0e-9
_SHA = re.compile(r"[0-9a-f]{64}")

# These are the existing fit and independent-refit qualification gates.  Search
# settings below never change them.  In particular, KL uses the stricter fit
# limit, while every nonempty pre-action stratum must pass its own gate.
FIT_MIN_LEAF_ROWS = 80
FIT_MIN_LEAF_EPISODES = 24
FIT_MAX_EPISODE_FRACTION = 0.20
FIT_MIN_TOP1_AGREEMENT = 0.96
FIT_MAX_MASKED_KL = 0.20
FIT_MIN_MASKED_MARGIN = 0.26
FIT_MIN_STRATUM_ROWS = 8
FIT_MIN_STRATUM_EPISODES = 3
FIT_MIN_STRATUM_AGREEMENT = 0.93
FIT_MAX_STRATUM_MASKED_KL = 0.35
INTERNAL_MIN_CORE_MASS = 0.35
INTERNAL_MIN_EXPECTED_ACTIONABLE_MASS = 0.30


@dataclass(frozen=True)
class FeatureSchema:
    """The feature count is injected, so an expanded 171-column schema works."""

    feature_names: tuple[str, ...]
    actions: tuple[str, ...]

    def __post_init__(self) -> None:
        mask_names = tuple(f"legal_{action.lower()}" for action in self.actions)
        if (len(self.feature_names) <= len(self.actions)
                or len(set(self.feature_names)) != len(self.feature_names)
                or len(set(self.actions)) != len(self.actions)
                or "WAIT" not in self.actions
                or not set(mask_names).issubset(self.feature_names)
                or not {"same_target_collision_streak_capped",
                        "robots_shortest_distance",
                        "self_charger_return_margin"}.issubset(self.feature_names)):
            raise ValueError("invalid injected pre-action Warehouse feature schema")

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def mask_feature_indices(self) -> tuple[int, ...]:
        return tuple(self.feature_names.index(f"legal_{action.lower()}")
                     for action in self.actions)

    @property
    def nonmask_feature_indices(self) -> tuple[int, ...]:
        mask = set(self.mask_feature_indices)
        return tuple(index for index in range(self.feature_dim)
                     if index not in mask)

    def sha256(self) -> str:
        payload = {"feature_names": self.feature_names, "actions": self.actions}
        return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CertificateSearchConfig:
    """Computation budget only; no quality or coverage gate is configurable."""

    beam_width: int = 3
    max_search_leaves: int = MAX_LEAVES
    max_axis_features: int = 18
    max_exact_candidates_per_round: int = 40
    max_exact_candidates_per_leaf: int = 12
    min_noncore_child_rows: int = 20
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    pure_movement_gates_only: bool = False

    def __post_init__(self) -> None:
        if (self.beam_width < 1
                or not 2 <= self.max_search_leaves <= MAX_LEAVES
                or self.max_axis_features < 1
                or self.max_exact_candidates_per_round < 1
                or self.max_exact_candidates_per_leaf < 1
                or self.min_noncore_child_rows < 1
                or not self.quantiles
                or any(not 0.0 < q < 1.0 for q in self.quantiles)):
            raise ValueError("invalid certificate tree search budget")


@dataclass(frozen=True)
class SparseProgramTree:
    schema: FeatureSchema
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
        d, a = self.schema.feature_dim, self.schema.action_count
        if (self.split_weights.shape != (MAX_LEAVES - 1, d)
                or self.split_bias.shape != (MAX_LEAVES - 1,)
                or self.left_refs.shape != (MAX_LEAVES - 1,)
                or self.right_refs.shape != (MAX_LEAVES - 1,)
                or self.leaf_logits.shape != (MAX_LEAVES, a)
                or self.core_mask.shape != (MAX_LEAVES,)
                or self.split_weights.dtype != np.float32
                or self.split_bias.dtype != np.float32
                or self.leaf_logits.dtype != np.float32
                or self.left_refs.dtype != np.int64
                or self.right_refs.dtype != np.int64
                or self.core_mask.dtype != np.bool_
                or not 1 <= self.leaf_count <= MAX_LEAVES
                or self.node_count != self.leaf_count - 1
                or np.any(np.count_nonzero(self.split_weights, axis=1)
                          > MAX_NODE_FEATURES)
                or np.any(self.core_mask[self.leaf_count:])
                or not np.isfinite(self.split_weights).all()
                or not np.isfinite(self.split_bias).all()
                or not np.isfinite(self.leaf_logits).all()):
            raise ValueError("invalid certificate sparse tree arrays")
        seen_nodes: set[int] = set()
        seen_leaves: set[int] = set()
        certificates: dict[int, set[int]] = {}

        def visit(ref: int, depth: int, legal: set[int]) -> None:
            if depth > MAX_DEPTH:
                raise ValueError("certificate sparse tree exceeds depth five")
            if ref < 0:
                leaf = -1 - ref
                if leaf >= self.leaf_count or leaf in seen_leaves:
                    raise ValueError("invalid certificate sparse tree leaf")
                seen_leaves.add(leaf)
                certificates[leaf] = legal
                return
            if ref >= self.node_count or ref in seen_nodes:
                raise ValueError("invalid certificate sparse tree node")
            seen_nodes.add(ref)
            left_legal, right_legal = set(legal), set(legal)
            left_cert, right_cert = certified_mask_branches(
                self.split_weights[ref], float(self.split_bias[ref]),
                self.schema.mask_feature_indices,
                certified_feature_bounds(self.schema.feature_names))
            left_legal.update(left_cert)
            right_legal.update(right_cert)
            visit(int(self.left_refs[ref]), depth + 1, left_legal)
            visit(int(self.right_refs[ref]), depth + 1, right_legal)

        visit(0 if self.node_count else -1, 0,
              {self.schema.actions.index("WAIT")})
        if (seen_nodes != set(range(self.node_count))
                or seen_leaves != set(range(self.leaf_count))
                or any(int(np.argmax(self.leaf_logits[leaf])) not in certificates[leaf]
                       for leaf in range(self.leaf_count) if self.core_mask[leaf])):
            raise ValueError("certificate sparse tree lacks path legality proof")
        for name in ("split_weights", "split_bias", "left_refs", "right_refs",
                     "leaf_logits", "core_mask"):
            source = np.ascontiguousarray(getattr(self, name))
            object.__setattr__(self, name, np.frombuffer(
                source.tobytes(), dtype=source.dtype).reshape(source.shape))

    def leaf_indices(self, encoded: np.ndarray) -> np.ndarray:
        x = np.asarray(encoded, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.schema.feature_dim or not np.isfinite(x).all():
            raise ValueError("invalid pre-action features for certificate sparse tree")
        validate_certification_domain(
            x, self.schema.mask_feature_indices,
            certified_feature_bounds(self.schema.feature_names))
        return hard_tree_leaf_indices(
            x, self.split_weights, self.split_bias, self.left_refs,
            self.right_refs, self.node_count, MAX_DEPTH)

    def content_sha256(self) -> str:
        digest = sha256(self.schema.sha256().encode())
        for values, dtype in ((self.split_weights, "<f4"),
                              (self.split_bias, "<f4"),
                              (self.left_refs, "<i8"),
                              (self.right_refs, "<i8"),
                              (self.leaf_logits, "<f4"),
                              (self.core_mask, "u1")):
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
        digest.update(np.asarray((self.node_count, self.leaf_count),
                                 dtype="<i8").tobytes())
        return digest.hexdigest()

    def topology_sha256(self) -> str:
        digest = sha256(self.schema.sha256().encode())
        for values, dtype in ((self.split_weights, "<f4"),
                              (self.split_bias, "<f4"),
                              (self.left_refs, "<i8"),
                              (self.right_refs, "<i8")):
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
        digest.update(np.asarray((self.node_count, self.leaf_count),
                                 dtype="<i8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class _LeafStats:
    rows: int
    episodes: int
    raw_action: int
    raw_illegal: int
    path_legal: bool
    agreement: float
    masked_kl: float
    minimum_margin: float
    max_episode_fraction: float
    strata: dict
    core: bool
    potential: float

    def report(self, leaf: int, actions: tuple[str, ...]) -> dict:
        return {"leaf": leaf, "rows": self.rows, "episodes": self.episodes,
                "raw_action": actions[self.raw_action],
                "raw_illegal": self.raw_illegal,
                "path_legal": self.path_legal,
                "top1_agreement": self.agreement,
                "masked_kl": self.masked_kl,
                "minimum_margin": self.minimum_margin,
                "max_episode_fraction": self.max_episode_fraction,
                "strata": self.strata, "fit_core_candidate": self.core}


@dataclass
class _State:
    weights: list[np.ndarray]
    bias: list[np.float32]
    left: list[int]
    right: list[int]
    rows: list[np.ndarray]
    depths: list[int]
    legal: list[frozenset[int]]
    logits: list[np.ndarray]
    stats: list[_LeafStats]

    def core_rows(self) -> int:
        return sum(len(rows) for rows, stat in zip(self.rows, self.stats, strict=True)
                   if stat.core)

    def potential(self) -> float:
        return sum(stat.potential for stat in self.stats)

    def mean_kl(self) -> float:
        total = sum(len(rows) for rows in self.rows)
        return sum(len(rows) * stat.masked_kl
                   for rows, stat in zip(self.rows, self.stats, strict=True)) / total


@dataclass(frozen=True)
class _Split:
    weight: np.ndarray
    bias: np.float32
    left_rows: np.ndarray
    right_rows: np.ndarray
    legal_action: int | None
    quick_gain: float
    label: str


def _validated_fit_rows(
    encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
    episode_ids: Sequence[str], schema: FeatureSchema, frozen_actor_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(encoded, dtype=np.float32)
    m = np.asarray(masks, dtype=np.float32)
    p = np.asarray(target_probs, dtype=np.float32)
    episodes = np.asarray([str(value) for value in episode_ids])
    if (_SHA.fullmatch(frozen_actor_sha256) is None
            or x.ndim != 2 or x.shape[1] != schema.feature_dim or len(x) < 2
            or m.shape != (len(x), schema.action_count) or p.shape != m.shape
            or episodes.shape != (len(x),) or any(not value for value in episodes)
            or not np.isfinite(x).all() or not np.isfinite(m).all()
            or not np.isfinite(p).all()
            or np.any((m != 0.0) & (m != 1.0))
            or np.any(m.sum(1) < 1.0)
            or not np.array_equal(x[:, schema.mask_feature_indices], m)
            or np.any(p < -1.0e-7) or np.any(p * (1.0 - m) > 1.0e-6)
            or not np.allclose(p.sum(1), 1.0, rtol=0.0, atol=1.0e-5)):
        raise ValueError("certificate fitter requires one frozen Actor's full masked fit target")
    validate_certification_domain(
        x, schema.mask_feature_indices,
        certified_feature_bounds(schema.feature_names))
    return x, m, p, episodes


def _masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    selected = np.where(masks > 0.5, logits, -1.0e9)
    normalizer = logsumexp(selected, axis=-1, keepdims=True)
    return np.exp(selected - normalizer)


def _fit_leaf_logits(target: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Fit all action probabilities after physical masking, not a top1 proxy."""
    count = masks.sum(0)
    prior = np.log((target.sum(0) + 0.2) / (count + 1.0))
    prior -= prior.mean()
    actions = target.shape[1]

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        selected = np.where(masks > 0.5, z[None, :], -1.0e9)
        log_normalizer = logsumexp(selected, axis=1)
        q = np.exp(selected - log_normalizer[:, None])
        offset = z - prior
        loss = float(np.mean(log_normalizer - np.sum(target * z[None], axis=1))
                     + 0.001 * np.mean(offset * offset))
        gradient = (q - target).mean(0) + (0.002 / actions) * offset
        return loss, gradient.astype(np.float64)

    optimized = minimize(objective, prior.astype(np.float64), jac=True,
                         method="L-BFGS-B", options={"maxiter": 50, "ftol": 1e-10})
    if not np.isfinite(optimized.fun):
        raise RuntimeError("complete masked leaf distribution fit failed")
    return np.asarray(optimized.x - optimized.x.mean(), dtype=np.float32)


def _strata(x: np.ndarray, schema: FeatureSchema) -> dict[str, np.ndarray]:
    names = schema.feature_names
    conflict = x[:, names.index("same_target_collision_streak_capped")] > 0
    near = x[:, names.index("robots_shortest_distance")] * 42.0 <= 2.0
    low_return = x[:, names.index("self_charger_return_margin")] * 50.0 <= 4.0
    return {"ordinary": ~(conflict | near | low_return), "conflict": conflict,
            "near_robots": near, "low_return_margin": low_return}


def _leaf_stats(
    rows: np.ndarray, logits: np.ndarray, legal: frozenset[int], depth: int,
    x: np.ndarray, masks: np.ndarray, target: np.ndarray, episodes: np.ndarray,
    schema: FeatureSchema,
) -> _LeafStats:
    m, p = masks[rows], target[rows]
    q = _masked_probabilities(np.broadcast_to(logits, p.shape), m)
    raw = int(np.argmax(logits))
    raw_illegal = int(np.sum(m[:, raw] == 0))
    agreement = float(np.mean(q.argmax(1) == p.argmax(1)))
    row_kl = np.sum(p * (np.log(np.maximum(p, _EPS))
                         - np.log(np.maximum(q, _EPS))), axis=1)
    kl = float(np.mean(row_kl))
    masked_logits = np.where(m > 0.5, logits[None, :], -1.0e9)
    ordered = np.sort(masked_logits, axis=1)
    margin = float(np.min(ordered[:, -1] - ordered[:, -2]))
    _, counts = np.unique(episodes[rows], return_counts=True)
    max_fraction = float(counts.max() / len(rows))
    strata = {}
    strata_pass = True
    for name, indicator in _strata(x[rows], schema).items():
        chosen = np.flatnonzero(indicator)
        if not len(chosen):
            continue
        episode_count = len(set(episodes[rows[chosen]]))
        stratum_agreement = float(np.mean(q[chosen].argmax(1) == p[chosen].argmax(1)))
        stratum_kl = float(np.mean(row_kl[chosen]))
        passes = bool(len(chosen) >= FIT_MIN_STRATUM_ROWS
                      and episode_count >= FIT_MIN_STRATUM_EPISODES
                      and stratum_agreement >= FIT_MIN_STRATUM_AGREEMENT
                      and stratum_kl <= FIT_MAX_STRATUM_MASKED_KL)
        strata[name] = {"rows": len(chosen), "episodes": episode_count,
                        "agreement": stratum_agreement,
                        "masked_kl": stratum_kl, "passes": passes}
        strata_pass &= passes
    supported = bool(len(rows) >= FIT_MIN_LEAF_ROWS
                     and len(counts) >= FIT_MIN_LEAF_EPISODES
                     and max_fraction <= FIT_MAX_EPISODE_FRACTION)
    core = bool(supported and raw_illegal == 0 and raw in legal
                and agreement >= FIT_MIN_TOP1_AGREEMENT
                and kl <= FIT_MAX_MASKED_KL
                and margin >= FIT_MIN_MASKED_MARGIN and strata_pass)
    # This only orders unfinished search states.  It cannot qualify a leaf.
    action_purity = float(np.max(np.bincount(p.argmax(1),
                                             minlength=schema.action_count)) / len(rows))
    can_still_gate = depth < MAX_DEPTH and len(rows) >= FIT_MIN_LEAF_ROWS
    legality_factor = 1.0 if raw in legal else (0.65 if can_still_gate else 0.0)
    support_factor = min(1.0, len(rows) / FIT_MIN_LEAF_ROWS) * min(
        1.0, len(counts) / FIT_MIN_LEAF_EPISODES)
    potential = (0.0 if core or depth >= MAX_DEPTH else
                 len(rows) * action_purity**4 * np.exp(-2.0 * max(0.0, kl))
                 * legality_factor * support_factor)
    return _LeafStats(len(rows), len(counts), raw, raw_illegal,
                      raw in legal, agreement, kl, margin, max_fraction,
                      strata, core, float(potential))


def _projection_basis(
    rows: np.ndarray, x: np.ndarray, target: np.ndarray, schema: FeatureSchema,
    config: CertificateSearchConfig,
) -> list[tuple[str, np.ndarray]]:
    nonmask = np.asarray(schema.nonmask_feature_indices, dtype=np.int64)
    obs = x[np.ix_(rows, nonmask)].astype(np.float64)
    response = target[rows].astype(np.float64)
    centered = obs - obs.mean(0)
    target_centered = response - response.mean(0)
    scale = np.maximum(centered.std(0), 1.0e-8)
    # A direct BLAS matmul can emit spurious FP warnings on constant synthetic
    # columns on some Accelerate builds; the deterministic einsum also keeps
    # this tiny feature-by-action reduction independent of BLAS threading.
    cross = np.einsum("nf,na->fa", centered, target_centered, optimize=False)
    covariance = cross / (len(rows) * scale[:, None])
    strength = np.linalg.norm(covariance, axis=1)
    selected = np.argsort(-strength, kind="stable")[:config.max_axis_features]
    required = [schema.feature_names.index(name) for name in (
        "same_target_collision_streak_capped", "robots_shortest_distance",
        "self_charger_return_margin")]
    indices = list(dict.fromkeys([*required, *nonmask[selected].tolist()]))
    basis: list[tuple[str, np.ndarray]] = []
    for index in indices:
        weight = np.zeros(schema.feature_dim, dtype=np.float32)
        weight[index] = 1.0
        basis.append((f"axis:{schema.feature_names[index]}", weight))
    # Local fit-only sparse projections can capture diagonal regions but never
    # use a mask bit.  A legal-action proof still needs its own pure mask gate.
    for action in range(schema.action_count):
        direction = covariance[:, action]
        chosen = np.argsort(-np.abs(direction), kind="stable")[:MAX_NODE_FEATURES]
        if len(chosen) < 2:
            continue
        weight = np.zeros(schema.feature_dim, dtype=np.float32)
        weight[nonmask[chosen]] = direction[chosen].astype(np.float32)
        norm = float(np.linalg.norm(weight))
        if norm > 1.0e-8:
            weight /= norm
            basis.append((f"sparse3:{schema.actions[action]}", weight))
    return basis


def _proxy(rows: np.ndarray, legal: frozenset[int], target: np.ndarray) -> float:
    if not len(rows):
        return 0.0
    labels = target[rows].argmax(1)
    counts = np.bincount(labels, minlength=target.shape[1])
    action = int(np.argmax(counts))
    purity = float(counts[action] / len(rows))
    return len(rows) * purity**4 * (1.0 if action in legal else 0.65)


def _candidate_splits(
    state: _State, leaf: int, x: np.ndarray, masks: np.ndarray,
    target: np.ndarray, schema: FeatureSchema, config: CertificateSearchConfig,
) -> list[_Split]:
    rows = state.rows[leaf]
    if state.depths[leaf] >= MAX_DEPTH or len(rows) < 2:
        return []
    old_proxy = _proxy(rows, state.legal[leaf], target)
    candidates: list[_Split] = []
    seen: set[bytes] = set()

    def add(label: str, weight: np.ndarray, threshold: float,
            action: int | None = None,
            allow_small_residual: bool = False) -> None:
        bias = np.float32(-threshold)
        scores = hard_tree_split_scores(
            x[rows], weight[None, :], np.asarray([bias], dtype=np.float32))[:, 0]
        right = scores > HARD_SPLIT_TIE_EPSILON
        left_rows, right_rows = rows[~right], rows[right]
        if not len(left_rows) or not len(right_rows):
            return
        if action is None and min(len(left_rows), len(right_rows)) < config.min_noncore_child_rows:
            if not (allow_small_residual
                    and max(len(left_rows), len(right_rows)) >= FIT_MIN_LEAF_ROWS):
                return
        if action is not None and len(right_rows) < FIT_MIN_LEAF_ROWS:
            return
        partition = right.tobytes()
        if partition in seen:
            return
        _, right_cert = certified_mask_branches(
            weight, float(bias), schema.mask_feature_indices,
            certified_feature_bounds(schema.feature_names))
        if action is not None and action not in right_cert:
            return
        seen.add(partition)
        right_legal = state.legal[leaf] | right_cert
        quick = (_proxy(left_rows, state.legal[leaf], target)
                 + _proxy(right_rows, right_legal, target) - old_proxy)
        if action is not None:
            top = target[right_rows].argmax(1)
            action_purity = float(np.mean(top == action))
            quick += len(right_rows) * action_purity**4
        candidates.append(_Split(weight, bias, left_rows, right_rows,
                                 action, quick, label))

    # Pure binary gates enter *before* topology choice, never as a post-fit
    # repair.  WAIT is structurally legal on all paths.
    for action in range(schema.action_count):
        if action in state.legal[leaf]:
            continue
        weight = np.zeros(schema.feature_dim, dtype=np.float32)
        weight[schema.mask_feature_indices[action]] = 1.0
        add(f"mask_gate:{schema.actions[action]}", weight, 0.5, action)
    # These two-feature conditions spend one node on both a public-state
    # partition and a proof that the selected movement is physically legal.
    # For x in [lo, hi], (hi-t)*mask + x-hi > 0 implies mask=1
    # and x>t; (t-lo)*mask - x+lo > 0 implies mask=1 and x<t.
    # The shared interval checker validates stored float32 coefficients.
    if not config.pure_movement_gates_only:
        for action in range(schema.action_count):
            if action in state.legal[leaf]:
                continue
            mask_index = schema.mask_feature_indices[action]
            legal_rows = rows[masks[rows, action] > 0.5]
            if len(legal_rows) < FIT_MIN_LEAF_ROWS:
                continue
            for feature, (lower, upper) in sorted(
                    certified_feature_bounds(schema.feature_names).items()):
                values = x[legal_rows, feature]
                if float(np.max(values) - np.min(values)) < 1.0e-5:
                    continue
                for quantile in config.quantiles:
                    threshold = float(np.float32(np.quantile(values, quantile)))
                    if not lower + 1.0e-5 < threshold < upper - 1.0e-5:
                        continue
                    above = np.zeros(schema.feature_dim, dtype=np.float32)
                    above[mask_index], above[feature] = upper - threshold, 1.0
                    add(f"legal_above:{schema.actions[action]}:"
                        f"{schema.feature_names[feature]}@{quantile:g}",
                        above, upper, action)
                    below = np.zeros(schema.feature_dim, dtype=np.float32)
                    below[mask_index], below[feature] = threshold - lower, -1.0
                    add(f"legal_below:{schema.actions[action]}:"
                        f"{schema.feature_names[feature]}@{quantile:g}",
                        below, -lower, action)
    pure_gates = [item for item in candidates
                  if item.label.startswith("mask_gate:")]
    guarded = sorted((item for item in candidates
                      if item.legal_action is not None
                      and not item.label.startswith("mask_gate:")),
                     key=lambda item: (-item.quick_gain, item.label))
    forced = pure_gates + guarded[:config.max_exact_candidates_per_leaf]
    if len(rows) >= 2 * config.min_noncore_child_rows:
        for label, weight in _projection_basis(rows, x, target, schema, config):
            values = np.sum(x[rows] * weight[None, :], axis=1)
            risk_axis = label in {
                "axis:same_target_collision_streak_capped",
                "axis:robots_shortest_distance",
                "axis:self_charger_return_margin",
            }
            if risk_axis:
                boundary = {
                    "axis:same_target_collision_streak_capped": 0.0,
                    "axis:robots_shortest_distance": 2.0 / 42.0,
                    "axis:self_charger_return_margin": 4.0 / 50.0,
                }[label]
                add(f"{label}@stratum", weight, boundary,
                    allow_small_residual=True)
            for quantile in config.quantiles:
                threshold = float(np.float32(np.quantile(values, quantile)))
                add(f"{label}@{quantile:g}", weight, threshold,
                    allow_small_residual=risk_axis)
    ordinary = sorted((item for item in candidates if item.legal_action is None),
                      key=lambda item: (-item.quick_gain, item.label))
    return forced + ordinary[:config.max_exact_candidates_per_leaf]


def _expand(
    state: _State, leaf: int, split: _Split, x: np.ndarray, masks: np.ndarray,
    target: np.ndarray, episodes: np.ndarray, schema: FeatureSchema,
) -> _State:
    weights, bias = state.weights.copy(), state.bias.copy()
    left, right = state.left.copy(), state.right.copy()
    rows, depths = state.rows.copy(), state.depths.copy()
    legal, logits = state.legal.copy(), state.logits.copy()
    stats = state.stats.copy()
    node = len(weights)
    new_leaf = len(rows)
    old_ref = -1 - leaf
    if node:
        found = [(index, side) for index in range(node)
                 for side, refs in (("left", left), ("right", right))
                 if refs[index] == old_ref]
        if len(found) != 1:
            raise RuntimeError("certificate search lost leaf parent")
        parent, side = found[0]
        (left if side == "left" else right)[parent] = node
    weights.append(split.weight)
    bias.append(split.bias)
    left.append(old_ref)
    right.append(-1 - new_leaf)
    rows[leaf] = split.left_rows
    rows.append(split.right_rows)
    depths[leaf] += 1
    depths.append(depths[leaf])
    right_legal = (legal[leaf] | frozenset((split.legal_action,))
                   if split.legal_action is not None else legal[leaf])
    legal.append(right_legal)
    logits[leaf] = _fit_leaf_logits(target[rows[leaf]], masks[rows[leaf]])
    logits.append(_fit_leaf_logits(target[rows[new_leaf]], masks[rows[new_leaf]]))
    stats[leaf] = _leaf_stats(rows[leaf], logits[leaf], legal[leaf], depths[leaf],
                              x, masks, target, episodes, schema)
    stats.append(_leaf_stats(rows[new_leaf], logits[new_leaf], legal[new_leaf],
                             depths[new_leaf], x, masks, target, episodes, schema))
    return _State(weights, bias, left, right, rows, depths, legal, logits, stats)


def _materialize(
    state: _State, schema: FeatureSchema, report: dict,
    episode_ids: np.ndarray,
) -> SparseProgramTree:
    weights = np.zeros((MAX_LEAVES - 1, schema.feature_dim), dtype=np.float32)
    bias = np.zeros(MAX_LEAVES - 1, dtype=np.float32)
    left = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
    right = np.zeros(MAX_LEAVES - 1, dtype=np.int64)
    logits = np.zeros((MAX_LEAVES, schema.action_count), dtype=np.float32)
    core = np.zeros(MAX_LEAVES, dtype=np.bool_)
    for index, weight in enumerate(state.weights):
        weights[index] = weight
        bias[index] = state.bias[index]
        left[index], right[index] = state.left[index], state.right[index]
    for index, value in enumerate(state.logits):
        logits[index] = value
        core[index] = state.stats[index].core
    return SparseProgramTree(schema, weights, bias, left, right,
                             len(state.weights), len(state.rows), logits, core,
                             report, frozenset(episode_ids.tolist()))


def _pure_mask_gate_actions(tree: SparseProgramTree) -> dict[int, frozenset[int]]:
    """Return only legality proven by a dedicated binary mask-bit ancestor."""
    proven: dict[int, frozenset[int]] = {}
    mask_indices = tree.schema.mask_feature_indices
    bounds = certified_feature_bounds(tree.schema.feature_names)

    def visit(ref: int, legal: frozenset[int]) -> None:
        if ref < 0:
            proven[-1 - ref] = legal
            return
        weights = tree.split_weights[ref]
        used = np.flatnonzero(weights)
        pure = len(used) == 1 and int(used[0]) in mask_indices
        left_cert, right_cert = (certified_mask_branches(
            weights, float(tree.split_bias[ref]), mask_indices, bounds)
            if pure else (frozenset(), frozenset()))
        visit(int(tree.left_refs[ref]), legal | left_cert)
        visit(int(tree.right_refs[ref]), legal | right_cert)

    visit(0 if tree.node_count else -1,
          frozenset({tree.schema.actions.index("WAIT")}))
    return proven


def fit_certificate_tree(
    encoded: np.ndarray, masks: np.ndarray, target_probs: np.ndarray,
    episode_ids: Sequence[str], *, schema: FeatureSchema,
    frozen_actor_sha256: str,
    target_provenance: str = "frozen_nn_masked_distribution",
    config: CertificateSearchConfig = CertificateSearchConfig(),
) -> SparseProgramTree:
    """Search topology using fit episodes only; return no independent verdict."""
    if target_provenance != "frozen_nn_masked_distribution":
        raise ValueError("v6.14 core target must be the full frozen NN distribution")
    x, m, p, episodes = _validated_fit_rows(
        encoded, masks, target_probs, episode_ids, schema, frozen_actor_sha256)
    all_rows = np.arange(len(x), dtype=np.int64)
    wait = schema.actions.index("WAIT")
    root_legal = frozenset((wait,))
    root_logits = _fit_leaf_logits(p, m)
    root_stats = _leaf_stats(all_rows, root_logits, root_legal, 0,
                             x, m, p, episodes, schema)
    root = _State([], [], [], [], [all_rows], [0], [root_legal],
                  [root_logits], [root_stats])
    beam = [root]
    best = root
    explored = 1
    round_diagnostics = []

    def beam_score(state: _State) -> tuple[float, float, float, int]:
        # Potential helps cross uncertified intermediate nodes.  Only strict
        # core mass decides the final topology; potential never qualifies core.
        return (state.core_rows() + 0.30 * state.potential(),
                float(state.core_rows()), -state.mean_kl(), -len(state.rows))

    def final_score(state: _State) -> tuple[int, float, int]:
        return state.core_rows(), -state.mean_kl(), -len(state.rows)

    for round_index in range(config.max_search_leaves - 1):
        proposed: list[tuple[float, int, int, _State, _Split]] = []
        for state_index, state in enumerate(beam):
            if len(state.rows) >= config.max_search_leaves:
                continue
            for leaf in range(len(state.rows)):
                if state.stats[leaf].core or state.depths[leaf] >= MAX_DEPTH:
                    continue
                for split in _candidate_splits(state, leaf, x, m, p, schema, config):
                    proposed.append((split.quick_gain, state_index, leaf,
                                     state, split))
        if not proposed:
            break
        # Keep all mandatory mask gates within the exact-fit budget when
        # possible; then spend remaining computation on fit-only proxy rank.
        gates = [item for item in proposed if item[4].legal_action is not None]
        regular = [item for item in proposed if item[4].legal_action is None]
        gates.sort(key=lambda item: (-item[0], item[1], item[2], item[4].label))
        regular.sort(key=lambda item: (-item[0], item[1], item[2], item[4].label))
        budget = config.max_exact_candidates_per_round
        chosen = gates[:budget] + regular[:max(0, budget - len(gates))]
        next_states = []
        for _, _, leaf, state, split in chosen:
            candidate = _expand(state, leaf, split, x, m, p, episodes, schema)
            explored += 1
            next_states.append(candidate)
            if final_score(candidate) > final_score(best):
                best = candidate
        next_states.sort(key=beam_score, reverse=True)
        beam = next_states[:config.beam_width]
        round_diagnostics.append({"round": round_index + 1,
                                  "proposal_count": len(proposed),
                                  "exact_fit_count": len(chosen),
                                  "beam_best_core_rows": beam[0].core_rows(),
                                  "beam_best_potential": beam[0].potential()})

    report = {
        "version": "warehouse-v6.14-certificate-tree-prototype.v1",
        "pure_movement_gates_only": config.pure_movement_gates_only,
        "status": "review_only_unregistered",
        "target_provenance": target_provenance,
        "frozen_actor_sha256": frozen_actor_sha256,
        "feature_schema_sha256": schema.sha256(),
        "feature_dim": schema.feature_dim,
        "action_count": schema.action_count,
        "fit_episode_ids_sha256": sha256(json.dumps(
            sorted(set(episodes)), separators=(",", ":")).encode()).hexdigest(),
        "fit_target_sha256": sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest(),
        "fit_rows": len(x), "fit_episode_count": len(set(episodes)),
        "fit_core_rows": best.core_rows(),
        "fit_core_mass": float(best.core_rows() / len(x)),
        "fit_expected_feedback_actionable_mass": float(sum(
            np.sum(p[rows, best.stats[leaf].raw_action] *
                   (p[rows].argmax(1) == best.stats[leaf].raw_action))
            for leaf, rows in enumerate(best.rows) if best.stats[leaf].core) / len(x)),
        "internal_development_mass_thresholds": {
            "core": INTERNAL_MIN_CORE_MASS,
            "expected_feedback_actionable": INTERNAL_MIN_EXPECTED_ACTIONABLE_MASS,
        },
        "fit_only_mass_diagnostic_pass": bool(
            best.core_rows() / len(x) >= INTERNAL_MIN_CORE_MASS),
        "node_count": len(best.weights), "leaf_count": len(best.rows),
        "maximum_depth": max(best.depths),
        "maximum_features_per_node": max(
            (int(np.count_nonzero(w)) for w in best.weights), default=0),
        "search_config": {key: getattr(config, key)
                          for key in config.__dataclass_fields__},
        "search_rounds": round_diagnostics,
        "explored_topologies": explored,
        "leaf_diagnostics": [stat.report(leaf, schema.actions)
                             for leaf, stat in enumerate(best.stats)],
        "topology_uses_fit_only": True,
        "calibration_eligibility_validation_used": False,
        "execution_action_override": False,
    }
    result = _materialize(best, schema, report, episodes)
    if config.pure_movement_gates_only:
        pure = _pure_mask_gate_actions(result)
        if any(int(np.argmax(result.leaf_logits[leaf])) not in pure[leaf]
               for leaf in range(result.leaf_count) if result.core_mask[leaf]):
            raise RuntimeError(
                "Warehouse certified core movement lacks pure legal mask gate")
        report["core_pure_gate_leaf_count"] = int(result.core_mask[:result.leaf_count].sum())
    # An export must route the exact fit rows to the leaves scored in search.
    routed = result.leaf_indices(x)
    for leaf, rows in enumerate(best.rows):
        if not np.array_equal(np.sort(np.flatnonzero(routed == leaf)), np.sort(rows)):
            raise RuntimeError("certificate tree float32 export changed fit routes")
    report["tree_content_sha256"] = result.content_sha256()
    return result


def to_current_core_tree_result(tree: SparseProgramTree):
    """Adapter for the copied v6.14 Actor tree, never registered here."""
    from backend.training.warehouse_v6_core_tree import CoreTreeResult
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.public_action_features_v617 import (
        TREE_FEATURE_DIM, TREE_FEATURE_NAMES, TREE_MASK_START,
    )
    from env.warehouse import public_action_features_v614 as legacy_features

    if (tree.schema.feature_dim not in (TREE_FEATURE_DIM, legacy_features.TREE_FEATURE_DIM)
            or tree.schema.feature_names not in (tuple(TREE_FEATURE_NAMES), tuple(legacy_features.TREE_FEATURE_NAMES))
            or tree.schema.actions != tuple(ACTIONS)
            or tree.schema.mask_feature_indices != tuple(
                range(TREE_MASK_START, TREE_MASK_START + len(ACTIONS)))):
        raise ValueError("certificate tree schema differs from runtime public schema")
    return CoreTreeResult(tree.split_weights, tree.split_bias,
                          tree.left_refs, tree.right_refs,
                          tree.node_count, tree.leaf_count,
                          tree.leaf_logits, tree.core_mask,
                          tree.report, tree.train_episode_ids)


def frozen_actor_topology(
    role_weights: Mapping[str, np.ndarray], *, schema: FeatureSchema,
    frozen_actor_sha256: str,
) -> SparseProgramTree:
    """Extract a final Actor's tree topology; do not copy its core verdict.

    The full role weight hash is checked before extraction.  This function is
    for offline program refit only and never changes the runtime Actor.
    """
    from backend.training.warehouse_v6_actor import (
        LEGACY_MODEL_VERSION, V616_MODEL_VERSION, MODEL_VERSION, role_sha256,
    )

    versions = ((MODEL_VERSION,) if "protected_leaf_mask" in role_weights else
                (V616_MODEL_VERSION, LEGACY_MODEL_VERSION))
    if (_SHA.fullmatch(frozen_actor_sha256) is None or frozen_actor_sha256 not in {
            role_sha256(role_weights, model_version=version) for version in versions}):
        raise ValueError("final frozen Actor role hash mismatch")
    keys = ("tree.split.weight", "tree.split.bias", "tree.left_refs",
            "tree.right_refs", "tree.node_count", "tree.leaf_count",
            "tree.leaf_logits")
    if any(key not in role_weights for key in keys):
        raise ValueError("final frozen Actor tree tensors absent")
    raw_nodes = np.asarray(role_weights["tree.node_count"])
    raw_leaves = np.asarray(role_weights["tree.leaf_count"])
    if (raw_nodes.shape != (1,) or raw_leaves.shape != (1,)
            or not np.isfinite(raw_nodes).all() or not np.isfinite(raw_leaves).all()
            or raw_nodes[0] != int(raw_nodes[0])
            or raw_leaves[0] != int(raw_leaves[0])):
        raise ValueError("final frozen Actor topology counts invalid")
    nodes, leaves = int(raw_nodes[0]), int(raw_leaves[0])
    report = {"version": "warehouse-v6.14-frozen-topology-extraction.v1",
              "status": "review_only_unregistered",
              "frozen_actor_sha256": frozen_actor_sha256,
              "feature_schema_sha256": schema.sha256(),
              "source": "final_frozen_full_actor_tree_topology",
              "core_mask_inherited": False}
    empty_core = np.zeros(MAX_LEAVES, dtype=np.bool_)
    tree = SparseProgramTree(
        schema,
        np.asarray(role_weights["tree.split.weight"], dtype=np.float32),
        np.asarray(role_weights["tree.split.bias"], dtype=np.float32),
        np.asarray(role_weights["tree.left_refs"], dtype=np.int64),
        np.asarray(role_weights["tree.right_refs"], dtype=np.int64),
        nodes, leaves,
        np.asarray(role_weights["tree.leaf_logits"], dtype=np.float32),
        empty_core, report, frozenset())
    report["topology_sha256"] = tree.topology_sha256()
    return tree


def refit_frozen_topology_fit_only(
    topology: SparseProgramTree, encoded: np.ndarray, masks: np.ndarray,
    target_probs: np.ndarray, episode_ids: Sequence[str], *,
    frozen_actor_sha256: str, expected_topology_sha256: str,
) -> SparseProgramTree:
    """Refit program leaves on fit episodes without selecting a new topology.

    A final Actor's topology may be fixed here.  Calibration, eligibility, and
    validation must then be handled by their existing independent interfaces.
    Unvisited residual leaves retain the extracted logits and cannot be core.
    """
    if (topology.report.get("source") != "final_frozen_full_actor_tree_topology"
            or topology.topology_sha256() != expected_topology_sha256
            or topology.report.get("frozen_actor_sha256") != frozen_actor_sha256):
        raise ValueError("final frozen Actor topology identity mismatch")
    schema = topology.schema
    x, m, p, episodes = _validated_fit_rows(
        encoded, masks, target_probs, episode_ids, schema, frozen_actor_sha256)
    routed = topology.leaf_indices(x)
    logits = topology.leaf_logits.copy()
    core = np.zeros(MAX_LEAVES, dtype=np.bool_)
    details = []
    path_legal: dict[int, frozenset[int]] = {}

    def visit(ref: int, legal: frozenset[int]) -> None:
        if ref < 0:
            path_legal[-1 - ref] = legal
            return
        left_cert, right_cert = certified_mask_branches(
            topology.split_weights[ref], float(topology.split_bias[ref]),
            schema.mask_feature_indices,
            certified_feature_bounds(schema.feature_names))
        left_legal = legal | left_cert
        right_legal = legal | right_cert
        visit(int(topology.left_refs[ref]), left_legal)
        visit(int(topology.right_refs[ref]), right_legal)

    visit(0 if topology.node_count else -1,
          frozenset((schema.actions.index("WAIT"),)))
    for leaf in range(topology.leaf_count):
        rows = np.flatnonzero(routed == leaf)
        if len(rows):
            logits[leaf] = _fit_leaf_logits(p[rows], m[rows])
            stat = _leaf_stats(rows, logits[leaf], path_legal[leaf],
                               MAX_DEPTH, x, m, p, episodes, schema)
            core[leaf] = stat.core
            details.append(stat.report(leaf, schema.actions))
        else:
            details.append({"leaf": leaf, "rows": 0,
                            "fit_core_candidate": False,
                            "reason": "no_fit_visitation"})
    report = {"version": "warehouse-v6.14-frozen-topology-fit-refit.v1",
              "status": "review_only_unregistered",
              "source": "final_frozen_full_actor_tree_topology",
              "frozen_actor_sha256": frozen_actor_sha256,
              "topology_sha256": expected_topology_sha256,
              "fit_target_sha256": sha256(np.ascontiguousarray(
                  p, dtype="<f4").tobytes()).hexdigest(),
              "fit_episode_ids_sha256": sha256(json.dumps(
                  sorted(set(episodes)), separators=(",", ":")).encode()).hexdigest(),
              "fit_core_mass": float(core[routed].mean()),
              "leaf_diagnostics": details,
              "topology_uses_fit_only": False,
              "topology_extracted_from_frozen_actor": True,
              "leaf_logits_use_fit_only": True,
              "calibration_eligibility_validation_used": False,
              "execution_action_override": False}
    result = SparseProgramTree(
        schema, topology.split_weights, topology.split_bias,
        topology.left_refs, topology.right_refs,
        topology.node_count, topology.leaf_count, logits, core,
        report, frozenset(episodes.tolist()))
    if result.topology_sha256() != expected_topology_sha256:
        raise RuntimeError("fit-only refit changed frozen Actor topology")
    report["tree_content_sha256"] = result.content_sha256()
    return result


__all__ = ["FeatureSchema", "CertificateSearchConfig", "SparseProgramTree",
           "fit_certificate_tree", "to_current_core_tree_result",
           "frozen_actor_topology", "refit_frozen_topology_fit_only"]
