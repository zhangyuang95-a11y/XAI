"""Independent sparse programs for frozen Warehouse dual Actors.

Programs approximate each complete masked neural policy on pre-action public
features. They are gradients and explanation evidence only. They never choose
an action sent to the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from env.warehouse import public_action_features_v614 as legacy_features
from env.warehouse.public_action_features_v617 import (
    ACTION_PHYSICS_CONFIG_SHA256, encode_actor_tree_features,
)
from .warehouse_v6_actor import (
    FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256,
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_TIE_EPSILON, certified_feature_bounds,
    certified_mask_branches, split_scores as hard_tree_split_scores,
    leaf_indices as hard_tree_leaf_indices,
    validate_certification_domain,
)


PROGRAM_VERSION = "warehouse-role-program.v6.14-public-affordance-exploratory"
V616_PROGRAM_VERSION = "warehouse-role-program.v6.16-joint-certified-topology"
JOINT_PROGRAM_VERSION = "warehouse-role-program.v6.17-risk-protected-topology"
SELECTIVE_PROGRAM_VERSION = "warehouse-role-program.v6.18-independent-selective-distribution"
PATH_CONSTRAINED_PROGRAM_VERSION = "warehouse-role-program.v6.19-path-constrained-distribution-tree"
AXIS_TREE_METHOD = "axis_tree_v1"
ACTOR_SPARSE_TOPOLOGY_METHOD = "actor_sparse_topology_v1"
V616_TOPOLOGY_METHOD = "actor_joint_certified_topology_v2"
JOINT_CERTIFIED_TOPOLOGY_METHOD = "actor_risk_stratified_protected_topology_v3"
INDEPENDENT_SELECTIVE_METHOD = "independent_selective_masked_distribution_v4"
PATH_CONSTRAINED_METHOD = "path_constrained_distribution_tree_v1"
PROGRAM_MAX_DEPTH = 5
PROGRAM_MAX_LEAVES = 16
PROGRAM_MAX_INTERNAL_NODES = PROGRAM_MAX_LEAVES - 1
MAX_SPLIT_FEATURES = 3
MIN_LEAF_SUPPORT = 20
MIN_EPISODE_SUPPORT = 3
MIN_COVERAGE = 0.20
FEEDBACK_AGREEMENT = 0.85
EXPLANATION_AGREEMENT = 0.90
MAX_COVERED_KL = 0.35
_EPSILON = 1e-8
_MASK_INDICES = tuple(FEATURE_NAMES.index(f"legal_{action.lower()}")
                      for action in ACTIONS)
_CERTIFICATION_BOUNDS = certified_feature_bounds(tuple(FEATURE_NAMES))


def _certified_tree_actions(
        split_weights: np.ndarray, split_bias: np.ndarray,
        left_refs: np.ndarray, right_refs: np.ndarray,
        root_ref: int, *, pure_mask_gates_only: bool) -> dict[int, frozenset[int]]:
    certificates: dict[int, frozenset[int]] = {}

    def visit(ref: int, proven: frozenset[int]) -> None:
        if ref < 0:
            certificates[-1 - ref] = proven
            return
        weights = split_weights[ref]
        used = np.flatnonzero(weights)
        pure_gate = (len(used) == 1 and int(used[0]) in _MASK_INDICES)
        left, right = (certified_mask_branches(
            weights, float(split_bias[ref]),
            _MASK_INDICES, _CERTIFICATION_BOUNDS)
            if not pure_mask_gates_only or pure_gate
            else (frozenset(), frozenset()))
        visit(int(left_refs[ref]), proven | left)
        visit(int(right_refs[ref]), proven | right)

    visit(int(root_ref), frozenset({ACTIONS.index("WAIT")}))
    return certificates


def certified_program_actions(program: "WarehouseV6Program") -> dict[int, frozenset[int]]:
    """Prove each program leaf's legal actions with the Actor's path rule."""
    program._structure()
    return _certified_tree_actions(
        program.split_weights, program.split_bias, program.left_refs,
        program.right_refs, program.root_ref,
        pure_mask_gates_only=_requires_path_certificate(program))


def certified_actor_actions(
        role: FrozenWarehouseV6Role) -> dict[int, frozenset[int]]:
    """Expose the same certificate for an already validated frozen Actor role."""
    weights = role.weights
    nodes = int(weights["tree.node_count"][0])
    return _certified_tree_actions(
        weights["tree.split.weight"][:nodes],
        weights["tree.split.bias"][:nodes],
        weights["tree.left_refs"][:nodes].astype(np.int64),
        weights["tree.right_refs"][:nodes].astype(np.int64),
        0 if nodes else -1, pure_mask_gates_only=True)


def _requires_path_certificate(program: "WarehouseV6Program") -> bool:
    return program.report.get("fit_method") in (
        JOINT_CERTIFIED_TOPOLOGY_METHOD, V616_TOPOLOGY_METHOD,
        INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)


def _requires_actor_topology(program: "WarehouseV6Program") -> bool:
    return program.report.get("fit_method") in (
        ACTOR_SPARSE_TOPOLOGY_METHOD, JOINT_CERTIFIED_TOPOLOGY_METHOD,
        V616_TOPOLOGY_METHOD)


def _sample_arrays(rows: Sequence[Mapping], name: str, feature_dim: int = FEATURE_DIM) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any("episode" not in row for row in rows):
        raise ValueError(f"{name} requires pre-action episode IDs and samples")
    obs = np.stack([np.asarray(row["obs"], dtype=np.float32) for row in rows])
    masks = np.stack([np.asarray(row["mask"], dtype=np.float32) for row in rows])
    if (obs.shape != (len(rows), OBSERVATION_DIM)
            or masks.shape != (len(rows), len(ACTIONS))
            or not np.isfinite(obs).all() or not np.isfinite(masks).all()
            or np.any((masks != 0.0) & (masks != 1.0))
            or np.any(masks.sum(axis=-1) < 1.0)):
        raise ValueError(f"{name} has invalid Warehouse public rows")
    encoder = encode_actor_tree_features if feature_dim == FEATURE_DIM else legacy_features.encode_actor_tree_features
    return encoder(obs, masks), masks


def _sampled_indices(rows: Sequence[Mapping]) -> np.ndarray:
    sampled = [row.get("sampled_action") for row in rows]
    if any(action not in ACTIONS for action in sampled):
        raise ValueError("program audit requires actual sampled NN actions")
    return np.asarray([ACTIONS.index(action) for action in sampled], dtype=np.int64)


def _masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    selected = np.where(masks > 0.5, logits, -1.0e9)
    shifted = selected - selected.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _wilson_lower(success_fraction: float, count: int) -> float:
    if count <= 0:
        return 0.0
    z = 1.645
    denominator = 1.0 + z * z / count
    centre = success_fraction + z * z / (2.0 * count)
    spread = z * np.sqrt((success_fraction * (1.0 - success_fraction)
                          + z * z / (4.0 * count)) / count)
    return float(max(0.0, (centre - spread) / denominator))


def _metric_context(program: "WarehouseV6Program", actor: FrozenWarehouseV6Role,
                    rows: Sequence[Mapping]) -> dict:
    if (program.feature_dim != actor.feature_dim
            or tuple(program.feature_names) != tuple(actor.feature_names)):
        raise RuntimeError("Warehouse program and Actor feature schema mismatch")
    x, masks = _sample_arrays(rows, "program audit", actor.feature_dim)
    if _requires_path_certificate(program):
        validate_certification_domain(x, _MASK_INDICES,
            certified_feature_bounds(tuple(actor.feature_names)))
    return {"actor": actor, "program": program, "rows": rows,
            "x": x, "masks": masks, "sampled": _sampled_indices(rows),
            "leaves": program.leaf_indices(x),
            "p": _masked_probabilities(actor.logits_encoded(x), masks)}


def _metrics(program: "WarehouseV6Program", actor: FrozenWarehouseV6Role,
             rows: Sequence[Mapping], selected_leaves: set[int],
             purpose: str = "feedback", *, context: Mapping | None = None) -> dict:
    if purpose not in ("feedback", "explanation"):
        raise ValueError("invalid program metric purpose")
    if (program.feature_dim != actor.feature_dim
            or tuple(program.feature_names) != tuple(actor.feature_names)):
        raise RuntimeError("Warehouse program and Actor feature schema mismatch")
    if context is not None:
        if (context["actor"] is not actor or context["program"] is not program
                or context["rows"] is not rows):
            raise ValueError("program metric context binding mismatch")
        x, masks = context["x"], context["masks"]
        sampled, leaves = context["sampled"], context["leaves"]
    else:
        x, masks = _sample_arrays(rows, "program audit", actor.feature_dim)
        if _requires_path_certificate(program):
            validate_certification_domain(x, _MASK_INDICES, certified_feature_bounds(tuple(actor.feature_names)))
        sampled = _sampled_indices(rows)
        leaves = program.leaf_indices(x)
    selected = np.isin(leaves, tuple(selected_leaves))
    count = int(selected.sum())
    independent = program.report.get("fit_method") in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)
    all_x = x if independent else None
    all_p = (context["p"] if context is not None else
             _masked_probabilities(actor.logits_encoded(x), masks)
             if independent else None)
    result = {
        "sample_count": len(rows), "rows": count,
        "coverage": count / len(rows), "actionable_rows": 0,
        "actionable_coverage": 0.0, "agreement": 0.0,
        "agreement_wilson_lower_90": 0.0, "kl": None,
        "illegal_count": 0, "masked_illegal_count": 0,
        "path_uncertified_count": 0,
        "certified_rows": 0, "distinct_episode_count": 0,
        "distinct_scenario_seed_count": 0,
        "max_episode_fraction": 0.0,
    }
    if independent:
        result.update({"physical_frame_count": len(rows), "excluded_frames": 0,
                       "expected_actionable_coverage": 0.,
                       "expected_actionable_rows": 0.,
                       "per_nn_action": {}, "public_strata": {}})
    if independent:
        # These counts describe every raw tree output before physical masking,
        # eligibility selection or per-frame action matching. Local eligibility
        # remains selective; its illegal_count keeps the historical meaning.
        all_raw_actions = program.leaf_logits[leaves].argmax(-1)
        all_raw_legal = masks[np.arange(len(rows)), all_raw_actions] > .5
        certificates = certified_program_actions(program)
        all_path_proven = np.fromiter(
            (int(action) in certificates[int(leaf)]
             for action, leaf in zip(all_raw_actions, leaves, strict=True)),
            dtype=np.bool_, count=len(rows))
        result.update({
            "whole_tree_raw_illegal_count": int(np.sum(~all_raw_legal)),
            "whole_tree_path_uncertified_count": int(np.sum(~all_path_proven)),
        })
    if not count:
        return result
    x, m, leaf, actions = x[selected], masks[selected], leaves[selected], sampled[selected]
    p = all_p[selected] if independent else _masked_probabilities(actor.logits_encoded(x), m)
    raw = program.leaf_logits[leaf]
    q = _masked_probabilities(raw, m)
    top = q.argmax(-1)
    nn_top = p.argmax(-1)
    raw_legal = m[np.arange(count), raw.argmax(-1)] > 0.5
    path_certified = np.ones(count, dtype=np.bool_)
    if _requires_path_certificate(program):
        certificates = certified_program_actions(program)
        path_certified = np.fromiter(
            (int(np.argmax(raw[index])) in certificates[int(leaf[index])]
             for index in range(count)), dtype=np.bool_, count=count)
    action_legal = m[np.arange(count), actions] > 0.5
    matched = raw_legal & path_certified & action_legal & (top == actions)
    if purpose == "feedback":
        matched &= actions == nn_top
    agreement = float(np.mean(top == nn_top))
    episodes = [str(row["episode"]) for row, keep in zip(rows, selected, strict=True) if keep]
    scenario_seeds = {str(row.get("base_seed", row["episode"]))
                      for row, keep in zip(rows, selected, strict=True) if keep}
    frequencies = {episode: episodes.count(episode) for episode in set(episodes)}
    result.update({
        "actionable_rows": int(matched.sum()),
        "actionable_coverage": float(matched.sum() / len(rows)),
        "agreement": agreement,
        "agreement_wilson_lower_90": _wilson_lower(agreement, count),
        "kl": float(np.mean(np.sum(p * (
            np.log(np.maximum(p, _EPSILON)) - np.log(np.maximum(q, _EPSILON))), axis=-1))),
        "illegal_count": int(np.sum(~raw_legal)),
        "masked_illegal_count": int(np.sum(m[np.arange(count), top] <= 0.5)),
        "path_uncertified_count": int(np.sum(~path_certified)),
        "distinct_episode_count": len(frequencies),
        "distinct_scenario_seed_count": len(scenario_seeds),
        "max_episode_fraction": max(frequencies.values()) / count,
    })
    if independent:
        valid = raw_legal & path_certified
        expected = valid * p[np.arange(count), top]
        if purpose == "feedback":
            expected *= top == nn_top
        result["expected_actionable_rows"] = float(expected.sum())
        result["expected_actionable_coverage"] = float(expected.sum()/len(rows))
        for action, label in enumerate(ACTIONS):
            chosen = nn_top == action
            denominator = int(np.sum(all_p.argmax(-1) == action))
            result["per_nn_action"][label] = {
                "rows": int(chosen.sum()),
                "physical_frames": denominator,
                "qualified_coverage": float(chosen.sum()/denominator) if denominator else None,
                "agreement": float(np.mean(top[chosen] == nn_top[chosen])) if chosen.any() else None,
                "actual_actionable_rows": int(matched[chosen].sum()),
                "expected_actionable_rows": float(expected[chosen].sum()),
            }
        names = tuple(actor.feature_names)
        conflict = x[:, names.index("same_target_collision_streak_capped")] > 0
        near = x[:, names.index("robots_shortest_distance")]*42 <= 2
        low = x[:, names.index("self_charger_return_margin")]*50 <= 4
        all_conflict = all_x[:, names.index("same_target_collision_streak_capped")] > 0
        all_near = all_x[:, names.index("robots_shortest_distance")]*42 <= 2
        all_low = all_x[:, names.index("self_charger_return_margin")]*50 <= 4
        denominators = {"ordinary": int(np.sum(~(all_conflict | all_near | all_low))),
                        "conflict": int(all_conflict.sum()),
                        "near_robots": int(all_near.sum()),
                        "low_return_margin": int(all_low.sum())}
        row_kl = np.sum(p*(np.log(np.maximum(p, _EPSILON))
                           - np.log(np.maximum(q, _EPSILON))), axis=-1)
        for label, chosen in {"ordinary": ~(conflict | near | low),
                              "conflict": conflict, "near_robots": near,
                              "low_return_margin": low}.items():
            result["public_strata"][label] = {
                "rows": int(chosen.sum()),
                "physical_frames": denominators[label],
                "qualified_coverage": float(chosen.sum()/denominators[label]) if denominators[label] else None,
                "agreement": float(np.mean(top[chosen] == nn_top[chosen])) if chosen.any() else None,
                "kl": float(row_kl[chosen].mean()) if chosen.any() else None,
                "illegal_count": int(np.sum(~raw_legal[chosen])),
                "actual_actionable_rows": int(matched[chosen].sum()),
                "expected_actionable_rows": float(expected[chosen].sum()),
            }
    return result


def _eligible(metric: Mapping, required_agreement: float) -> bool:
    # The same frozen program must survive the stricter development precheck.
    # Only eligibility episodes can decide a leaf's membership.
    required_agreement = (0.93 if required_agreement == FEEDBACK_AGREEMENT else 0.95)
    return bool(metric["rows"] >= MIN_LEAF_SUPPORT
                and metric["distinct_episode_count"] >= MIN_EPISODE_SUPPORT
                and metric["distinct_scenario_seed_count"] >= MIN_EPISODE_SUPPORT
                and metric["max_episode_fraction"] <= 0.5
                and metric["agreement"] >= required_agreement
                and metric["agreement_wilson_lower_90"] >= required_agreement - 0.08
                and metric["kl"] is not None and metric["kl"] <= 0.25
                and metric["illegal_count"] == 0
                and metric.get("path_uncertified_count", 0) == 0)


def _whole_tree_raw_legality_passed(metric: Mapping) -> bool:
    """Require global raw legality; unqualified path proofs stay diagnostic.

    The qualified path count must still be zero. Missing new fields cannot
    inherit the old selected-only interpretation on a new admission path.
    """
    return (isinstance(metric, Mapping)
            and type(metric.get("whole_tree_raw_illegal_count")) is int
            and metric["whole_tree_raw_illegal_count"] == 0
            and type(metric.get("whole_tree_path_uncertified_count")) is int
            and metric["whole_tree_path_uncertified_count"] >= 0
            and type(metric.get("path_uncertified_count")) is int
            and metric["path_uncertified_count"] == 0)


def _passes(metric: Mapping, required_agreement: float, *,
            development: bool = False, require_whole_tree_zero: bool = False) -> bool:
    minimum = 0.30 if development else MIN_COVERAGE
    agreement = (0.93 if required_agreement == FEEDBACK_AGREEMENT else 0.95
                 ) if development else required_agreement
    maximum_kl = 0.25 if development else MAX_COVERED_KL
    return bool((not require_whole_tree_zero or _whole_tree_raw_legality_passed(metric))
                and metric["coverage"] >= minimum
                and metric["actionable_coverage"] >= minimum
                and metric["agreement"] >= agreement
                and metric["kl"] is not None and metric["kl"] <= maximum_kl
                and metric["illegal_count"] == 0
                and metric.get("path_uncertified_count", 0) == 0)


@dataclass(frozen=True)
class WarehouseV6Program:
    role: str
    actor_role_sha256: str
    root_ref: int
    split_weights: np.ndarray
    split_bias: np.ndarray
    left_refs: np.ndarray
    right_refs: np.ndarray
    leaf_logits: np.ndarray
    feedback_leaves: frozenset[int]
    explanation_leaves: frozenset[int]
    report: dict

    @property
    def feature_dim(self):
        return self.split_weights.shape[1]

    @property
    def feature_names(self):
        return FEATURE_NAMES if self.feature_dim == FEATURE_DIM else legacy_features.TREE_FEATURE_NAMES

    @property
    def feature_schema_sha256(self):
        return (FEATURE_SCHEMA_SHA256 if self.feature_dim == FEATURE_DIM
                else legacy_features.ACTOR_FEATURE_SCHEMA_SHA256)

    def encode(self, obs, mask):
        encoder = encode_actor_tree_features if self.feature_dim == FEATURE_DIM else legacy_features.encode_actor_tree_features
        return encoder(obs, mask)

    def _structure(self) -> tuple[int, int, int]:
        n = int(self.split_weights.shape[0])
        leaves = int(self.leaf_logits.shape[0])
        if (self.role not in AGENTS or not self.actor_role_sha256
                or self.feature_dim not in (FEATURE_DIM, legacy_features.TREE_FEATURE_DIM)
                or self.split_bias.shape != (n,)
                or self.left_refs.shape != (n,) or self.right_refs.shape != (n,)
                or self.leaf_logits.shape != (leaves, len(ACTIONS))
                or n > PROGRAM_MAX_INTERNAL_NODES or not 1 <= leaves <= PROGRAM_MAX_LEAVES
                or any(not np.isfinite(value).all() for value in (
                    self.split_weights, self.split_bias, self.leaf_logits))
                or np.max(np.count_nonzero(self.split_weights, axis=-1), initial=0)
                    > MAX_SPLIT_FEATURES
                or any(leaf < 0 or leaf >= leaves for leaf in (
                    self.feedback_leaves | self.explanation_leaves))):
            raise RuntimeError("Warehouse program tensor or complexity contract mismatch")
        visited_nodes, visited_leaves = set(), set()
        if self.feature_dim == FEATURE_DIM:
            from env.warehouse.public_action_features_v617 import CONTEXT_INDEX
            source_cost = np.count_nonzero(self.split_weights, axis=-1)
            source_cost += self.split_weights[:, CONTEXT_INDEX] != 0
            if np.any(source_cost > MAX_SPLIT_FEATURES):
                raise RuntimeError("Warehouse program derived-context feature budget exceeded")
        actual_depth = 0
        def walk(ref: int, depth: int) -> None:
            nonlocal actual_depth
            if depth > PROGRAM_MAX_DEPTH:
                raise RuntimeError("Warehouse program tree exceeds maximum depth")
            if ref < 0:
                leaf = -1 - ref
                if leaf >= leaves or leaf in visited_leaves:
                    raise RuntimeError("Warehouse program leaf graph is invalid")
                visited_leaves.add(leaf)
                actual_depth = max(actual_depth, depth)
                return
            if ref >= n or ref in visited_nodes:
                raise RuntimeError("Warehouse program split graph is invalid")
            visited_nodes.add(ref)
            walk(int(self.left_refs[ref]), depth + 1)
            walk(int(self.right_refs[ref]), depth + 1)
        walk(int(self.root_ref), 0)
        if visited_nodes != set(range(n)) or visited_leaves != set(range(leaves)):
            raise RuntimeError("Warehouse program contains unreachable nodes or leaves")
        return n, leaves, actual_depth

    def _leaf_encoded(self, encoded: np.ndarray) -> tuple[int, list[dict]]:
        node, path = int(self.root_ref), []
        # One routing contract is shared by the Actor, core refit, and
        # explanatory program, including exact threshold boundary states.
        scores = hard_tree_split_scores(
            np.asarray(encoded, dtype=np.float32).reshape(1, self.feature_dim),
            self.split_weights, self.split_bias)[0]
        while node >= 0:
            weights = self.split_weights[node]
            value = float(scores[node])
            go_right = value > HARD_SPLIT_TIE_EPSILON
            nonzero = np.flatnonzero(weights != 0.0)
            path.append({
                "features": [{
                    "name": self.feature_names[int(index)],
                    "coefficient": float(weights[index]),
                    "observed": float(encoded[index]),
                } for index in nonzero],
                "bias": float(self.split_bias[node]), "value": value,
                "branch": "right" if go_right else "left",
            })
            node = int(self.right_refs[node] if go_right else self.left_refs[node])
        return -1 - node, path

    def leaf_indices(self, encoded: np.ndarray) -> np.ndarray:
        flat = np.asarray(encoded, dtype=np.float32).reshape(-1, self.feature_dim)
        if self.report.get("fit_method") in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD):
            return hard_tree_leaf_indices(
                flat, self.split_weights, self.split_bias, self.left_refs,
                self.right_refs, len(self.split_weights), PROGRAM_MAX_DEPTH,
            ).reshape(encoded.shape[:-1])
        return np.asarray([self._leaf_encoded(row)[0] for row in flat], dtype=np.int64
                          ).reshape(encoded.shape[:-1])

    def _leaf(self, sample: Mapping) -> tuple[int, list[dict]]:
        encoded = self.encode(np.asarray(sample["obs"], dtype=np.float32),
                                         np.asarray(sample["mask"], dtype=np.float32))
        return self._leaf_encoded(encoded)

    def margin_certified(self, leaf: int, mask: np.ndarray) -> bool:
        # This program has its own tree. An Actor-tree residual bound indexed
        # by this program leaf would be an invalid analytical certificate.
        return False

    def lookup(self, sample: Mapping) -> dict:
        leaf, path = self._leaf(sample)
        mask = np.asarray(sample["mask"], dtype=np.float32)
        if _requires_path_certificate(self):
            encoded = self.encode(
                np.asarray(sample["obs"], dtype=np.float32), mask)
            validate_certification_domain(
                encoded.reshape(1, self.feature_dim), _MASK_INDICES,
                certified_feature_bounds(tuple(self.feature_names)))
        probabilities = _masked_probabilities(self.leaf_logits[leaf][None], mask[None])[0]
        action = ACTIONS[int(probabilities.argmax())]
        raw_action = ACTIONS[int(self.leaf_logits[leaf].argmax())]
        path_certified = (ACTIONS.index(raw_action) in
                          certified_program_actions(self)[leaf]
                          if _requires_path_certificate(self) else False)
        return {
            "leaf_id": leaf, "path": path, "action": action,
            "raw_action": raw_action,
            "raw_action_legal": bool(mask[ACTIONS.index(raw_action)] > 0.5),
            "raw_action_path_certified": path_certified,
            "probabilities": dict(zip(ACTIONS, map(float, probabilities), strict=True)),
            "margin_certified": False,
            "certificate_kind": "independent_empirical_audit_only",
            "program_sha256": self.sha256(),
        }

    def covered(self, sample: Mapping, *, actor_action: str, purpose: str,
                actor_role: FrozenWarehouseV6Role | None = None) -> dict | None:
        if purpose not in ("feedback", "explanation"):
            raise ValueError("unknown Warehouse program purpose")
        # A participant-facing explanation must be checked against the role
        # object loaded from the current, content-addressed Actor. Training
        # feedback already checks the current Actor distribution in its PPO
        # caller, but may supply the same role for additional binding.
        if purpose == "explanation" and actor_role is None:
            return None
        if actor_role is not None and (
                not isinstance(actor_role, FrozenWarehouseV6Role)
                or actor_role.role_sha256 != self.actor_role_sha256
                or actor_role.feature_dim != self.feature_dim
                or tuple(actor_role.feature_names) != tuple(self.feature_names)):
            return None
        if self.report.get("fit_method") == PATH_CONSTRAINED_METHOD:
            _validate_constrained_program_leaves(self)
        leaves = self.feedback_leaves if purpose == "feedback" else self.explanation_leaves
        if (self.report.get("fit_method") in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)
                and not _whole_tree_raw_legality_passed(
                    self.report.get("validation", {}).get(purpose, {}))):
            return None
        if not self.report["validation"][purpose + "_passed"]:
            return None
        evidence = self.lookup(sample)
        mask = np.asarray(sample["mask"], dtype=np.float32)
        if (evidence["leaf_id"] not in leaves or not evidence["raw_action_legal"]
                or (_requires_path_certificate(self)
                    and not evidence["raw_action_path_certified"])
                or evidence["action"] != actor_action
                or mask[ACTIONS.index(actor_action)] <= 0.5):
            return None
        return {**evidence, "eligible": True, "purpose": purpose,
                "actor_role_sha256": self.actor_role_sha256}

    def to_dict(self) -> dict:
        return {
            "version": (PATH_CONSTRAINED_PROGRAM_VERSION if self.report.get("fit_method") == PATH_CONSTRAINED_METHOD else
                        SELECTIVE_PROGRAM_VERSION if self.report.get("fit_method") == INDEPENDENT_SELECTIVE_METHOD else
                        JOINT_PROGRAM_VERSION if self.report.get("fit_method") == JOINT_CERTIFIED_TOPOLOGY_METHOD else
                        V616_PROGRAM_VERSION if _requires_path_certificate(self) else PROGRAM_VERSION), "role": self.role,
            "actor_role_sha256": self.actor_role_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
            "root_ref": int(self.root_ref),
            "split_weights": self.split_weights.tolist(),
            "split_bias": self.split_bias.tolist(),
            "left_refs": self.left_refs.tolist(),
            "right_refs": self.right_refs.tolist(),
            "leaf_logits": self.leaf_logits.tolist(),
            "feedback_leaves": sorted(self.feedback_leaves),
            "explanation_leaves": sorted(self.explanation_leaves),
            "report": self.report,
        }

    def sha256(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False))
        return sha256(target.read_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping, *, actor_role_sha256: str) -> "WarehouseV6Program":
        version = data.get("version")
        method = data.get("report", {}).get("fit_method")
        schemas = {FEATURE_SCHEMA_SHA256: FEATURE_DIM,
                   legacy_features.ACTOR_FEATURE_SCHEMA_SHA256: legacy_features.TREE_FEATURE_DIM}
        schema = data.get("feature_schema_sha256")
        dim = schemas.get(schema)
        if (version not in (PROGRAM_VERSION, JOINT_PROGRAM_VERSION, V616_PROGRAM_VERSION,
                            SELECTIVE_PROGRAM_VERSION, PATH_CONSTRAINED_PROGRAM_VERSION)
                or (version in (JOINT_PROGRAM_VERSION, V616_PROGRAM_VERSION,
                                SELECTIVE_PROGRAM_VERSION, PATH_CONSTRAINED_PROGRAM_VERSION)) !=
                   (method in (JOINT_CERTIFIED_TOPOLOGY_METHOD, V616_TOPOLOGY_METHOD,
                               INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD))
                or (version == JOINT_PROGRAM_VERSION and method != JOINT_CERTIFIED_TOPOLOGY_METHOD)
                or (version == V616_PROGRAM_VERSION and method != V616_TOPOLOGY_METHOD)
                or (version == SELECTIVE_PROGRAM_VERSION and method != INDEPENDENT_SELECTIVE_METHOD)
                or (version == PATH_CONSTRAINED_PROGRAM_VERSION and method != PATH_CONSTRAINED_METHOD)
                or dim is None
                or data.get("action_physics_config_sha256") !=
                   ACTION_PHYSICS_CONFIG_SHA256
                or data.get("role") not in AGENTS
                or data.get("actor_role_sha256") != actor_role_sha256):
            raise RuntimeError("Warehouse program version, schema, role or Actor hash mismatch")
        raw_weights = np.asarray(data["split_weights"], dtype=np.float32)
        if raw_weights.size == 0:
            raw_weights = np.zeros((0, dim), dtype=np.float32)
        elif raw_weights.ndim != 2 or raw_weights.shape[1] != dim:
            raise RuntimeError("Warehouse program feature schema and split tensor mismatch")
        result = cls(
            str(data["role"]), actor_role_sha256, int(data["root_ref"]),
            raw_weights,
            np.asarray(data["split_bias"], dtype=np.float32),
            np.asarray(data["left_refs"], dtype=np.int64),
            np.asarray(data["right_refs"], dtype=np.int64),
            np.asarray(data["leaf_logits"], dtype=np.float32),
            frozenset(map(int, data["feedback_leaves"])),
            frozenset(map(int, data["explanation_leaves"])),
            dict(data["report"]),
        )
        _, leaves, depth = result._structure()
        report = result.report
        if (report.get("max_depth") != depth
                or report.get("leaf_count") != leaves
                or ("feature_schema_sha256" in report and
                    report["feature_schema_sha256"] != schema)
                or ("actor_role_sha256" in report and
                    report["actor_role_sha256"] != actor_role_sha256)
                or report.get("max_features_per_node", MAX_SPLIT_FEATURES)
                    > MAX_SPLIT_FEATURES):
            raise RuntimeError("Warehouse program report complexity mismatch")
        if _requires_path_certificate(result):
            certificates = certified_program_actions(result)
            selected = result.feedback_leaves | result.explanation_leaves
            if any(int(np.argmax(result.leaf_logits[leaf])) not in certificates[leaf]
                   for leaf in selected):
                raise RuntimeError("Warehouse program eligible leaf lacks path legality proof")
        if method == PATH_CONSTRAINED_METHOD:
            _validate_constrained_program_leaves(result)
        return result

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             actor_role_sha256: str) -> "WarehouseV6Program":
        raw = Path(path).read_bytes()
        if not expected_sha256 or sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("Warehouse program artifact hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256)


def _fit_sparse_tree(role: str, actor: FrozenWarehouseV6Role,
                     rows: Sequence[Mapping], x: np.ndarray,
                     masks: np.ndarray) -> tuple[dict, dict]:
    # Only the fit episodes determine splits and pruning. Soft duplicated
    # labels make the target the complete masked NN distribution, not a
    # teacher action or the Actor's existing tree branch.
    from sklearn import __version__ as sklearn_version
    from sklearn.tree import DecisionTreeClassifier
    p = _masked_probabilities(actor.logits_encoded(x), masks)
    classifier = DecisionTreeClassifier(
        criterion="log_loss", max_depth=PROGRAM_MAX_DEPTH,
        min_weight_fraction_leaf=0.005,
        random_state=6200 + AGENTS.index(role))
    classifier.fit(np.repeat(x, len(ACTIONS), axis=0),
                   np.tile(np.arange(len(ACTIONS)), len(x)),
                   sample_weight=p.reshape(-1))
    path = classifier.decision_path(x).tocsc()
    episodes = np.asarray([str(row["episode"]) for row in rows])
    scenarios = np.asarray([str(row.get("base_seed", row["episode"])) for row in rows])
    local = {}
    for node in range(classifier.tree_.node_count):
        index = path.indices[path.indptr[node]:path.indptr[node + 1]]
        logits = np.log(np.maximum(p[index].mean(0), 1e-8)).astype(np.float32)
        target, mask = p[index], masks[index]
        q = _masked_probabilities(np.broadcast_to(logits, target.shape), mask)
        predicted = q.argmax(-1)
        nn_top = target.argmax(-1)
        raw_legal = mask[np.arange(len(index)), int(logits.argmax())] > 0.5
        agreement = float(np.mean(predicted == nn_top))
        kl = float(np.mean(np.sum(target * (
            np.log(np.maximum(target, _EPSILON))
            - np.log(np.maximum(q, _EPSILON))), axis=-1)))
        qualified = (len(index) >= MIN_LEAF_SUPPORT
                     and len(set(episodes[index])) >= MIN_EPISODE_SUPPORT
                     and len(set(scenarios[index])) >= MIN_EPISODE_SUPPORT
                     and agreement >= 0.95 and kl <= 0.25
                     and bool(raw_legal.all()))
        # Use expected actionable mass under the complete NN distribution.
        # Realized samples enter only independent validation and audit counts.
        actionable = float(np.sum((raw_legal & (predicted == nn_top))
                                  * target[np.arange(len(index)), nn_top]))
        explanation = float(np.sum(raw_legal
                                   * target[np.arange(len(index)), predicted]))
        local[node] = {
            "score": actionable + explanation if qualified else 0,
            "rows": index,
        }
    left = classifier.tree_.children_left
    right = classifier.tree_.children_right
    @lru_cache(None)
    def optimize(node: int, budget: int) -> tuple[float, tuple[int, ...]]:
        if budget == 1:
            return local[node]["score"], (node,)
        if left[node] < 0:
            return -1, ()
        best = (-1, ())
        for left_budget in range(1, budget):
            a, nodes_a = optimize(int(left[node]), left_budget)
            b, nodes_b = optimize(int(right[node]), budget - left_budget)
            if a >= 0 and b >= 0 and a + b > best[0]:
                best = a + b, nodes_a + nodes_b
        return best
    score, _, sources = max(
        [(optimize(0, budget)[0], budget, optimize(0, budget)[1])
         for budget in range(1, PROGRAM_MAX_LEAVES + 1)],
        key=lambda value: (value[0], -value[1]))
    selected = set(sources)
    weights, bias, left_refs, right_refs, source_leaves = [], [], [], [], []
    def compact(source: int) -> int:
        if source in selected:
            index = len(source_leaves)
            source_leaves.append(source)
            return -1 - index
        index = len(weights)
        row = np.zeros(FEATURE_DIM, dtype=np.float32)
        row[int(classifier.tree_.feature[source])] = 4.0
        weights.append(row)
        bias.append(-4.0 * float(classifier.tree_.threshold[source]))
        left_refs.append(0)
        right_refs.append(0)
        left_refs[index] = compact(int(left[source]))
        right_refs[index] = compact(int(right[source]))
        return index
    root = compact(0)
    structure = {
        "root_ref": root,
        "split_weights": np.stack(weights).astype(np.float32)
        if weights else np.zeros((0, FEATURE_DIM), dtype=np.float32),
        "split_bias": np.asarray(bias, dtype=np.float32),
        "left_refs": np.asarray(left_refs, dtype=np.int64),
        "right_refs": np.asarray(right_refs, dtype=np.int64),
        "leaf_count": len(source_leaves),
    }
    report = {
        "sklearn_version": sklearn_version,
        "fit_classifier_nodes": int(classifier.tree_.node_count),
        "fit_classifier_leaves": int(classifier.get_n_leaves()),
        "fit_prune_score": float(score),
        "fit_selected_source_nodes": sorted(map(int, sources)),
        "split_sha256": sha256(
            structure["split_weights"].tobytes()
            + structure["split_bias"].tobytes()
            + structure["left_refs"].tobytes()
            + structure["right_refs"].tobytes()).hexdigest(),
    }
    return structure, report


def _actor_sparse_topology(actor: FrozenWarehouseV6Role) -> tuple[dict, dict]:
    """Copy only the frozen Actor's public, hard, three-feature tree structure.

    Leaf action logits are deliberately excluded: they must be fitted against
    the *complete* masked NN distribution on separate fit episodes.
    """
    weights = actor.weights
    nodes = int(weights["tree.node_count"][0])
    leaves = int(weights["tree.leaf_count"][0])
    structure = {
        "root_ref": 0 if nodes else -1,
        "split_weights": np.asarray(
            weights["tree.split.weight"][:nodes], dtype=np.float32).copy(),
        "split_bias": np.asarray(
            weights["tree.split.bias"][:nodes], dtype=np.float32).copy(),
        "left_refs": np.asarray(
            weights["tree.left_refs"][:nodes], dtype=np.int64).copy(),
        "right_refs": np.asarray(
            weights["tree.right_refs"][:nodes], dtype=np.int64).copy(),
        "leaf_count": leaves,
    }
    digest = sha256(json.dumps({
        "root_ref": structure["root_ref"],
        "leaf_count": leaves,
        "feature_schema_sha256": (FEATURE_SCHEMA_SHA256
                                  if actor.feature_dim == FEATURE_DIM
                                  else legacy_features.ACTOR_FEATURE_SCHEMA_SHA256),
        "split_weights": structure["split_weights"].tolist(),
        "split_bias": structure["split_bias"].tolist(),
        "left_refs": structure["left_refs"].tolist(),
        "right_refs": structure["right_refs"].tolist(),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return structure, {
        "topology_origin": "frozen_actor_public_tree",
        "actor_tree_structure_sha256": digest,
        "actor_role_sha256": actor.role_sha256,
    }


def actor_sparse_topology_matches(program: WarehouseV6Program,
                                  actor: FrozenWarehouseV6Role, *,
                                  require_actor_binding: bool = True) -> bool:
    """Check a claimed extracted program against its bound frozen Actor."""
    if program.report.get("fit_method") not in (
            ACTOR_SPARSE_TOPOLOGY_METHOD, JOINT_CERTIFIED_TOPOLOGY_METHOD, V616_TOPOLOGY_METHOD):
        return False
    structure, origin = _actor_sparse_topology(actor)
    report = program.report.get("fit_tree", {})
    return bool(
        (not require_actor_binding or
         program.actor_role_sha256 == actor.role_sha256)
        and report.get("topology_origin") == origin["topology_origin"]
        and report.get("actor_tree_structure_sha256") ==
            origin["actor_tree_structure_sha256"]
        and report.get("actor_role_sha256") == program.actor_role_sha256
        and program.root_ref == structure["root_ref"]
        and np.array_equal(program.split_weights, structure["split_weights"])
        and np.array_equal(program.split_bias, structure["split_bias"])
        and np.array_equal(program.left_refs, structure["left_refs"])
        and np.array_equal(program.right_refs, structure["right_refs"])
        and len(program.leaf_logits) == structure["leaf_count"])


def _fit_leaf_logits(actor: FrozenWarehouseV6Role, features: np.ndarray,
                     masks: np.ndarray, leaf_ids: np.ndarray,
                     leaf_count: int, *,
                     empty_leaf_logits: np.ndarray | None = None) -> np.ndarray:
    # Fitting runs in the training environment. Frozen program inference and
    # validation remain importable in the NumPy-only study runtime.
    import torch

    target = _masked_probabilities(actor.logits_encoded(features), masks)
    learned = np.zeros((leaf_count, len(ACTIONS)), dtype=np.float32)
    for leaf in range(leaf_count):
        selected = leaf_ids == leaf
        if not np.any(selected):
            if empty_leaf_logits is None:
                raise RuntimeError("Warehouse program fit leaf has no fit samples")
            # This placeholder is never certified. Calibration and eligibility
            # cannot turn a leaf unseen in fit into program evidence.
            learned[leaf] = empty_leaf_logits[leaf]
            continue
        if "protected_leaf_mask" in actor.weights and actor.weights["protected_leaf_mask"][leaf] > 0.5:
            from .warehouse_v617_risk_tree import fit_boxed_logits
            learned[leaf] = fit_boxed_logits(target[selected], masks[selected],
                                             actor.weights["tree.leaf_logits"][leaf])
            continue
        prior = torch.from_numpy(np.log(np.maximum(target[selected].mean(0), 1e-8))
                                 .astype(np.float32))
        candidate = torch.nn.Parameter(prior.clone())
        legal = torch.from_numpy(masks[selected] > 0.5)
        p = torch.from_numpy(target[selected].astype(np.float32))
        optimizer = torch.optim.LBFGS([candidate], lr=0.5, max_iter=50,
                                      line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            logq = torch.log_softmax(candidate[None].masked_fill(~legal, -1e9), -1)
            loss = -(p * logq).sum(-1).mean() + 0.001 * (candidate - prior).square().mean()
            loss.backward()
            return loss
        optimizer.step(closure)
        learned[leaf] = candidate.detach().numpy()
    return learned


def _calibrate_leaf_logits(logits: np.ndarray, actor: FrozenWarehouseV6Role,
                           x: np.ndarray, masks: np.ndarray,
                           leaves: np.ndarray, *,
                           fitted_leaves: frozenset[int] | None = None,
                           respect_actor_leaf_bounds: bool = True) -> np.ndarray:
    target = _masked_probabilities(actor.logits_encoded(x), masks)
    result = logits.copy()
    for leaf in range(len(logits)):
        if fitted_leaves is not None and leaf not in fitted_leaves:
            continue
        selected = leaves == leaf
        if not np.any(selected):
            continue
        best, best_kl = logits[leaf], float("inf")
        for temperature in (0.8, 0.9, 1.0, 1.1, 1.2):
            trial = logits[leaf] / temperature
            if (respect_actor_leaf_bounds and "protected_leaf_mask" in actor.weights
                    and actor.weights["protected_leaf_mask"][leaf] > 0.5):
                from .warehouse_v617_risk_tree import project_logit_box
                trial = project_logit_box(trial, actor.weights["tree.leaf_logits"][leaf])
            q = _masked_probabilities(
                np.broadcast_to(trial, (int(selected.sum()), len(ACTIONS))),
                masks[selected])
            p = target[selected]
            kl = float(np.mean(np.sum(p * (
                np.log(np.maximum(p, _EPSILON))
                - np.log(np.maximum(q, _EPSILON))), axis=-1)))
            if kl < best_kl:
                best, best_kl = trial, kl
        result[leaf] = best
    return result


def _normalized_program_structure(structure: Mapping) -> dict:
    """Canonical structural bytes only; no old leaf logits enter new fitting."""
    if not isinstance(structure, Mapping):
        raise ValueError("fixed structure must be a mapping")
    weights = np.asarray(structure["split_weights"], np.float32)
    if weights.size == 0 and weights.ndim != 2:
        dim = structure.get("feature_dim")
        if dim not in (FEATURE_DIM, legacy_features.TREE_FEATURE_DIM):
            raise ValueError("empty structure needs its feature_dim")
        weights = np.zeros((0, dim), np.float32)
    bias = np.asarray(structure["split_bias"], np.float32)
    raw_left, raw_right = (np.asarray(structure[key]) for key in ("left_refs", "right_refs"))
    if (weights.ndim != 2 or weights.shape[1] not in (FEATURE_DIM, legacy_features.TREE_FEATURE_DIM)
            or bias.shape != (len(weights),) or raw_left.shape != bias.shape or raw_right.shape != bias.shape
            or any(not np.isfinite(a).all() for a in (weights, bias, raw_left, raw_right))
            or np.any(raw_left != np.floor(raw_left)) or np.any(raw_right != np.floor(raw_right))
            or type(structure.get("root_ref")) not in (int, np.int64, np.int32)
            or type(structure.get("leaf_count")) not in (int, np.int64, np.int32)):
        raise ValueError("invalid fixed program structure tensors")
    return {"root_ref": int(structure["root_ref"]), "split_weights": weights.copy(),
            "split_bias": bias.copy(), "left_refs": raw_left.astype(np.int64),
            "right_refs": raw_right.astype(np.int64), "leaf_count": int(structure["leaf_count"]),
            "feature_dim": weights.shape[1]}


def program_structure_sha256(structure: Mapping) -> str:
    normalized = _normalized_program_structure(structure)
    payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in normalized.items()}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _validate_constrained_program_leaves(program: WarehouseV6Program) -> None:
    """Every stored leaf keeps its optimized, path-certified float32 top."""
    from .warehouse_v619_leaf_fit import (LEAF_EPSILON, LEAF_REGULARIZATION,
        LEAF_MAXITER, LEAF_TOLERANCE, CALIBRATION_TEMPERATURES, float32_preference_proof)
    certificates = certified_program_actions(program)
    count = len(program.leaf_logits)
    fits = program.report.get("constrained_leaf_fits", {})
    calibrations = program.report.get("constrained_leaf_calibration", {})
    fit_counts = program.report.get("fit_leaf_row_counts")
    if (not isinstance(fits, Mapping) or set(fits) != {str(i) for i in range(count)}
            or not isinstance(calibrations, Mapping) or set(calibrations) != set(fits)
            or not isinstance(fit_counts, list) or len(fit_counts) != count):
        raise RuntimeError("constrained program lacks complete per-leaf optimization evidence")
    selected = program.feedback_leaves | program.explanation_leaves
    for leaf in range(count):
        fit, calibration = fits[str(leaf)], calibrations[str(leaf)]
        action = fit.get("chosen_action")
        if (fit.get("success") is not True or fit.get("float32_feasible") is not True
                or fit.get("solver") != "SLSQP" or fit.get("solver_status") != 0
                or fit.get("epsilon") != LEAF_EPSILON or fit.get("regularization") != LEAF_REGULARIZATION
                or fit.get("maxiter") != LEAF_MAXITER or fit.get("tolerance") != LEAF_TOLERANCE
                or type(fit.get("row_count")) is not int or fit["row_count"] < 0
                or fit["row_count"] != fit_counts[leaf]
                or (leaf in selected and fit["row_count"] == 0)
                or calibration.get("chosen_action") != action
                or calibration.get("temperature") not in CALIBRATION_TEMPERATURES
                or not float32_preference_proof(program.leaf_logits[leaf], action,
                    certificates[leaf], epsilon=LEAF_EPSILON)["passed"]):
            raise RuntimeError("constrained program leaf lost optimization/path/float32 certificate")


def fit_program(role: str, actor: FrozenWarehouseV6Actor,
                fit: Sequence[Mapping], calibration: Sequence[Mapping],
                eligibility: Sequence[Mapping],
                validation: Sequence[Mapping], *,
                method: str = AXIS_TREE_METHOD,
                fixed_structure: Mapping | None = None,
                fixed_structure_sha256: str | None = None) -> WarehouseV6Program:
    if role not in AGENTS:
        raise ValueError("invalid Warehouse program role")
    if method not in (AXIS_TREE_METHOD, ACTOR_SPARSE_TOPOLOGY_METHOD,
                      JOINT_CERTIFIED_TOPOLOGY_METHOD,
                      INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD):
        raise ValueError("unknown Warehouse program fitting method")
    if method != PATH_CONSTRAINED_METHOD and (fixed_structure is not None or fixed_structure_sha256 is not None):
        raise ValueError("fixed structures require the new constrained method")
    if (fixed_structure is None) != (fixed_structure_sha256 is None):
        raise ValueError("fixed structure and SHA must be supplied together")
    parts = {"fit": list(fit), "calibration": list(calibration),
             "eligibility": list(eligibility), "validation": list(validation)}
    episode_ids = {name: {str(row["episode"]) for row in rows}
                   for name, rows in parts.items()}
    # A scenario can be reset more than once inside a fixed training block.
    # Distinct episode IDs from the same base seed must stay in one split.
    scenario_ids = {
        name: {str(row.get("base_seed", row["episode"])) for row in rows}
        for name, rows in parts.items()
    }
    role_actor = actor.roles[role]
    for name, rows in parts.items():
        encoded, _ = _sample_arrays(rows, name, role_actor.feature_dim)
        if method in (JOINT_CERTIFIED_TOPOLOGY_METHOD, INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD):
            validate_certification_domain(
                encoded, _MASK_INDICES,
                certified_feature_bounds(tuple(role_actor.feature_names)))
        _sampled_indices(rows)
        if method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) and name != "fit":
            # A repeated physical frame is a real observation. It must not be
            # dropped merely because its state equals an earlier state.
            if any("frame" in row for row in rows):
                if any("frame" not in row for row in rows):
                    raise ValueError("natural program validation frame metadata is incomplete")
                by_episode: dict[str, list[int]] = {}
                for row in rows:
                    by_episode.setdefault(str(row["episode"]), []).append(int(row["frame"]))
                for frames in by_episode.values():
                    ordered = sorted(frames)
                    if (ordered[0] != 0 or len(set(frames)) != len(frames)
                            or ordered != list(range(ordered[0], ordered[-1]+1))):
                        raise ValueError("natural program validation contains duplicated or omitted physical frames")
    if any(episode_ids[left] & episode_ids[right]
           for left in parts for right in parts if left < right):
        raise ValueError("Warehouse program episode splits overlap")
    if any(scenario_ids[left] & scenario_ids[right]
           for left in parts for right in parts if left < right):
        raise ValueError("Warehouse program scenario seed splits overlap")
    fit_x, fit_masks = _sample_arrays(parts["fit"], "fit", role_actor.feature_dim)
    actor_topology = method in (ACTOR_SPARSE_TOPOLOGY_METHOD,
                                JOINT_CERTIFIED_TOPOLOGY_METHOD)
    if method == PATH_CONSTRAINED_METHOD:
        if fixed_structure is not None:
            if program_structure_sha256(fixed_structure) != fixed_structure_sha256:
                raise ValueError("fixed program structure SHA mismatch")
            structure = _normalized_program_structure(fixed_structure)
            if structure["feature_dim"] != role_actor.feature_dim:
                raise ValueError("fixed structure feature dimension differs from bound Actor")
            tree_report = {"topology_origin": "fixed_development_structure",
                           "fixed_structure_sha256": fixed_structure_sha256,
                           "topology_uses_fit_only": True,
                           "calibration_eligibility_validation_used": False,
                           "actor_role_sha256": role_actor.role_sha256}
        else:
            from .warehouse_v619_tree import fit_legal_structure
            structure, tree_report = fit_legal_structure(
                fit_x, fit_masks, _masked_probabilities(role_actor.logits_encoded(fit_x), fit_masks),
                [str(row["episode"]) for row in parts["fit"]],
                feature_names=role_actor.feature_names, actor_role_sha256=role_actor.role_sha256,
                actions=ACTIONS)
    elif method == INDEPENDENT_SELECTIVE_METHOD:
        from .warehouse_v618_selective_tree import fit_independent_structure
        structure, tree_report = fit_independent_structure(
            fit_x, fit_masks,
            _masked_probabilities(role_actor.logits_encoded(fit_x), fit_masks),
            [str(row["episode"]) for row in parts["fit"]],
            feature_names=role_actor.feature_names,
            actor_role_sha256=role_actor.role_sha256, actions=ACTIONS)
    elif actor_topology:
        structure, tree_report = _actor_sparse_topology(role_actor)
    else:
        structure, tree_report = _fit_sparse_tree(role, role_actor, parts["fit"],
                                                  fit_x, fit_masks)
    split_hashes = {
        name: sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
        for name, ids in episode_ids.items()
    }
    report = {
        "version": (PATH_CONSTRAINED_PROGRAM_VERSION if method == PATH_CONSTRAINED_METHOD else SELECTIVE_PROGRAM_VERSION
                    if method == INDEPENDENT_SELECTIVE_METHOD else JOINT_PROGRAM_VERSION
                    if method == JOINT_CERTIFIED_TOPOLOGY_METHOD
                    else PROGRAM_VERSION), "role": role,
        "actor_role_sha256": role_actor.role_sha256,
        "feature_schema_sha256": (FEATURE_SCHEMA_SHA256
                                  if role_actor.feature_dim == FEATURE_DIM
                                  else legacy_features.ACTOR_FEATURE_SCHEMA_SHA256),
        "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "max_depth": PROGRAM_MAX_DEPTH,
        "leaf_count": structure["leaf_count"],
        "max_features_per_node": MAX_SPLIT_FEATURES,
        "split_episodes": {name: len(ids) for name, ids in episode_ids.items()},
        "split_episode_ids": {name: sorted(ids) for name, ids in episode_ids.items()},
        "split_episode_sha256": split_hashes,
        "split_scenario_seed_ids": {
            name: sorted(ids) for name, ids in scenario_ids.items()},
        "fit_tree": tree_report,
        "leaf_eligibility": {}, "validation": {},
        "certificate_kind": "independent_empirical_audit_only",
    }
    if actor_topology or method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD):
        report["fit_method"] = method
    provisional = WarehouseV6Program(
        role, role_actor.role_sha256, structure["root_ref"],
        structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"],
        np.zeros((structure["leaf_count"], len(ACTIONS)), dtype=np.float32),
        frozenset(), frozenset(), report)
    report["max_depth"] = provisional._structure()[2]
    fit_leaves = provisional.leaf_indices(fit_x)
    fit_counts = np.bincount(fit_leaves, minlength=structure["leaf_count"])
    fitted_leaves = frozenset(np.flatnonzero(fit_counts).tolist())
    if actor_topology:
        # Even though structure came from the Actor, the program's path must
        # agree with frozen inference on every fit row before fitting logits.
        if not np.array_equal(fit_leaves, role_actor.leaf(fit_x)):
            raise RuntimeError("extracted Warehouse Actor tree route mismatch")
        if method == JOINT_CERTIFIED_TOPOLOGY_METHOD:
            for name in ("calibration", "eligibility", "validation"):
                x, _ = _sample_arrays(parts[name], name)
                if not np.array_equal(provisional.leaf_indices(x),
                                      role_actor.leaf(x)):
                    raise RuntimeError(
                        f"extracted Warehouse Actor tree route mismatch in {name}")
        report["fit_leaf_row_counts"] = fit_counts.astype(int).tolist()
    if method == PATH_CONSTRAINED_METHOD:
        from .warehouse_v619_leaf_fit import fit_constrained_leaf_distribution
        certificates = certified_program_actions(provisional)
        target = _masked_probabilities(role_actor.logits_encoded(fit_x), fit_masks)
        if fixed_structure is None:
            # P2's winning leaves already optimized the same full NN target.
            # Reuse them after independent binding/proof checks; a second fit
            # would exceed the registered end-to-end 20k leaf-call budget.
            from .warehouse_v619_leaf_fit import float32_preference_proof
            fitted_reports = tree_report.get("fit_leaf_reports")
            target_sha = sha256(np.ascontiguousarray(target, dtype="<f4").tobytes()).hexdigest()
            if (tree_report.get("fit_target_sha256") != target_sha
                    or not isinstance(fitted_reports, list)
                    or len(fitted_reports) != structure["leaf_count"]):
                raise RuntimeError("constrained search leaf target/report binding mismatch")
            leaf_fits = []
            for leaf, item in enumerate(fitted_reports):
                optimizer_report = item.get("optimizer", {})
                logits = np.asarray(structure["initial_leaf_logits"][leaf], np.float32).copy()
                if (item.get("leaf") != leaf or item.get("rows") != int(fit_counts[leaf])
                        or optimizer_report.get("row_count") != int(fit_counts[leaf])
                        or set(item.get("path_legal_actions", ())) != set(certificates[leaf])
                        or item.get("raw_action") != optimizer_report.get("chosen_action")
                        or not float32_preference_proof(logits,
                            optimizer_report.get("chosen_action"), certificates[leaf])["passed"]):
                    raise RuntimeError("constrained search leaf route/count/certificate binding mismatch")
                leaf_fits.append((logits, dict(optimizer_report)))
            report["leaf_optimization_calls_outside_search"] = 0
            report["convex_solver_calls_outside_search"] = 0
        else:
            leaf_fits = [fit_constrained_leaf_distribution(
                target[fit_leaves == leaf], fit_masks[fit_leaves == leaf], certificates[leaf])
                for leaf in range(structure["leaf_count"])]
            report["leaf_optimization_calls_outside_search"] = len(leaf_fits)
            report["convex_solver_calls_outside_search"] = sum(
                item[1]["enumerated_action_solves"] for item in leaf_fits)
        learned = np.stack([result[0] for result in leaf_fits])
        report["constrained_leaf_fits"] = {str(leaf): result[1] for leaf, result in enumerate(leaf_fits)}
        report["fit_leaf_row_counts"] = fit_counts.astype(int).tolist()
        report["sampling_denominator"] = "all_recorded_physical_pre_action_frames"
        report["excluded_frames"] = 0
        report["fixed_structure_sha256"] = fixed_structure_sha256
        report["program_structure_sha256"] = program_structure_sha256(structure)
    elif method == INDEPENDENT_SELECTIVE_METHOD:
        # This is a new independent partition. Actor leaf numbers and their
        # legacy box bounds have no meaning here.
        from .warehouse_v618_selective_tree import fit_leaf_distribution
        target = _masked_probabilities(role_actor.logits_encoded(fit_x), fit_masks)
        learned = np.stack([fit_leaf_distribution(
            target[fit_leaves == leaf], fit_masks[fit_leaves == leaf])
            for leaf in range(structure["leaf_count"])])
        report["fit_leaf_row_counts"] = fit_counts.astype(int).tolist()
        report["sampling_denominator"] = "all_recorded_physical_pre_action_frames"
        report["excluded_frames"] = 0
    else:
        learned = _fit_leaf_logits(role_actor, fit_x, fit_masks, fit_leaves,
                                   structure["leaf_count"],
                                   empty_leaf_logits=(role_actor.weights[
                                       "tree.leaf_logits"][:structure["leaf_count"]]
                                       if actor_topology else None))
    cal_x, cal_masks = _sample_arrays(parts["calibration"], "calibration", role_actor.feature_dim)
    if method == PATH_CONSTRAINED_METHOD:
        from .warehouse_v619_leaf_fit import calibrate_constrained_leaf_distribution
        target = _masked_probabilities(role_actor.logits_encoded(cal_x), cal_masks)
        cal_leaves = provisional.leaf_indices(cal_x)
        calibrated_results = [calibrate_constrained_leaf_distribution(
            learned[leaf], target[cal_leaves == leaf], cal_masks[cal_leaves == leaf],
            certificates[leaf], chosen_action=leaf_fits[leaf][1]["chosen_action"])
            for leaf in range(structure["leaf_count"])]
        calibrated = np.stack([result[0] for result in calibrated_results])
        report["constrained_leaf_calibration"] = {
            str(leaf): result[1] for leaf, result in enumerate(calibrated_results)}
    else:
        calibrated = _calibrate_leaf_logits(
            learned, role_actor, cal_x, cal_masks, provisional.leaf_indices(cal_x),
            fitted_leaves=(fitted_leaves if actor_topology or method == INDEPENDENT_SELECTIVE_METHOD else None),
            respect_actor_leaf_bounds=(method != INDEPENDENT_SELECTIVE_METHOD))
    provisional = WarehouseV6Program(
        role, role_actor.role_sha256, structure["root_ref"],
        structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"], calibrated,
        frozenset(), frozenset(), report)
    path_certificates = (certified_program_actions(provisional)
                         if method in (JOINT_CERTIFIED_TOPOLOGY_METHOD,
                                       INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) else {})
    if path_certificates:
        report["leaf_path_certificates"] = {
            str(leaf): {
                "proven_actions": [ACTIONS[action] for action in sorted(actions)],
                "raw_top_action": ACTIONS[int(np.argmax(calibrated[leaf]))],
                "raw_top_path_certified":
                    int(np.argmax(calibrated[leaf])) in actions,
            }
            for leaf, actions in path_certificates.items()
        }
        report["path_certificate_rule"] = (
            "pure_legal_bit_ancestor+certified_mask_branches.v2")
    if method == PATH_CONSTRAINED_METHOD:
        _validate_constrained_program_leaves(provisional)
    feedback, explanation = set(), set()
    metric_kwargs = ({"context": _metric_context(
        provisional, role_actor, parts["eligibility"])}
        if method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) else {})
    # Eligibility alone locks the leaf sets. Internal validation is evaluated
    # once, after the selection, and cannot remove or add a leaf.
    for leaf in range(structure["leaf_count"]):
        f = _metrics(provisional, role_actor, parts["eligibility"], {leaf}, "feedback", **metric_kwargs)
        e = _metrics(provisional, role_actor, parts["eligibility"], {leaf}, "explanation", **metric_kwargs)
        report["leaf_eligibility"][str(leaf)] = {"feedback": f, "explanation": e}
        has_path_proof = (method not in (JOINT_CERTIFIED_TOPOLOGY_METHOD,
                                        INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) or
                          int(np.argmax(calibrated[leaf])) in
                          path_certificates[leaf])
        if leaf in fitted_leaves and has_path_proof and (method != JOINT_CERTIFIED_TOPOLOGY_METHOD or role_actor.weights["protected_leaf_mask"][leaf] > 0.5) and _eligible(f, FEEDBACK_AGREEMENT):
            feedback.add(leaf)
        if leaf in fitted_leaves and has_path_proof and (method != JOINT_CERTIFIED_TOPOLOGY_METHOD or role_actor.weights["protected_leaf_mask"][leaf] > 0.5) and _eligible(e, EXPLANATION_AGREEMENT):
            explanation.add(leaf)
    program = WarehouseV6Program(
        role, role_actor.role_sha256, structure["root_ref"],
        structure["split_weights"], structure["split_bias"],
        structure["left_refs"], structure["right_refs"], calibrated,
        frozenset(feedback), frozenset(explanation), report)
    metric_kwargs = ({"context": _metric_context(
        program, role_actor, parts["validation"])}
        if method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) else {})
    feedback_metric = _metrics(program, role_actor, parts["validation"],
                               feedback, "feedback", **metric_kwargs)
    explanation_metric = _metrics(program, role_actor, parts["validation"],
                                  explanation, "explanation", **metric_kwargs)
    report["validation"] = {
        "feedback": feedback_metric, "explanation": explanation_metric,
        "feedback_online_passed": bool(
            feedback_metric["rows"] > 0
            and feedback_metric["agreement"] >= FEEDBACK_AGREEMENT
            and feedback_metric["kl"] is not None
            and feedback_metric["kl"] <= MAX_COVERED_KL
            and feedback_metric["illegal_count"] == 0
            and feedback_metric["path_uncertified_count"] == 0
            and (method not in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) or
                 _whole_tree_raw_legality_passed(feedback_metric))),
        "explanation_online_passed": bool(
            explanation_metric["rows"] > 0
            and explanation_metric["agreement"] >= EXPLANATION_AGREEMENT
            and explanation_metric["kl"] is not None
            and explanation_metric["kl"] <= MAX_COVERED_KL
            and explanation_metric["illegal_count"] == 0
            and explanation_metric["path_uncertified_count"] == 0
            and (method not in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) or
                 _whole_tree_raw_legality_passed(explanation_metric))),
        "feedback_passed": _passes(feedback_metric, FEEDBACK_AGREEMENT,
            require_whole_tree_zero=method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)),
        "explanation_passed": _passes(explanation_metric, EXPLANATION_AGREEMENT,
            require_whole_tree_zero=method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)),
        "development_feedback_passed": _passes(
            feedback_metric, FEEDBACK_AGREEMENT, development=True,
            require_whole_tree_zero=method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)),
        "development_explanation_passed": _passes(
            explanation_metric, EXPLANATION_AGREEMENT, development=True,
            require_whole_tree_zero=method in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)),
    }
    return program


def current_actor_fidelity(program: WarehouseV6Program,
                           actor: FrozenWarehouseV6Actor,
                           rows: Sequence[Mapping], *,
                           purpose: str = "feedback") -> dict:
    """Recheck fixed validation rows against the current live Actor snapshot."""
    leaves = (set(program.feedback_leaves) if purpose == "feedback"
              else set(program.explanation_leaves))
    if program.report.get("fit_method") == PATH_CONSTRAINED_METHOD:
        _validate_constrained_program_leaves(program)
    metric = _metrics(program, actor.roles[program.role], rows, leaves, purpose)
    topology_matches = (not _requires_actor_topology(program) or
                        actor_sparse_topology_matches(
                            program, actor.roles[program.role],
                            require_actor_binding=False))
    metric["current_actor_topology_matches_program"] = topology_matches
    metric["current_actor_role_sha256"] = actor.role_sha256[program.role]
    metric["program_sha256"] = program.sha256()
    metric["passed"] = bool(
        topology_matches
        and program.report["validation"][purpose + "_passed"]
        and (program.report.get("fit_method") not in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD) or
             _whole_tree_raw_legality_passed(program.report["validation"].get(purpose, {})))
        and _passes(metric, FEEDBACK_AGREEMENT if purpose == "feedback"
                    else EXPLANATION_AGREEMENT, development=True,
                    require_whole_tree_zero=program.report.get("fit_method") in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)))
    return metric


def audit_frozen_program(program: WarehouseV6Program,
                         actor: FrozenWarehouseV6Actor,
                         rows: Sequence[Mapping]) -> dict:
    if actor.role_sha256[program.role] != program.actor_role_sha256:
        raise RuntimeError("Warehouse program Actor binding mismatch")
    if (program.report.get("fit_method") in (
            ACTOR_SPARSE_TOPOLOGY_METHOD, JOINT_CERTIFIED_TOPOLOGY_METHOD, V616_TOPOLOGY_METHOD)
            and not actor_sparse_topology_matches(
                program, actor.roles[program.role])):
        raise RuntimeError("Warehouse extracted program topology differs from Actor")
    if _requires_path_certificate(program):
        certificates = certified_program_actions(program)
        if any(int(np.argmax(program.leaf_logits[leaf])) not in certificates[leaf]
               for leaf in program.feedback_leaves | program.explanation_leaves):
            raise RuntimeError("Warehouse program eligible leaf lacks path legality proof")
    if program.report.get("fit_method") == PATH_CONSTRAINED_METHOD:
        _validate_constrained_program_leaves(program)
    ids = {str(row["episode"]) for row in rows}
    prior = program.report["split_episode_ids"]
    if any(ids & set(used) for used in prior.values()):
        raise ValueError("Warehouse independent audit reused fitting episodes")
    scenario_ids = {str(row.get("base_seed", row["episode"])) for row in rows}
    prior_scenarios = program.report["split_scenario_seed_ids"]
    if any(scenario_ids & set(used) for used in prior_scenarios.values()):
        raise ValueError("Warehouse independent audit reused fitting scenario seeds")
    role_actor = actor.roles[program.role]
    feedback = _metrics(program, role_actor, rows,
                        set(program.feedback_leaves), "feedback")
    explanation = _metrics(program, role_actor, rows,
                           set(program.explanation_leaves), "explanation")
    independent = program.report.get("fit_method") in (INDEPENDENT_SELECTIVE_METHOD, PATH_CONSTRAINED_METHOD)
    feedback_passed = bool(program.report["validation"]["feedback_passed"]
        and (not independent or _whole_tree_raw_legality_passed(
            program.report["validation"].get("feedback", {})))
        and _passes(feedback, FEEDBACK_AGREEMENT, require_whole_tree_zero=independent))
    explanation_passed = bool(
        program.report["validation"]["explanation_passed"]
        and (not independent or _whole_tree_raw_legality_passed(
            program.report["validation"].get("explanation", {})))
        and _passes(explanation, EXPLANATION_AGREEMENT, require_whole_tree_zero=independent))
    return {
        "audit_version": "warehouse-role-program-independent.v6.2",
        "role": program.role, "actor_role_sha256": program.actor_role_sha256,
        "program_sha256": program.sha256(),
        "feature_schema_sha256": program.feature_schema_sha256,
        "episode_ids": sorted(ids), "episode_count": len(ids),
        "sample_count": len(rows),
        "max_depth": int(program.report["max_depth"]),
        "leaf_count": int(program.report["leaf_count"]),
        "max_features_per_node": MAX_SPLIT_FEATURES,
        "certificate_kind": "independent_empirical_audit_only",
        "feedback": feedback, "explanation": explanation,
        "feedback_eligible": feedback_passed,
        "explanation_eligible": explanation_passed,
        "passed": bool(feedback_passed and explanation_passed),
    }


__all__ = [
    "PROGRAM_VERSION", "JOINT_PROGRAM_VERSION", "JOINT_CERTIFIED_TOPOLOGY_METHOD",
    "SELECTIVE_PROGRAM_VERSION", "INDEPENDENT_SELECTIVE_METHOD",
    "PATH_CONSTRAINED_PROGRAM_VERSION", "PATH_CONSTRAINED_METHOD", "program_structure_sha256",
    "WarehouseV6Program", "fit_program", "certified_actor_actions",
    "certified_program_actions", "actor_sparse_topology_matches",
    "audit_frozen_program", "current_actor_fidelity",
]
