"""Bounded sparse-oblique Warehouse program for frozen dual-Actor RCPD.

Every predicate is a threshold on a learned linear combination of at most
three public pre-action observation features.  Projection coefficients, tree
thresholds, leaf distributions, and leaf eligibility are learned only from
disjoint program-development episodes.  This program supplies gradients and
audited evidence; it never chooses an environment action for an Actor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from backend.training.warehouse_selective_v5 import (
    FEATURE_SETS as AXIS_FEATURE_SETS,
    observable_features,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, OBSERVATION_DIM, observation_feature_names


PROGRAM_VERSION = "warehouse-sparse-oblique-v5.0"
PRESET_VERSION = "warehouse-sparse-oblique-role-preset-v5.0"
PROJECTION_VERSION = "fit-only-action-probability-ridge-triples-v1"
ONLINE_GATE_VERSION = "disjoint-eligibility-and-validation-leaf-v2"
MAX_DEPTH = 5
MAX_LEAVES = 16
MAX_FEATURES_PER_NODE = 3
MIN_LEAF_SUPPORT = 20
MIN_COVERAGE = 0.20
FEEDBACK_AGREEMENT = 0.85
AI_EXPLANATION_AGREEMENT = 0.90
MAX_COVERED_KL = 0.35
_OBS_NAMES = observation_feature_names()
_RAW_ORDER_NAMES = tuple(name for name in _OBS_NAMES if name.startswith("order_"))
FEATURE_SETS = {
    name: tuple(axis_names) + _RAW_ORDER_NAMES
    for name, axis_names in AXIS_FEATURE_SETS.items()
}
ROLE_PRESETS = {
    "robot_1": {"feature_mode": "all", "hard_weight": 0.5,
                "calibration_fit_weight": 0.5},
    "robot_2": {"feature_mode": "no_previous", "hard_weight": 0.0,
                "calibration_fit_weight": 0.5},
}


def _features(sample: Mapping) -> dict[str, float]:
    result = observable_features(sample)
    obs = np.asarray(sample["obs"], dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,):
        raise ValueError("invalid Warehouse pre-action observation")
    result.update({name: float(value) for name, value in zip(_OBS_NAMES, obs, strict=True)
                   if name.startswith("order_")})
    return result


def _matrix(rows: Sequence[Mapping], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[entry[name] for name in names]
                       for row in rows for entry in (_features(row),)], dtype=np.float32)


def _actor_probabilities(rows: Sequence[Mapping],
                         oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]]) -> np.ndarray:
    result = []
    for row in rows:
        raw = oracle(row)
        vector = np.asarray([raw[action] for action in ACTIONS]
                            if isinstance(raw, Mapping) else raw, dtype=np.float64)
        legal = np.asarray(row["mask"], dtype=np.float64) > 0.5
        if (vector.shape != (len(ACTIONS),) or legal.shape != (len(ACTIONS),)
                or not np.isfinite(vector).all() or not np.any(legal)
                or np.any(vector < -1e-8) or np.any(vector[~legal] > 1e-6)
                or vector.sum() <= 0):
            raise ValueError("oracle must return a legal masked Actor distribution")
        vector = np.maximum(vector, 0.0) * legal
        result.append(vector / vector.sum())
    return np.stack(result)


def _episodes(rows: Sequence[Mapping], name: str) -> set[str]:
    if not rows or any("episode" not in row for row in rows):
        raise ValueError(f"{name} requires episode identities")
    return {str(row["episode"]) for row in rows}


def _metrics(probabilities: np.ndarray, leaf_ids: np.ndarray,
             means: Mapping[int, np.ndarray], masks: np.ndarray,
             selected: set[int] | frozenset[int]) -> dict:
    chosen = np.isin(leaf_ids, tuple(selected))
    count = int(chosen.sum())
    result = {"rows": count, "coverage": float(count / len(leaf_ids)),
              "agreement": 0.0, "kl": None, "illegal_count": 0}
    if not count:
        return result
    q = np.stack([means[int(leaf)] for leaf in leaf_ids[chosen]])
    p = probabilities[chosen]
    legal = masks[chosen] > 0.5
    actions = q.argmax(axis=1)
    result["illegal_count"] = int((~legal[np.arange(count), actions]).sum())
    result["agreement"] = float(np.mean(actions == p.argmax(axis=1)))
    q = np.maximum(q * legal, 1e-8)
    q /= q.sum(axis=1, keepdims=True)
    result["kl"] = float(np.mean(np.sum(
        p * (np.log(np.maximum(p, 1e-8)) - np.log(q)), axis=1)))
    return result


def _eligible_leaf(metrics: Mapping, agreement: float) -> bool:
    return (metrics["support"] >= MIN_LEAF_SUPPORT
            and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


def _passes(metrics: Mapping, agreement: float) -> bool:
    return (metrics["coverage"] >= MIN_COVERAGE
            and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


def _passes_online(metrics: Mapping, agreement: float) -> bool:
    return (metrics["rows"] > 0 and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


def _candidate_projections(x: np.ndarray, probabilities: np.ndarray,
                           hard_weight: float) -> list[dict]:
    """Learn at most three-feature projections using only the fit split."""
    result = [{"features": [index], "coefficients": [1.0]}
              for index in range(x.shape[1])]
    centered = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    hard = np.eye(len(ACTIONS), dtype=np.float64)[probabilities.argmax(axis=1)]
    for action_index in range(len(ACTIONS)):
        target = (probabilities[:, action_index]
                  + hard_weight * hard[:, action_index]) / (1.0 + hard_weight)
        target = target - target.mean()
        scores = np.abs(np.einsum("ij,i->j", centered, target,
                                   optimize=False)) / np.maximum(norms, 1e-8)
        ranked = np.argsort(-scores, kind="stable")[:7]
        for width in (2, 3):
            for feature_ids in combinations(sorted(map(int, ranked)), width):
                columns = centered[:, feature_ids]
                gram = np.einsum("ni,nj->ij", columns, columns,
                                 optimize=False)
                ridge = 1e-3 * float(np.trace(gram)) / width + 1e-6
                coefficients = np.linalg.solve(
                    gram + ridge * np.eye(width),
                    np.einsum("ij,i->j", columns, target,
                              optimize=False))
                scale = float(np.max(np.abs(coefficients)))
                if not np.isfinite(scale) or scale <= 1e-10:
                    continue
                coefficients /= scale
                result.append({"features": list(feature_ids),
                               "coefficients": [float(v) for v in coefficients]})
    return result


def _project(x: np.ndarray, projections: Sequence[Mapping]) -> np.ndarray:
    return np.column_stack([
        np.einsum("ij,j->i", x[:, projection["features"]],
                  np.asarray(projection["coefficients"], dtype=np.float32),
                  optimize=False)
        for projection in projections
    ]).astype(np.float32)


@dataclass(frozen=True)
class SparseObliqueProgram:
    role: str
    actor_role_sha256: str
    feature_names: tuple[str, ...]
    tree: dict
    projections: tuple[dict, ...]
    leaf_probabilities: dict[int, tuple[float, ...]]
    leaf_audit: dict[int, dict]
    leaf_validation_audit: dict[int, dict]
    feedback_leaves: frozenset[int]
    explanation_leaves: frozenset[int]
    report: dict
    _sha256_cache: str | None = field(default=None, init=False, repr=False,
                                      compare=False)

    def _leaf(self, sample: Mapping) -> tuple[int, list[dict]]:
        values = _features(sample)
        node, path = 0, []
        while self.tree["left"][node] >= 0:
            projection = self.projections[self.tree["projection"][node]]
            predicate = [
                {"feature": self.feature_names[index], "coefficient": coefficient,
                 "observed": values[self.feature_names[index]]}
                for index, coefficient in zip(projection["features"],
                                              projection["coefficients"], strict=True)
            ]
            # Match the float32 projection matrix used while fitting the tree.
            value = float(np.einsum("i,i->",
                np.asarray([entry["observed"] for entry in predicate], dtype=np.float32),
                np.asarray([entry["coefficient"] for entry in predicate],
                           dtype=np.float32), optimize=False))
            threshold = self.tree["threshold"][node]
            left = value <= threshold
            path.append({"terms": predicate, "operator": "<=" if left else ">",
                         "threshold": threshold, "observed": value})
            node = self.tree["left" if left else "right"][node]
        return node, path

    def lookup(self, sample: Mapping) -> dict:
        leaf, path = self._leaf(sample)
        distribution = self.leaf_probabilities[leaf]
        return {"leaf_id": leaf, "path": path,
                "action": ACTIONS[int(np.argmax(distribution))],
                "probabilities": dict(zip(ACTIONS, distribution, strict=True)),
                "leaf_audit": self.leaf_audit.get(leaf),
                "feedback_leaf": leaf in self.feedback_leaves,
                "explanation_leaf": leaf in self.explanation_leaves,
                "program_sha256": self.sha256()}

    execute = lookup

    def covered(self, sample: Mapping, *, actor_action: str,
                purpose: str) -> dict | None:
        if purpose not in ("feedback", "explanation"):
            raise ValueError("unknown sparse program purpose")
        if not self.report["validation"][purpose + "_online_passed"]:
            return None
        evidence = self.lookup(sample)
        leaves = (self.feedback_leaves if purpose == "feedback"
                  else self.explanation_leaves)
        mask = np.asarray(sample["mask"], dtype=np.float32)
        action = evidence["action"]
        if (evidence["leaf_id"] not in leaves or actor_action != action
                or mask[ACTIONS.index(action)] <= 0.5):
            return None
        return {**evidence, "eligible": True, "purpose": purpose,
                "validation": self.report["validation"][purpose]}

    def to_dict(self) -> dict:
        return {"version": PROGRAM_VERSION, "role": self.role,
                "actor_role_sha256": self.actor_role_sha256,
                "feature_names": list(self.feature_names),
                "actions": list(ACTIONS), "tree": self.tree,
                "projections": list(self.projections),
                "leaf_probabilities": {str(key): list(value)
                                       for key, value in self.leaf_probabilities.items()},
                "leaf_audit": {str(key): value for key, value in self.leaf_audit.items()},
                "leaf_validation_audit": {str(key): value
                                          for key, value in self.leaf_validation_audit.items()},
                "feedback_leaves": sorted(self.feedback_leaves),
                "explanation_leaves": sorted(self.explanation_leaves),
                "report": self.report}

    def sha256(self) -> str:
        if self._sha256_cache is None:
            digest = sha256(json.dumps(self.to_dict(), sort_keys=True,
                                       separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
            object.__setattr__(self, "_sha256_cache", digest)
        return self._sha256_cache

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), sort_keys=True,
                                     separators=(",", ":"),
                                     allow_nan=False), encoding="utf-8")
        digest = sha256(target.read_bytes()).hexdigest()
        if digest != self.sha256():
            raise RuntimeError("Warehouse sparse program changed after audit")
        return digest

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             actor_role_sha256: str) -> "SparseObliqueProgram":
        raw = Path(path).read_bytes()
        if sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("Warehouse sparse program artifact hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256)

    @classmethod
    def from_dict(cls, data: Mapping, *, actor_role_sha256: str) -> "SparseObliqueProgram":
        role = data.get("role")
        preset = ROLE_PRESETS.get(role)
        names = FEATURE_SETS[preset["feature_mode"]] if preset else ()
        if (data.get("version") != PROGRAM_VERSION or role not in AGENTS
                or data.get("actor_role_sha256") != actor_role_sha256
                or data.get("feature_names") != list(names)
                or data.get("actions") != list(ACTIONS)):
            raise RuntimeError("Warehouse sparse program binding mismatch")
        projections = tuple(data["projections"])
        if not 0 <= len(projections) <= MAX_LEAVES - 1:
            raise RuntimeError("Warehouse sparse projection count invalid")
        for projection in projections:
            indices = projection.get("features", [])
            coefficients = projection.get("coefficients", [])
            if (not 1 <= len(indices) <= MAX_FEATURES_PER_NODE
                    or len(indices) != len(coefficients)
                    or len(set(indices)) != len(indices)
                    or any(not isinstance(i, int) or not 0 <= i < len(names)
                           for i in indices)
                    or not np.isfinite(coefficients).all()
                    or max(abs(float(v)) for v in coefficients) <= 0):
                raise RuntimeError("Warehouse sparse node exceeds feature or coefficient bounds")
        tree = data["tree"]
        keys = ("left", "right", "projection", "threshold")
        size = len(tree["left"])
        if (not 1 <= size <= 2 * MAX_LEAVES - 1
                or any(len(tree[key]) != size for key in keys)):
            raise RuntimeError("Warehouse sparse tree structure invalid")
        leaves, seen, used, leaf_depths = set(), set(), set(), []

        def visit(node: int, depth: int) -> None:
            if node in seen or node < 0 or node >= size or depth > MAX_DEPTH:
                raise RuntimeError("Warehouse sparse tree depth or cycle invalid")
            seen.add(node)
            left, right = tree["left"][node], tree["right"][node]
            if left == right == -1:
                if tree["projection"][node] != -1:
                    raise RuntimeError("Warehouse sparse leaf has predicate")
                leaves.add(node)
                leaf_depths.append(depth)
                return
            index = tree["projection"][node]
            if (not isinstance(index, int) or not 0 <= index < len(projections)
                    or not np.isfinite(tree["threshold"][node])):
                raise RuntimeError("Warehouse sparse predicate invalid")
            used.add(index)
            visit(left, depth + 1)
            visit(right, depth + 1)

        visit(0, 0)
        if (len(seen) != size or len(leaves) > MAX_LEAVES
                or used != set(range(len(projections)))):
            raise RuntimeError("Warehouse sparse tree has unreachable or excess nodes")
        probabilities = {int(key): tuple(map(float, value))
                         for key, value in data["leaf_probabilities"].items()}
        if (set(probabilities) != leaves
                or any(len(values) != len(ACTIONS) or not np.isfinite(values).all()
                       or min(values) < 0 or abs(sum(values) - 1) > 1e-5
                       for values in probabilities.values())):
            raise RuntimeError("Warehouse sparse leaf distribution invalid")
        feedback = frozenset(map(int, data["feedback_leaves"]))
        explanation = frozenset(map(int, data["explanation_leaves"]))
        if not feedback <= leaves or not explanation <= feedback:
            raise RuntimeError("Warehouse sparse eligible leaves invalid")
        report = data["report"]
        if (report.get("role") != role or report.get("actor_role_sha256") != actor_role_sha256
                or report.get("preset_version") != PRESET_VERSION
                or report.get("projection_version") != PROJECTION_VERSION
                or report.get("online_gate_version") != ONLINE_GATE_VERSION
                or report.get("feature_mode") != preset["feature_mode"]
                or report.get("hard_weight") != preset["hard_weight"]
                or report.get("calibration_fit_weight") != preset["calibration_fit_weight"]
                or report.get("leaf_count") != len(leaves)
                or report.get("max_depth") != max(leaf_depths)
                or report.get("max_features_per_node") != MAX_FEATURES_PER_NODE):
            raise RuntimeError("Warehouse sparse audit provenance invalid")
        split_ids = report.get("split_episode_ids", {})
        parts = {"fit", "calibration", "eligibility", "validation"}
        if set(split_ids) != parts:
            raise RuntimeError("Warehouse sparse audit episode splits missing")
        if any((not isinstance(split_ids[name], list)
                or len(split_ids[name]) != report["split_episodes"][name]
                or sha256(json.dumps(sorted(split_ids[name])).encode()).hexdigest()
                   != report["split_episode_sha256"][name]) for name in parts):
            raise RuntimeError("Warehouse sparse audit split hash mismatch")
        if any(set(split_ids[left]) & set(split_ids[right])
               for left in parts for right in parts if left < right):
            raise RuntimeError("Warehouse sparse audit episode overlap")
        leaf_audit = {int(key): value for key, value in data["leaf_audit"].items()}
        leaf_validation_audit = {int(key): value
                                 for key, value in data["leaf_validation_audit"].items()}
        if set(leaf_audit) != leaves or set(leaf_validation_audit) != leaves:
            raise RuntimeError("Warehouse sparse audit leaves mismatch")
        expected_feedback = {leaf for leaf in leaves
                             if _eligible_leaf(leaf_audit[leaf], FEEDBACK_AGREEMENT)
                             and _eligible_leaf(leaf_validation_audit[leaf],
                                                FEEDBACK_AGREEMENT)}
        expected_explanation = {leaf for leaf in expected_feedback
                                if role == "robot_2"
                                and _eligible_leaf(leaf_audit[leaf],
                                                   AI_EXPLANATION_AGREEMENT)
                                and _eligible_leaf(leaf_validation_audit[leaf],
                                                   AI_EXPLANATION_AGREEMENT)}
        validation = report["validation"]
        if (feedback != expected_feedback or explanation != expected_explanation
                or validation["feedback_online_passed"] !=
                   _passes_online(validation["feedback"], FEEDBACK_AGREEMENT)
                or validation["explanation_online_passed"] !=
                   (role == "robot_2" and _passes_online(
                       validation["explanation"], AI_EXPLANATION_AGREEMENT))
                or validation["feedback_passed"] !=
                   _passes(validation["feedback"], FEEDBACK_AGREEMENT)
                or validation["explanation_passed"] !=
                   (role == "robot_2" and _passes(
                       validation["explanation"], AI_EXPLANATION_AGREEMENT))):
            raise RuntimeError("Warehouse sparse eligibility audit mismatch")
        return cls(role, actor_role_sha256, names, tree, projections, probabilities,
                   leaf_audit, leaf_validation_audit, feedback, explanation, report)

    load_json = load


# The existing versioned trainer can swap its import without changing calls.
SelectiveLeafProgram = SparseObliqueProgram


def fit_selective_program(role: str, fit: Sequence[Mapping],
                          calibration: Sequence[Mapping],
                          eligibility: Sequence[Mapping],
                          validation: Sequence[Mapping],
                          oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]],
                          *, actor_role_sha256: str, hard_weight: float | None = None,
                          feature_mode: str | None = None, random_seed: int = 0,
                          target_action: str | None = None) -> SparseObliqueProgram:
    if role not in AGENTS:
        raise ValueError("invalid Warehouse role")
    preset = ROLE_PRESETS[role]
    hard_weight = preset["hard_weight"] if hard_weight is None else hard_weight
    feature_mode = preset["feature_mode"] if feature_mode is None else feature_mode
    if (hard_weight != preset["hard_weight"]
            or feature_mode != preset["feature_mode"]
            or target_action not in (None, *ACTIONS)
            or len(actor_role_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in actor_role_sha256)):
        raise ValueError("Warehouse sparse role preset or Actor hash mismatch")
    parts = {name: list(rows) for name, rows in
             (("fit", fit), ("calibration", calibration),
              ("eligibility", eligibility), ("validation", validation))}
    episodes = {name: _episodes(rows, name) for name, rows in parts.items()}
    if any(episodes[left] & episodes[right]
           for left in episodes for right in episodes if left < right):
        raise ValueError("Warehouse sparse episode splits overlap")
    names = FEATURE_SETS[feature_mode]
    x = {name: _matrix(rows, names) for name, rows in parts.items()}
    p = {name: _actor_probabilities(rows, oracle) for name, rows in parts.items()}
    masks = {name: np.stack([np.asarray(row["mask"], dtype=np.float32)
                             for row in rows]) for name, rows in parts.items()}
    projections = _candidate_projections(x["fit"], p["fit"], hard_weight)
    projected = {name: _project(matrix, projections) for name, matrix in x.items()}
    if target_action is None:
        fit_y = p["fit"]
        if hard_weight:
            fit_y = np.concatenate((fit_y, np.eye(len(ACTIONS))[
                fit_y.argmax(axis=1)] * np.sqrt(hard_weight)), axis=1)
    else:
        action_index = ACTIONS.index(target_action)
        fit_y = p["fit"][:, action_index]
        if hard_weight:
            fit_y = (fit_y + hard_weight * (
                p["fit"].argmax(axis=1) == action_index)) / (1 + hard_weight)
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAVES,
                                 min_samples_leaf=8, random_state=random_seed)
    tree.fit(projected["fit"], fit_y)
    leaf_ids = {name: tree.apply(matrix) for name, matrix in projected.items()}
    means = {int(leaf): p["fit"][leaf_ids["fit"] == leaf].mean(axis=0)
             for leaf in set(leaf_ids["fit"])}
    for leaf in set(leaf_ids["calibration"]):
        weight = preset["calibration_fit_weight"]
        means[int(leaf)] = (weight * means[int(leaf)]
                            + (1 - weight) * p["calibration"][
                                leaf_ids["calibration"] == leaf].mean(axis=0))
    leaf_audit, leaf_validation_audit = {}, {}
    feedback_leaves, explanation_leaves = set(), set()
    for leaf in sorted(means):
        eligible = _metrics(p["eligibility"], leaf_ids["eligibility"],
                            means, masks["eligibility"], {leaf})
        eligible["support"] = eligible["rows"]
        leaf_audit[leaf] = eligible
        validated = _metrics(p["validation"], leaf_ids["validation"],
                             means, masks["validation"], {leaf})
        validated["support"] = validated["rows"]
        leaf_validation_audit[leaf] = validated
        if (_eligible_leaf(eligible, FEEDBACK_AGREEMENT)
                and _eligible_leaf(validated, FEEDBACK_AGREEMENT)):
            feedback_leaves.add(leaf)
        if (role == "robot_2"
                and _eligible_leaf(eligible, AI_EXPLANATION_AGREEMENT)
                and _eligible_leaf(validated, AI_EXPLANATION_AGREEMENT)):
            explanation_leaves.add(leaf)
    feedback = _metrics(p["validation"], leaf_ids["validation"],
                        means, masks["validation"], feedback_leaves)
    explanation = _metrics(p["validation"], leaf_ids["validation"],
                           means, masks["validation"], explanation_leaves)
    report = {"role": role, "actor_role_sha256": actor_role_sha256,
              "preset_version": PRESET_VERSION,
              "projection_version": PROJECTION_VERSION,
              "online_gate_version": ONLINE_GATE_VERSION,
              "feature_mode": feature_mode,
              "hard_weight": float(hard_weight),
              "target_action": target_action,
              "calibration_fit_weight": preset["calibration_fit_weight"],
              "max_depth": int(tree.get_depth()),
              "leaf_count": int(tree.get_n_leaves()),
              "max_features_per_node": MAX_FEATURES_PER_NODE,
              "min_leaf_support": MIN_LEAF_SUPPORT,
              "split_episodes": {name: len(ids) for name, ids in episodes.items()},
              "split_episode_ids": {name: sorted(ids) for name, ids in episodes.items()},
              "split_episode_sha256": {
                  name: sha256(json.dumps(sorted(ids)).encode()).hexdigest()
                  for name, ids in episodes.items()},
              "eligibility": {"feedback_leaf_count": len(feedback_leaves),
                              "explanation_leaf_count": len(explanation_leaves)},
              "validation": {
                  "global": _metrics(p["validation"], leaf_ids["validation"], means,
                                     masks["validation"], set(means)),
                  "feedback": feedback,
                  "feedback_online_passed": _passes_online(
                      feedback, FEEDBACK_AGREEMENT),
                  "feedback_passed": _passes(feedback, FEEDBACK_AGREEMENT),
                  "explanation": explanation,
                  "explanation_online_passed": (role == "robot_2" and
                      _passes_online(explanation, AI_EXPLANATION_AGREEMENT)),
                  "explanation_passed": (role == "robot_2" and
                      _passes(explanation, AI_EXPLANATION_AGREEMENT)),
              }}
    used = sorted({int(index) for index in tree.tree_.feature if index >= 0})
    compact_indices = {old: new for new, old in enumerate(used)}
    structure = {
        "left": [int(value) for value in tree.tree_.children_left],
        "right": [int(value) for value in tree.tree_.children_right],
        "projection": [compact_indices[int(value)] if value >= 0 else -1
                       for value in tree.tree_.feature],
        "threshold": [float(value) for value in tree.tree_.threshold],
    }
    return SparseObliqueProgram(
        role, actor_role_sha256, names, structure,
        tuple(projections[index] for index in used),
        {leaf: tuple(map(float, vector)) for leaf, vector in means.items()},
        leaf_audit, leaf_validation_audit, frozenset(feedback_leaves),
        frozenset(explanation_leaves), report)


fit_sparse_program = fit_selective_program


def audit_frozen_program(program: SparseObliqueProgram,
                         samples: Sequence[Mapping],
                         oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]]) -> dict:
    rows = list(samples)
    episode_ids = _episodes(rows, "independent_development_audit")
    prior = program.report["split_episode_ids"]
    if any(episode_ids & set(ids) for ids in prior.values()):
        raise ValueError("independent Warehouse sparse audit reused a fit episode")
    if any(not episode.isdecimal() or str(int(episode)) != episode
           for episode in episode_ids):
        raise ValueError("independent Warehouse sparse audit requires numeric seeds")
    development_seeds = sorted(int(episode) for episode in episode_ids)
    p = _actor_probabilities(rows, oracle)
    masks = np.stack([np.asarray(row["mask"], dtype=np.float32) for row in rows])
    leaf_ids = np.asarray([program._leaf(row)[0] for row in rows], dtype=np.int32)
    means = {leaf: np.asarray(value)
             for leaf, value in program.leaf_probabilities.items()}
    global_metrics = _metrics(p, leaf_ids, means, masks, set(means))
    feedback = _metrics(p, leaf_ids, means, masks, set(program.feedback_leaves))
    explanation = _metrics(p, leaf_ids, means, masks,
                           set(program.explanation_leaves))
    feedback_passed = (_passes(feedback, FEEDBACK_AGREEMENT)
                       and program.report["validation"]["feedback_online_passed"])
    explanation_passed = (program.role == "robot_2"
                          and _passes(explanation, AI_EXPLANATION_AGREEMENT)
                          and program.report["validation"]["explanation_online_passed"])
    validation = {"global": global_metrics, "feedback": feedback,
                  "explanation": explanation,
                  "feedback_passed": bool(feedback_passed),
                  "explanation_passed": bool(explanation_passed)}
    return {
        "audit_version": "warehouse-sparse-oblique-independent-dev-v5.0",
        "audit_source": "final_independent_development_rollouts",
        "role": program.role,
        "actor_role_sha256": program.actor_role_sha256,
        "program_sha256": program.sha256(),
        "episode_count": len(episode_ids),
        "rows": len(rows), "sample_count": len(rows),
        "development_seeds": development_seeds,
        "episode_ids": sorted(episode_ids),
        "episode_sha256": sha256(json.dumps(sorted(episode_ids)).encode()).hexdigest(),
        "program_depth": program.report["max_depth"],
        "program_leaf_count": program.report["leaf_count"],
        "max_features_per_node": MAX_FEATURES_PER_NODE,
        "feedback_leaves": sorted(program.feedback_leaves),
        "explanation_leaves": sorted(program.explanation_leaves),
        "global": global_metrics, "feedback": feedback,
        "explanation": explanation,
        "feedback_passed": bool(feedback_passed),
        "explanation_passed": bool(explanation_passed),
        "feedback_eligible": bool(feedback_passed),
        "explanation_eligible": bool(explanation_passed),
        "validation": validation,
        "passed": bool(feedback_passed and
                       (program.role != "robot_2" or explanation_passed)),
    }


audit_frozen_sparse_program = audit_frozen_program
