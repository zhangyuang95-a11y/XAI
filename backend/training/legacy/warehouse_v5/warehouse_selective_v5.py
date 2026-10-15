"""Experimental selective, role-specific Warehouse v5 program extractor.

The program observes only the frozen Actor's public pre-action observation and
physical action mask. A tree leaf may supply a gradient or an explanation only
after disjoint fit, calibration, eligibility-audit, and development episodes
pass the fixed selective audit. This module does not choose environment actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse.pure_nn import AGENTS, OBSERVATION_DIM, observation_feature_names


MAX_DEPTH = 5
MAX_LEAVES = 16
MIN_LEAF_SUPPORT = 20
MIN_COVERAGE = 0.20
FEEDBACK_AGREEMENT = 0.85
AI_EXPLANATION_AGREEMENT = 0.90
MAX_COVERED_KL = 0.35
_FEATURE_NAMES = observation_feature_names()
_BASE_NAMES = tuple(name for name in _FEATURE_NAMES
                    if name.startswith(("self_", "partner_", "elapsed_", "charger_")))
_TARGET_FIELDS = ("present", "delivery", "order_0", "order_1", "priority",
                  "distance", "partner_distance", "distance_advantage",
                  "row_delta", "column_delta", "partner_at_goal", "best_gain")
FEATURE_NAMES = tuple(sorted(set(_BASE_NAMES)
    | {f"legal_{action.lower()}" for action in ACTIONS}
    | {"charger_distance", "partner_charger_distance", "partner_distance"}
    | {f"charger_gain_{action.lower()}" for action in ACTIONS}
    | {f"charger_route_{action.lower()}" for action in ACTIONS}
    | {f"target_{rank}_{field}" for rank in (0, 1) for field in _TARGET_FIELDS}
    | {f"target_{rank}_gain_{action.lower()}" for rank in (0, 1) for action in ACTIONS}
    | {f"target_{rank}_route_{action.lower()}" for rank in (0, 1) for action in ACTIONS}))
FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "no_previous": tuple(name for name in FEATURE_NAMES if "_previous_action_" not in name),
}
PRESET_VERSION = "warehouse-priority-selective-online-leaf-gate-20261015.v3"
PROGRAM_VERSION = "warehouse-selective-leaf-v5.3"
ONLINE_GATE_VERSION = "disjoint-eligibility-and-validation-leaf-v2"
ROLE_PRESETS = {
    "robot_1": {"feature_mode": "all", "hard_weight": 0.5,
                "calibration_fit_weight": 0.5},
    "robot_2": {"feature_mode": "no_previous", "hard_weight": 0.0,
                "calibration_fit_weight": 0.5},
}
# This was selected on already-used development seeds, not preregistered and
# not evidence of held-out quality. The independent final audit remains unused.
PRESET_PROVENANCE = {
    "selection_source": "development_tuning",
    "raw_probe_pair_seed": 6001,
    "raw_probe_checkpoint_joint_steps": 100800,
    "fit_training_episodes": 50,
    "calibration_training_episodes": 25,
    "eligibility_training_episodes": 25,
    "development_seeds": list(range(240000, 240024)),
    "final_audit_seeds_used_for_selection": False,
    "heldout_seeds_used_for_selection": False,
    "raw_probe_robot_1_feedback": {"coverage": 0.2354, "agreement": 0.8717, "kl": 0.3231},
    "raw_probe_robot_2_feedback": {"coverage": 0.2399, "agreement": 0.9001, "kl": 0.2827},
    "raw_probe_robot_2_explanation": {"coverage": 0.2163, "agreement": 0.9037, "kl": 0.2765},
    "later_raw_selected_checkpoint_joint_steps": 160320,
    "later_raw_selected_preset_passed": False,
    "online_gate_change": "Feedback and AI explanation leaves separately pass eligibility and disjoint validation support, agreement, KL, and legality checks. Internal online gates have no aggregate coverage floor; only the independent frozen audit applies the 20% coverage floor.",
}


@lru_cache(maxsize=None)
def _distance(start: tuple[int, int], goal: tuple[int, int]) -> int:
    return shortest_path_distance(start, goal, STUDY_MAP_LAYOUT.layout_id)


def _point(features: Mapping[str, float], prefix: str) -> tuple[int, int]:
    return (int(round(features[prefix + "_row"] * (STUDY_MAP_LAYOUT.rows - 1))),
            int(round(features[prefix + "_column"] * (STUDY_MAP_LAYOUT.cols - 1))))


def _gains(start: tuple[int, int], target: tuple[int, int], mask: np.ndarray) -> tuple[dict[str, float], str]:
    here = _distance(start, target)
    gains = {}
    for index, action in enumerate(ACTIONS):
        if action == "WAIT":
            gain = 0
        else:
            delta = MOVE_DELTAS[action]
            after = (start[0] + delta[0], start[1] + delta[1])
            gain = here - _distance(after, target) if STUDY_MAP_LAYOUT.is_passable(after) else -2
        gains[action] = float(gain)
    best = max((action for index, action in enumerate(ACTIONS) if mask[index] > 0.5),
               key=lambda action: (gains[action], -ACTIONS.index(action)))
    return gains, best


def observable_features(sample: Mapping) -> dict[str, float]:
    """Named features from only `obs` and the physical `mask` before action."""
    obs = np.asarray(sample["obs"], dtype=np.float32)
    mask = np.asarray(sample["mask"], dtype=np.float32)
    if (obs.shape != (OBSERVATION_DIM,) or mask.shape != (len(ACTIONS),)
            or not np.isfinite(obs).all() or not np.isfinite(mask).all()
            or not np.any(mask > 0.5)):
        raise ValueError("invalid pre-action Warehouse observation or mask")
    source = dict(zip(_FEATURE_NAMES, map(float, obs), strict=True))
    me, partner = _point(source, "self"), _point(source, "partner")
    result = {name: source[name] for name in _BASE_NAMES}
    result.update({f"legal_{action.lower()}": float(mask[index] > 0.5)
                   for index, action in enumerate(ACTIONS)})
    carrying = any(source[f"self_carrying_task_{index}"] > 0.5 for index in (0, 1))
    targets = []
    for index in (0, 1):
        if source[f"self_carrying_task_{index}"] > 0.5:
            goal = _point(source, f"order_{index}_delivery")
            kind = 0
        elif not carrying and source[f"order_{index}_available"] > 0.5:
            goal = _point(source, f"order_{index}_pickup")
            kind = 1
        else:
            continue
        here, there = _distance(me, goal), _distance(partner, goal)
        # A target descriptor, not an action label or a controller decision.
        targets.append((kind, here - there, here, index, goal))
    targets.sort()
    for rank in (0, 1):
        prefix = f"target_{rank}_"
        if rank >= len(targets):
            result.update({prefix + field: 0.0 for field in _TARGET_FIELDS})
            result.update({prefix + f"{kind}_{action.lower()}": 0.0
                           for kind in ("gain", "route") for action in ACTIONS})
            continue
        kind, priority, here, index, goal = targets[rank]
        there = _distance(partner, goal)
        result.update({prefix + "present": 1.0, prefix + "delivery": float(kind == 0),
                       prefix + "order_0": float(index == 0),
                       prefix + "order_1": float(index == 1),
                       prefix + "priority": priority / 12.0,
                       prefix + "distance": here / 12.0,
                       prefix + "partner_distance": there / 12.0,
                       prefix + "distance_advantage": (there - here) / 12.0,
                       prefix + "row_delta": (goal[0] - me[0]) / max(1, STUDY_MAP_LAYOUT.rows - 1),
                       prefix + "column_delta": (goal[1] - me[1]) / max(1, STUDY_MAP_LAYOUT.cols - 1),
                       prefix + "partner_at_goal": float(partner == goal)})
        gains, best = _gains(me, goal, mask)
        for action in ACTIONS:
            result[prefix + f"gain_{action.lower()}"] = gains[action] / 12.0
            result[prefix + f"route_{action.lower()}"] = float(action == best)
        result[prefix + "best_gain"] = gains[best] / 12.0
    charger = STUDY_MAP_LAYOUT.charger_position
    result["charger_distance"] = _distance(me, charger) / 12.0
    result["partner_charger_distance"] = _distance(partner, charger) / 12.0
    result["partner_distance"] = _distance(me, partner) / 12.0
    gains, best = _gains(me, charger, mask)
    for action in ACTIONS:
        result[f"charger_gain_{action.lower()}"] = gains[action] / 12.0
        result[f"charger_route_{action.lower()}"] = float(action == best)
    if set(result) != set(FEATURE_NAMES):
        raise RuntimeError("Warehouse selective feature schema drifted")
    return result


def _matrix(samples: Sequence[Mapping], feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([[features[name] for name in feature_names]
                       for row in samples for features in (observable_features(row),)],
                      dtype=np.float32)


def _actor_probabilities(samples: Sequence[Mapping], oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]]) -> np.ndarray:
    vectors = []
    for row in samples:
        raw = oracle(row)
        vector = np.asarray([raw[action] for action in ACTIONS] if isinstance(raw, Mapping) else raw,
                            dtype=np.float64)
        mask = np.asarray(row["mask"], dtype=np.float64) > 0.5
        if (vector.shape != (len(ACTIONS),) or not np.isfinite(vector).all()
                or np.any(vector < -1e-8) or not np.any(mask)
                or np.any(vector[~mask] > 1e-6) or vector.sum() <= 0):
            raise ValueError("oracle must return a legal masked Actor distribution")
        vector = np.maximum(vector, 0.0) * mask
        vectors.append(vector / vector.sum())
    return np.stack(vectors)


def _episodes(samples: Sequence[Mapping], name: str) -> set[str]:
    if not samples or any("episode" not in row for row in samples):
        raise ValueError(f"{name} requires recorded episode identities")
    return {str(row["episode"]) for row in samples}


def _metrics(probabilities: np.ndarray, leaf_ids: np.ndarray,
             means: Mapping[int, np.ndarray], masks: np.ndarray,
             selected: set[int]) -> dict:
    chosen = np.isin(leaf_ids, tuple(selected))
    count = int(chosen.sum())
    result = {"rows": count, "coverage": float(count / len(leaf_ids)),
              "agreement": 0.0, "kl": None, "illegal_count": 0}
    if count == 0:
        return result
    q = np.stack([means[int(leaf)] for leaf in leaf_ids[chosen]])
    p = probabilities[chosen]
    legal = masks[chosen] > 0.5
    chosen_action = q.argmax(axis=1)
    result["illegal_count"] = int((~legal[np.arange(count), chosen_action]).sum())
    result["agreement"] = float(np.mean(chosen_action == p.argmax(axis=1)))
    q = np.maximum(q * legal, 1e-8)
    q /= q.sum(axis=1, keepdims=True)
    result["kl"] = float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-8)) - np.log(q)), axis=1)))
    return result


def _passes(metrics: Mapping, agreement: float) -> bool:
    return (metrics["coverage"] >= MIN_COVERAGE
            and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


def _eligible_leaf(metrics: Mapping, agreement: float) -> bool:
    return (metrics["support"] >= MIN_LEAF_SUPPORT
            and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


def _passes_online(metrics: Mapping, agreement: float) -> bool:
    """Conditional agreement is required online; final coverage is audited later."""
    return (metrics["rows"] > 0
            and metrics["agreement"] >= agreement
            and metrics["kl"] is not None and metrics["kl"] <= MAX_COVERED_KL
            and metrics["illegal_count"] == 0)


@dataclass(frozen=True)
class SelectiveLeafProgram:
    role: str
    actor_role_sha256: str
    feature_names: tuple[str, ...]
    tree: dict
    leaf_probabilities: dict[int, tuple[float, ...]]
    leaf_audit: dict[int, dict]
    leaf_validation_audit: dict[int, dict]
    feedback_leaves: frozenset[int]
    explanation_leaves: frozenset[int]
    report: dict
    _sha256_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def _leaf(self, sample: Mapping) -> tuple[int, list[dict]]:
        features = observable_features(sample)
        node, path = 0, []
        while self.tree["left"][node] >= 0:
            name = self.feature_names[self.tree["feature"][node]]
            threshold = self.tree["threshold"][node]
            value = features[name]
            left = value <= threshold
            path.append({"feature": name, "operator": "<=" if left else ">",
                         "threshold": threshold, "observed": value})
            node = self.tree["left" if left else "right"][node]
        return node, path

    def lookup(self, sample: Mapping) -> dict:
        """Tree identity and pre-action evidence; never an environment action."""
        leaf, path = self._leaf(sample)
        distribution = self.leaf_probabilities[leaf]
        return {"leaf_id": leaf, "path": path,
                "action": ACTIONS[int(np.argmax(distribution))],
                "probabilities": dict(zip(ACTIONS, distribution, strict=True)),
                "leaf_audit": self.leaf_audit.get(leaf),
                "feedback_leaf": leaf in self.feedback_leaves,
                "explanation_leaf": leaf in self.explanation_leaves,
                "program_sha256": self.sha256()}

    def execute(self, sample: Mapping) -> dict:
        """Execute the bounded tree on a current pre-action observation."""
        return self.lookup(sample)

    def covered(self, sample: Mapping, *, actor_action: str, purpose: str) -> dict | None:
        """Gate an audited leaf on the caller's current Actor action and legal mask.

        The development audit binds the Actor at extraction. During PPO the
        Actor may move, so the caller supplies its current action for every
        feedback row; a stale program stops contributing whenever it disagrees.
        Final QA separately requires the program and audit to bind the frozen
        selected Actor hash.
        """
        if purpose not in ("feedback", "explanation"):
            raise ValueError("unknown selective program purpose")
        gate = purpose + "_online_passed"
        if not self.report["validation"][gate]:
            return None
        evidence = self.lookup(sample)
        leaves = self.feedback_leaves if purpose == "feedback" else self.explanation_leaves
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
                "feature_names": list(self.feature_names), "actions": list(ACTIONS),
                "tree": self.tree,
                "leaf_probabilities": {str(key): list(value) for key, value in self.leaf_probabilities.items()},
                "leaf_audit": {str(key): value for key, value in self.leaf_audit.items()},
                "leaf_validation_audit": {str(key): value for key, value in self.leaf_validation_audit.items()},
                "feedback_leaves": sorted(self.feedback_leaves),
                "explanation_leaves": sorted(self.explanation_leaves),
                "report": self.report}

    def sha256(self) -> str:
        if self._sha256_cache is None:
            digest = sha256(json.dumps(self.to_dict(), sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            object.__setattr__(self, "_sha256_cache", digest)
        return self._sha256_cache

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False),
                          encoding="utf-8")
        digest = sha256(target.read_bytes()).hexdigest()
        if digest != self.sha256():
            raise RuntimeError("Warehouse selective program changed after its audit")
        return digest

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             actor_role_sha256: str) -> "SelectiveLeafProgram":
        raw = Path(path).read_bytes()
        if sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("Warehouse selective program artifact hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256)

    @classmethod
    def from_dict(cls, data: Mapping, *, actor_role_sha256: str) -> "SelectiveLeafProgram":
        """Restore checkpoint state without changing audited leaf identities."""
        preset = ROLE_PRESETS.get(data.get("role"))
        expected_features = FEATURE_SETS[preset["feature_mode"]] if preset else ()
        if (data.get("version") != PROGRAM_VERSION
                or data.get("role") not in AGENTS
                or data.get("actor_role_sha256") != actor_role_sha256
                or data.get("feature_names") != list(expected_features)
                or data.get("actions") != list(ACTIONS)):
            raise RuntimeError("Warehouse selective program binding mismatch")
        tree = data["tree"]
        keys = ("left", "right", "feature", "threshold")
        size = len(tree["left"])
        if size < 1 or size > 2 * MAX_LEAVES - 1 or any(len(tree[key]) != size for key in keys):
            raise RuntimeError("Warehouse selective tree structure invalid")
        leaves, seen = set(), set()
        def visit(node: int, depth: int) -> None:
            if node in seen or node < 0 or node >= size or depth > MAX_DEPTH:
                raise RuntimeError("Warehouse selective tree depth or cycle invalid")
            seen.add(node)
            left, right = tree["left"][node], tree["right"][node]
            if left == right == -1:
                leaves.add(node)
                return
            if (not isinstance(tree["feature"][node], int)
                    or not 0 <= tree["feature"][node] < len(expected_features)
                    or not np.isfinite(tree["threshold"][node])):
                raise RuntimeError("Warehouse selective tree predicate invalid")
            visit(left, depth + 1)
            visit(right, depth + 1)
        visit(0, 0)
        if len(seen) != size or len(leaves) > MAX_LEAVES:
            raise RuntimeError("Warehouse selective tree has unreachable nodes or excess leaves")
        probabilities = {int(key): tuple(map(float, value))
                         for key, value in data["leaf_probabilities"].items()}
        if (set(probabilities) != leaves
                or any(len(values) != len(ACTIONS) or not np.isfinite(values).all()
                       or min(values) < 0 or abs(sum(values) - 1.0) > 1e-5
                       for values in probabilities.values())):
            raise RuntimeError("Warehouse selective leaf distribution invalid")
        feedback = frozenset(map(int, data["feedback_leaves"]))
        explanation = frozenset(map(int, data["explanation_leaves"]))
        if not feedback <= leaves or not explanation <= leaves:
            raise RuntimeError("Warehouse selective eligible leaf identity invalid")
        report = data["report"]
        if (report.get("role") != data["role"]
                or report.get("actor_role_sha256") != actor_role_sha256
                or report.get("preset_version") != PRESET_VERSION
                or report.get("online_gate_version") != ONLINE_GATE_VERSION
                or report.get("preset_provenance") != PRESET_PROVENANCE
                or report.get("feature_mode") != preset["feature_mode"]
                or report.get("hard_weight") != preset["hard_weight"]
                or report.get("calibration_fit_weight") != preset["calibration_fit_weight"]
                or report.get("leaf_count") != len(leaves)
                or report.get("max_depth", MAX_DEPTH + 1) > MAX_DEPTH):
            raise RuntimeError("Warehouse selective audit provenance invalid")
        splits = report.get("split_episode_ids", {})
        names = {"fit", "calibration", "eligibility", "validation"}
        if set(splits) != names or any(not isinstance(splits[name], list) for name in names):
            raise RuntimeError("Warehouse selective audit split provenance missing")
        if any((len(splits[name]) != report["split_episodes"][name]
                or sha256(json.dumps(sorted(splits[name])).encode()).hexdigest()
                   != report["split_episode_sha256"][name]) for name in names):
            raise RuntimeError("Warehouse selective audit split hash mismatch")
        if any(set(splits[left]) & set(splits[right])
               for left in names for right in names if left < right):
            raise RuntimeError("Warehouse selective audit episodes overlap")
        leaf_audit = {int(key): value for key, value in data["leaf_audit"].items()}
        leaf_validation_audit = {int(key): value for key, value in data["leaf_validation_audit"].items()}
        if set(leaf_audit) != leaves or set(leaf_validation_audit) != leaves:
            raise RuntimeError("Warehouse selective audit leaves mismatch")
        expected_feedback = {leaf for leaf, audit in leaf_audit.items()
                             if (_eligible_leaf(audit, FEEDBACK_AGREEMENT)
                                 and _eligible_leaf(leaf_validation_audit[leaf],
                                                    FEEDBACK_AGREEMENT))}
        expected_explanation = {leaf for leaf in expected_feedback
                                if (data["role"] == "robot_2"
                                    and _eligible_leaf(leaf_audit[leaf],
                                                       AI_EXPLANATION_AGREEMENT)
                                    and _eligible_leaf(leaf_validation_audit[leaf],
                                                       AI_EXPLANATION_AGREEMENT))}
        validation = report["validation"]
        if (feedback != expected_feedback or explanation != expected_explanation
                or validation["feedback_online_passed"] !=
                   _passes_online(validation["feedback"], FEEDBACK_AGREEMENT)
                or validation["explanation_online_passed"] !=
                   (data["role"] == "robot_2" and
                    _passes_online(validation["explanation"],
                                   AI_EXPLANATION_AGREEMENT))
                or validation["feedback_passed"] != _passes(validation["feedback"], FEEDBACK_AGREEMENT)
                or validation["explanation_passed"] !=
                   (data["role"] == "robot_2" and _passes(validation["explanation"],
                                                           AI_EXPLANATION_AGREEMENT))):
            raise RuntimeError("Warehouse selective eligibility audit mismatch")
        return cls(data["role"], actor_role_sha256, expected_features, tree, probabilities,
                   leaf_audit, leaf_validation_audit,
                   feedback, explanation, report)

    load_json = load


def fit_selective_program(role: str, fit: Sequence[Mapping], calibration: Sequence[Mapping],
                          eligibility: Sequence[Mapping], validation: Sequence[Mapping],
                          oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]],
                          *, actor_role_sha256: str, hard_weight: float | None = None,
                          feature_mode: str | None = None,
                          random_seed: int = 0) -> SelectiveLeafProgram:
    """Fit, calibrate, audit and validate without sharing an episode across splits.

    `oracle` evaluates the frozen NN on each supplied pre-action sample. The
    caller owns the seed split and must reserve its final validation and
    held-out seeds outside these four inputs.
    """
    if role not in AGENTS:
        raise ValueError("invalid Warehouse role")
    preset = ROLE_PRESETS[role]
    hard_weight = preset["hard_weight"] if hard_weight is None else hard_weight
    feature_mode = preset["feature_mode"] if feature_mode is None else feature_mode
    if (hard_weight != preset["hard_weight"] or feature_mode != preset["feature_mode"]
            or len(actor_role_sha256) != 64
            or any(c not in "0123456789abcdef" for c in actor_role_sha256)):
        raise ValueError("Warehouse selective v5 role preset or Actor hash mismatch")
    parts = {name: list(rows) for name, rows in (("fit", fit), ("calibration", calibration),
                                                 ("eligibility", eligibility), ("validation", validation))}
    episodes = {name: _episodes(rows, name) for name, rows in parts.items()}
    if any(episodes[left] & episodes[right] for left in episodes for right in episodes if left < right):
        raise ValueError("selective program episode splits overlap")
    feature_names = FEATURE_SETS[feature_mode]
    x = {name: _matrix(rows, feature_names) for name, rows in parts.items()}
    p = {name: _actor_probabilities(rows, oracle) for name, rows in parts.items()}
    masks = {name: np.stack([np.asarray(row["mask"], dtype=np.float32) for row in rows])
             for name, rows in parts.items()}
    fit_y = p["fit"]
    if hard_weight:
        one_hot = np.eye(len(ACTIONS))[fit_y.argmax(axis=1)]
        fit_y = np.concatenate((fit_y, one_hot * np.sqrt(hard_weight)), axis=1)
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAVES,
                                 min_samples_leaf=8, random_state=random_seed)
    tree.fit(x["fit"], fit_y)
    leaf_ids = {name: tree.apply(matrix) for name, matrix in x.items()}
    means = {int(leaf): p["fit"][leaf_ids["fit"] == leaf].mean(axis=0)
             for leaf in set(leaf_ids["fit"])}
    for leaf in set(leaf_ids["calibration"]):
        fit_weight = preset["calibration_fit_weight"]
        means[int(leaf)] = (fit_weight * means[int(leaf)]
                            + (1.0 - fit_weight) *
                            p["calibration"][leaf_ids["calibration"] == leaf].mean(axis=0))
    leaf_audit = {}
    leaf_validation_audit = {}
    feedback_leaves, explanation_leaves = set(), set()
    for leaf in sorted(means):
        audited = _metrics(p["eligibility"], leaf_ids["eligibility"], means,
                           masks["eligibility"], {leaf})
        audited["support"] = audited["rows"]
        leaf_audit[leaf] = audited
        validated = _metrics(p["validation"], leaf_ids["validation"], means,
                             masks["validation"], {leaf})
        validated["support"] = validated["rows"]
        leaf_validation_audit[leaf] = validated
        if (_eligible_leaf(audited, FEEDBACK_AGREEMENT)
                and _eligible_leaf(validated, FEEDBACK_AGREEMENT)):
            feedback_leaves.add(leaf)
        if (role == "robot_2"
                and _eligible_leaf(audited, AI_EXPLANATION_AGREEMENT)
                and _eligible_leaf(validated, AI_EXPLANATION_AGREEMENT)):
            explanation_leaves.add(leaf)
    dev_feedback = _metrics(p["validation"], leaf_ids["validation"], means,
                            masks["validation"], feedback_leaves)
    dev_explanation = _metrics(p["validation"], leaf_ids["validation"], means,
                               masks["validation"], explanation_leaves)
    all_leaves = set(means)
    report = {"role": role, "actor_role_sha256": actor_role_sha256,
              "preset_version": PRESET_VERSION,
              "online_gate_version": ONLINE_GATE_VERSION,
              "preset_provenance": deepcopy(PRESET_PROVENANCE),
              "feature_mode": feature_mode,
              "calibration_fit_weight": preset["calibration_fit_weight"],
              "hard_weight": float(hard_weight), "max_depth": int(tree.get_depth()),
              "leaf_count": int(tree.get_n_leaves()), "min_leaf_support": MIN_LEAF_SUPPORT,
              "split_episodes": {name: len(ids) for name, ids in episodes.items()},
              "split_episode_ids": {name: sorted(ids) for name, ids in episodes.items()},
              "split_episode_sha256": {name: sha256(json.dumps(sorted(ids)).encode()).hexdigest()
                                       for name, ids in episodes.items()},
              "eligibility": {"feedback_leaf_count": len(feedback_leaves),
                              "explanation_leaf_count": len(explanation_leaves)},
              "validation": {"global": _metrics(p["validation"], leaf_ids["validation"],
                                                 means, masks["validation"], all_leaves),
                             "feedback": dev_feedback,
                             "feedback_online_passed": _passes_online(dev_feedback,
                                                                       FEEDBACK_AGREEMENT),
                             "feedback_passed": _passes(dev_feedback, FEEDBACK_AGREEMENT),
                             "explanation": dev_explanation,
                             "explanation_online_passed": (role == "robot_2" and
                                _passes_online(dev_explanation, AI_EXPLANATION_AGREEMENT)),
                             "explanation_passed": (role == "robot_2" and
                                                    _passes(dev_explanation, AI_EXPLANATION_AGREEMENT))}}
    structure = {"left": [int(v) for v in tree.tree_.children_left],
                 "right": [int(v) for v in tree.tree_.children_right],
                 "feature": [int(v) for v in tree.tree_.feature],
                 "threshold": [float(v) for v in tree.tree_.threshold]}
    return SelectiveLeafProgram(role, actor_role_sha256, feature_names, structure,
                                {leaf: tuple(map(float, vector)) for leaf, vector in means.items()},
                                leaf_audit, leaf_validation_audit, frozenset(feedback_leaves),
                                frozenset(explanation_leaves), report)


def audit_frozen_program(program: SelectiveLeafProgram, samples: Sequence[Mapping],
                         oracle: Callable[[Mapping], Sequence[float] | Mapping[str, float]]) -> dict:
    """Independent, read-only audit of fixed leaf IDs against a frozen Actor.

    This cannot refit or change the program, and rejects any episode already
    used to fit, calibrate, select leaves, or validate the program.
    """
    rows = list(samples)
    episode_ids = _episodes(rows, "independent_development_audit")
    prior = program.report["split_episode_ids"]
    if any(episode_ids & set(ids) for ids in prior.values()):
        raise ValueError("independent Warehouse program audit reused an earlier episode")
    if any(not episode.isdecimal() or str(int(episode)) != episode
           for episode in episode_ids):
        raise ValueError("independent Warehouse development audit requires numeric scenario seeds")
    development_seeds = sorted(int(episode) for episode in episode_ids)
    probabilities = _actor_probabilities(rows, oracle)
    masks = np.stack([np.asarray(row["mask"], dtype=np.float32) for row in rows])
    leaf_ids = np.asarray([program._leaf(row)[0] for row in rows], dtype=np.int32)
    means = {leaf: np.asarray(value) for leaf, value in program.leaf_probabilities.items()}
    global_metrics = _metrics(probabilities, leaf_ids, means, masks, set(means))
    feedback = _metrics(probabilities, leaf_ids, means, masks, set(program.feedback_leaves))
    explanation = _metrics(probabilities, leaf_ids, means, masks,
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
    return {"audit_version": "warehouse-selective-independent-dev-v5.3",
            "audit_source": "final_independent_development_rollouts",
            "role": program.role, "actor_role_sha256": program.actor_role_sha256,
            "program_sha256": program.sha256(),
            "episode_count": len(episode_ids), "rows": len(rows),
            "sample_count": len(rows), "development_seeds": development_seeds,
            "episode_ids": sorted(episode_ids),
            "episode_sha256": sha256(json.dumps(sorted(episode_ids)).encode()).hexdigest(),
            "program_depth": program.report["max_depth"],
            "program_leaf_count": program.report["leaf_count"],
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
                           (program.role != "robot_2" or explanation_passed))}
