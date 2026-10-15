"""Role-specific v6 Warehouse program certificates and disjoint audits.

The program describes a frozen Actor's masked distribution on validated
leaves. It can supply a training target or explanation evidence, but has no
interface that can advance the environment or choose an executed action.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import (
    FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256, OBSERVATION_DIM,
    encode_policy_features,
)
from .warehouse_v6_actor import (
    FrozenWarehouseV6Actor, FrozenWarehouseV6Role, MAX_SPLIT_FEATURES,
    MIN_RESIDUAL_BOUND, TREE_DEPTH, TREE_LEAVES, TREE_NODES,
)


PROGRAM_VERSION = "warehouse-role-program.v6.0"
MIN_LEAF_SUPPORT = 20
MIN_COVERAGE = 0.20
FEEDBACK_AGREEMENT = 0.85
EXPLANATION_AGREEMENT = 0.90
MAX_COVERED_KL = 0.35
_EPSILON = 1e-8


def _sample_arrays(rows: Sequence[Mapping], name: str):
    if not rows:
        raise ValueError(f"{name} requires samples")
    if any("episode" not in row for row in rows):
        raise ValueError(f"{name} requires pre-action episode IDs")
    obs = np.stack([np.asarray(row["obs"], dtype=np.float32) for row in rows])
    masks = np.stack([np.asarray(row["mask"], dtype=np.float32) for row in rows])
    if (obs.shape != (len(rows), OBSERVATION_DIM)
            or masks.shape != (len(rows), len(ACTIONS))
            or not np.isfinite(obs).all() or not np.isfinite(masks).all()
            or np.any(masks.sum(axis=-1) < 1.0)):
        raise ValueError(f"{name} has invalid Warehouse pre-action rows")
    return encode_policy_features(obs, masks), masks


def _masked_probabilities(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    chosen = np.where(masks > 0.5, logits, -1e9)
    shifted = chosen - chosen.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _fit_leaf_logits(actor: FrozenWarehouseV6Role, features: np.ndarray,
                     masks: np.ndarray, leaf_ids: np.ndarray) -> np.ndarray:
    """Fit program logits to the *full* Actor masked distribution by KL."""
    target = _masked_probabilities(actor.logits_encoded(features), masks)
    learned = np.asarray(actor.weights["tree.leaf_logits"], dtype=np.float32).copy()
    for leaf in range(TREE_LEAVES):
        selected = leaf_ids == leaf
        if not np.any(selected):
            continue
        prior = torch.from_numpy(learned[leaf].copy())
        candidate = torch.nn.Parameter(prior.clone())
        legal = torch.from_numpy(masks[selected] > 0.5)
        p = torch.from_numpy(target[selected].astype(np.float32))
        optimizer = torch.optim.LBFGS([candidate], lr=0.5, max_iter=50,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            predicted = torch.log_softmax(candidate[None].masked_fill(~legal, -1e9), -1)
            loss = -(p * predicted).sum(-1).mean() + 0.01 * (candidate - prior).square().mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        learned[leaf] = candidate.detach().numpy()
    return learned


def _calibrate_leaf_logits(logits: np.ndarray, actor: FrozenWarehouseV6Role,
                           features: np.ndarray, masks: np.ndarray,
                           leaf_ids: np.ndarray) -> np.ndarray:
    """Use disjoint episodes to choose a bounded temperature per visited leaf."""
    target = _masked_probabilities(actor.logits_encoded(features), masks)
    calibrated = logits.copy()
    for leaf in range(TREE_LEAVES):
        selected = leaf_ids == leaf
        if not np.any(selected):
            continue
        best, best_kl = logits[leaf], float("inf")
        for temperature in (0.8, 0.9, 1.0, 1.1, 1.2):
            trial = logits[leaf] / temperature
            q = _masked_probabilities(np.broadcast_to(trial, (int(selected.sum()), len(ACTIONS))),
                                      masks[selected])
            p = target[selected]
            kl = float(np.mean(np.sum(p * (np.log(np.maximum(p, _EPSILON))
                                              - np.log(np.maximum(q, _EPSILON))), axis=-1)))
            if kl < best_kl:
                best, best_kl = trial, kl
        calibrated[leaf] = best
    return calibrated


def _metrics(program: "WarehouseV6Program", actor: FrozenWarehouseV6Role,
             rows: Sequence[Mapping], selected_leaves: set[int]) -> dict:
    features, masks = _sample_arrays(rows, "program audit")
    leaves = actor.leaf(features)
    selected = np.isin(leaves, tuple(selected_leaves))
    result = {"rows": int(selected.sum()), "sample_count": len(rows),
              "coverage": float(selected.mean()), "agreement": 0.0,
              "kl": None, "illegal_count": 0, "masked_illegal_count": 0,
              "certified_rows": 0}
    if not np.any(selected):
        return result
    x, m, leaf = features[selected], masks[selected], leaves[selected]
    p = _masked_probabilities(actor.logits_encoded(x), m)
    q = _masked_probabilities(program.leaf_logits[leaf], m)
    proposed = q.argmax(-1)
    raw_proposed = program.leaf_logits[leaf].argmax(-1)
    result["agreement"] = float(np.mean(proposed == p.argmax(-1)))
    result["kl"] = float(np.mean(np.sum(p * (
        np.log(np.maximum(p, _EPSILON)) - np.log(np.maximum(q, _EPSILON))), axis=-1)))
    # Raw leaf action legality is the hard program gate. Merely masking q
    # would make this count zero by construction and hide an illegal proposal.
    result["illegal_count"] = int(np.sum(m[np.arange(len(m)), raw_proposed] <= 0.5))
    result["masked_illegal_count"] = int(np.sum(m[np.arange(len(m)), proposed] <= 0.5))
    result["certified_rows"] = int(sum(program.margin_certified(int(item), mask)
                                        for item, mask in zip(leaf, m, strict=True)))
    return result


def _eligible(metric: Mapping, required_agreement: float) -> bool:
    return bool(metric["rows"] >= MIN_LEAF_SUPPORT
                and metric["agreement"] >= required_agreement
                and metric["kl"] is not None and metric["kl"] <= MAX_COVERED_KL
                and metric["illegal_count"] == 0)


def _passes(metric: Mapping, required_agreement: float) -> bool:
    return bool(metric["coverage"] >= MIN_COVERAGE
                and metric["agreement"] >= required_agreement
                and metric["kl"] is not None and metric["kl"] <= MAX_COVERED_KL
                and metric["illegal_count"] == 0)


@dataclass(frozen=True)
class WarehouseV6Program:
    role: str
    actor_role_sha256: str
    split_weights: np.ndarray
    split_bias: np.ndarray
    actor_tree_logits: np.ndarray
    actor_residual_bounds: np.ndarray
    leaf_logits: np.ndarray
    feedback_leaves: frozenset[int]
    explanation_leaves: frozenset[int]
    report: dict

    def _leaf(self, sample: Mapping) -> tuple[int, list[dict]]:
        encoded = encode_policy_features(np.asarray(sample["obs"], dtype=np.float32),
                                         np.asarray(sample["mask"], dtype=np.float32))
        node, path = 0, []
        for _ in range(TREE_DEPTH):
            weights = self.split_weights[node]
            value = float(encoded @ weights + self.split_bias[node])
            feature_indices = np.flatnonzero(weights != 0)
            path.append({"features": [
                {"name": FEATURE_NAMES[int(i)], "coefficient": float(weights[i]),
                 "observed": float(encoded[i])} for i in feature_indices],
                "bias": float(self.split_bias[node]), "value": value,
                "branch": "left" if value <= 0 else "right"})
            node = 2 * node + 1 + int(value > 0)
        return node - TREE_NODES, path

    def margin_certified(self, leaf: int, mask: np.ndarray) -> bool:
        """Prove the full Actor and program argmax agree on this legal mask.

        The total deviation includes leaf calibration error and the Actor's
        learned, bounded residual. Audit remains mandatory even when true.
        """
        allowed = np.flatnonzero(np.asarray(mask) > 0.5)
        if not len(allowed):
            return False
        if len(allowed) == 1:
            return True
        logits = self.leaf_logits[leaf, allowed]
        sorted_logits = np.sort(logits)
        difference = float(np.max(np.abs(
            self.leaf_logits[leaf] - self.actor_tree_logits[leaf])))
        bound = float(self.actor_residual_bounds[leaf]) + difference
        return bool(sorted_logits[-1] - sorted_logits[-2] > 2.0 * bound + 1e-6)

    def lookup(self, sample: Mapping) -> dict:
        leaf, path = self._leaf(sample)
        mask = np.asarray(sample["mask"], dtype=np.float32)
        probabilities = _masked_probabilities(self.leaf_logits[leaf][None], mask[None])[0]
        action = ACTIONS[int(probabilities.argmax())]
        raw_action = ACTIONS[int(self.leaf_logits[leaf].argmax())]
        return {"leaf_id": leaf, "path": path, "action": action,
                "raw_action": raw_action,
                "raw_action_legal": bool(mask[ACTIONS.index(raw_action)] > 0.5),
                "probabilities": dict(zip(ACTIONS, map(float, probabilities), strict=True)),
                "margin_certified": self.margin_certified(leaf, mask),
                "program_sha256": self.sha256()}

    def covered(self, sample: Mapping, *, actor_action: str, purpose: str) -> dict | None:
        if purpose not in ("feedback", "explanation"):
            raise ValueError("unknown Warehouse v6 program purpose")
        leaves = self.feedback_leaves if purpose == "feedback" else self.explanation_leaves
        if not self.report["validation"][purpose + "_online_passed"]:
            return None
        evidence = self.lookup(sample)
        mask = np.asarray(sample["mask"], dtype=np.float32)
        if (evidence["leaf_id"] not in leaves or not evidence["raw_action_legal"]
                or evidence["action"] != actor_action
                or mask[ACTIONS.index(actor_action)] <= 0.5):
            return None
        return {**evidence, "eligible": True, "purpose": purpose,
                "actor_role_sha256": self.actor_role_sha256}

    def to_dict(self) -> dict:
        return {"version": PROGRAM_VERSION, "role": self.role,
                "actor_role_sha256": self.actor_role_sha256,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "split_weights": self.split_weights.tolist(),
                "split_bias": self.split_bias.tolist(),
                "actor_tree_logits": self.actor_tree_logits.tolist(),
                "actor_residual_bounds": self.actor_residual_bounds.tolist(),
                "leaf_logits": self.leaf_logits.tolist(),
                "feedback_leaves": sorted(self.feedback_leaves),
                "explanation_leaves": sorted(self.explanation_leaves),
                "report": self.report}

    def sha256(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False))
        return sha256(target.read_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping, *, actor_role_sha256: str) -> "WarehouseV6Program":
        if (data.get("version") != PROGRAM_VERSION
                or data.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or data.get("role") not in AGENTS
                or data.get("actor_role_sha256") != actor_role_sha256):
            raise RuntimeError("Warehouse v6 program version, schema, role, or Actor hash mismatch")
        result = cls(
            str(data["role"]), actor_role_sha256,
            np.asarray(data["split_weights"], dtype=np.float32),
            np.asarray(data["split_bias"], dtype=np.float32),
            np.asarray(data["actor_tree_logits"], dtype=np.float32),
            np.asarray(data["actor_residual_bounds"], dtype=np.float32),
            np.asarray(data["leaf_logits"], dtype=np.float32),
            frozenset(map(int, data["feedback_leaves"])),
            frozenset(map(int, data["explanation_leaves"])), dict(data["report"]),
        )
        if (result.split_weights.shape != (TREE_NODES, FEATURE_DIM)
                or result.split_bias.shape != (TREE_NODES,)
                or result.actor_tree_logits.shape != (TREE_LEAVES, len(ACTIONS))
                or result.actor_residual_bounds.shape != (TREE_LEAVES,)
                or result.leaf_logits.shape != (TREE_LEAVES, len(ACTIONS))
                or np.max(np.count_nonzero(result.split_weights, axis=-1)) > MAX_SPLIT_FEATURES
                or any(not np.isfinite(value).all() for value in (
                    result.split_weights, result.split_bias, result.actor_tree_logits,
                    result.actor_residual_bounds, result.leaf_logits))
                or np.any(result.actor_residual_bounds < MIN_RESIDUAL_BOUND)
                or any(leaf < 0 or leaf >= TREE_LEAVES
                       for leaf in result.feedback_leaves | result.explanation_leaves)
                or result.report.get("max_depth") != TREE_DEPTH
                or result.report.get("leaf_count") != TREE_LEAVES
                or result.report.get("max_features_per_node") != MAX_SPLIT_FEATURES):
            raise RuntimeError("Warehouse v6 program complexity or tensor contract mismatch")
        return result

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str,
             actor_role_sha256: str) -> "WarehouseV6Program":
        raw = Path(path).read_bytes()
        if not expected_sha256 or sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("Warehouse v6 program artifact hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256)


def fit_program(role: str, actor: FrozenWarehouseV6Actor,
                fit: Sequence[Mapping], calibration: Sequence[Mapping],
                eligibility: Sequence[Mapping], validation: Sequence[Mapping]) -> WarehouseV6Program:
    """Fit and validate one role program on four disjoint training episodes sets."""
    if role not in AGENTS:
        raise ValueError("invalid Warehouse v6 program role")
    parts = {"fit": list(fit), "calibration": list(calibration),
             "eligibility": list(eligibility), "validation": list(validation)}
    episode_ids = {name: {str(row["episode"]) for row in rows}
                   for name, rows in parts.items()}
    for name, rows in parts.items():
        _sample_arrays(rows, name)
    if any(episode_ids[left] & episode_ids[right]
           for left in parts for right in parts if left < right):
        raise ValueError("Warehouse v6 program episode splits overlap")
    role_actor = actor.roles[role]
    weight = role_actor.weights["tree.split.weight"]
    selected = np.argsort(-np.abs(weight), axis=-1, kind="stable")[:, :MAX_SPLIT_FEATURES]
    sparse = np.zeros_like(weight)
    np.put_along_axis(sparse, selected, 1.0, axis=-1)
    sparse *= weight
    fit_x, fit_m = _sample_arrays(parts["fit"], "fit")
    fitted = _fit_leaf_logits(role_actor, fit_x, fit_m, role_actor.leaf(fit_x))
    cal_x, cal_m = _sample_arrays(parts["calibration"], "calibration")
    calibrated = _calibrate_leaf_logits(fitted, role_actor, cal_x, cal_m,
                                        role_actor.leaf(cal_x))
    bounds = MIN_RESIDUAL_BOUND + (1.0 - MIN_RESIDUAL_BOUND) / (
        1.0 + np.exp(-role_actor.weights["leaf_bound_logits"]))
    split_hashes = {name: sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
                    for name, ids in episode_ids.items()}
    report = {"version": PROGRAM_VERSION, "role": role,
              "actor_role_sha256": role_actor.role_sha256,
              "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
              "max_depth": TREE_DEPTH, "leaf_count": TREE_LEAVES,
              "max_features_per_node": MAX_SPLIT_FEATURES,
              "split_episodes": {name: len(ids) for name, ids in episode_ids.items()},
              "split_episode_ids": {name: sorted(ids) for name, ids in episode_ids.items()},
              "split_episode_sha256": split_hashes,
              "leaf_eligibility": {}, "leaf_validation": {}, "validation": {}}
    provisional = WarehouseV6Program(
        role, role_actor.role_sha256, sparse, role_actor.weights["tree.split.bias"].copy(),
        role_actor.weights["tree.leaf_logits"].copy(), bounds, calibrated,
        frozenset(), frozenset(), report)
    feedback, explanation = set(), set()
    for leaf in range(TREE_LEAVES):
        allowed = {leaf}
        e = _metrics(provisional, role_actor, parts["eligibility"], allowed)
        v = _metrics(provisional, role_actor, parts["validation"], allowed)
        report["leaf_eligibility"][str(leaf)] = e
        report["leaf_validation"][str(leaf)] = v
        if _eligible(e, FEEDBACK_AGREEMENT) and _eligible(v, FEEDBACK_AGREEMENT):
            feedback.add(leaf)
        if _eligible(e, EXPLANATION_AGREEMENT) and _eligible(v, EXPLANATION_AGREEMENT):
            explanation.add(leaf)
    program = WarehouseV6Program(
        role, role_actor.role_sha256, sparse, role_actor.weights["tree.split.bias"].copy(),
        role_actor.weights["tree.leaf_logits"].copy(), bounds, calibrated,
        frozenset(feedback), frozenset(explanation), report)
    feedback_metric = _metrics(program, role_actor, parts["validation"], feedback)
    explanation_metric = _metrics(program, role_actor, parts["validation"], explanation)
    report["validation"] = {
        "feedback": feedback_metric, "explanation": explanation_metric,
        "feedback_online_passed": bool(feedback_metric["rows"] > 0
                                       and feedback_metric["agreement"] >= FEEDBACK_AGREEMENT
                                       and feedback_metric["kl"] is not None
                                       and feedback_metric["kl"] <= MAX_COVERED_KL
                                       and feedback_metric["illegal_count"] == 0),
        "explanation_online_passed": bool(explanation_metric["rows"] > 0
                                          and explanation_metric["agreement"] >= EXPLANATION_AGREEMENT
                                          and explanation_metric["kl"] is not None
                                          and explanation_metric["kl"] <= MAX_COVERED_KL
                                          and explanation_metric["illegal_count"] == 0),
        "feedback_passed": _passes(feedback_metric, FEEDBACK_AGREEMENT),
        "explanation_passed": _passes(explanation_metric, EXPLANATION_AGREEMENT),
    }
    return program


def audit_frozen_program(program: WarehouseV6Program,
                         actor: FrozenWarehouseV6Actor,
                         rows: Sequence[Mapping]) -> dict:
    if actor.role_sha256[program.role] != program.actor_role_sha256:
        raise RuntimeError("Warehouse v6 program Actor binding mismatch")
    ids = {str(row["episode"]) for row in rows}
    prior = program.report["split_episode_ids"]
    if any(ids & set(used) for used in prior.values()):
        raise ValueError("Warehouse v6 independent audit reused fitting episodes")
    role_actor = actor.roles[program.role]
    feedback = _metrics(program, role_actor, rows, set(program.feedback_leaves))
    explanation = _metrics(program, role_actor, rows, set(program.explanation_leaves))
    feedback_passed = bool(program.report["validation"]["feedback_online_passed"]
                           and _passes(feedback, FEEDBACK_AGREEMENT))
    explanation_passed = bool(program.report["validation"]["explanation_online_passed"]
                              and _passes(explanation, EXPLANATION_AGREEMENT))
    return {"audit_version": "warehouse-role-program-independent.v6.0",
            "role": program.role, "actor_role_sha256": program.actor_role_sha256,
            "program_sha256": program.sha256(),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "episode_ids": sorted(ids), "episode_count": len(ids),
            "sample_count": len(rows), "max_depth": TREE_DEPTH,
            "leaf_count": TREE_LEAVES, "max_features_per_node": MAX_SPLIT_FEATURES,
            "feedback": feedback, "explanation": explanation,
            "feedback_eligible": feedback_passed,
            "explanation_eligible": explanation_passed,
            "passed": bool(feedback_passed and
                           (program.role != "robot_2" or explanation_passed))}
