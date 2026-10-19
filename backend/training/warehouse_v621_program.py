"""Explicit shared-topology programs for the optional v6.21 Actor route.

No environment, trajectory collection or training authorization is provided.
Programs approximate the complete physically masked NN. They never select an
executed action. Actor core membership and empirical evidence qualification
are distinct immutable records. This method is not passed to a legacy method
dispatcher, so unknown-method fallback cannot omit whole-tree checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from .warehouse_v619_leaf_fit import (
    CALIBRATION_TEMPERATURES, LEAF_EPSILON, LEAF_MAXITER, LEAF_REGULARIZATION,
    LEAF_TOLERANCE, calibrate_constrained_leaf_distribution,
    fit_constrained_leaf_distribution, float32_preference_proof,
)
from .warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256
from .warehouse_v6_program import (
    ACTION_PHYSICS_CONFIG_SHA256, _eligible, _sample_arrays, _sampled_indices,
    _wilson_lower, program_structure_sha256,
)
from .warehouse_v6_tree_routing import (
    HARD_SPLIT_ROUTING_VERSION, certified_feature_bounds, certified_leaf_actions,
    leaf_indices, validate_certification_domain,
)
from .warehouse_v621_compile import _validated_structure


VERSION = "warehouse-role-program.v6.21-shared-core-box"
METHOD = "shared_actor_topology_core_box_distribution_v1"
BOX_EPSILON = .05
_MASKS = tuple(FEATURE_NAMES.index(f"legal_{a.lower()}") for a in ACTIONS)
_SPLITS = ("fit", "calibration", "eligibility", "validation")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _probs(logits, masks):
    z, m = np.asarray(logits, np.float64), np.asarray(masks, np.float64)
    if (m.ndim < 1 or m.shape[-1] != 5 or z.ndim < 1 or z.shape[-1] != 5
            or not np.isfinite(z).all() or not np.isfinite(m).all()
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)):
        raise ValueError("invalid complete NN logits or physical masks")
    try:
        z = np.broadcast_to(z, m.shape)
    except ValueError as error:
        raise ValueError("complete NN logits and physical mask shapes differ") from error
    legal = m > .5
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            maximum = np.where(legal, z, -np.inf).max(-1, keepdims=True)
            shifted = np.zeros_like(z)
            np.subtract(z, maximum, out=shifted, where=legal)
            shifted[~legal] = -np.inf
            exp = np.exp(shifted)
            result = exp/exp.sum(-1, keepdims=True)
    except FloatingPointError as error:
        raise RuntimeError("complete NN legal-logit normalization overflowed") from error
    if not np.isfinite(result).all() or np.any(result[~legal] != 0):
        raise RuntimeError("complete NN probabilities are not finite or physically masked")
    return result


def _distribution_inputs(target, masks):
    p, m = np.asarray(target, np.float64), np.asarray(masks, np.float64)
    if (p.ndim != 2 or p.shape[1] != 5 or m.shape != p.shape
            or any(not np.isfinite(a).all() for a in (p, m))
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or np.any(p < 0) or np.any(p[m == 0] != 0)
            or (len(p) and not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0))):
        raise ValueError("shared program requires the complete physically masked five-action NN distribution")
    return p, m


def _kl(p, q):
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    return np.sum(p*(np.log(np.maximum(p, 1e-300))-np.log(np.maximum(q, 1e-300))), axis=-1)


def _box_proof(logits, anchor, action, legal_actions):
    z, t = np.asarray(logits), np.asarray(anchor, np.float32)
    preference = float32_preference_proof(z, action, legal_actions)
    error = float(np.max(np.abs(z.astype(np.float64)-t.astype(np.float64))))
    center = float(abs(z.astype(np.float64).sum()))
    return {"passed": bool(preference["passed"] and error <= BOX_EPSILON and center <= 2e-6),
            "maximum_anchor_offset": error, "absolute_center_sum": center,
            "box_epsilon": BOX_EPSILON, "preference": preference}


def fit_core_leaf_distribution(target, masks, legal_actions, anchor):
    """One convex full-distribution fit, centered in the Actor anchor's box."""
    p, m = _distribution_inputs(target, masks)
    t = np.asarray(anchor, np.float32)
    action = int(t.argmax())
    if (t.shape != (5,) or not np.isfinite(t).all() or abs(float(t.astype(float).sum())) > 1e-4
            or not float32_preference_proof(t, action, legal_actions)["passed"]
            or any(np.any(m[:, a] == 0) for a in legal_actions)):
        raise ValueError("invalid core anchor, centering or path certificate")
    n = len(p)
    if not n:
        return t.copy(), {"success": True, "row_count": 0, "chosen_action": action,
            "solver": "SLSQP", "solver_status": 0, "enumerated_action_solves": 0,
            "empty_leaf": True, "float32_feasible": True,
            "proof": _box_proof(t, t, action, legal_actions),
            "epsilon": BOX_EPSILON, "maxiter": LEAF_MAXITER, "tolerance": LEAF_TOLERANCE,
            "regularization": LEAF_REGULARIZATION}
    patterns, inv = np.unique(m, axis=0, return_inverse=True)
    fractions = np.bincount(inv, weights=p.sum(-1), minlength=len(patterns))/n
    mean = p.mean(0)
    prior = np.log((p.sum(0)+.2)/(m.sum(0)+1.))
    prior -= prior.mean()
    def objective(z):
        value, gradient = -float(mean@z), -mean.copy()
        for pattern, fraction in zip(patterns, fractions, strict=True):
            active = pattern > .5
            normalizer = logsumexp(z[active])
            value += fraction*normalizer
            gradient[active] += fraction*np.exp(z[active]-normalizer)
        delta = z-prior
        return float(value+LEAF_REGULARIZATION*(delta@delta)), gradient+2*LEAF_REGULARIZATION*delta
    # Reserve enough room that serialization to float32 does not leave the box.
    safety = 16*np.finfo(np.float32).eps*max(1., float(np.abs(t).max()))
    width = BOX_EPSILON-safety
    if width <= 0:
        raise RuntimeError("float32 core anchor is too large for a certifiable program box")
    center = t.astype(np.float64)-t.astype(np.float64).mean()
    result = minimize(objective, center, method="SLSQP", jac=True,
        bounds=[(float(a)-width, float(a)+width) for a in t],
        constraints=({"type": "eq", "fun": lambda z: float(z.sum()),
                      "jac": lambda z: np.ones(5)},),
        options={"maxiter": LEAF_MAXITER, "ftol": LEAF_TOLERANCE, "disp": False})
    z = np.asarray(result.x-result.x.mean(), np.float32)
    proof = _box_proof(z, t, action, legal_actions)
    if not result.success or not proof["passed"]:
        raise RuntimeError("core box fit did not converge to a float32 certified solution")
    return z, {"success": True, "row_count": n, "chosen_action": action,
        "solver": "SLSQP", "solver_status": int(result.status),
        "iterations": int(result.nit), "enumerated_action_solves": 1,
        "empty_leaf": False, "float32_feasible": True, "proof": proof,
        "masked_kl": float(_kl(p, _probs(z, m)).mean()),
        "epsilon": BOX_EPSILON, "maxiter": LEAF_MAXITER, "tolerance": LEAF_TOLERANCE,
        "regularization": LEAF_REGULARIZATION, "target": "complete_masked_NN_distribution"}


def calibrate_core_leaf_distribution(logits, target, masks, legal_actions, anchor):
    p, m = _distribution_inputs(target, masks)
    z, t = np.asarray(logits, np.float32), np.asarray(anchor, np.float32)
    action = int(t.argmax())
    if not _box_proof(z, t, action, legal_actions)["passed"]:
        raise ValueError("core calibration input is outside its certified box")
    candidates, records = [], []
    for temperature in CALIBRATION_TEMPERATURES:
        # Scaling the entire anchor would silently violate the shared box.
        trial = np.asarray(t.astype(float)+(z.astype(float)-t.astype(float))/temperature, np.float32)
        proof = _box_proof(trial, t, action, legal_actions)
        value = float(_kl(p, _probs(trial, m)).mean()) if len(p) else 0.
        accepted = bool(proof["passed"] and np.isfinite(value))
        records.append({"temperature": temperature, "masked_kl": value, "accepted": accepted,
                        "proof": proof})
        if accepted:
            candidates.append((value, CALIBRATION_TEMPERATURES.index(temperature), trial, temperature))
    if not candidates:
        raise RuntimeError("no box- and path-preserving core calibration")
    if not len(p):
        calibrated, temperature, value = z.copy(), 1., 0.
    else:
        value, _, calibrated, temperature = min(candidates, key=lambda item: item[:2])
    return calibrated, {"temperature": temperature, "chosen_action": action,
                        "masked_kl": value, "empty_calibration_leaf": not len(p),
                        "candidate_reports": records,
                        "proof": _box_proof(calibrated, t, action, legal_actions)}


def _arrays(rows, name):
    if (not rows or any(not str(r.get("episode", "")) for r in rows)
            or any(type(r.get("base_seed")) is not int or r["base_seed"] < 0 for r in rows)
            or any("frame" not in r or type(r["frame"]) is not int or r["frame"] < 0 for r in rows)):
        raise ValueError("shared program requires complete pre-action physical frame metadata")
    frames, bases = {}, {}
    for row in rows:
        frames.setdefault(str(row["episode"]), []).append(row["frame"])
        bases.setdefault(str(row["episode"]), set()).add(row["base_seed"])
    if any(values != list(range(len(values))) for values in frames.values()) or any(len(v) != 1 for v in bases.values()):
        raise ValueError("shared program contains omitted, duplicated or reordered physical frames")
    x, masks = _sample_arrays(rows, name, FEATURE_DIM)
    validate_certification_domain(x, _MASKS, certified_feature_bounds(tuple(FEATURE_NAMES)))
    if not np.array_equal(x[:, _MASKS], masks) or np.any(masks[:, 4] != 1):
        raise ValueError("shared program physical masks differ from public features")
    sampled = _sampled_indices(rows)
    if np.any(masks[np.arange(len(rows)), sampled] == 0):
        raise ValueError("shared program sampled action is physically illegal")
    return x, masks, sampled


@dataclass(frozen=True)
class WarehouseV621Program:
    role: str
    actor_role_sha256: str
    structure: dict
    leaf_logits: np.ndarray
    core_mask: np.ndarray
    anchors: np.ndarray
    residual_bound: float
    feedback_leaves: frozenset[int]
    explanation_leaves: frozenset[int]
    report: dict

    def __post_init__(self):
        # Tensor inputs and external dictionaries cannot mutate a saved program
        # through an alias held by the caller after fitting/loading.
        copied = {}
        for key, value in self.structure.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.flags.writeable = False
            copied[key] = value
        object.__setattr__(self, "structure", copied)
        for name in ("leaf_logits", "core_mask", "anchors"):
            value = getattr(self, name).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        # Reports are intentionally shared while fit builds the provisional
        # program; the final evidence hash binds the complete resulting dict.

    @property
    def feature_dim(self):
        return FEATURE_DIM

    @property
    def feature_names(self):
        return FEATURE_NAMES

    def _structure(self):
        tree, capacity = _validated_structure(self.structure, FEATURE_NAMES)
        n = tree["leaf_count"]
        if (capacity["depth"] > 5 or n > 16 or capacity["maximum_source_features_per_node"] > 3
                or self.role not in AGENTS or not isinstance(self.actor_role_sha256, str)
                or len(self.actor_role_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.actor_role_sha256)
                or self.leaf_logits.shape != (n, 5) or self.anchors.shape != (n, 5)
                or self.core_mask.shape != (n,) or np.any((self.core_mask != 0) & (self.core_mask != 1))
                or any(not np.isfinite(a).all() for a in (self.leaf_logits, self.anchors))
                or self.leaf_logits.dtype != np.float32 or self.anchors.dtype != np.float32
                or not any(abs(self.residual_bound-b) < 1e-7 for b in (.35, .50))
                or not (self.feedback_leaves | self.explanation_leaves) <= set(range(n))):
            raise RuntimeError("shared program tensor, core, binding or complexity mismatch")
        return tree, capacity

    def certificates(self):
        tree, _ = self._structure()
        return certified_leaf_actions(tree["split_weights"], tree["split_bias"],
            tree["left_refs"], tree["right_refs"], len(tree["split_weights"]), tree["leaf_count"],
            _MASKS, explicit=True, wait_action=4)

    def leaf_indices(self, encoded):
        tree, capacity = self._structure()
        return leaf_indices(encoded, tree["split_weights"], tree["split_bias"],
            tree["left_refs"], tree["right_refs"], len(tree["split_weights"]), capacity["depth"])

    def validate(self):
        tree, capacity = self._structure()
        proof = self.certificates()
        r = self.report
        if (r.get("version") != VERSION or r.get("fit_method") != METHOD
                or r.get("actor_role_sha256") != self.actor_role_sha256
                or r.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or r.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256
                or r.get("program_structure_sha256") != program_structure_sha256(tree)
                or r.get("max_depth") != capacity["depth"] or r.get("leaf_count") != tree["leaf_count"]
                or r.get("core_anchor_sha256") != _digest(self.anchors.tolist())
                or r.get("core_mask") != self.core_mask.astype(int).tolist()
                or r.get("residual_bound") != self.residual_bound or r.get("program_box_epsilon") != BOX_EPSILON
                or r.get("core_KL_upper_bound") != (self.residual_bound+BOX_EPSILON)**2/2
                or r.get("core_membership_changes_with_qualification") is not False
                or r.get("excluded_frames") != 0):
            raise RuntimeError("unknown shared program method/schema or corrupted Actor/topology/core binding")
        fits, cals = r.get("leaf_fits", {}), r.get("leaf_calibrations", {})
        expected = {str(i) for i in range(tree["leaf_count"])}
        if set(fits) != expected or set(cals) != expected or len(r.get("fit_leaf_row_counts", [])) != len(expected):
            raise RuntimeError("shared program lacks complete leaf fit/calibration evidence")
        for leaf in range(tree["leaf_count"]):
            action = int(self.leaf_logits[leaf].argmax())
            fit, cal = fits[str(leaf)], cals[str(leaf)]
            if (not float32_preference_proof(self.leaf_logits[leaf], action, proof[leaf])["passed"]
                    or fit.get("success") is not True or fit.get("float32_feasible") is not True
                    or fit.get("solver") != "SLSQP" or fit.get("solver_status") != 0
                    or fit.get("maxiter") != LEAF_MAXITER or fit.get("tolerance") != LEAF_TOLERANCE
                    or fit.get("regularization") != LEAF_REGULARIZATION
                    or fit.get("epsilon") != (BOX_EPSILON if self.core_mask[leaf] else LEAF_EPSILON)
                    or type(fit.get("enumerated_action_solves")) is not int or fit["enumerated_action_solves"] < 0
                    or fit.get("chosen_action") != action or cal.get("chosen_action") != action
                    or cal.get("temperature") not in CALIBRATION_TEMPERATURES
                    or type(r["fit_leaf_row_counts"][leaf]) is not int or r["fit_leaf_row_counts"][leaf] < 0
                    or fit.get("row_count") != r["fit_leaf_row_counts"][leaf]
                    or (leaf in self.feedback_leaves | self.explanation_leaves and not r["fit_leaf_row_counts"][leaf])):
                raise RuntimeError("shared program leaf lost path, fit or float32 proof")
            if self.core_mask[leaf]:
                t = self.anchors[leaf]
                margin = float(t[action])-float(np.delete(t, action).max())
                if (action != int(t.argmax()) or margin+1e-6 < 2*(self.residual_bound+BOX_EPSILON)+.26
                        or not _box_proof(self.leaf_logits[leaf], t, action, proof[leaf])["passed"]):
                    raise RuntimeError("shared core anchor/margin/box certificate failed")
        if (r.get("leaf_optimization_calls") != tree["leaf_count"]
                or r.get("convex_solver_calls") != sum(item["enumerated_action_solves"] for item in fits.values())):
            raise RuntimeError("shared program leaf/solver accounting differs from actual per-leaf evidence")
        qualification = r.get("leaf_eligibility", {})
        if set(qualification) != expected:
            raise RuntimeError("shared program lacks complete independent leaf qualification")
        for purpose, selected, threshold in (("feedback", self.feedback_leaves, .85),
                                              ("explanation", self.explanation_leaves, .90)):
            qualified = frozenset(leaf for leaf in range(tree["leaf_count"])
                if r["fit_leaf_row_counts"][leaf] and _eligible(qualification[str(leaf)][purpose], threshold))
            if selected != qualified:
                raise RuntimeError("shared program qualification differs from independent eligibility evidence")
        for key in ("split_episode_ids", "split_scenario_seed_ids"):
            groups = r.get(key, {})
            if set(groups) != set(_SPLITS) or any(not isinstance(z, list) or not z for z in groups.values()):
                raise RuntimeError("shared program lacks four independent episode partitions")
            if any(set(groups[a]) & set(groups[b]) for a in _SPLITS for b in _SPLITS if a < b):
                raise RuntimeError("shared program episode partitions overlap")
        for purpose in ("feedback", "explanation"):
            metric = r.get("validation", {}).get(purpose, {})
            if any(type(metric.get(k)) is not int or metric[k] != 0 for k in (
                    "whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count", "core_KL_bound_violations")):
                raise RuntimeError("shared program lacks explicit zero whole-tree legality evidence")
            if (r["validation"].get(purpose+"_passed") is not _passes(metric, purpose)
                    or r["validation"].get("development_"+purpose+"_passed") is not _passes(metric, purpose, development=True)):
                raise RuntimeError("shared program validation gate evidence changed")
        return self

    def to_dict(self):
        self.validate()
        return {"version": VERSION, "method": METHOD, "role": self.role,
            "actor_role_sha256": self.actor_role_sha256, "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
            "structure": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.structure.items()},
            "leaf_logits": self.leaf_logits.tolist(), "core_mask": self.core_mask.astype(int).tolist(),
            "anchors": self.anchors.tolist(), "residual_bound": self.residual_bound,
            "feedback_leaves": sorted(self.feedback_leaves), "explanation_leaves": sorted(self.explanation_leaves),
            "report": self.report}

    def sha256(self):
        return _digest(self.to_dict())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data, *, actor_role_sha256):
        if (data.get("version") != VERSION or data.get("method") != METHOD
                or data.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or data.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256
                or data.get("actor_role_sha256") != actor_role_sha256):
            raise RuntimeError("shared program version, method, schema, physics or Actor hash mismatch")
        raw_structure = dict(data["structure"])
        if np.asarray(raw_structure["split_weights"]).shape == (0,):
            if raw_structure.get("feature_dim") != FEATURE_DIM:
                raise RuntimeError("empty shared program structure needs its explicit feature schema dimension")
            raw_structure["split_weights"] = np.zeros((0, FEATURE_DIM), np.float32)
        tree, _ = _validated_structure(raw_structure, FEATURE_NAMES)
        sets = [data[key] for key in ("feedback_leaves", "explanation_leaves")]
        if any(not isinstance(a, list) or any(type(v) is not int for v in a) or len(a) != len(set(a)) for a in sets):
            raise RuntimeError("invalid shared program qualification sets")
        result = cls(data["role"], actor_role_sha256, tree,
            np.asarray(data["leaf_logits"], np.float32), np.asarray(data["core_mask"], np.float32),
            np.asarray(data["anchors"], np.float32), data["residual_bound"],
            frozenset(sets[0]), frozenset(sets[1]), dict(data["report"]))
        return result.validate()

    @classmethod
    def load(cls, path, *, expected_sha256, actor_role_sha256):
        raw = Path(path).read_bytes()
        if not expected_sha256 or sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("shared program file hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256)


def _actor_parts(role_actor):
    # Importing a new method never silently accepts an old additive Actor.
    from .warehouse_v621_actor import MODEL_VERSION, FrozenWarehouseV621Role
    if (not isinstance(role_actor, FrozenWarehouseV621Role) or role_actor.model_version != MODEL_VERSION or role_actor.feature_dim != FEATURE_DIM
            or tuple(role_actor.feature_names) != tuple(FEATURE_NAMES)):
        raise RuntimeError("shared program requires the new explicitly versioned Actor")
    w = role_actor.weights
    nodes, count = int(w["tree.node_count"][0]), int(w["tree.leaf_count"][0])
    tree, capacity = _validated_structure({"root_ref": 0 if nodes else -1,
        "split_weights": w["tree.split.weight"][:nodes], "split_bias": w["tree.split.bias"][:nodes],
        "left_refs": w["tree.left_refs"][:nodes], "right_refs": w["tree.right_refs"][:nodes],
        "leaf_count": count}, FEATURE_NAMES)
    if capacity["depth"] > 5 or count > 16 or capacity["maximum_source_features_per_node"] > 3:
        raise RuntimeError("shared Actor exceeds the registered program complexity")
    return tree, np.asarray(w["core_leaf_mask"][:count], np.float32).copy(), \
        np.asarray(w["tree.leaf_logits"][:count], np.float32).copy(), float(w["core_residual_bound"][0])


def _binding(program, role_actor, *, require_role_sha):
    tree, core, anchors, bound = _actor_parts(role_actor)
    # A live PPO snapshot may change non-core tree logits. Only shared core
    # anchors support the analytic box bound; other changes are adjudicated by
    # the independent full-NN fidelity metrics on the fixed held-out rows.
    anchor_rows = np.ones(len(core), bool) if require_role_sha else core > .5
    return bool((not require_role_sha or role_actor.role_sha256 == program.actor_role_sha256)
        and program_structure_sha256(tree) == program_structure_sha256(program.structure)
        and np.array_equal(core, program.core_mask)
        and np.array_equal(anchors[anchor_rows], program.anchors[anchor_rows])
        and bound == program.residual_bound)


def _metric(program, role_actor, rows, selected_leaves, purpose):
    if purpose not in ("feedback", "explanation"):
        raise ValueError("unknown shared program evidence purpose")
    x, m, sampled = _arrays(rows, "shared program metrics")
    leaves = program.leaf_indices(x)
    p = _probs(role_actor.logits_encoded(x), m)
    q = _probs(program.leaf_logits[leaves], m)
    raw_top = program.leaf_logits[leaves].argmax(-1)
    top, nn = q.argmax(-1), p.argmax(-1)
    certified = program.certificates()
    raw_legal = m[np.arange(len(m)), raw_top] > .5
    path = np.asarray([int(a) in certified[int(l)] for a, l in zip(raw_top, leaves, strict=True)])
    chosen = np.isin(leaves, list(selected_leaves)); n = int(chosen.sum())
    valid = raw_legal & path
    matched = valid & (top == sampled)
    expected = valid*p[np.arange(len(p)), top]
    if purpose == "feedback":
        matched &= sampled == nn
        expected *= top == nn
    episode = np.asarray([str(r["episode"]) for r in rows])
    base = np.asarray([str(r.get("base_seed", r["episode"])) for r in rows])
    _, frequencies = np.unique(episode[chosen], return_counts=True)
    agree = float(np.mean(top[chosen] == nn[chosen])) if n else 0.
    kl = _kl(p, q)
    core_rows = program.core_mask[leaves] > .5
    bound = (program.residual_bound+BOX_EPSILON)**2/2
    return {"sample_count": len(rows), "physical_frame_count": len(rows), "excluded_frames": 0,
        "rows": n, "coverage": n/len(rows), "actionable_rows": int((matched & chosen).sum()),
        "actionable_coverage": float((matched & chosen).sum()/len(rows)),
        "expected_actionable_rows": float(expected[chosen].sum()),
        "expected_actionable_coverage": float(expected[chosen].sum()/len(rows)),
        "agreement": agree, "agreement_wilson_lower_90": _wilson_lower(agree, n),
        "kl": float(kl[chosen].mean()) if n else None,
        "illegal_count": int((~raw_legal & chosen).sum()), "masked_illegal_count": 0,
        "path_uncertified_count": int((~path & chosen).sum()),
        "whole_tree_raw_illegal_count": int((~raw_legal).sum()),
        "whole_tree_path_uncertified_count": int((~path).sum()),
        "core_KL_upper_bound": bound, "core_actual_max_kl": float(kl[core_rows].max()) if core_rows.any() else None,
        "core_KL_bound_violations": int(np.sum(core_rows & (kl > bound+5e-6))),
        "certified_rows": int((valid & chosen).sum()),
        "distinct_episode_count": len(frequencies), "distinct_scenario_seed_count": len(set(base[chosen])),
        "max_episode_fraction": float(frequencies.max()/n) if n else 0.,
        "whole_tree_agreement": float((top == nn).mean()), "whole_tree_kl": float(kl.mean()),
        "core_rows": int(program.core_mask[leaves].sum()),
        "per_nn_action": {a: {"physical_frames": int((nn == i).sum()),
            "rows": int(((nn == i) & chosen).sum()),
            "qualified_coverage": float(((nn == i) & chosen).sum()/(nn == i).sum()) if np.any(nn == i) else None,
            "actual_actionable_rows": int(((nn == i) & chosen & matched).sum())}
            for i, a in enumerate(ACTIONS)}}


def _passes(metric, purpose, *, development=False):
    agreement = (.93 if purpose == "feedback" else .95) if development else (.85 if purpose == "feedback" else .90)
    coverage = .30 if development else .20
    return bool(metric["coverage"] >= coverage and metric["actionable_coverage"] >= coverage
        and metric["agreement"] >= agreement and metric["kl"] is not None
        and metric["kl"] <= (.25 if development else .35)
        and all(metric[k] == 0 for k in ("illegal_count", "path_uncertified_count",
            "whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count", "core_KL_bound_violations")))


def fit_shared_program(role, actor, fit, calibration, eligibility, validation):
    if role not in AGENTS:
        raise ValueError("invalid shared program role")
    parts = dict(zip(_SPLITS, map(list, (fit, calibration, eligibility, validation)), strict=True))
    ids = {name: sorted({str(r["episode"]) for r in rows}) for name, rows in parts.items()}
    scenarios = {name: sorted({str(r.get("base_seed", r["episode"])) for r in rows}) for name, rows in parts.items()}
    if any(set(groups[a]) & set(groups[b]) for groups in (ids, scenarios)
           for a in _SPLITS for b in _SPLITS if a < b):
        raise ValueError("shared program fit/calibration/eligibility/validation overlap")
    arrays = {name: _arrays(rows, name) for name, rows in parts.items()}
    role_actor = actor.roles[role]
    tree, core, anchors, bound = _actor_parts(role_actor)
    n = tree["leaf_count"]
    report = {"version": VERSION, "fit_method": METHOD, "actor_role_sha256": role_actor.role_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256, "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "program_structure_sha256": program_structure_sha256(tree),
        "core_anchor_sha256": _digest(anchors.tolist()), "core_mask": core.astype(int).tolist(),
        "residual_bound": bound, "program_box_epsilon": BOX_EPSILON,
        "core_KL_upper_bound": (bound+BOX_EPSILON)**2/2,
        "core_bound_scope": "same_anchor_same_physical_mask_centered_per_action_logit_perturbations",
        "core_membership_changes_with_qualification": False,
        "split_episode_ids": ids, "split_scenario_seed_ids": scenarios, "excluded_frames": 0,
        "leaf_count": n, "max_depth": _validated_structure(tree, FEATURE_NAMES)[1]["depth"],
        "leaf_fits": {}, "leaf_calibrations": {}, "leaf_eligibility": {}, "validation": {},
        "full_NN_distribution_targets": True, "external_program_selects_execution_action": False,
        "routing_contract": HARD_SPLIT_ROUTING_VERSION}
    provisional = WarehouseV621Program(role, role_actor.role_sha256, tree,
        np.zeros((n, 5), np.float32), core, anchors, bound, frozenset(), frozenset(), report)
    certificates = provisional.certificates()
    targets = {name: _probs(role_actor.logits_encoded(x), m).astype(np.float32)
               for name, (x, m, _) in arrays.items()}
    routes = {name: provisional.leaf_indices(x) for name, (x, _, _) in arrays.items()}
    for name, (x, _, _) in arrays.items():
        if not np.array_equal(routes[name], role_actor.leaf(x)):
            raise RuntimeError("shared program does not preserve frozen Actor hard routing")
    counts = np.bincount(routes["fit"], minlength=n)
    report["fit_leaf_row_counts"] = counts.astype(int).tolist()
    report["fit_target_sha256"] = sha256(np.ascontiguousarray(targets["fit"], dtype="<f4").tobytes()).hexdigest()
    logits = []
    for leaf in range(n):
        take = routes["fit"] == leaf
        if core[leaf]:
            z, detail = fit_core_leaf_distribution(targets["fit"][take], arrays["fit"][1][take], certificates[leaf], anchors[leaf])
        else:
            z, detail = fit_constrained_leaf_distribution(targets["fit"][take], arrays["fit"][1][take], certificates[leaf])
        report["leaf_fits"][str(leaf)] = detail
        take = routes["calibration"] == leaf
        if core[leaf]:
            z, detail = calibrate_core_leaf_distribution(z, targets["calibration"][take], arrays["calibration"][1][take], certificates[leaf], anchors[leaf])
        else:
            z, detail = calibrate_constrained_leaf_distribution(z, targets["calibration"][take], arrays["calibration"][1][take], certificates[leaf], chosen_action=detail["chosen_action"])
        report["leaf_calibrations"][str(leaf)] = detail
        logits.append(z)
    provisional = WarehouseV621Program(role, role_actor.role_sha256, tree,
        np.stack(logits), core, anchors, bound, frozenset(), frozenset(), report)
    f, e = set(), set()
    for leaf in range(n):
        fm = _metric(provisional, role_actor, parts["eligibility"], {leaf}, "feedback")
        em = _metric(provisional, role_actor, parts["eligibility"], {leaf}, "explanation")
        report["leaf_eligibility"][str(leaf)] = {"feedback": fm, "explanation": em}
        if counts[leaf] and _eligible(fm, .85):
            f.add(leaf)
        if counts[leaf] and _eligible(em, .90):
            e.add(leaf)
    program = WarehouseV621Program(role, role_actor.role_sha256, tree,
        np.stack(logits), core, anchors, bound, frozenset(f), frozenset(e), report)
    report["leaf_optimization_calls"] = n
    report["convex_solver_calls"] = sum(z["enumerated_action_solves"] for z in report["leaf_fits"].values())
    report["validation"] = {purpose: _metric(program, role_actor, parts["validation"], leaves, purpose)
        for purpose, leaves in (("feedback", f), ("explanation", e))}
    for purpose in ("feedback", "explanation"):
        report["validation"][purpose+"_passed"] = _passes(report["validation"][purpose], purpose)
        report["validation"]["development_"+purpose+"_passed"] = _passes(report["validation"][purpose], purpose, development=True)
    return program.validate()


def program_metrics(program, actor, rows):
    program.validate()
    role_actor = actor.roles[program.role]
    if not _binding(program, role_actor, require_role_sha=True):
        raise RuntimeError("shared program is bound to a different Actor, core anchor or topology")
    return {purpose: _metric(program, role_actor, rows, leaves, purpose)
        for purpose, leaves in (("feedback", program.feedback_leaves), ("explanation", program.explanation_leaves))}


def current_actor_fidelity(program, actor, rows, *, purpose="feedback"):
    program.validate()
    if purpose not in ("feedback", "explanation"):
        raise ValueError("unknown shared program fidelity purpose")
    role_actor = actor.roles[program.role]
    episodes = {str(r["episode"]) for r in rows}
    scenes = {str(r.get("base_seed", r["episode"])) for r in rows}
    if any(episodes & set(program.report["split_episode_ids"][name])
           or scenes & set(program.report["split_scenario_seed_ids"][name])
           for name in ("fit", "calibration", "eligibility")):
        raise ValueError("current fidelity cannot reuse fit, calibration or eligibility episodes")
    same = _binding(program, role_actor, require_role_sha=False)
    leaves = program.feedback_leaves if purpose == "feedback" else program.explanation_leaves
    metric = _metric(program, role_actor, rows, leaves, purpose)
    return {**metric, "passed": bool(same and program.report["validation"][purpose+"_passed"]
        and _passes(metric, purpose, development=True)),
        "current_actor_topology_matches_program": same,
        "current_actor_role_sha256": role_actor.role_sha256, "program_sha256": program.sha256(),
        "purpose": purpose, "independent_validation_required": True}


def differentiable_feedback_loss(program, current_logits, encoded, masks, sampled_actions, *,
                                 current_fidelity, variant="regularized"):
    """KL(program || complete NN), after independent validation and action match.

    The denominator is all physical batch rows. No noise or surrogate gradient
    is injected: an identical distribution has zero loss and gradient. raw has
    exactly zero external program loss and gradient, regardless of evidence.
    """
    import torch
    if variant not in ("raw", "regularized"):
        raise ValueError("unknown shared program training variant")
    if (current_logits.ndim != 2 or current_logits.shape[1] != 5 or not len(current_logits)
            or not current_logits.is_floating_point() or not torch.isfinite(current_logits).all().item()):
        raise ValueError("feedback requires nonempty complete five-action logits")
    zero = current_logits.sum()*0
    empty = {"variant": variant, "eligible_rows": 0, "feedback_rows": 0, "loss": 0.,
             "external_program_gradient_enabled": False, "noise_or_artificial_gradient": False}
    if variant == "raw":
        return zero, empty
    program.validate()
    if (not isinstance(current_fidelity, Mapping) or current_fidelity.get("passed") is not True
            or current_fidelity.get("purpose") != "feedback"
            or current_fidelity.get("independent_validation_required") is not True
            or not isinstance(current_fidelity.get("current_actor_role_sha256"), str)
            or len(current_fidelity["current_actor_role_sha256"]) != 64
            or current_fidelity.get("program_sha256") != program.sha256()):
        return zero, empty
    x = np.asarray(encoded.detach().cpu() if torch.is_tensor(encoded) else encoded, np.float32)
    m = np.asarray(masks.detach().cpu() if torch.is_tensor(masks) else masks, np.float32)
    actions = np.asarray(sampled_actions.detach().cpu() if torch.is_tensor(sampled_actions) else sampled_actions)
    if (x.shape != (len(current_logits), FEATURE_DIM) or m.shape != (len(x), 5)
            or actions.shape != (len(x),) or not np.issubdtype(actions.dtype, np.integer)
            or np.any(actions < 0) or np.any(actions >= 5)
            or not torch.isfinite(current_logits).all().item()
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or not np.array_equal(x[:, _MASKS], m) or np.any(m[np.arange(len(m)), actions] != 1)):
        raise ValueError("feedback physical masks, actions or encoded features are inconsistent")
    validate_certification_domain(x, _MASKS, certified_feature_bounds(tuple(FEATURE_NAMES)))
    leaf = program.leaf_indices(x)
    legal = torch.as_tensor(m > .5, device=current_logits.device)
    def physical_log_softmax(logits):
        maximum = logits.masked_fill(~legal, -torch.inf).amax(-1, keepdim=True).detach()
        # Illegal raw values cannot overflow a subtraction used only to
        # normalize legal actions. Center legal logits before finite masking.
        shifted = torch.where(legal, logits, maximum)-maximum
        if not torch.isfinite(shifted).all().item():
            raise RuntimeError("complete NN legal-logit normalization overflowed")
        return torch.log_softmax(shifted.masked_fill(~legal, -1e9), -1)
    logp = physical_log_softmax(current_logits)
    qz = torch.as_tensor(program.leaf_logits[leaf], device=current_logits.device, dtype=current_logits.dtype)
    logq = physical_log_softmax(qz).detach()
    nn_top = logp.detach().argmax(-1).cpu().numpy()
    program_top = program.leaf_logits[leaf].argmax(-1)
    certificates = program.certificates()
    valid = np.asarray([int(a) in certificates[int(l)] for a, l in zip(program_top, leaf, strict=True)])
    eligible = np.isin(leaf, list(program.feedback_leaves)) & valid
    selected = eligible & (program_top == nn_top) & (actions == nn_top)
    select = torch.as_tensor(selected, device=current_logits.device, dtype=current_logits.dtype)
    loss = (torch.sum(logq.exp()*(logq-logp), -1)*select).mean()
    return loss, {**empty, "eligible_rows": int(eligible.sum()), "feedback_rows": int(selected.sum()),
        "loss": float(loss.detach()), "external_program_gradient_enabled": bool(selected.any()),
        "program_sha256": program.sha256(),
        "current_actor_role_sha256": current_fidelity["current_actor_role_sha256"],
        "denominator": "all_physical_batch_rows"}
