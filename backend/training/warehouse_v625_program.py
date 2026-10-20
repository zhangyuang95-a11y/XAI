"""Versioned sparse logistic surrogate leaves; never an execution controller.

The stored program uses float32 coefficients and an explicitly ordered float32
leaf forward pass.  A conservative box certificate proves every unconditionally
path-illegal action is strictly below one fixed path-legal witness.  Thus legal
actions can exchange preference within a leaf without changing execution rules.
Unknown public feature domains and uncertain certificates fail closed.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from env.warehouse.navigation import ACTIONS
from .warehouse_v6_actor import FEATURE_NAMES, FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
from .warehouse_v6_tree_routing import certified_feature_bounds, HARD_SPLIT_ROUTING_VERSION
from .warehouse_v622_program import (
    DEFAULT_LIMITS, SPLITS, _arrays, _certificates, _eligible, _role_actor, _route,
    _wilson, digest, gate_metrics, masked_probabilities, normalize_structure,
    plain, program_structure_sha256, source_feature_cost,
)
from .warehouse_v619_leaf_fit import ConstrainedLeafFitError, CALIBRATION_TEMPERATURES

VERSION = "warehouse-role-program.v6.25-sparse-logistic"
LEAF_VERSION = "warehouse-sparse-logistic-leaf.v1"
ARITHMETIC = "ordered-fp32-multiply-add-k3-raw-logit-argmax-first.v1"
CERTIFICATE_VERSION = "path-legal-fixed-witness-global-box-exact-corners.v1"
MAX_SOURCE_FEATURES = 3
MAX_COEFFICIENTS = 20
FEATURE_MENU = 8
MAX_FULL_FITS = 8
COEFFICIENT_BOUND = 32.
CERTIFICATE_MARGIN = 1.e-3
REGULARIZATION = 1.e-4
MAXITER = 200
TOLERANCE = 1.e-6
LEAF_NUMERIC_CONTRACT = {
    "arithmetic": ARITHMETIC, "weights_shape": "five_actions_by_support",
    "support_order": "sorted_feature_schema_indices", "coefficient_dtype": "float32_exact_json",
    "forward": "bias_then_support_order_float32_multiply_then_float32_add_no_fma",
    "raw_argmax": "first_index", "calibration": "coefficients_already_temperature_divided_and_recertified",
    "box": "certified_feature_bounds_plus_physical_legal_bits_0_1",
    "roundoff": "k0_zero_else_nextafter_gamma_2kplus2_times_abs_sum_plus_2kplus2_tiny32",
    "gamma": "n_eps32_divided_by_1_minus_n_eps32", "strict_margin": CERTIFICATE_MARGIN,
    "coefficient_absolute_bound": COEFFICIENT_BOUND,
}


def public_bounds(names):
    """Only immutable environmental bounds, never empirical fit min/max."""
    names = tuple(names)
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) for n in names):
        raise ValueError("unique public feature names required")
    bounds = certified_feature_bounds(names)
    for i, name in enumerate(names):
        if name in ("legal_" + a.lower() for a in ACTIONS):
            bounds[i] = (0., 1.)
    return bounds


def _support(value, names):
    if value is None:
        raise ValueError("explicit support required")
    result = []
    for v in value:
        if isinstance(v, str):
            if v not in names:
                raise ValueError("unknown support feature")
            v = names.index(v)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError("support indices must be integers")
        result.append(int(v))
    if result != sorted(set(result)) or any(not 0 <= i < len(names) for i in result):
        raise ValueError("support must be unique, increasing, and in schema")
    w = np.zeros(len(names), np.float32)
    w[result] = 1
    if source_feature_cost(w, names) > MAX_SOURCE_FEATURES or len(result) > MAX_SOURCE_FEATURES:
        raise ValueError("leaf exceeds three public source features")
    if any(i not in public_bounds(names) for i in result):
        raise ValueError("leaf support has no immutable public bound")
    return tuple(result)


def _legal(value):
    raw = tuple(value)
    if (not raw or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer))
                       or not 0 <= a < 5 for a in raw)):
        raise ValueError("path-proven action indices required")
    result = tuple(sorted(set(map(int, raw))))
    if 4 not in result:
        raise ValueError("Warehouse WAIT must be included in the path proof")
    return result


def _coefficients(leaf, names):
    if leaf.get("version") != LEAF_VERSION or leaf.get("arithmetic") != ARITHMETIC:
        raise ValueError("unknown sparse logistic leaf schema/arithmetic")
    if tuple(leaf.get("feature_names", ())) != tuple(names):
        raise ValueError("leaf full feature schema differs from its program")
    support = _support(leaf.get("support"), names)
    if leaf.get("support_names") != [names[i] for i in support]:
        raise ValueError("leaf support names do not match indices")
    bias, weights = np.asarray(leaf.get("bias"), np.float32), np.asarray(leaf.get("weights"), np.float32)
    if weights.shape == (0,) and not support:
        weights = np.zeros((5, 0), np.float32)
    if (bias.shape != (5,) or weights.shape != (5, len(support))
            or not np.isfinite(bias).all() or not np.isfinite(weights).all()
            or np.max(np.abs(bias), initial=0) > COEFFICIENT_BOUND
            or np.max(np.abs(weights), initial=0) > COEFFICIENT_BOUND):
        raise ValueError("invalid bounded float32 leaf coefficients")
    # JSON must store exact float32 values; silently rounding tampered assets
    # would certify a different program from the one that was serialized.
    if (not np.array_equal(np.asarray(leaf["bias"], np.float64), bias.astype(np.float64))
            or not np.array_equal(np.asarray(leaf["weights"], np.float64), weights.astype(np.float64))):
        raise ValueError("serialized coefficients are not exact float32 values")
    return support, bias, weights


def _rounding_error(bias, weights, support, names):
    if not support:
        return np.zeros(5, np.float64)
    bounds = public_bounds(names)
    magnitude = np.abs(bias.astype(np.float64))
    for j, index in enumerate(support):
        magnitude += np.abs(weights[:, j].astype(np.float64)) * max(map(abs, bounds[index]))
    operations = 2 * len(support) + 2
    u = np.finfo(np.float32).eps  # Twice unit roundoff, intentionally conservative.
    gamma = operations * u / (1 - operations * u)
    return np.nextafter(gamma * magnitude + operations * float(np.finfo(np.float32).tiny), np.inf)


def certify_leaf(leaf, names, legal_actions):
    """Recompute a strict raw-top certificate from stored coefficients.

    Exact rational corner gaps avoid trusting an LP solver's infeasibility
    flag.  Box domains ignore path correlations and so are conservative.
    This routine does not assert that arbitrary caller-supplied legal actions
    are valid path proofs; the complete program recomputes those from topology.
    """
    names, legal = tuple(names), _legal(legal_actions)
    support, bias, weights = _coefficients(leaf, names)
    witness = leaf.get("witness")
    if type(witness) is not int or witness not in legal:
        raise ValueError("fixed witness must have a path legality proof")
    bounds = public_bounds(names)
    corners = list(product(*(bounds[i] for i in support))) or [()]
    error = _rounding_error(bias, weights, support, names)
    checks = []
    for action in range(5):
        if action in legal:
            continue
        minimum = None
        for corner in corners:
            gap = Fraction.from_float(float(bias[witness])) - Fraction.from_float(float(bias[action]))
            for j, x in enumerate(corner):
                gap += (Fraction.from_float(float(weights[witness, j]))
                        - Fraction.from_float(float(weights[action, j]))) * Fraction.from_float(float(x))
            minimum = gap if minimum is None else min(minimum, gap)
        required = (Fraction.from_float(CERTIFICATE_MARGIN)
                    + Fraction.from_float(float(error[witness])) + Fraction.from_float(float(error[action])))
        checks.append({"action": action, "passed": minimum > required,
            "minimum_exact_gap": [minimum.numerator, minimum.denominator],
            "minimum_real_gap": float(minimum), "required_gap": float(required)})
    binding = {"version": LEAF_VERSION, "arithmetic": ARITHMETIC, "support": list(support),
        "support_names": [names[i] for i in support], "bias": bias.tolist(), "weights": weights.tolist(),
        "witness": witness, "legal_actions": list(legal),
        "bounds": [[names[i], *bounds[i]] for i in support]}
    return {"version": CERTIFICATE_VERSION, "passed": all(c["passed"] for c in checks),
        "proof_scope": "all_public_box_values_raw_top_in_path_proven_actions",
        "path_legal_actions": list(legal), "witness": witness, "bounds": binding["bounds"],
        "corner_count": len(corners), "coefficient_count": int(bias.size + weights.size),
        "source_feature_cost": sum(2 if names[i] == "public_hard_risk_context" else 1 for i in support),
        "float32_error_per_action": error.tolist(), "strict_margin": CERTIFICATE_MARGIN,
        "illegal_action_checks": checks, "binding_sha256": digest(binding),
        "tie_break": "numpy_argmax_first_index_strict_witness_gap_excludes_illegal_ties"}


def evaluate_leaf(leaf, x, masks):
    names = tuple(leaf.get("feature_names", ()))
    support, bias, weights = _coefficients(leaf, names)
    x, m = np.asarray(x, np.float32), np.asarray(masks, np.float32)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all() or m.shape != (len(x), 5):
        raise ValueError("invalid sparse logistic inference frame arrays")
    legal_names = ["legal_" + a.lower() for a in ACTIONS]
    for i, name in enumerate(names):
        if name in legal_names and np.any(x[:, i] != m[:, legal_names.index(name)]):
            raise ValueError("encoded public legal bit differs from physical mask")
    bounds = public_bounds(names)
    for i in support:
        lo, hi = bounds[i]
        if np.any((x[:, i] < lo) | (x[:, i] > hi)):
            raise ValueError("public feature is outside the certified inference domain")
    raw = np.broadcast_to(bias, (len(x), 5)).copy()
    for j, index in enumerate(support):
        raw = np.add(raw, np.multiply(x[:, index, None], weights[None, :, j], dtype=np.float32), dtype=np.float32)
    return raw, masked_probabilities(raw, m)


def _consume(budget):
    if budget is None:
        return None
    if hasattr(budget, "consume"):
        return budget.consume()
    if callable(budget):
        return budget()
    raise TypeError("leaf budget needs consume() or a callable")


def _check(budget):
    if budget is not None and hasattr(budget, "check"):
        budget.check()


def _finish(budget, token, success, detail):
    if budget is not None and hasattr(budget, "finish"):
        budget.finish(token, success=bool(success), detail=plain(detail))


def _inputs(x, masks, targets, names, legal):
    x, m, p = np.asarray(x, np.float32), np.asarray(masks, np.float64), np.asarray(targets, np.float64)
    if (x.ndim != 2 or x.shape[1] != len(names) or m.shape != (len(x), 5) or p.shape != m.shape
            or not np.isfinite(x).all() or not np.isfinite(m).all() or not np.isfinite(p).all()
            or np.any((m != 0) & (m != 1)) or np.any(m[:, 4] != 1)
            or np.any(p < 0) or np.any(p[m == 0] != 0)
            or len(p) and not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0)):
        raise ValueError("complete five-action masked NN distribution inputs required")
    if any(np.any(m[:, a] != 1) for a in legal):
        raise ValueError("path legality proof conflicts with observed physical masks")
    legal_names = ["legal_" + a.lower() for a in ACTIONS]
    for i, name in enumerate(names):
        if name in legal_names and np.any(x[:, i] != m[:, legal_names.index(name)]):
            raise ValueError("encoded public legal bit differs from physical mask")
    return x, m, p


def _candidate_supports(x, p, m, names, legal, initial):
    if not len(x):
        return [()], []
    prior_q = masked_probabilities(np.broadcast_to(initial, (len(x), 5)), m)
    residual = p - prior_q
    scores = []
    for index, (lower, upper) in public_bounds(names).items():
        col = x[:, index].astype(np.float64)
        if np.any((col < lower) | (col > upper)):
            raise ValueError("public fit feature lies outside immutable bounds")
        centered = col - col.mean()
        scale = float(np.linalg.norm(centered))
        if scale > 1.e-12:
            score = float(np.linalg.norm(centered @ residual) / scale)
            scores.append((score, index))
    menu = [i for _, i in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:FEATURE_MENU]]
    ranks = {i: score for score, i in scores}
    possible = []
    for size in (1, 2, 3):
        for support in combinations(sorted(menu), size):
            try:
                support = _support(support, names)
            except ValueError:
                continue
            possible.append((sum(ranks[i] for i in support), support))
    possible.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return [(), *[s for _, s in possible[:MAX_FULL_FITS - 1]]], menu


def fit_leaf(x, masks, targets, feature_names, legal_actions, mode="P1", support=None,
             initial_logits=None, budget=None, witness=None):
    """Fit bounded supports with a fixed legality witness and explicit charges.

    A supplied support (including []) performs exactly one full trial. Search
    mode performs at most eight trials, with the constant trial first. Every
    attempted scipy solve is consumed before entry and settled on exit.
    """
    names, legal = tuple(feature_names), _legal(legal_actions)
    public_bounds(names)
    if mode not in ("P0", "P1"):
        raise ValueError("mode must be P0 or P1")
    x, m, p = _inputs(x, masks, targets, names, legal)
    prior = np.log((p.sum(0) + .2) / (m.sum(0) + 1.))
    prior -= prior.mean()
    initial = prior if initial_logits is None else np.asarray(initial_logits, np.float64)
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError("finite five-action initial logits required")
    if witness is None:
        top = int(initial.argmax())
        witness = top if top in legal else max(legal, key=lambda a: (float(p[:, a].sum()), -a))
    if type(witness) is not int or witness not in legal:
        raise ValueError("fixed witness is not path-proven legal")
    if mode == "P0" and support is not None and len(support):
        raise ValueError("P0 has constant leaves")
    if support is not None:
        supports, menu = [_support(support, names)], []
    elif mode == "P0":
        supports, menu = [()], []
    else:
        supports, menu = _candidate_supports(x, p, m, names, legal, initial.astype(np.float32))
    candidates, records = [], []
    for chosen in supports:
        _check(budget)
        k = len(chosen)
        bounds = public_bounds(names)
        corners = list(product(*(bounds[i] for i in chosen))) or [()]
        # Fixed coefficient bounds yield a fixed roundoff reserve, preserving
        # convexity. Re-certification uses each stored solution's tighter bound.
        operations = 2 * k + 2
        u = np.finfo(np.float32).eps
        max_magnitude = COEFFICIENT_BOUND * (1 + sum(max(map(abs, bounds[i])) for i in chosen))
        error_reserve = operations * u / (1 - operations * u) * max_magnitude if k else 0.
        margin = CERTIFICATE_MARGIN + 2 * error_reserve + 1.e-5
        design = np.column_stack((np.ones(len(x)), x[:, chosen].astype(np.float64)))
        raw_legal = np.asarray([a in legal for a in range(5)])
        matrix = []
        for a in np.flatnonzero(~raw_legal):
            for corner in corners:
                row = np.zeros((5, k + 1), np.float64)
                row[witness] = (1., *corner)
                row[a] = -np.asarray((1., *corner))
                matrix.append(row.ravel())
        constraint = np.asarray(matrix, np.float64).reshape(-1, 5 * (k + 1))
        prior_parameters = np.zeros((5, k + 1), np.float64)
        prior_parameters[:, 0] = prior
        start = np.zeros_like(prior_parameters)
        start[:, 0] = np.clip(initial - initial.mean(), -COEFFICIENT_BOUND / 2, COEFFICIENT_BOUND / 2)
        if np.any(~raw_legal):
            start[witness, 0] = max(start[witness, 0], float(start[~raw_legal, 0].max()) + margin + .01)
        start -= start.mean(0)
        def objective(flat):
            _check(budget)
            values = flat.reshape(5, k + 1)
            logits = design @ values.T
            if len(p):
                masked = np.where(m > .5, logits, -np.inf)
                normalizers = logsumexp(masked, axis=-1)
                q = np.exp(masked - normalizers[:, None])
                ce = float(np.mean(p.sum(-1) * normalizers - np.sum(p * logits, axis=-1)))
                gradient = (q * p.sum(-1)[:, None] - p).T @ design / len(p)
            else:
                ce, gradient = 0., np.zeros_like(values)
            offset = values - prior_parameters
            return ce + REGULARIZATION * float(np.sum(offset * offset)), (gradient + 2 * REGULARIZATION * offset).ravel()
        constraints = []
        if len(constraint):
            constraints.append({"type": "ineq", "fun": lambda v, c=constraint: c @ v - margin,
                                "jac": lambda v, c=constraint: c})
        center = np.zeros((k + 1, 5 * (k + 1)), np.float64)
        for j in range(k + 1):
            center[j, j::k + 1] = 1
        constraints.append({"type": "eq", "fun": lambda v: center @ v, "jac": lambda v: center})
        token = _consume(budget)
        begun = time.monotonic()
        try:
            result = minimize(objective, start.ravel(), jac=True, method="SLSQP",
                bounds=[(-COEFFICIENT_BOUND, COEFFICIENT_BOUND)] * start.size,
                constraints=constraints, options={"maxiter": MAXITER, "ftol": TOLERANCE, "disp": False})
            values = np.asarray(result.x, np.float64).reshape(5, k + 1).astype(np.float32)
            leaf = {"version": LEAF_VERSION, "arithmetic": ARITHMETIC, "feature_names": list(names),
                "support": list(chosen), "support_names": [names[i] for i in chosen],
                "bias": values[:, 0].tolist(), "weights": values[:, 1:].tolist(), "witness": witness,
                "mode": mode, "fit_loss": float(result.fun)}
            certificate = certify_leaf(leaf, names, legal)
            leaf["certificate"] = certificate
            raw, q = evaluate_leaf(leaf, x, m)
            forward_kl = float(np.mean(np.sum(p * (np.log(np.maximum(p, 1.e-300))
                - np.log(np.maximum(q.astype(np.float64), 1.e-300))), axis=-1))) if len(p) else 0.
            accepted = bool(result.success and np.isfinite(result.fun) and certificate["passed"])
            record = {"support": list(chosen), "witness": witness, "success": accepted,
                "solver_success": bool(result.success), "solver": "SLSQP", "solver_status": int(result.status),
                "iterations": int(result.nit), "message": str(result.message), "objective": float(result.fun),
                "full_distribution_forward_kl": forward_kl, "certificate_passed": certificate["passed"],
                "convex_fixed_support_witness_problem": True, "elapsed_seconds": time.monotonic() - begun}
        except BaseException as error:
            _finish(budget, token, False, {"support": list(chosen), "witness": witness,
                "exception": type(error).__name__, "message": str(error), "elapsed_seconds": time.monotonic() - begun})
            raise
        _finish(budget, token, accepted, record)
        records.append(record)
        if accepted:
            candidates.append((float(result.fun), chosen, leaf))
    if not candidates:
        raise ConstrainedLeafFitError("No converged float32-certified sparse logistic leaf", candidate_reports=records)
    _, _, winner = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
    winner["report"] = {"mode": mode, "row_count": len(x), "empty_leaf": not bool(len(x)),
        "eligible_for_qualification": bool(len(x)), "candidate_reports": records,
        "full_fit_calls": len(records), "convex_solver_calls": len(records), "feature_menu": menu,
        "maximum_full_fits": MAX_FULL_FITS, "objective": "KL_complete_masked_NN_to_program_plus_prior_L2",
        "regularization": REGULARIZATION, "coefficient_bound": COEFFICIENT_BOUND,
        "fixed_witness": witness, "support_search_uses_fit_only": True,
        "nonconvex_support_and_witness_choice_claimed_convex": False}
    return winner


def calibrate_leaf(leaf, x, masks, targets, names, legal_actions):
    """Finite temperature calibration, preserving support and fixed witness."""
    names, legal = tuple(names), _legal(legal_actions)
    x, m, p = _inputs(x, masks, targets, names, legal)
    if not certify_leaf(leaf, names, legal)["passed"]:
        raise ValueError("calibration requires a current complete leaf certificate")
    choices, records = [], []
    for index, temperature in enumerate(CALIBRATION_TEMPERATURES):
        trial = deepcopy(leaf)
        trial["bias"] = (np.asarray(leaf["bias"], np.float32) / np.float32(temperature)).astype(np.float32).tolist()
        trial["weights"] = (np.asarray(leaf["weights"], np.float32) / np.float32(temperature)).astype(np.float32).tolist()
        try:
            proof = certify_leaf(trial, names, legal)
        except ValueError:
            proof = {"passed": False}
        loss = None
        if proof["passed"]:
            _, q = evaluate_leaf(trial, x, m)
            loss = float(np.mean(np.sum(p * (np.log(np.maximum(p, 1.e-300))
                - np.log(np.maximum(q.astype(np.float64), 1.e-300))), axis=-1))) if len(p) else 0.
            trial["certificate"] = proof
            choices.append((loss, index if len(p) else abs(temperature - 1), trial, temperature))
        records.append({"temperature": temperature, "full_distribution_forward_kl": loss, "certificate_passed": proof["passed"]})
    if not choices:
        raise ValueError("no certificate-preserving calibration temperature")
    loss, _, result, temperature = min(choices, key=lambda item: (item[0], item[1]))
    return result, {"temperature": temperature, "row_count": len(x), "empty_calibration_leaf": not bool(len(x)),
        "full_distribution_forward_kl": loss, "candidate_reports": records, "support_reselected": False,
        "witness_reselected": False, "certificate_recomputed_after_float32_scaling": True,
        "pre_calibration_leaf": deepcopy(leaf)}


@dataclass
class WarehouseV625Program:
    role: str
    actor_role_sha256: str
    structure: dict
    leaves: list
    feedback_leaves: frozenset
    explanation_leaves: frozenset
    report: dict
    mode: str = "P1"
    feature_names: tuple = tuple(FEATURE_NAMES)

    def leaf_indices(self, x):
        tree, capacity = normalize_structure(self.structure, self.feature_names)
        return _route(tree, np.asarray(x, np.float32), capacity["depth"])

    def certificates(self):
        return _certificates(self.structure, self.feature_names)

    def distributions(self, x, masks):
        x, masks = np.asarray(x, np.float32), np.asarray(masks, np.float32)
        routes = self.leaf_indices(x)
        raw, q = np.empty((len(x), 5), np.float32), np.empty((len(x), 5), np.float32)
        for i, leaf in enumerate(self.leaves):
            take = routes == i
            raw[take], q[take] = evaluate_leaf(leaf, x[take], masks[take])
        return routes, raw, q

    def validate(self):
        tree, capacity = normalize_structure(self.structure, self.feature_names)
        if (self.mode not in ("P0", "P1") or len(self.leaves) != tree["leaf_count"]
                or self.role not in ("robot_1", "robot_2") or not isinstance(self.actor_role_sha256, str)
                or len(self.actor_role_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.actor_role_sha256)
                or tuple(self.feature_names) != tuple(FEATURE_NAMES)):
            raise ValueError("invalid v625 program role/model/schema")
        proofs = self.certificates()
        for i, leaf in enumerate(self.leaves):
            computed = certify_leaf(leaf, self.feature_names, proofs[i])
            if not computed["passed"] or computed != leaf.get("certificate") or leaf.get("mode") != self.mode:
                raise ValueError("stored v625 leaf certificate is stale, uncertain or invalid")
            if self.mode == "P0" and leaf["support"]:
                raise ValueError("constant baseline contains a variable leaf")
        for chosen in (self.feedback_leaves, self.explanation_leaves):
            if (not isinstance(chosen, frozenset) or any(type(v) is not int for v in chosen)
                    or not chosen <= set(range(len(self.leaves)))):
                raise ValueError("invalid qualified leaf IDs")
        r = self.report
        if (r.get("version") != VERSION or r.get("actor_role_sha256") != self.actor_role_sha256
                or r.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or r.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256
                or r.get("program_structure_sha256") != program_structure_sha256(tree)
                or r.get("external_program_selects_execution_action") is not False
                or r.get("full_NN_distribution_targets") is not True
                or r.get("arithmetic") != ARITHMETIC):
            raise ValueError("program evidence lost its physical/model/method binding")
        groups = [r.get(key, {}) for key in ("split_episode_ids", "split_scenario_seed_ids")]
        for group in groups:
            if (set(group) != set(SPLITS) or any(not isinstance(v, list) or not v or v != sorted(set(v)) for v in group.values())
                    or any(set(group[a]) & set(group[b]) for a in SPLITS for b in SPLITS if a < b)):
                raise ValueError("whole-episode program evidence partitions overlap or are missing")
        for purpose, chosen, threshold in (("feedback", self.feedback_leaves, .93), ("explanation", self.explanation_leaves, .95)):
            expected = frozenset(i for i in range(len(self.leaves)) if r["fit_leaf_row_counts"][i]
                and _eligible(r["leaf_eligibility"][str(i)][purpose], threshold))
            if expected != chosen:
                raise ValueError("leaf qualification differs from independent eligibility evidence")
        validation = r.get("validation", {})
        if validation.get("gate") != gate_metrics(validation, phase="development"):
            raise ValueError("stored validation gate differs from unchanged numerical thresholds")
        return self

    def to_dict(self):
        self.validate()
        return plain({"version": VERSION, "role": self.role, "actor_role_sha256": self.actor_role_sha256,
            "structure": self.structure, "leaves": self.leaves, "feedback_leaves": sorted(self.feedback_leaves),
            "explanation_leaves": sorted(self.explanation_leaves), "report": self.report,
            "mode": self.mode, "feature_names": list(self.feature_names)})

    def sha256(self):
        return digest(self.to_dict())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with path.open("xb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        return sha256(raw).hexdigest()

    @classmethod
    def from_dict(cls, data, *, actor_role_sha256, role=None):
        required = {"version", "role", "actor_role_sha256", "structure", "leaves", "feedback_leaves", "explanation_leaves", "report", "mode", "feature_names"}
        if (not isinstance(data, Mapping) or set(data) != required or data["version"] != VERSION
                or data["actor_role_sha256"] != actor_role_sha256 or role is not None and data["role"] != role):
            raise ValueError("unknown v625 schema or model/role hash mismatch")
        tree, _ = normalize_structure(data["structure"], tuple(data["feature_names"]))
        for key in ("feedback_leaves", "explanation_leaves"):
            if not isinstance(data[key], list) or data[key] != sorted(set(data[key])) or any(type(v) is not int for v in data[key]):
                raise ValueError("invalid serialized qualification IDs")
        return cls(data["role"], actor_role_sha256, tree, deepcopy(data["leaves"]), frozenset(data["feedback_leaves"]),
            frozenset(data["explanation_leaves"]), deepcopy(data["report"]), data["mode"], tuple(data["feature_names"])).validate()

    @classmethod
    def load(cls, path, *, expected_sha256, actor_role_sha256, role=None):
        raw = Path(path).read_bytes()
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("v625 program file hash mismatch")
        return cls.from_dict(json.loads(raw), actor_role_sha256=actor_role_sha256, role=role)


def _metrics(program, rows, arrays, p, selected, purpose, *, predictions=None):
    x, m, sampled = arrays
    route, raw_logits, q = program.distributions(x, m) if predictions is None else predictions
    raw, top, nn = raw_logits.argmax(-1), q.argmax(-1), p.argmax(-1)
    proofs = program.certificates()
    legal = m[np.arange(len(m)), raw] > 0
    path = np.asarray([int(a) in proofs[int(l)] for a, l in zip(raw, route, strict=True)])
    chosen = np.isin(route, tuple(selected))
    n, valid = int(chosen.sum()), legal & path
    matched, expected = valid & (top == sampled), valid * p[np.arange(len(p)), top]
    if purpose == "feedback":
        matched &= sampled == nn
        expected *= top == nn
    kl = np.sum(p * (np.log(np.maximum(p, np.float32(1.e-8))) - np.log(np.maximum(q, np.float32(1.e-8)))), axis=-1)
    frequencies = Counter(str(r["episode"]) for r, keep in zip(rows, chosen, strict=True) if keep)
    scenes = {str(r["base_seed"]) for r, keep in zip(rows, chosen, strict=True) if keep}
    agreement = float((top[chosen] == nn[chosen]).mean()) if n else 0.
    result = {"sample_count": len(rows), "physical_frame_count": len(rows), "excluded_frames": 0,
        "rows": n, "coverage": n / len(rows), "actionable_rows": int((chosen & matched).sum()),
        "actionable_coverage": float((chosen & matched).sum() / len(rows)),
        "expected_actionable_rows": float(expected[chosen].sum()), "expected_actionable_coverage": float(expected[chosen].sum() / len(rows)),
        "agreement": agreement, "agreement_wilson_lower_90": _wilson(agreement, n),
        "kl": float(kl[chosen].mean()) if n else None, "illegal_count": int((chosen & ~legal).sum()),
        "masked_illegal_count": int((chosen & (m[np.arange(len(m)), top] == 0)).sum()),
        "path_uncertified_count": int((chosen & ~path).sum()), "whole_tree_raw_illegal_count": int((~legal).sum()),
        "whole_tree_path_uncertified_count": int((~path).sum()), "certified_rows": int((chosen & valid).sum()),
        "distinct_episode_count": len(frequencies), "distinct_scenario_seed_count": len(scenes),
        "max_episode_fraction": max(frequencies.values()) / n if n else 0.,
        "whole_tree_agreement": float((top == nn).mean()), "whole_tree_kl": float(kl.mean())}
    result["per_nn_action"] = {a: {"physical_frames": int((nn == i).sum()), "rows": int((chosen & (nn == i)).sum()),
        "qualified_coverage": float((chosen & (nn == i)).sum() / (nn == i).sum()) if (nn == i).any() else None,
        "agreement": float((top[chosen & (nn == i)] == nn[chosen & (nn == i)]).mean()) if (chosen & (nn == i)).any() else None,
        "actual_actionable_rows": int((chosen & matched & (nn == i)).sum()),
        "expected_actionable_rows": float(expected[chosen & (nn == i)].sum())} for i, a in enumerate(ACTIONS)}
    conflict = x[:, FEATURE_NAMES.index("same_target_collision_streak_capped")] > 0
    near = x[:, FEATURE_NAMES.index("robots_shortest_distance")] * np.float32(42) <= 2
    low = x[:, FEATURE_NAMES.index("self_charger_return_margin")] * np.float32(50) <= 4
    result["public_strata"] = {}
    low_battery = x[:, FEATURE_NAMES.index("self_battery_fraction")] <= np.float32(.2)
    for label, take in {"ordinary": ~(conflict | near | low), "conflict": conflict, "near_robots": near,
                       "low_return_margin": low, "low_battery_le_20_percent": low_battery}.items():
        selected_rows, denominator = take & chosen, int(take.sum())
        support = int(selected_rows.sum())
        result["public_strata"][label] = {"rows": support, "physical_frames": denominator,
            "qualified_coverage": support / denominator if denominator else None,
            "agreement": float((top[selected_rows] == nn[selected_rows]).mean()) if support else None,
            "kl": float(kl[selected_rows].mean()) if support else None,
            "illegal_count": int((selected_rows & ~legal).sum()), "actual_actionable_rows": int((selected_rows & matched).sum()),
            "expected_actionable_rows": float(expected[selected_rows].sum())}
    return result


class _FinalBudget:
    def __init__(self, specification, checkpoint_path=None):
        self.spec = dict(specification or {})
        self.maximum = self.spec.get("max_leaf_optimization_calls", self.spec.get("maximum_leaf_api_calls", 32))
        self.calls = 0
        self.records = []
        self.inflight = None
        self.path = Path(checkpoint_path or self.spec["checkpoint_path"]) if checkpoint_path or self.spec.get("checkpoint_path") else None
        if type(self.maximum) is not int or not 1 <= self.maximum <= 32:
            raise ValueError("final fitting requires a maximum of 32 full solves")
        prior = self.spec.get("prior_leaf_optimization_calls", 0)
        total = self.spec.get("max_total_leaf_optimization_calls", 20000)
        if type(prior) is not int or type(total) is not int or prior < 0 or total > 20000 or prior + self.maximum > total:
            raise ValueError("final and prior leaf calls exceed the registered shared budget")
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise ValueError("final fitting checkpoint is single-use; an existing attempt cannot retry")
            self._persist(initial=True)
    def _persist(self, initial=False):
        if self.path is None:
            return
        raw = json.dumps({"schema": "warehouse-v625-final-refit-predebit.v1", "maximum": self.maximum,
            "calls": self.calls, "inflight": self.inflight, "records": self.records,
            "prior_leaf_optimization_calls": self.spec.get("prior_leaf_optimization_calls", 0)},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if initial:
            with self.path.open("xb") as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
        else:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("wb") as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            os.replace(temporary, self.path)
        descriptor = os.open(str(self.path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    def check(self):
        deadline = self.spec.get("deadline_monotonic")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("v625 final leaf deadline exhausted")
    def consume(self):
        self.check()
        if self.calls >= self.maximum:
            raise RuntimeError("v625 final leaf call budget exhausted before solve")
        self.calls += 1
        self.inflight = self.calls
        self._persist()
        return self.calls
    def finish(self, token, *, success, detail):
        if token != self.inflight:
            raise RuntimeError("final leaf debit settled twice or out of order")
        self.records.append({"token": token, "success": success, "detail": detail})
        self.inflight = None
        self._persist()


def fit_program(role, actor, fit, calibration, eligibility, validation, *, structure, mode,
                fit_budget=None, initial_leaves=None, checkpoint_path=None):
    """One refit per final leaf; the search's supports and witnesses are fixed."""
    role_actor = _role_actor(actor, role)
    tree, capacity = normalize_structure(structure)
    n = tree["leaf_count"]
    if initial_leaves is None:
        initial_leaves = structure.get("initial_leaf_models")
    if mode == "P1" and (not isinstance(initial_leaves, list) or len(initial_leaves) != n):
        raise ValueError("P1 final fit requires search-selected supports and witnesses")
    if initial_leaves is not None and (not isinstance(initial_leaves, list) or len(initial_leaves) != n):
        raise ValueError("incomplete initial leaves")
    budget = _FinalBudget(fit_budget, checkpoint_path) if fit_budget is None or isinstance(fit_budget, Mapping) else fit_budget
    parts = dict(zip(SPLITS, map(list, (fit, calibration, eligibility, validation)), strict=True))
    arrays = {name: _arrays(rows) for name, rows in parts.items()}
    ids = {name: sorted({str(r["episode"]) for r in rows}) for name, rows in parts.items()}
    scenes = {name: sorted({str(r["base_seed"]) for r in rows}) for name, rows in parts.items()}
    if any(set(groups[a]) & set(groups[b]) for groups in (ids, scenes) for a in SPLITS for b in SPLITS if a < b):
        raise ValueError("program fitting partitions overlap by complete episode/scenario")
    proofs = _certificates(tree)
    routes = {s: _route(tree, a[0], capacity["depth"]) for s, a in arrays.items()}
    targets = {s: masked_probabilities(role_actor.logits_encoded(a[0]), a[1]) for s, a in arrays.items()}
    counts = np.bincount(routes["fit"], minlength=n)
    report = {"version": VERSION, "actor_role_sha256": role_actor.role_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256, "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
        "program_structure_sha256": program_structure_sha256(tree), "arithmetic": ARITHMETIC,
        "numeric_contract": deepcopy(LEAF_NUMERIC_CONTRACT),
        "routing_contract": HARD_SPLIT_ROUTING_VERSION, "full_NN_distribution_targets": True,
        "external_program_selects_execution_action": False, "excluded_frames": 0, "mode": mode,
        "leaf_count": n, "max_depth": capacity["depth"], "limits": {**DEFAULT_LIMITS.to_dict(),
            "maximum_leaf_source_features": 3, "maximum_leaf_coefficients": 20},
        "split_episode_ids": ids, "split_scenario_seed_ids": scenes,
        "physical_frames_per_split": {s: len(rows) for s, rows in parts.items()},
        "fit_leaf_row_counts": counts.astype(int).tolist(), "leaf_fits": {}, "leaf_calibrations": {}, "leaf_eligibility": {},
        "fit_target_sha256": sha256(np.ascontiguousarray(targets["fit"], dtype="<f4").tobytes()).hexdigest(),
        "leaf_optimization_calls": 0, "convex_solver_calls": 0, "status": "fitting"}
    leaves = []
    for i in range(n):
        _check(budget)
        take = routes["fit"] == i
        initial = initial_leaves[i] if initial_leaves is not None else None
        if initial is not None and not certify_leaf(initial, FEATURE_NAMES, proofs[i])["passed"]:
            raise ValueError("search selected leaf lacks a current path certificate")
        leaf = fit_leaf(arrays["fit"][0][take], arrays["fit"][1][take], targets["fit"][take], FEATURE_NAMES,
            proofs[i], mode=mode, support=initial["support"] if initial is not None else [],
            initial_logits=initial["bias"] if initial is not None else None,
            witness=initial["witness"] if initial is not None else None, budget=budget)
        report["leaf_fits"][str(i)] = deepcopy(leaf["report"])
        report["leaf_optimization_calls"] += leaf["report"]["full_fit_calls"]
        report["convex_solver_calls"] += leaf["report"]["convex_solver_calls"]
        take = routes["calibration"] == i
        leaf, detail = calibrate_leaf(leaf, arrays["calibration"][0][take], arrays["calibration"][1][take],
            targets["calibration"][take], FEATURE_NAMES, proofs[i])
        report["leaf_calibrations"][str(i)] = detail
        leaves.append(leaf)
    program = WarehouseV625Program(role, role_actor.role_sha256, tree, leaves, frozenset(), frozenset(), report, mode)
    chosen = {"feedback": set(), "explanation": set()}
    eligibility_prediction = program.distributions(arrays["eligibility"][0], arrays["eligibility"][1])
    for i in range(n):
        _check(budget)
        metric = {purpose: _metrics(program, parts["eligibility"], arrays["eligibility"], targets["eligibility"], {i}, purpose,
            predictions=eligibility_prediction)
            for purpose in chosen}
        report["leaf_eligibility"][str(i)] = metric
        for purpose, threshold in (("feedback", .93), ("explanation", .95)):
            if counts[i] and _eligible(metric[purpose], threshold):
                chosen[purpose].add(i)
    program.feedback_leaves, program.explanation_leaves = frozenset(chosen["feedback"]), frozenset(chosen["explanation"])
    validation_prediction = program.distributions(arrays["validation"][0], arrays["validation"][1])
    report["validation"] = {purpose: _metrics(program, parts["validation"], arrays["validation"], targets["validation"], selected, purpose,
        predictions=validation_prediction)
        for purpose, selected in (("feedback", program.feedback_leaves), ("explanation", program.explanation_leaves))}
    report["validation"]["gate"] = gate_metrics(report["validation"], phase="development")
    report["leaf_optimization_calls_outside_search"] = report["leaf_optimization_calls"]
    report["fit_budget"] = {"prior_leaf_optimization_calls": fit_budget.get("prior_leaf_optimization_calls", 0),
        "actual_final_leaf_optimization_calls": report["leaf_optimization_calls"],
        "actual_total_leaf_optimization_calls": fit_budget.get("prior_leaf_optimization_calls", 0) + report["leaf_optimization_calls"]} if isinstance(fit_budget, Mapping) else {
            "actual_final_leaf_optimization_calls": report["leaf_optimization_calls"]}
    report["status"] = "completed"
    _check(budget)
    return program.validate()


def program_metrics(program, actor, rows):
    program.validate()
    role_actor = _role_actor(actor, program.role)
    if role_actor.role_sha256 != program.actor_role_sha256:
        raise ValueError("program is bound to a different complete NN")
    rows = list(rows)
    arrays = _arrays(rows)
    target = masked_probabilities(role_actor.logits_encoded(arrays[0]), arrays[1])
    predictions = program.distributions(arrays[0], arrays[1])
    result = {purpose: _metrics(program, rows, arrays, target, selected, purpose, predictions=predictions)
        for purpose, selected in (("feedback", program.feedback_leaves), ("explanation", program.explanation_leaves))}
    result["per_leaf"] = {str(i): {purpose: _metrics(program, rows, arrays, target, {i}, purpose, predictions=predictions)
        for purpose in ("feedback", "explanation")} for i in range(len(program.leaves))}
    result["gate"] = gate_metrics(result, phase="development")
    return result
