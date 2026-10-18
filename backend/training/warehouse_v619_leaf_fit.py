"""Path-constrained five-action distribution fitting, never an action selector.

For each path-certified candidate a, the masked full-NN cross entropy plus
prior L2 is convex in centered logits, under linear strict-preference bounds.
Mask-pattern sufficient statistics evaluate exactly the all-frame objective.
The discrete choice of candidate a and topology search are not convex claims.
"""
from __future__ import annotations

import numpy as np


LEAF_EPSILON = 1e-3
LEAF_REGULARIZATION = 1e-4
LEAF_MAXITER = 200
LEAF_TOLERANCE = 1e-6
CALIBRATION_TEMPERATURES = (.8, .9, 1., 1.1, 1.2)


class ConstrainedLeafFitError(RuntimeError):
    def __init__(self, message, *, candidate_reports):
        super().__init__(message)
        self.enumerated_action_solves = len(candidate_reports)
        self.report = {"success": False, "enumerated_action_solves": self.enumerated_action_solves,
                       "candidate_reports": candidate_reports}


def float32_preference_proof(logits, chosen_action: int, legal_actions, *,
                            epsilon: float = LEAF_EPSILON) -> dict:
    """Prove the stored float32 leaf retains its certified raw top action."""
    z = np.asarray(logits)
    legal = frozenset(int(a) for a in legal_actions)
    if (z.ndim != 1 or z.dtype != np.float32 or not np.isfinite(z).all()
            or type(chosen_action) not in (int, np.int64, np.int32)
            or not 0 <= chosen_action < len(z) or chosen_action not in legal):
        return {"passed": False, "minimum_raw_gap": None}
    other = np.delete(z.astype(np.float64), int(chosen_action))
    gap = float(z[int(chosen_action)]) - float(other.max(initial=-np.inf))
    return {"passed": bool(int(z.argmax()) == int(chosen_action) and gap >= epsilon),
            "minimum_raw_gap": gap, "chosen_action": int(chosen_action),
            "epsilon": float(epsilon), "dtype": "float32"}


def fit_constrained_leaf_distribution(target, masks, legal_actions, *,
        epsilon: float = LEAF_EPSILON,
        regularization: float = LEAF_REGULARIZATION,
        maxiter: int = LEAF_MAXITER,
        tolerance: float = LEAF_TOLERANCE) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    p = np.asarray(target, np.float64)
    m = np.asarray(masks, np.float64)
    if (p.ndim != 2 or p.shape[1] != 5 or m.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(m).all()
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(p < 0) or np.any(p[m == 0] > 0)
            or (len(p) and not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0))
            or not np.isfinite(epsilon) or epsilon <= 0
            or not np.isfinite(regularization) or regularization <= 0
            or type(maxiter) is not int or maxiter < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("invalid constrained leaf full-distribution inputs/config")
    raw_legal = tuple(legal_actions)
    if (not raw_legal or any(type(a) not in (int, np.int64, np.int32)
                            or not 0 <= a < 5 for a in raw_legal)):
        raise ValueError("constrained leaf requires path-certified action indices")
    legal = tuple(sorted(set(int(a) for a in raw_legal)))
    # A path certificate is inconsistent if an observed physical mask rejects
    # its action. Such a caller error is never repaired by another action.
    if len(m) and any(np.any(m[:, a] == 0) for a in legal):
        raise ValueError("path-certified action conflicts with observed physical mask")
    count = len(p)
    mean_target = p.mean(0) if count else np.zeros(5, np.float64)
    patterns, inverse, frequencies = np.unique(m, axis=0, return_inverse=True, return_counts=True)
    # For unit distributions this is count/n. Retaining the actual floating
    # row sums makes the grouped objective exactly equal to -mean(p*log q),
    # including float32 NN probabilities whose stored sum differs by one ULP.
    fractions = (np.bincount(inverse, weights=p.sum(-1), minlength=len(patterns))/count
                 if count else np.zeros(0, np.float64))
    prior = np.log((p.sum(0)+.2)/(m.sum(0)+1.))
    prior -= prior.mean()
    entropy_constant = float(np.sum(np.where(p > 0, p*np.log(np.maximum(p, 1e-300)), 0))/count) if count else 0.
    safety = 8*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(prior))))
    constraint_margin = epsilon+safety

    def ce_gradient(z):
        ce, gradient = -float(mean_target@z), -mean_target.copy()
        for pattern, fraction in zip(patterns, fractions, strict=True):
            active = pattern > .5
            normalizer = logsumexp(z[active])
            ce += float(fraction*normalizer)
            gradient[active] += fraction*np.exp(z[active]-normalizer)
        return ce, gradient

    def objective(z):
        ce, gradient = ce_gradient(z)
        offset = z-prior
        return ce+regularization*float(offset@offset), gradient+2*regularization*offset

    candidates, records = [], []
    for action in legal:
        others = [a for a in range(5) if a != action]
        matrix = np.zeros((4, 5), np.float64)
        matrix[:, action] = 1.
        matrix[np.arange(4), others] = -1.
        initial = prior.copy()
        initial[action] = max(initial[action], float(initial[others].max())+constraint_margin)
        initial -= initial.mean()
        try:
            result = minimize(objective, initial, method="SLSQP", jac=True,
                constraints=(
                    {"type": "eq", "fun": lambda z: float(z.sum()),
                     "jac": lambda z: np.ones(5, np.float64)},
                    {"type": "ineq", "fun": lambda z, mat=matrix: mat@z-constraint_margin,
                     "jac": lambda z, mat=matrix: mat},
                ), options={"maxiter": maxiter, "ftol": tolerance, "disp": False})
        except Exception as exc:
            records.append({"action": action, "success": False, "solver_status": None,
                            "message": f"{type(exc).__name__}: {exc}",
                            "finite": False, "float32_feasible": False})
            continue
        z64 = np.asarray(result.x, np.float64)
        z32 = np.asarray(z64-z64.mean(), np.float32)
        proof = float32_preference_proof(z32, action, legal, epsilon=epsilon)
        feasible = (np.isfinite(z64).all() and abs(float(z64.sum())) <= tolerance
                    and float(np.min(matrix@z64-constraint_margin)) >= -tolerance
                    and proof["passed"])
        final_ce, _ = ce_gradient(z32.astype(np.float64))
        final_objective, _ = objective(z32.astype(np.float64))
        record = {"action": action, "success": bool(result.success),
                  "solver_status": int(result.status), "message": str(result.message),
                  "iterations": int(result.nit), "finite": bool(np.isfinite(z32).all()),
                  "float32_feasible": bool(feasible),
                  "objective": float(final_objective) if np.isfinite(final_objective) else None,
                  "masked_cross_entropy": float(final_ce) if np.isfinite(final_ce) else None,
                  "masked_kl": float(final_ce+entropy_constant) if np.isfinite(final_ce) else None,
                  "proof": proof}
        records.append(record)
        if result.success and feasible and np.isfinite(final_objective):
            candidates.append((float(final_objective), action, z32, record))
    if not candidates:
        raise ConstrainedLeafFitError(
            "No converged finite float32-feasible path-constrained leaf solution: "+repr(records),
            candidate_reports=records)
    _, action, logits, winner = min(candidates, key=lambda item: (item[0], item[1]))
    report = {"success": True, "row_count": count, "empty_leaf": not bool(count),
        "eligible_for_qualification": bool(count), "chosen_action": action,
        "objective": winner["objective"], "masked_cross_entropy": winner["masked_cross_entropy"],
        "masked_kl": winner["masked_kl"], "epsilon": float(epsilon),
        "effective_constraint_margin": float(constraint_margin),
        "float32_safety_margin": float(safety), "regularization": float(regularization),
        "solver": "SLSQP", "solver_status": winner["solver_status"],
        "iterations": winner["iterations"], "maxiter": maxiter, "tolerance": float(tolerance),
        "float32_feasible": True, "enumerated_action_solves": len(legal),
        "mask_pattern_count": len(patterns), "physical_fit_frame_count": count,
        "objective_scope": "exact_all_frame_masked_cross_entropy_plus_prior_l2",
        "prior_logits": prior.tolist(), "candidate_reports": records,
    }
    return logits, report


def calibrate_constrained_leaf_distribution(logits, target, masks, legal_actions, *,
        chosen_action: int, epsilon: float = LEAF_EPSILON) -> tuple[np.ndarray, dict]:
    from scipy.special import logsumexp
    z = np.asarray(logits)
    p, m = np.asarray(target, np.float64), np.asarray(masks, np.float64)
    if (not float32_preference_proof(z, chosen_action, legal_actions, epsilon=epsilon)["passed"]
            or p.ndim != 2 or p.shape[1] != 5 or m.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(m).all()
            or np.any((m != 0) & (m != 1)) or np.any(m.sum(-1) < 1)
            or np.any(p < 0) or np.any(p[m == 0] > 0)
            or (len(p) and not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0))):
        raise ValueError("invalid constrained calibration inputs/certificate")
    if len(m) and any(np.any(m[:, int(a)] == 0) for a in legal_actions):
        raise ValueError("path certificate conflicts with calibration physical mask")
    candidates, records = [], []
    for temperature in CALIBRATION_TEMPERATURES:
        trial = np.asarray(z/temperature, np.float32)
        proof = float32_preference_proof(trial, chosen_action, legal_actions, epsilon=epsilon)
        if len(p):
            masked = np.where(m > .5, trial.astype(np.float64), -np.inf)
            logq = masked-logsumexp(masked, axis=-1, keepdims=True)
            kl = float(np.mean(np.sum(np.where(p > 0,
                p*(np.log(np.maximum(p, 1e-300))-np.where(m > .5, logq, 0)), 0), axis=-1)))
        else:
            kl = 0.
        records.append({"temperature": temperature, "masked_kl": kl,
                        "float32_proof": proof, "accepted": bool(proof["passed"] and np.isfinite(kl))})
        if proof["passed"] and np.isfinite(kl):
            candidates.append((kl, CALIBRATION_TEMPERATURES.index(temperature), trial, temperature))
    if not candidates:
        raise RuntimeError("No certificate-preserving calibration temperature")
    # Empty leaves have no calibration target: retain the fitted prior rather
    # than selecting a temperature from arbitrary equal-loss ties.
    if not len(p):
        return z.copy(), {"temperature": 1., "empty_calibration_leaf": True,
                           "masked_kl": 0., "chosen_action": chosen_action,
                           "candidate_reports": records}
    kl, _, calibrated, temperature = min(candidates, key=lambda item: (item[0], item[1]))
    return calibrated, {"temperature": temperature, "empty_calibration_leaf": False,
                        "masked_kl": kl, "chosen_action": chosen_action,
                        "candidate_reports": records}
