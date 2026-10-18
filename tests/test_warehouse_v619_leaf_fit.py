"""Constrained optimization uses every masked probability and preserves proofs."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from backend.training.warehouse_v619_leaf_fit import (
    ConstrainedLeafFitError, calibrate_constrained_leaf_distribution,
    fit_constrained_leaf_distribution, float32_preference_proof,
)


def test_mask_pattern_statistics_exactly_match_full_frame_objective_and_never_rewrite_wait():
    p = np.tile(np.array([.7, .05, .05, .05, .15], np.float32), (24, 1))
    masks = np.ones_like(p)
    masks[12:, 0] = 0
    p[12:] = [0, .1, .1, .1, .7]
    before_p, before_m = p.copy(), masks.copy()
    from backend.training.warehouse_v618_selective_tree import fit_leaf_distribution
    unconstrained = fit_leaf_distribution(p, masks)
    assert unconstrained.argmax() == 0
    assert np.count_nonzero(masks[:, unconstrained.argmax()] == 0) == 12
    z, report = fit_constrained_leaf_distribution(p, masks, {4})
    assert z.dtype == np.float32 and z.argmax() == 4
    assert float32_preference_proof(z, 4, {4})['passed']
    assert report['physical_fit_frame_count'] == 24 and report['mask_pattern_count'] == 2
    assert report['enumerated_action_solves'] == 1 and report['solver_status'] == 0
    selected = np.where(masks > .5, z.astype(np.float64), -np.inf)
    logq = selected-logsumexp(selected, axis=-1, keepdims=True)
    ce = -np.mean(np.sum(np.where(p > 0, p.astype(np.float64)*np.where(masks > .5, logq, 0), 0), axis=-1))
    prior = np.asarray(report['prior_logits'])
    objective = ce+report['regularization']*np.sum((z-prior)**2)
    assert report['masked_cross_entropy'] == pytest.approx(ce, abs=1e-10)
    assert report['objective'] == pytest.approx(objective, abs=1e-10)
    assert np.ptp(z[1:4]) < 1e-5
    assert z[4]-z[0] < .1  # Optimized full distribution, not an arbitrary WAIT spike.
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(masks, before_m)


def test_same_argmax_but_changed_nonmax_mass_changes_fitted_distribution():
    m = np.ones((30, 5))
    p = np.tile([.45,.05,.1,.1,.3], (30,1))
    first, _ = fit_constrained_leaf_distribution(p, m, {0,4})
    p[:,1:3] = p[:,[2,1]]
    second, _ = fit_constrained_leaf_distribution(p, m, {0,4})
    assert first.argmax() == second.argmax() == 0
    assert not np.allclose(first, second)
    assert first[1] < first[2] and second[1] > second[2]


def test_empty_leaf_is_deterministic_optimized_prior_and_never_qualifies():
    first, report = fit_constrained_leaf_distribution(np.empty((0,5)), np.empty((0,5)), {0,4})
    second, _ = fit_constrained_leaf_distribution(np.empty((0,5)), np.empty((0,5)), {0,4})
    np.testing.assert_array_equal(first, second)
    assert report['chosen_action'] == 0  # Stable action tie order, not a WAIT fallback.
    assert report['empty_leaf'] and not report['eligible_for_qualification']
    assert report['enumerated_action_solves'] == 2
    assert float32_preference_proof(first, 0, {0,4})['passed']


def test_calibration_rejects_temperatures_that_erase_stored_float32_margin():
    mask = np.tile([1,0,0,0,1], (20,1))
    p = np.tile([.5,0,0,0,.5], (20,1))
    z, report = fit_constrained_leaf_distribution(p, mask, {0,4})
    calibrated, cal = calibrate_constrained_leaf_distribution(
        z, p, mask, {0,4}, chosen_action=report['chosen_action'])
    assert calibrated.argmax() == report['chosen_action']
    assert float32_preference_proof(calibrated, report['chosen_action'], {0,4})['passed']
    rejected = [row for row in cal['candidate_reports'] if not row['accepted']]
    assert any(row['temperature'] > 1 for row in rejected)


def test_nonconverged_solvers_reject_and_count_every_attempt(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'minimize', lambda *a, **k:
        SimpleNamespace(success=False,status=9,message='budget',nit=200,x=np.zeros(5)))
    with pytest.raises(ConstrainedLeafFitError) as error:
        fit_constrained_leaf_distribution(np.tile([.3,.1,.1,.1,.4],(3,1)),np.ones((3,5)),{0,4})
    assert error.value.enumerated_action_solves == 2
    assert len(error.value.report['candidate_reports']) == 2


def test_declared_path_action_conflicting_with_observed_mask_is_rejected():
    with pytest.raises(ValueError, match='conflicts'):
        fit_constrained_leaf_distribution(np.tile([0,.1,.1,.1,.7],(4,1)),
                                         np.tile([0,1,1,1,1],(4,1)), {0,4})
