"""Independent empty-support float32 centering regressions; no solver/reset."""
from hashlib import sha256

import numpy as np
import pytest

from backend.training import warehouse_v621_actor as actor
from backend.training import warehouse_v621_program as program
from backend.training.warehouse_v619_leaf_fit import float32_preference_proof


WITNESS = np.asarray([-3.8417201042175293, -5.141720294952393,
    3.3582801818847656, 1.1582801342010498, 4.466877460479736], np.float32)


def frozen_role(anchor):
    values = {name: np.zeros(shape, np.float32) for name, shape in actor.tensor_shapes().items()}
    values["tree.leaf_count"][0] = 1
    values["tree.leaf_logits"][0] = anchor
    values["core_leaf_mask"][0] = 1
    values["core_actions"].fill(-1)
    values["core_actions"][0] = 4
    values["core_residual_bound"][0] = .35
    values["core_margin_floor"][0] = actor.required_core_margin(.35)
    return actor.FrozenWarehouseV621Role(values)


def nearby_anchors():
    result = [WITNESS.copy()]
    for coordinate in range(5):
        for direction in (-np.inf, np.inf):
            t = WITNESS.copy()
            t[coordinate] = np.nextafter(t[coordinate], np.float32(direction))
            result.append(t)
    return result


def assert_box_center_action(z, t):
    assert z.dtype == np.float32 and np.isfinite(z).all()
    assert abs(float(z.astype(np.float64).sum())) <= 2e-6
    assert np.max(np.abs(z.astype(np.float64)-t.astype(np.float64))) <= .05
    assert float32_preference_proof(z, 4, {4})["passed"]


def test_witness_is_accepted_actor_but_needs_program_center_repair():
    role = frozen_role(WITNESS)
    t = role.weights["tree.leaf_logits"][0]
    assert 2e-6 < abs(float(t.astype(np.float64).sum())) < 1e-4
    assert abs(float(t.mean())) < 2e-5
    assert float(t[4])-float(t[:4].max()) > actor.required_core_margin(.35)
    assert role.certified_actions() == (frozenset({4}),)
    assert not program._box_proof(t, t, 4, {4})["passed"]


@pytest.mark.parametrize("anchor", nearby_anchors(), ids=[f"float-neighbor-{i}" for i in range(11)])
def test_empty_core_fit_and_calibration_repair_only_program_without_solver(anchor):
    role = frozen_role(anchor)
    t = role.weights["tree.leaf_logits"][0]
    before = {name: value.tobytes() for name, value in role.weights.items()}
    actor_sha = role.role_sha256
    z, report = program.fit_core_leaf_distribution(np.zeros((0, 5)), np.zeros((0, 5)), {4}, t)
    assert report["success"] and report["empty_leaf"]
    assert report["row_count"] == 0 and report["enumerated_action_solves"] == 0
    assert report["proof"]["passed"]
    assert_box_center_action(z, t)
    calibrated, calibration = program.calibrate_core_leaf_distribution(
        z, np.zeros((0, 5)), np.zeros((0, 5)), {4}, t)
    assert calibration["empty_calibration_leaf"]
    assert calibration["proof"]["passed"]
    assert_box_center_action(calibrated, t)
    assert {name: value.tobytes() for name, value in role.weights.items()} == before
    assert role.role_sha256 == actor_sha == actor.role_sha256(role.weights)


def test_repaired_empty_program_preserves_both_KL_bounds_for_all_masks_and_residual_corners():
    t = frozen_role(WITNESS).weights["tree.leaf_logits"][0]
    z, _ = program.fit_core_leaf_distribution(np.zeros((0, 5)), np.zeros((0, 5)), {4}, t)
    raw_nn, masks = [], []
    for mask_bits in range(16):
        mask = [float(bool(mask_bits & (1 << a))) for a in range(4)]+[1.]
        for residual_bits in range(32):
            r = np.asarray([.35 if residual_bits & (1 << a) else -.35 for a in range(5)], np.float64)
            raw_nn.append(t.astype(np.float64)+r)
            masks.append(mask)
    masks = np.asarray(masks, np.float64)
    p = actor.masked_probabilities(np.asarray(raw_nn), masks)
    q = actor.masked_probabilities(np.broadcast_to(z, p.shape), masks)
    assert np.max(program._kl(p, q)) <= .08+5e-6
    assert np.max(program._kl(q, p)) <= .08+5e-6
    assert np.all(p.argmax(-1) == 4) and np.all(q.argmax(-1) == 4)
