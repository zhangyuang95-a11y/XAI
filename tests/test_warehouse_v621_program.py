"""Synthetic tensor/row tests; never reset an environment or train a model."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from backend.training import warehouse_v621_program as p
from backend.training import warehouse_v621_actor as frozen
from backend.training import warehouse_v621_actor_training as training
from backend.training.warehouse_v6_tree_routing import certified_leaf_actions
from env.warehouse.public_observation_v6 import OBSERVATION_DIM, OBSERVATION_FEATURE_NAMES
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import MOVE_DELTAS


def model(*, bound=.35, mixed=False):
    torch.manual_seed(52)
    active = training.WarehouseV621Actor(core_residual_bound=bound)
    if mixed:
        w = np.zeros((1, p.FEATURE_DIM), np.float32)
        w[0, p.FEATURE_NAMES.index("legal_right")] = 1.
        tree = dict(root_ref=0, split_weights=w, split_bias=np.array([-.5], np.float32),
                    left_refs=np.array([-1]), right_refs=np.array([-2]), leaf_count=2)
        core, actions = [1, 0], [4, -1]
        logits = np.zeros((2, 5), np.float32)
    else:
        tree = dict(root_ref=-1, split_weights=np.zeros((0, p.FEATURE_DIM), np.float32),
                    split_bias=np.zeros(0, np.float32), left_refs=np.zeros(0, int),
                    right_refs=np.zeros(0, int), leaf_count=1)
        core, actions, logits = [1], [4], np.zeros((1, 5), np.float32)
    for role in p.AGENTS:
        active.install_structure(role, tree, core, actions, logits)
    return active


def parts(*, variable_masks=False):
    result = []
    for split in range(4):
        rows = []
        for episode in range(3):
            for frame in range(30):
                obs = np.zeros(OBSERVATION_DIM, np.float32)
                cells = STUDY_MAP_LAYOUT.passable_positions
                own, partner = cells[frame % len(cells)], cells[-1]
                index = {name: i for i, name in enumerate(OBSERVATION_FEATURE_NAMES)}
                for row, tiles in enumerate(STUDY_MAP_LAYOUT.tiles):
                    for column, tile in enumerate(tiles):
                        obs[index[f"wall_row_{row}_column_{column}"]] = float(tile == "#")
                positions = {"self": own, "partner": partner, "charger": STUDY_MAP_LAYOUT.charger_position,
                             "order_0_pickup": cells[0], "order_0_delivery": cells[1],
                             "order_1_pickup": cells[-2], "order_1_delivery": cells[-1]}
                for name, (row, column) in positions.items():
                    obs[index[name+"_row"]] = row/max(1, STUDY_MAP_LAYOUT.rows-1)
                    obs[index[name+"_column"]] = column/max(1, STUDY_MAP_LAYOUT.cols-1)
                obs[index["self_active"]] = obs[index["partner_active"]] = 1
                obs[index["self_battery_fraction"]] = obs[index["partner_battery_fraction"]] = 1
                mask = np.asarray([float(STUDY_MAP_LAYOUT.is_passable(
                    (own[0]+MOVE_DELTAS[a][0], own[1]+MOVE_DELTAS[a][1]))) if a in MOVE_DELTAS else 1.
                    for a in p.ACTIONS], np.float32)
                rows.append({"obs": obs, "mask": mask, "episode": f"s{split}-e{episode}",
                             "base_seed": 100*split+episode, "frame": frame, "sampled_action": "WAIT"})
        result.append(rows)
    return result


def fit(*, bound=.35, mixed=False):
    active = model(bound=bound, mixed=mixed)
    actor = frozen.FrozenWarehouseV621Actor.from_model(active)
    rows = parts(variable_masks=mixed)
    program = p.fit_shared_program("robot_1", actor, *rows)
    return active, actor, rows, program


@pytest.mark.parametrize("bound", [.35, .50])
def test_same_actor_structure_core_box_full_distribution_and_native_qualification(bound):
    active, actor, rows, program = fit(bound=bound)
    assert program.report["fit_method"] == p.METHOD
    assert program.report["full_NN_distribution_targets"]
    assert not program.report["core_membership_changes_with_qualification"]
    assert program.feedback_leaves == program.explanation_leaves == frozenset({0})
    assert program.report["leaf_optimization_calls"] == 1
    assert program.report["convex_solver_calls"] == 1
    assert program.report["core_KL_upper_bound"] == (program.residual_bound+.05)**2/2
    assert np.max(np.abs(program.leaf_logits-program.anchors)) <= .05
    assert abs(float(program.leaf_logits.sum())) < 2e-6
    assert program.report["leaf_fits"]["0"]["row_count"] == 90
    metrics = p.program_metrics(program, actor, rows[3])
    assert metrics["feedback"]["coverage"] == 1
    assert metrics["feedback"]["expected_actionable_coverage"] > .3
    assert metrics["feedback"]["whole_tree_raw_illegal_count"] == 0
    assert metrics["feedback"]["whole_tree_path_uncertified_count"] == 0
    assert metrics["feedback"]["agreement"] == 1
    assert metrics["feedback"]["kl"] < program.report["core_KL_upper_bound"]
    np.testing.assert_array_equal(program.core_mask, actor.roles["robot_1"].weights["core_leaf_mask"][:1])


def test_noncore_fits_only_path_certified_actions_and_does_not_change_core():
    active, actor, rows, program = fit(mixed=True)
    assert program.core_mask.tolist() == [1, 0]
    assert program.report["leaf_fits"]["1"]["solver"] == "SLSQP"
    assert int(program.leaf_logits[1].argmax()) in {3, 4}
    for action, cert in zip(program.leaf_logits.argmax(1), program.certificates(), strict=True):
        assert int(action) in cert
    assert program.report["leaf_fits"]["0"]["proof"]["passed"]
    metrics = p.program_metrics(program, actor, rows[3])
    assert metrics["feedback"]["whole_tree_raw_illegal_count"] == 0


def test_core_fit_uses_soft_targets_not_only_hard_action_and_box_calibration():
    anchor = np.array([-.4, -.4, -.4, -.4, 1.6], np.float32)
    masks = np.ones((20, 5), np.float32)
    target = np.tile([.1, .1, .1, .05, .65], (20, 1)).astype(np.float32)
    z, report = p.fit_core_leaf_distribution(target, masks, {4}, anchor)
    other = target.copy()
    other[:, 0] += .04
    other[:, 1] -= .04
    z2, _ = p.fit_core_leaf_distribution(other, masks, {4}, anchor)
    assert np.max(np.abs(z-z2)) > .01
    assert report["proof"]["passed"]
    calibrated, cal = p.calibrate_core_leaf_distribution(z, target, masks, {4}, anchor)
    assert np.max(np.abs(calibrated-anchor)) <= .05
    assert cal["proof"]["passed"]
    assert all(not row["accepted"] for row in cal["candidate_reports"] if row["proof"]["maximum_anchor_offset"] > .05)


def test_core_empty_leaf_has_no_solver_or_qualification_and_no_data_repair():
    anchor = np.array([-.4, -.4, -.4, -.4, 1.6], np.float32)
    z, report = p.fit_core_leaf_distribution(np.zeros((0, 5)), np.zeros((0, 5)), {4}, anchor)
    assert report["enumerated_action_solves"] == 0 and report["row_count"] == 0
    np.testing.assert_array_equal(z, anchor)
    invalid = np.array([[.2]*5], np.float32)
    with pytest.raises(ValueError, match="complete physically masked"):
        p.fit_core_leaf_distribution(invalid, [[0, 1, 1, 1, 1]], {4}, anchor)


def test_strict_save_load_sha_unknown_method_and_schema_tamper(tmp_path):
    _, _, _, program = fit()
    path = tmp_path/"program.json"
    digest = program.save(path)
    restored = p.WarehouseV621Program.load(path, expected_sha256=digest,
                                         actor_role_sha256=program.actor_role_sha256)
    assert restored.sha256() == program.sha256()
    with pytest.raises(FileExistsError):
        program.save(path)
    with pytest.raises(RuntimeError, match="file hash"):
        p.WarehouseV621Program.load(path, expected_sha256="0"*64, actor_role_sha256=program.actor_role_sha256)
    for key, bad in (("method", "unknown"), ("version", "old"), ("actor_role_sha256", "0"*64),
                     ("feature_schema_sha256", "0"*64), ("action_physics_config_sha256", "0"*64)):
        data = program.to_dict()
        data[key] = bad
        with pytest.raises(RuntimeError):
            p.WarehouseV621Program.from_dict(data, actor_role_sha256=program.actor_role_sha256)


@pytest.mark.parametrize("change", ["box", "topology", "qualification", "core_mask", "report_method", "whole_legality", "bound", "validation_gate", "overlap", "solver_count", "solver_recipe", "KL_bound"])
def test_load_rejects_invalid_boxes_paths_or_evidence(change):
    _, _, _, program = fit()
    data = program.to_dict()
    if change == "box":
        data["leaf_logits"][0][0] += .2
        data["leaf_logits"][0][1] -= .2
    elif change == "topology":
        data["structure"]["leaf_count"] = 2
    elif change == "qualification":
        data["feedback_leaves"] = []
    elif change == "core_mask":
        data["core_mask"] = [0]
    elif change == "report_method":
        data["report"]["fit_method"] = "unknown"
    elif change == "whole_legality":
        del data["report"]["validation"]["feedback"]["whole_tree_raw_illegal_count"]
    elif change == "bound":
        data["residual_bound"] = 1.
    elif change == "validation_gate":
        data["report"]["validation"]["feedback_passed"] = False
    elif change == "solver_count":
        data["report"]["convex_solver_calls"] += 1
    elif change == "solver_recipe":
        data["report"]["leaf_fits"]["0"]["regularization"] = 0.
    elif change == "KL_bound":
        data["report"]["core_KL_upper_bound"] = 1.
    else:
        data["report"]["split_episode_ids"]["validation"] = data["report"]["split_episode_ids"]["fit"]
    with pytest.raises((ValueError, RuntimeError)):
        p.WarehouseV621Program.from_dict(data, actor_role_sha256=program.actor_role_sha256)


@pytest.mark.parametrize("change", ["episode_overlap", "scene_overlap", "missing_frame", "duplicate_frame", "omitted_frame", "illegal_sample"])
def test_fit_rejects_leakage_or_incomplete_physical_frames(change):
    active = model()
    actor = frozen.FrozenWarehouseV621Actor.from_model(active)
    rows = parts()
    if change == "episode_overlap":
        rows[1][0]["episode"] = rows[0][0]["episode"]
    elif change == "scene_overlap":
        rows[1][0]["base_seed"] = rows[0][0]["base_seed"]
    elif change == "missing_frame":
        del rows[0][0]["frame"]
    elif change == "duplicate_frame":
        rows[0][1]["frame"] = 0
    elif change == "omitted_frame":
        rows[0][1]["frame"] = 2
    else:
        rows[0][0]["mask"][0] = 0
        rows[0][0]["sampled_action"] = "UP"
    with pytest.raises(ValueError):
        p.fit_shared_program("robot_1", actor, *rows)


def test_current_fidelity_allows_new_NN_hash_but_rejects_core_anchor_change():
    active, actor, rows, program = fit()
    with torch.no_grad():
        active.actors["robot_1"].action_residual[-1].weight.mul_(1.5)
    updated = frozen.FrozenWarehouseV621Actor.from_model(active)
    assert updated.role_sha256["robot_1"] != program.actor_role_sha256
    fidelity = p.current_actor_fidelity(program, updated, rows[3])
    assert fidelity["passed"]
    assert fidelity["current_actor_role_sha256"] == updated.role_sha256["robot_1"]
    with pytest.raises(RuntimeError, match="different Actor"):
        p.program_metrics(program, updated, rows[3])
    with pytest.raises(ValueError, match="cannot reuse"):
        p.current_actor_fidelity(program, updated, rows[0])
    with torch.no_grad():
        active.actors["robot_1"].tree.extra_margin[0] += .1
    changed = frozen.FrozenWarehouseV621Actor.from_model(active)
    assert not p.current_actor_fidelity(program, changed, rows[3])["passed"]


def test_real_residual_KL_gradient_raw_zero_and_failed_fidelity_zero():
    active, actor, rows, program = fit()
    with torch.no_grad():
        active.actors["robot_1"].action_residual[-1].weight.mul_(3)
    updated = frozen.FrozenWarehouseV621Actor.from_model(active)
    fidelity = p.current_actor_fidelity(program, updated, rows[3])
    assert fidelity["passed"]
    x, m, sampled = p._arrays(rows[3], "test")
    logits = active.actors["robot_1"](torch.from_numpy(x))
    loss, report = p.differentiable_feedback_loss(program, logits, x, m, sampled,
        current_fidelity=fidelity)
    assert loss.item() > 0 and report["feedback_rows"] == len(x)
    loss.backward()
    gradient = active.actors["robot_1"].action_residual[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not report["noise_or_artificial_gradient"]
    for variant, evidence in (("raw", fidelity), ("regularized", {**fidelity, "passed": False})):
        active.zero_grad()
        logits = active.actors["robot_1"](torch.from_numpy(x))
        loss, report = p.differentiable_feedback_loss(program, logits, x, m, sampled,
            current_fidelity=evidence, variant=variant)
        assert loss.item() == 0 and report["feedback_rows"] == 0
        loss.backward()
        assert active.actors["robot_1"].action_residual[-1].weight.grad.abs().sum() == 0


def test_identical_distributions_have_zero_loss_and_no_artificial_gradient():
    _, actor, rows, program = fit()
    fidelity = p.current_actor_fidelity(program, actor, rows[3])
    x, m, sampled = p._arrays(rows[3], "test")
    q = torch.tensor(program.leaf_logits[program.leaf_indices(x)], requires_grad=True)
    loss, report = p.differentiable_feedback_loss(program, q, x, m, sampled,
        current_fidelity=fidelity)
    assert loss.item() == 0
    loss.backward()
    assert q.grad.abs().max() < 1e-7


def test_feedback_requires_actual_sampled_NN_top_and_current_program_receipt():
    _, actor, rows, program = fit()
    fidelity = p.current_actor_fidelity(program, actor, rows[3])
    x, m, sampled = p._arrays(rows[3], "test")
    logits = torch.tensor(actor.roles["robot_1"].logits_encoded(x), requires_grad=True)
    for i in range(10):
        sampled[i] = np.flatnonzero(m[i, :4])[0]
    _, report = p.differentiable_feedback_loss(program, logits, x, m, sampled,
        current_fidelity=fidelity)
    assert report["feedback_rows"] == len(x)-10
    loss, report = p.differentiable_feedback_loss(program, logits, x, m, sampled,
        current_fidelity={**fidelity, "program_sha256": "0"*64})
    assert loss.item() == 0 and not report["external_program_gradient_enabled"]


def test_static_core_is_not_changed_by_failed_empirical_qualification():
    active = model()
    actor = frozen.FrozenWarehouseV621Actor.from_model(active)
    rows = parts()
    # Fit/calibration remain large; eligibility lacks native twenty-row support.
    rows[2] = [dict(rows[2][0], episode="small", frame=0, base_seed=200)]
    program = p.fit_shared_program("robot_1", actor, *rows)
    assert program.core_mask.tolist() == [1]
    assert program.feedback_leaves == program.explanation_leaves == frozenset()
    assert not program.report["validation"]["feedback_passed"]
    assert program.report["validation"]["feedback"]["core_rows"] == 90


def test_program_tensors_are_read_only_after_fit_and_loader():
    _, _, _, program = fit()
    for value in (program.core_mask, program.anchors, program.leaf_logits, program.structure["split_bias"]):
        assert not value.flags.writeable


def test_one_episode_cannot_switch_scenario_midway():
    actor = frozen.FrozenWarehouseV621Actor.from_model(model())
    rows = parts()
    rows[0][1]["base_seed"] = 999
    with pytest.raises(ValueError, match="omitted, duplicated or reordered"):
        p.fit_shared_program("robot_1", actor, *rows)


@pytest.mark.parametrize("bound", [.35, .50])
def test_box_KL_bound_for_all_physical_mask_patterns_and_residual_corners(bound):
    anchor = np.array([-.4, -.4, -.4, -.4, 1.6], np.float32)
    masks = np.repeat(np.array([[int(bool(bits & (1 << a))) for a in range(4)]+[1]
                               for bits in range(16)], np.float32), 32, axis=0)
    signs = np.tile(np.array([[1 if bits & (1 << a) else -1 for a in range(5)]
                             for bits in range(32)], float), (16, 1))
    targets = p._probs(anchor+bound*signs, masks).astype(np.float32)
    z, detail = p.fit_core_leaf_distribution(targets, masks, {4}, anchor)
    q = p._probs(z, masks)
    assert detail["proof"]["passed"]
    assert np.max(p._kl(targets, q)) <= (bound+.05)**2/2+1e-6
    assert np.max(p._kl(q, targets)) <= (bound+.05)**2/2+1e-6


def test_unbounded_negative_legal_logits_never_leak_probability_to_illegal_actions():
    logits = np.array([[-2e9, 0., -2e9-1., 0., -2e9-2.]], np.float64)
    masks = np.array([[1, 0, 1, 0, 1]], np.float32)
    result = p._probs(logits, masks)
    np.testing.assert_array_equal(result[0, [1, 3]], [0., 0.])
    assert result.sum() == pytest.approx(1.) and result.argmax() == 0
    expected = np.exp([0., -1., -2.]); expected /= expected.sum()
    np.testing.assert_allclose(result[0, [0, 2, 4]], expected)


@pytest.mark.parametrize('logits,masks', [
    ([np.nan, 0., 0., 0., 0.], [1, 1, 1, 1, 1]),
    ([np.inf, 0., 0., 0., 0.], [1, 1, 1, 1, 1]),
    ([0., 0., 0., 0., 0.], [0, 0, 0, 0, 0]),
    ([0., 0., 0., 0., 0.], [1, .5, 1, 1, 1])])
def test_program_probability_normalizer_rejects_invalid_logits_and_masks(logits, masks):
    with pytest.raises(ValueError):
        p._probs(logits, masks)


def test_program_probability_normalizer_rejects_unrepresentable_legal_range():
    with pytest.raises(RuntimeError, match='overflowed'):
        p._probs([1e308, -1e308, 0., 0., 0.], [1, 1, 1, 1, 1])


def test_feedback_normalizes_extreme_negative_legal_logits_before_masking():
    _, actor, rows, program = fit()
    fidelity = p.current_actor_fidelity(program, actor, rows[3])
    x, masks, sampled = p._arrays(rows[3], 'negative legal NN test')
    logits = program.leaf_logits[program.leaf_indices(x)].astype(np.float64)-2e9
    logits[masks == 0] = 0.
    value = torch.tensor(logits, requires_grad=True)
    loss, report = p.differentiable_feedback_loss(program, value, x, masks, sampled,
                                                 current_fidelity=fidelity)
    assert torch.isfinite(loss) and abs(float(loss.detach())) < 1e-12
    assert report['feedback_rows'] == len(rows[3])
    loss.backward()
    assert torch.isfinite(value.grad).all()
    assert value.grad[torch.from_numpy(masks == 0)].abs().sum() == 0


def test_live_noncore_anchor_drift_is_revalidated_while_core_anchor_drift_rejects():
    active = training.WarehouseV621Actor(core_residual_bound=.35)
    weight = np.zeros((1, p.FEATURE_DIM), np.float32)
    weight[0, p.FEATURE_NAMES.index('elapsed_fraction')] = 1
    structure = dict(root_ref=0, split_weights=weight, split_bias=np.array([-.5], np.float32),
                     left_refs=np.array([-1]), right_refs=np.array([-2]), leaf_count=2)
    for role in p.AGENTS:
        active.install_structure(role, structure, [1, 0], [4, -1], np.zeros((2, 5), np.float32))
    actor = frozen.FrozenWarehouseV621Actor.from_model(active)
    rows = parts()
    program = p.fit_shared_program('robot_1', actor, *rows)
    assert p.current_actor_fidelity(program, actor, rows[3])['passed']
    with torch.no_grad():
        active.actors['robot_1'].tree.leaf_logits[1, 0] += .01
    drifted = frozen.FrozenWarehouseV621Actor.from_model(active)
    assert drifted.role_sha256['robot_1'] != program.actor_role_sha256
    checked = p.current_actor_fidelity(program, drifted, rows[3])
    assert checked['current_actor_topology_matches_program'] and checked['passed']
    with pytest.raises(RuntimeError, match='different Actor'):
        p.program_metrics(program, drifted, rows[3])
    with torch.no_grad():
        active.actors['robot_1'].tree.extra_margin[0] += .1
    changed_core = frozen.FrozenWarehouseV621Actor.from_model(active)
    assert not p.current_actor_fidelity(program, changed_core, rows[3])['passed']


def test_empty_core_leaf_corrects_valid_actor_anchor_rounding_without_solver():
    anchor = np.array([-3.8417201042175293, -5.141720294952393, 3.3582801818847656,
                       1.1582801342010498, 4.466877460479736], np.float32)
    assert abs(float(anchor.astype(np.float64).sum())) > 2e-6
    assert np.isclose(anchor.mean(), 0., atol=2e-5)
    assert not p._box_proof(anchor, anchor, 4, {4})['passed']
    z, report = p.fit_core_leaf_distribution(np.zeros((0, 5)), np.zeros((0, 5)), {4}, anchor)
    assert report['success'] and report['float32_feasible'] and report['proof']['passed']
    assert report['empty_leaf'] and report['row_count'] == report['enumerated_action_solves'] == 0
    assert report['numeric_centering']['solver_calls'] == 0
    assert report['numeric_centering']['component_corrections']
    assert abs(float(z.astype(np.float64).sum())) <= 2e-6
    assert np.max(np.abs(z.astype(float)-anchor.astype(float))) <= .05
    assert z.argmax() == anchor.argmax() == 4
    calibrated, cal = p.calibrate_core_leaf_distribution(z, np.zeros((0, 5)),
        np.zeros((0, 5)), {4}, anchor)
    assert cal['empty_calibration_leaf'] and cal['proof']['passed']
    np.testing.assert_array_equal(calibrated, z)


def test_nonempty_fit_and_calibration_reprove_centering_in_original_anchor_box():
    anchor = np.array([-3.8417201042175293, -5.141720294952393, 3.3582801818847656,
                       1.1582801342010498, 4.466877460479736], np.float32)
    masks = np.ones((10, 5), np.float32)
    target = p._probs(anchor+[.01, -.02, .02, -.01, 0.], masks).astype(np.float32)
    z, report = p.fit_core_leaf_distribution(target, masks, {4}, anchor)
    assert report['proof']['passed'] and report['enumerated_action_solves'] == 1
    calibrated, cal = p.calibrate_core_leaf_distribution(z, target, masks, {4}, anchor)
    assert cal['proof']['passed']
    assert abs(float(calibrated.astype(float).sum())) <= 2e-6
    assert np.max(np.abs(calibrated.astype(float)-anchor.astype(float))) <= .05


def test_float32_centering_failure_is_typed_and_never_reported_success():
    # No stored value can change by <=.05 at this magnitude; this particular
    # anchor has sum 32, so a centered program is unrepresentable in the box.
    anchor = np.array([1e8, -1e8, 32., -16., 16.], np.float32)
    with pytest.raises(p.CoreBoxNumericalError) as info:
        p._center_float32_in_box(anchor, anchor, 0, {0})
    assert not info.value.report['success'] and not info.value.report['proof']['passed']
    assert info.value.report['proof']['preference']['passed']
    assert info.value.report['absolute_center_sum_limit'] == 2e-6
