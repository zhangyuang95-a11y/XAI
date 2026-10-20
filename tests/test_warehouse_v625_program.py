"""Scientific language/certificate boundaries, without environment rollouts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_v625_program as p


def fixture(n=41):
    x = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    m = np.tile([1, 1, 0, 0, 1], (n, 1)).astype(np.float32)
    z = np.column_stack((2 - 4 * x[:, 0], -2 + 4 * x[:, 0], np.full(n, -10), np.full(n, -10), np.full(n, -4))).astype(np.float32)
    return x, m, p.masked_probabilities(z, m)


def leaf(bias=(0, 0, -4, -4, -2), weights=((-2,), (2,), (0,), (0,), (0,)), witness=0):
    return {"version": p.LEAF_VERSION, "arithmetic": p.ARITHMETIC,
        "feature_names": ["self_row"], "support": [0], "support_names": ["self_row"],
        "bias": np.asarray(bias, np.float32).tolist(), "weights": np.asarray(weights, np.float32).tolist(),
        "witness": witness, "mode": "P1"}


class Budget:
    def __init__(self):
        self.debits, self.settlements = [], []
    def consume(self):
        token = len(self.debits) + 1
        self.debits.append(token)
        return token
    def finish(self, token, *, success, detail):
        self.settlements.append((token, success, detail))
    def check(self):
        pass


def test_linear_leaf_fits_variable_action_distribution_while_constant_cannot():
    x, m, target = fixture()
    budget = Budget()
    constant = p.fit_leaf(x, m, target, ("self_row",), {0, 1, 4}, mode="P0", budget=budget)
    linear = p.fit_leaf(x, m, target, ("self_row",), {0, 1, 4}, support=[0], budget=budget)
    assert linear["certificate"]["passed"]
    assert linear["report"]["candidate_reports"][0]["full_distribution_forward_kl"] < .01
    assert linear["report"]["candidate_reports"][0]["full_distribution_forward_kl"] < constant["report"]["candidate_reports"][0]["full_distribution_forward_kl"] / 10
    raw, q = p.evaluate_leaf(linear, x, m)
    assert set(raw.argmax(-1)) == {0, 1}
    assert np.allclose(q.sum(-1), 1)
    assert np.all(q[:, (2, 3)] == 0)
    assert budget.debits == [1, 2]
    assert [s[0] for s in budget.settlements] == [1, 2]


def test_auto_support_is_bounded_and_constant_incumbent_is_included():
    x, m, target = fixture()
    model = p.fit_leaf(x, m, target, ("self_row",), {0, 1, 4})
    assert model["report"]["full_fit_calls"] == 2
    assert model["report"]["candidate_reports"][0]["support"] == []
    assert model["report"]["full_fit_calls"] <= p.MAX_FULL_FITS


def test_certificate_allows_multiple_path_legal_top_actions():
    model = leaf(bias=(2, -2, -10, -10, -4), weights=((-4,), (4,), (0,), (0,), (0,)))
    certificate = p.certify_leaf(model, ("self_row",), {0, 1, 4})
    assert certificate["passed"]
    assert {c["action"] for c in certificate["illegal_action_checks"]} == {2, 3}
    assert certificate["corner_count"] == 2


def test_off_data_corner_violation_is_rejected():
    model = leaf(bias=(0, -2, -1, -4, -3), weights=((0,), (0,), (2,), (0,), (0,)))
    proof = p.certify_leaf(model, ("self_row",), {0, 1, 4})
    assert not proof["passed"]
    # x=0 alone would have looked safe; x=1 must also be certified.
    assert next(c for c in proof["illegal_action_checks"] if c["action"] == 2)["minimum_real_gap"] == -1


def test_illegal_argmax_tie_and_near_margin_are_rejected():
    model = leaf(bias=(0, -2, 0, -4, -3), weights=((0,),) * 5)
    assert not p.certify_leaf(model, ("self_row",), {0, 1, 4})["passed"]
    model["bias"][2] = float(np.float32(-.001))
    assert not p.certify_leaf(model, ("self_row",), {0, 1, 4})["passed"]


def test_float32_arithmetic_matches_explicit_order():
    x, m, _ = fixture()
    model = leaf()
    raw, _ = p.evaluate_leaf(model, x, m)
    expected = np.asarray(model["bias"], np.float32)[None] + x * np.asarray(model["weights"], np.float32)[:, 0][None]
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(raw[:3], p.evaluate_leaf(model, x[:3], m[:3])[0])


@pytest.mark.parametrize("support,names", [([0], ("unbounded_unknown",)), ([0, 1, 2, 3], ("self_row", "self_column", "partner_row", "partner_column")), ([0, 1, 2], ("public_hard_risk_context", "self_row", "self_column"))])
def test_unbounded_or_over_budget_supports_rejected(support, names):
    x = np.zeros((3, len(names)), np.float32)
    m = np.ones((3, 5), np.float32)
    with pytest.raises(ValueError):
        p.fit_leaf(x, m, m / 5, names, {4}, support=support)


def test_inference_domain_violation_rejected():
    x, m, _ = fixture()
    x[0, 0] = 1.01
    with pytest.raises(ValueError, match="domain"):
        p.evaluate_leaf(leaf(), x, m)


def test_legal_action_proof_conflicting_with_observed_mask_rejected():
    x, m, target = fixture()
    with pytest.raises(ValueError, match="conflicts"):
        p.fit_leaf(x, m, target, ("self_row",), {0, 2, 4}, support=[])


def test_witness_not_path_legal_rejected():
    with pytest.raises(ValueError, match="witness"):
        p.certify_leaf(leaf(witness=2), ("self_row",), {0, 1, 4})


def test_non_float32_serialized_coefficients_rejected():
    model = leaf()
    model["bias"][0] = .12345678912345
    with pytest.raises(ValueError, match="float32"):
        p.certify_leaf(model, ("self_row",), {0, 1, 4})


def test_calibration_reproves_and_freezes_support_and_witness():
    x, m, target = fixture()
    model = p.fit_leaf(x, m, target, ("self_row",), {0, 1, 4}, support=[0])
    calibrated, report = p.calibrate_leaf(model, x, m, target, ("self_row",), {0, 1, 4})
    assert calibrated["support"] == model["support"]
    assert calibrated["witness"] == model["witness"]
    assert calibrated["certificate"] == p.certify_leaf(calibrated, ("self_row",), {0, 1, 4})
    assert not report["support_reselected"]
    assert report["certificate_recomputed_after_float32_scaling"]
    assert report["pre_calibration_leaf"] == model


def test_empty_leaf_constant_is_explicitly_ineligible():
    model = p.fit_leaf(np.empty((0, 1), np.float32), np.empty((0, 5)), np.empty((0, 5)), ("self_row",), {4}, support=[])
    assert model["report"]["empty_leaf"]
    assert not model["report"]["eligible_for_qualification"]
    assert model["certificate"]["passed"]


def test_budget_consumed_before_optimizer_and_exception_settled(monkeypatch):
    x, m, target = fixture()
    budget = Budget()
    def fail(*args, **kwargs):
        assert budget.debits == [1]
        raise RuntimeError("engineering failure")
    monkeypatch.setattr(p, "minimize", fail)
    with pytest.raises(RuntimeError, match="engineering"):
        p.fit_leaf(x, m, target, ("self_row",), {4}, support=[], budget=budget)
    assert len(budget.settlements) == 1
    assert budget.settlements[0][1] is False


def test_final_predebit_checkpoint_is_small_durable_and_single_use(tmp_path):
    path = tmp_path / "final.json"
    budget = p._FinalBudget({"maximum_leaf_api_calls": 32, "prior_leaf_optimization_calls": 19968,
        "max_total_leaf_optimization_calls": 20000}, path)
    token = budget.consume()
    assert json.loads(path.read_text())["inflight"] == token
    budget.finish(token, success=True, detail={"x": 1})
    saved = json.loads(path.read_text())
    assert saved["inflight"] is None and saved["calls"] == 1
    assert path.stat().st_size < 1000
    with pytest.raises(ValueError, match="single-use"):
        p._FinalBudget({}, path)
    with pytest.raises(RuntimeError, match="twice"):
        budget.finish(token, success=True, detail={})


def test_final_budget_rejects_prior_plus_reserve_overflow():
    with pytest.raises(ValueError, match="shared budget"):
        p._FinalBudget({"prior_leaf_optimization_calls": 19969, "maximum_leaf_api_calls": 32})


def test_json_roundtrip_preserves_leaf_certificate_exactly():
    model = leaf()
    before = p.certify_leaf(model, ("self_row",), {0, 1, 4})
    after = p.certify_leaf(json.loads(json.dumps(model)), ("self_row",), {0, 1, 4})
    assert before == after


def test_p0_cannot_hide_variable_logit_leaf():
    x, m, target = fixture()
    with pytest.raises(ValueError, match="constant"):
        p.fit_leaf(x, m, target, ("self_row",), {0, 1, 4}, mode="P0", support=[0])


def complete_fixture(monkeypatch):
    from backend.training.warehouse_v6_actor import MODEL_VERSION, FEATURE_DIM
    names = tuple(p.FEATURE_NAMES)
    logits = np.asarray([-3, -3, -3, -3, 3], np.float32)
    role = SimpleNamespace(role_sha256="a" * 64, model_version=MODEL_VERSION,
        feature_dim=FEATURE_DIM, feature_names=names,
        logits_encoded=lambda x: np.broadcast_to(logits, (len(x), 5)).copy())
    actor = SimpleNamespace(roles={"robot_1": role})
    def arrays(rows):
        x = np.zeros((len(rows), len(names)), np.float32)
        m = np.ones((len(rows), 5), np.float32)
        for i, name in enumerate(names):
            if name.startswith("legal_"):
                x[:, i] = 1
        return x, m, np.full(len(rows), 4, np.int64)
    monkeypatch.setattr(p, "_arrays", arrays)
    parts = [[{"episode": f"{name}-{i // 30}", "base_seed": f"{name}-{i // 30}"} for i in range(90)]
             for name in p.SPLITS]
    tree = {"root_ref": -1, "split_weights": np.empty((0, len(names)), np.float32),
        "split_bias": np.empty(0, np.float32), "left_refs": [], "right_refs": [],
        "leaf_count": 1, "feature_dim": len(names)}
    return actor, parts, tree


def test_complete_program_single_final_fit_qualification_and_hash_roundtrip(monkeypatch, tmp_path):
    actor, parts, tree = complete_fixture(monkeypatch)
    program = p.fit_program("robot_1", actor, *parts, structure=tree, mode="P0",
        fit_budget={"maximum_leaf_api_calls": 32}, checkpoint_path=tmp_path / "fit.json")
    assert program.feedback_leaves == frozenset({0})
    assert program.explanation_leaves == frozenset({0})
    assert program.report["leaf_optimization_calls"] == 1
    assert program.report["leaf_optimization_calls_outside_search"] == 1
    metric = p.program_metrics(program, actor, parts[-1])
    assert metric["gate"]["passed"]
    assert metric["feedback"]["physical_frame_count"] == 90
    assert "low_battery_le_20_percent" in metric["feedback"]["public_strata"]
    path = tmp_path / "program.json"
    sha = program.save(path)
    loaded = p.WarehouseV625Program.load(path, expected_sha256=sha, actor_role_sha256="a" * 64)
    assert loaded.sha256() == program.sha256()
    with pytest.raises(ValueError, match="hash"):
        p.WarehouseV625Program.load(path, expected_sha256="0" * 64, actor_role_sha256="a" * 64)
    with pytest.raises(ValueError, match="hash"):
        p.WarehouseV625Program.load(path, expected_sha256=sha, actor_role_sha256="b" * 64)


def test_final_partition_overlap_rejected_before_solves(monkeypatch):
    actor, parts, tree = complete_fixture(monkeypatch)
    parts[-1][0] = deepcopy(parts[0][0])
    budget = Budget()
    with pytest.raises(ValueError, match="overlap"):
        p.fit_program("robot_1", actor, *parts, structure=tree, mode="P0", fit_budget=budget)
    assert not budget.debits


def test_final_p1_requires_fit_selected_supports(monkeypatch):
    actor, parts, tree = complete_fixture(monkeypatch)
    with pytest.raises(ValueError, match="search-selected"):
        p.fit_program("robot_1", actor, *parts, structure=tree, mode="P1")


def test_final_p1_uses_one_fixed_support_and_witness_trial(monkeypatch):
    actor, parts, tree = complete_fixture(monkeypatch)
    x, m, _ = p._arrays(parts[0])
    target = p.masked_probabilities(actor.roles["robot_1"].logits_encoded(x), m)
    initial = p.fit_leaf(x, m, target, p.FEATURE_NAMES, {4}, support=[p.FEATURE_NAMES.index("self_row")])
    budget = Budget()
    program = p.fit_program("robot_1", actor, *parts, structure=tree, mode="P1",
        initial_leaves=[initial], fit_budget=budget)
    assert len(budget.debits) == len(budget.settlements) == 1
    assert program.leaves[0]["support"] == initial["support"]
    assert program.leaves[0]["witness"] == initial["witness"]
    assert program.report["leaf_fits"]["0"]["full_fit_calls"] == 1


def test_mask_feature_mismatch_rejected_before_budget():
    budget = Budget()
    names = ("legal_up",)
    with pytest.raises(ValueError, match="legal bit"):
        p.fit_leaf(np.zeros((2, 1)), np.ones((2, 5)), np.full((2, 5), .2), names, {4}, support=[], budget=budget)
    assert budget.debits == []


def test_leaf_full_feature_schema_mismatch_rejected():
    with pytest.raises(ValueError, match="full feature schema"):
        p.certify_leaf(leaf(), ("self_column",), {0, 1, 4})
