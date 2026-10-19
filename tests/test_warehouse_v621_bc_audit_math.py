"""Synthetic arithmetic checks; no environment, native metric or model load."""
from copy import deepcopy
import ast
import inspect
import math

import numpy as np
import pytest

from experiments.warehouse import v621_bc_audit_math as audit


NAMES = ("legal_up", "legal_down", "legal_left", "legal_right", "legal_wait", "bounded")


def example():
    raw = {"structure": {"root_ref": 0, "split_weights": [[1, 0, 0, 0, 0, 0]],
        "split_bias": [-.5], "left_refs": [-1], "right_refs": [-2], "leaf_count": 2},
        "leaf_logits": [[0, 0, 0, 0, 2], [2, 0, 0, 0, 0]], "core_mask": [0, 1],
        "residual_bound": .35, "feedback_leaves": [1], "explanation_leaves": [0, 1]}
    masks = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1]])
    x = np.column_stack((masks, [.1, .2, .3, .4]))
    p = np.array([[.7, 0, 0, 0, .3], [.7, 0, 0, 0, .3], [0, 0, 0, 0, 1], [.2, 0, 0, 0, .8]])
    return dict(raw_program=raw, encoded=x, masks=masks, nn_p=p, sampled_indices=np.array([0, 4, 4, 0]),
        episodes=["a", "a", "b", "c"], scenarios=[10, 10, 20, 30],
        feature_names=NAMES, feature_bounds={"bounded": (0, 1)}, shared=True)


def kl_binary(p, q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def test_all_physical_denominators_sampled_and_expected_are_different():
    result = audit.audit_program(**example())
    f, e, c = result["feedback"], result["explanation"], result["context"]
    assert c["leaf_indices"].tolist() == [1, 1, 0, 1]
    assert c["path_certificates"] == ((4,), (0, 4))
    assert f["sample_count"] == 4 and f["excluded_frames"] == 0
    assert f["rows"] == 3 and f["coverage"] == .75
    assert f["actionable_rows"] == 1 and f["actionable_coverage"] == .25
    assert f["expected_actionable_rows"] == pytest.approx(1.4)
    assert f["expected_actionable_coverage"] == pytest.approx(.35)
    assert e["actionable_rows"] == 3 and e["actionable_coverage"] == .75
    assert e["expected_actionable_coverage"] == pytest.approx(.65)
    assert f["agreement"] == pytest.approx(2/3)
    assert e["agreement"] == .75
    assert f["distinct_episode_count"] == 2 and f["distinct_scenario_seed_count"] == 2
    assert f["max_episode_fraction"] == pytest.approx(2/3)
    assert f["per_nn_action"]["UP"] == {"physical_frames": 2, "rows": 2,
        "qualified_coverage": 1., "actual_actionable_rows": 1}
    assert f["per_nn_action"]["WAIT"]["qualified_coverage"] == .5
    assert f["per_nn_action"]["DOWN"]["qualified_coverage"] is None
    assert f["whole_tree_agreement"] == .75


def test_KL_direction_is_nn_to_program_not_training_loss_direction():
    data = example()
    result = audit.audit_program(**data)
    q = math.exp(2)/(1+math.exp(2))
    expected = (2*kl_binary(.7, q)+kl_binary(.2, q))/3
    assert result["feedback"]["kl"] == pytest.approx(expected, abs=1e-14)
    reverse = (2*kl_binary(q, .7)+kl_binary(q, .2))/3
    assert abs(expected-reverse) > .01
    assert result["feedback"]["whole_tree_kl"] == pytest.approx(expected*.75)
    assert result["feedback"]["core_KL_upper_bound"] == pytest.approx(.08)
    assert result["feedback"]["core_actual_max_kl"] == pytest.approx(kl_binary(.2, q))
    assert result["feedback"]["core_KL_bound_violations"] == sum(kl_binary(p, q) > .08+5e-6 for p in (.7, .7, .2))


def test_wilson_uses_selected_frame_count_and_real_agreement():
    result = audit.audit_program(**example())["feedback"]
    z, p, n = 1.645, 2/3, 3
    expected = (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)
    assert result["agreement_wilson_lower_90"] == pytest.approx(expected)


def test_qualification_does_not_hide_whole_tree_illegal_or_unproven_rows():
    data = example()
    data["raw_program"]["leaf_logits"][0] = [2, 0, 0, 0, 0]
    result = audit.audit_program(**data)
    f, e = result["feedback"], result["explanation"]
    assert f["illegal_count"] == 0 and f["path_uncertified_count"] == 0
    assert f["whole_tree_raw_illegal_count"] == 1
    assert f["whole_tree_path_uncertified_count"] == 1
    assert e["illegal_count"] == 1 and e["path_uncertified_count"] == 1
    assert e["masked_illegal_count"] == 0
    assert result["context"]["raw_top"][2] == 0
    assert result["context"]["program_top"][2] == 4


def test_empty_qualified_set_preserves_every_frame_and_none_KL():
    data = example()
    data["raw_program"]["feedback_leaves"] = []
    f = audit.audit_program(**data)["feedback"]
    assert f["sample_count"] == 4 and f["rows"] == 0
    assert f["kl"] is None and f["agreement"] == 0 and f["agreement_wilson_lower_90"] == 0
    assert f["actionable_coverage"] == 0 and f["expected_actionable_coverage"] == 0
    assert f["core_rows"] == 3  # core bound still covers non-qualified physical rows


def test_interval_proof_never_replaces_shared_explicit_guard_contract():
    data = example()
    data["raw_program"]["structure"]["split_weights"][0][-1] = 1
    data["feature_bounds"] = {"bounded": (0, .5)}
    c = audit.audit_program(**data)["context"]
    assert 0 in c["interval_path_certificates"][1]
    assert 0 not in c["path_certificates"][1]
    assert not c["path_certified"][[0, 1, 3]].any()
    data["shared"] = False
    assert 0 not in audit.audit_program(**data)["context"]["path_certificates"][1]


def test_interval_proof_requires_bounds_for_every_other_used_feature():
    data = example()
    data["raw_program"]["structure"]["split_weights"][0][-1] = 1
    data["feature_bounds"] = {}
    assert 0 not in audit.audit_program(**data)["context"]["interval_path_certificates"][1]


def test_negative_mask_guard_certifies_the_left_path():
    data = example()
    tree = data["raw_program"]["structure"]
    tree["split_weights"][0][0], tree["split_bias"][0] = -1, .5
    data["raw_program"]["leaf_logits"].reverse()
    data["raw_program"]["feedback_leaves"] = [0]
    assert audit.audit_program(**data)["context"]["path_certificates"] == ((0, 4), (4,))


def test_left_tie_and_batch_independence_from_stored_float32():
    data = example()
    tree = data["raw_program"]["structure"]
    tree["split_weights"] = [[0, 0, 0, 0, 0, 1]]
    tree["split_bias"] = [0]
    data["encoded"][:, -1] = [0, 1e-10, np.nextafter(np.float32(1e-10), np.float32(0)),
                                     np.nextafter(np.float32(1e-10), np.float32(np.inf))]
    result = audit.audit_program(**data)
    expected = (data["encoded"][:, -1].astype(np.float32).astype(float) > 1e-10).astype(int)
    assert np.array_equal(result["context"]["leaf_indices"], expected)
    for i in range(4):
        one = dict(data)
        for field in ("encoded", "masks", "nn_p", "sampled_indices", "episodes", "scenarios"):
            one[field] = np.asarray(data[field])[i:i+1]
        assert audit.audit_program(**one)["context"]["leaf_indices"][0] == expected[i]


@pytest.mark.parametrize("shared,epsilon", [(True, 1e-300), (False, 1e-8)])
def test_underflow_masked_KL_floors_and_illegal_probability_zero(shared, epsilon):
    data = example()
    data["shared"] = shared
    data["raw_program"]["leaf_logits"][1] = [1000, 1e9, 0, 0, -1000]
    result = audit.audit_program(**data)
    q = result["context"]["program_probabilities"]
    assert np.all(q[np.asarray(data["masks"]) == 0] == 0)
    expected = .7*math.log(.7)+.3*(math.log(.3)-math.log(epsilon))
    assert result["context"]["row_kl"][0] == pytest.approx(expected)
    assert np.isfinite(result["context"]["row_kl"]).all()


def test_control_common_metrics_use_legacy_floor_and_no_shared_bounds():
    data = example()
    data["shared"] = False
    data["raw_program"] = {**data["raw_program"].pop("structure"), **data["raw_program"]}
    result = audit.audit_program(**data)
    assert result["feedback"]["coverage"] == .75
    assert result["feedback"]["actionable_coverage"] == .25
    assert "core_KL_upper_bound" not in result["feedback"]
    assert result["feedback"]["certified_rows"] == 0  # historical control field
    assert result["feedback"]["per_nn_action"]["UP"]["expected_actionable_rows"] == pytest.approx(1.4)


def test_serialized_registered_float32_bound_is_not_rounded_to_nominal():
    data = example()
    actual = float(np.float32(.35))
    assert actual != .35
    data["raw_program"]["residual_bound"] = actual
    bound = audit.audit_program(**data)["feedback"]["core_KL_upper_bound"]
    assert bound == (actual+.05)**2/2
    assert bound != (.35+.05)**2/2
    data["raw_program"]["residual_bound"] = float(np.nextafter(actual, np.inf))
    with pytest.raises(ValueError, match="registered residual bound"):
        audit.audit_program(**data)


@pytest.mark.parametrize("change", ["mask_not_binary", "wait_illegal", "encoded_mask_mismatch", "illegal_p",
    "not_normalized", "nan_p", "illegal_sample", "short_episode", "outside_bound", "nan_logits",
    "duplicate_selected", "bad_ref", "too_many_sources", "wrong_core", "unregistered_b"])
def test_malformed_physical_or_program_evidence_fails_closed(change):
    data = example()
    raw, tree = data["raw_program"], data["raw_program"]["structure"]
    if change == "mask_not_binary": data["masks"][0, 0] = 2
    elif change == "wait_illegal": data["masks"][0, 4] = 0
    elif change == "encoded_mask_mismatch": data["encoded"][0, 0] = 0
    elif change == "illegal_p": data["nn_p"][0, 1] = .1
    elif change == "not_normalized": data["nn_p"][0, 0] = .5
    elif change == "nan_p": data["nn_p"][0, 0] = np.nan
    elif change == "illegal_sample": data["sampled_indices"][0] = 1
    elif change == "short_episode": data["episodes"] = ["a"]
    elif change == "outside_bound": data["encoded"][0, -1] = 2
    elif change == "nan_logits": raw["leaf_logits"][0][0] = np.nan
    elif change == "duplicate_selected": raw["feedback_leaves"] = [1, 1]
    elif change == "bad_ref": tree["left_refs"] = [0]
    elif change == "too_many_sources": tree["split_weights"][0][:4] = [1]*4
    elif change == "wrong_core": raw["core_mask"] = [0, 2]
    elif change == "unregistered_b": raw["residual_bound"] = .2
    with pytest.raises(ValueError): audit.audit_program(**data)


def test_hard_risk_context_counts_two_sources():
    data = example()
    data["feature_names"] = (*NAMES[:-1], "public_hard_risk_context")
    data["feature_bounds"] = {}
    data["raw_program"]["structure"]["split_weights"] = [[1, 1, 0, 0, 0, 1]]
    with pytest.raises(ValueError, match="three source"):
        audit.audit_program(**data)


def test_single_leaf_no_nodes_and_input_mapping_not_modified():
    data = example()
    data["raw_program"] = {"structure": {"root_ref": -1, "split_weights": [], "split_bias": [],
        "left_refs": [], "right_refs": [], "leaf_count": 1}, "leaf_logits": [[0, 0, 0, 0, 2]],
        "core_mask": [1], "residual_bound": .5, "feedback_leaves": [0], "explanation_leaves": [0]}
    original = deepcopy(data["raw_program"])
    result = audit.audit_program(**data)
    assert result["context"]["path_certificates"] == ((4,),)
    assert np.all(result["context"]["leaf_indices"] == 0)
    assert result["feedback"]["core_KL_upper_bound"] == pytest.approx(.15125)
    assert data["raw_program"] == original


def test_only_numpy_and_standard_library_imports_no_production_helpers():
    parsed = ast.parse(inspect.getsource(audit))
    modules = {node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(parsed) if isinstance(node, ast.Import) for alias in node.names}
    assert modules <= {"__future__", "collections", "collections.abc", "math", "numpy"}
    assert "certified_mask_branches" not in inspect.getsource(audit)
