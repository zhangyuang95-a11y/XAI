from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    R41DiagnosticBoostedTreeProgram,
    export_hist_gradient_boosting_classifier,
)


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1247)
    observations = rng.normal(size=(640, 5))
    score = (2.2 * observations[:, 0] - 1.1 * observations[:, 1]
             + 0.7 * observations[:, 2] * observations[:, 3]
             - 0.25 * observations[:, 4])
    labels = np.where(score > 0.15, "advance", "yield")
    return observations, labels


def _binary_model(*, learning_rate: float = 0.19, max_iter: int = 7):
    observations, labels = _binary_data()
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=7,
        min_samples_leaf=8,
        l2_regularization=0.03,
        early_stopping=False,
        random_state=41,
    ).fit(observations, labels)
    return model, observations


def _multiclass_model():
    rng = np.random.default_rng(90210)
    observations = rng.normal(size=(720, 6))
    scores = np.column_stack((
        1.8 * observations[:, 0] - 0.4 * observations[:, 3],
        -0.7 * observations[:, 0] + 1.6 * observations[:, 1]
        + 0.2 * observations[:, 4],
        -1.2 * observations[:, 1] + 1.5 * observations[:, 2]
        - 0.3 * observations[:, 5],
    ))
    labels = np.asarray(("dock", "queue", "reroute"))[np.argmax(scores, axis=1)]
    model = HistGradientBoostingClassifier(
        learning_rate=0.13,
        max_iter=6,
        max_leaf_nodes=8,
        min_samples_leaf=7,
        early_stopping=False,
        random_state=7,
    ).fit(observations, labels)
    return model, observations


def _threshold_probes(program, seed_rows: np.ndarray) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float64).copy() for row in seed_rows]
    for tree in program.to_dict()["trees"]:
        for node in tree["nodes"]:
            if node["kind"] != "split":
                continue
            feature = node["feature_index"]
            threshold = node["threshold"]
            for value in (np.nextafter(threshold, -np.inf), threshold,
                          np.nextafter(threshold, np.inf)):
                row = np.zeros(len(program.feature_names), dtype=np.float64)
                row[feature] = value
                rows.append(row)
    rows.extend((
        np.full(len(program.feature_names), -1e100, dtype=np.float64),
        np.full(len(program.feature_names), 1e100, dtype=np.float64),
    ))
    return np.asarray(rows, dtype=np.float64)


def _reachable_threshold_probes(program):
    """Construct threshold probes that are guaranteed to reach each node."""

    result = []
    for tree_index, tree in enumerate(program.to_dict()["trees"]):
        nodes = tree["nodes"]
        parents = {}
        for parent_index, node in enumerate(nodes):
            if node["kind"] == "split":
                parents[node["left"]] = (parent_index, True)
                parents[node["right"]] = (parent_index, False)
        for node_index, target in enumerate(nodes):
            if target["kind"] != "split":
                continue
            constraints = {}
            child = node_index
            while child:
                parent_index, took_left = parents[child]
                parent = nodes[parent_index]
                lower, upper = constraints.get(
                    parent["feature_index"], (-np.inf, np.inf))
                if took_left:
                    upper = min(upper, parent["threshold"])
                else:
                    lower = max(lower, parent["threshold"])
                constraints[parent["feature_index"]] = (lower, upper)
                child = parent_index

            base = np.zeros(len(program.feature_names), dtype=np.float64)
            for feature, (lower, upper) in constraints.items():
                base[feature] = (upper if np.isfinite(upper)
                                 else np.nextafter(lower, np.inf))
            feature = target["feature_index"]
            lower, upper = constraints.get(feature, (-np.inf, np.inf))
            for observed in (np.nextafter(target["threshold"], -np.inf),
                             target["threshold"],
                             np.nextafter(target["threshold"], np.inf)):
                if not lower < observed <= upper:
                    continue
                row = base.copy()
                row[feature] = observed
                result.append((row, tree_index, node_index, observed,
                               target["threshold"]))
    return result


def test_tree_predictor_leaf_values_already_include_learning_rate():
    unit, observations = _binary_model(learning_rate=1.0, max_iter=1)
    shrunk, _ = _binary_model(learning_rate=0.2, max_iter=1)
    unit_nodes = unit._predictors[0][0].nodes
    shrunk_nodes = shrunk._predictors[0][0].nodes
    unit_leaves = unit_nodes[unit_nodes["is_leaf"].astype(bool)]["value"]
    shrunk_leaves = shrunk_nodes[shrunk_nodes["is_leaf"].astype(bool)]["value"]

    assert np.array_equal(unit_nodes["is_leaf"], shrunk_nodes["is_leaf"])
    assert np.array_equal(unit_nodes["feature_idx"], shrunk_nodes["feature_idx"])
    assert np.allclose(shrunk_leaves, 0.2 * unit_leaves, rtol=2e-14, atol=2e-14)

    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        shrunk, feature_names=[f"public.axis_{index}" for index in range(5)])
    assert program.to_dict()["leaf_value_semantics"] == LEAF_VALUE_SEMANTICS
    expected = shrunk.decision_function(observations[:80])
    # No second learning-rate multiplication is needed: baseline plus exported
    # TreePredictor leaf values is sklearn's raw decision exactly.
    assert np.allclose(program.decision_function_batch(observations[:80]), expected,
                       rtol=0.0, atol=0.0)


def test_binary_numpy_runtime_matches_sklearn_on_finite_and_threshold_inputs():
    model, observations = _binary_model()
    program = export_hist_gradient_boosting_classifier(
        model,
        feature_names=[f"public.axis_{index}" for index in range(observations.shape[1])],
        action_names=["ADVANCE", "YIELD"],
        metadata={"purpose": "public diagnostic trace", "seed": np.int64(41)},
    )
    probes = _threshold_probes(program, observations[:37])

    expected_raw = model.decision_function(probes)
    raw_scores = program.raw_scores_batch(probes)
    raw_logits = program.raw_logits_batch(probes)
    assert raw_scores.shape == (len(probes), 1)
    assert np.allclose(raw_scores[:, 0], expected_raw, rtol=0.0, atol=2e-15)
    assert np.array_equal(raw_logits[:, 0], np.zeros(len(probes)))
    assert np.allclose(raw_logits[:, 1], expected_raw, rtol=0.0, atol=2e-15)
    assert np.allclose(program.softmax_batch(probes), model.predict_proba(probes),
                       rtol=2e-14, atol=2e-15)
    assert np.array_equal(program.predict_batch(probes), model.predict(probes))

    restored = R41DiagnosticBoostedTreeProgram.from_json(program.to_json())
    assert restored.to_dict() == json.loads(program.to_json())
    assert np.array_equal(restored.raw_logits_batch(probes), raw_logits)
    assert np.array_equal(restored.predict_batch(probes), model.predict(probes))


def test_multiclass_raw_logits_softmax_predictions_and_trace_are_additive():
    model, observations = _multiclass_model()
    names = [f"public.sensor_{index}" for index in range(observations.shape[1])]
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=names, action_names=["DOCK", "QUEUE", "REROUTE"])
    probes = _threshold_probes(program, observations[:31])

    expected_raw = model.decision_function(probes)
    assert np.allclose(program.raw_logits_batch(probes), expected_raw,
                       rtol=0.0, atol=4e-15)
    assert np.allclose(program.predict_proba_batch(probes), model.predict_proba(probes),
                       rtol=2e-14, atol=2e-15)
    assert np.array_equal(program.predict_batch(probes), model.predict(probes))

    row = probes[9]
    prediction, trace = program.predict_with_trace(dict(zip(names, row)))
    assert prediction == model.predict(row[None, :])[0]
    assert len(trace["trees"]) == model.n_iter_ * len(model.classes_)
    reconstructed = np.asarray(trace["baseline"], dtype=np.float64)
    for tree_trace in trace["trees"]:
        reconstructed[tree_trace["output_index"]] += tree_trace["contribution"]
        assert tree_trace["contribution_logits"][tree_trace["class_index"]] == (
            tree_trace["contribution"])
        for step in tree_trace["path"]:
            assert step["feature_name"] == names[step["feature_index"]]
            assert step["went_left"] == (step["observed"] <= step["threshold"])
    assert np.array_equal(reconstructed, np.asarray(trace["raw_scores"]))
    assert np.array_equal(np.asarray(trace["raw_logits"]),
                          program.raw_logits(row))
    assert np.allclose(trace["probabilities"], program.predict_proba(row),
                       rtol=0.0, atol=0.0)
    probabilities, probability_trace = program.predict_proba_with_trace(row)
    assert np.array_equal(probabilities, program.predict_proba(row))
    assert probability_trace == trace


def test_exact_threshold_goes_left_and_path_reports_public_predicate():
    model, _ = _binary_model(max_iter=2)
    names = [f"public.axis_{index}" for index in range(5)]
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=names)
    root = program.to_dict()["trees"][0]["nodes"][0]
    assert root["kind"] == "split"
    row = np.zeros(5, dtype=np.float64)
    row[root["feature_index"]] = root["threshold"]
    trace = program.trace(row)
    first_step = trace["trees"][0]["path"][0]
    assert first_step == {
        "node_index": 0,
        "feature_index": root["feature_index"],
        "feature_name": names[root["feature_index"]],
        "observed": root["threshold"],
        "operator": "<=",
        "threshold": root["threshold"],
        "went_left": True,
        "next_node": root["left"],
    }
    assert program.predict(row) == model.predict(row[None, :])[0]


def test_every_exported_split_is_exercised_at_its_reachable_float_boundaries():
    model, _ = _binary_model(max_iter=5)
    names = [f"public.axis_{index}" for index in range(5)]
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=names)
    probes = _reachable_threshold_probes(program)
    expected_split_count = sum(
        node["kind"] == "split" for tree in program.to_dict()["trees"]
        for node in tree["nodes"])
    assert len({(tree_index, node_index) for _, tree_index, node_index, _, _
                in probes}) == expected_split_count

    rows = np.asarray([row for row, *_ in probes])
    assert np.array_equal(program.predict_batch(rows), model.predict(rows))
    assert np.allclose(program.decision_function_batch(rows),
                       model.decision_function(rows), rtol=0.0, atol=0.0)
    for row, tree_index, node_index, observed, threshold in probes:
        path = program.trace(row)["trees"][tree_index]["path"]
        step = next(item for item in path if item["node_index"] == node_index)
        assert step["observed"] == observed
        assert step["went_left"] == (observed <= threshold)


def test_only_exact_finite_public_inputs_and_numeric_axis_trees_are_accepted():
    model, observations = _binary_model(max_iter=2)
    names = [f"public.axis_{index}" for index in range(5)]
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=names)
    good = dict(zip(names, observations[0]))
    assert program.predict(good) == model.predict(observations[:1])[0]

    with pytest.raises(ValueError, match="exactly its public features"):
        program.predict({**good, "hidden_state": 2.0})
    with pytest.raises(ValueError, match="exactly its public features"):
        program.predict({**good, "actor_logits": 2.0})
    with pytest.raises(ValueError, match="must be finite"):
        program.predict(np.asarray([np.nan, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="must be finite"):
        program.predict_batch(np.asarray([[np.inf, 0.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="hidden/logit"):
        R41DiagnosticBoostedTreeProgram.from_sklearn(
            model, feature_names=["public.0", "public.1", "hidden", "public.3",
                                  "public.4"])

    payload = program.to_dict()
    payload["feature_names"][0] = "actor_logits"
    with pytest.raises(ValueError, match="hidden/logit"):
        R41DiagnosticBoostedTreeProgram.from_dict(payload)
    payload = program.to_dict()
    payload["leaf_value_semantics"] = "multiply_by_learning_rate_again"
    with pytest.raises(ValueError, match="leaf-value semantics"):
        R41DiagnosticBoostedTreeProgram.from_dict(payload)
    payload = program.to_dict()
    payload["trees"][0]["nodes"][0]["threshold"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        R41DiagnosticBoostedTreeProgram.from_dict(payload)
    payload = program.to_dict()
    payload["classes"] = [True, 1]
    with pytest.raises(ValueError, match="class labels"):
        R41DiagnosticBoostedTreeProgram.from_dict(payload)

    categorical = HistGradientBoostingClassifier(
        categorical_features=[True, False], max_iter=2, min_samples_leaf=2,
        early_stopping=False, random_state=0,
    ).fit(np.asarray([[0, 0.1], [1, 0.2], [2, 0.3], [0, 0.4],
                      [1, 0.5], [2, 0.6]]), np.asarray([0, 1, 1, 0, 1, 1]))
    with pytest.raises(ValueError, match="Categorical splits"):
        R41DiagnosticBoostedTreeProgram.from_sklearn(
            categorical, feature_names=["public.category", "public.value"])


def test_payload_tree_data_is_defensively_copied():
    model, observations = _binary_model(max_iter=2)
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=[f"public.{index}" for index in range(5)])
    before = program.raw_logits_batch(observations[:5])
    payload = program.to_dict()
    saved = deepcopy(payload)
    leaf = next(node for tree in payload["trees"] for node in tree["nodes"]
                if node["kind"] == "leaf")
    leaf["value"] += 1000.0
    exported_tree = program.trees[0]
    exported_tree["nodes"][0] = {"kind": "leaf", "value": 99.0}
    assert np.array_equal(program.raw_logits_batch(observations[:5]), before)
    assert program.to_dict() == saved


def test_trace_rejects_nonfinite_accumulation_and_deep_tree_load_is_iterative():
    model, _ = _binary_model(max_iter=1)
    program = R41DiagnosticBoostedTreeProgram.from_sklearn(
        model, feature_names=[f"public.{index}" for index in range(5)])
    payload = program.to_dict()
    payload["baseline"] = [1e308]
    for tree in payload["trees"]:
        for node in tree["nodes"]:
            if node["kind"] == "leaf":
                node["value"] = 1e308
    overflowing = R41DiagnosticBoostedTreeProgram.from_dict(payload)
    with pytest.raises(ValueError, match="accumulation"):
        overflowing.trace(np.zeros(5))

    # A 1,100-level comb is deeper than Python's default recursion limit but
    # remains a valid, finite explicit tree.
    depth = 1100
    nodes = []
    for level in range(depth):
        internal_index = 2 * level
        nodes.extend((
            {"kind": "split", "feature_index": 0, "threshold": float(level),
             "missing_go_to_left": True, "left": internal_index + 1,
             "right": internal_index + 2},
            {"kind": "leaf", "value": 0.0},
        ))
    nodes.append({"kind": "leaf", "value": 0.0})
    deep_payload = program.to_dict()
    deep_payload["n_iterations"] = 1
    deep_payload["trees"] = [{"iteration": 0, "output_index": 0,
                              "nodes": nodes}]
    deep = R41DiagnosticBoostedTreeProgram.from_dict(deep_payload)
    assert len(deep.trace(np.full(5, 1e6))["trees"][0]["path"]) == depth
