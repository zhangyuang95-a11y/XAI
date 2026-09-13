from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    VERSION as BOOSTED_TREE_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    AGGREGATION,
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
    assemble_public_tree_program_v9,
)


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
LOGITS = {
    "base": (4.0, 0.0, 0.0, 0.0, 0.0),
    "narrow_passage": (0.0, 4.0, 0.0, 0.0, 0.0),
    "shared_pickup": (0.0, 0.0, 4.0, 0.0, 0.0),
    "shared_charger": (0.0, 0.0, 0.0, 4.0, 0.0),
}
WEIGHTS = {
    "narrow_passage": 0.2,
    "shared_pickup": 0.3,
    "shared_charger": 0.4,
}


def _base_feature_names() -> tuple[str, ...]:
    required = sorted(R41DiagnosticPublicRelationsV9._required_names())
    fillers = [f"public.filler.{index}"
               for index in range(197 - len(required))]
    names = tuple((*required, *fillers))
    assert len(names) == len(set(names)) == 197
    return names


def _tree_program(
    relations: R41DiagnosticPublicRelationsV9, logits,
) -> R41DiagnosticBoostedTreeProgram:
    split_feature = relations.feature_names.index("other.relative_row")
    trees = []
    for output_index, value in enumerate(logits):
        trees.append({
            "iteration": 0,
            "output_index": output_index,
            "nodes": [
                {"kind": "split", "feature_index": split_feature,
                 "threshold": 0.0, "missing_go_to_left": True,
                 "left": 1, "right": 2},
                {"kind": "leaf", "value": float(value)},
                {"kind": "leaf", "value": float(value)},
            ],
        })
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_TREE_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(range(len(ACTIONS))),
        "action_names": list(ACTIONS),
        "output_kind": "multiclass_logits",
        "baseline": [0.0] * len(ACTIONS),
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {},
    })


@pytest.fixture(scope="module")
def components():
    names = _base_feature_names()
    relations = R41DiagnosticPublicRelationsV9(names)
    programs = {name: _tree_program(relations, logits)
                for name, logits in LOGITS.items()}
    return names, relations, programs


@pytest.fixture(scope="module")
def program(components):
    names, _, programs = components
    return assemble_public_tree_program_v9(
        names,
        programs["base"],
        programs["narrow_passage"],
        programs["shared_pickup"],
        programs["shared_charger"],
        mix_weights=WEIGHTS,
        metadata={"candidate": "development-only", "seed": np.int64(8)},
    )


def _critical_rows(names: tuple[str, ...]) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    rows = np.zeros((5, 197), dtype=np.float32)
    # Begin with a row outside all critical routes.
    for row in rows:
        for action in ("UP", "DOWN", "LEFT"):
            row[index[f"self.neighbor.{action}.passable"]] = 1.0
        row[index["other.path_distance"]] = 1.0
        row[index["self.battery"]] = 0.8
        row[index["other.battery"]] = 0.8
        row[index["charger.self.path_distance"]] = 1.0
        row[index["charger.other.path_distance"]] = 1.0

    # row 1: narrow passage only
    rows[1, index["self.neighbor.LEFT.passable"]] = 0.0
    rows[1, index["other.path_distance"]] = np.float32(3.0 / 41.0)
    # row 2: shared pickup only
    rows[2, index["task.0.available"]] = 1.0
    rows[2, index["task.0.pickup.self.path_distance"]] = np.float32(4.0 / 41.0)
    rows[2, index["task.0.pickup.other.path_distance"]] = np.float32(4.0 / 41.0)
    # row 3: shared charger only
    rows[3, index["self.battery"]] = 0.3
    rows[3, index["charger.self.path_distance"]] = np.float32(5.0 / 41.0)
    rows[3, index["charger.other.path_distance"]] = np.float32(5.0 / 41.0)
    # row 4: all three routes overlap.
    rows[4] = 0.0
    rows[4, index["task.0.available"]] = 1.0
    return rows


def _softmax(logits) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max())
    return values / values.sum()


def _weighted_average(base, specialists) -> np.ndarray:
    numerator = np.asarray(base, dtype=np.float64).copy()
    denominator = 1.0
    for distribution, weight in specialists:
        numerator += weight * np.asarray(distribution, dtype=np.float64)
        denominator += weight
    return numerator / denominator


def test_general_constructor_binds_exact_relations_and_four_programs(program):
    payload = program.to_dict()
    assert len(program.base_feature_names) == 197
    assert len(program.feature_names) == 668
    assert payload["aggregation"] == AGGREGATION
    assert [item["group"] for item in payload["specialists"]] == list(GROUPS)
    assert [item["route"]["feature_name"] for item in payload["specialists"]] == [
        "derived.critical." + group for group in GROUPS
    ]
    assert program.mix_weights == WEIGHTS
    assert payload["metadata"] == {"candidate": "development-only", "seed": 8}
    assert all(len(item["program"]["feature_names"]) == 668
               for item in payload["specialists"])
    complexity = program.complexity()
    assert complexity == {
        "component_count": 4,
        "specialist_count": 3,
        "route_count": 3,
        "total_iterations": 4,
        "total_trees": 20,
        "total_nodes": 60,
        "split_nodes": 20,
        "leaf_nodes": 40,
        "maximum_tree_depth": 1,
        "maximum_component_iterations": 1,
        "unique_split_features": 1,
        "unique_split_feature_names": ["other.relative_row"],
        "components": {
            name: {
                "iterations": 1, "trees": 5, "nodes": 15,
                "split_nodes": 5, "leaf_nodes": 10,
                "maximum_tree_depth": 1, "unique_split_features": 1,
                "unique_split_feature_names": ["other.relative_row"],
            }
            for name in ("base", *GROUPS)
        },
    }


def test_batch_and_mapping_apply_only_triggered_routes_in_payload_order(
    program, components,
):
    names, relations, _ = components
    rows = _critical_rows(names)
    masks = relations.critical_masks(rows)
    assert masks["narrow_passage"].tolist() == [False, True, False, False, True]
    assert masks["shared_pickup"].tolist() == [False, False, True, False, True]
    assert masks["shared_charger"].tolist() == [False, False, False, True, True]

    distributions = {name: _softmax(logits) for name, logits in LOGITS.items()}
    expected = [distributions["base"]]
    for group in GROUPS:
        expected.append(_weighted_average(
            distributions["base"], [(distributions[group], WEIGHTS[group])]))
    overlap = _weighted_average(distributions["base"], [
        (distributions[group], WEIGHTS[group]) for group in GROUPS
    ])
    expected.append(overlap)
    expected = np.asarray((expected[0], expected[1], expected[2], expected[3],
                           expected[4]))

    actual = program.predict_proba_batch(rows)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
    assert np.array_equal(program.predict_batch(rows), np.argmax(expected, axis=1))
    mapping = dict(zip(names, rows[4]))
    mapped = program.predict_proba(mapping)
    np.testing.assert_array_equal(
        np.asarray([mapped[action] for action in ACTIONS]), actual[4])
    assert program.predict(mapping) == ACTIONS[int(np.argmax(expected[4]))]
    traces = program.trace_batch(rows)
    assert [trace["triggered_routes"] for trace in traces] == [
        [], ["narrow_passage"], ["shared_pickup"], ["shared_charger"],
        list(GROUPS),
    ]
    for index, trace in enumerate(traces):
        np.testing.assert_array_equal(trace["probabilities"], actual[index])
    assert all(item["program_trace"] is None
               for item in traces[0]["routes"])


def test_trace_reconstructs_overlap_order_and_every_triggered_tree_path(
    program, components,
):
    names, _, _ = components
    row = _critical_rows(names)[4]
    prediction, trace = program.predict_with_trace(dict(zip(names, row)))
    assert trace["triggered_routes"] == list(GROUPS)
    assert [item["group"] for item in trace["routes"]] == list(GROUPS)
    assert all(item["triggered"] for item in trace["routes"])
    assert len(trace["base"]["program_trace"]["trees"]) == len(ACTIONS)
    assert all(len(tree["path"]) == 1
               for tree in trace["base"]["program_trace"]["trees"])

    base = np.asarray(trace["base"]["distribution"])
    weighted_sum = base.copy()
    total_weight = 1.0
    for route_trace in trace["routes"]:
        assert route_trace["program_trace"] is not None
        assert len(route_trace["program_trace"]["trees"]) == len(ACTIONS)
        assert all(len(tree["path"]) == 1
                   for tree in route_trace["program_trace"]["trees"])
        weighted = (route_trace["mix_weight"]
                    * np.asarray(route_trace["specialist_distribution"]))
        np.testing.assert_allclose(route_trace["weighted_distribution"], weighted,
                                   rtol=0.0, atol=2e-16)
        weighted_sum += weighted
        total_weight += route_trace["mix_weight"]
    current = weighted_sum / total_weight
    np.testing.assert_array_equal(trace["weighted_probability_sum"], weighted_sum)
    assert trace["total_weight"] == total_weight
    np.testing.assert_array_equal(trace["final_probabilities"], current)
    assert trace["probabilities"] == trace["final_probabilities"]
    assert prediction == trace["prediction"] == program.predict(dict(zip(names, row)))
    probabilities, repeated = program.predict_proba_with_trace(dict(zip(names, row)))
    assert repeated == trace
    np.testing.assert_array_equal(
        [probabilities[action] for action in ACTIONS], current)


def test_custom_public_route_has_identical_batch_and_trace_boundary_semantics(
    components,
):
    names, _, programs = components
    filler = next(name for name in names if name.startswith("public.filler."))
    threshold = float(np.nextafter(np.float64(1.0), np.inf))
    routes = {
        "narrow_passage": {
            "feature_name": filler, "operator": "<", "threshold": threshold,
        },
        "shared_pickup": {
            "feature_name": "derived.critical.shared_pickup",
            "operator": ">", "threshold": 0.5,
        },
        "shared_charger": {
            "feature_name": "derived.critical.shared_charger",
            "operator": ">", "threshold": 0.5,
        },
    }
    subject = R41DiagnosticPublicTreeProgramV9.from_programs(
        names, programs["base"], programs["narrow_passage"],
        programs["shared_pickup"], programs["shared_charger"],
        mix_weights=WEIGHTS, routes=routes,
    )
    row = _critical_rows(names)[0]
    row[names.index(filler)] = np.float32(1.0)
    expected = _weighted_average(_softmax(LOGITS["base"]), [
        (_softmax(LOGITS["narrow_passage"]), WEIGHTS["narrow_passage"])
    ])
    np.testing.assert_allclose(subject.predict_proba_batch(row[None, :])[0], expected,
                               rtol=0.0, atol=2e-16)
    trace = subject.trace(dict(zip(names, row)))
    assert trace["triggered_routes"] == ["narrow_passage"]
    assert trace["routes"][0]["observed"] == 1.0


def test_json_round_trip_is_exact_and_payload_is_defensively_copied(
    program, components,
):
    names, _, _ = components
    rows = _critical_rows(names)
    encoded = program.to_json()
    restored = R41DiagnosticPublicTreeProgramV9.from_json(encoded)
    assert restored.to_dict() == json.loads(encoded)
    np.testing.assert_array_equal(
        restored.predict_proba_batch(rows), program.predict_proba_batch(rows))

    payload = program.to_dict()
    saved = deepcopy(payload)
    payload["specialists"][0]["mix_weight"] = 1.0
    route = program.routes[0]
    route["threshold"] = -100.0
    assert program.to_dict() == saved


def test_schema_program_bindings_routes_and_weights_are_strict(program):
    payload = program.to_dict()
    payload["aggregation"]["trace_order"] = list(reversed(GROUPS))
    with pytest.raises(ValueError, match="aggregation"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    payload = program.to_dict()
    payload["specialists"][0], payload["specialists"][1] = (
        payload["specialists"][1], payload["specialists"][0])
    with pytest.raises(ValueError, match="order"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    payload = program.to_dict()
    payload["specialists"][0]["mix_weight"] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    payload = program.to_dict()
    payload["specialists"][0]["route"]["feature_name"] = "actor.logits.0"
    with pytest.raises(ValueError, match="public feature"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    payload = program.to_dict()
    first, second = payload["base"]["feature_names"][:2]
    payload["base"]["feature_names"][:2] = [second, first]
    with pytest.raises(ValueError, match="exactly 668"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)
    payload = program.to_dict()
    payload["specialists"][2]["program"]["action_names"][0] = "OTHER"
    with pytest.raises(ValueError, match="action registry"):
        R41DiagnosticPublicTreeProgramV9.from_dict(payload)


def test_runtime_accepts_only_finite_raw_197_feature_inputs(program, components):
    names, _, programs = components
    row = _critical_rows(names)[0]
    mapping = dict(zip(names, row))
    with pytest.raises(ValueError, match="exactly 197"):
        program.predict({**mapping, "actor_logits": 1.0})
    missing = dict(mapping)
    del missing[names[0]]
    with pytest.raises(ValueError, match="exactly 197"):
        program.predict_proba(missing)
    with pytest.raises(ValueError, match="batch differs"):
        program.predict_proba_batch(np.zeros((1, 668), dtype=np.float32))
    changed = row.copy()
    changed[0] = np.nan
    with pytest.raises(ValueError, match="batch differs"):
        program.predict_batch(changed[None, :])
    with pytest.raises(ValueError, match="mix weight"):
        R41DiagnosticPublicTreeProgramV9.from_programs(
            names, programs["base"], programs["narrow_passage"],
            programs["shared_pickup"], programs["shared_charger"],
            mix_weights={**WEIGHTS, "narrow_passage": True},
        )

    numpy_weights = {group: np.float64(weight)
                     for group, weight in WEIGHTS.items()}
    rebuilt = R41DiagnosticPublicTreeProgramV9.from_programs(
        names, programs["base"], programs["narrow_passage"],
        programs["shared_pickup"], programs["shared_charger"],
        mix_weights=numpy_weights,
    )
    assert rebuilt.mix_weights == WEIGHTS
