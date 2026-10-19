"""Producer versus independent scalar audit, using synthetic public rows."""
from copy import deepcopy

import numpy as np
import pytest

from backend.training import warehouse_v622_program as producer
from scripts import audit_warehouse_v622_capacity as audit
from scripts import run_warehouse_v622_capacity as run
from tests.test_warehouse_v622_program import actor, parts, singleton, fit, rows, Role


@pytest.mark.parametrize("qualified", [True, False])
def test_independent_sections_match_actual_program_for_nonempty_and_empty_regions(qualified):
    model = actor()
    if not qualified:
        model.roles["robot_1"] = Role(probabilities=(.8, .05, .05, .05, .05))
    data = parts()
    program = fit(actor=model, data=data)
    raw = program.to_dict()
    for rows in data:
        native = producer.program_metrics(program, model, rows)
        independent = audit.context(raw, audit.encoded_data(model, "robot_1", rows), producer.FEATURE_NAMES)
        for purpose in ("feedback", "explanation"):
            actual = audit.section(independent, raw[purpose+"_leaves"], purpose)
            audit.math.assert_equal(actual, native[purpose])
    assert bool(program.feedback_leaves) is qualified


def test_independent_math_detects_path_failure_even_when_observed_actions_legal():
    program = fit()
    raw = program.to_dict()
    raw["leaf_logits"] = [[2., 0., 0., 0., 0.]]  # UP is observed legal but has no path guard.
    data = parts()[0]
    c = audit.context(raw, audit.encoded_data(actor(), "robot_1", data), producer.FEATURE_NAMES)
    metric = audit.section(c, [0], "explanation")
    assert metric["whole_tree_raw_illegal_count"] == 0
    assert metric["whole_tree_path_uncertified_count"] == len(data)


def test_d32_final_table_is_not_counted_as_a_new_optimizer_fit():
    search = {"leaf_optimization_calls": 150, "template_generation_leaf_optimization_calls": 100,
        "frozen_table_leaf_optimization_calls": 50,
        "certificate": {"global_function_class_optimality_gap": None}}
    report = {"leaf_count": 25, "leaf_optimization_calls_outside_search": 0}
    assert run.budget_audit(search, report, "D32")["passed"]
    report["leaf_optimization_calls_outside_search"] = 25
    assert not run.budget_audit(search, report, "D32")["passed"]


def test_incomplete_native_batch_cannot_unlock_method_selection():
    from experiments.warehouse import v622_protocol as v
    from tests.test_warehouse_v622_protocol import results
    rows = results()
    rows[0] = {**rows[0], "status": "scientific_budget_exhausted", "metrics": None, "leaf_count": 0}
    assert v.summarize_units(rows)["selected_arm"] == "B32"
    with pytest.raises(RuntimeError):
        v.summarize_units(rows[:-1])


@pytest.mark.parametrize("arm", ["A32", "D32"])
def test_production_search_schema_to_program_to_independent_audit(arm):
    """Exercise the real interfaces with tiny synthetic data and full limits."""
    from backend.training.warehouse_v622_search import fit_structure
    from experiments.warehouse import v622_protocol as v
    from experiments.warehouse.v622_capacity_audit_math import audit_search_budget, audit_frozen_table
    model = actor()
    data = parts()
    data[0] = rows("fit", n=120)
    x, masks, p, episodes = run.arrays(model, "robot_1", data[0])
    structure, search = fit_structure(arm, x, masks, p, episodes,
        feature_names=producer.FEATURE_NAMES, actor_role_sha256="a"*64)
    program = producer.fit_program("robot_1", model, *data, structure=structure,
        structure_sha256=producer.program_structure_sha256(structure), fit_method=v.METHODS[arm],
        fit_budget={"prior_leaf_optimization_calls": search["leaf_optimization_calls"],
            "max_total_leaf_optimization_calls": 20540 if arm=="D32" else 20000,
            "max_leaf_optimization_calls": 0 if arm=="D32" else 32,
            "provenance_sha256": search["report_sha256"]})
    inputs = (x, masks, p, episodes, producer.FEATURE_NAMES, "a"*64)
    assert audit_search_budget(search, arm, program.report, inputs)["passed"]
    assert run.budget_audit(search, program.report, arm)["passed"]
    if arm == "D32":
        assert audit_frozen_table(search, program.to_dict(), structure, inputs)["passed"]
