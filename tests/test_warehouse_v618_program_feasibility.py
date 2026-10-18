"""The diagnostic schedule and expected/realized gates remain explicit."""
from copy import deepcopy

import pytest

from scripts.diagnose_warehouse_v618_program_feasibility import (
    fixed_seed_plan, metric_gates,
)
from experiments.warehouse import protocol as p


def test_feasibility_plan_has_four_episode_disjoint_full_lists_and_hashes():
    plan = fixed_seed_plan()
    assert {key: len(value) for key, value in plan["splits"].items()} == {
        "fit": 50, "calibration": 25, "eligibility": 25, "validation": 24}
    assert plan["splits"]["fit"] == list(range(263600, 263650))
    assert plan["splits"]["validation"] == list(range(263700, 263724))
    assert plan["audit"] == list(range(263800, 263824))
    assert all(len(value) == 64 for value in plan["split_seed_sha256"].values())
    assert all(plan["split_episode_ids"][name] == list(map(str, values))
               for name, values in plan["splits"].items())
    assert plan["sealed_seed_overlap_count"] == 0
    assert plan["purpose_overlap_count"] == 0


@pytest.mark.parametrize("fit_base,audit_base", [
    (240100, 263800), (263600, 250000), (263600, 263700),
    (p.task_scenario_seed(p.FINAL_PROGRAM_AUDIT_SEEDS[0], p.TASKS[0]), 263800),
])
def test_feasibility_rejects_sealed_and_overlapping_scenes_before_rollout(fit_base, audit_base):
    with pytest.raises(ValueError, match="sealed|overlap"):
        fixed_seed_plan(fit_base, audit_base)


def test_bc_expected_coverage_does_not_count_as_realized_development_coverage():
    feedback = {"coverage": .36, "actionable_coverage": .29,
                "expected_actionable_coverage": .31, "rows": 100,
                "agreement": .94, "kl": .20, "illegal_count": 0,
                "path_uncertified_count": 0,
                "whole_tree_raw_illegal_count": 0,
                "whole_tree_path_uncertified_count": 0}
    explanation = {**feedback, "agreement": .96, "actionable_coverage": .33}
    result = metric_gates(feedback, explanation)
    assert result["bc_precheck_35_30"]["passed"]
    assert not result["development_30_realized"]["passed"]
    failed = deepcopy(feedback)
    failed["illegal_count"] = 1
    assert not metric_gates(failed, explanation)["bc_precheck_35_30"]["passed"]


def test_no_qualified_rows_reports_failure_without_inventing_kl():
    metric = {"coverage": 0., "actionable_coverage": 0.,
              "expected_actionable_coverage": 0., "rows": 0,
              "agreement": 0., "kl": None, "illegal_count": 0,
              "path_uncertified_count": 0,
              "whole_tree_raw_illegal_count": 0,
              "whole_tree_path_uncertified_count": 0}
    result = metric_gates(metric, metric)
    assert not result["bc_precheck_35_30"]["passed"]
    assert not result["development_30_realized"]["passed"]
