"""Admission and fair comparison checks; synthetic receipts, no environments."""
from copy import deepcopy

import pytest

from experiments.warehouse import v621_protocol as old
from experiments.warehouse import v622_protocol as v


def metrics(coverage=.5):
    row = {"coverage": coverage, "agreement": .99, "kl": .1,
        "expected_actionable_coverage": .45, "whole_tree_raw_illegal_count": 0,
        "whole_tree_path_uncertified_count": 0}
    return {"feedback": dict(row), "explanation": dict(row)}


def results(stage="primary"):
    return [{"job": job, "status": "completed", "budget_audit": {"passed": True},
             "metrics": {"validation": metrics()}, "leaf_count": 27}
            for job in v.jobs(stage)]


def test_complexity_is_versioned_and_whole_episode_folds_unchanged():
    assert v.LIMITS["maximum_depth"] == 6
    assert v.LIMITS["maximum_leaves"] == 32
    assert old.executable_recipe()["whole_tree_limits"]["maximum_leaves"] == 16
    assert v.grouped_folds() == old.grouped_folds()
    assert len(v.jobs()) == 12 and len(v.jobs("fallback")) == 6


def test_complete_batch_required_and_one_failed_role_blocks_a_method():
    rows = results()
    with pytest.raises(RuntimeError):
        v.summarize_units(rows[:-1])
    with pytest.raises(RuntimeError):
        v.summarize_units(rows[:-1]+[deepcopy(rows[0])])
    rows[0]["metrics"]["validation"] = metrics(.1)
    summary = v.summarize_units(rows)
    assert summary["selected_arm"] == "B32"
    assert not summary["arms"]["A32"]["all_role_fold_gates_passed"]


def test_failure_routes_fallback_then_closed_without_training_authority():
    for stage, next_stage in (("primary", "fallback"), ("fallback", "closed_failed_program_precheck")):
        rows = results(stage)
        for row in rows:
            row["metrics"]["validation"] = metrics(.1)
        summary = v.summarize_units(rows, stage=stage)
        assert summary["selected_arm"] is None and summary["next_stage"] == next_stage
        assert summary["PPO_joint_steps"] == 0 and not summary["pilot_opened"]


def test_retention_rejects_formally_good_but_regressed_policy():
    reference = {"mean_deliveries": 14.9333333333, "mean_collisions": 2.84,
                 "shutdown_episode_rate": 1/75}
    candidate = dict(reference, mean_deliveries=9.)
    assert not v.quality_retained(candidate, reference)["passed"]
    assert v.quality_retained(reference, reference)["passed"]
    assert not v.quality_retained(dict(reference, mean_collisions=3.85), reference)["passed"]
    assert not v.quality_retained(dict(reference, shutdown_episode_rate=.051), reference)["passed"]
    assert not v.quality_retained(dict(reference, mean_deliveries=float("inf")), reference)["passed"]


def test_training_controls_and_loss_denominators_are_explicit():
    r = v.executable_recipe()
    assert r["training"]["reference_loss_shared_between_raw_and_regularized"] is True
    assert r["training"]["raw_program_gradients"] == 0
    assert r["training"]["critic_only_joint_steps_in_budget"] == 4000
    assert r["training"]["program_loss_denominator"] == "all_physical_role_rows"
    assert not r["pilot_opened"]
