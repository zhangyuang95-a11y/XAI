"""Protect Warehouse v5.4 development/final evidence boundaries."""
from __future__ import annotations

from copy import deepcopy

import pytest

from backend.training import warehouse_dual_v54 as trainer
from backend.training import warehouse_dual_structured_v54 as structured_trainer
from backend.training.warehouse_v54_gate import development_preflight


def test_v54_seed_blocks_are_disjoint():
    blocks = [set(trainer.PILOT_DEVELOPMENT_SEEDS),
              set(trainer.DEVELOPMENT_SEEDS),
              set(trainer.PRELIMINARY_PROGRAM_AUDIT_SEEDS),
              set(trainer.FINAL_PROGRAM_AUDIT_SEEDS),
              {seed for values in trainer.HELDOUT_SEEDS_BY_TASK.values()
               for seed in values}]
    assert all(not left & right for index, left in enumerate(blocks)
               for right in blocks[index + 1:])
    assert trainer.DEFAULT_SEEDS == (6101, 6102, 6103)
    assert trainer.DEVELOPMENT_PILOT_SEED == 6099


def test_pilot_cannot_claim_final_sets(tmp_path):
    with pytest.raises(ValueError, match="development pilot cannot"):
        trainer.run_experiment(pair_seeds=(6099,), output_dir=tmp_path,
                               development_pilot=True, enable_heldout=True)
    with pytest.raises(ValueError, match="requires 3"):
        trainer.run_experiment(pair_seeds=(6099,), output_dir=tmp_path)


def test_development_preflight_requires_all_role_margins():
    metric = {"coverage": .26, "agreement": .94, "kl": .29,
              "illegal_count": 0, "rows": 100}
    acceptance = {
        "pair_seeds": [6099], "selected_pair_seed": 6099,
        "heldout_evaluated": False, "final_program_audit": {},
        "role_sha256": {"regularized": {"robot_1": "a", "robot_2": "b"}},
        "program_sha256": {"regularized": {"robot_1": "d", "robot_2": "e"}},
        "selected_regularized_feedback_rows": {"robot_1": 5, "robot_2": 7},
        "candidate_training": [{"pair_seed": 6099,
            "initial_role_sha256": {"robot_1": "a", "robot_2": "b"},
            "variants": {
                "raw": {"joint_steps": 200000, "actor_decisions": 400000,
                        "scenario_schedule_sha256": "c",
                        "initial_role_sha256": {"robot_1": "a", "robot_2": "b"},
                        "both_roles_changed": True, "post_policy_interventions": 0,
                        "feedback_rows": {"robot_1": 0, "robot_2": 0}},
                "regularized": {"joint_steps": 200000,
                                "actor_decisions": 400000,
                                "scenario_schedule_sha256": "c",
                                "initial_role_sha256": {"robot_1": "a", "robot_2": "b"},
                                "both_roles_changed": True,
                                "post_policy_interventions": 0}}}],
        "development": [{"pair_seed": 6099, "raw": {"passed": True},
                         "regularized": {"passed": True},
                         "preliminary_program_audit": {
                             "robot_1": {"feedback": metric,
                                         "audit_source": "preliminary_development_rollouts",
                                         "development_seeds": list(range(240024, 240048)),
                                         "sample_count": 1200,
                                         "actor_role_sha256": "a",
                                         "program_sha256": "d",
                                         "max_features_per_node": 3,
                                         "feedback_eligible": True},
                             "robot_2": {"feedback": metric,
                                         "audit_source": "preliminary_development_rollouts",
                                         "development_seeds": list(range(240024, 240048)),
                                         "sample_count": 1200,
                                         "actor_role_sha256": "b",
                                         "program_sha256": "e",
                                         "max_features_per_node": 3,
                                         "feedback_eligible": True,
                                         "explanation_eligible": True,
                                         "explanation": metric}}}],
    }
    assert development_preflight(acceptance)["passed"] is True
    failed = deepcopy(acceptance)
    failed["development"][0]["preliminary_program_audit"]["robot_2"][
        "explanation"] = {**metric, "coverage": .19}
    assert development_preflight(failed)["passed"] is False
    failed = deepcopy(acceptance)
    failed["final_program_audit"] = {"robot_1": {}}
    with pytest.raises(ValueError, match="cannot use final evidence"):
        development_preflight(failed)


@pytest.mark.parametrize("module", [trainer, structured_trainer])
def test_formal_program_preaudit_runs_only_after_quality_selection(module, monkeypatch):
    calls = []

    def audit(model, programs, *, pair_seed, seeds):
        calls.append((pair_seed, tuple(seeds)))
        return {"selected_seed": pair_seed}

    monkeypatch.setattr(module, "audit_selected_programs", audit)
    candidates = [
        {"seed": seed, "development_passed": passed, "min_quality": quality,
         "regularized": {"model": object(), "programs": {}},
         "selected_program_audit": {}}
        for seed, passed, quality in
        ((6101, True, 4.0), (6102, True, 5.0), (6103, False, 9.0))
    ]
    selected, passed = module._select_and_preaudit(
        candidates, development_pilot=False)
    assert passed is True
    assert selected["seed"] == 6102
    assert calls == [(6102, tuple(range(240072, 240096)))]
    assert candidates[0]["selected_program_audit"] == {}
    assert candidates[2]["selected_program_audit"] == {}

    calls.clear()
    for item in candidates:
        item["development_passed"] = False
        item["selected_program_audit"] = {}
    _, passed = module._select_and_preaudit(candidates, development_pilot=False)
    assert passed is False
    assert calls == []
