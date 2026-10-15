"""Real NN rollout smoke for exact online extraction and scene isolation."""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import torch

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import WarehouseV6Actor
from env.warehouse.pure_nn import AGENTS


def test_online_extractions_stop_at_boundaries_and_never_reuse_scenes(
        monkeypatch, tmp_path):
    """Use real Warehouse NN–NN transitions with a tiny synthetic split cycle.

    The smaller cycle exercises the exact 120-step barriers without claiming
    that four scenarios are enough for a scientific program validation.
    """
    purposes = {"fit": 1, "calibration": 1, "eligibility": 1, "validation": 1}
    monkeypatch.setattr(runner, "FINAL_PROGRAM_SPLIT_COUNTS", purposes)
    calls = []

    def extract(_model, buffers):
        seed_lists = [list(dict.fromkeys(int(row["base_seed"]) for row in buffers[role]))
                      for role in AGENTS]
        assert seed_lists[0] == seed_lists[1]
        assert len(seed_lists[0]) == 4
        calls.append(seed_lists[0])
        parts = dict(zip(purposes, ([seed] for seed in seed_lists[0]), strict=True))
        reports = {role: {"training_split_scenario_seeds": parts}
                   for role in AGENTS}
        return {role: None for role in AGENTS}, reports, {role: [] for role in AGENTS}

    def no_update(_model, _optimizer, _batch, _programs, **_kwargs):
        zeros = {role: 0 for role in AGENTS}
        return {"feedback_rows": zeros, "feedback_gradient_norm": zeros,
                "physical_no_legal_action_rows": zeros,
                "feedback_disabled_epochs": zeros,
                "feedback_window_actor_rows": zeros,
                "zero_action_coverage_rows": zeros}

    def quality(_model, _seeds):
        return {"summary": {"quality": 1.0, "formal_quality_passed": True,
                            "development_quality_passed": True}}

    monkeypatch.setattr(runner, "_fit_training_programs", extract)
    monkeypatch.setattr(runner, "ppo_update", no_update)
    monkeypatch.setattr(runner, "evaluate", quality)
    # Admission is edited concurrently in this repository and belongs to the
    # real source closure. This test isolates the rollout boundary contract.
    monkeypatch.setattr(runner, "_assert_source_closure", lambda _expected: None)
    torch.manual_seed(1301)
    initial = deepcopy(WarehouseV6Actor().state_dict())
    spec = {
        "candidate_id": "online-cohort-smoke", "pair_seed": 1301,
        "joint_steps": 1560, "output_dir": str(tmp_path),
        "scenario_namespace": "online-cohort-smoke",
        "checkpoint_seeds": [1302, 1303], "smoke": True,
        "source_closure_sha256": runner.source_closure_sha256(),
        "online_program_extraction_steps": [480, 960, 1440],
    }
    _, _, report = runner._train_variant(
        initial, variant="regularized", spec=spec, directory=tmp_path)
    assert len(calls) == 3
    assert len({seed for cohort in calls for seed in cohort}) == 12
    assert [item["joint_steps"] for item in report["program_extractions"]["robot_1"]] == [
        480, 960, 1440]
    assert [item["window_start_joint_step"] for item in report[
        "program_extractions"]["robot_1"]] == [0, 480, 960]
    assert report["online_program_cohorts_disjoint"] is True
    assert report["actor_decisions"] == 3120
    assert all(not row["post_policy_interventions"]
               for row in report["training_episodes"])


def test_online_program_split_uses_exact_scenario_quota_and_keeps_resets_together(
        monkeypatch):
    class StubProgram:
        report = {"validation": {"feedback_passed": False}}

        def sha256(self):
            return "1" * 64

    observed = {}

    def fit(role, _actor, fit_rows, calibration_rows,
            eligibility_rows, validation_rows):
        observed[role] = (fit_rows, calibration_rows, eligibility_rows,
                          validation_rows)
        return StubProgram()

    monkeypatch.setattr(runner, "fit_program", fit)
    # The grouping contract is independent of NN resampling, covered by the
    # actor/program parity tests. Keep this test focused on scenario purpose.
    monkeypatch.setattr(runner, "_resample_program_rows",
                        lambda _actor, _role, rows: list(rows))
    samples = []
    for seed in range(1000, 1130):
        for reset in range(2 if seed == 1007 else 1):
            samples.append({"episode": f"{seed}:{reset}", "base_seed": seed,
                            "episode_id": 1, "frame": 0,
                            "sampled_action": "WAIT"})
    actor = WarehouseV6Actor()
    buffers = {role: deque(deepcopy(samples), maxlen=20000) for role in AGENTS}
    programs, reports, _ = runner._fit_training_programs(actor, buffers)
    assert all(programs.values())
    assert set(observed) == set(AGENTS)
    for role in AGENTS:
        splits = reports[role]["training_split_scenario_seeds"]
        assert {name: len(values) for name, values in splits.items()} == {
            "fit": 50, "calibration": 25, "eligibility": 25, "validation": 24}
        assert splits["fit"][0] == 1006
        assert splits["validation"][-1] == 1129
        assert reports[role]["training_split_episode_counts"]["fit"] == 51
        assert set(row["episode"] for row in observed[role][0] if
                   row["base_seed"] == 1007) == {"1007:0", "1007:1"}
        assert all(not any(row["base_seed"] == 1007 for row in group)
                   for group in observed[role][1:])
