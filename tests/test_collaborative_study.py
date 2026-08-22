from __future__ import annotations

from collections import Counter

import pytest

import ui.collaborative_study as study_module
from ui.collaborative_study import (
    CollaborativeConditionAllocator,
    CollaborativeDeliveryStudy,
    CollaborativeStudyConfig,
    RoundSummary,
)


def _summary(round_name: str, seed: int, score: float) -> RoundSummary:
    return RoundSummary(
        round_name=round_name,
        seed=seed,
        score=score,
        steps=120,
        deliveries=3,
        robot_collisions=0,
        shutdowns=0,
        human_route_regret_units=2,
        mean_delivery_latency=18.5,
        terminal_reason="horizon",
    )


def _assignment(tmp_path, index: int = 0):
    allocator = CollaborativeConditionAllocator(
        randomization_seed=100,
        study_phase="pilot",
    )
    return allocator._assignment_for_index(f"p{index}", index)


def test_block_allocation_is_one_to_one_with_four_parallel_forms(tmp_path) -> None:
    allocator = CollaborativeConditionAllocator(
        randomization_seed=700,
        study_phase="pilot",
    )
    assignments = [
        allocator._assignment_for_index(f"p{index}", index)
        for index in range(24)
    ]
    for block in range(3):
        items = assignments[block * 8 : (block + 1) * 8]
        assert Counter(item.condition for item in items) == {
            "control": 4,
            "explanation": 4,
        }
        assert Counter((item.condition, item.form_id) for item in items) == {
            (condition, form_id): 1
            for condition in ("control", "explanation")
            for form_id in range(4)
        }
    assert all(item.controlled_agent == "robot_1" for item in assignments)
    assert all(item.target_agent == "robot_2" for item in assignments)
    assert all(
        len({item.demo_seed, item.task1_seed, item.task2_seed}) == 3
        for item in assignments
    )


def test_explanation_branch_allows_zero_questions_and_early_finish(tmp_path) -> None:
    events = []
    config = CollaborativeStudyConfig(event_sink=events.append)
    assignment = next(
        _assignment(tmp_path, index)
        for index in range(8)
        if _assignment(tmp_path, index).condition == "explanation"
    )
    study = CollaborativeDeliveryStudy(config)
    study.start(assignment, language="zh-CN")
    assert study.stage == "instructions"
    assert study.group_code == "A"
    study.begin_task1()
    study.finish_round(_summary("task1", assignment.task1_seed, 80))
    assert study.stage == "explanation"
    study.finish_explanation()
    assert study.stage == "task2"
    assert study.explanation_count == 0
    assert any(
        item["event"] == "explanation_exploration_completed"
        and item["explanation_count"] == 0
        for item in events
    )


def test_control_branch_waits_for_explicit_task2_confirmation(tmp_path) -> None:
    assignment = next(
        _assignment(tmp_path, index)
        for index in range(8)
        if _assignment(tmp_path, index).condition == "control"
    )
    events = []
    study = CollaborativeDeliveryStudy(
        CollaborativeStudyConfig(event_sink=events.append)
    )
    study.start(assignment, language="en")
    assert study.group_code == "B"
    study.begin_task1()
    study.finish_round(_summary("task1", assignment.task1_seed, -20))
    assert study.stage == "task1_complete"
    assert events[-1]["event"] == "task1_completion_presented"
    assert events[-1]["task1_score"] == -20
    with pytest.raises(RuntimeError):
        study.record_explanation(question="why?", report={}, response_seconds=1)
    with pytest.raises(RuntimeError):
        study.record_step({"frame": 121})
    study.begin_task2()
    assert study.stage == "task2"
    assert [item["event"] for item in events[-2:]] == [
        "task1_completion_acknowledged",
        "round_started",
    ]
    assert events[-1]["seed"] == assignment.task2_seed
    with pytest.raises(RuntimeError):
        study.begin_task2()


def test_explanation_branch_cannot_use_control_confirmation(tmp_path) -> None:
    assignment = next(
        _assignment(tmp_path, index)
        for index in range(8)
        if _assignment(tmp_path, index).condition == "explanation"
    )
    study = CollaborativeDeliveryStudy(
        CollaborativeStudyConfig(event_sink=lambda event: None)
    )
    study.start(assignment, language="en")
    study.begin_task1()
    study.finish_round(_summary("task1", assignment.task1_seed, 25))
    assert study.stage == "explanation"
    with pytest.raises(RuntimeError):
        study.begin_task2()


def test_ten_minute_timeout_and_score_delta(tmp_path, monkeypatch) -> None:
    clock = iter((100.0, 701.0, 701.0, 701.0))
    monkeypatch.setattr(study_module.time, "monotonic", lambda: next(clock))
    assignment = next(
        _assignment(tmp_path, index)
        for index in range(8)
        if _assignment(tmp_path, index).condition == "explanation"
    )
    study = CollaborativeDeliveryStudy(
        CollaborativeStudyConfig(event_sink=lambda event: None)
    )
    study.start(assignment, language="en")
    study.begin_task1()
    study.finish_round(_summary("task1", assignment.task1_seed, 10))
    assert study.explanation_time_expired
    study.finish_explanation()
    study.finish_round(_summary("task2", assignment.task2_seed, 65))
    assert study.stage == "survey"
    study.submit_survey(
        {
            "coordination_understanding": 4,
            "ai_predictability": 3,
            "interface_clarity": 5,
            "comment": "",
        }
    )
    assert study.stage == "completed"
    assert study.score_delta == 55


def test_wrong_round_or_survey_payload_is_rejected(tmp_path) -> None:
    assignment = _assignment(tmp_path, 0)
    study = CollaborativeDeliveryStudy(
        CollaborativeStudyConfig(event_sink=lambda event: None)
    )
    study.start(assignment, language="en")
    study.begin_task1()
    with pytest.raises(RuntimeError):
        study.finish_round(_summary("task2", assignment.task2_seed, 0))
