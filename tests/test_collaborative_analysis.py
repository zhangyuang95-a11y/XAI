from __future__ import annotations

import json

import pytest

from evaluation.trace_ablation import analyze_study_log


def _round(participant, condition, block, name, score, deliveries):
    return {
        "schema_version": "human-study-log.v5",
        "participant_id": participant,
        "condition": condition,
        "event": "round_completed",
        "round_name": name,
        "score": score,
        "deliveries": deliveries,
        "robot_collisions": 0,
        "shutdowns": 0,
        "human_route_regret_units": 1,
        "mean_delivery_latency": 20,
        "steps": 120,
        "terminal_reason": "horizon",
    }


def test_collaborative_difference_in_differences_analysis(tmp_path) -> None:
    events = []
    rows = [
        ("c1", "control", 0, 0, 10),
        ("c2", "control", 0, 10, 20),
        ("e1", "explanation", 0, 0, 40),
        ("e2", "explanation", 0, 10, 50),
    ]
    for participant, condition, block, task1, task2 in rows:
        events.append(
            {
                "schema_version": "human-study-log.v5",
                "participant_id": participant,
                "condition": condition,
                "event": "study_started",
                "assignment": {
                    "study_phase": "pilot",
                    "condition": condition,
                    "block_index": block,
                    "form_id": 0,
                },
            }
        )
        events.append(_round(participant, condition, block, "task1", task1, 1))
        events.append(_round(participant, condition, block, "task2", task2, 2))
        if condition == "explanation":
            events.extend(
                [
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "event": "explanation_presented",
                        "response_seconds": 2,
                    },
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "event": "explanation_exploration_completed",
                        "duration_seconds": 30,
                    },
                ]
            )
        events.append(
            {
                "participant_id": participant,
                "condition": condition,
                "event": "study_completed",
            }
        )
    path = tmp_path / "study.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in events) + "\n",
        encoding="utf-8",
    )
    result = analyze_study_log(
        path,
        study_phase="pilot",
        bootstrap_rounds=200,
        permutation_rounds=200,
        seed=9,
    )
    assert result["completed"] == 4
    assert result["mean_score_delta"] == {
        "control": 10,
        "explanation": 40,
    }
    assert result["itt_effect"] == 30
    assert len(result["bootstrap_95_ci"]) == 2
    assert 0 <= result["block_permutation_p"] <= 1
    assert result["explanation_uptake"] == 1
    assert result["mean_explanation_duration_seconds"] == 30
    assert result["secondary_mean_change"]["control"]["deliveries"] == 1
