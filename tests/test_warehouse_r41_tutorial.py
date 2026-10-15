from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.training.warehouse_r41_conflict_scenarios import build_manifest
from ui.warehouse_alignment_r41_tutorial import (
    build_neutral_tutorial,
    validate_neutral_tutorial,
)


@pytest.fixture(scope="module")
def conflict_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("r41-tutorial-scenes")
    old = root / "old.json"
    old.write_text(json.dumps({"splits": {"play": []}}), encoding="utf-8")
    return build_manifest(
        [old],
        counts={"train": 1, "conflict_validation": 1, "final_test": 1,
                "tutorial": 1, "play_candidates": 60},
    )


@pytest.fixture(scope="module")
def tutorial(conflict_manifest):
    return build_neutral_tutorial(conflict_manifest)


def test_neutral_tutorial_is_real_replay_with_every_required_event(
        conflict_manifest, tutorial):
    report = validate_neutral_tutorial(
        tutorial,
        tutorial_scene=conflict_manifest["splits"]["tutorial"][0],
    )
    assert report["passed"] is True
    assert report["uses_final_actor"] is False
    assert tutorial["uses_final_actor"] is False
    assert tutorial["source"] == "independent_neutral_ai_ai"
    assert tutorial["duration_ms"] == 380
    assert 2 <= len(tutorial["frames"]) <= 121
    assert all(report["coverage"][name] for name in (
        "pickup_frames", "delivery_frames", "simultaneous_movement_frames",
        "collision_frames", "wait_frames", "charge_frames",
    ))
    collision = tutorial["frames"][report["coverage"]["collision_frames"][0]]
    feedback = collision["state"]["public_feedback"]
    assert feedback["collision_kind"] == "same_target"
    assert feedback["submitted_actions"] == {"robot_1": "RIGHT", "robot_2": "LEFT"}
    assert feedback["executed_actions"] == {"robot_1": "WAIT", "robot_2": "WAIT"}


@pytest.mark.parametrize("mutation", ("state", "actions", "coverage", "binding", "actor"))
def test_neutral_tutorial_rejects_fabricated_spliced_or_relabelled_frames(
        conflict_manifest, tutorial, mutation):
    damaged = deepcopy(tutorial)
    if mutation == "state":
        damaged["frames"][5]["state"]["agents"][0]["position"] = [0, 3]
    elif mutation == "actions":
        damaged["frames"][5]["actions"]["robot_1"] = "WAIT"
    elif mutation == "coverage":
        damaged["coverage"]["pickup_frames"] = [1]
    elif mutation == "binding":
        damaged["bindings"]["tutorial_successor_state_sha256"] = "f" * 64
    else:
        damaged["uses_final_actor"] = True
    with pytest.raises(ValueError):
        validate_neutral_tutorial(
            damaged,
            tutorial_scene=conflict_manifest["splits"]["tutorial"][0],
        )

