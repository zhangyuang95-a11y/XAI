"""Behavioral regressions for the independent, task-driven teaching demo."""
from copy import deepcopy

import pytest

from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    diagnostic_scene_fingerprint,
    reset_diagnostic_scenario,
    validate_diagnostic_active_conflict,
)
from ui import warehouse_alignment_r42_tutorial as tutorial


def _scene(seed):
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=seed)
    return {
        "id": f"independent_tutorial_test_{seed}",
        "snapshot": env.snapshot(),
        "fingerprint": diagnostic_scene_fingerprint(env),
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "family_id": validate_diagnostic_active_conflict(
            env.state.tasks, config=env.config)["conflict_family_id"],
    }


@pytest.mark.parametrize("seed", [42001, 42002, 42003])
def test_tutorial_sustains_joint_delivery_in_all_three_windows(seed):
    scene = _scene(seed)
    manifest = {"splits": {"play": [scene] * 7}}
    payload = tutorial.build_tutorial(manifest)
    report = tutorial.validate_tutorial(payload, scene)
    assert min(report["delivery_windows"].values()) >= 2
    assert min(report["individual_deliveries"]) >= 3
    assert report["simultaneous_movement_count"] >= 60
    assert report["longest_no_task_progress"] <= 20
    assert payload["frames"][-1]["metrics"]["shutdowns"] == 0
    assert payload["uses_final_actor"] is False
    assert tutorial.build_tutorial(manifest) == payload


def test_physically_valid_idle_tail_cannot_pass_tutorial_acceptance():
    scene = _scene(42001)
    payload = tutorial.build_tutorial({"splits": {"play": [scene] * 7}})
    env = R41DiagnosticConflictWarehouseEnv()
    reset_diagnostic_scenario(env, scene)
    metrics = tutorial.base._metrics(env)
    frames = [tutorial.base._public_frame(env, metrics, {})]
    coverage = tutorial.base._empty_coverage()
    # The old acceptance admitted a physically valid 120-step recording with
    # progress confined to its opening. Retain enough early work to meet the
    # individual-delivery check, then reproduce that missing-progress failure.
    for index in range(1, 121):
        actions = (payload["frames"][index]["actions"] if index <= 60
                   else {"robot_1": "WAIT", "robot_2": "WAIT"})
        metrics, _ = tutorial._append(
            env, frames, coverage, metrics, actions, final=index == 120)
    padded = deepcopy(payload)
    padded.update(frames=frames, coverage=coverage)
    with pytest.raises(ValueError, match="sustain deliveries|40-step window"):
        tutorial.validate_tutorial(padded, scene)


def test_joint_teaching_decision_does_not_advance_environment():
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=42001)
    before = env.snapshot()
    goals = tutorial._teaching_goals(env, set())
    actions = tutorial._joint_teaching_action(env, goals)
    assert set(actions) == {"robot_1", "robot_2"}
    assert all(action in tutorial.base.ACTIONS for action in actions.values())
    assert env.snapshot() == before
