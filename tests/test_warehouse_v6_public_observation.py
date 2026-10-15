"""Public-input contract for the Warehouse v6 training and frozen Actors."""

from dataclasses import asdict
from copy import deepcopy
import json

import numpy as np
import pytest

from backend.training.warehouse_v6_teacher import public_bc_teacher
from domains.warehouse import turnbased as legacy_turnbased
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState, participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse.public_observation_v6 import (
    FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256, OBSERVATION_DIM,
    OBSERVATION_FEATURE_NAMES, action_masks, encode_policy_features,
    encode_public_observation, reconstruct_public_history,
    rebuild_public_pre_action_state_from_frames, public_pre_action_snapshot,
    rebuild_public_pre_action_state,
)


def _start(seed=260300):
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=seed)
    return env


def _roundtrip(state):
    payload = json.loads(json.dumps(legacy_turnbased._pack(asdict(state)), allow_nan=False))
    data = legacy_turnbased._unpack(payload)
    data["agents"] = [AgentState(**row) for row in data["agents"]]
    for key in ("tasks", "completed_tasks"):
        data[key] = [DeliveryTask(**row) for row in data[key]]
    return WarehouseState(**data)


def _public_frame(state):
    """Export only fields a v6 participant frame is allowed to contain."""
    roles = (("human", "robot_1"), ("ai", "robot_2"))
    return {
        "turn": state.frame, "episode_id": state.episode_id,
        "map_id": STUDY_MAP_LAYOUT.layout_id,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "previous_joint_actions": (None if state.frame == 0 else {
            role: {"proposed": state.by_id(agent_id).last_action.lower(),
                   "executed": state.by_id(agent_id).last_executed_action.lower()}
            for role, agent_id in roles}),
        "previous_collision_kind": state.last_robot_collision_kind,
        "same_target_collision_streak": state.same_target_collision_streak,
        **{role: {"x": state.by_id(agent_id).position[1],
                  "y": state.by_id(agent_id).position[0],
                  "battery": state.by_id(agent_id).battery,
                  "active": state.by_id(agent_id).active,
                  "heading": state.by_id(agent_id).heading.lower(),
                  "carrying": state.by_id(agent_id).carrying_task_id}
           for role, agent_id in roles},
        "orders": [{"id": task.task_id,
                    "pickup": list(reversed(task.pickup_position)),
                    "dropoff": list(reversed(task.delivery_position)),
                    "status": task.status,
                    "carrier": next((role for role, agent_id in roles
                                     if agent_id == task.carrier_agent_id), None)}
                   for task in state.tasks],
    }


def test_public_observation_has_stable_schema_and_derived_charge_capacity():
    state = _start().get_state()
    rows = encode_public_observation(state)
    masks = action_masks(state)
    features = encode_policy_features(rows, masks)
    assert rows.shape == (2, OBSERVATION_DIM)
    assert features.shape == (2, FEATURE_DIM)
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM
    assert len(FEATURE_SCHEMA_SHA256) == 64
    assert np.array_equal(features[:, :OBSERVATION_DIM], rows)
    assert np.array_equal(features[:, OBSERVATION_DIM:], masks)
    index = {name: i for i, name in enumerate(OBSERVATION_FEATURE_NAMES)}
    assert all(rows[:, index[f"self_last_submitted_{action.lower()}"]].sum() == 0
               for action in ACTIONS)
    assert np.array_equal(rows[:, index["same_target_collision_streak_capped"]],
                          np.zeros(2))

    for role, agent in enumerate(state.agents):
        distance = shortest_path_distance(agent.position, STUDY_MAP_LAYOUT.charger_position,
                                          STUDY_MAP_LAYOUT.layout_id)
        assert rows[role, index["self_charger_shortest_distance"]] == pytest.approx(
            distance / (STUDY_MAP_LAYOUT.rows * STUDY_MAP_LAYOUT.cols))
        assert rows[role, index["self_feasible_move_steps"]] == pytest.approx(
            agent.battery / 2 / 50)
        assert rows[role, index["self_charger_return_margin"]] == pytest.approx(
            (agent.battery / 2 - distance) / 50)


def test_hidden_goal_route_and_scoring_state_cannot_change_observation_or_bc_label():
    state = _start().get_state()
    before = encode_public_observation(state)
    masks = action_masks(state)
    labels = public_bc_teacher(before, masks)
    altered = deepcopy(state)
    altered.active_coordination_plan = {"phase": "invented", "priority_agent_id": "robot_2"}
    altered.coordination_plan_cooldown_until = 999
    altered.human_route_regret_units = 999.0
    altered.score_breakdown["human_detour"] = 999.0
    for agent in altered.agents:
        agent.route_commitment_task_id = "hidden-task"
        agent.navigation_goal_kind = "charge"
        agent.navigation_goal_position = (0, 0)
        agent.goal_type = "YIELDING"
        agent.goal_id = "hidden-task"
        agent.goal_since = 999
        agent.goal_switch_reason = "hidden"
        agent.yielding_plan_id = "hidden-plan"
        agent.recent_goal_types = ("hidden",)
        agent.charge_mode_active = True
    for task in altered.tasks:
        task.claimed_battery = 0.0
        task.shortest_safe_delivery_steps = 999.0
        task.safe_path_clearance_extension_steps = 999.0
    after = encode_public_observation(altered)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(masks, action_masks(altered))
    np.testing.assert_array_equal(labels, public_bc_teacher(after, action_masks(altered)))


def test_public_history_survives_persisted_frame_roundtrip():
    env = _start(seed=260301)
    public_frames = []
    for _ in range(20):
        before = env.get_state()
        public_frames.append(_public_frame(before))
        for role, agent in (("human", "robot_1"), ("ai", "robot_2")):
            reconstructed = reconstruct_public_history(public_frames, role)
            actual = before.by_id(agent)
            assert reconstructed == {
                "recent_positions": actual.recent_positions,
                "last_battery_delta": actual.last_battery_delta,
                "steps_since_charging": actual.steps_since_charging,
                "charger_wait_streak": actual.charger_wait_streak,
                "last_submitted_action": actual.last_action if before.frame else None,
                "last_executed_action": actual.last_executed_action if before.frame else None,
                "last_robot_collision_kind": before.last_robot_collision_kind,
                "same_target_collision_streak": before.same_target_collision_streak,
            }
        restored = _roundtrip(before)
        np.testing.assert_array_equal(encode_public_observation(before),
                                      encode_public_observation(restored))
        snapshot_rebuilt = rebuild_public_pre_action_state(
            public_pre_action_snapshot(before))
        frame_rebuilt = rebuild_public_pre_action_state_from_frames(public_frames)
        np.testing.assert_array_equal(encode_public_observation(before),
                                      encode_public_observation(snapshot_rebuilt))
        np.testing.assert_array_equal(encode_public_observation(before),
                                      encode_public_observation(frame_rebuilt))
        masks = action_masks(before)
        labels = public_bc_teacher(encode_public_observation(before), masks)
        assert all(masks[role, int(labels[role])] == 1.0 for role in range(2))
        actions = {agent: ACTIONS[int(labels[role])]
                   for role, agent in enumerate(("robot_1", "robot_2"))}
        env.step(actions)


def test_same_target_collision_history_exposes_proposals_not_cancelled_moves():
    env = _start(seed=260302)
    state = env.get_state()
    state.by_id("robot_1").position = (1, 1)
    state.by_id("robot_2").position = (1, 3)
    env.set_state(state)
    frames = [_public_frame(env.get_state())]
    names = {name: index for index, name in enumerate(OBSERVATION_FEATURE_NAMES)}
    for expected in (1, 2):
        _, _, _, _, info = env.step({"robot_1": "RIGHT", "robot_2": "LEFT"})
        assert info["robot_collision_kind"] == "same_target"
        assert info["executed_actions"] == {"robot_1": "WAIT", "robot_2": "WAIT"}
        current = env.get_state()
        assert current.same_target_collision_streak == expected
        rows = encode_public_observation(current)
        assert rows[0, names["self_last_submitted_right"]] == 1.0
        assert rows[0, names["partner_last_submitted_left"]] == 1.0
        assert rows[1, names["self_last_submitted_left"]] == 1.0
        assert rows[0, names["same_target_collision_streak_capped"]] == expected / 8
        frames.append(_public_frame(current))
        rebuilt = rebuild_public_pre_action_state_from_frames(frames)
        np.testing.assert_array_equal(rows, encode_public_observation(rebuilt))
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert env.state.same_target_collision_streak == 0
    frames.append(_public_frame(env.get_state()))
    assert reconstruct_public_history(frames, "human")["same_target_collision_streak"] == 0
    tampered = deepcopy(frames)
    tampered[-1]["same_target_collision_streak"] = 2
    with pytest.raises(ValueError, match="streak was altered"):
        rebuild_public_pre_action_state_from_frames(tampered)


def test_policy_features_reject_invalid_masks_and_nonfinite_input():
    state = _start().get_state()
    rows, masks = encode_public_observation(state), action_masks(state)
    bad = masks.copy()
    bad[0] = 0
    with pytest.raises(ValueError):
        encode_policy_features(rows, bad)
    bad = masks.copy()
    bad[0, 0] = 0.5
    with pytest.raises(ValueError):
        encode_policy_features(rows, bad)
    bad_rows = rows.copy()
    bad_rows[0, 0] = np.nan
    with pytest.raises(ValueError):
        encode_policy_features(bad_rows, masks)
