"""Development evidence for the public, multi-step BC teacher prototype."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from backend.training.warehouse_v6_teacher_lookahead import public_lookahead_teacher
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import encode_public_observation


def test_hidden_route_history_cannot_change_lookahead_label():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=260300)
    state = env.get_state()
    obs, masks = encode_public_observation(state), action_masks(state)
    before = public_lookahead_teacher(obs, masks)
    altered = deepcopy(state)
    altered.active_coordination_plan = {"priority_agent_id": "robot_2"}
    for robot in altered.agents:
        robot.route_commitment_task_id = "hidden"
        robot.navigation_goal_position = (0, 0)
        robot.yielding_plan_id = "hidden"
    np.testing.assert_array_equal(obs, encode_public_observation(altered))
    np.testing.assert_array_equal(before, public_lookahead_teacher(
        encode_public_observation(altered), action_masks(altered)))


def test_lookahead_clears_a_documented_corridor_deadlock():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=260300)
    state = env.get_state()
    state.participant_controlled_agent_id = "robot_1"
    env.set_state(state)
    while not (env.state.terminated or env.state.truncated):
        obs, masks = encode_public_observation(env.state), action_masks(env.state)
        labels = public_lookahead_teacher(obs, masks)
        assert all(masks[role, int(labels[role])] == 1.0 for role in range(2))
        commands = {agent: ACTIONS[int(labels[role])]
                    for role, agent in enumerate(AGENTS)}
        _, _, _, _, info = env.step(commands)
        assert info["proposed_actions"] == commands
        assert not info["action_override_agents"]
    assert env.state.total_deliveries >= 10
    assert env.state.robot_collision_events == 0
    assert env.state.shutdown_count == 0
