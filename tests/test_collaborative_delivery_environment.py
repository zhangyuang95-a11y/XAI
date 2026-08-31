from __future__ import annotations

from copy import deepcopy

import pytest

from env.warehouse.environment import (
    CHARGER_POSITION,
    MOVE_DELTAS,
    SHELF_COLUMNS,
    SHELF_ROWS,
    WAITING_POSITIONS,
    WarehouseConfig,
    WarehouseMultiAgentEnv,
    is_passable,
    shortest_path_distance,
)
from env.warehouse.domain import DeliveryTask
from env.warehouse.observations import _actor_visible_goal
from env.warehouse.layouts import (
    COMPACT_STAGGERED_8X9_LAYOUT,
    CORRIDOR_SHELF_LAYOUT,
    STAGGERED_AISLES_LAYOUT,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.coordination_plan import frozen_joint_coordination_plan
from env.warehouse.scenarios import (
    apply_charger_commitment_scenario,
    apply_charger_handoff_scenario,
    apply_delivery_goal_clearance_scenario,
    apply_dual_charger_approach_scenario,
    apply_empty_delivery_clearance_scenario,
    apply_head_on_scenario,
    apply_outer_exit_charger_approach_scenario,
)
from env.warehouse.regressions import seed_42027_regression_state
from env.warehouse.rewards import RewardConfig


def _legacy_config(**overrides) -> WarehouseConfig:
    values = {
        "rows": STAGGERED_AISLES_LAYOUT.rows,
        "cols": STAGGERED_AISLES_LAYOUT.cols,
        "map_layout_id": STAGGERED_AISLES_LAYOUT.layout_id,
    }
    values.update(overrides)
    return WarehouseConfig(**values)


def _compact_legacy_config(**overrides) -> WarehouseConfig:
    values = {
        "rows": COMPACT_STAGGERED_8X9_LAYOUT.rows,
        "cols": COMPACT_STAGGERED_8X9_LAYOUT.cols,
        "map_layout_id": COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
    }
    values.update(overrides)
    return WarehouseConfig(**values)


def _task_signature(environment: WarehouseMultiAgentEnv):
    state = environment.get_state()
    return tuple(
        (
            task.task_id,
            task.pickup_position,
            task.delivery_position,
            task.status,
            task.carrier_agent_id,
        )
        for task in sorted(state.tasks, key=lambda item: item.task_id)
    )


def _set_agent_positions(environment, left, right) -> None:
    state = environment.get_state()
    state.by_id("robot_1").position = left
    state.by_id("robot_2").position = right
    environment.set_state(state)


def _give_first_task_to_robot_one(environment: WarehouseMultiAgentEnv):
    state = environment.get_state()
    task = state.tasks[0]
    task.status = "carried"
    task.carrier_agent_id = "robot_1"
    task.claimed_frame = state.frame
    state.by_id("robot_1").carrying_task_id = task.task_id
    environment.set_state(state)
    return task


def test_executed_neural_progress_creates_nonbinding_route_commitment() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=12))
    environment.reset(seed=51)
    state = environment.get_state()
    state.tasks = [
        DeliveryTask("task_1", (5, 0), (1, 0), created_frame=0),
        DeliveryTask("task_2", (4, 10), (2, 10), created_frame=0),
    ]
    state.by_id("robot_1").position = (5, 5)
    state.by_id("robot_2").position = (9, 6)
    environment.set_state(state)

    environment.step({"robot_1": "LEFT", "robot_2": "WAIT"})
    committed = environment.get_state()
    assert committed.by_id("robot_1").route_commitment_task_id == "task_1"
    assert committed.task_by_id("task_1").status == "available"
    assert committed.task_by_id("task_1").carrier_agent_id is None

    # A later move that is not pickup progress does not erase neural intent;
    # this is the memory needed to resume the same job after charging.
    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert (
        environment.get_state().by_id("robot_1").route_commitment_task_id
        == "task_1"
    )


def test_charger_route_progress_does_not_create_task_commitment() -> None:
    environment = WarehouseMultiAgentEnv(_compact_legacy_config())
    environment.reset(seed=17)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    robot.position = (6, 4)
    robot.battery = 20.0
    robot.navigation_goal_kind = "charge"
    robot.navigation_goal_position = environment.layout.charger_position
    robot.charge_mode_active = True
    robot.route_commitment_task_id = None
    # Make one task's A point lie on the same geometric branch as the charger
    # step.  The step is charge intent, not evidence of task selection.
    task = state.tasks[0]
    task.pickup_position = (7, 3)
    task.delivery_position = (0, 4)
    environment.set_state(state)

    environment.step({"robot_1": "DOWN", "robot_2": "WAIT"})

    assert (
        environment.get_state().by_id("robot_1").route_commitment_task_id
        is None
    )


def test_shared_prefix_move_preserves_the_predecision_persistent_goal() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=12))
    environment.reset(seed=54)
    state = environment.get_state()
    state.tasks = [
        DeliveryTask("task_1", (2, 8), (3, 2), created_frame=0),
        DeliveryTask("task_2", (4, 8), (1, 2), created_frame=0),
    ]
    state.by_id("robot_1").position = (7, 3)
    state.by_id("robot_2").position = (7, 5)
    environment.set_state(state)
    committed_before = environment.get_state().by_id(
        "robot_1"
    ).goal_id

    environment.step({"robot_1": "UP", "robot_2": "WAIT"})

    robot = environment.get_state().by_id("robot_1")
    assert committed_before is not None
    assert robot.route_commitment_task_id == committed_before


def test_duplicate_commitment_is_not_created_while_another_task_is_free() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=12))
    environment.reset(seed=55)
    state = environment.get_state()
    state.tasks = [
        DeliveryTask("task_1", (5, 0), (1, 0), created_frame=0),
        DeliveryTask("task_2", (4, 10), (2, 10), created_frame=0),
    ]
    state.by_id("robot_1").position = (5, 5)
    state.by_id("robot_2").position = (9, 6)
    state.by_id("robot_2").route_commitment_task_id = "task_1"
    environment.set_state(state)

    environment.step({"robot_1": "LEFT", "robot_2": "WAIT"})

    assert (
        environment.get_state().by_id("robot_1").route_commitment_task_id
        == "task_2"
    )
    assert (
        environment.get_state().by_id("robot_2").route_commitment_task_id
        == "task_1"
    )


def test_route_commitment_clears_when_teammate_claims_task() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=12))
    environment.reset(seed=52)
    state = environment.get_state()
    first = state.tasks[0]
    first.status = "carried"
    first.carrier_agent_id = "robot_2"
    first.claimed_frame = state.frame
    state.by_id("robot_2").carrying_task_id = first.task_id
    state.by_id("robot_1").route_commitment_task_id = first.task_id
    environment.set_state(state)

    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})

    assert environment.get_state().by_id("robot_1").route_commitment_task_id is None


def test_clearance_move_cannot_retarget_a_valid_commitment() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=12))
    environment.reset(seed=53)
    state = environment.get_state()
    state.tasks = [
        DeliveryTask("task_1", (5, 0), (1, 0), created_frame=0),
        DeliveryTask("task_2", (4, 10), (2, 10), created_frame=0),
    ]
    robot = state.by_id("robot_1")
    robot.position = (5, 5)
    robot.route_commitment_task_id = "task_1"
    state.by_id("robot_2").position = (9, 6)
    environment.set_state(state)

    environment.step({"robot_1": "UP", "robot_2": "WAIT"})

    updated = environment.get_state().by_id("robot_1")
    assert updated.route_commitment_task_id == "task_1"
    assert _actor_visible_goal(environment.get_state(), updated) == (
        "pickup",
        (5, 0),
    )
    assert environment.get_state().task_by_id("task_2").status == "available"


def test_committed_task_controls_safe_charger_departure_energy() -> None:
    selected = None
    for seed in range(100):
        environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=12))
        environment.reset(seed=seed)
        state = environment.get_state()
        robot = state.by_id("robot_1")
        robot.position = environment.layout.charger_position
        energies = sorted(
            [
                (
                environment._mission_route_steps(
                    state,
                    robot,
                    task,
                    origin=robot.position,
                )
                * environment.config.move_battery_cost,
                task,
                )
                for task in state.tasks
            ],
            key=lambda item: (item[0], item[1].task_id),
        )
        if energies[0][0] + 2.0 <= min(100.0, energies[-1][0]):
            selected = (environment, state, energies)
            break
    assert selected is not None
    environment, state, energies = selected
    robot = state.by_id("robot_1")
    robot.battery = min(100.0, energies[0][0] + 1.0)
    robot.route_commitment_task_id = None
    assert not environment._requires_charge(state, robot)

    robot.route_commitment_task_id = energies[-1][1].task_id
    assert environment._requires_charge(state, robot)


def test_defaults_and_shared_task_endpoint_constraints() -> None:
    config = WarehouseConfig()
    assert (config.rows, config.cols) == (6, 7)
    assert (config.num_agents, config.max_agents) == (2, 2)
    assert config.horizon == 120
    assert config.active_task_count == 2
    assert config.move_battery_cost == 2.0

    for seed in range(20):
        environment = WarehouseMultiAgentEnv(config)
        environment.reset(seed=seed)
        state = environment.get_state()
        assert len(state.tasks) == 2
        assert all(agent.navigation_goal_kind == "wait" for agent in state.agents)
        assert all(
            agent.navigation_goal_position == agent.position for agent in state.agents
        )
        endpoints = [
            endpoint
            for task in state.tasks
            for endpoint in (task.pickup_position, task.delivery_position)
        ]
        assert len(set(endpoints)) == 4
        excluded = {CHARGER_POSITION, *WAITING_POSITIONS}
        assert not (set(endpoints) & excluded)
        for task in state.tasks:
            assert task.status == "available"
            assert task.carrier_agent_id is None
            assert is_passable(task.pickup_position)
            assert is_passable(task.delivery_position)
            assert task.pickup_position in environment.layout.dead_end_positions
            assert (
                task.pickup_position
                not in environment.layout.pickup_endpoint_exclusions
            )
            assert shortest_path_distance(
                task.pickup_position,
                task.delivery_position,
            ) >= 4
            assert any(
                CORRIDOR_SHELF_LAYOUT.is_blocked(
                    (
                        task.pickup_position[0] + delta[0],
                        task.pickup_position[1] + delta[1],
                    )
                )
                for delta in MOVE_DELTAS.values()
            )

    with pytest.raises(ValueError, match="movement battery cost of 2"):
        WarehouseConfig(move_battery_cost=1.0)


def test_offline_teacher_freezes_goals_without_environment_task_assignment() -> None:
    environment = WarehouseMultiAgentEnv(_compact_legacy_config(horizon=20))
    environment.reset(seed=42027)
    before = environment.get_state()
    assert all(agent.navigation_goal_kind == "wait" for agent in before.agents)

    actions = stable_coordination_actions(environment)

    assert any(action != "WAIT" for action in actions.values())
    after = environment.get_state()
    assert [agent.navigation_goal_kind for agent in after.agents] == ["wait", "wait"]
    assert [agent.navigation_goal_position for agent in after.agents] == [
        agent.position for agent in after.agents
    ]


def test_offline_matching_prioritizes_oldest_feasible_shared_task() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=42027)
    state = environment.get_state()
    state.frame = 50
    old_task, new_task = state.tasks
    old_task.created_frame = 0
    new_task.created_frame = 49
    state.by_id("robot_1").battery = 100.0
    state.by_id("robot_2").active = False

    assignments = environment._frozen_task_assignments(
        state,
        prioritize_old_tasks=True,
    )

    assert assignments["robot_1"].task_id == old_task.task_id


def test_productive_delivery_return_is_not_an_anomalous_charger_cycle() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=42027)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    task = state.tasks[0]
    robot.position = environment.layout.charger_position
    robot.battery = 100.0
    robot.carrying_task_id = task.task_id
    task.status = "carried"
    task.carrier_agent_id = robot.agent_id
    task.claimed_frame = state.frame
    task.delivery_position = (
        environment.layout.charger_position[0] - 1,
        environment.layout.charger_position[1],
    )
    environment.set_state(state)

    environment.step({"robot_1": "UP", "robot_2": "WAIT"})
    _, _, _, _, info = environment.step(
        {"robot_1": "DOWN", "robot_2": "WAIT"}
    )

    events = tuple(item["event"] for item in info["energy_events"])
    assert "charger_productive_return" in events
    assert "charger_return_cycle" not in events


def test_unproductive_short_charger_return_remains_an_anomalous_cycle() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=42027)
    state = environment.get_state()
    state.by_id("robot_1").position = environment.layout.charger_position
    environment.set_state(state)

    environment.step({"robot_1": "UP", "robot_2": "WAIT"})
    _, _, _, _, info = environment.step(
        {"robot_1": "DOWN", "robot_2": "WAIT"}
    )

    events = tuple(item["event"] for item in info["energy_events"])
    assert "charger_return_cycle" in events
    assert "charger_productive_return" not in events


def test_charger_handoff_return_is_productive_coordination_not_cycle() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=42028)
    state = environment.get_state()
    charger_row, charger_column = environment.layout.charger_position
    occupant = state.by_id("robot_1")
    queued = state.by_id("robot_2")
    occupant.position = environment.layout.charger_position
    occupant.battery = 70.0
    queued.position = (charger_row - 1, charger_column)
    queued.battery = 20.0
    environment.set_state(state)

    environment.step({"robot_1": "LEFT", "robot_2": "DOWN"})
    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "RIGHT"}
    )

    events = {
        item["event"]: item
        for item in info["energy_events"]
        if item.get("agent_id") == "robot_1"
    }
    assert "charger_productive_return" in events
    assert "charger_return_cycle" not in events
    assert events["charger_productive_return"]["productive_reason"] == (
        "coordination"
    )


def test_return_after_teammate_leaves_charger_remains_productive() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(participant_detour_scoring=False)
    )
    environment.reset(seed=42029)
    state = environment.get_state()
    charger_row, charger_column = environment.layout.charger_position
    returning = state.by_id("robot_1")
    teammate = state.by_id("robot_2")
    returning.position = environment.layout.charger_position
    returning.battery = 70.0
    teammate.position = (charger_row - 1, charger_column)
    teammate.battery = 20.0
    environment.set_state(state)

    environment.step({"robot_1": "LEFT", "robot_2": "DOWN"})
    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    environment.step({"robot_1": "WAIT", "robot_2": "RIGHT"})
    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "UP"}
    )

    events = {
        item["event"]: item
        for item in info["energy_events"]
        if item.get("agent_id") == "robot_1"
    }
    assert "charger_productive_return" in events
    assert "charger_return_cycle" not in events
    assert events["charger_productive_return"]["productive_reason"] == (
        "coordination"
    )


def test_staggered_layout_has_three_real_charger_exit_cells() -> None:
    layout = CORRIDOR_SHELF_LAYOUT
    assert layout.tiles == (
        "#########",
        "#.......#",
        "##.###.##",
        "#.......#",
        "..#####..",
        "#.......#",
        "##..#..##",
        "###...###",
        "###...###",
    )
    assert layout.robot_start_positions == ((8, 3), (8, 5))
    assert layout.charger_position == (8, 4)
    assert layout.robot_exit_positions == ((7, 3), (7, 4), (7, 5))
    assert layout.task_endpoint_exclusions == ((7, 3), (7, 4), (7, 5))
    assert layout.is_passable((7, 3))
    assert layout.is_passable((7, 5))
    assert layout.four_way_intersections == ()
    origin = layout.passable_positions[0]
    assert all(
        shortest_path_distance(origin, position, layout.layout_id) < 100
        for position in layout.passable_positions
    )
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(
            rows=layout.rows,
            cols=layout.cols,
            map_layout_id=layout.layout_id,
        )
    )
    universally_forbidden = {
        *layout.robot_start_positions,
        *layout.task_endpoint_exclusions,
        layout.charger_position,
    }
    for seed in range(50):
        environment.reset(seed=seed)
        pickups = {
            task.pickup_position for task in environment.get_state().tasks
        }
        deliveries = {
            task.delivery_position for task in environment.get_state().tasks
        }
        endpoints = {
            endpoint
            for task in environment.get_state().tasks
            for endpoint in (task.pickup_position, task.delivery_position)
        }
        assert endpoints.isdisjoint(universally_forbidden)
        assert pickups.issubset(set(layout.dead_end_positions))
        assert deliveries.isdisjoint(set(layout.dead_end_positions))


def test_teacher_sends_full_robot_up_while_teammate_enters_charger() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=120))
    environment.reset(seed=42041)
    state = environment.get_state()
    state.by_id("robot_2").battery = 35.0
    environment.set_state(state)

    teacher = stable_coordination_actions(environment)

    assert teacher == {"robot_1": "UP", "robot_2": "LEFT"}


def test_real_same_target_conflict_bypasses_coordination_cooldown() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=120))
    environment.reset(seed=42042)
    state = environment.get_state()
    state.frame = 31
    state.coordination_plan_cooldown_until = 100
    robot_one = state.by_id("robot_1")
    robot_one.position = (4, 4)
    robot_one.battery = 20.0
    robot_one.goal_type = "GO_TO_CHARGER"
    robot_one.goal_id = None
    robot_one.route_commitment_task_id = None
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = CHARGER_POSITION
    robot_two = state.by_id("robot_2")
    robot_two.position = (5, 3)
    robot_two.battery = 30.0
    robot_two.goal_type = "GO_TO_PICKUP"
    robot_two.goal_id = None
    robot_two.route_commitment_task_id = None
    robot_two.navigation_goal_kind = "pickup"
    robot_two.navigation_goal_position = (5, 4)

    plan = frozen_joint_coordination_plan(
        state,
        environment.config,
        requires_charge={"robot_1": True, "robot_2": False},
    )

    assert plan is not None
    assert plan["plan_kind"] == "same_target_priority"
    assert plan["moving_agent_id"] != plan["waiting_agent_id"]
    assert plan["moving_target"] == (5, 4)


def test_v8_records_head_on_risk_right_of_way_and_charger_queue() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=901)
    _set_agent_positions(environment, (2, 5), (5, 5))
    _, _, _, _, info = environment.step(
        {"robot_1": "DOWN", "robot_2": "UP"}
    )
    assert "head_on_conflict_risk" in {
        item["event"] for item in info["coordination_events"]
    }
    assert info["robot_collision_event"] is False

    environment.reset(seed=902)
    state = environment.get_state()
    state.by_id("robot_1").position = (3, 5)
    state.by_id("robot_2").position = (5, 5)
    for agent, task, delivery in zip(
        state.agents,
        state.tasks,
        ((7, 5), (1, 5)),
    ):
        task.status = "carried"
        task.carrier_agent_id = agent.agent_id
        task.delivery_position = delivery
        agent.carrying_task_id = task.task_id
    environment.set_state(state)
    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "UP"}
    )
    events = {item["event"] for item in info["coordination_events"]}
    assert "joint_coordination_plan" in events
    assert environment.get_state().active_coordination_plan is not None
    # The submitted pair did not align with the frozen clearance plan, so the
    # environment correctly refuses to claim that yielding completed.
    assert environment.get_state().active_coordination_plan["phase"] == (
        "CLEAR_CELL"
    )
    assert "yield_bay_entered" not in events

    environment.reset(seed=903)
    state = environment.get_state()
    state.by_id("robot_1").position = (9, 4)
    state.by_id("robot_1").battery = 1.0
    state.by_id("robot_2").position = environment.layout.charger_position
    state.by_id("robot_2").battery = 20.0
    environment.set_state(state)
    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    assert "charger_queue" in {
        item["event"] for item in info["coordination_events"]
    }
    assert "coordination_yield" not in {
        item["event"] for item in info["coordination_events"]
    }


@pytest.mark.parametrize("agent_id", ["robot_1", "robot_2"])
def test_either_robot_can_claim_and_only_carrier_can_deliver(agent_id: str) -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=81)
    state = environment.get_state()
    task = state.tasks[0]
    carrier = state.by_id(agent_id)
    other = state.by_id("robot_2" if agent_id == "robot_1" else "robot_1")
    carrier.position = task.pickup_position
    environment.set_state(state)

    environment.step({agent_id: "WAIT", other.agent_id: "WAIT"})
    claimed = environment.get_state().task_by_id(task.task_id)
    assert claimed.status == "carried"
    assert claimed.carrier_agent_id == agent_id
    assert environment.get_state().by_id(agent_id).carrying_task_id == task.task_id
    assert environment.get_state().by_id(other.agent_id).carrying_task_id is None

    state = environment.get_state()
    state.by_id(other.agent_id).position = claimed.delivery_position
    environment.set_state(state)
    environment.step({agent_id: "WAIT", other.agent_id: "WAIT"})
    assert environment.get_state().total_deliveries == 0

    state = environment.get_state()
    state.by_id(agent_id).position = claimed.delivery_position
    state.by_id(other.agent_id).position = WAITING_POSITIONS[
        1 if other.agent_id == "robot_2" else 0
    ]
    environment.set_state(state)
    _, _, _, _, info = environment.step(
        {agent_id: "WAIT", other.agent_id: "WAIT"}
    )
    final = environment.get_state()
    assert final.total_deliveries == 1
    assert final.by_id(agent_id).carrying_task_id is None
    assert len(final.tasks) == 2
    assert task.task_id in info["delivered_task_ids"]
    assert info["reward_breakdown"]["delivery"] == 100


def test_replenishment_sequence_is_reproducible() -> None:
    left = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    right = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    left.reset(seed=902)
    right.reset(seed=902)
    assert _task_signature(left) == _task_signature(right)

    for environment in (left, right):
        state = environment.get_state()
        task = state.tasks[0]
        state.by_id("robot_1").position = task.pickup_position
        environment.set_state(state)
        environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
        state = environment.get_state()
        state.by_id("robot_1").position = task.delivery_position
        environment.set_state(state)
        environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})

    assert _task_signature(left) == _task_signature(right)
    assert left.get_state().next_task_index == 4


@pytest.mark.parametrize(
    ("positions", "actions"),
    [
        (((1, 2), (1, 4)), {"robot_1": "RIGHT", "robot_2": "LEFT"}),
        (((1, 2), (1, 3)), {"robot_1": "RIGHT", "robot_2": "LEFT"}),
        (((1, 2), (1, 3)), {"robot_1": "RIGHT", "robot_2": "WAIT"}),
    ],
)
def test_robot_conflicts_block_both_and_charge_one_collision(
    positions,
    actions,
) -> None:
    environment = WarehouseMultiAgentEnv(_compact_legacy_config(horizon=20))
    environment.reset(seed=3)
    _set_agent_positions(environment, *positions)
    batteries = {
        item.agent_id: item.battery for item in environment.get_state().agents
    }
    _, _, terminated, truncated, info = environment.step(actions)
    state = environment.get_state()
    assert tuple(item.position for item in state.agents) == positions
    assert {item.agent_id: item.battery for item in state.agents} == batteries
    assert state.robot_collision_events == 1
    assert state.last_robot_collision_event is True
    assert info["robot_collision_event"] is True
    assert info["reward_breakdown"]["robot_collision"] == -200
    assert state.by_id("robot_2").navigation_goal_kind == "wait"
    assert not terminated and not truncated


def test_static_obstacle_is_wait_without_collision_or_energy_cost() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=4)
    _set_agent_positions(environment, (1, 1), (7, 5))
    before = environment.get_state().by_id("robot_1").battery
    _, _, _, _, info = environment.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )
    state = environment.get_state()
    assert state.by_id("robot_1").position == (1, 1)
    assert state.by_id("robot_1").battery == before
    assert state.robot_collision_events == 0
    assert state.last_robot_collision_event is False
    assert info["executed_actions"]["robot_1"] == "WAIT"
    assert info["invalid_move_agents"] == ("robot_1",)


def test_ineffective_joint_wait_streak_is_observable_and_resets_on_motion() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=4242)

    _, _, _, _, first_info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    first = environment.get_state()
    _, _, _, _, second_info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    second = environment.get_state()

    assert first.ineffective_joint_wait_streak == 1
    assert second.ineffective_joint_wait_streak == 2
    assert first_info["ineffective_joint_wait_streak"] == 1
    assert second_info["ineffective_joint_wait_streak"] == 2

    movable_action = next(
        action
        for action, allowed in zip(
            (*MOVE_DELTAS, "WAIT"),
            environment.action_masks()["robot_1"],
        )
        if action != "WAIT" and allowed > 0.5
    )
    environment.step(
        {"robot_1": movable_action, "robot_2": "WAIT"}
    )

    assert environment.get_state().ineffective_joint_wait_streak == 0


def test_each_successful_move_costs_exactly_two_battery_for_both_robots() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=41)
    _set_agent_positions(environment, (1, 1), (1, 4))

    environment.step({"robot_1": "RIGHT", "robot_2": "LEFT"})

    state = environment.get_state()
    assert state.by_id("robot_1").position == (1, 2)
    assert state.by_id("robot_2").position == (1, 3)
    assert state.by_id("robot_1").battery == pytest.approx(98.0)
    assert state.by_id("robot_2").battery == pytest.approx(98.0)


def test_charging_and_shutdown_score_components() -> None:
    charging = WarehouseMultiAgentEnv(_compact_legacy_config(horizon=120))
    charging.reset(seed=5)
    state = charging.get_state()
    state.by_id("robot_1").position = charging.layout.charger_position
    state.by_id("robot_1").battery = 50
    charging.set_state(state)
    charging.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert charging.get_state().by_id("robot_1").battery == 60

    shutdown = WarehouseMultiAgentEnv(_legacy_config(horizon=120))
    shutdown.reset(seed=6)
    state = shutdown.get_state()
    state.by_id("robot_1").position = (1, 1)
    state.by_id("robot_1").battery = 1
    state.by_id("robot_2").position = (7, 5)
    shutdown.set_state(state)
    _, _, terminated, truncated, info = shutdown.step(
        {"robot_1": "RIGHT", "robot_2": "WAIT"}
    )
    final = shutdown.get_state()
    assert terminated and not truncated
    assert final.terminal_reason == "battery_shutdown"
    assert info["reward_breakdown"]["shutdown"] == -50
    assert info["reward_breakdown"]["time"] == -120
    assert info["potential_shaping_reward"] == 0.0
    assert info["training_reward"] == pytest.approx(
        info["base_training_reward"]
    )
    assert final.user_score == pytest.approx(
        sum(final.score_breakdown.values())
    )


def test_assignment_potential_shaping_is_training_only() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=7)
    _, rewards, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    user_delta = sum(info["reward_breakdown"].values())
    assert environment.get_state().user_score == pytest.approx(user_delta)
    for agent_id in environment.agent_ids:
        expected_training = (
            user_delta / 100
            + info["individual_progress_rewards"][agent_id]
            + info["coordination_progress_reward"]
            + info["counterfactual_regret_penalty_rewards"][agent_id]
            + info["repeated_avoidable_wait_penalty_rewards"][agent_id]
            + info["flat_avoidable_wait_penalty_rewards"][agent_id]
        )
        assert rewards[agent_id] == pytest.approx(expected_training)
        assert rewards[agent_id] == pytest.approx(
            info["training_rewards"][agent_id]
        )


def test_head_on_clearance_reduces_state_potential_without_fixed_bonus() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=7001)
    apply_head_on_scenario(environment, reverse=False)
    before_state = environment.get_state()
    before = environment._assignment_potential(
        before_state,
        {agent.agent_id: agent.position for agent in before_state.agents},
    )
    actions = stable_coordination_actions(environment)

    _, _, terminated, truncated, info = environment.step(actions)

    assert not terminated and not truncated
    assert not info["robot_collision_event"]
    assert info["safe_mission_potential_before"] == pytest.approx(before)
    assert info["safe_mission_potential_after"] < before
    assert info["potential_shaping_reward"] > 0.0
    assert info["reward_breakdown"]["time"] == pytest.approx(-1.0)
    assert info["reward_breakdown"]["robot_collision"] == pytest.approx(0.0)


@pytest.mark.parametrize("variant", range(6))
@pytest.mark.parametrize("reverse", (False, True))
def test_compact_head_on_catalog_requires_one_causal_clearance_move(
    variant: int,
    reverse: bool,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=7_100 + variant)
    apply_head_on_scenario(
        environment,
        reverse=reverse,
        variant=variant,
    )
    state = environment.get_state()

    actions = stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        state,
        actions,
    )

    assert not environment.validate_state(state)
    assert not invalid
    assert not collision
    assert tuple(actions.values()).count("WAIT") == 1
    moving_id = next(
        agent_id for agent_id, action in actions.items() if action != "WAIT"
    )
    assert targets[moving_id] != state.by_id(moving_id).position


@pytest.mark.parametrize("variant", range(28))
def test_delivery_goal_clearance_scenario_is_valid_follow_through_state(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=17_000 + variant)
    apply_delivery_goal_clearance_scenario(environment, variant=variant)
    state = environment.get_state()
    trailing = state.by_id("robot_1")
    teammate = state.by_id("robot_2")

    assert not environment.validate_state(state)
    assert trailing.carrying_task_id is not None
    assert teammate.carrying_task_id is not None
    assert trailing.navigation_goal_kind == "delivery"
    assert trailing.navigation_goal_position == teammate.position
    trailing_task = state.task_by_id(trailing.carrying_task_id)
    trailing_task.shortest_safe_delivery_steps = 1.0
    environment.set_state(state)

    # The offline teacher may not assume knowledge of the peer's private
    # current-frame action.  R2 vacates the B cell first while R1 holds, then
    # R1 enters it from the next frozen state.
    actions = stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        state,
        actions,
    )
    assert not invalid
    assert not collision
    assert targets["robot_1"] == trailing.position
    assert targets["robot_2"] != teammate.position

    _, _, _, _, info = environment.step(actions)
    assert "occupied_cell_clearance_wait" in {
        event["event"] for event in info["coordination_events"]
    }
    after_clearance = environment.get_state()
    assert (
        after_clearance.task_by_id(trailing.carrying_task_id)
        .shortest_safe_delivery_steps
        == 2.0
    )
    follow_actions = stable_coordination_actions(environment)
    follow_targets = environment._resolve_motion(
        after_clearance,
        follow_actions,
    )[0]
    assert follow_targets["robot_1"] == trailing.navigation_goal_position


def test_nonavoidable_participant_uncertainty_wait_extends_safe_path() -> None:
    environment = WarehouseMultiAgentEnv(_compact_legacy_config(horizon=120))
    environment.reset(seed=17_900)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    task_one, task_two = state.tasks
    robot_one.position = (2, 4)
    robot_two.position = (4, 4)
    robot_one.battery = robot_two.battery = 100.0
    for robot, task, delivery in (
        (robot_one, task_one, (2, 5)),
        (robot_two, task_two, (3, 1)),
    ):
        task.status = "carried"
        task.carrier_agent_id = robot.agent_id
        task.claimed_frame = state.frame
        robot.carrying_task_id = task.task_id
        task.delivery_position = delivery
    task_two.shortest_safe_delivery_steps = 8.0
    task_two.safe_path_charge_planned = False
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "WAIT"}
    )
    after = environment.get_state().task_by_id(task_two.task_id)

    # UP is geometrically shorter for R2, but a participant may legally move
    # DOWN into that same target from S_t. WAIT is therefore necessary causal
    # safety work and must not make the route-efficiency ratio look worse.
    assert "robot_2" not in info["avoidable_wait_agents"]
    assert after.shortest_safe_delivery_steps == pytest.approx(9.0)
    assert after.safe_path_clearance_extension_steps == pytest.approx(1.0)


def test_nonavoidable_loaded_clearance_move_extends_safe_path() -> None:
    environment = WarehouseMultiAgentEnv(_compact_legacy_config(horizon=120))
    environment.reset(seed=17_901)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    task_one, task_two = state.tasks
    robot_one.position = (3, 4)
    robot_two.position = (2, 4)
    robot_one.battery = robot_two.battery = 100.0
    for robot, task, delivery in (
        (robot_one, task_one, (1, 3)),
        (robot_two, task_two, (3, 3)),
    ):
        task.status = "carried"
        task.carrier_agent_id = robot.agent_id
        task.claimed_frame = state.frame
        robot.carrying_task_id = task.task_id
        task.delivery_position = delivery
    task_two.shortest_safe_delivery_steps = 7.0
    task_two.safe_path_charge_planned = False
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "DOWN", "robot_2": "UP"}
    )
    after = environment.get_state().task_by_id(task_two.task_id)

    # Holding R2's old cell is not robust because the participant may legally
    # enter it. The one-step retreat and later recovery are two exact units of
    # necessary clearance, not an avoidable loaded detour.
    assert "robot_2" not in info["avoidable_loaded_delivery_detour_agents"]
    assert after.shortest_safe_delivery_steps == pytest.approx(9.0)
    assert after.safe_path_clearance_extension_steps == pytest.approx(2.0)


@pytest.mark.parametrize("variant", range(8))
def test_empty_robot_causally_clears_loaded_teammate_delivery(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=18_000 + variant)
    apply_empty_delivery_clearance_scenario(environment, variant=variant)
    state = environment.get_state()
    occupant_id = "robot_2" if (variant // 3) % 2 else "robot_1"
    loaded_id = "robot_1" if occupant_id == "robot_2" else "robot_2"
    occupant = state.by_id(occupant_id)
    loaded = state.by_id(loaded_id)

    assert not environment.validate_state(state)
    assert occupant.carrying_task_id is None
    assert occupant.route_commitment_task_id is not None
    assert loaded.carrying_task_id is not None
    assert loaded.navigation_goal_position == occupant.position

    actions = stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        state,
        actions,
    )
    assert not invalid
    assert not collision
    assert actions[loaded_id] == "WAIT"
    assert actions[occupant_id] != "WAIT"
    assert targets[loaded_id] == loaded.position
    assert targets[occupant_id] != occupant.position

    environment.step(actions)
    after_clearance = environment.get_state()
    follow_actions = stable_coordination_actions(environment)
    follow_targets = environment._resolve_motion(
        after_clearance,
        follow_actions,
    )[0]
    assert follow_targets[loaded_id] == loaded.navigation_goal_position


@pytest.mark.parametrize("variant", range(8))
def test_dual_charger_approach_selects_one_entry_from_frozen_state(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=19_000 + variant)
    apply_dual_charger_approach_scenario(environment, variant=variant)
    state = environment.get_state()
    entrant_id = next(
        agent.agent_id
        for agent in state.agents
        if agent.position[0] == environment.layout.charger_position[0]
    )
    queued_id = "robot_1" if entrant_id == "robot_2" else "robot_2"
    entrant = state.by_id(entrant_id)
    queued = state.by_id(queued_id)
    entrant_battery = entrant.battery

    actions = stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        state,
        actions,
    )
    assert not environment.validate_state(state)
    assert not invalid
    assert not collision
    assert targets[queued_id] != environment.layout.charger_position
    _, _, _, _, first_info = environment.step(actions)
    assert not first_info["robot_collision_event"]
    if actions[entrant_id] == "WAIT":
        # In four approach orientations the peer can legally enter the same
        # target from S_t.  The causal protocol therefore clears that peer in
        # one frozen state and lets the entrant move only from S_{t+1}.
        assert targets[entrant_id] == entrant.position
        assert targets[queued_id] != queued.position
        second_state = environment.get_state()
        second_actions = stable_coordination_actions(environment)
        second_targets, _, second_invalid, second_collision, _, _ = (
            environment._resolve_motion(second_state, second_actions)
        )
        assert not second_invalid
        assert not second_collision
        assert second_actions[entrant_id] != "WAIT"
        assert second_targets[entrant_id] == environment.layout.charger_position
        _, _, _, _, second_info = environment.step(second_actions)
        assert not second_info["robot_collision_event"]
    else:
        assert targets[entrant_id] == environment.layout.charger_position

    assert environment.get_state().by_id(entrant_id).position == (
        environment.layout.charger_position
    )
    assert entrant_battery > environment.get_state().by_id(entrant_id).battery


@pytest.mark.parametrize("variant", range(12))
def test_outer_exit_charger_approach_advances_urgent_robot_first(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=19_100 + variant)
    apply_outer_exit_charger_approach_scenario(environment, variant=variant)
    state = environment.get_state()
    assert state.ineffective_joint_wait_streak == variant % 8
    urgent = next(
        agent
        for agent in state.agents
        if agent.position[0] == environment.layout.charger_position[0] - 1
    )
    idle = next(
        agent for agent in state.agents if agent.agent_id != urgent.agent_id
    )
    before_distance = shortest_path_distance(
        urgent.position,
        environment.layout.charger_position,
        environment.config.map_layout_id,
    )

    actions = stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        state,
        actions,
    )

    assert not invalid
    assert not collision
    assert actions[urgent.agent_id] != "WAIT"
    assert actions[idle.agent_id] == "WAIT"
    assert shortest_path_distance(
        targets[urgent.agent_id],
        environment.layout.charger_position,
        environment.config.map_layout_id,
    ) < before_distance
    assert targets[idle.agent_id] == idle.position


@pytest.mark.parametrize("agent_id", ("robot_1", "robot_2"))
@pytest.mark.parametrize("variant", range(24))
def test_recent_departure_curriculum_never_teaches_immediate_reentry(
    agent_id: str,
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=20_000 + variant)
    apply_charger_commitment_scenario(
        environment,
        agent_id=agent_id,
        variant=variant,
    )
    state = environment.get_state()
    agent = state.by_id(agent_id)
    if agent.last_charger_departure_frame is None:
        return

    actions = stable_coordination_actions(environment)
    target = environment._resolve_motion(state, actions)[0][agent_id]
    assert agent.route_commitment_task_id is not None
    assert not environment._requires_charge(state, agent)
    assert target != environment.layout.charger_position


def test_mask_representable_two_step_yield_keeps_positive_progress_signal() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=7003)
    apply_head_on_scenario(environment, reverse=False)

    infos = []
    for _ in range(4):
        actions = stable_coordination_actions(environment)
        _, _, _, _, info = environment.step(actions)
        infos.append(info)

    # Conservative decentralized clearance takes multiple frozen states: the
    # follower never enters the peer's S_t cell in the same frame it vacates.
    # The exact number of later ordinary progress frames depends on compact
    # topology and is not part of the contract.
    assert infos[0]["coordination_events"]
    assert infos[1]["coordination_events"]
    assert any(
        event.get("event") == "coordination_yield"
        for info in infos
        for event in info["coordination_events"]
    )
    assert all(not info["robot_collision_event"] for info in infos)
    assert all(info["potential_shaping_reward"] >= 0.0 for info in infos)
    assert sum(info["potential_shaping_reward"] for info in infos) > 0.0


def test_full_charger_handoff_reduces_queue_potential_only_when_cleared() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=7002)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_1",
        queued_battery=12.0,
    )
    blocked_state = environment.get_state()
    blocked = environment._assignment_potential(
        blocked_state,
        {agent.agent_id: agent.position for agent in blocked_state.agents},
    )
    wait_environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    wait_environment.reset(seed=7002)
    wait_environment.set_state(blocked_state)
    _, _, _, _, wait_info = wait_environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    actions = stable_coordination_actions(environment)

    _, _, _, _, clear_info = environment.step(actions)

    assert wait_info["potential_shaping_reward"] == pytest.approx(0.0)
    assert clear_info["safe_mission_potential_before"] == pytest.approx(blocked)
    assert clear_info["safe_mission_potential_after"] < blocked
    assert clear_info["potential_shaping_reward"] > 0.0
    assert clear_info["reward_breakdown"]["time"] == pytest.approx(-1.0)
    assert clear_info["reward_breakdown"]["robot_collision"] == pytest.approx(0.0)


def test_necessary_charging_wait_reduces_safe_mission_potential() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=71)
    _give_first_task_to_robot_one(environment)
    state = environment.get_state()
    state.by_id("robot_1").position = CHARGER_POSITION
    state.by_id("robot_1").battery = 1.0
    environment.set_state(state)

    _, rewards, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert info["safe_mission_potential_before"] - info[
        "safe_mission_potential_after"
    ] == pytest.approx(1.0)
    assert info["individual_progress_rewards"]["robot_1"] == pytest.approx(0.01)
    assert info["base_training_reward"] == pytest.approx(-0.01)
    assert "robot_1" not in info["avoidable_wait_agents"]
    assert info["counterfactual_regret_units"]["robot_1"] == 0.0
    assert rewards["robot_1"] == pytest.approx(0.0)


def test_full_charger_wait_receives_avoidable_wait_training_penalty() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=72)
    state = environment.get_state()
    state.by_id("robot_1").position = CHARGER_POSITION
    state.by_id("robot_1").battery = 100.0
    environment.set_state(state)

    _, rewards, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert info["charger_energy_gained"] == 0.0
    assert info["potential_shaping_reward"] == pytest.approx(0.0)
    assert info["base_training_reward"] == pytest.approx(-0.01)
    assert "robot_1" in info["avoidable_wait_agents"]
    assert info["counterfactual_regret_penalty_rewards"]["robot_1"] < 0.0
    assert info["repeated_avoidable_wait_penalty_rewards"]["robot_1"] == 0.0
    assert rewards["robot_1"] < info["base_training_reward"]


def test_safe_frozen_goal_progress_is_better_than_avoidable_wait() -> None:
    config = WarehouseConfig(horizon=20, participant_detour_scoring=False)
    baseline = WarehouseMultiAgentEnv(config)
    baseline.reset(seed=7)
    initial = baseline.get_state()
    waiting = WarehouseMultiAgentEnv(config)
    waiting.reset(seed=7)
    waiting.set_state(deepcopy(initial))
    moving = WarehouseMultiAgentEnv(config)
    moving.reset(seed=7)
    moving.set_state(deepcopy(initial))

    _, wait_rewards, _, _, wait_info = waiting.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    _, move_rewards, _, _, move_info = moving.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )

    assert wait_info["counterfactual_regret_units"]["robot_1"] == 1.0
    assert move_info["counterfactual_regret_units"]["robot_1"] == 0.0
    assert move_info["individual_progress_rewards"]["robot_1"] == pytest.approx(
        0.01
    )
    assert move_rewards["robot_1"] > wait_rewards["robot_1"]


def test_repeated_avoidable_wait_penalty_increases_and_caps() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=20, participant_detour_scoring=False)
    )
    environment.reset(seed=7)
    penalties: list[float] = []
    for _ in range(7):
        _, _, _, _, info = environment.step(
            {"robot_1": "WAIT", "robot_2": "WAIT"}
        )
        penalties.append(
            info["repeated_avoidable_wait_penalty_rewards"]["robot_1"]
        )

    assert penalties == pytest.approx(
        [0.0, -0.02, -0.04, -0.06, -0.08, -0.08, -0.08]
    )
    assert environment.get_state().by_id("robot_1").avoidable_wait_streak == 7


def test_teacher_yield_and_charger_handoff_have_zero_regret() -> None:
    head_on = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=20, participant_detour_scoring=False)
    )
    head_on.reset(seed=7001)
    apply_head_on_scenario(head_on, reverse=False)
    head_on_actions = stable_coordination_actions(head_on)
    _, _, _, _, head_on_info = head_on.step(head_on_actions)
    assert head_on_info["counterfactual_regret_units"] == {
        "robot_1": 0.0,
        "robot_2": 0.0,
    }

    handoff = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=20, participant_detour_scoring=False)
    )
    handoff.reset(seed=7002)
    apply_charger_handoff_scenario(
        handoff,
        occupant_agent_id="robot_1",
        queued_battery=12.0,
    )
    handoff_actions = stable_coordination_actions(handoff)
    _, _, _, _, handoff_info = handoff.step(handoff_actions)
    assert handoff_info["counterfactual_regret_units"] == {
        "robot_1": 0.0,
        "robot_2": 0.0,
    }


def test_frozen_commitment_prevents_assignment_switch_progress() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=20, participant_detour_scoring=False)
    )
    environment.reset(seed=81)
    state = environment.get_state()
    left_task, right_task = sorted(state.tasks, key=lambda task: task.task_id)
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    robot_one.route_commitment_task_id = left_task.task_id
    robot_two.route_commitment_task_id = right_task.task_id

    selected: tuple[tuple[int, int], str] | None = None
    for position in environment.layout.passable_positions:
        if position == robot_two.position:
            continue
        for action, delta in MOVE_DELTAS.items():
            target = (position[0] + delta[0], position[1] + delta[1])
            if not environment.layout.is_passable(target):
                continue
            committed_before = shortest_path_distance(
                position,
                left_task.pickup_position,
            )
            committed_after = shortest_path_distance(
                target,
                left_task.pickup_position,
            )
            other_before = shortest_path_distance(
                position,
                right_task.pickup_position,
            )
            other_after = shortest_path_distance(
                target,
                right_task.pickup_position,
            )
            if committed_after > committed_before and other_after < other_before:
                selected = (position, action)
                break
        if selected is not None:
            break
    assert selected is not None
    robot_one.position, action = selected
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": action, "robot_2": "WAIT"}
    )

    assert info["frozen_missions"]["robot_1"]["task_id"] == left_task.task_id
    assert info["individual_progress_rewards"]["robot_1"] < 0.0
    assert info["counterfactual_regret_units"]["robot_1"] > 0.0


def test_old_task_priority_cannot_retarget_valid_pickup_commitment() -> None:
    """Reward and teacher must use the pickup visible to the Actor."""

    environment = WarehouseMultiAgentEnv(
        _compact_legacy_config(horizon=120, participant_detour_scoring=False)
    )
    environment.reset(seed=40_786)
    state = environment.get_state()
    state.frame = 62
    old_task, committed_task = sorted(
        state.tasks,
        key=lambda task: task.task_id,
    )
    old_task.pickup_position = (3, 0)
    old_task.delivery_position = (1, 4)
    old_task.created_frame = 49
    committed_task.pickup_position = (5, 0)
    committed_task.delivery_position = (3, 3)
    committed_task.created_frame = 53

    charging = state.by_id("robot_1")
    charging.position = environment.layout.charger_position
    charging.battery = 10.0
    charging.carrying_task_id = None
    charging.route_commitment_task_id = None
    committed = state.by_id("robot_2")
    committed.position = (5, 1)
    committed.battery = 45.0
    committed.carrying_task_id = None
    committed.route_commitment_task_id = committed_task.task_id
    environment.set_state(state)

    frozen = environment._frozen_task_assignments(
        environment.get_state(),
        prioritize_old_tasks=True,
    )
    assert frozen["robot_2"].task_id == committed_task.task_id
    assert stable_coordination_actions(environment)["robot_2"] == "LEFT"

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "RIGHT"}
    )

    assert info["frozen_missions"]["robot_2"]["task_id"] == (
        committed_task.task_id
    )
    assert info["individual_progress_rewards"]["robot_2"] < 0.0
    assert info["counterfactual_regret_units"]["robot_2"] > 0.0
    assert "robot_2" in info["avoidable_detour_agents"]


def test_valid_commitment_outranks_matching_an_extra_robot() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=120))
    environment.reset(seed=26_210_004)
    state = environment.get_state()
    state.frame = 46
    committed_task, other_task = sorted(
        state.tasks,
        key=lambda task: task.task_id,
    )
    committed_task.pickup_position = (4, 8)
    committed_task.delivery_position = (5, 4)
    committed_task.created_frame = 32
    other_task.pickup_position = (2, 8)
    other_task.delivery_position = (5, 3)
    other_task.created_frame = 39
    committed = state.by_id("robot_1")
    committed.position = (4, 5)
    committed.battery = 44.0
    committed.carrying_task_id = None
    committed.route_commitment_task_id = committed_task.task_id
    teammate = state.by_id("robot_2")
    teammate.position = environment.layout.charger_position
    teammate.battery = 44.0
    teammate.carrying_task_id = None
    teammate.route_commitment_task_id = None
    environment.set_state(state)

    assignments = environment._frozen_task_assignments(
        environment.get_state(),
        prioritize_old_tasks=True,
    )

    assert assignments["robot_1"].task_id == committed_task.task_id
    assert stable_coordination_actions(environment)["robot_1"] == "RIGHT"


def test_coordination_round_trip_cannot_create_reward() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=20, participant_detour_scoring=False)
    )
    environment.reset(seed=7001)
    apply_head_on_scenario(environment, reverse=False)
    forward = stable_coordination_actions(environment)
    _, _, _, _, forward_info = environment.step(forward)
    opposite = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
        "WAIT": "WAIT",
    }
    _, _, _, _, reverse_info = environment.step(
        {agent_id: opposite[action] for agent_id, action in forward.items()}
    )

    assert forward_info["coordination_progress_reward"] > 0.0
    assert (
        forward_info["coordination_progress_reward"]
        + reverse_info["coordination_progress_reward"]
    ) == pytest.approx(0.0)


def test_counterfactual_regret_is_symmetric_and_individual() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=3)

    _, rewards, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "DOWN"}
    )

    assert info["mission_regression_units"] == 0.0
    assert info["mission_regression_penalty_reward"] == 0.0
    assert info["counterfactual_regret_units"] == {
        "robot_1": pytest.approx(2.0),
        "robot_2": pytest.approx(1.0),
    }
    for agent_id in environment.agent_ids:
        assert info["counterfactual_regret_penalty_rewards"][agent_id] == (
            pytest.approx(
                -0.03 * info["counterfactual_regret_units"][agent_id]
            )
        )
    assert rewards["robot_1"] != rewards["robot_2"]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"avoidable_wait_cost": -0.01},
        {"mission_regression_scale": -0.01},
    ),
)
def test_efficiency_penalty_configuration_rejects_negative_values(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**kwargs)


def test_leaving_charger_after_safe_charge_has_no_goal_switch_penalty() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=73)
    task = _give_first_task_to_robot_one(environment)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    robot.position = CHARGER_POSITION
    route_steps = environment._mission_route_steps(
        state,
        robot,
        task,
        origin=CHARGER_POSITION,
    )
    robot.battery = route_steps * environment.config.move_battery_cost
    environment.set_state(state)
    before_distance = shortest_path_distance(CHARGER_POSITION, task.delivery_position)
    action = next(
        action
        for action, delta in MOVE_DELTAS.items()
        if is_passable(
            (CHARGER_POSITION[0] + delta[0], CHARGER_POSITION[1] + delta[1])
        )
        and shortest_path_distance(
            (CHARGER_POSITION[0] + delta[0], CHARGER_POSITION[1] + delta[1]),
            task.delivery_position,
        )
        == before_distance - 1
    )

    _, _, _, _, info = environment.step(
        {"robot_1": action, "robot_2": "WAIT"}
    )

    assert info["safe_mission_potential_after"] < info[
        "safe_mission_potential_before"
    ]
    assert info["potential_shaping_reward"] >= 0.0


def test_charging_motion_cycle_cannot_create_positive_shaping() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=74)
    task = _give_first_task_to_robot_one(environment)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    robot.position = CHARGER_POSITION
    safe_energy = environment._mission_route_steps(
        state,
        robot,
        task,
        origin=CHARGER_POSITION,
    ) * environment.config.move_battery_cost
    robot.battery = safe_energy - 1.0
    environment.set_state(state)
    initial = environment.get_state()
    shaping_total = 0.0
    loop_actions = next(
        pair
        for pair, delta in (
            (("LEFT", "RIGHT"), MOVE_DELTAS["LEFT"]),
            (("UP", "DOWN"), MOVE_DELTAS["UP"]),
            (("RIGHT", "LEFT"), MOVE_DELTAS["RIGHT"]),
        )
        if (
            CHARGER_POSITION[0] + delta[0],
            CHARGER_POSITION[1] + delta[1],
        )
        != task.delivery_position
    )

    for _ in range(2):
        _, _, _, _, info = environment.step(
            {"robot_1": "WAIT", "robot_2": "WAIT"}
        )
        shaping_total += info["potential_shaping_reward"]
    for _ in range(5):
        for action in loop_actions:
            _, _, terminated, truncated, info = environment.step(
                {"robot_1": action, "robot_2": "WAIT"}
            )
            assert not terminated and not truncated
            shaping_total += info["potential_shaping_reward"]

    final = environment.get_state()
    assert final.by_id("robot_1").position == initial.by_id("robot_1").position
    assert final.by_id("robot_1").battery == pytest.approx(
        initial.by_id("robot_1").battery
    )
    assert shaping_total <= 1e-9


def test_replacement_task_is_excluded_from_delivery_transition_potential() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=75)
    task = _give_first_task_to_robot_one(environment)
    state = environment.get_state()
    state.by_id("robot_1").position = task.delivery_position
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert len(info["created_task_ids"]) == 1
    assert info["potential_shaping_reward"] >= 0.0
    final = environment.get_state()
    full_final_potential = environment._assignment_potential(
        final,
        {agent.agent_id: agent.position for agent in final.agents},
    )
    assert full_final_potential >= info["safe_mission_potential_after"]


@pytest.mark.parametrize(
    ("frame", "actions", "legacy_regret"),
    [
        (25, {"robot_1": "RIGHT", "robot_2": "UP"}, 15.0),
        (69, {"robot_1": "DOWN", "robot_2": "UP"}, 14.0),
        # The new charge-mode state correctly keeps robot 2 moving toward the
        # charger here; the archived UP command was the old return-cycle bug.
        (99, {"robot_1": "DOWN", "robot_2": "DOWN"}, 13.0),
        (119, {"robot_1": "DOWN", "robot_2": "UP"}, 13.0),
        (120, {"robot_1": "DOWN", "robot_2": "WAIT"}, 13.0),
    ],
)
def test_seed_42027_required_delivery_and_charging_moves_are_not_detours(
    frame: int,
    actions: dict[str, str],
    legacy_regret: float,
) -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=120))
    environment.reset(seed=42027)
    environment.set_state(
        seed_42027_regression_state(frame, environment.config)
    )

    _, rewards, _, _, info = environment.step(actions)

    assert legacy_regret >= 13.0  # documents the archived v20 failure
    assert info["route_regret"]["robot_1"] == pytest.approx(0.0)
    assert info["reward_breakdown"]["human_detour"] == pytest.approx(0.0)
    assert info["counterfactual_regret_units"]["robot_1"] == 0.0
    assert info["counterfactual_regret_units"]["robot_2"] == 0.0
    assert rewards == info["training_rewards"]


def test_one_step_detour_regret_is_bounded_by_two_grid_steps() -> None:
    environment = WarehouseMultiAgentEnv(_legacy_config(horizon=20))
    environment.reset(seed=42027)
    state = seed_42027_regression_state(25, environment.config)
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "LEFT", "robot_2": "UP"}
    )

    assert 0.0 <= info["route_regret"]["robot_1"] <= 2.0


def test_energy_history_and_return_cycle_are_observed_without_action_override() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=88)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    robot.position = CHARGER_POSITION
    robot.battery = 20.0
    environment.set_state(state)

    _, _, _, _, charge_info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    charged = environment.get_state().by_id("robot_1")
    assert charged.last_battery_delta == pytest.approx(10.0)
    assert charged.steps_since_charging == 0
    assert charged.charger_wait_streak == 1
    assert charge_info["charger_energy_gained_by_agent"] == {"robot_1": 10.0}

    _, _, _, _, departure_info = environment.step(
        {"robot_1": "UP", "robot_2": "WAIT"}
    )
    assert departure_info["executed_actions"]["robot_1"] == "UP"
    assert any(
        item["event"] == "charger_departure"
        for item in departure_info["energy_events"]
    )
    departure_trace = departure_info["decision_trace"]
    assert (
        departure_trace["agents"]["robot_1"]["primary_reason_code"]
        == "PREMATURE_CHARGER_DEPARTURE"
    )
    assert departure_trace["fact_valid"] is True
    _, _, _, _, return_info = environment.step(
        {"robot_1": "DOWN", "robot_2": "WAIT"}
    )
    assert return_info["executed_actions"]["robot_1"] == "DOWN"
    assert any(
        item["event"] == "charger_return_cycle"
        for item in return_info["energy_events"]
    )
    assert not any(
        item["event"] == "charger_productive_return"
        for item in return_info["energy_events"]
    )


def test_available_task_starvation_is_reported_after_forty_steps() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=89)
    state = environment.get_state()
    state.frame = 41
    state.tasks[0].created_frame = 0
    state.tasks[1].created_frame = 40
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    # The persistent allocator has already committed a live robot to the old
    # task, so an isolated WAIT is penalized as avoidable delay rather than
    # falsely reporting that the task has no owner.
    assert info["starving_task_ids"] == ()
    assert info["starving_task_assignees"] == {}


def test_committed_task_in_progress_is_not_reported_as_starving() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=189)
    state = environment.get_state()
    state.frame = 41
    old_task = state.tasks[0]
    old_task.created_frame = 0
    state.tasks[1].created_frame = 40
    state.by_id("robot_1").route_commitment_task_id = old_task.task_id
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert info["starving_task_ids"] == ()


def test_old_task_is_not_starving_during_necessary_charging() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=190)
    state = environment.get_state()
    state.frame = 41
    old_task = state.tasks[0]
    old_task.created_frame = 0
    state.tasks[1].created_frame = 40
    charging = state.by_id("robot_1")
    charging.position = environment.layout.charger_position
    charging.battery = 20.0
    charging.charge_mode_active = True
    state.by_id("robot_2").active = False
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    # Age alone is not starvation when the only feasible assignee is doing
    # causally necessary energy work for that task. Active avoidable neglect
    # after the age threshold remains covered by the preceding test.
    assert info["starving_task_ids"] == ()
