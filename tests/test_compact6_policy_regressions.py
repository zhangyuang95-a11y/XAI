from __future__ import annotations

from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.coordination_priority import (
    coordination_priority,
    imminent_head_on_encounter,
)
from env.warehouse.domain import DeliveryTask, collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import _actor_action_mask
from backend.adapters.warehouse_explanations import WarehouseExplanationMixin


class _Explainer(WarehouseExplanationMixin):
    pass


def _compact_environment() -> WarehouseMultiAgentEnv:
    environment = WarehouseMultiAgentEnv(
        collaborative_study_config(participant_detour_scoring=False)
    )
    environment.reset(seed=31_105)
    return environment


def _install_frame_101_charger_reservation(
    environment: WarehouseMultiAgentEnv,
    *,
    parked: bool = False,
) -> None:
    """Install the pre-move facts behind the tutorial's 105 reversal."""

    state = environment.get_state()
    state.frame = 101 if not parked else 102
    state.tasks = [
        DeliveryTask("task_11", (0, 3), (2, 5), created_frame=100),
        DeliveryTask("task_12", (2, 6), (3, 1), created_frame=100),
    ]
    robot_one = state.by_id("robot_1")
    robot_one.position = (5, 4) if parked else environment.layout.charger_position
    robot_one.battery = 50.0 if parked else 52.0
    robot_one.carrying_task_id = None
    robot_one.route_commitment_task_id = "task_11"
    robot_one.goal_type = "GO_TO_PICKUP"
    robot_one.goal_id = "task_11"
    robot_one.navigation_goal_kind = "wait"
    robot_one.navigation_goal_position = robot_one.position
    robot_one.charge_mode_active = False

    robot_two = state.by_id("robot_2")
    robot_two.position = (1, 2)
    robot_two.battery = 31.0
    robot_two.carrying_task_id = None
    robot_two.route_commitment_task_id = None
    robot_two.goal_type = "GO_TO_CHARGER"
    robot_two.goal_id = None
    robot_two.navigation_goal_kind = "charge"
    robot_two.navigation_goal_position = environment.layout.charger_position
    robot_two.charge_mode_active = True
    environment.set_state(state)


def _install_frame_115_loaded_conflict(
    environment: WarehouseMultiAgentEnv,
) -> None:
    """Install the exact loaded-vs-empty aisle conflict before frame 116."""

    state = environment.get_state()
    state.frame = 115
    state.tasks = [
        DeliveryTask(
            "task_11",
            (0, 3),
            (2, 5),
            status="carried",
            carrier_agent_id="robot_1",
            created_frame=100,
            claimed_frame=113,
        ),
        DeliveryTask("task_12", (2, 6), (3, 1), created_frame=100),
    ]
    robot_one = state.by_id("robot_1")
    robot_one.position = (2, 3)
    robot_one.battery = 30.0
    robot_one.carrying_task_id = "task_11"
    robot_one.route_commitment_task_id = "task_11"
    robot_one.goal_type = "GO_TO_DROPOFF"
    robot_one.goal_id = "task_11"
    robot_one.navigation_goal_kind = "delivery"
    robot_one.navigation_goal_position = (2, 5)
    robot_one.charge_mode_active = False

    robot_two = state.by_id("robot_2")
    robot_two.position = (2, 2)
    robot_two.battery = 53.0
    robot_two.carrying_task_id = None
    robot_two.route_commitment_task_id = "task_12"
    robot_two.goal_type = "GO_TO_PICKUP"
    robot_two.goal_id = "task_12"
    robot_two.navigation_goal_kind = "wait"
    robot_two.navigation_goal_position = robot_two.position
    robot_two.charge_mode_active = False
    environment.set_state(state)


def _priority(environment: WarehouseMultiAgentEnv):
    state = environment.get_state()
    goals = {
        agent.agent_id: (
            state.task_by_id(agent.carrying_task_id).delivery_position
            if agent.carrying_task_id is not None
            else state.task_by_id(agent.route_commitment_task_id).pickup_position
            if agent.route_commitment_task_id is not None
            else agent.navigation_goal_position
        )
        for agent in state.agents
    }
    kinds = {
        agent.agent_id: (
            "delivery"
            if agent.carrying_task_id is not None
            else "pickup"
            if agent.route_commitment_task_id is not None
            else agent.navigation_goal_kind
        )
        for agent in state.agents
    }
    requires_charge = {
        agent.agent_id: agent.navigation_goal_kind == "charge"
        for agent in state.agents
    }
    return coordination_priority(
        state,
        environment.config,
        goal_positions=goals,
        goal_kinds=kinds,
        requires_charge=requires_charge,
        imminent_head_on=imminent_head_on_encounter(
            state,
            environment.config,
            goals,
        ),
    )


def test_charger_route_is_reserved_before_both_robots_enter_bottleneck() -> None:
    environment = _compact_environment()
    _install_frame_101_charger_reservation(environment)

    # Robot 1 parks on the right apron while robot 2 makes simultaneous
    # charger progress. It must not enter (5,2)->(4,2) and reverse later.
    assert stable_coordination_actions(environment) == {
        "robot_1": "RIGHT",
        "robot_2": "DOWN",
    }


def test_parked_robot_holds_until_charger_route_no_longer_conflicts() -> None:
    environment = _compact_environment()
    _install_frame_101_charger_reservation(environment, parked=True)

    assert stable_coordination_actions(environment) == {
        "robot_1": "WAIT",
        "robot_2": "DOWN",
    }


def test_ai_holds_queue_cell_while_participant_is_charging() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    participant = state.by_id("robot_1")
    participant.position = environment.layout.charger_position
    participant.navigation_goal_kind = "charge"
    participant.navigation_goal_position = environment.layout.charger_position
    participant.charge_mode_active = True
    participant.battery = 38.0
    actor = state.by_id("robot_2")
    actor.position = (4, 3)
    actor.navigation_goal_kind = "charge"
    actor.navigation_goal_position = environment.layout.charger_position
    actor.charge_mode_active = True
    actor.battery = 46.0
    state.active_coordination_plan = None

    mask = _actor_action_mask(state, actor, environment.config)

    assert dict(zip(ACTIONS, mask)) == {
        "UP": 0.0,
        "DOWN": 0.0,
        "LEFT": 0.0,
        "RIGHT": 0.0,
        "WAIT": 1.0,
    }


def test_goal_directions_do_not_create_false_head_on_after_routes_split() -> None:
    environment = WarehouseMultiAgentEnv(
        collaborative_study_config(participant_detour_scoring=False)
    )
    environment.reset(seed=51_001)

    assert stable_coordination_actions(environment) == {
        "robot_1": "UP",
        "robot_2": "UP",
    }
    environment.step({"robot_1": "UP", "robot_2": "UP"})

    # The robots are aligned two cells apart, but Robot 1's shortest route
    # turns upward while Robot 2's route turns left.  Their next cells are
    # distinct, so forcing Robot 2 down would be a false head-on yield and an
    # immediate UP->DOWN reversal.
    assert stable_coordination_actions(environment) == {
        "robot_1": "UP",
        "robot_2": "LEFT",
    }


def test_loaded_delivery_overrides_ordinary_single_lane_egress() -> None:
    environment = _compact_environment()
    _install_frame_115_loaded_conflict(environment)

    priority = _priority(environment)
    assert priority.agent_id == "robot_1"
    assert priority.basis == "loaded_delivery"
    assert stable_coordination_actions(environment) == {
        "robot_1": "RIGHT",
        "robot_2": "WAIT",
    }


def test_loaded_priority_move_preserves_exact_delivery_energy_boundary() -> None:
    environment = _compact_environment()
    _install_frame_115_loaded_conflict(environment)
    before = environment.get_state()
    task = before.task_by_id("task_11")
    required_before = environment._mission_route_steps(
        before,
        before.by_id("robot_1"),
        task,
        origin=before.by_id("robot_1").position,
    ) * environment.config.move_battery_cost

    environment.step({"robot_1": "RIGHT", "robot_2": "WAIT"})
    after = environment.get_state()
    required_after = environment._mission_route_steps(
        after,
        after.by_id("robot_1"),
        after.task_by_id("task_11"),
        origin=after.by_id("robot_1").position,
    ) * environment.config.move_battery_cost

    assert required_before == 30.0
    assert before.by_id("robot_1").battery == required_before
    assert after.by_id("robot_1").battery >= required_after


def test_wait_is_called_avoidable_only_after_safe_progress_is_proved() -> None:
    environment = _compact_environment()
    _install_frame_115_loaded_conflict(environment)
    state = environment.get_state()
    state.by_id("robot_1").position = (2, 4)
    state.by_id("robot_1").battery = 28.0
    state.by_id("robot_2").position = (3, 0)
    state.by_id("robot_2").route_commitment_task_id = "task_12"
    state.by_id("robot_2").goal_id = "task_12"
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    decision = info["decision_trace"]["agents"]["robot_1"]

    assert decision["primary_reason_code"] == (
        "AVOIDABLE_WAIT_SAFE_PROGRESS_AVAILABLE"
    )
    assert decision["wait_counterfactual"]["verified"] is True
    assert decision["wait_counterfactual"]["action"] == "RIGHT"


def test_explanation_uses_visible_a1_b1_slot_not_internal_task_11() -> None:
    environment = _compact_environment()
    _install_frame_115_loaded_conflict(environment)
    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "WAIT"}
    )
    trace = info["decision_trace"]
    explainer = _Explainer()

    chinese = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )
    english = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_1",
        focus="action",
        language="en",
    )

    assert chinese is not None and "A1" in chinese and "B1" in chinese
    assert english is not None and "A1" in english and "B1" in english
    assert "task_11" not in chinese
    assert "task_11" not in english


def test_loaded_priority_explanation_names_cargo_priority_and_exact_energy() -> None:
    environment = _compact_environment()
    _install_frame_115_loaded_conflict(environment)
    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "WAIT"}
    )
    trace = info["decision_trace"]
    explainer = _Explainer()

    chinese = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="collaboration",
        language="zh-CN",
    )
    english = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="collaboration",
        language="en",
    )
    energy = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_1",
        focus="energy",
        language="zh-CN",
    )

    assert chinese is not None and "载有从A1取得的货物" in chinese
    assert english is not None and "carrying cargo collected at A1" in english
    assert energy is not None and "当前电量30%" in energy
    assert "预计需30%" in energy


def test_wait_without_counterfactual_proof_is_not_called_inefficient() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    robot_one = state.by_id("robot_1")
    robot_one.position = (0, 3)
    robot_one.battery = environment.config.move_battery_cost
    robot_one.carrying_task_id = None
    robot_one.route_commitment_task_id = None
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = environment.layout.charger_position
    robot_one.goal_type = "GO_TO_CHARGER"
    robot_one.goal_id = None
    robot_one.charge_mode_active = True
    robot_two = state.by_id("robot_2")
    robot_two.position = (2, 6)
    robot_two.battery = 100.0
    environment.set_state(state)

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    trace = info["decision_trace"]
    decision = trace["agents"]["robot_1"]
    explanation = _Explainer()._decision_trace_explanation(
        trace,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )

    assert decision["wait_counterfactual"] is None
    assert decision["primary_reason_code"] != (
        "AVOIDABLE_WAIT_SAFE_PROGRESS_AVAILABLE"
    )
    assert explanation is not None and "没有支持" in explanation
    assert "低效" not in explanation
