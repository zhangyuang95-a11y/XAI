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
from env.warehouse.runtime_coordination import (
    causal_participant_actions,
    guard_participant_action,
    select_ai_ai_joint_actions,
    select_human_ai_action,
)
from env.warehouse.energy_management import charge_release_evidence
from env.warehouse.decision_protocol import distribution_decision_metadata
from core.policy_contracts import ActionDistribution
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


def test_future_charger_route_overlap_does_not_serialize_safe_progress() -> None:
    environment = _compact_environment()
    _install_frame_101_charger_reservation(environment)

    selected, evidence = select_ai_ai_joint_actions(
        environment,
        stable_coordination_actions(environment),
    )
    # A possible overlap several frames later is not an immediate conflict.
    # Both robots advance now; the atomic selector will re-evaluate the real
    # bottleneck from the next frozen state instead of parking Robot 1 early.
    assert selected == {
        "robot_1": "LEFT",
        "robot_2": "DOWN",
    }
    assert evidence["selected_joint_action"]["score_breakdown"][
        "progressing_agents"
    ] == 2
    assert evidence["selected_joint_action"]["score_breakdown"][
        "short_cycles"
    ] == 0


def test_robot_does_not_remain_parked_for_speculative_charger_overlap() -> None:
    environment = _compact_environment()
    _install_frame_101_charger_reservation(environment, parked=True)

    selected, evidence = select_ai_ai_joint_actions(
        environment,
        stable_coordination_actions(environment),
    )
    assert selected == {
        "robot_1": "UP",
        "robot_2": "DOWN",
    }
    assert evidence["selected_joint_action"]["score_breakdown"][
        "noncharging_waits"
    ] == 0


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


def test_wait_without_counterfactual_proof_is_honestly_called_inefficient() -> None:
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
    assert explanation is not None and "没有可验证" in explanation
    assert "低效" in explanation


def test_human_ai_unknown_action_wait_records_specific_counterfactual() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    human.position = (4, 2)
    ai = state.by_id("robot_2")
    ai.position = (4, 4)
    task = state.tasks[1]
    ai.carrying_task_id = None
    ai.navigation_goal_kind = "pickup"
    ai.navigation_goal_position = task.pickup_position
    ai.goal_type = "GO_TO_PICKUP"
    ai.goal_id = task.task_id
    ai.route_commitment_task_id = task.task_id
    environment.set_state(state)

    selected, runtime = select_human_ai_action(environment, "LEFT")
    assert selected == "WAIT"
    risky_left = next(
        item
        for item in runtime["ai_action_candidates"]
        if item["action"] == "LEFT"
    )
    assert {
        "participant_action": "RIGHT",
        "kind": "same_target",
    } in risky_left["collision_counterfactuals"]

    distribution = ActionDistribution(
        agent_id="robot_2",
        actions=tuple(ACTIONS),
        probabilities=(0.1, 0.1, 0.6, 0.1, 0.1),
        logits=(0.0, 0.0, 1.0, 0.0, 0.0),
        action_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        proposed_action="LEFT",
    )
    _, _, _, _, info = environment.step(
        {"robot_1": "UP", "robot_2": selected},
        decision_metadata=distribution_decision_metadata(
            {"robot_1": distribution, "robot_2": distribution},
            decision_source="test_human_ai_robust_selection",
            participant_overrides={"robot_1": "UP"},
            policy_actions={"robot_1": "UP", "robot_2": "LEFT"},
            selected_actions={"robot_1": "UP", "robot_2": selected},
            runtime_decision=runtime,
        ),
    )
    trace = info["decision_trace"]
    decision = trace["agents"]["robot_2"]
    assert decision["primary_reason_code"] == (
        "WAIT_FOR_UNKNOWN_PARTICIPANT_ACTION"
    )
    assert decision["human_action_uncertainty"]["collision_counterfactuals"]
    assert trace["fact_valid"] is True
    explanation = _Explainer()._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="action",
        language="en",
    )
    assert explanation is not None
    assert "did not know your current move" in explanation
    assert "trace" not in explanation.lower()


def test_participant_collision_action_reaches_environment_and_costs_200() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    state.by_id("robot_1").position = (4, 2)
    state.by_id("robot_2").position = (4, 4)
    environment.set_state(state)

    submitted, guard = guard_participant_action(environment, "RIGHT")

    assert submitted == "RIGHT"
    assert guard["requested_action"] == "RIGHT"
    assert guard["selected_action"] == "RIGHT"
    assert guard["blocked"] is False
    assert guard["collision_protection_applied"] is False

    _, _, _, _, info = environment.step(
        {"robot_1": submitted, "robot_2": "LEFT"}
    )
    after = environment.get_state()

    assert info["robot_collision_event"] is True
    assert info["robot_collision_kind"] == "same_target"
    assert info["reward_breakdown"]["robot_collision"] == -200
    assert after.robot_collision_events == 1
    assert after.score_breakdown["robot_collision"] == -200


def test_participant_standoff_retreat_is_explained_as_yield_not_detour() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    state.ineffective_joint_wait_streak = 1
    participant = state.by_id("robot_1")
    participant.position = (2, 3)
    participant.last_executed_action = "WAIT"
    participant.navigation_goal_kind = "delivery"
    participant.navigation_goal_position = (1, 0)
    ai = state.by_id("robot_2")
    ai.position = (2, 4)
    ai.battery = 74.0
    ai.carrying_task_id = "task_x"
    ai.route_commitment_task_id = "task_x"
    ai.goal_type = "GO_TO_DROPOFF"
    ai.goal_id = "task_x"
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = (1, 0)
    state.tasks = [
        DeliveryTask(
            "task_x",
            (0, 3),
            (1, 0),
            status="carried",
            carrier_agent_id="robot_2",
            created_frame=0,
            claimed_frame=0,
        ),
        DeliveryTask("task_y", (2, 6), (3, 0), created_frame=0),
    ]
    environment.set_state(state)

    selected, runtime = select_human_ai_action(environment, "WAIT")
    assert selected == "RIGHT"
    actions = {"robot_1": "WAIT", "robot_2": selected}
    distribution = ActionDistribution(
        agent_id="robot_2",
        actions=tuple(ACTIONS),
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.6),
        logits=(0.0, 0.0, 0.0, 0.0, 1.0),
        action_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        proposed_action="WAIT",
    )
    _, _, _, _, info = environment.step(
        actions,
        decision_metadata=distribution_decision_metadata(
            {"robot_1": distribution, "robot_2": distribution},
            decision_source="test_participant_standoff_clearance",
            participant_overrides={"robot_1": "WAIT"},
            policy_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
            selected_actions=actions,
            runtime_decision={**runtime, "selected_actions": actions},
        ),
    )
    trace = info["decision_trace"]
    decision = trace["agents"]["robot_2"]

    assert decision["primary_reason_code"] == "CLEAR_PARTICIPANT_STANDOFF"
    assert any(
        event["event"] == "participant_standoff_clearance"
        for event in decision["coordination_evidence"]
    )
    assert trace["fact_valid"] is True

    explainer = _Explainer()
    chinese = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    english = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="action",
        language="en",
    )

    assert chinese is not None
    assert "给机器人1让路" in chinese
    assert "狭窄通道" in chinese
    assert "继续前往B1交付" in chinese
    assert "绕路" not in chinese
    assert english is not None
    assert "yield to Robot 1" in english
    assert "narrow aisle" in english
    assert "continue toward B1" in english
    assert "detour" not in english


def test_counterfactual_safe_retreat_is_explained_as_avoidance() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    state.ineffective_joint_wait_streak = 0
    participant = state.by_id("robot_1")
    participant.position = (1, 0)
    participant.carrying_task_id = None
    participant.route_commitment_task_id = None
    participant.goal_type = "WAIT"
    participant.goal_id = None
    participant.navigation_goal_kind = "wait"
    participant.navigation_goal_position = participant.position
    ai = state.by_id("robot_2")
    ai.position = (1, 1)
    ai.battery = 80.0
    ai.carrying_task_id = "task_x"
    ai.route_commitment_task_id = "task_x"
    ai.goal_type = "GO_TO_DROPOFF"
    ai.goal_id = "task_x"
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = (1, 0)
    state.tasks = [
        DeliveryTask(
            "task_x",
            (0, 3),
            (1, 0),
            status="carried",
            carrier_agent_id="robot_2",
            created_frame=0,
            claimed_frame=0,
        ),
        DeliveryTask("task_y", (2, 6), (3, 0), created_frame=0),
    ]
    environment.set_state(state)

    selected, runtime = select_human_ai_action(environment, "WAIT")
    assert selected == "RIGHT"
    assert runtime["selected_ai_action"]["distance_after"] > (
        runtime["selected_ai_action"]["distance_before"]
    )
    actions = {"robot_1": "WAIT", "robot_2": selected}
    distribution = ActionDistribution(
        agent_id="robot_2",
        actions=tuple(ACTIONS),
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.6),
        logits=(0.0, 0.0, 0.0, 0.0, 1.0),
        action_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        proposed_action="WAIT",
    )
    _, _, _, _, info = environment.step(
        actions,
        decision_metadata=distribution_decision_metadata(
            {"robot_1": distribution, "robot_2": distribution},
            decision_source="test_counterfactual_safe_retreat",
            participant_overrides={"robot_1": "WAIT"},
            policy_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
            selected_actions=actions,
            runtime_decision={**runtime, "selected_actions": actions},
        ),
    )
    trace = info["decision_trace"]
    decision = trace["agents"]["robot_2"]

    assert decision["primary_reason_code"] == (
        "MOVE_TO_AVOID_UNKNOWN_PARTICIPANT_ACTION"
    )
    assert decision["human_action_uncertainty"]["riskier_progress_actions"]
    assert trace["fact_valid"] is True

    explainer = _Explainer()
    chinese = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    english = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="action",
        language="en",
    )

    assert chinese is not None
    assert "给机器人1让路" in chinese
    assert "继续前往B1交付" in chinese
    assert "记录中" not in chinese
    assert "低效" not in chinese
    assert english is not None
    assert "yield to Robot 1" in english
    assert "continue toward B1" in english
    assert "record" not in english.casefold()
    assert "inefficient" not in english.casefold()


def test_loaded_ai_priority_does_not_intercept_participant_action() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    state.by_id("robot_1").position = (4, 2)
    ai = state.by_id("robot_2")
    ai.position = (4, 4)
    task = state.tasks[1]
    task.status = "carried"
    task.carrier_agent_id = ai.agent_id
    task.claimed_frame = 0
    ai.carrying_task_id = task.task_id
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = task.delivery_position
    ai.goal_type = "GO_TO_DROPOFF"
    ai.goal_id = task.task_id
    ai.route_commitment_task_id = task.task_id
    environment.set_state(state)

    plan = environment.get_state().active_coordination_plan
    assert plan is not None
    assert plan["plan_kind"] == "participant_avoids_priority_cell"
    assert plan["priority_basis"] == "loaded_delivery"
    assert plan["moving_target"] == (4, 3)
    assert "RIGHT" in causal_participant_actions(environment)
    submitted, evidence = guard_participant_action(environment, "RIGHT")
    assert submitted == "RIGHT"
    assert evidence["blocked"] is False
    assert evidence["blocked_reason"] is None
    assert evidence["collision_protection_applied"] is False

    selected, runtime = select_human_ai_action(environment, "LEFT")
    assert runtime["participant_action_known_at_decision_time"] is False
    risky_left = next(
        item
        for item in runtime["ai_action_candidates"]
        if item["action"] == "LEFT"
    )
    assert {
        "participant_action": "RIGHT",
        "kind": "same_target",
    } in risky_left["collision_counterfactuals"]
    assert selected in ACTIONS


def test_joint_runtime_rejects_a_teacher_action_when_pareto_dominated() -> None:
    environment = _compact_environment()
    _install_frame_101_charger_reservation(environment, parked=True)
    policy_actions = {"robot_1": "RIGHT", "robot_2": "DOWN"}
    selected, runtime = select_ai_ai_joint_actions(environment, policy_actions)

    assert runtime["safe_joint_actions"]
    assert runtime["rejected_joint_actions"]
    assert all(
        item["reason"].startswith("collision:")
        or item["reason"] == "invalid_static_move"
        for item in runtime["rejected_joint_actions"]
    )
    assert selected == runtime["selected_actions"]


def test_exact_zero_energy_arrival_at_charger_remains_active_and_can_charge() -> None:
    environment = _compact_environment()
    state = environment.get_state()
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    robot_one.position = (5, 2)
    robot_one.battery = environment.config.move_battery_cost
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = environment.layout.charger_position
    robot_one.goal_type = "GO_TO_CHARGER"
    robot_one.goal_id = None
    robot_one.charge_mode_active = True
    robot_two.position = (0, 3)
    robot_two.battery = 100.0
    environment.set_state(state)

    _, _, terminated, truncated, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "WAIT"}
    )
    arrived = environment.get_state().by_id("robot_1")

    assert arrived.position == environment.layout.charger_position
    assert arrived.battery == 0.0
    assert arrived.active is True
    assert terminated is False
    assert truncated is False
    assert environment.get_state().shutdown_count == 0

    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    charged = environment.get_state().by_id("robot_1")
    assert charged.battery == environment.config.charge_per_wait
    assert charged.active is True


def _install_human_ai_charger_handoff(
    environment: WarehouseMultiAgentEnv,
) -> None:
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    human.position = (5, 4)
    human.battery = 20.0
    human.carrying_task_id = None
    human.route_commitment_task_id = None
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.goal_type = "GO_TO_CHARGER"
    human.goal_id = None
    human.charge_mode_active = True

    ai = state.by_id("robot_2")
    ai.position = environment.layout.charger_position
    ai.battery = 40.0
    ai.carrying_task_id = None
    ai.route_commitment_task_id = None
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.goal_type = "GO_TO_CHARGER"
    ai.goal_id = None
    ai.charge_mode_active = True
    state.active_coordination_plan = None
    environment.set_state(state)


def test_human_ai_charger_handoff_is_public_causal_and_two_phase() -> None:
    environment = _compact_environment()
    _install_human_ai_charger_handoff(environment)
    before = environment.get_state()
    plan = before.active_coordination_plan

    assert plan is not None
    assert plan["plan_kind"] == "charger_handoff_clearance"
    assert plan["phase"] == "CLEAR_CELL"
    assert plan["priority_agent_id"] == "robot_1"
    assert plan["moving_agent_id"] == "robot_2"
    assert plan["joint_actions"] == {
        "robot_1": "WAIT",
        "robot_2": "LEFT",
    }
    assert [step["phase"] for step in plan["planned_action_sequence"]] == [
        "CLEAR_CELL",
        "PASS_THROUGH",
    ]

    # The interface may recommend waiting, but it must not replace a
    # participant command. Any resulting robot conflict is scored by the
    # atomic environment transition.
    assert "LEFT" in causal_participant_actions(environment)
    submitted, guard = guard_participant_action(environment, "LEFT")
    assert submitted == "LEFT"
    assert guard["blocked"] is False
    assert guard["collision_protection_applied"] is False

    selected, runtime = select_human_ai_action(environment, "WAIT")
    assert selected == "LEFT"
    assert runtime["ai_is_planned_clearer"] is True
    plan_id = str(plan["plan_id"])
    environment.step({"robot_1": "WAIT", "robot_2": selected})

    middle = environment.get_state()
    ai = middle.by_id("robot_2")
    assert ai.position == (5, 2)
    assert ai.last_charger_departure_was_coordination is True
    assert ai.last_charger_departure_plan_id == plan_id
    assert middle.active_coordination_plan is not None
    assert middle.active_coordination_plan["phase"] == "PASS_THROUGH"

    selected, runtime = select_human_ai_action(environment, "UP")
    assert selected == "WAIT"
    assert runtime["ai_is_planned_waiter"] is True
    assert "LEFT" in causal_participant_actions(environment)
    environment.step({"robot_1": "LEFT", "robot_2": selected})
    after = environment.get_state()
    assert after.by_id("robot_1").position == environment.layout.charger_position
    assert after.active_coordination_plan is None


def test_visible_charger_contention_is_part_of_threshold_and_explanation() -> None:
    environment = _compact_environment()
    _install_human_ai_charger_handoff(environment)
    state = environment.get_state()
    ai = state.by_id("robot_2")
    evidence = charge_release_evidence(environment, state, ai)

    assert evidence["coordination_contention_energy"] == (
        environment.config.charge_per_wait
    )
    assert evidence["coordination_contention_steps"] == (
        environment.config.charge_per_wait
        / environment.config.move_battery_cost
    )
    assert evidence["release_threshold"] == min(
        100.0,
        evidence["route_energy"]
        + evidence["hysteresis_energy"]
        + evidence["coordination_contention_energy"],
    )

    # Use a stationary charger state to exercise the participant-facing
    # explanation independently from the handoff clearance action.
    state.active_coordination_plan = None
    state.by_id("robot_1").position = (4, 3)
    state.by_id("robot_2").position = environment.layout.charger_position
    state.by_id("robot_2").battery = 30.0
    environment.set_state(state)
    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    trace = info["decision_trace"]
    explainer = _Explainer()
    chinese = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="energy",
        language="zh-CN",
    )
    english = explainer._decision_trace_explanation(
        trace,
        target_agent="robot_2",
        focus="energy",
        language="en",
    )

    assert chinese is not None and "充电通道交接" in chinese
    assert "10%" in chinese
    assert english is not None and "visible charger handoff" in english
    assert "10%" in english
