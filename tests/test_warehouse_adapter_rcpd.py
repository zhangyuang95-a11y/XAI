from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from backend.adapters.base import Intervention
from backend.adapters.warehouse import WarehouseAdapter, WarehousePolicyState
from backend.nlp.explanation_generator import ExecutionGroundedExplanationGenerator
from backend.nlp.schemas import EvidenceBundle, QueryIntent, QueryPlan
from backend.simulation.query_engine import _common_ir_cache_key
from core.rcpd import RCPD, RCPDConfig
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from env.warehouse.layouts import COMPACT_STAGGERED_8X9_LAYOUT
from env.warehouse.observations import observation_dim
from ui.web_runtime import _study_question_focus


def _legacy_config(**overrides) -> WarehouseConfig:
    return WarehouseConfig(
        rows=COMPACT_STAGGERED_8X9_LAYOUT.rows,
        cols=COMPACT_STAGGERED_8X9_LAYOUT.cols,
        map_layout_id=COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
        **overrides,
    )


def _system(
    seed: int = 17,
) -> tuple[WarehouseMultiAgentEnv, WarehouseAdapter, MAPPOPolicy]:
    config = _legacy_config(horizon=12, seed=seed)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=seed)
    adapter = WarehouseAdapter(environment)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=24, seed=seed))
    return environment, adapter, policy


def test_required_program_groups_use_shared_tasks_energy_and_teammate() -> None:
    _, adapter, policy = _system()
    context = adapter.semantic_policy_context(adapter.snapshot(policy), "robot_2")
    groups = adapter.required_program_predicate_groups()

    assert set(groups) == {
        "shared_task_state",
        "energy_state",
        "multiagent_relation",
    }
    assert "self.carrying_shared_task" in groups["shared_task_state"]
    assert "self.battery_percent" in groups["energy_state"]
    assert "other.nearest_distance" in groups["multiagent_relation"]
    assert all(
        feature in context.features
        for features in groups.values()
        for feature in features
    )


def test_charging_wait_has_nonempty_action_explanation_without_private_goal() -> None:
    environment, adapter, policy = _system()
    snapshot = adapter.snapshot(policy)
    state = deepcopy(snapshot.state)
    robot = state.by_id("robot_2")
    robot.position = environment.layout.charger_position
    robot.battery = 27.0
    robot.navigation_goal_kind = "charge"
    robot.navigation_goal_position = environment.layout.charger_position
    decision = replace(
        snapshot,
        state=state,
        proposed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        executed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        metadata={
            **dict(snapshot.metadata),
            "decision_evidence_aligned": True,
            "action_resolution": {
                "robot_2": {
                    "requested_action": "WAIT",
                    "executed_action": "WAIT",
                    "environment_changed_action": False,
                    "blocked_reason": None,
                }
            },
            "environment_events": (
                {"event": "charging", "energy_gained": 10.0},
            ),
        },
    )
    facts = adapter.evidence_facts(decision, "robot_2", policy)
    evidence = EvidenceBundle(
        query_plan=QueryPlan(
            raw_text="机器人2为什么要等待？",
            intent=QueryIntent.EXPLANATORY,
            frame_reference=decision.frame,
            subjects=("robot_2",),
            requires_policy_query=True,
            requires_program_trace=True,
            target_variables=("robot_2.observed_action",),
            evidence_requirements=("state", "policy", "program_trace"),
            response_language="zh-CN",
            confidence=1.0,
        ),
        direct_result={
            "target": "robot_2",
            "proposed_action": "WAIT",
            "executed_action": "WAIT",
            "action_descriptions": adapter.action_descriptions(),
        },
        state_facts=tuple(
            {
                "fact_id": fact.fact_id,
                "predicate": fact.predicate,
                "arguments": fact.arguments,
                "value": fact.value,
                "factor_groups": fact.factor_groups,
            }
            for fact in facts
        ),
        policy_results={
            "target": "robot_2",
            "proposed_action": "WAIT",
            "executed_action": "WAIT",
            "action_descriptions": adapter.action_descriptions(),
        },
        # This reproduces the safe fallback used when a local RCPD trace is
        # withheld because it does not agree with the neural action.
        program_trace=(),
    )

    document = ExecutionGroundedExplanationGenerator(
        semantics=adapter
    ).generate_document(
        evidence,
        include_program_trace=False,
        language="zh-CN",
    )

    assert document.text
    assert "机器人2这一步执行了等待" in document.text
    assert "电量为27%" in document.text
    assert "本步后电量升至37%" in document.text
    assert "因此它在充电站等待充电" in document.text
    assert "充电完成后" in document.text
    assert "充电" in document.text


def test_action_constraint_is_natural_language_not_internal_trace_dump() -> None:
    _, adapter, _ = _system()
    text = adapter.explanation_verbalize_unit(
        {
            "predicate": "action_constraint",
            "arguments": ("robot_1",),
            "value": {
                "selected_action": "WAIT",
                "constrained_action": "DOWN",
                "active_reason_features": (
                    "candidate.DOWN.blocked_by_static_obstacle",
                ),
                "observed_meaning": {
                    "explanation_role": "action_feasibility",
                },
            },
        },
        "zh-CN",
    )

    assert text == "向下会被墙或货架阻挡"
    assert "action_constraint" not in text
    assert "candidate." not in text
    assert "{" not in text


def test_charging_explanation_orders_action_reason_energy_and_next_task() -> None:
    _, adapter, _ = _system()
    plan = QueryPlan(
        raw_text="机器人1为什么等待？",
        intent=QueryIntent.EXPLANATORY,
        frame_reference=42,
        subjects=("robot_1",),
        requires_policy_query=True,
        requires_program_trace=True,
        target_variables=("robot_1.observed_action",),
        response_language="zh-CN",
        confidence=1.0,
    )
    actions = {
        "target": "robot_1",
        "proposed_action": "WAIT",
        "executed_action": "WAIT",
        "action_descriptions": adapter.action_descriptions(),
    }
    evidence = EvidenceBundle(
        query_plan=plan,
        direct_result=actions,
        policy_results=actions,
        state_facts=(
            {
                "fact_id": "robot_1.charging_outcome",
                "predicate": "charging_outcome",
                "arguments": ("robot_1",),
                "value": {
                    "battery_before": 16.0,
                    "battery_after": 26.0,
                    "energy_gained": 10.0,
                    "charge_required": True,
                    "next_task": {
                        "kind": "delivery",
                        "task_id": "task_10",
                        "endpoint": (0, 4),
                        "endpoint_kind": "B",
                    },
                },
            },
        ),
        program_trace=(
            {
                "trace_type": "action_constraint",
                "program_selected_action": "WAIT",
                "constrained_action": "DOWN",
                "active_reason_features": (
                    "candidate.DOWN.blocked_by_static_obstacle",
                ),
                "observed_meaning": {
                    "explanation_role": "action_feasibility",
                    "action": "WAIT",
                    "constrained_action": "DOWN",
                    "reason_observations": (
                        {
                            "feature": "candidate.DOWN.blocked_by_static_obstacle",
                            "value": 1.0,
                        },
                    ),
                },
            },
        ),
    )

    text = ExecutionGroundedExplanationGenerator(
        semantics=adapter
    ).generate(
        evidence,
        include_program_trace=True,
        language="zh-CN",
    )

    action_index = text.index("机器人1这一步执行了等待")
    reason_index = text.index("向下会被墙或货架阻挡")
    energy_index = text.index("电量为16%")
    assert action_index < reason_index < energy_index
    assert "本步后电量升至26%" in text
    assert "任务10" in text
    assert "B点(0, 4)" in text
    assert all(
        token not in text
        for token in (
            "action_constraint",
            "candidate.",
            "trace_type",
            "path_index",
            "{",
        )
    )


def test_low_battery_robot_waiting_for_occupied_charger_gets_queue_explanation() -> None:
    environment, adapter, policy = _system()
    snapshot = adapter.snapshot(policy)
    state = deepcopy(snapshot.state)
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    charger_row, charger_column = environment.layout.charger_position
    queue_position = (charger_row - 1, charger_column)
    robot_one.position = queue_position
    robot_one.battery = 10.0
    robot_one.navigation_goal_kind = "charge"
    robot_one.navigation_goal_position = environment.layout.charger_position
    robot_two.position = environment.layout.charger_position
    robot_two.battery = 31.0
    robot_two.navigation_goal_kind = "charge"
    robot_two.navigation_goal_position = environment.layout.charger_position
    decision = replace(
        snapshot,
        state=state,
        proposed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        executed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        metadata={
            **dict(snapshot.metadata),
            "decision_evidence_aligned": True,
            "environment_events": (
                {"event": "charging", "energy_gained": 10.0},
            ),
        },
    )
    facts = adapter.evidence_facts(decision, "robot_1", policy)
    queue = next(
        fact for fact in facts if fact.predicate == "charger_queue_context"
    )
    assert queue.value["battery"] == 10.0
    assert queue.value["occupant_agent"] == "robot_2"
    assert queue.value["occupant_battery_before"] == 31.0
    assert queue.value["occupant_battery_after"] == 41.0

    actions = {
        "target": "robot_1",
        "proposed_action": "WAIT",
        "executed_action": "WAIT",
        "action_descriptions": adapter.action_descriptions(),
    }
    text = ExecutionGroundedExplanationGenerator(semantics=adapter).generate(
        EvidenceBundle(
            query_plan=QueryPlan(
                raw_text="为什么机器人1在这一帧等待？",
                intent=QueryIntent.EXPLANATORY,
                subjects=("robot_1",),
                requires_policy_query=True,
                requires_program_trace=True,
                target_variables=("robot_1.observed_action",),
                evidence_requirements=("study_focus:action",),
                response_language="zh-CN",
                confidence=1.0,
            ),
            direct_result=actions,
            policy_results=actions,
            state_facts=tuple(
                {
                    "fact_id": fact.fact_id,
                    "predicate": fact.predicate,
                    "arguments": fact.arguments,
                    "value": fact.value,
                    "factor_groups": fact.factor_groups,
                }
                for fact in facts
            ),
            program_trace=(
                {
                    "trace_type": "action_constraint",
                    "program_selected_action": "WAIT",
                    "constrained_action": "DOWN",
                    "active_reason_features": (
                        "candidate.DOWN.blocked_by_robot",
                    ),
                    "observed_meaning": {
                        "explanation_role": "action_feasibility",
                        "action": "WAIT",
                        "constrained_action": "DOWN",
                    },
                },
            ),
        ),
        include_program_trace=True,
        language="zh-CN",
    )

    assert "机器人1本步等待，因为它的电量仅有10%，已经需要充电" in text
    assert "唯一的充电站(7, 4)正由机器人2占用" in text
    assert "机器人2本步在站内等待充电，电量从31%升至41%" in text
    assert "只能在(6, 4)等待充电站空出" in text
    assert "会被墙或货架阻挡" not in text
    assert "选择等待的原因之一" not in text


def test_battery_departure_question_reports_threshold_waits_and_next_task() -> None:
    environment, adapter, policy = _system(seed=118)
    state = environment.get_state()
    robot = state.by_id("robot_2")
    task = state.tasks[0]
    robot.position = environment.layout.charger_position
    robot.battery = 11.0
    robot.carrying_task_id = task.task_id
    task.status = "carried"
    task.carrier_agent_id = robot.agent_id
    task.claimed_frame = state.frame
    environment.set_state(state)
    snapshot = adapter.snapshot(policy)
    decision = replace(
        snapshot,
        proposed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        executed_actions={"robot_1": "WAIT", "robot_2": "WAIT"},
        metadata={
            **dict(snapshot.metadata),
            "decision_evidence_aligned": True,
            "environment_events": ({"event": "charging", "energy_gained": 10.0},),
        },
    )
    fact = next(
        item
        for item in adapter.evidence_facts(decision, "robot_2", policy)
        if item.predicate == "energy_decision_context"
    )
    value = {**fact.value, "study_focus": "charge_threshold"}
    english = adapter.explanation_verbalize_unit(
        {
            "predicate": fact.predicate,
            "arguments": fact.arguments,
            "value": value,
        },
        "en",
    )
    chinese = adapter.explanation_verbalize_unit(
        {
            "predicate": fact.predicate,
            "arguments": fact.arguments,
            "value": value,
        },
        "zh-CN",
    )
    assert value["minimum_safe_departure_battery"] == 36.0
    assert value["charge_waits_remaining"] == 3
    assert value["projected_departure_battery"] == 41.0
    assert all(token in english for token in ("11%", "at least 36%", "3 more", "41%", "task 1"))
    assert all(token in chinese for token in ("11%", "至少需要36%", "3次", "41%", "任务1"))
    assert _study_question_focus("why did robot 2 stop") == "action"
    assert _study_question_focus(
        "what battery does robot 2 try to hold before trying to go out again?"
    ) == "charge_threshold"


def test_all_move_directions_explain_shared_task_progress_without_trace() -> None:
    directions = {
        "UP": ((3, 4), (2, 4), "向上"),
        "DOWN": ((2, 4), (3, 4), "向下"),
        "LEFT": ((1, 4), (1, 3), "向左"),
        "RIGHT": ((2, 4), (2, 5), "向右"),
    }
    for action, (origin, delivery_position, action_label) in directions.items():
        _, adapter, policy = _system()
        snapshot = adapter.snapshot(policy)
        state = deepcopy(snapshot.state)
        robot = state.by_id("robot_2")
        robot.position = origin
        task = state.tasks[0]
        task.delivery_position = delivery_position
        task.status = "carried"
        task.carrier_agent_id = "robot_2"
        robot.carrying_task_id = task.task_id
        robot.navigation_goal_kind = "delivery"
        robot.navigation_goal_position = delivery_position
        decision = replace(
            snapshot,
            state=state,
            proposed_actions={"robot_1": "WAIT", "robot_2": action},
            executed_actions={"robot_1": "WAIT", "robot_2": action},
            metadata={
                **dict(snapshot.metadata),
                "decision_evidence_aligned": True,
            },
        )
        facts = adapter.evidence_facts(decision, "robot_2", policy)
        movement = next(
            fact for fact in facts if fact.predicate == "movement_outcome"
        )
        assert movement.value["distance_before"] == 1
        assert movement.value["distance_after"] == 0

        actions = {
            "target": "robot_2",
            "proposed_action": action,
            "executed_action": action,
            "action_descriptions": adapter.action_descriptions(),
        }
        text = ExecutionGroundedExplanationGenerator(
            semantics=adapter
        ).generate(
            EvidenceBundle(
                query_plan=QueryPlan(
                    raw_text="为什么执行这个动作？",
                    intent=QueryIntent.EXPLANATORY,
                    frame_reference=decision.frame,
                    subjects=("robot_2",),
                    requires_policy_query=True,
                    requires_program_trace=True,
                    target_variables=("robot_2.observed_action",),
                    response_language="zh-CN",
                    confidence=1.0,
                ),
                direct_result=actions,
                policy_results=actions,
                state_facts=tuple(
                    {
                        "fact_id": fact.fact_id,
                        "predicate": fact.predicate,
                        "arguments": fact.arguments,
                        "value": fact.value,
                        "factor_groups": fact.factor_groups,
                    }
                    for fact in facts
                ),
                program_trace=(),
            ),
            include_program_trace=False,
            language="zh-CN",
        )

        assert f"机器人2这一步执行了{action_label}" in text
        assert "剩余距离从1格缩短到0格" in text
        assert "从而推进交付" in text
        assert all(
            token not in text
            for token in ("navigation_goal", "goal_kind", "candidate.", "{")
        )


def test_low_battery_carrier_move_explains_charging_before_delivery() -> None:
    environment, adapter, policy = _system()
    snapshot = adapter.snapshot(policy)
    state = deepcopy(snapshot.state)
    robot = state.by_id("robot_2")
    charger_row, charger_column = environment.layout.charger_position
    robot.position = (charger_row, charger_column - 1)
    state.by_id("robot_1").position = (charger_row - 1, charger_column - 1)
    robot.battery = 18.0
    task = state.tasks[0]
    task.delivery_position = (0, 4)
    task.status = "carried"
    task.carrier_agent_id = robot.agent_id
    robot.carrying_task_id = task.task_id
    robot.navigation_goal_kind = "charge"
    robot.navigation_goal_position = environment.layout.charger_position
    decision = replace(
        snapshot,
        state=state,
        proposed_actions={"robot_1": "WAIT", "robot_2": "RIGHT"},
        executed_actions={"robot_1": "WAIT", "robot_2": "RIGHT"},
        metadata={
            **dict(snapshot.metadata),
            "decision_evidence_aligned": True,
        },
    )
    facts = adapter.evidence_facts(decision, "robot_2", policy)
    movement = next(
        fact for fact in facts if fact.predicate == "movement_outcome"
    )
    assert movement.value["work"]["kind"] == "charge"
    assert movement.value["distance_before"] == 1
    assert movement.value["distance_after"] == 0

    actions = {
        "target": "robot_2",
        "proposed_action": "RIGHT",
        "executed_action": "RIGHT",
        "action_descriptions": adapter.action_descriptions(),
    }
    text = ExecutionGroundedExplanationGenerator(semantics=adapter).generate(
        EvidenceBundle(
            query_plan=QueryPlan(
                raw_text="为什么机器人2向右移动？",
                intent=QueryIntent.EXPLANATORY,
                frame_reference=decision.frame,
                subjects=("robot_2",),
                requires_policy_query=True,
                requires_program_trace=True,
                target_variables=("robot_2.observed_action",),
                response_language="zh-CN",
                confidence=1.0,
            ),
            direct_result=actions,
            policy_results=actions,
            state_facts=tuple(
                {
                    "fact_id": fact.fact_id,
                    "predicate": fact.predicate,
                    "arguments": fact.arguments,
                    "value": fact.value,
                    "factor_groups": fact.factor_groups,
                }
                for fact in facts
            ),
            program_trace=(),
        ),
        include_program_trace=False,
        language="zh-CN",
    )

    assert "机器人2这一步执行了向右" in text
    assert "电量为18%" in text
    assert "到充电站的剩余距离从1格缩短到0格" in text
    assert f"充电后继续将任务{task.task_id.removeprefix('task_')}送往B点(0, 4)" in text
    assert all(
        token not in text
        for token in ("navigation_goal", "goal_kind", "candidate.", "{")
    )


def test_nonprogress_move_reports_verified_policy_choice_without_false_progress() -> None:
    _, adapter, _ = _system()
    text = adapter.explanation_verbalize_unit(
        {
            "predicate": "movement_outcome",
            "arguments": ("robot_2",),
            "value": {
                "action": "DOWN",
                "position_before": (2, 4),
                "position_after": (3, 4),
                "distance_before": 3,
                "distance_after": 4,
                "selected_probability": 0.5605789,
                "policy_selected": True,
                "work": {
                    "kind": "delivery",
                    "task_id": "task_1",
                    "endpoint": (1, 2),
                },
            },
        },
        "zh-CN",
    )

    assert "它本帧执行了向下" in text
    assert "执行后这一步没有缩短" in text
    assert "选择概率最高" not in text
    assert "没有缩短" in text
    assert "剩余距离从3格变为4格" in text
    assert "推进交付" not in text


def test_task_question_only_reports_task_effect_and_omits_other_explanation_dimensions() -> None:
    _, adapter, _ = _system()
    actions = {
        "target": "robot_2",
        "proposed_action": "UP",
        "executed_action": "UP",
        "action_descriptions": adapter.action_descriptions(),
    }
    evidence = EvidenceBundle(
        query_plan=QueryPlan(
            raw_text="这个动作如何影响当前配送任务？",
            intent=QueryIntent.EXPLANATORY,
            subjects=("robot_2",),
            requires_policy_query=True,
            requires_program_trace=True,
            target_variables=("robot_2.observed_action",),
            evidence_requirements=("study_focus:task",),
            response_language="zh-CN",
            confidence=1.0,
        ),
        direct_result=actions,
        policy_results=actions,
        state_facts=(
            {
                "fact_id": "robot_2.movement_outcome",
                "predicate": "movement_outcome",
                "arguments": ("robot_2",),
                "value": {
                    "action": "UP",
                    "position_before": (4, 4),
                    "position_after": (3, 4),
                    "distance_before": 5,
                    "distance_after": 4,
                    "selected_probability": 0.7,
                    "policy_selected": True,
                    "work": {
                        "kind": "delivery",
                        "task_id": "task_1",
                        "endpoint": (1, 2),
                    },
                },
            },
        ),
        program_trace=(
            {
                "trace_type": "action_constraint",
                "program_selected_action": "UP",
                "constrained_action": "DOWN",
                "active_reason_features": (
                    "candidate.DOWN.blocked_by_static_obstacle",
                ),
                "observed_meaning": {
                    "explanation_role": "action_feasibility",
                    "action": "UP",
                    "constrained_action": "DOWN",
                },
            },
        ),
    )

    text = ExecutionGroundedExplanationGenerator(semantics=adapter).generate(
        evidence,
        include_program_trace=True,
        language="zh-CN",
    )

    assert text == (
        "它从(4, 4)移动到(3, 4)，使到任务1的B点(1, 2)的剩余距离"
        "从5格缩短到4格，从而推进交付。"
    )
    assert "墙或货架" not in text
    assert "电量" not in text
    assert "协作" not in text


def test_energy_question_reports_current_battery_and_whether_it_constrained_action() -> None:
    _, adapter, policy = _system()
    snapshot = adapter.snapshot(policy)
    state = deepcopy(snapshot.state)
    robot = state.by_id("robot_2")
    robot.battery = 88.0
    task = state.tasks[0]
    robot.navigation_goal_kind = "pickup"
    robot.navigation_goal_position = task.pickup_position
    decision = replace(
        snapshot,
        state=state,
        proposed_actions={"robot_1": "WAIT", "robot_2": "UP"},
        executed_actions={"robot_1": "WAIT", "robot_2": "UP"},
        metadata={**dict(snapshot.metadata), "decision_evidence_aligned": True},
    )
    facts = adapter.evidence_facts(decision, "robot_2", policy)
    actions = {
        "target": "robot_2",
        "proposed_action": "UP",
        "executed_action": "UP",
        "action_descriptions": adapter.action_descriptions(),
    }
    text = ExecutionGroundedExplanationGenerator(semantics=adapter).generate(
        EvidenceBundle(
            query_plan=QueryPlan(
                raw_text="当前电量和充电需求如何影响了这个动作？",
                intent=QueryIntent.EXPLANATORY,
                frame_reference=decision.frame,
                subjects=("robot_2",),
                requires_policy_query=True,
                requires_program_trace=True,
                target_variables=("robot_2.observed_action",),
                evidence_requirements=("study_focus:energy",),
                response_language="zh-CN",
                confidence=1.0,
            ),
            direct_result=actions,
            policy_results=actions,
            state_facts=tuple(
                {
                    "fact_id": fact.fact_id,
                    "predicate": fact.predicate,
                    "arguments": fact.arguments,
                    "value": fact.value,
                    "factor_groups": fact.factor_groups,
                }
                for fact in facts
            ),
            program_trace=(
                {
                    "trace_type": "action_constraint",
                    "program_selected_action": "UP",
                    "constrained_action": "DOWN",
                    "active_reason_features": (
                        "candidate.DOWN.blocked_by_static_obstacle",
                    ),
                    "observed_meaning": {
                        "explanation_role": "action_feasibility",
                        "action": "UP",
                        "constrained_action": "DOWN",
                    },
                },
            ),
        ),
        include_program_trace=True,
        language="zh-CN",
    )

    assert "执行向上时机器人2的当前电量为88%" in text
    assert "每次成功移动只消耗2点" in text
    assert "电量没有迫使它转去充电" in text
    assert "当前协作状态" not in text
    assert "墙或货架" not in text


def test_collaboration_and_allocation_questions_receive_separate_answers() -> None:
    _, adapter, policy = _system()
    snapshot = adapter.snapshot(policy)
    state = deepcopy(snapshot.state)
    task_one, task_two = state.tasks[:2]
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    task_two.status = "carried"
    task_two.carrier_agent_id = robot_one.agent_id
    robot_one.carrying_task_id = task_two.task_id
    robot_one.navigation_goal_kind = "delivery"
    robot_one.navigation_goal_position = task_two.delivery_position
    robot_two.navigation_goal_kind = "pickup"
    robot_two.navigation_goal_position = task_one.pickup_position
    decision = replace(
        snapshot,
        state=state,
        proposed_actions={"robot_1": "RIGHT", "robot_2": "UP"},
        executed_actions={"robot_1": "RIGHT", "robot_2": "UP"},
        metadata={**dict(snapshot.metadata), "decision_evidence_aligned": True},
    )
    facts = adapter.evidence_facts(decision, "robot_2", policy)
    actions = {
        "target": "robot_2",
        "proposed_action": "UP",
        "executed_action": "UP",
        "action_descriptions": adapter.action_descriptions(),
    }
    evidence = EvidenceBundle(
        query_plan=QueryPlan(
                raw_text="队友的位置或动作是否影响了这个决定？",
                intent=QueryIntent.EXPLANATORY,
                frame_reference=decision.frame,
                subjects=("robot_2",),
                requires_policy_query=True,
                requires_program_trace=True,
                target_variables=("robot_2.observed_action",),
                evidence_requirements=("study_focus:collaboration",),
                response_language="zh-CN",
                confidence=1.0,
        ),
        direct_result=actions,
        policy_results=actions,
        state_facts=tuple(
            {
                "fact_id": fact.fact_id,
                "predicate": fact.predicate,
                "arguments": fact.arguments,
                "value": fact.value,
                "factor_groups": fact.factor_groups,
            }
            for fact in facts
        ),
        program_trace=(
            {
                "trace_type": "action_constraint",
                "program_selected_action": "UP",
                "constrained_action": "DOWN",
                "active_reason_features": (
                    "candidate.DOWN.blocked_by_static_obstacle",
                ),
                "observed_meaning": {
                    "explanation_role": "action_feasibility",
                    "action": "UP",
                    "constrained_action": "DOWN",
                },
            },
        ),
    )
    generator = ExecutionGroundedExplanationGenerator(semantics=adapter)
    text = generator.generate(
        evidence,
        include_program_trace=True,
        language="zh-CN",
    )
    allocation_text = generator.generate(
        replace(
            evidence,
            query_plan=replace(
                evidence.query_plan,
                raw_text="为什么任务1的A点由机器人2去取？",
                evidence_requirements=("study_focus:allocation",),
            ),
        ),
        include_program_trace=True,
        language="zh-CN",
    )

    assert "没有直接影响" in text
    assert "机器人1位于" in text
    assert "执行向右" in text
    assert "任务1" not in text and "任务2" not in text
    assert "墙或货架" not in text
    assert f"机器人2前往任务1的A点{task_one.pickup_position}取货" in allocation_text
    assert f"机器人1已经承运任务2" in allocation_text
    assert "不能再认领另一项任务" in allocation_text
    assert "没有直接影响" not in allocation_text


def test_explanation_cache_isolated_by_study_question_focus() -> None:
    plan = QueryPlan(
        raw_text="为什么？",
        intent=QueryIntent.EXPLANATORY,
        subjects=("robot_2",),
        requires_policy_query=True,
        target_variables=("robot_2.observed_action",),
        evidence_requirements=("state", "study_focus:energy"),
        response_language="zh-CN",
        confidence=1.0,
    )
    direct = {
        "target": "robot_2",
        "proposed_action": "UP",
        "executed_action": "UP",
    }
    energy = EvidenceBundle(
        query_plan=plan,
        direct_result=direct,
        policy_results=direct,
    )
    collaboration = replace(
        energy,
        query_plan=replace(
            plan,
            evidence_requirements=("state", "study_focus:collaboration"),
        ),
    )

    def cache_key(evidence: EvidenceBundle) -> str:
        return _common_ir_cache_key(
            evidence,
            language="zh-CN",
            policy_hash="policy",
            program_hash="program",
            environment_id="warehouse",
        )

    assert cache_key(energy) != cache_key(collaboration)


def test_joint_assignment_explanation_uses_plain_language_not_internal_cost_units() -> None:
    _, adapter, _ = _system()
    text = adapter.explanation_verbalize_unit(
        {
            "predicate": "collaboration_context",
            "arguments": ("robot_2", "robot_1"),
            "value": {
                "target_agent": "robot_2",
                "teammate_agent": "robot_1",
                "target_role": {
                    "kind": "pickup",
                    "task_slot": 1,
                    "endpoint": (2, 5),
                },
                "teammate_role": {
                    "kind": "pickup",
                    "task_slot": 2,
                    "endpoint": (6, 4),
                },
                "joint_selected_safe_actions": 43.0,
                "joint_swapped_safe_actions": 45.0,
                "joint_selected_breakdown": {
                    "assignments": (
                        {
                            "agent_id": "robot_2",
                            "task_slot": 1,
                            "travel_cells": 20,
                            "charge_waits": 0,
                            "current_battery": 49.0,
                            "route_legs": (
                                {"kind": "current_to_pickup", "cells": 7},
                                {"kind": "pickup_to_delivery", "cells": 8},
                                {"kind": "delivery_to_charger", "cells": 5},
                            ),
                        },
                        {
                            "agent_id": "robot_1",
                            "task_slot": 2,
                            "travel_cells": 19,
                            "charge_waits": 0,
                            "current_battery": 88.0,
                            "route_legs": (
                                {"kind": "current_to_pickup", "cells": 6},
                                {"kind": "pickup_to_delivery", "cells": 7},
                                {"kind": "delivery_to_charger", "cells": 6},
                            ),
                        },
                    ),
                    "total_travel_cells": 39,
                    "total_charge_waits": 0,
                },
                "joint_swapped_breakdown": {
                    "assignments": (
                        {
                            "agent_id": "robot_2",
                            "task_slot": 2,
                            "travel_cells": 21,
                            "charge_waits": 0,
                            "current_battery": 49.0,
                            "route_legs": (
                                {"kind": "current_to_pickup", "cells": 8},
                                {"kind": "pickup_to_delivery", "cells": 7},
                                {"kind": "delivery_to_charger", "cells": 6},
                            ),
                        },
                        {
                            "agent_id": "robot_1",
                            "task_slot": 1,
                            "travel_cells": 20,
                            "charge_waits": 0,
                            "current_battery": 88.0,
                            "route_legs": (
                                {"kind": "current_to_pickup", "cells": 7},
                                {"kind": "pickup_to_delivery", "cells": 8},
                                {"kind": "delivery_to_charger", "cells": 5},
                            ),
                        },
                    ),
                    "total_travel_cells": 41,
                    "total_charge_waits": 0,
                },
                "teammate_constrained_actions": (),
            },
        },
        "zh-CN",
    )

    assert "机器人2负责任务1：当前位置到A点7格、A点到B点8格、B点到充电站5格" in text
    assert "机器人1负责任务2：当前位置到A点6格、A点到B点7格、B点到充电站6格" in text
    assert "团队合计行驶39格，无需额外充电等待" in text
    assert "交换任务后" in text
    assert "团队合计行驶41格，无需额外充电等待" in text
    assert "以上行驶距离包括取货、交付以及交付后前往充电站" in text
    assert "当前分工少走2格" in text
    assert "当前电量49%" in text
    assert "安全动作" not in text
    assert "43" not in text and "45" not in text


def test_semantic_context_separates_static_and_robot_blocking() -> None:
    _, adapter, policy = _system()
    base = adapter.snapshot(policy)

    static_state = deepcopy(base.state)
    static_target = static_state.by_id("robot_2")
    static_target.position = (1, 2)
    static_target.navigation_goal_position = (4, 2)
    static_target.navigation_goal_kind = "pickup"
    static_state.by_id("robot_1").position = (9, 1)
    static_snapshot = replace(base, state=static_state)

    robot_state = deepcopy(base.state)
    robot_target = robot_state.by_id("robot_2")
    robot_target.position = (2, 3)
    robot_target.navigation_goal_position = (4, 3)
    robot_target.navigation_goal_kind = "pickup"
    robot_state.by_id("robot_1").position = (3, 3)
    robot_snapshot = replace(base, state=robot_state)

    static = adapter.semantic_policy_context(static_snapshot, "robot_2")
    robot = adapter.semantic_policy_context(robot_snapshot, "robot_2")
    prefix = "candidate.DOWN"
    assert static.features[f"{prefix}.geometric_goal_progress"] == 1.0
    assert static.features[f"{prefix}.blocked_by_static_obstacle"] == 1.0
    assert static.features[f"{prefix}.blocked_by_robot"] == 0.0
    assert robot.features[f"{prefix}.geometric_goal_progress"] == 1.0
    assert robot.features[f"{prefix}.blocked_by_static_obstacle"] == 0.0
    assert robot.features[f"{prefix}.blocked_by_robot"] == 1.0
    assert robot.entity_bindings[f"{prefix}.blocker"] == "robot_1"


def test_current_intervention_contract_recomputes_observations_and_evidence() -> None:
    _, adapter, policy = _system()
    snapshot = replace(
        adapter.snapshot(policy),
        metadata={"decision_evidence_aligned": True},
    )
    edited = adapter.apply_interventions(
        snapshot,
        (Intervention("robot_2", "battery", 37.0),),
    )

    assert edited.state.by_id("robot_2").battery == 37.0
    assert edited.metadata["decision_evidence_aligned"] is False
    assert edited.action_distributions == {}
    assert edited.observations["robot_2"].shape == (
        observation_dim(_legacy_config()),
    )


def test_shared_objective_evidence_is_bound_to_live_task_and_frame() -> None:
    environment, adapter, policy = _system(seed=77)
    state = environment.get_state()
    task = state.tasks[0]
    state.by_id("robot_2").position = task.pickup_position
    environment.set_state(state)
    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    snapshot = adapter.snapshot(policy)

    reason = next(
        fact
        for fact in adapter.evidence_facts(snapshot, "robot_2", policy)
        if fact.fact_id == "robot_2.objective_reason"
    )
    assert reason.predicate == "shared_objective_selection_reason"
    assert reason.value["schema"] == "shared_objective_selection_reason.v2"
    assert reason.value["evidence_frame"] == snapshot.frame
    assert reason.value["selected_objective"]["task_id"] == task.task_id
    assert reason.value["task_state"]["carrying_task_id"] == task.task_id


def test_recorded_collision_resolution_is_available_as_action_evidence() -> None:
    environment, adapter, policy = _system()
    state = environment.get_state()
    state.by_id("robot_1").position = (1, 2)
    state.by_id("robot_2").position = (1, 4)
    environment.set_state(state)
    before = adapter.snapshot(policy)
    _, _, _, _, info = environment.step(
        {"robot_1": "RIGHT", "robot_2": "LEFT"}
    )
    decision = replace(
        before,
        proposed_actions={"robot_1": "RIGHT", "robot_2": "LEFT"},
        executed_actions=info["executed_actions"],
        metadata={
            **dict(before.metadata),
            "decision_evidence_aligned": True,
            "action_resolution": info["action_resolution"],
        },
    )

    fact = next(
        item
        for item in adapter.evidence_facts(decision, "robot_2", policy)
        if item.fact_id == "robot_2.action_resolution"
    )
    assert fact.value["blocked_reason"] == "same_target"
    assert fact.value["executed_action"] == "WAIT"


def test_local_state_sampler_returns_complete_valid_two_robot_states() -> None:
    environment, adapter, policy = _system()
    samples = adapter.sample_local_policy_states(
        adapter.snapshot(policy),
        "robot_2",
        count=12,
        seed=5,
    )

    assert len(samples) == 12
    assert all(not environment.validate_state(item.state) for item in samples)
    assert all(len(item.state.agents) == 2 for item in samples)
    assert all(len(item.state.tasks) == 2 for item in samples)


def test_warehouse_question_vocabulary_has_no_personal_mission_fields() -> None:
    _, adapter, _ = _system()
    serialized = str(adapter.question_vocabulary()).lower()

    assert "observed_action" in serialized
    assert "objective" in serialized
    assert "carrying_item" not in serialized
    assert "delivery_exit" not in serialized
    assert "quota" not in serialized


def test_rcpd_distils_current_shared_warehouse_actor() -> None:
    environment, adapter, policy = _system()
    states: list[WarehousePolicyState] = []
    observations = environment.observations()
    for _ in range(6):
        snapshot = adapter.snapshot(policy)
        states.extend(
            adapter.policy_state(snapshot, agent_id)
            for agent_id in environment.agent_ids
        )
        actions, _ = policy.act(
            observations,
            environment.global_state(),
            deterministic=False,
        )
        observations, _, terminated, truncated, _ = environment.step(actions)
        if terminated or truncated:
            break

    result = RCPD(
        RCPDConfig(max_depth=3, min_samples_leaf=2, random_seed=23)
    ).fit(
        states,
        lambda item: adapter.policy_distribution(
            item.snapshot,
            item.agent_id,
            policy,
        ),
        adapter.relational_features,
        validation_states=states,
    )

    assert result.metrics.sample_count == len(states)
    assert result.metrics.action_regret is not None
    assert set(result.program.action_names) == {
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "WAIT",
    }
