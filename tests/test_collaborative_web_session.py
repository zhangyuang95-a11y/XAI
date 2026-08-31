from __future__ import annotations

import pytest
import torch

from backend.adapters.warehouse import WarehouseAdapter
from env.warehouse.environment import ACTIONS, WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from ui.collaborative_study import CollaborativeConditionAllocator
from ui.tutorial import TutorialTrajectory
from ui.web_runtime import WarehouseWebSession, serialize_warehouse_state


class _NoopEngine:
    def precompile_frame_irs(self, _snapshot) -> None:
        return None


def _wait_policy() -> MAPPOPolicy:
    policy = MAPPOPolicy(
        WarehouseConfig(horizon=120),
        MAPPOConfig(hidden_dim=16, seed=2),
    )
    with torch.no_grad():
        for parameter in policy.network.actor.parameters():
            parameter.zero_()
        policy.network.actor[-1].bias[ACTIONS.index("WAIT")] = 10
    return policy


def _tutorial(policy: MAPPOPolicy, *, horizon: int = 1) -> TutorialTrajectory:
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=40_221)
    frames = WarehouseAdapter(environment).rollout(
        policy,
        horizon=horizon,
        deterministic=True,
    ).frames
    return TutorialTrajectory(
        seed=40_221,
        focus_agent="robot_2",
        frames=frames,
        milestones=(("demonstration_complete", 1, "robot_2"),),
    )


def _one_frame_tutorial(policy: MAPPOPolicy) -> TutorialTrajectory:
    return _tutorial(policy, horizon=1)


def test_public_web_state_uses_shared_tasks_not_personal_missions() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=9)
    payload = serialize_warehouse_state(
        environment.get_state(),
        selected_agent="robot_1",
        reveal_policy=False,
    )
    assert len(payload["tasks"]) == 2
    forbidden = {
        "pickup_shelf",
        "pickup_access",
        "home_position",
        "delivery_quota",
        "remaining_deliveries",
        "minimum_final_battery",
        "mission_ready",
    }
    assert not forbidden.intersection(payload["agents"][0])
    assert "delivery_target" not in payload
    assert "mission_ready_count" not in payload


def test_tutorial_can_end_early_and_starts_a_fresh_task1_with_audit() -> None:
    policy = _wait_policy()
    assignment = CollaborativeConditionAllocator(
        randomization_seed=91,
    )._assignment_for_index("early-demo", 0)
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=17,
        tutorial=_tutorial(policy, horizon=4),
    )
    session.start_study(assignment=assignment, language="zh-CN")
    assert "begin_task1" in session.view()["study"]["allowed_commands"]
    session.tutorial_advance()
    session.tutorial_advance()
    assert session.tutorial_complete is False

    task1 = session.begin_task1()["view"]

    assert session.human_study.stage == "task1"
    assert task1["state"]["frame"] == 0
    assert task1["timeline"]["trajectory_seed"] == assignment.task1_seed
    assert all(agent["battery"] == 100.0 for agent in task1["state"]["agents"])
    events = session.drain_events()
    early = next(event for event in events if event["event"] == "tutorial_ended_early")
    acknowledged = next(
        event for event in events if event["event"] == "tutorial_acknowledged"
    )
    assert not any(event["event"] == "tutorial_completed" for event in events)
    assert early == {
        "event": "tutorial_ended_early",
        "tutorial_seed": 40_221,
        "trajectory_kind": "continuous_partial_mission",
        "last_displayed_frame": 2,
        "displayed_frames": 3,
        "total_frames": 5,
        "remaining_frames": 2,
        "completion_fraction": 0.6,
    }
    assert acknowledged["tutorial_completed"] is False
    assert acknowledged["ended_early"] is True
    assert acknowledged["last_displayed_frame"] == 2
    assert acknowledged["displayed_frames"] == 3
    assert acknowledged["total_frames"] == 5
    assert acknowledged["remaining_frames"] == 2
    assert acknowledged["completion_fraction"] == 0.6


def test_completed_tutorial_audit_keeps_tutorial_frame_count_after_task_reset() -> None:
    policy = _wait_policy()
    assignment = CollaborativeConditionAllocator(
        randomization_seed=91,
    )._assignment_for_index("complete-demo", 1)
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=18,
        tutorial=_tutorial(policy, horizon=2),
    )
    session.start_study(assignment=assignment, language="en")
    session.tutorial_advance()
    session.tutorial_advance()

    session.begin_task1()

    events = session.drain_events()
    acknowledged = next(
        event for event in events if event["event"] == "tutorial_acknowledged"
    )
    assert acknowledged["tutorial_completed"] is True
    assert acknowledged["ended_early"] is False
    assert acknowledged["displayed_frames"] == 3
    assert acknowledged["total_frames"] == 3
    assert acknowledged["remaining_frames"] == 0
    assert acknowledged["completion_fraction"] == 1.0
    assert not any(event["event"] == "tutorial_ended_early" for event in events)


def test_group_a_live_questions_preserve_task1_state_and_direct_round_boundaries(
    tmp_path,
) -> None:
    policy = _wait_policy()
    assignment = next(
        CollaborativeConditionAllocator(
            randomization_seed=91,
        )._assignment_for_index(f"p{index}", index)
        for index in range(8)
        if CollaborativeConditionAllocator(
            randomization_seed=91,
        )._assignment_for_index(f"p{index}", index).condition == "explanation"
    )
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=5,
        tutorial=_one_frame_tutorial(policy),
    )
    session.start_study(assignment=assignment, language="en")
    assert session.view()["study"]["group_code"] == "A"
    assert session.view()["study"]["group_explanation_available"] is True
    session.tutorial_advance()
    assert session.tutorial_complete
    session.begin_task1()
    live = session.submit_human_action("RIGHT")["view"]
    assert live["transition"]["from_frame"] == 0
    assert live["transition"]["to_frame"] == 1
    live_motion = {
        item["id"]: item for item in live["transition"]["agents"]
    }
    assert live_motion["robot_1"]["proposed_action"] == "RIGHT"
    assert live_motion["robot_1"]["battery_delta"] == -2.0
    assert live_motion["robot_2"]["proposed_action"] is None
    before_question = session.environment.get_state()
    before_count = session.timeline.count
    response = session.explain_study(
        "What is Robot 2 trying to do?",
        target_agent="robot_2",
        question_kind="goal",
    )
    assert response["report"]["trajectory_kind"] == "human_ai_task1"
    assert response["report"]["agent_control"] == {
        "robot_1": "human", "robot_2": "ai",
    }
    assert session.environment.get_state() == before_question
    assert session.timeline.count == before_count
    for _ in range(119):
        session.submit_human_action("WAIT")
    assert session.human_study.stage == "task1_complete"
    assert session.human_study.round_summaries["task1"].steps == 120
    assert session.timeline.count == 121
    assert session.human_study.explanation_count == 1
    session.begin_task2()
    for _ in range(120):
        session.submit_human_action("WAIT")
    assert session.human_study.stage == "survey"
    assert session.human_study.round_summaries["task2"].steps == 120
    session.submit_survey(
        {
            "coordination_understanding": 4,
            "ai_predictability": 4,
            "interface_clarity": 4,
            "comment": "",
        }
    )
    assert session.human_study.stage == "completed"
    assert session.human_study.score_delta is not None


def test_control_transition_preserves_task1_until_task2_is_confirmed(tmp_path) -> None:
    policy = _wait_policy()
    allocator = CollaborativeConditionAllocator(randomization_seed=91)
    assignment = next(
        allocator._assignment_for_index(f"control-{index}", index)
        for index in range(8)
        if allocator._assignment_for_index(f"control-{index}", index).condition
        == "control"
    )
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=8,
        tutorial=_one_frame_tutorial(policy),
    )
    session.start_study(assignment=assignment, language="zh-CN")
    assert session.view()["study"]["group_code"] == "B"
    assert session.view()["study"]["group_explanation_available"] is False
    session.tutorial_advance()
    session.begin_task1()
    for _ in range(120):
        result = session.submit_human_action("WAIT")

    transition_view = result["view"]
    task1_state = session.environment.get_state()
    task1_tasks = tuple(
        (task.task_id, task.pickup_position, task.delivery_position)
        for task in task1_state.tasks
    )
    assert session.human_study.stage == "task1_complete"
    assert transition_view["state"]["frame"] == 120
    assert transition_view["study"]["progress"] == 120
    assert transition_view["study"]["condition"] == "control"
    assert transition_view["study"]["allowed_commands"] == [
        "set_language", "restart", "begin_task2",
    ]
    assert transition_view["transition"]["loop"] is False
    assert transition_view["state"]["policy_hidden"] is True
    assert session.timeline.count == 121
    with pytest.raises(RuntimeError):
        session.submit_human_action("WAIT")
    with pytest.raises(RuntimeError):
        session.select_frame(20)
    with pytest.raises(RuntimeError):
        session.explain_study("Why did robot 2 wait?")
    assert session.environment.get_state() == task1_state
    assert tuple(
        (task.task_id, task.pickup_position, task.delivery_position)
        for task in session.environment.get_state().tasks
    ) == task1_tasks

    task2 = session.begin_task2()["view"]
    assert session.human_study.stage == "task2"
    assert task2["state"]["frame"] == 0
    assert all(agent["battery"] == 100.0 for agent in task2["state"]["agents"])
    assert session.timeline.count == 1
    assert assignment.task2_seed != assignment.task1_seed
    assert tuple(
        (task.task_id, task.pickup_position, task.delivery_position)
        for task in session.environment.get_state().tasks
    ) != task1_tasks
    assert session.human_study.round_summaries["task1"].seed == assignment.task1_seed


def test_explanation_questions_bind_recent_task1_action_and_robot2_only() -> None:
    policy = _wait_policy()
    allocator = CollaborativeConditionAllocator(randomization_seed=91)
    assignment = next(
        allocator._assignment_for_index(f"explain-{index}", index)
        for index in range(8)
        if allocator._assignment_for_index(f"explain-{index}", index).condition
        == "explanation"
    )
    tutorial = _tutorial(policy, horizon=4)
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=12,
        tutorial=tutorial,
    )
    session.run_id = "question-seed-test"
    session.start_study(assignment=assignment, language="zh-CN")
    for _ in range(4):
        session.tutorial_advance()
    session.begin_task1()
    for _ in range(4):
        session.submit_human_action("WAIT")
    before = session.environment.get_state()
    response = session.explain_study(
        "机器人2为什么等待？",
        target_agent="robot_2",
        question_kind="wait",
    )
    report = response["report"]
    assert report["target_agent"] == "robot_2"
    assert report["trajectory_kind"] == "human_ai_task1"
    assert report["agent_control"] == {"robot_1": "human", "robot_2": "ai"}
    assert report["anchor_frame"] <= session.current.snapshot.frame
    assert max(report["context_frames"]) <= report["anchor_frame"]
    assert len(report["context_frames"]) <= 5
    assert session.environment.get_state() == before
    with pytest.raises(ValueError, match="Robot 2"):
        session.explain_study(
            "Why?", target_agent="robot_1", question_kind="action"
        )
    events = session.drain_events()
    assert any(
        event["event"] == "live_explanation_presented"
        and event["trajectory_kind"] == "human_ai_task1"
        for event in events
    )


def test_reference_trajectory_is_one_immutable_public_121_frame_payload() -> None:
    policy = _wait_policy()
    allocator = CollaborativeConditionAllocator(randomization_seed=92)
    assignment = next(
        allocator._assignment_for_index(f"reference-{index}", index)
        for index in range(8)
        if allocator._assignment_for_index(f"reference-{index}", index).condition
        == "explanation"
    )
    tutorial = _tutorial(policy, horizon=120)
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=13,
        tutorial=tutorial,
    )
    session.run_id = "reference-payload-test"
    session.start_study(assignment=assignment, language="en")
    demonstration = session.reference_trajectory_payload()
    assert demonstration["trajectory_kind"] == "ai_ai_demonstration"
    assert demonstration["agent_control"] == {"robot_1": "ai", "robot_2": "ai"}
    session.begin_task1()
    assert session.view()["timeline"]["trajectory_kind"] == "human_ai_task1"
    with pytest.raises(RuntimeError, match="instructions"):
        session.reference_trajectory_payload()
