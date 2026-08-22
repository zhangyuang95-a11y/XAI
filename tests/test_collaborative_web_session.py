from __future__ import annotations

import pytest
import torch

from backend.adapters.warehouse import WarehouseAdapter
from backend.nlp.schemas import QueryIntent, QueryPlan
from env.warehouse.environment import ACTIONS, WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from ui.collaborative_study import CollaborativeConditionAllocator
from ui.tutorial import TutorialTrajectory
from ui.web_runtime import WarehouseWebSession, serialize_warehouse_state


class _NoopEngine:
    def precompile_frame_irs(self, _snapshot) -> None:
        return None


class _ExplanationAnswer:
    user_visible_explanation = True
    scene_edit = None

    def __init__(self, target_agent: str, frame: int) -> None:
        self.target_agent = target_agent
        self.frame = frame

    def to_dict(self):
        text = f"{self.target_agent} explanation at decision frame {self.frame}."
        return {
            "explanation": text,
            "explanation_document": {"text": text},
        }


class _ExplanationPlanner:
    def __init__(self) -> None:
        self.last_diagnostics = {}

    def parse(self, question, *, selected_frame, environment_schema, cache_context):
        target = environment_schema["focus_entity"]
        # FACTUAL deliberately reproduces the old failure mode: the study
        # layer must still bind it to the selected action transition.
        return QueryPlan(
            raw_text=question,
            intent=QueryIntent.FACTUAL,
            frame_reference=selected_frame,
            subjects=(target,),
            requires_program_trace=False,
            target_variables=(f"{target}.objective",),
            response_language="zh-CN",
        )


class _ExplanationEngine(_NoopEngine):
    def __init__(self) -> None:
        self.planner = _ExplanationPlanner()
        self.calls = []

    def question_cache_context(self, _snapshot):
        return {}

    def execute_plan(self, plan, snapshot, *, language, seed, **_kwargs):
        self.calls.append(
            {
                "target": plan.prediction_targets[0],
                "frame": snapshot.frame,
                "language": language,
                "seed": seed,
                "requires_program_trace": plan.requires_program_trace,
                "requires_policy_query": plan.requires_policy_query,
                "target_variables": plan.target_variables,
                "evidence_requirements": plan.evidence_requirements,
            }
        )
        return _ExplanationAnswer(plan.prediction_targets[0], snapshot.frame)


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
    environment.reset(seed=42_026)
    frames = WarehouseAdapter(environment).rollout(
        policy,
        horizon=horizon,
        deterministic=True,
    ).frames
    return TutorialTrajectory(
        seed=42_026,
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
        "tutorial_seed": 42_026,
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


def test_full_explanation_session_has_exact_round_boundaries_and_immutable_replay(
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
    for _ in range(119):
        session.submit_human_action("WAIT")
    assert session.human_study.stage == "explanation"
    assert session.human_study.round_summaries["task1"].steps == 120
    frame_count = session.timeline.count
    replay = session.select_frame(40)
    assert replay["transition"]["loop"] is True
    assert {
        item["id"] for item in replay["transition"]["agents"]
    } == {"robot_1", "robot_2"}
    assert all(
        item["proposed_action"] is not None
        for item in replay["transition"]["agents"]
    )
    session.select_frame(0)
    assert session.timeline.index == 1
    session.select_frame(80)
    assert session.timeline.count == frame_count
    session.finish_explanation()
    assert session.human_study.explanation_count == 0
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


def test_explanation_questions_bind_any_action_frame_and_either_robot() -> None:
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
    task1_environment = session.environment
    for _ in range(120):
        session.submit_human_action("WAIT")
    assert session.human_study.stage == "explanation"
    assert session.environment is not task1_environment
    assert session.timeline.count == 5
    assert session.timeline.index == 1
    explanation_view = session.view()
    assert explanation_view["timeline"]["trajectory_kind"] == "ai_ai_reference"
    assert explanation_view["timeline"]["trajectory_seed"] == 42_026
    assert explanation_view["timeline"]["agent_control"] == {
        "robot_1": "ai",
        "robot_2": "ai",
    }
    assert explanation_view["state"]["frame"] == 1
    assert session.current.snapshot == tutorial.frames[0].next_snapshot
    assert explanation_view["state"]["policy_hidden"] is False
    assert explanation_view["study"]["round_summaries"]["task1"]["steps"] == 120
    assert 42_026 not in {assignment.task1_seed, assignment.task2_seed}
    engine = _ExplanationEngine()
    session.engine = engine
    transition_events = session.drain_events()
    reference_event = next(
        event
        for event in transition_events
        if event["event"] == "explanation_reference_loaded"
    )
    assert reference_event["trajectory_seed"] == 42_026
    assert reference_event["agent_control"] == {
        "robot_1": "ai",
        "robot_2": "ai",
    }

    session.select_frame(4)
    last_frame = session.explain_study(
        "队友的位置或动作是否影响了这个决定？", target_agent="robot_2"
    )
    first_report = last_frame["report"]
    assert first_report["selected_timeline_frame"] == 4
    assert first_report["decision_evidence_frame"] == 3
    assert first_report["target_agent"] == "robot_2"
    assert first_report["trajectory_kind"] == "ai_ai_reference"
    assert first_report["trajectory_seed"] == 42_026
    assert first_report["agent_control"] == {
        "robot_1": "ai",
        "robot_2": "ai",
    }
    assert last_frame["view"]["study"]["explanation_target_agent"] == "robot_2"

    session.select_frame(1)
    first_frame = session.explain_study(
        "为什么在这里采取这个动作？", target_agent="robot_1"
    )
    second_report = first_frame["report"]
    assert second_report["selected_timeline_frame"] == 1
    assert second_report["decision_evidence_frame"] == 0
    assert second_report["target_agent"] == "robot_1"
    assert second_report["question_seed"] != first_report["question_seed"]
    assert second_report["question_sequence"] == 2
    assert first_frame["view"]["study"]["explanation_target_agent"] == "robot_1"
    assert first_frame["view"]["last_explanation"]["target_agent"] == "robot_1"

    assert len(engine.calls) == 4  # English and Chinese for each question.
    assert {call["target"] for call in engine.calls[:2]} == {"robot_2"}
    assert {call["target"] for call in engine.calls[2:]} == {"robot_1"}
    assert {call["frame"] for call in engine.calls[:2]} == {3}
    assert {call["frame"] for call in engine.calls[2:]} == {0}
    assert all(call["requires_program_trace"] for call in engine.calls)
    assert all(call["requires_policy_query"] for call in engine.calls)
    assert {
        call["target_variables"] for call in engine.calls[:2]
    } == {("robot_2.observed_action",)}
    assert {
        call["target_variables"] for call in engine.calls[2:]
    } == {("robot_1.observed_action",)}
    assert all(
        {"state", "policy", "program_trace"}.issubset(
            call["evidence_requirements"]
        )
        for call in engine.calls
    )
    assert all(
        "study_focus:collaboration" in call["evidence_requirements"]
        for call in engine.calls[:2]
    )
    assert all(
        "study_focus:action" in call["evidence_requirements"]
        for call in engine.calls[2:]
    )
    assert len({call["seed"] for call in engine.calls[:2]}) == 1
    assert len({call["seed"] for call in engine.calls[2:]}) == 1

    audit_events = session.drain_events()
    explanation_events = [
        event for event in audit_events
        if event["event"] == "explanation_presented"
    ]
    assert [event["target_agent"] for event in explanation_events] == [
        "robot_2", "robot_1",
    ]
    assert len({event["question_seed"] for event in explanation_events}) == 2
    assert all(
        event["trajectory_kind"] == "ai_ai_reference"
        and event["trajectory_seed"] == 42_026
        and event["agent_control"] == {"robot_1": "ai", "robot_2": "ai"}
        for event in explanation_events
    )
    browse_events = [
        event for event in audit_events
        if event["event"] == "trajectory_frame_browsed"
    ]
    assert [event["environment_frame"] for event in browse_events] == [4, 1]
    assert all(
        event["study_stage"] == "explanation"
        and event["trajectory_kind"] == "ai_ai_reference"
        and event["trajectory_seed"] == 42_026
        and "round" not in event
        for event in browse_events
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
    session.begin_task1()
    for _ in range(120):
        session.submit_human_action("WAIT")
    assert session.human_study.stage == "explanation"

    first = session.reference_trajectory_payload()
    second = session.reference_trajectory_payload()
    assert first["trajectory_hash"] == second["trajectory_hash"]
    session.selected_agent = "robot_1"
    after_robot_selection = session.reference_trajectory_payload()
    session.selected_agent = "robot_2"
    after_other_selection = session.reference_trajectory_payload()
    assert after_robot_selection == first
    assert after_other_selection == first
    assert len(first["frames"]) == 121
    assert [frame["index"] for frame in first["frames"]] == list(range(121))
    assert first["agent_control"] == {"robot_1": "ai", "robot_2": "ai"}
    assert first["trajectory_seed"] == 42_026
    public_text = str(first)
    assert "goal_kind" not in public_text
    assert "goal_position" not in public_text
    assert "action_probabilities" not in public_text

    engine = _ExplanationEngine()
    session.engine = engine
    response = session.explain_study(
        "Why did robot 1 execute that action?",
        target_agent="robot_1",
        selected_frame=73,
        trajectory_hash=first["trajectory_hash"],
        question_kind="action",
    )
    assert response["report"]["selected_timeline_frame"] == 73
    assert response["report"]["target_agent"] == "robot_1"
    assert session.reference_trajectory_payload()["trajectory_hash"] == first["trajectory_hash"]
    assert all(
        "study_focus:action" in call["evidence_requirements"]
        for call in engine.calls
    )
    with pytest.raises(ValueError, match="changed"):
        session.explain_study(
            "Why did robot 2 wait?",
            target_agent="robot_2",
            selected_frame=1,
            trajectory_hash="stale",
            question_kind="action",
        )


def test_reference_trajectory_accepts_only_selection_only_legacy_hashes() -> None:
    policy = _wait_policy()
    allocator = CollaborativeConditionAllocator(randomization_seed=92)
    assignment = next(
        allocator._assignment_for_index(f"legacy-reference-{index}", index)
        for index in range(8)
        if allocator._assignment_for_index(
            f"legacy-reference-{index}", index
        ).condition == "explanation"
    )
    session = WarehouseWebSession(
        policy=policy,
        engine_factory=lambda adapter, actor: _NoopEngine(),
        seed=13,
        tutorial=_tutorial(policy, horizon=120),
    )
    session.run_id = "legacy-reference-hash-test"
    session.start_study(assignment=assignment, language="en")
    session.begin_task1()
    for _ in range(120):
        session.submit_human_action("WAIT")

    canonical = session.reference_trajectory_payload()["trajectory_hash"]
    legacy = {
        session._reference_identity_hash(
            session._reference_trajectory_identity(
                selected_agent_marker=agent_id,
            )
        )
        for agent_id in ("robot_1", "robot_2")
    }
    assert session.reference_trajectory_hash_is_compatible(canonical)
    assert all(
        session.reference_trajectory_hash_is_compatible(value)
        for value in legacy
    )
    assert not session.reference_trajectory_hash_is_compatible("unrelated")
