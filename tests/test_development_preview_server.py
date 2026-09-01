from __future__ import annotations

from copy import deepcopy
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

from core.policy_contracts import ActionDistribution
from backend.adapters.warehouse_explanations import WarehouseExplanationMixin
from backend.training.warehouse_study_acceptance import (
    _pareto_dominating_joint_actions,
)
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.runtime_coordination import select_ai_ai_joint_actions
import ui.development_preview_server as preview_server
from ui.development_preview_server import (
    DevelopmentPreviewSessions,
    DevelopmentPreviewState,
    Handler,
    PreviewFrame,
    TASK1_SEED,
    _initial_frame,
    _transition_payload,
    build_development_tutorial,
)


def _envelope(state: DevelopmentPreviewState, command: str, **payload):
    return {
        "operation_id": f"test-{state.version}-{command}",
        "run_id": state.run_id,
        "expected_stage": state.stage,
        "expected_state_version": state.version,
        "command": command,
        "payload": payload,
    }


def _start_task1(
    state: DevelopmentPreviewState,
    *,
    locale: str = "en",
    condition: str = "explanation",
) -> None:
    state.command(
        _envelope(
            state,
            "start",
            participant_id="preview-test",
            locale=locale,
            viewport_width=1440,
            condition_override=condition,
        )
    )
    state.command(_envelope(state, "begin_task1"))


def _append_explicit_transition(
    state: DevelopmentPreviewState,
    environment: WarehouseMultiAgentEnv,
    actions: dict[str, str],
) -> None:
    before = environment.get_state()
    _, rewards, _, _, info = environment.step(actions)
    after = environment.get_state()
    frame = PreviewFrame(
        state=after,
        actions=dict(actions),
        goal_overrides={},
        rewards=dict(rewards),
        info=deepcopy(dict(info)),
        events=(),
        transition=_transition_payload(
            before,
            after,
            actions,
            info,
            reveal_ai_action=False,
            loop=False,
        )
        | {"before_stage": "task1"},
        action_distributions={
            agent_id: ActionDistribution(
                agent_id=agent_id,
                actions=tuple(ACTIONS),
                probabilities=tuple(
                    1.0 if candidate == action else 0.0 for candidate in ACTIONS
                ),
                proposed_action=action,
            )
            for agent_id, action in actions.items()
        },
    )
    state.environment = environment
    state.round_frame = frame
    state.round_frames.append(frame)


def test_development_tutorial_is_one_productive_real_environment_round() -> None:
    frames = build_development_tutorial()

    assert len(frames) == 121
    assert [frame.state.frame for frame in frames] == list(range(121))
    assert frames[-1].state.terminal_reason == "horizon"
    assert frames[-1].state.total_deliveries > 0
    observed_events = {
        str(event.get("event", "")) for frame in frames for event in frame.events
    }
    assert {"claimed", "delivered", "charging", "coordination_yield"} <= observed_events


def test_seed_40786_reported_segments_use_undominated_atomic_joint_decisions() -> None:
    """Lock the reported failure windows to the shared causal runtime.

    The assertions deliberately inspect evidence, not a hard-coded action for
    a frame number.  Task replacement and lifecycle fixes can legitimately
    change the exact move while the invariant must remain: all 25 joint
    candidates are resolved from S_t, the selected safe move is undominated,
    and the physical/audit/explanation chain reports no regression.
    """

    frames = build_development_tutorial()
    reported_frames = (
        36,
        37,
        38,
        52,
        53,
        54,
        60,
        61,
        62,
        63,
        64,
        66,
        67,
        68,
        69,
        74,
        75,
        76,
        80,
        81,
        82,
        113,
        114,
        115,
    )
    explainer = WarehouseExplanationMixin()
    forbidden = (
        "decisiontrace",
        "decision trace",
        "reason_code",
        "日志缺失",
        "内部字段",
        "原因码",
    )

    for frame_index in reported_frames:
        frame = frames[frame_index]
        info = frame.info
        trace = info["decision_trace"]
        runtime = trace["runtime_decision"]
        considered = tuple(runtime["safe_joint_actions"]) + tuple(
            runtime["rejected_joint_actions"]
        )

        assert len(considered) == len(ACTIONS) ** 2
        assert runtime["same_frozen_state"] is True
        assert _pareto_dominating_joint_actions(runtime) == ()
        assert runtime["selected_actions"] == info["executed_actions"]
        assert trace["fact_valid"] is True
        assert trace["fact_validation_failures"] == ()
        assert info["robot_collision_event"] is False
        assert info["avoidable_wait_agents"] == ()
        assert info["avoidable_detour_agents"] == ()
        assert info["unexplained_reversal_agents"] == ()
        assert info["short_cycle_agents"] == ()
        assert info["invalid_goal_switch_agents"] == ()
        assert info["decision_audit"]["same_pre_move_state"] is True
        assert info["decision_audit"]["environment_step_calls"] == 1

        for agent_id in ("robot_1", "robot_2"):
            for language in ("en", "zh-CN"):
                text = explainer._decision_trace_explanation(
                    trace,
                    target_agent=agent_id,
                    focus="action",
                    language=language,
                )
                assert text is not None
                normalized = text.casefold()
                assert not any(token in normalized for token in forbidden)


@pytest.mark.parametrize(
    ("reported_frame", "positions", "safe_alternative"),
    (
        (36, ((2, 3), (1, 2)), ("LEFT", "RIGHT")),
        (37, ((2, 3), (1, 3)), ("LEFT", "DOWN")),
        (52, ((3, 1), (4, 2)), ("RIGHT", "DOWN")),
        (64, ((2, 2), (3, 2)), ("DOWN", "DOWN")),
        (66, ((3, 2), (3, 1)), ("DOWN", "RIGHT")),
        (67, ((4, 2), (3, 1)), ("DOWN", "RIGHT")),
        (74, ((4, 3), (5, 2)), ("LEFT", "RIGHT")),
        (80, ((3, 1), (3, 2)), ("RIGHT", "UP")),
        (82, ((3, 2), (4, 2)), ("UP", "UP")),
        (113, ((2, 3), (1, 3)), ("RIGHT", "DOWN")),
        (115, ((2, 2), (2, 3)), ("RIGHT", "RIGHT")),
    ),
)
def test_reported_prefix_joint_alternatives_are_recognized_as_atomically_safe(
    reported_frame: int,
    positions: tuple[tuple[int, int], tuple[int, int]],
    safe_alternative: tuple[str, str],
) -> None:
    """Preserve the original failure-state geometry after trajectory repair.

    Correcting an early decision changes every later position, so the old
    frame number cannot be asserted against the repaired trajectory itself.
    These are the pre-fix S_t positions reconstructed from seed 40786.  The
    test proves the public selector still enumerates the reported pipeline
    move as safe; no production branch reads the frame label or this fixture.
    """

    environment = WarehouseMultiAgentEnv(collaborative_study_config())
    environment.reset(seed=reported_frame)
    state = environment.get_state()
    for agent, position in zip(state.agents, positions):
        agent.position = position
        agent.battery = 100.0
        agent.active = True
    environment.set_state(state)

    _, runtime = select_ai_ai_joint_actions(
        environment,
        {"robot_1": "WAIT", "robot_2": "WAIT"},
    )
    safe_pairs = {
        (
            str(item["actions"]["robot_1"]),
            str(item["actions"]["robot_2"]),
        )
        for item in runtime["safe_joint_actions"]
    }

    assert safe_alternative in safe_pairs


def test_development_preview_uses_six_by_seven_production_geometry() -> None:
    state = DevelopmentPreviewState()
    state.command(
        _envelope(
            state,
            "start",
            participant_id="map-test",
            locale="en",
            condition_override="explanation",
        )
    )
    view = state.view()

    assert view["map"]["layout_id"] == STUDY_MAP_LAYOUT.layout_id
    assert view["map"]["rows"] == 6
    assert view["map"]["cols"] == 7
    assert view["study"]["tutorial"]["total_frames"] == 121
    reference = state.reference_trajectory()
    assert reference["trajectory_kind"] == "ai_ai_demonstration"
    assert len(reference["frames"]) == 121


def test_ai_ai_trajectory_is_not_available_after_instructions() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)

    with pytest.raises(RuntimeError, match="instructions demonstration"):
        state.reference_trajectory()


def test_http_preview_state_is_isolated_between_browser_pages(monkeypatch) -> None:
    sessions = DevelopmentPreviewSessions(max_sessions=4)
    monkeypatch.setattr(preview_server, "SESSIONS", sessions)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)

    def request(method: str, page_id: str, body: dict | None = None) -> dict:
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"X-Warehouse-Page": page_id}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        connection.request(
            method,
            "/api/study/command" if body else "/api/view",
            body=encoded,
            headers=headers,
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        return payload

    try:
        assert request("GET", "browser-page-a")["study"]["stage"] == "idle"
        assert request("GET", "browser-page-b")["study"]["stage"] == "idle"
        request(
            "POST",
            "browser-page-a",
            {
                "operation_id": "start-page-a",
                "run_id": None,
                "expected_stage": "idle",
                "expected_state_version": 0,
                "command": "start",
                "payload": {"participant_id": "page-a", "locale": "en"},
            },
        )
        assert request("GET", "browser-page-a")["study"]["stage"] == "instructions"
        assert request("GET", "browser-page-b")["study"]["stage"] == "idle"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_interactive_round_does_not_end_after_two_steps() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state, locale="zh-CN")

    state.command(_envelope(state, "human_action", action="WAIT"))
    result = state.command(_envelope(state, "human_action", action="WAIT"))

    assert result["view"]["study"]["stage"] == "task1"
    assert result["view"]["state"]["frame"] == 2
    assert result["view"]["study"]["total"] == 120


def test_task1_flows_directly_to_task2_without_explanation_stage() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state, condition="control")

    for _ in range(120):
        state.command(_envelope(state, "human_action", action="WAIT"))

    assert state.stage == "task1_complete"
    assert state.commands() == ("set_language", "restart", "begin_task2")
    state.command(_envelope(state, "begin_task2"))
    assert state.stage == "task2"


def test_live_question_uses_real_task1_prefix_and_does_not_mutate_game_state() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)
    for action in ("UP", "WAIT", "LEFT"):
        state.command(_envelope(state, "human_action", action=action))

    before = deepcopy(state.environment.get_state())
    result = state.command(
        _envelope(
            state,
            "ask_explanation",
            target_agent="robot_2",
            question="Why did you do that?",
            question_kind="action",
        )
    )
    after = state.environment.get_state()
    report = result["view"]["last_explanation"]

    assert after == before
    assert report["trajectory_kind"] == "human_ai_task1"
    assert report["agent_control"] == {"robot_1": "human", "robot_2": "ai"}
    assert report["target_agent"] == "robot_2"
    assert len(report["context_frames"]) <= 5
    assert max(report["context_frames"]) <= report["current_frame"]
    assert report["fact_validation"]["future_frames_used"] is False


def test_live_question_rejects_replay_frame_and_other_robot() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)
    state.command(_envelope(state, "human_action", action="WAIT"))

    with pytest.raises(ValueError, match="cannot select"):
        state.command(
            _envelope(
                state,
                "ask_explanation",
                target_agent="robot_2",
                selected_frame=1,
                question="Why?",
                question_kind="action",
            )
        )
    with pytest.raises(ValueError, match="Robot 2"):
        state.command(
            {
                **_envelope(
                    state,
                    "ask_explanation",
                    target_agent="robot_1",
                    question="Why?",
                    question_kind="action",
                ),
                "operation_id": "different-operation",
            }
        )


def test_live_questions_are_group_a_task1_only() -> None:
    control = DevelopmentPreviewState()
    _start_task1(control, condition="control")
    assert "ask_explanation" not in control.commands()
    with pytest.raises(RuntimeError):
        control.command(
            _envelope(
                control,
                "ask_explanation",
                target_agent="robot_2",
                question="Why?",
                question_kind="action",
            )
        )

    task2 = DevelopmentPreviewState()
    _start_task1(task2)
    task2.stage = "task2"
    assert "ask_explanation" not in task2.commands()


def test_language_switch_preserves_answer_evidence_and_game_state() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state, locale="en")
    state.command(_envelope(state, "human_action", action="WAIT"))
    state.command(
        _envelope(
            state,
            "ask_explanation",
            target_agent="robot_2",
            question="What is Robot 2 doing?",
            question_kind="action",
        )
    )
    english = state.view()["last_explanation"]
    before = deepcopy(state.environment.get_state())

    state.command(_envelope(state, "set_language", locale="zh-CN"))
    chinese = state.view()["last_explanation"]

    assert state.environment.get_state() == before
    assert state.explanation_count == 1
    assert english["structured_evidence"] == chinese["structured_evidence"]
    assert english["answer_en"] == chinese["answer_en"]
    assert english["answer_zh"] == chinese["answer_zh"]
    assert english["explanation"] != chinese["explanation"]


def test_next_participant_action_is_attached_to_question_log() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)
    state.command(_envelope(state, "human_action", action="WAIT"))
    state.command(
        _envelope(
            state,
            "ask_explanation",
            target_agent="robot_2",
            question="Why did Robot 2 wait?",
            question_kind="wait",
        )
    )
    assert state.question_log[-1]["post_question_action"] is None

    state.command(_envelope(state, "human_action", action="RIGHT"))

    assert state.question_log[-1]["post_question_action"] == "RIGHT"
    assert state.question_log[-1]["post_question_frame"] == 2


def test_collision_answer_uses_same_pre_state_and_no_foresight() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)
    environment = WarehouseMultiAgentEnv(collaborative_study_config())
    environment.reset(seed=TASK1_SEED)
    scenario = environment.get_state()
    scenario.by_id("robot_1").position = (4, 2)
    scenario.by_id("robot_2").position = (4, 4)
    environment.set_state(scenario)
    state.environment = environment
    state.round_frame = _initial_frame(environment)
    state.round_frames = [state.round_frame]
    _append_explicit_transition(
        state,
        environment,
        {"robot_1": "RIGHT", "robot_2": "LEFT"},
    )

    result = state.command(
        _envelope(
            state,
            "ask_explanation",
            target_agent="robot_2",
            question="Why did we just collide?",
            question_kind="collision",
        )
    )
    english = result["view"]["last_explanation"]["answer_en"]
    chinese = result["view"]["last_explanation"]["answer_zh"]

    assert "simultaneously" in english
    assert "shared pre-move state" in english
    assert "did not observe your current action first" in english
    assert "同时" in chinese
    assert "没有提前看到" in chinese


def test_no_recent_collision_answer_is_explicit_and_anchored_to_last_five() -> None:
    state = DevelopmentPreviewState()
    _start_task1(state)
    state.command(_envelope(state, "human_action", action="WAIT"))

    result = state.command(
        _envelope(
            state,
            "ask_explanation",
            target_agent="robot_2",
            question="Why did we collide?",
            question_kind="collision",
        )
    )

    assert result["view"]["last_explanation"]["answer_en"] == (
        "No collision occurred in the last five steps."
    )
