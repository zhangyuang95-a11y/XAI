from __future__ import annotations

from copy import deepcopy
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

from core.policy_contracts import ActionDistribution
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
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
