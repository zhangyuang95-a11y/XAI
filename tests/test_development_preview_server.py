from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading

from env.warehouse.layouts import DEFAULT_MAP_LAYOUT
import ui.development_preview_server as preview_server
from ui.development_preview_server import (
    DevelopmentPreviewSessions,
    DevelopmentPreviewState,
    Handler,
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


def test_development_tutorial_is_one_productive_real_environment_round() -> None:
    frames = build_development_tutorial()

    assert len(frames) == 121
    assert [frame.state.frame for frame in frames] == list(range(121))
    assert frames[-1].state.terminal_reason == "horizon"
    assert frames[-1].state.total_deliveries > 0
    observed_events = {
        str(event.get("event", ""))
        for frame in frames
        for event in frame.events
    }
    assert {"claimed", "delivered", "charging", "coordination_yield"} <= observed_events


def test_development_preview_uses_canonical_geometry_and_121_demo_frames() -> None:
    state = DevelopmentPreviewState()
    view = state.view()

    assert view["map"]["layout_id"] == DEFAULT_MAP_LAYOUT.layout_id
    assert view["map"]["rows"] == DEFAULT_MAP_LAYOUT.rows
    assert view["map"]["cols"] == DEFAULT_MAP_LAYOUT.cols
    assert view["study"]["tutorial"]["total_frames"] == 121
    assert len(state.reference_trajectory()["frames"]) == 121


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
        connection.request(method, "/api/study/command" if body else "/api/view", body=encoded, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        return payload

    try:
        first = request("GET", "browser-page-a")
        second = request("GET", "browser-page-b")
        assert first["study"]["stage"] == "idle"
        assert second["study"]["stage"] == "idle"

        started = request(
            "POST",
            "browser-page-a",
            {
                "operation_id": "start-page-a",
                "run_id": None,
                "expected_stage": "idle",
                "expected_state_version": 0,
                "command": "start",
                "payload": {
                    "participant_id": "page-a",
                    "locale": "en",
                    "viewport_width": 1440,
                },
            },
        )
        assert started["view"]["study"]["stage"] == "instructions"
        assert request("GET", "browser-page-a")["study"]["stage"] == "instructions"
        assert request("GET", "browser-page-b")["study"]["stage"] == "idle"
        assert sessions.state_for("browser-page-a").tutorial_frames is sessions.state_for(
            "browser-page-b"
        ).tutorial_frames
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_interactive_round_does_not_end_after_two_steps() -> None:
    state = DevelopmentPreviewState()
    state.command(
        _envelope(
            state,
            "start",
            participant_id="preview-test",
            locale="zh-CN",
            viewport_width=1440,
            condition_override="explanation",
        )
    )
    state.command(_envelope(state, "begin_task1"))

    state.command(_envelope(state, "human_action", action="WAIT"))
    result = state.command(_envelope(state, "human_action", action="WAIT"))

    assert result["view"]["study"]["stage"] == "task1"
    assert result["view"]["state"]["frame"] == 2
    assert result["view"]["study"]["progress"] == 2
    assert result["view"]["study"]["total"] == 120
    assert state.environment.get_state().episode_id == 1


def test_interactive_round_finishes_only_at_real_terminal_boundary() -> None:
    state = DevelopmentPreviewState()
    state.command(
        _envelope(
            state,
            "start",
            participant_id="preview-horizon",
            locale="en",
            viewport_width=1440,
            condition_override="control",
        )
    )
    state.command(_envelope(state, "begin_task1"))

    for _ in range(119):
        state.command(_envelope(state, "human_action", action="WAIT"))
    assert state.stage == "task1"
    assert state.environment.get_state().frame == 119

    result = state.command(_envelope(state, "human_action", action="WAIT"))
    assert result["view"]["study"]["stage"] == "task1_complete"
    assert result["view"]["study"]["round_summaries"]["task1"]["steps"] == 120
    assert result["view"]["study"]["round_summaries"]["task1"]["terminal_reason"] == "horizon"


def test_development_explanation_is_grounded_in_selected_action_evidence() -> None:
    state = DevelopmentPreviewState()
    state.stage = "explanation"
    state.locale = "zh-CN"
    selected = next(
        index
        for index, frame in enumerate(state.tutorial_frames)
        if index > 0 and frame.actions.get("robot_2") != "WAIT"
    )

    result = state.command(
        _envelope(
            state,
            "ask_explanation",
            selected_frame=selected,
            target_agent="robot_2",
            question="为什么机器人 2 执行这个动作？",
            question_kind="action",
        )
    )
    report = result["view"]["last_explanation"]

    assert "机器人2这一步执行了" in report["explanation"]
    assert "正式的 RCPD" not in report["explanation"]
    assert report["selected_timeline_frame"] == selected
    assert report["decision_evidence_frame"] == selected - 1
    assert report["explanation_mode"] == "deterministic_environment_evidence"


def test_action_explanation_states_recorded_goal_progress_and_outcome() -> None:
    state = DevelopmentPreviewState()

    assert state.tutorial_frames[65].goal_overrides["robot_1"] == (6, 10)
    text = state._grounded_development_explanation(
        index=65,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )

    assert "临时导航目标是任务2的A点(6, 10)" in text
    assert "从当前位置(5, 5)到该目标的最短可通行路径为6格" in text
    assert "执行了向下" in text
    assert "从(5, 5)移动到(6, 5)" in text
    assert "剩余距离从6格缩短到5格" in text


def test_development_energy_explanation_uses_real_charging_frame() -> None:
    state = DevelopmentPreviewState()
    state.stage = "explanation"
    state.locale = "zh-CN"
    selected, agent_id = next(
        (index, str(event["agent_id"]))
        for index, frame in enumerate(state.tutorial_frames)
        for event in frame.events
        if event.get("event") == "charging"
    )

    result = state.command(
        _envelope(
            state,
            "ask_explanation",
            selected_frame=selected,
            target_agent=agent_id,
            question="电量和充电需求如何影响这个动作？",
            question_kind="energy",
        )
    )
    text = result["view"]["last_explanation"]["explanation"]

    assert "电量" in text
    assert "%" in text
    assert "充电" in text
