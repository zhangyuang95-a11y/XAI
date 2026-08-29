from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading

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
    _initial_frame,
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


def _scenario_preview(
    environment: WarehouseMultiAgentEnv,
    actions: dict[str, str],
) -> DevelopmentPreviewState:
    """Build explanation evidence from one explicit real transition."""

    before = _initial_frame(environment)
    _, rewards, _, _, info = environment.step(actions)
    outcome = PreviewFrame(
        state=environment.get_state(),
        actions=dict(actions),
        goal_overrides={},
        rewards=dict(rewards),
        info=dict(info),
        events=(),
        transition=None,
        action_distributions={
            agent_id: ActionDistribution(
                agent_id=agent_id,
                actions=tuple(ACTIONS),
                probabilities=tuple(
                    1.0 if candidate == action else 0.0
                    for candidate in ACTIONS
                ),
                proposed_action=action,
            )
            for agent_id, action in actions.items()
        },
    )
    return DevelopmentPreviewState(tutorial_frames=(before, outcome))


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

    assert view["map"]["layout_id"] == STUDY_MAP_LAYOUT.layout_id
    assert view["map"]["rows"] == STUDY_MAP_LAYOUT.rows
    assert view["map"]["cols"] == STUDY_MAP_LAYOUT.cols
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

    assert "机器人2本帧执行了" in report["explanation"]
    assert "当前目标" in report["explanation"]
    assert "正式的 RCPD" not in report["explanation"]
    assert report["selected_timeline_frame"] == selected
    assert report["decision_evidence_frame"] == selected - 1
    assert report["explanation_mode"] == "deterministic_environment_evidence"


def test_action_explanation_states_recorded_goal_progress_and_outcome() -> None:
    state = DevelopmentPreviewState()
    selected = next(
        index
        for index, frame in enumerate(state.tutorial_frames)
        if index > 0 and frame.actions.get("robot_1") not in {None, "WAIT"}
    )
    text = state._grounded_development_explanation(
        index=selected,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )

    assert "机器人1本帧执行了" in text
    assert "决策前，机器人1的当前目标是" in text
    assert "选择概率最高" not in text


def test_action_explanation_prioritizes_clearing_charger_for_teammate() -> None:
    environment = WarehouseMultiAgentEnv(collaborative_study_config())
    environment.reset(seed=20_901)
    scenario = environment.get_state()
    charger = environment.layout.charger_position
    scenario.by_id("robot_1").position = charger
    scenario.by_id("robot_1").battery = 100.0
    scenario.by_id("robot_2").position = (charger[0], charger[1] + 1)
    scenario.by_id("robot_2").battery = 10.0
    scenario.by_id("robot_2").navigation_goal_kind = "charge"
    scenario.by_id("robot_2").navigation_goal_position = charger
    environment.set_state(scenario)
    state = _scenario_preview(
        environment,
        {"robot_1": "UP", "robot_2": "LEFT"},
    )
    selected = 1

    assert any(
        fact.predicate == "collaboration_context"
        and fact.value.get("charger_clearance")
        for fact in state._explanation_snapshot(selected)[0].evidence_facts(
            state._explanation_snapshot(selected)[1], "robot_1", None
        )
    )

    text = state._grounded_development_explanation(
        index=selected,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )

    assert "执行后的结果是让出充电站" in text
    assert "低电量" in text
    assert "队友能够进入" in text
    assert "决策前，机器人1的当前目标是" in text
    assert "同一决策前状态同时选动作" in text
    assert "机器人2事先不知道机器人1本帧会怎么走" in text
    assert "选择概率最高" not in text


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


def test_collision_explanation_is_direct_concise_and_simultaneous() -> None:
    environment = WarehouseMultiAgentEnv(collaborative_study_config())
    environment.reset(seed=20_902)
    scenario = environment.get_state()
    scenario.by_id("robot_1").position = (1, 2)
    scenario.by_id("robot_2").position = (1, 4)
    environment.set_state(scenario)
    state = _scenario_preview(
        environment,
        {"robot_1": "RIGHT", "robot_2": "LEFT"},
    )
    selected = 1

    assert state.tutorial_frames[selected].info["robot_collision_event"] is True
    english = state._grounded_development_explanation(
        index=selected,
        target_agent="robot_2",
        focus="collision",
        language="en",
    )
    chinese = state._grounded_development_explanation(
        index=selected,
        target_agent="robot_2",
        focus="collision",
        language="zh-CN",
    )

    assert english.startswith("Yes:")
    assert "same pre-move state" in english
    assert "did not know Robot 1's current move" in english
    assert "current goal" in english
    assert len(english.split()) <= 80
    assert english.count(".") <= 3
    assert "同一决策前状态同时选动作" in chinese
    assert "当前目标" in chinese
    assert chinese.count("。") <= 3
