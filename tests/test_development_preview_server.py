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

    assert "机器人2" in report["explanation"]
    assert any(
        reason in report["explanation"]
        for reason in ("是为了", "是因为", "冻结状态")
    )
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

    assert "机器人1" in text
    assert "是为了" in text
    assert "选择概率最高" not in text


def test_action_explanations_use_direct_reasons_on_reported_frames() -> None:
    state = DevelopmentPreviewState()
    expected_by_reason = {
        "CLEAR_TEAMMATE_ROUTE": ("是为了", "清空"),
        "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE": ("等待", "被机器人"),
        "PRIORITY_ROUTE_PROGRESS": ("因为", "等待让行"),
        "PICKUP_ROUTE_PROGRESS": ("A点取货", "缩短到"),
        "DELIVERY_ROUTE_PROGRESS": ("送到B点", "缩短到"),
        "CHARGER_ROUTE_PROGRESS": ("前往充电站", "当前电量"),
    }

    found: set[str] = set()
    for frame, record in enumerate(state.tutorial_frames[1:], start=1):
        trace = record.info["decision_trace"]
        for agent_id, decision in trace["agents"].items():
            reason = decision["primary_reason_code"]
            if reason not in expected_by_reason or reason in found:
                continue
            expected_fragments = expected_by_reason[reason]
            found.add(reason)
            text = state._grounded_development_explanation(
                index=frame,
                target_agent=agent_id,
                focus="action",
                language="zh-CN",
            )
            assert all(fragment in text for fragment in expected_fragments)
            assert "当前目标是等待" not in text
            assert text.count("。") <= 2

    assert found == set(expected_by_reason)


def test_all_tutorial_action_explanations_are_short_and_reason_bearing() -> None:
    state = DevelopmentPreviewState()
    reason_markers = (
        "是为了",
        "是因为",
        "是在",
        "但因",
        "属于重新定位",
        "策略",
        "冻结状态",
        "因为决策前",
        "因为",
        "没有推进",
        "没有可验证",
    )

    for frame in range(1, len(state.tutorial_frames)):
        for agent_id in ("robot_1", "robot_2"):
            action = state.tutorial_frames[frame].actions[agent_id]
            text = state._grounded_development_explanation(
                index=frame,
                target_agent=agent_id,
                focus="action",
                language="zh-CN",
            )
            assert len(text) <= 120, (frame, agent_id, text)
            assert text.count("。") <= 2, (frame, agent_id, text)
            assert any(marker in text for marker in reason_markers), (
                frame,
                agent_id,
                text,
            )
            if action != "WAIT":
                assert "当前目标是等待" not in text, (frame, agent_id, text)


def test_charger_departure_explanations_state_the_operational_reason() -> None:
    state = DevelopmentPreviewState()

    for frame, record in enumerate(state.tutorial_frames):
        departures = tuple(
            event
            for event in record.events
            if event.get("event") == "charger_departure"
        )
        for event in departures:
            text = state._grounded_development_explanation(
                index=frame,
                target_agent=str(event["agent_id"]),
                focus="action",
                language="zh-CN",
            )
            assert "重新定位" not in text
            assert any(
                reason in text
                for reason in ("离开充电站", "充电已经完成", "A点取货")
            ), (frame, text)


def test_tutorial_task_moves_have_direct_progress_reasons() -> None:
    state = DevelopmentPreviewState()
    seen: set[str] = set()
    for frame, record in enumerate(state.tutorial_frames[1:], start=1):
        for agent_id, decision in record.info["decision_trace"]["agents"].items():
            reason = decision["primary_reason_code"]
            if reason not in {"PICKUP_ROUTE_PROGRESS", "DELIVERY_ROUTE_PROGRESS"}:
                continue
            seen.add(reason)
            text = state._grounded_development_explanation(
                index=frame,
                target_agent=agent_id,
                focus="action",
                language="zh-CN",
            )
            assert ("A点取货" if reason == "PICKUP_ROUTE_PROGRESS" else "送到B点") in text
            assert "缩短到" in text
            assert "非必要绕路" not in text
            assert "冻结状态无法支持" not in text
    assert seen == {"PICKUP_ROUTE_PROGRESS", "DELIVERY_ROUTE_PROGRESS"}


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
    plan = environment.get_state().active_coordination_plan
    assert plan is not None
    state = _scenario_preview(
        environment,
        {
            str(plan["waiting_agent_id"]): "WAIT",
            str(plan["moving_agent_id"]): str(plan["moving_action"]),
        },
    )
    selected = 1

    trace = state.tutorial_frames[selected].info["decision_trace"]
    decision = trace["agents"][str(plan["moving_agent_id"])]
    assert decision["primary_reason_code"] == "CLEAR_TEAMMATE_ROUTE"
    assert decision["joint_coordination_plan"]["reason_code"].startswith(
        "critical_charger"
    )

    text = state._grounded_development_explanation(
        index=selected,
        target_agent="robot_1",
        focus="action",
        language="zh-CN",
    )

    assert "离开充电站" in text
    assert "是为了" in text
    assert "低电量" in text
    assert "机器人2" in text
    assert "清空" in text
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


def test_tutorial_explanations_route_question_focus_and_show_energy_math() -> None:
    state = DevelopmentPreviewState()

    charging = state._grounded_development_explanation(
        index=2,
        target_agent="robot_2",
        focus="energy",
        language="zh-CN",
    )
    assert all(
        fragment in charging
        for fragment in ("当前电量33%", "预计需64%", "另留4%", "充至68%", "升至43%")
    )

    task_departure = state._grounded_development_explanation(
        index=6,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    assert all(
        fragment in task_departure
        for fragment in ("任务1的A点", "当前电量73%", "预计需64%", "电量足够")
    )

    allocation = state._grounded_development_explanation(
        index=23,
        target_agent="robot_1",
        focus="allocation",
        language="zh-CN",
    )
    assert "任务1已由机器人2取走" in allocation
    assert "当前目标是任务3" in allocation

    focused = {
        focus: state._grounded_development_explanation(
            index=6,
            target_agent="robot_2",
            focus=focus,
            language="zh-CN",
        )
        for focus in ("action", "energy", "task", "collaboration", "allocation")
    }
    assert len(set(focused.values())) == len(focused)


def test_tutorial_coordination_explains_loaded_priority_and_only_real_waits() -> None:
    state = DevelopmentPreviewState()

    loaded_priority = state._grounded_development_explanation(
        index=37,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    assert "载有任务3货物" in loaded_priority
    assert "前往B点交付" in loaded_priority
    assert "优先" not in loaded_priority or "先通过" in loaded_priority

    clearance = state._grounded_development_explanation(
        index=38,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    assert "下一格(5, 4)被机器人1占用" in clearance
    assert "正在清空该格" in clearance
    assert (
        state.tutorial_frames[38]
        .info["decision_trace"]["agents"]["robot_2"]["primary_reason_code"]
        == "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE"
    )

    parallel = state.tutorial_frames[40]
    assert parallel.actions == {"robot_1": "RIGHT", "robot_2": "UP"}
    assert parallel.info["decision_trace"]["agents"]["robot_2"][
        "joint_coordination_plan"
    ] is None
    parallel_text = state._grounded_development_explanation(
        index=40,
        target_agent="robot_2",
        focus="action",
        language="zh-CN",
    )
    assert "任务4的A点" in parallel_text
    assert "缩短到7格" in parallel_text


def test_decision_trace_exposes_release_threshold_components() -> None:
    state = DevelopmentPreviewState()
    charging = state.tutorial_frames[2].info["decision_trace"]["agents"][
        "robot_2"
    ]["charging_state"]

    assert charging["task_id"] == "task_1"
    assert charging["pickup_steps"] == 10.0
    assert charging["delivery_steps"] == 10.0
    assert charging["return_steps"] == 6.0
    assert charging["mission_reserve_steps"] == 6.0
    assert charging["route_energy"] == 64.0
    assert charging["hysteresis_energy"] == 4.0
    assert charging["release_threshold"] == 68.0


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

    assert english.startswith("A collision occurred:")
    assert "same pre-move state" in english
    assert "neither knew the other's current action" in english
    assert len(english.split()) <= 80
    assert english.count(".") <= 3
    assert "同一决策前状态同时选动作" in chinese
    assert chinese.count("。") <= 3
