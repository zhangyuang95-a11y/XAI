from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import run as run_entry
from env.warehouse.layouts import DEFAULT_MAP_LAYOUT
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from tests.browser_fixture_server import FixtureState
from ui.collaborative_study import CollaborativeConditionAllocator
from ui.web_runtime import (
    WarehouseWebApplication,
    WarehouseWebSession,
    serialize_warehouse_state,
    warehouse_map_payload,
)
from ui.web_server import (
    API_ROUTES,
    WarehouseHTTPServer,
    build_parser,
    bundled_index_html,
)


def test_web_state_serialization_uses_shared_tasks_and_can_hide_policy() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=10))
    environment.reset(seed=2026)
    visible = serialize_warehouse_state(
        environment.get_state(),
        selected_agent="robot_1",
        actions={"robot_1": "UP"},
    )
    hidden = serialize_warehouse_state(
        environment.get_state(),
        selected_agent="robot_1",
        actions={"robot_1": "UP"},
        reveal_policy=False,
    )

    assert len(visible["agents"]) == 2
    assert len(visible["tasks"]) == 2
    assert visible["agents"][0]["proposed_action"] == "UP"
    assert hidden["policy_hidden"] is True
    assert all(agent["proposed_action"] is None for agent in hidden["agents"])
    assert all("action_probabilities" not in agent for agent in hidden["agents"])
    assert all("goal_kind" not in agent for agent in visible["agents"])
    assert all("goal_position" not in agent for agent in visible["agents"])
    assert all("goal_kind" not in agent for agent in hidden["agents"])
    assert all("goal_position" not in agent for agent in hidden["agents"])
    assert "delivery_target" not in hidden
    assert "mission_ready_count" not in hidden


def test_web_map_payload_matches_current_fixed_geometry() -> None:
    payload = warehouse_map_payload()

    assert payload["rows"] == 6
    assert payload["cols"] == 7
    assert "delivery_position" not in payload
    assert payload["charger_position"] == [5, 3]
    assert "yield_bays" not in payload
    assert payload["robot_start_positions"] == [[5, 2], [5, 4]]
    assert payload["robot_exit_positions"] == [[4, 2], [4, 3], [4, 4]]
    assert len(payload["shelves"]) == 23
    assert payload["shared_delivery_tasks"] is True


def test_browser_fixture_uses_the_production_map_geometry() -> None:
    fixture = FixtureState()

    assert fixture.map_payload() == warehouse_map_payload()
    for frame in range(24):
        assert DEFAULT_MAP_LAYOUT.is_passable(tuple(fixture.path_position(frame)))
        assert DEFAULT_MAP_LAYOUT.is_passable(
            tuple(fixture.path_position(frame, reverse=True))
        )
    for task in fixture.snapshot()["tasks"]:
        assert DEFAULT_MAP_LAYOUT.is_passable(tuple(task["pickup_position"]))
        assert DEFAULT_MAP_LAYOUT.is_passable(tuple(task["delivery_position"]))


class _FakeApplication:
    def dispatch(self, session_id, operation, payload):
        return session_id or "test-session", {
            "operation": operation,
            "payload": dict(payload),
        }


def test_http_server_serves_bundled_page_health_and_command_api() -> None:
    server = WarehouseHTTPServer(("127.0.0.1", 0), _FakeApplication())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode("utf-8")
        assert response.status == 200
        assert "PolicyLens" in html
        assert 'data-bundled-asset="styles.css"' in html
        assert 'data-bundled-asset="app.js"' in html
        assert "script-src 'self' 'nonce-" in response.getheader(
            "Content-Security-Policy"
        )

        connection.request("GET", "/api/health")
        response = connection.getresponse()
        assert json.loads(response.read())["status"] == "ok"

        envelope = {
            "operation_id": "op-1",
            "run_id": None,
            "expected_stage": "idle",
            "expected_state_version": 0,
            "command": "start",
            "payload": {},
        }
        body = json.dumps(envelope).encode()
        connection.request(
            "POST",
            "/api/study/command",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["operation"] == "study_command"

        connection.request("POST", "/api/study/posttest", body=b"{}")
        response = connection.getresponse()
        response.read()
        assert response.status == 404
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_only_current_study_write_endpoint_is_exposed() -> None:
    assert API_ROUTES == {
        "/api/view": "view",
        "/api/study/command": "study_command",
        "/api/study/reference-trajectory": "reference_trajectory",
    }


def test_web_client_retries_transient_tunnel_failures() -> None:
    source = (Path(__file__).parents[1] / "ui" / "web" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "const API_MAX_ATTEMPTS = 4" in source
    assert "status >= 520 && status <= 530" in source
    assert "await delay(350 * (2 ** (attempt - 1)))" in source
    assert 'new Error(tr("temporaryNetworkError"))' in source


def test_web_client_defaults_the_registration_page_to_english() -> None:
    root = Path(__file__).parents[1] / "ui" / "web"
    source = (root / "app.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")

    assert 'const DEFAULT_LOCALE = "en"' in source
    assert 'setLanguage(DEFAULT_LOCALE, false)' in source
    assert 'requestedStage !== "idle" && view.study?.locale' in source
    assert "PolicyLens · Two-Robot Collaborative Delivery Study" in html
    assert 'id="languageButtonLabel">中</span>' in html
    assert 'data-i18n="participantSetup">Participant setup</span>' in html
    assert 'data-i18n="start">Start study</button>' in html
    assert '<kbd data-i18n="spaceKey">Space</kbd>' in html


def test_web_cli_seed_library_defaults_to_checkpoint_sibling() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "candidate/warehouse_mappo.pt",
            "--transformer-model",
            "test-model",
        ]
    )

    # None is intentional: WarehouseWebApplication resolves the seed library
    # beside the selected checkpoint, avoiding cross-version artifact mixing.
    assert args.parallel_seed_library is None
    assert args.study_db is None
    assert args.test_condition_selector is False

    development = build_parser().parse_args(
        ["--transformer-model", "test-model", "--test-condition-selector"]
    )
    assert development.test_condition_selector is True


def test_default_and_formal_entrypoints_cannot_be_confused(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr("ui.web_server.main", lambda: captured.append(list(run_entry.sys.argv)))
    monkeypatch.setattr(
        run_entry,
        "_default_ui_artifacts",
        lambda: (Path("test-model.pt"), Path("test-program.json")),
    )

    monkeypatch.setattr(run_entry.sys, "argv", ["run.py"])
    run_entry.main()
    assert "--test-condition-selector" in captured[-1]

    monkeypatch.setattr(
        run_entry.sys,
        "argv",
        ["run_formal_ui.py", "--test-condition-selector", "--formal-study"],
    )
    run_entry.main()
    assert "--test-condition-selector" not in captured[-1]
    assert "--formal-study" not in captured[-1]


def test_default_entrypoint_loads_only_a_formally_accepted_bundle(tmp_path) -> None:
    bundle = run_entry.CollaborativeArtifactPaths.under(tmp_path, "candidate")
    bundle.root.mkdir(parents=True)
    for path in (
        bundle.model,
        bundle.rcpd_program,
        bundle.parallel_seed_pairs,
        bundle.reference_trajectory,
    ):
        path.write_text("{}", encoding="utf-8")
    bundle.training_summary.write_text(
        json.dumps(
            {
                "model_version": "model-v1",
                "program_regularization": {"explanation_eligible": True},
            }
        ),
        encoding="utf-8",
    )
    evaluation = {
        "model_version": "model-v1",
        "checkpoint": str(bundle.model.resolve()),
        "episodes_per_condition": 200,
        "acceptance_checks": {
            name: True for name in run_entry.FORMAL_ACCEPTANCE_CHECKS
        },
        "formal_candidate": False,
        "artifact_hashes": {
            "model": run_entry.file_sha256(bundle.model),
            "program": run_entry.file_sha256(bundle.rcpd_program),
            "training_summary": run_entry.file_sha256(bundle.training_summary),
            "parallel_seed_library": run_entry.file_sha256(
                bundle.parallel_seed_pairs
            ),
            "reference_trajectory": run_entry.file_sha256(
                bundle.reference_trajectory
            ),
        },
    }
    bundle.formal_evaluation.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    assert run_entry._accepted_ui_artifacts(bundle) is None

    evaluation["formal_candidate"] = True
    bundle.formal_evaluation.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    assert run_entry._accepted_ui_artifacts(bundle) == (
        bundle.model,
        bundle.rcpd_program,
    )

    bundle.rcpd_program.write_text('{"tampered": true}', encoding="utf-8")
    assert run_entry._accepted_ui_artifacts(bundle) is None


class _AssignmentStore:
    def __init__(self, *, existing=None, runs=()):
        self.existing = existing
        self.runs = list(runs)

    def participant_assignment(self, _participant_key):
        return self.existing

    def assignments(self):
        return [self.existing] if self.existing else []

    def run_assignments(self):
        return list(self.runs)


def _assignment_application(*, test_mode: bool, store: _AssignmentStore):
    application = WarehouseWebApplication.__new__(WarehouseWebApplication)
    application.test_condition_selector = test_mode
    application.study_store = store
    application.study_allocator = CollaborativeConditionAllocator(
        randomization_seed=41000,
        study_phase="pilot",
    )
    return application


def test_development_conditions_follow_runs_not_participant_identifier() -> None:
    store = _AssignmentStore()
    application = _assignment_application(test_mode=True, store=store)

    first, first_key = application._assignment_for_participant("same-id")
    store.runs.append(first.to_dict())
    second, second_key = application._assignment_for_participant("same-id")

    assert first_key == second_key == "same-id"
    assert first.enrollment_index == 0
    assert second.enrollment_index == 1
    assert first.condition == "control"
    assert second.condition == "explanation"
    forced, _ = application._assignment_for_participant(
        "same-id", condition_override="control"
    )
    assert forced.condition == "control"


def test_formal_mode_keeps_assignment_and_rejects_condition_override() -> None:
    allocator = CollaborativeConditionAllocator(randomization_seed=41000)
    existing = allocator._assignment_for_index("original", 0).to_dict()
    application = _assignment_application(
        test_mode=False,
        store=_AssignmentStore(existing=existing),
    )

    repeated, _ = application._assignment_for_participant("Original")
    assert repeated.condition == existing["condition"]
    assert repeated.enrollment_index == existing["enrollment_index"]
    with pytest.raises(ValueError, match="development test mode"):
        application._assignment_for_participant(
            "Original", condition_override="explanation"
        )


def test_bundled_page_contains_current_flow_and_no_old_experiment_ui() -> None:
    html = bundled_index_html("test-nonce").decode("utf-8")

    for text in (
        "双机器人协作配送实验",
        "AI–AI 协作演示",
        "开始任务 1",
        "询问机器人2",
        "结束问卷",
    ):
        assert text in html
    lowered = html.lower()
    assert "pretest" not in lowered
    assert "posttest" not in lowered
    assert "free exploration" not in lowered
    assert "自由提问模式" not in html
    assert 'id="workflowExplanation"' not in html
    assert 'id="explanationPanel"' not in html


def test_frontend_uses_one_command_per_action_and_current_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "ui" / "web" / "app.js").read_text(encoding="utf-8")
    html = (root / "ui" / "web" / "index.html").read_text(encoding="utf-8")
    styles = (root / "ui" / "web" / "styles.css").read_text(encoding="utf-8")

    assert 'command("human_action", { action' in source
    assert 'command("begin_task2")' in source
    assert 'task1_complete: "task1CompletePanel"' in source
    assert "if (state.view?.study?.test_condition_selector === true)" in source
    assert 'payload.condition_override = $("testConditionSelector").value;' in source
    assert 'condition_override: $("testConditionSelector").value' not in source
    assert 'id="testConditionSelector"' in html
    assert 'value="explanation" data-i18n="conditionExplanation" selected' in html
    assert "Development test condition" in html
    assert 'id="testConditionStatus"' in html
    assert 'id="assignmentGroupBanner"' in html
    assert "您已分配到 A 组（有解释）" in source
    assert "您已分配到 B 组（无解释）" in source
    assert "You are assigned to Group A (explanations)" in source
    assert "You are assigned to Group B (no explanations)" in source
    assert (root / "run_formal_ui.py").exists()
    run_source = (root / "run.py").read_text(encoding="utf-8")
    assert 'sys.argv.append("--test-condition-selector")' in run_source
    assert '"--formal-study" in sys.argv' in run_source
    assert 'id="questionTarget"' not in html
    assert 'target_agent: "robot_2"' in source
    assert 'id="presetQuestions"' in html
    assert 'data-question-key="presetWhyAction"' in html
    assert 'data-question-key="presetWhyWait"' in html
    assert 'data-question-key="presetHumanInfluence"' in html
    assert 'data-question-key="presetGoal"' in html
    assert 'submitExplanationQuestion(tr(button.dataset.questionKey), button.dataset.questionKind)' in source
    assert "locale: DEFAULT_LOCALE" in source
    assert 'command("timeline_select"' not in source
    assert 'document.querySelectorAll("#presetQuestions button")' in source
    assert "AI–AI 解释轨迹" not in html
    assert "Robot 1 (participant)" not in source
    assert "选择任务 1 的一个实际执行帧" not in source
    assert "Select an executed Task 1 frame" not in source
    assert 'view.timeline?.agent_control || {}' in source
    assert 'aria-label="AI-AI reference trajectory frame"' not in html
    assert "locale: requestedLocale" in source
    assert "Task 2 has no live questions" in source
    assert "任务 2 不提供即时提问" in source
    assert "成功移动耗电 2" in source
    assert "A successful move costs 2 battery" in source
    assert "AI–AI 协作演示（可提前结束）" in source
    assert "提前结束演示并开始任务 1" in source
    assert "Finish demo early and begin Task 1" in source
    assert 'id="taskCards"' not in html
    assert "Active tasks" not in html
    assert "活动任务" not in html
    assert "function renderTasks" not in source
    assert 'const canBeginTask1 = allowed("begin_task1")' in source
    assert 'state.busy || !tutorial.complete' not in source
    assert 'pendingBeginTask1: false' in source
    assert 'button.id === "beginTask1Button"' in source
    assert 'button.disabled = !allowed("begin_task1") || state.pendingBeginTask1' in source
    assert 'if (state.pendingBeginTask1)' in source
    assert 'await command("begin_task1")' in source
    assert 'id="beginTask1Button" type="button" data-i18n="endDemoEarly"' in html
    assert "Task 1 complete" in html
    assert "Begin Task 2" in html
    assert "ArrowUp" in source and '" ":"WAIT"' in source
    assert '"/api/study/command"' in source
    assert "/api/study/posttest" not in source
    assert "enter_free_mode" not in source
    assert "agent.goal_kind" not in source
    assert "agent.goal_position" not in source
    assert "color-scheme: light" in styles
    assert "color-scheme: dark" not in styles
    assert "min-width: 1024px" not in styles
    assert "requestAnimationFrame" in source
    assert "function animateOnce" in source
    assert "function animateLoop" in source
    assert "interpolateTransition" in source
    assert "await animateOnce(beforeView, view.transition, 400)" in source
    assert "paintView(view, true)" in source
    assert "transition.before_state" in source
    assert "elapsed < 600" in source
    assert "elapsed < 850" in source
    assert "elapsed < 925" in source
    assert "setTimeout(resolve, 220)" not in source
    loop_source = source.split("function animateLoop", 1)[1].split(
        "function renderRobots", 1
    )[0]
    assert 'command("' not in loop_source


def test_blinded_explanation_payload_hides_internal_trace_fields() -> None:
    session = WarehouseWebSession.__new__(WarehouseWebSession)
    session.last_explanation_report = {
        "explanation": "Displayed explanation.",
        "raw_explanation": "Hidden raw candidate.",
        "explanation_mode": "rcpd_trace",
        "trace_audit": {"eligible": True},
        "claims": [{"text": "internal"}],
        "verdicts": [{"status": "supported"}],
    }
    session.human_study = SimpleNamespace(assignment=object())

    public = session._public_explanation_report()

    assert public["explanation"] == "Displayed explanation."
    assert public["explanation_mode"] == "blinded"
    assert public["claims"] == ()
    assert public["verdicts"] == ()
    assert "raw_explanation" not in public
    assert "trace_audit" not in public
