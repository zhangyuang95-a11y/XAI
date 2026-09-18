from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

import pytest

from domains.pong.config import PongConfig
from domains.pong.environment.engine import PongEnvironment, predict_next_contact
from domains.pong.policies.rule_demo import ControllerDecision, RuleDemoController
from domains.pong.study import PongStudySession
from domains.pong.web.server import PongApplication, PongHTTPServer


def short_config(**overrides: object) -> PongConfig:
    values = {
        "fixed_hz": 60,
        "duration_seconds": 1.0,
        "ball_speed_y_per_second": 2.0,
        "ball_speed_x_per_second": 2.0,
        "paddle_speed_per_second": 5.0,
    }
    values.update(overrides)
    return PongConfig(**values)


def ball(env: PongEnvironment, ball_id: str):
    return next(item for item in env.balls if item.ball_id == ball_id)


def put_ball_near_catch(env: PongEnvironment, ball_id: str, *, x: float) -> None:
    item = ball(env, ball_id)
    item.x = x
    item.y = env.config.paddle_y - item.height_cells - 0.01
    item.vx = 0.0
    item.vy = env.config.ball_speed_y_per_second
    item.descending_encounter = False
    item.pending_miss = False


def next_encounter(env: PongEnvironment, *, limit: int = 8) -> dict[str, object]:
    for _ in range(limit):
        transition = env.step("stay", "stay")
        event = next((item for item in transition.events if item.get("event") == "encounter"), None)
        if event:
            return event
    raise AssertionError("expected an encounter")


def hide_other_balls(env: PongEnvironment, keep: set[str]) -> None:
    for item in env.balls:
        if item.ball_id not in keep:
            item.y = 0.0
            item.vy = -env.config.ball_speed_y_per_second


def test_default_is_a_24x14_continuous_five_ball_game() -> None:
    config = PongConfig()
    config.validate()
    assert (config.grid_columns, config.grid_rows, config.paddle_y) == (24, 14, 11)
    assert config.fixed_hz == 60
    assert config.paddle_width == 4
    assert config.ball_ids == ("A1", "A2", "A3", "B1", "B2")
    env = PongEnvironment(config, seed=4)
    public = env.frame().to_dict()
    assert all(isinstance(item["x"], float) for item in public["balls"])
    assert next(item for item in public["balls"] if item["ball_id"] == "A1")["occupied_cells"]["width"] == 1
    large = next(item for item in public["balls"] if item["ball_id"] == "B1")
    assert large["occupied_cells"]["width"] == large["occupied_cells"]["height"] == 2


def test_browser_loop_keeps_fixed_physics_steps_independent_of_rendering() -> None:
    app = (Path(__file__).parents[1] / "domains" / "pong" / "web" / "app.js").read_text()
    assert "function simulationTick()" in app
    assert "game.step(currentAction);" in app
    assert "const interval = SPEC.fixedDt * 1000;" in app
    assert "requestAnimationFrame(animationLoop);" in app
    assert "while (physicsAccumulator >= interval" in app
    assert "模型观察签名与当前 Pong v2 物理规则不兼容" in app
    assert "浏览器缺少模型所需特征" in app


def test_paddles_move_continuously_and_never_leave_the_board() -> None:
    env = PongEnvironment(short_config(), seed=4)
    start = env.player_x
    env.step("right", "stay")
    assert env.player_x == pytest.approx(start + 5 / 60)
    for _ in range(400):
        env.step("left", "right")
    maximum = env.config.width - env.config.paddle_width
    assert env.player_x == pytest.approx(0.0)
    assert env.ai_x == pytest.approx(maximum)


def test_small_ball_catch_uses_the_visible_continuous_contact_coordinate() -> None:
    env = PongEnvironment(short_config(), seed=4)
    hide_other_balls(env, {"A1"})
    env.player_x, env.ai_x = 4.0, 19.0
    put_ball_near_catch(env, "A1", x=7.0)
    event = next_encounter(env)
    assert event["ball_id"] == "A1"
    assert event["outcome"] == "caught"
    assert float(event["coverage"]["contact"]) == pytest.approx(7.0)


def test_large_ball_requires_two_distinct_paddles_at_two_actual_contacts() -> None:
    env = PongEnvironment(short_config(), seed=4)
    hide_other_balls(env, {"B1"})
    env.player_x, env.ai_x = 7.0, 11.0
    put_ball_near_catch(env, "B1", x=10.0)
    caught = next_encounter(env)
    assert caught["outcome"] == "caught"
    assert caught["coverage"]["distinct_paddles"] is True

    env = PongEnvironment(short_config(), seed=4)
    hide_other_balls(env, {"B1"})
    env.player_x, env.ai_x = 8.0, 18.0
    put_ball_near_catch(env, "B1", x=10.0)
    assert next_encounter(env)["outcome"] == "missed"


def test_large_miss_scores_three_once_then_the_persistent_ball_bounces() -> None:
    env = PongEnvironment(short_config(duration_seconds=4.0), seed=4)
    hide_other_balls(env, {"B1"})
    env.player_x, env.ai_x = 0.0, 18.0
    put_ball_near_catch(env, "B1", x=10.0)
    assert next_encounter(env)["outcome"] == "missed"
    for _ in range(120):
        transition = env.step("stay", "stay")
        if any(item.get("event") == "miss_scored" for item in transition.events):
            break
    assert env.missed_balls == 3
    assert env.missed_by_type["large"] == 3
    assert ball(env, "B1").vy < 0


def test_prediction_and_engine_share_the_same_continuous_contact_geometry() -> None:
    env = PongEnvironment(short_config(), seed=4)
    hide_other_balls(env, {"B1"})
    put_ball_near_catch(env, "B1", x=10.0)
    prediction = predict_next_contact(ball(env, "B1"), env.frame(), env.config)
    assert prediction is not None
    event = next_encounter(env)
    assert event["contact_cells"] == pytest.approx(list(prediction.contact_cells))


def test_prediction_remains_exact_after_a_side_wall_reflection() -> None:
    env = PongEnvironment(short_config(duration_seconds=5.0), seed=4)
    hide_other_balls(env, {"A1"})
    item = ball(env, "A1")
    item.x, item.y, item.vx, item.vy = 21.8, 5.0, 2.0, 2.0
    prediction = predict_next_contact(item, env.frame(), env.config)
    assert prediction is not None
    for _ in range(240):
        transition = env.step("stay", "stay")
        event = next((value for value in transition.events if value.get("event") == "encounter"), None)
        if event:
            assert event["contact_cells"] == pytest.approx(list(prediction.contact_cells))
            break
    else:
        raise AssertionError("expected a reflected contact")


def test_snapshot_restores_continuous_positions_and_replays_identically() -> None:
    first = PongEnvironment(short_config(duration_seconds=3.0), seed=10)
    actions = [("left", "right"), ("stay", "left"), ("right", "stay")] * 8
    for player, ai in actions[:5]:
        first.step(player, ai)
    second = PongEnvironment(short_config(duration_seconds=3.0), seed=999)
    second.restore(deepcopy(first.snapshot()))
    for player, ai in actions[5:]:
        assert first.step(player, ai).to_dict() == second.step(player, ai).to_dict()


def test_controller_hands_a_small_ball_to_player_when_already_covered() -> None:
    env = PongEnvironment(short_config(), seed=2)
    hide_other_balls(env, {"A1", "A2"})
    env.player_x, env.ai_x = 4.0, 18.0
    put_ball_near_catch(env, "A1", x=5.0)
    put_ball_near_catch(env, "A2", x=19.0)
    decision = RuleDemoController(env.config).choose(env.frame())
    assert decision.target_ball_id == "A2"
    assert decision.intent_type in {"catch_small", "hold_target"}


def test_large_commitment_survives_a_newer_small_ball_until_encounter() -> None:
    env = PongEnvironment(short_config(duration_seconds=4.0), seed=2)
    hide_other_balls(env, {"B1", "A1"})
    env.player_x, env.ai_x = 7.0, 11.0
    put_ball_near_catch(env, "B1", x=10.0)
    # Give B1 more than one frame so the controller first commits to it.
    ball(env, "B1").y -= 0.20
    controller = RuleDemoController(env.config)
    first = controller.choose(env.frame())
    assert first.target_ball_id == "B1"
    put_ball_near_catch(env, "A1", x=15.0)
    second = controller.choose(env.frame())
    assert second.target_ball_id == "B1"
    assert second.commitment_state in {"approaching_large", "waiting_for_large"}


def test_small_first_check_counts_waiting_until_the_small_ball_is_caught() -> None:
    env = PongEnvironment(short_config(duration_seconds=4.0), seed=3)
    hide_other_balls(env, {"B1", "A1"})
    env.player_x, env.ai_x = 7.0, 11.0
    b1, a1 = ball(env, "B1"), ball(env, "A1")
    b1.x, b1.y, b1.vx, b1.vy = 10.0, 7.0, 0.0, 2.0  # B1 reaches the line in 1 s.
    a1.x, a1.y, a1.vx, a1.vy = 5.0, 7.0, 0.0, 2.0   # A1 reaches it later.
    controller = RuleDemoController(env.config)
    candidates = controller._candidates(env.frame())
    large = next(item for item in candidates if item.prediction.ball_id == "B1")
    first_small, feasible = controller._small_first_feasibility(large, candidates)
    assert first_small == "A1"
    assert feasible is False


def test_group_a_bubble_has_stable_assignment_without_seconds_and_group_b_is_hidden() -> None:
    a = PongStudySession(group="A", config=short_config(), seed=3)
    bubble = a.summary()["intent_bubble"]
    assert "秒" not in bubble["text"] + bubble["detail_text"]
    assert "no_safe_progress" not in bubble["text"]
    assert a.summary()["continuous_motion"] is True
    b = PongStudySession(group="B", config=short_config(), seed=3)
    assert "intent_bubble" not in b.summary()


def test_same_competing_opportunity_is_recorded_once_not_once_per_physics_frame() -> None:
    session = PongStudySession(group="A", config=short_config(duration_seconds=2.0), seed=3)
    # The controller can see the same pair for many 60 Hz updates.  The replay
    # should retain the decision point once, rather than flood the event list.
    decision = ControllerDecision(
        "stay", "test", target_ball_id="A1", opportunity_id="A1:1",
        conflict_ball_ids=("A1", "A2"),
    )
    for _ in range(2):
        before = session.environment.frame()
        transition = session.environment.step("stay", "stay")
        session._record_review_events(before, transition, decision)
    conflicts = [event for event in session.review_events if event["type"] == "competing_opportunities"]
    assert len(conflicts) == 1


def test_group_a_review_and_group_b_skip_remain_isolated_from_task2() -> None:
    config = short_config(duration_seconds=0.05)
    a = PongStudySession(group="A", config=config, seed=1)
    while not a.environment.terminal:
        a.tick("stay")
    assert a.review(1)["allowed"] is True
    a.advance_task()
    assert a.summary()["task"] == 2
    assert a.ask("为什么", 1)["allowed"] is False
    assert "intent_bubble" not in a.summary()

    b = PongStudySession(group="B", config=config, seed=1)
    while not b.environment.terminal:
        b.tick("stay")
    assert b.review(1)["allowed"] is False
    b.advance_task()
    assert b.summary()["task"] == 2


def test_application_runs_at_the_configured_fixed_rate_and_restarts_task2() -> None:
    app = PongApplication()
    started = app.start({"group": "B", "participant_id": "test"})
    session_id = started["session_id"]
    with app._lock:
        session = app._sessions[session_id]
        session.environment.phase = "terminal"
        session.frame_history.append(session._public_frame(session.environment.frame()))
    result = app.task2(session_id)
    assert result["task"] == 2 and result["frame"]["terminal"] is False
    time.sleep(0.04)
    assert app._threads[session_id].is_alive()


def test_http_task1_to_task2_lifecycle_restarts_the_worker() -> None:
    app = PongApplication()
    server = PongHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = Request(root + "/api/start", data=json.dumps({"group": "B"}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=2) as response:
            started = json.loads(response.read())
            cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        session_id = started["session_id"]
        with app._lock:
            app._sessions[session_id].environment.phase = "terminal"
        task2 = Request(root + "/api/task2", data=b"{}",
                        headers={"Content-Type": "application/json", "Cookie": cookie}, method="POST")
        with urlopen(task2, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["task"] == 2 and payload["frame"]["terminal"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_invalid_action_and_unsupported_nn_are_explicit() -> None:
    env = PongEnvironment(short_config(), seed=2)
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step("teleport", "stay")
    assert env.snapshot() == before
    with pytest.raises(ValueError, match="explicit frozen nn_policy"):
        PongStudySession(group="A", config=short_config(control_mode="frozen_nn"))
