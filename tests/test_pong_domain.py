from __future__ import annotations

from copy import deepcopy
import json
import threading
import time
from urllib.request import Request, urlopen

import pytest

from domains.pong.config import PongConfig
from domains.pong.environment.engine import PongEnvironment, predict_next_contact
from domains.pong.environment.model import BallKind
from domains.pong.policies.rule_demo import RuleDemoController
from domains.pong.study import PongStudySession
from domains.pong.web.server import PongApplication, PongHTTPServer


def short_config(**overrides: object) -> PongConfig:
    values = {"fixed_hz": 8, "duration_seconds": 3.0, "ball_speed_y_per_second": 4.0,
              "ball_speed_x_per_second": 4.0, "paddle_speed_per_second": 4.0}
    values.update(overrides)
    return PongConfig(**values)


def put_ball_near_catch(env: PongEnvironment, ball_id: str, *, x: int) -> None:
    ball = next(item for item in env.balls if item.ball_id == ball_id)
    ball.x = x
    ball.y = int(env.config.paddle_y) - ball.height_cells - 1
    ball.vx = 0
    ball.vy = 1
    ball.motion_phase = env.config.ball_step_interval_updates - 1
    ball.descending_encounter = False
    ball.pending_miss = False


def next_encounter(env: PongEnvironment) -> dict[str, object]:
    for _ in range(3):
        transition = env.step("stay", "stay")
        event = next((item for item in transition.events if item.get("event") == "encounter"), None)
        if event:
            return event
    raise AssertionError("expected a catch-line encounter")


def test_four_persistent_balls_and_integer_occupied_cells() -> None:
    env = PongEnvironment(short_config(), seed=4)
    assert [ball.ball_id for ball in env.balls] == ["A1", "A2", "A3", "B1", "B2"]
    assert [ball.kind.value for ball in env.balls] == ["small", "small", "small", "large", "large"]
    frame = env.frame().to_dict()
    for ball in frame["balls"]:
        assert ball["grid_x"] == ball["x"]
        assert ball["occupied_cells"]["left"] == ball["grid_x"]
        assert isinstance(ball["grid_x"], int)
    assert next(ball for ball in frame["balls"] if ball["ball_id"] == "A1")["occupied_cells"] == {"left": frame["balls"][0]["grid_x"], "top": frame["balls"][0]["grid_y"], "width": 1, "height": 1}
    large = next(ball for ball in frame["balls"] if ball["ball_id"] == "B1")
    assert large["occupied_cells"]["width"] == large["occupied_cells"]["height"] == 2
    assert env.config.paddle_width == 4 and env.config.paddle_height_cells == 1


def test_small_ball_catch_uses_exact_displayed_cell() -> None:
    env = PongEnvironment(short_config(), seed=4)
    env.player_x, env.ai_x = 4, 24
    put_ball_near_catch(env, "A1", x=7)
    event = next_encounter(env)
    assert event["ball_id"] == "A1"
    assert event["outcome"] == "caught"
    assert event["coverage"]["contact"] == 7
    assert env.successful_opportunities == 1


def test_paddles_are_clamped_to_the_visible_board() -> None:
    env = PongEnvironment(short_config(), seed=4)
    for _ in range(40):
        env.step("left", "right")
    maximum = int(env.config.width) - int(env.config.paddle_width)
    assert env.player_x == 0
    assert env.ai_x == maximum
    assert env.player_x + int(env.config.paddle_width) <= int(env.config.width)
    assert env.ai_x + int(env.config.paddle_width) <= int(env.config.width)


def test_small_ball_outside_paddle_is_missed_then_bounces_and_persists() -> None:
    env = PongEnvironment(short_config(), seed=4)
    env.player_x, env.ai_x = 0, 24
    put_ball_near_catch(env, "A1", x=12)
    assert next_encounter(env)["outcome"] == "missed"
    for _ in range(8):
        transition = env.step("stay", "stay")
        if any(item.get("event") == "miss_scored" for item in transition.events):
            break
    ball = next(item for item in env.balls if item.ball_id == "A1")
    assert env.missed_balls == 1
    assert ball.vy < 0 and ball.ball_id == "A1"


def test_large_ball_miss_counts_as_three_and_keeps_bouncing() -> None:
    env = PongEnvironment(short_config(), seed=4)
    env.player_x, env.ai_x = 0, 24
    put_ball_near_catch(env, "B1", x=12)
    assert next_encounter(env)["outcome"] == "missed"
    penalty = None
    for _ in range(8):
        transition = env.step("stay", "stay")
        scored = next((item for item in transition.events if item.get("event") == "miss_scored"), None)
        if scored is not None:
            penalty = scored
            break
    ball = next(item for item in env.balls if item.ball_id == "B1")
    assert penalty is not None and penalty["miss_penalty"] == 3
    assert env.missed_balls == 3
    assert env.missed_by_type["large"] == 3
    assert ball.vy < 0


def test_ball_public_state_retains_adjacent_anchors_for_smooth_rendering() -> None:
    env = PongEnvironment(short_config(), seed=4)
    ball = next(item for item in env.balls if item.ball_id == "A1")
    ball.x, ball.y, ball.vx, ball.vy = 7, 4, 1, 1
    ball.motion_phase = env.config.ball_step_interval_updates - 1
    env.step("stay", "stay")
    public = next(item for item in env.frame().to_dict()["balls"] if item["ball_id"] == "A1")
    assert public["previous_grid_x"] == 7
    assert public["grid_x"] == 8
    assert public["motion_phase"] == 0


def test_large_ball_needs_two_distinct_paddles_on_actual_lower_corners() -> None:
    env = PongEnvironment(short_config(), seed=4)
    # B1 occupies x=10..11: its lower contacts are 10 and 11.
    env.player_x, env.ai_x = 7, 11
    put_ball_near_catch(env, "B1", x=10)
    event = next_encounter(env)
    assert event["outcome"] == "caught"
    assert event["contact_cells"] == [10, 11]
    assert event["coverage"]["distinct_paddles"] is True

    env = PongEnvironment(short_config(), seed=4)
    env.player_x, env.ai_x = 8, 20  # only robot1 can cover the two contact cells
    put_ball_near_catch(env, "B1", x=10)
    assert next_encounter(env)["outcome"] == "missed"


def test_prediction_and_engine_share_the_same_contact_cells() -> None:
    env = PongEnvironment(short_config(), seed=4)
    put_ball_near_catch(env, "B1", x=10)
    prediction = predict_next_contact(next(ball for ball in env.balls if ball.ball_id == "B1"), env.frame(), env.config)
    assert prediction is not None
    event = next_encounter(env)
    assert event["contact_cells"] == list(prediction.contact_cells)


def test_snapshot_restores_motion_phase_and_replays_identically() -> None:
    first = PongEnvironment(short_config(), seed=10)
    actions = [("left", "right"), ("stay", "left"), ("right", "stay")] * 4
    for player, ai in actions[:5]:
        first.step(player, ai)
    second = PongEnvironment(short_config(), seed=999)
    second.restore(deepcopy(first.snapshot()))
    for player, ai in actions[5:]:
        assert first.step(player, ai).to_dict() == second.step(player, ai).to_dict()


def test_controller_checks_other_balls_when_player_already_covers_small_ball() -> None:
    env = PongEnvironment(short_config(), seed=2)
    env.player_x, env.ai_x = 4, 21
    put_ball_near_catch(env, "A1", x=5)
    put_ball_near_catch(env, "A2", x=22)
    decision = RuleDemoController(env.config).choose(env.frame())
    assert decision.target_ball_id == "A2"
    assert decision.intent_type in {"catch_small", "hold_target"}


def test_large_side_is_based_on_predicted_contact_not_current_ball_centre() -> None:
    env = PongEnvironment(short_config(), seed=3)
    env.player_x, env.ai_x = 2, 22
    ball = next(item for item in env.balls if item.ball_id == "B1")
    ball.x, ball.y, ball.vx, ball.vy, ball.motion_phase = 9, 12, 1, 1, 0
    decision = RuleDemoController(env.config).choose(env.frame())
    if decision.target_ball_id == "B1":
        prediction = predict_next_contact(ball, env.frame(), env.config)
        assert prediction is not None
        expected = prediction.contact_cells[0] if decision.contact_side == "left" else prediction.contact_cells[1]
        assert decision.target_x <= expected < decision.target_x + int(env.config.paddle_width)


def test_review_uses_named_before_and_after_evidence_without_fallback() -> None:
    session = PongStudySession(group="A", config=short_config(duration_seconds=0.25), seed=5)
    while not session.environment.terminal:
        session.tick("stay")
    response = session.review(1)
    assert response["allowed"] is True
    assert response["decision_evidence"]["frame"] == 1
    assert response["result_evidence"]["frame"] == 0
    answer = session.ask("为什么向这边移动？", 1)
    assert answer["evidence_binding"] == "decision"
    result = session.ask("刚才为什么漏接？", 1)
    assert result["evidence_binding"] == "result"


def test_group_a_review_and_group_b_skip_are_isolated_from_task2() -> None:
    config = short_config(duration_seconds=0.25)
    a = PongStudySession(group="A", config=config, seed=1)
    while not a.environment.terminal:
        a.tick("stay")
    assert a.summary()["review_available"] is True
    assert a.review(1)["allowed"] is True
    a.advance_task()
    assert a.summary()["task"] == 2
    assert a.ask("为什么", 1)["allowed"] is False
    assert "intent_bubble" not in a.summary()

    b = PongStudySession(group="B", config=config, seed=1)
    while not b.environment.terminal:
        b.tick("stay")
    assert b.summary()["review_available"] is False
    assert b.review(1)["allowed"] is False
    b.advance_task()
    assert b.summary()["task"] == 2


def test_live_a_bubble_is_rule_labelled_and_b_is_hidden() -> None:
    a = PongStudySession(group="A", config=short_config(), seed=3)
    bubble = a.summary()["intent_bubble"]
    assert bubble["distance_label"].startswith("距离")
    assert "no_safe_progress" not in bubble["text"]
    assert "秒后到接球线" in bubble["detail_text"] or "没有我能及时承担" in bubble["text"]
    assert "ball_step_interval_updates" in a.summary()
    b = PongStudySession(group="B", config=short_config(), seed=3)
    assert "intent_bubble" not in b.summary()


def test_large_ball_bubble_states_distances_time_and_small_ball_tradeoff() -> None:
    session = PongStudySession(group="A", config=short_config(), seed=4)
    env = session.environment
    env.player_x, env.ai_x = 7, 11
    put_ball_near_catch(env, "B1", x=10)
    for ball_id in ("A1", "A2", "A3", "B2"):
        ball = next(item for item in env.balls if item.ball_id == ball_id)
        ball.vy = -1
    decision = session.controller.choose(env.frame())
    assert decision.target_ball_id == "B1"
    session._update_intent_bubble(decision, env.frame())
    bubble = session.summary()["intent_bubble"]
    assert "机器人1需要同时覆盖另一侧" in bubble["text"]
    assert "我还差" in bubble["detail_text"]
    assert "秒后到接球线" in bubble["detail_text"]


def test_task2_restarts_one_continuous_application_worker() -> None:
    app = PongApplication()
    started = app.start({"group": "B", "participant_id": "test"})
    session_id = started["session_id"]
    with app._lock:
        session = app._sessions[session_id]
        session.environment.phase = "terminal"
        session.frame_history.append(session._public_frame(session.environment.frame()))
    result = app.task2(session_id)
    assert result["task"] == 2 and result["frame"]["terminal"] is False
    time.sleep(0.02)
    assert app._threads[session_id].is_alive()


def test_http_task1_to_task2_lifecycle_restarts_the_worker() -> None:
    app = PongApplication()
    server = PongHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        start_request = Request(
            root + "/api/start", data=json.dumps({"group": "B"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(start_request, timeout=2) as response:
            started = json.loads(response.read())
            cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        session_id = started["session_id"]
        with app._lock:
            app._sessions[session_id].environment.phase = "terminal"
        task2_request = Request(root + "/api/task2", data=b"{}",
                                headers={"Content-Type": "application/json", "Cookie": cookie}, method="POST")
        with urlopen(task2_request, timeout=2) as response:
            task2 = json.loads(response.read())
        assert task2["task"] == 2 and task2["frame"]["terminal"] is False
        assert app._threads[session_id].is_alive()
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
    with pytest.raises(NotImplementedError):
        PongStudySession(group="A", config=short_config(control_mode="frozen_nn"))
