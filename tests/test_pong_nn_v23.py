"""Small, fixed checks of the v2.3 controller and evidence path."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from domains.pong.config import PongConfig
from domains.pong.environment.engine import PongEnvironment
from domains.pong.policies.coordinated import CoordinatedPongController, VERSION

ROOT = Path(__file__).resolve().parents[1]
NODE = Path("/Users/zhangyuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")


def test_controller_chooses_inward_action_at_boundary_and_records_actual_source() -> None:
    config = PongConfig()
    env = PongEnvironment(config, seed=260918)
    env.ai_x = 0.0
    controller = CoordinatedPongController(config)
    decision = controller.choose(env, "left")
    assert decision.rule_version == VERSION
    assert decision.evidence["discarded_neural_action"] == "left"
    assert decision.controller_selected_action in {"right", "stay"}
    assert decision.evidence["excluded_future_player_action"] is True


def test_ready_large_ball_strict_coverage_edge_is_repaired_before_contact() -> None:
    config = PongConfig()
    env = PongEnvironment(config, seed=7010)
    env.reset_training_episode(seed=7010, scenario_name="large_both_ready", active_ball_ids={"B1"})
    controllers = [CoordinatedPongController(config, agent=agent) for agent in ("player", "ai")]
    outcome = None
    for _ in range(8):
        actions = [controller.choose(env, "stay").controller_selected_action for controller in controllers]
        for _ in range(6):
            transition = env.step(*actions)
            event = next((item for item in transition.events
                          if item.get("event") == "encounter" and item.get("ball_id") == "B1"), None)
            if event is not None:
                outcome = event["outcome"]
                break
        if outcome is not None:
            break
    assert outcome == "caught"


@pytest.mark.skipif(not NODE.exists(), reason="bundled Node runtime unavailable")
def test_browser_and_python_choose_same_actions_on_fixed_scenes() -> None:
    config = PongConfig()
    cases = []
    expected = []
    for seed in range(260918, 260922):
        env = PongEnvironment(config, seed=seed)
        controller = CoordinatedPongController(config)
        for proposal in ("left", "stay", "right"):
            decision = controller.choose(env, proposal)
            cases.append({"seed": seed, "frame_index": env.frame_index, "player_x": env.player_x,
                          "ai_x": env.ai_x, "balls": [ball.to_dict() for ball in env.balls],
                          "proposal": proposal})
            expected.append((decision.controller_selected_action, decision.evidence["target_ball_id"],
                             decision.evidence["contact_side"]))
            controller.reset()
            for _ in range(6):
                env.step("stay", "stay")
    result = subprocess.run([str(NODE), str(ROOT / "scripts/check_pong_v23_browser.cjs")],
                            input=json.dumps({"cases": cases}), text=True, capture_output=True, check=True)
    observed = json.loads(result.stdout)
    assert [(item["action"], item["evidence"]["target_ball_id"],
             item["evidence"]["contact_side"]) for item in observed] == expected


@pytest.mark.skipif(not NODE.exists(), reason="bundled Node runtime unavailable")
def test_freeform_questions_use_question_and_actual_event() -> None:
    result = subprocess.run([str(NODE), str(ROOT / "tests/pong_v23_explanation.cjs")],
                            text=True, capture_output=True, check=True)
    assert "question-aware Pong explanations: passed" in result.stdout
