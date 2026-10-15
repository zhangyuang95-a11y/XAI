"""Evidence agrees with unchanged physical outcomes, not just answer strings."""
from copy import deepcopy
from hashlib import sha256
import json

from domains.warehouse import turnbased as w
from domains.warehouse.build_historical_demo import build, ASSET
from env.warehouse.frozen_missions import frozen_training_missions


def test_frozen_demo_exactly_reproduces_original_render_source_in_isolation():
    rebuilt = build()
    recorded = w.demonstration()
    assert rebuilt == recorded
    assert sha256((w.ROOT / ASSET).read_bytes()).hexdigest() == w.CONFIG["historical_demo"]["sha256"]
    assert len(recorded["frames"]) == 121
    assert recorded["frames"][0]["ai"]["battery"] == 35
    assert recorded["frames"][1]["demo_actions"] == {"human": "up", "ai": "left"}
    assert recorded["frames"][-1]["score"]["raw_score"] == 1180
    assert recorded["frames"][-1]["score"]["metrics"]["deliveries"] == 13
    assert recorded["frames"][-1]["score"]["metrics"]["collision_events"] == 0
    assert recorded["provenance"]["historical_deployment_timestamp_verified"] is False
    assert recorded["provenance"]["kind"] == "historical_source_replay"
    assert "snapshot" not in json.dumps(recorded)
    # The original demo score is never remapped into the task's +10 scale.
    delivery = next(f for f in recorded["frames"] if f["score"]["metrics"]["deliveries"])
    assert delivery["score"]["breakdown"]["delivery"] == 100
    assert "+10 per delivery" in " ".join(w.rules())


def test_wait_explains_specific_potential_joint_action_without_predicting_human():
    state = w.step(w.initial_state(1, 2), "up")
    decision = w.decide(state)
    assert decision["action"] == "wait"
    assert "If I moved left and you moved right" in decision["reason_en"]
    assert "column 3, row 4" in decision["reason_en"]
    assert "如果我向左、你向右" in decision["reason_zh"]
    env = w._restore(state)
    _, _, _, collision, kind, attempted = env._resolve_motion(env.state, {"robot_1": "RIGHT", "robot_2": "LEFT"})
    assert collision and kind == "same_target"
    assert attempted["robot_1"] == attempted["robot_2"] == (4, 3)
    assert state["last_actions"]["human"] == "up"  # conditional RIGHT is not an observed command


def test_action_safety_claims_exhaust_all_five_commands_and_include_residual_risk():
    saw_residual_wait_risk = False
    for task in (1, 2, 3):
        state = w.initial_state(1, task)
        for _ in range(15):
            before = deepcopy(state)
            decision = w.decide(state)
            env = w._restore(state)
            expected = [action.lower() for action in w.ACTIONS if env._resolve_motion(
                env.state, {"robot_1": action, "robot_2": decision["action"].upper()})[3]]
            assert [c["human_action"] for c in decision["collision_checks"]["cases"]] == expected
            assert set(decision["collision_checks"]["human_commands_checked"]) == set(w.legal_actions(state))
            assert ("none of your five commands" in decision["reason_en"]) == (not expected)
            if expected and decision["action"] == "wait":
                saw_residual_wait_risk = True
                assert "If I waited and you moved" in decision["reason_en"]
                assert "都不会" not in decision["reason_zh"]
            assert state == before
            state = w.step(state, w.human_advisor(state))
    assert saw_residual_wait_risk


def test_pickup_delivery_charger_names_and_route_distances_match_current_board():
    state = w.initial_state(1, 2)
    saw_pickup = saw_delivery = saw_charger = False
    for _ in range(65):
        decision = w.decide(state)
        env = w._restore(state)
        mission = frozen_training_missions(env, env.state)["robot_2"]
        assert decision["goal_details"]["position"] == list(reversed(mission.goal_position))
        c = decision["controller_trace"]["selected_ai_action"]
        assert c["distance_before"] == w._distance(w._pos(state["ai"]), decision["goal_details"]["position"])
        assert c["distance_after"] == w._distance(tuple(reversed(c["target"])), decision["goal_details"]["position"])
        if mission.goal_kind == "charge":
            saw_charger = True
            assert decision["goal_details"]["label"] == "charger"
            assert decision["goal_details"]["position"] == list(w.CHARGER)
        elif mission.task:
            slot = next(i+1 for i,t in enumerate(env.state.tasks) if t.task_id == mission.task.task_id)
            expected = ("A" if mission.goal_kind == "pickup" else "B") + str(slot)
            assert decision["goal_details"]["label"] == expected
            coordinate = state["orders"][slot-1]["pickup" if expected.startswith("A") else "dropoff"]
            assert decision["goal_details"]["position"] == coordinate
            saw_pickup |= expected.startswith("A")
            saw_delivery |= expected.startswith("B")
        if decision["action"] != "wait":
            c = decision["controller_trace"]["selected_ai_action"]
            assert f"from {c['distance_before']} to {c['distance_after']} moves" in decision["reason_en"]
            if decision["collision_checks"]["cases"]:
                assert "If my movement completes" in decision["reason_en"]
        state = w.step(state, w.human_advisor(state))
    assert saw_pickup and saw_delivery and saw_charger


def test_collision_animation_uses_real_attempted_and_cancelled_positions_only():
    state = w.initial_state(1, 1)
    for _ in range(20):
        env = w._restore(state)
        decision = w.decide(state)
        cases = w._collision_cases(env, decision["action"])
        if cases:
            chosen = cases[0]
            after = w.step(state, chosen["human_action"])
            public = w.public_state(after)
            animation = public["collision_animation"]
            assert animation["kind"] == chosen["kind"]
            for actor in ("human", "ai"):
                assert animation[actor] == chosen[actor]
                assert animation[actor]["to"] == animation[actor]["from"]
                assert animation[actor]["to"] == [after[actor]["x"], after[actor]["y"]]
            assert any(animation[a]["attempted"] != animation[a]["to"] for a in ("human", "ai"))
            event = next(e for e in after["events"] if e["type"] == "collision")
            assert event["collision_animation"] == animation
            assert w.score(after)["breakdown"]["robot_collision"] == -10
            assert "controller_trace" not in json.dumps(public)
            break
        assert "collision_animation" not in w.public_state(state)
        state = w.step(state, w.human_advisor(state))
    else:
        raise AssertionError("No actual collision candidate found in preserved trajectory")


def test_animation_reports_same_cell_swap_and_stationary_collisions_from_physics():
    for positions, actions, kind in (
        (((5, 2), (5, 4)), {"robot_1": "RIGHT", "robot_2": "LEFT"}, "same_target"),
        (((5, 2), (5, 3)), {"robot_1": "RIGHT", "robot_2": "LEFT"}, "swap"),
        (((5, 2), (5, 3)), {"robot_1": "RIGHT", "robot_2": "WAIT"}, "occupied_stationary"),
    ):
        env = w._restore(w.initial_state(1, 1))
        raw = env.get_state()
        for agent, position in zip(raw.agents, positions):
            agent.position = position
        env.set_state(raw)
        before = env.get_state()
        info = env.step(actions)[-1]
        event = next(e for e in w._events(before, env.state, info) if e["type"] == "collision")
        motion = event["collision_animation"]
        assert motion["kind"] == kind
        for actor, key in w._ACTORS.items():
            assert motion[actor]["from"] == list(reversed(before.by_id(key).position))
            assert motion[actor]["attempted"] == list(reversed(info["intended_targets"][key]))
            assert motion[actor]["to"] == motion[actor]["from"]
        assert env.state.user_score - before.user_score == -11  # unchanged collision + time
