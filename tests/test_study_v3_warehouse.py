"""Mechanism checks plus fixed-AI human-only feasibility certificates."""
from copy import deepcopy
import json

import pytest

from domains.warehouse import turnbased as w


def charge_fixture():
    state = w.initial_state(100, 2)
    state["human"].update(x=1, y=5, battery=90)
    state["ai"].update(x=4, y=2, battery=40)
    state["orders"][3].update(pickup=[7, 1], dropoff=[1, 4])
    return state


def crossing_fixture():
    state = w.initial_state(100, 2)
    state["human"].update(x=3, y=3, battery=90)
    state["ai"].update(x=5, y=3, battery=90, carrying="A1")
    state["orders"][3].update(status="carried", dropoff=[1, 3])
    return state


def test_path_distance_uses_walls():
    # The central wall blocks the apparent four-step straight route.
    assert w._distance((2, 5), (6, 5)) == 8
    assert w._distance((1, 5), (4, 2)) == 6


def test_decision_and_facts_never_mutate_state_or_read_extra_future():
    state = charge_fixture()
    before = deepcopy(state)
    decision = w.decide(state)
    assert w.decide(state) == decision
    w.facts(state, decision)
    assert state == before
    other = deepcopy(state)
    other["future_schedule"] = [{"private_future": "changed"}]
    other["unsubmitted_human_action"] = "right"
    other["group"] = "B"
    assert w.decide(other) == decision
    assert w.public_state(other) == w.public_state(state)


def test_charging_threshold_and_departure_are_exact():
    state = charge_fixture()
    # Independent sum: charger→pickup 4 + pickup→dropoff 9 +
    # dropoff→charger 5 + two reserve moves = 20 moves = 60%.
    assert w._energy_required(state, "ai") == 60
    assert w.decide(state)["action"] == "wait"
    state = w.step(state, "wait")
    assert state["ai"]["battery"] == 50
    state = w.step(state, "wait")
    assert state["ai"]["battery"] == 60
    assert w.decide(state)["reason_code"] == "leave_charger"
    assert w.decide(state)["action"] != "wait"
    assert w._pos(w.step(state, "wait")["ai"]) != w.CHARGER


def test_wait_unlocks_when_possible_human_conflict_removed():
    state = crossing_fixture()
    assert w.decide(state)["action"] == "wait"
    next_state = w.step(state, "left")
    assert w.decide(next_state)["action"] == "left"


def test_same_risk_active_clearance_releases_occupied_route():
    state = crossing_fixture()
    state["human"].update(x=4, y=3)
    state["ai"]["carrying"] = None
    state["orders"][3].update(status="available", pickup=[1, 3])
    state["orders"][0].update(pickup=[7, 3])
    decision = w.decide(state)
    assert decision["action"] in ("up", "down", "right")
    assert decision["reason_code"] == "clear_shared_route"
    assert decision["facts"][1]["id"] == "ai_risk"
    next_state = w.step(state, "wait")
    assert w._pos(next_state["ai"]) != (5, 3)
    assert next_state["collision_events"] == 0
    assert "simultaneous" in decision["reason_en"]


def test_charger_has_safe_escape_even_when_person_blocks_one_exit():
    state = charge_fixture()
    state["ai"]["battery"] = 90
    state["human"].update(x=4, y=3, battery=6)
    decision = w.decide(state)
    assert decision["action"] in ("left", "right")
    next_state = w.step(state, "wait")
    assert w._pos(next_state["ai"]) != w.CHARGER


def test_loaded_crossing_commitment_is_stable_and_observably_released():
    state = crossing_fixture()
    state["human"].update(x=4, y=3)
    state["orders"][0].update(pickup=[7, 3])
    assert w.decide(state)["reason_code"] == "committed_crossing_wait"
    for _ in range(3):
        state = w.step(state, "wait")
        assert state["last_actions"]["ai"] == "wait"
    # A human retreat through two real steps clears the possible next-cell
    # conflict. The policy advances without needing any special command.
    state = w.step(state, "left")
    state = w.step(state, "left")
    assert w.decide(state)["action"] == "left"
    assert w._pos(w.step(state, "wait")["ai"]) == (4, 3)


def test_charge_shortfall_answers_arithmetic_without_user_inference():
    state = charge_fixture()
    fact = next(row for row in w.facts(state) if row["id"] == "ai_charge_shortfall")
    assert "20% more" in fact["en"]
    assert "2 charging turns" in fact["en"]
    assert "还差20%" in fact["zh"]
    assert state["turn"] == 0 and state["ai"]["battery"] == 40


def test_collision_once_preserves_human_submission_and_no_move_energy():
    state = crossing_fixture()
    state["human"].update(x=4, y=3)
    original = deepcopy(state)
    state = w.step(state, "right")
    assert state["last_actions"]["human"] == "right"
    assert state["human"] == original["human"]
    assert state["ai"] == original["ai"]
    assert state["collision_events"] == 1
    assert len([e for e in state["events"] if e["type"] == "collision"]) == 1
    assert w.score(state)["raw_score"] == -201


def test_shutdown_is_new_event_once_and_charge_can_recover_at_station():
    state = charge_fixture()
    state["human"].update(x=1, y=5, battery=4)
    state = w.step(state, "up")
    assert state["human"]["battery"] == 1
    assert state["shutdown_events"] == 1
    assert w.legal_actions(state) == ["wait"]
    state = w.step(state, "wait")
    assert state["shutdown_events"] == 1
    state["human"].update(x=4, y=2, battery=1)
    state["ai"].update(x=7, y=5, battery=90)
    state = w.step(state, "wait")
    assert state["human"]["battery"] == 11
    assert not state["human"]["shutdown"]


def test_charge_caps_and_only_confirmed_wait_charges():
    state = charge_fixture()
    state["human"].update(x=4, y=2, battery=95)
    state["ai"].update(x=7, y=5, battery=90)
    next_state = w.step(state, "wait")
    assert next_state["human"]["battery"] == 100
    assert next(e for e in next_state["events"] if e["actor"] == "human")["amount"] == 5
    assert w.step(state, "left")["human"]["battery"] == 92


def test_pickup_delivery_finite_and_score_not_group_dependent():
    state = charge_fixture()
    state["human"].update(x=1, y=5)
    state["orders"][0].update(pickup=[1, 4], dropoff=[2, 4])
    state = w.step(state, "up")
    assert state["human"]["carrying"] == "H1"
    state = w.step(state, "right")
    assert state["deliveries"] == 1
    assert state["human"]["carrying"] is None
    assert w.score(state)["task_score"] == pytest.approx(100 / 6, abs=1e-6)
    state["group"] = "A"
    assert w.score(state)["task_score"] == pytest.approx(100 / 6, abs=1e-6)
    assert len(state["orders"]) == 6


def test_terminal_illegal_stale_decision_rejections():
    state = w.initial_state(100, 1)
    with pytest.raises(ValueError, match="illegal"):
        w.step(state, "teleport")
    decision = w.decide(state)
    decision["action"] = "wait"
    with pytest.raises(ValueError, match="decision"):
        w.step(state, "wait", decision)
    state["turn"] = 119
    state = w.step(state, "wait")
    assert state["terminal"]
    with pytest.raises(ValueError, match="complete"):
        w.step(state, "wait")
    assert w.legal_actions(state) == []


def test_public_projection_whitelist_and_json_roundtrip():
    state = charge_fixture()
    state["policy_memory"] = {"secret": "should not be public"}
    public = w.public_state(state)
    assert not {"policy_memory", "seed", "last_actions", "decision"}.intersection(public)
    assert set(public["ai"]) == {"x", "y", "battery", "carrying"}
    assert json.loads(json.dumps(public)) == public
    assert "secret" not in json.dumps(public)


@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("seed", w.CONFIG["development_seeds"] + w.CONFIG["heldout_seeds"])
def test_fixed_ai_reaches_all_six_deliveries_in_original_budget(seed, task):
    state = w.initial_state(seed, task)
    while not state["terminal"]:
        decision = w.decide(state)
        action = w.human_advisor(state)
        next_state = w.step(state, action, decision)
        assert next_state["last_actions"]["ai"] == decision["action"]
        assert next_state == w.step(deepcopy(state), action, deepcopy(decision))
        state = next_state
    assert state["deliveries"] == 6
    assert state["turn"] <= 120
    assert state["collision_events"] == state["shutdown_events"] == 0


def test_demonstration_is_real_execution_with_failure_and_repair():
    demonstration = w.demonstration()
    assert 4 <= len(demonstration["captions"]) <= 6
    frames = demonstration["frames"]
    assert [frame["turn"] for frame in frames] == list(range(len(frames)))
    events = [event["type"] for frame in frames for event in frame["events"]]
    assert {"collision", "charge", "pickup", "delivery", "complete"} <= set(events)
    assert frames[-1]["score"]["task_score"] == 100
    assert all("reason" not in caption["en"].lower() for caption in demonstration["captions"])


def test_comprehension_answers_match_verified_fixtures():
    assert w.decide(charge_fixture())["action"] == "wait"
    assert w.decide(crossing_fixture())["reason_code"] == "committed_crossing_wait"
    state = charge_fixture()
    state["ai"]["battery"] = 60
    assert w.decide(state)["reason_code"] == "leave_charger"
    for language in ("en", "zh"):
        items = w.comprehension(language)
        assert len(items) == 3
        assert [item["answer"] for item in items] == [0, 1, 2]
