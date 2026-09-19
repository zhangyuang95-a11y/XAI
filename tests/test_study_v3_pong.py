"""Study-only fixtures; no participant records or treatment-effect claims."""
from copy import deepcopy
import itertools
import json
import random

import pytest

from domains.pong import turnbased as pong


def scenario(*, human=2, ai=6, contacts=(2, 6), turns=4, ordinary=None, small_turns=None):
    balls = [pong._ball("fixture-team", "cooperative", list(contacts), turns)]
    if ordinary is not None:
        balls.append(pong._ball("fixture-small", "ordinary", [ordinary], small_turns or turns))
    return pong._new_state([balls], seed=999, task=2, human=human, ai=ai)


def play(state, policy=pong.human_advisor):
    while not state["terminal"]:
        state = pong.step(state, policy(state))
    return state


def test_public_contract_and_configuration():
    for task, waves in ((1, 6), (2, 8), (3, 8)):
        state = pong.initial_state(730100, task)
        assert state["wave_count"] == waves
        assert state["lanes"] == 9
        assert all(1 <= len(w) <= 2 for w in state["_schedule"])
        assert all(4 <= max(b["remaining"] for b in w) <= 6 for w in state["_schedule"])
        public = pong.public_state(state)
        assert {"human", "ai", "wave", "wave_count", "balls", "score"} <= public.keys()
        rendered = json.dumps(public)
        for private in ("policy_memory", "_schedule", '"seed"', "reason_en", "alternatives", "assignment_switches"):
            assert private not in rendered
        assert len(pong.rules()) == len(pong.rules("zh")) == 6
    with pytest.raises(ValueError):
        pong.initial_state(1, 4)


def test_simultaneous_movement_precedes_arrival_and_counts_once():
    state = scenario(human=1, ai=7, turns=1, ordinary=2)
    nxt = pong.step(state, "right")
    assert (nxt["human"]["x"], nxt["ai"]["x"]) == (2, 6)
    assert nxt["raw_score"] == 4
    assert [event["points"] for event in nxt["events"]] == [3, 1]
    assert len(nxt["events"]) == 2
    assert nxt["terminal"]
    with pytest.raises(ValueError):
        pong.step(nxt, "wait")


def test_two_paddles_on_one_team_contact_do_not_catch():
    state = scenario(human=2, ai=2, turns=1)
    nxt = pong.step(state, "wait")
    assert nxt["raw_score"] == 0
    assert nxt["events"][0]["type"] == "missed"
    assert nxt["metrics"]["misses"] == 1


def test_ordinary_overlap_scores_one_not_two():
    state = scenario(human=2, ai=2, turns=1, ordinary=2)
    nxt = pong.step(state, "wait")
    assert nxt["raw_score"] == 1
    assert nxt["metrics"]["ordinary_caught"] == 1
    assert nxt["metrics"]["cooperative_caught"] == 0


def test_boundary_and_stale_decision_rejected():
    state = scenario(human=0)
    assert pong.legal_actions(state) == ["wait", "right"]
    with pytest.raises(ValueError):
        pong.step(state, "left")
    with pytest.raises(ValueError):
        pong.step(state, "teleport")
    forged = pong.decide(state)
    forged["action"] = "left" if forged["action"] != "left" else "right"
    with pytest.raises(ValueError):
        pong.step(state, "wait", forged)


def test_same_state_decision_does_not_see_human_unsubmitted_action():
    state = scenario(human=4, ai=4)
    decision = pong.decide(state)
    positions = {pong.step(state, action, decision)["ai"]["x"] for action in pong.legal_actions(state)}
    assert len(positions) == 1


def test_fixed_tie_break_and_commitment_survive_small_human_move():
    state = scenario(human=4, ai=4)
    first = pong.decide(state)
    assert first["memory"]["commitment"]["ai_contact"] == 6
    moved = pong.step(state, "left")
    assert pong.decide(moved)["memory"] == first["memory"]
    assert not pong.decide(moved)["assignment_changed"]


def test_required_switch_only_after_previous_assignment_impossible():
    state = scenario(human=6, ai=4, contacts=(3, 6), turns=2)
    state["policy_memory"] = {"commitment": {"ball_id": "fixture-team", "ai_contact": 6, "human_contact": 3}}
    decision = pong.decide(state)
    assert decision["reason_code"] == "switch_unreachable_assignment"
    assert decision["memory"]["commitment"]["ai_contact"] == 3
    nxt = pong.step(state, "wait")
    assert nxt["metrics"]["assignment_switches"] == 1
    assert pong.step(nxt, "wait")["raw_score"] == 3


def test_infeasible_cooperation_changes_to_reachable_ordinary():
    state = scenario(human=7, ai=6, turns=1, ordinary=7)
    decision = pong.decide(state)
    assert decision["reason_code"] == "ordinary_after_infeasible_team"
    assert decision["action"] == "right"
    nxt = pong.step(state, "wait")
    assert nxt["raw_score"] == 1


def test_safe_ordinary_detour_returns_for_team_ball():
    state = scenario(ordinary=5, small_turns=2)
    assert pong.decide(state)["reason_code"] == "small_then_return"
    result = play(state, lambda _: "wait")
    assert result["raw_score"] == 4


def test_costly_detour_is_rejected_and_true_human_tradeoff():
    state = scenario(ordinary=4, small_turns=3, turns=4)
    assert pong.decide(state)["action"] == "wait"
    assert pong.decide(state)["reason_code"] == "assign_team_ball"
    held = play(state, lambda _: "wait")
    chasing = deepcopy(state)
    for action in ("right", "right", "wait", "left"):
        chasing = pong.step(chasing, action)
    assert held["raw_score"] == 3
    assert chasing["raw_score"] == 1
    assert pong.wave_feasibility(state)["reachable_points"] == 3


def test_no_reachable_job_means_wait_no_random_fallback():
    state = scenario(human=0, ai=8, contacts=(3, 5), turns=1)
    decision = pong.decide(state)
    assert decision["action"] == "wait"
    assert decision["goal"] == "hold_position"


def test_actual_step_alone_updates_policy_memory_and_evidence_is_pure():
    state = scenario()
    original = deepcopy(state)
    decision = pong.decide(state)
    assert pong.facts(state, decision)
    pong.public_state(state)
    pong.wave_feasibility(state)
    pong.human_advisor(state)
    assert state == original
    assert not state["policy_memory"]
    assert pong.step(state, "wait")["policy_memory"]


def test_hidden_future_cannot_change_decision_facts_or_human_proxy():
    first = pong.initial_state(730103, 2)
    second = deepcopy(first)
    second["seed"] = -999999
    for wave in second["_schedule"][1:]:
        for ball in wave:
            ball["contacts"] = [8 - x for x in ball["contacts"]]
            ball["remaining"] = 99
    assert pong.decide(first) == pong.decide(second)
    assert pong.facts(first) == pong.facts(second)
    assert pong.human_advisor(first) == pong.human_advisor(second)
    assert pong.public_state(first) == pong.public_state(second)


def test_wave_positions_continue_and_schedule_never_adapts_to_score():
    state = pong.initial_state(730108, 2)
    other = deepcopy(state)
    first_duration = max(b["remaining"] for b in state["balls"])
    for _ in range(first_duration):
        state = pong.step(state, pong.human_advisor(state))
        other = pong.step(other, "wait")
    assert state["wave"] == other["wave"] == 2
    assert state["balls"] == other["balls"]
    assert state["human"]["x"] == state["events"][0]["human_x"]
    assert state["ai"]["x"] == state["events"][0]["ai_x"]


def test_all_wait_completes_without_unbounded_blocking():
    state = play(pong.initial_state(730104, 2), lambda _: "wait")
    assert state["turn"] == state["max_turns"]
    assert not state["balls"]
    assert 0 <= pong.score(state)["task_score"] <= 100


def test_fixed_seed_replay_equal_at_every_step():
    first = pong.initial_state(730117, 3)
    second = deepcopy(first)
    rng = random.Random(112)
    while not first["terminal"]:
        action = rng.choice(pong.legal_actions(first))
        first, second = pong.step(first, action), pong.step(second, action)
        assert first == second


def test_search_matches_independent_exhaustive_full_engine_enumeration():
    state = scenario(human=3, ai=6, contacts=(2, 6), turns=4, ordinary=4, small_turns=3)
    feasible_scores = []
    for sequence in itertools.product(("wait", "left", "right"), repeat=4):
        candidate = deepcopy(state)
        try:
            for action in sequence:
                candidate = pong.step(candidate, action)
        except ValueError:
            continue
        feasible_scores.append(candidate["raw_score"])
    optimum = max(feasible_scores)
    assert pong.wave_feasibility(state)["reachable_points"] == optimum
    assert pong.global_reachable_upper_bound(state)["reachable_points"] == optimum


@pytest.mark.parametrize("split", ("development_seeds", "heldout_seeds"))
@pytest.mark.parametrize("task", (1, 2, 3))
def test_24_seed_feasibility_with_actual_fixed_ai(split, task):
    seeds = pong.CONFIG[split]
    assert len(seeds) == len(set(seeds)) == 24
    assert set(pong.CONFIG["development_seeds"]).isdisjoint(pong.CONFIG["heldout_seeds"])
    for seed in seeds:
        result = play(pong.initial_state(seed, task))
        assert result["metrics"]["cooperative_caught"] == result["metrics"]["cooperative_total"]
        assert pong.score(result)["task_score"] >= 90


def test_demonstration_captions_have_real_matching_public_outcomes():
    demo = pong.demonstration()
    assert 4 <= len(demo["captions"]) <= 6
    frames = demo["frames"]
    assert [f["turn"] for f in frames] == list(range(len(frames)))
    expected = {2: ("ordinary", "caught", 1), 4: ("cooperative", "caught", 3),
                8: ("cooperative", "missed", 0), 12: ("cooperative", "caught", 3)}
    for index, (kind, outcome, points) in expected.items():
        assert any(e.get("kind") == kind and e["type"] == outcome and e["points"] == points for e in frames[index]["events"])
    assert frames[-1]["terminal"]
    assert all("reason" not in json.dumps(frame) and "policy_memory" not in json.dumps(frame) for frame in frames)


def test_comprehension_answers_are_grounded_in_separate_actual_scenarios():
    predicted = scenario(human=1, ai=8, turns=4)
    assert pong.decide(predicted)["action"] == "left"
    holding = scenario(turns=2)
    assert pong.decide(holding)["action"] == "wait"
    assert pong.decide(holding)["target_lane"] == holding["ai"]["x"]
    changed = scenario(human=7, ai=6, turns=1, ordinary=7)
    assert pong.decide(changed)["action"] == "right"
    assert [item["answer"] for item in pong.comprehension()] == [0, 1, 2]
    assert [item["answer"] for item in pong.comprehension("zh")] == [0, 1, 2]


def test_evidence_contains_true_times_distances_and_no_internal_jargon():
    state = scenario(human=1, ai=7, turns=3, ordinary=4, small_turns=2)
    evidence = pong.facts(state)
    assert any("3 turns" in fact["en"] for fact in evidence)
    assert any("lane 8" in fact["en"] for fact in evidence)
    texts = " ".join(row[lang] for row in evidence for lang in ("en", "zh"))
    for term in ("NN", "embedding", "checkpoint", "logit", "reward shaping", "神经网络"):
        assert term not in texts


def test_score_is_unmodified_nonnegative_fraction_of_all_scheduled_points():
    state = scenario(ordinary=4, small_turns=3)
    result = play(state, lambda _: "wait")
    score = pong.score(result)
    assert score["raw_score"] == 3
    assert score["metrics"]["total_possible_points"] == 4
    assert score["task_score"] == 75
