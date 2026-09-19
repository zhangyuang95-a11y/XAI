"""Parity against the actual historical policy and preserved Warehouse physics."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import subprocess
import types

import pytest

from domains.warehouse import turnbased as w
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.navigation import ACTIONS


def historic_from_git():
    raw = subprocess.check_output(["git", "show", w.CONFIG["controller_commit"] + ":env/warehouse/runtime_coordination.py"], cwd=w.ROOT)
    module = types.ModuleType("env.warehouse.independent_history_test")
    module.__package__ = "env.warehouse"
    exec(compile(raw, "git:de16551:runtime_coordination", "exec"), module.__dict__)
    return raw, module


def old_initial(task):
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=w.CONFIG["task_seeds"][str(task)])
    s = env.get_state(); s.participant_controlled_agent_id = "robot_1"; env.set_state(s)
    return env


def old_step(env, history, seed, human):
    s = env.get_state()
    proposed, dist = w._actor().act(env.observations(), deterministic=False,
        base_seed=seed, decision_key=(s.episode_id, s.frame))
    action, runtime = history.select_human_ai_action(env, proposed["robot_2"])
    submitted, guard = history.guard_participant_action(env, human.upper())
    actions = {**proposed, "robot_1": submitted, "robot_2": action}
    runtime = {**runtime, "participant_action_guard": guard, "selected_actions": dict(actions)}
    info = env.step(actions, decision_metadata=distribution_decision_metadata(dist,
        decision_source="participant_plus_robust_numpy_actor", participant_overrides={"robot_1": submitted},
        policy_actions=proposed, selected_actions=actions, runtime_decision=runtime))[-1]
    return action, runtime, info


def test_exact_history_module_actor_and_dependencies_are_pinned():
    raw, _ = historic_from_git()
    assert raw == (w.ROOT / "env/warehouse/historical_sep2_coordination.py").read_bytes()
    assert sha256(raw).hexdigest() == "ebcbf38c6365752a73b2175ab44750f3231da369421424f5cccd4aada23729c8"
    assert w._actor().artifact_sha256 == "96762a46f59abd24a10b1abedf8dc325d72c85f3af39424c33e2dcba4ef5ffd3"
    for path, digest in w.CONFIG["source_sha256"].items():
        assert sha256((w.ROOT / path).read_bytes()).hexdigest() == digest
    assert w.CONFIG["historical_deployment_timestamp_verified"] is False


@pytest.mark.parametrize("task", (1, 2, 3))
def test_original_map_initial_state_task_seeds_and_all_step_physics_match(task):
    _, history = historic_from_git()
    state = w.initial_state(100, task)
    original = old_initial(task)
    assert w.WIDTH == 7 and w.HEIGHT == 6
    assert w._restore(state).state == original.state
    # Enrollment IDs do not secretly vary the exact requested legacy scenarios.
    assert w.public_state(w.initial_state(987654, task)) == w.public_state(state)
    while not state["terminal"]:
        decision = w.decide(state)
        # Include both informed commands and periodic deliberate waits; real
        # unchanged physics handles every transition, task replacement and score.
        human = "wait" if state["turn"] % 19 == 0 else w.human_advisor(state)
        chosen, runtime, _ = old_step(original, history, state["scenario_seed"], human)
        assert decision["action"] == chosen.lower()
        assert decision["controller_trace"]["selected_ai_action"] == w._plain(runtime["selected_ai_action"])
        state = w.step(state, human, decision)
        restored = w._restore(json.loads(json.dumps(state)))
        assert asdict(restored.state) == asdict(original.state)
        assert restored.get_rng_state() == original.get_rng_state()
        assert restored._episode_counter == original._episode_counter
        assert w.score(state)["task_score"] == original.state.user_score
    assert state["turn"] == 120 or state["shutdown_events"] > 0
    assert sum(w.score(state)["breakdown"].values()) == w.score(state)["task_score"]


def test_snapshot_preserves_tuples_active_handoff_memory_and_future_jobs():
    state = w.initial_state(100, 2)
    saw_plan = saw_replacement = False
    original_orders = {o["id"] for o in state["orders"]}
    for _ in range(120):
        env = w._restore(state)
        saw_plan |= env.state.active_coordination_plan is not None
        saw_replacement |= bool({o["id"] for o in state["orders"]} - original_orders)
        restored = json.loads(json.dumps(state))
        assert w._snapshot(w._restore(restored)) == state["snapshot"]
        assert w.decide(restored) == w.decide(state)
        if state["terminal"]: break
        action = w.human_advisor(state)
        assert w.step(state, action) == w.step(restored, action)
        state = w.step(state, action)
    assert saw_plan and saw_replacement


def test_read_operations_and_parallel_decisions_are_pure_and_group_blind():
    state = w.initial_state(11, 2)
    before = deepcopy(state)
    expected = w.decide(state)
    w.facts(state, expected); w.human_advisor(state); w.public_state(state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(x == expected for x in pool.map(w.decide, [state] * 12))
    assert state == before
    changed = deepcopy(state)
    changed.update(group="B", future_schedule=["unrelated"], unsubmitted_human_action="left")
    assert w.decide(changed) == expected
    assert w.public_state(changed) == w.public_state(state)


def test_recognized_wall_command_is_preserved_and_costs_no_movement_battery():
    state = w.initial_state(100, 1)
    assert set(w.legal_actions(state)) == {"up", "down", "left", "right", "wait"}
    after = w.step(state, "down")  # human starts on the bottom row
    assert after["last_actions"]["human"] == "down"
    assert after["human"]["battery"] == 100
    assert w._pos(after["human"]) == (2, 5)
    assert any(e["type"] == "blocked" for e in after["events"])
    assert w.score(after)["breakdown"]["time"] == -1


def test_shared_charger_penalty_uses_latest_physics_and_new_score_not_old_controller_score():
    # Direct physical fixture: both stationary on distinct cells, occupant
    # >60 after WAIT, partner <20 two map steps away, safe departure exists.
    env = old_initial(1)
    s = env.get_state()
    s.by_id("robot_1").position = env.layout.charger_position
    s.by_id("robot_1").battery = 65
    s.by_id("robot_2").position = (4, 2)
    s.by_id("robot_2").battery = 18
    env.set_state(s)
    before = env.get_state()
    info = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})[-1]
    state = w._state(env, 1, 100, w._events(before, env.state, info))
    assert w.score(state)["breakdown"]["shared_charger_occupancy"] == -5
    assert w.score(state)["task_score"] == env.state.user_score
    assert w.score(state)["score_max"] is None
    assert w.score(state)["task_score"] < 0
    fact = next(f for f in w.facts(state) if f["id"] == "score_shared_charger_occupancy")
    assert "-5" in fact["en"]
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert env.state.score_breakdown["shared_charger_occupancy"] == -5


def test_low_battery_shutdown_debits_entire_remaining_step_budget():
    env = old_initial(1)
    s = env.get_state()
    s.by_id("robot_1").battery = 2
    env.set_state(s)
    state = w._state(env, 1, 100)
    state = w.step(state, "up")
    assert state["terminal"]
    assert w.score(state)["breakdown"]["time"] == -120
    assert w.score(state)["breakdown"]["shutdown"] == -5
    assert w.legal_actions(state) == []
    with pytest.raises(ValueError, match="complete"):
        w.step(state, "wait")


def test_illegal_or_stale_decisions_are_rejected():
    state = w.initial_state(100, 2)
    with pytest.raises(ValueError, match="illegal"):
        w.step(state, "teleport")
    decision = w.decide(state)
    decision["reason_en"] += " fabricated"
    with pytest.raises(ValueError, match="decision"):
        w.step(state, "wait", decision)
    bad = deepcopy(state); bad["turn"] += 1
    with pytest.raises(ValueError, match="boundary"):
        w.decide(bad)


def test_public_projection_has_shared_jobs_and_no_private_decision_or_rng():
    state = w.initial_state(100, 2)
    state["human"]["private_note"] = "SECRET"
    public = w.public_state(state)
    assert not {"snapshot", "seed", "scenario_seed", "policy_memory", "last_actions", "controller_trace"} & set(public)
    assert "SECRET" not in json.dumps(public)
    assert all(o["owner"] == "shared" for o in public["orders"])
    assert public["chargers"] == [[3, 5]]
    assert len(public["orders"]) == 2
    assert json.loads(json.dumps(public)) == public


def test_demonstration_runs_real_transitions_and_six_neutral_captions():
    demo = w.demonstration()
    assert len(demo["captions"]) == 6
    assert [f["turn"] for f in demo["frames"]] == list(range(len(demo["frames"])))
    assert {"charge", "pickup", "delivery", "collision", "complete"} <= {e["type"] for f in demo["frames"] for e in f["events"]}
    assert demo["frames"][-1]["terminal"]
    assert demo["frames"][-1]["score"]["metrics"]["deliveries"] > 0
    assert not any("neural" in c["en"].lower() or "policy" in c["en"].lower() for c in demo["captions"])


def test_comprehension_is_bilingual_and_tracks_actual_physical_rules():
    assert [q["answer"] for q in w.comprehension()] == [0, 1, 2]
    assert len(w.comprehension("zh")) == 3
    env = old_initial(1)
    state = env.get_state(); state.by_id("robot_1").position = env.layout.charger_position; state.by_id("robot_1").battery = 40; env.set_state(state)
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert env.state.by_id("robot_1").battery == 50
    assert "no fixed 100" in " ".join(w.rules())


def test_question_bank_is_reproducible_bilingual_and_not_a_model_understanding_claim():
    from domains.warehouse.build_qa_cases import build
    recorded = json.loads((w.ROOT / "domains/warehouse/qa_cases.json").read_text())
    assert recorded == build()
    assert len(recorded) >= 60
    assert len({c["question"] for c in recorded}) == len(recorded)
    assert {c["language"] for c in recorded} == {"en", "zh"}
    assert {c["expected_kind"] for c in recorded} == {"facts", "counterfactual", "clarification"}
    assert all(c["evaluation_kind"] == "injected_plan_composition_not_language_understanding" for c in recorded)
