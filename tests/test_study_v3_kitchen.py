"""Mechanics boundaries and independent fixed-AI feasibility checks.

Microstates below are explicitly test fixtures, never participant samples.
Full-run tests begin at frozen initial_state and control the human role only.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from domains.kitchen import engine as e


def item(recipe="tomato", stage="chopped", item_id="fixture-item"):
    return {"id": item_id, "ingredient": recipe, "stage": stage, "chop_progress": 0 if stage == "raw" else 2}


def fixture():
    state = e.initial_state(1000, 2)
    for order in state["orders"]:
        order["deadline"] = 190
    state["max_turns"] = 200
    return state


def action_decision(action):
    return {"action": action, "memory": {}, "reason_code": "test_fixture", "reason_en": "Test action", "reason_zh": "测试动作"}


def tick(state, human="wait", ai="wait"):
    return e.step(state, human, action_decision(ai))


def run(seed, task, waits=()):
    state = e.initial_state(seed, task)
    trajectory = [deepcopy(state)]
    decisions = []
    actions = []
    while not state["terminal"]:
        decisions.append(e.decide(state))
        human_action = "wait" if state["turn"] in waits else e.human_advisor(state)
        actions.append(human_action)
        state = e.step(state, human_action)
        trajectory.append(state)
    return trajectory, decisions, actions


class KitchenMechanics(unittest.TestCase):
    def test_movement_walls_stations_and_work_area(self):
        state = fixture()
        self.assertNotIn("interact_handoff", e.legal_actions(state))
        state["human"].update(x=3, y=3)
        self.assertNotIn("right", e.legal_actions(state))
        with self.assertRaises(ValueError):
            e.step(state, "right")
        state["human"].update(x=2, y=1)
        self.assertNotIn("left", e.legal_actions(state))
        self.assertNotIn("right", e.legal_actions(state))
        self.assertIn("take_tomato", e.legal_actions(state))

    def test_one_portion_two_chops_and_progress_survives_queries(self):
        state = fixture()
        state["human"].update(x=2, y=1)
        state = tick(state, "take_tomato")
        state = tick(state, "chop")
        self.assertEqual(state["human"]["holding"]["chop_progress"], 1)
        saved = deepcopy(state)
        e.facts(state)
        e.rules("zh")
        e.public_state(state)
        self.assertEqual(state, saved)
        state = tick(state, "chop")
        self.assertEqual(state["human"]["holding"]["stage"], "chopped")
        self.assertNotIn("chop", e.legal_actions(state))

    def test_cooking_exact_twelve_and_fourteen_full_turns(self):
        for recipe, duration in (("tomato", 12), ("onion", 14)):
            state = fixture()
            state["ai"].update(x=6, y=1, holding=item(recipe))
            state = tick(state, ai="interact_pot1")
            self.assertEqual(state["pots"][0]["remaining"], duration)
            for remaining in range(duration - 1, 0, -1):
                state = tick(state)
                self.assertEqual(state["pots"][0]["remaining"], remaining)
                self.assertEqual(state["pots"][0]["status"], "cooking")
            state = tick(state)
            self.assertEqual(state["pots"][0]["status"], "ready")
            self.assertEqual(state["pots"][0]["ready_age"], 0)
            self.assertEqual(state["turn"], duration + 1)

    def test_six_full_ready_turns_and_removal_at_boundary(self):
        state = fixture()
        state["ai"].update(x=6, y=1)
        state["pots"][0].update(status="ready", item=item(stage="cooked"), remaining=0, ready_age=0)
        for age in range(1, 6):
            state = tick(state)
            self.assertEqual(state["pots"][0]["status"], "ready")
            self.assertEqual(state["pots"][0]["ready_age"], age)
        removed = tick(state, ai="interact_pot1")
        self.assertEqual(removed["pots"][0]["status"], "empty")
        self.assertEqual(removed["metrics"]["burnt"], 0)
        burned = tick(state)
        self.assertEqual(burned["pots"][0]["status"], "burnt")
        self.assertEqual(burned["metrics"]["burnt"], 1)
        cleared = tick(burned, ai="interact_pot1")
        self.assertEqual(cleared["pots"][0]["status"], "empty")
        self.assertEqual(cleared["metrics"]["waste"], 1)
        self.assertEqual(cleared["metrics"]["burnt"], 1)

    def test_removed_soup_has_no_spoil_timer(self):
        state = fixture()
        state["human"]["holding"] = item(stage="cooked")
        for _ in range(20):
            state = tick(state)
        self.assertEqual(state["human"]["holding"]["stage"], "cooked")
        self.assertEqual(state["metrics"]["burnt"], 0)

    def test_simultaneous_handoff_placements_and_pickups_both_fail(self):
        for placement in (True, False):
            state = fixture()
            state["human"].update(x=3, y=3)
            state["ai"].update(x=5, y=3)
            if placement:
                state["human"]["holding"] = item(item_id="h")
                state["ai"]["holding"] = item(stage="cooked", item_id="a")
            else:
                state["handoff"] = item(item_id="counter")
            before = deepcopy(state)
            state = tick(state, "interact_handoff", "interact_handoff")
            self.assertEqual(state["human"], before["human"])
            self.assertEqual(state["ai"], before["ai"])
            self.assertEqual(state["handoff"], before["handoff"])
            self.assertEqual(state["turn"], before["turn"] + 1)
            self.assertEqual(state["metrics"]["handoff_conflicts"], 1)
            self.assertEqual([ev["type"] for ev in state["events"]], ["handoff_conflict"])

    def test_new_handoff_item_cannot_be_collected_same_turn(self):
        state = fixture()
        state["human"].update(x=3, y=3, holding=item())
        state["ai"].update(x=5, y=3)
        self.assertNotIn("interact_handoff", e.legal_actions(state, "ai"))
        with self.assertRaises(ValueError):
            tick(state, "interact_handoff", "interact_handoff")
        state = e.step(state, "interact_handoff")
        self.assertIsNotNone(state["handoff"])
        self.assertIsNone(state["ai"]["holding"])
        state = e.step(state, "wait")
        self.assertIsNone(state["handoff"])
        self.assertEqual(state["ai"]["holding"]["stage"], "chopped")

    def test_buffer_capacity_and_human_can_recover_own_item(self):
        state = fixture()
        state["human"].update(x=3, y=3, holding=item())
        state = tick(state, "interact_handoff")
        state = tick(state, "interact_handoff")
        self.assertIsNone(state["handoff"])
        state = tick(state, "left")
        state = tick(state, "interact_buffer")
        self.assertIsNone(state["human"]["holding"])
        self.assertIsNotNone(state["buffers"]["human"])
        state["human"]["holding"] = item(item_id="second")
        self.assertNotIn("interact_buffer", e.legal_actions(state))
        state = tick(state, "discard")
        state = tick(state, "interact_buffer")
        self.assertEqual(state["human"]["holding"]["id"], "fixture-item")

    def test_discard_every_held_stage_at_any_legal_position(self):
        for actor in ("human", "ai"):
            for stage in ("raw", "chopped", "cooked", "plated"):
                state = fixture()
                state[actor]["holding"] = item(stage=stage)
                self.assertIn("discard", e.legal_actions(state, actor))
                state = tick(state, "discard" if actor == "human" else "wait", "discard" if actor == "ai" else "wait")
                self.assertIsNone(state[actor]["holding"])
                self.assertEqual(state["metrics"]["waste"], 1)
                self.assertEqual(e.score(state)["task_score"], 0)

    def test_full_hands_and_counters_have_a_public_recovery_path(self):
        state = fixture()
        state["human"].update(x=3, y=3, holding=item(stage="plated", item_id="held_h"))
        state["ai"].update(x=5, y=3, holding=item(stage="cooked", item_id="held_a"))
        state["handoff"] = item(item_id="handoff")
        state["buffers"] = {"human": item(item_id="buffer_h"), "ai": item(item_id="buffer_a")}
        state = e.step(state, "discard")
        state = e.step(state, "interact_handoff")
        self.assertIsNone(state["handoff"])
        # AI's already chosen wait cannot be replaced by a magically same-turn
        # delivery. It can hand over on the following legal turn.
        state = e.step(state, "discard")
        self.assertIsNotNone(state["handoff"])
        self.assertEqual(state["handoff"]["stage"], "cooked")
        self.assertIsNone(state["ai"]["holding"])

    def test_serve_on_deadline_then_expire_other_order(self):
        state = fixture()
        state["turn"] = 4
        state["orders"][0].update(ingredient="tomato", deadline=5)
        state["orders"][1].update(ingredient="tomato", deadline=5)
        state["human"].update(x=2, y=5, holding=item(stage="plated"))
        state = tick(state, "serve")
        self.assertEqual(state["orders"][0]["status"], "completed")
        self.assertEqual(state["orders"][0]["served_turn"], 5)
        self.assertEqual(state["orders"][1]["status"], "expired")
        self.assertEqual(e.score(state)["raw_score"], 1)

    def test_wrong_or_expired_order_cannot_score_twice(self):
        state = fixture()
        for order in state["orders"]:
            order["ingredient"] = "onion"
        state["human"].update(x=2, y=5, holding=item(stage="plated"))
        state = tick(state, "serve")
        self.assertIsNotNone(state["human"]["holding"])
        self.assertEqual(state["metrics"]["completed_orders"], 0)
        self.assertEqual(state["events"][0]["type"], "serve_rejected")
        state["orders"][0]["ingredient"] = "tomato"
        state = tick(state, "serve")
        self.assertEqual(state["metrics"]["completed_orders"], 1)
        state["human"]["holding"] = item(stage="plated", item_id="extra")
        state = tick(state, "serve")
        self.assertEqual(state["metrics"]["completed_orders"], 1)

    def test_budget_end_and_terminal_rejection(self):
        state = fixture()
        state["max_turns"] = 1
        state = e.step(state, "wait")
        self.assertTrue(state["terminal"])
        self.assertEqual(state["termination_reason"], "turn_budget")
        self.assertEqual(e.legal_actions(state), [])
        with self.assertRaises(ValueError):
            e.step(state, "wait")


class KitchenControllerAndEvidence(unittest.TestCase):
    def test_group_absent_from_api_and_ai_cannot_operate_human_stations(self):
        import inspect
        self.assertNotIn("group", inspect.signature(e.decide).parameters)
        state = fixture()
        self.assertTrue(all(a not in e.legal_actions(state, "ai") for a in ("take_tomato", "chop", "plate", "serve")))
        while not state["terminal"]:
            state = e.step(state, "wait")
        self.assertEqual(state["metrics"]["completed_orders"], 0)
        self.assertEqual(state["next_item_id"], 1)

    def test_decision_and_facts_ignore_hidden_future_and_do_not_mutate(self):
        state = e.initial_state(1000, 3)
        changed = deepcopy(state)
        changed["_future_orders"] = [{"id": "SECRET_future", "ingredient": "onion", "arrival": 123, "deadline": 500}]
        saved = deepcopy(state)
        self.assertEqual(e.decide(state), e.decide(changed))
        self.assertEqual(e.facts(state), e.facts(changed))
        self.assertEqual(e.human_advisor(state), e.human_advisor(changed))
        self.assertEqual(e.public_state(state), e.public_state(changed))
        self.assertEqual(state, saved)

    def test_public_projection_omits_decision_and_future(self):
        state = e.initial_state(1000, 3)
        state["policy_memory"] = {"hidden_marker": "SECRET_POLICY"}
        view = json.dumps(e.public_state(state))
        for forbidden in ("SECRET_POLICY", "policy_memory", "reason_code", "_future_orders", '"seed"', "emergency_rescues"):
            self.assertNotIn(forbidden, view)
        self.assertEqual(len(e.public_state(state)["orders"]), 2)

    def test_known_order_only_appears_after_arrival(self):
        state = e.initial_state(1000, 3)
        upcoming = deepcopy(state["_future_orders"][0])
        for _ in range(upcoming["arrival"] - 1):
            state = e.step(state, "wait")
        self.assertNotIn(upcoming["id"], [o["id"] for o in e.public_state(state)["orders"]])
        state = e.step(state, "wait")
        self.assertIn(upcoming["id"], [o["id"] for o in e.public_state(state)["orders"]])

    def test_emergency_free_hand_then_rescue_is_persistent_and_real(self):
        trajectory, decisions, _ = run(2000, 2, waits=(9, 10, 11, 12))
        triggers = [(state, d) for state, d in zip(trajectory, decisions) if d.get("emergency")]
        self.assertTrue(triggers)
        before, decision = triggers[0]
        self.assertEqual(before["turn"], 24)
        self.assertEqual(decision["emergency"]["normal_plan_turns"], 9)
        self.assertEqual(decision["emergency"]["burn_in"], 8)
        self.assertEqual(decision["emergency"]["rescue_turns"], 5)
        self.assertEqual(decision["action"], "interact_handoff")
        self.assertEqual(decisions[25]["action"], "up")
        self.assertTrue(any(ev["type"] == "pot_removed" and ev["pot"] == "pot1" for ev in trajectory[29]["events"]))
        self.assertEqual(trajectory[-1]["metrics"]["burnt"], 0)
        self.assertGreaterEqual(trajectory[-1]["metrics"]["emergency_rescues"], 1)

    def test_buffered_cooked_soup_releases_after_counter_clears(self):
        state = fixture()
        state["ai"].update(x=6, y=3)
        state["buffers"]["ai"] = item(stage="cooked")
        state["handoff"] = item(item_id="block")
        # An empty pot and prepared item is useful alternative work: AI accepts
        # it rather than pretending that the blocked soup prevents all work.
        self.assertNotEqual(e.decide(state)["action"], "wait")
        state["handoff"] = None
        self.assertEqual(e.decide(state)["action"], "interact_buffer")
        state = e.step(state, "wait")
        state = e.step(state, "wait")
        state = e.step(state, "wait")
        self.assertEqual(state["handoff"]["stage"], "cooked")

    def test_unstarted_ingredients_prioritize_earliest_due_order(self):
        state = fixture()
        state["orders"][0].update(ingredient="tomato", deadline=80)
        state["orders"][1].update(ingredient="onion", deadline=100)
        state["handoff"] = item("tomato")
        state["buffers"]["ai"] = item("onion", item_id="later")
        self.assertEqual(e.decide(state)["goal"], "handoff")

    def test_load_commitment_does_not_change_for_new_different_order(self):
        state = fixture()
        state["ai"]["holding"] = item("onion")
        state["policy_memory"] = {"load_pot": "pot2"}
        before = e.decide(state)
        state["orders"].append({"id": "new_visible", "ingredient": "tomato", "arrival": 0, "deadline": 30, "status": "pending", "served_turn": None})
        after = e.decide(state)
        self.assertEqual(before["action"], after["action"])
        self.assertEqual(after["goal"], "pot2")

    def test_rules_and_comprehension_claims_match_engine(self):
        self.assertEqual(len(e.rules()), len(e.rules("zh")))
        self.assertEqual(len(e.comprehension()), 3)
        state = fixture()
        state["ai"].update(x=6, y=5)
        state["pots"][1].update(status="ready", item=item(stage="cooked"), remaining=0, ready_age=5)
        state["handoff"] = item("onion")
        self.assertEqual(e.decide(state)["action"], "interact_pot2")
        state = fixture()
        state["ai"]["holding"] = item("onion", "cooked")
        state["handoff"] = item("tomato")
        self.assertEqual(e.decide(state)["goal"], "ai_buffer")
        state["buffers"]["ai"] = item()
        self.assertEqual(e.decide(state)["action"], "wait")
        state["handoff"] = None
        self.assertEqual(e.decide(state)["goal"], "handoff")

    def test_all_facts_are_bilingual_and_no_implementation_jargon(self):
        import re
        trajectory, _, _ = run(2000, 2, waits=(9, 10, 11, 12))
        for state in trajectory[::5]:
            facts = e.facts(state)
            self.assertEqual(len({f["id"] for f in facts}), len(facts))
            for fact in facts:
                self.assertTrue(fact["en"] and fact["zh"])
                self.assertIsNone(re.search(r"\b(?:NN|PPO|logit|embedding|checkpoint|Q-value|policy head|reward shaping)\b", fact["en"], re.I))


class KitchenFeasibility(unittest.TestCase):
    def test_48_frozen_seeds_use_only_legal_human_actions_and_fixed_ai(self):
        config = e._configuration()
        dev, hold = config["development_seeds"], config["held_out_seeds"]
        self.assertEqual(len(dev), 24)
        self.assertEqual(len(hold), 24)
        self.assertFalse(set(dev) & set(hold))
        for task in (1, 2, 3):
            for seed in dev + hold:
                with self.subTest(task=task, seed=seed):
                    trajectory, decisions, actions = run(seed, task)
                    final = trajectory[-1]
                    self.assertGreaterEqual(final["metrics"]["completed_orders"], 4 if task == 1 else 5)
                    self.assertEqual(final["metrics"]["burnt"], 0)
                    if task == 2:
                        self.assertEqual(final["metrics"]["completed_orders"], 6)
                        self.assertGreater(final["metrics"]["parallel_cooking_turns"], 0)
                        # Item IDs prove both simultaneously cooking items
                        # eventually appear in actual served events.
                        overlap = next(s for s in trajectory if all(p["status"] == "cooking" for p in s["pots"]))
                        in_both = {p["item"]["id"] for p in overlap["pots"]}
                        served = {ev["item"]["id"] for s in trajectory for ev in s["events"] if ev["type"] == "served"}
                        self.assertTrue(in_both <= served)
                    for before, after, decision, action in zip(trajectory, trajectory[1:], decisions, actions):
                        self.assertIn(action, e.legal_actions(before))
                        self.assertEqual(after, e.step(before, action, decision))
                        if action in e.MOVES:
                            dx, dy = e.MOVES[action]
                            self.assertEqual((after["human"]["x"], after["human"]["y"]), (before["human"]["x"] + dx, before["human"]["y"] + dy))

    def test_demo_is_real_has_six_captions_success_and_conflict(self):
        demo = e.demonstration()
        self.assertEqual(len(demo["captions"]), 6)
        self.assertEqual([f["turn"] for f in demo["frames"]], list(range(len(demo["frames"]))))
        types = {ev["type"] for frame in demo["frames"] for ev in frame["events"]}
        self.assertTrue({"served", "handoff_conflict", "chop", "pot_ready"} <= types)
        self.assertGreater(demo["frames"][-1]["score"]["raw_score"], 0)
        for caption in demo["captions"]:
            self.assertIn(caption["index"], range(len(demo["frames"])))
            self.assertTrue(caption["en"] and caption["zh"])


if __name__ == "__main__":
    unittest.main()
