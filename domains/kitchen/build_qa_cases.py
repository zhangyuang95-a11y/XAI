"""Build bilingual evidence fixtures from genuine fixed-controller trajectories.

Expected plans are test oracles for the semantic stage, not model-generated
answers. These tests validate ground truth, never claim open-ended accuracy.
"""
from copy import deepcopy
import json
from pathlib import Path
from . import engine as e
from .validation import simulate


def build():
    regular = simulate(1000, 2, capture=True)
    emergency = simulate(2000, 2, forced_waits=(9, 10, 11, 12), capture=True)
    frames = [row["state"] for row in regular["replay"]] + [regular["final_state"]]
    urgent = [row["state"] for row in emergency["replay"]]
    def first(predicate):
        return next(state for state in frames if predicate(state))
    states = {
        "start": frames[0],
        "one_chop": first(lambda s: s["human"]["holding"] and s["human"]["holding"]["chop_progress"] == 1),
        "prepared": first(lambda s: s["human"]["holding"] and s["human"]["holding"]["stage"] == "chopped"),
        "handoff_raw": first(lambda s: s["handoff"] and s["handoff"]["stage"] == "chopped"),
        "ai_prepared": first(lambda s: s["ai"]["holding"] and s["ai"]["holding"]["stage"] == "chopped"),
        "tomato_load": first(lambda s: s["pots"][0]["status"] == "cooking" and s["pots"][0]["remaining"] == 12),
        "both_pots": first(lambda s: all(p["status"] == "cooking" for p in s["pots"])),
        "onion_load": first(lambda s: s["pots"][1]["status"] == "cooking" and s["pots"][1]["remaining"] == 14),
        "ready": first(lambda s: s["pots"][0]["status"] == "ready"),
        "ai_cooked": first(lambda s: s["ai"]["holding"] and s["ai"]["holding"]["stage"] == "cooked"),
        "handoff_cooked": first(lambda s: s["handoff"] and s["handoff"]["stage"] == "cooked"),
        "human_cooked": first(lambda s: s["human"]["holding"] and s["human"]["holding"]["stage"] == "cooked"),
        "plated": first(lambda s: s["human"]["holding"] and s["human"]["holding"]["stage"] == "plated"),
        "served": first(lambda s: any(ev["type"] == "served" for ev in s["events"])),
        "emergency": urgent[24], "rescue_continues": urgent[25],
        "after_rescue": urgent[29],
    }
    # Reproduce the demo's first real simultaneous-pickup failure.
    state = e.initial_state(1000, 1)
    while True:
        action = e.human_advisor(state)
        clash = (e.decide(state)["action"] == "interact_handoff" and state["handoff"]
                 and not state["human"]["holding"] and "interact_handoff" in e.legal_actions(state))
        state = e.step(state, "interact_handoff" if clash else action)
        if clash:
            states["conflict"] = state
            break
    specs = [
        ("holding_start", "start", "What am I carrying right now?", "我现在手里拿着什么？", ["human_holding"], [{"path": "human.holding", "equals": None}]),
        ("role", "start", "Can you finish an order while I just wait?", "如果我一直等着，你能独自完成订单吗？", ["public_rule0"], []),
        ("time", "start", "How many turns do we have left in this task?", "这个任务还剩多少回合？", ["current_turn"], [{"path": "max_turns", "equals": 160}]),
        ("controls", "start", "Which actions are available where I am standing?", "我现在站的位置可以做哪些动作？", ["human_legal_actions"], []),
        ("preparation_progress", "one_chop", "I chopped once. Is this ingredient ready for the stove?", "我已经切了一次，这份原料可以下锅了吗？", ["human_holding", "public_rule3"], [{"path": "human.holding.chop_progress", "equals": 1}, {"path": "human.holding.stage", "equals": "raw"}]),
        ("prepared_portion", "prepared", "Does this prepared portion contain everything needed for one soup?", "这一份切好的原料够做一份汤吗？", ["human_holding", "public_rule3"], [{"path": "human.holding.stage", "equals": "chopped"}]),
        ("handoff_contents", "handoff_raw", "What's occupying the handoff counter?", "交接台上是什么占着位置？", ["handoff"], [{"path": "handoff.stage", "equals": "chopped"}]),
        ("accept_next", "handoff_raw", "Are you going to take that prepared ingredient now?", "你下一步准备拿交接台上的备料吗？", ["ai_next_action", "ai_reason"], []),
        ("load_choice", "ai_prepared", "Where are you taking the ingredient in your hand?", "你手里的备料准备放到哪里？", ["ai_reason", "ai_holding"], [{"path": "ai.holding.stage", "equals": "chopped"}]),
        ("tomato_countdown", "tomato_load", "Exactly how long until the tomato soup is cooked?", "番茄汤还要几回合才煮好？", ["pot1"], [{"fact_id": "pot1", "contains": "12"}]),
        ("loading_turn", "tomato_load", "Did putting it into the pot already use one cooking turn?", "刚才下锅那一步算不算已经煮了一回合？", ["pot1", "public_rule3"], [{"fact_id": "pot1", "contains": "12"}]),
        ("two_pots", "both_pots", "Are both stoves cooking, or is one still empty?", "两个炉灶都在煮吗，还是有一个空着？", ["pot1", "pot2"], []),
        ("onion_duration", "onion_load", "Why does this onion soup show fourteen turns?", "为什么洋葱汤显示还要十四回合？", ["pot2", "public_rule3"], [{"fact_id": "pot2", "contains": "14"}]),
        ("ready_window", "ready", "It's just become ready. How many more turns before it burns?", "汤刚煮好，还能留在锅里几回合才烧坏？", ["pot1", "public_rule4"], [{"fact_id": "pot1", "contains": "6"}]),
        ("ready_not_done", "ready", "Is ready soup already worth an order point?", "汤煮好了就算完成订单、得分了吗？", ["public_rule6", "current_score"], [{"path": "metrics.completed_orders", "equals": 0}]),
        ("removed_food", "ai_cooked", "You're carrying the soup now. Can it still burn in your hand?", "你已经把汤拿出来了，拿在手里还会烧坏吗？", ["ai_holding", "public_rule4"], [{"path": "ai.holding.stage", "equals": "cooked"}]),
        ("handoff_pickup", "handoff_cooked", "What's on the counter, and what could I do next?", "交接台上是什么，我下一步可以做什么？", ["handoff", "human_available_option"], [{"path": "handoff.stage", "equals": "cooked"}]),
        ("plating_required", "human_cooked", "Can I serve this soup immediately without a plate?", "我手里的熟汤不装盘可以直接上菜吗？", ["human_holding", "public_rule6"], [{"path": "human.holding.stage", "equals": "cooked"}]),
        ("plated_serve", "plated", "The soup is plated. What action is available now?", "汤已经装盘了，现在可以做什么？", ["human_holding", "human_available_option"], [{"path": "human.holding.stage", "equals": "plated"}]),
        ("completed_score", "served", "How many orders did we actually complete?", "我们现在实际完成了几个订单？", ["current_score", "order1"], [{"path": "metrics.completed_orders", "equals": 1}]),
        ("deadline_inclusive", "start", "An order is due on turn 45. Is serving during turn 45 too late?", "订单截止第45回合，在第45回合上菜算晚了吗？", ["public_rule6", "order1"], []),
        ("duplicates", "start", "When two orders ask for the same soup, which one does serving complete?", "如果两个订单要同一种汤，上菜时会完成哪个订单？", ["public_rule6"], []),
        ("discard_recovery", "prepared", "Can I throw away what I'm holding if all counters are full? Is there an extra penalty?", "如果台面都满了，我可以丢掉手里的东西吗，会额外扣分吗？", ["public_rule2", "public_rule7"], []),
        ("conflict_fact", "conflict", "We both used the counter just now. Why didn't either of us get the item?", "刚才我们同时用了交接台，为什么谁都没拿到物品？", ["event0", "public_rule5"], [{"path": "metrics.handoff_conflicts", "equals": 1}]),
        ("conflict_repair", "conflict", "How does waiting let the other person use the counter after that failed transfer?", "刚才交接失败后，一方等待、另一方使用台面会怎样？", ["public_rule5"], []),
        ("pause_time", "tomato_load", "Will the soup burn while I type this question or switch language?", "我输入问题或切换语言时，汤会继续煮到烧坏吗？", ["public_rule1"], []),
        ("initial_wait", "start", "Why are you not cooking anything yet?", "你为什么现在还没有开始煮东西？", ["ai_reason", "handoff", "pot1", "pot2"], []),
        ("urgent_reason", "emergency", "Why put that ingredient down instead of carrying it to the other stove?", "为什么先把备料放下，而不是拿到另一口炉灶去？", ["ai_reason", "alternative2"], []),
        ("rescue_commitment", "rescue_continues", "You just put the ingredient down. What are you doing next, and why?", "你刚把备料放下，接下来做什么，为什么？", ["ai_reason", "ai_next_action"], []),
        ("distance", "start", "How far are you from stove 1, including the interaction to take food out?", "你距离炉灶1有多远，取出食物还要算一次交互吗？", ["pot1_distance"], [{"fact_id": "pot1_distance", "contains": "2"}]),
        ("false_premise", "ai_prepared", "Why are you holding cooked soup already?", "你为什么已经拿着熟汤了？", ["ai_holding"], [{"path": "ai.holding.stage", "equals": "chopped"}]),
        ("multiple_questions", "tomato_load", "What am I holding, and how many turns does stove 1 need?", "我手里是什么，炉灶1还要几回合？", ["human_holding", "pot1"], [{"fact_id": "pot1", "contains": "12"}]),
    ]
    cases = []
    for name, state_key, en, zh, ids, claims in specs:
        snapshot = states[state_key]
        available = {fact["id"] for fact in e.facts(snapshot)}
        assert set(ids) <= available, (name, set(ids) - available)
        for language, question in (("en", en), ("zh", zh)):
            plan = {"language": language, "binding": {"task": snapshot["task"], "turn": snapshot["turn"]},
                    "intents": [{"kind": "facts", "evidence_ids": ids}], "clarification": None,
                    "premise": "contradicted" if name == "false_premise" else "supported"}
            cases.append({"case_id": f"kitchen_{name}_{language}", "domain": "kitchen", "question": question,
                          "language": language, "state": deepcopy(snapshot), "expected_fact_ids": ids,
                          "expected_claims": deepcopy(claims), "expected_kind": "facts", "expected_plan": plan})
    for name, reason, en, zh in [
        ("which_one", "ambiguous_object", "What about that other thing?", "那另一个东西呢？"),
        ("history_binding", "select_frame", "Why did you change your plan earlier in Task 1?", "你为什么在 Task 1 之前那步改变了计划？"),
        ("outside_task", "unsupported_question", "What is the capital of Canada?", "加拿大的首都是哪里？"),
    ]:
        snapshot = states["tomato_load"]
        for language, question in (("en", en), ("zh", zh)):
            cases.append({"case_id": f"kitchen_{name}_{language}", "domain": "kitchen", "question": question,
                          "language": language, "state": deepcopy(snapshot), "expected_fact_ids": [],
                          "expected_claims": [], "expected_kind": "clarification",
                          "expected_plan": {"language": language, "binding": {"task": snapshot["task"], "turn": snapshot["turn"]},
                                            "intents": [], "clarification": reason, "premise": "unclear"}})
    for language, question in (("en", "If I wait for one turn now, will the team earn any points during that turn?"),
                               ("zh", "如果我现在等一回合，我们会在这一回合得分吗？")):
        snapshot = states["tomato_load"]
        cases.append({"case_id": f"kitchen_wait_counterfactual_{language}", "domain": "kitchen", "question": question,
                      "language": language, "state": deepcopy(snapshot), "expected_fact_ids": [],
                      "expected_claims": [{"simulation_path": "raw_score_delta", "equals": 0}], "expected_kind": "counterfactual",
                      "expected_plan": {"language": language, "binding": {"task": snapshot["task"], "turn": snapshot["turn"]},
                                        "intents": [{"kind": "counterfactual", "actions": ["wait"], "horizon": 1, "evidence_ids": []}],
                                        "clarification": None, "premise": "supported"}})
    # Short continuation retains explicit prior dialogue, rather than claiming
    # that the words alone identify an object.
    for language, question, previous in (("en", "And stove 2?", "How many turns does stove 1 need?"), ("zh", "那炉灶2呢？", "炉灶1还需要几回合？")):
        snapshot = states["both_pots"]
        cases.append({"case_id": f"kitchen_context_followup_{language}", "domain": "kitchen", "question": question,
                      "language": language, "state": deepcopy(snapshot), "previous_dialogue": [{"question": previous, "answer": "The selected stove's visible cooking timer was discussed."}],
                      "expected_fact_ids": ["pot2"], "expected_claims": [], "expected_kind": "facts",
                      "expected_plan": {"language": language, "binding": {"task": snapshot["task"], "turn": snapshot["turn"]},
                                        "intents": [{"kind": "facts", "evidence_ids": ["pot2"]}], "clarification": None, "premise": "supported"}})
    return cases


if __name__ == "__main__":
    cases = build()
    Path(__file__).with_name("qa_cases.json").write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(cases)} kitchen QA cases from actual fixed-controller snapshots.")
