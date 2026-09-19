"""Rebuild frozen Warehouse composition fixtures; no provider is called.

The questions and expected numbers are hand-specified below. State snapshots
come from real preserved-engine execution or explicit validated charger fixtures.
Injected plans verify evidence composition, not free-language understanding.
Run: python -m domains.warehouse.build_qa_cases
"""
from copy import deepcopy
import json
from pathlib import Path

from . import turnbased as w


def fixture(mode):
    state = w.initial_state(123, 2)
    env = w._restore(state)
    raw = env.get_state()
    ai, human = raw.by_id("robot_2"), raw.by_id("robot_1")
    ai.position = env.layout.charger_position; ai.battery = 40; ai.charge_mode_active = True
    human.position = (1, 0) if mode == "charge" else (5, 4)
    human.battery = 100 if mode == "charge" else 20
    env.set_state(raw)
    return w._state(env, 2, 123)


def build():
    initial = w.initial_state(123, 2)
    charge, handoff = fixture("charge"), fixture("handoff")
    after_wait, after_up = w.step(initial, "wait"), w.step(initial, "up")
    after_handoff = w.step(handoff, "wait")
    assert w.decide(charge)["action"] == "wait"
    assert w.decide(handoff)["action"] == "left"
    assert after_handoff["ai"]["x"] == 2 and after_handoff["ai"]["y"] == 5
    assert w.score(after_wait)["raw_score"] == -3
    cases = []

    def pair(key, state, questions, ids=(), claims=(), *, purpose="observation", subject="shared",
             premise="supported", clarification=None, actions=None, history=(), previous=()):
        for lang, question in zip(("en", "zh"), questions):
            selected_ids = list(ids)
            selected_claims = deepcopy(list(claims))
            for claim in selected_claims:
                if "contains_pair" in claim:
                    claim["contains"] = claim.pop("contains_pair")[lang == "zh"]
            intents = [] if clarification else [{"kind": "counterfactual" if actions else "facts", "subject": subject,
                "purpose": purpose, "evidence_ids": selected_ids}]
            if actions:
                intents[0].update(actions=actions, horizon=len(actions))
            plan = {"language": lang, "binding": {"task": state["task"], "turn": state["turn"]},
                    "premise": premise, "clarification": clarification, "intents": intents}
            cases.append({"case_id": "warehouse_restored_"+key+"_"+lang, "domain": "warehouse",
                "question": question, "language": lang, "category": key,
                "evaluation_kind": "injected_plan_composition_not_language_understanding",
                "state": deepcopy(state), "decision": w.decide(state), "expected_fact_ids": selected_ids,
                "expected_claims": selected_claims, "expected_plan": plan,
                "expected_kind": "clarification" if clarification else "counterfactual" if actions else "facts",
                "public_history": deepcopy(list(history)), "previous_dialogue": deepcopy(list(previous))})

    pair("next_action", initial, ("Which direction will you move next?", "你下一步往哪个方向移动？"),
         ["system:ai_action"], [{"decision_path":"action","equals":"up"}], purpose="action", subject="ai")
    pair("initial_reason", initial, ("Why did you choose that next move?", "为什么选择这个下一步动作？"),
         ["system:ai_reason"], [{"decision_path":"reason_code","equals":"reduce_possible_collision"}], purpose="reason", subject="ai")
    pair("human_location", initial, ("Where am I starting?", "我从哪里开始？"), ["human_state"], [{"path":"human.x","equals":2},{"path":"human.y","equals":5}], subject="human")
    pair("ai_location", initial, ("Where are you now?", "你现在在哪里？"), ["ai_state"], [{"path":"ai.x","equals":4},{"path":"ai.y","equals":5}], subject="ai")
    pair("human_energy", initial, ("Do I have enough battery for one move to the charger?", "我的电量够走一步到充电站吗？"), ["human_state","human_charger_distance"], [{"path":"human.battery","equals":100},{"fact_id":"human_charger_distance","contains_pair":["1 moves, costing 2", "1步，耗电2"]}], subject="human")
    pair("ai_energy", initial, ("How much charge do you currently have?", "你目前有多少电量？"), ["ai_state"], [{"path":"ai.battery","equals":100}], subject="ai")
    pair("budget", initial, ("How many game steps are left?", "还剩多少游戏步数？"), ["time"], [{"path":"max_turns","equals":120},{"path":"turn","equals":0}])
    pair("initial_score", initial, ("What is our score before we move?", "移动之前我们的分数是多少？"), ["score"], [{"fact_id":"score","contains_pair":["score is 0", "原始分为0"]}])
    pair("negative_score", after_wait, ("Is a negative score possible here?", "这里的分数可以为负吗？"), ["score","public_rule_4"], [{"fact_id":"score","contains_pair":["score is -3", "原始分为-3"]}], purpose="rule")
    pair("waiting_cost", after_wait, ("Why did one wait cost us three points?", "为什么等待一步扣了我们三分？"), ["score_time","score_human_detour","public_rule_6"], [{"fact_id":"score_time","contains_pair":["-1", "-1"]},{"fact_id":"score_human_detour","contains_pair":["-2", "-2"]}], purpose="rule")
    pair("raw_scale", initial, ("Is this percentage progress out of one hundred?", "这里的得分是百分制进度吗？"), ["public_rule_4"], [{"fact_id":"public_rule_4","contains_pair":["no fixed 100-point maximum", "不是满分100分"]}], premise="contradicted", purpose="rule")
    pair("ownership", initial, ("Which parcels belong only to me?", "哪些包裹只属于我？"), ["public_rule_1"], [{"path":"orders.0.owner","equals":"shared"},{"path":"orders.1.owner","equals":"shared"}], premise="contradicted", purpose="rule")
    pair("first_order", initial, ("Where are the pickup and destination for task one?", "第一个订单的取货点和送货点在哪里？"), ["order_task_1"], [{"path":"orders.0.id","equals":"task_1"}])
    pair("second_order", initial, ("Tell me about the other shared job.", "说说另一个共享订单。"), ["order_task_2"], [{"path":"orders.1.id","equals":"task_2"}])
    pair("replenishment", initial, ("Do we stop receiving jobs after two deliveries?", "完成两次配送后就不再出现订单了吗？"), ["public_rule_1"], [{"fact_id":"public_rule_1","contains_pair":["creates another shared job", "补充新订单"]}], premise="contradicted", purpose="rule")
    pair("controls", initial, ("What does the space bar do in Warehouse?", "Warehouse里空格键做什么？"), ["public_rule_2"], [{"fact_id":"public_rule_2","contains_pair":["Space waits", "空格等待"]}], purpose="rule")
    pair("collision_rule", initial, ("What happens if we try to swap our cells?", "如果我们试图交换位置会发生什么？"), ["public_rule_2","public_rule_4"], [{"fact_id":"public_rule_4","contains_pair":["−10 per collision", "每碰撞−10"]}], purpose="rule")
    pair("move_cost_false", after_up, ("Why did you spend three battery on that move?", "你刚才为什么移动耗电三点？"), ["public_rule_3","ai_state"], [{"path":"ai.battery","equals":98}], premise="contradicted", purpose="rule")
    pair("shutdown_rule", initial, ("Does running out of battery end the round?", "电量耗尽会结束这一局吗？"), ["public_rule_3","public_rule_4"], [{"fact_id":"public_rule_4","contains_pair":["unused steps", "剩余步数"]}], purpose="rule")
    pair("charge_cap", charge, ("Can the charger take battery above one hundred?", "充电可以超过一百点电量吗？"), ["public_rule_3"], [{"fact_id":"public_rule_3","contains_pair":["capped at 100", "上限100"]}], purpose="rule")
    pair("charger_penalty", initial, ("When do we lose five points for holding the charger?", "什么情况下占用充电站扣五分？"), ["public_rule_5"], [{"fact_id":"public_rule_5","contains_pair":["above 60", "超过60"]},{"fact_id":"public_rule_5","contains_pair":["below 20", "低于20"]}], purpose="rule")
    pair("charger_repeat", initial, ("Will the charger penalty repeat on every wait?", "占桩扣分会在每一步等待时重复吗？"), ["public_rule_5"], [{"fact_id":"public_rule_5","contains_pair":["once per continuous stay", "每次连续占用"]}], premise="contradicted", purpose="rule")
    pair("charging_reason", charge, ("Why are you waiting on the charger now?", "你为什么现在在充电格等待？"), ["system:ai_reason","ai_charge_shortfall"], [{"decision_path":"action","equals":"wait"},{"path":"ai.battery","equals":40}], purpose="reason", subject="ai")
    pair("charging_shortfall", charge, ("Exactly how much more battery do you need before the current threshold?", "离当前出发电量门槛还差多少点？"), ["ai_charge_shortfall"], [{"fact_id":"ai_charge_shortfall","contains_pair":["16 more", "还差16"]}], subject="ai")
    pair("charging_duration", charge, ("How many charging steps would that missing battery take?", "补足这些电量需要几步充电？"), ["ai_charge_shortfall"], [{"fact_id":"ai_charge_shortfall","contains_pair":["2 charging steps", "2个充电步"]}], subject="ai")
    pair("charging_calculation", charge, ("Where does your departure battery threshold come from?", "你的离开充电站电量门槛如何计算？"), ["ai_charge_shortfall","ai_charge_calculation"], [{"fact_id":"ai_charge_shortfall","contains_pair":["threshold is 56", "门槛为56"]},{"fact_id":"ai_charge_calculation","contains_pair":["6 reserve moves", "6步安全余量"]}], subject="ai")
    pair("handoff_action", handoff, ("Which way will you leave the charger?", "你会朝哪个方向离开充电站？"), ["system:ai_action"], [{"decision_path":"action","equals":"left"}], purpose="action", subject="ai")
    pair("handoff_reason", handoff, ("Why make room instead of remaining on the charger?", "为什么要让开而不是继续占着充电站？"), ["system:ai_reason","alternative_wait"], [{"decision_path":"controller_trace.ai_is_planned_clearer","equals":True}], purpose="reason", subject="ai")
    pair("handoff_alternative", handoff, ("What if you waited there instead of moving left?", "如果你不左移而是继续在那里等待，会怎样？"), ["alternative_wait"], [{"fact_id":"alternative_wait","contains_pair":["1 of your possible moves could conflict", "有1种可能发生冲突"]}], purpose="comparison", subject="ai")
    pair("handoff_followup", handoff, ("For that handoff, which cell will you clear?", "这次交接你会腾出哪一个格子？"), ["ai_state","system:ai_action"], [{"path":"ai.x","equals":3},{"path":"ai.y","equals":5},{"decision_path":"action","equals":"left"}], previous=[{"question":"What will you do next?","answer":"I will move left from the shared charger to make space."}], subject="ai")
    for key, question, actions, score, x, y, battery in [
        ("simulate_wait", ("If I wait one step now, what changes?", "如果我现在等待一步，会怎样？"), ["wait"], -3, 2, 5, 100),
        ("simulate_up", ("If I go up now, where do I end up and what does it cost?", "如果我现在向上走，最后在哪、得分如何变化？"), ["up"], -1, 2, 4, 98),
        ("simulate_wall", ("If I press down at this bottom edge, do I lose battery?", "我在底部边缘按下，会耗电吗？"), ["down"], -1, 2, 5, 100),
        ("simulate_two_waits", ("What happens if I wait for two steps?", "如果我连续等两步会怎样？"), ["wait","wait"], -6, 2, 5, 100),
        ("simulate_charger_entry", ("What if I move right onto the charger rather than up?", "如果我不是向上而是右移到充电站，会怎样？"), ["right"], -5, 3, 5, 98),
    ]:
        pair(key, initial, question, claims=[{"simulation_path":"raw_score_delta","equals":score},
            {"simulation_path":"task_score_delta","equals":score},{"simulation_path":"human.x","equals":x},
            {"simulation_path":"human.y","equals":y},{"simulation_path":"human.battery","equals":battery}],
            actions=actions, purpose="comparison", subject="human")
    pair("past_human_position", after_up, ("Where was I at step zero?", "第零步我在哪里？"), ["history:task2:turn0:human_position"],
         [{"fact_id":"history:task2:turn0:human_position","contains_pair":["column 2, row 5", "第2列、第5行"]}], history=[w.public_state(initial)], subject="human")
    pair("past_ai_position", after_up, ("Where were you before this move?", "这次移动之前你在哪里？"), ["history:task2:turn0:ai_position"],
         [{"fact_id":"history:task2:turn0:ai_position","contains_pair":["column 4, row 5", "第4列、第5行"]}], history=[w.public_state(initial)], subject="ai")
    pair("ambiguous_object", initial, ("What about that one?", "那一个呢？"), clarification="ambiguous_object")
    pair("missing_past_state", after_up, ("Why did you move that way in the previous task?", "你在上一个任务为什么这样走？"), clarification="select_frame")
    pair("unsupported_topic", initial, ("What will the weather be tomorrow?", "明天的天气怎么样？"), clarification="unsupported_question")
    pair("questions_free", charge, ("Will asking this question charge your battery automatically?", "问这个问题会让你自动充电吗？"), ["ai_charge_shortfall","public_rule_0"], [{"path":"turn","equals":0},{"path":"ai.battery","equals":40}], premise="contradicted")
    assert len(cases) >= 60 and len({c["question"] for c in cases}) == len(cases)
    return cases


if __name__ == "__main__":
    target = Path(__file__).with_name("qa_cases.json")
    cases = build()
    target.write_text("[\n" + ",\n".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases) + "\n]\n")
    print(f"Wrote {len(cases)} composition fixtures; no real-provider or human result is claimed.")
