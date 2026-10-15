"""Build small, current-version Kitchen QA fixtures from frozen NN execution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from . import engine_v5_frozen as physics
from . import neural_domain as kitchen


def build():
    state = kitchen.initial_state(1000, 2)
    selected = {"initial": deepcopy(state)}
    while not state["terminal"] and len(selected) < 4:
        decision = kitchen.decide(state)
        if decision["action_id"].startswith("interact_order") and "order_load" not in selected:
            selected["order_load"] = deepcopy(state)
        if any(pot["status"] == "ready" for pot in state["pots"]) and "ready_pan" not in selected:
            selected["ready_pan"] = deepcopy(state)
        if state["handoff"] and state["handoff"]["stage"] == "finished" and "handoff" not in selected:
            selected["handoff"] = deepcopy(state)
        state = kitchen.step(state, physics.simulation_partner(state), decision)
    if set(selected) != {"initial", "order_load", "ready_pan", "handoff"}:
        raise AssertionError("Frozen NN trace did not reach the expected QA states")
    output = []

    def add(key, name, question_en, question_zh, evidence, claims=(), kind="facts"):
        for language, question in (("en", question_en), ("zh", question_zh)):
            snapshot = selected[key]
            case = {"case_id": f"kitchen_nn_v1_{name}_{language}", "domain": "kitchen",
                    "question": question, "language": language, "state": deepcopy(snapshot),
                    "expected_fact_ids": list(evidence), "expected_claims": list(claims),
                    "expected_kind": kind}
            if kind == "counterfactual":
                case["expected_plan"] = {"language": language,
                    "binding": {"task": snapshot["task"], "turn": snapshot["turn"]},
                    "premise": "supported", "clarification": None,
                    "intents": [{"kind": "counterfactual", "subject": "human", "purpose": "comparison",
                                 "evidence_ids": [], "actions": ["wait"], "horizon": 1}]}
            output.append(case)

    add("initial", "action", "What will you do next?", "你下一步会做什么？",
        ["system:ai_action"], [{"decision_path": "action_id", "equals": kitchen.decide(selected["initial"])["action_id"]}])
    add("initial", "reason", "Why did you choose that action?", "为什么选择这个动作？",
        ["system:ai_reason", "ai_policy_source"], [{"decision_path": "reason_code", "equals": "neural_selection"}])
    add("order_load", "order", "Which order will this pan load serve?", "这次下锅绑定哪份订单？",
        ["system:ai_action"], [{"decision_path": "order_id", "equals": kitchen.decide(selected["order_load"])["order_id"]}])
    ready = next(pot["id"] for pot in selected["ready_pan"]["pots"] if pot["status"] == "ready")
    add("ready_pan", "pan", "What is happening at the ready stove?", "已炒好的炉灶现在是什么状态？",
        [ready], [{"path": "pots." + str(0 if ready == "pot1" else 1) + ".status", "equals": "ready"}])
    add("handoff", "handoff", "What is on the handoff counter?", "交接台上有什么？",
        ["handoff"], [{"path": "handoff.stage", "equals": "finished"}])
    add("initial", "wait", "If I wait one turn, what happens?", "如果我等一回合，会怎样？",
        [], [{"simulation_path": "steps_completed", "equals": 1}], "counterfactual")
    return output


def main():
    target = Path(__file__).with_name("neural_qa_cases.json")
    cases = build()
    target.write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(cases)} neural Kitchen QA cases to {target}")


if __name__ == "__main__":
    main()
