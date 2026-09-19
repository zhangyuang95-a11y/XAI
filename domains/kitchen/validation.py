"""Developer-only proxy screen. No proxy is a human participant or an A/B group.

Run with python -m domains.kitchen.validation. Output stays inside this domain.
All proxies choose only a human legal action; engine.step owns the fixed AI.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
from statistics import mean

from . import engine as e


def serial_greedy(state):
    held = state["human"]["holding"]
    if held:
        if not e._pending(state, held["ingredient"]):
            return "discard"
        target, action = {"raw": ("chop", "chop"), "chopped": ("handoff", "interact_handoff"),
                          "cooked": ("plate", "plate"), "plated": ("serve", "serve")}[held["stage"]]
        if target == "handoff" and state["handoff"]:
            return e._approach(state, "human", "human_buffer", "interact_buffer") if state["buffers"]["human"] is None else "wait"
        return e._approach(state, "human", target, action)
    if state["handoff"] and state["handoff"]["stage"] in ("cooked", "plated"):
        return e._approach(state, "human", "handoff", "interact_handoff")
    if state["buffers"]["human"]:
        return e._approach(state, "human", "human_buffer", "interact_buffer")
    if any(e._all_items(state)):
        return e._approach(state, "human", "handoff", "wait")
    recipe = e._next_needed_ingredient(state)
    return e._approach(state, "human", "ingredients", "take_" + recipe) if recipe else "wait"


class PublicHistoryPartner:
    """Learns from completed public handoffs, never calls decide or facts.

    Starts with a serial recipe workflow. After observing the first AI handoff,
    it prepares the next ingredient while a pot cooks, and after observing
    cooked food it preserves the output lane. Timing comes from visible pots.
    This limited heuristic is not a model of all possible human learning.
    """
    def __init__(self):
        self.seen_output = False

    def action(self, state):
        self.seen_output |= any(event["type"] == "item_placed" and event.get("actor") == "ai"
                                and event.get("item", {}).get("stage") in ("cooked", "plated") for event in state["events"])
        if not self.seen_output:
            return serial_greedy(state)
        held, counter = state["human"]["holding"], state["handoff"]
        buffer = state["buffers"]["human"]
        outputs = any(it and it["stage"] in ("cooked", "plated") for it in (counter, state["ai"]["holding"], state["buffers"]["ai"]))
        pots = [p for p in state["pots"] if p["status"] in ("cooking", "ready")]
        soon = any(p["status"] == "ready" or p["remaining"] <= 4 for p in pots)
        if held:
            if held["stage"] != "chopped":
                return serial_greedy(state)
            if outputs or len(pots) == 2 and soon:
                return e._approach(state, "human", "human_buffer", "interact_buffer") if buffer is None else "discard"
            if counter is None:
                return e._approach(state, "human", "handoff", "interact_handoff")
            return e._approach(state, "human", "human_buffer", "interact_buffer") if buffer is None else "wait"
        if counter and counter["stage"] in ("cooked", "plated"):
            return e._approach(state, "human", "handoff", "interact_handoff")
        if outputs or any(p["status"] == "ready" for p in pots):
            if counter and state["ai"]["holding"] and state["ai"]["holding"]["stage"] in ("cooked", "plated"):
                return e._approach(state, "human", "handoff", "interact_handoff")
            return e._approach(state, "human", "handoff", "wait")
        if buffer and counter is None and not (len(pots) == 2 and soon):
            return e._approach(state, "human", "human_buffer", "interact_buffer")
        recipe = e._next_needed_ingredient(state)
        if recipe and buffer is None:
            return e._approach(state, "human", "ingredients", "take_" + recipe)
        return e._approach(state, "human", "handoff", "wait")


def simulate(seed, task, proxy="coordinated", forced_waits=(), capture=False):
    state = e.initial_state(seed, task)
    rng = random.Random(seed * 1021 + task)
    learner = PublicHistoryPartner()
    replay = []
    while not state["terminal"]:
        if state["turn"] in forced_waits:
            action = "wait"
        elif proxy == "random":
            action = rng.choice(e.legal_actions(state))
        elif proxy == "serial_greedy":
            action = serial_greedy(state)
        elif proxy == "public_history":
            action = learner.action(state)
        else:
            action = e.human_advisor(state)
        decision = e.decide(state)
        if capture:
            replay.append({"turn": state["turn"], "human_action": action, "ai_decision": decision,
                           "state": deepcopy(state)})
        state = e.step(state, action)
    result = {"seed": seed, "task": task, "proxy": proxy, "score": e.score(state),
              "emergency_rescues": state["metrics"]["emergency_rescues"]}
    if capture:
        result.update(replay=replay, final_state=state, forced_human_wait_turns=list(forced_waits))
    return result


def screen():
    config = e._configuration()
    all_results = []
    for split in ("development", "held_out"):
        for seed in config[split + "_seeds"]:
            for task in (1, 2, 3):
                for proxy in ("random", "serial_greedy", "public_history", "coordinated"):
                    all_results.append(dict(simulate(seed, task, proxy), split=split))
    groups = defaultdict(list)
    for row in all_results:
        groups[(row["split"], row["task"], row["proxy"])].append(row["score"]["task_score"])
    summary = [{"split": split, "task": task, "proxy": proxy, "n_scenarios": len(scores),
                "mean_task_score": mean(scores), "min_task_score": min(scores), "max_task_score": max(scores)}
               for (split, task, proxy), scores in groups.items()]
    report = {"version": e.VERSION, "scenario_version": e.SCENARIO_VERSION,
              "interpretation": "Development proxies only. No human participants, human A/B effects, confidence intervals or 50% success claim.",
              "summary": summary, "runs": all_results}
    folder = Path(__file__).resolve().parent
    (folder / "validation_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    traces = {"parallel_cooking_and_both_served": simulate(2000, 2, capture=True),
              "natural_emergency_after_human_delay": simulate(2000, 2, forced_waits=(9, 10, 11, 12), capture=True)}
    (folder / "validation_replays.json").write_text(json.dumps(traces, ensure_ascii=False, separators=(",", ":")) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(screen(), ensure_ascii=False, indent=2))
