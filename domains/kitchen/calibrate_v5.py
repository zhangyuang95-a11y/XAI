"""Freeze v5 menus/deadlines from development trajectories, then validate held-out.

Run ``python -m domains.kitchen.calibrate_v5``. All partners are deterministic
human-action proxies; this is not evidence of a human explanation treatment.
"""
from copy import deepcopy
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from . import engine as e

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/study_v3_kitchen.json"
EVIDENCE = ROOT.parent / "analysis/three_domain_revision_20260920_v35"


def rollout(seed, task):
    state = e.initial_state(seed, task)
    prepared_ages, raw_ages, pans = {}, {}, set()
    events, score_delta = [], 0
    while not state["terminal"]:
        for food in e._all_items(state):
            if food and food["stage"] in ("raw", "prepared") and food.get("freshness_basis") != "in_pan":
                ages = prepared_ages if food["stage"] == "prepared" else raw_ages
                ages[food["ingredient"]] = max(ages.get(food["ingredient"], 0), state["turn"] - food["freshness_started_turn"])
        state = e.step(state, e.human_advisor(state))
        for event in state["events"]:
            events.append(event["type"])
            if event["type"] == "pot_loaded": pans.add(event["pot"])
            if event["type"] == "score_delta": score_delta += event["delta"]
    assert state["raw_score"] == score_delta
    return {"seed": seed, "task": task, "completed": state["metrics"]["completed_orders"], "assigned": 5,
            "turns": state["turn"], "score": state["raw_score"], "metrics": deepcopy(state["metrics"]),
            "menu": [order["recipe"] for order in state["orders"]],
            "served_turns": {order["id"]: order["served_turn"] for order in state["orders"]},
            "maximum_prepared_age": prepared_ages, "maximum_raw_age": raw_ages, "used_pans": sorted(pans),
            "events": sorted(set(events)), "final_state_sha256": sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()}


def build():
    old = json.loads(CONFIG_PATH.read_text())
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    historical = EVIDENCE / "kitchen_previous_release_config.json"
    if old.get("rules_version") != e.VERSION and not historical.exists():
        historical.write_text(json.dumps(old,ensure_ascii=False,indent=2)+"\n")
    config = {"version": e.SCENARIO_VERSION, "scenario_version": e.SCENARIO_VERSION, "rules_version": e.VERSION,
              "study_mode": "pilot_pending_human_validation", "development_seeds": old["development_seeds"],
              "held_out_seeds": old["held_out_seeds"], "task_budgets": {str(t):1000 for t in (1,2,3)},
              "order_deadlines": {str(t):[1000]*5 for t in (1,2,3)}, "scenarios": {},
              "public_mechanics": {"width":9,"height":7,"orders_per_task":5,"full_menu_visible_from_start":True,
                 "prepare_turns":e.PREPARE_TURNS,"cook_turns":e.COOK_TURNS,"mixing_turns":2,"ready_full_turns_before_burn":8,
                 "handoff_capacity":1,"held_capacity":1,"human_buffer_capacity":1,"ai_raw_slots":2,"dedicated_temporary_plate_slots":2,
                 "trash_position":[4,4],"facing_required":True,"same_turn_handoff_conflict":"both_fail",
                 "score":{"served":100,"step":-1,"single_component_discard":-3,"combined_dish_discard":-10,"scale":"raw","negative_allowed":True},
                 "freshness":{"raw_after_acquisition":120,"prepared_after_completion":e.PREPARED_FRESH_TURNS,
                              "spoilage":"after actions on the expiry turn; item remains visible and unusable","pan_clock":"cooking/burning replaces oxidation","cooked_and_finished_oxidation":False},
                 "blocked_output":"occupied handoff commits a held finished dish to physical bin disposal; clearing later does not cancel",
                 "output_priority":"empty AI hand collects ready finished food before new ingredients; never takes delivered output back",
                 "conservation":"previous components + acquired = current components + served + explicit bin disposal"},
              "historical_evidence":"Previous release config and calibration retained in analysis/three_domain_revision_20260920_v35/kitchen_previous_release_config.json and Git history."}
    original = e._configuration
    e._configuration = lambda: config
    try:
        development_probe = [rollout(seed,t) for seed in config["development_seeds"] for t in (1,2,3)]
        assert all(r["completed"]==5 for r in development_probe), "Development probe failed; no held-out tuning allowed"
        for task in (1,2,3):
            rows=[r for r in development_probe if r["task"]==task]
            config["task_budgets"][str(task)] = ceil(max(r["turns"] for r in rows)*1.25/10)*10
            config["order_deadlines"][str(task)] = [ceil(max(r["served_turns"][f"order{i}"] for r in rows)*1.25/10)*10 for i in range(1,6)]
        config["scenarios"]={str(seed):{str(t):e._scenario(seed,t) for t in (1,2,3)} for seed in config["development_seeds"]+config["held_out_seeds"]}
        development=[rollout(seed,t) for seed in config["development_seeds"] for t in (1,2,3)]
        assert all(r["completed"]==5 for r in development), "Frozen development replay failed"
        held_out=[rollout(seed,t) for seed in config["held_out_seeds"] for t in (1,2,3)]
        summary={}
        for name,rows in (("development",development),("held_out",held_out)):
            summary[name]={str(t):{"runs":len([r for r in rows if r["task"]==t]),
                "all_complete":all(r["completed"]==5 for r in rows if r["task"]==t),
                "minimum_turns":min(r["turns"] for r in rows if r["task"]==t),"maximum_turns":max(r["turns"] for r in rows if r["task"]==t),
                "minimum_score":min(r["score"] for r in rows if r["task"]==t),"maximum_score":max(r["score"] for r in rows if r["task"]==t),
                "burnt":sum(r["metrics"]["burnt"] for r in rows if r["task"]==t),"spoiled":sum(r["metrics"]["spoiled"] for r in rows if r["task"]==t),
                "waste":sum(r["metrics"]["waste"] for r in rows if r["task"]==t)} for t in (1,2,3)}
        results={"evidence_type":"actual fixed-AI plus deterministic human-action proxy; no human participants", "engine_version":e.VERSION,
                 "scenario_version":e.SCENARIO_VERSION,"calibration_method":"Development only: per-task worst completion and per-menu-position worst serving turn, multiplied by 1.25, rounded upward to 10. Frozen before held-out evaluation.",
                 "task_budgets":config["task_budgets"],"order_deadlines":config["order_deadlines"],"summary":summary,
                 "development_probe":development_probe,"development_runs":development,"held_out_runs":held_out,"human_relative_gain":None}
        config["calibration_log"]=[{"version":e.VERSION,"method":results["calibration_method"],"human_effect":"not_measured","group_independent":True}]
        config["calibration_results"]={key:value for key,value in results.items() if key not in ("development_probe","development_runs","held_out_runs")}
        (EVIDENCE/"kitchen_calibration.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
        assert all(r["completed"]==5 for r in held_out), "Held-out failure preserved; revise only against development protocol"
        CONFIG_PATH.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
        return results["summary"]
    finally:
        e._configuration=original
        original.cache_clear()


if __name__ == "__main__":
    print(json.dumps(build(),indent=2))
