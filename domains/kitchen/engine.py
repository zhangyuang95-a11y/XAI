"""Pure deterministic kitchen mechanics and an observation-bound fixed teammate.

Neither decisions nor evidence queries mutate a state. The controller receives no
participant/group and never consults the private future order schedule.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random

DOMAIN = "kitchen"
VERSION = "kitchen-v3.1.0"
SCENARIO_VERSION = "kitchen-scenarios-v3.1.0"
WIDTH, HEIGHT = 9, 7
COOK_TURNS = {"tomato": 12, "onion": 14}
BURN_TURNS = 6
CHOP_TURNS = 2
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
LABELS = {"tomato": ("tomato", "番茄"), "onion": ("onion", "洋葱")}
STATIONS = [
    {"id": "ingredients", "x": 1, "y": 1, "kind": "ingredients", "label_en": "Ingredients", "label_zh": "原料柜"},
    {"id": "chop", "x": 3, "y": 1, "kind": "chop", "label_en": "Chopping board", "label_zh": "切菜台"},
    {"id": "human_buffer", "x": 1, "y": 3, "kind": "buffer", "label_en": "Your counter", "label_zh": "你的暂存台"},
    {"id": "plate", "x": 1, "y": 5, "kind": "plate", "label_en": "Plates", "label_zh": "装盘台"},
    {"id": "serve", "x": 3, "y": 5, "kind": "serve", "label_en": "Serving hatch", "label_zh": "上菜口"},
    {"id": "handoff", "x": 4, "y": 3, "kind": "handoff", "label_en": "Handoff counter", "label_zh": "交接台"},
    {"id": "pot1", "x": 7, "y": 1, "kind": "pot", "label_en": "Stove 1", "label_zh": "炉灶 1"},
    {"id": "ai_buffer", "x": 7, "y": 3, "kind": "buffer", "label_en": "AI counter", "label_zh": "AI 暂存台"},
    {"id": "pot2", "x": 7, "y": 5, "kind": "pot", "label_en": "Stove 2", "label_zh": "炉灶 2"},
]
STATION_BY_ID = {s["id"]: s for s in STATIONS}
WALLS = [[x, y] for y in range(HEIGHT) for x in range(WIDTH)
         if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1) or (x == 4 and y != 3)]
BLOCKED = frozenset(tuple(p) for p in WALLS) | frozenset((s["x"], s["y"]) for s in STATIONS)
FLOOR = frozenset((x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) not in BLOCKED)


def _pos(actor):
    return actor["x"], actor["y"]


def _adjacent(actor, station_id):
    station = STATION_BY_ID[station_id]
    return abs(actor["x"] - station["x"]) + abs(actor["y"] - station["y"]) == 1


@lru_cache(maxsize=512)
def _route(position, station_id):
    """Shortest legal path to any reachable interaction square, fixed tie order."""
    target = STATION_BY_ID[station_id]
    goals = {(target["x"] + dx, target["y"] + dy) for dx, dy in MOVES.values()} & FLOOR
    queue = deque([(position, ())])
    seen = {position}
    while queue:
        point, actions = queue.popleft()
        if point in goals:
            return actions
        for action, (dx, dy) in MOVES.items():
            candidate = point[0] + dx, point[1] + dy
            if candidate in FLOOR and candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, actions + (action,)))
    raise ValueError("Station is not reachable by this actor")


def _route_end(position, route):
    x, y = position
    for action in route:
        dx, dy = MOVES[action]
        x, y = x + dx, y + dy
    return x, y


def _approach(state, actor, station_id, interaction):
    route = _route(_pos(state[actor]), station_id)
    return route[0] if route else interaction


def _name(item, language="en"):
    if item is None:
        return "nothing" if language == "en" else "没有物品"
    base = LABELS[item["ingredient"]][language != "en"]
    stages = {
        "en": {"raw": "raw {0}", "chopped": "prepared {0}", "cooked": "cooked {0} soup", "plated": "plated {0} soup"},
        "zh": {"raw": "未切好的{0}", "chopped": "切好的{0}", "cooked": "煮好的{0}汤", "plated": "装盘的{0}汤"},
    }
    return stages[language][item["stage"]].format(base)


def _event(state, kind, en, zh, **extra):
    state["events"].append({"type": kind, "en": en, "zh": zh, **extra})


@lru_cache(maxsize=1)
def _configuration():
    path = Path(__file__).resolve().parents[2] / "configs" / "study_v3_kitchen.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _scenario(seed, task):
    config = _configuration()
    saved = config.get("scenarios", {}).get(str(seed), {}).get(str(task))
    if saved:
        return deepcopy(saved)
    rng = random.Random(int(seed) * 71 + int(task) * 10007)
    count = 4 if task == 1 else 6
    deadlines = [45, 75, 105, 140] if task == 1 else [45, 70, 95, 120, 145, 160]
    # Task 3 introduces only public-on-arrival orders. Their future recipe and
    # release time are kept out of public_state, decide, facts and advisor.
    arrivals = [0] * count if task != 3 else [0, 0, 24, 48, 72, 96]
    recipes = [rng.choice(["tomato", "onion"]) for _ in range(count)]
    if task == 2:
        # Close but manageable readiness windows arise naturally from cooking.
        recipes[0:2] = ["tomato", "onion"]
    starts = [(2, 3), (2, 2), (3, 3), (2, 4)]
    human_start = starts[rng.randrange(len(starts))] if task == 3 else (2, 3)
    return {"max_turns": 140 if task == 1 else 160,
            "human_start": list(human_start), "ai_start": [6, 3],
            "orders": [{"id": f"order{i + 1}", "ingredient": recipe, "arrival": arrival, "deadline": deadline}
                       for i, (recipe, arrival, deadline) in enumerate(zip(recipes, arrivals, deadlines))]}


def initial_state(seed: int, task: int) -> dict:
    if task not in (1, 2, 3):
        raise ValueError("Task must be 1, 2, or 3")
    scenario = _scenario(int(seed), task)
    revealed = [dict(order, status="pending", served_turn=None) for order in scenario["orders"] if order["arrival"] <= 0]
    hidden = [deepcopy(order) for order in scenario["orders"] if order["arrival"] > 0]
    return {"domain": DOMAIN, "version": VERSION, "scenario_version": SCENARIO_VERSION,
            "seed": int(seed), "task": task, "turn": 0, "max_turns": scenario["max_turns"],
            "terminal": False, "termination_reason": None, "events": [], "policy_memory": {},
            "human": {"x": scenario["human_start"][0], "y": scenario["human_start"][1], "holding": None},
            "ai": {"x": scenario["ai_start"][0], "y": scenario["ai_start"][1], "holding": None},
            "handoff": None, "buffers": {"human": None, "ai": None},
            "pots": [{"id": name, "x": STATION_BY_ID[name]["x"], "y": STATION_BY_ID[name]["y"],
                      "status": "empty", "item": None, "remaining": 0, "ready_age": 0}
                     for name in ("pot1", "pot2")],
            "orders": revealed, "_future_orders": hidden, "total_orders": len(scenario["orders"]),
            "next_item_id": 1,
            "metrics": {"completed_orders": 0, "burnt": 0, "expired": 0, "waste": 0,
                        "blocked_waits": 0, "human_waits": 0, "handoff_conflicts": 0,
                        "parallel_cooking_turns": 0, "emergency_rescues": 0}}


def legal_actions(state, actor="human") -> list[str]:
    if actor not in ("human", "ai"):
        raise ValueError("Unknown actor")
    if state["terminal"]:
        return []
    who = state[actor]
    held = who["holding"]
    result = ["wait"]
    for action, (dx, dy) in MOVES.items():
        if (who["x"] + dx, who["y"] + dy) in FLOOR:
            result.append(action)
    if held:
        result.append("discard")
    for station_id, item in (("handoff", state["handoff"]), (f"{actor}_buffer", state["buffers"][actor])):
        if _adjacent(who, station_id) and ((held is None) != (item is None)):
            result.append("interact_handoff" if station_id == "handoff" else "interact_buffer")
    if actor == "human":
        if not held and _adjacent(who, "ingredients"):
            result.extend(["take_tomato", "take_onion"])
        if held and held["stage"] == "raw" and _adjacent(who, "chop"):
            result.append("chop")
        if held and held["stage"] == "cooked" and _adjacent(who, "plate"):
            result.append("plate")
        if held and held["stage"] == "plated" and _adjacent(who, "serve"):
            result.append("serve")
    else:
        for pot in state["pots"]:
            if not _adjacent(who, pot["id"]):
                continue
            if ((pot["status"] == "empty" and held and held["stage"] == "chopped")
                    or (pot["status"] == "ready" and held is None)
                    or pot["status"] == "burnt"):
                result.append("interact_" + pot["id"])
    return result


def _pending(state, ingredient=None):
    return sorted([o for o in state["orders"] if o["status"] == "pending" and o["deadline"] > state["turn"]
                   and (ingredient is None or o["ingredient"] == ingredient)], key=lambda o: (o["deadline"], o["id"]))


def _useful(state, item):
    return item is not None and item["stage"] != "raw" and bool(_pending(state, item["ingredient"]))


def _burn_delay(pot):
    if pot["status"] == "cooking":
        return pot["remaining"] + BURN_TURNS
    if pot["status"] == "ready":
        return BURN_TURNS - pot["ready_age"]
    return 100000


def _removal_delay(pot, travel_and_interaction):
    """Earliest removal also waits until a cooking pot is actually ready."""
    earliest_ready_interaction = pot["remaining"] + 1 if pot["status"] == "cooking" else 1
    return max(travel_and_interaction, earliest_ready_interaction)


def _macro(state, kind, target, interaction, en, zh, **extra):
    route = _route(_pos(state["ai"]), target)
    return {"kind": kind, "target": target, "action": route[0] if route else interaction,
            "duration": len(route) + 1, "end": _route_end(_pos(state["ai"]), route),
            "reason_en": en, "reason_zh": zh, **extra}


def _free_hand_options(state):
    held = state["ai"]["holding"]
    if not held:
        return []
    options = []
    if state["handoff"] is None:
        options.append(_macro(state, "deliver" if held["stage"] in ("cooked", "plated") else "store_handoff",
                              "handoff", "interact_handoff",
                              "I am putting down the food to free my hand.", "我先把食物放下，腾出手。"))
    if state["buffers"]["ai"] is None:
        options.append(_macro(state, "store", "ai_buffer", "interact_buffer",
                              "I am using my empty counter to free my hand.", "我先使用空着的暂存台，腾出手。"))
    if held["stage"] == "chopped":
        for pot in state["pots"]:
            if pot["status"] == "empty" and _useful(state, held):
                options.append(_macro(state, "load", pot["id"], "interact_" + pot["id"],
                                      "I am putting the prepared ingredient in an empty pot.", "我把切好的原料放进空锅。"))
    return options


def _loadable_pots(state, item, position=None, pickup_delay=0):
    if not _useful(state, item) or item["stage"] != "chopped":
        return []
    position = position or _pos(state["ai"])
    result = []
    for pot in state["pots"]:
        if pot["status"] != "empty":
            continue
        route = _route(position, pot["id"])
        endpoint = _route_end(position, route)
        return_moves = len(_route(endpoint, "handoff"))
        # Optimistic public lower bound: load, cook, remove, deliver, human
        # pickup, plate and serve. Human can already be at the counter.
        earliest_serve = state["turn"] + pickup_delay + len(route) + 1 + COOK_TURNS[item["ingredient"]] + 1 + return_moves + 1 + 1 + 3 + 1 + 1
        if any(o["deadline"] >= earliest_serve for o in _pending(state, item["ingredient"])):
            result.append((len(route), pot["id"]))
    return [name for _, name in sorted(result)]


def _normal_plan(state):
    held = state["ai"]["holding"]
    memory = state.get("policy_memory", {})
    if held:
        if not _useful(state, held):
            return {"kind": "discard", "target": "", "action": "discard", "duration": 1, "end": _pos(state["ai"]),
                    "reason_en": f"The {_name(held)} in my hand cannot fulfill any currently available order, so I am clearing my hand.",
                    "reason_zh": f"我手里的{_name(held, 'zh')}无法完成当前有效订单，所以先清空手中物品。"}
        if held["stage"] in ("cooked", "plated"):
            if state["handoff"] is None:
                return _macro(state, "deliver", "handoff", "interact_handoff",
                              f"I am bringing the {_name(held)} to the empty handoff counter.", f"我要把{_name(held, 'zh')}送到空着的交接台。")
            if state["buffers"]["ai"] is None:
                return _macro(state, "store", "ai_buffer", "interact_buffer",
                              f"The handoff counter holds {_name(state['handoff'])}. I am putting the soup on my empty counter to free my hand.",
                              f"交接台上有{_name(state['handoff'], 'zh')}。我先把汤放到空着的暂存台，腾出手。")
            return {"kind": "blocked", "target": "handoff", "action": "wait", "duration": 1000, "end": _pos(state["ai"]),
                    "reason_en": f"I am holding {_name(held)}. The handoff counter holds {_name(state['handoff'])}, and my counter is full, so I cannot put down the soup.",
                    "reason_zh": f"我手里是{_name(held, 'zh')}。交接台上有{_name(state['handoff'], 'zh')}，我的暂存台也满了，所以现在无法放下汤。"}
        pots = _loadable_pots(state, held)
        if pots:
            target = memory.get("load_pot") if memory.get("load_pot") in pots else pots[0]
            return _macro(state, "load", target, "interact_" + target,
                          f"I am taking the prepared {LABELS[held['ingredient']][0]} to stove {target[-1]} to start cooking.",
                          f"我要把切好的{LABELS[held['ingredient']][1]}放进炉灶 {target[-1]} 开始煮。", load_pot=target)
        if state["buffers"]["ai"] is None:
            return _macro(state, "store", "ai_buffer", "interact_buffer",
                          "No available pot can finish this ingredient in time right now. I am putting it on my counter to free my hand.",
                          "目前没有可及时完成这份原料的空锅。我先把它放到暂存台，腾出手。")
        # This is a documented recovery rule, never a random waste action.
        return {"kind": "discard", "target": "", "action": "discard", "duration": 1, "end": _pos(state["ai"]),
                "reason_en": "I cannot start this prepared ingredient now, and my counter is full. I am discarding it to free my hand for the pots.",
                "reason_zh": "这份备料现在无法下锅，我的暂存台也满了。我先丢弃它，腾出手处理炉灶。"}
    ready = sorted([p for p in state["pots"] if p["status"] == "ready"], key=lambda p: (_burn_delay(p), p["id"]))
    if ready:
        reachable = [p for p in ready if len(_route(_pos(state["ai"]), p["id"])) + 1 <= _burn_delay(p)]
        pot = (reachable or ready)[0]
        turns = len(_route(_pos(state["ai"]), pot["id"])) + 1
        if turns > _burn_delay(pot):
            return _macro(state, "rescue_late", pot["id"], "interact_" + pot["id"],
                          f"The soup on stove {pot['id'][-1]} will burn in {_burn_delay(pot)} turns, but reaching and emptying it needs {turns} turns from here. I cannot save it in time; I am approaching the stove so I can clear it afterwards.",
                          f"炉灶 {pot['id'][-1]} 的汤再留 {_burn_delay(pot)} 回合会烧坏，但从这里到达并出锅需要 {turns} 回合，已经来不及救下。我先走向炉灶，之后清理。")
        return _macro(state, "rescue", pot["id"], "interact_" + pot["id"],
                      f"The soup on stove {pot['id'][-1]} is cooked. I am taking it off the heat before it burns.",
                      f"炉灶 {pot['id'][-1]} 的汤煮好了。我要在烧坏前把它取出来。")
    buffered = state["buffers"]["ai"]
    if buffered and (not _useful(state, buffered) or state["handoff"] is None and buffered["stage"] in ("cooked", "plated")):
        return _macro(state, "fetch_buffer", "ai_buffer", "interact_buffer",
                      "I am picking up the food from my counter so I can finish its handoff or clear unusable food.",
                      "我要从暂存台取回食物，继续交接或清理无法使用的食物。")
    handoff = state["handoff"]
    prepared_options = []
    for source, item in (("ai_buffer", buffered), ("handoff", handoff)):
        if not item or item["stage"] != "chopped":
            continue
        pickup_route = _route(_pos(state["ai"]), source)
        pickup_position = _route_end(_pos(state["ai"]), pickup_route)
        loadable = _loadable_pots(state, item, pickup_position, len(pickup_route) + 1)
        if loadable:
            due = _pending(state, item["ingredient"])[0]["deadline"]
            prepared_options.append((due, source, loadable[0], item))
    if prepared_options:
        _, source, target, item = min(prepared_options, key=lambda candidate: candidate[:2])
        where_en, where_zh = ("my counter", "我的暂存台") if source == "ai_buffer" else ("the handoff counter", "交接台")
        return _macro(state, "fetch_buffer" if source == "ai_buffer" else "accept", source,
                      "interact_buffer" if source == "ai_buffer" else "interact_handoff",
                      f"I am collecting the {_name(item)} from {where_en} for stove {target[-1]}. Its matching order has the earliest deadline among ingredients I can start now.",
                      f"我要从{where_zh}取走{_name(item, 'zh')}，放到炉灶 {target[-1]}。在现在可开工的备料中，它对应的订单最早截止。", load_pot=target)
    if handoff and handoff["stage"] not in ("cooked", "plated"):
        if not _useful(state, handoff):
            return _macro(state, "accept", "handoff", "interact_handoff",
                          f"I am collecting the {_name(handoff)} from the handoff counter.",
                          f"我要从交接台取走{_name(handoff, 'zh')}。", load_pot=None)
    burnt = [p for p in state["pots"] if p["status"] == "burnt"]
    if burnt:
        pot = burnt[0]
        return _macro(state, "clear_pot", pot["id"], "interact_" + pot["id"],
                      f"The food on stove {pot['id'][-1]} has burnt. I am clearing that pot so it can be used again.",
                      f"炉灶 {pot['id'][-1]} 的食物烧坏了。我要清空它，让这口锅可以重新使用。")
    cooking = sorted([p for p in state["pots"] if p["status"] == "cooking"], key=lambda p: (_burn_delay(p), p["id"]))
    if cooking:
        pot = cooking[0]
        route_to_handoff = _route(_pos(state["ai"]), "handoff")
        handoff_position = _route_end(_pos(state["ai"]), route_to_handoff)
        safe_return = len(route_to_handoff) + len(_route(handoff_position, pot["id"])) + 1
        human_item = state["human"]["holding"]
        preparation_possible = (human_item and human_item["stage"] in ("raw", "chopped")) or _next_needed_ingredient(state) is not None
        if (state["handoff"] is None and any(p["status"] == "empty" for p in state["pots"])
                and preparation_possible and safe_return <= _burn_delay(pot)):
            return _macro(state, "await_next_ingredient", "handoff", "wait",
                          f"Stove {pot['id'][-1]} still needs {pot['remaining']} cooking turns and the other pot is empty. I can wait near the handoff for another prepared portion and still return before this soup burns.",
                          f"炉灶 {pot['id'][-1]} 还要煮 {pot['remaining']} 回合，另一口锅是空的。我可以在交接台附近等下一份备料，并在这锅汤烧坏前返回。")
        return _macro(state, "watch_pot", pot["id"], "wait",
                      f"Stove {pot['id'][-1]} needs {pot['remaining']} more cooking turns. I am getting into position to take out the soup when it is ready.",
                      f"炉灶 {pot['id'][-1]} 还要煮 {pot['remaining']} 回合。我先到旁边，等煮好后出锅。")
    if buffered and buffered["stage"] in ("cooked", "plated"):
        return {"kind": "blocked", "target": "handoff", "action": "wait", "duration": 1000, "end": _pos(state["ai"]),
                "reason_en": f"The cooked soup is on my counter, but the handoff counter holds {_name(handoff)}. It needs a free place before I can hand over the soup.",
                "reason_zh": f"熟汤在我的暂存台上，但交接台上有{_name(handoff, 'zh')}。需要先腾出交接空间才能送出汤。"}
    approaching = bool(_route(_pos(state["ai"]), "handoff"))
    return _macro(state, "await_ingredient", "handoff", "wait",
                  "There is no prepared ingredient I can cook and no soup ready to hand over. " + ("I am moving next to the handoff counter to wait for prepared food." if approaching else "I am waiting next to the handoff counter for prepared food."),
                  "现在没有可下锅的备料，也没有可以交接的熟汤。" + ("我正在走向交接台旁，准备等候备料。" if approaching else "我已在交接台旁等候备料。"))


def decide(state) -> dict:
    if state["terminal"]:
        return {"action": "wait", "reason_code": "terminal", "reason_en": "This task has ended.", "reason_zh": "本任务已结束。",
                "goal": "Finished", "memory": deepcopy(state["policy_memory"]), "alternatives": [], "facts": []}
    plan = _normal_plan(state)
    held = state["ai"]["holding"]
    committed_id = state.get("policy_memory", {}).get("emergency_pot")
    committed = next((p for p in state["pots"] if p["id"] == committed_id and p["status"] in ("cooking", "ready")), None)
    if committed and held is None:
        # Freeing the hand is only the first part of rescuing a pot. Retain
        # that commitment instead of picking up the same stored ingredient
        # again and oscillating between the handoff counter and the pot.
        plan = _macro(state, "emergency", committed["id"],
                      "interact_" + committed["id"] if committed["status"] == "ready" else "wait",
                      f"I am continuing to stove {committed['id'][-1]} to take its soup off the heat. It will burn in {_burn_delay(committed)} turns if left in the pot.",
                      f"我继续去炉灶 {committed['id'][-1]} 取出汤。汤再留 {_burn_delay(committed)} 回合会烧坏。")
    danger = []
    for pot in state["pots"]:
        if pot["status"] not in ("cooking", "ready") or plan["target"] == pot["id"] and plan["kind"] in ("rescue", "watch_pot", "emergency"):
            continue
        normal_delay = _removal_delay(pot, plan["duration"] + len(_route(tuple(plan["end"]), pot["id"])) + 1)
        deadline = _burn_delay(pot)
        if normal_delay <= deadline:
            continue
        if not held:
            direct_delay = _removal_delay(pot, len(_route(_pos(state["ai"]), pot["id"])) + 1)
            if direct_delay <= deadline:
                danger.append((deadline, pot["id"], direct_delay, normal_delay, None))
        else:
            choices = []
            for option in _free_hand_options(state):
                delay = _removal_delay(pot, option["duration"] + len(_route(tuple(option["end"]), pot["id"])) + 1)
                if delay <= deadline:
                    choices.append((delay, option["duration"], option["target"], option))
            if choices:
                delay, _, _, freeing = min(choices, key=lambda c: c[:3])
                danger.append((deadline, pot["id"], delay, normal_delay, freeing))
    emergency = None
    if danger:
        deadline, pot_id, direct_delay, normal_delay, freeing = min(danger, key=lambda d: (d[0], d[1]))
        emergency = {"pot": pot_id, "burn_in": deadline, "rescue_turns": direct_delay, "normal_plan_turns": normal_delay}
        if freeing:
            plan = deepcopy(freeing)
            plan["reason_en"] = f"Stove {pot_id[-1]} will burn in {deadline} turns if its soup stays in the pot. I must free my hand first; if the transfers succeed, this route lets me take out the soup in {direct_delay} turns."
            plan["reason_zh"] = f"炉灶 {pot_id[-1]} 的汤再留 {deadline} 回合会烧坏。我需要先腾出手；如果交接顺利，这条路线可在 {direct_delay} 回合内完成出锅。"
        else:
            pot = next(p for p in state["pots"] if p["id"] == pot_id)
            plan = _macro(state, "emergency", pot_id, "interact_" + pot_id if pot["status"] == "ready" else "wait",
                          f"I am going to stove {pot_id[-1]} first. Continuing the other job would take {normal_delay} turns before removal, but this soup will burn in {deadline} turns if left in the pot.",
                          f"我先去炉灶 {pot_id[-1]}。继续另一项工作后再出锅需要 {normal_delay} 回合，但这锅汤再留 {deadline} 回合就会烧坏。")
    action = plan["action"]
    if action not in legal_actions(state, "ai"):
        raise AssertionError(f"Controller planned an illegal action: {action}")
    memory = {"goal_kind": plan["kind"], "target": plan["target"],
              "load_pot": plan.get("load_pot", state.get("policy_memory", {}).get("load_pot")),
              "emergency_pot": emergency["pot"] if emergency else committed["id"] if committed else None}
    alternatives = []
    if action != "wait":
        alternatives.append({"action": "wait", "en": "Waiting would leave my position and held item unchanged for this turn; pot timers would still advance.",
                             "zh": "等待会让我本回合的位置和手持物品不变，但锅的计时仍会推进。"})
    if plan.get("target") in STATION_BY_ID:
        route = _route(_pos(state["ai"]), plan["target"])
        alternatives.append({"action": action, "en": f"The selected work station is {len(route)} movement turns from my current position.",
                             "zh": f"从我现在的位置到选定工位需要移动 {len(route)} 回合。"})
    if emergency:
        alternatives.append({"action": "continue_previous_job", "en": f"Finishing the other job before emptying stove {emergency['pot'][-1]} would require {emergency['normal_plan_turns']} turns, beyond its {emergency['burn_in']}-turn remaining window.",
                             "zh": f"先完成另一项工作再给炉灶 {emergency['pot'][-1]} 出锅需要 {emergency['normal_plan_turns']} 回合，超过剩余 {emergency['burn_in']} 回合的窗口。"})
    return {"action": action, "reason_code": "emergency" if emergency else plan["kind"],
            "reason_en": plan["reason_en"], "reason_zh": plan["reason_zh"],
            "goal": plan["target"] or plan["kind"], "memory": memory,
            "alternatives": alternatives, "facts": [], "emergency": emergency}


def _interact(state, actor, action, new_turn, newly_loaded):
    who = state[actor]
    held = who["holding"]
    if action == "discard":
        who["holding"] = None
        state["metrics"]["waste"] += 1
        _event(state, "waste", f"{'You' if actor == 'human' else 'AI'} discarded {_name(held)}.",
               f"{'你' if actor == 'human' else 'AI'}丢弃了{_name(held, 'zh')}。", actor=actor, item=deepcopy(held))
    elif action.startswith("take_"):
        ingredient = action[5:]
        item = {"id": f"item{state['next_item_id']}", "ingredient": ingredient, "stage": "raw", "chop_progress": 0}
        state["next_item_id"] += 1
        who["holding"] = item
        _event(state, "ingredient_taken", f"You took one {ingredient} portion.", f"你取了一份{LABELS[ingredient][1]}。", actor=actor, item=deepcopy(item))
    elif action == "chop":
        held["chop_progress"] += 1
        if held["chop_progress"] == CHOP_TURNS:
            held["stage"] = "chopped"
        _event(state, "chop", f"Preparation: {held['chop_progress']} of {CHOP_TURNS} chopping turns.",
               f"切菜进度：{held['chop_progress']}/{CHOP_TURNS}。", item=deepcopy(held))
    elif action in ("interact_handoff", "interact_buffer"):
        on_handoff = action == "interact_handoff"
        old = state["handoff"] if on_handoff else state["buffers"][actor]
        location_en = "handoff counter" if on_handoff else "storage counter"
        location_zh = "交接台" if on_handoff else "暂存台"
        if held:
            if on_handoff:
                state["handoff"] = held
            else:
                state["buffers"][actor] = held
            who["holding"] = None
            _event(state, "item_placed", f"{'You' if actor == 'human' else 'AI'} placed {_name(held)} on the {location_en}.",
                   f"{'你' if actor == 'human' else 'AI'}把{_name(held, 'zh')}放到{location_zh}。", actor=actor, station="handoff" if on_handoff else actor + "_buffer", item=deepcopy(held))
        else:
            who["holding"] = old
            if on_handoff:
                state["handoff"] = None
            else:
                state["buffers"][actor] = None
            _event(state, "item_taken", f"{'You' if actor == 'human' else 'AI'} took {_name(old)} from the {location_en}.",
                   f"{'你' if actor == 'human' else 'AI'}从{location_zh}取走{_name(old, 'zh')}。", actor=actor, station="handoff" if on_handoff else actor + "_buffer", item=deepcopy(old))
    elif action.startswith("interact_pot"):
        pot_id = action[len("interact_"):]
        pot = next(p for p in state["pots"] if p["id"] == pot_id)
        if pot["status"] == "burnt":
            wasted = deepcopy(pot["item"])
            pot.update(status="empty", item=None, remaining=0, ready_age=0)
            state["metrics"]["waste"] += 1
            _event(state, "pot_cleared", f"AI cleared burnt food from stove {pot_id[-1]}.", f"AI 清除了炉灶 {pot_id[-1]} 中烧坏的食物。", pot=pot_id, item=wasted)
        elif pot["status"] == "empty":
            pot.update(status="cooking", item=held, remaining=COOK_TURNS[held["ingredient"]], ready_age=0)
            who["holding"] = None
            newly_loaded.add(pot_id)
            _event(state, "pot_loaded", f"AI started {held['ingredient']} soup on stove {pot_id[-1]}; it takes {pot['remaining']} full cooking turns.",
                   f"AI 在炉灶 {pot_id[-1]} 开始煮{LABELS[held['ingredient']][1]}汤，需要完整烹饪 {pot['remaining']} 回合。", pot=pot_id, item=deepcopy(held))
        elif pot["status"] == "ready":
            who["holding"] = pot["item"]
            who["holding"]["stage"] = "cooked"
            removed = deepcopy(who["holding"])
            pot.update(status="empty", item=None, remaining=0, ready_age=0)
            if state["policy_memory"].get("emergency_pot") == pot_id:
                state["metrics"]["emergency_rescues"] += 1
            _event(state, "pot_removed", f"AI took cooked soup off stove {pot_id[-1]}.", f"AI 从炉灶 {pot_id[-1]} 取出了熟汤。", pot=pot_id, item=removed)
    elif action == "plate":
        held["stage"] = "plated"
        _event(state, "plated", f"You plated the {held['ingredient']} soup.", f"你把{LABELS[held['ingredient']][1]}汤装好了盘。", item=deepcopy(held))
    elif action == "serve":
        matching = sorted([o for o in state["orders"] if o["status"] == "pending" and o["ingredient"] == held["ingredient"] and o["deadline"] >= new_turn], key=lambda o: (o["deadline"], o["id"]))
        if matching:
            order = matching[0]
            order.update(status="completed", served_turn=new_turn)
            who["holding"] = None
            state["metrics"]["completed_orders"] += 1
            _event(state, "served", f"You completed {order['id'].replace('order', 'order ')} with {held['ingredient']} soup.",
                   f"你用{LABELS[held['ingredient']][1]}汤完成了订单 {order['id'].replace('order', '')}。", order_id=order["id"], item=deepcopy(held))
        else:
            _event(state, "serve_rejected", "This plated soup does not match an order that can still be served. You keep holding it.",
                   "这份装盘汤不符合仍可交付的订单；物品仍在你手中。", item=deepcopy(held))


def step(state, human_action, decision=None) -> dict:
    if state["terminal"]:
        raise ValueError("This task has ended")
    if human_action not in legal_actions(state):
        raise ValueError("That action is not available in this position")
    decision = deepcopy(decision) if decision is not None else decide(state)
    if decision["action"] not in legal_actions(state, "ai"):
        raise ValueError("AI action is not legal in the saved pre-action state")
    nxt = deepcopy(state)
    nxt["events"] = []
    new_turn = state["turn"] + 1
    nxt["policy_memory"] = deepcopy(decision["memory"])
    actions = {"human": human_action, "ai": decision["action"]}
    conflict = actions["human"] == actions["ai"] == "interact_handoff"
    newly_loaded = set()
    # Legal interactions refer to s_t. Both actors stay within disjoint areas;
    # interactions are never made legal by the other actor's same-turn action.
    for actor, action in actions.items():
        if action in MOVES:
            dx, dy = MOVES[action]
            nxt[actor]["x"] += dx
            nxt[actor]["y"] += dy
        elif action == "wait":
            if actor == "human":
                nxt["metrics"]["human_waits"] += 1
            elif decision["reason_code"] == "blocked":
                nxt["metrics"]["blocked_waits"] += 1
        elif not (conflict and action == "interact_handoff"):
            _interact(nxt, actor, action, new_turn, newly_loaded)
    if conflict:
        nxt["metrics"]["handoff_conflicts"] += 1
        _event(nxt, "handoff_conflict", "Both teammates tried to use the handoff counter at once. Neither transfer happened; one turn passed.",
               "双方同回合使用交接台，两次交互都失败，物品不变，仍消耗一回合。")
    # A fresh load does not count its loading turn; a fresh ready pot does not
    # count its cooking-completion turn toward the six full ready turns.
    for pot in nxt["pots"]:
        if pot["id"] in newly_loaded:
            continue
        if pot["status"] == "cooking":
            pot["remaining"] -= 1
            if pot["remaining"] == 0:
                pot["status"] = "ready"
                pot["item"]["stage"] = "cooked"
                pot["ready_age"] = 0
                _event(nxt, "pot_ready", f"Soup on stove {pot['id'][-1]} is ready. It will burn after six more full turns left in the pot.",
                       f"炉灶 {pot['id'][-1]} 的汤煮好了；继续留在锅里满 6 回合后会烧坏。", pot=pot["id"])
        elif pot["status"] == "ready":
            pot["ready_age"] += 1
            if pot["ready_age"] >= BURN_TURNS:
                pot["status"] = "burnt"
                nxt["metrics"]["burnt"] += 1
                _event(nxt, "pot_burnt", f"Soup on stove {pot['id'][-1]} burnt after six full ready turns.",
                       f"炉灶 {pot['id'][-1]} 的汤煮好后留满 6 回合，烧坏了。", pot=pot["id"])
    nxt["turn"] = new_turn
    for order in nxt["orders"]:
        if order["status"] == "pending" and order["deadline"] <= new_turn:
            order["status"] = "expired"
            nxt["metrics"]["expired"] += 1
            _event(nxt, "order_expired", f"Order {order['id'].replace('order', '')} expired after this turn's serving opportunity.",
                   f"订单 {order['id'].replace('order', '')} 在本回合上菜机会结束后过期。", order_id=order["id"])
    future = []
    for order in nxt["_future_orders"]:
        if order["arrival"] <= new_turn:
            nxt["orders"].append(dict(order, status="pending", served_turn=None))
            _event(nxt, "order_arrived", f"New order: {order['ingredient']} soup, due on turn {order['deadline']}.",
                   f"新订单：{LABELS[order['ingredient']][1]}汤，截止回合 {order['deadline']}。", order=deepcopy(order))
        else:
            future.append(order)
    nxt["_future_orders"] = future
    if all(p["status"] in ("cooking", "ready") for p in nxt["pots"]):
        nxt["metrics"]["parallel_cooking_turns"] += 1
    if new_turn >= nxt["max_turns"]:
        nxt["terminal"], nxt["termination_reason"] = True, "turn_budget"
    elif not nxt["_future_orders"] and all(o["status"] != "pending" for o in nxt["orders"]):
        nxt["terminal"], nxt["termination_reason"] = True, "all_orders_resolved"
    if nxt["terminal"]:
        _event(nxt, "task_ended", "The task has ended.", "本任务已结束。")
    return nxt


def score(state) -> dict:
    completed = state["metrics"]["completed_orders"]
    metrics = deepcopy(state["metrics"])
    # A controller trigger count belongs to the private audit, not the public
    # score projection in stages where explanations are unavailable.
    metrics.pop("emergency_rescues", None)
    metrics.update(total_orders=state["total_orders"], elapsed_turns=state["turn"])
    return {"task_score": 100.0 * completed / state["total_orders"], "raw_score": completed, "metrics": metrics}


def public_state(state) -> dict:
    pots = []
    for pot in state["pots"]:
        pots.append({"id": pot["id"], "x": pot["x"], "y": pot["y"], "status": pot["status"],
                     "ingredient": pot["item"]["ingredient"] if pot["item"] else None,
                     "item": deepcopy(pot["item"]), "remaining": pot["remaining"], "ready_age": pot["ready_age"],
                     "burn_in": _burn_delay(pot) if pot["status"] in ("cooking", "ready") else None})
    orders = [dict(deepcopy(order), remaining=max(0, order["deadline"] - state["turn"])) for order in state["orders"]]
    # Explicit allowlist: never leak memory, decision reasons, hidden schedule,
    # seed, an advisor action, or any explanation via ordinary participant view.
    return {"domain": DOMAIN, "version": VERSION, "task": state["task"], "turn": state["turn"],
            "max_turns": state["max_turns"], "terminal": state["terminal"], "events": deepcopy(state["events"]),
            "score": score(state), "width": WIDTH, "height": HEIGHT, "walls": deepcopy(WALLS),
            "stations": deepcopy(STATIONS), "human": deepcopy(state["human"]), "ai": deepcopy(state["ai"]),
            "pots": pots, "orders": orders, "total_orders": state["total_orders"],
            "handoff": deepcopy(state["handoff"]), "buffers": deepcopy(state["buffers"])}


def rules(language="en") -> list[str]:
    en = [
        "Work together: you prepare ingredients, plate soup and serve; your AI teammate cooks and hands over food. Neither teammate can finish an order alone.",
        "Choose one move, one available interaction, or Wait, then confirm. Both teammates act from the same starting turn; questions and replay do not advance time.",
        "Each teammate holds one item. Use adjacent stations; counters hold one item each. You can discard a held item anywhere for one turn, without an extra score penalty.",
        "One tomato or onion portion makes one soup. Chop it with two interactions. Tomato takes 12 cooking turns; onion takes 14. Loading does not count as a cooking turn.",
        "Ready soup burns after six additional full turns in the pot. The turn it becomes ready does not count. Removed soup no longer burns. Clearing a burnt pot takes one interaction.",
        "The handoff counter holds one item. If both teammates interact with it on the same turn, both transfers fail. An item put down this turn cannot be picked up on the same turn.",
        "Take cooked soup, plate it, then serve a matching order. An order due on turn T may still be served during turn T, before expiry. The earliest due matching order is served first.",
        "Task score is 100 times correctly completed, on-time orders divided by the fixed number of orders. Waste, burnt food and expired orders do not add hidden penalties.",
    ]
    zh = [
        "双方合作出餐：你负责备料、装盘和上菜，AI 队友负责烹饪与交接。任何一方都不能独自完成订单。",
        "选择移动、一个可用交互或等待，再确认。双方按同一个回合前状态行动；提问和回放不推进时间。",
        "双方各拿一件物品，工位须从相邻位置交互，台面各放一件物品。任意位置都可花一回合丢弃手持物品，不另加扣分。",
        "一份番茄或洋葱原料可做一份汤，需两次切菜交互。番茄煮 12 回合，洋葱煮 14 回合；下锅当回合不计烹饪时间。",
        "熟汤继续留在锅中满 6 回合会烧坏，刚变熟当回合不计入。取出后不再烧坏；清空烧坏的锅需一次交互。",
        "交接台只放一件物品。双方同回合交互时，两次交互都失败。本回合新放下的物品不能在同回合被接走。",
        "取回熟汤，装盘，再提交匹配订单。截止回合 T 仍可在该回合上菜，之后才过期。同菜优先交最早截止订单。",
        "任务分数为：按时正确完成订单数除以本局固定订单总数，再乘 100。浪费、烧坏和过期没有额外隐藏扣分。",
    ]
    return zh if language == "zh" else en


def facts(state, decision=None) -> list[dict]:
    decision = decide(state) if decision is None else decision
    rows = [{"id": "current_turn", "en": f"This is Task {state['task']}, turn {state['turn']}, with {state['max_turns'] - state['turn']} turns remaining.",
             "zh": f"当前是 Task {state['task']}，回合 {state['turn']}，剩余 {state['max_turns'] - state['turn']} 回合。"},
            {"id": "ai_reason", "en": decision["reason_en"], "zh": decision["reason_zh"]},
            {"id": "ai_next_action", "en": f"My next action is {action_label(decision['action'], 'en').lower()}.",
             "zh": f"我下一步会{action_label(decision['action'], 'zh')}。"}]
    rows.append({"id": "current_score", "en": f"The team has completed {state['metrics']['completed_orders']} of {state['total_orders']} orders on time. The current task score is {score(state)['task_score']:.2f} out of 100.",
                 "zh": f"双方已按时完成 {state['metrics']['completed_orders']}/{state['total_orders']} 个订单，当前任务分数是 {score(state)['task_score']:.2f}/100。"})
    for actor in ("human", "ai"):
        who = state[actor]
        rows.append({"id": actor + "_holding", "en": f"{'You are' if actor == 'human' else 'I am'} holding {_name(who['holding'])} at column {who['x']}, row {who['y']}.",
                     "zh": f"{'你' if actor == 'human' else '我'}在第 {who['x']} 列、第 {who['y']} 行，手里是{_name(who['holding'], 'zh')}。"})
    if not state["terminal"]:
        suggestion = human_advisor(state)
        rows.append({"id": "human_available_option",
                     "en": f"One available next action for you is: {action_label(suggestion, 'en').lower()}. This is a suggestion; you still choose and confirm your own action.",
                     "zh": f"你下一步可选择：{action_label(suggestion, 'zh')}。这是一项建议，仍由你自行选择并确认动作。"})
        rows.append({"id": "human_legal_actions", "en": "Your available actions here are: " + ", ".join(action_label(a, "en").lower() for a in legal_actions(state)) + ".",
                     "zh": "你当前位置可用的动作是：" + "、".join(action_label(a, "zh") for a in legal_actions(state)) + "。"})
    for name, item in [("handoff", state["handoff"]), ("human_buffer", state["buffers"]["human"]), ("ai_buffer", state["buffers"]["ai"])]:
        st = STATION_BY_ID[name]
        rows.append({"id": name, "en": f"The {st['label_en'].lower()} holds {_name(item)}.", "zh": f"{st['label_zh']}上是{_name(item, 'zh')}。"})
    for pot in state["pots"]:
        number = pot["id"][-1]
        if pot["status"] == "cooking":
            en = f"Stove {number} is cooking {pot['item']['ingredient']} soup: {pot['remaining']} full turns until ready, then six additional turns before it burns if not removed."
            zh = f"炉灶 {number} 正在煮{LABELS[pot['item']['ingredient']][1]}汤，还需 {pot['remaining']} 回合变熟，之后再留满 6 回合会烧坏。"
        elif pot["status"] == "ready":
            en = f"Soup on stove {number} is ready and will burn in {_burn_delay(pot)} turns if not removed."
            zh = f"炉灶 {number} 的汤已煮好，再留 {_burn_delay(pot)} 回合会烧坏。"
        else:
            en = f"Stove {number} is {'empty' if pot['status'] == 'empty' else 'holding burnt food'} ."
            zh = f"炉灶 {number}{'是空的' if pot['status'] == 'empty' else '里有烧坏的食物'}。"
        rows.append({"id": pot["id"], "en": en, "zh": zh})
        moves = len(_route(_pos(state["ai"]), pot["id"]))
        rows.append({"id": pot["id"] + "_distance", "en": f"From my current position, reaching stove {number} needs {moves} movement turns. Taking ready soup out then needs one interaction, and my hand must be empty.",
                     "zh": f"从我现在的位置到炉灶 {number} 需要移动 {moves} 回合；取出熟汤还需要一次交互，且手必须是空的。"})
    for order in state["orders"]:
        rows.append({"id": order["id"], "en": f"Order {order['id'].replace('order', '')} is for {order['ingredient']} soup, due on turn {order['deadline']}; it is {order['status']}.",
                     "zh": f"订单 {order['id'].replace('order', '')} 需要{LABELS[order['ingredient']][1]}汤，截止回合 {order['deadline']}，状态为{ {'pending': '待完成', 'completed': '已完成', 'expired': '已过期'}[order['status']]}。"})
    for index, alternative in enumerate(decision["alternatives"]):
        rows.append({"id": f"alternative{index}", "en": alternative["en"], "zh": alternative["zh"]})
    for index, event in enumerate(state["events"]):
        rows.append({"id": f"event{index}", "en": event["en"], "zh": event["zh"]})
    for index, (en, zh) in enumerate(zip(rules("en"), rules("zh"))):
        rows.append({"id": f"public_rule{index}", "en": en, "zh": zh})
    return rows


def action_label(action, language="en"):
    pairs = {"up": ("Move up", "向上移动"), "down": ("Move down", "向下移动"),
             "left": ("Move left", "向左移动"), "right": ("Move right", "向右移动"),
             "wait": ("Wait", "等待"), "discard": ("Discard held item", "丢弃手持物品"),
             "interact_handoff": ("Use handoff counter", "使用交接台"), "interact_buffer": ("Use storage counter", "使用暂存台"),
             "interact_pot1": ("Use stove 1", "使用炉灶 1"), "interact_pot2": ("Use stove 2", "使用炉灶 2"),
             "take_tomato": ("Take tomato", "拿取番茄"), "take_onion": ("Take onion", "拿取洋葱"),
             "chop": ("Chop ingredient", "切菜"), "plate": ("Plate soup", "装盘"), "serve": ("Serve soup", "上菜")}
    return pairs[action][language == "zh"]


def _all_items(state):
    return [state["human"]["holding"], state["ai"]["holding"], state["handoff"],
            state["buffers"]["human"], state["buffers"]["ai"]] + [p["item"] for p in state["pots"] if p["status"] != "burnt"]


def _next_needed_ingredient(state):
    counts = {name: 0 for name in COOK_TURNS}
    for item in _all_items(state):
        if item:
            counts[item["ingredient"]] += 1
    for order in _pending(state):
        recipe = order["ingredient"]
        if counts[recipe]:
            counts[recipe] -= 1
        else:
            return recipe
    return None


def human_advisor(state) -> str:
    """Development partner: legal human moves only, currently observed state.

    Uses the same fixed AI's current decision, without replacing its action or
    looking at undisclosed orders. It handles the entire public production chain.
    """
    if state["terminal"]:
        return "wait"
    held, handoff = state["human"]["holding"], state["handoff"]
    buffered = state["buffers"]["human"]
    ai_decision = decide(state)
    if held:
        if not _pending(state, held["ingredient"]):
            return "discard"
        if held["stage"] == "plated":
            return _approach(state, "human", "serve", "serve")
        if held["stage"] == "cooked":
            return _approach(state, "human", "plate", "plate")
        if held["stage"] == "raw":
            return _approach(state, "human", "chop", "chop")
        cooked_needs_handoff = any(item and item["stage"] in ("cooked", "plated")
                                    for item in (handoff, state["ai"]["holding"], state["buffers"]["ai"]))
        active_pots = [p for p in state["pots"] if p["status"] in ("cooking", "ready")]
        imminent = any(p["status"] == "ready" or p["remaining"] <= 4 for p in active_pots)
        # A single prepared portion can wait on the human counter while the
        # handoff lane is used for cooked soup. Never issue the AI an action.
        if cooked_needs_handoff or (len(active_pots) == 2 and imminent):
            if buffered is None:
                return _approach(state, "human", "human_buffer", "interact_buffer")
            return "discard"
        if handoff is None:
            return _approach(state, "human", "handoff", "interact_handoff")
        if buffered is None:
            return _approach(state, "human", "human_buffer", "interact_buffer")
        return "wait"
    if handoff and handoff["stage"] in ("cooked", "plated"):
        return _approach(state, "human", "handoff", "interact_handoff")
    if buffered and buffered["stage"] in ("cooked", "plated"):
        return _approach(state, "human", "human_buffer", "interact_buffer")
    # Do not fetch a stored prepared portion until there is room to deliver it;
    # return to the handoff first while a cooked output is in transit.
    ai_food = state["ai"]["holding"]
    cooked_coming = ai_food and ai_food["stage"] in ("cooked", "plated") or any(p["status"] == "ready" for p in state["pots"]) or (state["buffers"]["ai"] and state["buffers"]["ai"]["stage"] in ("cooked", "plated"))
    if cooked_coming:
        if handoff and handoff["stage"] not in ("cooked", "plated") and ai_decision["action"] != "interact_handoff":
            return _approach(state, "human", "handoff", "interact_handoff")
        return _approach(state, "human", "handoff", "wait")
    active = [p for p in state["pots"] if p["status"] in ("cooking", "ready")]
    if buffered and handoff is None and not (len(active) == 2 and any(p["remaining"] <= 7 for p in active)):
        return _approach(state, "human", "human_buffer", "interact_buffer")
    needed = _next_needed_ingredient(state)
    if needed and buffered is None:
        return _approach(state, "human", "ingredients", "take_" + needed)
    return _approach(state, "human", "handoff", "wait")


def demonstration() -> dict:
    """A genuine deterministic trajectory, with public mechanics captions only."""
    state = initial_state(1000, 1)
    frames = [public_state(state)]
    captions = [{"index": 0, "en": "Prepare, cook, plate and serve together. Each teammate works on one side of the kitchen.",
                 "zh": "双方合作备料、烹饪、装盘和上菜，每人负责厨房的一侧。"}]
    captioned = set()
    conflict_done = False
    for _ in range(state["max_turns"]):
        if state["terminal"]:
            break
        action = human_advisor(state)
        ai_action = decide(state)["action"]
        if (not conflict_done and ai_action == "interact_handoff" and state["handoff"]
                and "interact_handoff" in legal_actions(state) and not state["human"]["holding"]):
            action, conflict_done = "interact_handoff", True
        state = step(state, action)
        frames.append(public_state(state))
        types = {event["type"] for event in state["events"]}
        mappings = [
            ("chop", "Use two chopping interactions for one ingredient portion. Each confirmed action advances one turn.", "一份原料要切两次；每次确认的动作推进一回合。"),
            ("handoff_conflict", "Both teammates used the one-item handoff counter together: neither transfer happened. On the next turn one can wait while the other uses it.", "双方同时使用单物品交接台，两次交互都失败；下一回合一方可以等待，让另一方使用。"),
            ("pot_ready", "Cooked soup must leave the stove before six more full turns pass. Once removed it no longer burns.", "汤煮好后需在额外 6 回合内出锅，取出后不再烧坏。"),
            ("served", "A correctly plated soup delivered by the order deadline completes an order for the team.", "正确装盘并在截止回合前上菜，双方共同完成一个订单。"),
        ]
        for kind, en, zh in mappings:
            if kind in types and kind not in captioned:
                captions.append({"index": len(frames) - 1, "en": en, "zh": zh})
                captioned.add(kind)
        if "served" in captioned and conflict_done:
            break
    captions.append({"index": len(frames) - 1, "en": "The demonstration is complete. In the tasks, choose and confirm each action yourself; replay and questions do not use game turns.",
                     "zh": "演示完成。正式任务请自行选择并确认每步动作；回放和提问不消耗游戏回合。"})
    return {"frames": frames, "captions": captions}


def comprehension(language="en") -> list[dict]:
    items = [
        {"id": "kitchen_predict", "text": "AI is next to stove 2 with an empty hand. Its cooked soup will burn this turn, while the handoff counter has a fresh prepared onion. What will AI do next?",
         "options": ["Take the cooked soup off stove 2", "Walk to collect the onion", "Wait for the onion to disappear"], "answer": 0},
        {"id": "kitchen_reason", "text": "AI holds cooked onion soup, the handoff counter holds a prepared tomato, and AI's storage counter is empty. AI walks toward its storage counter. What best explains this?",
         "options": ["It can serve orders directly from that counter", "It can put down the soup there and free its hand", "Putting soup there earns extra points"], "answer": 1},
        {"id": "kitchen_change", "text": "AI holds cooked tomato soup and both counters on its side are occupied. If you clear the handoff counter, what becomes possible?",
         "options": ["AI can hand over the cooked soup", "The soup is automatically served", "The kitchen stops advancing turns"], "answer": 0},
    ]
    if language != "zh":
        return items
    translated = [
        ("AI 空手站在炉灶 2 旁，锅里的熟汤本回合就会烧坏，而交接台上有新备好的洋葱。AI 下一步会做什么？", ["从炉灶 2 取出熟汤", "走去取洋葱", "等待洋葱消失"]),
        ("AI 手里拿着熟洋葱汤，交接台上有切好的番茄，AI 暂存台是空的。AI 走向暂存台，最合理的原因是什么？", ["它可以直接从暂存台上菜", "它可以把汤放下，腾出手", "把汤放在那里能额外得分"]),
        ("AI 手里拿着熟番茄汤，它一侧的两个台面都被占用了。如果你清空交接台，会使什么成为可能？", ["AI 可以交接熟汤", "汤会自动上菜", "厨房会停止推进回合"]),
    ]
    return [dict(item, text=text, options=options) for item, (text, options) in zip(items, translated)]
