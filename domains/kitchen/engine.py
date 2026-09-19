"""Deterministic, facing-aware cooperative two-recipe kitchen.

The controller observes only the current kitchen and revealed orders. One human
command commits one simultaneous turn; animations and questions are not clocks.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random

DOMAIN = "kitchen"
VERSION = "kitchen-v4.0.0"
SCENARIO_VERSION = "kitchen-scenarios-v4.0.0"
WIDTH, HEIGHT = 9, 7
PREPARE_TURNS = CHOP_TURNS = 2
COOK_TURNS = {"egg": 4, "meat": 6, "tomato": 4, "pepper": 4}
MIX_TURNS, BURN_TURNS = 2, 8
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
LABELS = {"egg": ("egg", "鸡蛋"), "tomato": ("tomato", "番茄"), "meat": ("meat", "肉"), "pepper": ("pepper", "辣椒")}
RECIPES = {"egg_tomato": {"protein": "egg", "vegetable": "tomato", "en": "tomato and egg stir-fry", "zh": "番茄炒鸡蛋"},
           "pepper_meat": {"protein": "meat", "vegetable": "pepper", "en": "pepper and meat stir-fry", "zh": "辣椒炒肉"}}
INGREDIENT_RECIPE = {ingredient: recipe for recipe, spec in RECIPES.items() for ingredient in (spec["protein"], spec["vegetable"])}
STATIONS = [
    {"id": "egg", "x": 1, "y": 1, "kind": "ingredient", "ingredient": "egg", "label_en": "Eggs", "label_zh": "鸡蛋柜"},
    {"id": "tomato", "x": 3, "y": 1, "kind": "ingredient", "ingredient": "tomato", "label_en": "Tomatoes", "label_zh": "番茄柜"},
    {"id": "meat", "x": 1, "y": 2, "kind": "ingredient", "ingredient": "meat", "label_en": "Meat", "label_zh": "肉柜"},
    {"id": "pepper", "x": 3, "y": 2, "kind": "ingredient", "ingredient": "pepper", "label_en": "Peppers", "label_zh": "辣椒柜"},
    {"id": "prep", "x": 1, "y": 3, "kind": "prep", "label_en": "Preparation", "label_zh": "备料台"},
    {"id": "human_buffer", "x": 1, "y": 4, "kind": "buffer", "label_en": "Your counter", "label_zh": "你的暂存台"},
    {"id": "plate", "x": 1, "y": 5, "kind": "plate", "label_en": "Serving plates", "label_zh": "正式装盘台"},
    {"id": "serve", "x": 3, "y": 5, "kind": "serve", "label_en": "Serving hatch", "label_zh": "上菜口"},
    {"id": "handoff", "x": 4, "y": 3, "kind": "handoff", "label_en": "Handoff counter", "label_zh": "交接台"},
    {"id": "pot1", "x": 7, "y": 1, "kind": "pot", "label_en": "Stove 1", "label_zh": "炉灶 1"},
    {"id": "protein1", "x": 7, "y": 2, "kind": "protein_buffer", "pot_id": "pot1", "label_en": "Stove 1 temporary plate", "label_zh": "炉灶 1 临时盘位"},
    {"id": "ai_raw", "x": 7, "y": 3, "kind": "raw_buffer", "capacity": 2, "label_en": "Two ingredient slots", "label_zh": "双槽原料暂存台"},
    {"id": "protein2", "x": 7, "y": 4, "kind": "protein_buffer", "pot_id": "pot2", "label_en": "Stove 2 temporary plate", "label_zh": "炉灶 2 临时盘位"},
    {"id": "pot2", "x": 7, "y": 5, "kind": "pot", "label_en": "Stove 2", "label_zh": "炉灶 2"},
]
STATION_BY_ID = {station["id"]: station for station in STATIONS}
STATION_AT = {(station["x"], station["y"]): station for station in STATIONS}
WALLS = [[x, y] for y in range(HEIGHT) for x in range(WIDTH)
         if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1) or (x == 4 and y != 3)]
BLOCKED = frozenset(tuple(p) for p in WALLS) | frozenset(STATION_AT)
FLOOR = frozenset((x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) not in BLOCKED)


def _pos(actor):
    return actor["x"], actor["y"]


def _front(actor):
    dx, dy = MOVES[actor["facing"]]
    return STATION_AT.get((actor["x"] + dx, actor["y"] + dy))


@lru_cache(maxsize=8192)
def _route(position, facing, station_id):
    """Shortest actual action sequence, including a blocked move to face a counter."""
    target = STATION_BY_ID[station_id]
    initial = (position[0], position[1], facing)
    queue, seen = deque([(initial, ())]), {initial}
    while queue:
        (x, y, direction), actions = queue.popleft()
        dx, dy = MOVES[direction]
        if (x + dx, y + dy) == (target["x"], target["y"]):
            return actions
        for action, (dx, dy) in MOVES.items():
            point = (x + dx, y + dy)
            nx, ny = point if point in FLOOR else (x, y)
            nxt = nx, ny, action
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, actions + (action,)))
    raise ValueError("Station is unreachable from this work area")


def _route_end(position, facing, route):
    x, y = position
    for action in route:
        dx, dy = MOVES[action]
        if (x + dx, y + dy) in FLOOR:
            x, y = x + dx, y + dy
        facing = action
    return x, y, facing


def _distance(state, actor, target):
    who = state[actor]
    return len(_route(_pos(who), who["facing"], target))


def _approach(state, actor, target, interaction="interact"):
    who = state[actor]
    route = _route(_pos(who), who["facing"], target)
    return route[0] if route else interaction


def _recipe_name(recipe, language="en"):
    return RECIPES[recipe]["zh" if language == "zh" else "en"]


def _name(item, language="en"):
    if item is None:
        return "nothing" if language == "en" else "无物品"
    zh = language == "zh"
    stage = item["stage"]
    if stage in ("finished", "plated", "mixing"):
        recipe = _recipe_name(item["recipe"], language)
        return ({"finished": f"{recipe} in an output container", "plated": f"{recipe} on a serving plate", "mixing": f"combined {recipe}"} if not zh else
                {"finished": f"出锅容器中的{recipe}", "plated": f"正式餐盘中的{recipe}", "mixing": f"已合入主料的{recipe}"})[stage]
    ingredient = LABELS[item["ingredient"]][zh]
    descriptions = {"raw": (f"raw {ingredient}", f"未备好的{ingredient}"), "prepared": (f"prepared {ingredient}", f"备好的{ingredient}"),
                    "cooked_protein": (f"cooked {ingredient}" + (" on a temporary plate" if item.get("container") == "temporary_plate" else ""), ("临时盘中的" if item.get("container") == "temporary_plate" else "") + f"熟{ingredient}"),
                    "cooked_vegetable": (f"cooked {ingredient}", f"炒熟的{ingredient}")}
    return descriptions[stage][zh]


def _event(state, kind, en, zh, **extra):
    state["events"].append({"type": kind, "en": en, "zh": zh, **extra})


def _pending(state, recipe=None):
    return sorted((order for order in state["orders"] if order["status"] == "pending" and order["deadline"] > state["turn"]
                   and (recipe is None or order["recipe"] == recipe)), key=lambda order: (order["deadline"], order["id"]))


def _order(state, order_id):
    return next((order for order in _pending(state) if order["id"] == order_id), None)


def _all_items(state):
    return [state["human"]["holding"], state["ai"]["holding"], state["handoff"], state["buffers"]["human"],
            *state["buffers"]["ai_raw"], *state["buffers"]["protein"].values(),
            *(pot["item"] for pot in state["pots"] if pot["status"] != "burnt")]


def _contains(item, ingredient):
    return ingredient in item.get("ingredients", [item["ingredient"]])


def _choose_order(state, item):
    if item.get("order_id"):
        return _order(state, item["order_id"])
    for order in _pending(state, item["recipe"]):
        already = any(other and other["id"] != item["id"] and other.get("order_id") == order["id"] and _contains(other, item["ingredient"])
                      for other in _all_items(state))
        if not already:
            return order
    return None


def _useful(state, item):
    return item is not None and _choose_order(state, item) is not None


@lru_cache(maxsize=1)
def _configuration():
    return json.loads((Path(__file__).resolve().parents[2] / "configs" / "study_v3_kitchen.json").read_text())


def _scenario(seed, task):
    config = _configuration()
    saved = config.get("scenarios", {}).get(str(seed), {}).get(str(task))
    if saved:
        return deepcopy(saved)
    rng = random.Random(seed * 71 + task * 10007)
    count, budget = (4, 240) if task == 1 else (6, 360)
    budget = config.get("task_budgets", {}).get(str(task), budget)
    recipes = [rng.choice(list(RECIPES)) for _ in range(count)]
    recipes[:2] = ["egg_tomato", "pepper_meat"]
    if task == 3 and rng.randrange(2):
        recipes[:2] = recipes[:2][::-1]
    arrivals = [0] * count if task != 3 else [0, 0, 32, 64, 96, 128]
    deadlines = ([130, 160, 225, budget] if task == 1 else [130, 170, 245, 280, 345, budget])
    return {"max_turns": budget, "human_start": [2, 3] if task != 3 else list(rng.choice([(2, 3), (2, 2), (2, 4)])),
            "ai_start": [6, 3], "orders": [{"id": f"order{i + 1}", "recipe": recipe, "arrival": arrivals[i], "deadline": deadlines[i]}
                                            for i, recipe in enumerate(recipes)]}


def initial_state(seed: int, task: int):
    if task not in (1, 2, 3):
        raise ValueError("Task must be 1, 2, or 3")
    scene = _scenario(int(seed), task)
    return {"domain": DOMAIN, "version": VERSION, "scenario_version": SCENARIO_VERSION, "seed": int(seed), "task": task,
            "turn": 0, "max_turns": scene["max_turns"], "terminal": False, "termination_reason": None, "events": [], "policy_memory": {},
            "human": {"x": scene["human_start"][0], "y": scene["human_start"][1], "facing": "right", "holding": None},
            "ai": {"x": scene["ai_start"][0], "y": scene["ai_start"][1], "facing": "right", "holding": None},
            "handoff": None, "buffers": {"human": None, "ai_raw": [None, None], "protein": {"pot1": None, "pot2": None}},
            "pots": [{"id": name, "x": STATION_BY_ID[name]["x"], "y": STATION_BY_ID[name]["y"], "status": "empty", "phase": "idle",
                      "recipe": None, "order_id": None, "item": None, "remaining": 0, "ready_age": 0} for name in ("pot1", "pot2")],
            "orders": [dict(order, status="pending", served_turn=None) for order in scene["orders"] if order["arrival"] == 0],
            "_future_orders": [deepcopy(order) for order in scene["orders"] if order["arrival"] > 0], "total_orders": len(scene["orders"]), "next_item_id": 1,
            "metrics": {"completed_orders": 0, "burnt": 0, "expired": 0, "waste": 0, "blocked_waits": 0, "human_waits": 0,
                        "handoff_conflicts": 0, "parallel_cooking_turns": 0, "parallel_recipe_turns": 0, "wrong_order_buffered": 0}}


def _pot(state, pot_id):
    return next(pot for pot in state["pots"] if pot["id"] == pot_id)


def _loadable(state, item):
    if not item or item["stage"] != "prepared" or not _useful(state, item):
        return []
    order = _choose_order(state, item)
    result = []
    spec = RECIPES[item["recipe"]]
    for pot in state["pots"]:
        if pot["status"] != "empty":
            continue
        if (item["ingredient"] == spec["protein"] and pot["phase"] == "idle" and state["buffers"]["protein"][pot["id"]] is None
                and not any(p["order_id"] == order["id"] for p in state["pots"])):
            result.append(pot)
        elif item["ingredient"] == spec["vegetable"] and pot["phase"] == "await_vegetable" and pot["order_id"] == order["id"]:
            result.append(pot)
    return sorted(result, key=lambda pot: (_distance(state, "ai", pot["id"]), pot["id"]))


def _descriptor(state, actor="human", slot=None):
    """Physical front-cell interaction. Optional AI raw-slot selection is audited."""
    who, held = state[actor], state[actor]["holding"]
    station = _front(who)
    if not station:
        return None
    target, kind = station["id"], station["kind"]
    out = {"station": target}
    if target == "handoff" or target == "human_buffer" and actor == "human":
        item = state["handoff"] if target == "handoff" else state["buffers"]["human"]
        if (held is None) == (item is None):
            return None
        return dict(out, kind="take" if held is None else "put")
    if actor == "human":
        if kind == "ingredient" and held is None:
            return dict(out, kind="take_ingredient", ingredient=station["ingredient"])
        if target == "prep" and held and held["stage"] == "raw":
            return dict(out, kind="prepare")
        if target == "plate" and held and held["stage"] == "finished":
            return dict(out, kind="plate")
        if target == "serve" and held and held["stage"] == "plated":
            return dict(out, kind="serve")
        return None
    if target == "ai_raw":
        choices = ([i for i, item in enumerate(state["buffers"]["ai_raw"]) if item is None] if held and held["stage"] in ("raw", "prepared") else
                   [i for i, item in enumerate(state["buffers"]["ai_raw"]) if item is not None] if held is None else [])
        if slot is not None:
            choices = [slot] if slot in choices else []
        return dict(out, kind="take" if held is None else "put", slot=choices[0]) if choices else None
    if kind == "protein_buffer":
        item, pot_id = state["buffers"]["protein"][station["pot_id"]], station["pot_id"]
        if held and item is None and held.get("pot_id") == pot_id and held["stage"] in ("cooked_protein", "finished"):
            return dict(out, kind="put", pot_id=pot_id)
        if held is None and item:
            return dict(out, kind="take", pot_id=pot_id)
        return None
    if kind == "pot":
        pot = _pot(state, target)
        if held is None and (pot["status"] == "burnt" or pot["order_id"] and not _order(state, pot["order_id"])):
            return dict(out, kind="clear_pot")
        if pot in _loadable(state, held):
            return dict(out, kind="load")
        if held is None and pot["status"] == "ready" and pot["phase"] in ("protein", "mix"):
            return dict(out, kind="remove")
        if (pot["status"] == "ready" and pot["phase"] == "vegetable" and held and held["stage"] == "cooked_protein"
                and held.get("was_buffered") and held.get("order_id") == pot["order_id"] and held.get("pot_id") == pot["id"]
                and held["recipe"] == pot["recipe"] and held["ingredient"] == RECIPES[pot["recipe"]]["protein"]):
            return dict(out, kind="combine")
    return None


def legal_actions(state, actor="human"):
    if actor not in ("human", "ai"):
        raise ValueError("Unknown actor")
    if state["terminal"]:
        return []
    result = ["up", "down", "left", "right", "wait"]
    if _descriptor(state, actor):
        result.append("interact")
    if state[actor]["holding"]:
        result.append("discard")
    return result


def interaction_label(state, actor="human", language="en", slot=None):
    descriptor = _descriptor(state, actor, slot)
    zh = language == "zh"
    if not descriptor:
        station, held = _front(state[actor]), state[actor]["holding"]
        if station is None:
            return "请先面向一个工位" if zh else "Face a work station first"
        if actor == "human":
            if station["kind"] == "ingredient":
                return "你的手里已有物品，请先放下" if zh else "Your hands are full; put down your item first"
            if station["id"] == "prep":
                return ("这份原料已经备好" if zh else "This ingredient is already prepared") if held else ("先拿一份原料" if zh else "Pick up an ingredient first")
            if station["id"] == "plate":
                return "只有出锅容器中的成品才能转装正式餐盘" if zh else "Only a finished dish in an output container can be transferred onto a serving plate"
            if station["id"] == "serve":
                return "先将成品转装正式餐盘，才能上菜" if zh else "Transfer a finished dish onto a serving plate before serving"
            if station["id"] in ("handoff", "human_buffer"):
                item = state["handoff"] if station["id"] == "handoff" else state["buffers"]["human"]
                if held and item:
                    return "手中和台上都有物品，不能互换或覆盖" if zh else "Your hand and this counter are both occupied; items cannot be swapped or overwritten"
                return "这个台面是空的" if zh else "This counter is empty"
        return "面前工位现在没有可执行的交互" if zh else "No interaction is currently available at the station in front"
    labels = {"take": ("Pick up", "拿起"), "put": ("Put down", "放下"), "prepare": ("Prepare ingredient", "备料"),
              "plate": ("Transfer onto a serving plate", "转装正式餐盘"), "serve": ("Serve dish", "上菜"), "load": ("Add prepared ingredient", "加入备料"),
              "remove": ("Take cooked food off heat", "出锅"), "combine": ("Return the cooked protein to the pan", "把熟主料倒回锅"), "clear_pot": ("Clear the pan", "清理锅")}
    if descriptor["kind"] in ("put", "take"):
        station = STATION_BY_ID[descriptor["station"]]
        if descriptor["kind"] == "put":
            item = state[actor]["holding"]
        elif descriptor["station"] == "handoff":
            item = state["handoff"]
        elif descriptor["station"] == "human_buffer":
            item = state["buffers"]["human"]
        elif descriptor["station"] == "ai_raw":
            item = state["buffers"]["ai_raw"][descriptor["slot"]]
        else:
            item = state["buffers"]["protein"][descriptor["pot_id"]]
        if zh:
            return ("放下" if descriptor["kind"] == "put" else "取回") + _name(item, "zh") + "（" + station["label_zh"] + "）"
        return ("Put " if descriptor["kind"] == "put" else "Pick up ") + _name(item) + (" on " if descriptor["kind"] == "put" else " from ") + station["label_en"].lower()
    if descriptor["kind"] == "take_ingredient":
        ingredient = LABELS[descriptor["ingredient"]][zh]
        return f"取一份{ingredient}" if zh else f"Take one {ingredient} portion"
    if descriptor["kind"] == "prepare" and state[actor]["holding"]["ingredient"] == "egg":
        return "打散鸡蛋" if zh else "Whisk egg"
    return labels[descriptor["kind"]][zh]


def _burn_delay(pot):
    if pot["status"] == "cooking":
        return pot["remaining"] + BURN_TURNS
    if pot["status"] == "ready":
        return BURN_TURNS - pot["ready_age"]
    return 100000


def _plan(state, target, en, zh, reason="work", slot=None, at_target="interact"):
    action = _approach(state, "ai", target, at_target)
    return {"action": action, "goal": target, "reason_code": reason, "reason_en": en, "reason_zh": zh, "slot": slot}


def _wait(en, zh, reason="waiting"):
    return {"action": "wait", "goal": "wait", "reason_code": reason, "reason_en": en, "reason_zh": zh, "slot": None}


def _holding_plan(state):
    item = state["ai"]["holding"]
    if not _useful(state, item):
        result = _wait("The food in my hand cannot fill a current order. I am discarding it to free my hand.", "手中食物不能完成当前订单，我先丢弃它以腾出手。", "discard_unusable")
        result["action"] = "discard"
        return result
    if item["stage"] == "cooked_protein":
        pid = item["pot_id"]
        pot = _pot(state, pid)
        if not item["was_buffered"]:
            return _plan(state, "protein" + pid[-1], f"I cooked the {LABELS[item['ingredient']][0]} first. I must put its temporary plate on stove {pid[-1]}'s counter before cooking the vegetable in that pan.",
                         f"我先炒熟了{LABELS[item['ingredient']][1]}，现在要把临时盘放到炉灶 {pid[-1]} 的专用台，再用这口锅炒蔬菜。", "store_protein")
        if pot["phase"] == "vegetable" and pot["status"] in ("cooking", "ready"):
            return _plan(state, pid, f"The cooked {LABELS[item['ingredient']][0]} has already rested on its temporary counter. I am returning it to stove {pid[-1]} when the vegetable is ready, then combining them for two turns.",
                         f"熟{LABELS[item['ingredient']][1]}已经在临时台暂存。我把它带回炉灶 {pid[-1]}，等蔬菜炒好后倒回锅中，再合炒两回合。", "return_protein", at_target="interact" if pot["status"] == "ready" else "wait")
        return _plan(state, "protein" + pid[-1], "I am returning this cooked component to its dedicated temporary counter while its matching pan waits for the vegetable.", "对应锅还在等蔬菜，我把熟主料放回专用临时盘位。", "store_protein")
    if item["stage"] == "finished":
        if state["handoff"] is None:
            return _plan(state, "handoff", "The dish is finished in its output container. I am handing it over so you can transfer it to a serving plate and serve it.", "这道菜已在出锅容器中完成。我送到交接台，由你转装正式餐盘后上菜。", "deliver")
        if state["buffers"]["protein"][item["pot_id"]] is None:
            return _plan(state, "protein" + item["pot_id"][-1], "The handoff counter is occupied. I am putting the finished dish on its pan's temporary counter to free my hand.", "交接台被占用，我先把成品放到对应锅的临时台，腾出手。", "store_finished")
        return _wait("I am holding a finished dish and both its temporary counter and the handoff counter are occupied. Clear the handoff counter so I can deliver it.", "我拿着成品，对应临时台和交接台都被占用；清空交接台后我才能交付。", "blocked")
    loads = _loadable(state, item)
    if loads:
        pot = loads[0]
        return _plan(state, pot["id"], f"I am taking the prepared {LABELS[item['ingredient']][0]} to stove {pot['id'][-1]} for the next required cooking stage.",
                     f"我要把备好的{LABELS[item['ingredient']][1]}放进炉灶 {pot['id'][-1]}，进行这道菜的下一道烹饪工序。", "load")
    empty = [i for i, value in enumerate(state["buffers"]["ai_raw"]) if value is None]
    if empty:
        spec = RECIPES[item["recipe"]]
        vegetable = item["ingredient"] == spec["vegetable"]
        return _plan(state, "ai_raw", f"The {LABELS[item['ingredient']][0]} cannot enter its pan yet. " + (f"I cook and temporarily plate {spec['protein']} first, so I am storing this vegetable in a free ingredient slot." if vegetable else "I am storing it in a free ingredient slot until its pan is available."),
                     f"这份{LABELS[item['ingredient']][1]}现在还不能下锅。" + (f"我需要先炒熟并暂存{LABELS[spec['protein']][1]}，所以把蔬菜放到空原料槽。" if vegetable else "我先把它放到空原料槽，等锅可用。"),
                     "store_early_vegetable" if vegetable else "store_ingredient", slot=empty[0])
    return _wait("Both ingredient slots are full, and the ingredient in my hand cannot enter a pan yet. I am waiting for capacity to become available.", "两个原料槽都满了，手中原料现在又不能下锅，我在等待腾出空间。", "blocked")


def _empty_plan(state):
    ready = sorted((pot for pot in state["pots"] if pot["status"] == "ready"), key=lambda p: (_burn_delay(p), p["id"]))
    for pot in ready:
        if pot["phase"] in ("protein", "mix"):
            turns = _distance(state, "ai", pot["id"]) + 1
            lateness = turns > _burn_delay(pot)
            return _plan(state, pot["id"], f"Stove {pot['id'][-1]} has cooked food. Reaching, facing and removing it needs {turns} turns; it will burn in {_burn_delay(pot)} turns." + (" I cannot reach it before burning, but I can clear the pan afterwards." if lateness else " I am taking it off the heat."),
                         f"炉灶 {pot['id'][-1]} 的食物已炒好；到达、转向并出锅需 {turns} 回合，再留 {_burn_delay(pot)} 回合会糊。" + ("已经来不及救下，我先过去，之后清理。" if lateness else "我先把它取出。"), "remove_ready")
        component = state["buffers"]["protein"][pot["id"]]
        if component:
            target = "protein" + pot["id"][-1]
            return _plan(state, target, f"The vegetable on stove {pot['id'][-1]} is ready. I must retrieve its already-cooked protein from the temporary plate and return it to this pan.",
                         f"炉灶 {pot['id'][-1]} 的蔬菜炒好了，我需要从临时盘取回已熟主料，倒回这口锅。", "fetch_protein")
    # Finished food stored under handoff pressure must leave before this pan can
    # start another order; its dedicated counter is never overwritten.
    for pid, item in state["buffers"]["protein"].items():
        if item and (not _useful(state, item) or item["stage"] == "finished" and state["handoff"] is None):
            return _plan(state, "protein" + pid[-1], "I am collecting food from the temporary counter to complete its handoff or clear food that no current order needs.", "我从临时台取回食物，继续交接，或清理当前订单不再需要的食物。", "fetch_output")
    for pot in state["pots"]:
        if pot["status"] == "burnt" or pot["order_id"] and not _order(state, pot["order_id"]):
            return _plan(state, pot["id"], "This pan contains burnt food or belongs to an expired order. I am clearing it so the pan can be used again.", "这口锅的食物已糊，或对应订单已过期。我清理它，使锅可以重新使用。", "clear_pot")
    # Retrieve protein while the vegetable cooks only if it will be ready by
    # the earliest return. The explicit route includes final facing actions.
    for pot in sorted(state["pots"], key=lambda p: (_burn_delay(p), p["id"])):
        component = state["buffers"]["protein"][pot["id"]]
        if pot["phase"] == "vegetable" and pot["status"] == "cooking" and component:
            target = "protein" + pot["id"][-1]
            route = _route(_pos(state["ai"]), state["ai"]["facing"], target)
            x, y, face = _route_end(_pos(state["ai"]), state["ai"]["facing"], route)
            return_time = len(route) + 1 + len(_route((x, y), face, pot["id"])) + 1
            if pot["remaining"] <= return_time:
                return _plan(state, target, f"Stove {pot['id'][-1]}'s vegetable needs {pot['remaining']} more cooking turns. I am fetching its stored cooked protein so I can combine them when ready.",
                             f"炉灶 {pot['id'][-1]} 的蔬菜还需 {pot['remaining']} 回合，我先取回暂存的熟主料，以便炒好后合炒。", "fetch_protein")
    candidates = []
    for source, slot, item in [("handoff", None, state["handoff"]), *(("ai_raw", i, it) for i, it in enumerate(state["buffers"]["ai_raw"]))]:
        if not item:
            continue
        if source == "ai_raw" and not _useful(state, item):
            return _plan(state, source, "I am clearing an ingredient that cannot fill a current order.", "我先清理不能完成当前订单的原料。", "clear_ingredient", slot=slot)
        if item["stage"] != "prepared" or not _useful(state, item):
            continue
        loads = _loadable(state, item)
        if loads:
            order = _choose_order(state, item)
            candidates.append((0 if loads[0]["phase"] == "await_vegetable" else 1, order["deadline"], source, slot, item))
    if candidates:
        _, _, source, slot, item = min(candidates, key=lambda row: row[:3])
        return _plan(state, source, f"I am collecting the prepared {LABELS[item['ingredient']][0]} because a matching pan can use it for its next cooking stage.",
                     f"有对应的锅可以开始下一道工序，我去取备好的{LABELS[item['ingredient']][1]}。", "accept_ingredient", slot=slot)
    item = state["handoff"]
    if item and item["stage"] == "prepared" and _useful(state, item) and None in state["buffers"]["ai_raw"]:
        return _plan(state, "handoff", "I am accepting this prepared ingredient. If its cooking stage cannot start yet, I will put it in a free ingredient slot.", "我先接收这份备料；如果还没轮到它下锅，就放入空原料槽暂存。", "accept_early")
    if item and _handoff_needs_human(state):
        if item["stage"] == "raw":
            return _plan(state, "handoff", f"The handoff counter contains raw {LABELS[item['ingredient']][0]}. Only you can prepare it. Take it back and use the preparation counter twice before returning it.",
                         f"交接台上是未备好的{LABELS[item['ingredient']][1]}，备料只能由你完成。请取回它，在备料台操作两次后再交接。", "await_preparation", at_target="wait")
        return _plan(state, "handoff", "The handoff item cannot be used by a current order. Please take it back and clear or store it so the counter is available.",
                     "交接台上的物品不能用于当前订单，请取回并清理或暂存它，腾出交接台。", "blocked_unusable_handoff", at_target="wait")
    cooking = sorted((p for p in state["pots"] if p["status"] == "cooking"), key=lambda p: (_burn_delay(p), p["id"]))
    if cooking:
        pot = cooking[0]
        return _plan(state, pot["id"], f"Stove {pot['id'][-1]} still needs {pot['remaining']} cooking turns. No other currently available ingredient can start now, so I am waiting near this pan.",
                     f"炉灶 {pot['id'][-1]} 还需 {pot['remaining']} 回合；当前没有其他备料能开始下锅，我先在这口锅附近等待。", "watch_pot", at_target="wait")
    needed = next((RECIPES[p["recipe"]]["vegetable"] for p in state["pots"] if p["phase"] == "await_vegetable"), None)
    reason_en = f"The next missing ingredient for an active pan is prepared {needed}." if needed else "I need prepared protein to start an available recipe; an early vegetable stays in storage."
    reason_zh = f"进行中的锅还缺备好的{LABELS[needed][1]}。" if needed else "开始新菜需要备好的鸡蛋或肉；提前送来的蔬菜先留在暂存台。"
    return _plan(state, "handoff", reason_en + " I am waiting near the handoff counter.", reason_zh + "我在交接台附近等待。", "await_ingredient", at_target="wait")


def decide(state):
    if state["terminal"]:
        plan = _wait("The task has ended.", "任务已结束。", "terminal")
    else:
        plan = _holding_plan(state) if state["ai"]["holding"] else _empty_plan(state)
        # Do not accept more work while a ready protein/final dish is within a
        # reachable rescue window. Free a held ingredient on its real counter,
        # then retain a rescue commitment until that specific pan is emptied.
        ready = [p for p in state["pots"] if p["status"] == "ready" and p["phase"] in ("protein", "mix")]
        commitment = state.get("policy_memory", {}).get("rescue_pot")
        ready.sort(key=lambda p: (p["id"] != commitment, _burn_delay(p), p["id"]))
        held = state["ai"]["holding"]
        if ready and held and held["stage"] in ("prepared", "cooked_protein"):
            pot = ready[0]
            target = "ai_raw" if held["stage"] == "prepared" else "protein" + held["pot_id"][-1]
            empty = [i for i, value in enumerate(state["buffers"]["ai_raw"]) if value is None] if target == "ai_raw" else []
            room = bool(empty) if target == "ai_raw" else state["buffers"]["protein"][held["pot_id"]] is None
            if room:
                plan = _plan(state, target, f"Stove {pot['id'][-1]} has ready food. I must free my hand on this counter before I can remove it.",
                             f"炉灶 {pot['id'][-1]} 的食物已熟，我必须先把手中物品放到这个台面，再去出锅。", "free_for_rescue", slot=empty[0] if empty else None)
                commitment = pot["id"]
        if not ready:
            commitment = None
        if plan["action"] not in legal_actions(state, "ai"):
            raise AssertionError(f"Illegal AI action {plan}: {state['ai']}")
        if plan["action"] == "interact" and _descriptor(state, "ai", plan.get("slot")) is None:
            raise AssertionError("AI selected an unavailable physical interaction")
    memory = {"goal": plan["goal"], "rescue_pot": commitment if not state["terminal"] else None}
    alternatives = []
    if plan["action"] != "wait":
        alternatives.append({"action": "wait", "en": "Waiting would preserve my position and held item for this turn, but every active pan would still advance by one cooking or ready turn.",
                             "zh": "等待会保持我本回合的位置和手持物品，但每口活动的锅仍推进一回合烹饪或待出锅计时。"})
    return dict(plan, memory=memory, alternatives=alternatives, facts=[],
                action_label_en=interaction_label(state, "ai", slot=plan.get("slot")) if plan["action"] == "interact" else action_label(plan["action"]),
                action_label_zh=interaction_label(state, "ai", "zh", slot=plan.get("slot")) if plan["action"] == "interact" else action_label(plan["action"], "zh"))


def _assign(state, item):
    order = _choose_order(state, item)
    if order and item.get("order_id") is None:
        item["order_id"] = order["id"]


def _reset_pot(pot):
    pot.update(status="empty", phase="idle", recipe=None, order_id=None, item=None, remaining=0, ready_age=0)


def _interact(state, actor, descriptor, new_turn, newly_loaded):
    who = state[actor]
    held, target, kind = who["holding"], descriptor["station"], descriptor["kind"]
    if kind == "take_ingredient":
        ingredient = descriptor["ingredient"]
        iid = f"item{state['next_item_id']}"
        state["next_item_id"] += 1
        who["holding"] = {"id": iid, "ingredient": ingredient, "ingredients": [ingredient], "recipe": INGREDIENT_RECIPE[ingredient], "stage": "raw",
                          "prepare_progress": 0, "components": [iid], "order_id": None, "pot_id": None, "container": None, "was_buffered": False}
        _event(state, "ingredient_taken", f"You took one {ingredient} portion.", f"你取了一份{LABELS[ingredient][1]}。", actor=actor, item=deepcopy(who["holding"]))
    elif kind == "prepare":
        held["prepare_progress"] += 1
        if held["prepare_progress"] >= PREPARE_TURNS:
            held["stage"] = "prepared"
        verb = "Whisking" if held["ingredient"] == "egg" else "Chopping"
        _event(state, "prepared", f"{verb}: {held['prepare_progress']} of {PREPARE_TURNS} preparation turns.",
               f"{'打散' if held['ingredient'] == 'egg' else '切配'}进度：{held['prepare_progress']}/{PREPARE_TURNS}。", actor=actor, item=deepcopy(held))
    elif kind in ("put", "take"):
        if target == "handoff":
            old = state["handoff"]
            state["handoff"] = held
        elif target == "human_buffer":
            old = state["buffers"]["human"]
            state["buffers"]["human"] = held
        elif target == "ai_raw":
            old = state["buffers"]["ai_raw"][descriptor["slot"]]
            state["buffers"]["ai_raw"][descriptor["slot"]] = held
            if held and held["ingredient"] == RECIPES[held["recipe"]]["vegetable"] and not _loadable(state, held):
                state["metrics"]["wrong_order_buffered"] += 1
        else:
            pid = descriptor["pot_id"]
            old = state["buffers"]["protein"][pid]
            state["buffers"]["protein"][pid] = held
            if held and held["stage"] == "cooked_protein":
                held["was_buffered"] = True
                pot = _pot(state, pid)
                if pot["phase"] == "await_protein_store":
                    pot["phase"] = "await_vegetable"
        who["holding"] = old
        if old and actor == "ai":
            _assign(state, old)
        item = held if held else old
        _event(state, "item_placed" if held else "item_taken", f"{'You' if actor == 'human' else 'AI'} {'put' if held else 'picked up'} {_name(item)} {'on' if held else 'from'} {STATION_BY_ID[target]['label_en'].lower()}.",
               f"{'你' if actor == 'human' else 'AI'}{'放下' if held else '取回'}了{_name(item, 'zh')}（{STATION_BY_ID[target]['label_zh']}）。", actor=actor, station=target, slot=descriptor.get("slot"), item=deepcopy(item))
    elif kind == "load":
        pot = _pot(state, target)
        _assign(state, held)
        protein = held["ingredient"] == RECIPES[held["recipe"]]["protein"]
        pot.update(status="cooking", phase="protein" if protein else "vegetable", recipe=held["recipe"], order_id=held["order_id"], item=held,
                   remaining=COOK_TURNS[held["ingredient"]], ready_age=0)
        held["pot_id"] = pot["id"]
        who["holding"] = None
        newly_loaded.add(pot["id"])
        _event(state, "pot_loaded", f"AI started cooking {held['ingredient']} on stove {pot['id'][-1]}; {pot['remaining']} full cooking turns are required.",
               f"AI 在炉灶 {pot['id'][-1]} 开始炒{LABELS[held['ingredient']][1]}，需要完整烹饪 {pot['remaining']} 回合。", pot=target, phase=pot["phase"], item=deepcopy(held))
    elif kind == "remove":
        pot = _pot(state, target)
        who["holding"] = pot["item"]
        if pot["phase"] == "protein":
            who["holding"].update(stage="cooked_protein", container="temporary_plate", was_buffered=False)
            pot.update(status="empty", phase="await_protein_store", item=None, remaining=0, ready_age=0)
        else:
            who["holding"].update(stage="finished", container="output_container")
            _reset_pot(pot)
        _event(state, "pot_removed", f"AI took {_name(who['holding'])} off stove {target[-1]}.",
               f"AI 从炉灶 {target[-1]} 取出了{_name(who['holding'], 'zh')}。", pot=target, item=deepcopy(who["holding"]))
    elif kind == "combine":
        pot = _pot(state, target)
        vegetable = pot["item"]
        held.update(stage="mixing", container=None, ingredients=[RECIPES[held["recipe"]]["protein"], RECIPES[held["recipe"]]["vegetable"]],
                    components=held["components"] + vegetable["components"])
        pot.update(status="cooking", phase="mix", item=held, remaining=MIX_TURNS, ready_age=0)
        who["holding"] = None
        newly_loaded.add(target)
        _event(state, "components_combined", f"AI returned the cooked protein to stove {target[-1]} with its cooked vegetable. The dish now needs two full mixing turns.",
               f"AI 把已熟主料倒回炉灶 {target[-1]}，与炒熟的蔬菜合炒，还需完整 2 回合。", pot=target, item=deepcopy(held))
    elif kind == "clear_pot":
        pot = _pot(state, target)
        item = deepcopy(pot["item"])
        if item:
            state["metrics"]["waste"] += len(item["components"])
        protein = state["buffers"]["protein"][target]
        if pot["phase"] == "vegetable" and protein and _order(state, pot["order_id"]):
            pot.update(status="empty", phase="await_vegetable", item=None, remaining=0, ready_age=0)
        else:
            _reset_pot(pot)
        _event(state, "pot_cleared", f"AI cleared stove {target[-1]}.", f"AI 清理了炉灶 {target[-1]}。", pot=target, item=item)
    elif kind == "plate":
        held.update(stage="plated", container="serving_plate")
        _event(state, "plated", "You transferred the finished dish from its output container onto a serving plate.", "你把成品从出锅容器转装到了正式餐盘。", actor=actor, item=deepcopy(held))
    elif kind == "serve":
        matches = sorted((order for order in state["orders"] if order["status"] == "pending" and order["recipe"] == held["recipe"] and order["id"] == held.get("order_id") and order["deadline"] >= new_turn), key=lambda o: (o["deadline"], o["id"]))
        if matches and len(held["components"]) == 2 and set(held["ingredients"]) == {RECIPES[held["recipe"]]["protein"], RECIPES[held["recipe"]]["vegetable"]}:
            order = matches[0]
            order.update(status="completed", served_turn=new_turn)
            who["holding"] = None
            state["metrics"]["completed_orders"] += 1
            _event(state, "served", f"You served {_recipe_name(held['recipe'])} for {order['id']}.", f"你为订单 {order['id']} 上了{_recipe_name(held['recipe'], 'zh')}。", order_id=order["id"], item=deepcopy(held))
        else:
            _event(state, "serve_rejected", "No current order accepts this plated dish. You keep holding it.", "当前没有可接受这份餐盘的订单；物品仍留在你手中。", item=deepcopy(held))


def step(state, human_action, decision=None):
    if state["terminal"]:
        raise ValueError("This task has ended")
    if human_action not in legal_actions(state):
        raise ValueError("That action is not available in front of you")
    decision = deepcopy(decision) if decision is not None else decide(state)
    if decision["action"] not in legal_actions(state, "ai"):
        raise ValueError("AI action is not legal in the saved state")
    actions = {"human": human_action, "ai": decision["action"]}
    descriptors = {actor: _descriptor(state, actor, decision.get("slot") if actor == "ai" else None) if action == "interact" else None
                   for actor, action in actions.items()}
    if any(actions[actor] == "interact" and descriptors[actor] is None for actor in actions):
        raise ValueError("Selected interaction is not available in front")
    nxt = deepcopy(state)
    nxt["events"] = []
    nxt["policy_memory"] = deepcopy(decision.get("memory", {}))
    new_turn, newly_loaded = state["turn"] + 1, set()
    conflict = all(descriptors[actor] and descriptors[actor]["station"] == "handoff" for actor in actions)
    for actor, action in actions.items():
        who = nxt[actor]
        if action in MOVES:
            dx, dy = MOVES[action]
            who["facing"] = action
            if (who["x"] + dx, who["y"] + dy) in FLOOR:
                who["x"], who["y"] = who["x"] + dx, who["y"] + dy
        elif action == "wait":
            if actor == "human":
                nxt["metrics"]["human_waits"] += 1
            elif decision.get("reason_code") == "blocked":
                nxt["metrics"]["blocked_waits"] += 1
        elif action == "discard":
            item = who["holding"]
            nxt["metrics"]["waste"] += len(item["components"])
            who["holding"] = None
            _event(nxt, "waste", f"{'You' if actor == 'human' else 'AI'} discarded {_name(item)}.", f"{'你' if actor == 'human' else 'AI'}丢弃了{_name(item, 'zh')}。", actor=actor, item=deepcopy(item))
        elif not conflict:
            _interact(nxt, actor, descriptors[actor], new_turn, newly_loaded)
    if conflict:
        nxt["metrics"]["handoff_conflicts"] += 1
        _event(nxt, "handoff_conflict", "Both teammates used the handoff counter at once. Neither item moved; one turn passed.", "双方同时使用交接台，物品都未转移，但推进了一回合。")
    for pot in nxt["pots"]:
        if pot["id"] in newly_loaded:
            continue
        if pot["status"] == "cooking":
            pot["remaining"] -= 1
            if pot["remaining"] == 0:
                pot["status"], pot["ready_age"] = "ready", 0
                if pot["phase"] == "vegetable":
                    pot["item"]["stage"] = "cooked_vegetable"
                elif pot["phase"] == "protein":
                    pot["item"]["stage"] = "cooked_protein"
                _event(nxt, "pot_ready", f"Stove {pot['id'][-1]}'s {pot['phase']} stage is ready. It burns after eight more full turns unless removed or correctly combined.",
                       f"炉灶 {pot['id'][-1]} 这一阶段炒好了；再留满 8 回合，且未出锅或按菜谱合炒，就会烧糊。", pot=pot["id"], phase=pot["phase"])
        elif pot["status"] == "ready":
            pot["ready_age"] += 1
            if pot["ready_age"] >= BURN_TURNS:
                pot["status"] = "burnt"
                nxt["metrics"]["burnt"] += 1
                _event(nxt, "pot_burnt", f"Food on stove {pot['id'][-1]} burnt after eight full ready turns.", f"炉灶 {pot['id'][-1]} 的食物炒好后留满 8 回合，烧糊了。", pot=pot["id"], item=deepcopy(pot["item"]))
    nxt["turn"] = new_turn
    for order in nxt["orders"]:
        if order["status"] == "pending" and order["deadline"] <= new_turn:
            order["status"] = "expired"
            nxt["metrics"]["expired"] += 1
            _event(nxt, "order_expired", f"{order['id']} expired after this turn's serving opportunity.", f"订单 {order['id']} 在本回合上菜机会结束后过期。", order_id=order["id"])
    future = []
    for order in nxt["_future_orders"]:
        if order["arrival"] <= new_turn:
            nxt["orders"].append(dict(order, status="pending", served_turn=None))
            _event(nxt, "order_arrived", f"New order: {_recipe_name(order['recipe'])}, due on turn {order['deadline']}.", f"新订单：{_recipe_name(order['recipe'], 'zh')}，截止回合 {order['deadline']}。", order=deepcopy(order))
        else:
            future.append(order)
    nxt["_future_orders"] = future
    if all(p["status"] == "cooking" for p in nxt["pots"]):
        nxt["metrics"]["parallel_cooking_turns"] += 1
    if all(p["phase"] != "idle" for p in nxt["pots"]):
        nxt["metrics"]["parallel_recipe_turns"] += 1
    if new_turn >= nxt["max_turns"]:
        nxt["terminal"], nxt["termination_reason"] = True, "turn_budget"
    elif not future and all(o["status"] != "pending" for o in nxt["orders"]):
        nxt["terminal"], nxt["termination_reason"] = True, "all_orders_resolved"
    return nxt


def score(state):
    return {"task_score": 100.0 * state["metrics"]["completed_orders"] / state["total_orders"], "raw_score": state["metrics"]["completed_orders"],
            "metrics": dict(deepcopy(state["metrics"]), total_orders=state["total_orders"], elapsed_turns=state["turn"])}


def public_state(state):
    pots = [dict(deepcopy(pot), ingredient=pot["item"]["ingredient"] if pot["item"] else None,
                 burn_in=_burn_delay(pot) if pot["status"] in ("cooking", "ready") else None) for pot in state["pots"]]
    return {"domain": DOMAIN, "version": VERSION, "task": state["task"], "turn": state["turn"], "max_turns": state["max_turns"],
            "terminal": state["terminal"], "events": deepcopy(state["events"]), "score": score(state), "width": WIDTH, "height": HEIGHT,
            "walls": deepcopy(WALLS), "stations": deepcopy(STATIONS), "human": deepcopy(state["human"]), "ai": deepcopy(state["ai"]),
            "pots": pots, "handoff": deepcopy(state["handoff"]), "buffers": deepcopy(state["buffers"]),
            "orders": [dict(deepcopy(order), remaining=max(0, order["deadline"] - state["turn"])) for order in state["orders"]],
            "total_orders": state["total_orders"], "recipes": deepcopy(RECIPES),
            "interaction": {"available": _descriptor(state) is not None, "station": (_front(state["human"]) or {}).get("id"),
                            "label_en": interaction_label(state), "label_zh": interaction_label(state, language="zh")}}


def _needed(state):
    """Allocate every existing portion once, then list missing visible ingredients."""
    assigned = {(item["order_id"], ingredient) for item in _all_items(state) if item and item.get("order_id")
                for ingredient in item.get("ingredients", [item["ingredient"]])}
    free = {ingredient: 0 for ingredient in LABELS}
    for item in _all_items(state):
        if item and item.get("order_id") is None:
            for ingredient in item.get("ingredients", [item["ingredient"]]):
                free[ingredient] += 1
    needed = []
    for order in _pending(state):
        spec = RECIPES[order["recipe"]]
        for kind in ("protein", "vegetable"):
            ingredient = spec[kind]
            if (order["id"], ingredient) in assigned:
                continue
            if free[ingredient]:
                free[ingredient] -= 1
            else:
                active = any(p["order_id"] == order["id"] for p in state["pots"])
                needed.append((order, ingredient, kind, active))
    return needed


def _handoff_needs_human(state):
    item = state["handoff"]
    return bool(item and (item["stage"] not in ("prepared", "finished", "plated") or not _useful(state, item)))


def _next_missing_input(state):
    needs = _needed(state)
    if not needs:
        return None
    active_count = sum(p["phase"] != "idle" for p in state["pots"])
    first_pending = min(int(order["id"][5:]) for order in _pending(state))
    batch_end = ((first_pending - 1) // 2 + 1) * 2
    open_proteins = [row for row in needs if row[2] == "protein" and not row[3] and int(row[0]["id"][5:]) <= batch_end] if active_count < 2 else []
    return min(open_proteins or needs, key=lambda row: (not row[3], row[0]["deadline"], row[2] != "protein"))


def _next_input_portion(state):
    """Visible-state ingredient goal, distinct from an immediate movement."""
    if state["terminal"]:
        return None
    sources = [("handoff counter", "交接台", state["handoff"] if _handoff_needs_human(state) else None),
               ("your hand", "你手中", state["human"]["holding"]),
               ("your storage counter", "你的暂存台", state["buffers"]["human"])]
    for en, zh, item in sources:
        if item and item["stage"] in ("raw", "prepared") and _useful(state, item):
            order = _choose_order(state, item)
            return {"ingredient": item["ingredient"], "recipe": item["recipe"], "order_id": order["id"],
                    "source_en": en, "source_zh": zh, "prepare_remaining": max(0, PREPARE_TURNS - item["prepare_progress"]), "existing": True}
    needed = _next_missing_input(state)
    if needed:
        order, ingredient, _, _ = needed
        return {"ingredient": ingredient, "recipe": order["recipe"], "order_id": order["id"],
                "source_en": f"the {ingredient} cupboard", "source_zh": LABELS[ingredient][1] + "柜", "prepare_remaining": 2, "existing": False}
    return None


def human_advisor(state):
    """Developer partner chooses human actions only; no undisclosed order access."""
    if state["terminal"]:
        return "wait"
    held, handoff, buffer = state["human"]["holding"], state["handoff"], state["buffers"]["human"]
    ai = decide(state)
    output = any(item and item["stage"] == "finished" for item in (state["ai"]["holding"], *state["buffers"]["protein"].values())) or any(p["phase"] == "mix" for p in state["pots"])
    if held:
        if not _useful(state, held):
            return "discard"
        if held["stage"] == "plated":
            return _approach(state, "human", "serve")
        if held["stage"] == "finished":
            return _approach(state, "human", "plate")
        if held["stage"] == "raw":
            return _approach(state, "human", "prep")
        if _handoff_needs_human(state):
            # With hand, storage and handoff all occupied, one explicit discard
            # is the only legal way to free a hand; never overwrite another item.
            return _approach(state, "human", "human_buffer") if buffer is None else "discard"
        if handoff is None and not output:
            return _approach(state, "human", "handoff")
        if buffer is None:
            return _approach(state, "human", "human_buffer")
        if handoff is None:
            return _approach(state, "human", "handoff")
        return "wait"
    if _handoff_needs_human(state):
        return _approach(state, "human", "handoff")
    if handoff and handoff["stage"] in ("finished", "plated"):
        return _approach(state, "human", "handoff")
    if buffer and buffer["stage"] in ("finished", "plated"):
        return _approach(state, "human", "human_buffer")
    if output:
        # Recover a blocking input only when the AI is not simultaneously taking
        # it; place it on the human counter, then collect the finished output.
        if handoff and ai["action"] != "interact" and buffer is None:
            return _approach(state, "human", "handoff")
        return _approach(state, "human", "handoff", "wait")
    if buffer and handoff is None:
        return _approach(state, "human", "human_buffer")
    needed = _next_missing_input(state)
    if needed and buffer is None:
        _, ingredient, _, _ = needed
        return _approach(state, "human", ingredient)
    return _approach(state, "human", "handoff", "wait")


def action_label(action, language="en"):
    pairs = {"up": ("Move or face up", "向上移动或转向"), "down": ("Move or face down", "向下移动或转向"),
             "left": ("Move or face left", "向左移动或转向"), "right": ("Move or face right", "向右移动或转向"),
             "wait": ("Wait one turn", "等待一回合"), "interact": ("Interact in front", "与面前工位交互"), "discard": ("Discard held item", "丢弃手持物品")}
    return pairs[action][language == "zh"]


def rules(language="en"):
    pairs = [
        ("You collect and prepare ingredients, transfer finished food to a serving plate, and serve. Your AI teammate operates the pans. Neither role can finish an order alone.", "你取料、备料、将成品转装正式餐盘并上菜；AI 操作炉灶。任何一方都不能独立完成订单。"),
        ("WASD moves one square and changes facing. A blocked move turns in place and still uses one turn. E interacts only with the station directly in front; Space waits one turn. Both teammates act together. No key press means no game time passes.", "WASD 移动一格并改变朝向；方向被挡时原地转向，仍消耗一回合。E 只操作正前方工位，空格等待一回合。双方同步行动；不操作就不推进游戏时间。"),
        ("The four cupboards supply egg, tomato, meat and pepper separately. Prepare each portion twice at the preparation counter: whisk egg and chop the other ingredients.", "四个独立原料柜分别提供鸡蛋、番茄、肉和辣椒。每份都要在备料台操作两次：鸡蛋打散，其余切配。"),
        ("There are two recipes: tomato and egg stir-fry, and pepper and meat stir-fry. Cook the protein first (egg: 4 turns; meat: 6), remove it onto a temporary plate and place that plate on its pan's dedicated counter. Then cook the vegetable for 4 turns, return the cooked protein to that pan, and mix for 2 turns.", "两道菜为番茄炒鸡蛋和辣椒炒肉：先炒主料（鸡蛋 4 回合、肉 6 回合），出锅到临时盘并实际放到对应锅的专用台；再炒蔬菜 4 回合，倒回已熟主料，合炒 2 回合。"),
        ("A finished dish comes from the AI in an output container. Take it to the serving-plate counter before serving. Temporary plates, output containers and final serving plates are distinct and supplied without depletion.", "AI 用出锅容器交付成品。你拿走后必须到正式装盘台转装餐盘才能上菜。临时盘、出锅容器和正式餐盘彼此不同，无限供应。"),
        ("Each teammate holds one item. The handoff and human counter each hold one item. AI has two prepared-ingredient slots and one dedicated temporary-plate slot per pan. No swaps or overwrites occur. A held item can be discarded for one turn.", "每人只能拿一件物品；交接台和人类暂存台各一件。AI 有两个备料槽，以及每口锅一个专用临时盘位。不能交换或覆盖台面物品，手持物品可花一回合丢弃。"),
        ("Food ready in a pan burns after eight additional full turns. Its ready turn is not counted. Loading and combining do not count their own turn as a cooking turn. Removed food no longer burns.", "锅中食物炒好后再留满 8 回合会糊，刚炒好当回合不算。下锅或开始合炒当回合不计烹饪时间。出锅食物不再烧糊。"),
        ("If both teammates use the handoff counter in the same turn, neither transfer succeeds. Food placed this turn cannot be taken by the other teammate until a later turn.", "双方同回合使用交接台，两次交互都失败。本回合放下的食物不能被另一方同回合取走。"),
        ("Each dish stays bound to the order it was cooked for. Score is 100 times correct, on-time orders divided by the fixed order count. Serving on an order's deadline turn is accepted before that order expires. Burnt food and waste add no hidden score penalties. Questions and replay do not use turns.", "每份菜始终绑定开工时对应的订单。得分为按时正确完成订单数除以固定订单总数再乘 100。截止回合仍可上菜，之后才过期。烧糊和浪费没有额外隐藏扣分；提问和回放不消耗回合。"),
    ]
    return [pair[language == "zh"] for pair in pairs]


def _physical_food_locations(state):
    """All visible food, including burnt contents; never consult future orders."""
    result = []
    for actor, en, zh in (("human", "your hand", "你手中"), ("ai", "my hand", "我手中")):
        who = state[actor]
        if who["holding"]:
            result.append((who["holding"], f"{en} at column {who['x']}, row {who['y']}", f"{zh}（第 {who['x']} 列、第 {who['y']} 行）"))
    for item, en, zh in [(state["handoff"], "the handoff counter at (4, 3)", "交接台（4,3）"),
                         (state["buffers"]["human"], "your storage counter at (1, 4)", "你的暂存台（1,4）")]:
        if item:
            result.append((item, en, zh))
    for i, item in enumerate(state["buffers"]["ai_raw"]):
        if item:
            result.append((item, f"ingredient slot {i + 1} at (7, 3)", f"原料槽 {i + 1}（7,3）"))
    for pid, item in state["buffers"]["protein"].items():
        if item:
            station = STATION_BY_ID["protein" + pid[-1]]
            result.append((item, f"stove {pid[-1]}'s temporary plate counter at ({station['x']}, {station['y']})", f"炉灶 {pid[-1]} 的临时盘位（{station['x']},{station['y']}）"))
    for pot in state["pots"]:
        if pot["item"]:
            result.append((pot["item"], f"stove {pot['id'][-1]}'s pan at ({pot['x']}, {pot['y']})" + (" (burnt)" if pot["status"] == "burnt" else ""),
                           f"炉灶 {pot['id'][-1]} 锅内（{pot['x']},{pot['y']}）" + ("（已糊）" if pot["status"] == "burnt" else "")))
    return result


def facts(state, decision=None):
    decision = decide(state) if decision is None else decision
    rows = [{"id": "ai_reason", "en": decision["reason_en"], "zh": decision["reason_zh"]},
            {"id": "ai_next_action", "en": "The task has ended; there is no executable next action." if state["terminal"] else "My next action is to " + decision.get("action_label_en", action_label(decision["action"])).lower() + ".", "zh": "任务已结束，没有可执行的下一步动作。" if state["terminal"] else "我下一步会" + decision.get("action_label_zh", action_label(decision["action"], "zh")) + "。"},
            {"id": "current_turn", "en": f"Task {state['task']}, turn {state['turn']}; {state['max_turns'] - state['turn']} turns remain.", "zh": f"Task {state['task']}，回合 {state['turn']}，剩余 {state['max_turns'] - state['turn']} 回合。"}]
    for actor in ("human", "ai"):
        who = state[actor]
        rows.append({"id": actor + "_holding", "en": f"{'You are' if actor == 'human' else 'I am'} at column {who['x']}, row {who['y']}, facing {who['facing']}, holding {_name(who['holding'])}.",
                     "zh": f"{'你' if actor == 'human' else '我'}在第 {who['x']} 列、第 {who['y']} 行，朝{ {'up':'上','down':'下','left':'左','right':'右'}[who['facing']] }，手里是{_name(who['holding'], 'zh')}。"})
    for name, item in [("handoff", state["handoff"]), ("human_buffer", state["buffers"]["human"]), *((f"raw_slot{i + 1}", it) for i, it in enumerate(state["buffers"]["ai_raw"])), *((f"{pid}_temporary_plate", it) for pid, it in state["buffers"]["protein"].items())]:
        rows.append({"id": name, "en": f"{name.replace('_', ' ')} holds {_name(item)}.", "zh": f"{ {'handoff':'交接台','human_buffer':'你的暂存台','raw_slot1':'原料槽 1','raw_slot2':'原料槽 2','pot1_temporary_plate':'炉灶 1 临时盘位','pot2_temporary_plate':'炉灶 2 临时盘位'}[name] }上是{_name(item, 'zh')}。"})
    for pot in state["pots"]:
        phase_en = {"idle": "no active recipe", "protein": "first protein cooking", "await_protein_store": "cooked protein must be put on its temporary counter", "await_vegetable": "waiting for the prepared vegetable", "vegetable": "vegetable cooking", "mix": "final mixing"}[pot["phase"]]
        phase_zh = {"idle": "没有进行中的菜", "protein": "先炒主料", "await_protein_store": "熟主料需要放到临时台", "await_vegetable": "等待蔬菜备料", "vegetable": "炒蔬菜", "mix": "最后合炒"}[pot["phase"]]
        rows.append({"id": pot["id"], "en": f"Stove {pot['id'][-1]}: {phase_en}; status {pot['status']}; {pot['remaining']} cooking turns remain." + (f" The food burns in {_burn_delay(pot)} turns if not handled." if pot["status"] in ("cooking", "ready") else ""),
                     "zh": f"炉灶 {pot['id'][-1]}：{phase_zh}，状态为{ {'empty':'空锅','cooking':'烹饪中','ready':'此阶段已熟','burnt':'已烧糊'}[pot['status']] }，还需烹饪 {pot['remaining']} 回合。" + (f"若不处理，{_burn_delay(pot)} 回合后烧糊。" if pot["status"] in ("cooking", "ready") else "")})
        turns = _distance(state, "ai", pot["id"]) + 1
        rows.append({"id": pot["id"] + "_distance", "en": f"Reaching, facing and interacting with stove {pot['id'][-1]} needs {turns} turns from my current position, before any extra hand-freeing or cooking wait.",
                     "zh": f"从我当前状态到炉灶 {pot['id'][-1]}，包括移动、朝向和一次交互，需要 {turns} 回合；额外腾手或等待食物炒熟不包含在内。"})
    for pot in state["pots"]:
        content = _name(pot["item"])
        content_zh = _name(pot["item"], "zh")
        if pot["status"] == "burnt":
            content = "burnt food (" + content + ")"
            content_zh = "已烧糊的食物（" + content_zh + "）"
        if pot["item"]:
            constituents_en = ", ".join(LABELS[name][0] for name in pot["item"]["ingredients"])
            constituents_zh = "、".join(LABELS[name][1] for name in pot["item"]["ingredients"])
            ids = ", ".join(pot["item"]["components"])
            content += f"; ingredients: {constituents_en}; original portions: {ids}"
            content_zh += f"；原料组成：{constituents_zh}；原始份号：{ids}"
        job_en = f" for {pot['order_id']} ({_recipe_name(pot['recipe'])})" if pot["recipe"] else " with no assigned recipe"
        job_zh = f"，对应 {pot['order_id']}（{_recipe_name(pot['recipe'], 'zh')}）" if pot["recipe"] else "，尚未分配菜谱"
        rows.append({"id": pot["id"] + "_contents", "en": f"Stove {pot['id'][-1]}'s pan at ({pot['x']}, {pot['y']}) holds {content}{job_en}.",
                     "zh": f"炉灶 {pot['id'][-1]} 锅内（{pot['x']},{pot['y']}）是{content_zh}{job_zh}。"})
    locations = _physical_food_locations(state)
    for ingredient, (en_name, zh_name) in LABELS.items():
        located = [(item, en, zh) for item, en, zh in locations if ingredient in item["ingredients"]]
        if located:
            en_parts = [f"{_name(item)} is in/on {en}" + (f" for {item['order_id']}" if item.get("order_id") else " (not yet assigned to an order)") for item, en, zh in located]
            zh_parts = [f"{_name(item, 'zh')}在{zh}" + (f"，对应 {item['order_id']}" if item.get("order_id") else "，尚未绑定订单") for item, en, zh in located]
            text_en = f"Current {en_name} portions: " + "; ".join(en_parts) + "."
            text_zh = f"当前的{zh_name}：" + "；".join(zh_parts) + "。"
        else:
            text_en, text_zh = f"No collected {en_name} portion is currently in a hand, counter or pan. The cupboard still supplies it.", f"当前手中、台面或锅内没有已取出的{zh_name}；原料柜仍可取料。"
        rows.append({"id": "ingredient_location_" + ingredient, "en": text_en, "zh": text_zh})
    missing = _needed(state)
    rows.append({"id": "missing_ingredients", "en": "Portions not yet present anywhere for currently visible orders: " + ("; ".join(f"{order['id']}: {ingredient}" for order, ingredient, _, _ in missing) if missing else "none") + ". Raw portions already collected still need preparation; this does not include future orders.",
                 "zh": "当前已显示订单中，厨房还没有对应份料的原料：" + ("；".join(f"{order['id']}：{LABELS[ingredient][1]}" for order, ingredient, _, _ in missing) if missing else "无") + "。已取出的生料仍需备料；这里不包含未来订单。"})
    next_input = _next_input_portion(state)
    if next_input:
        name_en, name_zh = LABELS[next_input["ingredient"]]
        todo_en = f"complete {next_input['prepare_remaining']} preparation interactions" if next_input["prepare_remaining"] else "it is already prepared"
        todo_zh = f"还要备料 {next_input['prepare_remaining']} 次" if next_input["prepare_remaining"] else "它已经备好"
        rows.append({"id": "next_input_ingredient", "en": f"The next input portion to work on is {name_en} for {next_input['order_id']} ({_recipe_name(next_input['recipe'])}), from {next_input['source_en']}; {todo_en}. Hand it over when the shared counter is free, giving a finished output room to pass first. This is an ingredient goal, separate from your immediate next movement.",
                     "zh": f"下一份应处理的输入原料是{next_input['source_zh']}的{name_zh}，用于 {next_input['order_id']}（{_recipe_name(next_input['recipe'], 'zh')}）；{todo_zh}。等交接台有空位再递交，先给待交付成品留出通道。这是原料目标，与眼前下一步移动不同。"})
    else:
        rows.append({"id": "next_input_ingredient", "en": "The task has ended; there is no next ingredient to supply." if state["terminal"] else "All ingredient portions required by currently visible orders are already present. Do not collect an extra portion now; attend to preparation, handoff, plating or serving already in progress.",
                     "zh": "任务已结束，无需再递交原料。" if state["terminal"] else "当前可见订单所需的原料份都已在厨房中，无需再拿新原料；请处理进行中的备料、交接、装盘或上菜。"})
    if _handoff_needs_human(state):
        handoff = state["handoff"]
        rows.append({"id": "handoff_recovery", "en": "The handoff contains " + _name(handoff) + ". " + ("Take it back, prepare it at your station, then return the prepared portion." if handoff["stage"] == "raw" and _useful(state, handoff) else "No current order can use it in this form; take it back and clear or store it.") + " If holding something, use your empty storage counter first; if both your hand and storage are full, an explicit discard can free a hand without overwriting food.",
                     "zh": "交接台上是" + _name(handoff, "zh") + "。" + ("请取回，到备料台备好后重新交接。" if handoff["stage"] == "raw" and _useful(state, handoff) else "当前订单无法使用这份物品，请取回清理或暂存。") + "若手中有物品，先放到空暂存台；若手和暂存台都满，可明确丢弃手持物品腾手，不覆盖其他食物。"})
    for order in state["orders"]:
        rows.append({"id": order["id"], "en": f"{order['id']} needs {_recipe_name(order['recipe'])}, due on turn {order['deadline']}; status {order['status']}.",
                     "zh": f"订单 {order['id']} 需要{_recipe_name(order['recipe'], 'zh')}，截止回合 {order['deadline']}，状态为{ {'pending':'待完成','completed':'已完成','expired':'已过期'}[order['status']] }。"})
    if not state["terminal"]:
        suggestion = human_advisor(state)
        rows.append({"id": "human_available_option", "en": "One available next action is: " + action_label(suggestion).lower() + ". It is a suggestion, not a guarantee of the final score.", "zh": "你下一步可选择" + action_label(suggestion, "zh") + "。这是一项建议，不保证最终分数。"})
        rows.append({"id": "front_interaction", "en": "Your front-cell interaction is: " + interaction_label(state) + ".", "zh": "你面前当前可做的交互是：" + interaction_label(state, language="zh") + "。"})
    for i, (en, zh) in enumerate(zip(rules(), rules("zh"))):
        rows.append({"id": f"public_rule{i}", "en": en, "zh": zh})
    for i, event in enumerate(state["events"]):
        rows.append({"id": f"event{i}", "en": event["en"], "zh": event["zh"]})
    for i, alt in enumerate(decision["alternatives"]):
        rows.append({"id": f"alternative{i}", "en": alt["en"], "zh": alt["zh"]})
    return rows


def demonstration():
    state = initial_state(1000, 1)
    frames, captions = [public_state(state)], [{"index": 0, "en": "WASD moves and faces; E uses only the station directly in front. Four cupboards supply four separate ingredients.", "zh": "WASD 移动并转向；E 只操作正前方工位。四个独立原料柜各提供一种原料。"}]
    wanted = [("prepared", "Prepare each portion twice: whisk eggs or chop the other ingredients.", "每份原料备料两次：打散鸡蛋或切配其他原料。"),
              ("item_placed", "The cooked protein rests on its pan's temporary plate counter while the vegetable uses that same pan.", "熟主料实际放到对应锅的临时盘位，再用同一口锅炒蔬菜。"),
              ("components_combined", "Return that cooked protein to the cooked vegetable, then combine for two full turns.", "把已熟主料倒回炒好的蔬菜，合炒完整两回合。"),
              ("plated", "The AI's output container is different from your final serving plate. Transfer the finished dish here before serving.", "AI 的出锅容器与正式餐盘不同。你需要在这里把成品转装正式餐盘，再上菜。")]
    seen = set()
    for _ in range(state["max_turns"]):
        if state["terminal"]:
            break
        state = step(state, human_advisor(state))
        frames.append(public_state(state))
        for kind, en, zh in wanted:
            matching = [event for event in state["events"] if event["type"] == kind]
            if kind == "item_placed":
                matching = [event for event in matching if event.get("station", "").startswith("protein") and event.get("item", {}).get("stage") == "cooked_protein"]
            if matching and kind not in seen:
                captions.append({"index": len(frames) - 1, "en": en, "zh": zh})
                seen.add(kind)
        if any(event["type"] == "served" for event in state["events"]):
            break
    captions.append({"index": len(frames) - 1, "en": "A correct dish served on time completes an order. You are ready to coordinate both recipes with your teammate.", "zh": "按时上菜完成订单。现在你可以与队友配合制作两道菜。"})
    return {"frames": frames, "captions": captions}


def comprehension(language="en"):
    rows = [
        {"id": "kitchen_predict", "text": "You supplied prepared tomato before egg. The AI holds the tomato, both pans are idle, and an ingredient slot is empty. What will it do?", "options": ["Store the tomato while waiting for prepared egg", "Serve tomato alone", "Skip the egg cooking stage"], "answer": 0},
        {"id": "kitchen_reason", "text": "Cooked meat is on stove 2's temporary plate counter. The pepper in stove 2 is ready. Why does the AI fetch that meat?", "options": ["To discard a finished order", "To return it to the pepper and finish mixing", "To give you raw meat to chop"], "answer": 1},
        {"id": "kitchen_change", "text": "You collect a finished dish in the AI's output container. What is still necessary before serving?", "options": ["Transfer it onto a formal serving plate", "Cook its protein again", "Give it back to the ingredient cupboard"], "answer": 0},
    ]
    if language == "zh":
        translations = [
            ("你先交了备好的番茄，还没交鸡蛋。AI 拿着番茄，两口锅空闲，原料槽有空位。它会做什么？", ["暂存番茄，等待备好的鸡蛋", "只上番茄", "跳过炒鸡蛋工序"]),
            ("熟肉在炉灶 2 的临时盘位，炉灶 2 的辣椒已炒好。AI 为什么取回肉？", ["丢弃已完成订单", "把肉倒回辣椒中完成合炒", "把生肉交给你切"]),
            ("你拿到了 AI 出锅容器中的成品，上菜前还需要做什么？", ["转装正式餐盘", "重新炒一次主料", "放回原料柜"]),
        ]
        rows = [dict(row, text=text, options=options) for row, (text, options) in zip(rows, translations)]
    return rows
