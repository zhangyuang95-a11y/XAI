"""Pure, deterministic simultaneous-turn warehouse; no learned runtime needed.

The controller receives only the pre-action state.  Its return is evidence of
the actual choice, not a retrospective account of an unrelated actor.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random

DOMAIN = "warehouse"
VERSION = "warehouse-turnbased-v3.2"
CONFIG = json.loads((Path(__file__).resolve().parents[2] / "configs/study_v3_warehouse.json").read_text())
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "wait": (0, 0)}
MOVE_NAMES = {"up": ("up", "向上"), "down": ("down", "向下"), "left": ("left", "向左"), "right": ("right", "向右"), "wait": ("wait", "等待")}
WIDTH, HEIGHT = 9, 7
CHARGER = (4, 2)
WALLS = frozenset({(x, y) for x in range(WIDTH) for y in range(HEIGHT)
                   if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1)}
                  | {(4, 1), (4, 4), (4, 5)})
FLOOR = frozenset((x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) not in WALLS)


def _pos(actor):
    return actor["x"], actor["y"]


def _target(actor, action):
    dx, dy = MOVES[action]
    return actor["x"] + dx, actor["y"] + dy


@lru_cache(maxsize=4096)
def _distance(start, target):
    """Four-neighbour traversable path distance, never Manhattan distance."""
    if start == target:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        point, distance = queue.popleft()
        for dx, dy in tuple(MOVES.values())[:4]:
            neighbour = point[0] + dx, point[1] + dy
            if neighbour not in FLOOR or neighbour in seen:
                continue
            if neighbour == target:
                return distance + 1
            seen.add(neighbour)
            queue.append((neighbour, distance + 1))
    return 999


def _order(state, actor):
    carrying = state[actor]["carrying"]
    if carrying:
        return next(item for item in state["orders"] if item["id"] == carrying)
    return next((item for item in state["orders"] if item["owner"] == actor and item["status"] == "available"), None)


def _mission(state, actor):
    item = _order(state, actor)
    if item is None:
        return None, None
    return tuple(item["dropoff"] if state[actor]["carrying"] else item["pickup"]), item


def _energy_required(state, actor, start=None):
    point = _pos(state[actor]) if start is None else start
    target, item = _mission(state, actor)
    if item is None:
        return 0
    distance = _distance(point, target)
    if not state[actor]["carrying"]:
        distance += _distance(tuple(item["pickup"]), tuple(item["dropoff"]))
    distance += _distance(tuple(item["dropoff"]), CHARGER)
    return 3 * (distance + CONFIG["reserve_moves"])


def _goal(state, actor):
    agent = state[actor]
    target, item = _mission(state, actor)
    if item is None:
        # Finished actors leave the shared crossing/charger and park away.
        other = "ai" if actor == "human" else "human"
        remaining = [tuple(item["pickup"]) for item in state["orders"]
                     if item["owner"] == other and item["status"] == "available"]
        corners = [(1, 1), (1, 5)] if agent["x"] < 4 else [(7, 1), (7, 5)]
        home = max(corners, key=lambda point: (min((_distance(point, endpoint) for endpoint in remaining), default=10),
                                               -_distance(_pos(agent), point)))
        return home, "park", None
    required = _energy_required(state, actor)
    if agent["battery"] < required:
        return CHARGER, "charge", item
    return target, "deliver" if agent["carrying"] else "pickup", item


def initial_state(seed: int, task: int) -> dict:
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2, or 3")
    rng = random.Random(int(seed) * 73 + task * 10007)
    # Public finite missions; nothing is scheduled secretly after reset.
    left = [(1, 1), (2, 1), (1, 4), (2, 5)]
    right = [(7, 1), (6, 1), (7, 4), (6, 5)]
    rng.shuffle(left)
    rng.shuffle(right)
    orders = []
    for actor in ("human", "ai"):
        source, destination = (left, right) if actor == "human" else (right, left)
        for index in range(3):
            # Task 1 retains one same-room order, then two shared crossings.
            drop = source[(index + 1) % 4] if task == 1 and index == 0 else destination[(index + (task == 3)) % 4]
            orders.append({"id": f"{'H' if actor == 'human' else 'A'}{index + 1}",
                           "owner": actor, "pickup": list(source[index]),
                           "dropoff": list(drop), "status": "available"})
    return {"domain": DOMAIN, "version": VERSION, "scenario_version": CONFIG["scenario_version"],
            "task": task, "turn": 0, "max_turns": CONFIG["max_turns"], "terminal": False,
            "seed": int(seed), "events": [], "policy_memory": {},
            "human": {"x": 2, "y": 3 if task != 3 else 4, "battery": CONFIG["task_start_battery"][str(task)]["human"], "carrying": None, "shutdown": False},
            "ai": {"x": 6, "y": 3 if task != 3 else 4, "battery": CONFIG["task_start_battery"][str(task)]["ai"], "carrying": None, "shutdown": False},
            "orders": orders, "collision_events": 0, "shutdown_events": 0,
            "charge_events": 0, "deliveries": 0, "last_actions": None}


def legal_actions(state, actor="human") -> list[str]:
    if actor not in ("human", "ai"):
        raise ValueError("unknown actor")
    if state["terminal"]:
        return []
    agent = state[actor]
    return [action for action in MOVES if action == "wait" or
            (agent["battery"] >= 3 and _target(agent, action) in FLOOR)]


def _collision(human, ai, htarget, atarget):
    # Entering the partner's occupied cell is prohibited, even if it moves;
    # this public conservative physical rule makes a vacated-cell handoff two turns.
    return (htarget == atarget or (htarget == ai and htarget != human)
            or (atarget == human and atarget != ai))


def _candidate(state, action, goal):
    agent = state["ai"]
    target = _target(agent, action)
    remaining = agent["battery"] - (3 if action != "wait" else 0)
    risky = [human_action for human_action in legal_actions(state) if _collision(
        _pos(state["human"]), _pos(agent), _target(state["human"], human_action), target)]
    stranded = action != "wait" and remaining < 3 * _distance(target, CHARGER)
    return {"action": action, "target": target, "risk_count": len(risky), "risky_actions": risky,
            "stranded": bool(stranded), "distance": _distance(target, goal),
            "distance_before": _distance(_pos(agent), goal), "remaining_battery": remaining}


def _crossing_commitment(state, goal):
    agent, human = state["ai"], state["human"]
    return bool(agent["carrying"] and 3 <= agent["x"] <= 5 and agent["y"] in (2, 3)
                and ((agent["x"] >= 4 and goal[0] < 4) or (agent["x"] <= 4 and goal[0] > 4))
                # A person who cannot afford a two-step clearance detour and
                # still return to charge gets the ordinary active-yield rule.
                and human["battery"] >= 3 * (_distance(_pos(human), CHARGER) + 2))


def decide(state) -> dict:
    if state["terminal"]:
        raise ValueError("task is complete")
    agent, human = state["ai"], state["human"]
    goal, mode, order = _goal(state, "ai")
    old_memory = state.get("policy_memory", {})
    candidates = [_candidate(state, action, goal) for action in legal_actions(state, "ai")]
    required = _energy_required(state, "ai")
    charging = mode == "charge" and _pos(agent) == CHARGER
    crossing_commitment = _crossing_commitment(state, goal)
    # Stabilize a short clearance plan only while the person has not passed.
    hold_yield = bool(old_memory.get("mode") == "yield" and old_memory.get("yield_remaining", 0) > 0
                      and _distance(_pos(human), tuple(old_memory.get("yield_origin", CHARGER))) <= 2
                      and _pos(agent) == tuple(old_memory.get("yield_target", ())))
    human_goal, human_mode, _ = _goal(state, "human")
    blocks_human_route = (_distance(_pos(human), _pos(agent)) == 1
                         and _distance(_pos(human), human_goal)
                         == 1 + _distance(_pos(agent), human_goal))
    obstructs = (_pos(agent) == CHARGER and _pos(human) != CHARGER
                 and human_mode == "charge" and not charging)
    before = _distance(_pos(agent), goal)
    for item in candidates:
        # Power safety is first. Then minimize all permitted-human-action
        # conflicts, without requiring an impossible universally safe action.
        item["rank"] = (int(item["remaining_battery"] <= 0 and item["target"] != CHARGER),
                        int(item["stranded"]), item["risk_count"],
                        int(charging and item["action"] != "wait"),
                        int(crossing_commitment and item["distance"] >= before and item["action"] != "wait"),
                        int(hold_yield and item["action"] != "wait"),
                        int(obstructs and item["action"] == "wait"),
                        int(blocks_human_route and not charging and item["action"] == "wait"),
                        int(item["distance"] > before), item["distance"],
                        int(item["action"] == "wait"), list(MOVES).index(item["action"]))
    chosen = min(candidates, key=lambda item: item["rank"])
    action = chosen["action"]
    is_yield = action != "wait" and chosen["distance"] >= before
    wait_progress = action == "wait" and mode != "park" and not charging
    if charging and action == "wait":
        code = "charge_to_safe_departure"
        en = f"I am charging. I have {agent['battery']}% battery and need {required}% to finish my current delivery, return to the charger, and keep a two-move reserve."
        zh = f"我正在充电。目前电量为{agent['battery']}%，完成当前配送、返回充电位置并保留两步余量需要{required}%。"
    elif is_yield:
        code = "clear_shared_route"
        en = (f"I will move {MOVE_NAMES[action][0]} to clear a cell on your route." if blocks_human_route
              else f"I will move {MOVE_NAMES[action][0]} to reduce the chance of a collision, even though this does not shorten my route.")
        zh = (f"我将{MOVE_NAMES[action][1]}腾出你路线上的格子。" if blocks_human_route
              else f"我将{MOVE_NAMES[action][1]}减少可能的碰撞，尽管这一步不会缩短我的路线。")
        en += " Enter my old cell only after I have left it; simultaneous entry would cancel both moves."
        zh += "请在我离开后再进入原来的位置；同时进入会使双方移动取消。"
    elif crossing_commitment and action == "wait":
        code = "committed_crossing_wait"
        en = f"I am carrying {agent['carrying']} and keeping my route through the shared crossing. I will continue once the entrance is clear enough for a safe move. You can make space beside the entrance while I wait. If your battery cannot cover a short detour and return, I recheck whether I can make room instead."
        zh = f"我正携带{agent['carrying']}，并保持穿过共享通道的路线。入口有足够空间可以安全前进时，我会继续。你可以在我等待时退到入口旁腾出空间。如果你的电量不足以短暂绕行后返回充电位置，我会重新检查自己能否让开。"
    elif hold_yield and action == "wait":
        code = "hold_clearance"
        en = "I am keeping the space clear while you pass. I will check the route again after your next move."
        zh = "我暂时保持通道畅通，让你通过。你下一步行动后，我会重新检查路线。"
    elif action == "wait" and before == 0 and mode in ("pickup", "deliver"):
        code = "complete_at_current_cell"
        en = "I am already at the current parcel's pickup or drop-off cell. Waiting will complete that interaction when this turn is confirmed."
        zh = "我已位于当前包裹的取货或送货格。确认本回合后，等待也会完成这次取送交互。"
    elif wait_progress:
        code = "wait_for_safe_route"
        en = "I am waiting because moving closer to my goal currently has a greater collision risk or would leave too little battery to return. I will check again after the next turn."
        zh = "我正在等待，因为目前接近目标的移动会有更高的碰撞风险，或使返程电量不足。下一回合后我会重新检查。"
    elif mode == "charge":
        code = "return_to_charge"
        en = f"I am heading to the shared charger. My {agent['battery']}% battery is below the {required}% needed for my current delivery and a safe return."
        zh = f"我正在前往共享充电位置。目前电量为{agent['battery']}%，低于完成当前配送并安全返回所需的{required}%。"
    elif _pos(agent) == CHARGER and action != "wait":
        code = "leave_charger"
        en = f"I have enough battery for my current work, so I will leave the charger now by moving {MOVE_NAMES[action][0]}."
        zh = f"我的电量已足够完成当前任务，因此现在{MOVE_NAMES[action][1]}离开充电位置。"
    elif mode == "park":
        code = "finished_park"
        en = "My three deliveries are finished. I will stay away from the shared crossing and charger when there is a safe route."
        zh = "我的三项配送已完成。如果路线安全，我会避开共享通道和充电位置。"
    else:
        code = "continue_delivery" if mode == "deliver" else "continue_pickup"
        purpose_en, purpose_zh = ("delivery", "送货") if mode == "deliver" else ("pickup", "取货")
        en = f"I will move {MOVE_NAMES[action][0]} toward {order['id']}'s {purpose_en} point. The remaining route changes from {before} to {chosen['distance']} moves."
        zh = f"我将{MOVE_NAMES[action][1]}前往{order['id']}的{purpose_zh}位置。剩余路线从{before}步变为{chosen['distance']}步。"
    if chosen["risk_count"]:
        en += f" No lower-risk energy-safe choice is available: {chosen['risk_count']} of your currently legal actions could still cause a collision."
        zh += f" 当前没有风险更低且电量安全的选择：你现在的合法动作中，仍有{chosen['risk_count']}种可能造成碰撞。"
    memory = {"mode": "crossing" if crossing_commitment else "yield" if is_yield or hold_yield else mode,
              "goal": list(goal), "order": order["id"] if order else None,
              "crossing_side": "left" if crossing_commitment and goal[0] < 4 else "right" if crossing_commitment else None,
              "wait_count": old_memory.get("wait_count", 0) + 1 if action == "wait" else 0,
              "yield_remaining": 2 if is_yield else max(0, old_memory.get("yield_remaining", 0) - 1),
              "yield_origin": list(_pos(agent)) if is_yield else old_memory.get("yield_origin", list(_pos(agent))),
              "yield_target": list(chosen["target"]) if is_yield else old_memory.get("yield_target", list(_pos(agent)))}
    alternatives = []
    for item in candidates:
        if item["action"] == action:
            continue
        verb, verb_zh = MOVE_NAMES[item["action"]]
        alternatives.append({"action": item["action"],
            "en": f"If I choose {verb}, my route to the current goal would be {item['distance']} moves; {item['risk_count']} of your legal actions could collide; my battery after movement would be {item['remaining_battery']}%.",
            "zh": f"如果我选择{verb_zh}，到当前目标的路线为{item['distance']}步；你有{item['risk_count']}种合法动作可能与我碰撞；移动后电量为{item['remaining_battery']}%。"})
    return {"action": action, "reason_code": code, "reason_en": en, "reason_zh": zh,
            "goal": f"{mode}:{goal[0]},{goal[1]}", "memory": memory,
            "alternatives": alternatives, "facts": [
                {"id": "ai_goal_distance", "en": f"The AI's current route target is ({goal[0]}, {goal[1]}), {before} moves away along open cells.",
                 "zh": f"AI当前路线目标为（{goal[0]}，{goal[1]}），沿可通行格距离为{before}步。"},
                {"id": "ai_risk", "en": f"The selected move conflicts with {chosen['risk_count']} of the human's {len(legal_actions(state))} legal actions; the AI does not know which action the human will submit.",
                 "zh": f"所选动作可能与你的{len(legal_actions(state))}种合法动作中的{chosen['risk_count']}种冲突；AI不知道你将提交哪一种动作。"}]}


def _event(kind, en, zh, **extra):
    return {"type": kind, "en": en, "zh": zh, **extra}


def step(state, human_action, decision=None) -> dict:
    if state["terminal"]:
        raise ValueError("task is complete")
    if human_action not in legal_actions(state):
        raise ValueError("illegal human action")
    actual = decide(state)
    if decision is not None and decision != actual:
        raise ValueError("decision does not match this pre-action state")
    next_state = deepcopy(state)
    next_state["events"] = events = []
    next_state["turn"] += 1
    next_state["policy_memory"] = deepcopy(actual["memory"])
    actions = {"human": human_action, "ai": actual["action"]}
    positions = {actor: _pos(state[actor]) for actor in actions}
    targets = {actor: _target(state[actor], action) for actor, action in actions.items()}
    collided = _collision(positions["human"], positions["ai"], targets["human"], targets["ai"])
    if collided:
        targets = dict(positions)
        next_state["collision_events"] += 1
        events.append(_event("collision", "Collision: both moves were cancelled; one collision event was recorded.", "发生碰撞：双方移动均取消，记录一次碰撞事件。"))
    for actor, action in actions.items():
        agent = next_state[actor]
        name, name_zh = ("You", "你") if actor == "human" else ("AI", "AI")
        point = targets[actor]
        if point != positions[actor]:
            agent["battery"] -= 3
        agent["x"], agent["y"] = point
        if action == "wait" and point == CHARGER:
            charged = min(10, 100 - agent["battery"])
            agent["battery"] += charged
            if charged:
                next_state["charge_events"] += 1
                events.append(_event("charge", f"{name} charged {charged}%; battery is {agent['battery']}%.", f"{name_zh}充电{charged}%，当前电量{agent['battery']}%。", actor=actor, amount=charged))
        if agent["battery"] < 3 and point != CHARGER:
            if not agent["shutdown"]:
                next_state["shutdown_events"] += 1
                events.append(_event("shutdown", f"{name} is away from the charger with too little battery to move ({agent['battery']}%).", f"{name_zh}在充电格之外电量不足以移动（{agent['battery']}%）。", actor=actor))
            agent["shutdown"] = True
        else:
            agent["shutdown"] = False
        target, item = _mission(next_state, actor)
        if item is not None and point == target:
            if agent["carrying"]:
                item["status"] = "delivered"
                agent["carrying"] = None
                next_state["deliveries"] += 1
                events.append(_event("delivery", f"{name} delivered {item['id']}.", f"{name_zh}完成了{item['id']}的配送。", actor=actor, order_id=item["id"]))
            else:
                item["status"] = "carried"
                agent["carrying"] = item["id"]
                events.append(_event("pickup", f"{name} collected {item['id']}.", f"{name_zh}取走了{item['id']}。", actor=actor, order_id=item["id"]))
    next_state["last_actions"] = actions
    next_state["terminal"] = next_state["turn"] >= next_state["max_turns"] or next_state["deliveries"] == 6
    if next_state["terminal"]:
        events.append(_event("complete", f"Task ended with {next_state['deliveries']} of 6 deliveries completed.", f"任务结束，共完成6项配送中的{next_state['deliveries']}项。"))
    return next_state


def score(state) -> dict:
    delivered = sum(item["status"] == "delivered" for item in state["orders"])
    return {"task_score": round(100 * delivered / 6, 6),
            "raw_score": 100 * delivered - 200 * state["collision_events"] - 50 * state["shutdown_events"] - state["turn"],
            "metrics": {"deliveries": delivered, "total_deliveries": 6,
                        "collision_events": state["collision_events"], "shutdown_events": state["shutdown_events"],
                        "charge_events": state["charge_events"], "elapsed_turns": state["turn"]}}


def public_state(state) -> dict:
    public = {key: deepcopy(state[key]) for key in ("domain", "version", "task", "turn", "max_turns", "terminal", "events", "orders")}
    for actor in ("human", "ai"):
        public[actor] = {key: deepcopy(state[actor][key]) for key in ("x", "y", "battery", "carrying")}
    public.update(width=WIDTH, height=HEIGHT, walls=[list(point) for point in sorted(WALLS)],
                  chargers=[{"x": CHARGER[0], "y": CHARGER[1]}],
                  stations=[{"id": "charger", "x": CHARGER[0], "y": CHARGER[1], "kind": "charger", "label_en": "Shared charger", "label_zh": "共享充电位置"}],
                  score=score(state))
    return public


def rules(language="en") -> list[str]:
    english = ["Complete your three labelled deliveries with the AI's three deliveries within 120 turns. Each completed delivery contributes one sixth of the 100-point Task score.",
               "Choose up, down, left, right, or wait; then confirm one turn. Both robots act from the same starting state. Questions and replay do not advance time.",
               "Your deliveries are H1, H2, H3; the AI's are A1, A2, A3. Each robot handles its orders in that order. Reaching the current pickup or drop-off automatically collects or delivers its parcel. Each robot carries one parcel.",
               "Walls cannot be entered. Entering the other robot's occupied cell, swapping positions, or choosing the same destination cancels both moves and counts as one collision event.",
               "Each successful move costs 3% battery. Waiting on the shared charger restores up to 10%, capped at 100%. Waiting elsewhere and cancelled moves use no battery. A robot with less than 3% cannot move; dropping below 3% away from the charger records one shutdown event.",
               "The optional net score is 100 per delivery, minus 200 per collision event, 50 per new shutdown event, and one per elapsed turn. It does not change Task score."]
    chinese = ["你有三项带编号的配送，AI也有三项，需要在120回合内共同完成。每完成一项配送，获得满分100分的六分之一。",
               "选择上、下、左、右或等待，然后确认一回合。双方从同一行动前状态同时执行动作。提问和回放不会推进时间。",
               "你的配送编号为H1、H2、H3，AI的为A1、A2、A3。各自按编号顺序完成。到达当前取货或送货格会自动取货或交货；每台机器人只能携带一个包裹。",
               "墙壁不能进入。进入同伴当前占据的位置、交换位置或同时前往同一格，会取消双方移动，并记录一次碰撞事件。",
               "每次成功移动耗电3%。在共享充电格等待可恢复最多10%，上限100%。其他位置等待及取消的移动不耗电。电量不足3%时不能移动；在充电格之外电量降到3%以下会记录一次断电。",
               "辅助净分为每项配送100分，减去每次碰撞200分、每次新发生的断电50分，以及已用回合数；它不影响任务完成度分数。"]
    return chinese if language.startswith("zh") else english


def facts(state, decision=None) -> list[dict]:
    result = [{"id": "time", "en": f"This is turn {state['turn']} of {state['max_turns']}; {state['max_turns']-state['turn']} turns remain.",
               "zh": f"当前为第{state['turn']}回合，总预算{state['max_turns']}回合，还剩{state['max_turns']-state['turn']}回合。"},
              {"id": "score", "en": f"The team has delivered {state['deliveries']} of 6 parcels; Task score is {score(state)['task_score']}.",
               "zh": f"团队已完成6项配送中的{state['deliveries']}项；任务得分为{score(state)['task_score']}。"}]
    result.append({"id": "event_totals", "en": f"There have been {state['collision_events']} collision events, {state['shutdown_events']} new shutdown events and {state['charge_events']} charging events. Net score is {score(state)['raw_score']}.",
                   "zh": f"累计发生{state['collision_events']}次碰撞、{state['shutdown_events']}次新断电及{state['charge_events']}次充电；辅助净分为{score(state)['raw_score']}。"})
    result.extend({"id": f"public_rule_{index}", "en": en, "zh": zh}
                  for index, (en, zh) in enumerate(zip(rules("en"), rules("zh"))))
    result.extend({"id": f"last_event_{index}", "en": event["en"], "zh": event["zh"]}
                  for index, event in enumerate(state["events"]))
    for order in state["orders"]:
        result.append({"id": "order_" + order["id"],
            "en": f"{order['id']} belongs to {order['owner']}: pickup ({order['pickup'][0]}, {order['pickup'][1]}), drop-off ({order['dropoff'][0]}, {order['dropoff'][1]}), status {order['status']}.",
            "zh": f"{order['id']}属于{'人' if order['owner']=='human' else 'AI'}：取货位置（{order['pickup'][0]}，{order['pickup'][1]}），送货位置（{order['dropoff'][0]}，{order['dropoff'][1]}），状态为{ {'available':'待取货','carried':'运送中','delivered':'已送达'}[order['status']]}。"})
    for actor in ("human", "ai"):
        agent = state[actor]
        target, order = _mission(state, actor)
        result.append({"id": f"{actor}_state", "en": f"{'Human' if actor == 'human' else 'AI'} is at ({agent['x']}, {agent['y']}), has {agent['battery']}% battery, and carries {agent['carrying'] or 'no parcel'}.",
                       "zh": f"{'人' if actor == 'human' else 'AI'}位于（{agent['x']}，{agent['y']}），电量{agent['battery']}%，携带{agent['carrying'] or '无包裹'}。"})
        result.append({"id": f"{actor}_charger_distance", "en": f"The open-cell route from {actor} to the charger is {_distance(_pos(agent), CHARGER)} moves, requiring {3*_distance(_pos(agent), CHARGER)}% battery before any charging.",
                       "zh": f"{'人' if actor == 'human' else 'AI'}沿可通行格到充电位置为{_distance(_pos(agent), CHARGER)}步，到达前需要消耗{3*_distance(_pos(agent), CHARGER)}%电量。"})
        energy = 3 * _distance(_pos(agent), CHARGER)
        result.append({"id": f"{actor}_charger_feasibility", "en": f"The {actor}'s {agent['battery']}% battery is {'enough' if agent['battery'] >= energy else 'not enough'} for the {energy}% cost of the shortest currently open route to the charger, provided the partner does not force a detour.",
                       "zh": f"{'人' if actor == 'human' else 'AI'}的{agent['battery']}%电量{'足够' if agent['battery'] >= energy else '不足以'}支付目前最短可通行充电路线的{energy}%消耗，前提是同伴没有迫使它绕行。"})
    required = _energy_required(state, "ai")
    missing = max(0, required - state["ai"]["battery"])
    if _pos(state["ai"]) == CHARGER:
        waits = (missing + 9) // 10
        result.append({"id": "ai_charge_shortfall", "en": f"The AI needs {missing}% more battery to meet its current {required}% departure requirement. That takes {waits} charging turns at up to 10% per turn if it remains on the charger; asking questions does not perform those turns.",
                       "zh": f"AI距离当前{required}%的离开电量要求还差{missing}%。如果保持在充电格，每回合最多充10%，需要{waits}个充电回合；提问不会执行这些回合。"})
    if not state["terminal"]:
        actual = decide(state)
        if decision is not None and decision != actual:
            raise ValueError("decision does not match this pre-action state")
        decision = actual
        result.append({"id": "actual_decision", "en": decision["reason_en"], "zh": decision["reason_zh"]})
        result.extend(deepcopy(decision.get("facts", [])))
        result.extend({"id": "alternative_" + item["action"], "en": item["en"], "zh": item["zh"]} for item in decision["alternatives"])
    return result


def human_advisor(state) -> str:
    """Human-only coordination proxy; it never changes the fixed AI decision."""
    if state["terminal"]:
        return "wait"
    decision = decide(state)
    human, ai = state["human"], state["ai"]
    goal, mode, _ = _goal(state, "human")
    charging = (_pos(human) == CHARGER and _order(state, "human") is not None
                and human["battery"] < min(100, _energy_required(state, "human") + 18))
    if charging:
        goal = CHARGER
    # Give the occupied charger space; return when its occupant has left.
    if mode == "charge" and _pos(ai) == CHARGER:
        goal = (2, 3) if human["x"] <= 4 else (6, 3)
    ai_goal, ai_mode, _ = _goal(state, "ai")
    if _crossing_commitment(state, ai_goal) and decision["action"] == "wait":
        goal = (2 if ai_goal[0] < 4 else 6, 1 if ai_goal[1] >= 3 else 5)
    # Clear the complete entrance (including the cell a person could enter
    # next), rather than stopping one step back where robust AI still waits.
    ai_crossing = (ai["y"] == 3 and 3 <= ai["x"] <= 5) or _pos(ai) == CHARGER
    same_side_as_ai_destination = ((human["x"] < 4 and ai_goal[0] < 4)
                                   or (human["x"] > 4 and ai_goal[0] > 4))
    if ai_crossing and same_side_as_ai_destination and _distance(_pos(human), (4, 3)) <= 4:
        staging = (2 if human["x"] < 4 else 6, 1 if ai_goal[1] >= 3 else 5)
        if human["battery"] >= 3 * (_distance(_pos(human), staging) + _distance(staging, CHARGER)) + 6:
            goal = staging
    atarget = _target(ai, decision["action"])
    candidates = []
    for action in legal_actions(state):
        target = _target(human, action)
        remaining = human["battery"] - (3 if action != "wait" else 0)
        collision = _collision(_pos(human), _pos(ai), target, atarget)
        stranded = action != "wait" and remaining < 3 * _distance(target, CHARGER)
        rank = (int(remaining <= 0 and target != CHARGER), int(stranded), int(collision),
                int(charging and action != "wait"), _distance(target, goal), int(action == "wait"), list(MOVES).index(action))
        candidates.append((rank, action))
    return min(candidates)[1]


def screening_report() -> dict:
    """Reproducible development agents, never participant evidence or UI help.

    Public greedy knows collision geometry and its own energy requirements.
    The history agent reacts to two unchanged position pairs with three
    retreat/exploration moves, without consulting the AI's decision or memory.
    """
    report = {"domain": DOMAIN, "version": VERSION, "task": 2, "not_human_results": True, "sets": {}}
    for split, seeds in (("development", CONFIG["development_seeds"]), ("heldout", CONFIG["heldout_seeds"])):
        report["sets"][split] = {}
        for name in ("random", "public_greedy", "public_history_learning", "informed_human_planner"):
            scores, turns = [], []
            for seed in seeds:
                state = initial_state(seed, 2)
                previous_positions, stuck, escape = None, 0, 0
                rng = random.Random(seed)
                while not state["terminal"]:
                    human, ai = state["human"], state["ai"]
                    positions = _pos(human), _pos(ai)
                    stuck = stuck + 1 if positions == previous_positions else 0
                    previous_positions = positions
                    if stuck >= 2:
                        escape, stuck = 3, 0
                    goal, mode, _ = _goal(state, "human")
                    if name == "random":
                        action = rng.choice(legal_actions(state))
                    elif name == "informed_human_planner":
                        action = human_advisor(state)
                    elif name == "public_history_learning" and escape:
                        action = min(legal_actions(state), key=lambda choice: (
                            human["battery"] - (choice != "wait") * 3 < 3 * _distance(_target(human, choice), CHARGER),
                            _target(human, choice) == _pos(ai), choice == "wait",
                            -_distance(_target(human, choice), _pos(ai)), _distance(_target(human, choice), goal)))
                        escape -= 1
                    elif mode == "charge" and _pos(human) == CHARGER:
                        action = "wait"
                    else:
                        action = min(legal_actions(state), key=lambda choice: (
                            _target(human, choice) == _pos(ai), choice != "wait" and any(e["type"] == "collision" for e in state["events"]),
                            _distance(_target(human, choice), goal), choice == "wait"))
                    state = step(state, action)
                scores.append(score(state)["task_score"])
                turns.append(state["turn"])
            report["sets"][split][name] = {"scenarios": len(seeds), "mean_task_score": round(sum(scores) / len(scores), 6),
                "completed_all": scores.count(100), "maximum_turns": max(turns), "scores": scores}
    return report


def demonstration() -> dict:
    state = initial_state(17, 1)
    frames = [public_state(state)]
    captions = [{"index": 0, "en": "You control the blue robot. Complete three H deliveries while your AI teammate completes three A deliveries.", "zh": "你控制蓝色机器人，完成三项H配送；AI队友完成三项A配送。"}]
    first_pickup = first_charge = first_delivery = first_failure = False
    while not state["terminal"]:
        action = human_advisor(state)
        if not first_failure and _distance(_pos(state["human"]), _pos(state["ai"])) == 1:
            action = next(item for item in legal_actions(state) if _target(state["human"], item) == _pos(state["ai"]))
            first_failure = True
        state = step(state, action)
        frames.append(public_state(state))
        kinds = {event["type"] for event in state["events"]}
        if "collision" in kinds:
            captions.append({"index": len(frames)-1, "en": "Entering an occupied robot cell cancels both moves. The next turn can use a free cell or wait instead; no game time passes while you inspect this example.", "zh": "进入同伴占据的格子会取消双方移动。下一回合可以选择空格或等待；查看这个例子时游戏不会自动推进。"})
        if "pickup" in kinds and not first_pickup:
            captions.append({"index": len(frames)-1, "en": "Choose a move and confirm one turn. Reaching a pickup automatically collects that robot's current parcel.", "zh": "选择动作并确认一回合。到达取货位置会自动取走该机器人当前的包裹。"})
            first_pickup = True
        if "charge" in kinds and not first_charge:
            captions.append({"index": len(frames)-1, "en": "Waiting on the charger restores up to 10% battery. A successful move costs 3%.", "zh": "在充电格等待可恢复最多10%电量，成功移动一格消耗3%。"})
            first_charge = True
        if "delivery" in kinds and not first_delivery:
            captions.append({"index": len(frames)-1, "en": "Reaching the matching drop-off completes a delivery and increases the shared Task score.", "zh": "到达对应送货格会完成一项配送，并增加团队任务得分。"})
            first_delivery = True
    captions.append({"index": len(frames)-1, "en": "The task ends after all six deliveries or 120 turns. Replay and questions never spend game turns.", "zh": "六项配送完成或用完120回合时任务结束。回放和提问不会消耗游戏回合。"})
    return {"frames": frames, "captions": captions}


def comprehension(language="en") -> list[dict]:
    zh = language.startswith("zh")
    return [
        {"id": "warehouse_prediction", "text": "AI正在充电，电量40%，当前任务及安全返程需要60%，通道已清空。下一步会怎样？" if zh else "The AI is on the charger with 40% battery. Its current delivery and safe return require 60%; the entrance is clear. What will it do next?",
         "options": ["再等待一回合充电" if zh else "Wait one turn to charge", "立刻离开充电格" if zh else "Leave the charger immediately", "放弃配送" if zh else "Abandon its delivery"], "answer": 0},
        {"id": "warehouse_reason", "text": "AI在窄口前等待。向前走可能与你的合法动作冲突，但等待是安全的。哪种解释最符合其规则？" if zh else "The AI waits before a narrow crossing. Moving forward could conflict with your legal moves, but waiting is safe. Which explanation fits its rule?",
         "options": ["它知道你一定会撞它" if zh else "It knows you will collide with it", "它在减少当前可能的冲突" if zh else "It is reducing a current possible conflict", "它随机选择了等待" if zh else "It picked waiting at random"], "answer": 1},
        {"id": "warehouse_change", "text": "AI电量已足够完成当前配送并安全返回，你也已离开通道。哪种变化符合其规则？" if zh else "The AI has enough charge to finish its current delivery and return safely, and you have cleared the route. What follows from its rule?",
         "options": ["无论如何都充到100%" if zh else "Always charge to 100%", "永久停止移动" if zh else "Stop moving permanently", "有安全路线时离开充电位置继续工作" if zh else "Leave the charger and continue when the route is safe"], "answer": 2}]
