"""Exact historical Warehouse controller over the preserved pre-v3 physics.

Only this adapter translates the shared study API. Simulation and action choice
remain in the frozen legacy implementation; JSON snapshots preserve its complete
state and random generator without invoking the goal-recomputing set_state().
"""
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from math import ceil
import json
from pathlib import Path

from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState, participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.numpy_policy import NumpyWarehousePolicy
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.energy_management import charge_release_evidence
from env.warehouse.frozen_missions import frozen_training_missions
from env.warehouse import historical_sep2_coordination as historical

DOMAIN = "warehouse"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/study_v3_warehouse.json").read_text())
VERSION = CONFIG["version"]
WIDTH, HEIGHT = STUDY_MAP_LAYOUT.cols, STUDY_MAP_LAYOUT.rows
CHARGER = tuple(reversed(STUDY_MAP_LAYOUT.charger_position))
WALLS = frozenset((c, r) for r, c in STUDY_MAP_LAYOUT.blocked_positions)
FLOOR = frozenset((c, r) for r, c in STUDY_MAP_LAYOUT.passable_positions)
MOVES = {a.lower(): (dc, dr) for a, (dr, dc) in MOVE_DELTAS.items()} | {"wait": (0, 0)}
MOVE_NAMES = {"up": ("up", "向上"), "down": ("down", "向下"), "left": ("left", "向左"), "right": ("right", "向右"), "wait": ("wait", "等待")}
_ACTORS = {"human": "robot_1", "ai": "robot_2"}


def _plain(value):
    """Canonical, finite JSON, also used to detach all returned structures."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _pack(value):
    # Preserve tuples even inside untyped coordination-plan dictionaries.
    if isinstance(value, tuple):
        return {"$tuple": [_pack(x) for x in value]}
    if isinstance(value, list):
        return [_pack(x) for x in value]
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    return value


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            return tuple(_unpack(x) for x in value["$tuple"])
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(x) for x in value]
    return value


@lru_cache(maxsize=1)
def _actor():
    for relative, expected in CONFIG["source_sha256"].items():
        if sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Warehouse frozen source or actor mismatch: " + relative)
    return NumpyWarehousePolicy.load(ROOT / "output/deployment/warehouse_mappo_v68_6x7_actor.npz")


def _snapshot(env):
    return {"version": CONFIG["snapshot_version"], "state": _pack(asdict(env.state)),
            "rng": _pack(env.get_rng_state()), "episode_counter": env._episode_counter}


def _restore(state):
    if state.get("version") != VERSION:
        raise ValueError("warehouse state version mismatch")
    data = state["snapshot"]
    if data["version"] != CONFIG["snapshot_version"]:
        raise ValueError("warehouse snapshot version mismatch")
    raw = _unpack(data["state"])
    raw["agents"] = [AgentState(**x) for x in raw["agents"]]
    for field in ("tasks", "completed_tasks"):
        raw[field] = [DeliveryTask(**x) for x in raw[field]]
    env = WarehouseMultiAgentEnv(participant_study_config())
    # set_state is an intervention API: it recomputes goals and coordination.
    # Persistence must restore the already committed decision state exactly.
    env.state = WarehouseState(**raw)
    env._episode_counter = int(data["episode_counter"])
    env.set_rng_state(_unpack(data["rng"]))
    errors = env.validate_state(env.state)
    if errors:
        raise ValueError("invalid warehouse snapshot: " + str(errors))
    if env.state.frame != state["turn"] or bool(env.state.terminated or env.state.truncated) != state["terminal"]:
        raise ValueError("warehouse snapshot boundary mismatch")
    return env


def _actor_view(agent):
    return {"x": agent.position[1], "y": agent.position[0], "battery": agent.battery,
            "carrying": agent.carrying_task_id, "heading": agent.heading.lower(), "active": agent.active}


def _orders(env):
    return [{"id": t.task_id, "owner": "shared", "pickup": list(reversed(t.pickup_position)),
             "dropoff": list(reversed(t.delivery_position)), "status": t.status,
             "carrier": next((k for k, v in _ACTORS.items() if v == t.carrier_agent_id), None)}
            for t in env.state.tasks]


def _state(env, task, seed, events=None, last_actions=None, charges=0):
    s = env.state
    return {"domain": DOMAIN, "version": VERSION, "scenario_version": CONFIG["scenario_version"],
            "task": task, "seed": int(seed), "scenario_seed": CONFIG["task_seeds"][str(task)],
            "turn": s.frame, "max_turns": env.config.horizon, "terminal": bool(s.terminated or s.truncated),
            "events": _plain(events or []), "policy_memory": {}, "snapshot": _plain(_snapshot(env)),
            "human": _actor_view(s.by_id("robot_1")), "ai": _actor_view(s.by_id("robot_2")),
            "orders": _orders(env), "deliveries": s.total_deliveries,
            "collision_events": s.robot_collision_events, "shutdown_events": s.shutdown_count,
            "charge_events": charges, "last_actions": last_actions}


def initial_state(seed: int, task: int) -> dict:
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2, or 3")
    _actor()
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=CONFIG["task_seeds"][str(task)])
    s = env.get_state()
    s.participant_controlled_agent_id = "robot_1"
    # Exactly the old start-round lifecycle, only used at initial creation.
    env.set_state(s)
    return _state(env, task, seed)


def legal_actions(state, actor="human"):
    if actor not in _ACTORS:
        raise ValueError("unknown actor")
    # The original controls submit wall moves too; physics records/cancels
    # them. Never mask out a recognized human command or rewrite it to WAIT.
    return [] if state["terminal"] else [a.lower() for a in ACTIONS]


def _distance(start, target):
    return shortest_path_distance(tuple(reversed(start)), tuple(reversed(target)), STUDY_MAP_LAYOUT.layout_id)


def _pos(actor):
    return actor["x"], actor["y"]


def _target(actor, action):
    dx, dy = MOVES[action]
    return actor["x"] + dx, actor["y"] + dy


def _selection(env, seed):
    s = env.get_state()
    proposed, distributions = _actor().act(env.observations(), deterministic=False,
        base_seed=seed, decision_key=(s.episode_id, s.frame))
    action, trace = historical.select_human_ai_action(env, proposed["robot_2"])
    return action, trace, proposed, distributions


def _reason(env, action, trace):
    ai = env.state.by_id("robot_2")
    selected = trace["selected_ai_action"]
    candidates = trace["ai_action_candidates"]
    safer_than = [c for c in candidates if len(c["collision_counterfactuals"]) > len(selected["collision_counterfactuals"])]
    verb, zhverb = MOVE_NAMES[action.lower()]
    en, zh = f"I will {verb if action == 'WAIT' else 'move ' + verb}. ", f"我会{zhverb}。"
    if action == "WAIT" and ai.position == env.layout.charger_position and ai.battery < 100:
        threshold = charge_release_evidence(env, env.state, ai)["release_threshold"]
        code = "charging"
        en += f"Waiting here restores up to 10 battery. I have {ai.battery:g}; the current departure threshold is {threshold:g}."
        zh += f"在这里等待最多恢复10点电量。我目前有{ai.battery:g}点，当前离开充电位置的电量门槛为{threshold:g}点。"
    elif safer_than:
        c = min(safer_than, key=lambda x: x["distance_after"])
        n, chosen_n = len(c["collision_counterfactuals"]), len(selected["collision_counterfactuals"])
        code = "reduce_possible_collision"
        alternative_label = "Waiting" if c["action"] == "WAIT" else "Moving " + c["action"].lower()
        en += f"{alternative_label} could conflict with {n} of your currently possible moves; my chosen action conflicts with {chosen_n}. I choose before knowing your next command."
        if selected["satisfies_planned_clearance"]:
            en += " This also carries out the current plan to clear a space for your route."
            zh += "这也执行了当前为你的路线腾出空间的计划。"
        zh += f"{MOVE_NAMES[c['action'].lower()][1]}可能与你当前可选动作中的{n}种发生冲突；所选动作对应{chosen_n}种。我选择时还不知道你下一步的指令。"
    elif trace["ai_is_planned_waiter"] and action == "WAIT":
        code = "hold_clearance"
        en += "The current shared-route plan has me hold this space while you pass. I will check the situation again after the joint move."
        zh += "当前通道配合计划要求我暂时保持这个位置，让你通过。双方完成行动后，我会重新检查局面。"
    elif selected["satisfies_planned_clearance"] or (trace["physical_clearance_required"] and action != "WAIT"):
        code = "clear_shared_route"
        en += "This step creates space for the current shared-route handoff. It may move me away from my delivery destination temporarily."
        zh += "这一步为当前的通道交接腾出空间，可能暂时远离我的配送目标。"
    elif selected["distance_after"] < selected["distance_before"]:
        code = "mission_progress"
        en += f"Among the safer available choices, this reduces my current route from {selected['distance_before']} to {selected['distance_after']} moves."
        zh += f"在优先考虑安全的可选动作中，这一步把当前路线从{selected['distance_before']}步缩短到{selected['distance_after']}步。"
    else:
        code = "conservative_selection"
        en += f"I first compare possible conflicts, then battery safety and the current handoff plan, before route progress. This choice leaves {selected['distance_after']} moves to the current target."
        zh += f"我先比较可能的冲突，再考虑电量安全和当前交接计划，最后考虑路线推进。所选动作执行后，到当前目标还有{selected['distance_after']}步。"
    return code, en, zh


def decide(state):
    if state["terminal"]:
        return {"action": "wait", "reason_code": "complete", "reason_en": "The task has finished.",
                "reason_zh": "任务已经结束。", "goal": "complete", "memory": {}, "alternatives": [], "facts": []}
    env = _restore(state)
    action, trace, _, _ = _selection(env, state["scenario_seed"])
    code, en, zh = _reason(env, action, trace)
    goal = historical._committed_goal(env, env.state, env.state.by_id("robot_2"))
    alternatives = []
    for c in trace["ai_action_candidates"]:
        if c["action"] == action:
            continue
        n = len(c["collision_counterfactuals"])
        alternatives.append({"action": c["action"].lower(),
            "en": f"If I choose {c['action'].lower()}, my target would be column {c['target'][1]}, row {c['target'][0]}; the route would have {c['distance_after']} moves left, and {n} of your possible moves could conflict. This is a possibility, not a prediction of your choice.",
            "zh": f"如果我选择{MOVE_NAMES[c['action'].lower()][1]}，目标为第{c['target'][1]}列、第{c['target'][0]}行；路线还剩{c['distance_after']}步，你的可选动作中有{n}种可能发生冲突。这表示可能性，不是预测你的选择。"})
    return {"action": action.lower(), "reason_code": code, "reason_en": en, "reason_zh": zh,
            "goal": f"column {goal[1]}, row {goal[0]}", "memory": {}, "alternatives": alternatives,
            "facts": [{"id": "ai_goal_distance", "en": f"My current route target is column {goal[1]}, row {goal[0]}; {trace['selected_ai_action']['distance_before']} moves away before this turn.",
                       "zh": f"我当前路线目标为第{goal[1]}列、第{goal[0]}行；本回合行动前相距{trace['selected_ai_action']['distance_before']}步。"}],
            "controller_trace": _plain(trace), "snapshot_sha256": sha256(json.dumps(state["snapshot"], sort_keys=True).encode()).hexdigest()}


def _events(before, after, info):
    rows = []
    def add(kind, en, zh, **more):
        rows.append({"type": kind, "en": en, "zh": zh, **more})
    if info.get("robot_collision_event") or after.robot_collision_events > before.robot_collision_events:
        add("collision", "A robot collision cancelled the conflicting movement and cost 10 points.", "机器人碰撞取消了冲突移动，并扣除10分。")
    for actor, key in _ACTORS.items():
        a, b = before.by_id(key), after.by_id(key)
        who, whozh = ("You", "你") if actor == "human" else ("The AI", "AI")
        if b.battery > a.battery:
            add("charge", f"{who} gained {b.battery-a.battery:g} battery by waiting at the charger.", f"{whozh}在充电位置等待，恢复{b.battery-a.battery:g}点电量。", actor=actor, amount=b.battery-a.battery)
        if a.carrying_task_id is None and b.carrying_task_id is not None:
            add("pickup", f"{who} picked up {b.carrying_task_id} at its A point.", f"{whozh}在A点取走了{b.carrying_task_id}。", actor=actor, order_id=b.carrying_task_id)
        if a.active and not b.active:
            add("shutdown", f"{who} ran out of battery. The task ends; unused steps still count in the time cost.", f"{whozh}电量耗尽，任务结束；未使用步数仍计入时间扣分。", actor=actor)
    for t in after.completed_tasks[len(before.completed_tasks):]:
        add("delivery", f"{t.task_id} reached its B point: +10 points, before step and other costs. A new shared job replaces it.", f"{t.task_id}已送达B点：获得10分，另计步数及其他扣分。系统补充一个新的共享订单。", order_id=t.task_id)
    prior = {t.task_id for t in before.tasks}
    for t in after.tasks:
        if t.task_id not in prior:
            add("order_arrived", "A new shared delivery job is now available.", "一个新的共享配送订单已出现。", order_id=t.task_id)
    for e in after.last_rule_events:
        if e.get("event") == "shared_charger_occupancy":
            actor = next(k for k, v in _ACTORS.items() if v == e["agent_id"])
            add("shared_charger_occupancy", "The charger was held above 60 battery while the nearby partner had below 20 and a safe exit existed: −5 points for this stay.", "占用者电量超过60，同伴电量低于20且距充电位置不超过两步，并存在安全出口，本次连续占用扣5分。", actor=actor, score_delta=e["score_delta"])
    for key in info.get("invalid_move_agents", ()):
        add("blocked", "A requested move was blocked by the map; no movement battery was spent.", "请求的移动被地图阻挡，没有消耗移动电量。", actor="human" if key == "robot_1" else "ai")
    if after.terminated or after.truncated:
        add("complete", "This task has ended. Continue to the next stage when ready.", "本任务已结束，准备好后可进入下一阶段。")
    return rows


def step(state, human_action, decision=None):
    if state["terminal"]:
        raise ValueError("task complete")
    if human_action not in legal_actions(state):
        raise ValueError("illegal human action")
    actual_decision = decide(state)
    if decision is not None and decision != actual_decision:
        raise ValueError("decision does not match this pre-action state")
    env = _restore(state)
    before = env.get_state()
    selected, trace, proposed, distributions = _selection(env, state["scenario_seed"])
    human, guard = historical.guard_participant_action(env, human_action.upper())
    actions = {**proposed, "robot_1": human, "robot_2": selected}
    runtime = {**trace, "participant_action_guard": guard, "selected_actions": dict(actions)}
    _, _, _, _, info = env.step(actions, decision_metadata=distribution_decision_metadata(distributions,
        decision_source="participant_plus_robust_numpy_actor", participant_overrides={"robot_1": human},
        policy_actions=proposed, selected_actions=actions, runtime_decision=runtime))
    events = _events(before, env.state, info)
    result = _state(env, state["task"], state["seed"], events,
        {"human": human.lower(), "ai": selected.lower()}, state["charge_events"] + sum(e["type"] == "charge" for e in events))
    result["scenario_seed"] = state["scenario_seed"]
    return result


def score(state):
    env = _restore(state)
    s = env.state
    return {"task_score": float(s.user_score), "raw_score": float(s.user_score), "score_max": None,
            "score_scale": "raw", "breakdown": dict(s.score_breakdown),
            "metrics": {"deliveries": s.total_deliveries, "collision_events": s.robot_collision_events,
                        "shutdown_events": s.shutdown_count, "charge_events": state["charge_events"],
                        "elapsed_turns": s.frame, "human_detour_units": s.human_route_regret_units,
                        "shared_charger_penalties": int(-s.score_breakdown["shared_charger_occupancy"] / 5),
                        "shared_charger_penalty_points": s.score_breakdown["shared_charger_occupancy"],
                        "terminal_reason": s.terminal_reason}}


def public_state(state):
    env = _restore(state)
    public = {k: deepcopy(state[k]) for k in ("domain", "version", "task", "turn", "max_turns", "terminal", "events")}
    public.update({k: _actor_view(env.state.by_id(v)) for k, v in _ACTORS.items()})
    public.update(width=WIDTH, height=HEIGHT, walls=[list(p) for p in sorted(WALLS)], orders=_orders(env),
        chargers=[list(CHARGER)], stations=[{"id": "charger", "x": CHARGER[0], "y": CHARGER[1], "kind": "charger", "label_en": "Shared charger", "label_zh": "共享充电位置"}],
        score=score(state), map_id=STUDY_MAP_LAYOUT.layout_id)
    return public


def rules(language="en"):
    pairs = [
        ("Work with the AI for up to 120 steps. Both robots act simultaneously from the same starting state. Questions and replay do not spend game steps.", "与AI共同工作最多120步。双方根据同一行动前状态同时执行。提问与回放不消耗步数。"),
        ("Two shared A-to-B jobs remain active. Either empty robot can claim a parcel by reaching A; reaching its matching B delivers it and creates another shared job. Carry one parcel at a time.", "地图保持两个有效共享A到B订单。空手机器人到达A点自动认领包裹，到达对应B点送达并补充新订单；每次携带一个包裹。"),
        ("WASD or arrow keys move; Space waits. Walls block movement. Conflicting destinations, swaps, or entry into a stationary robot cause a robot collision; your submitted command is not changed to avoid it.", "WASD或方向键移动，空格等待。墙壁阻挡移动。同一目标、交换位置或进入静止同伴格会导致机器人碰撞；系统不会为了避碰替换你的指令。"),
        ("A successful move uses 2 battery. Waiting at the charger adds up to 10, capped at 100. Cancelled moves and waits elsewhere use no battery. Reaching the charger at exactly zero is safe; zero elsewhere ends the task.", "成功移动耗电2点。在充电格等待最多恢复10点，上限100。取消的移动和其他位置等待不耗电。电量恰好为0时到达充电格安全，其他位置电量耗尽会结束任务。"),
        ("Raw team score: +10 per delivery, −10 per collision, −5 per robot shutdown, −1 per budget step, and −2 per human detour unit. A shutdown charges the unused steps too. The score can be negative and has no fixed 100-point maximum.", "团队原始分：每配送+10，每碰撞−10，每台断电−5，每预算步−1，每人类绕路单位−2。断电时剩余步数也会扣除。分数可以为负，不是满分100分。"),
        ("A shared-charger violation costs 5 points once per continuous stay: the occupant waits and ends above 60 battery, an active partner below 20 is within two path steps, a safe exit exists, and no collision occurs. Leaving resets this one-stay rule.", "共享充电站违规每次连续占用扣5分：占用者等待后电量超过60，仍可行动的同伴电量低于20且距充电站不超过两步，存在安全出口且本步无碰撞。离开后重新开始计算。"),
        ("A human detour unit is extra remaining route distance compared with the best collision-free move toward the same current target, capped at two units per step. Waiting can count when safe progress was possible. Charger waits, blocked wall moves, collisions, and necessary movement out of the AI's next cell are exempt.", "人类绕路单位是相对当前同一目标、与可避免碰撞的最佳一步相比多出的剩余距离，每步上限两单位。有安全前进机会时，等待也可能计算在内。充电格等待、撞墙被阻挡、碰撞以及必须离开AI下一目标格的移动免于此项扣分。"),
    ]
    return [p[language.startswith("zh")] for p in pairs]


def facts(state, decision=None):
    env = _restore(state)
    result = [{"id": "time", "en": f"Step {state['turn']} of 120; {120-state['turn']} budget steps remain.", "zh": f"已用{state['turn']}步，预算120步，还剩{120-state['turn']}步。"},
              {"id": "score", "en": f"Raw team score is {env.state.user_score:g}; {env.state.total_deliveries} deliveries completed. This score can be negative and has no fixed maximum.", "zh": f"团队原始分为{env.state.user_score:g}；已完成{env.state.total_deliveries}次配送。分数可为负，没有固定满分。"}]
    labels = {"delivery": ("Delivery points", "配送得分"), "robot_collision": ("Collision costs", "碰撞扣分"), "shutdown": ("Shutdown costs", "断电扣分"), "time": ("Budget-step costs", "预算步数扣分"), "human_detour": ("Human detour costs", "人类绕路扣分"), "shared_charger_occupancy": ("Charger occupancy costs", "占桩扣分")}
    for k, amount in env.state.score_breakdown.items():
        result.append({"id": "score_"+k, "en": f"{labels[k][0]}: {amount:g}.", "zh": f"{labels[k][1]}：{amount:g}。"})
    for actor, key in _ACTORS.items():
        a = env.state.by_id(key)
        d = shortest_path_distance(a.position, env.layout.charger_position, env.config.map_layout_id)
        who, whozh = ("You", "你") if actor == "human" else ("The AI", "AI")
        result.append({"id": actor+"_state", "en": f"{who} {'are' if actor == 'human' else 'is'} at column {a.position[1]}, row {a.position[0]}, with {a.battery:g} battery and carrying {a.carrying_task_id or 'no parcel'}.", "zh": f"{whozh}位于第{a.position[1]}列、第{a.position[0]}行，电量{a.battery:g}，携带{a.carrying_task_id or '无包裹'}。"})
        result.append({"id": actor+"_charger_distance", "en": f"The shortest map route from {who.lower()} to the charger is {d} moves, costing {2*d} battery if the partner does not force a detour.", "zh": f"{whozh}到充电位置的地图最短路线为{d}步，耗电{2*d}点，前提是同伴没有造成绕行。"})
    ai = env.state.by_id("robot_2")
    if ai.position == env.layout.charger_position:
        evidence = charge_release_evidence(env, env.state, ai)
        threshold = evidence["release_threshold"]
        missing = max(0, threshold-ai.battery)
        waits = int(ceil(missing / env.config.charge_per_wait))
        result.append({"id": "ai_charge_shortfall",
            "en": f"I have {ai.battery:g} battery on the charger; the current departure threshold is {threshold:g}. I need {missing:g} more to reach that threshold. Staying here to charge would take {waits} charging steps at up to 10 per step if the same situation continues; a handoff can require leaving earlier. Questions do not execute these steps.",
            "zh": f"我在充电位置有{ai.battery:g}点电量，当前离开门槛为{threshold:g}点，还差{missing:g}点。如果当前情况保持不变且继续留在这里充电，以每步最多恢复10点计算，需要{waits}个充电步；交接可能要求我提前离开。提问不会执行这些步。"})
        result.append({"id": "ai_charge_calculation",
            "en": f"The current charging calculation uses {evidence['pickup_steps']:g} moves to pickup, {evidence['delivery_steps']:g} for delivery, {evidence['return_steps']:g} to return and {evidence['mission_reserve_steps']:g} reserve moves, at 2 battery per move. It adds {evidence['hysteresis_energy']:g} battery to avoid immediately returning and {evidence['coordination_contention_energy']:g} for a currently visible charger handoff; the total is capped at 100. An empty robot checks all currently available jobs because its partner may claim one first.",
            "zh": f"当前充电计算包括取货{evidence['pickup_steps']:g}步、配送{evidence['delivery_steps']:g}步、返程{evidence['return_steps']:g}步及{evidence['mission_reserve_steps']:g}步安全余量，每步耗电2点；另外预留{evidence['hysteresis_energy']:g}点避免离开后立刻返回，以及{evidence['coordination_contention_energy']:g}点应对当前可见的充电交接，总量上限100。空手机器人检查所有当前可用订单，因为同伴可能抢先取走其中一个。"})
    for order in _orders(env):
        result.append({"id": "order_"+order["id"], "en": f"{order['id']} is a shared job: A at column {order['pickup'][0]}, row {order['pickup'][1]}; B at column {order['dropoff'][0]}, row {order['dropoff'][1]}; status {order['status']}, carrier {order['carrier'] or 'none'}.", "zh": f"{order['id']}是共享订单：A点为第{order['pickup'][0]}列、第{order['pickup'][1]}行；B点为第{order['dropoff'][0]}列、第{order['dropoff'][1]}行；状态{order['status']}，承运者{order['carrier'] or '无'}。"})
    result.extend({"id": f"last_event_{i}", "en": e["en"], "zh": e["zh"]} for i, e in enumerate(state["events"]))
    result.extend({"id": f"public_rule_{i}", "en": en, "zh": zh} for i, (en, zh) in enumerate(zip(rules(), rules("zh"))))
    if not state["terminal"]:
        actual = decide(state)
        if decision is not None and decision != actual:
            raise ValueError("decision does not match this pre-action state")
        result.append({"id": "actual_decision", "en": actual["reason_en"], "zh": actual["reason_zh"]})
        result.extend(actual["facts"])
        result.extend({"id": "alternative_"+a["action"], "en": a["en"], "zh": a["zh"]} for a in actual["alternatives"])
    return result


def _human_advisor(state, *, proactive=True):
    """Present-state feasibility helper; never replaces the fixed AI."""
    if state["terminal"]:
        return "wait"
    env = _restore(state)
    actual, trace, _, _ = _selection(env, state["scenario_seed"])
    # Historical joint guidance proposes only the HUMAN command. Its proposed
    # AI command is discarded; scoring below uses the actual fixed AI action.
    suggested = historical.stable_coordination_actions(env)["robot_1"]
    h = env.state.by_id("robot_1")
    goal = (env._frozen_route_goal(env.state, h.agent_id) if proactive else None) or historical._committed_goal(env, env.state, h)
    ai = env.state.by_id("robot_2")
    # The conservative teammate tests every possible human command. Merely
    # waiting two cells in front of it can keep its next cell contested. A
    # human-side retreat creates actual separation without changing that AI.
    needs_room = proactive and actual == "WAIT" and any(
        c["distance_after"] < c["distance_before"] and c["collision_counterfactuals"]
        for c in trace["ai_action_candidates"])
    candidates = []
    for a in ACTIONS:
        targets, _, invalid, collision, _, _ = env._resolve_motion(env.state, {"robot_1": a, "robot_2": actual})
        target = targets["robot_1"]
        remaining = h.battery - (2 if target != h.position else 0)
        return_cost = 2 * shortest_path_distance(target, env.layout.charger_position, env.config.map_layout_id)
        charging = h.position == env.layout.charger_position and env._requires_charge(env.state, h)
        next_ai_blocked = 0
        if proactive and ai.carrying_task_id and not collision and not invalid:
            branch = _restore(state)
            known_jobs = {job.task_id for job in branch.state.tasks}
            branch.step({"robot_1": a, "robot_2": actual})
            # Do not inspect a newly sampled job before it has become public.
            if not (branch.state.terminated or branch.state.truncated) and {job.task_id for job in branch.state.tasks} <= known_jobs:
                following, _, _, _ = _selection(branch, state["scenario_seed"])
                next_ai = branch.state.by_id("robot_2")
                fixed_goal = historical._committed_goal(branch, branch.state, next_ai)
                dr, dc = MOVE_DELTAS.get(following, (0, 0))
                next_position = (next_ai.position[0]+dr, next_ai.position[1]+dc)
                distance = shortest_path_distance(next_ai.position, fixed_goal, env.config.map_layout_id)
                next_ai_blocked = int(distance > 0 and shortest_path_distance(next_position, fixed_goal, env.config.map_layout_id) >= distance)
        rank = (int(collision), int("robot_1" in invalid), int(remaining < return_cost),
                int(charging and a != "WAIT"), next_ai_blocked,
                max(0, 3 - shortest_path_distance(target, ai.position, env.config.map_layout_id)) if needs_room else 0,
                shortest_path_distance(target, goal, env.config.map_layout_id) if proactive else int(a != suggested),
                int(a != suggested) if proactive else shortest_path_distance(target, goal, env.config.map_layout_id), int(a == "WAIT"))
        candidates.append((rank, a))
    return min(candidates)[1].lower()


def human_advisor(state):
    return _human_advisor(state)


def demonstration():
    # Full real trajectory on the original neutral demo seed; captions are
    # selected from confirmed outcomes, never an explanation of hidden plans.
    state = initial_state(CONFIG["demo_seed"], 1)
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=CONFIG["demo_seed"])
    original = env.get_state(); original.participant_controlled_agent_id = "robot_1"; env.set_state(original)
    state = _state(env, 1, CONFIG["demo_seed"]); state["scenario_seed"] = CONFIG["demo_seed"]
    frames = [public_state(state)]
    captions = [{"index": 0, "en": "You control the blue robot. The orange robot is your AI teammate. Two shared A-to-B jobs are available.", "zh": "你控制蓝色机器人，橙色机器人是AI队友。双方共用两个A到B订单。"}]
    seen = set()
    demonstrated_collision = False
    for _ in range(120):
        if state["terminal"]: break
        action = _human_advisor(state, proactive=False)
        if not demonstrated_collision and state["turn"] >= 65:
            trial = _restore(state)
            ai_action = decide(state)["action"].upper()
            colliding = [a for a in historical.causal_participant_actions(trial)
                         if trial._resolve_motion(trial.state, {"robot_1": a, "robot_2": ai_action})[3]]
            if colliding:
                action = colliding[0].lower()
                demonstrated_collision = True
            else:
                # A short deliberately poor approach supplies a real public
                # collision example; only the demonstration's human commands
                # change. The historical AI remains untouched.
                human = trial.state.by_id("robot_1")
                partner = trial.state.by_id("robot_2")
                action = min(historical.causal_participant_actions(trial), key=lambda a: (
                    shortest_path_distance(tuple(reversed(_target(_actor_view(human), a.lower()))),
                                           partner.position, trial.config.map_layout_id), a == "WAIT")).lower()
        state = step(state, action); frames.append(public_state(state))
        for event in state["events"]:
            if event["type"] in {"pickup", "delivery", "charge", "collision"} and event["type"] not in seen:
                captions.append({"index": len(frames)-1, "en": event["en"], "zh": event["zh"]}); seen.add(event["type"])
    captions.extend([
        {"index": len(frames)-1, "en": "A task ends at 120 steps or a battery shutdown. Questions and replay do not spend steps. Raw score can be negative.", "zh": "任务在120步或电量耗尽时结束。提问和回放不消耗步数。原始分可能为负。"}])
    while len(captions) < 6:
        i = min(len(frames)-1, 2 + len(captions))
        captions.insert(-1, {"index": i, "en": "A successful move uses 2 battery; waiting at the charger restores up to 10. Plan a return before the battery runs out.", "zh": "成功移动消耗2点电量，在充电格等待最多恢复10点。请在电量耗尽前安排返回充电位置。"})
    return {"frames": frames, "captions": sorted(captions, key=lambda c: c["index"])}


def comprehension(language="en"):
    zh = language.startswith("zh")
    return [
        {"id": "warehouse_prediction", "text": "机器人在充电格等待，电量40。没有碰撞时，本步后电量是多少？" if zh else "A robot waits on the charger with 40 battery. There is no collision. What is its battery after the step?", "options": ["50", "40", "38"], "answer": 0},
        {"id": "warehouse_reason", "text": "AI不知道你下一步的指令。向前走可能发生冲突，而等待没有可能冲突。保守队友优先考虑什么？" if zh else "The AI does not know your next command. Moving forward could conflict, while waiting has no possible conflict. What does the conservative teammate prioritize?", "options": ["预知你的动作" if zh else "Predicting your command with certainty", "减少可能冲突" if zh else "Reducing possible conflicts", "优先追求配送速度" if zh else "Delivery speed before conflicts"], "answer": 1},
        {"id": "warehouse_change", "text": "一个包裹已送达B点，任务尚未结束。接下来订单如何变化？" if zh else "A parcel reaches its B point before the task ends. What happens to jobs?", "options": ["同一包裹继续留在手上" if zh else "The robot keeps the same parcel", "不再有新订单" if zh else "No further jobs can appear", "补充一个共享订单" if zh else "A replacement shared job appears"], "answer": 2},
    ]


def screening_report():
    # Fixed historical task seed; do not pretend enrollment IDs are scenarios.
    results = {}
    for task in (1, 2, 3):
        state = initial_state(100, task)
        while not state["terminal"]:
            state = step(state, human_advisor(state))
        results[str(task)] = score(state)
    return {"domain": DOMAIN, "version": VERSION, "not_human_results": True, "fixed_task_seeds": CONFIG["task_seeds"], "tasks": results}
