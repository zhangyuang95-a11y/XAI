"""Authoritative, deterministic cooperative Pong for the three-domain study.

Only ``step`` sees the hidden wave schedule. Decisions, facts and the human
advisory proxy project the current wave first. No participant/group input is
accepted. Lanes are displayed as 1–9 while coordinates are stored as 0–8.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any

DOMAIN = "pong"
VERSION = "pong-turnbased.v3.0"
CONFIG = json.loads((Path(__file__).resolve().parents[2] / "configs/study_v3_pong.json").read_text())
LANES = CONFIG["lanes"]
WEIGHTS = CONFIG["weights"]
_DELTAS = {"left": -1, "wait": 0, "right": 1}


def _ball(identifier: str, kind: str, contacts: list[int], remaining: int) -> dict:
    return {"id": identifier, "kind": kind, "contacts": list(contacts),
            "remaining": remaining, "initial_remaining": remaining}


def _schedule(seed: int, task: int) -> list[list[dict]]:
    """Pre-sample all waves; performance and group cannot alter this schedule."""
    rng = random.Random(int(seed) * 1009 + task * 71)
    count = CONFIG["task_waves"][str(task)]
    schedule = []
    for index in range(count):
        left = rng.choice([1, 2, 3])
        right = left + rng.choice([3, 4])
        duration = [5, 6, 4, 5, 6, 4, 6, 5][index]
        if task == 3:
            left, right = 8 - right, 8 - left
        balls = [_ball(f"w{index + 1}-team", "cooperative", [left, right], duration)]
        include_small = (index % 3 == 1) if task == 1 else index % 4 != 3
        if include_small:
            # Some detours return in time; others genuinely forgo a team ball.
            safe = index % 2 == 0
            side = rng.choice([left, right])
            small_x = side + (1 if side == left else -1) if safe else (left + right) // 2
            small_time = 2 if safe else max(2, duration - 1)
            balls.append(_ball(f"w{index + 1}-small", "ordinary", [small_x], small_time))
        schedule.append(balls)
    return schedule


def _new_state(schedule: list[list[dict]], *, seed: int, task: int,
               human: int, ai: int) -> dict:
    all_balls = [ball for wave in schedule for ball in wave]
    return {"domain": DOMAIN, "version": VERSION, "scenario_version": CONFIG["scenario_version"],
            "task": task, "turn": 0, "max_turns": sum(max(b["remaining"] for b in w) for w in schedule),
            "terminal": False, "events": [], "policy_memory": {}, "seed": seed,
            "human": {"x": human}, "ai": {"x": ai}, "lanes": LANES,
            "wave": 1, "wave_count": len(schedule), "balls": deepcopy(schedule[0]),
            "_schedule": deepcopy(schedule), "raw_score": 0,
            "total_possible_points": sum(WEIGHTS[b["kind"]] for b in all_balls),
            "metrics": {"cooperative_caught": 0, "ordinary_caught": 0,
                        "cooperative_total": sum(b["kind"] == "cooperative" for b in all_balls),
                        "ordinary_total": sum(b["kind"] == "ordinary" for b in all_balls),
                        "misses": 0, "assignment_switches": 0,
                        "assignment_releases": 0, "waiting_turns": 0}}


def initial_state(seed: int, task: int) -> dict:
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2 or 3")
    human, ai = ((6, 2) if task == 3 else (2, 6))
    return _new_state(_schedule(seed, task), seed=int(seed), task=task, human=human, ai=ai)


def legal_actions(state: dict, actor: str = "human") -> list[str]:
    if actor not in ("human", "ai"):
        raise ValueError("actor must be human or ai")
    if state["terminal"]:
        return []
    x = state[actor]["x"]
    return [action for action in ("wait", "left", "right") if 0 <= x + _DELTAS[action] < LANES]


def _move_towards(position: int, target: int) -> str:
    return "left" if target < position else "right" if target > position else "wait"


def _observation(state: dict) -> dict:
    """AI capability boundary: deliberately excludes seed and hidden schedule."""
    return {"human": deepcopy(state["human"]), "ai": deepcopy(state["ai"]),
            "balls": deepcopy(state["balls"]), "policy_memory": deepcopy(state["policy_memory"]),
            "terminal": state["terminal"]}


def _decision_from_observation(obs: dict) -> dict:
    if obs["terminal"]:
        return {"action": "wait", "reason_code": "task_complete",
                "reason_en": "This task has finished. There is no next action to execute.",
                "reason_zh": "本任务已经结束，没有下一步需要执行。", "goal": "task_complete",
                "memory": {}, "alternatives": [], "assignment_changed": False,
                "assignment_released": False, "target_lane": None}
    human, ai = obs["human"]["x"], obs["ai"]["x"]
    team = next((b for b in obs["balls"] if b["kind"] == "cooperative"), None)
    smalls = [b for b in obs["balls"] if b["kind"] == "ordinary"]
    previous = obs["policy_memory"].get("commitment")
    committed = None
    comparisons = []
    release = False
    switched = False
    if team:
        left, right = team["contacts"]
        remaining = team["remaining"]
        options = []
        for own, partner in ((left, right), (right, left)):
            feasible = abs(ai - own) <= remaining and abs(human - partner) <= remaining
            comparisons.append({"action": _move_towards(ai, own),
                "en": f"Covering lane {own + 1} needs {abs(ai - own)} moves from me and {abs(human - partner)} moves from you to lane {partner + 1}; {remaining} turns remain. " + ("Both sides can still be reached if we move there." if feasible else "At least one side cannot be reached in time."),
                "zh": f"我到第{own + 1}道需{abs(ai - own)}步，你到第{partner + 1}道需{abs(human - partner)}步；还剩{remaining}回合。" + ("如果各自前往，两侧都还来得及。" if feasible else "至少一侧已经来不及。")})
            if feasible:
                options.append({"ball_id": team["id"], "ai_contact": own,
                                "human_contact": partner})
        if previous and previous.get("ball_id") == team["id"]:
            committed = next((o for o in options if o == previous), None)
            release = committed is None
        if committed is None and options:
            # Shortest total travel; fixed right-side AI assignment breaks ties.
            committed = min(options, key=lambda o: (abs(ai - o["ai_contact"]) +
                abs(human - o["human_contact"]), -o["ai_contact"]))
            switched = bool(previous and previous.get("ball_id") == team["id"] and previous != committed)
    target = None
    target_ball = None
    code = "no_reachable_ball"
    reason_en = "I will stay here because no currently visible ball has a reachable catching assignment."
    reason_zh = "当前可见的球没有能及时完成的接球分工，我会守在这里。"
    if committed:
        target, target_ball = committed["ai_contact"], team["id"]
        own_lane, partner_lane = target + 1, committed["human_contact"] + 1
        code = "switch_unreachable_assignment" if switched else "keep_assignment" if previous == committed else "assign_team_ball"
        reason_en = f"I will cover lane {own_lane} for {team['id']}; you need to cover lane {partner_lane} when it arrives in {team['remaining']} turns. "
        reason_zh = f"我负责{team['id']}的第{own_lane}道；它将在{team['remaining']}回合后到达，你需要覆盖第{partner_lane}道。"
        if switched:
            reason_en += "Our previous division can no longer reach both sides, so I changed sides."
            reason_zh += "原分工已经无法让双方及时到达，所以我换到了另一侧。"
        else:
            reason_en += "I will keep this side while both assigned positions remain reachable."
            reason_zh += "只要双方的分工仍然可达，我就会保持这一侧。"
        for small in sorted(smalls, key=lambda b: (b["remaining"], b["id"])):
            location, at = small["contacts"][0], small["remaining"]
            travel, back = abs(ai - location), abs(location - target)
            gap = team["remaining"] - at
            safe = at <= team["remaining"] and travel <= at and back <= gap
            if at > team["remaining"]:
                comparisons.append({"action": _move_towards(ai, location),
                    "en": f"{small['id']} arrives after the team ball. I will finish the current team-ball assignment before reconsidering it.",
                    "zh": f"{small['id']}在合作球之后到达。我会先完成当前合作分工，再考虑接它。"})
                continue
            comparisons.append({"action": _move_towards(ai, location),
                "en": f"Catching {small['id']} at lane {location + 1} takes {travel} moves before its {at}-turn arrival. Returning to my team-ball side takes {back} moves, with {max(0, gap)} turns between arrivals. " + ("I can catch it and return in time." if safe else "I cannot take this detour and still keep my team-ball assignment."),
                "zh": f"普通球{small['id']}将在{at}回合后到第{location + 1}道，我到那里需{travel}步；返回合作位置需{back}步，两球到达相隔{max(0, gap)}回合。" + ("我能接完再及时返回。" if safe else "我无法接它并保持原合作分工。")})
            # Do not leave the cooperative job merely to duplicate a partner
            # already standing on the ordinary contact point.
            if safe and human != location:
                target, target_ball, code = location, small["id"], "small_then_return"
                reason_en = f"I can catch {small['id']} at lane {location + 1} and still cover lane {own_lane} when {team['id']} arrives. Please cover lane {partner_lane} for the team ball."
                reason_zh = f"我能先在第{location + 1}道接{small['id']}，再及时回到第{own_lane}道接{team['id']}。请你负责合作球的第{partner_lane}道。"
                break
    else:
        reachable = [b for b in smalls if abs(ai - b["contacts"][0]) <= b["remaining"]]
        if reachable:
            selected = min(reachable, key=lambda b: (b["remaining"], abs(ai - b["contacts"][0]), b["id"]))
            target, target_ball = selected["contacts"][0], selected["id"]
            code = "ordinary_after_infeasible_team" if team else "catch_ordinary"
            reason_en = f"I will catch {target_ball} at lane {target + 1}. " + ("No division can get us to both team-ball contacts in time from our current positions." if team else "This visible ordinary ball is within reach.")
            reason_zh = f"我将到第{target + 1}道接{target_ball}。" + ("从双方目前位置出发，已经没有能及时覆盖合作球两侧的分工。" if team else "这颗可见的普通球还来得及接住。")
        elif release:
            code = "release_infeasible_assignment"
            reason_en = "Our team-ball assignment is no longer reachable, and I cannot reach another visible catch, so I will wait."
            reason_zh = "合作分工已经来不及完成，也没有其他可见的球可以及时接住，我会等待。"
    action = "wait" if target is None or obs["terminal"] else _move_towards(ai, target)
    return {"action": action, "reason_code": code, "reason_en": reason_en,
            "reason_zh": reason_zh, "goal": target_ball or "hold_position",
            "memory": {"commitment": committed} if committed else {},
            "alternatives": comparisons, "assignment_changed": switched,
            "assignment_released": release, "target_lane": target}


def decide(state: dict) -> dict:
    return _decision_from_observation(_observation(state))


def _event(event_type: str, en: str, zh: str, **fields: Any) -> dict:
    return {"type": event_type, "en": en, "zh": zh, **fields}


def _settle_visible(obs: dict, human_action: str, decision: dict) -> tuple[dict, int, list[dict]]:
    """Pure physical step shared by authoritative execution and exact search."""
    nxt = deepcopy(obs)
    nxt["human"]["x"] += _DELTAS[human_action]
    nxt["ai"]["x"] += _DELTAS[decision["action"]]
    nxt["policy_memory"] = deepcopy(decision["memory"])
    points, events, remaining = 0, [], []
    for ball in nxt["balls"]:
        ball["remaining"] -= 1
        if ball["remaining"]:
            remaining.append(ball)
            continue
        human, ai = nxt["human"]["x"], nxt["ai"]["x"]
        caught = ({human, ai} == set(ball["contacts"])) if ball["kind"] == "cooperative" else ball["contacts"][0] in (human, ai)
        earned = WEIGHTS[ball["kind"]] if caught else 0
        points += earned
        events.append(_event("caught" if caught else "missed",
            f"{ball['id']} {'caught' if caught else 'missed'}: +{earned} points.",
            f"{ball['id']}{'接住' if caught else '漏接'}：+{earned}分。",
            ball_id=ball["id"], kind=ball["kind"], points=earned,
            contacts=list(ball["contacts"]), human_x=human, ai_x=ai))
    nxt["balls"] = remaining
    if not any(b["id"] == nxt["policy_memory"].get("commitment", {}).get("ball_id") for b in remaining):
        nxt["policy_memory"] = {}
    return nxt, points, events


def step(state: dict, human_action: str, decision: dict | None = None) -> dict:
    if state["terminal"]:
        raise ValueError("This task is complete.")
    if human_action not in legal_actions(state):
        raise ValueError("Illegal human action.")
    expected = decide(state)
    if decision is None:
        decision = expected
    elif decision != expected:
        raise ValueError("The AI decision does not match the submitted state.")
    result = deepcopy(state)
    projected, gained, events = _settle_visible(_observation(state), human_action, decision)
    for key in ("human", "ai", "balls", "policy_memory"):
        result[key] = projected[key]
    result["turn"] += 1
    result["raw_score"] += gained
    result["events"] = events
    result["metrics"]["waiting_turns"] += decision["action"] == "wait"
    result["metrics"]["assignment_switches"] += bool(decision["assignment_changed"])
    result["metrics"]["assignment_releases"] += bool(decision["assignment_released"])
    for event in events:
        if event["type"] == "caught":
            result["metrics"][event["kind"] + "_caught"] += 1
        else:
            result["metrics"]["misses"] += 1
    if not result["balls"]:
        result["policy_memory"] = {}
        if result["wave"] >= result["wave_count"]:
            result["terminal"] = True
        else:
            result["wave"] += 1
            result["balls"] = deepcopy(result["_schedule"][result["wave"] - 1])
            result["events"].append(_event("wave_started", f"Wave {result['wave']} started.", f"第{result['wave']}波开始。", wave=result["wave"]))
    if result["turn"] >= result["max_turns"]:
        result["terminal"] = True
    return result


def score(state: dict) -> dict:
    denominator = state["total_possible_points"]
    return {"task_score": round(100 * state["raw_score"] / denominator, 6) if denominator else 0.0,
            "raw_score": state["raw_score"], "metrics": deepcopy(state["metrics"]) |
            {"total_possible_points": denominator,
             "cooperative_success_rate": state["metrics"]["cooperative_caught"] / max(1, state["metrics"]["cooperative_total"]),
             "ordinary_success_rate": state["metrics"]["ordinary_caught"] / max(1, state["metrics"]["ordinary_total"]),
             "elapsed_turns": state["turn"]}}


def public_state(state: dict) -> dict:
    public = {key: deepcopy(state[key]) for key in ("domain", "version", "task", "turn", "max_turns",
        "terminal", "events", "human", "ai", "lanes", "wave", "wave_count", "balls")}
    public["score"] = score(state)
    # Private decision counters are researcher-only; public scores describe play.
    for key in ("assignment_switches", "assignment_releases"):
        public["score"]["metrics"].pop(key, None)
    return public


def rules(language: str = "en") -> list[str]:
    en = ["There are nine lanes. Move one lane left or right, or wait, each turn. Paddles may share a lane without a penalty.",
          "Both paddles move at the same time. Then every visible ball counts down by one turn; balls at zero are scored.",
          "An ordinary ball earns 1 point when either paddle covers its contact lane. Covering it twice still earns only 1 point.",
          "A team ball earns 3 points only when the two paddles cover its two different contact lanes at the same arrival turn.",
          "A missed ball earns 0 points. Balls score once. A new wave appears after all balls in the current wave finish; paddle positions carry over.",
          "Task score is 100 times points earned divided by all scheduled ball points. Some balls may require competing choices, so 100 is not always achievable. Reading and replay do not advance turns."]
    zh = ["共有九道。每回合向左或向右移动一道，或等待。双方球拍可重叠，不额外扣分。",
          "双方先同时移动，随后可见球的剩余回合减一，减到零的球立即结算。",
          "普通球由任一球拍覆盖接触位置即可得1分；双方同时覆盖仍只得1分。",
          "合作球必须在同一到达回合由两个球拍分别覆盖两个不同的接触位置，才能得3分。",
          "漏接得0分，每球只结算一次。本波全部球结算后出现下一波，双方沿用当前位置。",
          "任务分数为100乘实得分除以本局所有球的总分值。部分球可能存在取舍，因此未必能得100分。阅读和回放不推进回合。"]
    return zh if language.startswith("zh") else en


def facts(state: dict, decision: dict | None = None) -> list[dict]:
    actual = decide(state)
    if decision is not None and decision != actual:
        raise ValueError("Decision evidence does not match this state.")
    evidence = [{"id": "position", "en": f"You are at lane {state['human']['x'] + 1}; your teammate is at lane {state['ai']['x'] + 1}.",
                 "zh": f"你在第{state['human']['x'] + 1}道，队友在第{state['ai']['x'] + 1}道。"},
                {"id": "decision", "en": actual["reason_en"], "zh": actual["reason_zh"]},
                {"id": "next_action", "en": "This task is finished; there is no next action." if state["terminal"] else "The teammate's next action is " + {"left": "move left", "right": "move right", "wait": "wait"}[actual["action"]] + ".",
                 "zh": "任务已经结束，没有下一步动作。" if state["terminal"] else "队友下一步将" + {"left": "左移", "right": "右移", "wait": "等待"}[actual["action"]] + "。"}]
    for ball in state["balls"]:
        lanes = ", ".join(str(x + 1) for x in ball["contacts"])
        evidence.append({"id": "ball:" + ball["id"],
            "en": f"{ball['id']} is a {'team' if ball['kind'] == 'cooperative' else 'ordinary'} ball worth {WEIGHTS[ball['kind']]} points, arriving at lane(s) {lanes} in {ball['remaining']} turns.",
            "zh": f"{ball['id']}是{'合作' if ball['kind'] == 'cooperative' else '普通'}球，值{WEIGHTS[ball['kind']]}分，将在{ball['remaining']}回合后到达第{lanes}道。"})
    evidence.extend({"id": f"comparison:{i}", "en": row["en"], "zh": row["zh"]} for i, row in enumerate(actual["alternatives"]))
    evidence.extend({"id": f"public_rule:{i}", "en": en, "zh": zh} for i, (en, zh) in enumerate(zip(rules(), rules("zh"))))
    evidence.extend({"id": f"event:{i}", "en": e["en"], "zh": e["zh"]} for i, e in enumerate(state["events"]))
    return evidence


def _search_key(obs: dict) -> str:
    return json.dumps(obs, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=60000)
def _solve_wave(key: str) -> tuple[int, tuple[str, ...]]:
    obs = json.loads(key)
    if not obs["balls"]:
        return 0, ()
    decision = _decision_from_observation(obs)
    candidates = []
    for action in legal_actions(obs):
        nxt, points, _ = _settle_visible(obs, action, decision)
        later, path = _solve_wave(_search_key(nxt))
        candidates.append((points + later, (action,) + path))
    # Prefer less unnecessary motion among equally scoring solutions.
    return max(candidates, key=lambda p: (p[0], -sum(a != "wait" for a in p[1]), tuple(-("wait", "left", "right").index(a) for a in p[1])))


def wave_feasibility(state: dict) -> dict:
    """Exact current-wave optimum with the actual AI and human-only control."""
    points, path = _solve_wave(_search_key(_observation(state)))
    return {"reachable_points": points, "public_wave_points": sum(WEIGHTS[b["kind"]] for b in state["balls"]),
            "human_actions": list(path), "scope": "current_visible_wave", "ai_policy": VERSION}


def human_advisor(state: dict) -> str:
    if state["terminal"]:
        return "wait"
    path = wave_feasibility(state)["human_actions"]
    return path[0] if path else "wait"


def greedy_human(state: dict) -> str:
    """Public-state nearest-arrival proxy; never controls or sees AI plans."""
    if not state["balls"]:
        return "wait"
    x = state["human"]["x"]
    ball = min(state["balls"], key=lambda b: (b["remaining"], b["id"]))
    return _move_towards(x, min(ball["contacts"], key=lambda at: (abs(x - at), at)))


def history_learning_human(state: dict, history: list[dict]) -> str:
    """A simple public-history proxy: learn AI side choices at past catches."""
    team = next((b for b in state["balls"] if b["kind"] == "cooperative"), None)
    if not team:
        return greedy_human(state)
    examples = [event for frame in history for event in frame.get("events", [])
                if event.get("kind") == "cooperative" and event.get("ai_x") in event.get("contacts", [])]
    if not examples:
        return greedy_human(state)
    right_count = sum(e["ai_x"] == max(e["contacts"]) for e in examples)
    ai_right = right_count * 2 >= len(examples)
    human_contact = min(team["contacts"]) if ai_right else max(team["contacts"])
    for ball in state["balls"]:
        if ball["kind"] != "ordinary":
            continue
        at = ball["contacts"][0]
        if (abs(state["human"]["x"] - at) <= ball["remaining"] and
                abs(at - human_contact) <= team["remaining"] - ball["remaining"]):
            return _move_towards(state["human"]["x"], at)
    return _move_towards(state["human"]["x"], human_contact)


def global_reachable_upper_bound(state: dict) -> dict:
    """Research-only exact upper bound using future waves, never a player aid.

    This offline verifier may know the fixed schedule. Its choices do not feed
    decide(), facts(), demonstration(), or human_advisor(). Branches retain the
    real fixed AI. Different scores at an identical physical state merge by max.
    """
    initial = deepcopy(state)
    frontier = {(_search_key(_observation(initial)), initial["wave"]): initial}
    for _ in range(initial["max_turns"] - initial["turn"]):
        next_frontier = {}
        for current in frontier.values():
            if current["terminal"]:
                choices = [current]
            else:
                decision = decide(current)
                choices = [step(current, a, decision) for a in legal_actions(current)]
            for nxt in choices:
                key = (_search_key(_observation(nxt)), nxt["wave"])
                if key not in next_frontier or nxt["raw_score"] > next_frontier[key]["raw_score"]:
                    next_frontier[key] = nxt
        frontier = next_frontier
        if all(s["terminal"] for s in frontier.values()):
            break
    best = max(frontier.values(), key=lambda s: s["raw_score"])
    return {"reachable_points": best["raw_score"], "task_score_upper_bound": score(best)["task_score"],
            "scope": "offline_full_schedule_upper_bound", "ai_policy": VERSION,
            "participant_visible": False}


def demonstration() -> dict:
    schedule = [[_ball(f"demo{i}-team", "cooperative", [2, 6], 4)] for i in range(1, 4)]
    schedule[0].append(_ball("demo1-small", "ordinary", [5], 2))
    state = _new_state(schedule, seed=0, task=1, human=2, ai=6)
    frames = [public_state(state)]
    for _ in range(4):
        state = step(state, "wait")
        frames.append(public_state(state))
    for action in ("right", "right", "wait", "wait"):
        state = step(state, action)
        frames.append(public_state(state))
    while not state["terminal"]:
        state = step(state, human_advisor(state))
        frames.append(public_state(state))
    captions = [
        (0, "Cover the falling balls' contact lanes. Each ball shows how many turns remain.", "覆盖球的接触位置。每个球显示还有几回合到达。"),
        (1, "Both paddles move first; then the balls count down by one. Waiting is also an action.", "双方先移动，随后球的剩余回合减一。等待也是一个动作。"),
        (2, "One paddle covered the ordinary ball's lane: the team earned 1 point.", "一块球拍覆盖普通球的位置，团队得到1分。"),
        (4, "Two paddles covered different sides together: the team ball earned 3 points.", "两块球拍同时覆盖不同两侧，合作球得到3分。"),
        (8, "One contact lane was uncovered when the next team ball arrived: it earned 0 points.", "下一颗合作球到达时有一侧未被覆盖，因此得到0分。"),
        (12, "The following wave used the same rules. Covering both sides together succeeded again.", "再下一波仍使用相同规则，同时覆盖两侧后再次成功。")]
    return {"frames": frames, "captions": [{"index": i, "en": en, "zh": zh} for i, en, zh in captions]}


def comprehension(language: str = "en") -> list[dict]:
    en = [
        {"id": "pong-next-move", "text": "There are four turns left. You are in lane 2 and your teammate is in lane 9. A team ball will contact lanes 3 and 7, with no other balls. What will your teammate do next?", "options": ["Move left", "Move right", "Wait"], "answer": 0},
        {"id": "pong-hold", "text": "You cover lane 3 and your teammate covers lane 7. A team ball arrives at those lanes in two turns, with no ordinary balls. Why would your teammate wait?", "options": ["Waiting adds bonus points", "Its assigned contact lane is already covered", "It has stopped responding"], "answer": 1},
        {"id": "pong-change", "text": "One turn remains. You are in lane 8 and your teammate is in lane 7. A team ball arrives at lanes 3 and 7, and an ordinary ball arrives at lane 8. What will your teammate do next?", "options": ["Move left toward lane 3", "Wait for an extra turn", "Move right to catch the ordinary ball"], "answer": 2}]
    zh = [
        {"id": "pong-next-move", "text": "还剩四回合，你在第2道，队友在第9道。合作球将在第3和第7道到达，没有其他球。队友下一步会怎么做？", "options": ["左移", "右移", "等待"], "answer": 0},
        {"id": "pong-hold", "text": "你覆盖第3道，队友覆盖第7道。合作球将在两回合后到达这两道，没有普通球。队友为什么等待？", "options": ["等待可以额外加分", "已经覆盖自己负责的接触位置", "已经停止响应"], "answer": 1},
        {"id": "pong-change", "text": "还剩一回合，你在第8道，队友在第7道。合作球将在第3和第7道到达，普通球将在第8道到达。队友下一步会怎么做？", "options": ["向第3道左移", "多等一回合", "右移去接普通球"], "answer": 2}]
    return zh if language.startswith("zh") else en
