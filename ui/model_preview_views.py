"""Exact carried display/event helpers; no legacy action selector is imported."""
import json
_ACTORS = {'human':'robot_1', 'ai':'robot_2'}

def _plain(value):
    """Canonical, finite JSON, also used to detach all returned structures."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

def _actor_view(agent):
    return {"x": agent.position[1], "y": agent.position[0], "battery": agent.battery,
            "carrying": agent.carrying_task_id, "heading": agent.heading.lower(), "active": agent.active}

def _orders(env):
    return [{"id": t.task_id, "owner": "shared", "pickup": list(reversed(t.pickup_position)),
             "dropoff": list(reversed(t.delivery_position)), "status": t.status,
             "carrier": next((k for k, v in _ACTORS.items() if v == t.carrier_agent_id), None)}
            for t in env.state.tasks]

def _events(before, after, info):
    rows = []
    def add(kind, en, zh, **more):
        rows.append({"type": kind, "en": en, "zh": zh, **more})
    if info.get("robot_collision_event") or after.robot_collision_events > before.robot_collision_events:
        animation = {"kind": info.get("robot_collision_kind") or after.last_robot_collision_kind}
        for actor, key in _ACTORS.items():
            animation[actor] = {"from": list(reversed(before.by_id(key).position)),
                                "attempted": list(reversed(info["intended_targets"][key])),
                                "to": list(reversed(after.by_id(key).position))}
        add("collision", "A robot collision cancelled the conflicting movement and cost 10 points.",
            "机器人碰撞取消了冲突移动，并扣除10分。", collision_animation=animation)
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
