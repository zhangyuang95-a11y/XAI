"""Warehouse-specific explanation vocabulary and deterministic verbalization."""

from __future__ import annotations

from typing import Any, Mapping


from .warehouse_context import (
    _action_zh,
    _goal_label,
)


class WarehouseExplanationMixin:
    """Natural-language semantics over observable warehouse evidence."""

    def explanation_entity_label(self, entity_id: str, language: str) -> str:
        if language == "zh-CN":
            return "机器人1" if entity_id == "robot_1" else "机器人2"
        return entity_id.replace("_", " ").title()

    def explanation_action_label(self, action: str, language: str) -> str:
        return _action_zh(action) if language == "zh-CN" else str(action).lower()

    def explanation_objective_label(self, objective: str, language: str) -> str:
        return _goal_label(objective, language)

    def concise_study_explanation(
        self,
        snapshot: Any,
        *,
        target_agent: str,
        policy: Any,
        focus: str,
        language: str,
    ) -> str:
        """Answer one study question directly in at most three sentences.

        All reasons are state facts available before the joint decision. Any
        movement, charging, or collision statement is explicitly an observed
        outcome of the simultaneous transition.
        """

        facts = tuple(self.evidence_facts(snapshot, target_agent, policy))
        by_predicate = {fact.predicate: fact.value for fact in facts}
        robot = self.explanation_entity_label(target_agent, language)
        action = str(by_predicate.get("executed_action", "WAIT"))
        action_label = self.explanation_action_label(action, language)
        objective = by_predicate.get("shared_objective_selection_reason", {})
        objective = objective if isinstance(objective, Mapping) else {}
        selected = objective.get("selected_objective", {})
        selected = selected if isinstance(selected, Mapping) else {}
        goal_kind = str(selected.get("id", "wait"))
        task_id = str(selected.get("task_id", "") or "")
        tasks = tuple(
            item
            for item in objective.get("active_shared_tasks", ())
            if isinstance(item, Mapping)
        )
        task_slot = next(
            (
                index
                for index, item in enumerate(tasks, start=1)
                if str(item.get("task_id", "")) == task_id
            ),
            None,
        )

        if language == "zh-CN":
            if goal_kind == "pickup" and task_slot:
                goal = f"任务{task_slot}的A取货点"
            elif goal_kind == "delivery" and task_slot:
                goal = f"任务{task_slot}的B交付点"
            elif goal_kind == "charge":
                goal = "充电站"
            else:
                goal = _goal_label(goal_kind, language)
            goal_sentence = f"决策前，{robot}的当前目标是{goal}。"
        else:
            if goal_kind == "pickup" and task_slot:
                goal = f"task {task_slot} pickup A"
            elif goal_kind == "delivery" and task_slot:
                goal = f"task {task_slot} drop-off B"
            elif goal_kind == "charge":
                goal = "the charger"
            else:
                goal = _goal_label(goal_kind, language)
            goal_sentence = f"Before deciding, {robot}'s current goal was {goal}."

        resolution = by_predicate.get("action_resolution_reason", {})
        resolution = resolution if isinstance(resolution, Mapping) else {}
        collision_kind = str(
            resolution.get("collision_kind")
            or resolution.get("blocked_reason")
            or ""
        )
        collision = collision_kind in {
            "same_target", "swap", "occupied_stationary", "robot_collision"
        }
        collaboration = by_predicate.get("collaboration_context", {})
        collaboration = collaboration if isinstance(collaboration, Mapping) else {}
        charger_clearance = bool(collaboration.get("charger_clearance", False))
        teammate_battery = float(collaboration.get("teammate_battery", 0.0) or 0.0)
        movement = by_predicate.get("movement_outcome", {})
        movement = movement if isinstance(movement, Mapping) else {}
        distance_before = int(movement.get("distance_before", 0) or 0)
        distance_after = int(movement.get("distance_after", distance_before) or 0)

        simultaneous_zh = (
            "两台机器人都只根据同一决策前状态同时选动作，"
            "机器人2事先不知道机器人1本帧会怎么走。"
        )
        simultaneous_en = (
            "Both robots chose simultaneously from the same pre-move state, "
            "so Robot 2 did not know Robot 1's current move."
        )
        if focus == "collision":
            collision_label_zh = {
                "same_target": "双方进入同一格",
                "swap": "双方交换位置",
                "occupied_stationary": "一方进入了队友本帧未离开的格子",
                "robot_collision": "机器人之间",
            }.get(collision_kind, "机器人之间")
            collision_label_en = {
                "same_target": "both robots targeted the same cell",
                "swap": "the robots tried to swap cells",
                "occupied_stationary": (
                    "one robot entered a cell the teammate did not leave"
                ),
                "robot_collision": "the robot paths conflicted",
            }.get(collision_kind, "the robot paths conflicted")
            if language == "zh-CN":
                first = (
                    f"是的，本帧因{collision_label_zh}发生碰撞，环境阻止了双方移动。"
                    if collision
                    else "本帧没有实际碰撞，但同时决策意味着双方仍存在同格或换位风险。"
                )
                return f"{first}{simultaneous_zh}{goal_sentence}"
            first = (
                f"Yes: {collision_label_en}, so the environment blocked both moves."
                if collision
                else "No collision occurred in this frame, but simultaneous choices still created same-cell or swap risk."
            )
            return f"{first} {simultaneous_en} {goal_sentence}"

        if charger_clearance:
            if language == "zh-CN":
                direct = (
                    f"{robot}{action_label}，执行后的结果是让出充电站，"
                    f"使低电量（{teammate_battery:g}%）的队友能够进入。"
                )
                return f"{direct}{goal_sentence}{simultaneous_zh}"
            direct = (
                f"{robot} moved {action_label}; the observed result was to vacate the charger "
                f"for its low-battery teammate ({teammate_battery:g}%)."
            )
            return f"{direct} {goal_sentence} {simultaneous_en}"

        if focus in {"energy", "charge_threshold"}:
            energy = by_predicate.get("energy_decision_context", {})
            energy = energy if isinstance(energy, Mapping) else {}
            battery = float(energy.get("battery", 0.0) or 0.0)
            requires_charge = bool(energy.get("requires_charge", False))
            if language == "zh-CN":
                direct = (
                    f"{robot}本帧{action_label}；决策前电量为{battery:g}%"
                    f"，当时{'需要充电' if requires_charge else '尚不需要充电'}。"
                )
                return f"{direct}{goal_sentence}"
            direct = (
                f"{robot} executed {action_label}; before deciding it had {battery:g}% battery and "
                f"{'needed to charge' if requires_charge else 'did not yet need to charge'}."
            )
            return f"{direct} {goal_sentence}"

        if language == "zh-CN":
            effect = (
                f"，执行后到目标的距离从{distance_before}格缩短到{distance_after}格"
                if distance_after < distance_before
                else ""
            )
            return f"{robot}本帧执行了{action_label}{effect}。{goal_sentence}"
        effect = (
            f", and after execution its distance to the goal fell from {distance_before} to {distance_after} cells"
            if distance_after < distance_before
            else ""
        )
        return f"{robot} executed {action_label}{effect}. {goal_sentence}"

    def explanation_predicate_schema(self) -> Mapping[str, Any]:
        return {
            "shared_objective_selection_reason": {
                "arguments": ("robot", "objective", "task", "frame"),
                "value_schema": "shared_objective_selection_reason.v2",
            },
            "executed_action": {"arguments": ("robot",), "unit": "action"},
            "proposed_action": {"arguments": ("robot",), "unit": "action"},
            "action_resolution_reason": {
                "arguments": ("robot", "requested_action", "executed_action"),
                "unit": "transition_reason",
            },
            "charging_outcome": {
                "arguments": ("robot",),
                "unit": "battery_points",
            },
            "charger_queue_context": {
                "arguments": ("robot", "charger_occupant"),
                "unit": "single_charger_queue",
            },
            "movement_outcome": {
                "arguments": ("robot",),
                "unit": "grid_path_progress",
            },
            "energy_decision_context": {
                "arguments": ("robot",),
                "unit": "battery_decision_context",
            },
            "collaboration_context": {
                "arguments": ("robot", "teammate"),
                "unit": "shared_task_assignment",
            },
            "battery": {"arguments": ("robot",), "unit": "percent"},
            "position": {"arguments": ("robot",), "unit": "grid_coordinate"},
            "shared_task": {"arguments": ("task",), "unit": "task_state"},
        }

    def explanation_verbalize_unit(
        self,
        unit: Mapping[str, Any],
        language: str,
    ) -> str:
        predicate = str(unit.get("predicate", "fact"))
        arguments = tuple(str(item) for item in unit.get("arguments", ()))
        value = unit.get("value")
        robot = self.explanation_entity_label(arguments[0], language) if arguments else ""
        if predicate in {
            "executed_action",
            "final_action",
            "proposed_action",
            "action_proposed",
            "action",
        }:
            raw_action = (
                value.get("action", "")
                if isinstance(value, Mapping)
                else value
            )
            action = self.explanation_action_label(str(raw_action), language)
            if predicate in {"proposed_action", "action_proposed"}:
                return (
                    f"{robot}原本选择了{action}"
                    if language == "zh-CN"
                    else f"{robot} originally selected {action}"
                )
            return (
                f"{robot}这一步执行了{action}"
                if language == "zh-CN"
                else f"{robot} executed {action} on this step"
            )
        if predicate == "charging_outcome" and isinstance(value, Mapping):
            before = float(value.get("battery_before", 0.0))
            after = float(value.get("battery_after", before))
            before_text = f"{before:g}"
            after_text = f"{after:g}"
            task = value.get("next_task", {})
            task = task if isinstance(task, Mapping) else {}
            task_id = str(task.get("task_id", ""))
            task_slot = task.get("task_slot")
            task_label = (
                f"任务{task_slot}"
                if language == "zh-CN" and task_slot
                else f"task {task_slot}"
                if task_slot
                else f"任务{task_id.removeprefix('task_')}"
                if language == "zh-CN" and task_id
                else task_id.replace("_", " ")
            )
            endpoint = task.get("endpoint")
            endpoint_text = str(tuple(endpoint)) if endpoint is not None else ""
            kind = str(task.get("kind", ""))
            charge_required = bool(value.get("charge_required", False))
            if language == "zh-CN":
                context = (
                    f"；充电完成后，它将继续把{task_label}的货物送往B点{endpoint_text}"
                    if kind == "delivery" and task_label and endpoint_text
                    else f"；按当前任务分配，充电完成后它将前往{task_label}的A点{endpoint_text}取货"
                    if kind == "pickup" and task_label and endpoint_text
                    else "；充电完成后，它将重新参与共享配送任务"
                )
                reason = (
                    "，不足以安全完成后续配送，因此它在充电站等待充电"
                    if charge_required
                    else "，在充电站等待使它能够补充电量"
                )
                return (
                    f"执行该动作时{robot}的电量为{before_text}%{reason}；"
                    f"本步后电量升至{after_text}%{context}"
                )
            context = (
                f"; after charging, it will continue carrying {task_label} to point B {endpoint_text}"
                if kind == "delivery" and task_label and endpoint_text
                else f"; under the current assignment, after charging it will collect {task_label} at point A {endpoint_text}"
                if kind == "pickup" and task_label and endpoint_text
                else "; after charging, it will rejoin the shared delivery tasks"
            )
            reason = (
                ", which was not enough to complete the remaining delivery safely, so it waited at the charger"
                if charge_required
                else ", and waiting at the charger replenished its battery"
            )
            return (
                f"{robot} had {before_text}% battery when it took this action{reason}; "
                f"its battery reached {after_text}% after the step{context}"
            )
        if predicate == "charger_queue_context" and isinstance(value, Mapping):
            occupant_id = str(value.get("occupant_agent", ""))
            occupant = self.explanation_entity_label(occupant_id, language)
            battery = float(value.get("battery", 0.0))
            position = tuple(value.get("position", ()))
            charger = tuple(value.get("charger_position", ()))
            occupant_before = float(value.get("occupant_battery_before", 0.0))
            occupant_after = float(
                value.get("occupant_battery_after", occupant_before)
            )
            if language == "zh-CN":
                return (
                    f"{robot}本步等待，因为它的电量仅有{battery:g}%，已经需要充电；"
                    f"但唯一的充电站{charger}正由{occupant}占用。{occupant}本步在站内"
                    f"等待充电，电量从{occupant_before:g}%升至{occupant_after:g}%，"
                    f"因此{robot}暂时不能进入，只能在{position}等待充电站空出"
                )
            return (
                f"{robot} waited because its battery was only {battery:g}% and it needed "
                f"to charge, but the only charger at {charger} was occupied by {occupant}. "
                f"{occupant} waited on the charger and its battery rose from "
                f"{occupant_before:g}% to {occupant_after:g}%, so {robot} could not enter "
                f"yet and had to wait at {position} for the charger to become free"
            )
        if predicate == "movement_outcome" and isinstance(value, Mapping):
            before_distance = int(value.get("distance_before", 0))
            after_distance = int(value.get("distance_after", before_distance))
            action = str(value.get("action", ""))
            action_label = self.explanation_action_label(action, language)
            selected_probability = float(value.get("selected_probability", 0.0))
            policy_selected = bool(value.get("policy_selected", False))
            work = value.get("work", {})
            work = work if isinstance(work, Mapping) else {}
            kind = str(work.get("kind", "reposition"))
            task_id = str(work.get("task_id", ""))
            task_slot = work.get("task_slot")
            task_label = (
                f"任务{task_slot}"
                if language == "zh-CN" and task_slot
                else f"task {task_slot}"
                if task_slot
                else f"任务{task_id.removeprefix('task_')}"
                if language == "zh-CN" and task_id
                else task_id.replace("_", " ")
            )
            endpoint = work.get("endpoint")
            endpoint_text = str(tuple(endpoint)) if endpoint is not None else ""
            before_position = tuple(value.get("position_before", ()))
            after_position = tuple(value.get("position_after", ()))
            if language == "zh-CN":
                if kind in {"pickup", "delivery"} and task_label and endpoint_text:
                    endpoint_kind = "A" if kind == "pickup" else "B"
                    purpose = "取货" if kind == "pickup" else "交付"
                    if after_distance < before_distance:
                        return (
                            f"它从{before_position}移动到{after_position}，使到{task_label}的"
                            f"{endpoint_kind}点{endpoint_text}的"
                            f"剩余距离从{before_distance}格缩短到{after_distance}格，"
                            f"从而推进{purpose}"
                        )
                    if policy_selected:
                        return (
                            f"它本帧执行了{action_label}，从{before_position}移动到"
                            f"{after_position}；执行后这一步没有缩短到"
                            f"{task_label}的{endpoint_kind}点"
                            f"{endpoint_text}的路线，剩余距离从{before_distance}格变为"
                            f"{after_distance}格"
                        )
                    return (
                        f"它从{before_position}移动到{after_position}；这一步没有缩短到"
                        f"{task_label}的{endpoint_kind}点{endpoint_text}"
                        f"的路线，剩余距离从{before_distance}格变为{after_distance}格；"
                        "现有可验证证据不能支持更具体的动作原因"
                    )
                if kind == "charge":
                    battery = float(work.get("battery_before", 0.0))
                    battery_text = f"{battery:g}"
                    next_task = work.get("next_task", {})
                    next_task = next_task if isinstance(next_task, Mapping) else {}
                    next_id = str(next_task.get("task_id", ""))
                    next_slot = next_task.get("task_slot")
                    next_label = (
                        f"任务{next_slot}"
                        if next_slot
                        else f"任务{next_id.removeprefix('task_')}"
                        if next_id
                        else "后续共享任务"
                    )
                    next_endpoint = next_task.get("endpoint")
                    next_endpoint_text = (
                        str(tuple(next_endpoint))
                        if next_endpoint is not None
                        else ""
                    )
                    next_kind = str(next_task.get("kind", ""))
                    following = (
                        f"，充电后继续将{next_label}送往B点{next_endpoint_text}"
                        if next_kind == "delivery" and next_endpoint_text
                        else f"，充电后前往{next_label}的A点{next_endpoint_text}取货"
                        if next_kind == "pickup" and next_endpoint_text
                        else "，充电后继续参与共享配送"
                    )
                    return (
                        f"此时它的电量为{battery_text}%；它从{before_position}移动到"
                        f"{after_position}，使到充电站的"
                        f"剩余距离从{before_distance}格缩短到{after_distance}格"
                        f"{following}"
                    )
                return f"这样做使它从{before_position}移动到{after_position}"
            if kind in {"pickup", "delivery"} and task_label and endpoint_text:
                endpoint_kind = "A" if kind == "pickup" else "B"
                purpose = "pickup" if kind == "pickup" else "delivery"
                if after_distance < before_distance:
                    return (
                        f"It moved from {before_position} to {after_position}, reducing "
                        f"the remaining distance to {task_label} point "
                        f"{endpoint_kind} {endpoint_text} from {before_distance} to "
                        f"{after_distance} cells, advancing {purpose}"
                    )
                if policy_selected:
                    return (
                        f"It executed {action_label}, moving from {before_position} to "
                        f"{after_position}; after execution this step "
                        f"did not shorten the route to {task_label} "
                        f"point {endpoint_kind} {endpoint_text}, changing the remaining "
                        f"distance from {before_distance} to {after_distance} cells"
                    )
                return (
                    f"It moved from {before_position} to {after_position}; this step did "
                    f"not shorten the route to {task_label} point "
                    f"{endpoint_kind} {endpoint_text}; the remaining distance changed "
                    f"from {before_distance} to {after_distance} cells, and the verified "
                    "evidence does not support a more specific reason for the action"
                )
            if kind == "charge":
                battery = float(work.get("battery_before", 0.0))
                return (
                    f"With {battery:g}% battery, it moved from {before_position} to "
                    f"{after_position}, reducing the remaining distance "
                    f"to the charger from {before_distance} to {after_distance} cells"
                )
            return (
                f"This moved it from {tuple(value.get('position_before', ())) } "
                f"to {tuple(value.get('position_after', ())) }"
            )
        if predicate == "energy_decision_context" and isinstance(value, Mapping):
            battery = float(value.get("battery", 0.0))
            move_cost = float(value.get("move_battery_cost", 0.0))
            required = value.get("required_safe_energy")
            minimum_departure = value.get("minimum_safe_departure_battery")
            waits_remaining = int(value.get("charge_waits_remaining", 0))
            projected_departure = float(
                value.get("projected_departure_battery", battery)
            )
            requires_charge = bool(value.get("requires_charge", False))
            study_focus = str(value.get("study_focus", ""))
            action = self.explanation_action_label(
                str(value.get("executed_action", "")), language
            )
            role = value.get("task_role", {})
            role = role if isinstance(role, Mapping) else {}
            slot = role.get("task_slot")
            endpoint = role.get("endpoint")
            endpoint_text = str(tuple(endpoint)) if endpoint is not None else ""
            if study_focus == "charge_threshold":
                if language == "zh-CN":
                    next_work = (
                        f"随后把任务{slot}送到B点{endpoint_text}"
                        if role.get("kind") == "delivery" and slot
                        else f"随后前往任务{slot}的A点{endpoint_text}取货"
                        if role.get("kind") == "pickup" and slot
                        else "随后继续参与共享配送"
                    )
                    if minimum_departure is None:
                        return (
                            f"{robot}当前电量为{battery:g}%。当前没有可核实的已分配"
                            "配送路线，因此不能给出一个具体的最低离站电量"
                        )
                    if waits_remaining > 0:
                        return (
                            f"{robot}当前电量为{battery:g}%，安全离开充电站至少需要"
                            f"{float(minimum_departure):g}%（包括完成路线并保留2格余量）。"
                            f"它还需等待充电{waits_remaining}次，每次增加"
                            f"{float(value.get('charge_per_wait', 0.0)):g}点；预计达到"
                            f"{projected_departure:g}%后离开，{next_work}"
                        )
                    return (
                        f"{robot}当前电量为{battery:g}%，已达到安全离站所需的最低"
                        f"{float(minimum_departure):g}%，无需继续等待充电；{next_work}"
                    )
                next_work = (
                    f"then deliver task {slot} at B {endpoint_text}"
                    if role.get("kind") == "delivery" and slot
                    else f"then collect task {slot} at A {endpoint_text}"
                    if role.get("kind") == "pickup" and slot
                    else "then rejoin the shared delivery work"
                )
                if minimum_departure is None:
                    return (
                        f"{robot} currently had {battery:g}% battery. There was no "
                        "verifiable assigned route, so a specific minimum departure "
                        "battery cannot be stated"
                    )
                if waits_remaining > 0:
                    return (
                        f"{robot} had {battery:g}% battery and needed at least "
                        f"{float(minimum_departure):g}% before safely leaving the charger "
                        "(including the two-cell reserve). It needed "
                        f"{waits_remaining} more charging wait"
                        f"{'s' if waits_remaining != 1 else ''} at "
                        f"{float(value.get('charge_per_wait', 0.0)):g} points each, "
                        f"reaching about {projected_departure:g}%; it would {next_work}"
                    )
                return (
                    f"{robot} had {battery:g}% battery, already meeting the minimum safe "
                    f"departure level of {float(minimum_departure):g}%. It did not need "
                    f"another charging wait and could {next_work}"
                )
            if language == "zh-CN":
                task_context = (
                    f"完成任务{slot}到B点{endpoint_text}的安全路线"
                    if role.get("kind") == "delivery" and slot and endpoint_text
                    else f"完成任务{slot}从A点{endpoint_text}开始的安全路线"
                    if role.get("kind") == "pickup" and slot and endpoint_text
                    else "继续安全配送"
                )
                if requires_charge:
                    required_text = (
                        f"，而{task_context}预计需要约{float(required):g}点电量"
                        if required is not None
                        else ""
                    )
                    return (
                        f"执行{action}时{robot}的当前电量为{battery:g}%{required_text}；"
                        "电量不足以安全完成后续路线，因此充电需求确实影响了这个决定"
                    )
                required_text = (
                    f"，{task_context}预计需要约{float(required):g}点电量"
                    if required is not None
                    else ""
                )
                return (
                    f"执行{action}时{robot}的当前电量为{battery:g}%{required_text}；"
                    f"当前电量足够，且每次成功移动只消耗{move_cost:g}点，"
                    "所以电量没有迫使它转去充电，本步主要由配送与协作需要决定"
                )
            task_context = (
                f"the safe route for task {slot} to point B {endpoint_text}"
                if role.get("kind") == "delivery" and slot and endpoint_text
                else f"the safe route for task {slot} beginning at point A {endpoint_text}"
                if role.get("kind") == "pickup" and slot and endpoint_text
                else "continued safe delivery"
            )
            if requires_charge:
                required_text = (
                    f", while {task_context} required about {float(required):g} battery points"
                    if required is not None
                    else ""
                )
                return (
                    f"{robot} had {battery:g}% battery when it executed {action}"
                    f"{required_text}; the battery was insufficient for the remaining safe "
                    "route, so charging needs did affect this decision"
                )
            required_text = (
                f", and {task_context} required about {float(required):g} battery points"
                if required is not None
                else ""
            )
            return (
                f"{robot} had {battery:g}% battery when it executed {action}"
                f"{required_text}; this was sufficient and each successful move costs "
                f"{move_cost:g}, so battery did not force a charging detour and the step "
                "was mainly determined by delivery and coordination needs"
            )
        if predicate == "collaboration_context" and isinstance(value, Mapping):
            teammate_id = str(value.get("teammate_agent", ""))
            teammate = self.explanation_entity_label(teammate_id, language)
            target_role = value.get("target_role", {})
            target_role = target_role if isinstance(target_role, Mapping) else {}
            teammate_role = value.get("teammate_role", {})
            teammate_role = teammate_role if isinstance(teammate_role, Mapping) else {}

            def role_text(owner: str, role: Mapping[str, Any]) -> str:
                kind = str(role.get("kind", "wait"))
                slot = role.get("task_slot")
                endpoint = role.get("endpoint")
                endpoint_text = str(tuple(endpoint)) if endpoint is not None else ""
                if language == "zh-CN":
                    if kind == "pickup" and slot:
                        return f"{owner}前往任务{slot}的A点{endpoint_text}取货"
                    if kind == "delivery" and slot:
                        return f"{owner}已经承运任务{slot}，正在前往B点{endpoint_text}交付"
                    if kind == "charge":
                        return f"{owner}需要先处理充电"
                    return f"{owner}当前等待"
                if kind == "pickup" and slot:
                    return f"{owner} is collecting task {slot} at point A {endpoint_text}"
                if kind == "delivery" and slot:
                    return f"{owner} is already carrying task {slot} to point B {endpoint_text}"
                if kind == "charge":
                    return f"{owner} needs to recharge first"
                return f"{owner} is currently waiting"

            target_text = role_text(robot, target_role)
            teammate_text = role_text(teammate, teammate_role)
            selected_cost = value.get("joint_selected_safe_actions")
            swapped_cost = value.get("joint_swapped_safe_actions")
            selected_breakdown = value.get("joint_selected_breakdown")
            selected_breakdown = (
                selected_breakdown if isinstance(selected_breakdown, Mapping) else None
            )
            swapped_breakdown = value.get("joint_swapped_breakdown")
            swapped_breakdown = (
                swapped_breakdown if isinstance(swapped_breakdown, Mapping) else None
            )

            def route_breakdown_text(
                breakdown: Mapping[str, Any],
                *,
                current: bool,
            ) -> str:
                assignment_texts: list[str] = []
                for raw_entry in breakdown.get("assignments", ()):
                    if not isinstance(raw_entry, Mapping):
                        continue
                    owner = self.explanation_entity_label(
                        str(raw_entry.get("agent_id", "")),
                        language,
                    )
                    slot = raw_entry.get("task_slot")
                    leg_texts: list[str] = []
                    for raw_leg in raw_entry.get("route_legs", ()):
                        if not isinstance(raw_leg, Mapping):
                            continue
                        kind = str(raw_leg.get("kind", ""))
                        cells = int(raw_leg.get("cells", 0))
                        if language == "zh-CN":
                            leg_label = {
                                "current_to_pickup": "当前位置到A点",
                                "pickup_to_delivery": "A点到B点",
                                "current_to_delivery": "当前位置到B点",
                                "current_to_charger": "当前位置到充电站",
                                "charger_to_pickup": "充电站到A点",
                                "charger_to_delivery": "充电站到B点",
                                "delivery_to_charger": "B点到充电站",
                            }.get(kind, "路线")
                            leg_texts.append(f"{leg_label}{cells}格")
                        else:
                            leg_label = {
                                "current_to_pickup": "current position to A",
                                "pickup_to_delivery": "A to B",
                                "current_to_delivery": "current position to B",
                                "current_to_charger": "current position to charger",
                                "charger_to_pickup": "charger to A",
                                "charger_to_delivery": "charger to B",
                                "delivery_to_charger": "B to charger",
                            }.get(kind, "route")
                            leg_texts.append(f"{leg_label}: {cells} cells")
                    travel = int(raw_entry.get("travel_cells", 0))
                    waits = int(raw_entry.get("charge_waits", 0))
                    battery = float(raw_entry.get("current_battery", 0.0))
                    if language == "zh-CN":
                        charge_text = (
                            f"需要额外等待充电{waits}次"
                            if waits
                            else "无需额外充电等待"
                        )
                        assignment_texts.append(
                            f"{owner}负责任务{slot}：{'、'.join(leg_texts)}，共{travel}格"
                            f"（当前电量{battery:g}%，{charge_text}）"
                        )
                    else:
                        charge_text = (
                            f"{waits} additional charging waits"
                            if waits
                            else "no additional charging wait"
                        )
                        assignment_texts.append(
                            f"{owner} on task {slot}: {', '.join(leg_texts)}, {travel} cells "
                            f"in total (current battery {battery:g}%, {charge_text})"
                        )
                total_travel = int(breakdown.get("total_travel_cells", 0))
                total_waits = int(breakdown.get("total_charge_waits", 0))
                if language == "zh-CN":
                    label = "当前分工" if current else "交换任务后"
                    wait_summary = (
                        f"，另需必要充电等待{total_waits}次"
                        if total_waits
                        else "，无需额外充电等待"
                    )
                    return (
                        f"{label}：{'；'.join(assignment_texts)}；团队合计行驶"
                        f"{total_travel}格{wait_summary}"
                    )
                label = "Current assignment" if current else "After swapping tasks"
                wait_summary = (
                    f", plus {total_waits} necessary charging waits"
                    if total_waits
                    else ", with no additional charging wait"
                )
                return (
                    f"{label}: {'; '.join(assignment_texts)}; team travel totals "
                    f"{total_travel} cells{wait_summary}"
                )

            numeric_comparison = None
            if selected_breakdown is not None and swapped_breakdown is not None:
                selected_travel = int(selected_breakdown.get("total_travel_cells", 0))
                swapped_travel = int(swapped_breakdown.get("total_travel_cells", 0))
                selected_waits = int(selected_breakdown.get("total_charge_waits", 0))
                swapped_waits = int(swapped_breakdown.get("total_charge_waits", 0))
                details = (
                    f"{route_breakdown_text(selected_breakdown, current=True)}。"
                    f"{route_breakdown_text(swapped_breakdown, current=False)}。"
                )
                if language == "zh-CN":
                    if selected_waits == swapped_waits and selected_travel < swapped_travel:
                        conclusion = f"因此当前分工少走{swapped_travel - selected_travel}格"
                    elif selected_waits < swapped_waits:
                        conclusion = (
                            f"因此当前分工少{swapped_waits - selected_waits}次必要充电等待"
                        )
                    elif selected_waits == swapped_waits and selected_travel == swapped_travel:
                        conclusion = "两种分工的预计行驶距离和充电等待次数相同"
                    else:
                        conclusion = "综合行驶距离与必要充电等待，系统保留当前分工"
                    numeric_comparison = (
                        f"{details}以上行驶距离包括取货、交付以及交付后前往充电站。"
                        f"{conclusion}"
                    )
                else:
                    if selected_waits == swapped_waits and selected_travel < swapped_travel:
                        conclusion = (
                            f"the current assignment therefore saves "
                            f"{swapped_travel - selected_travel} travel cells"
                        )
                    elif selected_waits < swapped_waits:
                        conclusion = (
                            f"the current assignment therefore saves "
                            f"{swapped_waits - selected_waits} necessary charging waits"
                        )
                    elif selected_waits == swapped_waits and selected_travel == swapped_travel:
                        conclusion = (
                            "the two assignments have the same estimated travel and charging waits"
                        )
                    else:
                        conclusion = (
                            "considering both travel distance and necessary charging waits, "
                            "the system keeps the current assignment"
                        )
                    numeric_comparison = (
                        f"{details}These distances include collection, delivery, and travel "
                        f"from B to the charger after delivery. Therefore, {conclusion}"
                    )
            study_focus = str(value.get("study_focus", ""))
            constrained = tuple(
                self.explanation_action_label(str(item), language)
                for item in value.get("teammate_constrained_actions", ())
            )
            teammate_position = tuple(value.get("teammate_position", ()))
            teammate_action = self.explanation_action_label(
                str(value.get("teammate_executed_action", "WAIT")),
                language,
            )
            executed_action = self.explanation_action_label(
                str(value.get("executed_action", "WAIT")),
                language,
            )
            enabled_teammate_action = bool(
                value.get("enabled_teammate_action", False)
            )
            charger_clearance = bool(value.get("charger_clearance", False))
            if enabled_teammate_action:
                target_position = tuple(value.get("target_position", ()))
                target_position_after = tuple(
                    value.get("target_position_after", ())
                )
                teammate_target = tuple(
                    value.get("teammate_target_position", ())
                )
                teammate_battery = float(value.get("teammate_battery", 0.0))
                before_distance = value.get("target_distance_before")
                after_distance = value.get("target_distance_after")
                route_effect = ""
                if before_distance is not None and after_distance is not None:
                    before_distance = int(before_distance)
                    after_distance = int(after_distance)
                    if language == "zh-CN":
                        if after_distance > before_distance:
                            route_effect = (
                                f"；这使{robot}到自身长期任务目标的距离从"
                                f"{before_distance}格增加到{after_distance}格，属于为队友"
                                "进行的协作让行"
                            )
                        else:
                            route_effect = (
                                f"；本步后{robot}到自身长期任务目标的距离从"
                                f"{before_distance}格变为{after_distance}格"
                            )
                    elif after_distance > before_distance:
                        route_effect = (
                            f"; this increased {robot}'s distance to its own longer-term "
                            f"task target from {before_distance} to {after_distance} cells, "
                            "so the move was a cooperative yield for its teammate"
                        )
                    else:
                        route_effect = (
                            f"; after the move, {robot}'s distance to its own longer-term "
                            f"task target changed from {before_distance} to "
                            f"{after_distance} cells"
                        )
                if language == "zh-CN":
                    if charger_clearance:
                        return (
                            f"本帧的直接协作原因是让出充电站：{teammate}的电量仅剩"
                            f"{teammate_battery:g}%，正准备进入{robot}占用的充电位"
                            f"{teammate_target}；{robot}因此从{target_position}执行"
                            f"{executed_action}到{target_position_after}，使{teammate}能够"
                            f"进入充电站并准备充电{route_effect}"
                        )
                    return (
                        f"本帧的直接协作原因是为{teammate}腾出位置{target_position}："
                        f"{robot}执行{executed_action}到{target_position_after}，使"
                        f"{teammate}能够移动到{teammate_target}{route_effect}"
                    )
                if charger_clearance:
                    return (
                        f"The immediate coordination reason was to clear the charger: "
                        f"{teammate} had only {teammate_battery:g}% battery and was moving "
                        f"into the charging cell {teammate_target} occupied by {robot}; "
                        f"{robot} therefore moved {executed_action} from {target_position} "
                        f"to {target_position_after}, allowing {teammate} to enter the "
                        f"charger and prepare to charge{route_effect}"
                    )
                return (
                    f"The immediate coordination reason was to clear {target_position} "
                    f"for {teammate}: {robot} moved {executed_action} to "
                    f"{target_position_after}, allowing {teammate} to move to "
                    f"{teammate_target}{route_effect}"
                )
            if study_focus == "collaboration":
                directly_limited = bool(
                    value.get("teammate_directly_limited_action", False)
                )
                if language == "zh-CN":
                    if directly_limited:
                        simultaneous = (
                            f"{teammate}本步的{teammate_action}与该动作同时执行，"
                            "不能被当作事先已经腾出的空间"
                            if str(value.get("teammate_executed_action", "WAIT")) != "WAIT"
                            else f"{teammate}本步等待，因此该位置不会腾出"
                        )
                        return (
                            f"有直接影响。做出本步决定时，{teammate}位于"
                            f"{teammate_position}，使{robot}原本提议的"
                            f"{self.explanation_action_label(str(value.get('proposed_action', 'WAIT')), language)}"
                            f"不可执行；{robot}实际执行了{executed_action}。{simultaneous}"
                        )
                    return (
                        f"没有直接影响。做出本步决定时，{teammate}位于"
                        f"{teammate_position}并执行{teammate_action}，没有占用或冲突于"
                        f"{robot}执行{executed_action}所需的格子"
                    )
                if directly_limited:
                    simultaneous = (
                        f"{teammate}'s {teammate_action} happened simultaneously and could "
                        "not be treated as space already vacated"
                        if str(value.get("teammate_executed_action", "WAIT")) != "WAIT"
                        else f"{teammate} waited, so that cell did not become free"
                    )
                    return (
                        f"Yes, directly. At decision time {teammate} was at "
                        f"{teammate_position}, making {robot}'s proposed "
                        f"{self.explanation_action_label(str(value.get('proposed_action', 'WAIT')), language)} unavailable; "
                        f"{robot} therefore executed {executed_action}. "
                        f"{simultaneous}"
                    )
                return (
                    f"No, not directly. At decision time {teammate} was at "
                    f"{teammate_position} and executed {teammate_action}; this neither "
                    f"occupied nor conflicted with the cell needed for {robot}'s "
                    f"{executed_action}"
                )
            if study_focus == "task":
                kind = str(target_role.get("kind", "wait"))
                slot = target_role.get("task_slot")
                endpoint = target_role.get("endpoint")
                endpoint_text = str(tuple(endpoint)) if endpoint is not None else ""
                if language == "zh-CN":
                    if kind == "delivery" and slot:
                        return (
                            f"本步{executed_action}没有完成交付；{robot}仍承运任务{slot}，"
                            f"后续需要到B点{endpoint_text}交付"
                        )
                    if kind == "pickup" and slot:
                        return (
                            f"本步{executed_action}没有完成取货；{robot}仍需前往任务{slot}"
                            f"的A点{endpoint_text}认领货物"
                        )
                    return f"本步{executed_action}没有改变当前配送任务状态"
                if kind == "delivery" and slot:
                    return (
                        f"The {executed_action} did not complete a delivery; {robot} still "
                        f"carries task {slot} and must deliver it at B {endpoint_text}"
                    )
                if kind == "pickup" and slot:
                    return (
                        f"The {executed_action} did not complete a pickup; {robot} still "
                        f"needs to claim task {slot} at A {endpoint_text}"
                    )
                return f"The {executed_action} did not change the current delivery-task state"
            if language == "zh-CN":
                if selected_cost is not None and swapped_cost is not None:
                    if numeric_comparison is not None:
                        allocation_reason = numeric_comparison
                    elif float(selected_cost) < float(swapped_cost):
                        allocation_reason = (
                            "系统比较了两种分工所需的取货、交付路线以及必要充电；"
                            "当前分工让团队总体少走路，因此没有交换两台机器人的任务"
                        )
                    else:
                        allocation_reason = (
                            "系统比较了两种分工所需的取货、交付路线以及必要充电；"
                            "两种分工同样合适，因此按固定顺序采用了当前分工"
                        )
                elif target_role.get("kind") == "pickup" and teammate_role.get("kind") == "delivery":
                    allocation_reason = (
                        f"由于{teammate}已经携带一件货物、不能再认领另一项任务，"
                        f"剩余可认领任务由{robot}负责"
                    )
                elif target_role.get("kind") == "delivery" and teammate_role.get("kind") == "pickup":
                    allocation_reason = (
                        f"由于{robot}已经携带一件货物、不能再去取另一项任务的A点，"
                        f"该取货任务由{teammate}负责"
                    )
                elif target_role.get("kind") == "charge":
                    allocation_reason = (
                        f"{robot}此时不能安全承担新的配送路线，因此可认领任务优先由"
                        f"{teammate}处理"
                    )
                else:
                    allocation_reason = "两台机器人依据当前承运状态与安全路线共同分工"
                if study_focus == "allocation":
                    return f"{target_text}；{teammate_text}。{allocation_reason}"
                physical = (
                    f"队友的位置或预计动作还会限制{ '、'.join(constrained) }，"
                    "因此直接影响了本步可选动作"
                    if constrained
                    else "队友没有直接阻挡本步实际动作；它的主要影响体现在共享任务分工"
                )
                return f"当前协作状态是：{target_text}；{teammate_text}。{allocation_reason}；{physical}"
            if selected_cost is not None and swapped_cost is not None:
                if numeric_comparison is not None:
                    allocation_reason = numeric_comparison
                elif float(selected_cost) < float(swapped_cost):
                    allocation_reason = (
                        "the system compared the pickup and delivery routes, including any "
                        "necessary charging, for both assignments; the current assignment "
                        "requires less total travel, so the robots did not swap tasks"
                    )
                else:
                    allocation_reason = (
                        "the system compared the pickup and delivery routes, including any "
                        "necessary charging, for both assignments; they were equally suitable, "
                        "so the fixed tie-breaking order kept the current assignment"
                    )
            elif target_role.get("kind") == "pickup" and teammate_role.get("kind") == "delivery":
                allocation_reason = (
                    f"because {teammate} already carries one item and cannot claim another, "
                    f"the remaining available task is assigned to {robot}"
                )
            elif target_role.get("kind") == "delivery" and teammate_role.get("kind") == "pickup":
                allocation_reason = (
                    f"because {robot} already carries one item and cannot collect another A "
                    f"point, that pickup is assigned to {teammate}"
                )
            elif target_role.get("kind") == "charge":
                allocation_reason = (
                    f"{robot} cannot safely take a new delivery route yet, so available work "
                    f"is handled by {teammate} first"
                )
            else:
                allocation_reason = "the robots divide work using their carrying state and safe routes"
            if study_focus == "allocation":
                return f"{target_text}; {teammate_text}. {allocation_reason}"
            physical = (
                f"the teammate's position or predicted action also constrained "
                f"{', '.join(constrained)}, directly affecting the feasible actions"
                if constrained
                else "the teammate did not directly block the executed action; its main effect was the shared-task assignment"
            )
            return f"The collaboration state was: {target_text}; {teammate_text}. {allocation_reason}; {physical}"
        if predicate in {"action_constraint", "executed_path_condition"}:
            details = value if isinstance(value, Mapping) else {}
            selected = self.explanation_action_label(
                str(details.get("selected_action", "")), language
            )
            constrained = self.explanation_action_label(
                str(details.get("constrained_action", "")), language
            )
            meaning = details.get("observed_meaning", {})
            meaning = meaning if isinstance(meaning, Mapping) else {}
            reason_features = tuple(
                str(item)
                for item in details.get(
                    "active_reason_features",
                    meaning.get("reason_features", ()),
                )
            )
            if any(item.endswith("blocked_by_static_obstacle") for item in reason_features):
                return (
                    f"{constrained}会被墙或货架阻挡"
                    if language == "zh-CN"
                    else f"moving {constrained} was blocked by a wall or shelf"
                )
            if any(item.endswith("blocked_by_robot") for item in reason_features):
                return (
                    f"{constrained}会被队友占据的格子阻挡"
                    if language == "zh-CN"
                    else f"moving {constrained} was blocked by the teammate's cell"
                )
            if any(item.endswith("predicted_same_cell_conflict") for item in reason_features):
                return (
                    f"{constrained}可能让两台机器人进入同一格"
                    if language == "zh-CN"
                    else f"moving {constrained} could send both robots into the same cell"
                )
            if any(item.endswith("predicted_swap_conflict") for item in reason_features):
                return (
                    f"{constrained}可能造成两台机器人交换位置冲突"
                    if language == "zh-CN"
                    else f"moving {constrained} could cause a robot swap conflict"
                )
            role = str(meaning.get("explanation_role", ""))
            progress = float(details.get("geometric_goal_progress", 0.0) or 0.0)
            if role == "action_objective_effect" and progress > 0.0:
                return (
                    f"{selected}能够推进当前配送任务"
                    if language == "zh-CN"
                    else f"{selected} advanced the current delivery task"
                )
            if role == "action_route_efficiency":
                return (
                    f"{selected}减少了不必要的绕路"
                    if language == "zh-CN"
                    else f"{selected} reduced unnecessary detour"
                )
            return (
                f"当前可见条件更支持{selected}"
                if language == "zh-CN"
                else f"the visible conditions supported {selected}"
            )
        if predicate == "battery":
            return f"{robot}电量为{value}%" if language == "zh-CN" else f"{robot} battery was {value}%"
        if predicate == "position":
            return f"{robot}位于{value}" if language == "zh-CN" else f"{robot} was at {value}"
        if predicate == "action_resolution_reason" and isinstance(value, Mapping):
            requested = self.explanation_action_label(
                str(value.get("requested_action", "WAIT")), language
            )
            executed = self.explanation_action_label(
                str(value.get("executed_action", "WAIT")), language
            )
            teammate_requested = self.explanation_action_label(
                str(value.get("teammate_requested_action", "WAIT")), language
            )
            target = tuple(value.get("intended_target", ()))
            teammate_target = tuple(value.get("teammate_intended_target", ()))
            kind = str(value.get("collision_kind") or value.get("blocked_reason") or "")
            study_focus = str(value.get("study_focus", ""))
            if study_focus == "collision" and kind not in {
                "same_target", "swap", "occupied_stationary", "robot_collision"
            }:
                return (
                    f"没有直接的机器人冲突。{robot}提议并实际执行了{executed}，"
                    f"其目标格{target}没有与队友的目标格{teammate_target}发生同格、"
                    "交换或占位冲突"
                    if language == "zh-CN"
                    else f"There was no direct robot conflict. {robot} proposed and "
                    f"executed {executed}; its target {target} did not create a same-cell, "
                    f"swap, or occupied-cell conflict with the teammate's target "
                    f"{teammate_target}"
                )
            if language == "zh-CN":
                kind_text = {
                    "same_target": "双方争抢同一目标格",
                    "swap": "双方试图交换位置",
                    "occupied_stationary": "目标格仍由等待的队友占用",
                    "static_obstacle": "目标格是墙或货架",
                }.get(kind, "该动作受到环境约束")
                return (
                    f"{robot}提议{requested}、目标格为{target}；队友提议"
                    f"{teammate_requested}、目标格为{teammate_target}。{kind_text}，"
                    f"因此{robot}实际执行{executed}"
                )
            kind_text = {
                "same_target": "both robots targeted the same cell",
                "swap": "the robots attempted to swap positions",
                "occupied_stationary": "the target cell remained occupied by the waiting teammate",
                "static_obstacle": "the target cell was a wall or shelf",
            }.get(kind, "the action was constrained by the environment")
            return (
                f"{robot} proposed {requested} toward {target}; its teammate proposed "
                f"{teammate_requested} toward {teammate_target}. {kind_text}, so "
                f"{robot} actually executed {executed}"
            )
        if predicate == "shared_objective_selection_reason" and isinstance(value, Mapping):
            selected = value.get("selected_objective", {})
            selected = selected if isinstance(selected, Mapping) else {}
            objective = str(selected.get("id", "wait"))
            task_id = str(selected.get("task_id", "") or "")
            target = selected.get("target_position")
            target_text = str(tuple(target)) if target is not None else ""
            tasks = tuple(
                item
                for item in value.get("active_shared_tasks", ())
                if isinstance(item, Mapping)
            )
            task_slot = next(
                (
                    index
                    for index, item in enumerate(tasks, start=1)
                    if str(item.get("task_id", "")) == task_id
                ),
                None,
            )
            task_state = value.get("task_state", {})
            task_state = task_state if isinstance(task_state, Mapping) else {}
            current = task_state.get("current_position")
            current_text = str(tuple(current)) if current is not None else ""
            conditions = tuple(
                item
                for item in value.get("decision_conditions", ())
                if isinstance(item, Mapping)
            )
            route_distance = next(
                (
                    int(item.get("value", 0))
                    for item in conditions
                    if item.get("name") == "route_distance"
                ),
                None,
            )
            if language == "zh-CN":
                if objective == "pickup" and task_slot and target_text:
                    goal = f"当前的临时导航目标是任务{task_slot}的A点{target_text}"
                elif objective == "delivery" and task_slot and target_text:
                    goal = f"当前的配送目标是任务{task_slot}的B点{target_text}"
                elif objective == "charge" and target_text:
                    goal = f"当前目标是充电站{target_text}"
                else:
                    goal = f"当前目标是{_goal_label(objective, language)}"
                route = (
                    f"；从当前位置{current_text}到该目标的最短可通行路径为"
                    f"{route_distance}格"
                    if current_text and route_distance is not None
                    else ""
                )
                return f"{robot}{goal}{route}"
            if objective == "pickup" and task_slot and target_text:
                goal = f"current temporary navigation target was task {task_slot} point A {target_text}"
            elif objective == "delivery" and task_slot and target_text:
                goal = f"current delivery target was task {task_slot} point B {target_text}"
            elif objective == "charge" and target_text:
                goal = f"current target was the charger at {target_text}"
            else:
                goal = f"current objective was {_goal_label(objective, language)}"
            route = (
                f"; the shortest passable route from {current_text} to that target was "
                f"{route_distance} cells"
                if current_text and route_distance is not None
                else ""
            )
            return f"{robot}'s {goal}{route}"
        return f"{predicate}: {value}"
