"""Warehouse-specific explanation vocabulary and deterministic verbalization."""

from __future__ import annotations

from typing import Any, Mapping

from env.warehouse.transition_audit import human_ai_moving_safety_reason

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

    def _decision_trace_explanation(
        self,
        trace: Mapping[str, Any],
        *,
        target_agent: str,
        focus: str,
        language: str,
    ) -> str | None:
        """Answer the requested question from one fact-validated joint trace."""

        agents = trace.get("agents", {})
        if not isinstance(agents, Mapping):
            return None
        decision = agents.get(target_agent)
        if not isinstance(decision, Mapping):
            return None
        robot = self.explanation_entity_label(target_agent, language)
        if not bool(trace.get("fact_valid", False)):
            return (
                f"无法可靠确认{robot}这一步的具体原因；请不要据此推断它在让路或避碰。"
                if language == "zh-CN"
                else f"A reliable cause for {robot}'s action could not be confirmed; it should not be interpreted as yielding or collision avoidance."
            )
        teammate_id = next(
            (str(agent_id) for agent_id in agents if str(agent_id) != target_agent),
            "",
        )
        teammate = self.explanation_entity_label(teammate_id, language)
        teammate_decision = agents.get(teammate_id, {})
        teammate_decision = (
            teammate_decision if isinstance(teammate_decision, Mapping) else {}
        )

        reason = str(decision.get("primary_reason_code", ""))
        selected = str(decision.get("selected_action", "WAIT"))
        resolved = str(decision.get("resolved_action", selected))
        # Older persisted Human-AI traces can carry the former generic detour
        # label even though their frozen runtime candidate set proves that the
        # move was a safer response to an unknown participant action.  Derive
        # the public explanation from that evidence, not the stale label.
        if reason == "POLICY_MISSION_DETOUR" and target_agent == "robot_2":
            runtime = trace.get("runtime_decision", {})
            runtime = runtime if isinstance(runtime, Mapping) else {}
            reason = human_ai_moving_safety_reason(runtime, resolved) or reason
        action = self.explanation_action_label(resolved, language)
        effect = decision.get("direct_effect", {})
        effect = effect if isinstance(effect, Mapping) else {}
        before_distance = int(effect.get("distance_before", 0) or 0)
        after_distance = int(effect.get("distance_after", before_distance) or 0)
        before_battery = float(effect.get("battery_before", 0.0) or 0.0)
        after_battery = float(effect.get("battery_after", before_battery) or before_battery)
        frozen_goal = decision.get("frozen_goal", {})
        frozen_goal = frozen_goal if isinstance(frozen_goal, Mapping) else {}
        resulting_goal = decision.get("resulting_goal", {})
        resulting_goal = resulting_goal if isinstance(resulting_goal, Mapping) else {}
        goal_id = str(
            frozen_goal.get("goal_id")
            or resulting_goal.get("goal_id")
            or ""
        )
        charging = decision.get("charging_state", {})
        charging = charging if isinstance(charging, Mapping) else {}
        release_threshold = float(
            charging.get("release_threshold", 0.0) or 0.0
        )
        plan = decision.get("joint_coordination_plan")
        plan = plan if isinstance(plan, Mapping) else {}
        feasibility = tuple(
            item
            for item in decision.get("battery_feasibility", ())
            if isinstance(item, Mapping)
        )
        selected_energy = next(
            (
                item
                for item in feasibility
                if str(item.get("task_id", "")) == goal_id
            ),
            None,
        )
        current_battery = float(
            charging.get("battery", before_battery) or before_battery
        )
        tasks = tuple(
            item for item in trace.get("tasks", ()) if isinstance(item, Mapping)
        )
        tasks_by_id = {str(item.get("task_id", "")): item for item in tasks}
        task_slots = {
            str(item.get("task_id", "")): index
            for index, item in enumerate(tasks, start=1)
        }

        def task_slot(task_id: object) -> int | None:
            return task_slots.get(str(task_id or ""))

        def task_label(task_id: object) -> str:
            slot = task_slot(task_id)
            if slot is None:
                return "当前A–B任务" if language == "zh-CN" else "the current A–B task"
            if language == "zh-CN":
                return f"A{slot}/B{slot}任务"
            return f"the A{slot}–B{slot} task"

        def pickup_label(task_id: object) -> str:
            slot = task_slot(task_id)
            return (
                f"A{slot}" if slot is not None else ("当前A点" if language == "zh-CN" else "the current A point")
            )

        def delivery_label(task_id: object) -> str:
            slot = task_slot(task_id)
            return (
                f"B{slot}" if slot is not None else ("当前B点" if language == "zh-CN" else "the current B point")
            )

        def priority_clause() -> tuple[str, str]:
            priority_id = str(plan.get("priority_agent_id", ""))
            priority = self.explanation_entity_label(priority_id, language)
            basis = str(plan.get("priority_basis", ""))
            priority_goal = str(plan.get("priority_goal_id", ""))
            if language == "zh-CN":
                if basis == "loaded_delivery":
                    return priority, (
                        f"载有从{pickup_label(priority_goal)}取得的货物并正前往{delivery_label(priority_goal)}交付"
                    )
                if basis in {
                    "urgent_charger_route",
                    "critical_charger_route",
                    "lower_energy_charger_waiter",
                }:
                    return priority, "电量较低并需要优先前往充电站"
                if basis == "charger_exit":
                    return priority, "已经充足电并需要离开充电站"
                if basis == "single_lane_egress":
                    return priority, "位于单通道深处并需要先驶出"
                return priority, "当前路线具有通行优先级"
            if basis == "loaded_delivery":
                return priority, (
                    f"was carrying cargo collected at {pickup_label(priority_goal)} to {delivery_label(priority_goal)}"
                )
            if basis in {
                "urgent_charger_route",
                "critical_charger_route",
                "lower_energy_charger_waiter",
            }:
                return priority, "had low battery and needed the charger first"
            if basis == "charger_exit":
                return priority, "had enough charge and needed to leave the charger"
            if basis == "single_lane_egress":
                return priority, "was deeper in the single lane and needed to exit first"
            return priority, "had right of way on the current route"

        uncertainty = decision.get("human_action_uncertainty", {})
        uncertainty = uncertainty if isinstance(uncertainty, Mapping) else {}
        collision_counterfactuals = tuple(
            item
            for item in uncertainty.get("collision_counterfactuals", ())
            if isinstance(item, Mapping)
        )

        if focus == "collision" and collision_counterfactuals:
            example = collision_counterfactuals[0]
            ai_action = self.explanation_action_label(
                str(example.get("ai_action", "WAIT")), language
            )
            human_action = self.explanation_action_label(
                str(example.get("participant_action", "WAIT")), language
            )
            if language == "zh-CN":
                return f"{robot}在决策时不知道你本帧会怎么走；如果它{ai_action}而你{human_action}，双方会发生{str(example.get('kind', '路径冲突'))}，因此它选择了安全动作。"
            return f"At decision time, {robot} did not know your current move. If it moved {ai_action} while you moved {human_action}, the robots could have had a {str(example.get('kind', 'path conflict'))}, so it chose the safe action."

        # Question routing comes before action-template routing.  Previously
        # every non-collision question returned the same action explanation.
        if focus == "allocation":
            visible_task_one_id = (
                str(tasks[0].get("task_id", "")) if tasks else ""
            )
            task_one = tasks_by_id.get(visible_task_one_id)
            if task_one is not None:
                carrier_id = str(task_one.get("carrier_agent_id") or "")
                if carrier_id:
                    carrier = self.explanation_entity_label(carrier_id, language)
                    if language == "zh-CN":
                        if carrier_id == target_agent:
                            return f"{robot}已经在A1取得货物，正在前往B1交付。"
                        current = task_label(goal_id) if goal_id else "其他任务"
                        return f"A1的货物已由{carrier}取走并正送往B1，{robot}不能再次领取；其当前目标是{current}。"
                    if carrier_id == target_agent:
                        return f"{robot} had already collected the cargo at A1 and was delivering it to B1."
                    current = task_label(goal_id) if goal_id else "another task"
                    return f"{carrier} had already collected the cargo at A1, so {robot} could not claim it again; its current goal was {current}."
                if goal_id == visible_task_one_id:
                    if language == "zh-CN":
                        return f"{robot}选择A1取货，因为A1/B1是其已锁定的当前任务。"
                    return f"{robot} was heading to A1 because the A1–B1 task was its locked pickup goal."
                if str(frozen_goal.get("navigation_kind", "")) == "charge":
                    least_required = min(
                        (
                            float(item.get("required_energy", 0.0) or 0.0)
                            for item in feasibility
                        ),
                        default=0.0,
                    )
                    if language == "zh-CN":
                        return f"A1仍可领取，但{robot}当前仅{current_battery:g}%电量，最省电的安全配送也需{least_required:g}%，因此先去充电。"
                    return f"A1 was still available, but {robot} had only {current_battery:g}% battery and even the least costly safe delivery required {least_required:g}%, so it charged first."
            if language == "zh-CN":
                return f"{robot}当前锁定的是{task_label(goal_id)}，不是A1/B1任务。"
            return f"{robot}'s locked goal was {task_label(goal_id)}, not the A1–B1 task."

        if focus in {"energy", "charge_threshold"}:
            route_energy = float(charging.get("route_energy", 0.0) or 0.0)
            hysteresis_energy = float(
                charging.get("hysteresis_energy", 0.0) or 0.0
            )
            coordination_energy = float(
                charging.get("coordination_contention_energy", 0.0) or 0.0
            )
            controlling_task = task_label(charging.get("task_id"))
            if reason == "CONTINUE_CHARGING":
                if language == "zh-CN":
                    contention = (
                        f"，并为已知的充电通道交接预留{coordination_energy:g}%"
                        if coordination_energy > 0
                        else ""
                    )
                    return f"{robot}继续充电：当前电量{before_battery:g}%，{controlling_task}完整配送及安全返航预计需{route_energy:g}%，另留{hysteresis_energy:g}%防止离站后折返{contention}，因此需充至{release_threshold:g}%；本步升至{after_battery:g}%。"
                contention = (
                    f", plus {coordination_energy:g}% for the visible charger handoff"
                    if coordination_energy > 0
                    else ""
                )
                return f"{robot} kept charging: it had {before_battery:g}%; {controlling_task} required {route_energy:g}% for delivery and a safe return, plus {hysteresis_energy:g}% to prevent an immediate return{contention}, so the release level was {release_threshold:g}%. This step raised it to {after_battery:g}%."
            if selected_energy is not None:
                required = float(selected_energy.get("required_energy", 0.0) or 0.0)
                slack = float(selected_energy.get("energy_slack", 0.0) or 0.0)
                if language == "zh-CN":
                    enough = "足够" if slack >= 0 else "不足"
                    return f"{robot}当前电量{current_battery:g}%，{task_label(goal_id)}完整配送、返航及安全余量预计需{required:g}%，电量{enough}。"
                enough = "enough" if slack >= 0 else "not enough"
                return f"{robot} had {current_battery:g}% battery; {task_label(goal_id)} required {required:g}% for delivery, return, and safety reserve, so its battery was {enough}."
            if feasibility:
                least_required = min(
                    float(item.get("required_energy", 0.0) or 0.0)
                    for item in feasibility
                )
                if language == "zh-CN":
                    return f"{robot}当前电量{current_battery:g}%，最省电的可选安全配送仍需{least_required:g}%，因此需要先充电。"
                return f"{robot} had {current_battery:g}% battery, while even the least costly safe delivery required {least_required:g}%, so it needed to charge first."

        if focus == "collaboration":
            if reason in {
                "CLEAR_PARTICIPANT_STANDOFF",
                "CLEAR_PARTICIPANT_ROUTE",
                "MOVE_TO_AVOID_UNKNOWN_PARTICIPANT_ACTION",
            }:
                if language == "zh-CN":
                    return f"队友影响了本步：{robot}{action}为{teammate}让出通道空间，避免双方在狭窄区域发生冲突。"
                return f"The teammate affected this step: {robot} moved {action} to leave aisle space for {teammate} and avoid a conflict in the narrow area."
            if plan:
                priority, basis = priority_clause()
                if str(plan.get("waiting_agent_id", "")) == target_agent:
                    if language == "zh-CN":
                        return f"队友影响了本步：{robot}等待，让{basis}的{priority}先通过。"
                    return f"The teammate affected this step: {robot} waited because {priority} {basis}."
                if language == "zh-CN":
                    return f"队友影响了本步：{robot}因{basis}而优先通过，{teammate}等待让行。"
                return f"The teammate affected this step: {robot} moved first because it {basis}, while {teammate} waited."
            resolution = decision.get("action_resolution", {})
            resolution = resolution if isinstance(resolution, Mapping) else {}
            if language == "zh-CN":
                return f"队友没有迫使{robot}改变动作；本帧没有联合让行计划或目标格冲突。"
            return f"The teammate did not force {robot} to change action; there was no joint yield plan or target-cell conflict."

        if focus == "task":
            goal_kind = str(frozen_goal.get("goal_type", ""))
            if (
                goal_kind == "GO_TO_PICKUP"
                and after_distance < before_distance
            ):
                if language == "zh-CN":
                    return f"这个动作推进了{task_label(goal_id)}取货：到A点的剩余距离从{before_distance}格缩短到{after_distance}格。"
                return f"This advanced pickup for {task_label(goal_id)}: the remaining distance to A fell from {before_distance} to {after_distance} cells."
            if (
                goal_kind == "GO_TO_DROPOFF"
                and after_distance < before_distance
            ):
                if language == "zh-CN":
                    return f"这个动作推进了{task_label(goal_id)}交付：到B点的剩余距离从{before_distance}格缩短到{after_distance}格。"
                return f"This advanced delivery for {task_label(goal_id)}: the remaining distance to B fell from {before_distance} to {after_distance} cells."
            if resolved == "WAIT" and goal_id:
                if language == "zh-CN":
                    return f"{robot}本帧没有推进{task_label(goal_id)}，剩余距离仍为{before_distance}格。"
                return f"{robot} did not advance {task_label(goal_id)} on this step; {before_distance} cells remained."
            if language == "zh-CN":
                return f"本步用于充电或通道协调，没有直接推进{task_label(goal_id)}。"
            return f"This step handled charging or aisle coordination and did not directly advance {task_label(goal_id)}."

        if reason == "SAFETY_RULE_BLOCKED":
            resolution = decision.get("action_resolution", {})
            resolution = resolution if isinstance(resolution, Mapping) else {}
            blocked = str(resolution.get("blocked_reason") or "joint_conflict")
            proposed = self.explanation_action_label(selected, language)
            if language == "zh-CN":
                return f"{robot}原本选择{proposed}，但安全执行层因{blocked}阻止了移动；这不是机器人主动等待。"
            return f"{robot} selected {proposed}, but the safety resolver blocked it because of {blocked}; this was not an intentional wait."

        if reason == "WAIT_FOR_OCCUPIED_ROUTE_CLEARANCE":
            clearing_id = str(
                plan.get("clearing_agent_id")
                or plan.get("moving_agent_id")
                or ""
            )
            clearing = self.explanation_entity_label(clearing_id, language)
            cell = tuple(plan.get("occupied_position", ()))
            _, basis = priority_clause()
            if language == "zh-CN":
                return f"{robot}等待，因为通往{task_label(goal_id)}的下一格{cell}被{clearing}占用；{clearing}{basis}，正在清空该格。"
            return f"{robot} waited because {clearing} occupied the next cell {cell} on its route; {clearing}, which {basis}, was clearing that cell."

        if reason == "WAIT_FOR_CONFLICTING_TARGET":
            priority, basis = priority_clause()
            cell = tuple(plan.get("moving_target", ()))
            if language == "zh-CN":
                return f"{robot}等待，因为两台机器人的下一步都需要进入{cell}；{basis}的{priority}先行，避免同格冲突。"
            return f"{robot} waited because both robots needed cell {cell} next; {priority}, which {basis}, moved first to avoid a same-cell conflict."

        if reason == "CLEAR_TEAMMATE_ROUTE":
            priority_id = str(plan.get("priority_agent_id", ""))
            priority = self.explanation_entity_label(priority_id, language)
            purpose = (
                "充电路线"
                if str(plan.get("reason_code", "")).startswith("critical_charger")
                else "当前路线"
            )
            position_before = tuple(effect.get("position_before", ()))
            position_after = tuple(effect.get("position_after", ()))
            left_charger = bool(
                charging.get("at_charger", False)
                and position_before != position_after
            )
            if language == "zh-CN":
                if left_charger:
                    return f"{robot}{action}离开充电站，是为了给低电量的{priority}清空{purpose}的下一格。"
                return f"{robot}{action}是为了给{priority}清空{purpose}的下一格。"
            return f"{robot} moved {action} to clear the next cell on {priority}'s route."

        if reason in {
            "CLEAR_PARTICIPANT_STANDOFF",
            "CLEAR_PARTICIPANT_ROUTE",
            "MOVE_TO_AVOID_UNKNOWN_PARTICIPANT_ACTION",
        }:
            goal_type = str(frozen_goal.get("goal_type", ""))
            if goal_type == "GO_TO_DROPOFF" or decision.get("carrying_task_id"):
                target_zh = delivery_label(goal_id)
                target_en = delivery_label(goal_id)
                continuation_zh = f"继续前往{target_zh}交付"
                continuation_en = f"continue toward {target_en} to deliver the cargo"
            elif goal_type == "GO_TO_PICKUP":
                target_zh = pickup_label(goal_id)
                target_en = pickup_label(goal_id)
                continuation_zh = f"继续前往{target_zh}取货"
                continuation_en = f"continue toward {target_en} to collect the cargo"
            else:
                continuation_zh = "继续执行当前任务"
                continuation_en = "continue its current task"
            if language == "zh-CN":
                return f"{robot}{action}是为了给{teammate}让路，避免双方在狭窄通道发生冲突。虽然这一步暂时远离当前目标，但避让后它会{continuation_zh}。"
            return f"{robot} moved {action} to yield to {teammate} and avoid a conflict in the narrow aisle. Although this temporarily moved it away from its current goal, it will {continuation_en} after yielding."

        if reason == "WAIT_FOR_PRIORITY_PASSAGE":
            priority, basis = priority_clause()
            if language == "zh-CN":
                return f"{robot}等待是为了让{basis}的{priority}先通过狭窄通道。"
            return f"{robot} waited so that {priority}, which {basis}, could pass through the narrow aisle."

        if reason == "PRIORITY_ROUTE_PROGRESS":
            left_charger = bool(
                charging.get("at_charger", False)
                and tuple(effect.get("position_before", ()))
                != tuple(effect.get("position_after", ()))
            )
            if language == "zh-CN":
                if str(frozen_goal.get("navigation_kind", "")) == "charge":
                    least_required = min(
                        (
                            float(item.get("required_energy", 0.0) or 0.0)
                            for item in feasibility
                        ),
                        default=0.0,
                    )
                    return f"{robot}{action}进入充电站，是因为当前电量{current_battery:g}%，而最省电的安全配送仍需{least_required:g}%；{teammate}等待保留入口。"
                priority, basis = priority_clause()
                if left_charger:
                    return f"{robot}{action}离开充电站，因为它{basis}；{teammate}等待让行。"
                return f"{robot}{action}继续前进，因为它{basis}；{teammate}等待让行。"
            if str(frozen_goal.get("navigation_kind", "")) == "charge":
                least_required = min(
                    (
                        float(item.get("required_energy", 0.0) or 0.0)
                        for item in feasibility
                    ),
                    default=0.0,
                )
                return f"{robot} moved {action} into the charger because it had {current_battery:g}% battery while even the least costly safe delivery required {least_required:g}%; {teammate} waited to keep the entrance clear."
            _, basis = priority_clause()
            if left_charger:
                return f"{robot} moved {action} out of the charger because it {basis}; {teammate} waited to yield."
            return f"{robot} moved {action} first because it {basis}; {teammate} waited to yield."

        if reason == "CONTINUE_CHARGING":
            route_energy = float(charging.get("route_energy", 0.0) or 0.0)
            hysteresis_energy = float(charging.get("hysteresis_energy", 0.0) or 0.0)
            coordination_energy = float(
                charging.get("coordination_contention_energy", 0.0) or 0.0
            )
            if language == "zh-CN":
                contention = (
                    f"，并为已知的充电通道交接预留{coordination_energy:g}%"
                    if coordination_energy > 0
                    else ""
                )
                return f"{robot}等待是为了继续充电：完整配送及安全返航预计需{route_energy:g}%，另留{hysteresis_energy:g}%防止折返{contention}；当前{before_battery:g}%，本步升至{after_battery:g}%。"
            contention = (
                f", plus {coordination_energy:g}% for the visible charger handoff"
                if coordination_energy > 0
                else ""
            )
            return f"{robot} waited to keep charging: a complete delivery and safe return required {route_energy:g}%, plus {hysteresis_energy:g}% to prevent an immediate return{contention}; battery rose from {before_battery:g}% to {after_battery:g}%."

        if reason == "LEAVE_CHARGER_THRESHOLD_MET":
            if language == "zh-CN":
                return f"{robot}{action}离开充电站，因为决策前电量{before_battery:g}%已达到安全离站阈值{release_threshold:g}%。"
            return f"{robot} left the charger by moving {action} because its {before_battery:g}% battery had reached the safe release threshold of {release_threshold:g}%."

        if reason == "PREMATURE_CHARGER_DEPARTURE":
            if language == "zh-CN":
                return f"{robot}在电量仅{before_battery:g}%、低于安全离站所需{release_threshold:g}%时仍{action}离开充电站；这是一次过早离站的低效决策。"
            return f"{robot} moved {action} away from the charger with only {before_battery:g}% battery, below the safe release requirement of {release_threshold:g}%; this was an inefficient premature departure."

        if reason == "CHARGER_ROUTE_PROGRESS":
            least_required = min(
                (
                    float(item.get("required_energy", 0.0) or 0.0)
                    for item in feasibility
                ),
                default=0.0,
            )
            if language == "zh-CN":
                return f"{robot}{action}前往充电站，是因为当前电量{current_battery:g}%，最省电的安全配送仍需{least_required:g}%；剩余距离从{before_distance}格缩短到{after_distance}格。"
            return f"{robot} moved {action} toward the charger because it had {current_battery:g}% battery while even the least costly safe delivery required {least_required:g}%; the remaining route fell from {before_distance} to {after_distance} cells."

        if reason == "WAIT_FOR_CHARGER_AVAILABILITY":
            if language == "zh-CN":
                return f"{robot}等待是因为充电站当前被另一台机器人占用；它会在充电位释放后继续前进。"
            return f"{robot} waited because the charger was occupied; it can continue after the charging cell is released."

        if reason == "WAIT_FOR_UNKNOWN_PARTICIPANT_ACTION":
            example = next(
                (
                    item
                    for item in collision_counterfactuals
                    if str(item.get("ai_action", "")) in {"UP", "DOWN", "LEFT", "RIGHT"}
                ),
                collision_counterfactuals[0] if collision_counterfactuals else {},
            )
            ai_action = self.explanation_action_label(
                str(example.get("ai_action", "WAIT")), language
            )
            human_action = self.explanation_action_label(
                str(example.get("participant_action", "WAIT")), language
            )
            if language == "zh-CN":
                return f"{robot}等待以避免潜在碰撞。决策时它不知道你本帧的动作；如果它{ai_action}而你{human_action}，双方可能进入冲突位置。"
            return f"{robot} waited to avoid a potential collision. It did not know your current move at decision time; moving {ai_action} while you moved {human_action} could put both robots in conflict."

        if reason == "DELIVERY_ROUTE_PROGRESS":
            if language == "zh-CN":
                return f"{robot}{action}，把从{pickup_label(goal_id)}取得的货物送往{delivery_label(goal_id)}；剩余路线从{before_distance}格缩短到{after_distance}格。"
            return f"{robot} moved {action} to take the cargo collected at {pickup_label(goal_id)} to {delivery_label(goal_id)}; the remaining route fell from {before_distance} to {after_distance} cells."

        if reason == "PICKUP_ROUTE_PROGRESS":
            if language == "zh-CN":
                return f"{robot}{action}前往{pickup_label(goal_id)}取货；剩余路线从{before_distance}格缩短到{after_distance}格。"
            return f"{robot} moved {action} toward {pickup_label(goal_id)} to collect the cargo; the remaining route fell from {before_distance} to {after_distance} cells."

        if reason == "ENERGY_SAFE_TASK_SELECTION":
            safe_tasks = tuple(
                item
                for item in decision.get("battery_feasibility", ())
                if isinstance(item, Mapping) and bool(item.get("safe", False))
            )
            selected_task = goal_id or (
                str(safe_tasks[0].get("task_id", "")) if len(safe_tasks) == 1 else ""
            )
            if language == "zh-CN":
                target = f"{task_label(selected_task)}的A点" if selected_task else "电量可安全完成的A点"
                selected_record = next(
                    (
                        item
                        for item in safe_tasks
                        if str(item.get("task_id", "")) == selected_task
                    ),
                    safe_tasks[0] if safe_tasks else {},
                )
                required = float(selected_record.get("required_energy", 0.0) or 0.0)
                return f"{robot}{action}是为了前往{target}取货；当前电量{current_battery:g}%，完整配送、返航及安全余量预计需{required:g}%，电量足够。"
            selected_record = next(
                (
                    item
                    for item in safe_tasks
                    if str(item.get("task_id", "")) == selected_task
                ),
                safe_tasks[0] if safe_tasks else {},
            )
            required = float(selected_record.get("required_energy", 0.0) or 0.0)
            target = f"point A for {task_label(selected_task)}" if selected_task else "an energy-feasible pickup point"
            return f"{robot} moved {action} toward {target}; it had {current_battery:g}% battery and the complete delivery, return, and safety reserve required {required:g}%, so the battery was sufficient."

        if reason == "POLICY_MISSION_DETOUR":
            if language == "zh-CN":
                return f"{robot}{action}没有缩短前往当前目标的路线，也没有起到避碰、让行或充电作用，因此这是一次低效移动。"
            return f"{robot}'s {action} did not shorten the route to its current goal or serve collision avoidance, yielding, or charging, so it was an inefficient move."

        if reason == "AVOIDABLE_WAIT_SAFE_PROGRESS_AVAILABLE":
            counterfactual = decision.get("wait_counterfactual", {})
            counterfactual = counterfactual if isinstance(counterfactual, Mapping) else {}
            alternative = self.explanation_action_label(
                str(counterfactual.get("action", "WAIT")), language
            )
            if language == "zh-CN":
                return f"{robot}本可安全{alternative}并继续推进{task_label(goal_id)}，因此这次等待没有必要。"
            return f"{robot} could safely move {alternative} and continue {task_label(goal_id)}, so this wait was unnecessary."

        if reason in {
            "WAIT_NO_VERIFIED_CAUSE",
            "POLICY_WAIT_NO_VERIFIED_COORDINATION_CAUSE",
        }:
            if language == "zh-CN":
                return f"{robot}本帧等待，但没有任务、电量或安全原因支持这次等待；这是一次低效决策。"
            return f"No task, battery, or safety need justified {robot}'s wait; it was an inefficient decision."

        if language == "zh-CN":
            return f"{robot}执行了{action}，但没有可验证的任务、电量或安全原因；这是一次低效决策。"
        return f"{robot} moved {action} without a verifiable task, battery, or safety reason; it was an inefficient decision."

    def concise_study_explanation(
        self,
        snapshot: Any,
        *,
        target_agent: str,
        policy: Any,
        focus: str,
        language: str,
    ) -> str:
        """Answer with one direct reason and, at most, one key fact.

        Causes are limited to the frozen pre-decision state.  Current joint
        actions and the resulting movement are described only as outcomes.
        """

        raw_trace = snapshot.metadata.get("decision_trace", {})
        if (
            isinstance(raw_trace, Mapping)
            and raw_trace.get("agents")
        ):
            trace_text = self._decision_trace_explanation(
                raw_trace,
                target_agent=target_agent,
                focus=focus,
                language=language,
            )
            if trace_text is not None:
                return trace_text

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
                goal = f"任务{task_slot}的A点取货"
                goal_target = f"任务{task_slot}的A点"
            elif goal_kind == "delivery" and task_slot:
                goal = f"任务{task_slot}的B点交付"
                goal_target = f"任务{task_slot}的B点"
            elif goal_kind == "charge":
                goal = "前往充电站"
                goal_target = "充电站"
            else:
                goal = _goal_label(goal_kind, language)
                goal_target = goal
        else:
            if goal_kind == "pickup" and task_slot:
                goal = f"collect task {task_slot} at A"
            elif goal_kind == "delivery" and task_slot:
                goal = f"deliver task {task_slot} at B"
            elif goal_kind == "charge":
                goal = "reach the charger"
            else:
                goal = _goal_label(goal_kind, language)
            goal_target = goal

        resolution = by_predicate.get("action_resolution_reason", {})
        resolution = resolution if isinstance(resolution, Mapping) else {}
        collision_kind = str(
            resolution.get("collision_kind")
            or resolution.get("blocked_reason")
            or ""
        )
        collision = collision_kind in {
            "same_target",
            "swap",
            "occupied_stationary",
            "robot_collision",
        }
        collision_zh = {
            "same_target": "双方试图进入同一格",
            "swap": "双方试图交换位置",
            "occupied_stationary": "它试图进入队友未离开的格子",
            "robot_collision": "双方路径冲突",
        }.get(collision_kind, "双方路径存在冲突")
        collision_en = {
            "same_target": "both robots targeted the same cell",
            "swap": "the robots tried to swap cells",
            "occupied_stationary": "it targeted a cell the teammate did not leave",
            "robot_collision": "the robot paths conflicted",
        }.get(collision_kind, "the robot paths conflicted")

        simultaneous_zh = "两台机器人只依据同一决策前状态同时选动作，事先不知道对方本帧动作。"
        simultaneous_en = (
            "Both robots chose simultaneously from the same pre-move state; "
            "neither knew the other's current action."
        )
        if focus == "collision":
            if language == "zh-CN":
                first = (
                    f"本帧发生碰撞：{collision_zh}，因此环境阻止了移动。"
                    if collision
                    else "本帧没有实际碰撞；风险来自双方可能选择同一格或互换位置。"
                )
                return f"{first}{simultaneous_zh}"
            first = (
                f"A collision occurred: {collision_en}, so the environment blocked the move."
                if collision
                else "No collision occurred; the risk was a same-cell target or a position swap."
            )
            return f"{first} {simultaneous_en}"

        collaboration = by_predicate.get("collaboration_context", {})
        collaboration = collaboration if isinstance(collaboration, Mapping) else {}
        teammate_id = str(collaboration.get("teammate_agent", ""))
        teammate = self.explanation_entity_label(teammate_id, language)
        teammate_battery = float(collaboration.get("teammate_battery", 0.0) or 0.0)
        target_battery = float(collaboration.get("target_battery", 0.0) or 0.0)
        teammate_role = collaboration.get("teammate_role", {})
        teammate_role = teammate_role if isinstance(teammate_role, Mapping) else {}
        movement = by_predicate.get("movement_outcome", {})
        movement = movement if isinstance(movement, Mapping) else {}
        work = movement.get("work", {})
        work = work if isinstance(work, Mapping) else {}
        distance_before = int(movement.get("distance_before", 0) or 0)
        distance_after = int(movement.get("distance_after", distance_before) or 0)
        energy = by_predicate.get("energy_decision_context", {})
        energy = energy if isinstance(energy, Mapping) else {}
        battery = float(energy.get("battery", target_battery) or 0.0)
        requires_charge = bool(energy.get("requires_charge", False))
        required_energy = energy.get("required_safe_energy")
        charging = by_predicate.get("charging_outcome", {})
        charging = charging if isinstance(charging, Mapping) else {}
        queue = by_predicate.get("charger_queue_context", {})
        queue = queue if isinstance(queue, Mapping) else {}

        if focus in {"energy", "charge_threshold"}:
            if charging:
                before = float(charging.get("battery_before", battery) or 0.0)
                after = float(charging.get("battery_after", before) or before)
                if language == "zh-CN":
                    return (
                        f"{robot}等待是为了继续充电，以达到安全配送所需电量。"
                        f"本步电量从{before:g}%升至{after:g}%。"
                    )
                return (
                    f"{robot} waited to keep charging toward the level needed for a safe "
                    f"delivery. This step raised it from {before:g}% to {after:g}%."
                )
            if language == "zh-CN":
                if requires_charge:
                    evidence = (
                        f"当前仅{battery:g}%，下一条安全配送路线约需{float(required_energy):g}点电量"
                        if required_energy is not None
                        else f"当前仅{battery:g}%，不足以安全完成下一次配送"
                    )
                    return f"充电需求影响了{robot}的{action_label}：{evidence}。"
                enough = (
                    f"，下一条安全路线约需{float(required_energy):g}点"
                    if required_energy is not None
                    else ""
                )
                return f"{robot}当前电量为{battery:g}%{enough}，电量并未迫使它改变本步动作。"
            if requires_charge:
                evidence = (
                    f"it had {battery:g}% while the next safe route required about {float(required_energy):g} points"
                    if required_energy is not None
                    else f"it had only {battery:g}%, insufficient for the next safe delivery"
                )
                return f"Charging needs affected {robot}'s {action_label}: {evidence}."
            enough = (
                f"; the next safe route required about {float(required_energy):g} points"
                if required_energy is not None
                else ""
            )
            return f"{robot} had {battery:g}% battery{enough}, so energy did not force this action."

        if collision and bool(resolution.get("environment_changed_action", False)):
            proposed = self.explanation_action_label(
                str(resolution.get("requested_action", action)), language
            )
            if language == "zh-CN":
                return f"{robot}原本选择{proposed}，但因{collision_zh}被环境改为等待。{simultaneous_zh}"
            return f"{robot} chose {proposed}, but the environment changed it to wait because {collision_en}. {simultaneous_en}"

        if charging:
            before = float(charging.get("battery_before", battery) or 0.0)
            after = float(charging.get("battery_after", before) or before)
            if language == "zh-CN":
                return f"{robot}等待是为了继续充电，以达到安全配送所需电量。本步电量{before:g}%→{after:g}%。"
            return f"{robot} waited to keep charging toward the level needed for a safe delivery. Battery rose from {before:g}% to {after:g}%."

        if queue:
            occupant_id = str(queue.get("occupant_agent", teammate_id))
            occupant = self.explanation_entity_label(occupant_id, language)
            if language == "zh-CN":
                return f"{robot}等待是因为需要充电，但唯一的充电站正被{occupant}占用。当前电量为{battery:g}%。"
            return f"{robot} waited because it needed to charge, but {occupant} occupied the only charger. Its battery was {battery:g}%."

        if action == "WAIT" and collaboration.get("occupied_clearance_wait"):
            occupied = collaboration.get("occupied_clearance_wait", {})
            occupied = occupied if isinstance(occupied, Mapping) else {}
            cell = tuple(occupied.get("occupied_position", ()))
            if language == "zh-CN":
                return f"{robot}等待是为了先让{teammate}通过；通往当前目标的下一格{cell}在决策前被队友占用。"
            return f"{robot} waited to let {teammate} pass; the next cell {cell} toward its goal was occupied before the decision."

        occupied_goal_cells = tuple(collaboration.get("occupied_goal_cells", ()))
        if action == "WAIT" and occupied_goal_cells:
            cell = tuple(occupied_goal_cells[0])
            if language == "zh-CN":
                return f"{robot}等待是为了让{teammate}先通过；通往{goal_target}的下一格{cell}在决策前被队友占用。"
            return f"{robot} waited to let {teammate} pass; the next cell {cell} toward its goal was occupied before the decision."

        if action == "WAIT" and bool(collaboration.get("teammate_has_charge_priority", False)):
            if language == "zh-CN":
                return f"{robot}等待是为了让电量更低的{teammate}优先前往充电站。两者都需充电，{teammate}为{teammate_battery:g}%，{robot}为{target_battery:g}%。"
            return f"{robot} waited to give the lower-battery {teammate} charger priority. Both needed to charge: {teammate_battery:g}% versus {target_battery:g}%."

        teammate_yield = collaboration.get("teammate_yielded_for_target")
        if action == "WAIT" and isinstance(teammate_yield, Mapping):
            if language == "zh-CN":
                return f"{robot}等待是为了给相邻的{teammate}留出通道空间，降低双方挤入同一区域的风险。决策前两者相距{int(collaboration.get('teammate_distance', 0))}格。"
            return f"{robot} waited to leave aisle space for nearby {teammate}, reducing the risk of both entering the same narrow area. They were {int(collaboration.get('teammate_distance', 0))} cells apart before deciding."

        if (
            action == "WAIT"
            and bool(collaboration.get("teammate_requires_charge", False))
            and str(teammate_role.get("kind", "")) == "charge"
            and int(collaboration.get("teammate_distance", 99)) <= 3
        ):
            if language == "zh-CN":
                return f"{robot}等待是为了避免阻挡低电量的{teammate}，让其先前往充电站。决策前，{teammate}仅有{teammate_battery:g}%电量。"
            return f"{robot} waited to avoid obstructing low-battery {teammate} on the way to the charger. Before deciding, {teammate} had {teammate_battery:g}% battery."

        if action == "WAIT" and bool(collaboration.get("goal_advance_near_teammate", False)):
            if language == "zh-CN":
                return f"{robot}等待是为了与{teammate}保持间距，让队友先通过狭窄通道。"
            return f"{robot} waited to keep spacing and let {teammate} clear the narrow aisle first."

        coordination_yield = collaboration.get("coordination_yield")
        if isinstance(coordination_yield, Mapping):
            if str(teammate_role.get("kind", "")) == "charge":
                purpose_zh = f"给低电量的{teammate}让出通道，使其前往充电站"
                purpose_en = f"clear the aisle for low-battery {teammate} to reach the charger"
            else:
                purpose_zh = f"给{teammate}让路，避免双方在狭窄通道发生冲突"
                purpose_en = f"yield to {teammate} and avoid a narrow-aisle conflict"
            if language == "zh-CN":
                return f"{robot}{action_label}是为了{purpose_zh}。决策前，{teammate}电量为{teammate_battery:g}%。"
            return f"{robot} chose {action_label} to {purpose_en}. Before the decision, {teammate} had {teammate_battery:g}% battery."

        if bool(collaboration.get("charger_clearance", False)):
            if language == "zh-CN":
                return f"{robot}{action_label}是为了让出充电站，供低电量的{teammate}进入。决策前，{teammate}仅有{teammate_battery:g}%电量。"
            return f"{robot} chose {action_label} to vacate the charger for low-battery {teammate}. Before the decision, {teammate} had {teammate_battery:g}% battery."

        kind = str(work.get("kind", goal_kind))
        charger_departure = collaboration.get("charger_departure")
        if isinstance(charger_departure, Mapping):
            if bool(collaboration.get("teammate_requires_charge", False)):
                if language == "zh-CN":
                    own_status = (
                        f"自身仍需充电，但{teammate}电量更低"
                        if bool(collaboration.get("target_requires_charge", False))
                        else "自身已不需要继续充电"
                    )
                    return f"{robot}{action_label}是为了离开充电站，把唯一充电位让给电量更低的{teammate}。{own_status}；两者电量分别为{target_battery:g}%和{teammate_battery:g}%。"
                own_status = (
                    f"it still needed charge, but {teammate} was lower"
                    if bool(collaboration.get("target_requires_charge", False))
                    else "it no longer needed to keep charging"
                )
                return f"{robot} chose {action_label} to leave the charger free for lower-battery {teammate}. It had {target_battery:g}% versus {teammate_battery:g}%; {own_status}."
            if kind == "reposition":
                if language == "zh-CN":
                    return f"{robot}{action_label}是因为本轮充电已经完成，需要离站并重新参与共享取货。决策前电量为{target_battery:g}%，已不需要继续充电。"
                return f"{robot} chose {action_label} because this charging round was complete and it needed to rejoin shared pickup work. Before deciding, it had {target_battery:g}% and no longer needed to charge."

        if action in {"UP", "DOWN", "LEFT", "RIGHT"} and kind in {
            "pickup",
            "delivery",
            "charge",
        }:
            if distance_after < distance_before:
                if language == "zh-CN":
                    return f"{robot}{action_label}是为了{goal}；本步将剩余距离从{distance_before}格缩短到{distance_after}格。"
                return f"{robot} chose {action_label} to {goal}; this reduced the remaining route from {distance_before} to {distance_after} cells."
            selected_probability = float(
                movement.get("selected_probability", 0.0) or 0.0
            )
            policy_selected = bool(movement.get("policy_selected", False))
            highest_action = self.explanation_action_label(
                str(movement.get("highest_probability_action", "")), language
            )
            highest_probability = float(
                movement.get("highest_probability", 0.0) or 0.0
            )
            if language == "zh-CN":
                route_change = (
                    f"将距离从{distance_before}格增加到{distance_after}格"
                    if distance_after > distance_before
                    else f"未减少{distance_before}格的剩余距离"
                )
                if not policy_selected:
                    return f"{robot}的目标是{goal}，但{action_label}是策略随机采样出的低概率动作（{selected_probability * 100:.1f}%），{route_change}。这是非必要绕路，不是任务或安全要求。"
                return f"{robot}的目标是{goal}，但策略最高概率动作{action_label}（{highest_probability * 100:.1f}%）{route_change}。这是策略产生的非必要绕路，不是任务或安全要求。"
            route_change = (
                f"increased the route from {distance_before} to {distance_after} cells"
                if distance_after > distance_before
                else f"did not reduce the remaining {distance_before}-cell route"
            )
            if not policy_selected:
                return f"{robot}'s goal was to {goal}, but {action_label} was a low-probability stochastic sample ({selected_probability * 100:.1f}%) that {route_change}. This was an unnecessary detour, not a task or safety requirement."
            return f"{robot}'s goal was to {goal}, but the policy's highest-probability action, {highest_action} ({highest_probability * 100:.1f}%), {route_change}. This was a policy-generated unnecessary detour, not a task or safety requirement."

        pickup_progress = tuple(
            item
            for item in movement.get("available_pickup_progress", ())
            if isinstance(item, Mapping)
        )
        if action in {"UP", "DOWN", "LEFT", "RIGHT"} and pickup_progress:
            improved = tuple(
                item
                for item in pickup_progress
                if int(item.get("distance_after", 0))
                < int(item.get("distance_before", 0))
            )
            nearest_before = min(
                int(item.get("distance_before", 0)) for item in pickup_progress
            )
            nearest_after = min(
                int(item.get("distance_after", 0)) for item in pickup_progress
            )
            if nearest_after < nearest_before:
                if language == "zh-CN":
                    return f"{robot}{action_label}是为了靠近尚未领取的共享取货区；最近A点的距离从{nearest_before}格缩短到{nearest_after}格。此时尚未承诺具体任务。"
                return f"{robot} chose {action_label} to approach the unclaimed shared pickup area; the nearest A point fell from {nearest_before} to {nearest_after} cells. No specific task was committed yet."
            if improved:
                candidate = min(
                    improved,
                    key=lambda item: (
                        int(item.get("distance_after", 0)),
                        int(item.get("task_slot", 0)),
                    ),
                )
                slot = int(candidate.get("task_slot", 0))
                before = int(candidate.get("distance_before", 0))
                after = int(candidate.get("distance_after", 0))
                if language == "zh-CN":
                    return f"{robot}{action_label}是在靠近任务{slot}的A点候选方向，距离从{before}格缩短到{after}格。任务尚未承诺，因此不能断言它最终会领取该货物。"
                return f"{robot} chose {action_label} along a candidate route toward task {slot}'s A point, reducing that distance from {before} to {after} cells. The task was not committed, so this does not prove the final pickup choice."

        if action == "WAIT":
            if goal_kind == "wait":
                if language == "zh-CN":
                    return f"{robot}保持位置，因为决策前没有已确认的取货、交付、充电或让行目标。"
                return f"{robot} held position because no pickup, delivery, charging, or yielding objective was established before the decision."
            if language == "zh-CN":
                return f"{robot}当前需要{goal}，但冻结状态未显示必须等待的安全或任务原因；这是一次未被证据充分解释的停顿。"
            return f"{robot} needed to {goal}, but the frozen state shows no safety or task reason that required waiting; this pause is not fully explained by the evidence."

        if language == "zh-CN":
            return f"{robot}{action_label}属于重新定位；冻结状态没有显示明确的取货、交付、充电或让行理由。"
        return f"{robot}'s {action_label} was a repositioning move; the frozen state shows no specific pickup, delivery, charging, or yielding reason."

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
                            f"{float(minimum_departure):g}%（包括任务路线、返航安全余量和充电迟滞）。"
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
                        "(including the route, return reserve, and charging hysteresis). It needed "
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
