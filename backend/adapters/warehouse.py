"""Collaborative-delivery implementation of the generic environment adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random
from typing import Any, Mapping, Sequence

from backend.adapters.base import (
    ActionDistribution,
    CandidateIntervention,
    EvidenceFact,
    EnvironmentAdapter,
    EnvironmentSnapshot,
    Intervention,
    PolicyProtocol,
    RolloutFrame,
    RolloutResult,
    SemanticPolicyContext,
)
from env.warehouse.contracts import RCPD_PROGRAM_VERSION
from env.warehouse.environment import (
    ACTIONS,
    MOVE_DELTAS,
    AgentState,
    WarehouseMultiAgentEnv,
    WarehouseState,
    all_passable_positions,
    is_passable,
    legal_action_mask,
    shortest_path_distance,
)
from env.warehouse.observations import (
    _actor_visible_goal,
    all_local_observations,
    global_observation,
    observation_schema,
)


WAREHOUSE_PROGRAM_VERSION = RCPD_PROGRAM_VERSION


from .warehouse_context import (
    WarehousePolicyState,
    _action_zh,
    _charging_state,
    _collaboration_context,
    _energy_decision_context,
    _goal_label,
    _manhattan,
    _movement_work_context,
    _post_charge_task_context,
    _task_record,
    _task_state,
    _transition_events,
)
from .warehouse_explanations import WarehouseExplanationMixin


class WarehouseAdapter(WarehouseExplanationMixin, EnvironmentAdapter):
    """Expose only the two-robot shared-task semantics used by the study."""

    def __init__(self, environment: WarehouseMultiAgentEnv) -> None:
        self.environment = environment

    def observation_schema(self) -> Mapping[str, Any]:
        return observation_schema(self.environment.config)

    def action_schema(self) -> Sequence[str]:
        return ACTIONS

    def entity_schema(self) -> Mapping[str, Any]:
        tasks = self.environment.state.tasks if self.environment.state else ()
        return {
            "agent": {
                "ids": self.environment.agent_ids,
                "references": {
                    "robot_1": ("robot_1", "robot 1", "机器人1", "参与者机器人"),
                    "robot_2": ("robot_2", "robot 2", "机器人2", "AI机器人"),
                },
                "properties": {
                    "position": {"type": "coordinate", "editable": True},
                    "battery": {"type": "percentage", "editable": True},
                    "carrying_task_id": {"type": "string_or_null", "editable": False},
                    "heading": {"type": "action", "editable": True},
                    "last_action": {"type": "action", "editable": True},
                    "last_executed_action": {"type": "action", "editable": False},
                    "active": {"type": "boolean", "editable": True},
                    "deliveries_completed": {"type": "integer", "editable": False},
                    "objective": {
                        "type": "objective",
                        "values": tuple(self.objective_descriptions()),
                        "editable": False,
                    },
                },
            },
            "task": {
                "ids": tuple(task.task_id for task in tasks),
                "properties": {
                    "pickup_position": "coordinate",
                    "delivery_position": "coordinate",
                    "status": "available_or_carried",
                    "carrier_agent_id": "robot_id_or_null",
                    "created_frame": "integer",
                    "claimed_frame": "integer_or_null",
                },
            },
            "charger": {
                "id": "charger",
                "position": self.environment.layout.charger_position,
            },
        }

    def snapshot(
        self,
        policy: PolicyProtocol | None = None,
    ) -> EnvironmentSnapshot:
        state = self.environment.get_state()
        observations = deepcopy(self.environment.observations())
        global_state = deepcopy(self.environment.global_state())
        distributions: Mapping[str, ActionDistribution] = {}
        proposed: Mapping[str, str] = {}
        if policy is not None:
            proposed, distributions = policy.act(
                observations,
                global_state,
                deterministic=True,
            )
        return EnvironmentSnapshot(
            environment=self.environment.environment_name,
            frame=state.frame,
            state=state,
            environment_rng_state=deepcopy(self.environment.get_rng_state()),
            policy_rng_state=(
                deepcopy(policy.get_rng_state()) if policy is not None else None
            ),
            observations=observations,
            global_state=global_state,
            actions={agent.agent_id: agent.last_executed_action for agent in state.agents},
            action_distributions=deepcopy(distributions),
            action_masks=deepcopy(self.environment.action_masks()),
            proposed_actions=dict(proposed),
            executed_actions={
                agent.agent_id: agent.last_executed_action
                for agent in state.agents
            },
            rewards=dict(state.last_rewards),
            metadata={
                "contract_version": "collaborative_snapshot_v1",
                "episode_id": state.episode_id,
                "task_state": _task_state(state),
                "shared_tasks": [_task_record(task) for task in state.tasks],
                "charging_state": _charging_state(
                    state, self.environment.layout.charger_position
                ),
                "user_score": state.user_score,
                "score_breakdown": dict(state.score_breakdown),
                "robot_collision_events": state.robot_collision_events,
                "shutdown_count": state.shutdown_count,
                "decision_evidence_aligned": False,
            },
            checkpoint_id=f"episode_{state.episode_id}:frame_{state.frame}",
        )

    def restore(
        self,
        snapshot: EnvironmentSnapshot,
        policy: PolicyProtocol | None = None,
    ) -> None:
        if snapshot.environment != self.environment.environment_name:
            raise ValueError(
                f"Cannot restore {snapshot.environment} into "
                f"{self.environment.environment_name}."
            )
        self.environment.set_state(deepcopy(snapshot.state))
        self.environment.set_rng_state(deepcopy(snapshot.environment_rng_state))
        if policy is not None and snapshot.policy_rng_state is not None:
            policy.set_rng_state(deepcopy(snapshot.policy_rng_state))

    @staticmethod
    def _set_agent_property(
        agent: AgentState,
        property_name: str,
        value: Any,
    ) -> None:
        if property_name == "position":
            coordinate = tuple(int(item) for item in value)
            if len(coordinate) != 2 or not is_passable(coordinate):
                raise ValueError("position must be a passable grid coordinate")
            agent.position = coordinate
        elif property_name == "battery":
            battery = float(value)
            if not 0.0 <= battery <= 100.0:
                raise ValueError("battery must be in [0, 100]")
            agent.battery = battery
            agent.active = battery > 0.0
        elif property_name == "heading":
            heading = str(value)
            if heading not in MOVE_DELTAS:
                raise ValueError("heading must be a movement action")
            agent.heading = heading
        elif property_name == "last_action":
            action = str(value)
            if action not in ACTIONS:
                raise ValueError("last_action must be a warehouse action")
            agent.last_action = action
        elif property_name == "active":
            agent.active = bool(value)
            if not agent.active:
                agent.battery = 0.0
        else:
            raise ValueError(f"Unsupported editable robot property: {property_name}")

    def validate_intervention(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> tuple[bool, tuple[str, ...]]:
        state: WarehouseState = deepcopy(snapshot.state)
        errors: list[str] = []
        for item in interventions:
            try:
                agent = state.by_id(item.entity_id)
                self._set_agent_property(agent, item.property_name, item.value)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        if not errors:
            self.environment._refresh_navigation_goals(state)
            errors.extend(self.environment.validate_state(state))
        return not errors, tuple(errors)

    def apply_interventions(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> EnvironmentSnapshot:
        valid, errors = self.validate_intervention(snapshot, interventions)
        if not valid:
            raise ValueError("Invalid intervention: " + "; ".join(errors))
        state: WarehouseState = deepcopy(snapshot.state)
        for item in interventions:
            self._set_agent_property(
                state.by_id(item.entity_id),
                item.property_name,
                item.value,
            )
        return self.refresh_snapshot(
            replace(
                snapshot,
                state=state,
                metadata={
                    **dict(snapshot.metadata),
                    "intervened": True,
                    "decision_evidence_aligned": False,
                    "environment_events": (),
                    "action_resolution": {},
                },
            )
        )

    def refresh_snapshot(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> EnvironmentSnapshot:
        state: WarehouseState = deepcopy(snapshot.state)
        self.environment._refresh_navigation_goals(state)
        observations = all_local_observations(state, self.environment.config)
        global_state = global_observation(state, self.environment.config)
        masks = {
            agent.agent_id: legal_action_mask(state, agent)
            for agent in state.agents
        }
        return replace(
            snapshot,
            state=state,
            observations=deepcopy(observations),
            global_state=deepcopy(global_state),
            action_masks=masks,
            action_distributions={},
            proposed_actions={},
            executed_actions={},
            metadata={
                **dict(snapshot.metadata),
                "task_state": _task_state(state),
                "shared_tasks": [_task_record(task) for task in state.tasks],
                "charging_state": _charging_state(
                    state, self.environment.layout.charger_position
                ),
            },
        )

    def compile_relational_constraints(
        self,
        snapshot: EnvironmentSnapshot,
        constraints: Sequence[Mapping[str, Any]],
    ) -> tuple[Sequence[Intervention], tuple[str, ...]]:
        del snapshot
        if not constraints:
            return (), ()
        return (), (
            "Relational scene editing is not part of the collaborative study.",
        )

    def recompute_observations(self) -> Mapping[str, Any]:
        return self.environment.observations()

    def global_state(self) -> Any:
        return self.environment.global_state()

    def rollout(
        self,
        policy: PolicyProtocol,
        *,
        horizon: int,
        deterministic: bool = False,
    ) -> RolloutResult:
        frames: list[RolloutFrame] = []
        counts = {
            agent_id: {action: 0 for action in ACTIONS}
            for agent_id in self.environment.agent_ids
        }
        terminal_reason: str | None = None
        for _ in range(max(0, int(horizon))):
            if self.environment.state is None:
                break
            if self.environment.state.terminated or self.environment.state.truncated:
                terminal_reason = self.environment.state.terminal_reason
                break
            observations = self.environment.observations()
            global_state = self.environment.global_state()
            masks = self.environment.action_masks()
            decision = self.snapshot(policy)
            proposed_actions, distributions = policy.act(
                observations,
                global_state,
                deterministic=deterministic,
            )
            actions = dict(proposed_actions)
            decision = replace(
                decision,
                observations=deepcopy(observations),
                global_state=deepcopy(global_state),
                action_distributions=deepcopy(distributions),
                action_masks=deepcopy(masks),
                proposed_actions=dict(proposed_actions),
                metadata={
                    **dict(decision.metadata),
                    "action_execution": "independent_simultaneous_mappo_actor",
                    "submitted_actions": dict(actions),
                },
            )
            _, rewards, terminated, truncated, info = self.environment.step(actions)
            next_snapshot = self.snapshot(policy)
            for agent_id, action in info.get("executed_actions", actions).items():
                counts[agent_id][action] += 1
            frames.append(
                RolloutFrame(
                    frame=decision.frame,
                    snapshot=decision,
                    observations=deepcopy(observations),
                    actions=dict(proposed_actions),
                    distributions=dict(distributions),
                    reward=dict(rewards),
                    done=terminated or truncated,
                    info=dict(info),
                    proposed_actions=dict(proposed_actions),
                    executed_actions=dict(info.get("executed_actions", actions)),
                    action_masks=deepcopy(masks),
                    next_snapshot=next_snapshot,
                    reward_breakdown=dict(info.get("reward_breakdown", {})),
                    task_state=_task_state(next_snapshot.state),
                    charging_state={
                        **_charging_state(
                            next_snapshot.state,
                            self.environment.layout.charger_position,
                        ),
                        "used": bool(info.get("charger_used", False)),
                        "energy_gained": float(info.get("charger_energy_gained", 0.0)),
                    },
                    environment_events=_transition_events(info),
                    rng_state={
                        "environment": deepcopy(decision.environment_rng_state),
                        "policy": deepcopy(decision.policy_rng_state),
                    },
                    checkpoint_id=decision.checkpoint_id,
                )
            )
            if terminated or truncated:
                terminal_reason = info.get("terminal_reason")
                break
        frequencies = {
            agent_id: {
                action: count / max(1, sum(action_counts.values()))
                for action, count in action_counts.items()
            }
            for agent_id, action_counts in counts.items()
        }
        return RolloutResult(tuple(frames), frequencies, terminal_reason)

    def render(self, state: Any | None = None) -> Any:
        return self.environment.render_ascii(state)

    def policy_state(
        self,
        snapshot: EnvironmentSnapshot,
        agent_id: str,
    ) -> WarehousePolicyState:
        snapshot.state.by_id(agent_id)
        return WarehousePolicyState(snapshot=snapshot, agent_id=agent_id)

    def default_target_entity(self, snapshot: EnvironmentSnapshot) -> str:
        del snapshot
        return "robot_2"

    def policy_distribution(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Mapping[str, float]:
        cached = snapshot.action_distributions.get(target_entity)
        if cached is not None:
            return dict(zip(cached.actions, cached.probabilities))
        _, distributions = policy.act(
            snapshot.observations,
            snapshot.global_state,
            deterministic=True,
        )
        distribution = distributions[target_entity]
        return dict(zip(distribution.actions, distribution.probabilities))

    def relational_features(
        self,
        policy_state: WarehousePolicyState,
    ) -> Mapping[str, float]:
        state: WarehouseState = policy_state.snapshot.state
        agent = state.by_id(policy_state.agent_id)
        other = next(item for item in state.agents if item.agent_id != agent.agent_id)
        layout_id = self.environment.config.map_layout_id
        layout = self.environment.layout
        nearest_dead_end = min(
            (
                shortest_path_distance(agent.position, bay, layout_id)
                for bay in layout.dead_end_positions
                if bay != other.position
            ),
            default=layout.rows * layout.cols,
        )
        same_corridor_axis = (
            agent.position[0] == other.position[0]
            or agent.position[1] == other.position[1]
        )
        self_priority = (
            agent.carrying_task_id is not None
            and other.carrying_task_id is None
        ) or (
            (agent.carrying_task_id is None)
            == (other.carrying_task_id is None)
            and agent.agent_id == "robot_1"
        )
        features: dict[str, float] = {
            "self.row": float(agent.position[0]),
            "self.column": float(agent.position[1]),
            "self.battery_percent": float(agent.battery),
            "self.active": float(agent.active),
            "self.carrying_shared_task": float(agent.carrying_task_id is not None),
            "self.at_charger": float(agent.position == layout.charger_position),
            "self.deliveries_completed": float(agent.deliveries_completed),
            "goal.is_pickup": float(agent.goal_kind == "pickup"),
            "goal.is_delivery": float(agent.goal_kind == "delivery"),
            "goal.is_charge": float(agent.goal_kind == "charge"),
            "goal.is_wait": float(agent.goal_kind == "wait"),
            "goal.distance": float(
                shortest_path_distance(agent.position, agent.goal_position, layout_id)
            ),
            "goal.row_delta": float(agent.goal_position[0] - agent.position[0]),
            "goal.column_delta": float(agent.goal_position[1] - agent.position[1]),
            "other.nearest_distance": float(_manhattan(agent.position, other.position)),
            "other.nearest_row_delta": float(other.position[0] - agent.position[0]),
            "other.nearest_column_delta": float(other.position[1] - agent.position[1]),
            "other.nearest_battery_percent": float(other.battery),
            "other.carrying_shared_task": float(other.carrying_task_id is not None),
            "other.active": float(other.active),
            "charger.distance": float(
                shortest_path_distance(agent.position, layout.charger_position, layout_id)
            ),
            "charger.occupied": float(
                any(item.position == layout.charger_position for item in state.agents)
            ),
            "corridor.self_in_topological_dead_end": float(
                agent.position in layout.dead_end_positions
            ),
            "corridor.teammate_in_topological_dead_end": float(
                other.position in layout.dead_end_positions
            ),
            "corridor.nearest_topological_dead_end_distance": float(
                nearest_dead_end
            ),
            "corridor.same_axis": float(same_corridor_axis),
            "corridor.head_on_risk": float(
                same_corridor_axis and _manhattan(agent.position, other.position) <= 4
            ),
            "corridor.self_has_priority": float(self_priority),
            "corridor.teammate_has_priority": float(not self_priority),
            "team.delivery_count": float(state.total_deliveries),
            "team.robot_collision_count": float(state.robot_collision_events),
            "team.shutdown_count": float(state.shutdown_count),
            "score.human_detour_units": float(state.human_route_regret_units),
            "time.progress_fraction": float(
                state.frame / max(1, self.environment.config.horizon)
            ),
        }
        for action in ACTIONS:
            features[f"self.previous_action.{action}"] = float(
                agent.last_action == action
            )
            features[f"other.previous_action.{action}"] = float(
                other.last_action == action
            )
        for slot, task in enumerate(
            sorted(state.tasks, key=lambda item: item.task_id),
            start=1,
        ):
            prefix = f"task.slot_{slot}"
            features.update(
                {
                    f"{prefix}.available": float(task.status == "available"),
                    f"{prefix}.carried_by_self": float(
                        task.carrier_agent_id == agent.agent_id
                    ),
                    f"{prefix}.carried_by_other": float(
                        task.carrier_agent_id == other.agent_id
                    ),
                    f"{prefix}.pickup_distance": float(
                        shortest_path_distance(agent.position, task.pickup_position, layout_id)
                    ),
                    f"{prefix}.delivery_distance": float(
                        shortest_path_distance(agent.position, task.delivery_position, layout_id)
                    ),
                    f"{prefix}.route_distance": float(
                        shortest_path_distance(
                            task.pickup_position,
                            task.delivery_position,
                            layout_id,
                        )
                    ),
                }
            )

        legal_mask = legal_action_mask(state, agent, layout_id)
        current_distance = shortest_path_distance(
            agent.position, agent.goal_position, layout_id
        )
        progress_by_action: dict[str, float] = {}
        for index, action in enumerate(ACTIONS):
            delta = MOVE_DELTAS.get(action, (0, 0))
            proposed = (
                agent.position[0] + delta[0],
                agent.position[1] + delta[1],
            )
            static_legal = action == "WAIT" or is_passable(proposed, layout_id)
            blocked_by_robot = action != "WAIT" and proposed == other.position
            legal = bool(legal_mask[index])
            effective = proposed if legal else agent.position
            other_delta = MOVE_DELTAS.get(other.last_action, (0, 0))
            other_target = (
                other.position[0] + other_delta[0],
                other.position[1] + other_delta[1],
            )
            same_cell = action != "WAIT" and proposed == other_target
            swap = (
                action != "WAIT"
                and proposed == other.position
                and other_target == agent.position
            )
            effective_progress = float(
                current_distance
                - shortest_path_distance(effective, agent.goal_position, layout_id)
            )
            geometric_progress = float(
                _manhattan(agent.position, agent.goal_position)
                - _manhattan(proposed, agent.goal_position)
            )
            prefix = f"candidate.{action}"
            features.update(
                {
                    f"{prefix}.legal": float(legal),
                    f"{prefix}.geometric_goal_progress": geometric_progress,
                    f"{prefix}.effective_goal_progress": effective_progress,
                    f"{prefix}.goal_progress": effective_progress,
                    f"{prefix}.blocked_by_static_obstacle": float(
                        action != "WAIT" and not static_legal
                    ),
                    f"{prefix}.blocked_by_robot": float(blocked_by_robot),
                    f"{prefix}.predicted_same_cell_conflict": float(same_cell),
                    f"{prefix}.predicted_swap_conflict": float(swap),
                    f"{prefix}.charger_progress": float(
                        shortest_path_distance(
                            agent.position, layout.charger_position, layout_id
                        )
                        - shortest_path_distance(
                            effective, layout.charger_position, layout_id
                        )
                    ),
                    f"{prefix}.enters_topological_dead_end": float(
                        effective in layout.dead_end_positions
                        and effective != agent.position
                    ),
                    f"{prefix}.nearest_other_distance_after": float(
                        _manhattan(effective, other.position)
                    ),
                }
            )
            progress_by_action[action] = effective_progress
        best = max(progress_by_action.values(), default=0.0)
        for action, progress in progress_by_action.items():
            features[f"candidate.{action}.avoidable_detour"] = max(
                0.0,
                best - progress,
            )
        return features

    def semantic_policy_context(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> SemanticPolicyContext:
        state: WarehouseState = snapshot.state
        agent = state.by_id(target_entity)
        other = next(item for item in state.agents if item.agent_id != target_entity)
        features = dict(self.relational_features(self.policy_state(snapshot, target_entity)))
        bindings: dict[str, str] = {
            "teammate": other.agent_id,
            "nearest_agent": other.agent_id,
        }
        reasons: dict[str, tuple[str, ...]] = {}
        tags: set[str] = set()
        for action in ACTIONS:
            prefix = f"candidate.{action}"
            active = tuple(
                reason
                for reason in (
                    "blocked_by_static_obstacle",
                    "blocked_by_robot",
                    "predicted_same_cell_conflict",
                    "predicted_swap_conflict",
                )
                if features.get(f"{prefix}.{reason}", 0.0) > 0.5
            )
            reasons[action] = active
            if "blocked_by_robot" in active:
                bindings[f"{prefix}.blocker"] = other.agent_id
                tags.add("occupied_progress")
            if "predicted_same_cell_conflict" in active:
                bindings[f"{prefix}.conflicting_agent"] = other.agent_id
                tags.add("same_cell_conflict")
            if "predicted_swap_conflict" in active:
                bindings[f"{prefix}.conflicting_agent"] = other.agent_id
                tags.add("swap_conflict")
        if not tags:
            tags.add("ordinary")
        provenance = {
            feature: (
                "action_mask"
                if feature.endswith(".legal")
                else "shared_tasks"
                if feature.startswith("task.")
                else "teammate_observation"
                if feature.startswith("other.")
                else "local_geometry"
                if feature.startswith(("candidate.", "corridor."))
                else "self_observation"
            )
            for feature in features
        }
        return SemanticPolicyContext(
            features=features,
            entity_bindings=bindings,
            feature_provenance=provenance,
            scenario_tags=tuple(sorted(tags)),
            action_constraint_reasons=reasons,
        )

    def semantic_policy_features(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> Mapping[str, float]:
        return self.semantic_policy_context(snapshot, target_entity).features

    def semantic_feature_entity_bindings(
        self,
        feature: str,
        context: SemanticPolicyContext,
    ) -> Mapping[str, str]:
        if feature.startswith("other."):
            return {
                "teammate": context.entity_bindings.get("teammate", "robot_1")
            }
        if feature.startswith("candidate."):
            action = feature.split(".", 2)[1]
            prefix = f"candidate.{action}."
            return {
                role: entity
                for role, entity in context.entity_bindings.items()
                if role.startswith(prefix)
            }
        return {}

    def semantic_feature_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        descriptions: dict[str, Mapping[str, str]] = {
            "self.battery_percent": {"zh": "当前电量", "en": "current battery"},
            "self.carrying_shared_task": {"zh": "是否承运共享任务", "en": "carrying a shared task"},
            "goal.distance": {"zh": "到当前目标的最短路径", "en": "shortest path to current goal"},
            "other.nearest_distance": {"zh": "与队友的距离", "en": "distance to teammate"},
            "other.nearest_battery_percent": {"zh": "队友电量", "en": "teammate battery"},
            "team.delivery_count": {"zh": "团队配送数", "en": "team delivery count"},
        }
        for slot in (1, 2):
            for suffix, zh, en in (
                ("available", "任务可认领", "task available"),
                ("carried_by_self", "任务由本机器人承运", "task carried by this robot"),
                ("carried_by_other", "任务由队友承运", "task carried by teammate"),
                ("pickup_distance", "到任务A点的距离", "distance to task A"),
                ("delivery_distance", "到任务B点的距离", "distance to task B"),
                ("route_distance", "任务A到B的距离", "task A-to-B distance"),
            ):
                descriptions[f"task.slot_{slot}.{suffix}"] = {"zh": zh, "en": en}
        for action in ACTIONS:
            for suffix, zh, en in (
                ("legal", "动作是否合法", "whether the action is legal"),
                ("geometric_goal_progress", "几何目标进展", "geometric goal progress"),
                ("effective_goal_progress", "实际可行目标进展", "effective goal progress"),
                ("blocked_by_static_obstacle", "是否被墙或货架阻挡", "blocked by wall or shelf"),
                ("blocked_by_robot", "是否被队友占位阻挡", "blocked by teammate occupancy"),
                ("predicted_same_cell_conflict", "是否预测同格冲突", "predicted same-cell conflict"),
                ("predicted_swap_conflict", "是否预测交换冲突", "predicted swap conflict"),
                ("avoidable_detour", "相对最优动作的绕路量", "detour versus best feasible action"),
            ):
                descriptions[f"candidate.{action}.{suffix}"] = {"zh": zh, "en": en}
        return descriptions

    def semantic_feature_observation(
        self,
        feature: str,
        value: float,
    ) -> Mapping[str, Any]:
        description = self.semantic_feature_descriptions().get(feature, {})
        return {
            "feature": feature,
            "value": float(value),
            "zh": description.get("zh", feature),
            "en": description.get("en", feature),
        }

    def action_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        return {
            action: {"zh": _action_zh(action), "en": action.lower()}
            for action in ACTIONS
        }

    def objective_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        return {
            goal: {
                "zh": _goal_label(goal, "zh-CN"),
                "en": _goal_label(goal, "en"),
            }
            for goal in ("pickup", "delivery", "charge", "wait")
        }

    def question_vocabulary(self) -> Mapping[str, Any]:
        return {
            "query_variables": {
                "observed_action": {
                    "kind": "action",
                    "aliases": (
                        "recorded action",
                        "executed action",
                        "实际动作",
                        "为什么这样做",
                    ),
                },
                "objective": {
                    "kind": "objective",
                    "aliases": ("objective", "goal", "task", "目标", "任务"),
                },
            },
            "objectives": {
                key: {**value, "aliases": (key, value["zh"], value["en"])}
                for key, value in self.objective_descriptions().items()
            },
            "action_values": {
                key: {**value, "aliases": (key, value["zh"], value["en"])}
                for key, value in self.action_descriptions().items()
            },
        }


    def action_legality_features(self) -> Mapping[str, str]:
        return {action: f"candidate.{action}.legal" for action in ACTIONS}

    def action_constraint_reason_features(self) -> Mapping[str, Mapping[str, str]]:
        reasons = (
            "blocked_by_static_obstacle",
            "blocked_by_robot",
            "predicted_same_cell_conflict",
            "predicted_swap_conflict",
        )
        return {
            action: {
                reason: f"candidate.{action}.{reason}"
                for reason in reasons
            }
            for action in ACTIONS
        }

    def required_program_predicate_groups(self) -> Mapping[str, tuple[str, ...]]:
        return {
            "shared_task_state": (
                "self.carrying_shared_task",
                "task.slot_1.available",
                "task.slot_1.carried_by_self",
                "task.slot_1.carried_by_other",
                "task.slot_2.available",
                "task.slot_2.carried_by_self",
                "task.slot_2.carried_by_other",
            ),
            "energy_state": ("self.battery_percent",),
            "multiagent_relation": (
                "other.nearest_distance",
                "other.nearest_battery_percent",
                *tuple(
                    f"candidate.{action}.{reason}"
                    for action in ACTIONS
                    for reason in (
                        "blocked_by_robot",
                        "predicted_same_cell_conflict",
                        "predicted_swap_conflict",
                    )
                ),
            ),
        }

    def relation_role_definitions(self) -> Mapping[str, str]:
        return {
            "teammate": "the other robot visible at the decision frame",
            "candidate.<ACTION>.blocker": "the teammate occupying the candidate destination",
            "candidate.<ACTION>.conflicting_agent": "the teammate whose prior action predicts a conflict",
        }

    def _decision_evidence_state(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> WarehouseState:
        """Apply recorded decision goals to an explanation-only state copy.

        Available warehouse tasks remain publicly unassigned, so an empty
        robot's persistent navigation goal is deliberately ``wait``.  Offline
        reference controllers can nevertheless use a temporary frozen task
        match for one decision.  When that exact match was recorded in the
        snapshot, explanations must use it instead of inventing a reason from
        the public wait goal.
        """

        state: WarehouseState = deepcopy(snapshot.state)
        if not bool(snapshot.metadata.get("decision_evidence_aligned", False)):
            return state
        raw_overrides = snapshot.metadata.get("decision_goal_overrides", {})
        if not isinstance(raw_overrides, Mapping):
            return state
        for raw_agent_id, raw_position in raw_overrides.items():
            try:
                agent = state.by_id(str(raw_agent_id))
                goal = tuple(int(item) for item in raw_position)
            except (KeyError, TypeError, ValueError):
                continue
            if len(goal) != 2 or not self.environment.layout.is_passable(goal):
                continue
            agent.navigation_goal_position = goal
            if goal == self.environment.layout.charger_position:
                agent.navigation_goal_kind = "charge"
            elif agent.carrying_task_id is not None:
                agent.navigation_goal_kind = "delivery"
            elif any(
                task.status == "available" and task.pickup_position == goal
                for task in state.tasks
            ):
                agent.navigation_goal_kind = "pickup"
            elif goal == agent.position:
                agent.navigation_goal_kind = "wait"
        return state

    def _objective_fact(
        self,
        state: WarehouseState,
        agent: AgentState,
    ) -> EvidenceFact:
        # Preserve the live task-list order because the UI labels these slots
        # A1/B1 and A2/B2 independently of monotonically increasing task IDs.
        tasks = list(state.tasks)
        visible_goal_kind, visible_goal_position = _actor_visible_goal(state, agent)
        selected = next(
            (task for task in tasks if task.task_id == agent.carrying_task_id),
            None,
        )
        if selected is None:
            selected = next(
                (
                    task
                    for task in tasks
                    if task.status == "available"
                    and task.pickup_position == visible_goal_position
                ),
                None,
            )
        route_distance = shortest_path_distance(
            agent.position,
            visible_goal_position,
            self.environment.config.map_layout_id,
        )
        requirements = [
            {
                "key": "objective.selected_objective",
                "semantic_name": "selected_objective",
                "role": "objective_reason",
                "group": "objective",
                "value": visible_goal_kind,
                "selection_basis": "shared_task_context",
                "decision_features": [f"goal.is_{visible_goal_kind}"],
                "fact_verbalizations": [
                    f"机器人当前目标是{_goal_label(visible_goal_kind, 'zh-CN')}。",
                    f"The robot's current objective is {_goal_label(visible_goal_kind, 'en')}.",
                ],
            },
            {
                "key": "objective.target_position",
                "semantic_name": "target_position",
                "role": "objective_reason",
                "group": "route",
                "value": visible_goal_position,
                "selection_basis": "shared_task_context",
                "decision_features": ["goal.distance", "candidate."],
                "fact_verbalizations": [
                    f"目标坐标是{visible_goal_position}。",
                    f"The target is at {visible_goal_position}.",
                ],
            },
            {
                "key": "objective.route_distance",
                "semantic_name": "route_distance",
                "role": "objective_reason",
                "group": "route",
                "value": route_distance,
                "selection_basis": "shared_task_context",
                "decision_features": ["goal.distance"],
                "fact_verbalizations": [
                    f"到目标的最短可通行路径为{route_distance}格。",
                    f"The shortest passable route to the target is {route_distance} cells.",
                ],
            },
        ]
        if selected is not None:
            slot = tasks.index(selected) + 1
            requirements.append(
                {
                    "key": "objective.shared_task",
                    "semantic_name": "shared_task",
                    "role": "objective_reason",
                    "group": "task_status",
                    "value": selected.task_id,
                    "selection_basis": "shared_task_context",
                    "decision_features": [f"task.slot_{slot}."],
                    "fact_verbalizations": [
                        f"证据绑定任务{selected.task_id}，A={selected.pickup_position}，B={selected.delivery_position}。",
                        f"The evidence is bound to task {selected.task_id}, A={selected.pickup_position}, B={selected.delivery_position}.",
                    ],
                }
            )
        value = {
            "schema": "shared_objective_selection_reason.v2",
            "evidence_frame": int(state.frame),
            "selected_objective": {
                "id": visible_goal_kind,
                "target_position": visible_goal_position,
                "task_id": selected.task_id if selected else None,
            },
            "task_state": {
                "current_position": agent.position,
                "battery_percent": round(float(agent.battery), 1),
                "carrying_task_id": agent.carrying_task_id,
                "deliveries_completed": int(agent.deliveries_completed),
            },
            "active_shared_tasks": [_task_record(task) for task in tasks],
            "locations": {"charger": self.environment.layout.charger_position},
            "route_considered": [
                {"stage": "current_position", "position": agent.position},
                {"stage": visible_goal_kind, "position": visible_goal_position},
            ],
            "decision_conditions": [
                {"name": "carrying_task_id", "value": agent.carrying_task_id},
                {"name": "battery_percent", "value": round(float(agent.battery), 1)},
                {"name": "route_distance", "value": route_distance},
            ],
            "explanation_requirements": requirements,
            "shared_explanation_requirements": list(requirements),
        }
        return EvidenceFact(
            fact_id=f"{agent.agent_id}.objective_reason",
            predicate="shared_objective_selection_reason",
            arguments=(
                agent.agent_id,
                visible_goal_kind,
                selected.task_id if selected else "none",
                int(state.frame),
            ),
            value=value,
            factor_groups=("objective_reason", "goal", "shared_task", "rationale", "state"),
            verbalizations=(),
            value_verbalizations=(
                visible_goal_kind,
                _goal_label(visible_goal_kind, "zh-CN"),
                _goal_label(visible_goal_kind, "en"),
            ),
        )

    def evidence_facts(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Sequence[EvidenceFact]:
        state = self._decision_evidence_state(snapshot)
        agent = state.by_id(target_entity)
        probabilities = self.policy_distribution(snapshot, target_entity, policy)
        argmax = max(probabilities, key=probabilities.__getitem__)
        aligned = bool(snapshot.metadata.get("decision_evidence_aligned", False))
        proposed = snapshot.proposed_actions.get(target_entity, argmax) if aligned else argmax
        executed = snapshot.executed_actions.get(target_entity, proposed) if aligned else proposed
        facts: list[EvidenceFact] = [
            EvidenceFact(
                fact_id=f"{target_entity}.executed_action",
                predicate="executed_action",
                arguments=(target_entity,),
                value=executed,
                factor_groups=("action", "action_reason"),
                verbalizations=(
                    f"{target_entity} executed {executed}.",
                    f"{target_entity}实际执行了{_action_zh(str(executed))}。",
                ),
                value_verbalizations=(str(executed), _action_zh(str(executed))),
            ),
            EvidenceFact(
                fact_id=f"{target_entity}.proposed_action",
                predicate="proposed_action",
                arguments=(target_entity,),
                value=proposed,
                factor_groups=("action",),
                verbalizations=(
                    f"{target_entity} proposed {proposed}.",
                    f"{target_entity}提出了{_action_zh(str(proposed))}。",
                ),
                value_verbalizations=(str(proposed), _action_zh(str(proposed))),
            ),
            EvidenceFact(
                fact_id=f"{target_entity}.battery",
                predicate="battery",
                arguments=(target_entity,),
                value=round(agent.battery, 1),
                factor_groups=("energy", "rationale", "state"),
                verbalizations=(
                    f"{target_entity} battery was {agent.battery:.1f}%.",
                    f"{target_entity}电量为{agent.battery:.1f}%。",
                ),
                value_verbalizations=(f"{agent.battery:.1f}%",),
            ),
            EvidenceFact(
                fact_id=f"{target_entity}.position",
                predicate="position",
                arguments=(target_entity,),
                value=agent.position,
                factor_groups=("location", "state"),
                verbalizations=(
                    f"{target_entity} was at {agent.position}.",
                    f"{target_entity}位于{agent.position}。",
                ),
            ),
            self._objective_fact(state, agent),
        ]
        decision_trace = snapshot.metadata.get("decision_trace", {})
        if isinstance(decision_trace, Mapping):
            agent_trace = dict(
                decision_trace.get("agents", {})
            ).get(target_entity)
            if isinstance(agent_trace, Mapping):
                facts.append(
                    EvidenceFact(
                        fact_id=f"{target_entity}.decision_trace",
                        predicate="decision_trace",
                        arguments=(target_entity,),
                        value={
                            "schema_version": decision_trace.get(
                                "schema_version"
                            ),
                            "decision_frame": decision_trace.get(
                                "decision_frame"
                            ),
                            "outcome_frame": decision_trace.get(
                                "outcome_frame"
                            ),
                            "pre_state_hash": decision_trace.get(
                                "pre_state_hash"
                            ),
                            "post_state_hash": decision_trace.get(
                                "post_state_hash"
                            ),
                            "fact_valid": bool(
                                decision_trace.get("fact_valid", False)
                            ),
                            "fact_validation_failures": tuple(
                                decision_trace.get(
                                    "fact_validation_failures", ()
                                )
                            ),
                            **dict(agent_trace),
                        },
                        factor_groups=(
                            "decision_trace",
                            "action_reason",
                            "rationale",
                        ),
                        verbalizations=(),
                        value_verbalizations=(),
                    )
                )
        resolution = dict(snapshot.metadata.get("action_resolution", {})).get(
            target_entity,
            {},
        )
        if aligned and resolution:
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.action_resolution",
                    predicate="action_resolution_reason",
                    arguments=(target_entity, str(proposed), str(executed)),
                    value=dict(resolution),
                    factor_groups=("action", "action_reason", "coordination"),
                    verbalizations=(
                        f"The environment resolved {proposed} to {executed} because {resolution.get('blocked_reason') or 'it was feasible'}.",
                        f"环境把{_action_zh(str(proposed))}解析为{_action_zh(str(executed))}，原因是{resolution.get('blocked_reason') or '动作可执行'}。",
                    ),
                )
            )
        environment_events = snapshot.metadata.get("environment_events", ())
        charging_event = next(
            (
                event
                for event in environment_events
                if isinstance(event, Mapping)
                and str(event.get("event", "")) == "charging"
                and float(event.get("energy_gained", 0.0)) > 0.0
            ),
            None,
        )
        teammate = next(
            item for item in state.agents if item.agent_id != target_entity
        )
        if (
            aligned
            and str(executed) == "WAIT"
            and agent.position != self.environment.layout.charger_position
            and self.environment._requires_charge(state, agent)
            and teammate.position == self.environment.layout.charger_position
            and str(snapshot.executed_actions.get(teammate.agent_id, "WAIT")) == "WAIT"
            and charging_event is not None
        ):
            energy_gained = float(charging_event.get("energy_gained", 0.0))
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.charger_queue_context",
                    predicate="charger_queue_context",
                    arguments=(target_entity, teammate.agent_id),
                    value={
                        "battery": float(agent.battery),
                        "position": agent.position,
                        "charger_position": self.environment.layout.charger_position,
                        "charger_distance": shortest_path_distance(
                            agent.position,
                            self.environment.layout.charger_position,
                            self.environment.config.map_layout_id,
                        ),
                        "occupant_agent": teammate.agent_id,
                        "occupant_battery_before": float(teammate.battery),
                        "occupant_battery_after": min(
                            100.0,
                            float(teammate.battery) + energy_gained,
                        ),
                        "occupant_energy_gained": energy_gained,
                        "executed_action": str(executed),
                    },
                    factor_groups=(
                        "action",
                        "action_reason",
                        "energy",
                        "coordination",
                        "multiagent",
                    ),
                    verbalizations=(),
                )
            )
        if (
            aligned
            and str(executed) == "WAIT"
            and agent.position == self.environment.layout.charger_position
            and charging_event is not None
        ):
            energy_gained = float(charging_event.get("energy_gained", 0.0))
            battery_before = float(agent.battery)
            battery_after = min(100.0, battery_before + energy_gained)
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.charging_outcome",
                    predicate="charging_outcome",
                    arguments=(target_entity,),
                    value={
                        "battery_before": battery_before,
                        "energy_gained": energy_gained,
                        "battery_after": battery_after,
                        "charge_required": self.environment._requires_charge(
                            state,
                            agent,
                        ),
                        "next_task": _post_charge_task_context(
                            state,
                            agent,
                            self.environment,
                        ),
                    },
                    factor_groups=("action", "action_reason", "energy", "transition"),
                    verbalizations=(
                        f"{target_entity} recovered {energy_gained:g} battery points by waiting at the charger.",
                        f"{target_entity}在充电站等待后恢复了{energy_gained:g}点电量。",
                    ),
                    value_verbalizations=(f"{energy_gained:g}",),
                )
            )
        if aligned and str(executed) in MOVE_DELTAS:
            row_delta, column_delta = MOVE_DELTAS[str(executed)]
            position_before = agent.position
            position_after = (
                position_before[0] + row_delta,
                position_before[1] + column_delta,
            )
            work = _movement_work_context(
                state,
                agent,
                self.environment,
            )
            endpoint = work.get("endpoint")
            distance_before = (
                shortest_path_distance(
                    position_before,
                    tuple(endpoint),
                    self.environment.config.map_layout_id,
                )
                if endpoint is not None
                else 0
            )
            distance_after = (
                shortest_path_distance(
                    position_after,
                    tuple(endpoint),
                    self.environment.config.map_layout_id,
                )
                if endpoint is not None
                else 0
            )
            available_pickup_progress = tuple(
                {
                    "task_id": task.task_id,
                    "task_slot": slot,
                    "endpoint": task.pickup_position,
                    "distance_before": shortest_path_distance(
                        position_before,
                        task.pickup_position,
                        self.environment.config.map_layout_id,
                    ),
                    "distance_after": shortest_path_distance(
                        position_after,
                        task.pickup_position,
                        self.environment.config.map_layout_id,
                    ),
                }
                for slot, task in enumerate(state.tasks, start=1)
                if task.status == "available"
            )
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.movement_outcome",
                    predicate="movement_outcome",
                    arguments=(target_entity,),
                    value={
                        "action": str(executed),
                        "position_before": position_before,
                        "position_after": position_after,
                        "distance_before": distance_before,
                        "distance_after": distance_after,
                        "selected_probability": float(
                            probabilities.get(str(executed), 0.0)
                        ),
                        "highest_probability_action": str(argmax),
                        "highest_probability": float(
                            probabilities.get(str(argmax), 0.0)
                        ),
                        "policy_selected": str(proposed) == str(argmax),
                        "work": work,
                        "available_pickup_progress": available_pickup_progress,
                    },
                    factor_groups=(
                        "action",
                        "action_reason",
                        "route",
                        "transition",
                    ),
                    verbalizations=(),
                )
            )
        if aligned:
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.energy_decision_context",
                    predicate="energy_decision_context",
                    arguments=(target_entity,),
                    value=_energy_decision_context(
                        state,
                        agent,
                        self.environment,
                        executed_action=str(executed),
                    ),
                    factor_groups=(
                        "action",
                        "action_reason",
                        "energy",
                        "shared_task",
                    ),
                    verbalizations=(),
                )
            )
            teammate_id = next(
                item.agent_id
                for item in state.agents
                if item.agent_id != target_entity
            )
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.collaboration_context",
                    predicate="collaboration_context",
                    arguments=(target_entity, teammate_id),
                    value=_collaboration_context(
                        state,
                        agent,
                        self.environment,
                        proposed_action=str(proposed),
                        executed_action=str(executed),
                        executed_actions=snapshot.executed_actions,
                        action_resolution=resolution,
                        environment_events=tuple(environment_events),
                    ),
                    factor_groups=(
                        "action",
                        "action_reason",
                        "coordination",
                        "shared_task",
                        "multiagent",
                    ),
                    verbalizations=(),
                )
            )
        for task in sorted(state.tasks, key=lambda item: item.task_id):
            facts.append(
                EvidenceFact(
                    fact_id=f"{task.task_id}.shared_task",
                    predicate="shared_task",
                    arguments=(task.task_id,),
                    value=_task_record(task),
                    factor_groups=("shared_task", "state", "rationale"),
                    verbalizations=(
                        f"{task.task_id}: A={task.pickup_position}, B={task.delivery_position}, status={task.status}, carrier={task.carrier_agent_id or 'none'}.",
                        f"{task.task_id}：A点{task.pickup_position}，B点{task.delivery_position}，状态{task.status}，承运者{task.carrier_agent_id or '无'}。",
                    ),
                )
            )
        teammate = next(item for item in state.agents if item.agent_id != target_entity)
        facts.extend(
            (
                EvidenceFact(
                    fact_id=f"{teammate.agent_id}.position",
                    predicate="position",
                    arguments=(teammate.agent_id,),
                    value=teammate.position,
                    factor_groups=("coordination", "location", "state"),
                    verbalizations=(
                        f"{teammate.agent_id} was at {teammate.position}.",
                        f"{teammate.agent_id}位于{teammate.position}。",
                    ),
                ),
                EvidenceFact(
                    fact_id=f"{teammate.agent_id}.previous_action",
                    predicate="proposed_action",
                    arguments=(teammate.agent_id,),
                    value=teammate.last_action,
                    factor_groups=("coordination", "state"),
                    verbalizations=(
                        f"{teammate.agent_id}'s previous action was {teammate.last_action}.",
                        f"{teammate.agent_id}上一步动作为{_action_zh(teammate.last_action)}。",
                    ),
                ),
            )
        )
        for action, probability in probabilities.items():
            facts.append(
                EvidenceFact(
                    fact_id=f"{target_entity}.probability.{action}",
                    predicate="action_probability",
                    arguments=(target_entity, action),
                    value=round(float(probability), 6),
                    factor_groups=("action", "comparison"),
                    verbalizations=(
                        f"The policy probability of {action} was {probability:.3f}.",
                        f"策略选择{_action_zh(action)}的概率为{probability:.3f}。",
                    ),
                )
            )
        return tuple(facts)

    def factor_group_descriptions(self) -> Mapping[str, Sequence[str]]:
        return {
            "action": ("recorded proposal and executed action", "记录的提议与实际动作"),
            "objective_reason": ("current shared-task objective", "当前共享任务目标"),
            "shared_task": ("live A-to-B task state", "实时A到B任务状态"),
            "coordination": ("teammate position and conflict evidence", "队友位置与冲突证据"),
            "energy": ("battery and charging evidence", "电量与充电证据"),
        }

    def sample_local_policy_states(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        *,
        count: int,
        seed: int,
    ) -> Sequence[EnvironmentSnapshot]:
        rng = random.Random(seed)
        generated: list[EnvironmentSnapshot] = []
        passable = list(all_passable_positions())
        attempts = 0
        while len(generated) < max(0, int(count)) and attempts < max(20, count * 20):
            attempts += 1
            state: WarehouseState = deepcopy(snapshot.state)
            target = state.by_id(target_entity)
            teammate = next(item for item in state.agents if item.agent_id != target_entity)
            operation = rng.randrange(4)
            if operation == 0:
                target.battery = float(rng.randrange(5, 101, 5))
                target.active = True
            elif operation == 1:
                candidates = [
                    position
                    for position in passable
                    if position != teammate.position
                ]
                target.position = rng.choice(candidates)
            elif operation == 2:
                candidates = [
                    position
                    for position in passable
                    if position != target.position
                ]
                teammate.position = rng.choice(candidates)
            else:
                teammate.last_action = rng.choice(ACTIONS)
            self.environment._refresh_navigation_goals(state)
            if self.environment.validate_state(state):
                continue
            generated.append(
                self.refresh_snapshot(
                    replace(
                        snapshot,
                        state=state,
                        metadata={
                            **dict(snapshot.metadata),
                            "sampled_local_state": True,
                            "decision_evidence_aligned": False,
                        },
                    )
                )
            )
        return tuple(generated)

    def sample_policy_counterfactuals(
        self,
        policy_state: WarehousePolicyState,
        count: int,
        rng: random.Random,
    ) -> Sequence[WarehousePolicyState]:
        snapshots = self.sample_local_policy_states(
            policy_state.snapshot,
            policy_state.agent_id,
            count=count,
            seed=rng.randrange(2**31),
        )
        return tuple(
            WarehousePolicyState(snapshot=item, agent_id=policy_state.agent_id)
            for item in snapshots
        )

    def claim_ontology(self) -> Mapping[str, Any]:
        return {
            "predicates": tuple(self.explanation_predicate_schema()),
            "actions": ACTIONS,
            "objectives": tuple(self.objective_descriptions()),
        }

    def policy_entity_references(self) -> Mapping[str, Sequence[str]]:
        return {
            "robot_1": ("robot_1", "robot 1", "机器人1", "participant robot"),
            "robot_2": ("robot_2", "robot 2", "机器人2", "AI robot"),
        }

    def policy_entity_properties(self) -> Mapping[str, Any]:
        return dict(self.entity_schema()["agent"]["properties"])

    def canonicalize_claim_constraint(
        self,
        claim: Any,
        matched_fact: EvidenceFact,
        semantic_matcher: Any,
    ) -> Mapping[str, Any] | None:
        del semantic_matcher
        return {
            "predicate": matched_fact.predicate,
            "arguments": matched_fact.arguments,
            "value": getattr(claim, "value", matched_fact.value),
        }

    def claim_fact_compatible(self, claim: Any, fact: EvidenceFact) -> bool:
        predicate = str(getattr(claim, "predicate", "")).strip()
        return not predicate or predicate == fact.predicate

    def claim_value_consistent(
        self,
        claim: Any,
        fact: EvidenceFact,
        semantic_matcher: Any,
    ) -> bool:
        del semantic_matcher
        value = getattr(claim, "value", None)
        return value is None or str(value).lower() == str(fact.value).lower()

    def counterfactual_action_fact(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        action: str,
        distribution: Mapping[str, float],
        interventions: Sequence[Intervention],
    ) -> EvidenceFact:
        del snapshot
        return EvidenceFact(
            fact_id=f"{target_entity}.recomputed_action",
            predicate="recomputed_action",
            arguments=(target_entity,),
            value=action,
            factor_groups=("counterfactual", "action"),
            verbalizations=(
                f"After {len(interventions)} validated edit(s), the policy chose {action} with probability {distribution.get(action, 0.0):.3f}.",
                f"经过{len(interventions)}个有效修改后，策略以{distribution.get(action, 0.0):.3f}的概率选择{_action_zh(action)}。",
            ),
        )

    def sample_states_from_constraints(
        self,
        claims: Sequence[Any],
        *,
        count: int,
        seed: int,
        base_snapshot: EnvironmentSnapshot | None = None,
    ) -> Sequence[EnvironmentSnapshot]:
        del claims
        snapshot = base_snapshot or self.snapshot()
        return self.sample_local_policy_states(
            snapshot,
            "robot_2",
            count=count,
            seed=seed,
        )

    def causal_intervention_candidates(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> Sequence[CandidateIntervention]:
        state: WarehouseState = snapshot.state
        agent = state.by_id(target_entity)
        teammate = next(item for item in state.agents if item.agent_id != target_entity)
        candidates: list[CandidateIntervention] = [
            CandidateIntervention(
                candidate_id="battery_plus_20",
                factor="energy",
                description="increase current battery by 20 percentage points",
                interventions=(
                    Intervention(
                        target_entity,
                        "battery",
                        min(100.0, agent.battery + 20.0),
                    ),
                ),
            )
        ]
        for action in ACTIONS:
            candidates.append(
                CandidateIntervention(
                    candidate_id=f"teammate_previous_action_{action.lower()}",
                    factor="coordination",
                    description=f"set teammate's previous action to {action}",
                    interventions=(
                        Intervention(teammate.agent_id, "last_action", action),
                    ),
                )
            )
        return tuple(candidates)

    def recourse_intervention_candidates(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        desired_action: str,
    ) -> Sequence[CandidateIntervention]:
        del desired_action
        return self.causal_intervention_candidates(snapshot, target_entity)

    def sample_evaluation_states(
        self,
        mode: str,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        *,
        count: int,
        seed: int,
    ) -> Sequence[EnvironmentSnapshot]:
        del mode
        return self.sample_local_policy_states(
            snapshot,
            target_entity,
            count=count,
            seed=seed,
        )
