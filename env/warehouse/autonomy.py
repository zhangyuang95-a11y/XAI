"""Local same-weight autonomy ablations; never selected by the default runtime.

The physical environment, task claim/delivery, energy, score and legacy audit
state are unchanged. Only observations supplied by this subclass differ. The
old checkpoint learned with much richer features, so zeroing those features is
an out-of-distribution diagnostic, not evidence for a newly trained pure-RL AI.
The caller must also bypass post-policy action selectors and disable the
Actor's analytic priors; an observation projection alone cannot do that.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .domain import WarehouseConfig, WarehouseState
from .environment import WarehouseMultiAgentEnv
from .layouts import get_map_layout
from .navigation import ACTIONS, legal_action_mask, shortest_path_distance
from .observations import (
    NAVIGATION_GOAL_KINDS, _local_patch, _normalize_delta,
    all_local_observations, observation_dim,
)


AUTONOMY_OBSERVATION_VERSION = "warehouse_public_static_mask_ablation_v1"
STATIC_MASK_OBSERVATION_VERSION = "warehouse_static_mask_only_ablation_v1"


def projection_layout(config: WarehouseConfig) -> dict[str, Any]:
    """Expose exact legacy-compatible ranges; each pair is [start, stop)."""
    actions, goals, agents, tasks = (
        len(ACTIONS), len(NAVIGATION_GOAL_KINDS), config.max_agents,
        config.active_task_count,
    )
    own_size = 14 + goals + 5 + agents + 2 * actions + 6
    other_size = 19 + goals + 3 * actions
    canonical_robot_size = 4 + 1 + tasks + 1 + tasks + goals + 3 + actions
    canonical_size = agents * canonical_robot_size + tasks * 9
    coordination_size = 8 + 2 * actions + 2 * (9 + 6 * tasks) * actions + actions ** 2
    task_start = own_size
    other_start = task_start + tasks * 23
    canonical_start = other_start + (agents - 1) * other_size
    coordination_start = canonical_start + canonical_size
    patch_start = coordination_start + coordination_size
    own_mask_start = patch_start + (2 * config.local_patch_radius + 1) ** 2
    local_dim = own_mask_start + actions
    if local_dim != observation_dim(config):
        raise ValueError("Warehouse observation layout changed; re-audit the projection")
    own_recent_start = 13 + goals + 3 + agents + 2 + actions
    other_goal_offset = 5 + actions
    other_mask_offset = other_goal_offset + goals + 7
    other_recent_offset = other_mask_offset + actions
    control_ranges = [
        (10, 13),  # heuristic ineffective-joint-wait classification
        (13, 13 + goals + 3),
        (own_recent_start + actions + 3, own_recent_start + actions + 5),
        (coordination_start, patch_start),
    ]
    task_ranges = []
    for i in range(tasks):
        begin = task_start + i * 23
        task_ranges.append((begin, begin + 23))
        control_ranges.extend(((begin + 5, begin + 7), (begin + 17, begin + 23)))
    other_ranges, other_masks = [], []
    for i in range(agents - 1):
        begin = other_start + i * other_size
        other_ranges.append((begin, begin + other_size))
        other_masks.append((begin + other_mask_offset, begin + other_mask_offset + actions))
        control_ranges.extend((
            (begin + other_goal_offset, begin + other_goal_offset + goals + 4),
            (begin + other_goal_offset + goals + 6, begin + other_goal_offset + goals + 7),
            (begin + other_recent_offset + actions + 3, begin + other_recent_offset + actions + 5),
        ))
    canonical_robots = []
    for i in range(agents):
        begin = canonical_start + i * canonical_robot_size
        canonical_robots.append((begin, begin + canonical_robot_size))
        control_ranges.append((begin + 5 + tasks, begin + canonical_robot_size - actions))
    return {
        "version": AUTONOMY_OBSERVATION_VERSION,
        "local_dim": local_dim, "global_dim": agents * local_dim + 8,
        "own": (0, own_size), "own_recent_start": own_recent_start,
        "tasks": tuple(task_ranges), "teammates": tuple(other_ranges),
        "teammate_masks": tuple(other_masks), "teammate_recent_offset": other_recent_offset,
        "canonical_robots": tuple(canonical_robots),
        "canonical_tasks_start": canonical_start + agents * canonical_robot_size,
        "coordination": (coordination_start, patch_start),
        "patch": (patch_start, own_mask_start), "own_mask": (own_mask_start, local_dim),
        "zeroed_control_ranges": tuple(control_ranges),
        "physics_and_user_score": "unchanged_base_environment",
        "legacy_goal_and_plan_audits": "retained_in_environment_not_exposed_to_actor",
        "old_weights_out_of_distribution": True,
    }


def _recent_public_energy(state: WarehouseState, agent: Any, config: WarehouseConfig) -> tuple[float, ...]:
    age = (config.horizon if agent.last_charger_departure_frame is None
           else max(0, state.frame - agent.last_charger_departure_frame))
    return (
        *(float(agent.last_executed_action == action) for action in ACTIONS),
        max(-1., min(1., agent.last_battery_delta / config.charge_per_wait)),
        min(1., agent.steps_since_charging / max(1, config.horizon)),
        min(1., agent.charger_wait_streak / 10.),
        0., 0.,  # charge-mode policy and avoidable-wait judgement are omitted
        min(1., age / max(1, config.horizon)), float(age <= 4),
    )


def _autonomy_row(state: WarehouseState, agent_id: str, config: WarehouseConfig,
                  ranges: dict[str, Any]) -> np.ndarray:
    """Encode public facts directly, without calling goal/plan/mask heuristics."""
    row = np.zeros(ranges["local_dim"], dtype=np.float32)
    agent = state.by_id(agent_id)
    layout = get_map_layout(config.map_layout_id)
    others = sorted((a for a in state.agents if a.agent_id != agent_id), key=lambda a: a.agent_id)
    teammate = others[0]
    goal_count, action_count = len(NAVIGATION_GOAL_KINDS), len(ACTIONS)
    scale = float(config.rows * config.cols)
    distance = lambda start, goal: shortest_path_distance(start, goal, config.map_layout_id)
    row[:10] = (
        agent.position[0] / max(1, config.rows - 1),
        agent.position[1] / max(1, config.cols - 1), agent.battery / 100.,
        float(agent.active), float(agent.carrying_task_id is not None),
        _normalize_delta(layout.charger_position[0] - agent.position[0], config.rows),
        _normalize_delta(layout.charger_position[1] - agent.position[1], config.cols),
        float(agent.position == layout.charger_position),
        min(1., state.frame / max(1, config.horizon)), float(state.last_robot_collision_event),
    )
    identity_start = 13 + goal_count + 3
    agent_index = int(agent_id.rsplit("_", 1)[1]) - 1
    row[identity_start:identity_start + config.max_agents] = [
        float(i == agent_index) for i in range(config.max_agents)]
    control_start = identity_start + config.max_agents
    row[control_start:control_start + 2] = (
        float(state.participant_controlled_agent_id == agent_id),
        float(state.participant_controlled_agent_id is not None and state.participant_controlled_agent_id != agent_id),
    )
    row[control_start + 2:control_start + 2 + action_count] = [
        float(agent.last_action == action) for action in ACTIONS]
    recent = _recent_public_energy(state, agent, config)
    row[ranges["own_recent_start"]:ranges["own_recent_start"] + len(recent)] = recent
    tasks = sorted(state.tasks, key=lambda task: task.task_id)[:config.active_task_count]
    for task, (begin, _) in zip(tasks, ranges["tasks"]):
        row[begin:begin + 5] = (
            float(task.status == "available"), float(task.status == "carried"),
            float(task.carrier_agent_id is None), float(task.carrier_agent_id == agent_id),
            float(task.carrier_agent_id is not None and task.carrier_agent_id != agent_id),
        )
        row[begin + 7:begin + 17] = (
            _normalize_delta(task.pickup_position[0] - agent.position[0], config.rows),
            _normalize_delta(task.pickup_position[1] - agent.position[1], config.cols),
            _normalize_delta(task.delivery_position[0] - agent.position[0], config.rows),
            _normalize_delta(task.delivery_position[1] - agent.position[1], config.cols),
            distance(task.pickup_position, task.delivery_position) / scale,
            distance(agent.position, task.pickup_position) / scale,
            distance(agent.position, task.delivery_position) / scale,
            min(1., max(0, state.frame - task.created_frame) / max(1, config.horizon)),
            distance(teammate.position, task.pickup_position) / scale,
            distance(teammate.position, task.delivery_position) / scale,
        )
    for other, (begin, _), mask_range in zip(others, ranges["teammates"], ranges["teammate_masks"]):
        row[begin:begin + 5] = (
            _normalize_delta(other.position[0] - agent.position[0], config.rows),
            _normalize_delta(other.position[1] - agent.position[1], config.cols),
            other.battery / 100., float(other.active), float(other.carrying_task_id is not None),
        )
        row[begin + 5:begin + 5 + action_count] = [float(other.last_action == action) for action in ACTIONS]
        goal_begin = begin + 5 + action_count
        row[goal_begin + goal_count + 4:goal_begin + goal_count + 6] = (
            float(other.position == layout.charger_position),
            max(-1., min(1., (other.battery - distance(other.position, layout.charger_position)
                              * config.move_battery_cost) / 100.)),
        )
        row[slice(*mask_range)] = legal_action_mask(state, other, config.map_layout_id)
        recent = _recent_public_energy(state, other, config)
        offset = begin + ranges["teammate_recent_offset"]
        row[offset:offset + len(recent)] = recent
    task_ids = [task.task_id for task in tasks]
    for other, (begin, end) in zip(sorted(state.agents, key=lambda a: a.agent_id), ranges["canonical_robots"]):
        carry_slot = task_ids.index(other.carrying_task_id) if other.carrying_task_id in task_ids else -1
        row[begin:begin + 5 + config.active_task_count] = (
            other.position[0] / max(1, config.rows - 1), other.position[1] / max(1, config.cols - 1),
            other.battery / 100., float(other.active), float(carry_slot < 0),
            *(float(carry_slot == i) for i in range(config.active_task_count)),
        )
        row[end - action_count:end] = [float(other.last_action == action) for action in ACTIONS]
    for i, task in enumerate(tasks):
        begin = ranges["canonical_tasks_start"] + i * 9
        row[begin:begin + 9] = (
            float(task.status == "available"), float(task.status == "carried"), float(task.carrier_agent_id is None),
            float(task.carrier_agent_id == "robot_1"), float(task.carrier_agent_id == "robot_2"),
            task.pickup_position[0] / max(1, config.rows - 1), task.pickup_position[1] / max(1, config.cols - 1),
            task.delivery_position[0] / max(1, config.rows - 1), task.delivery_position[1] / max(1, config.cols - 1),
        )
    row[slice(*ranges["patch"])] = _local_patch(state, agent_id, config)
    row[slice(*ranges["own_mask"])] = legal_action_mask(state, agent, config.map_layout_id)
    if not np.isfinite(row).all():
        raise ValueError("Non-finite autonomy observation")
    return row


def autonomy_observations(state: WarehouseState, config: WarehouseConfig) -> dict[str, np.ndarray]:
    """Pure fixed-width public-state projection, with no rule-derived goals."""
    ranges = projection_layout(config)
    return {agent.agent_id: _autonomy_row(state, agent.agent_id, config, ranges) for agent in state.agents}


def static_mask_observations(state: WarehouseState, config: WarehouseConfig) -> dict[str, np.ndarray]:
    """Intermediate ablation: replace masks only; retain all original features."""
    ranges = projection_layout(config)
    rows = all_local_observations(state, config)
    for agent in state.agents:
        row = rows[agent.agent_id]
        row[slice(*ranges["own_mask"])] = legal_action_mask(state, agent, config.map_layout_id)
        others = sorted((a for a in state.agents if a.agent_id != agent.agent_id), key=lambda a: a.agent_id)
        for other, mask_range in zip(others, ranges["teammate_masks"]):
            row[slice(*mask_range)] = legal_action_mask(state, other, config.map_layout_id)
    return rows


class AutonomyWarehouseEnv(WarehouseMultiAgentEnv):
    """Same physics/score, projected Actor inputs; not a training certification.

    Base reset/step/set_state still maintain goal/plan and reward audit fields,
    including the original goal-dependent participant detour score. They are
    deliberately retained so identical submitted actions produce identical
    physical states and scores. Callers must not expose those audit fields as
    learned intentions or use them to replace actions in the autonomy mode.
    """
    observation_version = AUTONOMY_OBSERVATION_VERSION

    def observations(self) -> dict[str, np.ndarray]:
        self._require_state()
        return autonomy_observations(self.state, self.config)

    def global_state(self) -> np.ndarray:
        self._require_state()
        state = self.state
        locals_by_id = self.observations()
        tail = (
            state.frame / max(1, self.config.horizon),
            min(1., state.total_deliveries / max(1, self.config.horizon // self.config.minimum_task_distance)),
            sum(agent.active for agent in state.agents) / self.config.max_agents,
            float(any(agent.position == self.layout.charger_position for agent in state.agents)),
            min(1., state.robot_collision_events / max(1, self.config.horizon)),
            sum(task.status == "available" for task in state.tasks) / self.config.active_task_count,
            sum(task.status == "carried" for task in state.tasks) / self.config.active_task_count,
            max(-1., min(1., state.user_score / 1000.)),
        )
        return np.concatenate((locals_by_id["robot_1"], locals_by_id["robot_2"], np.asarray(tail, np.float32)))
