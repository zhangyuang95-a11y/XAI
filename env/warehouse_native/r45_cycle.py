"""r4.5 public energy and charger-cycle observation extension."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from env.warehouse_native.observations import task_order
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.r44_charger import R44SharedChargerMixin
from env.warehouse_native.r45_energy import task_energy_budget, selected_energy_budget


VERSION = "warehouse-r45-cycle-observation.v1"
_FIELDS = (
    "task0.required", "task0.margin", "task1.required", "task1.margin",
    "selected.required", "selected.margin", "departure.age",
    "departure.delivery_progress", "departure.carrying_changed",
    "recent.position_repeat", "steps_since_charging",
    "carrying.delivery_distance",
    *(f"carrying.delivery_neighbor_distance.{action}"
      for action in ("UP", "DOWN", "LEFT", "RIGHT")),
)
R45_OBSERVATION_FEATURE_NAMES = (
    "energy_cycle.last_robot_collision",
    *(f"energy_cycle.{role}.last_requested.{action}"
      for role in ("self", "other") for action in ACTIONS),
) + tuple(
    f"energy_cycle.{role}.{field}"
    for role in ("self", "other") for field in _FIELDS
)


def _clip(value, lower=-1.0, upper=1.0):
    return max(lower, min(upper, float(value)))


class R45EnergyCycleMixin(R44SharedChargerMixin):
    @property
    def observation_size(self):
        return int(super().observation_size) + len(R45_OBSERVATION_FEATURE_NAMES)

    @property
    def feature_names(self):
        return tuple(super().feature_names) + R45_OBSERVATION_FEATURE_NAMES

    def _cycle_values(self, agent_id):
        agent = self.state.by_id(agent_id)
        active = [task for task in task_order(self.state) if task.active]
        budgets = [task_energy_budget(self, agent_id, task) for task in active[:2]]
        values = []
        for index in range(2):
            row = budgets[index] if index < len(budgets) else None
            values.extend((
                _clip(row["required_battery"] / 100.0, 0.0, 2.0) if row else 0.0,
                _clip(row["battery_margin"] / 100.0) if row else 0.0,
            ))
        selected = selected_energy_budget(self, agent_id)
        values.extend((
            _clip(selected["required_battery"] / 100.0, 0.0, 2.0)
            if selected else 0.0,
            _clip(selected["battery_margin"] / 100.0) if selected else 0.0,
        ))
        age = (self.config.horizon if agent.last_charger_departure_frame is None
               else max(0, self.state.frame - agent.last_charger_departure_frame))
        delivery_progress = (
            agent.last_charger_departure_frame is not None
            and agent.deliveries_completed > agent.deliveries_at_last_charger_departure
        )
        carrying_changed = (
            agent.last_charger_departure_frame is not None
            and agent.carrying_task_id != agent.carrying_task_at_last_charger_departure
        )
        recent = tuple(agent.recent_positions)[-6:]
        repeats = 0.0 if len(recent) < 2 else 1.0 - len(set(recent)) / len(recent)
        values.extend((
            min(1.0, age / max(1.0, float(self.config.horizon))),
            float(delivery_progress), float(carrying_changed), repeats,
            min(1.0, agent.steps_since_charging / 12.0),
        ))
        if agent.carrying_task_id:
            delivery = self.state.task_by_id(
                agent.carrying_task_id
            ).delivery_position
            distance = shortest_path_distance(
                agent.position, delivery, self.config.map_layout_id
            )
            values.append(min(1.0, float(distance) / 12.0))
            for action in ("UP", "DOWN", "LEFT", "RIGHT"):
                delta = MOVE_DELTAS[action]
                target = (agent.position[0] + delta[0],
                          agent.position[1] + delta[1])
                neighbor_distance = (
                    shortest_path_distance(
                        target, delivery, self.config.map_layout_id
                    ) if self.layout.is_passable(target) else 12
                )
                values.append(min(1.0, float(neighbor_distance) / 12.0))
        else:
            values.extend((0.0,) * 5)
        return values

    def observations(self):
        base = super().observations()
        result = {}
        for role, agent_id in enumerate(self.agent_ids):
            other_id = self.agent_ids[1 - role]
            extra = np.asarray(
                [float(self.state.last_robot_collision_event)]
                + [float(self.state.by_id(agent_id).last_action == action)
                   for action in ACTIONS]
                + [float(self.state.by_id(other_id).last_action == action)
                   for action in ACTIONS]
                + self._cycle_values(agent_id) + self._cycle_values(other_id),
                dtype=np.float32,
            )
            result[agent_id] = np.concatenate((base[agent_id], extra)).astype(
                np.float32, copy=False
            )
        return result

    def step(self, actions, *, decision_metadata=None):
        before = deepcopy(self.state)
        _observations, rewards, terminated, truncated, info = super().step(
            actions, decision_metadata=decision_metadata
        )
        charger = self.layout.charger_position
        for agent in self.state.agents:
            previous = before.by_id(agent.agent_id)
            recent = (*tuple(previous.recent_positions), tuple(agent.position))[-6:]
            agent.recent_positions = tuple(recent)
            if previous.position == charger and agent.position != charger:
                agent.last_charger_departure_frame = int(self.state.frame)
                agent.deliveries_at_last_charger_departure = int(
                    agent.deliveries_completed
                )
                agent.team_deliveries_at_last_charger_departure = int(
                    self.state.total_deliveries
                )
                agent.carrying_task_at_last_charger_departure = (
                    agent.carrying_task_id
                )
        return self.observations(), rewards, terminated, truncated, info


__all__ = ["VERSION", "R45_OBSERVATION_FEATURE_NAMES", "R45EnergyCycleMixin"]
