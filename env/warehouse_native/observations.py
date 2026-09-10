"""Compact full-public features. There is deliberately no policy action mask.

Task slots are sorted by numerical task ID, without ranking or assigning a task.
Every task and charger gets the same geometric features. Distances neither select
a goal nor encode a next action. Positions use the warehouse's (row, column).
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np

from env.warehouse.domain import WarehouseConfig, WarehouseState, collaborative_study_config
from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance

OBSERVATION_VERSION = "warehouse-native-public-v1"


def task_order(state):
    return sorted(state.tasks, key=lambda t: int(t.task_id.rsplit("_", 1)[-1]))


@lru_cache(maxsize=32)
def observation_names(config: WarehouseConfig | None = None) -> tuple[str, ...]:
    config = config or collaborative_study_config()
    names = ["role.robot_1", "role.robot_2", "time.remaining", "team.deliveries"]
    for label in ("self", "other"):
        names += [f"{label}.{f}" for f in ("row", "column", "battery", "active", "carrying", "deliveries")]
        names += [f"{label}.last_executed.{a}" for a in ACTIONS]
        names += [f"{label}.carrying_task.{i}" for i in range(config.active_task_count)]
        names += [f"{label}.neighbor.{a}.passable" for a in MOVE_DELTAS]
    names += ["other.relative_row", "other.relative_column", "other.path_distance"]
    names += ["charger.row", "charger.column", "charger.occupied_self", "charger.occupied_other"]
    destinations = ["charger"]
    for i in range(config.active_task_count):
        names += [f"task.{i}.{f}" for f in ("exists", "available", "carried_self", "carried_other", "pickup_row", "pickup_column", "delivery_row", "delivery_column", "age", "pickup_delivery_distance")]
        destinations += [f"task.{i}.pickup", f"task.{i}.delivery"]
    for target in destinations:
        for who in ("self", "other"):
            names += [f"{target}.{who}.{f}" for f in ("relative_row", "relative_column", "path_distance")]
            names += [f"{target}.{who}.neighbor_distance.{a}" for a in MOVE_DELTAS]
    names += [f"map.{r}.{c}.passable" for r in range(config.rows) for c in range(config.cols)]
    return tuple(names)


def observation_size(config: WarehouseConfig | None = None) -> int:
    return len(observation_names(config))


def public_observations(state: WarehouseState, config: WarehouseConfig) -> dict[str, np.ndarray]:
    layout = get_map_layout(config.map_layout_id)
    scale = float(max(config.rows * config.cols - 1, 1))
    row_scale, col_scale = max(1, config.rows - 1), max(1, config.cols - 1)
    tasks = task_order(state)
    def dist(a, b):
        distance = shortest_path_distance(a, b, config.map_layout_id)
        return min(1.0, distance / scale) if math.isfinite(distance) else 1.0
    result = {}
    for role, agent in enumerate(state.agents):
        other = state.agents[1-role]
        values = [float(role == 0), float(role == 1), max(0., 1-state.frame/config.horizon), min(1., state.total_deliveries/config.horizon)]
        for person in (agent, other):
            values += [person.position[0]/row_scale, person.position[1]/col_scale, person.battery/100., float(person.active), float(person.carrying_task_id is not None), min(1., person.deliveries_completed/config.horizon)]
            values += [float(person.last_executed_action == a) for a in ACTIONS]
            values += [float(i < len(tasks) and person.carrying_task_id == tasks[i].task_id) for i in range(config.active_task_count)]
            values += [float(layout.is_passable((person.position[0]+dr,person.position[1]+dc))) for dr,dc in MOVE_DELTAS.values()]
        values += [(other.position[0]-agent.position[0])/row_scale, (other.position[1]-agent.position[1])/col_scale, dist(agent.position,other.position)]
        charger = layout.charger_position
        values += [charger[0]/row_scale, charger[1]/col_scale, float(agent.position==charger), float(other.position==charger)]
        targets = [charger]
        for i in range(config.active_task_count):
            if i < len(tasks):
                task = tasks[i]
                values += [1., float(task.status=="available"), float(task.carrier_agent_id==agent.agent_id), float(task.carrier_agent_id==other.agent_id), task.pickup_position[0]/row_scale, task.pickup_position[1]/col_scale, task.delivery_position[0]/row_scale, task.delivery_position[1]/col_scale, min(1., (state.frame-task.created_frame)/config.horizon), dist(task.pickup_position,task.delivery_position)]
                targets += [task.pickup_position, task.delivery_position]
            else:
                values += [0.] * 10
                targets += [None, None]
        for target in targets:
            for person in (agent,other):
                if target is None:
                    values += [0.] * 7
                    continue
                values += [(target[0]-person.position[0])/row_scale, (target[1]-person.position[1])/col_scale, dist(person.position,target)]
                for dr,dc in MOVE_DELTAS.values():
                    neighbor = (person.position[0]+dr,person.position[1]+dc)
                    values.append(dist(neighbor,target) if layout.is_passable(neighbor) else 1.)
        values += [float(symbol==".") for line in layout.tiles for symbol in line]
        result[agent.agent_id] = np.asarray(values,dtype=np.float32)
    return result
