"""Training-only r4.4 physics with an independent v1 neural observation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from env.warehouse.navigation import ACTIONS
from backend.warehouse_r41_diagnostic_online_runtime import DEFAULT_REWARD_CONFIG
from backend.warehouse_r44_runtime import R44WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names, public_observations

from .energy import ENERGY_VERSION, task_budgets


ENVIRONMENT_VERSION = "warehouse-nn-training-environment.v1"
SNAPSHOT_KEY = "warehouse_nn_history"
_TASK_FIELDS = (
    "eligible", "reachable", "direct_required", "from_charger_required",
    "direct_margin", "charge_needed", "single_charge_feasible",
    "mid_charge_feasible", "pre_mid_charge_required_battery",
    "post_mid_charge_required_battery", "pickup_steps", "delivery_steps",
    "return_steps", "charger_steps",
)
HISTORY_FIELDS = (
    "frames_since_pickup", "frames_since_delivery", "frames_since_charge",
    "frames_since_departure", "frames_since_collision", "departure_battery",
    "recent_position_repeat",
)


def extra_feature_names(active_task_count: int = 2) -> tuple[str, ...]:
    values: list[str] = []
    for role in ("self", "other"):
        for index in range(active_task_count):
            values.extend(f"warehouse_nn.{role}.task.{index}.{name}" for name in _TASK_FIELDS)
        values.extend(f"warehouse_nn.{role}.history.{name}" for name in HISTORY_FIELDS)
        values.extend(f"warehouse_nn.{role}.history.last_submitted.{action}"
                      for action in ACTIONS)
    return tuple(values)


class WarehouseNNEnv(R44WarehouseEnv):
    """Isolated environment contract; never imported by the online service."""

    def __init__(self, environment_config: Mapping[str, Any] | None = None):
        overrides = dict(environment_config or {})
        self._environment_overrides = deepcopy(overrides)
        config = collaborative_study_config(**overrides)
        reward_config = deepcopy(DEFAULT_REWARD_CONFIG)
        reward_config["collision_training_cost"] = 0.05
        super().__init__(
            config,
            reward_config,
            collision_cost=float(reward_config["collision_training_cost"]),
            mode="observed",
        )
        self._nn_history = self._empty_history()

    def _empty_history(self) -> dict[str, dict[str, Any]]:
        return {
            agent_id: {
                "last_pickup": None,
                "last_delivery": None,
                "last_charge": None,
                "last_departure": None,
                "last_collision": None,
                "departure_battery": None,
                "last_submitted_action": "WAIT",
                "positions": [],
            }
            for agent_id in self.agent_ids
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        # Use the compact public geometry as the base and append the new
        # versioned features.  r4.x history/rule internals do not become hidden
        # targets for this new Actor.
        return tuple(observation_names(self.config)) + extra_feature_names(
            self.config.active_task_count
        )

    @property
    def observation_size(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def _age(frame: int, value: int | None, horizon: int) -> float:
        return 1.0 if value is None else min(1.0, max(0, frame - value) / max(1, horizon))

    def _role_values(self, agent_id: str) -> list[float]:
        scale_steps = float(max(1, self.config.rows * self.config.cols))
        result: list[float] = []
        budgets = task_budgets(self, agent_id)
        for index in range(self.config.active_task_count):
            if index >= len(budgets):
                result.extend([0.0] * len(_TASK_FIELDS))
                continue
            row = budgets[index]
            norm_battery = lambda value: 0.0 if value is None else max(-2.0, min(2.0, float(value) / 100.0))
            norm_steps = lambda value: 0.0 if value is None else min(1.0, float(value) / scale_steps)
            result.extend((
                float(row["eligible"]), float(row["reachable"]),
                norm_battery(row["direct_required_battery"]),
                norm_battery(row["from_charger_required_battery"]),
                norm_battery(row["direct_margin"]),
                norm_battery(row["charge_needed"]),
                float(row["single_charge_feasible"]),
                float(row["mid_charge_feasible"]),
                norm_battery(row["pre_mid_charge_required_battery"]),
                norm_battery(row["post_mid_charge_required_battery"]),
                norm_steps(row["pickup_steps"]), norm_steps(row["delivery_steps"]),
                norm_steps(row["return_steps"]), norm_steps(row["charger_steps"]),
            ))
        history = self._nn_history.get(agent_id, {})
        positions = [tuple(item) for item in history.get("positions", [])][-6:]
        repeat = 0.0 if len(positions) < 2 else 1.0 - len(set(positions)) / len(positions)
        frame = int(self.state.frame)
        result.extend((
            self._age(frame, history.get("last_pickup"), self.config.horizon),
            self._age(frame, history.get("last_delivery"), self.config.horizon),
            self._age(frame, history.get("last_charge"), self.config.horizon),
            self._age(frame, history.get("last_departure"), self.config.horizon),
            self._age(frame, history.get("last_collision"), self.config.horizon),
            0.0 if history.get("departure_battery") is None else float(history["departure_battery"]) / 100.0,
            float(repeat),
        ))
        result.extend(float(history.get("last_submitted_action") == action)
                      for action in ACTIONS)
        return result

    def observations(self) -> dict[str, np.ndarray]:
        self._require_state()
        base = public_observations(self.state, self.config)
        result = {}
        for role, agent_id in enumerate(self.agent_ids):
            other_id = self.agent_ids[1 - role]
            extra = np.asarray(
                self._role_values(agent_id) + self._role_values(other_id),
                dtype=np.float32,
            )
            result[agent_id] = np.concatenate((base[agent_id], extra)).astype(
                np.float32, copy=False
            )
        return result

    def global_state(self) -> np.ndarray:
        observations = self.observations()
        return np.concatenate([observations[key] for key in self.agent_ids]).astype(np.float32)

    def reset(self, *, seed=None):
        self._nn_history = self._empty_history()
        _observations, info = super().reset(seed=seed)
        for agent in self.state.agents:
            self._nn_history[agent.agent_id]["positions"] = [tuple(agent.position)]
        info["warehouse_nn_environment_version"] = ENVIRONMENT_VERSION
        info["energy_version"] = ENERGY_VERSION
        return self.observations(), info

    def step(self, actions, *, decision_metadata=None):
        before = deepcopy(self.state)
        _observations, rewards, terminated, truncated, info = super().step(
            actions, decision_metadata=decision_metadata
        )
        frame = int(self.state.frame)
        for agent_id in self.agent_ids:
            self._nn_history[agent_id]["last_submitted_action"] = str(
                actions.get(agent_id, "WAIT")
            )
            if info.get("robot_collision"):
                self._nn_history[agent_id]["last_collision"] = frame
        for event in info.get("events", ()):  # Events are authoritative facts.
            agent_id = event.get("agent_id")
            if agent_id not in self._nn_history:
                continue
            if event.get("event") == "pickup":
                self._nn_history[agent_id]["last_pickup"] = frame
            elif event.get("event") == "delivery":
                self._nn_history[agent_id]["last_delivery"] = frame
            elif event.get("event") == "charge":
                self._nn_history[agent_id]["last_charge"] = frame
        charger = self.layout.charger_position
        for agent in self.state.agents:
            old = before.by_id(agent.agent_id)
            history = self._nn_history[agent.agent_id]
            if old.position == charger and agent.position != charger:
                history["last_departure"] = frame
                history["departure_battery"] = float(old.battery)
            history["positions"] = (
                [tuple(item) for item in history.get("positions", [])]
                + [tuple(agent.position)]
            )[-6:]
        info["warehouse_nn_environment_version"] = ENVIRONMENT_VERSION
        info["energy_version"] = ENERGY_VERSION
        return self.observations(), rewards, terminated, truncated, info

    def snapshot(self):
        payload = super().snapshot()
        payload[SNAPSHOT_KEY] = {
            "version": ENVIRONMENT_VERSION,
            "history": deepcopy(self._nn_history),
        }
        return payload

    def restore(self, payload, **kwargs):
        if not isinstance(payload, Mapping):
            raise ValueError("Warehouse NN snapshot must be an object")
        private = deepcopy(payload.get(SNAPSHOT_KEY))
        if not isinstance(private, dict) or private.get("version") != ENVIRONMENT_VERSION:
            raise ValueError("Warehouse NN observation history is missing or incompatible")
        base = deepcopy(dict(payload))
        del base[SNAPSHOT_KEY]
        super().restore(base, **kwargs)
        history = private.get("history")
        if not isinstance(history, dict) or set(history) != set(self.agent_ids):
            raise ValueError("Warehouse NN history identities differ")
        self._nn_history = deepcopy(history)
        # Force all feature calculations once so incompatible saved content
        # fails during restore rather than later in a rollout.
        self.observations()

    def branch(self):
        clone = type(self)(self._environment_overrides)
        clone.restore(self.snapshot(), require_feedback=True)
        return clone


__all__ = [
    "ENVIRONMENT_VERSION", "SNAPSHOT_KEY", "HISTORY_FIELDS",
    "extra_feature_names", "WarehouseNNEnv",
]
