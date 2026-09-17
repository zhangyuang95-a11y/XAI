"""Training-only warehouse physics with the independent r4.7 observation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from env.warehouse.navigation import ACTIONS
from backend.warehouse_r41_diagnostic_online_runtime import DEFAULT_REWARD_CONFIG
from backend.warehouse_r44_runtime import R44WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names, public_observations

from .energy import energy_version, departure_budget, task_budgets


ENVIRONMENT_VERSION = "warehouse-nn-training-environment.r4.7"
R46_COMPAT_ENVIRONMENT_VERSION = "warehouse-nn-training-environment.r4.6-continuation"
R48_ENVIRONMENT_VERSION = "warehouse-nn-training-environment.r4.8"
R49_ENVIRONMENT_VERSION = "warehouse-nn-training-environment.r4.9"
SNAPSHOT_KEY = "warehouse_nn_history"
_TASK_FIELDS = (
    "identity", "eligible", "reachable", "carried_self", "carried_other",
    "ownership_changed", "became_carried_other",
    "direct_required", "from_charger_required",
    "direct_margin", "charge_needed", "single_charge_feasible",
    "mid_charge_feasible", "pre_mid_charge_required_battery",
    "post_mid_charge_required_battery", "pickup_steps", "delivery_steps",
    "return_steps", "charger_steps",
)
_R46_TASK_FIELDS = (
    "eligible", "reachable", "direct_required", "from_charger_required",
    "direct_margin", "charge_needed", "single_charge_feasible",
    "mid_charge_feasible", "pre_mid_charge_required_battery",
    "post_mid_charge_required_battery", "pickup_steps", "delivery_steps",
    "return_steps", "charger_steps",
)
HISTORY_FIELDS = (
    "frames_since_pickup", "frames_since_delivery", "frames_since_charge",
    "frames_since_departure", "frames_since_collision", "departure_battery",
    "recent_position_repeat", "frames_since_progress", "wait_streak",
    "charger_occupancy_streak", "safe_charger_departure",
    "minimum_departure_required", "sufficient_for_any_work",
)
_R46_HISTORY_FIELDS = (
    "frames_since_pickup", "frames_since_delivery", "frames_since_charge",
    "frames_since_departure", "frames_since_collision", "departure_battery",
    "recent_position_repeat",
)


def extra_feature_names(active_task_count: int = 2, *, r46_compatible: bool = False) -> tuple[str, ...]:
    values: list[str] = []
    task_fields = _R46_TASK_FIELDS if r46_compatible else _TASK_FIELDS
    history_fields = _R46_HISTORY_FIELDS if r46_compatible else HISTORY_FIELDS
    for role in ("self", "other"):
        for index in range(active_task_count):
            values.extend(f"warehouse_nn.{role}.task.{index}.{name}" for name in task_fields)
        values.extend(f"warehouse_nn.{role}.history.{name}" for name in history_fields)
        values.extend(f"warehouse_nn.{role}.history.last_submitted.{action}"
                      for action in ACTIONS)
        if not r46_compatible:
            values.extend(f"warehouse_nn.{role}.history.last_executed.{action}"
                          for action in ACTIONS)
    return tuple(values)


class WarehouseNNEnv(R44WarehouseEnv):
    """Isolated environment contract; never imported by the online service."""

    def __init__(self, environment_config: Mapping[str, Any] | None = None):
        overrides = dict(environment_config or {})
        self._observation_contract = str(overrides.pop("observation_contract", "r4.7"))
        if self._observation_contract not in {"r4.6", "r4.7", "r4.8", "r4.9"}:
            raise ValueError("Unknown warehouse NN observation contract")
        self._environment_overrides = deepcopy({
            **overrides, "observation_contract": self._observation_contract,
        })
        self._environment_version = (
            R49_ENVIRONMENT_VERSION if self._observation_contract == "r4.9"
            else R48_ENVIRONMENT_VERSION if self._observation_contract == "r4.8"
            else R46_COMPAT_ENVIRONMENT_VERSION if self._observation_contract == "r4.6"
            else ENVIRONMENT_VERSION
        )
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
                "last_executed_action": "WAIT",
                "positions": [],
                "wait_streak": 0,
                "charger_occupancy_streak": 0,
                "task_transitions": {},
            }
            for agent_id in self.agent_ids
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        # Use the compact public geometry as the base and append the new
        # versioned features.  r4.x history/rule internals do not become hidden
        # targets for this new Actor.
        names = tuple(observation_names(self.config)) + extra_feature_names(
            self.config.active_task_count,
            r46_compatible=self._observation_contract == "r4.6",
        )
        # The released r4.6 Actor has neither a contract marker nor the
        # r4.7+ task/history additions.  A continuation must reproduce its
        # exact 257-float input layout before it can safely load its weights.
        marker = () if self._observation_contract in {"r4.6", "r4.7"} else (
            f"warehouse_nn.contract.{self._observation_contract}",
        )
        return names + marker

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
        history = self._nn_history.get(agent_id, {})
        legacy = self._observation_contract == "r4.6"
        for index in range(self.config.active_task_count):
            if index >= len(budgets):
                result.extend([0.0] * len(_R46_TASK_FIELDS if legacy else _TASK_FIELDS))
                continue
            row = budgets[index]
            norm_battery = lambda value: 0.0 if value is None else max(-2.0, min(2.0, float(value) / 100.0))
            norm_steps = lambda value: 0.0 if value is None else min(1.0, float(value) / scale_steps)
            transition = history.get("task_transitions", {}).get(row["task_id"], {})
            # The numerical suffix is only a stable public identity.  It is not
            # a target assignment and is deliberately weakly normalised.
            suffix = str(row["task_id"]).rsplit("_", 1)[-1]
            try:
                task_identity = min(1.0, max(0.0, float(int(suffix)) / 1000.0))
            except ValueError:
                task_identity = 0.0
            values = (
                task_identity, float(row["eligible"]), float(row["reachable"]),
                float(row["carried_self"]), float(row["carried_other"]),
                float(transition.get("ownership_changed", False)),
                float(transition.get("became_carried_other", False)),
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
            )
            # r4.6 Actor inputs end at the original public energy fields.
            result.extend(values[5:] if legacy else values)
        positions = [tuple(item) for item in history.get("positions", [])][-6:]
        repeat = 0.0 if len(positions) < 2 else 1.0 - len(set(positions)) / len(positions)
        frame = int(self.state.frame)
        departure = departure_budget(self, agent_id)
        agent = self.state.by_id(agent_id)
        safe_departure = bool(
            agent.position == self.layout.charger_position
            and bool(self._safe_departures(self.state, agent_id))
        )
        last_progress = max(
            int(history.get("last_pickup") or 0),
            int(history.get("last_delivery") or 0),
        )
        history_values = (
            self._age(frame, history.get("last_pickup"), self.config.horizon),
            self._age(frame, history.get("last_delivery"), self.config.horizon),
            self._age(frame, history.get("last_charge"), self.config.horizon),
            self._age(frame, history.get("last_departure"), self.config.horizon),
            self._age(frame, history.get("last_collision"), self.config.horizon),
            0.0 if history.get("departure_battery") is None else float(history["departure_battery"]) / 100.0,
            float(repeat),
            min(1.0, max(0.0, float(frame - last_progress) / max(1, self.config.horizon))),
            min(1.0, float(history.get("wait_streak", 0)) / max(1, self.config.horizon)),
            min(1.0, float(history.get("charger_occupancy_streak", 0)) / max(1, self.config.horizon)),
            float(safe_departure),
            norm_battery(departure["minimum_departure_required_battery"]),
            float(departure["sufficient_for_any_work"]),
        )
        result.extend(history_values[:7] if legacy else history_values)
        result.extend(float(history.get("last_submitted_action") == action)
                      for action in ACTIONS)
        if not legacy:
            result.extend(float(history.get("last_executed_action") == action)
                          for action in ACTIONS)
        return result

    @staticmethod
    def _task_state_map(state) -> dict[str, tuple[str, str | None]]:
        return {
            str(task.task_id): (str(task.status), task.carrier_agent_id)
            for task in state.tasks
        }

    def record_public_state_transition(self, before_state) -> None:
        """Refresh only facts caused by a public task-state transition.

        Targeted curricula call this after creating a legal state.  Normal
        gameplay calls it after the authoritative environment transition.  It
        never selects or changes an action.
        """
        before = self._task_state_map(before_state)
        after = self._task_state_map(self.state)
        for agent_id, history in self._nn_history.items():
            transitions: dict[str, dict[str, bool]] = {}
            for task_id, new_value in after.items():
                old_value = before.get(task_id)
                transitions[task_id] = {
                    "ownership_changed": old_value is not None and old_value != new_value,
                    "became_carried_other": bool(
                        old_value is not None
                        and old_value[0] == "available"
                        and new_value[0] == "carried"
                        and new_value[1] not in (None, agent_id)
                    ),
                }
            history["task_transitions"] = transitions

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
            contract = (np.asarray([1.0], dtype=np.float32)
                        if self._observation_contract not in {"r4.6", "r4.7"}
                        else np.empty(0, dtype=np.float32))
            result[agent_id] = np.concatenate((base[agent_id], extra, contract)).astype(
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
        self.record_public_state_transition(self.state)
        info["warehouse_nn_environment_version"] = self._environment_version
        info["energy_version"] = energy_version(self)
        return self.observations(), info

    def step(self, actions, *, decision_metadata=None):
        before = deepcopy(self.state)
        before_history = deepcopy(self._nn_history)
        _observations, rewards, terminated, truncated, info = super().step(
            actions, decision_metadata=decision_metadata
        )
        frame = int(self.state.frame)
        self.record_public_state_transition(before)
        for agent_id in self.agent_ids:
            self._nn_history[agent_id]["last_submitted_action"] = str(
                actions.get(agent_id, "WAIT")
            )
            self._nn_history[agent_id]["last_executed_action"] = str(
                info.get("executed_actions", {}).get(agent_id, "WAIT")
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
            requested_wait = str(actions.get(agent.agent_id, "WAIT")) == "WAIT"
            if requested_wait:
                history["wait_streak"] = int(history.get("wait_streak", 0)) + 1
            else:
                history["wait_streak"] = 0
            charged_here = (
                old.position == charger and agent.position == charger and requested_wait
            )
            history["charger_occupancy_streak"] = (
                int(history.get("charger_occupancy_streak", 0)) + 1 if charged_here else 0
            )
        info["warehouse_nn_environment_version"] = self._environment_version
        info["energy_version"] = energy_version(self)
        # Training rewards need the prior public-history facts to determine
        # whether a WAIT followed a task transition.  This is not exposed to
        # the runtime and cannot influence action selection.
        info["warehouse_nn_before_history"] = before_history
        return self.observations(), rewards, terminated, truncated, info

    def snapshot(self):
        payload = super().snapshot()
        payload[SNAPSHOT_KEY] = {
            "version": self._environment_version,
            "history": deepcopy(self._nn_history),
        }
        return payload

    def restore(self, payload, **kwargs):
        if not isinstance(payload, Mapping):
            raise ValueError("Warehouse NN snapshot must be an object")
        private = deepcopy(payload.get(SNAPSHOT_KEY))
        if not isinstance(private, dict) or private.get("version") != self._environment_version:
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
    "ENVIRONMENT_VERSION", "R48_ENVIRONMENT_VERSION", "R49_ENVIRONMENT_VERSION",
    "SNAPSHOT_KEY", "HISTORY_FIELDS",
    "extra_feature_names", "WarehouseNNEnv",
]
