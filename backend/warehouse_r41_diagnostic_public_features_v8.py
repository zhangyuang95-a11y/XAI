"""Deterministic public relation features for the r4.1 diagnostic program.

The frozen Actor receives a 197-value public observation.  This transformer
keeps those values and adds only arithmetic or Boolean relations between named
public fields.  It has no access to Actor parameters, logits, hidden state,
actions, intervention metadata, physical hashes, or labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


VERSION = "warehouse-r41-diagnostic-public-relations.v8"
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
EPSILON = np.float32(1e-7)


class R41DiagnosticPublicRelationsV8:
    """Expand the exact public observation with stable geometric relations."""

    def __init__(self, base_feature_names: Sequence[str]):
        names = tuple(base_feature_names)
        if (not names or len(names) != len(set(names))
                or any(not isinstance(name, str) or not name for name in names)
                or any("logit" in name.lower() or "hidden" in name.lower()
                       for name in names)):
            raise ValueError("Public base feature registry differs")
        self.base_feature_names = names
        self._index = {name: index for index, name in enumerate(names)}
        required = self._required_names()
        missing = sorted(required - set(names))
        if missing:
            raise ValueError("Public base feature registry is missing: " + ", ".join(missing))
        probe = np.zeros((1, len(names)), dtype=np.float32)
        _, derived = self._transform(probe)
        self.derived_feature_names = tuple(derived)
        self.feature_names = (*self.base_feature_names, *self.derived_feature_names)
        if len(self.feature_names) != len(set(self.feature_names)):
            raise RuntimeError("Derived public feature names are not unique")

    @staticmethod
    def _required_names() -> set[str]:
        result = {
            "other.relative_row", "other.relative_column", "other.path_distance",
            "self.battery", "other.battery",
        }
        for action in ACTIONS:
            result.add(f"self.neighbor.{action}.passable")
            result.add(f"history.other.submitted.{action}")
        result.add("history.other.submitted.WAIT")
        for who in ("self", "other"):
            result.add(f"charger.{who}.path_distance")
            for action in ACTIONS:
                result.add(f"charger.{who}.neighbor_distance.{action}")
        for task in range(2):
            result.add(f"task.{task}.available")
            for who in ("self", "other"):
                result.add(f"task.{task}.carried_{who}")
                for endpoint in ("pickup", "delivery"):
                    prefix = f"task.{task}.{endpoint}.{who}"
                    result.add(prefix + ".path_distance")
                    for action in ACTIONS:
                        result.add(prefix + f".neighbor_distance.{action}")
        return result

    def _column(self, values: np.ndarray, name: str) -> np.ndarray:
        return values[:, self._index[name]]

    def _transform(self, values: np.ndarray) -> tuple[np.ndarray, list[str]]:
        columns: list[np.ndarray] = [values]
        names: list[str] = []

        def add(value: Any, name: str) -> None:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 1:
                array = array[:, None]
                local_names = [name]
            elif array.ndim == 2:
                local_names = [f"{name}.{index}" for index in range(array.shape[1])]
            else:
                raise ValueError("Derived public feature has invalid rank")
            if len(array) != len(values):
                raise ValueError("Derived public feature row count differs")
            columns.append(array)
            names.extend(local_names)

        degree = sum(
            self._column(values, f"self.neighbor.{action}.passable")
            for action in ACTIONS
        )
        add(degree, "derived.self.degree")
        narrow = ((degree <= np.float32(2.0))
                  & (self._column(values, "other.path_distance")
                     <= np.float32(3.0 / 41.0) + EPSILON))
        add(narrow, "derived.critical.narrow_passage")
        relative_row = self._column(values, "other.relative_row")
        relative_column = self._column(values, "other.relative_column")
        add(np.abs(relative_row), "derived.other.abs_relative_row")
        add(np.abs(relative_column), "derived.other.abs_relative_column")
        for action, predicate in (
            ("UP", relative_row < 0), ("DOWN", relative_row > 0),
            ("LEFT", relative_column < 0), ("RIGHT", relative_column > 0),
        ):
            add(predicate, f"derived.other.direction.{action}")

        targets = ["charger.self", "charger.other"] + [
            f"task.{task}.{endpoint}.{who}"
            for task in range(2)
            for endpoint in ("pickup", "delivery")
            for who in ("self", "other")
        ]
        for target in targets:
            current = self._column(values, target + ".path_distance")
            neighbor = np.stack([
                self._column(values, target + f".neighbor_distance.{action}")
                for action in ACTIONS
            ], axis=1)
            add(current[:, None] - neighbor, target + ".improvement")
            add(neighbor <= neighbor.min(axis=1, keepdims=True) + EPSILON,
                target + ".best")

        shared_pickup = np.zeros(len(values), dtype=np.bool_)
        for task in range(2):
            shared_pickup |= (
                (self._column(values, f"task.{task}.available") > .5)
                & (self._column(values, f"task.{task}.pickup.self.path_distance")
                   <= np.float32(4.0 / 41.0) + EPSILON)
                & (self._column(values, f"task.{task}.pickup.other.path_distance")
                   <= np.float32(4.0 / 41.0) + EPSILON)
            )
        self_charger = self._column(values, "charger.self.path_distance")
        other_charger = self._column(values, "charger.other.path_distance")
        shared_charger = (
            ((self_charger <= np.float32(3.0 / 41.0) + EPSILON)
             & (other_charger <= np.float32(3.0 / 41.0) + EPSILON))
            | ((np.minimum(self._column(values, "self.battery"),
                           self._column(values, "other.battery")) <= .3 + EPSILON)
               & (np.maximum(self_charger, other_charger)
                  <= np.float32(5.0 / 41.0) + EPSILON))
        )
        add(shared_pickup, "derived.critical.shared_pickup")
        add(shared_charger, "derived.critical.shared_charger")

        for who in ("self", "other"):
            current_goals: list[np.ndarray] = []
            neighbor_goals: list[np.ndarray] = []
            for task in range(2):
                carried = self._column(values, f"task.{task}.carried_{who}") > .5
                available = self._column(values, f"task.{task}.available") > .5
                pickup = self._column(
                    values, f"task.{task}.pickup.{who}.path_distance")
                delivery = self._column(
                    values, f"task.{task}.delivery.{who}.path_distance")
                current = np.where(carried, delivery,
                                   np.where(available, pickup, np.float32(2.0)))
                neighbors = np.stack([
                    np.where(
                        carried,
                        self._column(values,
                            f"task.{task}.delivery.{who}.neighbor_distance.{action}"),
                        np.where(
                            available,
                            self._column(values,
                                f"task.{task}.pickup.{who}.neighbor_distance.{action}"),
                            np.float32(2.0),
                        ),
                    )
                    for action in ACTIONS
                ], axis=1)
                current_goals.append(current)
                neighbor_goals.append(neighbors)
                add(current, f"derived.task.{task}.{who}.active_goal.distance")
                add(current[:, None] - neighbors,
                    f"derived.task.{task}.{who}.active_goal.improvement")
            current_matrix = np.stack(current_goals, axis=1)
            neighbor_tensor = np.stack(neighbor_goals, axis=1)
            nearest = current_matrix.min(axis=1)
            nearest_neighbor = neighbor_tensor.min(axis=1)
            add(nearest, f"derived.{who}.nearest_active_goal.distance")
            add(nearest[:, None] - nearest_neighbor,
                f"derived.{who}.nearest_active_goal.improvement")
            add(nearest_neighbor <= nearest_neighbor.min(axis=1, keepdims=True) + EPSILON,
                f"derived.{who}.nearest_active_goal.best")

        other_distance = self._column(values, "other.path_distance")
        geometry = np.stack([relative_row, relative_column, other_distance, degree], axis=1)
        for action in (*ACTIONS, "WAIT"):
            history = self._column(values, f"history.other.submitted.{action}")
            add(history[:, None] * geometry,
                f"derived.history.other.{action}.geometry")
        for action in ACTIONS:
            legal = self._column(values, f"self.neighbor.{action}.passable")
            add(legal * other_distance, f"derived.legal.{action}.other_distance")

        result = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Derived public features are non-finite")
        return result, names

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != len(self.base_feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Public observation batch differs")
        result, names = self._transform(values)
        if tuple(names) != self.derived_feature_names:
            raise RuntimeError("Derived public feature registry changed")
        return result

    def transform_mapping(self, observation: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(observation, Mapping):
            raise ValueError("Public observation must be a mapping")
        try:
            vector = np.asarray([
                float(observation[name]) for name in self.base_feature_names
            ], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every public base feature is required") from exc
        if vector.shape != (len(self.base_feature_names),) or not np.isfinite(vector).all():
            raise ValueError("Public observation vector differs")
        expanded = self.transform_batch(vector[None, :])[0]
        return {name: float(value) for name, value in zip(self.feature_names, expanded)}

    def critical_masks(self, observations: np.ndarray) -> dict[str, np.ndarray]:
        expanded = self.transform_batch(observations)
        indices = {name: index for index, name in enumerate(self.feature_names)}
        return {
            name: expanded[:, indices[f"derived.critical.{name}"]] > .5
            for name in ("narrow_passage", "shared_pickup", "shared_charger")
        }

    def contract(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "base_feature_names": list(self.base_feature_names),
            "derived_feature_names": list(self.derived_feature_names),
            "prediction_inputs": ["public_observation"],
            "actor_logits_input": False,
            "actor_hidden_state_input": False,
            "intervention_metadata_input": False,
            "label_input": False,
        }


__all__ = ["VERSION", "R41DiagnosticPublicRelationsV8"]
