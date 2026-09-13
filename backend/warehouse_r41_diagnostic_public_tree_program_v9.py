"""Traceable v9 public-tree composition with a NumPy-only runtime.

The program accepts only the frozen 197-value public observation.  It derives
the registered 653 public relation features deterministically, evaluates a
base explicit boosted tree, then applies three routed specialist trees in the
order recorded by the payload.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.warehouse_r41_diagnostic_boosted_tree import (
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)


VERSION = "warehouse-r41-diagnostic-public-tree-program.v9"
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")


def _aggregation() -> dict[str, Any]:
    return {
        "kind": "active_specialist_weighted_probability_average",
        "trace_order": list(GROUPS),
        "base_weight": 1.0,
        "normalizer": "base_weight_plus_active_specialist_weights",
        "order_independent": True,
    }


AGGREGATION = _aggregation()

_TOP_FIELDS = frozenset((
    "version", "relations", "action_names", "aggregation", "base",
    "specialists", "metadata",
))
_SPECIALIST_FIELDS = frozenset(("group", "route", "mix_weight", "program"))
_ROUTE_FIELDS = frozenset(("feature_name", "operator", "threshold"))
_OPERATORS = frozenset(("<", "<=", ">", ">=", "==", "!="))


def _finite_float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _builder_float(value: Any, label: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    return _finite_float(value, label)


def _json_value(value: Any, path: str = "metadata") -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError(path + " keys must be strings")
            normalized[key] = _json_value(child, path + "." + key)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(child, f"{path}[{index}]")
                for index, child in enumerate(value)]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(path + " must contain only finite JSON values")


def _as_program(value: Any, label: str) -> R41DiagnosticBoostedTreeProgram:
    if isinstance(value, R41DiagnosticBoostedTreeProgram):
        return R41DiagnosticBoostedTreeProgram.from_dict(value.to_dict())
    if isinstance(value, Mapping):
        try:
            return R41DiagnosticBoostedTreeProgram.from_dict(value)
        except ValueError as exc:
            raise ValueError(label + " boosted-tree program differs") from exc
    raise ValueError(label + " must be an explicit boosted-tree program")


def _default_routes() -> dict[str, dict[str, Any]]:
    return {
        group: {
            "feature_name": "derived.critical." + group,
            "operator": ">",
            "threshold": 0.5,
        }
        for group in GROUPS
    }


def _normalize_route(
    value: Any, *, feature_names: Sequence[str], label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROUTE_FIELDS:
        raise ValueError(label + " public route schema differs")
    feature_name = value.get("feature_name")
    operator = value.get("operator")
    if (type(feature_name) is not str or feature_name not in feature_names
            or "hidden" in feature_name.casefold()
            or "logit" in feature_name.casefold()):
        raise ValueError(label + " route must use one registered public feature")
    if type(operator) is not str or operator not in _OPERATORS:
        raise ValueError(label + " public route operator differs")
    return {
        "feature_name": feature_name,
        "operator": operator,
        "threshold": _finite_float(value.get("threshold"), label + " route threshold"),
    }


class R41DiagnosticPublicTreeProgramV9:
    """Weighted average of one base and three routed public boosted trees."""

    def __init__(self, payload: Mapping[str, Any]):
        if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
            raise ValueError("Public-tree v9 top-level schema differs")
        if payload.get("version") != VERSION:
            raise ValueError("Public-tree v9 version differs")

        relations_payload = payload.get("relations")
        if not isinstance(relations_payload, Mapping):
            raise ValueError("Public-tree v9 relations contract differs")
        base_names = relations_payload.get("base_feature_names")
        try:
            self.relations = R41DiagnosticPublicRelationsV9(base_names)
        except (TypeError, ValueError) as exc:
            raise ValueError("Public-tree v9 relations contract differs") from exc
        if (len(self.relations.base_feature_names) != 197
                or len(self.relations.feature_names) != 653
                or dict(relations_payload) != self.relations.contract()):
            raise ValueError("Public-tree v9 requires the exact 197-to-653 relation contract")
        self.base_feature_names = self.relations.base_feature_names
        self.feature_names = self.relations.feature_names
        self._feature_index = {
            name: index for index, name in enumerate(self.feature_names)
        }

        action_names = payload.get("action_names")
        if (not isinstance(action_names, list) or len(action_names) < 2
                or any(type(name) is not str or not name for name in action_names)
                or len(set(action_names)) != len(action_names)):
            raise ValueError("Public-tree v9 action registry differs")
        self.action_names = tuple(action_names)
        if payload.get("aggregation") != _aggregation():
            raise ValueError("Public-tree v9 aggregation contract differs")

        self.base_program = _as_program(payload.get("base"), "Base")
        self._validate_program(self.base_program, "Base")
        specialists = payload.get("specialists")
        if not isinstance(specialists, list) or len(specialists) != len(GROUPS):
            raise ValueError("Public-tree v9 specialist registry differs")

        normalized_specialists: list[dict[str, Any]] = []
        specialist_programs: dict[str, R41DiagnosticBoostedTreeProgram] = {}
        for index, item in enumerate(specialists):
            group = GROUPS[index]
            if (not isinstance(item, Mapping) or set(item) != _SPECIALIST_FIELDS
                    or item.get("group") != group):
                raise ValueError("Public-tree v9 specialist order differs")
            route = _normalize_route(
                item.get("route"), feature_names=self.feature_names, label=group)
            mix_weight = _finite_float(item.get("mix_weight"), group + " mix weight")
            if not 0.0 <= mix_weight <= 1.0:
                raise ValueError(group + " mix weight must be in [0, 1]")
            program = _as_program(item.get("program"), group)
            self._validate_program(program, group)
            if program.classes != self.base_program.classes:
                raise ValueError(group + " class registry differs from base")
            specialist_programs[group] = program
            normalized_specialists.append({
                "group": group,
                "route": route,
                "mix_weight": mix_weight,
                "program": program.to_dict(),
            })

        metadata = _json_value(payload.get("metadata"))
        if not isinstance(metadata, dict):
            raise ValueError("Public-tree v9 metadata must be an object")
        self.metadata = metadata
        self._specialists = tuple(normalized_specialists)
        self._specialist_programs = specialist_programs
        self._payload = {
            "version": VERSION,
            "relations": self.relations.contract(),
            "action_names": list(self.action_names),
            "aggregation": _aggregation(),
            "base": self.base_program.to_dict(),
            "specialists": deepcopy(normalized_specialists),
            "metadata": deepcopy(self.metadata),
        }

    def _validate_program(
        self, program: R41DiagnosticBoostedTreeProgram, label: str,
    ) -> None:
        if program.feature_names != self.feature_names:
            raise ValueError(label + " program must consume exactly 653 public features")
        if program.action_names != self.action_names:
            raise ValueError(label + " action registry differs")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "R41DiagnosticPublicTreeProgramV9":
        return cls(payload)

    @classmethod
    def from_json(cls, payload: str) -> "R41DiagnosticPublicTreeProgramV9":
        if type(payload) is not str:
            raise ValueError("Public-tree v9 JSON payload must be text")
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Public-tree v9 JSON payload is invalid") from exc
        return cls(decoded)

    @classmethod
    def from_programs(
        cls,
        base_feature_names: Sequence[str],
        base_program: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
        narrow_passage: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
        shared_pickup: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
        shared_charger: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
        *,
        mix_weights: Mapping[str, float],
        routes: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "R41DiagnosticPublicTreeProgramV9":
        """Assemble already-fitted/exported candidates into the v9 runtime."""

        relations = R41DiagnosticPublicRelationsV9(base_feature_names)
        if len(relations.base_feature_names) != 197 or len(relations.feature_names) != 653:
            raise ValueError("Public-tree v9 requires exactly 197 public base features")
        programs = {
            "base": _as_program(base_program, "Base"),
            "narrow_passage": _as_program(narrow_passage, "narrow_passage"),
            "shared_pickup": _as_program(shared_pickup, "shared_pickup"),
            "shared_charger": _as_program(shared_charger, "shared_charger"),
        }
        if not isinstance(mix_weights, Mapping) or set(mix_weights) != set(GROUPS):
            raise ValueError("Public-tree v9 requires one mix weight for every group")
        route_values = _default_routes() if routes is None else routes
        if not isinstance(route_values, Mapping) or set(route_values) != set(GROUPS):
            raise ValueError("Public-tree v9 requires one public route for every group")
        normalized_routes = {}
        for group in GROUPS:
            route = route_values[group]
            if not isinstance(route, Mapping):
                raise ValueError(group + " public route schema differs")
            route = dict(route)
            if isinstance(route.get("threshold"), np.generic):
                route["threshold"] = route["threshold"].item()
            normalized_routes[group] = _normalize_route(
                route, feature_names=relations.feature_names, label=group)
        return cls({
            "version": VERSION,
            "relations": relations.contract(),
            "action_names": list(programs["base"].action_names),
            "aggregation": _aggregation(),
            "base": programs["base"].to_dict(),
            "specialists": [{
                "group": group,
                "route": normalized_routes[group],
                "mix_weight": _builder_float(
                    mix_weights[group], group + " mix weight"),
                "program": programs[group].to_dict(),
            } for group in GROUPS],
            "metadata": {} if metadata is None else _json_value(metadata),
        })

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def to_json(self) -> str:
        return json.dumps(self._payload, allow_nan=False, separators=(",", ":"),
                          sort_keys=True)

    @property
    def routes(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(item["route"]) for item in self._specialists)

    @property
    def mix_weights(self) -> dict[str, float]:
        return {item["group"]: item["mix_weight"] for item in self._specialists}

    def _mapping_vector(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(observation, Mapping):
            raise ValueError("Public-tree v9 observation must be a mapping")
        if set(observation) != set(self.base_feature_names):
            raise ValueError("Public-tree v9 requires exactly 197 raw public features")
        try:
            vector = np.asarray([
                observation[name] for name in self.base_feature_names
            ], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Public-tree v9 public observation differs") from exc
        if vector.shape != (197,) or not np.isfinite(vector).all():
            raise ValueError("Public-tree v9 public observation must be finite")
        return vector

    @staticmethod
    def _compare(values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
        if operator == "<":
            return values < threshold
        if operator == "<=":
            return values <= threshold
        if operator == ">":
            return values > threshold
        if operator == ">=":
            return values >= threshold
        if operator == "==":
            return values == threshold
        return values != threshold

    def _route_mask(self, expanded: np.ndarray, route: Mapping[str, Any]) -> np.ndarray:
        observed = np.asarray(
            expanded[:, self._feature_index[route["feature_name"]]],
            dtype=np.float64,
        )
        return self._compare(observed, route["operator"], route["threshold"])

    def _predict_expanded(self, expanded: np.ndarray) -> np.ndarray:
        weighted_sum = self.base_program.predict_proba_batch(expanded)
        total_weight = np.ones(len(expanded), dtype=np.float64)
        for item in self._specialists:
            mask = self._route_mask(expanded, item["route"])
            if not np.any(mask):
                continue
            specialist = self._specialist_programs[item["group"]]
            local = specialist.predict_proba_batch(expanded[mask])
            weighted_sum[mask] += item["mix_weight"] * local
            total_weight[mask] += item["mix_weight"]
        probabilities = weighted_sum / total_weight[:, None]
        if (not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0,
                                   rtol=0.0, atol=2e-15)):
            raise ValueError("Public-tree v9 final probabilities are invalid")
        return probabilities

    def predict_proba_batch(self, observations: Any) -> np.ndarray:
        expanded = self.relations.transform_batch(observations)
        return self._predict_expanded(expanded)

    def predict_batch(self, observations: Any) -> np.ndarray:
        return np.argmax(self.predict_proba_batch(observations), axis=1).astype(np.int64)

    def predict_proba(self, observation: Mapping[str, Any]) -> dict[str, float]:
        vector = self._mapping_vector(observation)
        probabilities = self.predict_proba_batch(vector[None, :])[0]
        return {name: float(probabilities[index])
                for index, name in enumerate(self.action_names)}

    def predict(self, observation: Mapping[str, Any]) -> str:
        vector = self._mapping_vector(observation)
        index = int(self.predict_batch(vector[None, :])[0])
        return self.action_names[index]

    @staticmethod
    def _component_complexity(
        program: R41DiagnosticBoostedTreeProgram,
    ) -> dict[str, Any]:
        payload = program.to_dict()
        tree_count = len(payload["trees"])
        node_count = 0
        split_count = 0
        leaf_count = 0
        maximum_depth = 0
        used_features: set[str] = set()
        for tree in payload["trees"]:
            nodes = tree["nodes"]
            node_count += len(nodes)
            pending = [(0, 0)]
            while pending:
                node_index, depth = pending.pop()
                maximum_depth = max(maximum_depth, depth)
                node = nodes[node_index]
                if node["kind"] == "leaf":
                    leaf_count += 1
                    continue
                split_count += 1
                used_features.add(program.feature_names[node["feature_index"]])
                pending.append((node["left"], depth + 1))
                pending.append((node["right"], depth + 1))
        return {
            "iterations": program.n_iterations,
            "trees": tree_count,
            "nodes": node_count,
            "split_nodes": split_count,
            "leaf_nodes": leaf_count,
            "maximum_tree_depth": maximum_depth,
            "unique_split_features": len(used_features),
            "unique_split_feature_names": sorted(used_features),
        }

    def complexity(self) -> dict[str, Any]:
        """Recompute all capacity counts from the frozen explicit nodes."""

        components = {
            "base": self._component_complexity(self.base_program),
            **{
                group: self._component_complexity(
                    self._specialist_programs[group])
                for group in GROUPS
            },
        }
        all_features = {
            name
            for component in components.values()
            for name in component["unique_split_feature_names"]
        }
        return {
            "component_count": 1 + len(GROUPS),
            "specialist_count": len(GROUPS),
            "route_count": len(self._specialists),
            "total_iterations": sum(
                component["iterations"] for component in components.values()),
            "total_trees": sum(
                component["trees"] for component in components.values()),
            "total_nodes": sum(
                component["nodes"] for component in components.values()),
            "split_nodes": sum(
                component["split_nodes"] for component in components.values()),
            "leaf_nodes": sum(
                component["leaf_nodes"] for component in components.values()),
            "maximum_tree_depth": max(
                component["maximum_tree_depth"]
                for component in components.values()),
            "maximum_component_iterations": max(
                component["iterations"] for component in components.values()),
            "unique_split_features": len(all_features),
            "unique_split_feature_names": sorted(all_features),
            "components": components,
        }

    def _trace_expanded(self, expanded: np.ndarray) -> dict[str, Any]:
        base_trace = self.base_program.trace(expanded)
        base_distribution = np.asarray(base_trace["probabilities"], dtype=np.float64)
        weighted_sum = base_distribution.copy()
        total_weight = 1.0
        route_traces: list[dict[str, Any]] = []
        triggered: list[str] = []
        for item in self._specialists:
            route = item["route"]
            observed = float(expanded[self._feature_index[route["feature_name"]]])
            active = bool(self._compare(
                np.asarray([observed]), route["operator"], route["threshold"])[0])
            program_trace = None
            specialist_distribution = None
            weighted_distribution = None
            if active:
                triggered.append(item["group"])
                program_trace = self._specialist_programs[item["group"]].trace(expanded)
                specialist_distribution = np.asarray(
                    program_trace["probabilities"], dtype=np.float64)
                weighted_distribution = item["mix_weight"] * specialist_distribution
                weighted_sum += weighted_distribution
                total_weight += item["mix_weight"]
            route_traces.append({
                "group": item["group"],
                "route": deepcopy(route),
                "observed": observed,
                "triggered": active,
                "mix_weight": item["mix_weight"],
                "specialist_distribution": (
                    None if specialist_distribution is None
                    else [float(value) for value in specialist_distribution]
                ),
                "weighted_distribution": (
                    None if weighted_distribution is None
                    else [float(value) for value in weighted_distribution]
                ),
                "program_trace": program_trace,
            })
        final_probabilities = weighted_sum / total_weight
        if (not np.isfinite(final_probabilities).all()
                or not np.isclose(final_probabilities.sum(), 1.0,
                                  rtol=0.0, atol=2e-15)):
            raise ValueError("Public-tree v9 final probabilities are invalid")
        prediction_index = int(np.argmax(final_probabilities))
        return {
            "relations_version": self.relations.contract()["version"],
            "base_feature_count": len(self.base_feature_names),
            "expanded_feature_count": len(self.feature_names),
            "aggregation": _aggregation(),
            "base": {
                "distribution": list(base_trace["probabilities"]),
                "weight": 1.0,
                "weighted_distribution": list(base_trace["probabilities"]),
                "program_trace": base_trace,
            },
            "routes": route_traces,
            "triggered_routes": triggered,
            "weighted_probability_sum": [float(value) for value in weighted_sum],
            "total_weight": float(total_weight),
            "final_probabilities": [float(value) for value in final_probabilities],
            "probabilities": [float(value) for value in final_probabilities],
            "prediction_index": prediction_index,
            "prediction": self.action_names[prediction_index],
        }

    def trace(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        vector = self._mapping_vector(observation)
        expanded = self.relations.transform_batch(vector[None, :])[0]
        return self._trace_expanded(expanded)

    def trace_batch(self, observations: Any) -> tuple[dict[str, Any], ...]:
        expanded = self.relations.transform_batch(observations)
        return tuple(self._trace_expanded(row) for row in expanded)

    def predict_with_trace(
        self, observation: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        trace = self.trace(observation)
        return trace["prediction"], trace

    def predict_proba_with_trace(
        self, observation: Mapping[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        trace = self.trace(observation)
        probabilities = {
            name: trace["final_probabilities"][index]
            for index, name in enumerate(self.action_names)
        }
        return probabilities, trace


def assemble_public_tree_program_v9(
    base_feature_names: Sequence[str],
    base_program: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
    narrow_passage: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
    shared_pickup: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
    shared_charger: R41DiagnosticBoostedTreeProgram | Mapping[str, Any],
    *,
    mix_weights: Mapping[str, float],
    routes: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> R41DiagnosticPublicTreeProgramV9:
    return R41DiagnosticPublicTreeProgramV9.from_programs(
        base_feature_names, base_program, narrow_passage, shared_pickup,
        shared_charger, mix_weights=mix_weights, routes=routes, metadata=metadata,
    )


__all__ = [
    "VERSION", "GROUPS", "AGGREGATION", "R41DiagnosticPublicTreeProgramV9",
    "assemble_public_tree_program_v9",
]
