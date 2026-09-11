"""Dependency-light inference for the diagnostic r4.1 model-tree program.

The routing tree and every leaf classifier consume only the public 197-value
observation.  The frozen neural Actor remains the controller; this program is
used only to audit and render explanations.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.program import ProgramTraceStep


VERSION = "warehouse-r41-diagnostic-model-tree-program.v2"
_TOP_FIELDS = frozenset((
    "version", "action_names", "feature_names", "router", "leaf_models",
    "metadata",
))
_ROUTER_FIELDS = frozenset(("depth", "leaf_count", "nodes"))
_INTERNAL_FIELDS = frozenset((
    "kind", "feature_index", "threshold", "left", "right",
))
_LEAF_NODE_FIELDS = frozenset(("kind", "model_index"))
_MODEL_FIELDS = frozenset((
    "router_node", "classes", "feature_indices", "coefficients", "intercepts",
))
_FORBIDDEN_METADATA_FRAGMENTS = (
    "actor_logits", "actor_logit", "hidden_state", "hidden_activation",
    "branch_action", "branch_metadata",
)


def _plain_float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _validate_metadata(value: Any, path: str = "metadata") -> None:
    """Reject accidental leakage of Actor-internal or intervention labels."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Model-tree metadata keys must be strings")
            lowered = key.casefold()
            if any(fragment in lowered for fragment in _FORBIDDEN_METADATA_FRAGMENTS):
                raise ValueError(path + " contains prohibited internal metadata")
            _validate_metadata(child, path + "." + key)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_metadata(child, f"{path}[{index}]")
    elif value is None or type(value) in (str, bool, int):
        return
    elif type(value) is float and math.isfinite(value):
        return
    else:
        raise ValueError(path + " is not finite canonical JSON metadata")


class R41DiagnosticModelTreeProgram:
    """Immutable public-input routing tree with sparse linear-softmax leaves."""

    def __init__(self, payload: Mapping[str, Any]):
        if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
            raise ValueError("Diagnostic model-tree top-level schema differs")
        if payload.get("version") != VERSION:
            raise ValueError("Diagnostic model-tree version differs")
        actions = payload.get("action_names")
        features = payload.get("feature_names")
        if (not isinstance(actions, list) or not actions
                or any(not isinstance(item, str) or not item for item in actions)
                or len(set(actions)) != len(actions)
                or not isinstance(features, list) or not features
                or any(not isinstance(item, str) or not item for item in features)
                or len(set(features)) != len(features)):
            raise ValueError("Diagnostic model-tree action/feature registry differs")
        self.action_names = tuple(actions)
        self.feature_names = tuple(features)
        self.metadata = deepcopy(payload.get("metadata"))
        if not isinstance(self.metadata, dict):
            raise ValueError("Diagnostic model-tree metadata must be an object")
        _validate_metadata(self.metadata)

        router = payload.get("router")
        if not isinstance(router, Mapping) or set(router) != _ROUTER_FIELDS:
            raise ValueError("Diagnostic model-tree router schema differs")
        depth = router.get("depth")
        leaf_count = router.get("leaf_count")
        nodes = router.get("nodes")
        if (type(depth) is not int or depth < 0 or depth > 12
                or type(leaf_count) is not int or leaf_count < 1 or leaf_count > 256
                or not isinstance(nodes, list) or len(nodes) != 2 * leaf_count - 1):
            raise ValueError("Diagnostic model-tree router bounds differ")

        normalized_nodes: list[dict[str, Any]] = []
        leaf_node_indices: list[int] = []
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping) or node.get("kind") not in ("split", "leaf"):
                raise ValueError("Diagnostic model-tree router node is malformed")
            if node["kind"] == "split":
                if set(node) != _INTERNAL_FIELDS:
                    raise ValueError("Diagnostic model-tree split-node schema differs")
                feature_index = node.get("feature_index")
                left = node.get("left")
                right = node.get("right")
                threshold = _plain_float(node.get("threshold"), "router threshold")
                if (type(feature_index) is not int
                        or not 0 <= feature_index < len(self.feature_names)
                        or type(left) is not int or type(right) is not int
                        or left == right or not 0 <= left < len(nodes)
                        or not 0 <= right < len(nodes)):
                    raise ValueError("Diagnostic model-tree split-node value differs")
                normalized_nodes.append({"kind": "split", "feature_index": feature_index,
                    "threshold": threshold, "left": left, "right": right})
            else:
                if set(node) != _LEAF_NODE_FIELDS or type(node.get("model_index")) is not int:
                    raise ValueError("Diagnostic model-tree leaf-node schema differs")
                normalized_nodes.append({"kind": "leaf", "model_index": node["model_index"]})
                leaf_node_indices.append(index)

        # The router must be one rooted, acyclic, full binary tree.  Requiring
        # all nodes to be reachable prevents unused payload data from becoming
        # an unverified covert input channel.
        visited: set[int] = set()
        active: set[int] = set()
        observed_depth = 0

        def walk(index: int, level: int) -> None:
            nonlocal observed_depth
            if index in active:
                raise ValueError("Diagnostic model-tree router has a cycle")
            if index in visited:
                raise ValueError("Diagnostic model-tree router has a shared child")
            active.add(index); visited.add(index)
            node = normalized_nodes[index]
            if node["kind"] == "leaf":
                observed_depth = max(observed_depth, level)
            else:
                walk(node["left"], level + 1)
                walk(node["right"], level + 1)
            active.remove(index)

        walk(0, 0)
        if (len(visited) != len(normalized_nodes) or len(leaf_node_indices) != leaf_count
                or observed_depth != depth):
            raise ValueError("Diagnostic model-tree router topology differs")

        models = payload.get("leaf_models")
        if not isinstance(models, list) or len(models) != leaf_count:
            raise ValueError("Diagnostic model-tree leaf-model registry differs")
        normalized_models: list[dict[str, Any]] = []
        model_nodes: set[int] = set()
        for model_index, model in enumerate(models):
            if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
                raise ValueError("Diagnostic model-tree leaf-model schema differs")
            router_node = model.get("router_node")
            classes = model.get("classes")
            indices = model.get("feature_indices")
            coefficients = model.get("coefficients")
            intercepts = model.get("intercepts")
            if (type(router_node) is not int or router_node not in leaf_node_indices
                    or router_node in model_nodes
                    or normalized_nodes[router_node]["model_index"] != model_index
                    or not isinstance(classes, list) or not classes
                    or classes != sorted(classes)
                    or len(set(classes)) != len(classes)
                    or any(type(item) is not int or not 0 <= item < len(self.action_names)
                           for item in classes)
                    or not isinstance(indices, list) or indices != sorted(indices)
                    or len(set(indices)) != len(indices)
                    or any(type(item) is not int or not 0 <= item < len(self.feature_names)
                           for item in indices)
                    or not isinstance(coefficients, list)
                    or len(coefficients) != len(classes)
                    or not isinstance(intercepts, list)
                    or len(intercepts) != len(classes)):
                raise ValueError("Diagnostic model-tree leaf-model values differ")
            normalized_coefficients = []
            for row in coefficients:
                if not isinstance(row, list) or len(row) != len(indices):
                    raise ValueError("Diagnostic model-tree coefficient shape differs")
                normalized_coefficients.append([
                    _plain_float(item, "leaf coefficient") for item in row
                ])
            normalized_intercepts = [
                _plain_float(item, "leaf intercept") for item in intercepts
            ]
            if len(classes) == 1 and (indices or coefficients != [[]]):
                raise ValueError("A constant diagnostic model-tree leaf carries coefficients")
            model_nodes.add(router_node)
            normalized_models.append({
                "router_node": router_node,
                "classes": list(classes),
                "feature_indices": list(indices),
                "coefficients": normalized_coefficients,
                "intercepts": normalized_intercepts,
            })
        if model_nodes != set(leaf_node_indices):
            raise ValueError("Diagnostic model-tree leaf coverage differs")

        self.router_depth = depth
        self.router_leaf_count = leaf_count
        self._nodes = tuple(normalized_nodes)
        self._models = tuple(normalized_models)
        self._payload = {
            "version": VERSION,
            "action_names": list(self.action_names),
            "feature_names": list(self.feature_names),
            "router": {"depth": depth, "leaf_count": leaf_count,
                       "nodes": deepcopy(normalized_nodes)},
            "leaf_models": deepcopy(normalized_models),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "R41DiagnosticModelTreeProgram":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        if not isinstance(features, Mapping):
            raise ValueError("Diagnostic model-tree features must be a mapping")
        try:
            result = np.asarray([float(features[name]) for name in self.feature_names],
                                dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Diagnostic model-tree requires every public feature") from exc
        if result.shape != (len(self.feature_names),) or not np.isfinite(result).all():
            raise ValueError("Diagnostic model-tree public feature vector differs")
        return result

    def _route(self, vector: np.ndarray) -> tuple[dict[str, Any], tuple[ProgramTraceStep, ...]]:
        index = 0
        trace: list[ProgramTraceStep] = []
        while self._nodes[index]["kind"] == "split":
            node = self._nodes[index]
            observed = float(vector[node["feature_index"]])
            result = observed <= node["threshold"]
            trace.append(ProgramTraceStep(
                self.feature_names[node["feature_index"]], "<=",
                node["threshold"], observed, result,
            ))
            index = node["left"] if result else node["right"]
        return self._models[self._nodes[index]["model_index"]], tuple(trace)

    def _scores(self, vector: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
        classes = model["classes"]
        if len(classes) == 1:
            return np.asarray([0.0], dtype=np.float64)
        selected = vector[np.asarray(model["feature_indices"], dtype=np.int64)]
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        intercepts = np.asarray(model["intercepts"], dtype=np.float64)
        # An explicit reduction is deterministic across BLAS providers and
        # avoids platform-specific matmul warning noise for these tiny sparse
        # leaf matrices.
        return np.sum(coefficients * selected[None, :], axis=1) + intercepts

    def predict_proba(self, features: Mapping[str, float]) -> dict[str, float]:
        vector = self._vector(features)
        model, _ = self._route(vector)
        scores = self._scores(vector, model)
        scores -= np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        result = {action: 0.0 for action in self.action_names}
        for class_index, probability in zip(model["classes"], probabilities):
            result[self.action_names[class_index]] = float(probability)
        return result

    def predict(self, features: Mapping[str, float]) -> str:
        vector = self._vector(features)
        model, _ = self._route(vector)
        scores = self._scores(vector, model)
        return self.action_names[model["classes"][int(np.argmax(scores))]]

    def trace(self, features: Mapping[str, float]) -> tuple[ProgramTraceStep, ...]:
        _, trace = self._route(self._vector(features))
        return trace

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Diagnostic model-tree observation batch differs")
        probabilities = self.predict_proba_batch(values)
        return np.argmax(probabilities, axis=1).astype(np.uint8)

    def predict_proba_batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Diagnostic model-tree observation batch differs")
        routed = np.empty(len(values), dtype=np.int16)
        pending: list[tuple[int, np.ndarray]] = [
            (0, np.arange(len(values), dtype=np.int64))]
        while pending:
            node_index, row_indices = pending.pop()
            node = self._nodes[node_index]
            if node["kind"] == "leaf":
                routed[row_indices] = node["model_index"]
                continue
            take_left = values[row_indices, node["feature_index"]] <= node["threshold"]
            if np.any(take_left):
                pending.append((node["left"], row_indices[take_left]))
            if np.any(~take_left):
                pending.append((node["right"], row_indices[~take_left]))
        result = np.zeros((len(values), len(self.action_names)), dtype=np.float64)
        for model_index, model in enumerate(self._models):
            row_indices = np.flatnonzero(routed == model_index)
            if not len(row_indices):
                continue
            classes = np.asarray(model["classes"], dtype=np.int64)
            if len(classes) == 1:
                result[row_indices, classes[0]] = 1.0
                continue
            selected = values[np.ix_(row_indices, np.asarray(
                model["feature_indices"], dtype=np.int64))]
            coefficients = np.asarray(model["coefficients"], dtype=np.float64)
            scores = np.empty((len(row_indices), len(classes)), dtype=np.float64)
            for class_index in range(len(classes)):
                scores[:, class_index] = np.sum(
                    selected * coefficients[class_index][None, :], axis=1
                ) + float(model["intercepts"][class_index])
            scores -= np.max(scores, axis=1, keepdims=True)
            local = np.exp(scores); local /= local.sum(axis=1, keepdims=True)
            result[np.ix_(row_indices, classes)] = local
        if (not np.isfinite(result).all()
                or not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=2e-12)):
            raise ValueError("Diagnostic model-tree batch probabilities are invalid")
        return result

    def complexity(self) -> dict[str, Any]:
        feature_references = sum(len(model["feature_indices"])
                                 for model in self._models)
        coefficient_count = sum(
            len(model["classes"]) * len(model["feature_indices"])
            for model in self._models
        )
        nonzero = sum(
            int(value != 0.0)
            for model in self._models
            for row in model["coefficients"] for value in row
        )
        used = {index for model in self._models for index in model["feature_indices"]}
        router_features = {
            node["feature_index"] for node in self._nodes if node["kind"] == "split"
        }
        return {
            "router_nodes": len(self._nodes),
            "router_depth": self.router_depth,
            "router_leaves": self.router_leaf_count,
            "router_predicates": len(router_features),
            "leaf_models": len(self._models),
            "constant_leaf_models": sum(len(model["classes"]) == 1
                                        for model in self._models),
            "leaf_class_count": sum(len(model["classes"]) for model in self._models),
            "leaf_feature_references": feature_references,
            "unique_leaf_features": len(used),
            "coefficient_capacity": coefficient_count,
            "nonzero_coefficients": nonzero,
            "maximum_features_per_leaf": max(
                (len(model["feature_indices"]) for model in self._models), default=0),
            "available_features": len(self.feature_names),
            "actions": len(self.action_names),
        }


__all__ = ["VERSION", "R41DiagnosticModelTreeProgram"]
