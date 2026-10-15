"""NumPy runtime for an explicit ``HistGradientBoostingClassifier`` program.

The exporter reads a fitted sklearn estimator while sklearn is available
offline.  The resulting JSON payload contains only public feature predicates,
the raw-score baseline, and the already-shrunk value of every leaf.  Loading
and inference require only NumPy and the Python standard library.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "warehouse-r41-diagnostic-boosted-tree.v1"
MODEL_KIND = "sklearn_hist_gradient_boosting_classifier"
LEAF_VALUE_SEMANTICS = "tree_predictor_post_learning_rate_additive_raw_score"

_TOP_FIELDS = frozenset((
    "version", "kind", "feature_names", "classes", "action_names",
    "output_kind", "baseline", "leaf_value_semantics", "n_iterations",
    "trees", "metadata",
))
_TREE_FIELDS = frozenset(("iteration", "output_index", "nodes"))
_SPLIT_FIELDS = frozenset((
    "kind", "feature_index", "threshold", "missing_go_to_left", "left",
    "right",
))
_LEAF_FIELDS = frozenset(("kind", "value"))
_FORBIDDEN_INPUT_FRAGMENTS = ("hidden", "logit")


def _plain_float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _plain_label(value: Any, label: str) -> str | bool | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(label + " must be a finite JSON scalar")


def _json_value(value: Any, path: str = "metadata") -> Any:
    """Copy metadata into a finite, deterministic JSON value."""

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


def _normalize_feature_names(value: Any, expected: int | None = None) -> list[str]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(name) is not str or not name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Boosted-tree feature registry differs")
    names = list(value)
    if expected is not None and len(names) != expected:
        raise ValueError("Boosted-tree feature count differs from estimator")
    for name in names:
        lowered = name.casefold()
        if any(fragment in lowered for fragment in _FORBIDDEN_INPUT_FRAGMENTS):
            raise ValueError("Boosted-tree inputs may not contain hidden/logit features")
    return names


def _normalize_action_names(value: Any, expected: int) -> list[str]:
    if (not isinstance(value, (list, tuple)) or len(value) != expected
            or any(type(name) is not str or not name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Boosted-tree action registry differs")
    return list(value)


def _validate_tree_topology(nodes: Sequence[Mapping[str, Any]]) -> None:
    # Explicit enter/exit frames avoid Python's recursion limit for legitimate
    # deep trees while still distinguishing cycles from shared children.
    state = np.zeros(len(nodes), dtype=np.uint8)  # 0 unseen, 1 active, 2 done
    stack: list[tuple[int, bool]] = [(0, False)]
    while stack:
        index, exiting = stack.pop()
        if exiting:
            state[index] = 2
            continue
        if state[index] == 1:
            raise ValueError("Boosted-tree nodes contain a cycle")
        if state[index] == 2:
            raise ValueError("Boosted-tree nodes contain a shared child")
        state[index] = 1
        stack.append((index, True))
        node = nodes[index]
        if node["kind"] == "split":
            stack.append((node["right"], False))
            stack.append((node["left"], False))
    if int(np.count_nonzero(state)) != len(nodes):
        raise ValueError("Boosted-tree nodes contain unreachable data")


class R41DiagnosticBoostedTreeProgram:
    """Immutable explicit-tree classifier with traceable NumPy inference."""

    def __init__(self, payload: Mapping[str, Any]):
        if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
            raise ValueError("Boosted-tree top-level schema differs")
        if payload.get("version") != VERSION or payload.get("kind") != MODEL_KIND:
            raise ValueError("Boosted-tree version or model kind differs")

        self.feature_names = tuple(_normalize_feature_names(
            payload.get("feature_names")))
        raw_classes = payload.get("classes")
        if not isinstance(raw_classes, list) or len(raw_classes) < 2:
            raise ValueError("Boosted-tree class registry differs")
        classes = [_plain_label(value, "class label") for value in raw_classes]
        typed_classes = {(type(value), value) for value in classes}
        if (len(typed_classes) != len(classes)
                or len({type(value) for value in classes}) != 1):
            raise ValueError(
                "Boosted-tree class labels must be unique and use one scalar type")
        self.classes = tuple(classes)
        self.action_names = tuple(_normalize_action_names(
            payload.get("action_names"), len(classes)))

        expected_kind = "binary_logit" if len(classes) == 2 else "multiclass_logits"
        if payload.get("output_kind") != expected_kind:
            raise ValueError("Boosted-tree output kind differs")
        self.output_kind = expected_kind
        self.output_count = 1 if len(classes) == 2 else len(classes)
        baseline = payload.get("baseline")
        if not isinstance(baseline, list) or len(baseline) != self.output_count:
            raise ValueError("Boosted-tree baseline shape differs")
        normalized_baseline = [
            _plain_float(value, "baseline") for value in baseline
        ]
        self.baseline = tuple(normalized_baseline)
        if payload.get("leaf_value_semantics") != LEAF_VALUE_SEMANTICS:
            raise ValueError("Boosted-tree leaf-value semantics differ")

        n_iterations = payload.get("n_iterations")
        trees = payload.get("trees")
        if (type(n_iterations) is not int or n_iterations < 0
                or not isinstance(trees, list)
                or len(trees) != n_iterations * self.output_count):
            raise ValueError("Boosted-tree sequence dimensions differ")

        normalized_trees: list[dict[str, Any]] = []
        for tree_index, tree in enumerate(trees):
            expected_iteration = tree_index // self.output_count
            expected_output = tree_index % self.output_count
            if (not isinstance(tree, Mapping) or set(tree) != _TREE_FIELDS
                    or tree.get("iteration") != expected_iteration
                    or tree.get("output_index") != expected_output):
                raise ValueError("Boosted-tree sequence order differs")
            raw_nodes = tree.get("nodes")
            if not isinstance(raw_nodes, list) or not raw_nodes:
                raise ValueError("Boosted-tree node registry differs")
            normalized_nodes: list[dict[str, Any]] = []
            for node in raw_nodes:
                if not isinstance(node, Mapping):
                    raise ValueError("Boosted-tree node is malformed")
                if node.get("kind") == "leaf":
                    if set(node) != _LEAF_FIELDS:
                        raise ValueError("Boosted-tree leaf schema differs")
                    normalized_nodes.append({
                        "kind": "leaf",
                        "value": _plain_float(node.get("value"), "leaf value"),
                    })
                    continue
                if node.get("kind") != "split" or set(node) != _SPLIT_FIELDS:
                    raise ValueError("Boosted-tree split schema differs")
                feature_index = node.get("feature_index")
                left = node.get("left")
                right = node.get("right")
                missing_go_to_left = node.get("missing_go_to_left")
                if (type(feature_index) is not int
                        or not 0 <= feature_index < len(self.feature_names)
                        or type(left) is not int or type(right) is not int
                        or left == right
                        or not 0 <= left < len(raw_nodes)
                        or not 0 <= right < len(raw_nodes)
                        or type(missing_go_to_left) is not bool):
                    raise ValueError("Boosted-tree split values differ")
                normalized_nodes.append({
                    "kind": "split",
                    "feature_index": feature_index,
                    "threshold": _plain_float(node.get("threshold"),
                                               "split threshold"),
                    "missing_go_to_left": missing_go_to_left,
                    "left": left,
                    "right": right,
                })
            _validate_tree_topology(normalized_nodes)
            normalized_trees.append({
                "iteration": expected_iteration,
                "output_index": expected_output,
                "nodes": normalized_nodes,
            })

        self.n_iterations = n_iterations
        self.metadata = _json_value(payload.get("metadata"))
        if not isinstance(self.metadata, dict):
            raise ValueError("Boosted-tree metadata must be an object")
        self._trees = tuple(normalized_trees)
        self._payload = {
            "version": VERSION,
            "kind": MODEL_KIND,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "action_names": list(self.action_names),
            "output_kind": self.output_kind,
            "baseline": list(self.baseline),
            "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
            "n_iterations": self.n_iterations,
            "trees": deepcopy(normalized_trees),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "R41DiagnosticBoostedTreeProgram":
        return cls(payload)

    @classmethod
    def from_json(cls, payload: str) -> "R41DiagnosticBoostedTreeProgram":
        if type(payload) is not str:
            raise ValueError("Boosted-tree JSON payload must be text")
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Boosted-tree JSON payload is invalid") from exc
        return cls(decoded)

    @classmethod
    def from_sklearn(
        cls,
        estimator: Any,
        *,
        feature_names: Sequence[str] | None = None,
        action_names: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "R41DiagnosticBoostedTreeProgram":
        """Export a fitted numeric ``HistGradientBoostingClassifier``.

        ``TreePredictor.nodes['value']`` is copied without another learning-rate
        multiplication: sklearn applies shrinkage before creating each
        ``TreePredictor``.
        """

        required = ("classes_", "n_features_in_", "_baseline_prediction",
                    "_predictors")
        if estimator is None or any(not hasattr(estimator, name) for name in required):
            raise ValueError("A fitted HistGradientBoostingClassifier is required")
        if estimator.__class__.__name__ != "HistGradientBoostingClassifier":
            raise ValueError("A fitted HistGradientBoostingClassifier is required")
        n_features = int(estimator.n_features_in_)
        if feature_names is None:
            learned_names = getattr(estimator, "feature_names_in_", None)
            if learned_names is None:
                feature_names = [f"feature_{index}" for index in range(n_features)]
            else:
                feature_names = [str(name) for name in learned_names]
        normalized_features = _normalize_feature_names(feature_names, n_features)

        classes = [_plain_label(value, "class label")
                   for value in np.asarray(estimator.classes_).tolist()]
        if len(classes) < 2:
            raise ValueError("Boosted-tree classifier requires at least two classes")
        if action_names is None:
            action_names = [str(value) for value in classes]
        normalized_actions = _normalize_action_names(action_names, len(classes))
        output_count = 1 if len(classes) == 2 else len(classes)

        declared_outputs = getattr(estimator, "n_trees_per_iteration_", None)
        if declared_outputs is None:
            declared_outputs = getattr(estimator, "_n_trees_per_iteration", None)
        if type(declared_outputs) not in (int, np.int32, np.int64):
            raise ValueError("Boosted-tree estimator output count is unavailable")
        if int(declared_outputs) != output_count:
            raise ValueError("Boosted-tree estimator output count differs")
        categorical = getattr(estimator, "_is_categorical_remapped", None)
        if categorical is not None and bool(np.any(np.asarray(categorical))):
            raise ValueError("Categorical splits are not public axis thresholds")

        baseline_array = np.asarray(estimator._baseline_prediction,
                                    dtype=np.float64).reshape(-1)
        if baseline_array.shape != (output_count,) or not np.isfinite(baseline_array).all():
            raise ValueError("Boosted-tree estimator baseline differs")
        predictors = estimator._predictors
        if not isinstance(predictors, (list, tuple)):
            raise ValueError("Boosted-tree estimator sequence is unavailable")

        serialized_trees: list[dict[str, Any]] = []
        required_node_fields = {
            "value", "feature_idx", "num_threshold", "missing_go_to_left",
            "left", "right", "is_leaf",
        }
        for iteration, predictors_at_iteration in enumerate(predictors):
            if len(predictors_at_iteration) != output_count:
                raise ValueError("Boosted-tree estimator iteration width differs")
            for output_index, predictor in enumerate(predictors_at_iteration):
                raw_nodes = getattr(predictor, "nodes", None)
                names = set(getattr(getattr(raw_nodes, "dtype", None),
                                    "names", ()) or ())
                if raw_nodes is None or not required_node_fields <= names or len(raw_nodes) == 0:
                    raise ValueError("Unsupported sklearn TreePredictor layout")
                if "is_categorical" in names and bool(np.any(raw_nodes["is_categorical"])):
                    raise ValueError("Categorical splits are not public axis thresholds")
                nodes: list[dict[str, Any]] = []
                for raw_node in raw_nodes:
                    if bool(raw_node["is_leaf"]):
                        nodes.append({
                            "kind": "leaf",
                            "value": float(raw_node["value"]),
                        })
                    else:
                        nodes.append({
                            "kind": "split",
                            "feature_index": int(raw_node["feature_idx"]),
                            "threshold": float(raw_node["num_threshold"]),
                            "missing_go_to_left": bool(raw_node["missing_go_to_left"]),
                            "left": int(raw_node["left"]),
                            "right": int(raw_node["right"]),
                        })
                serialized_trees.append({
                    "iteration": iteration,
                    "output_index": output_index,
                    "nodes": nodes,
                })

        return cls({
            "version": VERSION,
            "kind": MODEL_KIND,
            "feature_names": normalized_features,
            "classes": classes,
            "action_names": normalized_actions,
            "output_kind": "binary_logit" if len(classes) == 2
                           else "multiclass_logits",
            "baseline": [float(value) for value in baseline_array],
            "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
            "n_iterations": len(predictors),
            "trees": serialized_trees,
            "metadata": {} if metadata is None else _json_value(metadata),
        })

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def to_json(self) -> str:
        return json.dumps(self._payload, allow_nan=False, separators=(",", ":"),
                          sort_keys=True)

    @property
    def trees(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(tree) for tree in self._trees)

    def _vector(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        if isinstance(observation, Mapping):
            if set(observation) != set(self.feature_names):
                raise ValueError("Boosted-tree prediction requires exactly its public features")
            try:
                vector = np.asarray([observation[name] for name in self.feature_names],
                                    dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Boosted-tree public feature vector differs") from exc
        else:
            try:
                vector = np.asarray(observation, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Boosted-tree public feature vector differs") from exc
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("Boosted-tree public feature vector must be finite")
        return vector

    def _batch(self, observations: Any) -> np.ndarray:
        try:
            values = np.asarray(observations, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Boosted-tree public feature batch differs") from exc
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Boosted-tree public feature batch must be finite")
        return values

    @staticmethod
    def _tree_values_batch(tree: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=np.float64)
        pending: list[tuple[int, np.ndarray]] = [
            (0, np.arange(len(values), dtype=np.int64))
        ]
        nodes = tree["nodes"]
        while pending:
            node_index, row_indices = pending.pop()
            if not len(row_indices):
                continue
            node = nodes[node_index]
            if node["kind"] == "leaf":
                result[row_indices] = node["value"]
                continue
            take_left = (values[row_indices, node["feature_index"]]
                         <= node["threshold"])
            if np.any(take_left):
                pending.append((node["left"], row_indices[take_left]))
            if np.any(~take_left):
                pending.append((node["right"], row_indices[~take_left]))
        return result

    def raw_scores_batch(self, observations: Any) -> np.ndarray:
        """Return sklearn's native raw shape (one column for binary)."""

        values = self._batch(observations)
        scores = np.empty((len(values), self.output_count), dtype=np.float64)
        scores[:] = np.asarray(self.baseline, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for tree in self._trees:
                scores[:, tree["output_index"]] += self._tree_values_batch(tree, values)
        if not np.isfinite(scores).all():
            raise ValueError("Boosted-tree raw-score accumulation is not finite")
        return scores

    def raw_scores(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        return self.raw_scores_batch(self._vector(observation)[None, :])[0]

    def decision_function_batch(self, observations: Any) -> np.ndarray:
        result = self.raw_scores_batch(observations)
        return result[:, 0] if self.output_count == 1 else result

    def decision_function(self, observation: Mapping[str, Any] | Sequence[float]) -> Any:
        result = self.raw_scores(observation)
        return float(result[0]) if self.output_count == 1 else result

    def _expand_logits(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.output_count != 1:
            return raw_scores.copy()
        return np.column_stack((np.zeros(len(raw_scores), dtype=np.float64),
                                raw_scores[:, 0]))

    def raw_logits_batch(self, observations: Any) -> np.ndarray:
        """Return one logit per class; binary logits are ``[0, raw_margin]``."""

        return self._expand_logits(self.raw_scores_batch(observations))

    def raw_logits(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        return self.raw_logits_batch(self._vector(observation)[None, :])[0]

    def softmax_batch(self, observations: Any) -> np.ndarray:
        logits = self.raw_logits_batch(observations)
        if not len(logits):
            return np.empty((0, len(self.classes)), dtype=np.float64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def softmax(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        return self.softmax_batch(self._vector(observation)[None, :])[0]

    def predict_proba_batch(self, observations: Any) -> np.ndarray:
        return self.softmax_batch(observations)

    def predict_proba(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        return self.softmax(observation)

    def predict_indices_batch(self, observations: Any) -> np.ndarray:
        scores = self.raw_scores_batch(observations)
        if self.output_count == 1:
            return (scores[:, 0] > 0.0).astype(np.int64)
        return np.argmax(scores, axis=1).astype(np.int64)

    def predict_batch(self, observations: Any) -> np.ndarray:
        indices = self.predict_indices_batch(observations)
        return np.asarray(self.classes)[indices]

    def predict(self, observation: Mapping[str, Any] | Sequence[float]) -> Any:
        index = int(self.predict_indices_batch(self._vector(observation)[None, :])[0])
        return self.classes[index]

    def predict_action(self, observation: Mapping[str, Any] | Sequence[float]) -> str:
        index = int(self.predict_indices_batch(self._vector(observation)[None, :])[0])
        return self.action_names[index]

    def _trace_tree(self, tree_index: int, vector: np.ndarray) -> dict[str, Any]:
        tree = self._trees[tree_index]
        node_index = 0
        path: list[dict[str, Any]] = []
        while tree["nodes"][node_index]["kind"] == "split":
            node = tree["nodes"][node_index]
            observed = float(vector[node["feature_index"]])
            went_left = observed <= node["threshold"]
            next_node = node["left"] if went_left else node["right"]
            path.append({
                "node_index": node_index,
                "feature_index": node["feature_index"],
                "feature_name": self.feature_names[node["feature_index"]],
                "observed": observed,
                "operator": "<=",
                "threshold": node["threshold"],
                "went_left": bool(went_left),
                "next_node": next_node,
            })
            node_index = next_node
        contribution = float(tree["nodes"][node_index]["value"])
        contribution_logits = [0.0] * len(self.classes)
        class_index = 1 if self.output_count == 1 else tree["output_index"]
        contribution_logits[class_index] = contribution
        return {
            "tree_index": tree_index,
            "iteration": tree["iteration"],
            "output_index": tree["output_index"],
            "class_index": class_index,
            "path": path,
            "leaf_node": node_index,
            "contribution": contribution,
            "contribution_logits": contribution_logits,
        }

    def trace(self, observation: Mapping[str, Any] | Sequence[float]) -> dict[str, Any]:
        """Return every path and additive contribution for one prediction."""

        vector = self._vector(observation)
        tree_traces = [self._trace_tree(index, vector)
                       for index in range(len(self._trees))]
        raw_scores = np.asarray(self.baseline, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for tree_trace in tree_traces:
                raw_scores[tree_trace["output_index"]] += tree_trace["contribution"]
        if not np.isfinite(raw_scores).all():
            raise ValueError("Boosted-tree raw-score accumulation is not finite")
        raw_logits = self._expand_logits(raw_scores[None, :])[0]
        shifted = raw_logits - np.max(raw_logits)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        if self.output_count == 1:
            prediction_index = int(raw_scores[0] > 0.0)
        else:
            prediction_index = int(np.argmax(raw_scores))
        baseline_logits = self._expand_logits(
            np.asarray(self.baseline, dtype=np.float64)[None, :])[0]
        return {
            "baseline": list(self.baseline),
            "baseline_logits": [float(value) for value in baseline_logits],
            "trees": tree_traces,
            "raw_scores": [float(value) for value in raw_scores],
            "raw_logits": [float(value) for value in raw_logits],
            "probabilities": [float(value) for value in probabilities],
            "prediction_index": prediction_index,
            "prediction": self.classes[prediction_index],
            "action": self.action_names[prediction_index],
        }

    def trace_batch(self, observations: Any) -> tuple[dict[str, Any], ...]:
        values = self._batch(observations)
        return tuple(self.trace(row) for row in values)

    def predict_with_trace(
        self, observation: Mapping[str, Any] | Sequence[float]
    ) -> tuple[Any, dict[str, Any]]:
        trace = self.trace(observation)
        return trace["prediction"], trace

    def predict_proba_with_trace(
        self, observation: Mapping[str, Any] | Sequence[float]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        trace = self.trace(observation)
        return np.asarray(trace["probabilities"], dtype=np.float64), trace


def export_hist_gradient_boosting_classifier(
    estimator: Any,
    *,
    feature_names: Sequence[str] | None = None,
    action_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> R41DiagnosticBoostedTreeProgram:
    """Convenience wrapper for :meth:`R41DiagnosticBoostedTreeProgram.from_sklearn`."""

    return R41DiagnosticBoostedTreeProgram.from_sklearn(
        estimator, feature_names=feature_names, action_names=action_names,
        metadata=metadata,
    )


DiagnosticBoostedTreeProgram = R41DiagnosticBoostedTreeProgram
export_sklearn_hgb = export_hist_gradient_boosting_classifier


__all__ = [
    "VERSION", "MODEL_KIND", "LEAF_VALUE_SEMANTICS",
    "R41DiagnosticBoostedTreeProgram", "DiagnosticBoostedTreeProgram",
    "export_hist_gradient_boosting_classifier", "export_sklearn_hgb",
]
