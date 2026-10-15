"""Compact, dependency-light inference for the r4.1 public program ensemble.

Every router predicate and every sparse leaf coefficient consumes only the
published 197-value observation.  The frozen neural Actor remains the sole
controller.  This module exists only for post-hoc audit and explanation.
"""
from __future__ import annotations

import base64
import bz2
from copy import deepcopy
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.warehouse_r41_diagnostic_model_tree import (
    R41DiagnosticModelTreeProgram,
    VERSION as MEMBER_VERSION,
    _validate_metadata,
)
from core.program import ProgramTraceStep


VERSION = "warehouse-r41-diagnostic-model-tree-ensemble.v2"
ENVELOPE_VERSION = "warehouse-r41-diagnostic-model-tree-ensemble-envelope.v2"
AGGREGATION = "mean_probability"
QMAX = 127
MIN_MEMBER_COUNT = 4
MAX_MEMBER_COUNT = 8
_TOP_FIELDS = frozenset((
    "version", "action_names", "feature_names", "aggregation", "members",
    "metadata",
))
_AGGREGATION_FIELDS = frozenset((
    "kind", "member_count", "coefficient_encoding", "coefficient_qmax",
    "router_threshold_encoding", "intercept_encoding",
))
_MEMBER_FIELDS = frozenset(("seed", "router", "leaf_models"))
_MODEL_FIELDS = frozenset((
    "router_node", "classes", "feature_count", "feature_indices_b64",
    "coefficients_q_b64", "coefficient_scales", "intercepts",
))


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _decode(value: Any, *, expected: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(label + " must be canonical base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError(label + " must be canonical base64") from exc
    if len(raw) != expected or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(label + " byte length or encoding differs")
    return raw


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


class R41DiagnosticModelTreeEnsemble:
    """Mean-probability ensemble of four to eight explicit, quantized model trees."""

    def __init__(self, payload: Mapping[str, Any]):
        if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
            raise ValueError("Diagnostic ensemble top-level schema differs")
        if payload.get("version") != VERSION:
            raise ValueError("Diagnostic ensemble version differs")
        actions = payload.get("action_names")
        features = payload.get("feature_names")
        if (not isinstance(actions, list) or not actions
                or any(not isinstance(item, str) or not item for item in actions)
                or len(set(actions)) != len(actions)
                or not isinstance(features, list) or not features
                or len(features) > 255
                or any(not isinstance(item, str) or not item for item in features)
                or len(set(features)) != len(features)):
            raise ValueError("Diagnostic ensemble action/feature registry differs")
        self.action_names = tuple(actions)
        self.feature_names = tuple(features)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Diagnostic ensemble metadata must be an object")
        _validate_metadata(metadata)
        self.metadata = deepcopy(metadata)

        members = payload.get("members")
        if (not isinstance(members, list)
                or not MIN_MEMBER_COUNT <= len(members) <= MAX_MEMBER_COUNT):
            raise ValueError("Diagnostic ensemble member count differs")
        aggregation = payload.get("aggregation")
        expected_aggregation = {
            "kind": AGGREGATION,
            "member_count": len(members),
            "coefficient_encoding": "symmetric_per_class_int8",
            "coefficient_qmax": QMAX,
            "router_threshold_encoding": "float32",
            "intercept_encoding": "float32",
        }
        if (not isinstance(aggregation, Mapping)
                or set(aggregation) != _AGGREGATION_FIELDS
                or not MIN_MEMBER_COUNT <= aggregation.get("member_count", 0) <= MAX_MEMBER_COUNT
                or dict(aggregation) != expected_aggregation):
            raise ValueError("Diagnostic ensemble aggregation contract differs")
        normalized_members: list[dict[str, Any]] = []
        programs: list[R41DiagnosticModelTreeProgram] = []
        seeds: set[int] = set()
        for member_index, member in enumerate(members):
            if not isinstance(member, Mapping) or set(member) != _MEMBER_FIELDS:
                raise ValueError("Diagnostic ensemble member schema differs")
            seed = member.get("seed")
            if type(seed) is not int or seed < 0 or seed in seeds:
                raise ValueError("Diagnostic ensemble member seed differs")
            seeds.add(seed)
            compact_models = member.get("leaf_models")
            if not isinstance(compact_models, list) or not compact_models:
                raise ValueError("Diagnostic ensemble compact leaves differ")
            expanded_models: list[dict[str, Any]] = []
            normalized_models: list[dict[str, Any]] = []
            for model in compact_models:
                if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
                    raise ValueError("Diagnostic ensemble compact leaf schema differs")
                router_node = model.get("router_node")
                classes = model.get("classes")
                feature_count = model.get("feature_count")
                scales = model.get("coefficient_scales")
                intercepts = model.get("intercepts")
                if (type(router_node) is not int or router_node < 0
                        or not isinstance(classes, list) or not classes
                        or classes != sorted(classes) or len(set(classes)) != len(classes)
                        or any(type(item) is not int
                               or not 0 <= item < len(self.action_names)
                               for item in classes)
                        or type(feature_count) is not int
                        or not 0 <= feature_count <= len(self.feature_names)
                        or not isinstance(scales, list) or len(scales) != len(classes)
                        or not isinstance(intercepts, list)
                        or len(intercepts) != len(classes)):
                    raise ValueError("Diagnostic ensemble compact leaf values differ")
                indices_raw = _decode(model.get("feature_indices_b64"),
                    expected=feature_count, label="feature indices")
                indices = np.frombuffer(indices_raw, dtype=np.uint8).astype(
                    np.int64, copy=False)
                if (len(indices) and (np.any(indices >= len(self.feature_names))
                        or np.any(indices[1:] <= indices[:-1]))):
                    raise ValueError("Diagnostic ensemble feature indices differ")
                coefficient_count = len(classes) * feature_count
                coefficient_raw = _decode(model.get("coefficients_q_b64"),
                    expected=coefficient_count, label="quantized coefficients")
                quantized = np.frombuffer(coefficient_raw, dtype=np.int8).reshape(
                    len(classes), feature_count)
                normalized_scales = [_finite(value, "coefficient scale")
                                     for value in scales]
                normalized_intercepts = [_finite(value, "leaf intercept")
                                         for value in intercepts]
                if (any(value <= 0.0 for value in normalized_scales)
                        or any(float(np.float32(value)) != value
                               for value in (*normalized_scales,
                                             *normalized_intercepts))):
                    raise ValueError("Diagnostic ensemble float32 payload differs")
                coefficients = (
                    quantized.astype(np.float64)
                    * np.asarray(normalized_scales, dtype=np.float64)[:, None]
                )
                if len(classes) == 1 and (feature_count != 0 or coefficient_count != 0):
                    raise ValueError("Constant diagnostic ensemble leaf has coefficients")
                normalized_models.append({
                    "router_node": router_node,
                    "classes": list(classes),
                    "feature_count": feature_count,
                    "feature_indices_b64": model["feature_indices_b64"],
                    "coefficients_q_b64": model["coefficients_q_b64"],
                    "coefficient_scales": normalized_scales,
                    "intercepts": normalized_intercepts,
                })
                expanded_models.append({
                    "router_node": router_node,
                    "classes": list(classes),
                    "feature_indices": indices.tolist(),
                    "coefficients": coefficients.tolist(),
                    "intercepts": normalized_intercepts,
                })
            member_metadata = {
                **deepcopy(self.metadata),
                "ensemble_member_index": member_index,
                "ensemble_member_seed": seed,
            }
            expanded = {
                "version": MEMBER_VERSION,
                "action_names": list(self.action_names),
                "feature_names": list(self.feature_names),
                "router": deepcopy(member.get("router")),
                "leaf_models": expanded_models,
                "metadata": member_metadata,
            }
            program = R41DiagnosticModelTreeProgram.from_dict(expanded)
            programs.append(program)
            normalized_members.append({
                "seed": seed,
                "router": deepcopy(program.to_dict()["router"]),
                "leaf_models": normalized_models,
            })
        self.member_seeds = tuple(member["seed"] for member in normalized_members)
        self._members = tuple(programs)
        self._payload = {
            "version": VERSION,
            "action_names": list(self.action_names),
            "feature_names": list(self.feature_names),
            "aggregation": expected_aggregation,
            "members": normalized_members,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "R41DiagnosticModelTreeEnsemble":
        return cls(payload)

    @classmethod
    def from_unquantized_members(
        cls, members: Sequence[tuple[int, Mapping[str, Any]]], *,
        metadata: Mapping[str, Any],
    ) -> "R41DiagnosticModelTreeEnsemble":
        if not MIN_MEMBER_COUNT <= len(members) <= MAX_MEMBER_COUNT:
            raise ValueError("Four to eight unquantized members are required")
        first = members[0][1]
        compact = []
        for seed, member in members:
            program = R41DiagnosticModelTreeProgram.from_dict(member)
            if (program.action_names != tuple(first["action_names"])
                    or program.feature_names != tuple(first["feature_names"])):
                raise ValueError("Unquantized member registries differ")
            leaves = []
            for model in member["leaf_models"]:
                coefficients = np.asarray(model["coefficients"], dtype=np.float64)
                if coefficients.size:
                    scales = np.max(np.abs(coefficients), axis=1) / QMAX
                    scales[scales == 0.0] = 1.0
                    scales = scales.astype(np.float32)
                    quantized = np.rint(
                        coefficients / scales.astype(np.float64)[:, None]
                    ).clip(-QMAX, QMAX).astype(np.int8)
                else:
                    scales = np.ones(len(model["classes"]), dtype=np.float32)
                    quantized = np.empty((len(model["classes"]), 0), dtype=np.int8)
                indices = np.asarray(model["feature_indices"], dtype=np.uint8)
                leaves.append({
                    "router_node": int(model["router_node"]),
                    "classes": list(model["classes"]),
                    "feature_count": int(len(indices)),
                    "feature_indices_b64": _encode(indices.tobytes()),
                    "coefficients_q_b64": _encode(quantized.tobytes()),
                    "coefficient_scales": [float(value) for value in scales],
                    "intercepts": [float(value) for value in np.asarray(
                        model["intercepts"], dtype=np.float32)],
                })
            compact.append({"seed": int(seed), "router": deepcopy(member["router"]),
                            "leaf_models": leaves})
        return cls({
            "version": VERSION,
            "action_names": list(first["action_names"]),
            "feature_names": list(first["feature_names"]),
            "aggregation": {
                "kind": AGGREGATION,
                "member_count": len(members),
                "coefficient_encoding": "symmetric_per_class_int8",
                "coefficient_qmax": QMAX,
                "router_threshold_encoding": "float32",
                "intercept_encoding": "float32",
            },
            "members": compact,
            "metadata": deepcopy(dict(metadata)),
        })

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def predict_proba_batch(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Diagnostic ensemble observation batch differs")
        probabilities = np.mean([
            member.predict_proba_batch(values) for member in self._members
        ], axis=0)
        if (not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0,
                                   rtol=0.0, atol=2e-12)):
            raise ValueError("Diagnostic ensemble probabilities are invalid")
        return probabilities

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba_batch(observations), axis=1).astype(np.uint8)

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        if not isinstance(features, Mapping):
            raise ValueError("Diagnostic ensemble features must be a mapping")
        try:
            vector = np.asarray([float(features[name]) for name in self.feature_names],
                                dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Diagnostic ensemble requires every public feature") from exc
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("Diagnostic ensemble public feature vector differs")
        return vector

    def predict_proba(self, features: Mapping[str, float]) -> dict[str, float]:
        probabilities = self.predict_proba_batch(self._vector(features)[None, :])[0]
        return {action: float(probabilities[index])
                for index, action in enumerate(self.action_names)}

    def predict(self, features: Mapping[str, float]) -> str:
        probabilities = self.predict_proba_batch(self._vector(features)[None, :])[0]
        return self.action_names[int(np.argmax(probabilities))]

    def trace(self, features: Mapping[str, float]) -> tuple[ProgramTraceStep, ...]:
        # Concatenating all member paths makes every predicate that contributed to
        # the mean-probability decision inspectable.  Participant prose uses at
        # most the first three translated predicates in its folded detail.
        return tuple(step for member in self._members
                     for step in member.trace(features))

    def member_traces(self, features: Mapping[str, float]) -> tuple[tuple[ProgramTraceStep, ...], ...]:
        return tuple(member.trace(features) for member in self._members)

    def complexity(self) -> dict[str, Any]:
        members = [member.complexity() for member in self._members]
        return {
            "member_count": len(members),
            "member_seeds": list(self.member_seeds),
            "aggregation": AGGREGATION,
            "router_nodes": sum(row["router_nodes"] for row in members),
            "router_leaves": sum(row["router_leaves"] for row in members),
            "maximum_member_router_depth": max(row["router_depth"] for row in members),
            "maximum_member_router_leaves": max(row["router_leaves"] for row in members),
            "nonzero_quantized_coefficients": sum(
                int(np.count_nonzero(np.frombuffer(base64.b64decode(
                    model["coefficients_q_b64"]), dtype=np.int8)))
                for member in self._payload["members"]
                for model in member["leaf_models"]
            ),
            "maximum_features_per_leaf": max(
                model["feature_count"] for member in self._payload["members"]
                for model in member["leaf_models"]
            ),
            "available_features": len(self.feature_names),
            "actions": len(self.action_names),
        }


def program_content_sha256(program: R41DiagnosticModelTreeEnsemble) -> str:
    """Hash the canonical compact payload without importing training code."""
    import json
    raw = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def make_program_envelope(program: R41DiagnosticModelTreeEnsemble) -> dict[str, Any]:
    """Return a deterministic, bounded transport envelope for release files."""
    raw = _canonical_bytes(program.to_dict())
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Diagnostic ensemble payload exceeds its decoded budget")
    compressed = bz2.compress(raw, compresslevel=9)
    if len(compressed) > 512 * 1024:
        raise ValueError("Diagnostic ensemble payload exceeds its compressed budget")
    return {
        "version": ENVELOPE_VERSION,
        "encoding": "canonical-json-bz2-base64",
        "decoded_bytes": len(raw),
        "payload_sha256": sha256(raw).hexdigest(),
        "payload_b64": _encode(compressed),
    }


def load_program_envelope(value: Mapping[str, Any]) -> R41DiagnosticModelTreeEnsemble:
    """Authenticate, bound, decode, and validate a release program envelope."""
    import json
    if (not isinstance(value, Mapping)
            or set(value) != {"version", "encoding", "decoded_bytes",
                              "payload_sha256", "payload_b64"}
            or value.get("version") != ENVELOPE_VERSION
            or value.get("encoding") != "canonical-json-bz2-base64"
            or type(value.get("decoded_bytes")) is not int
            or not 0 < value["decoded_bytes"] <= 2 * 1024 * 1024
            or not isinstance(value.get("payload_sha256"), str)
            or len(value["payload_sha256"]) != 64):
        raise ValueError("Diagnostic ensemble envelope schema differs")
    encoded = value.get("payload_b64")
    if not isinstance(encoded, str) or len(encoded) > 700_000:
        raise ValueError("Diagnostic ensemble envelope is oversized")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Diagnostic ensemble envelope base64 differs") from exc
    if (len(compressed) > 512 * 1024
            or base64.b64encode(compressed).decode("ascii") != encoded):
        raise ValueError("Diagnostic ensemble compressed payload differs")
    try:
        raw = bz2.decompress(compressed)
    except OSError as exc:
        raise ValueError("Diagnostic ensemble compressed payload is invalid") from exc
    if (len(raw) != value["decoded_bytes"]
            or sha256(raw).hexdigest() != value["payload_sha256"]):
        raise ValueError("Diagnostic ensemble decoded identity differs")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Diagnostic ensemble decoded JSON differs") from exc
    if not isinstance(payload, dict) or _canonical_bytes(payload) != raw:
        raise ValueError("Diagnostic ensemble payload is not canonical")
    return R41DiagnosticModelTreeEnsemble.from_dict(payload)


__all__ = [
    "VERSION", "ENVELOPE_VERSION", "AGGREGATION", "QMAX",
    "MIN_MEMBER_COUNT", "MAX_MEMBER_COUNT",
    "R41DiagnosticModelTreeEnsemble", "program_content_sha256",
    "make_program_envelope", "load_program_envelope",
]
