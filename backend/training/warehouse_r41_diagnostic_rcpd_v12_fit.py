"""Development-only fitting for the v12 diagnostic explanation program.

V12 keeps the v9 public base program and shared-pickup replacement, and adds
one preregistered shared-charger specialist.  The specialist is trained only
on unique endpoints of effective shared-charger pairs in the supplied fit
partition.  It is an explanation program component; it never controls the
runtime neural policy.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_fit as v9
from backend.training.warehouse_native_common import digest
from backend.warehouse_r41_diagnostic_boosted_tree import (
    R41DiagnosticBoostedTreeProgram,
    export_hist_gradient_boosting_classifier,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v12-fit.v1"
CONFIG_VERSION = "warehouse-r41-diagnostic-rcpd-v12-fit-config.v1"
CHARGER_SPECIALIST_VERSION = (
    "warehouse-r41-diagnostic-rcpd-v12-shared-charger-endpoint-specialist.v1")
CHARGER_ROUTE_FEATURE = "derived.critical.shared_charger"
ACTIONS = v9.ACTIONS
CLASSES = v9.CLASSES
ReplacementSupportError = v9.ReplacementSupportError
REPLACEMENT_VERSION = v9.REPLACEMENT_VERSION
REPLACEMENT_ROUTE_FEATURE = v9.REPLACEMENT_ROUTE_FEATURE
_replacement_route_mask = v9._replacement_route_mask
_partition_support = v9._partition_support

_CONFIG_FIELDS = frozenset((
    "version", "pair_pool_multiplier", "wait_endpoint_share", "model",
    "shared_pickup_replacement", "shared_charger_endpoint_specialist",
))
_SPECIALIST_FIELDS = frozenset((
    "version", "enabled", "group", "combination", "mix_weight", "route",
    "fit_source", "estimator", "endpoint_population",
    "endpoint_wait_share", "sample_weight_source", "model",
))
_ROUTE_FIELDS = frozenset(("feature_name", "operator", "threshold"))
_MODEL_FIELDS = frozenset((
    "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
    "l2_regularization", "max_depth", "max_bins", "random_state",
))


def _base_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": v9.CONFIG_VERSION,
        "pair_pool_multiplier": value["pair_pool_multiplier"],
        "wait_endpoint_share": value["wait_endpoint_share"],
        "model": deepcopy(value["model"]),
        "shared_pickup_replacement": deepcopy(
            value["shared_pickup_replacement"]),
    }


def _model_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_FIELDS:
        raise ValueError("V12 charger-specialist model schema differs")

    def finite(name: str, low: float, high: float) -> float:
        raw = value.get(name)
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError("V12 charger-specialist model differs: " + name)
        result = float(raw)
        if not low <= result <= high:
            raise ValueError("V12 charger-specialist model differs: " + name)
        return result

    integers = {name: value.get(name) for name in (
        "max_iter", "max_leaf_nodes", "min_samples_leaf", "max_bins",
        "random_state",
    )}
    if (any(type(item) is not int for item in integers.values())
            or not 1 <= integers["max_iter"] <= 400
            or not 2 <= integers["max_leaf_nodes"] <= 255
            or not 5 <= integers["min_samples_leaf"] <= 1000
            or not 16 <= integers["max_bins"] <= 255
            or not 0 <= integers["random_state"] <= 2**31 - 1):
        raise ValueError("V12 charger-specialist integer model differs")
    depth = value.get("max_depth")
    if depth is not None and (type(depth) is not int or not 2 <= depth <= 20):
        raise ValueError("V12 charger-specialist max_depth differs")
    return {
        "learning_rate": finite("learning_rate", 1e-4, 1.0),
        "max_iter": integers["max_iter"],
        "max_leaf_nodes": integers["max_leaf_nodes"],
        "min_samples_leaf": integers["min_samples_leaf"],
        "l2_regularization": finite("l2_regularization", 0.0, 1000.0),
        "max_depth": depth,
        "max_bins": integers["max_bins"],
        "random_state": integers["random_state"],
    }


def normalize_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the one auditable v12 fit configuration family."""
    if (not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS
            or value.get("version") != CONFIG_VERSION):
        raise ValueError("V12 fit config schema differs")
    base = dict(value)
    base["version"] = v9.CONFIG_VERSION
    base.pop("shared_charger_endpoint_specialist")
    normalized_base = v9.normalize_config(base)
    specialist = value.get("shared_charger_endpoint_specialist")
    route = specialist.get("route") if isinstance(specialist, Mapping) else None
    model = specialist.get("model") if isinstance(specialist, Mapping) else None
    if (not isinstance(specialist, Mapping)
            or set(specialist) != _SPECIALIST_FIELDS
            or specialist.get("version") != CHARGER_SPECIALIST_VERSION
            or specialist.get("enabled") is not True
            or specialist.get("group") != "shared_charger"
            or specialist.get("combination") != "weighted_average"
            or type(specialist.get("mix_weight")) not in (int, float)
            or not math.isfinite(float(specialist["mix_weight"]))
            or float(specialist["mix_weight"]) != 1.0
            or specialist.get("fit_source")
                != "fit_partition_effective_shared_charger_pair_endpoints"
            or specialist.get("estimator")
                != "hist_gradient_boosting_classifier"
            or specialist.get("endpoint_population")
                != "unique_pair_endpoint_rows"
            or type(specialist.get("endpoint_wait_share")) not in (int, float)
            or not math.isfinite(float(specialist["endpoint_wait_share"]))
            or float(specialist["endpoint_wait_share"]) != 0.8
            or specialist.get("sample_weight_source")
                != "fit_only_global_weights_with_specialist_0.8_wait_share"
            or not isinstance(route, Mapping) or set(route) != _ROUTE_FIELDS
            or route.get("feature_name") != CHARGER_ROUTE_FEATURE
            or route.get("operator") != ">"
            or type(route.get("threshold")) not in (int, float)
            or float(route["threshold"]) != 0.5):
        raise ValueError("V12 shared-charger endpoint specialist differs")
    if normalized_base["wait_endpoint_share"] != 0.5:
        raise ValueError("V12 base program requires frozen v11 0.5 WAIT share")
    return {
        "version": CONFIG_VERSION,
        "pair_pool_multiplier": normalized_base["pair_pool_multiplier"],
        "wait_endpoint_share": normalized_base["wait_endpoint_share"],
        "model": normalized_base["model"],
        "shared_pickup_replacement": normalized_base[
            "shared_pickup_replacement"],
        "shared_charger_endpoint_specialist": {
            "version": CHARGER_SPECIALIST_VERSION,
            "enabled": True,
            "group": "shared_charger",
            "combination": "weighted_average",
            "mix_weight": 1.0,
            "route": {
                "feature_name": CHARGER_ROUTE_FEATURE,
                "operator": ">", "threshold": 0.5,
            },
            "fit_source": (
                "fit_partition_effective_shared_charger_pair_endpoints"),
            "estimator": "hist_gradient_boosting_classifier",
            "endpoint_population": "unique_pair_endpoint_rows",
            "endpoint_wait_share": 0.8,
            "sample_weight_source": (
                "fit_only_global_weights_with_specialist_0.8_wait_share"),
            "model": _model_config(model),
        },
    }


def shared_charger_endpoint_indices(
    pairs: np.ndarray, pair_bits: np.ndarray,
) -> np.ndarray:
    """Return unique endpoints of fit-local effective charger pairs."""
    pairs = np.asarray(pairs)
    pair_bits = np.asarray(pair_bits)
    if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
            or pair_bits.shape != (len(pairs),)
            or pairs.dtype.kind not in "iu" or pair_bits.dtype.kind not in "iu"):
        raise ValueError("V12 effective-pair projection differs")
    bit = 1 << GROUPS.index("shared_charger")
    selected = (pair_bits & bit) != 0
    if not np.any(selected):
        raise ReplacementSupportError(
            "V12 fit partition has no effective shared-charger pair")
    return np.unique(pairs[selected].reshape(-1)).astype(np.int64, copy=False)


def _fit_charger_specialist(
    arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray, *,
    relations: R41DiagnosticPublicRelationsV9,
    scene_families: Mapping[str, str], config: Mapping[str, Any],
    binding_sha256: str,
) -> tuple[R41DiagnosticBoostedTreeProgram, dict[str, Any]]:
    normalized = normalize_config(config)
    base_config = _base_config(normalized)
    specialist_weight_config = deepcopy(base_config)
    specialist_weight_config["wait_endpoint_share"] = normalized[
        "shared_charger_endpoint_specialist"]["endpoint_wait_share"]
    weights, pairs, pair_bits, weight_audit = v9.build_fit_weights(
        arrays, fit_mask, scene_families=scene_families,
        config=specialist_weight_config)
    indices = shared_charger_endpoint_indices(pairs, pair_bits)
    selected = np.asarray(fit_mask, dtype=np.bool_)
    if (selected.shape != (len(arrays["observations"]),)
            or np.any(~selected[indices])):
        raise RuntimeError("V12 charger endpoints escaped the fit partition")
    labels = np.asarray(arrays["action_indices"])[indices].astype(
        np.int64, copy=True)
    if tuple(map(int, np.unique(labels))) != CLASSES:
        raise RuntimeError("V12 charger endpoints do not cover all actions")
    expanded = relations.transform_batch(
        np.asarray(arrays["observations"])[indices])
    sample_weights = np.asarray(weights[indices], dtype=np.float64)
    if (sample_weights.shape != (len(indices),)
            or not np.isfinite(sample_weights).all()
            or np.any(sample_weights <= 0.0)):
        raise RuntimeError("V12 charger endpoint weights differ")
    settings = normalized["shared_charger_endpoint_specialist"]["model"]
    estimator = HistGradientBoostingClassifier(
        loss="log_loss", early_stopping=False,
        learning_rate=settings["learning_rate"],
        max_iter=settings["max_iter"],
        max_leaf_nodes=settings["max_leaf_nodes"],
        min_samples_leaf=settings["min_samples_leaf"],
        l2_regularization=settings["l2_regularization"],
        max_depth=settings["max_depth"], max_bins=settings["max_bins"],
        random_state=settings["random_state"],
    )
    estimator.fit(expanded, labels, sample_weight=sample_weights)
    program = export_hist_gradient_boosting_classifier(
        estimator, feature_names=relations.feature_names,
        action_names=ACTIONS,
        metadata={
            "version": VERSION,
            "purpose": "fit-only shared-charger endpoint specialist",
            "binding_sha256": binding_sha256,
            "fit_config_sha256": digest(normalized),
            "fit_source": (
                "fit_partition_effective_shared_charger_pair_endpoints"),
            "held_labels_used": False,
            "runtime_action_override": False,
        },
    )
    sample = np.arange(min(4096, len(indices)))
    native = estimator.predict_proba(expanded[sample])
    explicit = program.predict_proba_batch(expanded[sample])
    maximum_error = float(np.max(np.abs(native - explicit)))
    if (maximum_error > 1e-12 or not np.array_equal(
            np.argmax(native, axis=1), np.argmax(explicit, axis=1))):
        raise RuntimeError("V12 charger sklearn/explicit parity differs")
    scenes = v9._decode(
        np.asarray(arrays["scene_fingerprints"]), "scene fingerprints")
    episodes = v9._decode(np.asarray(arrays["episode_ids"]), "episode ids")
    diagnostics = {
        "fit_endpoint_rows": len(indices),
        "fit_endpoint_scenes": len(set(map(str, scenes[indices]))),
        "fit_endpoint_episodes": len(set(map(str, episodes[indices]))),
        "fit_action_counts": {
            action: int(np.sum(labels == index))
            for index, action in enumerate(ACTIONS)
        },
        "fit_pair_occurrences": int(np.sum(
            (pair_bits & (1 << GROUPS.index("shared_charger"))) != 0)),
        "sample_weight_mass": float(math.fsum(map(float, sample_weights))),
        "sample_weight_source": normalized[
            "shared_charger_endpoint_specialist"]["sample_weight_source"],
        "base_wait_endpoint_share": normalized["wait_endpoint_share"],
        "specialist_endpoint_wait_share": normalized[
            "shared_charger_endpoint_specialist"]["endpoint_wait_share"],
        "fit_weights_content_sha256": weight_audit["content_sha256"],
        "model_iterations": program.n_iterations,
        "sklearn_explicit_max_probability_error": maximum_error,
        "sklearn_explicit_actions_equal": True,
        "held_labels_used": False,
        "runtime_action_override": False,
    }
    return program, diagnostics


def fit_program(
    arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray, *,
    relations: R41DiagnosticPublicRelationsV9,
    scene_families: Mapping[str, str], config: Mapping[str, Any],
    binding_sha256: str,
) -> tuple[R41DiagnosticPublicTreeProgramV9, dict[str, Any]]:
    """Fit the v9 base/pickup components and the fit-only charger component."""
    normalized = normalize_config(config)
    base, diagnostics = v9.fit_program(
        arrays, fit_mask, relations=relations,
        scene_families=scene_families, config=_base_config(normalized),
        binding_sha256=binding_sha256)
    charger, charger_diagnostics = _fit_charger_specialist(
        arrays, fit_mask, relations=relations,
        scene_families=scene_families, config=normalized,
        binding_sha256=binding_sha256)
    payload = base.to_dict()
    matches = [item for item in payload["specialists"]
               if item.get("group") == "shared_charger"]
    if len(matches) != 1:
        raise RuntimeError("V12 base program charger component differs")
    matches[0].update({
        "route": deepcopy(
            normalized["shared_charger_endpoint_specialist"]["route"]),
        "combination": "weighted_average",
        "mix_weight": 1.0,
        "program": charger.to_dict(),
    })
    payload["metadata"] = {
        **payload.get("metadata", {}),
        "version": VERSION,
        "fit_config_sha256": digest(normalized),
        "shared_charger_endpoint_specialist": True,
        "shared_charger_specialist_fit_only": True,
        "program_feedback_into_actor": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    program = R41DiagnosticPublicTreeProgramV9(payload)
    diagnostics = {
        **deepcopy(diagnostics),
        "version": VERSION,
        "fit_config_sha256": digest(normalized),
        "shared_charger_endpoint_specialist": charger_diagnostics,
    }
    return program, diagnostics


def predict_in_batches(
    program: R41DiagnosticPublicTreeProgramV9, observations: np.ndarray, *,
    batch_size: int = 16_384,
) -> np.ndarray:
    return v9.predict_in_batches(
        program, observations, batch_size=batch_size)


__all__ = [
    "VERSION", "CONFIG_VERSION", "CHARGER_SPECIALIST_VERSION",
    "CHARGER_ROUTE_FEATURE", "ACTIONS", "CLASSES",
    "ReplacementSupportError", "REPLACEMENT_VERSION",
    "REPLACEMENT_ROUTE_FEATURE", "normalize_config",
    "shared_charger_endpoint_indices", "fit_program", "predict_in_batches",
    "_replacement_route_mask", "_partition_support",
]
