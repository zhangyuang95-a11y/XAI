"""Development-only fitting primitives for the r4.1 diagnostic v9 program.

Every estimator consumes deterministic public features.  Actor action and
probability fields are fit targets and audit targets only; neither is stored
as a runtime feature.  Whole-scene masks are supplied by the selector so this
module cannot choose or inspect the fresh outer split.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as pair_weights_v8
from backend.training.warehouse_native_common import digest
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS,
    MODEL_KIND,
    R41DiagnosticBoostedTreeProgram,
    VERSION as BOOSTED_TREE_VERSION,
    export_hist_gradient_boosting_classifier,
)
from backend.warehouse_r41_diagnostic_public_features_v9 import (
    R41DiagnosticPublicRelationsV9,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    GROUPS,
    R41DiagnosticPublicTreeProgramV9,
    assemble_public_tree_program_v9,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v9-fit.v1"
ACTIONS = tuple(v8.ACTIONS)
CLASSES = tuple(range(len(ACTIONS)))
CONFIG_VERSION = "warehouse-r41-diagnostic-rcpd-v9-fit-config.v3"
REPLACEMENT_VERSION = (
    "warehouse-r41-diagnostic-rcpd-v9-shared-pickup-replacement.v1")
REPLACEMENT_ROUTE_FEATURE = (
    "derived.v9.route.shared_pickup_stationary_delivery_block")
_MODEL_FIELDS = frozenset((
    "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
    "l2_regularization", "max_depth", "max_bins", "random_state",
))
_CONFIG_FIELDS = frozenset((
    "version", "pair_pool_multiplier", "wait_endpoint_share", "model",
    "shared_pickup_replacement",
))
_REPLACEMENT_FIELDS = frozenset((
    "version", "enabled", "group", "combination", "route", "fit_source",
    "estimator", "minimum_rows_per_partition",
    "minimum_scenes_per_partition", "minimum_episodes_per_partition",
))
_REPLACEMENT_ROUTE_FIELDS = frozenset((
    "feature_name", "operator", "threshold",
))


class ReplacementSupportError(ValueError):
    """A preregistered replacement route lacks one fold's public support."""


def normalize_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
        raise ValueError("V9 fit config schema differs")
    model = value.get("model")
    if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
        raise ValueError("V9 model config schema differs")

    def number(name: str, *, low: float, high: float) -> float:
        raw = value.get(name)
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError("V9 fit config " + name + " differs")
        result = float(raw)
        if not low <= result <= high:
            raise ValueError("V9 fit config " + name + " differs")
        return result

    def model_number(name: str, *, low: float, high: float) -> float:
        raw = model.get(name)
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError("V9 model config " + name + " differs")
        result = float(raw)
        if not low <= result <= high:
            raise ValueError("V9 model config " + name + " differs")
        return result

    if value.get("version") != CONFIG_VERSION:
        raise ValueError("V9 fit config version differs")
    replacement = value.get("shared_pickup_replacement")
    route = replacement.get("route") if isinstance(replacement, Mapping) else None
    if (not isinstance(replacement, Mapping)
            or set(replacement) != _REPLACEMENT_FIELDS
            or replacement.get("version") != REPLACEMENT_VERSION
            or type(replacement.get("enabled")) is not bool
            or replacement.get("group") != "shared_pickup"
            or replacement.get("combination") != "replacement"
            or replacement.get("fit_source") != "fit_only_public_route_rows"
            or replacement.get("estimator") != "fit_only_weighted_majority"
            or not isinstance(route, Mapping)
            or set(route) != _REPLACEMENT_ROUTE_FIELDS
            or route.get("feature_name") != REPLACEMENT_ROUTE_FEATURE
            or route.get("operator") != ">"
            or type(route.get("threshold")) not in (int, float)
            or float(route["threshold"]) != 0.5):
        raise ValueError("V9 shared-pickup replacement config differs")
    minimums = {
        name: replacement.get(name) for name in (
            "minimum_rows_per_partition", "minimum_scenes_per_partition",
            "minimum_episodes_per_partition",
        )
    }
    if (any(type(item) is not int for item in minimums.values())
            or not 1 <= minimums["minimum_rows_per_partition"] <= 1_000_000
            or not 1 <= minimums["minimum_scenes_per_partition"] <= 10_000
            or not 1 <= minimums["minimum_episodes_per_partition"] <= 1_000_000):
        raise ValueError("V9 shared-pickup replacement support differs")
    integers = {
        name: model.get(name) for name in (
            "max_iter", "max_leaf_nodes", "min_samples_leaf", "max_bins",
            "random_state",
        )
    }
    if (any(type(item) is not int for item in integers.values())
            or not 1 <= integers["max_iter"] <= 400
            or not 2 <= integers["max_leaf_nodes"] <= 255
            or not 5 <= integers["min_samples_leaf"] <= 1000
            or not 16 <= integers["max_bins"] <= 255
            or not 0 <= integers["random_state"] <= 2**31 - 1):
        raise ValueError("V9 integer model config differs")
    depth = model.get("max_depth")
    if depth is not None and (type(depth) is not int or not 2 <= depth <= 20):
        raise ValueError("V9 model max_depth differs")
    return {
        "version": CONFIG_VERSION,
        "pair_pool_multiplier": number(
            "pair_pool_multiplier", low=1.0,
            high=pair_weights_v8.MAX_PAIR_POOL_MULTIPLIER),
        "wait_endpoint_share": number(
            "wait_endpoint_share", low=0.5, high=0.8),
        "shared_pickup_replacement": {
            "version": REPLACEMENT_VERSION,
            "enabled": replacement["enabled"],
            "group": "shared_pickup",
            "combination": "replacement",
            "route": {
                "feature_name": REPLACEMENT_ROUTE_FEATURE,
                "operator": ">",
                "threshold": 0.5,
            },
            "fit_source": "fit_only_public_route_rows",
            "estimator": "fit_only_weighted_majority",
            **minimums,
        },
        "model": {
            "learning_rate": model_number("learning_rate", low=1e-4, high=1.0),
            "max_iter": integers["max_iter"],
            "max_leaf_nodes": integers["max_leaf_nodes"],
            "min_samples_leaf": integers["min_samples_leaf"],
            "l2_regularization": model_number(
                "l2_regularization", low=0.0, high=1000.0),
            "max_depth": depth,
            "max_bins": integers["max_bins"],
            "random_state": integers["random_state"],
        },
    }


def _decode(values: np.ndarray, label: str) -> np.ndarray:
    if values.ndim != 1 or values.dtype.kind != "S":
        raise ValueError(label + " differs")
    try:
        return np.char.decode(values, "ascii").astype(str)
    except UnicodeDecodeError as error:
        raise ValueError(label + " differs") from error


def build_fit_weights(
    arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray, *,
    scene_families: Mapping[str, str], config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Balance only fit rows and add mass to effective pair endpoints."""
    normalized = normalize_config(config)
    count = len(arrays["observations"])
    selected = np.asarray(fit_mask)
    if selected.shape != (count,) or selected.dtype != np.dtype(np.bool_) \
            or not np.any(selected):
        raise ValueError("V9 fit mask differs")
    scenes = _decode(np.asarray(arrays["scene_fingerprints"]), "scene fingerprints")
    hashes = _decode(np.asarray(arrays["observation_hashes"]), "observation hashes")
    labels = np.asarray(arrays["action_indices"])
    bits = np.asarray(arrays["group_bits"])
    if (labels.shape != (count,) or labels.dtype.kind not in "iu"
            or bits.shape != (count,) or bits.dtype.kind not in "iu"):
        raise ValueError("V9 fit labels or public critical bits differ")
    # Advanced indexing happens before validation so held-fold labels are never
    # read, validated, summarized, or allowed to affect candidate selection.
    fit_labels = labels[np.flatnonzero(selected)].astype(np.int64, copy=True)
    if np.any(fit_labels < 0) or np.any(fit_labels >= len(ACTIONS)):
        raise ValueError("V9 fit labels differ")
    if any(scene not in scene_families for scene in set(map(str, scenes[selected]))):
        raise ValueError("V9 scene-family registry is incomplete")

    # Recompute the established leakage-safe v8 hierarchy separately for every
    # whole-scene fold.  Its implementation indexes fit labels only.  Reusing
    # weights stored by an earlier split would indirectly expose that split's
    # validation labels to this selector.
    fold_arrays = dict(arrays)
    fold_arrays["split_validation"] = (~selected).astype(np.bool_)
    established = pair_weights_v8.build_pair_weights(
        fold_arrays, scene_families=scene_families,
        use_action_factor=True,
        pair_pool_multiplier=normalized["pair_pool_multiplier"],
    )
    weights = np.asarray(established["weights"], dtype=np.float64)
    weights[~selected] = 0.0
    pairs = np.asarray(established["pairs"], dtype=np.int64)
    occurrences = established["audit"]["pair_occurrences"]
    # The established builder copied group_bits only for fit pair occurrences.
    # Reuse that audit projection instead of invoking a helper which validates
    # the complete row-aligned critical-bit array, including held-fold rows.
    pair_bits = np.asarray(
        [item["group_bits"] for item in occurrences], dtype=np.uint8)

    # Preserve the established family -> scene -> partner -> critical-bit pair
    # allocation exactly.  The only v9 change is a preregistered redistribution
    # within each occurrence from equal endpoints to WAIT/branch shares.  No
    # clipping follows, so occurrence mass and the hierarchy remain intact.
    old_pair = np.asarray(established["pair_contribution"], dtype=np.float64)
    old_pair[~selected] = 0.0
    old_pair_mass = float(math.fsum(map(float, old_pair[selected])))
    occurrence_mass = float(math.fsum(
        float(item["weighted_occurrence_mass"]) for item in occurrences))
    replacement = np.zeros(count, dtype=np.float64)
    wait_share = normalized["wait_endpoint_share"]
    branch_share = 1.0 - wait_share
    scale = old_pair_mass / occurrence_mass if occurrence_mass else 1.0
    if len(pairs) != len(occurrences) or (len(pairs) and occurrence_mass <= 0.0):
        raise ValueError("V9 established pair occurrence audit differs")
    for pair_index, (pair, item) in enumerate(zip(pairs, occurrences)):
        wait_row, branch_row = map(int, pair)
        if (int(item["pair_index"]) != pair_index
                or int(item["wait_row"]) != wait_row
                or int(item["branch_row"]) != branch_row):
            raise ValueError("V9 established pair occurrence rows differ")
        mass = float(item["weighted_occurrence_mass"]) * scale
        replacement[wait_row] += mass * wait_share
        replacement[branch_row] += mass * branch_share
    new_pair_mass = float(math.fsum(map(float, replacement[selected])))
    tolerance = max(1e-9, 32 * np.finfo(np.float64).eps
                    * max(1.0, old_pair_mass, new_pair_mass))
    if abs(old_pair_mass - new_pair_mass) > tolerance:
        raise ValueError("V9 pair endpoint redistribution changed pair mass")
    weights = weights - old_pair + replacement
    weights[~selected] = 0.0
    if (not np.isfinite(weights).all() or np.any(weights[selected] <= 0.0)
            or np.any(weights[~selected] != 0.0)):
        raise ValueError("V9 fit weights are invalid")
    audit = {
        "version": VERSION + ".weights.v1",
        "fit_rows": int(np.sum(selected)),
        "unique_fit_observations": len(set(map(str, hashes[selected]))),
        "effective_pair_count": len(pairs),
        "pair_pool_multiplier": normalized["pair_pool_multiplier"],
        "wait_endpoint_share": wait_share,
        "branch_endpoint_share": branch_share,
        "pair_mass_before_redistribution": old_pair_mass,
        "pair_mass_after_redistribution": new_pair_mass,
        "pair_mass_preserved": True,
        "pair_mass_preservation_tolerance": tolerance,
        "final_weight_mass": float(math.fsum(map(float, weights))),
        "post_redistribution_clipping": False,
        "configuration": deepcopy(normalized),
        "established_pair_weight_contract_sha256": digest(
            pair_weights_v8.contract()),
        "established_pair_weight_audit_content_sha256": digest(
            established["audit"]),
        "validation_labels_used": False,
        "outer_labels_used": False,
        "protected_final_access": False,
    }
    audit["content_sha256"] = digest(audit)
    return weights, pairs, pair_bits, audit


def constant_component(
    relations: R41DiagnosticPublicRelationsV9,
) -> R41DiagnosticBoostedTreeProgram:
    """Return a tiny valid component for zero-weight specialist routes."""
    trees = [{
        "iteration": 0,
        "output_index": output,
        "nodes": [{"kind": "leaf", "value": 0.0}],
    } for output in CLASSES]
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_TREE_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(CLASSES),
        "action_names": list(ACTIONS),
        "output_kind": "multiclass_logits",
        "baseline": [0.0] * len(ACTIONS),
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {
            "purpose": "zero-weight schema component",
            "prediction_input": "deterministic public features",
        },
    })


def _replacement_route_mask(
    observations: np.ndarray, relations: R41DiagnosticPublicRelationsV9,
    *, batch_size: int = 16_384,
) -> np.ndarray:
    """Evaluate the frozen conjunctive route from raw public rows only."""
    values = np.asarray(observations)
    if values.ndim != 2 or values.shape[1] != 197:
        raise ValueError("V9 replacement public observations differ")
    feature_index = relations.feature_names.index(REPLACEMENT_ROUTE_FEATURE)
    parts = [
        relations.transform_batch(values[start:start + batch_size])[:, feature_index] > .5
        for start in range(0, len(values), batch_size)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.bool_)


def _partition_support(mask: np.ndarray, scenes: np.ndarray,
                       episodes: np.ndarray) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    return {
        "rows": len(selected),
        "scenes": len(set(map(str, scenes[selected]))),
        "episodes": len(set(map(str, episodes[selected]))),
        "scene_registry_sha256": digest(sorted(set(map(str, scenes[selected])))),
        "episode_registry_sha256": digest(sorted(set(map(str, episodes[selected])))),
    }


def _replacement_support(
    arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray,
    relations: R41DiagnosticPublicRelationsV9, config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    route = _replacement_route_mask(arrays["observations"], relations)
    scenes = _decode(np.asarray(arrays["scene_fingerprints"]), "scene fingerprints")
    episodes = _decode(np.asarray(arrays["episode_ids"]), "episode ids")
    selected = np.asarray(fit_mask, dtype=np.bool_)
    if route.shape != selected.shape or scenes.shape != route.shape \
            or episodes.shape != route.shape:
        raise ValueError("V9 replacement support registry differs")
    replacement = config["shared_pickup_replacement"]
    fit = _partition_support(route & selected, scenes, episodes)
    validation = (_partition_support(route & ~selected, scenes, episodes)
                  if np.any(~selected) else None)
    minimum = {
        "rows": replacement["minimum_rows_per_partition"],
        "scenes": replacement["minimum_scenes_per_partition"],
        "episodes": replacement["minimum_episodes_per_partition"],
    }
    for label, support in (("fit", fit), ("validation", validation)):
        if support is None:
            continue
        if any(support[name] < minimum[name] for name in minimum):
            raise ReplacementSupportError(
                "V9 replacement route lacks preregistered " + label + " support")
    return route, {
        "version": REPLACEMENT_VERSION + ".support.v1",
        "route": deepcopy(replacement["route"]),
        "minimum_per_partition": minimum,
        "fit": fit,
        "validation": validation,
        "support_uses_action_labels": False,
        "runtime_identifiers_used": False,
        "protected_final_access": False,
    }


def fit_only_weighted_majority_component(
    relations: R41DiagnosticPublicRelationsV9, labels: np.ndarray,
    weights: np.ndarray, *, binding_sha256: str,
) -> tuple[R41DiagnosticBoostedTreeProgram, dict[str, Any]]:
    """Fit a constant replacement class using only fit-route labels."""
    selected_labels = np.asarray(labels, dtype=np.int64)
    selected_weights = np.asarray(weights, dtype=np.float64)
    if (selected_labels.ndim != 1 or selected_weights.shape != selected_labels.shape
            or not len(selected_labels) or not np.isfinite(selected_weights).all()
            or np.any(selected_weights <= 0)
            or np.any(selected_labels < 0) or np.any(selected_labels >= len(ACTIONS))):
        raise ValueError("V9 replacement fit-only rows differ")
    counts = np.bincount(
        selected_labels, weights=selected_weights, minlength=len(ACTIONS)).astype(np.float64)
    # Add the same fixed pseudocount to every action.  It keeps logits finite
    # while preserving the deterministic weighted-majority class.
    probabilities = (counts + 1e-9) / (float(counts.sum()) + 1e-9 * len(ACTIONS))
    baseline = np.log(probabilities)
    majority = int(np.argmax(counts))
    trees = [{
        "iteration": 0,
        "output_index": output,
        "nodes": [{"kind": "leaf", "value": 0.0}],
    } for output in CLASSES]
    program = R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_TREE_VERSION,
        "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(CLASSES),
        "action_names": list(ACTIONS),
        "output_kind": "multiclass_logits",
        "baseline": [float(value) for value in baseline],
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1,
        "trees": trees,
        "metadata": {
            "purpose": "fit-only public-route replacement explanation specialist",
            "binding_sha256": binding_sha256,
            "fit_source": "fit_only_public_route_rows",
            "estimator": "fit_only_weighted_majority",
            "majority_action": ACTIONS[majority],
            "held_labels_used": False,
            "runtime_action_override": False,
        },
    })
    return program, {
        "fit_rows": len(selected_labels),
        "weighted_action_mass": {
            action: float(counts[index]) for index, action in enumerate(ACTIONS)},
        "majority_action": ACTIONS[majority],
        "held_labels_used": False,
    }


def fit_program(
    arrays: Mapping[str, np.ndarray], fit_mask: np.ndarray, *,
    relations: R41DiagnosticPublicRelationsV9,
    scene_families: Mapping[str, str], config: Mapping[str, Any],
    binding_sha256: str,
) -> tuple[R41DiagnosticPublicTreeProgramV9, dict[str, Any]]:
    """Fit one explicit public HGB program on the supplied development mask."""
    normalized = normalize_config(config)
    weights, pairs, pair_bits, weight_audit = build_fit_weights(
        arrays, fit_mask, scene_families=scene_families, config=normalized)
    indices = np.flatnonzero(fit_mask)
    labels = np.asarray(arrays["action_indices"])[indices]
    if tuple(map(int, np.unique(labels))) != CLASSES:
        raise ValueError("V9 fit rows do not cover all actions")
    expanded = relations.transform_batch(np.asarray(arrays["observations"])[indices])
    settings = normalized["model"]
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
    estimator.fit(expanded, labels, sample_weight=weights[indices])
    base = export_hist_gradient_boosting_classifier(
        estimator, feature_names=relations.feature_names,
        action_names=ACTIONS,
        metadata={
            "version": VERSION,
            "binding_sha256": binding_sha256,
            "fit_config_sha256": digest(normalized),
            "prediction_input": "public observation relations only",
            "actor_logits_used_as_program_input": False,
            "actor_hidden_states_used": False,
            "runtime_action_override": False,
        },
    )
    sample = np.arange(min(4096, len(indices)))
    native = estimator.predict_proba(expanded[sample])
    explicit = base.predict_proba_batch(expanded[sample])
    max_error = float(np.max(np.abs(native - explicit)))
    if max_error > 1e-12 or not np.array_equal(
            np.argmax(native, axis=1), np.argmax(explicit, axis=1)):
        raise RuntimeError("V9 sklearn/explicit tree parity differs")
    dummy = constant_component(relations)
    specialists = {group: dummy for group in GROUPS}
    routes = {
        group: {
            "feature_name": "derived.critical." + group,
            "operator": ">", "threshold": 0.5,
        } for group in GROUPS
    }
    mix_weights = {group: 0.0 for group in GROUPS}
    combinations = {group: "weighted_average" for group in GROUPS}
    replacement_diagnostics: dict[str, Any] = {
        "enabled": False,
        "route": deepcopy(normalized["shared_pickup_replacement"]["route"]),
        "fit_source": "fit_only_public_route_rows",
        "held_labels_used": False,
    }
    if normalized["shared_pickup_replacement"]["enabled"]:
        route_mask, support = _replacement_support(
            arrays, fit_mask, relations, normalized)
        route_indices = np.flatnonzero(route_mask & fit_mask)
        route_labels = np.asarray(arrays["action_indices"])[route_indices].astype(
            np.int64, copy=True)
        replacement_program, majority_audit = fit_only_weighted_majority_component(
            relations, route_labels, weights[route_indices],
            binding_sha256=binding_sha256)
        specialists["shared_pickup"] = replacement_program
        routes["shared_pickup"] = deepcopy(
            normalized["shared_pickup_replacement"]["route"])
        mix_weights["shared_pickup"] = 1.0
        combinations["shared_pickup"] = "replacement"
        replacement_diagnostics = {
            "enabled": True,
            "combination": "replacement",
            "support": support,
            "fit": majority_audit,
            "held_labels_used": False,
            "runtime_action_override": False,
        }
    program = assemble_public_tree_program_v9(
        relations.base_feature_names, base,
        specialists["narrow_passage"], specialists["shared_pickup"],
        specialists["shared_charger"],
        mix_weights=mix_weights, routes=routes, combinations=combinations,
        metadata={
            "version": VERSION,
            "binding_sha256": binding_sha256,
            "fit_config_sha256": digest(normalized),
            "public_feature_contract_sha256": digest(relations.contract()),
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "replacement_specialist_explanation_only": True,
            "formal_ready": False,
        },
    )
    diagnostics = {
        "fit_rows": len(indices),
        "fit_action_counts": {
            action: int(np.sum(labels == index))
            for index, action in enumerate(ACTIONS)
        },
        "model_iterations": base.n_iterations,
        "sklearn_explicit_max_probability_error": max_error,
        "sklearn_explicit_actions_equal": True,
        "fit_weights": weight_audit,
        "fit_pair_count": len(pairs),
        "fit_pair_group_counts": {
            group: int(np.sum((pair_bits & (1 << index)) != 0))
            for index, group in enumerate(GROUPS)
        },
        "shared_pickup_replacement": replacement_diagnostics,
    }
    return program, diagnostics


def predict_in_batches(
    program: R41DiagnosticPublicTreeProgramV9, observations: np.ndarray,
    *, batch_size: int = 16_384,
) -> np.ndarray:
    parts = [program.predict_proba_batch(observations[start:start + batch_size])
             for start in range(0, len(observations), batch_size)]
    return np.concatenate(parts) if parts else np.empty((0, len(ACTIONS)))


__all__ = [
    "VERSION", "CONFIG_VERSION", "REPLACEMENT_VERSION",
    "REPLACEMENT_ROUTE_FEATURE", "ReplacementSupportError", "ACTIONS",
    "CLASSES", "normalize_config", "build_fit_weights", "constant_component",
    "fit_only_weighted_majority_component", "fit_program", "predict_in_batches",
]
