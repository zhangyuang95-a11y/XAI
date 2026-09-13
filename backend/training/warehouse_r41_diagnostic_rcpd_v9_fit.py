"""Development-only fitting primitives for the r4.1 diagnostic v9 program.

Every estimator consumes deterministic public features.  Actor action and
probability fields are fit targets and audit targets only; neither is stored
as a runtime feature.  Whole-scene masks are supplied by the selector so this
module cannot choose or inspect the fresh outer split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
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
CONFIG_VERSION = "warehouse-r41-diagnostic-rcpd-v9-fit-config.v1"
_MODEL_FIELDS = frozenset((
    "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
    "l2_regularization", "max_depth", "max_bins", "random_state",
))
_CONFIG_FIELDS = frozenset((
    "version", "pair_mass_fraction", "duplicate_power", "balance_power",
    "maximum_row_weight", "model",
))


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
        "pair_mass_fraction": number("pair_mass_fraction", low=0.0, high=4.0),
        "duplicate_power": number("duplicate_power", low=0.0, high=1.0),
        "balance_power": number("balance_power", low=0.0, high=1.0),
        "maximum_row_weight": number("maximum_row_weight", low=1.0, high=100.0),
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


def _partner(episode: str) -> str:
    # Stored episode IDs end with the registered public partner name.
    value = episode.rsplit(":", 1)[-1]
    return value if value in v7.PARTNERS else "unknown"


def _balanced_factor(values: np.ndarray, selected: np.ndarray,
                     power: float) -> np.ndarray:
    result = np.ones(len(values), dtype=np.float64)
    chosen = values[selected]
    if not len(chosen) or power == 0.0:
        return result
    counts = Counter(map(str, chosen))
    target = len(chosen) / max(1, len(counts))
    result[selected] = np.asarray([
        (target / counts[str(item)]) ** power for item in chosen
    ], dtype=np.float64)
    return result


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
    episodes = _decode(np.asarray(arrays["episode_ids"]), "episode IDs")
    labels = np.asarray(arrays["action_indices"])
    bits = np.asarray(arrays["group_bits"])
    if (labels.shape != (count,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= len(ACTIONS))
            or bits.shape != (count,) or bits.dtype.kind not in "iu"):
        raise ValueError("V9 fit labels or public critical bits differ")
    if any(scene not in scene_families for scene in set(map(str, scenes[selected]))):
        raise ValueError("V9 scene-family registry is incomplete")

    weights = np.ones(count, dtype=np.float64)
    duplicate_counts = Counter(map(str, hashes[selected]))
    power = normalized["duplicate_power"]
    if power:
        weights[selected] *= np.asarray([
            duplicate_counts[str(value)] ** (-power) for value in hashes[selected]
        ], dtype=np.float64)
    balance_power = normalized["balance_power"]
    families = np.asarray([
        scene_families.get(str(scene), "excluded") for scene in scenes
    ], dtype="U32")
    partners = np.asarray([_partner(str(value)) for value in episodes], dtype="U32")
    exact_bits = bits.astype(str)
    weights *= _balanced_factor(labels.astype(str), selected, balance_power)
    weights *= _balanced_factor(families, selected, balance_power / 2.0)
    weights *= _balanced_factor(partners, selected, balance_power / 2.0)
    weights *= _balanced_factor(exact_bits, selected, balance_power / 2.0)
    weights[~selected] = 0.0
    base_sum = float(math.fsum(map(float, weights[selected])))
    if not math.isfinite(base_sum) or base_sum <= 0.0:
        raise ValueError("V9 base fit weight mass differs")
    weights[selected] *= float(np.sum(selected)) / base_sum

    pairs = v7._effective_pairs(arrays, selected)
    pair_bits = v8._pair_group_bits(arrays, pairs)
    pair_addition = np.zeros(count, dtype=np.float64)
    pair_fraction = normalized["pair_mass_fraction"]
    strata: dict[tuple[str, str, int, str, int], list[tuple[int, int]]] = defaultdict(list)
    for pair_index, (wait_row, branch_row) in enumerate(pairs):
        family = scene_families[str(scenes[wait_row])]
        partner = _partner(str(episodes[wait_row]))
        for endpoint, row in (("wait", int(wait_row)), ("branch", int(branch_row))):
            strata[(family, partner, int(pair_bits[pair_index]), endpoint,
                    int(labels[row]))].append((pair_index, row))
    if pair_fraction and not strata:
        raise ValueError("V9 fit has no effective intervention pairs")
    pair_mass = float(np.sum(selected)) * pair_fraction
    endpoint_share = {"wait": 0.62, "branch": 0.38}
    by_endpoint = {
        endpoint: [key for key in strata if key[3] == endpoint]
        for endpoint in endpoint_share
    }
    for endpoint, share in endpoint_share.items():
        keys = by_endpoint[endpoint]
        if not keys:
            continue
        per_stratum = pair_mass * share / len(keys)
        for key in keys:
            occurrences = strata[key]
            per_occurrence = per_stratum / len(occurrences)
            for _, row in occurrences:
                pair_addition[row] += per_occurrence
    weights += pair_addition
    positive = weights[selected]
    median = float(np.median(positive))
    maximum = normalized["maximum_row_weight"] * max(median, 1e-12)
    clipped_rows = int(np.sum(positive > maximum))
    weights[selected] = np.minimum(positive, maximum)
    weights[~selected] = 0.0
    if (not np.isfinite(weights).all() or np.any(weights[selected] <= 0.0)
            or np.any(weights[~selected] != 0.0)):
        raise ValueError("V9 fit weights are invalid")
    audit = {
        "version": VERSION + ".weights.v1",
        "fit_rows": int(np.sum(selected)),
        "unique_fit_observations": len(duplicate_counts),
        "effective_pair_count": len(pairs),
        "pair_strata": len(strata),
        "base_mass_before_pair_addition": float(np.sum(selected)),
        "requested_pair_mass": pair_mass,
        "actual_pair_addition_before_cap": float(math.fsum(map(float, pair_addition))),
        "final_weight_mass": float(math.fsum(map(float, weights))),
        "clipped_rows": clipped_rows,
        "configuration": deepcopy(normalized),
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
    program = assemble_public_tree_program_v9(
        relations.base_feature_names, base, dummy, dummy, dummy,
        mix_weights={group: 0.0 for group in GROUPS},
        metadata={
            "version": VERSION,
            "binding_sha256": binding_sha256,
            "fit_config_sha256": digest(normalized),
            "public_feature_contract_sha256": digest(relations.contract()),
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
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
    "VERSION", "CONFIG_VERSION", "ACTIONS", "CLASSES", "normalize_config",
    "build_fit_weights", "constant_component", "fit_program",
    "predict_in_batches",
]
