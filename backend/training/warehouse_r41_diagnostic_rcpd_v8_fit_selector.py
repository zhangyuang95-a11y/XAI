"""One-shot, fit-only hyperparameter selection for diagnostic RCPD v8.

This producer is deliberately narrower than the v8 candidate fitter.  It
removes every row outside a caller-frozen fit registry before validating an
Actor label, fitting a model, or computing a metric.  It then freezes one
scene-grouped inner holdout and compares four pre-registered mixtures of one
fixed shared-charger specialist.  No outer-development or final-test metric is
an input to the selection rule.

The selected config is intended for one subsequent, independently bound v8
candidate fit.  This module never evaluates that outer candidate itself.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as weight_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as outer_api
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    assemble_public_tree_program_v8,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-rcpd-v8-fit-selector.v1"
SCOPE_VERSION = "warehouse-r41-diagnostic-rcpd-v8-fit-scope.v1"
STATUS_SELECTED = "passed_fit_only_inner_selection"
STATUS_FAILED = "failed_fit_only_inner_selection"
ROOT = Path(__file__).resolve().parents[2]

MIX_CANDIDATES = (0.0, 0.25, 0.5, 1.0)
INNER_HOLDOUT_SCENES = 64
INNER_HOLDOUT_FAMILY_QUOTAS = {
    "conflict_family_01": 11,
    "conflict_family_02": 11,
    "conflict_family_03": 11,
    "conflict_family_04": 11,
    "conflict_family_05": 10,
    "conflict_family_06": 10,
}
INNER_ORDER_SALT = (
    "r41-diagnostic-v8-fit-only-shared-charger-inner-holdout-20260913"
)
MIN_CHARGER_DIRECTION_IMPROVEMENT = 0.01
MAX_OTHER_METRIC_DEGRADATION = 0.002
CHARGER_MODEL = {
    "learning_rate": 0.1,
    "max_iter": 70,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 10,
    "l2_regularization": 0.1,
    "max_depth": None,
    "max_bins": 255,
    "random_state": 967,
}
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "one-shot fit-only shared-charger config selection",
        "source": {
            "rows": "authenticated diagnostic RCPD v8 development rows",
            "fit_scope": "separately frozen label-blind scene registry",
            "previously_exposed_outer_rows_removed_before_label_validation": True,
            "source_archive_values_decompressed_before_row_projection": True,
            "outer_labels_used_for_projection_fit_or_selection": False,
            "outer_probabilities_used_for_projection_fit_or_selection": False,
            "fresh_outer_binding": (
                "exact identity-only registry/report, used only for exclusion "
                "and provenance; no outer Actor rows exist at selection time"
            ),
            "final_rows_accessed": False,
            "final_labels_accessed": False,
        },
        "inner_split": {
            "unit": "whole scene",
            "order": "ascending sha256(frozen salt, family, fingerprint)",
            "salt": INNER_ORDER_SALT,
            "holdout_scenes": INNER_HOLDOUT_SCENES,
            "family_quotas": deepcopy(INNER_HOLDOUT_FAMILY_QUOTAS),
            "validation_wins_exact_public_observation_overlap": True,
            "episode_or_intervention_pair_crosses_split": False,
        },
        "frozen_model_change": {
            "component": "shared_charger",
            "model": deepcopy(CHARGER_MODEL),
            "mix_candidates": list(MIX_CANDIDATES),
            "base_narrow_pickup_models_unchanged": True,
            "narrow_pickup_mix_weights": 0.0,
            "pair_pool_multiplier_unchanged": True,
            "action_factor_unchanged": True,
        },
        "selection": {
            "all_nine_v8_gates_must_pass": True,
            "minimum_shared_charger_direction_gain_over_zero_mix": (
                MIN_CHARGER_DIRECTION_IMPROVEMENT
            ),
            "maximum_degradation_for_each_other_gate_metric": (
                MAX_OTHER_METRIC_DEGRADATION
            ),
            "tie_break": "smallest shared_charger mix weight",
            "no_candidate_means_stop_before_outer": True,
        },
        "runtime_action_override": False,
        "actor_changed": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return local_source_hashes((
        Path(__file__).resolve(),
        ROOT / "scripts/build_warehouse_r41_diagnostic_rcpd_v8_fit_selector.py",
    ))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _read_json(value: str | Path, label: str) -> dict[str, Any]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    try:
        raw = path.read_bytes()
    except UnicodeDecodeError as exc:
        raise ValueError(label + " must be UTF-8 JSON") from exc
    parsed = _parse_json_bytes(raw, label)
    if file_hash(path) != sha256(raw).hexdigest():
        raise RuntimeError(label + " changed while it was read")
    return parsed


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs, label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(label + " must be UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(label + " must be one JSON object")
    return parsed


def _unique_pairs(pairs: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in " + label)
        result[key] = value
    return result


def _write_json(path: Path, value: Any) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name("." + path.name + ".tmp.npz")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        np.savez_compressed(temporary, **{
            name: np.ascontiguousarray(arrays[name]) for name in sorted(arrays)
        })
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    summary = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        summary[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": sha256(memoryview(value).cast("B")).hexdigest(),
        }
    return digest(summary)


def _load_npz(value: str | Path, label: str) -> dict[str, np.ndarray]:
    # The authenticated source is loaded once, but no label/probability value is
    # inspected until _project_fit_only has physically removed every excluded
    # scene and the resulting NPZ has been re-opened.
    return v8._load_npz(_regular(value, label, maximum=MAX_NPZ_BYTES), label)


def normalize_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "version", "source_report_sha256", "source_rows_sha256",
        "eligible_fit_scene_fingerprints", "inner_candidate_scenes",
        "previously_exposed_outer_scene_fingerprints",
        "fresh_outer_scene_fingerprints", "fresh_outer_registry_file_sha256",
        "fresh_outer_registry_content_sha256", "fresh_outer_report_file_sha256",
        "fresh_outer_report_content_sha256", "label_blind",
        "final_rows_accessed", "final_labels_accessed", "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Fit-only selector scope schema differs")
    if value.get("version") != SCOPE_VERSION:
        raise ValueError("Fit-only selector scope version differs")
    expected_content = digest({
        key: item for key, item in value.items() if key != "content_sha256"
    })
    if (value.get("content_sha256") != expected_content
            or value.get("label_blind") is not True
            or value.get("final_rows_accessed") is not False
            or value.get("final_labels_accessed") is not False):
        raise ValueError("Fit-only selector scope assurance differs")
    source_report = _sha(value.get("source_report_sha256"), "source report")
    source_rows = _sha(value.get("source_rows_sha256"), "source rows")
    fresh_registry_file = _sha(
        value.get("fresh_outer_registry_file_sha256"), "fresh outer registry")
    fresh_registry_content = _sha(
        value.get("fresh_outer_registry_content_sha256"),
        "fresh outer registry content")
    fresh_report_file = _sha(
        value.get("fresh_outer_report_file_sha256"), "fresh outer report")
    fresh_report_content = _sha(
        value.get("fresh_outer_report_content_sha256"),
        "fresh outer report content")

    def hashes(raw: Any, label: str) -> list[str]:
        if (not isinstance(raw, list)
                or any(type(item) is not str or _HEX.fullmatch(item) is None
                       for item in raw)
                or len(set(raw)) != len(raw)):
            raise ValueError(label + " must be unique SHA-256 fingerprints")
        return list(raw)

    eligible = hashes(
        value.get("eligible_fit_scene_fingerprints"), "eligible fit scenes")
    exposed = hashes(
        value.get("previously_exposed_outer_scene_fingerprints"),
        "previously exposed outer scenes")
    fresh = hashes(
        value.get("fresh_outer_scene_fingerprints"), "fresh outer scenes")
    raw_candidates = value.get("inner_candidate_scenes")
    if not isinstance(raw_candidates, list):
        raise ValueError("inner_candidate_scenes must be a list")
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        if (not isinstance(raw, Mapping)
                or set(raw) != {"fingerprint", "family_id"}
                or type(raw.get("fingerprint")) is not str
                or _HEX.fullmatch(raw["fingerprint"]) is None
                or type(raw.get("family_id")) is not str
                or raw["family_id"] not in INNER_HOLDOUT_FAMILY_QUOTAS
                or raw["fingerprint"] in seen):
            raise ValueError("inner_candidate_scenes identity differs")
        seen.add(raw["fingerprint"])
        candidates.append({
            "fingerprint": raw["fingerprint"],
            "family_id": raw["family_id"],
        })
    eligible_set = set(eligible)
    exposed_set = set(exposed)
    fresh_set = set(fresh)
    if (len(fresh) != outer_api.FRESH_OUTER_SCENE_COUNT
            or not eligible_set or exposed_set & fresh_set
            or eligible_set & exposed_set or eligible_set & fresh_set
            or seen != eligible_set):
        raise ValueError(
            "Fit-only selector scope scene populations overlap or differ")
    return {
        "version": SCOPE_VERSION,
        "source_report_sha256": source_report,
        "source_rows_sha256": source_rows,
        "eligible_fit_scene_fingerprints": eligible,
        "inner_candidate_scenes": candidates,
        "previously_exposed_outer_scene_fingerprints": exposed,
        "fresh_outer_scene_fingerprints": fresh,
        "fresh_outer_registry_file_sha256": fresh_registry_file,
        "fresh_outer_registry_content_sha256": fresh_registry_content,
        "fresh_outer_report_file_sha256": fresh_report_file,
        "fresh_outer_report_content_sha256": fresh_report_content,
        "label_blind": True,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "content_sha256": expected_content,
    }


def candidate_configs(source_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = v8.normalize_config(source_config)
    if (source["pair_pool_multiplier"] != 16.0
            or source["use_action_factor"] is not True
            or any(source["mix_weights"][group] != 0.0 for group in v8.GROUPS)):
        raise ValueError("Fit-only selector requires the frozen v8 base config")
    configs = []
    for weight in MIX_CANDIDATES:
        item = deepcopy(source)
        item["models"]["shared_charger"] = deepcopy(CHARGER_MODEL)
        item["mix_weights"] = {
            "narrow_passage": 0.0,
            "shared_pickup": 0.0,
            "shared_charger": weight,
        }
        configs.append(v8.normalize_config(item))
    return configs


def _inner_holdout(
    candidate_scenes: Sequence[Mapping[str, str]],
    *, quotas: Mapping[str, int] = INNER_HOLDOUT_FAMILY_QUOTAS,
    salt: str = INNER_ORDER_SALT,
) -> tuple[list[str], dict[str, int]]:
    if (not isinstance(salt, str) or not salt
            or any(type(value) is not int or value < 1 for value in quotas.values())):
        raise ValueError("Inner holdout protocol differs")
    by_family: dict[str, list[str]] = {family: [] for family in quotas}
    seen: set[str] = set()
    for row in candidate_scenes:
        fingerprint = str(row["fingerprint"])
        family = str(row["family_id"])
        if (family not in by_family or _HEX.fullmatch(fingerprint) is None
                or fingerprint in seen):
            raise ValueError("Inner holdout scene registry differs")
        seen.add(fingerprint)
        by_family[family].append(fingerprint)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for family, quota in quotas.items():
        ordered = sorted(by_family[family], key=lambda fingerprint: digest({
            "salt": salt, "family_id": family, "fingerprint": fingerprint,
        }))
        if len(ordered) < quota:
            raise ValueError("Inner holdout family quota unavailable: " + family)
        chosen = ordered[:quota]
        selected.extend(chosen)
        counts[family] = len(chosen)
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("Inner holdout size differs")
    return selected, counts


def _project_fit_only(
    source: Mapping[str, np.ndarray], scope: Mapping[str, Any],
    *, quotas: Mapping[str, int] = INNER_HOLDOUT_FAMILY_QUOTAS,
    salt: str = INNER_ORDER_SALT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Copy only frozen fit scenes, then create the inner validation split.

    The function intentionally examines only split/scene/observation identities
    until excluded rows are gone.  Action labels and Actor probabilities are
    copied by row index but never branched on or validated here.
    """
    if set(source) != v8._ROW_FIELDS:
        raise ValueError("Fit-only source row fields differ")
    count = len(source["split_validation"])
    if (source["split_validation"].shape != (count,)
            or source["split_validation"].dtype != np.dtype(np.bool_)
            or source["scene_fingerprints"].shape != (count,)
            or source["scene_fingerprints"].dtype != np.dtype("S64")
            or source["observation_hashes"].shape != (count,)
            or source["observation_hashes"].dtype != np.dtype("S64")
            or any(np.asarray(value).shape[0] != count for value in source.values())):
        raise ValueError("Fit-only source identity arrays differ")
    normalized = normalize_scope(scope)
    scenes = v8._decode(source["scene_fingerprints"], "Fit-only source scenes")
    source_scene_set = set(map(str, scenes))
    old_outer = set(map(str, scenes[source["split_validation"]]))
    expected_old_outer = set(
        normalized["previously_exposed_outer_scene_fingerprints"])
    fresh_outer = set(normalized["fresh_outer_scene_fingerprints"])
    eligible = set(normalized["eligible_fit_scene_fingerprints"])
    if (old_outer != expected_old_outer
            or fresh_outer & source_scene_set
            or eligible != source_scene_set - old_outer
            or eligible & old_outer or eligible & fresh_outer):
        raise ValueError("Fit-only source/scope scene identity differs")

    inner_scenes, family_counts = _inner_holdout(
        normalized["inner_candidate_scenes"], quotas=quotas, salt=salt)
    inner_set = set(inner_scenes)
    if not inner_set.issubset(eligible):
        raise ValueError("Inner holdout contains a non-fit scene")

    eligible_rows = np.isin(scenes, list(eligible))
    indices = np.flatnonzero(eligible_rows)
    projected = {
        name: np.asarray(value)[indices].copy() for name, value in source.items()
    }
    projected_scenes = v8._decode(
        projected["scene_fingerprints"], "Projected fit scenes")
    inner_mask = np.isin(projected_scenes, list(inner_set))
    projected["split_validation"] = inner_mask.astype(np.bool_)

    validation_hashes = set(map(
        bytes, projected["observation_hashes"][projected["split_validation"]]))
    keep = np.asarray([
        bool(validation) or bytes(observation_hash) not in validation_hashes
        for observation_hash, validation in zip(
            projected["observation_hashes"], projected["split_validation"])
    ], dtype=np.bool_)
    removed_overlap = int(np.sum(~keep))
    projected = {
        name: np.asarray(value)[keep].copy() for name, value in projected.items()
    }
    projected_scenes = v8._decode(
        projected["scene_fingerprints"], "Deduplicated projected scenes")
    split = projected["split_validation"]
    fit_scenes = set(map(str, projected_scenes[~split]))
    validation_scenes = set(map(str, projected_scenes[split]))
    episodes = v8._decode(projected["episode_ids"], "Projected fit episodes")
    anchors = v8._decode(projected["anchor_ids"], "Projected fit anchors")
    fit_episodes = set(map(str, episodes[~split]))
    validation_episodes = set(map(str, episodes[split]))
    fit_anchors = {str(value) for value in anchors[~split] if str(value)}
    validation_anchors = {str(value) for value in anchors[split] if str(value)}
    if (validation_scenes != inner_set
            or fit_scenes != eligible - inner_set
            or fit_episodes & validation_episodes
            or fit_anchors & validation_anchors
            or set(map(bytes, projected["observation_hashes"][~split]))
                & set(map(bytes, projected["observation_hashes"][split]))
            or set(map(str, projected_scenes)) & (old_outer | fresh_outer)):
        raise ValueError("Fit-only projection isolation differs")
    audit = {
        "source_rows": count,
        "source_scenes": len(source_scene_set),
        "previously_exposed_outer_scenes_removed": len(old_outer),
        "fresh_outer_scenes_registered_absent_from_source": len(fresh_outer),
        "eligible_fit_scenes": len(eligible),
        "inner_fit_scenes": len(fit_scenes),
        "inner_holdout_scenes": len(validation_scenes),
        "inner_holdout_family_counts": family_counts,
        "inner_fit_rows": int(np.sum(~split)),
        "inner_holdout_rows": int(np.sum(split)),
        "inner_fit_rows_removed_for_exact_holdout_overlap": removed_overlap,
        "scene_identity_overlap": 0,
        "episode_identity_overlap": 0,
        "nonempty_anchor_identity_overlap": 0,
        "exact_observation_overlap": 0,
        "outer_labels_or_probabilities_used_for_projection": False,
    }
    return projected, audit


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "overall": float(metrics["overall"]["fidelity"]),
        "nonwait": float(metrics["nonwait"]["fidelity"]),
        "direction_overall": float(
            metrics["effective_intervention_direction"]["fidelity"]),
    }
    for group in v8.GROUPS:
        result["critical_" + group] = float(
            metrics["critical"][group]["fidelity"])
        result["direction_" + group] = float(
            metrics["effective_intervention_direction"]["by_group"][group][
                "fidelity"])
    return result


def choose_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (len(candidates) != len(MIX_CANDIDATES)
            or tuple(float(row.get("mix_weight", -1.0)) for row in candidates)
                != MIX_CANDIDATES):
        raise ValueError("Fit-only candidate registry differs")
    baseline_values = _metric_vector(candidates[0]["metrics"])
    charger_key = "direction_shared_charger"
    decisions = []
    qualified = []
    for row in candidates:
        weight = float(row["mix_weight"])
        values = _metric_vector(row["metrics"])
        gate = v8._gate(row["metrics"])
        gain = values[charger_key] - baseline_values[charger_key]
        degradations = {
            name: baseline_values[name] - value
            for name, value in values.items() if name != charger_key
        }
        checks = {
            "all_nine_v8_gates": bool(gate["passed"]),
            "minimum_charger_direction_gain": (
                gain + 1e-15 >= MIN_CHARGER_DIRECTION_IMPROVEMENT),
            "other_metrics_within_degradation_limit": all(
                amount <= MAX_OTHER_METRIC_DEGRADATION + 1e-15
                for amount in degradations.values()
            ),
        }
        eligible = all(checks.values())
        decision = {
            "mix_weight": weight,
            "gate": gate,
            "metric_values": values,
            "shared_charger_direction_gain": gain,
            "other_metric_degradations": degradations,
            "selection_checks": checks,
            "eligible": eligible,
        }
        decisions.append(decision)
        if eligible:
            qualified.append(decision)
    selected = min(qualified, key=lambda row: row["mix_weight"]) \
        if qualified else None
    return {
        "status": STATUS_SELECTED if selected is not None else STATUS_FAILED,
        "baseline_mix_weight": 0.0,
        "candidate_count": len(decisions),
        "candidates": decisions,
        "selected_mix_weight": (
            selected["mix_weight"] if selected is not None else None),
        "selection_rule": deepcopy(contract()["selection"]),
        "outer_evaluation_performed": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }


def _candidate_probabilities(
    fitted_program: Any, observations: np.ndarray,
    source_probabilities: np.ndarray, validation_mask: np.ndarray,
    fit_config: Mapping[str, Any], mix_weight: float,
) -> np.ndarray:
    config = deepcopy(fit_config)
    config["mix_weights"] = {
        "narrow_passage": 0.0,
        "shared_pickup": 0.0,
        "shared_charger": float(mix_weight),
    }
    wrapper = assemble_public_tree_program_v8(
        fitted_program.base_feature_names,
        fitted_program.base_program,
        fitted_program._specialist_programs["narrow_passage"],
        fitted_program._specialist_programs["shared_pickup"],
        fitted_program._specialist_programs["shared_charger"],
        mix_weights=config["mix_weights"], routes=v8._fixed_routes(),
        metadata=deepcopy(fitted_program.metadata),
    )
    result = np.asarray(source_probabilities, dtype=np.float64).copy()
    result[validation_mask] = v8._predict_in_batches(
        wrapper, observations[validation_mask])
    return result


def _select_projected(
    arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
    source_config: Mapping[str, Any], source_identity: Mapping[str, Any],
    selector_binding: str, scene_families: Mapping[str, str],
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    # This is the first operation allowed to inspect projected labels/probabilities.
    v8._validate_base_shapes(arrays, actor, "Physical fit-only selector rows")
    relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
    configs = candidate_configs(source_config)
    fit_config = deepcopy(configs[0])
    weight_result = weight_api.build_pair_weights(
        arrays,
        scene_families=scene_families,
        use_action_factor=fit_config["use_action_factor"],
        pair_pool_multiplier=fit_config["pair_pool_multiplier"],
    )
    mutable = {name: np.asarray(value).copy() for name, value in arrays.items()}
    mutable["weights"] = weight_result["weights"].copy()
    fit_pair_bits = v8._pair_group_bits(mutable, weight_result["pairs"])
    program, diagnostics = v8._fit_program(
        mutable, relations=relations, weights=weight_result,
        config=fit_config, binding=selector_binding,
        source_identity=source_identity, pair_group_bits=fit_pair_bits,
    )
    validation = mutable["split_validation"]
    validation_pairs = v7._effective_pairs(mutable, validation)
    validation_pair_bits = v8._pair_group_bits(mutable, validation_pairs)
    scored: list[dict[str, Any]] = []
    for mix, config in zip(MIX_CANDIDATES, configs):
        probabilities = _candidate_probabilities(
            program, mutable["observations"], mutable["probabilities"],
            validation, fit_config, mix)
        metrics = v8._metrics_from_probabilities(
            probabilities, mutable, validation,
            pairs=validation_pairs, pair_group_bits=validation_pair_bits,
        )
        scored.append({
            "mix_weight": mix,
            "config_sha256": digest(config),
            "metrics": metrics,
        })
    result = choose_candidate(scored)
    result.update({
        "fit_diagnostics": diagnostics,
        "weight_audit": weight_result["audit"],
        "fit_pair_group_bits_sha256": _arrays_digest({
            "pair_group_bits": fit_pair_bits,
        }),
        "validation_pair_group_bits_sha256": _arrays_digest({
            "pair_group_bits": validation_pair_bits,
        }),
    })
    return result, program, configs


def _validate_source_evidence(
    directory: Path, expected_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    report_path = _regular(
        directory / "report.json", "Source v8 report", maximum=MAX_JSON_BYTES)
    if file_hash(report_path) != _sha(expected_report_sha256, "source report"):
        raise ValueError("Source v8 report hash differs")
    report = _read_json(report_path, "Source v8 report")
    artifacts = report.get("evidence_artifacts")
    if (report.get("version") != v8.VERSION
            or report.get("execution", {}).get("final_rows_accessed") is not False
            or report.get("execution", {}).get("final_labels_accessed") is not False
            or not isinstance(artifacts, Mapping)):
        raise ValueError("Source v8 evidence assurance differs")
    required = (
        "rows.npz", "fit_config.json", "program.json", "weights_audit.json",
    )
    paths = {"report": report_path}
    for name in required:
        path = _regular(
            directory / name, "Source v8 " + name,
            maximum=MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES)
        if artifacts.get(name) != file_hash(path):
            raise ValueError("Source v8 evidence artifact differs: " + name)
        paths[name] = path
    return report, paths


def _snapshot_build_inputs(
    *, source_evidence: str | Path, expected_source_report_sha256: str,
    actor_path: str | Path, fit_scope_path: str | Path,
    expected_fit_scope_sha256: str, fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
) -> ImmutableInputSnapshot:
    """Freeze every build input from one authenticated no-follow read."""
    source_directory = Path(source_evidence).expanduser().absolute()
    if (not source_directory.is_dir() or source_directory.is_symlink()
            or source_directory.resolve() != source_directory):
        raise ValueError("Source v8 evidence directory is unsafe")
    report_path = source_directory / "report.json"
    report_sha256 = _sha(expected_source_report_sha256, "source report")
    report = _parse_json_bytes(read_authenticated_bytes(
        report_path, label="Source v8 report",
        expected_sha256=report_sha256, maximum=MAX_JSON_BYTES,
    ), "Source v8 report")
    artifacts = report.get("evidence_artifacts")
    required = (
        "rows.npz", "fit_config.json", "program.json", "weights_audit.json",
    )
    if (report.get("version") != v8.VERSION
            or not isinstance(artifacts, Mapping)
            or any(_HEX.fullmatch(str(artifacts.get(name))) is None
                   for name in required)):
        raise ValueError("Source v8 snapshot evidence differs")
    actor_sha256 = _sha(
        report.get("bindings", {}).get("actor_file_sha256"), "source Actor")
    scope_sha256 = _sha(expected_fit_scope_sha256, "fit scope")
    originals = {
        "source_report": report_path,
        "source_rows": source_directory / "rows.npz",
        "source_config": source_directory / "fit_config.json",
        "source_program": source_directory / "program.json",
        "source_weights_audit": source_directory / "weights_audit.json",
        "actor": actor_path,
        "fit_scope": fit_scope_path,
        "fresh_outer_registry": fresh_outer_registry_path,
        "fresh_outer_report": fresh_outer_report_path,
    }
    expected = {
        "source_report": report_sha256,
        "source_rows": str(artifacts["rows.npz"]),
        "source_config": str(artifacts["fit_config.json"]),
        "source_program": str(artifacts["program.json"]),
        "source_weights_audit": str(artifacts["weights_audit.json"]),
        "actor": actor_sha256,
        "fit_scope": scope_sha256,
        "fresh_outer_registry": _sha(
            expected_fresh_outer_registry_sha256, "fresh outer registry"),
        "fresh_outer_report": _sha(
            expected_fresh_outer_report_sha256, "fresh outer report"),
    }
    return ImmutableInputSnapshot(
        originals,
        expected_sha256=expected,
        relative_names={
            "source_report": "source/report.json",
            "source_rows": "source/rows.npz",
            "source_config": "source/fit_config.json",
            "source_program": "source/program.json",
            "source_weights_audit": "source/weights_audit.json",
            "actor": "actor/actor.npz",
            "fit_scope": "scope/fit_scope.json",
            "fresh_outer_registry": "fresh_outer/development_expansion.json",
            "fresh_outer_report": "fresh_outer/report.json",
        },
        maximum_bytes={
            "source_report": MAX_JSON_BYTES,
            "source_rows": MAX_NPZ_BYTES,
            "source_config": MAX_JSON_BYTES,
            "source_program": MAX_JSON_BYTES,
            "source_weights_audit": MAX_JSON_BYTES,
            "actor": MAX_NPZ_BYTES,
            "fit_scope": MAX_JSON_BYTES,
            "fresh_outer_registry": MAX_JSON_BYTES,
            "fresh_outer_report": MAX_JSON_BYTES,
        },
        prefix="warehouse-r41-v8-fit-selector-inputs-",
    )


def _validate_fresh_outer_binding(
    registry: Mapping[str, Any], report: Mapping[str, Any], *,
    registry_file_sha256: str, report_file_sha256: str,
    source_rows_sha256: str,
) -> set[str]:
    """Authenticate the identity-only outer registry without Actor evidence."""
    registry_sha = _sha(registry_file_sha256, "fresh outer registry")
    report_sha = _sha(report_file_sha256, "fresh outer report")
    registry_content = digest({
        key: value for key, value in registry.items() if key != "content_sha256"
    })
    report_content = digest({
        key: value for key, value in report.items() if key != "content_sha256"
    })
    boundaries = registry.get("information_boundary")
    scenes = registry.get("development_validation")
    identities = registry.get("selected_outer_identities")
    selection = report.get("selection")
    if (registry.get("version") != outer_api.VERSION
            or registry.get("status") != outer_api.STATUS
            or registry.get("contract") != outer_api.contract()
            or registry.get("content_sha256") != registry_content
            or report.get("version") != outer_api.REPORT_VERSION
            or report.get("status") != outer_api.STATUS
            or report.get("content_sha256") != report_content
            or report.get("registry_file_sha256") != registry_sha
            or report.get("registry_content_sha256") != registry_content
            or report.get("bindings") != registry.get("bindings")
            or report.get("producer_sources") != registry.get("producer_sources")
            or registry.get("producer_sources") != outer_api.producer_sources()
            or registry.get("producer_sources_sha256")
                != digest(registry.get("producer_sources"))
            or report.get("producer_sources_sha256")
                != registry.get("producer_sources_sha256")
            or report.get("information_boundary") != boundaries
            or report.get("statistics") != registry.get("statistics")
            or registry.get("formal_ready") is not False
            or report.get("formal_ready") is not False
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or registry.get("final_labels_used_for_selection") is not False
            or registry.get("runtime_action_override") is not False
            or not isinstance(boundaries, Mapping)
            or boundaries.get("outer_actor_rows_collected") is not False
            or boundaries.get("outer_candidate_scored") is not False
            or boundaries.get("actor_loaded_or_inferred") is not False
            or boundaries.get("observations_generated") is not False
            or boundaries.get("protected_final_access") is not False
            or registry.get("bindings", {}).get("source_rows_file_sha256")
                != _sha(source_rows_sha256, "source rows")
            or not isinstance(scenes, list)
            or len(scenes) != outer_api.FRESH_OUTER_SCENE_COUNT
            or not isinstance(identities, list)
            or len(identities) != outer_api.FRESH_OUTER_SCENE_COUNT
            or not isinstance(selection, Mapping)
            or selection.get("salt") != outer_api.SELECTION_SALT
            or selection.get("family_quotas") != outer_api.FAMILY_QUOTAS):
        raise ValueError("Fresh outer identity-only registry/report differs")
    identity_keys = {"batch_index", "family_id", "seed", "fingerprint"}
    scene_identity = [
        {key: row.get(key) for key in identity_keys}
        for row in scenes if isinstance(row, Mapping)
    ]
    if (len(scene_identity) != len(scenes)
            or scene_identity != identities
            or any(not isinstance(row, Mapping) or set(row) != identity_keys
                   for row in identities)
            or any(row.get("split") != "development_validation" for row in scenes)
            or Counter(row.get("family_id") for row in scenes)
                != Counter(outer_api.FAMILY_QUOTAS)):
        raise ValueError("Fresh outer scene identity or family registry differs")
    fingerprints = [row.get("fingerprint") for row in identities]
    if (any(type(value) is not str or _HEX.fullmatch(value) is None
            for value in fingerprints)
            or len(set(fingerprints)) != len(fingerprints)
            or selection.get("selected_identity_sha256")
                != digest(identities)):
        raise ValueError("Fresh outer identity selection binding differs")
    # report_sha is intentionally consumed above as an exact caller binding;
    # its bytes are frozen by ImmutableInputSnapshot before this function runs.
    if not report_sha:
        raise RuntimeError("Unreachable empty report SHA")
    return set(map(str, fingerprints))


def _scene_families_from_weight_audit(
    value: Mapping[str, Any], *, expected_scenes: set[str],
) -> dict[str, str]:
    """Recover the already-authenticated public scene-family registry.

    Only the scene and family strings are projected from the prior weight audit;
    masses, action factors, labels, and pair statistics are not inputs.
    """
    try:
        rows = value["balance"]["combined_training"]["scene_totals"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Source v8 weight-audit family registry is missing") from exc
    if not isinstance(rows, list):
        raise ValueError("Source v8 weight-audit family registry differs")
    result: dict[str, str] = {}
    for row in rows:
        if (not isinstance(row, Mapping)
                or type(row.get("scene")) is not str
                or _HEX.fullmatch(row["scene"]) is None
                or type(row.get("family")) is not str
                or row["family"] not in INNER_HOLDOUT_FAMILY_QUOTAS):
            raise ValueError("Source v8 weight-audit scene family differs")
        previous = result.setdefault(row["scene"], row["family"])
        if previous != row["family"]:
            raise ValueError("Source v8 weight audit assigns two scene families")
    if set(result) != expected_scenes:
        raise ValueError("Source v8 weight-audit fit scene population differs")
    return result


def _validate_scope_family_registry(
    scope: Mapping[str, Any], *, authenticated_families: Mapping[str, str],
) -> dict[str, str]:
    """Require the inner split's family labels to equal authenticated labels."""
    normalized = normalize_scope(scope)
    scoped = {
        str(row["fingerprint"]): str(row["family_id"])
        for row in normalized["inner_candidate_scenes"]
    }
    actual = {
        str(fingerprint): str(family)
        for fingerprint, family in authenticated_families.items()
    }
    if scoped != actual:
        raise ValueError(
            "Fit-only scope family registry differs from authenticated v8 audit")
    return actual


def prepare_scope(
    *, source_evidence: str | Path, expected_source_report_sha256: str,
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze the label-blind fit population and six-family inner registry."""
    source_directory = Path(source_evidence).expanduser().absolute()
    if (not source_directory.is_dir() or source_directory.is_symlink()
            or source_directory.resolve() != source_directory):
        raise ValueError("Source v8 evidence directory is unsafe")
    _, paths = _validate_source_evidence(
        source_directory, expected_source_report_sha256)
    # Access only the two identity arrays.  In particular, action_indices and
    # probabilities are never decompressed by this scope producer.
    with np.load(paths["rows.npz"], allow_pickle=False) as archive:
        if set(archive.files) != v8._ROW_FIELDS:
            raise ValueError("Source v8 row archive schema differs")
        split = archive["split_validation"].copy()
        raw_scenes = archive["scene_fingerprints"].copy()
    if (split.ndim != 1 or split.dtype != np.dtype(np.bool_)
            or raw_scenes.shape != split.shape
            or raw_scenes.dtype != np.dtype("S64")):
        raise ValueError("Source v8 scope identity arrays differ")
    scenes = v8._decode(raw_scenes, "Source v8 scope scenes")
    eligible = set(map(str, scenes[~split]))
    exposed = set(map(str, scenes[split]))
    registry_file = _regular(
        fresh_outer_registry_path, "Fresh outer registry", maximum=MAX_JSON_BYTES)
    report_file = _regular(
        fresh_outer_report_path, "Fresh outer report", maximum=MAX_JSON_BYTES)
    registry_sha256 = _sha(
        expected_fresh_outer_registry_sha256, "fresh outer registry")
    report_sha256 = _sha(
        expected_fresh_outer_report_sha256, "fresh outer report")
    if (file_hash(registry_file) != registry_sha256
            or file_hash(report_file) != report_sha256):
        raise ValueError("Exact fresh outer registry/report bytes required")
    registry = _read_json(registry_file, "Fresh outer registry")
    outer_report = _read_json(report_file, "Fresh outer report")
    fresh = sorted(_validate_fresh_outer_binding(
        registry, outer_report,
        registry_file_sha256=registry_sha256,
        report_file_sha256=report_sha256,
        source_rows_sha256=file_hash(paths["rows.npz"]),
    ))
    if set(fresh) & set(map(str, scenes)):
        raise ValueError("Fresh outer registry overlaps source v8 rows")
    audit = _read_json(paths["weights_audit.json"], "Source v8 weight audit")
    families = _scene_families_from_weight_audit(
        audit, expected_scenes=eligible)
    payload: dict[str, Any] = {
        "version": SCOPE_VERSION,
        "source_report_sha256": expected_source_report_sha256,
        "source_rows_sha256": file_hash(paths["rows.npz"]),
        "eligible_fit_scene_fingerprints": sorted(eligible),
        "inner_candidate_scenes": [
            {"fingerprint": fingerprint, "family_id": families[fingerprint]}
            for fingerprint in sorted(eligible)
        ],
        "previously_exposed_outer_scene_fingerprints": sorted(exposed),
        "fresh_outer_scene_fingerprints": fresh,
        "fresh_outer_registry_file_sha256": registry_sha256,
        "fresh_outer_registry_content_sha256": registry["content_sha256"],
        "fresh_outer_report_file_sha256": report_sha256,
        "fresh_outer_report_content_sha256": outer_report["content_sha256"],
        "label_blind": True,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }
    payload["content_sha256"] = digest(payload)
    normalized = normalize_scope(payload)
    destination = Path(output).expanduser().absolute()
    if (not destination.parent.is_dir() or destination.parent.is_symlink()
            or destination.parent.resolve() != destination.parent
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Fit-only scope destination is unsafe or exists")
    _write_json(destination, normalized)
    return normalized


def _build_frozen(
    *, source_evidence: str | Path, expected_source_report_sha256: str,
    actor_path: str | Path, fit_scope_path: str | Path,
    expected_fit_scope_sha256: str,
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
    output: str | Path,
    expected_sources: Mapping[str, str], prepublish: Callable[[], None],
) -> dict[str, Any]:
    sources = producer_sources()
    if sources != dict(expected_sources):
        raise RuntimeError("Fit-only selector sources changed before build")
    source_directory = Path(source_evidence).expanduser().absolute()
    if (not source_directory.is_dir() or source_directory.is_symlink()
            or source_directory.resolve() != source_directory):
        raise ValueError("Source v8 evidence directory is unsafe")
    source_report, paths = _validate_source_evidence(
        source_directory, expected_source_report_sha256)
    actor_file = _regular(actor_path, "Frozen Actor", maximum=MAX_NPZ_BYTES)
    if file_hash(actor_file) != source_report.get("bindings", {}).get(
            "actor_file_sha256"):
        raise ValueError("Selector Actor differs from source v8 evidence")
    scope_file = _regular(
        fit_scope_path, "Fit-only selector scope", maximum=MAX_JSON_BYTES)
    if file_hash(scope_file) != _sha(expected_fit_scope_sha256, "fit scope"):
        raise ValueError("Fit-only selector scope hash differs")
    scope = normalize_scope(_read_json(scope_file, "Fit-only selector scope"))
    if (scope["source_report_sha256"] != expected_source_report_sha256
            or scope["source_rows_sha256"] != file_hash(paths["rows.npz"])):
        raise ValueError("Fit-only scope/source v8 binding differs")
    fresh_registry_file = _regular(
        fresh_outer_registry_path, "Fresh outer registry", maximum=MAX_JSON_BYTES)
    fresh_report_file = _regular(
        fresh_outer_report_path, "Fresh outer report", maximum=MAX_JSON_BYTES)
    fresh_registry_sha256 = _sha(
        expected_fresh_outer_registry_sha256, "fresh outer registry")
    fresh_report_sha256 = _sha(
        expected_fresh_outer_report_sha256, "fresh outer report")
    if (file_hash(fresh_registry_file) != fresh_registry_sha256
            or file_hash(fresh_report_file) != fresh_report_sha256
            or scope["fresh_outer_registry_file_sha256"]
                != fresh_registry_sha256
            or scope["fresh_outer_report_file_sha256"] != fresh_report_sha256):
        raise ValueError("Fit-only scope/fresh outer file binding differs")
    fresh_registry = _read_json(
        fresh_registry_file, "Fresh outer registry")
    fresh_report = _read_json(fresh_report_file, "Fresh outer report")
    fresh_fingerprints = _validate_fresh_outer_binding(
        fresh_registry, fresh_report,
        registry_file_sha256=fresh_registry_sha256,
        report_file_sha256=fresh_report_sha256,
        source_rows_sha256=file_hash(paths["rows.npz"]),
    )
    if (fresh_fingerprints != set(scope["fresh_outer_scene_fingerprints"])
            or scope["fresh_outer_registry_content_sha256"]
                != fresh_registry["content_sha256"]
            or scope["fresh_outer_report_content_sha256"]
                != fresh_report["content_sha256"]):
        raise ValueError("Fit-only scope/fresh outer identity binding differs")
    source_config = v8.normalize_config(_read_json(
        paths["fit_config.json"], "Source v8 fit config"))
    configs = candidate_configs(source_config)
    source_weight_audit = _read_json(
        paths["weights_audit.json"], "Source v8 weight audit")
    all_source_fit_scenes = set(scope["eligible_fit_scene_fingerprints"])
    all_scene_families = _scene_families_from_weight_audit(
        source_weight_audit, expected_scenes=all_source_fit_scenes)
    _validate_scope_family_registry(
        scope, authenticated_families=all_scene_families)

    destination = Path(output).expanduser().absolute()
    parent = destination.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Fit-only selector destination is unsafe or exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        source_arrays = _load_npz(paths["rows.npz"], "Source v8 rows")
        projected, projection_audit = _project_fit_only(source_arrays, scope)
        del source_arrays
        fit_only_path = temporary / "fit_only_rows.npz"
        _write_npz(fit_only_path, projected)
        del projected
        # All fitting receives a physically separate archive containing no old
        # or fresh outer scene.
        fit_only = _load_npz(fit_only_path, "Physical fit-only rows")
        actor = NumPyNativeActor(actor_file)
        source_identity = v8._program_source_identity(source_report["bindings"])
        fit_scene_set = set(map(str, v8._decode(
            fit_only["scene_fingerprints"], "Physical fit-only scenes")))
        scene_families = {
            scene: all_scene_families[scene] for scene in fit_scene_set
        }
        selector_binding = digest({
            "version": VERSION,
            "contract_sha256": digest(contract()),
            "source_report_sha256": expected_source_report_sha256,
            "fit_scope_sha256": expected_fit_scope_sha256,
            "fit_only_rows_semantic_sha256": _arrays_digest(fit_only),
            "config_registry_sha256": digest(configs),
            "producer_sources_sha256": digest(sources),
        })
        selection, fitted_program, configs = _select_projected(
            fit_only, actor=actor, source_config=source_config,
            source_identity=source_identity, selector_binding=selector_binding,
            scene_families=scene_families)
        selected_weight = selection["selected_mix_weight"]
        selected_config = next(
            (config for mix, config in zip(MIX_CANDIDATES, configs)
             if mix == selected_weight), None)
        config_registry = {
            "version": VERSION,
            "candidate_count": len(configs),
            "candidates": [
                {"mix_weight": mix, "config": config,
                 "config_sha256": digest(config)}
                for mix, config in zip(MIX_CANDIDATES, configs)
            ],
            "selection_fields": ["models.shared_charger", "mix_weights.shared_charger"],
            "all_other_fields_frozen": True,
        }
        selected_payload = {
            "version": VERSION,
            "status": selection["status"],
            "selected_mix_weight": selected_weight,
            "selected_config": selected_config,
            "selected_config_sha256": (
                digest(selected_config) if selected_config is not None else None),
            "outer_evaluation_performed": False,
            "final_rows_accessed": False,
            "final_labels_accessed": False,
        }
        _write_json(temporary / "fit_scope.json", scope)
        _write_json(temporary / "config_registry.json", config_registry)
        _write_json(temporary / "inner_split_audit.json", projection_audit)
        _write_json(temporary / "inner_selection.json", selection)
        _write_json(temporary / "selected_config.json", selected_payload)
        _write_json(temporary / "inner_fit_program.json", fitted_program.to_dict())
        evidence_names = (
            "fit_only_rows.npz", "fit_scope.json", "config_registry.json",
            "inner_split_audit.json", "inner_selection.json",
            "selected_config.json", "inner_fit_program.json",
        )
        evidence = {
            name: file_hash(temporary / name) for name in evidence_names
        }
        report = {
            "version": VERSION,
            "status": selection["status"],
            "contract": contract(),
            "bindings": {
                "source_v8_report_sha256": expected_source_report_sha256,
                "source_v8_rows_sha256": file_hash(paths["rows.npz"]),
                "source_v8_config_sha256": file_hash(paths["fit_config.json"]),
                "source_v8_program_sha256": file_hash(paths["program.json"]),
                "source_v8_weights_audit_sha256": file_hash(
                    paths["weights_audit.json"]),
                "actor_file_sha256": file_hash(actor_file),
                "fit_scope_file_sha256": expected_fit_scope_sha256,
                "fit_scope_content_sha256": scope["content_sha256"],
                "fresh_outer_registry_file_sha256": fresh_registry_sha256,
                "fresh_outer_registry_content_sha256": fresh_registry[
                    "content_sha256"],
                "fresh_outer_report_file_sha256": fresh_report_sha256,
                "fresh_outer_report_content_sha256": fresh_report[
                    "content_sha256"],
                "selector_binding_sha256": selector_binding,
                "producer_sources_sha256": digest(sources),
            },
            "projection": projection_audit,
            "selection": selection,
            "selected_config_sha256": selected_payload["selected_config_sha256"],
            "outer_evaluation_performed": False,
            "outer_labels_used_for_projection_fit_or_selection": False,
            "outer_probabilities_used_for_projection_fit_or_selection": False,
            "final_rows_accessed": False,
            "final_labels_accessed": False,
            "runtime_action_override": False,
            "actor_changed": False,
            "formal_ready": False,
            "sources": sources,
            "evidence_artifacts": evidence,
        }
        _write_json(temporary / "report.json", report)
        prepublish()
        os.rename(temporary, destination)
        temporary = None
        return deepcopy(report)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def build(
    *, source_evidence: str | Path, expected_source_report_sha256: str,
    actor_path: str | Path, fit_scope_path: str | Path,
    expected_fit_scope_sha256: str,
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Run selection entirely from immutable copies, then recheck originals."""
    sources = producer_sources()
    with _snapshot_build_inputs(
        source_evidence=source_evidence,
        expected_source_report_sha256=expected_source_report_sha256,
        actor_path=actor_path,
        fit_scope_path=fit_scope_path,
        expected_fit_scope_sha256=expected_fit_scope_sha256,
        fresh_outer_registry_path=fresh_outer_registry_path,
        expected_fresh_outer_registry_sha256=(
            expected_fresh_outer_registry_sha256),
        fresh_outer_report_path=fresh_outer_report_path,
        expected_fresh_outer_report_sha256=expected_fresh_outer_report_sha256,
    ) as snapshot:
        def prepublish() -> None:
            snapshot.verify()
            if producer_sources() != sources:
                raise RuntimeError(
                    "Fit-only selector sources changed during transaction")

        return _build_frozen(
            source_evidence=snapshot.root / "source",
            expected_source_report_sha256=expected_source_report_sha256,
            actor_path=snapshot.paths["actor"],
            fit_scope_path=snapshot.paths["fit_scope"],
            expected_fit_scope_sha256=expected_fit_scope_sha256,
            fresh_outer_registry_path=snapshot.paths["fresh_outer_registry"],
            expected_fresh_outer_registry_sha256=(
                expected_fresh_outer_registry_sha256),
            fresh_outer_report_path=snapshot.paths["fresh_outer_report"],
            expected_fresh_outer_report_sha256=(
                expected_fresh_outer_report_sha256),
            output=output,
            expected_sources=sources,
            prepublish=prepublish,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scope_parser = commands.add_parser("prepare-scope")
    scope_parser.add_argument("--source-evidence", required=True)
    scope_parser.add_argument("--expected-source-report-sha256", required=True)
    scope_parser.add_argument("--fresh-outer-registry", required=True)
    scope_parser.add_argument(
        "--expected-fresh-outer-registry-sha256", required=True)
    scope_parser.add_argument("--fresh-outer-report", required=True)
    scope_parser.add_argument(
        "--expected-fresh-outer-report-sha256", required=True)
    scope_parser.add_argument("--output", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--source-evidence", required=True)
    select_parser.add_argument("--expected-source-report-sha256", required=True)
    select_parser.add_argument("--actor", required=True)
    select_parser.add_argument("--fit-scope", required=True)
    select_parser.add_argument("--expected-fit-scope-sha256", required=True)
    select_parser.add_argument("--fresh-outer-registry", required=True)
    select_parser.add_argument(
        "--expected-fresh-outer-registry-sha256", required=True)
    select_parser.add_argument("--fresh-outer-report", required=True)
    select_parser.add_argument(
        "--expected-fresh-outer-report-sha256", required=True)
    select_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-scope":
        scope = prepare_scope(
            source_evidence=args.source_evidence,
            expected_source_report_sha256=args.expected_source_report_sha256,
            fresh_outer_registry_path=args.fresh_outer_registry,
            expected_fresh_outer_registry_sha256=(
                args.expected_fresh_outer_registry_sha256),
            fresh_outer_report_path=args.fresh_outer_report,
            expected_fresh_outer_report_sha256=(
                args.expected_fresh_outer_report_sha256),
            output=args.output,
        )
        print(canonical({
            "status": "frozen_fit_only_scope",
            "scope": str(Path(args.output).resolve()),
            "scope_sha256": file_hash(Path(args.output).resolve()),
            "fit_scenes": len(scope["eligible_fit_scene_fingerprints"]),
            "previously_exposed_outer_scenes": len(
                scope["previously_exposed_outer_scene_fingerprints"]),
        }))
        return 0
    report = build(
        source_evidence=args.source_evidence,
        expected_source_report_sha256=args.expected_source_report_sha256,
        actor_path=args.actor, fit_scope_path=args.fit_scope,
        expected_fit_scope_sha256=args.expected_fit_scope_sha256,
        fresh_outer_registry_path=args.fresh_outer_registry,
        expected_fresh_outer_registry_sha256=(
            args.expected_fresh_outer_registry_sha256),
        fresh_outer_report_path=args.fresh_outer_report,
        expected_fresh_outer_report_sha256=(
            args.expected_fresh_outer_report_sha256),
        output=args.output,
    )
    print(canonical({
        "status": report["status"],
        "selected_config_sha256": report["selected_config_sha256"],
        "report": str(Path(args.output).resolve() / "report.json"),
    }))
    return 0 if report["status"] == STATUS_SELECTED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "SCOPE_VERSION", "STATUS_SELECTED", "STATUS_FAILED",
    "MIX_CANDIDATES", "INNER_HOLDOUT_SCENES",
    "INNER_HOLDOUT_FAMILY_QUOTAS", "INNER_ORDER_SALT", "CHARGER_MODEL",
    "MIN_CHARGER_DIRECTION_IMPROVEMENT", "MAX_OTHER_METRIC_DEGRADATION",
    "contract", "producer_sources", "normalize_scope", "candidate_configs",
    "choose_candidate", "prepare_scope", "build", "main", "_inner_holdout",
    "_project_fit_only", "_arrays_digest",
]
