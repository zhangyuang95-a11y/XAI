"""Development-only RCPD v8 producer for the frozen diagnostic Actor.

The producer fits one public-observation HistGradientBoosting base program and
three public-predicate specialists.  It consumes the authenticated v7
development rows plus the program-blind v8 development expansion.  The 64
expansion validation scenes win on exact float32 public-observation overlap;
every matching fit row is removed before weights or models are constructed.

There is deliberately no final/holdout input.  The fit config is extracted
only from a hash-authenticated, fit-only selector report and its bound scope;
there is no direct config input.  Fitted sklearn objects are never saved: the
only executable artifact is the explicit JSON tree program used by the NumPy
runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training import warehouse_r41_diagnostic_rcpd as legacy
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as expansion_api
from backend.training import warehouse_r41_diagnostic_expansion_rows_v8 as expansion_rows_api
from backend.training import warehouse_r41_diagnostic_prior_rows_v8 as prior_rows_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
    read_authenticated_bytes,
)
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as weight_api
from backend.warehouse_r41_diagnostic_boosted_tree import (
    R41DiagnosticBoostedTreeProgram,
    VERSION as BOOSTED_TREE_VERSION,
    export_hist_gradient_boosting_classifier,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    AGGREGATION,
    GROUPS,
    R41DiagnosticPublicTreeProgramV8,
    VERSION as PUBLIC_TREE_VERSION,
    assemble_public_tree_program_v8,
)
from env.warehouse.navigation import ACTIONS as ENV_ACTIONS
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-rcpd.v8"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_VERSION = "warehouse-r41-diagnostic-rcpd-v8-fit-config.v1"
STATUS_PASSED = "passed_development_gates"
STATUS_FAILED = "failed_development_gates"
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
CLASSES = (0, 1, 2, 3, 4)
COMPONENTS = ("base", *GROUPS)
MIN_OVERALL = 0.90
MIN_NONWAIT = 0.90
MIN_CRITICAL = 0.85
MIN_DIRECTION = 0.85
MIN_EVIDENCE_SCENES = 10
BASE_FEATURE_COUNT = 197
EXPANDED_FEATURE_COUNT = 349
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_COMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NPZ_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROGRAM_BYTES = 512 * 1024 * 1024
PREDICTION_BATCH_SIZE = 16_384
TRACE_SAMPLE_LIMIT = 16
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FIELDS = frozenset(v7._FIELDS)
_CONFIG_FIELDS = frozenset((
    "version", "pair_pool_multiplier", "use_action_factor", "models",
    "mix_weights",
))
_MODEL_FIELDS = frozenset((
    "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
    "l2_regularization", "max_depth", "max_bins", "random_state",
))
_SELECTOR_REQUIRED_ARTIFACTS = frozenset((
    "fit_only_rows.npz", "fit_scope.json", "config_registry.json",
    "inner_split_audit.json", "inner_selection.json", "selected_config.json",
    "inner_fit_program.json",
))
_CANDIDATE_SELECTOR_ARTIFACTS = frozenset((
    "fit_selector_report.json", "fit_selector_scope.json",
    "fit_selector_selected_config.json",
))


if tuple(ENV_ACTIONS) != ACTIONS:
    raise RuntimeError("Warehouse environment action registry changed")


def _fixed_routes() -> dict[str, dict[str, Any]]:
    return {
        group: {
            "feature_name": "derived.critical." + group,
            "operator": ">",
            "threshold": 0.5,
        }
        for group in GROUPS
    }


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "development-only candidate extraction",
        "source_rows": {
            "prior": "authenticated warehouse-r41-diagnostic-rcpd.v7 rows",
            "fit": (
                "all prior v7 development scenes plus the 128-scene v8 "
                "fit supplement"
            ),
            "validation": "64 frozen v8 development-validation scenes only",
            "validation_wins_exact_public_observation_overlap": True,
            "final_rows_accessed": False,
            "final_labels_accessed": False,
            "historical_final_overlap_check": (
                "deferred to irrevocable claim-bound exclusion phase"
            ),
        },
        "prediction_inputs": {
            "base_public_features": BASE_FEATURE_COUNT,
            "derived_public_features": EXPANDED_FEATURE_COUNT - BASE_FEATURE_COUNT,
            "expanded_public_features": EXPANDED_FEATURE_COUNT,
            "actor_logits": False,
            "actor_hidden_state": False,
            "intervention_metadata": False,
            "physical_hash": False,
            "action_label": False,
        },
        "actions": list(ACTIONS),
        "classes": list(CLASSES),
        "components": list(COMPONENTS),
        "routes": _fixed_routes(),
        "aggregation": deepcopy(AGGREGATION),
        "fit": {
            "estimator": "sklearn HistGradientBoostingClassifier",
            "loss": "log_loss",
            "early_stopping": False,
            "base_rows": "all fit rows",
            "specialist_rows": (
                "fit rows in the public critical group plus both endpoints of "
                "effective pairs whose ordinary pre-action source anchor is in "
                "that group; the endpoint union is used only when exact-overlap "
                "removal omitted that ordinary source row"
            ),
            "validation_labels_used_for_fit": False,
            "candidate_search_inside_producer": False,
        },
        "weights": {
            "version": weight_api.VERSION,
            "contract_sha256": digest(weight_api.contract()),
            "validation_weight": 1.0,
            "validation_labels_accessed": False,
        },
        "config_selection": {
            "source": "authenticated fit-only selector evidence",
            "direct_config_input": False,
            "selector_report_scope_and_selected_config_copied": True,
            "all_selector_artifacts_authenticated_before_fit": True,
            "inner_program_metrics_and_selection_deterministically_refit": True,
            "fresh_outer_registry_bound_without_outer_labels": True,
        },
        "effective_pair_group_assignment": (
            "ordinary pre-action source anchor; endpoint union only when the "
            "ordinary row was omitted by exact-overlap removal"
        ),
        "serialization": {
            "kind": "explicit JSON axis-threshold trees",
            "pickle": False,
            "runtime": "NumPy and Python standard library",
            "traceable_paths": True,
        },
        "thresholds": {
            "overall": MIN_OVERALL,
            "nonwait": MIN_NONWAIT,
            "critical": MIN_CRITICAL,
            "effective_intervention_direction": MIN_DIRECTION,
            "effective_intervention_direction_by_group": MIN_DIRECTION,
            "minimum_evidence_scenes": MIN_EVIDENCE_SCENES,
        },
        "actor_changed": False,
        "runtime_action_override": False,
        "independent_final_audit_required": True,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    sources = local_source_hashes((Path(__file__).resolve(),))
    # The selector's standalone CLI is part of the scientific producer even
    # though it is not imported by this module.  Merge its own authenticated
    # closure so the pre/post-fit source check also catches selector drift.
    from backend.training import (  # noqa: PLC0415
        warehouse_r41_diagnostic_rcpd_v8_fit_selector as selector_api,
    )
    for path, source_sha256 in selector_api.producer_sources().items():
        if path in sources and sources[path] != source_sha256:
            raise RuntimeError("Selector/source closure hash disagreement: " + path)
        sources[path] = source_sha256
    for path, sha256 in designation_binding.designation.source_closure().items():
        if path in sources and sources[path] != sha256:
            raise RuntimeError("Designation/source closure hash disagreement: " + path)
        sources[path] = sha256
    return dict(sorted(sources.items()))


def _fixed_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash every canonical input before semantic authentication."""
    result: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if (not path.is_file() or path.is_symlink() or path.resolve() != path):
            raise RuntimeError(
                "Diagnostic v8 fixed input changed during authentication: " + name)
        result[name] = file_hash(path)
    return result


def _integrity_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "producer_sources": producer_sources(),
        "fixed_input_sha256": _fixed_input_hashes(paths),
    }


def _verify_integrity(
    snapshot: Mapping[str, Any], paths: Mapping[str, Path], *, phase: str,
) -> None:
    try:
        current_sources = producer_sources()
        current_inputs = _fixed_input_hashes(paths)
    except BaseException as error:
        raise RuntimeError(
            "Diagnostic v8 sources or fixed inputs changed during " + phase
        ) from error
    if (current_sources != snapshot.get("producer_sources")
            or current_inputs != snapshot.get("fixed_input_sha256")):
        raise RuntimeError(
            "Diagnostic v8 sources or fixed inputs changed during " + phase)


def _validate_designation_v2(
    *, designation_path: Path, actor_path: Path, protocol_path: Path,
    actor: NumPyNativeActor, protocol: Mapping[str, Any],
    designation_original_path: Path,
    designation_snapshot_components: Mapping[str, Path],
    designation_original_components: Mapping[str, Path],
) -> dict[str, Any]:
    saved = designation_binding.read_bound_designation_snapshot(
        designation_path,
        original_path=designation_original_path,
        components=designation_snapshot_components,
        original_components=designation_original_components,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    bindings = saved["bindings"]
    if (bindings.get("actor_sha256") != file_hash(actor_path)
            or bindings.get("actor_parameters_sha256")
                != actor.metadata.get("actor_parameters_sha256")
            or bindings.get("protocol_file_sha256") != file_hash(protocol_path)
            or bindings.get("protocol_content_sha256") != digest(protocol)):
        raise ValueError("RCPD v8 designation v2 input binding differs")
    return saved


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str, *, maximum: int | None = None) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or (maximum is not None and path.stat().st_size > maximum)):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _manifest_validation_path(manifest_path: Path) -> Path:
    validation = _regular(
        manifest_path.parent / "validation.json", "Manifest validation",
        maximum=MAX_JSON_BYTES)
    if (validation.parent != manifest_path.parent
            or file_hash(validation)
                != manifest_binding.EXPECTED_VALIDATION_SHA256):
        raise ValueError("Exact canonical manifest validation bytes required")
    return validation


def _json_pairs(label: str):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result
    return pairs


def _read_json(value: str | Path, label: str) -> dict[str, Any]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_pairs(label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(label + " must be UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(label + " must be one JSON object")
    return parsed


def _read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_pairs(label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(label + " must be one JSON object")
    return parsed


def _selector_snapshot_inputs(
    evidence: str | Path, *, expected_report_sha256: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve all selector artifacts needed for deterministic refit.

    The report is read with ``O_NOFOLLOW`` under its caller-supplied exact
    hash.  Its artifact registry supplies the hashes used by the subsequent
    immutable snapshot; semantic authentication happens only inside that
    snapshot.
    """
    directory = Path(evidence).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Fit-only selector evidence directory is unsafe")
    report_path = _regular(
        directory / "report.json", "Fit-only selector report",
        maximum=MAX_JSON_BYTES)
    report_sha256 = _sha(expected_report_sha256, "fit-only selector report")
    raw = read_authenticated_bytes(
        report_path, label="fit-only selector report",
        expected_sha256=report_sha256, maximum=MAX_JSON_BYTES)
    report = _read_json_bytes(raw, "Fit-only selector report")
    artifacts = report.get("evidence_artifacts")
    if (not isinstance(artifacts, Mapping)
            or not _SELECTOR_REQUIRED_ARTIFACTS.issubset(artifacts)
            or any(type(artifacts[name]) is not str
                   or _HEX.fullmatch(artifacts[name]) is None
                   for name in _SELECTOR_REQUIRED_ARTIFACTS)):
        raise ValueError("Fit-only selector artifact registry differs")
    names = {
        "fit_only_rows.npz": "selector_fit_only_rows",
        "fit_scope.json": "selector_scope",
        "config_registry.json": "selector_config_registry",
        "inner_split_audit.json": "selector_inner_split_audit",
        "inner_selection.json": "selector_inner_selection",
        "selected_config.json": "selector_selected_config",
        "inner_fit_program.json": "selector_inner_fit_program",
    }
    originals = {"selector_report": report_path}
    expected = {"selector_report": report_sha256}
    for artifact_name, snapshot_name in names.items():
        originals[snapshot_name] = _regular(
            directory / artifact_name, "Fit-only selector " + artifact_name,
            maximum=(MAX_NPZ_COMPRESSED_BYTES
                     if artifact_name.endswith(".npz") else MAX_JSON_BYTES),
        )
        expected[snapshot_name] = artifacts[artifact_name]
    return originals, expected


def _canonical_json_file_sha256(value: Any) -> str:
    return sha256((canonical(value) + "\n").encode("utf-8")).hexdigest()


def _number(value: Any, label: str, *, lower: float, upper: float,
            lower_inclusive: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or type(value) not in (int, float):
        raise ValueError(label + " must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    lower_ok = result >= lower if lower_inclusive else result > lower
    if not lower_ok or result > upper:
        bracket = "[" if lower_inclusive else "("
        raise ValueError(f"{label} must be in {bracket}{lower}, {upper}]")
    return result


def _integer(value: Any, label: str, *, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")
    return value


def _designation_component_paths(designation_path: Path) -> dict[str, Path]:
    """Resolve all five fixed designation inputs through the shared adapter."""
    result: dict[str, Path] = {}
    for name, path in designation_binding.resolve_bound_components(
            designation_path).items():
        result["designation_" + name] = _regular(
            path, "designation " + name,
            maximum=(MAX_NPZ_COMPRESSED_BYTES if name == "actor"
                     else MAX_JSON_BYTES))
    return result


def _resolved_designation_components(designation_path: Path) -> dict[str, Path]:
    raw = read_authenticated_bytes(
        designation_path, label="diagnostic Actor designation",
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
        maximum=MAX_JSON_BYTES,
    )
    return designation_binding.resolve_bound_components_from_bytes(
        raw, original_path=designation_path,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )


def _snapshot_expected_hashes(
    *, expected_expansion_registry_sha256: str,
    expected_expansion_report_sha256: str,
    expected_prior_rows_report_sha256: str,
    expected_expansion_rows_report_sha256: str,
    expected_selector_report_sha256: str,
    expected_selector_scope_sha256: str,
    expected_selector_fit_only_rows_sha256: str,
    expected_selector_config_registry_sha256: str,
    expected_selector_inner_split_audit_sha256: str,
    expected_selector_inner_selection_sha256: str,
    expected_selector_inner_fit_program_sha256: str,
    expected_selector_selected_config_sha256: str,
) -> dict[str, str]:
    return {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "expansion_registry": expected_expansion_registry_sha256,
        "expansion_registry_report": expected_expansion_report_sha256,
        "selector_report": expected_selector_report_sha256,
        "selector_scope": expected_selector_scope_sha256,
        "selector_fit_only_rows": expected_selector_fit_only_rows_sha256,
        "selector_config_registry": expected_selector_config_registry_sha256,
        "selector_inner_split_audit": (
            expected_selector_inner_split_audit_sha256),
        "selector_inner_selection": expected_selector_inner_selection_sha256,
        "selector_inner_fit_program": expected_selector_inner_fit_program_sha256,
        "selector_selected_config": expected_selector_selected_config_sha256,
        "previous_development": expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256,
        "prior_reauth_report": expected_prior_rows_report_sha256,
        "prior_rows": prior_rows_api.EXPECTED_SOURCE_ROWS_SHA256,
        "prior_source_report": prior_rows_api.EXPECTED_SOURCE_REPORT_SHA256,
        "expansion_reauth_report": expected_expansion_rows_report_sha256,
        "expansion_collection_report": (
            expansion_rows_api.EXPECTED_SOURCE_COLLECTION_REPORT_SHA256),
        "expansion_rows": expansion_rows_api.EXPECTED_SOURCE_ROWS_SHA256,
        "designation_actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "designation_protocol": (
            designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256),
        "designation_training_ledger": (
            designation_binding.designation.EXPECTED_LEDGER_SHA256),
        "designation_dual_evaluation": (
            designation_binding.designation.EXPECTED_DUAL_EVALUATION_SHA256),
        "designation_failure_closeout": (
            designation_binding.designation.EXPECTED_CLOSEOUT_SHA256),
    }


def normalize_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
        raise ValueError("Diagnostic v8 fit-config schema differs")
    if value.get("version") != CONFIG_VERSION:
        raise ValueError("Diagnostic v8 fit-config version differs")
    if type(value.get("use_action_factor")) is not bool:
        raise ValueError("use_action_factor must be a bool")
    multiplier = _number(
        value.get("pair_pool_multiplier"), "pair_pool_multiplier",
        lower=0.0, upper=weight_api.MAX_PAIR_POOL_MULTIPLIER,
        lower_inclusive=False,
    )
    raw_models = value.get("models")
    if not isinstance(raw_models, Mapping) or set(raw_models) != set(COMPONENTS):
        raise ValueError("Diagnostic v8 requires exactly the base and three specialists")
    models: dict[str, dict[str, Any]] = {}
    for name in COMPONENTS:
        raw = raw_models[name]
        if not isinstance(raw, Mapping) or set(raw) != _MODEL_FIELDS:
            raise ValueError(name + " HGB config schema differs")
        depth = raw.get("max_depth")
        if depth is not None:
            depth = _integer(depth, name + ".max_depth", lower=1, upper=32)
        models[name] = {
            "learning_rate": _number(
                raw.get("learning_rate"), name + ".learning_rate",
                lower=0.0, upper=1.0, lower_inclusive=False),
            "max_iter": _integer(
                raw.get("max_iter"), name + ".max_iter", lower=1, upper=2000),
            "max_leaf_nodes": _integer(
                raw.get("max_leaf_nodes"), name + ".max_leaf_nodes",
                lower=2, upper=512),
            "min_samples_leaf": _integer(
                raw.get("min_samples_leaf"), name + ".min_samples_leaf",
                lower=1, upper=100_000),
            "l2_regularization": _number(
                raw.get("l2_regularization"), name + ".l2_regularization",
                lower=0.0, upper=1_000_000.0),
            "max_depth": depth,
            "max_bins": _integer(
                raw.get("max_bins"), name + ".max_bins", lower=2, upper=255),
            "random_state": _integer(
                raw.get("random_state"), name + ".random_state",
                lower=0, upper=2**32 - 1),
        }
    raw_mix = value.get("mix_weights")
    if not isinstance(raw_mix, Mapping) or set(raw_mix) != set(GROUPS):
        raise ValueError("Diagnostic v8 requires exactly three specialist mix weights")
    mix = {
        group: _number(raw_mix[group], group + " mix weight", lower=0.0, upper=1.0)
        for group in GROUPS
    }
    return {
        "version": CONFIG_VERSION,
        "pair_pool_multiplier": multiplier,
        "use_action_factor": value["use_action_factor"],
        "models": models,
        "mix_weights": mix,
    }


def _load_npz(value: str | Path, label: str) -> dict[str, np.ndarray]:
    path = _regular(value, label, maximum=MAX_NPZ_COMPRESSED_BYTES)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            expected = {name + ".npy" for name in _ROW_FIELDS}
            if ({item.filename for item in infos} != expected
                    or len(infos) != len(expected)
                    or any(item.is_dir() or item.file_size < 0
                           or item.compress_size < 0 for item in infos)
                    or sum(item.file_size for item in infos) > MAX_NPZ_EXPANDED_BYTES):
                raise ValueError(label + " archive contents differ")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _ROW_FIELDS:
                raise ValueError(label + " array schema differs")
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(label + " is not a safe NPZ archive") from exc


def _write_exclusive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Diagnostic v8 output path is unsafe")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_exclusive(path, (canonical(value) + "\n").encode("utf-8"))


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("Diagnostic v8 output archive already exists")
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_exclusive(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("Diagnostic v8 raw evidence destination exists")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        with os.fdopen(source_fd, "rb", closefd=True) as incoming, \
                os.fdopen(destination_fd, "wb", closefd=True) as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _decode(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim != 1 or array.dtype.kind not in "SU":
        raise ValueError(label + " must be a one-dimensional text array")
    try:
        return array.astype(str) if array.dtype.kind == "U" \
            else np.char.decode(array, "ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(label + " contains non-ASCII data") from exc


def _validate_base_shapes(arrays: Mapping[str, np.ndarray], actor: NumPyNativeActor,
                          label: str) -> None:
    if set(arrays) != _ROW_FIELDS:
        raise ValueError(label + " row fields differ")
    count = len(arrays["observations"])
    expected_dtypes = {
        "action_indices": np.dtype(np.uint8),
        "weights": np.dtype(np.float32),
        "observation_hashes": np.dtype("S64"),
        "scene_fingerprints": np.dtype("S64"),
        "episode_ids": np.dtype("S180"),
        "frames": np.dtype(np.int16),
        "group_bits": np.dtype(np.uint8),
        "kinds": np.dtype("S16"),
        "anchor_ids": np.dtype("S240"),
        "branch_actions": np.dtype("S8"),
        "physical_hashes": np.dtype("S64"),
        "source_state_hashes": np.dtype("S64"),
        "submitted_equal": np.dtype(np.bool_),
        "trajectory_done": np.dtype(np.bool_),
        "split_validation": np.dtype(np.bool_),
    }
    if (count <= 0
            or arrays["observations"].shape != (count, BASE_FEATURE_COUNT)
            or arrays["observations"].dtype != np.dtype(np.float32)
            or arrays["probabilities"].shape != (count, len(ACTIONS))
            or arrays["probabilities"].dtype != np.dtype(np.float32)
            or any(arrays[name].shape != (count,) or arrays[name].dtype != dtype
                   for name, dtype in expected_dtypes.items())):
        raise ValueError(label + " row shapes or dtypes differ")
    observations = arrays["observations"]
    probabilities = arrays["probabilities"]
    labels = arrays["action_indices"]
    if (not np.isfinite(observations).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(1), 1.0, rtol=0.0, atol=2e-6)
            or np.any(labels >= len(ACTIONS))
            or not np.all(arrays["submitted_equal"])
            or np.any(arrays["frames"] < 0)
            or np.any(arrays["group_bits"] >= (1 << len(GROUPS)))):
        raise ValueError(label + " numeric or authority evidence differs")
    expected_hashes = np.asarray(
        [legacy._obs_hash(row) for row in observations], dtype="S64")
    if not np.array_equal(expected_hashes, arrays["observation_hashes"]):
        raise ValueError(label + " public-observation hashes differ")
    # Authentication may read validation labels.  Model fitting below receives
    # only explicitly sliced fit labels and never receives this full array.
    logits = actor.logits(observations)
    actor_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    actor_probabilities /= actor_probabilities.sum(axis=1, keepdims=True)
    if (not np.allclose(probabilities, actor_probabilities, rtol=8e-6, atol=4e-6)
            or not np.array_equal(
                labels, actor_probabilities.argmax(1).astype(np.uint8))):
        raise ValueError(label + " rows differ from the frozen Actor")
    kinds = _decode(arrays["kinds"], label + " kinds")
    anchors = _decode(arrays["anchor_ids"], label + " anchors")
    branches = _decode(arrays["branch_actions"], label + " branches")
    physical = _decode(arrays["physical_hashes"], label + " physical hashes")
    sources = _decode(arrays["source_state_hashes"], label + " state hashes")
    ordinary = kinds == "ordinary"
    intervention = kinds == "intervention"
    if (not np.all(ordinary | intervention)
            or np.any(arrays["trajectory_done"][intervention])
            or not np.all(branches[ordinary] == "")
            or not np.all(physical[ordinary] == "")
            or np.any(anchors[intervention] == "")
            or not set(map(str, branches[intervention])).issubset(ACTIONS)
            or any(_HEX.fullmatch(str(item)) is None for item in physical[intervention])
            or any(_HEX.fullmatch(str(item)) is None for item in sources)):
        raise ValueError(label + " row-kind schema differs")


def _scene_family_map(manifest: Mapping[str, Any],
                      expansion: Mapping[str, Any],
                      previous_development: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    candidate_batches = manifest.get("candidate_batches")
    if not isinstance(candidate_batches, list):
        raise ValueError("Diagnostic manifest candidate population differs")
    public_development_rows: list[Mapping[str, Any]] = []
    for batch in candidate_batches:
        if not isinstance(batch, list):
            raise ValueError("Diagnostic manifest candidate batch differs")
        public_development_rows.extend(batch)
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Diagnostic manifest split registry differs")
    # Intentionally name only development splits.  The final split is neither
    # indexed nor used as a family source by this producer.
    for name in ("train", "conflict_validation"):
        rows = splits.get(name)
        if not isinstance(rows, list):
            raise ValueError("Diagnostic manifest development split differs")
        public_development_rows.extend(rows)
    previous_rows = previous_development.get("scenes")
    if not isinstance(previous_rows, list):
        raise ValueError("Previous development scene registry differs")
    public_development_rows.extend(previous_rows)
    for row in public_development_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Diagnostic manifest candidate differs")
        fingerprint = row.get("fingerprint")
        family = row.get("family_id")
        if (_HEX.fullmatch(str(fingerprint)) is None
                or type(family) is not str or not family):
            raise ValueError("Diagnostic scene identity differs")
        registered_family = result.setdefault(str(fingerprint), family)
        if registered_family != family:
            raise ValueError("One scene belongs to two conflict families")
    for row in (*expansion["fit_supplement"],
                *expansion["development_validation"]):
        fingerprint = str(row["fingerprint"])
        family = str(row["family_id"])
        if result.get(fingerprint) != family:
            raise ValueError("Expansion scene family differs from source manifest")
    return result


def _validate_expansion_registry(
    value: Mapping[str, Any], *, registry_path: Path, actor: NumPyNativeActor,
    manifest_path: Path, designation_path: Path, prior_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (value.get("version") != expansion_api.VERSION
            or value.get("status") != expansion_api.STATUS
            or value.get("content_sha256") != digest({
                key: item for key, item in value.items() if key != "content_sha256"
            })
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("final_labels_used_for_selection") is not False
            or value.get("runtime_action_override") is not False):
        raise ValueError("Frozen development expansion registry differs")
    bindings = value.get("bindings")
    if (not isinstance(bindings, Mapping)
            or bindings.get("actor_sha256") != actor.artifact_sha256
            or bindings.get("actor_parameters_sha256")
                != actor.metadata["actor_parameters_sha256"]
            or bindings.get("source_manifest_file_sha256") != file_hash(manifest_path)
            or bindings.get("designation_file_sha256") != file_hash(designation_path)
            or bindings.get("previous_development_file_sha256")
                != prior_report["bindings"]["development_supplement_file_sha256"]
            or bindings.get("contract_sha256") != digest(expansion_api.contract())):
        raise ValueError("Development expansion bindings differ")
    fit = value.get("fit_supplement")
    validation = value.get("development_validation")
    if (not isinstance(fit, list)
            or len(fit) != expansion_api.FIT_SUPPLEMENT_SCENES
            or not isinstance(validation, list)
            or len(validation) != expansion_api.VALIDATION_SCENES):
        raise ValueError("Development expansion split size differs")
    seen: set[str] = set()
    for split_name, rows in (("fit_supplement", fit),
                             ("development_validation", validation)):
        for row in rows:
            if (not isinstance(row, Mapping)
                    or row.get("split") != split_name
                    or _HEX.fullmatch(str(row.get("fingerprint"))) is None
                    or type(row.get("id")) is not str
                    or type(row.get("family_id")) is not str
                    or row["fingerprint"] in seen):
                raise ValueError("Development expansion scene registry differs")
            seen.add(str(row["fingerprint"]))
    if len(seen) != len(fit) + len(validation):
        raise ValueError("Development expansion scene identities overlap")
    return deepcopy(fit), deepcopy(validation)


def _row_key(arrays: Mapping[str, np.ndarray], index: int) -> tuple[bytes, ...]:
    return tuple(bytes(np.asarray(arrays[name][index]).tobytes()) for name in (
        "observation_hashes", "scene_fingerprints", "episode_ids", "frames",
        "kinds", "anchor_ids", "branch_actions", "physical_hashes",
        "source_state_hashes", "trajectory_done", "submitted_equal",
    ))


def _subset(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[indices].copy() for name, value in arrays.items()}


def _validate_expansion_rows(
    arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
    fit_scenes: Sequence[Mapping[str, Any]],
    validation_scenes: Sequence[Mapping[str, Any]],
    prior_arrays: Mapping[str, np.ndarray], relations: R41DiagnosticPublicRelationsV8,
) -> str:
    _validate_base_shapes(arrays, actor, "Development expansion rows")
    scenes = _decode(arrays["scene_fingerprints"], "Expansion scenes")
    split = arrays["split_validation"]
    fit_fingerprints = {str(row["fingerprint"]) for row in fit_scenes}
    validation_fingerprints = {str(row["fingerprint"]) for row in validation_scenes}
    expansion_fingerprints = fit_fingerprints | validation_fingerprints
    prior_fingerprints = set(map(str, _decode(
        prior_arrays["scene_fingerprints"], "Prior scenes")))
    supplied = set(map(str, scenes))
    if supplied == expansion_fingerprints:
        layout = "expansion_only"
    elif supplied == expansion_fingerprints | prior_fingerprints:
        layout = "already_combined"
    else:
        raise ValueError("Expansion row archive scene population differs")
    if (set(map(str, scenes[split])) != validation_fingerprints
            or set(map(str, scenes[~split]))
                != (fit_fingerprints if layout == "expansion_only"
                    else fit_fingerprints | prior_fingerprints)
            or np.any(np.isin(scenes[~split], list(validation_fingerprints)))):
        raise ValueError("Expansion row split differs from frozen registry")
    episodes = _decode(arrays["episode_ids"], "Expansion episodes")
    expected_expansion_episodes = {
        f"{row['id']}:{row['fingerprint']}:{partner}"
        for row in (*fit_scenes, *validation_scenes) for partner in v7.PARTNERS
    }
    expected_validation_episodes = {
        f"{row['id']}:{row['fingerprint']}:{partner}"
        for row in validation_scenes for partner in v7.PARTNERS
    }
    if not expected_expansion_episodes.issubset(set(map(str, episodes))):
        raise ValueError("Expansion row episode matrix is incomplete")
    expansion_mask = np.isin(scenes, list(expansion_fingerprints))
    if set(map(str, episodes[expansion_mask])) != expected_expansion_episodes:
        raise ValueError("Expansion rows contain an unregistered episode")
    validation_hashes = set(map(bytes, arrays["observation_hashes"][split]))
    if set(map(bytes, arrays["observation_hashes"][~split])) & validation_hashes:
        raise ValueError("Expansion rows violate validation-wins deduplication")
    critical = relations.critical_masks(arrays["observations"])
    derived_bits = np.zeros(len(scenes), dtype=np.uint8)
    for index, group in enumerate(GROUPS):
        derived_bits |= critical[group].astype(np.uint8) << index
    if not np.array_equal(derived_bits, arrays["group_bits"]):
        raise ValueError("Expansion critical groups differ from public observations")
    kinds = _decode(arrays["kinds"], "Expansion kinds")
    anchors = _decode(arrays["anchor_ids"], "Expansion anchors")
    branches = _decode(arrays["branch_actions"], "Expansion branches")
    frames = arrays["frames"]
    ordinary = expansion_mask & (kinds == "ordinary")
    expected_anchors = np.asarray([
        (f"{episodes[index]}:{int(frames[index])}"
         if arrays["group_bits"][index] != 0
         and (not split[index] or int(frames[index]) % 5 == 0)
         else "")
        for index in np.flatnonzero(ordinary)
    ], dtype="U240")
    if not np.array_equal(anchors[ordinary], expected_anchors):
        raise ValueError("Expansion intervention schedule differs")
    # The frozen validation side is never deduplicated and therefore retains
    # complete trajectories and all five branch endpoints per critical anchor.
    validation_ordinary = ordinary & split
    for episode in sorted(expected_validation_episodes):
        indices = np.flatnonzero(validation_ordinary & (episodes == episode))
        if not len(indices):
            raise ValueError("Expansion validation trajectory is missing")
        ordered = indices[np.argsort(frames[indices], kind="stable")]
        if (not np.array_equal(frames[ordered], np.arange(len(ordered)))
                or np.any(arrays["trajectory_done"][ordered[:-1]])
                or not bool(arrays["trajectory_done"][ordered[-1]])):
            raise ValueError("Expansion validation trajectory is incomplete")
    anchor_rows: dict[str, list[int]] = {}
    for index in np.flatnonzero(expansion_mask & (kinds == "intervention")):
        anchor_rows.setdefault(str(anchors[index]), []).append(int(index))
    for anchor, indices in anchor_rows.items():
        if (len({bool(split[index]) for index in indices}) != 1
                or len({str(scenes[index]) for index in indices}) != 1
                or len({str(episodes[index]) for index in indices}) != 1):
            raise ValueError("Expansion intervention anchor crosses a boundary")
        branch_counts = Counter(map(str, branches[indices]))
        if (split[indices[0]] and (not branch_counts
                or not set(branch_counts).issubset(ACTIONS)
                or any(count != 1 for count in branch_counts.values()))):
            # A branch that immediately terminates has no next-decision row,
            # so the complete retained matrix is a unique subset of actions.
            raise ValueError("Expansion validation anchor branch matrix differs")
    if layout == "already_combined":
        prior_scenes = np.isin(scenes, list(prior_fingerprints))
        expected_prior_indices = np.flatnonzero([
            bytes(value) not in validation_hashes
            for value in prior_arrays["observation_hashes"]
        ])
        supplied_prior_indices = np.flatnonzero(prior_scenes)
        expected_keys = Counter(
            _row_key(prior_arrays, int(index)) for index in expected_prior_indices)
        supplied_keys = Counter(
            _row_key(arrays, int(index)) for index in supplied_prior_indices)
        if expected_keys != supplied_keys:
            raise ValueError("Combined archive changed the authenticated prior-row projection")
    return layout


def _validation_wins_merge(
    prior_arrays: Mapping[str, np.ndarray], expansion_arrays: Mapping[str, np.ndarray],
    *, layout: str, expansion_fingerprints: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if layout == "already_combined":
        rows = {name: value.copy() for name, value in expansion_arrays.items()}
        input_fit = int(np.sum(~rows["split_validation"]))
        input_validation = int(np.sum(rows["split_validation"]))
        removed_prior = len(prior_arrays["observations"]) - int(np.sum(np.isin(
            _decode(rows["scene_fingerprints"], "Combined scenes"),
            list(set(map(str, _decode(prior_arrays["scene_fingerprints"], "Prior scenes"))))
        )))
        accounting = {
            "source_layout": layout,
            "raw_prior_rows": len(prior_arrays["observations"]),
            "raw_expansion_archive_rows": len(expansion_arrays["observations"]),
            "raw_expansion_only_rows": int(np.sum(np.isin(
                _decode(rows["scene_fingerprints"], "Combined scenes"),
                list(expansion_fingerprints)))),
            "fit_rows_removed_for_exact_validation_overlap": int(removed_prior),
            "retained_fit_rows": input_fit,
            "retained_validation_rows": input_validation,
            "retained_rows": len(rows["observations"]),
        }
        return rows, accounting

    prior = {name: value.copy() for name, value in prior_arrays.items()}
    prior["split_validation"][:] = False
    validation_hashes = set(map(
        bytes,
        expansion_arrays["observation_hashes"][expansion_arrays["split_validation"]],
    ))
    keep_prior = np.asarray([
        bytes(value) not in validation_hashes for value in prior["observation_hashes"]
    ], dtype=np.bool_)
    keep_expansion_fit = np.asarray([
        bool(is_validation) or bytes(value) not in validation_hashes
        for value, is_validation in zip(
            expansion_arrays["observation_hashes"],
            expansion_arrays["split_validation"],
        )
    ], dtype=np.bool_)
    pieces = (_subset(prior, np.flatnonzero(keep_prior)),
              _subset(expansion_arrays, np.flatnonzero(keep_expansion_fit)))
    rows = {
        name: np.concatenate([piece[name] for piece in pieces], axis=0)
        for name in _ROW_FIELDS
    }
    removed = int(np.sum(~keep_prior) + np.sum(~keep_expansion_fit))
    accounting = {
        "source_layout": layout,
        "raw_prior_rows": len(prior_arrays["observations"]),
        "raw_expansion_archive_rows": len(expansion_arrays["observations"]),
        "raw_expansion_only_rows": len(expansion_arrays["observations"]),
        "fit_rows_removed_for_exact_validation_overlap": removed,
        "retained_fit_rows": int(np.sum(~rows["split_validation"])),
        "retained_validation_rows": int(np.sum(rows["split_validation"])),
        "retained_rows": len(rows["observations"]),
    }
    return rows, accounting


def _validate_combined_rows(
    arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
    fit_fingerprints: set[str], validation_fingerprints: set[str],
    relations: R41DiagnosticPublicRelationsV8,
) -> None:
    _validate_base_shapes(arrays, actor, "Combined v8 rows")
    split = arrays["split_validation"]
    scenes = _decode(arrays["scene_fingerprints"], "Combined scenes")
    if (set(map(str, scenes[~split])) != fit_fingerprints
            or set(map(str, scenes[split])) != validation_fingerprints
            or set(map(bytes, arrays["observation_hashes"][~split]))
                & set(map(bytes, arrays["observation_hashes"][split]))):
        raise ValueError("Combined v8 scene split or observation isolation differs")
    critical = relations.critical_masks(arrays["observations"])
    expected_bits = np.zeros(len(scenes), dtype=np.uint8)
    for index, group in enumerate(GROUPS):
        expected_bits |= critical[group].astype(np.uint8) << index
    if not np.array_equal(expected_bits, arrays["group_bits"]):
        raise ValueError("Combined v8 critical groups differ")


def _pair_group_bits(
    arrays: Mapping[str, np.ndarray], pairs: np.ndarray,
) -> np.ndarray:
    """Assign each intervention pair to its ordinary pre-action source group.

    Intervention endpoints describe post-action states, so their critical flags
    can legitimately differ from the state where the intervention was issued.
    The ordinary row sharing the anchor is the authoritative public source.  An
    exact-observation validation-wins removal can omit that fit row; only then do
    we use the deterministic union of the two endpoint masks, matching the v8
    pair-weight contract.
    """
    raw_pairs = np.asarray(pairs)
    if (raw_pairs.ndim != 2 or raw_pairs.shape[1:] != (2,)
            or raw_pairs.dtype.kind not in "iu"):
        raise ValueError("Diagnostic v8 pairs must be a P-by-2 integer array")
    pairs64 = raw_pairs.astype(np.int64, copy=False)
    count = len(arrays["group_bits"])
    if len(pairs64) and (np.any(pairs64 < 0) or np.any(pairs64 >= count)):
        raise ValueError("Diagnostic v8 pair endpoint index is outside the rows")
    split = np.asarray(arrays["split_validation"])
    group_bits = np.asarray(arrays["group_bits"])
    if (split.shape != (count,) or split.dtype != np.dtype(np.bool_)
            or group_bits.shape != (count,) or group_bits.dtype.kind not in "iu"
            or np.any(group_bits < 0)
            or np.any(group_bits >= (1 << len(GROUPS)))):
        raise ValueError("Diagnostic v8 pair grouping arrays differ")
    decoded = {
        name: _decode(np.asarray(arrays[name]), "Pair grouping " + name)
        for name in ("kinds", "anchor_ids", "branch_actions",
                     "scene_fingerprints", "episode_ids")
    }
    if any(values.shape != (count,) for values in decoded.values()):
        raise ValueError("Diagnostic v8 pair grouping metadata is not row aligned")
    kinds = decoded["kinds"]
    anchors = decoded["anchor_ids"]
    branches = decoded["branch_actions"]
    scenes = decoded["scene_fingerprints"]
    episodes = decoded["episode_ids"]

    # Include the split in the identity so validation evidence can never borrow
    # a same-text anchor from fit.  Duplicate ordinary identities are treated as
    # unavailable and take the documented endpoint-union fallback.
    ordinary_by_identity: dict[tuple[str, bool], int] = {}
    duplicate_identities: set[tuple[str, bool]] = set()
    for index in np.flatnonzero(kinds == "ordinary"):
        anchor = str(anchors[index])
        if not anchor:
            continue
        identity = (anchor, bool(split[index]))
        if identity in ordinary_by_identity:
            duplicate_identities.add(identity)
        else:
            ordinary_by_identity[identity] = int(index)
    for identity in duplicate_identities:
        ordinary_by_identity.pop(identity, None)

    result = np.empty(len(pairs64), dtype=np.uint8)
    for pair_index, (raw_wait, raw_branch) in enumerate(pairs64):
        wait = int(raw_wait)
        branch = int(raw_branch)
        if (wait == branch or kinds[wait] != "intervention"
                or kinds[branch] != "intervention"
                or not str(anchors[wait])
                or anchors[wait] != anchors[branch]
                or split[wait] != split[branch]
                or scenes[wait] != scenes[branch]
                or episodes[wait] != episodes[branch]
                or branches[wait] != "WAIT"
                or branches[branch] not in ACTIONS[:-1]):
            raise ValueError("Diagnostic v8 intervention pair identity differs")
        identity = (str(anchors[wait]), bool(split[wait]))
        source = ordinary_by_identity.get(identity)
        if (source is not None
                and (scenes[source] != scenes[wait]
                     or episodes[source] != episodes[wait])):
            source = None
        result[pair_index] = np.uint8(
            group_bits[source] if source is not None
            else int(group_bits[wait]) | int(group_bits[branch]))
    return result


def _validated_pair_group_bits(
    pairs: np.ndarray, pair_group_bits: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(pair_group_bits)
    if (raw.shape != (len(pairs),) or raw.dtype.kind not in "iu"
            or np.any(raw < 0) or np.any(raw >= (1 << len(GROUPS)))):
        raise ValueError("Diagnostic v8 pair_group_bits differs")
    return raw.astype(np.uint8, copy=False)


def _component_fit_masks(
    arrays: Mapping[str, np.ndarray], pairs: np.ndarray,
    pair_group_bits: np.ndarray,
) -> dict[str, np.ndarray]:
    fit = ~arrays["split_validation"]
    bits = arrays["group_bits"]
    pair_bits = _validated_pair_group_bits(pairs, pair_group_bits)
    if len(pairs) and not np.all(fit[np.asarray(pairs).reshape(-1)]):
        raise ValueError("Diagnostic v8 specialist pair includes validation rows")
    result = {"base": fit.copy()}
    for group_index, group in enumerate(GROUPS):
        selected = fit & ((bits & (1 << group_index)) != 0)
        pair_selected = ((pair_bits & (1 << group_index)) != 0) \
            if len(pairs) else np.empty(0, dtype=np.bool_)
        if np.any(pair_selected):
            selected[np.unique(pairs[pair_selected].reshape(-1))] = True
        result[group] = selected
    return result


def _fit_program(
    arrays: Mapping[str, np.ndarray], *, relations: R41DiagnosticPublicRelationsV8,
    weights: Mapping[str, Any], config: Mapping[str, Any], binding: str,
    source_identity: Mapping[str, Any], pair_group_bits: np.ndarray,
) -> tuple[R41DiagnosticPublicTreeProgramV8, dict[str, Any]]:
    expanded = relations.transform_batch(arrays["observations"])
    if expanded.shape != (len(arrays["observations"]), EXPANDED_FEATURE_COUNT):
        raise RuntimeError("Diagnostic v8 expanded feature shape differs")
    masks = _component_fit_masks(
        arrays, weights["pairs"], pair_group_bits)
    fit_labels = arrays["action_indices"]  # sliced before each estimator sees it
    programs: dict[str, R41DiagnosticBoostedTreeProgram] = {}
    diagnostics: dict[str, Any] = {}
    for name in COMPONENTS:
        indices = np.flatnonzero(masks[name])
        labels = fit_labels[indices].copy()
        if tuple(map(int, np.unique(labels))) != CLASSES:
            raise ValueError(name + " fit rows do not cover the exact five classes")
        settings = config["models"][name]
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
        estimator.fit(
            expanded[indices], labels,
            sample_weight=weights["weights"][indices].astype(np.float64),
        )
        if tuple(map(int, estimator.classes_)) != CLASSES:
            raise RuntimeError(name + " fitted class order differs")
        metadata = {
            "diagnostic_rcpd_version": VERSION,
            "diagnostic_rcpd_binding_sha256": binding,
            "component": name,
            "fit_rows": len(indices),
            "fit_config": deepcopy(settings),
            "prediction_input": "349 deterministic public features",
            "validation_labels_used_for_fit": False,
            "actor_logits_used_as_program_input": False,
            "actor_hidden_states_used": False,
            "runtime_action_override": False,
        }
        program = export_hist_gradient_boosting_classifier(
            estimator, feature_names=relations.feature_names,
            action_names=ACTIONS, metadata=metadata,
        )
        if (program.classes != CLASSES or program.action_names != ACTIONS
                or program.feature_names != relations.feature_names):
            raise RuntimeError(name + " explicit program identity differs")
        # Full candidate scoring below runs through the explicit runtime.  This
        # parity check uses a fixed fit/validation sample to avoid a second
        # four-model pass over the entire half-million-row archive.
        fit_sample = indices[:4096]
        validation_sample = np.flatnonzero(arrays["split_validation"])[:4096]
        parity_indices = np.unique(np.concatenate((fit_sample, validation_sample)))
        native = estimator.predict_proba(expanded[parity_indices])
        explicit = program.predict_proba_batch(expanded[parity_indices])
        maximum_error = float(np.max(np.abs(native - explicit)))
        actions_equal = bool(np.array_equal(
            np.argmax(native, axis=1), np.argmax(explicit, axis=1)))
        if maximum_error > 1e-12 or not actions_equal:
            raise RuntimeError(name + " sklearn/explicit-tree parity differs")
        programs[name] = program
        diagnostics[name] = {
            "fit_rows": len(indices),
            "fit_action_counts": {
                action: int(np.sum(labels == index))
                for index, action in enumerate(ACTIONS)
            },
            "iterations": program.n_iterations,
            "parity_rows": len(parity_indices),
            "sklearn_explicit_max_probability_error": maximum_error,
            "sklearn_explicit_actions_equal": actions_equal,
        }
    wrapper = assemble_public_tree_program_v8(
        relations.base_feature_names,
        programs["base"], programs["narrow_passage"],
        programs["shared_pickup"], programs["shared_charger"],
        mix_weights=config["mix_weights"], routes=_fixed_routes(),
        metadata={
            "diagnostic_rcpd_version": VERSION,
            "diagnostic_rcpd_binding_sha256": binding,
            "fit_config_sha256": digest(config),
            "public_feature_contract_sha256": digest(relations.contract()),
            "pair_weight_contract_sha256": digest(weight_api.contract()),
            "actions": list(ACTIONS), "classes": list(CLASSES),
            "native_source_actor_sha256": source_identity[
                "native_source_actor_sha256"],
            "source_actor_parameters_sha256": source_identity[
                "source_actor_parameters_sha256"],
            "source_full_manifest_bindings": deepcopy(
                source_identity["source_full_manifest_bindings"]),
            "runtime_controller": "native_neural_actor_only",
            "runtime_action_override": False,
            "program_feedback_into_actor": False,
            "formal_ready": False,
        },
    )
    return wrapper, diagnostics


def _predict_in_batches(program: R41DiagnosticPublicTreeProgramV8,
                        observations: np.ndarray) -> np.ndarray:
    parts = [
        program.predict_proba_batch(observations[start:start + PREDICTION_BATCH_SIZE])
        for start in range(0, len(observations), PREDICTION_BATCH_SIZE)
    ]
    return np.concatenate(parts, axis=0) if parts \
        else np.empty((0, len(ACTIONS)), dtype=np.float64)


def _metrics_from_probabilities(
    probabilities: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    pairs: np.ndarray,
    pair_group_bits: np.ndarray,
) -> dict[str, Any]:
    """Apply the v7 metrics with source-anchored intervention groups."""
    if (probabilities.shape != (len(arrays["observations"]), len(ACTIONS))
            or not np.isfinite(probabilities).all()
            or not np.allclose(
                probabilities.sum(1), 1.0, rtol=0.0, atol=2e-12)):
        raise ValueError("Diagnostic v8 candidate probabilities differ")
    selected_mask = np.asarray(mask)
    if (selected_mask.shape != (len(arrays["observations"]),)
            or selected_mask.dtype != np.dtype(np.bool_)):
        raise ValueError("Diagnostic v8 metric mask differs")
    canonical_pairs = v7._effective_pairs(arrays, selected_mask)
    if not np.array_equal(np.asarray(pairs), canonical_pairs):
        raise ValueError("Diagnostic v8 effective-pair metric coverage differs")
    pair_bits = _validated_pair_group_bits(canonical_pairs, pair_group_bits)
    if not np.array_equal(pair_bits, _pair_group_bits(arrays, canonical_pairs)):
        raise ValueError("Diagnostic v8 effective-pair group assignment differs")

    predictions = np.argmax(probabilities, axis=1).astype(np.uint8)
    labels = arrays["action_indices"]
    correct = predictions == labels
    scenes = _decode(arrays["scene_fingerprints"], "Metric scenes")
    bits = arrays["group_bits"]

    def stat(selected: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(selected_mask & selected)
        return {
            "rows": int(len(indices)),
            "scenes": len(set(map(str, scenes[indices]))),
            "fidelity": float(correct[indices].mean()) if len(indices) else 0.0,
        }

    critical = {
        name: stat((bits & (1 << index)) != 0)
        for index, name in enumerate(GROUPS)
    }
    pair_correct = (
        correct[canonical_pairs[:, 0]] & correct[canonical_pairs[:, 1]]
    ) if len(canonical_pairs) else np.empty(0, dtype=np.bool_)
    direction_by_group = {}
    for index, name in enumerate(GROUPS):
        selected = (pair_bits & (1 << index)) != 0
        direction_by_group[name] = {
            "pairs": int(np.sum(selected)),
            "scenes": len(set(map(
                str, scenes[canonical_pairs[selected, 0]])))
                if np.any(selected) else 0,
            "fidelity": float(pair_correct[selected].mean())
                if np.any(selected) else 0.0,
        }
    target = arrays["probabilities"][selected_mask]
    approximate = probabilities[selected_mask]
    mean_kl = float(np.mean(np.sum(target * (
        np.log(target.clip(1e-8))
        - np.log(approximate.clip(1e-8))), axis=-1)))
    return {
        "overall": stat(np.ones(len(selected_mask), dtype=np.bool_)),
        "nonwait": stat(labels != ACTIONS.index("WAIT")),
        "critical": critical,
        "effective_intervention_direction": {
            "pairs": int(len(canonical_pairs)),
            "scenes": len(set(map(
                str, scenes[canonical_pairs[:, 0]])))
                if len(canonical_pairs) else 0,
            "fidelity": float(pair_correct.mean()) if len(pair_correct) else 0.0,
            "by_group": direction_by_group,
        },
        "mean_kl": mean_kl,
    }


def _metrics(
    probabilities: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    pairs: np.ndarray,
    pair_group_bits: np.ndarray,
) -> dict[str, Any]:
    return _metrics_from_probabilities(
        probabilities, arrays, arrays["split_validation"],
        pairs=pairs, pair_group_bits=pair_group_bits)


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall": metrics["overall"]["fidelity"] >= MIN_OVERALL,
        "nonwait": metrics["nonwait"]["fidelity"] >= MIN_NONWAIT,
        "effective_intervention_direction": (
            metrics["effective_intervention_direction"]["scenes"]
                >= MIN_EVIDENCE_SCENES
            and metrics["effective_intervention_direction"]["fidelity"]
                >= MIN_DIRECTION
        ),
    }
    for group in GROUPS:
        critical = metrics["critical"][group]
        direction = metrics["effective_intervention_direction"]["by_group"][group]
        checks[group] = (
            critical["scenes"] >= MIN_EVIDENCE_SCENES
            and critical["fidelity"] >= MIN_CRITICAL
        )
        checks["effective_intervention_direction_" + group] = (
            direction["scenes"] >= MIN_EVIDENCE_SCENES
            and direction["fidelity"] >= MIN_DIRECTION
        )
    return {"checks": checks, "passed": all(checks.values())}


def _strata(probabilities: np.ndarray,
            arrays: Mapping[str, np.ndarray],
            scene_families: Mapping[str, str],
            *, pairs: np.ndarray,
            pair_group_bits: np.ndarray) -> dict[str, Any]:
    validation = arrays["split_validation"]
    predictions = np.argmax(probabilities, axis=1)
    labels = arrays["action_indices"]
    correct = predictions == labels
    scenes = _decode(arrays["scene_fingerprints"], "Metric scenes")
    episodes = _decode(arrays["episode_ids"], "Metric episodes")
    canonical_pairs = v7._effective_pairs(arrays, validation)
    if not np.array_equal(np.asarray(pairs), canonical_pairs):
        raise ValueError("Diagnostic v8 stratum pair coverage differs")
    pair_bits = _validated_pair_group_bits(canonical_pairs, pair_group_bits)
    if not np.array_equal(pair_bits, _pair_group_bits(arrays, canonical_pairs)):
        raise ValueError("Diagnostic v8 stratum pair groups differ")
    pairs = canonical_pairs
    result: dict[str, Any] = {}
    memberships: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "partner": {}, "family": {}, "critical": {},
    }
    for partner in v7.PARTNERS:
        membership = np.asarray([
            episode.rsplit(":", 1)[-1] == partner for episode in episodes
        ])
        memberships["partner"][partner] = (
            membership,
            membership[pairs[:, 0]] if len(pairs)
            else np.empty(0, dtype=np.bool_),
        )
    for family in sorted(set(scene_families.values())):
        membership = np.asarray([
            scene_families.get(str(scene)) == family for scene in scenes
        ])
        memberships["family"][family] = (
            membership,
            membership[pairs[:, 0]] if len(pairs)
            else np.empty(0, dtype=np.bool_),
        )
    for index, group in enumerate(GROUPS):
        memberships["critical"][group] = (
            (arrays["group_bits"] & (1 << index)) != 0,
            (pair_bits & (1 << index)) != 0,
        )
    for category, values in memberships.items():
        rows = {}
        for name, (membership, pair_membership) in values.items():
            selected_rows = validation & membership
            selected_pairs = pair_membership
            pair_correct = (
                correct[pairs[selected_pairs, 0]]
                & correct[pairs[selected_pairs, 1]]
            ) if np.any(selected_pairs) else np.empty(0, dtype=np.bool_)
            rows[name] = {
                "rows": int(np.sum(selected_rows)),
                "scenes": len(set(map(str, scenes[selected_rows]))),
                "fidelity": float(correct[selected_rows].mean())
                    if np.any(selected_rows) else 0.0,
                "effective_pairs": int(np.sum(selected_pairs)),
                "direction_fidelity": float(pair_correct.mean())
                    if len(pair_correct) else 0.0,
            }
        result[category] = rows
    return result


def _trace_audit(program: R41DiagnosticPublicTreeProgramV8,
                 arrays: Mapping[str, np.ndarray],
                 probabilities: np.ndarray) -> dict[str, Any]:
    validation = arrays["split_validation"]
    bits = arrays["group_bits"]
    labels = arrays["action_indices"]
    candidates: list[int] = []
    selectors = [validation, validation & (labels != ACTIONS.index("WAIT"))]
    selectors.extend(
        validation & ((bits & (1 << index)) != 0)
        for index in range(len(GROUPS))
    )
    for selector in selectors:
        for index in np.flatnonzero(selector)[:4]:
            if int(index) not in candidates:
                candidates.append(int(index))
            if len(candidates) >= TRACE_SAMPLE_LIMIT:
                break
        if len(candidates) >= TRACE_SAMPLE_LIMIT:
            break
    rows = []
    for index in candidates:
        trace = program.trace_batch(arrays["observations"][index:index + 1])[0]
        trace_probabilities = np.asarray(trace["final_probabilities"], dtype=np.float64)
        if (not np.allclose(trace_probabilities, probabilities[index],
                            rtol=0.0, atol=2e-15)
                or trace["prediction_index"] != int(np.argmax(probabilities[index]))):
            raise RuntimeError("Diagnostic v8 trace does not reproduce batch inference")
        rows.append({
            "row_index": index,
            "observation_sha256": bytes(
                arrays["observation_hashes"][index]).decode("ascii"),
            "prediction_index": int(trace["prediction_index"]),
            "prediction": trace["prediction"],
            "triggered_routes": list(trace["triggered_routes"]),
            "trace_sha256": digest(trace),
        })
    if not rows:
        raise ValueError("Diagnostic v8 validation set has no trace sample")
    return {
        "selection": "first four rows from fixed validation strata, capped at 16",
        "rows": rows,
        "all_trace_probabilities_match_batch": True,
        "all_trace_actions_match_batch": True,
    }


def _authenticate_inputs(
    *, actor_path: Path, protocol_path: Path, manifest_path: Path,
    designation_path: Path, expansion_registry_path: Path,
    expected_expansion_registry_sha256: str, expansion_report_path: Path,
    expected_expansion_report_sha256: str, prior_rows_output: Path,
    expected_prior_rows_report_sha256: str, expansion_rows_output: Path,
    expected_expansion_rows_report_sha256: str,
    previous_development_path: Path, selector_report_path: Path,
    expected_selector_report_sha256: str, selector_scope_path: Path,
    selector_fit_only_rows_path: Path, selector_config_registry_path: Path,
    selector_inner_split_audit_path: Path,
    selector_inner_selection_path: Path,
    selector_inner_fit_program_path: Path,
    selector_selected_config_path: Path,
    designation_original_path: Path,
    designation_snapshot_components: Mapping[str, Path],
    designation_original_components: Mapping[str, Path],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    if producer_sources() != dict(sources):
        raise RuntimeError("Diagnostic v8 sources changed during authentication")
    for directory, label in (
        (prior_rows_output, "prior-row reauthentication"),
        (expansion_rows_output, "expansion-row reauthentication"),
    ):
        if (not directory.is_dir() or directory.is_symlink()
                or directory.resolve() != directory):
            raise ValueError(label + " directory is unsafe")
    for expected, actual, label in (
        (expected_expansion_registry_sha256, file_hash(expansion_registry_path),
         "development expansion registry"),
        (expected_expansion_report_sha256, file_hash(expansion_report_path),
         "development expansion report"),
        (expected_selector_report_sha256, file_hash(selector_report_path),
         "fit-only selector report"),
        (expected_prior_rows_report_sha256,
         file_hash(prior_rows_output / "report.json"),
         "prior-row reauthentication report"),
        (expected_expansion_rows_report_sha256,
         file_hash(expansion_rows_output / "report.json"),
         "expansion-row reauthentication report"),
    ):
        if _sha(expected, label) != actual:
            raise ValueError(label + " hash differs")
    actor = NumPyNativeActor(actor_path)
    if actor.obs_dim != BASE_FEATURE_COUNT:
        raise ValueError("Frozen Actor public observation dimension differs")
    protocol = _read_json(protocol_path, "Training protocol")
    manifest = manifest_binding.read_saved_manifest(
        manifest_path, actor_path=actor_path, replay_scope="development")
    designation = _read_json(designation_path, "Diagnostic Actor designation")
    if manifest.get("frozen_actor") != {
        "sha256": actor.artifact_sha256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
    }:
        raise ValueError("Development manifest/Actor identity differs")
    _validate_designation_v2(
        designation_path=designation_path,
        actor_path=actor_path, protocol_path=protocol_path,
        actor=actor, protocol=protocol,
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
    )
    runtime = manifest_binding.build_runtime(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path)
    prior_paths = {
        "saved_report": prior_rows_output / "report.json",
        "saved_rows": prior_rows_output / "rows.npz",
        "embedded_source_report": prior_rows_output / "source_v7_report.json",
        "actor": actor_path,
        "protocol": protocol_path,
        "manifest": manifest_path,
        "manifest_validation": manifest_path.parent / "validation.json",
        "designation": designation_path,
        "supplement": previous_development_path,
        "source_report": prior_rows_output / "source_v7_report.json",
        "source_rows": prior_rows_output / "rows.npz",
    }
    prior_report = prior_rows_api._read_saved_artifacts_snapshot(
        paths=prior_paths,
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=prior_rows_api.producer_sources(),
    )
    if (prior_report.get("version") != prior_rows_api.VERSION
            or prior_report.get("status") != prior_rows_api.STATUS):
        raise ValueError("Exact prior-row reauthentication receipt required")
    expansion = _read_json(expansion_registry_path, "Development expansion registry")
    fit_scenes, validation_scenes = _validate_expansion_registry(
        expansion, registry_path=expansion_registry_path, actor=actor,
        manifest_path=manifest_path, designation_path=designation_path,
        prior_report=prior_report,
    )
    expansion_paths = {
        "saved_report": expansion_rows_output / "report.json",
        "embedded_rows": expansion_rows_output / "expansion_rows.npz",
        "embedded_collection_report": (
            expansion_rows_output / "source_collection_report.json"),
        "actor": actor_path,
        "protocol": protocol_path,
        "manifest": manifest_path,
        "manifest_validation": manifest_path.parent / "validation.json",
        "designation": designation_path,
        "expansion_registry": expansion_registry_path,
        "expansion_report": expansion_report_path,
        "previous": previous_development_path,
        "source_collection_report": (
            expansion_rows_output / "source_collection_report.json"),
        "source_rows": expansion_rows_output / "expansion_rows.npz",
        "prior_rows": prior_rows_output / "rows.npz",
    }
    expansion_rows_report = expansion_rows_api._read_saved_artifacts_snapshot(
        paths=expansion_paths,
        expected_expansion_registry_sha256=expected_expansion_registry_sha256,
        expected_expansion_report_sha256=expected_expansion_report_sha256,
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=expansion_rows_api.producer_sources(),
    )
    if (expansion_rows_report.get("version") != expansion_rows_api.VERSION
            or expansion_rows_report.get("status") != expansion_rows_api.STATUS):
        raise ValueError("Exact expansion-row reauthentication receipt required")
    # Local import makes the selector source part of this producer's source
    # closure while avoiding a module-initialization cycle (the selector uses
    # this module's fit/config primitives).
    from backend.training import (  # noqa: PLC0415
        warehouse_r41_diagnostic_rcpd_v8_fit_selector as selector_api,
    )
    expected_selector_paths = {
        "fit_only_rows.npz": selector_fit_only_rows_path,
        "fit_scope.json": selector_scope_path,
        "config_registry.json": selector_config_registry_path,
        "inner_split_audit.json": selector_inner_split_audit_path,
        "inner_selection.json": selector_inner_selection_path,
        "selected_config.json": selector_selected_config_path,
        "inner_fit_program.json": selector_inner_fit_program_path,
    }
    if any(path.parent != selector_report_path.parent
           for path in expected_selector_paths.values()):
        raise ValueError("Fit-only selector immutable snapshot layout differs")
    selector = selector_api.authenticate_selected_config_snapshot(
        evidence_directory=selector_report_path.parent,
        expected_report_sha256=expected_selector_report_sha256,
        actor_path=actor_path,
        source_full_manifest_bindings=runtime.source_full_manifest_bindings,
        fresh_outer_registry_path=expansion_registry_path,
        expected_fresh_outer_registry_sha256=(
            expected_expansion_registry_sha256),
        fresh_outer_report_path=expansion_report_path,
        expected_fresh_outer_report_sha256=expected_expansion_report_sha256,
    )
    config = normalize_config(selector["config"])
    previous_development = _read_json(
        previous_development_path, "Previous development supplement")
    if (file_hash(previous_development_path)
            != prior_report["bindings"][
                "development_supplement_file_sha256"]):
        raise ValueError("Previous development supplement hash differs")
    relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
    if (len(relations.base_feature_names) != BASE_FEATURE_COUNT
            or len(relations.feature_names) != EXPANDED_FEATURE_COUNT):
        raise ValueError("Diagnostic v8 public feature schema differs")
    return {
        "actor": actor, "protocol": protocol, "manifest": manifest,
        "designation": designation, "prior_report": prior_report,
        "expansion_rows_report": expansion_rows_report,
        "expansion": expansion, "fit_scenes": fit_scenes,
        "validation_scenes": validation_scenes, "config": config,
        "selector": selector,
        "relations": relations,
        "previous_development": previous_development,
        "source_full_manifest_bindings": deepcopy(
            runtime.source_full_manifest_bindings),
    }


def _bindings(
    *, paths: Mapping[str, Path], authenticated: Mapping[str, Any],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    actor: NumPyNativeActor = authenticated["actor"]
    relations: R41DiagnosticPublicRelationsV8 = authenticated["relations"]
    expansion = authenticated["expansion"]
    prior_report = authenticated["prior_report"]
    expansion_rows_report = authenticated["expansion_rows_report"]
    selector = authenticated["selector"]
    return {
        "actor_file_sha256": file_hash(paths["actor"]),
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "actor_feature_names_sha256": digest(list(actor.metadata["feature_names"])),
        "source_full_manifest_bindings": deepcopy(
            authenticated["source_full_manifest_bindings"]),
        "source_full_manifest_bindings_sha256": digest(
            authenticated["source_full_manifest_bindings"]),
        "protocol_file_sha256": file_hash(paths["protocol"]),
        "protocol_content_sha256": digest(authenticated["protocol"]),
        "manifest_file_sha256": file_hash(paths["manifest"]),
        "manifest_content_sha256": manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256,
        "manifest_semantic_sha256": manifest_binding.EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "designation_file_sha256": file_hash(paths["designation"]),
        "designation_semantic_sha256": digest(authenticated["designation"]),
        "prior_rows_reauthentication_receipt_file_sha256": file_hash(
            paths["prior_reauth_report"]),
        "prior_rows_reauthentication_receipt_semantic_sha256": digest(
            prior_report),
        "prior_v7_source_report_file_sha256": file_hash(
            paths["prior_source_report"]),
        "prior_v7_source_report_semantic_sha256": prior_report["bindings"][
            "source_v7_report_semantic_sha256"],
        "prior_v7_rows_file_sha256": file_hash(paths["prior_rows"]),
        "prior_v7_rows_semantic_sha256": prior_report["bindings"][
            "source_v7_rows_semantic_sha256"],
        "expansion_registry_file_sha256": file_hash(paths["expansion_registry"]),
        "expansion_registry_content_sha256": expansion["content_sha256"],
        "expansion_registry_semantic_sha256": digest(expansion),
        "expansion_registry_report_file_sha256": file_hash(
            paths["expansion_registry_report"]),
        "expansion_registry_report_semantic_sha256": (
            expansion_rows_report["bindings"][
                "expansion_report_semantic_sha256"]),
        "expansion_rows_reauthentication_receipt_file_sha256": file_hash(
            paths["expansion_reauth_report"]),
        "expansion_rows_reauthentication_receipt_semantic_sha256": digest(
            expansion_rows_report),
        "expansion_source_collection_report_file_sha256": file_hash(
            paths["expansion_collection_report"]),
        "expansion_source_collection_report_semantic_sha256": (
            expansion_rows_report["bindings"][
                "source_collection_report_semantic_sha256"]),
        "expansion_rows_file_sha256": file_hash(paths["expansion_rows"]),
        "expansion_rows_semantic_sha256": expansion_rows_report["bindings"][
            "source_rows_semantic_sha256"],
        "previous_development_file_sha256": file_hash(
            paths["previous_development"]),
        "previous_development_semantic_sha256": digest(
            authenticated["previous_development"]),
        "fit_selector_report_file_sha256": file_hash(
            paths["selector_report"]),
        "fit_selector_report_semantic_sha256": digest(selector["report"]),
        "fit_selector_scope_file_sha256": file_hash(
            paths["selector_scope"]),
        "fit_selector_scope_content_sha256": selector["scope"][
            "content_sha256"],
        "fit_selector_selected_config_file_sha256": file_hash(
            paths["selector_selected_config"]),
        "fit_selector_selected_config_semantic_sha256": digest(
            selector["selected_config_record"]),
        "fit_selector_selected_config_sha256": digest(
            authenticated["config"]),
        "fit_selector_evidence_artifacts_sha256": digest(
            selector["report"]["evidence_artifacts"]),
        "fit_selector_strict_refit_receipt_sha256": selector[
            "strict_refit_receipt_sha256"],
        "fit_selector_fresh_outer_registry_file_sha256": selector["report"][
            "bindings"]["fresh_outer_registry_file_sha256"],
        "fit_selector_fresh_outer_report_file_sha256": selector["report"][
            "bindings"]["fresh_outer_report_file_sha256"],
        "fit_config_file_sha256": _canonical_json_file_sha256(
            authenticated["config"]),
        "fit_config_content_sha256": digest(authenticated["config"]),
        "public_feature_contract_sha256": digest(relations.contract()),
        "public_feature_registry_sha256": digest(list(relations.feature_names)),
        "pair_weight_version": weight_api.VERSION,
        "pair_weight_contract_sha256": digest(weight_api.contract()),
        "pair_weight_source_sha256": sources[
            Path(weight_api.__file__).resolve().relative_to(ROOT).as_posix()],
        "program_version": PUBLIC_TREE_VERSION,
        "component_program_version": BOOSTED_TREE_VERSION,
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }


def _build_into(
    destination: Path, *, paths: Mapping[str, Path],
    authenticated: Mapping[str, Any], sources: Mapping[str, str],
) -> dict[str, Any]:
    actor = authenticated["actor"]
    relations = authenticated["relations"]
    fit_scenes = authenticated["fit_scenes"]
    validation_scenes = authenticated["validation_scenes"]
    config = authenticated["config"]
    manifest = authenticated["manifest"]
    bindings = _bindings(paths=paths, authenticated=authenticated, sources=sources)
    prior_arrays = _load_npz(paths["prior_rows"], "Prior v7 rows")
    # Reconstruct exact v7 scene lists without touching final trajectories.
    old_train_scenes = deepcopy([
        *manifest["splits"]["train"], *manifest["splits"]["conflict_validation"]
    ])
    previous_path = paths.get("previous_development")
    if previous_path is None:
        raise ValueError("Previous development registry is required")
    previous = _read_json(previous_path, "Previous development supplement")
    old_validation_scenes = deepcopy(previous.get("scenes", []))
    if (file_hash(previous_path)
            != authenticated["prior_report"]["bindings"][
                "development_supplement_file_sha256"]):
        raise ValueError("Previous development supplement hash differs")
    v7._validate_arrays(
        prior_arrays, actor=actor, train_scenes=old_train_scenes,
        validation_scenes=old_validation_scenes,
    )
    expansion_arrays = _load_npz(paths["expansion_rows"], "Expansion rows")
    layout = _validate_expansion_rows(
        expansion_arrays, actor=actor, fit_scenes=fit_scenes,
        validation_scenes=validation_scenes, prior_arrays=prior_arrays,
        relations=relations,
    )
    expansion_fingerprints = {
        str(row["fingerprint"]) for row in (*fit_scenes, *validation_scenes)
    }
    arrays, accounting = _validation_wins_merge(
        prior_arrays, expansion_arrays, layout=layout,
        expansion_fingerprints=expansion_fingerprints,
    )
    prior_fingerprints = set(map(str, _decode(
        prior_arrays["scene_fingerprints"], "Prior scenes")))
    fit_fingerprints = prior_fingerprints | {
        str(row["fingerprint"]) for row in fit_scenes
    }
    validation_fingerprints = {
        str(row["fingerprint"]) for row in validation_scenes
    }
    _validate_combined_rows(
        arrays, actor=actor, fit_fingerprints=fit_fingerprints,
        validation_fingerprints=validation_fingerprints, relations=relations,
    )
    scene_families = _scene_family_map(
        manifest, authenticated["expansion"], previous)
    weight_result = weight_api.build_pair_weights(
        arrays, scene_families=scene_families,
        use_action_factor=config["use_action_factor"],
        pair_pool_multiplier=config["pair_pool_multiplier"],
    )
    arrays["weights"] = weight_result["weights"].copy()
    fit_pair_group_bits = _pair_group_bits(arrays, weight_result["pairs"])
    validation_pairs = v7._effective_pairs(
        arrays, arrays["split_validation"])
    validation_pair_group_bits = _pair_group_bits(arrays, validation_pairs)
    binding = digest({
        "bindings": bindings,
        "combined_rows_semantic_sha256": _arrays_digest(arrays),
        "pair_weights_semantic_sha256": _weights_digest(weight_result),
    })
    program, fit_diagnostics = _fit_program(
        arrays, relations=relations, weights=weight_result,
        config=config, binding=binding,
        source_identity=_program_source_identity(bindings),
        pair_group_bits=fit_pair_group_bits,
    )
    probabilities = _predict_in_batches(program, arrays["observations"])
    metrics = _metrics(
        probabilities, arrays, pairs=validation_pairs,
        pair_group_bits=validation_pair_group_bits)
    gate = _gate(metrics)
    strata = _strata(
        probabilities, arrays, scene_families, pairs=validation_pairs,
        pair_group_bits=validation_pair_group_bits)
    trace_audit = _trace_audit(program, arrays, probabilities)
    program_payload = program.to_dict()
    _validate_program_identity(
        program, relations=relations, config=config, binding=binding,
        source_identity=_program_source_identity(bindings))

    _copy_exclusive(paths["prior_rows"], destination / "prior_v7_rows.npz")
    _copy_exclusive(
        paths["prior_reauth_report"],
        destination / "prior_rows_reauthentication_report.json")
    _copy_exclusive(
        paths["prior_source_report"], destination / "source_v7_report.json")
    _copy_exclusive(paths["expansion_rows"], destination / "expansion_rows.npz")
    _copy_exclusive(
        paths["expansion_collection_report"],
        destination / "source_expansion_collection_report.json")
    _copy_exclusive(
        paths["expansion_reauth_report"],
        destination / "expansion_rows_reauthentication_report.json")
    _copy_exclusive(
        paths["expansion_registry"], destination / "development_expansion.json")
    _copy_exclusive(
        paths["expansion_registry_report"],
        destination / "development_expansion_report.json")
    _copy_exclusive(
        paths["selector_report"], destination / "fit_selector_report.json")
    _copy_exclusive(
        paths["selector_scope"], destination / "fit_selector_scope.json")
    _copy_exclusive(
        paths["selector_selected_config"],
        destination / "fit_selector_selected_config.json")
    _write_json(destination / "fit_config.json", config)
    _write_npz(destination / "rows.npz", arrays)
    _write_npz(destination / "pairs.npz", {
        "pairs": weight_result["pairs"],
        "pair_contribution": weight_result["pair_contribution"],
    })
    _write_json(destination / "weights_audit.json", weight_result["audit"])
    _write_json(destination / "program.json", program_payload)
    if (destination / "program.json").stat().st_size > MAX_PROGRAM_BYTES:
        raise ValueError("Diagnostic v8 explicit program exceeds the evidence limit")
    inputs = {
        "version": VERSION,
        "bindings": bindings,
        "contract": contract(),
        "fit_config": config,
        "sources": sources,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "historical_final_overlap_check_deferred_to_claim": True,
        "formal_ready": False,
    }
    _write_json(destination / "inputs.json", inputs)
    candidate = {
        "version": VERSION,
        "selection_status": (
            "frozen_development_candidate" if gate["passed"]
            else "failed_development_candidate"
        ),
        "binding_sha256": binding,
        "fit_config": config,
        "metrics": metrics,
        "gate": gate,
        "strata": strata,
        "fit_diagnostics": fit_diagnostics,
        "complexity": program.complexity(),
        "trace_audit": trace_audit,
        "program_content_sha256": digest(program_payload),
        "runtime_action_override": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "historical_final_overlap_check_deferred_to_claim": True,
        "formal_ready": False,
    }
    _write_json(destination / "candidate.json", candidate)
    evidence_names = (
        "inputs.json", "prior_rows_reauthentication_report.json",
        "prior_v7_rows.npz", "source_v7_report.json",
        "expansion_rows_reauthentication_report.json", "expansion_rows.npz",
        "source_expansion_collection_report.json",
        "development_expansion.json", "development_expansion_report.json",
        *sorted(_CANDIDATE_SELECTOR_ARTIFACTS),
        "fit_config.json", "rows.npz", "pairs.npz",
        "weights_audit.json", "program.json", "candidate.json",
    )
    evidence = {name: file_hash(destination / name) for name in evidence_names}
    report = {
        "version": VERSION,
        "status": STATUS_PASSED if gate["passed"] else STATUS_FAILED,
        "explanation_eligible": bool(gate["passed"]),
        "formal_ready": False,
        "bindings": bindings,
        "diagnostic_rcpd_binding_sha256": binding,
        "row_accounting": accounting,
        "fit_scene_count": len(fit_fingerprints),
        "validation_scene_count": len(validation_fingerprints),
        "exact_observation_overlap": 0,
        "candidate": candidate,
        "program_file_sha256": evidence["program.json"],
        "program_content_sha256": digest(program_payload),
        "program_json_bytes": (destination / "program.json").stat().st_size,
        "evidence_artifacts": evidence,
        "execution": {
            "actor_queries_for_row_authentication": len(arrays["observations"]),
            "hgb_fits": len(COMPONENTS),
            "ppo_joint_steps": 0,
            "optimizer_updates": 0,
            "actor_changed": False,
            "runtime_action_overrides": 0,
            "final_rows_accessed": False,
            "final_labels_accessed": False,
        },
    }
    if (_bindings(paths=paths, authenticated=authenticated, sources=sources)
            != bindings or producer_sources() != sources):
        raise RuntimeError("Diagnostic v8 inputs or producer sources changed during fit")
    _write_json(destination / "report.json", report)
    return report


def _arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    summary = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        summary[name] = {
            "dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": __import__("hashlib").sha256(
                memoryview(value).cast("B")).hexdigest(),
        }
    return digest(summary)


def _weights_digest(value: Mapping[str, Any]) -> str:
    return digest({
        "weights": _arrays_digest({"weights": value["weights"]}),
        "pairs": _arrays_digest({"pairs": value["pairs"]}),
        "pair_contribution": _arrays_digest({
            "pair_contribution": value["pair_contribution"]}),
        "audit": value["audit"],
    })


def _validate_program_identity(
    program: R41DiagnosticPublicTreeProgramV8, *,
    relations: R41DiagnosticPublicRelationsV8, config: Mapping[str, Any],
    binding: str, source_identity: Mapping[str, Any],
) -> None:
    if (program.action_names != ACTIONS
            or program.base_program.classes != CLASSES
            or program.base_feature_names != relations.base_feature_names
            or program.feature_names != relations.feature_names
            or program.routes != tuple(_fixed_routes()[group] for group in GROUPS)
            or program.mix_weights != config["mix_weights"]
            or program.to_dict().get("aggregation") != AGGREGATION
            or program.metadata.get("diagnostic_rcpd_version") != VERSION
            or program.metadata.get("diagnostic_rcpd_binding_sha256") != binding
            or program.metadata.get("fit_config_sha256") != digest(config)
            or program.metadata.get("public_feature_contract_sha256")
                != digest(relations.contract())
            or program.metadata.get("pair_weight_contract_sha256")
                != digest(weight_api.contract())
            or program.metadata.get("native_source_actor_sha256")
                != source_identity["native_source_actor_sha256"]
            or program.metadata.get("source_actor_parameters_sha256")
                != source_identity["source_actor_parameters_sha256"]
            or program.metadata.get("source_full_manifest_bindings")
                != source_identity["source_full_manifest_bindings"]
            or program.metadata.get("runtime_controller")
                != "native_neural_actor_only"
            or program.metadata.get("program_feedback_into_actor") is not False
            or program.metadata.get("runtime_action_override") is not False):
        raise ValueError("Diagnostic v8 program identity differs")
    for name, component in (
        ("base", program.base_program),
        *((group, program._specialist_programs[group]) for group in GROUPS),
    ):
        metadata = component.metadata
        if (component.classes != CLASSES or component.action_names != ACTIONS
                or component.feature_names != relations.feature_names
                or metadata.get("component") != name
                or metadata.get("diagnostic_rcpd_binding_sha256") != binding
                or metadata.get("fit_config") != config["models"][name]
                or metadata.get("validation_labels_used_for_fit") is not False
                or metadata.get("actor_logits_used_as_program_input") is not False
                or metadata.get("actor_hidden_states_used") is not False
                or metadata.get("runtime_action_override") is not False):
            raise ValueError(name + " explicit program binding differs")


def _program_source_identity(bindings: Mapping[str, Any]) -> dict[str, Any]:
    source_full = bindings.get("source_full_manifest_bindings")
    if (not isinstance(source_full, Mapping)
            or bindings.get("source_full_manifest_bindings_sha256")
                != digest(source_full)):
        raise ValueError("Diagnostic v8 source manifest binding differs")
    return {
        "native_source_actor_sha256": bindings["actor_file_sha256"],
        "source_actor_parameters_sha256": bindings["actor_parameters_sha256"],
        "source_full_manifest_bindings": deepcopy(dict(source_full)),
    }


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    prior_rows_output: str | Path,
    expected_prior_rows_report_sha256: str,
    previous_development_path: str | Path,
    expansion_rows_output: str | Path,
    expected_expansion_rows_report_sha256: str,
    selector_evidence: str | Path,
    expected_selector_report_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    selector_originals, selector_expected = _selector_snapshot_inputs(
        selector_evidence,
        expected_report_sha256=expected_selector_report_sha256)
    originals = {
        "actor": _regular(actor_path, "Frozen Actor"),
        "protocol": _regular(protocol_path, "Training protocol", maximum=MAX_JSON_BYTES),
        "manifest": _regular(manifest_path, "Diagnostic manifest", maximum=MAX_JSON_BYTES),
        "designation": _regular(designation_path, "Actor designation", maximum=MAX_JSON_BYTES),
        "expansion_registry": _regular(
            expansion_registry_path, "Development expansion registry",
            maximum=MAX_JSON_BYTES),
        "expansion_registry_report": _regular(
            expansion_report_path, "Development expansion report",
            maximum=MAX_JSON_BYTES),
        "previous_development": _regular(
            previous_development_path, "Previous development supplement",
            maximum=MAX_JSON_BYTES),
    }
    originals.update(selector_originals)
    originals["manifest_validation"] = _manifest_validation_path(
        originals["manifest"])
    components = _resolved_designation_components(originals["designation"])
    originals.update({"designation_" + name: path
                      for name, path in components.items()})
    if (components["actor"] != originals["actor"]
            or components["protocol"] != originals["protocol"]):
        raise ValueError("Explicit Actor/protocol differ from designation registry")
    prior_output = Path(prior_rows_output).expanduser().absolute()
    expansion_rows_directory = Path(
        expansion_rows_output).expanduser().absolute()
    originals["prior_reauth_report"] = _regular(
        prior_output / "report.json", "Prior-row reauthentication report",
        maximum=MAX_JSON_BYTES)
    originals["prior_rows"] = _regular(
        prior_output / "rows.npz", "Prior v7 rows", maximum=MAX_NPZ_COMPRESSED_BYTES)
    originals["prior_source_report"] = _regular(
        prior_output / "source_v7_report.json", "Historical v7 source report",
        maximum=MAX_JSON_BYTES)
    originals["expansion_reauth_report"] = _regular(
        expansion_rows_directory / "report.json",
        "Expansion-row reauthentication report", maximum=MAX_JSON_BYTES)
    originals["expansion_collection_report"] = _regular(
        expansion_rows_directory / "source_collection_report.json",
        "Expansion source collection report", maximum=MAX_JSON_BYTES)
    originals["expansion_rows"] = _regular(
        expansion_rows_directory / "expansion_rows.npz",
        "Development expansion rows", maximum=MAX_NPZ_COMPRESSED_BYTES)
    destination = Path(output).expanduser().absolute()
    parent = destination.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Diagnostic v8 destination is unsafe or already exists")
    lock = parent / ("." + destination.name + ".lock")
    lock_fd = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with ImmutableInputSnapshot(
            originals,
            expected_sha256=_snapshot_expected_hashes(
                expected_expansion_registry_sha256=_sha(
                    expected_expansion_registry_sha256,
                    "development expansion registry"),
                expected_expansion_report_sha256=_sha(
                    expected_expansion_report_sha256,
                    "development expansion report"),
                expected_prior_rows_report_sha256=_sha(
                    expected_prior_rows_report_sha256,
                    "prior-row reauthentication report"),
                expected_expansion_rows_report_sha256=_sha(
                    expected_expansion_rows_report_sha256,
                    "expansion-row reauthentication report"),
                expected_selector_report_sha256=selector_expected[
                    "selector_report"],
                expected_selector_scope_sha256=selector_expected[
                    "selector_scope"],
                expected_selector_fit_only_rows_sha256=selector_expected[
                    "selector_fit_only_rows"],
                expected_selector_config_registry_sha256=selector_expected[
                    "selector_config_registry"],
                expected_selector_inner_split_audit_sha256=selector_expected[
                    "selector_inner_split_audit"],
                expected_selector_inner_selection_sha256=selector_expected[
                    "selector_inner_selection"],
                expected_selector_inner_fit_program_sha256=selector_expected[
                    "selector_inner_fit_program"],
                expected_selector_selected_config_sha256=selector_expected[
                    "selector_selected_config"],
            ),
            relative_names={
                "manifest": "manifest/manifest.json",
                "manifest_validation": "manifest/validation.json",
                "prior_reauth_report": "prior/report.json",
                "prior_rows": "prior/rows.npz",
                "prior_source_report": "prior/source_v7_report.json",
                "expansion_reauth_report": "expansion/report.json",
                "expansion_collection_report": (
                    "expansion/source_collection_report.json"),
                "expansion_rows": "expansion/expansion_rows.npz",
                "selector_report": "selector/report.json",
                "selector_fit_only_rows": "selector/fit_only_rows.npz",
                "selector_scope": "selector/fit_scope.json",
                "selector_config_registry": "selector/config_registry.json",
                "selector_inner_split_audit": (
                    "selector/inner_split_audit.json"),
                "selector_inner_selection": "selector/inner_selection.json",
                "selector_inner_fit_program": "selector/inner_fit_program.json",
                "selector_selected_config": "selector/selected_config.json",
            },
            maximum_bytes={
                "prior_rows": MAX_NPZ_COMPRESSED_BYTES,
                "expansion_rows": MAX_NPZ_COMPRESSED_BYTES,
                "selector_fit_only_rows": MAX_NPZ_COMPRESSED_BYTES,
            },
            prefix="warehouse-r41-rcpd-v8-inputs-",
        ) as frozen:
            paths = frozen.paths
            snapshot_components = {
                name: paths["designation_" + name] for name in components
            }
            authenticated = _authenticate_inputs(
                actor_path=paths["actor"], protocol_path=paths["protocol"],
                manifest_path=paths["manifest"],
                designation_path=paths["designation"],
                expansion_registry_path=paths["expansion_registry"],
                expected_expansion_registry_sha256=(
                    expected_expansion_registry_sha256),
                expansion_report_path=paths["expansion_registry_report"],
                expected_expansion_report_sha256=(
                    expected_expansion_report_sha256),
                prior_rows_output=paths["prior_reauth_report"].parent,
                expected_prior_rows_report_sha256=(
                    expected_prior_rows_report_sha256),
                expansion_rows_output=paths["expansion_reauth_report"].parent,
                expected_expansion_rows_report_sha256=(
                    expected_expansion_rows_report_sha256),
                previous_development_path=paths["previous_development"],
                selector_report_path=paths["selector_report"],
                expected_selector_report_sha256=(
                    expected_selector_report_sha256),
                selector_scope_path=paths["selector_scope"],
                selector_fit_only_rows_path=paths["selector_fit_only_rows"],
                selector_config_registry_path=paths[
                    "selector_config_registry"],
                selector_inner_split_audit_path=paths[
                    "selector_inner_split_audit"],
                selector_inner_selection_path=paths[
                    "selector_inner_selection"],
                selector_inner_fit_program_path=paths[
                    "selector_inner_fit_program"],
                selector_selected_config_path=paths[
                    "selector_selected_config"],
                designation_original_path=originals["designation"],
                designation_snapshot_components=snapshot_components,
                designation_original_components=components,
                sources=sources,
            )
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError(
                    "Diagnostic v8 sources changed during authentication")
            temporary = Path(tempfile.mkdtemp(
                prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
            os.chmod(temporary, 0o700)
            report = _build_into(
                temporary, paths=paths, authenticated=authenticated,
                sources=sources)
            directory_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            frozen.verify()
            if producer_sources() != sources:
                raise RuntimeError("Diagnostic v8 sources changed during publication")
            if destination.exists() or destination.is_symlink():
                raise ValueError("Diagnostic v8 destination appeared during build")
            os.rename(temporary, destination)
            temporary = None
            try:
                frozen.verify()
                if producer_sources() != sources:
                    raise RuntimeError(
                        "Diagnostic v8 sources changed during publication")
                artifacts = report.get("evidence_artifacts", {})
                if (not isinstance(artifacts, Mapping)
                        or any(file_hash(destination / name) != expected
                               for name, expected in artifacts.items())
                        or file_hash(destination / "report.json")
                            != sha256(
                                (canonical(report) + "\n").encode("utf-8")
                            ).hexdigest()):
                    raise RuntimeError("Published diagnostic v8 evidence differs")
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return deepcopy(report)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)


def _read_saved_report_snapshot(
    output: str | Path, *, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    expected_prior_rows_report_sha256: str,
    previous_development_path: str | Path,
    expected_expansion_rows_report_sha256: str,
    expected_selector_report_sha256: str, require_passed: bool = True,
    refit: bool = True,
    designation_original_path: Path | None = None,
    designation_snapshot_components: Mapping[str, Path] | None = None,
    designation_original_components: Mapping[str, Path] | None = None,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed, and optionally refit all four models from raw evidence."""
    if type(refit) is not bool:
        raise ValueError("refit must be a bool")
    directory = Path(output).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Diagnostic v8 evidence directory is unsafe")
    report_path = _regular(directory / "report.json", "Diagnostic v8 report",
                           maximum=MAX_JSON_BYTES)
    if _sha(expected_report_sha256, "Diagnostic v8 report") != file_hash(report_path):
        raise ValueError("Diagnostic v8 report hash differs")
    expected_names = {
        "inputs.json", "prior_rows_reauthentication_report.json",
        "prior_v7_rows.npz", "source_v7_report.json",
        "expansion_rows_reauthentication_report.json", "expansion_rows.npz",
        "source_expansion_collection_report.json",
        "development_expansion.json", "development_expansion_report.json",
        "fit_config.json", "rows.npz", "pairs.npz",
        "weights_audit.json", "program.json", "candidate.json",
    } | _CANDIDATE_SELECTOR_ARTIFACTS
    integrity_paths = {"saved_report": report_path}
    for name in sorted(expected_names):
        integrity_paths["saved_" + name] = _regular(
            directory / name, "Diagnostic v8 " + name,
            maximum=(MAX_NPZ_COMPRESSED_BYTES if name.endswith(".npz")
                     else MAX_JSON_BYTES),
        )
    integrity_paths.update({
        "external_actor": _regular(actor_path, "Frozen Actor"),
        "external_protocol": _regular(
            protocol_path, "Training protocol", maximum=MAX_JSON_BYTES),
        "external_manifest": _regular(
            manifest_path, "Diagnostic manifest", maximum=MAX_JSON_BYTES),
        "external_designation": _regular(
            designation_path, "Actor designation", maximum=MAX_JSON_BYTES),
        "external_expansion_registry": _regular(
            expansion_registry_path, "Development expansion registry",
            maximum=MAX_JSON_BYTES),
        "external_expansion_registry_report": _regular(
            expansion_report_path, "Development expansion report",
            maximum=MAX_JSON_BYTES),
        "external_previous_development": _regular(
            previous_development_path, "Previous development supplement",
            maximum=MAX_JSON_BYTES),
    })
    integrity_paths["external_manifest_validation"] = (
        _manifest_validation_path(integrity_paths["external_manifest"]))
    if (designation_original_path is None
            or designation_snapshot_components is None
            or designation_original_components is None
            or sources is None):
        raise ValueError("Immutable diagnostic v8 reader snapshot required")
    if producer_sources() != dict(sources):
        raise RuntimeError("Diagnostic v8 sources changed before saved-report read")
    integrity_paths.update({
        "external_designation_" + name: path
        for name, path in designation_snapshot_components.items()
    })
    integrity = _integrity_snapshot(integrity_paths)
    report = _read_json(report_path, "Diagnostic v8 report")
    artifacts = report.get("evidence_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise ValueError("Diagnostic v8 evidence artifact registry differs")
    for name, expected in artifacts.items():
        path = _regular(
            directory / name, "Diagnostic v8 " + name,
            maximum=(MAX_NPZ_COMPRESSED_BYTES if name.endswith(".npz")
                     else MAX_JSON_BYTES),
        )
        if path.parent != directory or _sha(expected, name) != file_hash(path):
            raise ValueError("Diagnostic v8 evidence artifact changed: " + name)
    if report.get("version") != VERSION:
        raise ValueError("Diagnostic v8 report version differs")
    inputs = _read_json(directory / "inputs.json", "Diagnostic v8 inputs")
    config = normalize_config(_read_json(
        directory / "fit_config.json", "Diagnostic v8 fit config"))
    if config != inputs.get("fit_config"):
        raise ValueError("Diagnostic v8 normalized fit config differs")
    # Reauthenticate embedded raw files against caller-frozen identities.
    if (file_hash(directory / "source_expansion_collection_report.json")
            != expansion_rows_api.EXPECTED_SOURCE_COLLECTION_REPORT_SHA256
            or file_hash(directory / "expansion_rows.npz")
                != expansion_rows_api.EXPECTED_SOURCE_ROWS_SHA256
            or file_hash(inputs_path := directory / "inputs.json")
                != artifacts["inputs.json"]):
        raise ValueError("Diagnostic v8 embedded raw evidence differs")
    config_file_hash = inputs.get("bindings", {}).get("fit_config_file_sha256")
    if (config_file_hash != _canonical_json_file_sha256(config)
            or file_hash(directory / "fit_config.json") != config_file_hash
            or inputs["bindings"].get("fit_config_content_sha256")
                != digest(config)):
        raise ValueError("Diagnostic v8 fit-config binding differs")
    external_paths = {
        "actor": _regular(actor_path, "Frozen Actor"),
        "protocol": _regular(protocol_path, "Training protocol", maximum=MAX_JSON_BYTES),
        "manifest": _regular(manifest_path, "Diagnostic manifest", maximum=MAX_JSON_BYTES),
        "designation": _regular(designation_path, "Actor designation", maximum=MAX_JSON_BYTES),
        "expansion_registry": _regular(
            expansion_registry_path, "Development expansion registry",
            maximum=MAX_JSON_BYTES),
        "expansion_registry_report": _regular(
            expansion_report_path, "Development expansion report",
            maximum=MAX_JSON_BYTES),
        "expansion_rows": directory / "expansion_rows.npz",
        "selector_report": directory / "fit_selector_report.json",
        "selector_scope": directory / "fit_selector_scope.json",
        "selector_selected_config": (
            directory / "fit_selector_selected_config.json"),
        "previous_development": _regular(
            previous_development_path, "Previous development supplement",
            maximum=MAX_JSON_BYTES),
        "prior_reauth_report": (
            directory / "prior_rows_reauthentication_report.json"),
        "prior_source_report": directory / "source_v7_report.json",
        "prior_rows": directory / "prior_v7_rows.npz",
        "expansion_reauth_report": (
            directory / "expansion_rows_reauthentication_report.json"),
        "expansion_collection_report": (
            directory / "source_expansion_collection_report.json"),
    }
    # Inputs contain the original config bytes hash, while the canonical config
    # is itself bound semantically.  The reader does not need the external file.
    actor = NumPyNativeActor(external_paths["actor"])
    manifest = manifest_binding.read_saved_manifest(
        external_paths["manifest"], actor_path=external_paths["actor"],
        replay_scope="development")
    protocol = _read_json(external_paths["protocol"], "Training protocol")
    designation = _read_json(external_paths["designation"], "Actor designation")
    if manifest.get("frozen_actor") != {
        "sha256": actor.artifact_sha256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
    }:
        raise ValueError("Development manifest/Actor identity differs")
    _validate_designation_v2(
        designation_path=external_paths["designation"],
        actor_path=external_paths["actor"], protocol_path=external_paths["protocol"],
        actor=actor, protocol=protocol,
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
    )
    runtime = manifest_binding.build_runtime(
        actor_path=external_paths["actor"],
        protocol_path=external_paths["protocol"],
        manifest_path=external_paths["manifest"])
    expansion = _read_json(
        external_paths["expansion_registry"], "Development expansion registry")
    if (file_hash(external_paths["expansion_registry"])
            != _sha(expected_expansion_registry_sha256, "expansion registry")
            or file_hash(directory / "development_expansion.json")
                != expected_expansion_registry_sha256
            or file_hash(external_paths["expansion_registry_report"])
                != _sha(expected_expansion_report_sha256, "expansion report")
            or file_hash(directory / "development_expansion_report.json")
                != expected_expansion_report_sha256
            or expansion.get("content_sha256")
                != inputs["bindings"]["expansion_registry_content_sha256"]):
        raise ValueError("Diagnostic v8 expansion registry binding differs")
    previous = _read_json(
        external_paths["previous_development"], "Previous development supplement")
    if (file_hash(external_paths["previous_development"])
            != inputs["bindings"]["previous_development_file_sha256"]
            or digest(previous)
                != inputs["bindings"]["previous_development_semantic_sha256"]):
        raise ValueError("Diagnostic v8 previous-development binding differs")
    fit_scenes, validation_scenes = _validate_expansion_registry(
        expansion, registry_path=external_paths["expansion_registry"], actor=actor,
        manifest_path=external_paths["manifest"],
        designation_path=external_paths["designation"],
        prior_report={"bindings": {
            "development_supplement_file_sha256": file_hash(
                external_paths["previous_development"])
        }},
    )
    prior_report = prior_rows_api._read_saved_artifacts_snapshot(
        paths={
            "saved_report": external_paths["prior_reauth_report"],
            "saved_rows": external_paths["prior_rows"],
            "embedded_source_report": external_paths["prior_source_report"],
            "actor": external_paths["actor"],
            "protocol": external_paths["protocol"],
            "manifest": external_paths["manifest"],
            "manifest_validation": external_paths["manifest"].parent
                / "validation.json",
            "designation": external_paths["designation"],
            "supplement": external_paths["previous_development"],
            "source_report": external_paths["prior_source_report"],
            "source_rows": external_paths["prior_rows"],
        },
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=prior_rows_api.producer_sources(),
    )
    expansion_rows_report = expansion_rows_api._read_saved_artifacts_snapshot(
        paths={
            "saved_report": external_paths["expansion_reauth_report"],
            "embedded_rows": external_paths["expansion_rows"],
            "embedded_collection_report": external_paths[
                "expansion_collection_report"],
            "actor": external_paths["actor"],
            "protocol": external_paths["protocol"],
            "manifest": external_paths["manifest"],
            "manifest_validation": external_paths["manifest"].parent
                / "validation.json",
            "designation": external_paths["designation"],
            "expansion_registry": external_paths["expansion_registry"],
            "expansion_report": external_paths["expansion_registry_report"],
            "previous": external_paths["previous_development"],
            "source_collection_report": external_paths[
                "expansion_collection_report"],
            "source_rows": external_paths["expansion_rows"],
            "prior_rows": external_paths["prior_rows"],
        },
        expected_expansion_registry_sha256=expected_expansion_registry_sha256,
        expected_expansion_report_sha256=expected_expansion_report_sha256,
        designation_original_path=designation_original_path,
        designation_snapshot_components=designation_snapshot_components,
        designation_original_components=designation_original_components,
        sources=expansion_rows_api.producer_sources(),
    )
    from backend.training import (  # noqa: PLC0415
        warehouse_r41_diagnostic_rcpd_v8_fit_selector as selector_api,
    )
    selector = selector_api.authenticate_embedded_selected_config_snapshot(
        report_path=external_paths["selector_report"],
        expected_report_sha256=expected_selector_report_sha256,
        scope_path=external_paths["selector_scope"],
        selected_config_path=external_paths["selector_selected_config"],
        actor_file_sha256=file_hash(external_paths["actor"]),
        fresh_outer_registry_path=external_paths["expansion_registry"],
        expected_fresh_outer_registry_sha256=(
            expected_expansion_registry_sha256),
        fresh_outer_report_path=external_paths["expansion_registry_report"],
        expected_fresh_outer_report_sha256=expected_expansion_report_sha256,
    )
    if normalize_config(selector["config"]) != config:
        raise ValueError("Diagnostic v8 selector/config binding differs")
    relations = R41DiagnosticPublicRelationsV8(actor.metadata["feature_names"])
    live_authenticated = {
        "actor": actor,
        "protocol": protocol,
        "manifest": manifest,
        "designation": designation,
        "prior_report": prior_report,
        "expansion_rows_report": expansion_rows_report,
        "expansion": expansion,
        "fit_scenes": fit_scenes,
        "validation_scenes": validation_scenes,
        "config": config,
        "selector": selector,
        "relations": relations,
        "previous_development": previous,
        "source_full_manifest_bindings": deepcopy(
            runtime.source_full_manifest_bindings),
    }
    if _bindings(
        paths=external_paths,
        authenticated=live_authenticated,
        sources=sources,
    ) != inputs.get("bindings"):
        raise ValueError("Diagnostic v8 complete input binding differs")
    if (inputs["bindings"].get(
            "prior_rows_reauthentication_receipt_file_sha256")
            != file_hash(external_paths["prior_reauth_report"])
            or inputs["bindings"].get(
                "prior_rows_reauthentication_receipt_semantic_sha256")
                != digest(prior_report)
            or inputs["bindings"].get("prior_v7_source_report_file_sha256")
                != file_hash(external_paths["prior_source_report"])
            or inputs["bindings"].get(
                "prior_v7_source_report_semantic_sha256")
                != prior_report["bindings"][
                    "source_v7_report_semantic_sha256"]
            or inputs["bindings"].get("prior_v7_rows_file_sha256")
                != file_hash(external_paths["prior_rows"])
            or inputs["bindings"].get("prior_v7_rows_semantic_sha256")
                != prior_report["bindings"][
                    "source_v7_rows_semantic_sha256"]
            or inputs["bindings"].get(
                "expansion_rows_reauthentication_receipt_file_sha256")
                != file_hash(external_paths["expansion_reauth_report"])
            or inputs["bindings"].get(
                "expansion_rows_reauthentication_receipt_semantic_sha256")
                != digest(expansion_rows_report)
            or inputs["bindings"].get(
                "expansion_source_collection_report_file_sha256")
                != file_hash(external_paths["expansion_collection_report"])
            or inputs["bindings"].get(
                "expansion_source_collection_report_semantic_sha256")
                != expansion_rows_report["bindings"][
                    "source_collection_report_semantic_sha256"]
            or inputs["bindings"].get("expansion_rows_file_sha256")
                != file_hash(external_paths["expansion_rows"])
            or inputs["bindings"].get("expansion_rows_semantic_sha256")
                != expansion_rows_report["bindings"][
                    "source_rows_semantic_sha256"]
            or inputs["bindings"].get(
                "expansion_registry_report_semantic_sha256")
                != expansion_rows_report["bindings"][
                    "expansion_report_semantic_sha256"]):
        raise ValueError("Diagnostic v8 row reauthentication binding differs")
    if (inputs.get("contract") != contract()
            or inputs.get("sources") != sources
            or inputs["bindings"]["producer_sources_sha256"]
                != digest(sources)
            or inputs["bindings"]["actor_file_sha256"]
                != file_hash(external_paths["actor"])
            or inputs["bindings"]["protocol_file_sha256"]
                != file_hash(external_paths["protocol"])
            or inputs["bindings"]["manifest_file_sha256"]
                != file_hash(external_paths["manifest"])
            or inputs["bindings"]["designation_file_sha256"]
                != file_hash(external_paths["designation"])
            or inputs["bindings"].get("expansion_registry_report_file_sha256")
                != file_hash(external_paths["expansion_registry_report"])
            or inputs["bindings"]["public_feature_contract_sha256"]
                != digest(relations.contract())
            or inputs["bindings"]["public_feature_registry_sha256"]
                != digest(list(relations.feature_names))
            or inputs["bindings"]["pair_weight_contract_sha256"]
                != digest(weight_api.contract())
            or inputs["bindings"]["pair_weight_source_sha256"]
                != sources[
                    Path(weight_api.__file__).resolve().relative_to(ROOT).as_posix()]
            or inputs["bindings"].get("source_full_manifest_bindings")
                != runtime.source_full_manifest_bindings
            or inputs["bindings"].get("source_full_manifest_bindings_sha256")
                != digest(runtime.source_full_manifest_bindings)):
        raise ValueError("Diagnostic v8 input/source binding differs")
    arrays = _load_npz(directory / "rows.npz", "Diagnostic v8 combined rows")
    prior_arrays = _load_npz(directory / "prior_v7_rows.npz", "Prior v7 rows")
    v7._validate_arrays(
        prior_arrays, actor=actor,
        train_scenes=deepcopy([
            *manifest["splits"]["train"],
            *manifest["splits"]["conflict_validation"],
        ]),
        validation_scenes=deepcopy(previous.get("scenes", [])),
    )
    expansion_arrays = _load_npz(directory / "expansion_rows.npz", "Expansion rows")
    layout = _validate_expansion_rows(
        expansion_arrays, actor=actor, fit_scenes=fit_scenes,
        validation_scenes=validation_scenes, prior_arrays=prior_arrays,
        relations=relations,
    )
    expected_arrays, accounting = _validation_wins_merge(
        prior_arrays, expansion_arrays, layout=layout,
        expansion_fingerprints={
            str(row["fingerprint"]) for row in (*fit_scenes, *validation_scenes)
        },
    )
    family_map = _scene_family_map(manifest, expansion, previous)
    weight_result = weight_api.build_pair_weights(
        expected_arrays, scene_families=family_map,
        use_action_factor=config["use_action_factor"],
        pair_pool_multiplier=config["pair_pool_multiplier"],
    )
    expected_arrays["weights"] = weight_result["weights"].copy()
    fit_pair_group_bits = _pair_group_bits(
        expected_arrays, weight_result["pairs"])
    validation_pairs = v7._effective_pairs(
        expected_arrays, expected_arrays["split_validation"])
    validation_pair_group_bits = _pair_group_bits(
        expected_arrays, validation_pairs)
    if (_arrays_digest(arrays) != _arrays_digest(expected_arrays)
            or _read_json(directory / "weights_audit.json", "Weights audit")
                != weight_result["audit"]):
        raise ValueError("Diagnostic v8 merged rows or weights differ")
    with np.load(directory / "pairs.npz", allow_pickle=False) as pair_archive:
        if set(pair_archive.files) != {"pairs", "pair_contribution"}:
            raise ValueError("Diagnostic v8 pair evidence schema differs")
        if (not np.array_equal(pair_archive["pairs"], weight_result["pairs"])
                or not np.array_equal(
                    pair_archive["pair_contribution"],
                    weight_result["pair_contribution"])):
            raise ValueError("Diagnostic v8 pair evidence differs")
    binding = digest({
        "bindings": inputs["bindings"],
        "combined_rows_semantic_sha256": _arrays_digest(expected_arrays),
        "pair_weights_semantic_sha256": _weights_digest(weight_result),
    })
    saved_program = R41DiagnosticPublicTreeProgramV8.from_dict(
        _read_json(directory / "program.json", "Diagnostic v8 program"))
    _validate_program_identity(
        saved_program, relations=relations, config=config, binding=binding,
        source_identity=_program_source_identity(inputs["bindings"]))
    program = saved_program
    fit_diagnostics = report["candidate"]["fit_diagnostics"]
    if refit:
        program, fit_diagnostics = _fit_program(
            expected_arrays, relations=relations, weights=weight_result,
            config=config, binding=binding,
            source_identity=_program_source_identity(inputs["bindings"]),
            pair_group_bits=fit_pair_group_bits,
        )
        if program.to_dict() != saved_program.to_dict():
            raise ValueError("Diagnostic v8 program differs from authenticated refit")
    probabilities = _predict_in_batches(program, expected_arrays["observations"])
    metrics = _metrics(
        probabilities, expected_arrays, pairs=validation_pairs,
        pair_group_bits=validation_pair_group_bits)
    gate = _gate(metrics)
    candidate = {
        "version": VERSION,
        "selection_status": (
            "frozen_development_candidate" if gate["passed"]
            else "failed_development_candidate"
        ),
        "binding_sha256": binding,
        "fit_config": config,
        "metrics": metrics,
        "gate": gate,
        "strata": _strata(
            probabilities, expected_arrays, family_map,
            pairs=validation_pairs,
            pair_group_bits=validation_pair_group_bits),
        "fit_diagnostics": fit_diagnostics,
        "complexity": program.complexity(),
        "trace_audit": _trace_audit(program, expected_arrays, probabilities),
        "program_content_sha256": digest(program.to_dict()),
        "runtime_action_override": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
        "historical_final_overlap_check_deferred_to_claim": True,
        "formal_ready": False,
    }
    stored_candidate = _read_json(
        directory / "candidate.json", "Diagnostic v8 candidate")
    if stored_candidate != candidate or report.get("candidate") != candidate:
        raise ValueError("Diagnostic v8 candidate metrics or trace evidence differs")
    expected_report = {
        "version": VERSION,
        "status": STATUS_PASSED if gate["passed"] else STATUS_FAILED,
        "explanation_eligible": bool(gate["passed"]),
        "formal_ready": False,
        "bindings": inputs["bindings"],
        "diagnostic_rcpd_binding_sha256": binding,
        "row_accounting": accounting,
        "fit_scene_count": len(set(map(str, _decode(
            expected_arrays["scene_fingerprints"][~expected_arrays["split_validation"]],
            "Fit scenes")))),
        "validation_scene_count": len(set(map(str, _decode(
            expected_arrays["scene_fingerprints"][expected_arrays["split_validation"]],
            "Validation scenes")))),
        "exact_observation_overlap": 0,
        "candidate": candidate,
        "program_file_sha256": artifacts["program.json"],
        "program_content_sha256": digest(program.to_dict()),
        "program_json_bytes": (directory / "program.json").stat().st_size,
        "evidence_artifacts": dict(artifacts),
        "execution": {
            "actor_queries_for_row_authentication": len(expected_arrays["observations"]),
            "hgb_fits": len(COMPONENTS),
            "ppo_joint_steps": 0, "optimizer_updates": 0,
            "actor_changed": False, "runtime_action_overrides": 0,
            "final_rows_accessed": False, "final_labels_accessed": False,
        },
    }
    if report != expected_report:
        raise ValueError("Diagnostic v8 report differs from recomputed evidence")
    if require_passed and report["status"] != STATUS_PASSED:
        raise ValueError("Diagnostic v8 did not meet its development gates")
    _verify_integrity(integrity, integrity_paths, phase="saved-report return")
    return deepcopy(report)


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    expansion_registry_path: str | Path,
    expected_expansion_registry_sha256: str,
    expansion_report_path: str | Path,
    expected_expansion_report_sha256: str,
    expected_prior_rows_report_sha256: str,
    previous_development_path: str | Path,
    expected_expansion_rows_report_sha256: str,
    expected_selector_report_sha256: str, require_passed: bool = True,
    refit: bool = True,
) -> dict[str, Any]:
    """Authenticate every input from one immutable, hash-bound snapshot."""
    sources = producer_sources()
    if type(refit) is not bool:
        raise ValueError("refit must be a bool")
    directory = Path(output).expanduser().absolute()
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        raise ValueError("Diagnostic v8 evidence directory is unsafe")
    report_path = _regular(
        directory / "report.json", "Diagnostic v8 report",
        maximum=MAX_JSON_BYTES)
    try:
        report_raw = read_authenticated_bytes(
            report_path, label="Diagnostic v8 report",
            expected_sha256=_sha(
                expected_report_sha256, "Diagnostic v8 report"),
            maximum=MAX_JSON_BYTES,
        )
    except ValueError:
        raise ValueError("Diagnostic v8 report hash differs") from None
    try:
        report_identity = json.loads(
            report_raw.decode("utf-8"),
            object_pairs_hook=_json_pairs("Diagnostic v8 report"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in Diagnostic v8 report: "
                           + token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Diagnostic v8 report must be strict UTF-8 JSON") from error
    expected_names = {
        "inputs.json", "prior_rows_reauthentication_report.json",
        "prior_v7_rows.npz", "source_v7_report.json",
        "expansion_rows_reauthentication_report.json", "expansion_rows.npz",
        "source_expansion_collection_report.json",
        "development_expansion.json", "development_expansion_report.json",
        "fit_config.json", "rows.npz", "pairs.npz",
        "weights_audit.json", "program.json", "candidate.json",
    } | _CANDIDATE_SELECTOR_ARTIFACTS
    artifacts = (report_identity.get("evidence_artifacts")
                 if isinstance(report_identity, Mapping) else None)
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_names
            or any(_HEX.fullmatch(str(value)) is None
                   for value in artifacts.values())):
        raise ValueError("Diagnostic v8 evidence artifact registry differs")
    originals: dict[str, Path] = {
        "saved_report": report_path,
        "external_actor": _regular(actor_path, "Frozen Actor"),
        "external_protocol": _regular(
            protocol_path, "Training protocol", maximum=MAX_JSON_BYTES),
        "external_manifest": _regular(
            manifest_path, "Diagnostic manifest", maximum=MAX_JSON_BYTES),
        "external_designation": _regular(
            designation_path, "Actor designation", maximum=MAX_JSON_BYTES),
        "external_expansion_registry": _regular(
            expansion_registry_path, "Development expansion registry",
            maximum=MAX_JSON_BYTES),
        "external_expansion_registry_report": _regular(
            expansion_report_path, "Development expansion report",
            maximum=MAX_JSON_BYTES),
        "external_previous_development": _regular(
            previous_development_path, "Previous development supplement",
            maximum=MAX_JSON_BYTES),
    }
    originals["external_manifest_validation"] = _manifest_validation_path(
        originals["external_manifest"])
    components = _resolved_designation_components(
        originals["external_designation"])
    if (components["actor"] != originals["external_actor"]
            or components["protocol"] != originals["external_protocol"]):
        raise ValueError("Explicit Actor/protocol differ from designation registry")
    originals.update({"external_designation_" + name: path
                      for name, path in components.items()})
    expected: dict[str, str] = {
        "saved_report": expected_report_sha256,
        "external_actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "external_protocol": (
            designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256),
        "external_manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "external_manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "external_designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "external_expansion_registry": expected_expansion_registry_sha256,
        "external_expansion_registry_report": expected_expansion_report_sha256,
        "external_previous_development": (
            expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256),
        "external_designation_actor": (
            designation_binding.designation.EXPECTED_ACTOR_SHA256),
        "external_designation_protocol": (
            designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256),
        "external_designation_training_ledger": (
            designation_binding.designation.EXPECTED_LEDGER_SHA256),
        "external_designation_dual_evaluation": (
            designation_binding.designation.EXPECTED_DUAL_EVALUATION_SHA256),
        "external_designation_failure_closeout": (
            designation_binding.designation.EXPECTED_CLOSEOUT_SHA256),
    }
    relative_names = {
        "saved_report": "candidate/report.json",
        "external_manifest": "external_manifest/manifest.json",
        "external_manifest_validation": "external_manifest/validation.json",
    }
    maximum_bytes: dict[str, int] = {}
    for name in sorted(expected_names):
        key = "saved_" + name
        originals[key] = _regular(
            directory / name, "Diagnostic v8 " + name,
            maximum=(MAX_NPZ_COMPRESSED_BYTES if name.endswith(".npz")
                     else MAX_JSON_BYTES))
        expected[key] = str(artifacts[name])
        relative_names[key] = "candidate/" + name
        if name.endswith(".npz"):
            maximum_bytes[key] = MAX_NPZ_COMPRESSED_BYTES
    with ImmutableInputSnapshot(
        originals, expected_sha256=expected,
        relative_names=relative_names, maximum_bytes=maximum_bytes,
        prefix="warehouse-r41-rcpd-v8-reader-inputs-",
    ) as frozen:
        snapshot_components = {
            name: frozen.paths["external_designation_" + name]
            for name in components
        }
        result = _read_saved_report_snapshot(
            frozen.root / "candidate",
            expected_report_sha256=expected_report_sha256,
            actor_path=frozen.paths["external_actor"],
            protocol_path=frozen.paths["external_protocol"],
            manifest_path=frozen.paths["external_manifest"],
            designation_path=frozen.paths["external_designation"],
            expansion_registry_path=frozen.paths["external_expansion_registry"],
            expected_expansion_registry_sha256=(
                expected_expansion_registry_sha256),
            expansion_report_path=frozen.paths[
                "external_expansion_registry_report"],
            expected_expansion_report_sha256=(
                expected_expansion_report_sha256),
            expected_prior_rows_report_sha256=(
                expected_prior_rows_report_sha256),
            previous_development_path=frozen.paths[
                "external_previous_development"],
            expected_expansion_rows_report_sha256=(
                expected_expansion_rows_report_sha256),
            expected_selector_report_sha256=expected_selector_report_sha256,
            require_passed=require_passed, refit=refit,
            designation_original_path=originals["external_designation"],
            designation_snapshot_components=snapshot_components,
            designation_original_components=components,
            sources=sources,
        )
        frozen.verify()
        if producer_sources() != sources:
            raise RuntimeError(
                "Diagnostic v8 sources changed during saved-report return")
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--expansion-registry", required=True)
    parser.add_argument("--expected-expansion-registry-sha256", required=True)
    parser.add_argument("--expansion-report", required=True)
    parser.add_argument("--expected-expansion-report-sha256", required=True)
    parser.add_argument("--prior-rows-output", required=True)
    parser.add_argument("--expected-prior-rows-report-sha256", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--expansion-rows-output", required=True)
    parser.add_argument(
        "--expected-expansion-rows-report-sha256", required=True)
    parser.add_argument("--selector-evidence", required=True)
    parser.add_argument("--expected-selector-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        expansion_registry_path=args.expansion_registry,
        expected_expansion_registry_sha256=args.expected_expansion_registry_sha256,
        expansion_report_path=args.expansion_report,
        expected_expansion_report_sha256=args.expected_expansion_report_sha256,
        prior_rows_output=args.prior_rows_output,
        expected_prior_rows_report_sha256=(
            args.expected_prior_rows_report_sha256),
        previous_development_path=args.previous_development,
        expansion_rows_output=args.expansion_rows_output,
        expected_expansion_rows_report_sha256=(
            args.expected_expansion_rows_report_sha256),
        selector_evidence=args.selector_evidence,
        expected_selector_report_sha256=(
            args.expected_selector_report_sha256),
        output=args.output,
    )
    print(canonical({
        "status": result["status"],
        "report": str((Path(args.output).resolve() / "report.json")),
        "program_sha256": result["program_file_sha256"],
        "metrics": result["candidate"]["metrics"],
    }))
    return 0 if result["status"] == STATUS_PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "CONFIG_VERSION", "STATUS_PASSED", "STATUS_FAILED",
    "ACTIONS", "CLASSES", "COMPONENTS", "GROUPS",
    "MIN_OVERALL", "MIN_NONWAIT", "MIN_CRITICAL", "MIN_DIRECTION",
    "contract", "producer_sources", "normalize_config", "build",
    "read_saved_report", "main", "_validation_wins_merge", "_gate",
    "_validate_program_identity",
]
