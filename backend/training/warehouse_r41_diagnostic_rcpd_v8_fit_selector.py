"""One-shot, fit-only hyperparameter selection for diagnostic RCPD v8.

This producer is deliberately narrower than the v8 candidate fitter.  It
removes every row outside a caller-frozen fit registry before validating an
Actor label, fitting a model, or computing a metric.  It then freezes one
scene-grouped inner holdout and compares four pre-registered mixtures of one
fixed shared-charger specialist after one frozen, pair-aligned narrow-passage
fit revision.  No outer-development or final-test metric is an input to the
selection rule.

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


VERSION = "warehouse-r41-diagnostic-rcpd-v8-fit-selector.v2"
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
GATE_THRESHOLDS = {
    "overall": v8.MIN_OVERALL,
    "nonwait": v8.MIN_NONWAIT,
    "critical_narrow_passage": v8.MIN_CRITICAL,
    "critical_shared_pickup": v8.MIN_CRITICAL,
    "critical_shared_charger": v8.MIN_CRITICAL,
    "direction_overall": v8.MIN_DIRECTION,
    "direction_narrow_passage": v8.MIN_DIRECTION,
    "direction_shared_pickup": v8.MIN_DIRECTION,
    "direction_shared_charger": v8.MIN_DIRECTION,
}
NARROW_MODEL = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 5,
    "l2_regularization": 0.1,
    "max_depth": None,
    "max_bins": 255,
    "random_state": 947,
}
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

# Append-only account of fit-only inner-development trials that led to the
# single revision below.  These figures are descriptive evidence, never an
# additional selector input.  In particular, no fresh outer or final Actor row
# was collected or scored while making this choice.
DEVELOPMENT_DIAGNOSIS = {
    "version": "warehouse-r41-diagnostic-rcpd-v8-fit-diagnosis.v1",
    "failed_selector_report_sha256": (
        "3baff28bd2befdf9d55a3200d5706c05d6c8fa4b176057ae85a7827c55ccf4af"
    ),
    "failure": {
        "only_failed_gate": "direction_narrow_passage",
        "fidelity": 0.829456,
        "threshold": v8.MIN_DIRECTION,
        "pair_count": 4556,
        "ordinary_anchor_endpoint_fidelity": 0.85448,
        "changed_branch_endpoint_fidelity": 0.96730,
        "dominant_public_pattern": (
            "collision recovery after a submitted move was cancelled by an "
            "occupied stationary teammate"
        ),
    },
    "fit_only_trials": [
        {
            "name": "full_row_narrow_hgb_70_63",
            "best_direction_narrow_passage": 0.836699,
            "passed": False,
        },
        {
            "name": "full_row_wait_overlap_hgb_100_127",
            "best_direction_narrow_passage": 0.839333,
            "passed": False,
        },
        {
            "name": "pair_endpoint_hgb_70_63_symmetric",
            "best_direction_narrow_passage": 0.837138,
            "passed": False,
        },
        {
            "name": "public_group_direct_routing_and_bit_3_5_specialists",
            "best_direction_narrow_passage_below": 0.845,
            "passed": False,
        },
        {
            "name": "pair_endpoint_hgb_100_127_wait_two_to_one_pool_24",
            "pair_pool_multiplier": 24.0,
            "direction_narrow_passage": 0.8410886742756805,
            "passed": False,
        },
        {
            "name": "pair_endpoint_hgb_100_127_wait_two_to_one_pool_32",
            "pair_pool_multiplier": 32.0,
            "direction_narrow_passage": 0.852721685689201,
            "nine_gate_minimum_margin": 0.002721685689201,
            "passed": True,
        },
        {
            "name": "pair_endpoint_hgb_100_127_wait_two_to_one",
            "mix_weights": {
                "narrow_passage": 1.0,
                "shared_pickup": 1.0,
                "shared_charger": 1.0,
            },
            "nine_gate_minimum_margin": 0.0060140474100087715,
            "metrics": {
                "overall": 0.9193212723187891,
                "nonwait": 0.9132090739138927,
                "critical_narrow_passage": 0.9035665229953109,
                "critical_shared_pickup": 0.915086637031661,
                "critical_shared_charger": 0.9318798134287883,
                "direction_overall": 0.8662744243480788,
                "direction_narrow_passage": 0.8560140474100087,
                "direction_shared_pickup": 0.8661087866108786,
                "direction_shared_charger": 0.8725578988226161,
            },
            "passed": True,
        },
    ],
    "decision": {
        "narrow_fit_population": deepcopy(v8.NARROW_PAIR_ENDPOINT_FIT),
        "narrow_model": deepcopy(NARROW_MODEL),
        "fixed_narrow_mix_weight": 1.0,
        "fixed_pickup_mix_weight": 1.0,
        "charger_mix_candidates": list(MIX_CANDIDATES),
        "pair_pool_multiplier": 16.0,
        "pair_pool_decision": (
            "kept frozen source multiplier because its passing fit-only inner "
            "minimum margin 0.006014047 exceeded pool 32 margin 0.002721686; pool "
            "24 failed"
        ),
    },
    "fit_rows_only": True,
    "inner_validation_used_for_development_selection": True,
    "fresh_outer_rows_accessed": False,
    "fresh_outer_labels_accessed": False,
    "final_rows_accessed": False,
    "final_labels_accessed": False,
}
# The selector is a one-shot continuation of this exact failed v8 candidate.
# These identities are code constants rather than caller-controlled CLI values:
# accepting another source report would reopen the already-used development
# evidence and would also let a caller change the base or pickup capacity, or
# choose a different starting population for the one frozen narrow revision.
FROZEN_SOURCE_V8_REPORT_SHA256 = (
    "2781af5fe845796a99907ceb0bfcc8bd6e0150ad2fe1f4e64ad2d15650bb2eef"
)
FROZEN_SOURCE_V8_ROWS_SHA256 = (
    "d3e8d505aefa5573825a454fc118f7a3b801bbc239498d7416fef744bd2535c0"
)
FROZEN_SOURCE_V8_CONFIG_FILE_SHA256 = (
    "d53e225e6b0a37da68e44e89bc0052629ad3dd064ceb566342c8af34413b3cec"
)
FROZEN_SOURCE_V8_CONFIG_CONTENT_SHA256 = (
    "8a3d26ce27258caae3b8040f61c79a4444c4ca3405c931652431e29752e7e9e9"
)
FROZEN_SOURCE_V8_PROGRAM_SHA256 = (
    "5f2b197e70c67f4f5c4cd672806dcff256f2c3d7b5facaba828413e40937a4a3"
)
FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256 = (
    "00ac0026b0e62c0f6413cf3ba6cd7db849d5d23652a3ef71d580217509425e5d"
)
FROZEN_ACTOR_FILE_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
FROZEN_ACTOR_PARAMETERS_SHA256 = (
    "fc9095d0c0e230f4edf0004d66be42e0d176be166d0b071a0576d9954c057ea3"
)
# This scope was produced label-blind from the fixed source rows and the fresh
# identity-only outer registry.  Pinning it prevents a caller from relabelling
# scenes across the six families and thereby changing the inner split.
FROZEN_FIT_SCOPE_SHA256 = (
    "e578f21ff845a7d5ea15e695f5c5d22aa7533692aa38598963cf71c7a8ea3804"
)
FROZEN_SOURCE_ROW_COUNT = 516_565
FROZEN_SOURCE_SCENE_COUNT = 448
FROZEN_SOURCE_ELIGIBLE_ROW_COUNT = 480_642
FROZEN_SOURCE_CONFIG = {
    "version": v8.CONFIG_VERSION,
    "pair_pool_multiplier": 16.0,
    "use_action_factor": True,
    "models": {
        "base": {
            "learning_rate": 0.1, "max_iter": 70, "max_leaf_nodes": 63,
            "min_samples_leaf": 10, "l2_regularization": 0.1,
            "max_depth": None, "max_bins": 255, "random_state": 941,
        },
        "narrow_passage": {
            "learning_rate": 0.1, "max_iter": 1, "max_leaf_nodes": 2,
            "min_samples_leaf": 10, "l2_regularization": 0.1,
            "max_depth": None, "max_bins": 255, "random_state": 947,
        },
        "shared_pickup": {
            "learning_rate": 0.1, "max_iter": 1, "max_leaf_nodes": 2,
            "min_samples_leaf": 10, "l2_regularization": 0.1,
            "max_depth": None, "max_bins": 255, "random_state": 953,
        },
        "shared_charger": {
            "learning_rate": 0.1, "max_iter": 1, "max_leaf_nodes": 2,
            "min_samples_leaf": 10, "l2_regularization": 0.1,
            "max_depth": None, "max_bins": 255, "random_state": 967,
        },
    },
    "mix_weights": {group: 0.0 for group in v8.GROUPS},
}
FROZEN_SOURCE_IDENTITY = {
    "native_source_actor_sha256": FROZEN_ACTOR_FILE_SHA256,
    "source_actor_parameters_sha256": FROZEN_ACTOR_PARAMETERS_SHA256,
    "source_full_manifest_bindings": {
        "conflict_families_sha256": (
            "2769631844e8038b61b1b330e6afa1e528b91dd026c702926604e2495d1a0129"
        ),
        "conflict_validation_version": (
            "warehouse-r41-diagnostic-conflict-scene-validation.v3"
        ),
        "diagnostic_conflict_graph_sha256": (
            "2c02e6cbbed0b83bde25ddcc6b78911906ffeec82f69f8e3b927f9f137caa2d1"
        ),
        "diagnostic_contract_sha256": (
            "84fce104c5699ca8376f5c9f3fc6bf5a2a8d636b809addee352bc665a7641db0"
        ),
        "diagnostic_contract_version": "warehouse-r41-diagnostic-conflict.v2",
        "manifest_content_sha256": (
            "99bf5a84f409e1411f2ee6da9b8c785aa2bec1fe6fc55161a84b412f260d57c9"
        ),
        "manifest_file_sha256": (
            "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
        ),
        "manifest_semantic_sha256": (
            "fa8875550bfecf2ea5fa3d47634e2f1f3c064fc8ed42db19f63af7c9fd521c2c"
        ),
        "manifest_version": "warehouse-r41-diagnostic-conflict-scene-manifest.v3",
    },
}
if digest(FROZEN_SOURCE_CONFIG) != FROZEN_SOURCE_V8_CONFIG_CONTENT_SHA256:
    raise RuntimeError("Frozen fit-only selector source config constant differs")
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
EVIDENCE_ARTIFACT_NAMES = frozenset((
    "source_v8_report.json", "source_v8_rows.npz",
    "fit_only_rows.npz", "fit_scope.json", "config_registry.json",
    "inner_split_audit.json", "inner_selection.json", "selected_config.json",
    "inner_fit_program.json",
))
SELECTED_CONFIG_FIELDS = frozenset((
    "version", "status", "selected_mix_weight", "selected_config",
    "selected_config_sha256", "outer_evaluation_performed",
    "final_rows_accessed", "final_labels_accessed",
))
STRICT_REFIT_BINDING_FIELDS = frozenset((
    "source_v8_rows_semantic_sha256",
    "fit_only_rows_semantic_sha256", "config_registry_content_sha256",
    "inner_split_audit_content_sha256", "inner_selection_content_sha256",
    "selected_config_content_sha256", "inner_fit_program_content_sha256",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": (
            "one-shot fit-only narrow-pair endpoint revision and "
            "shared-charger config selection"
        ),
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
            "narrow_passage": {
                "model": deepcopy(NARROW_MODEL),
                "fit_population": deepcopy(v8.NARROW_PAIR_ENDPOINT_FIT),
                "mix_weight": 1.0,
            },
            "shared_pickup": {
                "model_unchanged": True,
                "mix_weight": 1.0,
            },
            "shared_charger": {
                "model": deepcopy(CHARGER_MODEL),
                "mix_candidates": list(MIX_CANDIDATES),
            },
            "base_model_unchanged": True,
            "pair_pool_multiplier_unchanged": True,
            "action_factor_unchanged": True,
        },
        "selection": {
            "all_nine_v8_gates_must_pass": True,
            "primary": "maximum minimum margin across all nine v8 gates",
            "gate_thresholds": deepcopy(GATE_THRESHOLDS),
            "tie_break": [
                "lower active specialist count",
                "higher overall fidelity",
                "smaller shared_charger mix weight",
            ],
            "no_candidate_means_stop_before_outer": True,
        },
        "development_diagnosis_sha256": digest(DEVELOPMENT_DIAGNOSIS),
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


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Copy one already-snapshotted input without following either symlink."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(source_fd, "rb", closefd=True) as incoming, \
                os.fdopen(destination_fd, "wb", closefd=True) as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


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
    frozen = v8.normalize_config(FROZEN_SOURCE_CONFIG)
    if source != frozen:
        raise ValueError(
            "Fit-only selector requires the exact frozen v8 source config")
    configs = []
    for weight in MIX_CANDIDATES:
        item = deepcopy(source)
        item["models"]["narrow_passage"] = deepcopy(NARROW_MODEL)
        item["models"]["shared_charger"] = deepcopy(CHARGER_MODEL)
        item["mix_weights"] = {
            "narrow_passage": 1.0,
            "shared_pickup": 1.0,
            "shared_charger": weight,
        }
        configs.append(v8.normalize_config(item))
    return configs


def _config_registry(configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [v8.normalize_config(item) for item in configs]
    preregistered = candidate_configs(FROZEN_SOURCE_CONFIG)
    if normalized != preregistered:
        raise ValueError("Fit-only candidate configs differ from preregistration")
    return {
        "version": VERSION,
        "candidate_count": len(normalized),
        "candidates": [
            {"mix_weight": mix, "config": config,
             "config_sha256": digest(config)}
            for mix, config in zip(MIX_CANDIDATES, normalized)
        ],
        "selection_fields": [
            "models.narrow_passage", "models.shared_charger",
            "mix_weights.narrow_passage", "mix_weights.shared_pickup",
            "mix_weights.shared_charger",
        ],
        "all_other_fields_frozen": True,
    }


def _selected_config_record(
    selection: Mapping[str, Any], configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_weight = selection.get("selected_mix_weight")
    selected_config = next(
        (v8.normalize_config(config)
         for mix, config in zip(MIX_CANDIDATES, configs)
         if mix == selected_weight), None)
    return {
        "version": VERSION,
        "status": selection.get("status"),
        "selected_mix_weight": selected_weight,
        "selected_config": selected_config,
        "selected_config_sha256": (
            digest(selected_config) if selected_config is not None else None),
        "outer_evaluation_performed": False,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }


def _selector_binding(
    *, scope_file_sha256: str, fit_only_rows_semantic_sha256: str,
    source_rows_semantic_sha256: str,
    sources: Mapping[str, str], configs: Sequence[Mapping[str, Any]],
) -> str:
    return digest({
        "version": VERSION,
        "contract_sha256": digest(contract()),
        "source_report_sha256": FROZEN_SOURCE_V8_REPORT_SHA256,
        "source_rows_file_sha256": FROZEN_SOURCE_V8_ROWS_SHA256,
        "source_rows_semantic_sha256": source_rows_semantic_sha256,
        "fit_scope_sha256": scope_file_sha256,
        "fit_only_rows_semantic_sha256": fit_only_rows_semantic_sha256,
        "config_registry_sha256": digest([
            v8.normalize_config(config) for config in configs
        ]),
        "producer_sources_sha256": digest(dict(sources)),
    })


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
    decisions = []
    qualified = []
    for row in candidates:
        weight = float(row["mix_weight"])
        values = _metric_vector(row["metrics"])
        gate = v8._gate(row["metrics"])
        margins = {
            name: values[name] - threshold
            for name, threshold in GATE_THRESHOLDS.items()
        }
        minimum_margin = min(margins.values())
        active_specialists = 2 + int(weight > 0.0)
        checks = {
            "all_nine_v8_gates": bool(gate["passed"]),
            "all_nine_margins_nonnegative": minimum_margin >= -1e-15,
        }
        eligible = all(checks.values())
        decision = {
            "mix_weight": weight,
            "config_sha256": row.get("config_sha256"),
            "validation_probabilities_sha256": row.get(
                "validation_probabilities_sha256"),
            "validation_predictions_sha256": row.get(
                "validation_predictions_sha256"),
            "gate": gate,
            "metric_values": values,
            "gate_margins": margins,
            "minimum_gate_margin": minimum_margin,
            "active_specialist_count": active_specialists,
            "selection_checks": checks,
            "eligible": eligible,
        }
        decisions.append(decision)
        if eligible:
            qualified.append(decision)
    selected = max(
        qualified,
        key=lambda row: (
            row["minimum_gate_margin"],
            -row["active_specialist_count"],
            row["metric_values"]["overall"],
            -row["mix_weight"],
        ),
    ) if qualified else None
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
    """Return one candidate's probabilities on the inner-validation rows.

    ``_metrics_from_probabilities`` accepts a full-row matrix even though it
    scores only its supplied mask.  The unscored rows retain the source Actor's
    class distribution, but must be renormalized after their float32-to-float64
    conversion: source rows use a 2e-6 tolerance while candidate matrices use
    2e-12.  The validation rows are then replaced by the program candidate and
    are the only rows used as candidate evidence.
    """
    config = deepcopy(fit_config)
    config["mix_weights"] = {
        "narrow_passage": 1.0,
        "shared_pickup": 1.0,
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
    selected = np.asarray(validation_mask)
    if (selected.shape != (len(observations),)
            or selected.dtype != np.dtype(np.bool_)):
        raise ValueError("Fit-only candidate validation mask differs")
    result = np.asarray(source_probabilities, dtype=np.float64).copy()
    if (result.shape != (len(observations), len(v8.ACTIONS))
            or not np.isfinite(result).all() or np.any(result < 0.0)):
        raise ValueError("Fit-only source Actor probabilities differ")
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Fit-only source Actor probability mass differs")
    result /= totals
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
            "validation_probabilities_sha256": _arrays_digest({
                "probabilities": np.asarray(
                    probabilities[validation], dtype=np.float64),
            }),
            "validation_predictions_sha256": _arrays_digest({
                "predictions": np.asarray(
                    np.argmax(probabilities[validation], axis=1),
                    dtype=np.uint8),
            }),
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
    if (_sha(expected_report_sha256, "source report")
            != FROZEN_SOURCE_V8_REPORT_SHA256):
        raise ValueError("Fit-only selector source report is not preregistered")
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
    frozen_artifacts = {
        "rows.npz": FROZEN_SOURCE_V8_ROWS_SHA256,
        "fit_config.json": FROZEN_SOURCE_V8_CONFIG_FILE_SHA256,
        "program.json": FROZEN_SOURCE_V8_PROGRAM_SHA256,
        "weights_audit.json": FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256,
    }
    if (report.get("bindings", {}).get("fit_config_content_sha256")
            != FROZEN_SOURCE_V8_CONFIG_CONTENT_SHA256
            or any(artifacts.get(name) != expected
                   for name, expected in frozen_artifacts.items())):
        raise ValueError("Source v8 preregistered artifact binding differs")
    # Use the artifact filename as the key, matching the remaining source
    # evidence mapping.  During ``build`` this path already points into the
    # no-follow ImmutableInputSnapshot, so publication copies the exact report
    # bytes authenticated at transaction start.
    paths = {"report.json": report_path}
    for name in required:
        path = _regular(
            directory / name, "Source v8 " + name,
            maximum=MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES)
        if artifacts.get(name) != file_hash(path):
            raise ValueError("Source v8 evidence artifact differs: " + name)
        paths[name] = path
    if (v8.normalize_config(_read_json(
            paths["fit_config.json"], "Source v8 fit config"))
            != v8.normalize_config(FROZEN_SOURCE_CONFIG)):
        raise ValueError("Source v8 preregistered config differs")
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
    if report_sha256 != FROZEN_SOURCE_V8_REPORT_SHA256:
        raise ValueError("Fit-only selector source report is not preregistered")
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
            or artifacts.get("rows.npz") != FROZEN_SOURCE_V8_ROWS_SHA256
            or artifacts.get("fit_config.json")
                != FROZEN_SOURCE_V8_CONFIG_FILE_SHA256
            or artifacts.get("program.json") != FROZEN_SOURCE_V8_PROGRAM_SHA256
            or artifacts.get("weights_audit.json")
                != FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256
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


def _projected_fit_only_audit(
    arrays: Mapping[str, np.ndarray], *, scope: Mapping[str, Any],
    actor: NumPyNativeActor,
) -> dict[str, Any]:
    """Recompute every derivable inner-split isolation claim."""
    normalized = normalize_scope(scope)
    v8._validate_base_shapes(arrays, actor, "Authenticated fit-only rows")
    scenes = v8._decode(
        arrays["scene_fingerprints"], "Authenticated fit-only scenes")
    split = arrays["split_validation"]
    eligible = set(normalized["eligible_fit_scene_fingerprints"])
    exposed = set(
        normalized["previously_exposed_outer_scene_fingerprints"])
    fresh = set(normalized["fresh_outer_scene_fingerprints"])
    inner, family_counts = _inner_holdout(
        normalized["inner_candidate_scenes"])
    inner_set = set(inner)
    fit_scenes = set(map(str, scenes[~split]))
    validation_scenes = set(map(str, scenes[split]))
    all_scenes = set(map(str, scenes))
    episodes = v8._decode(
        arrays["episode_ids"], "Authenticated fit-only episodes")
    anchors = v8._decode(
        arrays["anchor_ids"], "Authenticated fit-only anchors")
    fit_episodes = set(map(str, episodes[~split]))
    validation_episodes = set(map(str, episodes[split]))
    fit_anchors = {str(value) for value in anchors[~split] if str(value)}
    validation_anchors = {str(value) for value in anchors[split] if str(value)}
    fit_observations = set(map(bytes, arrays["observation_hashes"][~split]))
    validation_observations = set(map(
        bytes, arrays["observation_hashes"][split]))
    if (all_scenes != eligible
            or validation_scenes != inner_set
            or fit_scenes != eligible - inner_set
            or all_scenes & (exposed | fresh)
            or fit_episodes & validation_episodes
            or fit_anchors & validation_anchors
            or fit_observations & validation_observations):
        raise ValueError("Authenticated fit-only inner split differs")
    removed_overlap = FROZEN_SOURCE_ELIGIBLE_ROW_COUNT - len(scenes)
    if removed_overlap < 0:
        raise ValueError("Authenticated fit-only row count exceeds source")
    return {
        "source_rows": FROZEN_SOURCE_ROW_COUNT,
        "source_scenes": FROZEN_SOURCE_SCENE_COUNT,
        "previously_exposed_outer_scenes_removed": len(exposed),
        "fresh_outer_scenes_registered_absent_from_source": len(fresh),
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


def _strict_refit_receipt(
    *, report_file_sha256: str, report: Mapping[str, Any],
) -> str:
    artifacts = report["evidence_artifacts"]
    bindings = report["bindings"]
    return digest({
        "version": VERSION,
        "report_file_sha256": report_file_sha256,
        "selector_binding_sha256": bindings["selector_binding_sha256"],
        "source_v8_report_file_sha256": artifacts["source_v8_report.json"],
        "source_v8_rows_file_sha256": artifacts["source_v8_rows.npz"],
        "source_v8_rows_semantic_sha256": bindings[
            "source_v8_rows_semantic_sha256"],
        "fit_only_rows_file_sha256": artifacts["fit_only_rows.npz"],
        "fit_only_rows_semantic_sha256": bindings[
            "fit_only_rows_semantic_sha256"],
        "config_registry_file_sha256": artifacts["config_registry.json"],
        "config_registry_content_sha256": bindings[
            "config_registry_content_sha256"],
        "inner_split_audit_file_sha256": artifacts["inner_split_audit.json"],
        "inner_split_audit_content_sha256": bindings[
            "inner_split_audit_content_sha256"],
        "inner_selection_file_sha256": artifacts["inner_selection.json"],
        "inner_selection_content_sha256": bindings[
            "inner_selection_content_sha256"],
        "inner_fit_program_file_sha256": artifacts["inner_fit_program.json"],
        "inner_fit_program_content_sha256": bindings[
            "inner_fit_program_content_sha256"],
        "selected_config_file_sha256": artifacts["selected_config.json"],
        "selected_config_content_sha256": bindings[
            "selected_config_content_sha256"],
    })


def _authenticate_frozen_source_projection(
    *, source_report_path: str | Path, source_rows_path: str | Path,
    fit_only_rows_path: str | Path, expected_fit_only_rows_sha256: str,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Recreate the fit-only archive from the fixed failed-v8 source rows."""
    originals = {
        "source_report": _regular(
            source_report_path, "Frozen source v8 report",
            maximum=MAX_JSON_BYTES),
        "source_rows": _regular(
            source_rows_path, "Frozen source v8 rows", maximum=MAX_NPZ_BYTES),
        "fit_only_rows": _regular(
            fit_only_rows_path, "Fit-only selector rows",
            maximum=MAX_NPZ_BYTES),
    }
    expected = {
        "source_report": FROZEN_SOURCE_V8_REPORT_SHA256,
        "source_rows": FROZEN_SOURCE_V8_ROWS_SHA256,
        "fit_only_rows": _sha(
            expected_fit_only_rows_sha256, "fit-only selector rows"),
    }
    with ImmutableInputSnapshot(
        originals, expected_sha256=expected,
        relative_names={
            "source_report": "source/report.json",
            "source_rows": "source/rows.npz",
            "fit_only_rows": "selector/fit_only_rows.npz",
        },
        maximum_bytes={
            "source_report": MAX_JSON_BYTES,
            "source_rows": MAX_NPZ_BYTES,
            "fit_only_rows": MAX_NPZ_BYTES,
        },
        prefix="warehouse-r41-selector-source-projection-",
    ) as frozen:
        source_report = _parse_json_bytes(read_authenticated_bytes(
            frozen.paths["source_report"], label="Frozen source v8 report",
            expected_sha256=FROZEN_SOURCE_V8_REPORT_SHA256,
            maximum=MAX_JSON_BYTES,
        ), "Frozen source v8 report")
        artifacts = source_report.get("evidence_artifacts")
        if (source_report.get("version") != v8.VERSION
                or source_report.get("status") != v8.STATUS_FAILED
                or not isinstance(artifacts, Mapping)
                or artifacts.get("rows.npz") != FROZEN_SOURCE_V8_ROWS_SHA256
                or artifacts.get("fit_config.json")
                    != FROZEN_SOURCE_V8_CONFIG_FILE_SHA256
                or artifacts.get("program.json")
                    != FROZEN_SOURCE_V8_PROGRAM_SHA256
                or artifacts.get("weights_audit.json")
                    != FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256
                or source_report.get("bindings", {}).get("actor_file_sha256")
                    != FROZEN_ACTOR_FILE_SHA256):
            raise ValueError("Frozen source v8 report lineage differs")
        source = _load_npz(
            frozen.paths["source_rows"], "Frozen source v8 rows")
        source_rows_semantic_sha256 = _arrays_digest(source)
        projected, projection = _project_fit_only(source, scope)
        fit_only = _load_npz(
            frozen.paths["fit_only_rows"], "Fit-only selector rows")
        if (set(projected) != set(fit_only)
                or any(projected[name].dtype != fit_only[name].dtype
                       or projected[name].shape != fit_only[name].shape
                       or not np.array_equal(projected[name], fit_only[name])
                       for name in projected)):
            raise ValueError(
                "Fit-only rows are not the exact fixed-source projection")
        projected_semantic_sha256 = _arrays_digest(projected)
        if projected_semantic_sha256 != _arrays_digest(fit_only):
            raise ValueError("Fit-only source projection semantic identity differs")
        frozen.verify()
    return {
        "source_rows_semantic_sha256": source_rows_semantic_sha256,
        "projected_rows_semantic_sha256": projected_semantic_sha256,
        "projection": projection,
    }


def authenticate_embedded_selected_config_snapshot(
    *, report_path: str | Path, expected_report_sha256: str,
    scope_path: str | Path, selected_config_path: str | Path,
    source_report_path: str | Path, source_rows_path: str | Path,
    fit_only_rows_path: str | Path,
    actor_file_sha256: str,
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
) -> dict[str, Any]:
    """Authenticate compact selector metadata and its fixed source projection."""
    report_file = _regular(
        report_path, "Fit-only selector report", maximum=MAX_JSON_BYTES)
    report_sha256 = _sha(expected_report_sha256, "fit-only selector report")
    if file_hash(report_file) != report_sha256:
        raise ValueError("Fit-only selector report hash differs")
    report = _read_json(report_file, "Fit-only selector report")
    artifacts = report.get("evidence_artifacts")
    bindings = report.get("bindings")
    sources = producer_sources()
    if (report.get("version") != VERSION
            or report.get("status") != STATUS_SELECTED
            or report.get("contract") != contract()
            or report.get("development_diagnosis") != DEVELOPMENT_DIAGNOSIS
            or report.get("sources") != sources
            or not isinstance(bindings, Mapping)
            or bindings.get("producer_sources_sha256")
                != digest(sources)
            or any(type(bindings.get(name)) is not str
                   or _HEX.fullmatch(bindings[name]) is None
                   for name in STRICT_REFIT_BINDING_FIELDS)
            or report.get("outer_evaluation_performed") is not False
            or report.get("outer_labels_used_for_projection_fit_or_selection")
                is not False
            or report.get(
                "outer_probabilities_used_for_projection_fit_or_selection")
                is not False
            or report.get("final_rows_accessed") is not False
            or report.get("final_labels_accessed") is not False
            or report.get("runtime_action_override") is not False
            or report.get("actor_changed") is not False
            or report.get("formal_ready") is not False
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != EVIDENCE_ARTIFACT_NAMES
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in artifacts.values())):
        raise ValueError("Fit-only selector report assurance differs")

    scope_file = _regular(
        scope_path, "Fit-only selector scope", maximum=MAX_JSON_BYTES)
    selected_file = _regular(
        selected_config_path, "Fit-only selected config", maximum=MAX_JSON_BYTES)
    source_report_file = _regular(
        source_report_path, "Frozen source v8 report", maximum=MAX_JSON_BYTES)
    source_rows_file = _regular(
        source_rows_path, "Frozen source v8 rows", maximum=MAX_NPZ_BYTES)
    fit_only_rows_file = _regular(
        fit_only_rows_path, "Fit-only selector rows", maximum=MAX_NPZ_BYTES)
    if (file_hash(scope_file) != artifacts["fit_scope.json"]
            or file_hash(selected_file) != artifacts["selected_config.json"]
            or file_hash(source_report_file)
                != artifacts["source_v8_report.json"]
            or file_hash(source_rows_file) != artifacts["source_v8_rows.npz"]
            or file_hash(fit_only_rows_file) != artifacts["fit_only_rows.npz"]
            or artifacts["source_v8_report.json"]
                != FROZEN_SOURCE_V8_REPORT_SHA256
            or artifacts["source_v8_rows.npz"]
                != FROZEN_SOURCE_V8_ROWS_SHA256):
        raise ValueError("Fit-only selector artifact hash differs")
    scope = normalize_scope(_read_json(scope_file, "Fit-only selector scope"))
    selected = _read_json(selected_file, "Fit-only selected config")
    if set(selected) != SELECTED_CONFIG_FIELDS:
        raise ValueError("Fit-only selected-config schema differs")
    if (report_sha256 == ""  # keeps the exact report hash visibly consumed
            or bindings.get("source_v8_report_sha256")
                != FROZEN_SOURCE_V8_REPORT_SHA256
            or bindings.get("source_v8_rows_sha256")
                != FROZEN_SOURCE_V8_ROWS_SHA256
            or bindings.get("source_v8_config_sha256")
                != FROZEN_SOURCE_V8_CONFIG_FILE_SHA256
            or bindings.get("source_v8_program_sha256")
                != FROZEN_SOURCE_V8_PROGRAM_SHA256
            or bindings.get("source_v8_weights_audit_sha256")
                != FROZEN_SOURCE_V8_WEIGHTS_AUDIT_SHA256
            or bindings.get("fit_scope_file_sha256")
                != FROZEN_FIT_SCOPE_SHA256
            or artifacts["fit_scope.json"] != FROZEN_FIT_SCOPE_SHA256
            or bindings.get("fit_scope_file_sha256") != artifacts["fit_scope.json"]
            or bindings.get("fit_scope_content_sha256")
                != scope["content_sha256"]
            or bindings.get("actor_file_sha256")
                != _sha(actor_file_sha256, "candidate Actor")
            or bindings.get("actor_file_sha256") != FROZEN_ACTOR_FILE_SHA256
            or scope["source_report_sha256"]
                != bindings.get("source_v8_report_sha256")
            or scope["source_rows_sha256"]
                != bindings.get("source_v8_rows_sha256")):
        raise ValueError("Fit-only selector scope/source binding differs")

    registry_file = _regular(
        fresh_outer_registry_path, "Fresh outer registry",
        maximum=MAX_JSON_BYTES)
    outer_report_file = _regular(
        fresh_outer_report_path, "Fresh outer report", maximum=MAX_JSON_BYTES)
    registry_sha256 = _sha(
        expected_fresh_outer_registry_sha256, "fresh outer registry")
    outer_report_sha256 = _sha(
        expected_fresh_outer_report_sha256, "fresh outer report")
    if (file_hash(registry_file) != registry_sha256
            or file_hash(outer_report_file) != outer_report_sha256
            or bindings.get("fresh_outer_registry_file_sha256")
                != registry_sha256
            or bindings.get("fresh_outer_report_file_sha256")
                != outer_report_sha256
            or scope["fresh_outer_registry_file_sha256"] != registry_sha256
            or scope["fresh_outer_report_file_sha256"]
                != outer_report_sha256):
        raise ValueError("Fit-only selector fresh-outer file binding differs")
    registry = _read_json(registry_file, "Fresh outer registry")
    outer_report = _read_json(outer_report_file, "Fresh outer report")
    fresh = _validate_fresh_outer_binding(
        registry, outer_report,
        registry_file_sha256=registry_sha256,
        report_file_sha256=outer_report_sha256,
        source_rows_sha256=scope["source_rows_sha256"],
    )
    if (fresh != set(scope["fresh_outer_scene_fingerprints"])
            or bindings.get("fresh_outer_registry_content_sha256")
                != registry["content_sha256"]
            or bindings.get("fresh_outer_report_content_sha256")
                != outer_report["content_sha256"]
            or scope["fresh_outer_registry_content_sha256"]
                != registry["content_sha256"]
            or scope["fresh_outer_report_content_sha256"]
                != outer_report["content_sha256"]):
        raise ValueError("Fit-only selector fresh-outer identity binding differs")

    source_projection = _authenticate_frozen_source_projection(
        source_report_path=source_report_file,
        source_rows_path=source_rows_file,
        fit_only_rows_path=fit_only_rows_file,
        expected_fit_only_rows_sha256=artifacts["fit_only_rows.npz"],
        scope=scope,
    )
    if (bindings.get("source_v8_rows_semantic_sha256")
            != source_projection["source_rows_semantic_sha256"]
            or bindings.get("fit_only_rows_semantic_sha256")
                != source_projection["projected_rows_semantic_sha256"]):
        raise ValueError("Fit-only selector fixed-source projection differs")

    config = v8.normalize_config(selected.get("selected_config"))
    mix = selected.get("selected_mix_weight")
    selection = report.get("selection")
    expected_configs = candidate_configs(FROZEN_SOURCE_CONFIG)
    expected_config = next(
        (candidate for weight, candidate in zip(MIX_CANDIDATES, expected_configs)
         if weight == mix), None)
    if (type(mix) not in (int, float) or isinstance(mix, bool)
            or float(mix) not in MIX_CANDIDATES
            or selected.get("version") != VERSION
            or selected.get("status") != STATUS_SELECTED
            or selected.get("selected_config_sha256") != digest(config)
            or report.get("selected_config_sha256") != digest(config)
            or not isinstance(selection, Mapping)
            or selection.get("status") != STATUS_SELECTED
            or selection.get("selected_mix_weight") != float(mix)
            or selection.get("outer_evaluation_performed") is not False
            or selected.get("outer_evaluation_performed") is not False
            or selected.get("final_rows_accessed") is not False
            or selected.get("final_labels_accessed") is not False
            or config != expected_config
            or config["mix_weights"]["shared_charger"] != float(mix)
            or config["mix_weights"]["narrow_passage"] != 1.0
            or config["mix_weights"]["shared_pickup"] != 1.0
            or config["models"]["narrow_passage"] != NARROW_MODEL
            or config["models"]["shared_charger"] != CHARGER_MODEL
            or config["pair_pool_multiplier"] != 16.0
            or config["use_action_factor"] is not True):
        raise ValueError("Fit-only selector chosen config binding differs")
    return {
        "config": config,
        "report": deepcopy(report),
        "scope": scope,
        "selected_config_record": deepcopy(selected),
        "source_projection": source_projection,
        "report_file_sha256": report_sha256,
        "scope_file_sha256": artifacts["fit_scope.json"],
        "selected_config_file_sha256": artifacts["selected_config.json"],
        "strict_refit_receipt_sha256": _strict_refit_receipt(
            report_file_sha256=report_sha256, report=report),
    }


def authenticate_selected_config_snapshot(
    *, evidence_directory: str | Path | None, expected_report_sha256: str,
    evidence_artifact_paths: Mapping[str, str | Path] | None = None,
    actor_path: str | Path,
    source_full_manifest_bindings: Mapping[str, Any],
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
) -> dict[str, Any]:
    """Authenticate one selector from a single immutable evidence snapshot."""
    if evidence_artifact_paths is None:
        if evidence_directory is None:
            raise ValueError("Fit-only selector evidence directory is required")
        directory = Path(evidence_directory).expanduser().absolute()
        if (not directory.is_dir() or directory.is_symlink()
                or directory.resolve() != directory):
            raise ValueError("Fit-only selector evidence directory is unsafe")
        supplied = {
            "report.json": directory / "report.json",
            **{name: directory / name for name in EVIDENCE_ARTIFACT_NAMES},
        }
    else:
        if evidence_directory is not None or set(evidence_artifact_paths) != {
                "report.json", *EVIDENCE_ARTIFACT_NAMES}:
            raise ValueError("Exact fit-only selector artifact paths required")
        supplied = dict(evidence_artifact_paths)

    report_path = _regular(
        supplied["report.json"], "Fit-only selector report",
        maximum=MAX_JSON_BYTES)
    report_sha256 = _sha(expected_report_sha256, "fit-only selector report")
    report_raw = read_authenticated_bytes(
        report_path, label="Fit-only selector report",
        expected_sha256=report_sha256, maximum=MAX_JSON_BYTES)
    report = _parse_json_bytes(report_raw, "Fit-only selector report")
    artifacts = report.get("evidence_artifacts")
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != EVIDENCE_ARTIFACT_NAMES
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in artifacts.values())):
        raise ValueError("Fit-only selector complete artifact registry differs")

    originals: dict[str, Path] = {
        "selector_report": report_path,
        "actor": _regular(actor_path, "Frozen selector Actor", maximum=MAX_NPZ_BYTES),
        "fresh_outer_registry": _regular(
            fresh_outer_registry_path, "Fresh outer registry",
            maximum=MAX_JSON_BYTES),
        "fresh_outer_report": _regular(
            fresh_outer_report_path, "Fresh outer report",
            maximum=MAX_JSON_BYTES),
    }
    expected = {
        "selector_report": report_sha256,
        "actor": FROZEN_ACTOR_FILE_SHA256,
        "fresh_outer_registry": _sha(
            expected_fresh_outer_registry_sha256, "fresh outer registry"),
        "fresh_outer_report": _sha(
            expected_fresh_outer_report_sha256, "fresh outer report"),
    }
    relative_names = {
        "selector_report": "selector/report.json",
        "fresh_outer_registry": "outer/development_expansion.json",
        "fresh_outer_report": "outer/development_expansion_report.json",
    }
    maximum_bytes: dict[str, int] = {"actor": MAX_NPZ_BYTES}
    frozen_artifact_keys: dict[str, str] = {}
    for name in sorted(EVIDENCE_ARTIFACT_NAMES):
        key = "selector_" + name.replace(".", "_")
        path = _regular(
            supplied[name], "Fit-only selector " + name,
            maximum=MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES)
        originals[key] = path
        expected[key] = str(artifacts[name])
        relative_names[key] = "selector/" + name
        if name.endswith(".npz"):
            maximum_bytes[key] = MAX_NPZ_BYTES
        frozen_artifact_keys[name] = key

    sources = producer_sources()
    with ImmutableInputSnapshot(
        originals, expected_sha256=expected, relative_names=relative_names,
        maximum_bytes=maximum_bytes,
        prefix="warehouse-r41-selector-strict-reader-",
    ) as frozen:
        result = _authenticate_selected_config_snapshot_from_paths(
            evidence_directory=None,
            evidence_artifact_paths={
                "report.json": frozen.paths["selector_report"],
                **{
                    name: frozen.paths[key]
                    for name, key in frozen_artifact_keys.items()
                },
            },
            expected_report_sha256=report_sha256,
            actor_path=frozen.paths["actor"],
            source_full_manifest_bindings=source_full_manifest_bindings,
            fresh_outer_registry_path=frozen.paths["fresh_outer_registry"],
            expected_fresh_outer_registry_sha256=expected[
                "fresh_outer_registry"],
            fresh_outer_report_path=frozen.paths["fresh_outer_report"],
            expected_fresh_outer_report_sha256=expected["fresh_outer_report"],
        )
        frozen.verify()
    if producer_sources() != sources:
        raise RuntimeError("Fit-only selector sources changed during strict refit")
    return result


def _authenticate_selected_config_snapshot_from_paths(
    *, evidence_directory: str | Path | None, expected_report_sha256: str,
    evidence_artifact_paths: Mapping[str, str | Path] | None = None,
    actor_path: str | Path,
    source_full_manifest_bindings: Mapping[str, Any],
    fresh_outer_registry_path: str | Path,
    expected_fresh_outer_registry_sha256: str,
    fresh_outer_report_path: str | Path,
    expected_fresh_outer_report_sha256: str,
) -> dict[str, Any]:
    """Strictly refit and authenticate a fit-only selector decision.

    Every selector artifact is hash checked.  The four preregistered mixtures,
    all nine gate metrics, the selection rule, prediction summaries, and the
    explicit program are then recomputed from the physically projected rows.
    No fresh-outer observation, Actor probability, or label is read.
    """
    if evidence_artifact_paths is None:
        if evidence_directory is None:
            raise ValueError("Fit-only selector evidence directory is required")
        directory = Path(evidence_directory).expanduser().absolute()
        if (not directory.is_dir() or directory.is_symlink()
                or directory.resolve() != directory):
            raise ValueError("Fit-only selector evidence directory is unsafe")
        supplied = {
            "report.json": directory / "report.json",
            **{name: directory / name for name in EVIDENCE_ARTIFACT_NAMES},
        }
    else:
        if evidence_directory is not None or set(evidence_artifact_paths) != {
                "report.json", *EVIDENCE_ARTIFACT_NAMES}:
            raise ValueError("Exact fit-only selector artifact paths required")
        supplied = dict(evidence_artifact_paths)
        directory = None
    report_path = _regular(
        supplied["report.json"], "Fit-only selector report",
        maximum=MAX_JSON_BYTES)
    report_sha256 = _sha(expected_report_sha256, "fit-only selector report")
    report = _read_json(report_path, "Fit-only selector report")
    artifacts = report.get("evidence_artifacts")
    if (file_hash(report_path) != report_sha256
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != EVIDENCE_ARTIFACT_NAMES):
        raise ValueError("Fit-only selector complete artifact registry differs")
    paths: dict[str, Path] = {}
    for name in sorted(EVIDENCE_ARTIFACT_NAMES):
        path = _regular(
            supplied[name], "Fit-only selector " + name,
            maximum=MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES,
        )
        if ((directory is not None and path.parent != directory)
                or file_hash(path) != artifacts.get(name)):
            raise ValueError("Fit-only selector artifact hash differs: " + name)
        paths[name] = path

    summary = authenticate_embedded_selected_config_snapshot(
        report_path=report_path, expected_report_sha256=report_sha256,
        scope_path=paths["fit_scope.json"],
        selected_config_path=paths["selected_config.json"],
        source_report_path=paths["source_v8_report.json"],
        source_rows_path=paths["source_v8_rows.npz"],
        fit_only_rows_path=paths["fit_only_rows.npz"],
        actor_file_sha256=file_hash(actor_path),
        fresh_outer_registry_path=fresh_outer_registry_path,
        expected_fresh_outer_registry_sha256=(
            expected_fresh_outer_registry_sha256),
        fresh_outer_report_path=fresh_outer_report_path,
        expected_fresh_outer_report_sha256=expected_fresh_outer_report_sha256,
    )
    if dict(source_full_manifest_bindings) != FROZEN_SOURCE_IDENTITY[
            "source_full_manifest_bindings"]:
        raise ValueError("Fit-only selector source manifest identity differs")
    actor_file = _regular(actor_path, "Frozen selector Actor", maximum=MAX_NPZ_BYTES)
    if file_hash(actor_file) != FROZEN_ACTOR_FILE_SHA256:
        raise ValueError("Fit-only selector Actor is not preregistered")
    actor = NumPyNativeActor(actor_file)
    if (actor.artifact_sha256 != FROZEN_ACTOR_FILE_SHA256
            or actor.metadata.get("actor_parameters_sha256")
                != FROZEN_ACTOR_PARAMETERS_SHA256):
        raise ValueError("Fit-only selector Actor identity differs")

    configs = candidate_configs(FROZEN_SOURCE_CONFIG)
    config_registry = _read_json(
        paths["config_registry.json"], "Fit-only config registry")
    expected_registry = _config_registry(configs)
    if config_registry != expected_registry:
        raise ValueError("Fit-only candidate registry differs from preregistration")
    fit_only = _load_npz(paths["fit_only_rows.npz"], "Fit-only selector rows")
    projection = _projected_fit_only_audit(
        fit_only, scope=summary["scope"], actor=actor)
    saved_projection = _read_json(
        paths["inner_split_audit.json"], "Fit-only inner split audit")
    if (summary["source_projection"]["projection"] != projection
            or saved_projection != projection
            or report.get("projection") != projection):
        raise ValueError("Fit-only inner split audit differs from rows")

    sources = producer_sources()
    rows_semantic = _arrays_digest(fit_only)
    binding = _selector_binding(
        scope_file_sha256=FROZEN_FIT_SCOPE_SHA256,
        fit_only_rows_semantic_sha256=rows_semantic,
        source_rows_semantic_sha256=summary["source_projection"][
            "source_rows_semantic_sha256"],
        sources=sources, configs=configs)
    bindings = report["bindings"]
    if (report.get("sources") != sources
            or bindings.get("producer_sources_sha256") != digest(sources)
            or bindings.get("selector_binding_sha256") != binding
            or bindings.get("fit_only_rows_semantic_sha256") != rows_semantic
            or bindings.get("config_registry_content_sha256")
                != digest(config_registry)
            or bindings.get("inner_split_audit_content_sha256")
                != digest(saved_projection)):
        raise ValueError("Fit-only selector strict source binding differs")

    scene_families = {
        str(row["fingerprint"]): str(row["family_id"])
        for row in summary["scope"]["inner_candidate_scenes"]
    }
    selection, program, recomputed_configs = _select_projected(
        fit_only, actor=actor, source_config=FROZEN_SOURCE_CONFIG,
        source_identity=FROZEN_SOURCE_IDENTITY, selector_binding=binding,
        scene_families=scene_families)
    if recomputed_configs != configs:
        raise RuntimeError("Fit-only selector refit candidate configs changed")
    saved_selection = _read_json(
        paths["inner_selection.json"], "Fit-only inner selection")
    if (saved_selection != selection or report.get("selection") != selection
            or bindings.get("inner_selection_content_sha256")
                != digest(saved_selection)):
        raise ValueError("Fit-only selector metrics or selection differ from refit")
    saved_program_payload = _read_json(
        paths["inner_fit_program.json"], "Fit-only inner program")
    saved_program = v8.R41DiagnosticPublicTreeProgramV8.from_dict(
        saved_program_payload)
    recomputed_program_payload = program.to_dict()
    if (saved_program.to_dict() != saved_program_payload
            or saved_program_payload != recomputed_program_payload
            or bindings.get("inner_fit_program_content_sha256")
                != digest(saved_program_payload)):
        raise ValueError("Fit-only selector program differs from deterministic refit")
    expected_selected = _selected_config_record(selection, configs)
    if (summary["selected_config_record"] != expected_selected
            or bindings.get("selected_config_content_sha256")
                != digest(expected_selected)
            or selection.get("status") != STATUS_SELECTED):
        raise ValueError("Fit-only selected config differs from refit selection")
    result = dict(summary)
    result.update({
        "selection": deepcopy(selection),
        "projection": deepcopy(projection),
        "strict_refit_performed": True,
        "strict_refit_receipt_sha256": _strict_refit_receipt(
            report_file_sha256=report_sha256, report=report),
    })
    return result


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
    scope_file_sha256 = sha256(
        (canonical(normalized) + "\n").encode("utf-8")).hexdigest()
    if scope_file_sha256 != FROZEN_FIT_SCOPE_SHA256:
        raise ValueError("Prepared fit-only scope differs from preregistration")
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
    if (_sha(expected_fit_scope_sha256, "fit scope")
            != FROZEN_FIT_SCOPE_SHA256
            or file_hash(scope_file) != FROZEN_FIT_SCOPE_SHA256):
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
        source_rows_semantic_sha256 = _arrays_digest(source_arrays)
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
        fit_only_rows_semantic_sha256 = _arrays_digest(fit_only)
        selector_binding = _selector_binding(
            scope_file_sha256=FROZEN_FIT_SCOPE_SHA256,
            fit_only_rows_semantic_sha256=fit_only_rows_semantic_sha256,
            source_rows_semantic_sha256=source_rows_semantic_sha256,
            sources=sources, configs=configs)
        selection, fitted_program, configs = _select_projected(
            fit_only, actor=actor, source_config=source_config,
            source_identity=source_identity, selector_binding=selector_binding,
            scene_families=scene_families)
        config_registry = _config_registry(configs)
        selected_payload = _selected_config_record(selection, configs)
        _write_json(temporary / "fit_scope.json", scope)
        _write_json(temporary / "config_registry.json", config_registry)
        _write_json(temporary / "inner_split_audit.json", projection_audit)
        _write_json(temporary / "inner_selection.json", selection)
        _write_json(temporary / "selected_config.json", selected_payload)
        _write_json(temporary / "inner_fit_program.json", fitted_program.to_dict())
        _copy_exclusive(
            paths["report.json"], temporary / "source_v8_report.json")
        _copy_exclusive(
            paths["rows.npz"], temporary / "source_v8_rows.npz")
        evidence_names = (
            "source_v8_report.json", "source_v8_rows.npz",
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
            "development_diagnosis": deepcopy(DEVELOPMENT_DIAGNOSIS),
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
                "source_v8_rows_semantic_sha256": (
                    source_rows_semantic_sha256),
                "fit_only_rows_semantic_sha256": (
                    fit_only_rows_semantic_sha256),
                "config_registry_content_sha256": digest(config_registry),
                "inner_split_audit_content_sha256": digest(projection_audit),
                "inner_selection_content_sha256": digest(selection),
                "selected_config_content_sha256": digest(selected_payload),
                "inner_fit_program_content_sha256": digest(
                    fitted_program.to_dict()),
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
    if (_sha(expected_source_report_sha256, "source report")
            != FROZEN_SOURCE_V8_REPORT_SHA256
            or _sha(expected_fit_scope_sha256, "fit scope")
                != FROZEN_FIT_SCOPE_SHA256):
        raise ValueError("Fit-only selector preregistered source/scope differs")
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
    "INNER_HOLDOUT_FAMILY_QUOTAS", "INNER_ORDER_SALT", "GATE_THRESHOLDS",
    "NARROW_MODEL", "CHARGER_MODEL", "DEVELOPMENT_DIAGNOSIS",
    "FROZEN_SOURCE_V8_REPORT_SHA256", "FROZEN_FIT_SCOPE_SHA256",
    "FROZEN_SOURCE_CONFIG",
    "contract", "producer_sources", "normalize_scope", "candidate_configs",
    "choose_candidate", "prepare_scope", "authenticate_selected_config_snapshot",
    "authenticate_embedded_selected_config_snapshot",
    "build", "main", "_inner_holdout",
    "_project_fit_only", "_arrays_digest",
]
