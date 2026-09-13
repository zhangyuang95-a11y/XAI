"""Release-neutral validators for the fixed diagnostic v8 row evidence.

This module intentionally has no dependency on the v8 fitter or on either row
reauthentication producer.  Those three modules may therefore share the exact
same array and registry validation without forming a producer-source cycle.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as expansion_api
from backend.training import warehouse_r41_diagnostic_rcpd as legacy
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as v7
from backend.training.warehouse_native_common import digest, file_hash
from backend.warehouse_r41_diagnostic_public_features_v8 import R41DiagnosticPublicRelationsV8
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import GROUPS
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor

BASE_FEATURE_COUNT = 197
MAX_NPZ_COMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NPZ_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
_ROW_FIELDS = frozenset(v7._FIELDS)
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > maximum):
        raise ValueError(label + " must be a canonical bounded regular file")
    return path


def arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    summary: dict[str, Any] = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        summary[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": __import__("hashlib").sha256(
                memoryview(value).cast("B")).hexdigest(),
        }
    return digest(summary)


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
    prior_arrays: Mapping[str, np.ndarray] | None,
    relations: R41DiagnosticPublicRelationsV8,
) -> str:
    _validate_base_shapes(arrays, actor, "Development expansion rows")
    scenes = _decode(arrays["scene_fingerprints"], "Expansion scenes")
    split = arrays["split_validation"]
    fit_fingerprints = {str(row["fingerprint"]) for row in fit_scenes}
    validation_fingerprints = {str(row["fingerprint"]) for row in validation_scenes}
    expansion_fingerprints = fit_fingerprints | validation_fingerprints
    prior_fingerprints = (
        set()
        if prior_arrays is None
        else set(map(str, _decode(
            prior_arrays["scene_fingerprints"], "Prior scenes")))
    )
    supplied = set(map(str, scenes))
    if supplied == expansion_fingerprints:
        layout = "expansion_only"
    elif prior_arrays is not None and supplied == (
            expansion_fingerprints | prior_fingerprints):
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
        if prior_arrays is None:  # Defensive: the branch is unreachable above.
            raise ValueError("Combined expansion rows require prior rows")
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



__all__ = [
    "BASE_FEATURE_COUNT", "MAX_NPZ_COMPRESSED_BYTES",
    "MAX_NPZ_EXPANDED_BYTES", "_load_npz", "_decode",
    "arrays_digest",
    "_validate_base_shapes", "_validate_expansion_registry",
    "_validate_expansion_rows",
]
