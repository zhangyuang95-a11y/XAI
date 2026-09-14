"""Publish the label-blind ordered observation-hash projection for v13 outer.

The fresh outer scene identities must already be frozen by the v13 registry.
This producer then replays one fixed schedule with the frozen Actor, but it
publishes only one-way observation/row-identity hashes and a receipt.  Raw
observations, Actor actions, and Actor probabilities never enter either
published JSON document.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_v7
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_split as registry_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import ImmutableInputSnapshot
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-outer-hash-projection.v13"
STATUS = "frozen_label_blind_ordered_outer_hash_projection"
RECEIPT_VERSION = VERSION + ".receipt.v1"
SCENE_OFFSET = registry_api.SCENE_OFFSET
SCENE_COUNT = registry_api.FRESH_OUTER_SCENE_COUNT
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
PROJECTION_NAME = "ordered_outer_observation_hashes.json"
RECEIPT_NAME = "projection_receipt.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PROJECTION_FIELDS = frozenset((
    "version", "status", "identity", "schedule", "projection",
    "information_boundary", "content_sha256",
))
_IDENTITY_FIELDS = frozenset((
    "registry_file_sha256", "registry_content_sha256",
    "selected_identity_sha256",
))
_SCHEDULE_FIELDS = frozenset((
    "scene_offset", "scene_count", "partners", "dense_critical",
    "critical_anchor_period", "environment_steps",
))
_PAYLOAD_FIELDS = frozenset((
    "row_count", "unique_observation_count", "ordered_observation_hashes",
    "ordered_observation_hashes_sha256", "unique_observation_hashes",
    "unique_observation_hashes_sha256", "ordered_row_identity_hashes",
    "ordered_row_identity_hashes_sha256", "ordered_replay_sha256",
))
_BOUNDARY_FIELDS = frozenset((
    "registry_identity_frozen_before_replay",
    "actor_inference_used_only_to_advance_fixed_replay",
    "raw_observations_published", "actor_actions_published",
    "actor_probabilities_published", "program_accessed",
    "protected_final_access", "participant_data_accessed",
    "runtime_action_override", "formal_ready",
))
_RECOVERY_BOUNDARY = registry_api.RECOVERY_BOUNDARY
_RECOVERY_BINDINGS = registry_api.RECOVERY_BINDINGS


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "population": "fresh v13 identity-frozen development outer only",
        "scene_count": SCENE_COUNT,
        "scene_offset": SCENE_OFFSET,
        "partners": list(rows_v7.PARTNERS),
        "dense_critical": False,
        "critical_anchor_period": 5,
        "fixed_schedule": True,
        "published_payload": [
            "ordered observation SHA-256", "sorted unique observation SHA-256",
            "ordered row-identity SHA-256", "registry and replay identity",
        ],
        "raw_observations_published": False,
        "actor_actions_published": False,
        "actor_probabilities_published": False,
        "program_access": False,
        "protected_final_access": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: child for key, child in value.items()
                                   if key != "content_sha256"}))


def _frozen_sources_valid(value: Any, claimed_sha256: Any) -> bool:
    """Validate a producer closure captured when an artifact was written.

    Historical evidence is authenticated by its caller-supplied file hash and
    immutable content digest.  Its frozen source receipt must remain internally
    valid, but later serving or audit code may evolve without invalidating the
    already anchored evidence.
    """
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            type(path) is str and bool(path)
            and type(source_sha256) is str
            and _HEX.fullmatch(source_sha256) is not None
            for path, source_sha256 in value.items()
        )
        and type(claimed_sha256) is str
        and _HEX.fullmatch(claimed_sha256) is not None
        and digest(dict(value)) == claimed_sha256
    )


def _regular(value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a nonempty bounded canonical regular file")
    return path


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), label)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_snapshot(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    registry_path: str | Path, registry_report_path: str | Path,
    expected_registry_sha256: str, expected_registry_report_sha256: str,
) -> tuple[ImmutableInputSnapshot, dict[str, Path], dict[str, Path], Path]:
    actor = _regular(actor_path, "frozen Actor", maximum=MAX_NPZ_BYTES)
    protocol = _regular(protocol_path, "frozen protocol")
    manifest = _regular(manifest_path, "runtime manifest")
    designation = _regular(designation_path, "Actor designation")
    registry = _regular(registry_path, "fresh v13 outer registry")
    registry_report = _regular(registry_report_path, "fresh v13 outer registry report")
    components = designation_binding.resolve_bound_components(designation)
    if components["actor"] != actor or components["protocol"] != protocol:
        raise ValueError("Explicit Actor/protocol must be designation components")
    validation = _regular(manifest.parent / "validation.json", "manifest validation")
    expected = {
        "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
        "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "manifest": manifest_binding.EXPECTED_MANIFEST_SHA256,
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation": designation_binding.EXPECTED_DESIGNATION_SHA256,
        "registry": _sha(expected_registry_sha256, "fresh v13 registry"),
        "registry_report": _sha(
            expected_registry_report_sha256, "fresh v13 registry report"),
    }
    originals: dict[str, Path] = {
        "actor": actor, "protocol": protocol, "manifest": manifest,
        "manifest_validation": validation, "designation": designation,
        "registry": registry, "registry_report": registry_report,
    }
    for name, path in components.items():
        key = "designation_" + name
        originals[key] = path
        expected[key] = {
            "actor": designation_binding.designation.EXPECTED_ACTOR_SHA256,
            "protocol": designation_binding.designation.EXPECTED_PROTOCOL_FILE_SHA256,
            "training_ledger": designation_binding.designation.EXPECTED_LEDGER_SHA256,
            "dual_evaluation": designation_binding.designation.EXPECTED_DUAL_EVALUATION_SHA256,
            "failure_closeout": designation_binding.designation.EXPECTED_CLOSEOUT_SHA256,
        }[name]
    snapshot = ImmutableInputSnapshot(
        originals, expected_sha256=expected,
        relative_names={"manifest": "manifest/manifest.json",
                        "manifest_validation": "manifest/validation.json"},
        maximum_bytes={"actor": MAX_NPZ_BYTES,
                       "designation_actor": MAX_NPZ_BYTES},
        prefix="warehouse-r41-v13-outer-projection-inputs-",
    )
    return snapshot, snapshot.paths, components, designation


def _validate_registry(
    *, paths: Mapping[str, Path], component_originals: Mapping[str, Path],
    designation_original: Path,
) -> tuple[NumPyNativeActor, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    designation = designation_binding.read_bound_designation_snapshot(
        paths["designation"], original_path=designation_original,
        components={name: paths["designation_" + name]
                    for name in component_originals},
        original_components=component_originals,
        expected_sha256=designation_binding.EXPECTED_DESIGNATION_SHA256,
    )
    actor = NumPyNativeActor(paths["actor"])
    registry = _strict_json(paths["registry"], "fresh v13 outer registry")
    report = _strict_json(paths["registry_report"], "fresh v13 outer registry report")
    registry_sources = registry.get("producer_sources")
    scenes = registry.get("development_outer")
    identities = registry.get("selected_outer_identities")
    if (registry.get("version") != registry_api.VERSION
            or registry.get("status") != registry_api.STATUS
            or not _content_valid(registry)
            or registry.get("contract") != registry_api.contract()
            or not isinstance(scenes, list) or len(scenes) != SCENE_COUNT
            or not isinstance(identities, list) or len(identities) != SCENE_COUNT
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("action_labels_access") is not False
            or registry.get("probabilities_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or registry.get("formal_ready") is not False
            or not _frozen_sources_valid(
                registry_sources, registry.get("producer_sources_sha256"))):
        raise ValueError("Exact identity-frozen fresh v13 outer registry required")
    public_identities: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, (scene, identity) in enumerate(zip(scenes, identities)):
        if not isinstance(scene, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("Fresh v13 outer scene identity differs")
        public = {key: identity.get(key) for key in (
            "batch_index", "family_id", "seed", "fingerprint")}
        if (any(type(public[key]) is not int for key in ("batch_index", "seed"))
                or type(public["family_id"]) is not str
                or type(public["fingerprint"]) is not str
                or _HEX.fullmatch(public["fingerprint"]) is None
                or scene.get("seed") != public["seed"]
                or scene.get("fingerprint") != public["fingerprint"]
                or scene.get("family_id") != public["family_id"]
                or scene.get("id") != f"diagnostic_v13_fresh_outer_{index:04d}"
                or (public["seed"], public["fingerprint"]) in seen):
            raise ValueError("Fresh v13 outer materialised identity differs")
        seen.add((public["seed"], public["fingerprint"]))
        public_identities.append(public)
    bindings = registry.get("bindings")
    exclusion_counts = registry.get("exclusion_counts")
    exclusion_digests = registry.get("exclusion_digests")
    statistics = registry.get("statistics")
    boundary = registry.get("information_boundary")
    selection = report.get("selection")
    recovery_bindings_valid = (
        isinstance(bindings, Mapping)
        and set(bindings) == _RECOVERY_BINDINGS
        and all(
            type(bindings.get(name)) is str
            and _HEX.fullmatch(bindings[name]) is not None
            for name in _RECOVERY_BINDINGS
        )
    )
    recovery_statistics_valid = (
        isinstance(exclusion_counts, Mapping)
        and isinstance(exclusion_digests, Mapping)
        and isinstance(statistics, Mapping)
        and exclusion_counts.get("consumed_v9_outer_identities")
            == registry_api.FRESH_OUTER_SCENE_COUNT
        and exclusion_counts.get("consumed_v10_outer_identities")
            == registry_api.FRESH_OUTER_SCENE_COUNT
        and exclusion_counts.get("consumed_v11_outer_identities")
            == registry_api.FRESH_OUTER_SCENE_COUNT
        and exclusion_counts.get("consumed_v12_outer_identities")
            == registry_api.FRESH_OUTER_SCENE_COUNT
        and exclusion_counts.get("burned_v12_final_identities")
            == registry_api.FRESH_OUTER_SCENE_COUNT
        and exclusion_counts.get("union_candidate_identities_excluded") == 785
        and all(statistics.get(name) == value
                for name, value in exclusion_counts.items())
        and statistics.get("fixed_candidate_scene_count")
            == registry_api.FIXED_CANDIDATE_SCENE_COUNT
        and statistics.get("remaining_candidate_scene_count")
            == registry_api.EXPECTED_REMAINING_SCENE_COUNT
        and statistics.get("remaining_family_counts")
            == registry_api.EXPECTED_REMAINING_FAMILY_COUNTS
        and statistics.get("selected_outer_scene_count") == SCENE_COUNT
        and statistics.get("selected_outer_family_counts")
            == registry_api.FAMILY_QUOTAS
        and statistics.get("selected_exposed_seed_overlap") == 0
        and statistics.get("selected_exposed_fingerprint_overlap") == 0
        and type(statistics.get(
            "combined_promoted_unique_observation_count")) is int
        and statistics["combined_promoted_unique_observation_count"] > 0
        and exclusion_digests.get("consumed_v9_outer_identities_sha256")
            == bindings.get("consumed_v9_selected_identity_sha256")
        and exclusion_digests.get("consumed_v10_outer_identities_sha256")
            == bindings.get("consumed_v10_selected_identity_sha256")
        and exclusion_digests.get("consumed_v11_outer_identities_sha256")
            == bindings.get("consumed_v11_selected_identity_sha256")
        and exclusion_digests.get(
            "consumed_v10_outer_observation_hashes_sha256")
            == bindings.get("consumed_v10_outer_observation_hashes_sha256")
        and exclusion_digests.get(
            "consumed_v11_outer_observation_hashes_sha256")
            == bindings.get("consumed_v11_outer_observation_hashes_sha256")
        and exclusion_digests.get("consumed_v12_outer_identities_sha256")
            == bindings.get("consumed_v12_outer_selected_identity_sha256")
        and exclusion_digests.get(
            "consumed_v12_outer_observation_hashes_sha256")
            == bindings.get("consumed_v12_outer_observation_hashes_sha256")
        and exclusion_digests.get("burned_v12_final_identities_sha256")
            == bindings.get("burned_v12_final_selected_identity_sha256")
        and exclusion_digests.get(
            "burned_v12_final_observation_hashes_sha256")
            == bindings.get("burned_v12_final_observation_hashes_sha256")
        and isinstance(statistics.get("hash_screen"), Mapping)
        and statistics["hash_screen"].get(
            "selected_historical_observation_overlap") == 0
        and statistics["hash_screen"].get("selected_scene_count")
            == SCENE_COUNT
    )
    if (not isinstance(bindings, Mapping)
            or not recovery_bindings_valid
            or not recovery_statistics_valid
            or bindings.get("actor_sha256") != file_hash(paths["actor"])
            or bindings.get("actor_parameters_sha256")
                != actor.metadata["actor_parameters_sha256"]
            or bindings.get("manifest_file_sha256") != file_hash(paths["manifest"])
            or bindings.get("contract_sha256")
                != digest(registry_api.contract())
            or report.get("version") != registry_api.VERSION
            or report.get("status") != registry_api.STATUS
            or not _content_valid(report)
            or report.get("registry_file_sha256") != file_hash(paths["registry"])
            or report.get("registry_content_sha256") != registry["content_sha256"]
            or report.get("bindings") != bindings
            or not isinstance(selection, Mapping)
            or selection.get("salt") != registry_api.SELECTION_SALT
            or selection.get("family_quotas") != registry_api.FAMILY_QUOTAS
            or selection.get("exclusion_digests") != exclusion_digests
            or report.get("statistics") != statistics
            or boundary != _RECOVERY_BOUNDARY
            or report.get("information_boundary") != boundary
            or report.get("producer_sources") != registry_sources
            or not _frozen_sources_valid(
                report.get("producer_sources"),
                report.get("producer_sources_sha256"))
            or selection.get("selected_identity_sha256")
                != digest(public_identities)
            or report.get("formal_ready") is not False
            or designation.get("bindings", {}).get("actor_sha256")
                != actor.artifact_sha256):
        raise ValueError("Fresh v13 outer registry/report binding differs")
    return actor, [deepcopy(dict(row)) for row in scenes], registry, report


def _replay_outer(
    *, actor_path: Path, protocol_path: Path, manifest_path: Path,
    scenes: Sequence[Mapping[str, Any]], progress_label: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    runtime = manifest_binding.build_runtime(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path)
    rows, environment_steps = rows_v7._collect(
        runtime, scenes, scene_offset=SCENE_OFFSET, dense_critical=False,
        progress_label=progress_label)
    arrays, accounting = _projection_rows_to_arrays(rows)
    actor = NumPyNativeActor(actor_path)
    _validate_projection_replay_arrays(arrays, actor=actor, scenes=scenes)
    return arrays, accounting, environment_steps


def _projection_rows_to_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Encode one validation-only replay without invoking fit-weight logic.

    The legacy evidence encoder expects at least one development-fit row because
    it normalises class-balancing weights.  A fresh outer projection has no fit
    partition by construction.  Its placeholder weights exist only to retain
    the authenticated row schema and are never published or used for fitting.
    """
    if not rows:
        raise ValueError("Fresh outer projection replay is empty")
    actions = rows_v7.ACTIONS
    arrays = {
        "observations": np.stack(
            [row["observation"] for row in rows]).astype(np.float32),
        "probabilities": np.stack(
            [row["probabilities"] for row in rows]).astype(np.float32),
        "action_indices": np.asarray(
            [actions.index(row["action"]) for row in rows], dtype=np.uint8),
        "weights": np.ones(len(rows), dtype=np.float32),
        "observation_hashes": np.asarray([
            rows_v7.legacy._obs_hash(row["observation"]) for row in rows
        ], dtype="S64"),
        "scene_fingerprints": np.asarray(
            [row["scene"] for row in rows], dtype="S64"),
        "episode_ids": np.asarray(
            [row["episode"] for row in rows], dtype="S180"),
        "frames": np.asarray([row["frame"] for row in rows], dtype=np.int16),
        "group_bits": np.asarray([
            rows_v7.legacy._group_bits(row["groups"]) for row in rows
        ], dtype=np.uint8),
        "kinds": np.asarray([row["kind"] for row in rows], dtype="S16"),
        "anchor_ids": np.asarray(
            [row["anchor"] for row in rows], dtype="S240"),
        "branch_actions": np.asarray(
            [row["branch_action"] for row in rows], dtype="S8"),
        "physical_hashes": np.asarray(
            [row["physical_hash"] for row in rows], dtype="S64"),
        "source_state_hashes": np.asarray(
            [row["source_state_hash"] for row in rows], dtype="S64"),
        "submitted_equal": np.asarray(
            [row["submitted_equal"] for row in rows], dtype=np.bool_),
        "trajectory_done": np.asarray(
            [row["trajectory_done"] for row in rows], dtype=np.bool_),
        "split_validation": np.ones(len(rows), dtype=np.bool_),
    }
    accounting = {
        "raw_train_rows": 0,
        "raw_validation_rows": len(rows),
        "raw_collection_environment_steps": 0,
    }
    return arrays, accounting


def _validate_projection_replay_arrays(
    arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
    scenes: Sequence[Mapping[str, Any]],
) -> None:
    """Validate Actor parity and fixed replay coordinates without fit weights."""
    if set(arrays) != rows_v7._FIELDS:
        raise ValueError("Fresh outer projection row schema differs")
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
            or arrays["observations"].shape != (count, actor.obs_dim)
            or arrays["observations"].dtype != np.dtype(np.float32)
            or arrays["probabilities"].shape != (count, len(rows_v7.ACTIONS))
            or arrays["probabilities"].dtype != np.dtype(np.float32)
            or any(arrays[name].shape != (count,)
                   or arrays[name].dtype != dtype
                   for name, dtype in expected_dtypes.items())
            or not np.all(arrays["split_validation"])
            or not np.array_equal(
                arrays["weights"], np.ones(count, dtype=np.float32))):
        raise ValueError("Fresh outer projection array shape or split differs")

    observations = arrays["observations"]
    probabilities = arrays["probabilities"]
    labels = arrays["action_indices"]
    if (not np.isfinite(observations).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(
                probabilities.sum(1), 1.0, rtol=0.0, atol=2e-6)
            or np.any(labels >= len(rows_v7.ACTIONS))
            or not np.all(arrays["submitted_equal"])
            or np.any(arrays["frames"] < 0)
            or np.any(arrays["group_bits"] >= (1 << len(rows_v7.GROUPS)))):
        raise ValueError("Fresh outer projection numeric evidence differs")
    expected_hashes = np.asarray([
        rows_v7.legacy._obs_hash(row) for row in observations
    ], dtype="S64")
    if not np.array_equal(expected_hashes, arrays["observation_hashes"]):
        raise ValueError("Fresh outer projection observation hashes differ")
    logits = actor.logits(observations)
    expected_probabilities = np.exp(
        logits - logits.max(axis=1, keepdims=True))
    expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
    if (not np.allclose(
            probabilities, expected_probabilities, rtol=8e-6, atol=4e-6)
            or not np.array_equal(
                labels, expected_probabilities.argmax(1).astype(np.uint8))):
        raise ValueError("Fresh outer projection rows differ from frozen Actor")

    scene_fingerprints = rows_v7._decode(arrays["scene_fingerprints"])
    episode_ids = rows_v7._decode(arrays["episode_ids"])
    registered_fingerprints = {row["fingerprint"] for row in scenes}
    expected_episodes = {
        f"{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in scenes for partner in rows_v7.PARTNERS
    }
    if (len(registered_fingerprints) != len(scenes)
            or set(map(str, scene_fingerprints)) != registered_fingerprints
            or set(map(str, episode_ids)) != expected_episodes):
        raise ValueError("Fresh outer projection registered replay differs")

    kinds = rows_v7._decode(arrays["kinds"])
    anchors = rows_v7._decode(arrays["anchor_ids"])
    branches = rows_v7._decode(arrays["branch_actions"])
    physical = rows_v7._decode(arrays["physical_hashes"])
    source_states = rows_v7._decode(arrays["source_state_hashes"])
    ordinary = kinds == "ordinary"
    intervention = kinds == "intervention"
    if (not np.all(ordinary | intervention)
            or np.any(arrays["trajectory_done"][intervention])
            or not np.all(branches[ordinary] == "")
            or not np.all(physical[ordinary] == "")
            or np.any(anchors[intervention] == "")
            or not set(map(str, branches[intervention])).issubset(
                rows_v7.ACTIONS)
            or any(_HEX.fullmatch(str(value)) is None
                   for value in physical[intervention])
            or any(_HEX.fullmatch(str(value)) is None
                   for value in source_states)):
        raise ValueError("Fresh outer projection row-kind schema differs")
    frames = arrays["frames"]
    bits = arrays["group_bits"]
    ordinary_indices = np.flatnonzero(ordinary)
    expected_anchors = np.asarray([
        (f"{episode_ids[index]}:{int(frames[index])}"
         if bits[index] != 0 and int(frames[index]) % 5 == 0 else "")
        for index in ordinary_indices
    ], dtype="U240")
    if not np.array_equal(anchors[ordinary], expected_anchors):
        raise ValueError("Fresh outer projection intervention schedule differs")
    for episode in expected_episodes:
        indices = np.flatnonzero(ordinary & (episode_ids == episode))
        ordered = indices[np.argsort(frames[indices], kind="stable")]
        if (not len(ordered)
                or not np.array_equal(
                    frames[ordered], np.arange(len(ordered)))
                or np.any(arrays["trajectory_done"][ordered[:-1]])
                or not bool(arrays["trajectory_done"][ordered[-1]])):
            raise ValueError("Fresh outer projection trajectory is incomplete")


def _validate_registry_hash_screen_replay(
    arrays: Mapping[str, np.ndarray], *, registry: Mapping[str, Any],
) -> None:
    """Reproduce the per-scene commitments used by the v13 hash screen."""
    statistics = registry.get("statistics")
    screen = statistics.get("hash_screen") \
        if isinstance(statistics, Mapping) else None
    selected = screen.get("selected") if isinstance(screen, Mapping) else None
    scenes = registry.get("development_outer")
    if (not isinstance(screen, Mapping) or not _content_valid(screen)
            or not isinstance(selected, list) or len(selected) != SCENE_COUNT
            or not isinstance(scenes, list) or len(scenes) != SCENE_COUNT):
        raise ValueError("Fresh v13 registry hash-screen audit differs")
    observation_hashes = _decode_ascii(
        arrays["observation_hashes"], "observation hashes")
    scene_fingerprints = _decode_ascii(
        arrays["scene_fingerprints"], "scene fingerprints")
    if len(observation_hashes) != len(scene_fingerprints):
        raise ValueError("Fresh v13 registry hash-screen replay differs")
    for scene, audit in zip(scenes, selected):
        if not isinstance(scene, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("Fresh v13 registry hash-screen audit differs")
        fingerprint = scene.get("fingerprint")
        hashes = [value for value, row_fingerprint in zip(
            observation_hashes, scene_fingerprints)
                  if row_fingerprint == fingerprint]
        expected = {
            "family_id": scene.get("family_id"),
            "fingerprint": fingerprint,
            "seed": scene.get("seed"),
            "row_count": len(hashes),
            "unique_observation_count": len(set(hashes)),
            "observation_hashes_sha256": digest(hashes),
            "historical_observation_overlap": 0,
        }
        if not hashes or dict(audit) != expected:
            raise ValueError("Fresh v13 registry hash-screen replay differs")


def _decode_ascii(array: np.ndarray, label: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be a one-dimensional byte-string array")
    try:
        return list(map(str, np.char.decode(array, "ascii")))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be ASCII") from error


def _ordered_identity_hashes(arrays: Mapping[str, np.ndarray]) -> list[str]:
    count = len(arrays["observations"])
    text = {name: _decode_ascii(arrays[name], name) for name in (
        "scene_fingerprints", "episode_ids", "kinds", "anchor_ids",
        "branch_actions", "physical_hashes", "source_state_hashes")}
    return [digest({
        "scene_fingerprint": text["scene_fingerprints"][index],
        "episode_id": text["episode_ids"][index],
        "frame": int(arrays["frames"][index]),
        "group_bits": int(arrays["group_bits"][index]),
        "kind": text["kinds"][index],
        "anchor_id": text["anchor_ids"][index],
        "branch_action": text["branch_actions"][index],
        "physical_sha256": text["physical_hashes"][index],
        "source_state_sha256": text["source_state_hashes"][index],
        "trajectory_done": bool(arrays["trajectory_done"][index]),
        "split_validation": bool(arrays["split_validation"][index]),
    }) for index in range(count)]


def projection_from_arrays(
    arrays: Mapping[str, np.ndarray], *, registry_file_sha256: str,
    registry_content_sha256: str, selected_identity_sha256: str,
    environment_steps: int,
) -> dict[str, Any]:
    if set(arrays) != rows_v7._FIELDS:
        raise ValueError("Outer replay row schema differs")
    ordered = _decode_ascii(arrays["observation_hashes"], "observation hashes")
    if (not ordered or any(_HEX.fullmatch(value) is None for value in ordered)
            or not np.all(arrays["split_validation"])):
        raise ValueError("Outer replay observation-hash projection differs")
    row_identities = _ordered_identity_hashes(arrays)
    unique = sorted(set(ordered))
    value: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "identity": {
            "registry_file_sha256": _sha(registry_file_sha256, "registry"),
            "registry_content_sha256": _sha(
                registry_content_sha256, "registry content"),
            "selected_identity_sha256": _sha(
                selected_identity_sha256, "selected outer identity"),
        },
        "schedule": {
            "scene_offset": SCENE_OFFSET,
            "scene_count": SCENE_COUNT,
            "partners": list(rows_v7.PARTNERS),
            "dense_critical": False,
            "critical_anchor_period": 5,
            "environment_steps": int(environment_steps),
        },
        "projection": {
            "row_count": len(ordered),
            "unique_observation_count": len(unique),
            "ordered_observation_hashes": ordered,
            "ordered_observation_hashes_sha256": digest(ordered),
            "unique_observation_hashes": unique,
            "unique_observation_hashes_sha256": digest(unique),
            "ordered_row_identity_hashes": row_identities,
            "ordered_row_identity_hashes_sha256": digest(row_identities),
            "ordered_replay_sha256": digest(list(zip(ordered, row_identities))),
        },
        "information_boundary": {
            "registry_identity_frozen_before_replay": True,
            "actor_inference_used_only_to_advance_fixed_replay": True,
            "raw_observations_published": False,
            "actor_actions_published": False,
            "actor_probabilities_published": False,
            "program_accessed": False,
            "protected_final_access": False,
            "participant_data_accessed": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
    }
    value["content_sha256"] = digest(value)
    return value


def validate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if (set(value) != _PROJECTION_FIELDS
            or value.get("version") != VERSION or value.get("status") != STATUS
            or not _content_valid(value)):
        raise ValueError("Outer hash projection identity differs")
    identity, schedule, projection = (
        value.get("identity"), value.get("schedule"), value.get("projection"))
    boundary = value.get("information_boundary")
    if (not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS
            or not isinstance(schedule, Mapping) or set(schedule) != _SCHEDULE_FIELDS
            or not isinstance(projection, Mapping) or set(projection) != _PAYLOAD_FIELDS
            or not isinstance(boundary, Mapping) or set(boundary) != _BOUNDARY_FIELDS
            or any(_HEX.fullmatch(str(identity.get(name, ""))) is None for name in (
                "registry_file_sha256", "registry_content_sha256",
                "selected_identity_sha256"))
            or schedule.get("scene_offset") != SCENE_OFFSET
            or schedule.get("scene_count") != SCENE_COUNT
            or schedule.get("partners") != list(rows_v7.PARTNERS)
            or schedule.get("dense_critical") is not False
            or schedule.get("critical_anchor_period") != 5
            or type(schedule.get("environment_steps")) is not int
            or schedule["environment_steps"] <= 0):
        raise ValueError("Outer hash projection schedule differs")
    ordered = projection.get("ordered_observation_hashes")
    unique = projection.get("unique_observation_hashes")
    row_ids = projection.get("ordered_row_identity_hashes")
    if (not isinstance(ordered, list) or not ordered
            or not isinstance(unique, list) or unique != sorted(set(ordered))
            or not isinstance(row_ids, list) or len(row_ids) != len(ordered)
            or len(set(row_ids)) != len(row_ids)
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in [*ordered, *unique, *row_ids])
            or projection.get("row_count") != len(ordered)
            or projection.get("unique_observation_count") != len(unique)
            or projection.get("ordered_observation_hashes_sha256") != digest(ordered)
            or projection.get("unique_observation_hashes_sha256") != digest(unique)
            or projection.get("ordered_row_identity_hashes_sha256") != digest(row_ids)
            or projection.get("ordered_replay_sha256")
                != digest(list(zip(ordered, row_ids)))):
        raise ValueError("Outer hash projection payload differs")
    if (boundary.get("registry_identity_frozen_before_replay") is not True
            or boundary.get("actor_inference_used_only_to_advance_fixed_replay") is not True
            or boundary.get("raw_observations_published") is not False
            or boundary.get("actor_actions_published") is not False
            or boundary.get("actor_probabilities_published") is not False
            or boundary.get("program_accessed") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("participant_data_accessed") is not False
            or boundary.get("runtime_action_override") is not False
            or boundary.get("formal_ready") is not False):
        raise ValueError("Outer hash projection information boundary differs")
    return deepcopy(dict(value))


def _receipt(
    *, projection: Mapping[str, Any], projection_file_sha256: str,
    paths: Mapping[str, Path], registry: Mapping[str, Any],
    registry_report: Mapping[str, Any], sources: Mapping[str, str],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": RECEIPT_VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "actor_sha256": file_hash(paths["actor"]),
            "actor_parameters_sha256": NumPyNativeActor(paths["actor"]).metadata[
                "actor_parameters_sha256"],
            "protocol_sha256": file_hash(paths["protocol"]),
            "runtime_manifest_sha256": file_hash(paths["manifest"]),
            "designation_sha256": file_hash(paths["designation"]),
            "fresh_outer_registry_sha256": file_hash(paths["registry"]),
            "fresh_outer_registry_content_sha256": registry["content_sha256"],
            "fresh_outer_registry_report_sha256": file_hash(paths["registry_report"]),
            "fresh_outer_registry_report_content_sha256": registry_report[
                "content_sha256"],
            "outer_hash_projection_sha256": projection_file_sha256,
            "outer_hash_projection_content_sha256": projection["content_sha256"],
            "ordered_replay_sha256": projection["projection"][
                "ordered_replay_sha256"],
            "contract_sha256": digest(contract()),
            "source_closure_sha256": digest(sources),
        },
        "projection": {
            "row_count": projection["projection"]["row_count"],
            "unique_observation_count": projection["projection"][
                "unique_observation_count"],
            "ordered_observation_hashes_sha256": projection["projection"][
                "ordered_observation_hashes_sha256"],
            "unique_observation_hashes_sha256": projection["projection"][
                "unique_observation_hashes_sha256"],
            "ordered_row_identity_hashes_sha256": projection["projection"][
                "ordered_row_identity_hashes_sha256"],
        },
        "sources": deepcopy(dict(sources)),
        "raw_observations_published": False,
        "actor_actions_published": False,
        "actor_probabilities_published": False,
        "program_accessed": False,
        "protected_final_access": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    registry_path: str | Path, registry_report_path: str | Path,
    expected_registry_sha256: str, expected_registry_report_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    snapshot, paths, components, designation_original = _resolve_snapshot(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        registry_path=registry_path, registry_report_path=registry_report_path,
        expected_registry_sha256=expected_registry_sha256,
        expected_registry_report_sha256=expected_registry_report_sha256)
    destination = Path(output).expanduser().absolute()
    temporary: Path | None = None
    published = False
    lock_path: Path | None = None
    lock_fd: int | None = None
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or destination.parent.resolve() != destination.parent:
            raise ValueError("Projection output parent is unsafe")
        lock_path = destination.parent / ("." + destination.name + ".lock")
        lock_fd = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600)
        actor, scenes, registry, report = _validate_registry(
            paths=paths, component_originals=components,
            designation_original=designation_original)
        arrays, _accounting, environment_steps = _replay_outer(
            actor_path=paths["actor"], protocol_path=paths["protocol"],
            manifest_path=paths["manifest"], scenes=scenes,
            progress_label="fresh_v13_outer_hash_projection")
        _validate_registry_hash_screen_replay(arrays, registry=registry)
        selected_identity_sha = report["selection"]["selected_identity_sha256"]
        projection = projection_from_arrays(
            arrays, registry_file_sha256=file_hash(paths["registry"]),
            registry_content_sha256=registry["content_sha256"],
            selected_identity_sha256=selected_identity_sha,
            environment_steps=environment_steps)
        validate_projection(projection)
        temporary = Path(tempfile.mkdtemp(
            prefix=".warehouse-r41-v13-outer-projection-", dir=destination.parent))
        projection_path = temporary / PROJECTION_NAME
        _write_exclusive(projection_path, _json_bytes(projection))
        receipt = _receipt(
            projection=projection, projection_file_sha256=file_hash(projection_path),
            paths=paths, registry=registry, registry_report=report, sources=sources)
        _write_exclusive(temporary / RECEIPT_NAME, _json_bytes(receipt))
        # Published projection is deliberately label blind.
        raw = _json_bytes(projection) + _json_bytes(receipt)
        forbidden = (b'"observations"', b'"probabilities"', b'"action_indices"')
        if any(token in raw for token in forbidden):
            raise RuntimeError("Projection publication contains a forbidden outer payload")
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Projection source closure changed before publication")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Projection source closure changed at publication")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        temporary = None
        published = True
        snapshot.verify()
        if producer_sources() != sources:
            raise RuntimeError("Projection source closure changed after publication")
        return deepcopy(receipt)
    except BaseException:
        if published:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)
        snapshot.__exit__(None, None, None)


def read_saved_projection(
    *, projection_path: str | Path, receipt_path: str | Path,
    expected_projection_sha256: str, expected_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection_file = _regular(projection_path, "v13 outer hash projection")
    receipt_file = _regular(receipt_path, "v13 outer projection receipt")
    if (file_hash(projection_file) != _sha(
            expected_projection_sha256, "v13 outer hash projection")
            or file_hash(receipt_file) != _sha(
                expected_receipt_sha256, "v13 outer projection receipt")):
        raise ValueError("Exact v13 outer projection artifact pair required")
    projection = validate_projection(_strict_json(
        projection_file, "v13 outer hash projection"))
    receipt = _strict_json(receipt_file, "v13 outer projection receipt")
    sources = receipt.get("sources")
    bindings = receipt.get("bindings")
    if (receipt.get("version") != RECEIPT_VERSION
            or receipt.get("status") != STATUS or not _content_valid(receipt)
            or receipt.get("contract") != contract()
            or not isinstance(bindings, Mapping)
            or bindings.get("outer_hash_projection_sha256")
                != file_hash(projection_file)
            or bindings.get("outer_hash_projection_content_sha256")
                != projection["content_sha256"]
            or bindings.get("ordered_replay_sha256")
                != projection["projection"]["ordered_replay_sha256"]
            or not _frozen_sources_valid(
                sources, bindings.get("source_closure_sha256"))
            or receipt.get("raw_observations_published") is not False
            or receipt.get("actor_actions_published") is not False
            or receipt.get("actor_probabilities_published") is not False
            or receipt.get("program_accessed") is not False
            or receipt.get("protected_final_access") is not False
            or receipt.get("formal_ready") is not False):
        raise ValueError("V13 outer projection receipt differs")
    return projection, deepcopy(receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-report", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build(
        actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        registry_path=args.registry, registry_report_path=args.registry_report,
        expected_registry_sha256=args.expected_registry_sha256,
        expected_registry_report_sha256=args.expected_registry_report_sha256,
        output=args.output)
    print(canonical({"status": receipt["status"],
                     "rows": receipt["projection"]["row_count"],
                     "output": str(Path(args.output).expanduser().absolute())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "RECEIPT_VERSION", "SCENE_OFFSET", "SCENE_COUNT",
    "PROJECTION_NAME", "RECEIPT_NAME", "contract", "producer_sources",
    "projection_from_arrays", "validate_projection", "build",
    "read_saved_projection", "main",
]
