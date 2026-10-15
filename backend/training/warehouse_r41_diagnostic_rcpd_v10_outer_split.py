"""Freeze a replacement outer set after the v9 attempt was consumed.

The replacement is selected from the same fixed 2,160-scene identity
population as v9.  Every identity excluded by v9 remains excluded, and the 64
scenes claimed by the failed v9 one-shot attempt are additionally excluded.
Selection uses only seed, public scene fingerprint, family, exclusion
membership, and a fixed new salt.  It never reads the consumed v9 row archive
or any action, probability, program prediction, or score.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_outer_attempt_closeout_v10 as attempt_closeout_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_split as v9
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v10-fresh-outer-registry.v1"
REPORT_VERSION = VERSION
STATUS = "frozen_identity_only_pending_outer_collection"
ROOT = Path(__file__).resolve().parents[2]
SELECTION_SALT = "warehouse-r41-v10-fresh-development-outer-20260914-v1"
FAMILY_IDS = tuple(v9.FAMILY_IDS)
FAMILY_QUOTAS = dict(v9.FAMILY_QUOTAS)
FRESH_OUTER_SCENE_COUNT = 64
SOURCE_ROW_SCENE_COUNT = v9.SOURCE_ROW_SCENE_COUNT
FIXED_CANDIDATE_SCENE_COUNT = v9.FIXED_CANDIDATE_SCENE_COUNT
EXPECTED_REMAINING_SCENE_COUNT = 1631
EXPECTED_REMAINING_FAMILY_COUNTS = {
    "conflict_family_01": 268,
    "conflict_family_02": 269,
    "conflict_family_03": 271,
    "conflict_family_04": 268,
    "conflict_family_05": 277,
    "conflict_family_06": 278,
}
SCENE_OFFSET = 512
MAX_JSON_BYTES = v9.MAX_JSON_BYTES
_HEX = re.compile(r"[0-9a-f]{64}\Z")

# The still-valid development and earlier closeout remain unchanged.
EXPECTED_MANIFEST_SHA256 = v9.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = v9.EXPECTED_MANIFEST_VALIDATION_SHA256
EXPECTED_FAILED_ROWS_SHA256 = v9.EXPECTED_FAILED_ROWS_SHA256
EXPECTED_ORIGINAL_EXPANSION_SHA256 = v9.EXPECTED_ORIGINAL_EXPANSION_SHA256
EXPECTED_CURRENT_OUTER_REGISTRY_SHA256 = v9.EXPECTED_CURRENT_OUTER_REGISTRY_SHA256
EXPECTED_CURRENT_OUTER_REPORT_SHA256 = v9.EXPECTED_CURRENT_OUTER_REPORT_SHA256
EXPECTED_FORMAL_SELECTION_SHA256 = v9.EXPECTED_FORMAL_SELECTION_SHA256
EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 = v9.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256
EXPECTED_RETIRED_PROJECTION_SHA256 = v9.EXPECTED_RETIRED_PROJECTION_SHA256
EXPECTED_RETIRED_REPORT_SHA256 = v9.EXPECTED_RETIRED_REPORT_SHA256


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _rank(row: Mapping[str, Any]) -> str:
    return digest({
        "salt": SELECTION_SALT,
        "family_id": row["family_id"],
        "fingerprint": row["fingerprint"],
    })


def _public_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in (
        "batch_index", "family_id", "seed", "fingerprint")}


def _remaining_and_selected(
    candidates: Sequence[Mapping[str, Any]], *, excluded_seeds: set[int],
    excluded_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if len(candidates) != FIXED_CANDIDATE_SCENE_COUNT:
        raise ValueError("Fixed candidate population count differs")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        seed, fingerprint = v9._identity(raw, "fixed v10 candidate")
        family = raw.get("family_id")
        batch_index = raw.get("batch_index")
        family_offset = raw.get("family_offset")
        if (family not in FAMILY_IDS or type(batch_index) is not int
                or type(family_offset) is not int):
            raise ValueError("Fixed v10 candidate identity differs")
        if seed not in excluded_seeds and fingerprint not in excluded_fingerprints:
            by_family[str(family)].append(deepcopy(dict(raw)))
    remaining = [row for family in FAMILY_IDS for row in by_family[family]]
    counts = dict(sorted(Counter(row["family_id"] for row in remaining).items()))
    if (len(remaining) != EXPECTED_REMAINING_SCENE_COUNT
            or counts != EXPECTED_REMAINING_FAMILY_COUNTS):
        raise ValueError("Fixed replacement candidate remaining population differs")
    selected: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (
            _rank(row), row["fingerprint"]))
        quota = FAMILY_QUOTAS[family]
        if len(ordered) < quota:
            raise ValueError("Insufficient replacement identities for " + family)
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (FAMILY_IDS.index(row["family_id"]), _rank(row)))
    if (len(selected) != FRESH_OUTER_SCENE_COUNT
            or dict(sorted(Counter(row["family_id"] for row in selected).items()))
                != FAMILY_QUOTAS):
        raise RuntimeError("Replacement outer family quota differs")
    return remaining, selected, counts


def _materialize_scene(identity: Mapping[str, Any], index: int) -> dict[str, Any]:
    seed, expected_fingerprint = v9._identity(identity, "selected v10 outer")
    batch_index, family = identity.get("batch_index"), identity.get("family_id")
    if type(batch_index) is not int or family not in FAMILY_IDS:
        raise ValueError("Selected v10 outer batch identity differs")
    env = R41DiagnosticConflictWarehouseEnv()
    v9.retired_api._identity_only_reset(env, seed=seed)
    reset_family = conflict_family_id(env.state.tasks)
    if reset_family != family:
        raise ValueError("Selected v10 outer family differs at reset")
    v9.scenes_api._candidate_start_state(
        env, seed=seed, batch_index=batch_index, family_id=reset_family)
    if diagnostic_scene_fingerprint(env) != expected_fingerprint:
        raise ValueError("Selected v10 outer fingerprint differs")
    scene = v9.scenes_api._scene_from_environment(
        env, scene_id=f"diagnostic_v10_fresh_outer_{index:04d}",
        split="development_outer", seed=seed, batch_index=batch_index)
    if (scene.get("fingerprint") != expected_fingerprint
            or scene.get("family_id") != family):
        raise ValueError("Materialised v10 outer identity differs")
    return scene


def replay_exclusion_closure(
    *, manifest_path: str | Path, failed_rows_path: str | Path,
    original_expansion_path: str | Path,
    current_outer_registry_path: str | Path,
    current_outer_report_path: str | Path,
    failure_closeout_path: str | Path,
    expected_failure_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    formal_selection_path: str | Path,
    previous_development_path: str | Path,
    retired_projection_path: str | Path,
    consumed_v9_attempt_closeout_path: str | Path,
    expected_consumed_v9_attempt_closeout_sha256: str,
    permanent_v9_attempt_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate and replay every public identity exclusion source."""
    manifest = v9._authenticate_opaque(
        manifest_path, "frozen diagnostic manifest",
        expected_sha256=EXPECTED_MANIFEST_SHA256)
    manifest_validation = v9._authenticate_opaque(
        manifest.parent / "validation.json", "manifest validation receipt",
        expected_sha256=EXPECTED_MANIFEST_VALIDATION_SHA256)
    rows_file, row_fingerprints = v9._scene_fingerprints_only(
        failed_rows_path, expected_sha256=EXPECTED_FAILED_ROWS_SHA256)
    original_file, original = v9._strict_json(
        original_expansion_path, "original development expansion",
        expected_sha256=EXPECTED_ORIGINAL_EXPANSION_SHA256)
    trace_seeds, trace_fingerprints, old_outer_seeds, old_outer_fingerprints = (
        v9._source_expansion(original))
    current_file, current = v9._strict_json(
        current_outer_registry_path, "consumed v8 outer registry",
        expected_sha256=EXPECTED_CURRENT_OUTER_REGISTRY_SHA256)
    current_report_file, current_report = v9._strict_json(
        current_outer_report_path, "consumed v8 outer report",
        expected_sha256=EXPECTED_CURRENT_OUTER_REPORT_SHA256)
    current_identities, current_seeds, current_fingerprints, registered_old = (
        v9._current_outer(
            current, current_report,
            registry_sha256=EXPECTED_CURRENT_OUTER_REGISTRY_SHA256))
    if registered_old != old_outer_fingerprints:
        raise ValueError("Consumed v8 registry does not bind original outer")
    old_closeout = v9.closeout_api.read_saved_closeout(
        failure_closeout_path,
        expected_closeout_sha256=expected_failure_closeout_sha256,
        permanent_registry=permanent_closeout_registry)
    old_closeout_source = old_closeout.get("source", {})
    old_closeout_identities = old_closeout.get("consumed_outer", {}).get(
        "identities")
    if (old_closeout_source.get("v8_combined_rows_sha256")
            != EXPECTED_FAILED_ROWS_SHA256
            or old_closeout_source.get("v8_outer_registry_sha256")
                != EXPECTED_CURRENT_OUTER_REGISTRY_SHA256
            or old_closeout_source.get("v8_outer_registry_report_sha256")
                != EXPECTED_CURRENT_OUTER_REPORT_SHA256
            or old_closeout_identities != current_identities
            or old_closeout.get("disposition", {}).get(
                "eligible_for_outer_claim") is not False):
        raise ValueError("Failed v8 closeout binding differs")
    formal_file, formal = v9._strict_json(
        formal_selection_path, "formal X/Y selection",
        expected_sha256=EXPECTED_FORMAL_SELECTION_SHA256)
    formal_seeds, formal_fingerprints = v9._formal_identities(formal)
    previous_file, previous = v9._strict_json(
        previous_development_path, "previous development supplement",
        expected_sha256=EXPECTED_PREVIOUS_DEVELOPMENT_SHA256)
    previous_seeds, previous_fingerprints = v9._previous_identities(previous)
    retired_file = v9._regular(
        retired_projection_path, "retired identity projection",
        maximum=MAX_JSON_BYTES)
    retired_report_file = v9._regular(
        retired_file.parent / "report.json", "retired projection report",
        maximum=MAX_JSON_BYTES)
    if (file_hash(retired_file) != EXPECTED_RETIRED_PROJECTION_SHA256
            or file_hash(retired_report_file) != EXPECTED_RETIRED_REPORT_SHA256):
        raise ValueError("Exact retired identity projection pair required")
    # This is a frozen historical projection.  Authenticate its exact bytes
    # and its own recorded source closure rather than comparing that closure
    # with today's unrelated runtime sources.
    _, retired = v9._strict_json(
        retired_file, "retired identity projection",
        expected_sha256=EXPECTED_RETIRED_PROJECTION_SHA256)
    _, retired_report = v9._strict_json(
        retired_report_file, "retired projection report",
        expected_sha256=EXPECTED_RETIRED_REPORT_SHA256)
    if (retired.get("version") != v9.retired_api.VERSION
            or not v9._content_valid(retired)
            or retired_report.get("version") != v9.retired_api.REPORT_VERSION
            or retired_report.get("status") != v9.retired_api.STATUS
            or not v9._content_valid(retired_report)
            or retired_report.get("projection_file_sha256")
                != EXPECTED_RETIRED_PROJECTION_SHA256
            or retired_report.get("projection_content_sha256")
                != retired["content_sha256"]
            or retired_report.get("global_identity_count")
                != v9.retired_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES
            or retired_report.get("producer_sources_sha256")
                != digest(dict(sorted(
                    retired_report.get("producer_sources", {}).items())))):
        raise ValueError("Frozen retired identity projection pair differs")
    retired_seeds, retired_fingerprints = v9._unique_identities(
        retired.get("exposed_identities"),
        expected_count=v9.retired_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        label="retired exposed projection")
    attempt_closeout_file = v9._regular(
        consumed_v9_attempt_closeout_path, "consumed v9 attempt closeout",
        maximum=MAX_JSON_BYTES)
    attempt_closeout = attempt_closeout_api.read_saved_closeout(
        attempt_closeout_file,
        expected_closeout_sha256=expected_consumed_v9_attempt_closeout_sha256,
        permanent_attempt_registry=permanent_v9_attempt_registry)
    attempt_identities = attempt_closeout["consumed_outer"]["identities"]
    if (attempt_closeout.get("disposition", {}).get(
            "outer_identity_reuse_permitted") is not False
            or attempt_closeout["consumed_outer"].get("scene_count")
                != FRESH_OUTER_SCENE_COUNT
            or attempt_closeout["consumed_outer"].get("identities_sha256")
                != digest(attempt_identities)):
        raise ValueError("Consumed v9 attempt closeout differs")

    candidates = v9._candidate_identity_population()
    by_fingerprint = {row["fingerprint"]: row for row in candidates}
    if len(by_fingerprint) != len(candidates):
        raise ValueError("Fixed candidate fingerprint map differs")
    attempt_public = [_public_identity(row) for row in attempt_identities]
    candidate_public = {
        (row["seed"], row["fingerprint"]): _public_identity(row)
        for row in candidates
    }
    if (any(candidate_public.get((row["seed"], row["fingerprint"])) != row
            for row in attempt_public)
            or len({row["seed"] for row in attempt_public})
                != FRESH_OUTER_SCENE_COUNT):
        raise ValueError("Consumed v9 identities are not exact fixed candidates")
    row_candidate_fingerprints = row_fingerprints & set(by_fingerprint)
    row_candidate_seeds = {
        int(by_fingerprint[fingerprint]["seed"])
        for fingerprint in row_candidate_fingerprints
    }
    excluded_seeds = (
        trace_seeds | old_outer_seeds | current_seeds | formal_seeds
        | previous_seeds | retired_seeds | row_candidate_seeds
        | {int(row["seed"]) for row in old_closeout_identities}
        | {int(row["seed"]) for row in attempt_public}
    )
    excluded_fingerprints = (
        trace_fingerprints | old_outer_fingerprints | current_fingerprints
        | registered_old | formal_fingerprints | previous_fingerprints
        | retired_fingerprints | row_fingerprints
        | {str(row["fingerprint"]) for row in old_closeout_identities}
        | {str(row["fingerprint"]) for row in attempt_public}
    )
    return {
        "candidates": candidates,
        "excluded_seeds": excluded_seeds,
        "excluded_fingerprints": excluded_fingerprints,
        "v9_consumed_identities": attempt_public,
        "row_fingerprints": row_fingerprints,
        "trace_fingerprints": trace_fingerprints,
        "current_identities": current_identities,
        "formal_fingerprints": formal_fingerprints,
        "previous_fingerprints": previous_fingerprints,
        "retired_fingerprints": retired_fingerprints,
        "row_candidate_fingerprints": row_candidate_fingerprints,
        "old_closeout": old_closeout,
        "paths": {
            "manifest": manifest,
            "manifest_validation": manifest_validation,
            "rows": rows_file,
            "original": original_file,
            "current": current_file,
            "current_report": current_report_file,
            "formal": formal_file,
            "previous": previous_file,
            "retired": retired_file,
            "retired_report": retired_report_file,
            "attempt_closeout": attempt_closeout_file,
        },
        "attempt_closeout": attempt_closeout,
    }


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh replacement outer after consumed v9 one-shot attempt",
        "selection_salt": SELECTION_SALT,
        "fixed_candidate_population": FIXED_CANDIDATE_SCENE_COUNT,
        "required_remaining_population": EXPECTED_REMAINING_SCENE_COUNT,
        "required_remaining_family_counts": deepcopy(
            EXPECTED_REMAINING_FAMILY_COUNTS),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "selection_inputs": [
            "seed", "scene fingerprint", "family", "exclusion membership"],
        "consumed_v9_outer_permanently_excluded": True,
        "selection_uses_consumed_v9_rows": False,
        "prior_validation_wins_projection": (
            "deduplicated v8 and consumed-v9 one-way observation hashes"),
        "selection_uses_prior_validation_wins_projection": False,
        "selection_uses_actor": False,
        "selection_uses_observations": False,
        "selection_uses_actions_probabilities_program_or_metrics": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def _prior_validation_wins_projection(
    *, old_closeout: Mapping[str, Any],
    attempt_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    old = old_closeout.get("consumed_outer", {}).get(
        "observation_hash_projection", {})
    consumed = attempt_closeout.get("consumed_outer", {}).get(
        "observation_hash_projection", {})
    old_hashes = old.get("outer_observation_hashes")
    consumed_hashes = consumed.get("outer_observation_hashes")
    if (not isinstance(old_hashes, list) or not isinstance(consumed_hashes, list)
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in [*old_hashes, *consumed_hashes])):
        raise ValueError("Prior outer observation-hash projections differ")
    combined = sorted(set(old_hashes) | set(consumed_hashes))
    projection: dict[str, Any] = {
        "version": VERSION + ".validation-wins-exclusion.v1",
        "source_v8_closeout_content_sha256": old_closeout["content_sha256"],
        "source_v8_projection_content_sha256": old["content_sha256"],
        "source_v9_closeout_content_sha256": attempt_closeout[
            "content_sha256"],
        "source_v9_projection_content_sha256": consumed[
            "source_projection_content_sha256"],
        "outer_observation_hashes": combined,
        "unique_outer_observation_hash_count": len(combined),
        "outer_observation_hashes_sha256": digest(combined),
        "component_unique_counts": {
            "consumed_v8": len(old_hashes),
            "consumed_v9": len(consumed_hashes),
        },
        "component_hashes_sha256": {
            "consumed_v8": digest(old_hashes),
            "consumed_v9": digest(consumed_hashes),
        },
        "selector_rule": (
            "remove fit rows matching any prior outer observation hash before "
            "reading actions, probabilities, or raw observations"),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
        "selection_used_this_projection": False,
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    return projection


def create_registry(**kwargs: Any) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = producer_sources()
    closure = replay_exclusion_closure(**kwargs)
    remaining, selected, remaining_counts = _remaining_and_selected(
        closure["candidates"], excluded_seeds=closure["excluded_seeds"],
        excluded_fingerprints=closure["excluded_fingerprints"])
    selected_public = [_public_identity(row) for row in selected]
    if (any(row["seed"] in closure["excluded_seeds"]
            or row["fingerprint"] in closure["excluded_fingerprints"]
            for row in selected_public)
            or len({row["seed"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT):
        raise RuntimeError("Replacement outer overlaps an exposed identity")
    scenes = [_materialize_scene(row, index)
              for index, row in enumerate(selected)]
    if [(row["seed"], row["fingerprint"]) for row in scenes] != [
            (row["seed"], row["fingerprint"]) for row in selected]:
        raise RuntimeError("Replacement outer materialisation changed frozen set")
    projection = _prior_validation_wins_projection(
        old_closeout=closure["old_closeout"],
        attempt_closeout=closure["attempt_closeout"])
    projection_file_sha256 = sha256(
        (canonical(projection) + "\n").encode("utf-8")).hexdigest()
    paths = closure["paths"]
    attempt_closeout = closure["attempt_closeout"]
    bindings = {
        "actor_sha256": v9.manifest_binding.FROZEN_ACTOR_SHA256,
        "actor_parameters_sha256": (
            v9.manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256),
        "manifest_file_sha256": file_hash(paths["manifest"]),
        "manifest_validation_file_sha256": file_hash(
            paths["manifest_validation"]),
        "original_expansion_file_sha256": file_hash(paths["original"]),
        "failed_rows_file_sha256": file_hash(paths["rows"]),
        "consumed_outer_registry_file_sha256": file_hash(paths["current"]),
        "consumed_outer_report_file_sha256": file_hash(paths["current_report"]),
        "failure_closeout_file_sha256": kwargs[
            "expected_failure_closeout_sha256"],
        "failure_closeout_content_sha256": closure["old_closeout"][
            "content_sha256"],
        "consumed_v9_attempt_closeout_file_sha256": kwargs[
            "expected_consumed_v9_attempt_closeout_sha256"],
        "consumed_v9_attempt_closeout_content_sha256": attempt_closeout[
            "content_sha256"],
        "consumed_v9_selected_identity_sha256": attempt_closeout[
            "consumed_outer"]["identities_sha256"],
        "formal_selection_file_sha256": file_hash(paths["formal"]),
        "previous_development_file_sha256": file_hash(paths["previous"]),
        "retired_projection_file_sha256": file_hash(paths["retired"]),
        "retired_projection_report_file_sha256": file_hash(
            paths["retired_report"]),
        "outer_observation_hash_projection_file_sha256": (
            projection_file_sha256),
        "outer_observation_hash_projection_content_sha256": projection[
            "content_sha256"],
        "contract_sha256": digest(contract()),
    }
    exclusion_counts = {
        "source_row_scene_fingerprints": len(closure["row_fingerprints"]),
        "source_rows_in_fixed_candidate_population": len(
            closure["row_candidate_fingerprints"]),
        "original_expansion_trace_identities": len(
            closure["trace_fingerprints"]),
        "consumed_outer_identities": len(closure["current_identities"]),
        "consumed_v9_outer_identities": len(
            closure["v9_consumed_identities"]),
        "formal_xy_identities": len(closure["formal_fingerprints"]),
        "previous_development_identities": len(
            closure["previous_fingerprints"]),
        "retired_exposed_identities": len(closure["retired_fingerprints"]),
        "union_candidate_identities_excluded": (
            len(closure["candidates"]) - len(remaining)),
    }
    exclusion_digests = {
        "excluded_seeds_sha256": digest(sorted(closure["excluded_seeds"])),
        "excluded_scene_fingerprints_sha256": digest(sorted(
            closure["excluded_fingerprints"])),
        "source_row_scene_fingerprints_sha256": digest(sorted(
            closure["row_fingerprints"])),
        "original_expansion_trace_fingerprints_sha256": digest(sorted(
            closure["trace_fingerprints"])),
        "consumed_outer_identities_sha256": digest(
            closure["current_identities"]),
        "consumed_v9_outer_identities_sha256": digest(
            closure["v9_consumed_identities"]),
        "retired_exposed_fingerprints_sha256": digest(sorted(
            closure["retired_fingerprints"])),
    }
    statistics = {
        "fixed_candidate_scene_count": len(closure["candidates"]),
        "remaining_candidate_scene_count": len(remaining),
        "remaining_family_counts": remaining_counts,
        "selected_outer_scene_count": len(scenes),
        "selected_outer_family_counts": dict(sorted(Counter(
            row["family_id"] for row in scenes).items())),
        "selected_exposed_seed_overlap": 0,
        "selected_exposed_fingerprint_overlap": 0,
        **exclusion_counts,
    }
    boundary = {
        "consumed_v9_attempt_closed_before_selection": True,
        "consumed_v9_identities_excluded": True,
        "all_previously_exposed_identities_excluded": True,
        "identity_selection_frozen_before_materialisation": True,
        "failed_rows_fields_read": ["scene_fingerprints"],
        "consumed_v9_rows_opened": False,
        "observations_actions_probabilities_or_labels_read": False,
        "observation_hash_projection_used_for_identity_selection": False,
        "prior_v8_and_v9_observation_hashes_published_for_validation_wins": True,
        "actor_loaded_or_inferred": False,
        "program_predictions_or_metrics_read_for_selection": False,
        "protected_final_access": False,
        "formal_ready": False,
    }
    registry: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": bindings,
        "development_outer": scenes,
        "selected_outer_identities": selected_public,
        "exclusion_counts": exclusion_counts,
        "exclusion_digests": exclusion_digests,
        "statistics": statistics,
        "information_boundary": boundary,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "formal_ready": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
    }
    registry["content_sha256"] = digest(registry)
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "status": STATUS,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": deepcopy(bindings),
        "selection": {
            "salt": SELECTION_SALT,
            "family_quotas": deepcopy(FAMILY_QUOTAS),
            "selected_identity_sha256": digest(selected_public),
            "fixed_remaining_identity_sha256": digest([
                _public_identity(row) for row in sorted(
                    remaining, key=lambda item: (
                        item["family_id"], item["batch_index"],
                        item["family_offset"]))]),
            "exclusion_digests": deepcopy(exclusion_digests),
        },
        "statistics": deepcopy(statistics),
        "information_boundary": deepcopy(boundary),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    if producer_sources() != sources:
        raise RuntimeError("v10 registry source closure changed during selection")
    return registry, report, projection


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    registry, report, projection = create_registry(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v10 registry output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v10 registry output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / "outer_observation_hashes.json", projection)
        if file_hash(temporary / "outer_observation_hashes.json") != report[
                "bindings"]["outer_observation_hash_projection_file_sha256"]:
            raise RuntimeError("v10 historical projection bytes differ")
        _write_exclusive(temporary / "development_outer.json", registry)
        report["registry_file_sha256"] = file_hash(
            temporary / "development_outer.json")
        report["content_sha256"] = digest({
            key: child for key, child in report.items()
            if key != "content_sha256"})
        _write_exclusive(temporary / "report.json", report)
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(report)


def read_saved_registry(
    registry_path: str | Path, report_path: str | Path, *,
    expected_registry_sha256: str, expected_report_sha256: str,
    consumed_v9_attempt_closeout_path: str | Path,
    expected_consumed_v9_attempt_closeout_sha256: str,
    permanent_v9_attempt_registry: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate a saved replacement registry and its consumed-v9 root."""
    registry_file, registry = v9._strict_json(
        registry_path, "saved v10 outer registry",
        expected_sha256=expected_registry_sha256)
    report_file, report = v9._strict_json(
        report_path, "saved v10 outer registry report",
        expected_sha256=expected_report_sha256)
    closeout = attempt_closeout_api.read_saved_closeout(
        consumed_v9_attempt_closeout_path,
        expected_closeout_sha256=expected_consumed_v9_attempt_closeout_sha256,
        permanent_attempt_registry=permanent_v9_attempt_registry)
    scenes = registry.get("development_outer")
    identities = registry.get("selected_outer_identities")
    if (registry.get("version") != VERSION or registry.get("status") != STATUS
            or not v9._content_valid(registry)
            or registry.get("contract") != contract()
            or registry.get("formal_ready") is not False
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("action_labels_access") is not False
            or registry.get("probabilities_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or not isinstance(scenes, list) or len(scenes) != 64
            or not isinstance(identities, list) or len(identities) != 64
            or registry.get("producer_sources_sha256")
                != digest(dict(sorted(
                    registry.get("producer_sources", {}).items())))):
        raise ValueError("Saved v10 outer registry semantics differ")
    public: list[dict[str, Any]] = []
    for index, (scene, raw) in enumerate(zip(scenes, identities)):
        if not isinstance(raw, Mapping) or set(raw) != {
                "batch_index", "family_id", "seed", "fingerprint"}:
            raise ValueError("Saved v10 outer identity fields differ")
        identity = _public_identity(raw)
        v9._identity(identity, "saved v10 outer identity")
        if (not isinstance(scene, Mapping)
                or scene.get("id")
                    != f"diagnostic_v10_fresh_outer_{index:04d}"
                or any(scene.get(name) != identity[name]
                       for name in ("family_id", "seed", "fingerprint"))):
            raise ValueError("Saved v10 outer materialisation differs")
        public.append(identity)
    bindings = registry.get("bindings")
    statistics = registry.get("statistics")
    boundary = registry.get("information_boundary")
    if (not isinstance(bindings, Mapping)
            or bindings.get("failed_rows_file_sha256")
                != EXPECTED_FAILED_ROWS_SHA256
            or bindings.get("failure_closeout_file_sha256")
                != "90fb466abaec7fb49a2b6cfff1d79fa74d0f3365fc8102785bb5c1a9dfd0e4c0"
            or bindings.get("consumed_v9_attempt_closeout_file_sha256")
                != expected_consumed_v9_attempt_closeout_sha256
            or bindings.get("consumed_v9_attempt_closeout_content_sha256")
                != closeout["content_sha256"]
            or bindings.get("consumed_v9_selected_identity_sha256")
                != closeout["consumed_outer"]["identities_sha256"]
            or bindings.get("contract_sha256") != digest(contract())
            or not isinstance(statistics, Mapping)
            or statistics.get("remaining_candidate_scene_count") != 1631
            or statistics.get("remaining_family_counts")
                != EXPECTED_REMAINING_FAMILY_COUNTS
            or statistics.get("consumed_v9_outer_identities") != 64
            or statistics.get("union_candidate_identities_excluded") != 529
            or statistics.get("selected_outer_scene_count") != 64
            or statistics.get("selected_outer_family_counts") != FAMILY_QUOTAS
            or statistics.get("selected_exposed_seed_overlap") != 0
            or statistics.get("selected_exposed_fingerprint_overlap") != 0
            or not isinstance(boundary, Mapping)
            or boundary.get("consumed_v9_attempt_closed_before_selection")
                is not True
            or boundary.get("consumed_v9_identities_excluded") is not True
            or boundary.get("consumed_v9_rows_opened") is not False
            or boundary.get("observations_actions_probabilities_or_labels_read")
                is not False):
        raise ValueError("Saved v10 exclusion closure differs")
    selection = report.get("selection")
    if (report.get("version") != REPORT_VERSION
            or report.get("status") != STATUS
            or not v9._content_valid(report)
            or report.get("formal_ready") is not False
            or report.get("registry_file_sha256") != file_hash(registry_file)
            or report.get("registry_content_sha256")
                != registry["content_sha256"]
            or report.get("bindings") != bindings
            or report.get("statistics") != statistics
            or report.get("information_boundary") != boundary
            or report.get("producer_sources")
                != registry.get("producer_sources")
            or report.get("producer_sources_sha256")
                != registry.get("producer_sources_sha256")
            or not isinstance(selection, Mapping)
            or selection.get("salt") != SELECTION_SALT
            or selection.get("family_quotas") != FAMILY_QUOTAS
            or selection.get("selected_identity_sha256") != digest(public)
            or selection.get("exclusion_digests")
                != registry.get("exclusion_digests")
            or file_hash(report_file) != expected_report_sha256):
        raise ValueError("Saved v10 registry report differs")
    consumed_keys = {
        (row["seed"], row["fingerprint"])
        for row in closeout["consumed_outer"]["identities"]
    }
    if consumed_keys & {(row["seed"], row["fingerprint"]) for row in public}:
        raise ValueError("Saved v10 registry reuses a consumed v9 identity")
    return deepcopy(registry), deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--failed-rows", required=True)
    parser.add_argument("--original-expansion", required=True)
    parser.add_argument("--current-outer-registry", required=True)
    parser.add_argument("--current-outer-report", required=True)
    parser.add_argument("--failure-closeout", required=True)
    parser.add_argument("--expected-failure-closeout-sha256", required=True)
    parser.add_argument("--permanent-closeout-registry", required=True)
    parser.add_argument("--formal-selection", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--retired-projection", required=True)
    parser.add_argument("--consumed-v9-attempt-closeout", required=True)
    parser.add_argument(
        "--expected-consumed-v9-attempt-closeout-sha256", required=True)
    parser.add_argument("--permanent-v9-attempt-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(
        manifest_path=args.manifest,
        failed_rows_path=args.failed_rows,
        original_expansion_path=args.original_expansion,
        current_outer_registry_path=args.current_outer_registry,
        current_outer_report_path=args.current_outer_report,
        failure_closeout_path=args.failure_closeout,
        expected_failure_closeout_sha256=args.expected_failure_closeout_sha256,
        permanent_closeout_registry=args.permanent_closeout_registry,
        formal_selection_path=args.formal_selection,
        previous_development_path=args.previous_development,
        retired_projection_path=args.retired_projection,
        consumed_v9_attempt_closeout_path=args.consumed_v9_attempt_closeout,
        expected_consumed_v9_attempt_closeout_sha256=(
            args.expected_consumed_v9_attempt_closeout_sha256),
        permanent_v9_attempt_registry=args.permanent_v9_attempt_registry,
        output=args.output,
    )
    print(canonical({
        "status": report["status"],
        "registry": str(Path(args.output).resolve() / "development_outer.json"),
        "selected_identity_sha256": report["selection"][
            "selected_identity_sha256"],
        "remaining_candidate_scene_count": report["statistics"][
            "remaining_candidate_scene_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "REPORT_VERSION", "STATUS", "SELECTION_SALT", "FAMILY_IDS",
    "FAMILY_QUOTAS", "FRESH_OUTER_SCENE_COUNT", "SCENE_OFFSET",
    "EXPECTED_REMAINING_SCENE_COUNT", "EXPECTED_REMAINING_FAMILY_COUNTS",
    "contract", "producer_sources", "replay_exclusion_closure",
    "create_registry", "build", "read_saved_registry", "main",
    "_remaining_and_selected", "_materialize_scene",
    "_prior_validation_wins_projection",
]
