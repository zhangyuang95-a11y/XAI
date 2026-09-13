"""Materialise the one protected v9 warehouse final split after its claim.

This program is intentionally launched only by
``warehouse_r41_diagnostic_final_once_v9``.  Its first filesystem read is the
already-created permanent attempt anchor.  Only after that anchor has been
strictly authenticated does it read the deployment configuration and the
committed private salt.

The final identities come from the frozen 2,160-scene public candidate
population.  Every identity exposed by development, the failed v8 campaign,
the fresh v9 outer, the formal X/Y tasks, and retired holdouts is excluded.
Selection uses the v4 salt-ranking and family-quota rules.  The frozen Actor
may execute inside the fixed workload replay, but raw actions, logits, and
probabilities are neither inspected as ranking inputs nor written to the
material artifact.  Workload failure or public-observation overlap rejects a
candidate; the program is never opened and the Actor is never changed.
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
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_api
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as v4
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v9 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_once as outer_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_split as registry_api
from backend.training import warehouse_r41_diagnostic_retired_identity_projection_v8 as retired_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_r41_diagnostic_workload_screen import screen_scene


VERSION = "warehouse-r41-diagnostic-final-materializer.v9"
MATERIAL_VERSION = "warehouse-r41-diagnostic-final-material.v9"
MATERIAL_STATUS = "materialized_after_irrevocable_final_claim"
CONFIG_VERSION = VERSION + ".config.v1"
CONFIG_ENV = "WAREHOUSE_R41_V9_FINAL_MATERIALIZER_CONFIG"
FINAL_ONCE_VERSION = "warehouse-r41-diagnostic-final-once.v9"
ANCHOR_VERSION = FINAL_ONCE_VERSION + ".attempt-anchor.v1"
ANCHOR_STATUS = "final_attempt_irrevocably_claimed"
FINAL_SCENE_COUNT = 64
FAMILY_IDS = tuple(registry_api.FAMILY_IDS)
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
PRIVATE_SALT_DOMAIN = v4.HOLDOUT_SALT_DOMAIN
PRIVATE_SALT_COMMITMENT = v4.HOLDOUT_SALT_COMMITMENT
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ANCHOR_FIELDS = frozenset((
    "version", "status", "attempt_key", "attempt_key_inputs", "bindings",
    "candidate_and_outer_authenticated_before_claim",
    "final_identity_or_rows_accessed_before_claim", "retry_allowed",
    "formal_ready", "content_sha256",
))
_ATTEMPT_INPUT_FIELDS = frozenset((
    "scheme", "candidate_lock_sha256", "candidate_lock_content_sha256",
    "actor_sha256", "protocol_sha256", "runtime_manifest_sha256",
    "designation_sha256", "program_sha256",
    "public_feature_contract_sha256", "candidate_source_closure_sha256",
    "outer_result_sha256", "outer_attempt_key",
))
_ANCHOR_BINDING_FIELDS = frozenset((
    "outer_result_content_sha256",
    "candidate_runtime_source_closure_sha256",
    "final_controller_source_closure_sha256",
    "final_materializer_source_closure_sha256",
    "actor_feature_names_sha256",
))
_CONFIG_PATH_FIELDS = frozenset((
    "permanent_final_registry", "candidate_lock", "actor", "protocol",
    "runtime_manifest", "designation", "development_rows",
    "fresh_outer_registry", "fresh_outer_registry_report",
    "fresh_outer_hash_projection", "fresh_outer_hash_projection_receipt",
    "failed_rows", "original_expansion", "consumed_outer_registry",
    "consumed_outer_report", "formal_selection", "previous_development",
    "retired_identity_projection", "prior_outer_hash_projection",
    "private_salt",
))
_PRIOR_OUTER_PROJECTION_FIELDS = frozenset((
    "version", "source_closeout_content_sha256",
    "source_projection_content_sha256", "outer_observation_hashes",
    "unique_outer_observation_hash_count", "selector_rule",
    "raw_observations_included", "actions_included",
    "probabilities_included", "labels_included", "formal_ready",
    "content_sha256",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "invoked_only_after_permanent_final_claim": True,
        "private_salt_read_after_authenticated_claim": True,
        "whole_scene_count": FINAL_SCENE_COUNT,
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "private_salt_commitment": PRIVATE_SALT_COMMITMENT,
        "selection_inputs": [
            "committed salt", "scene identity", "family",
            "exposure membership", "fixed workload pass",
            "public observation-hash isolation",
        ],
        "raw_actor_actions_used_as_ranking_input": False,
        "actor_logits_or_probabilities_used_as_ranking_input": False,
        "actor_executes_only_inside_fixed_public_workload_replay": True,
        "actor_outputs_written_to_material_artifact": False,
        "program_access": False,
        "program_predictions_access": False,
        "candidate_adaptation": False,
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
    return (
        type(claimed) is str
        and _HEX.fullmatch(claimed) is not None
        and claimed == digest({key: child for key, child in value.items()
                               if key != "content_sha256"})
    )


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
        raise ValueError(label + " must contain one JSON object")
    return value


def _regular(
    value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES,
) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a nonempty bounded canonical regular file")
    return path


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _read_once(
    value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES,
) -> tuple[Path, bytes, str]:
    path = Path(value).expanduser().absolute()
    if path.resolve() != path or path.parent.resolve() != path.parent.absolute():
        raise ValueError(label + " path must be canonical")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_size <= 0
                or info.st_size > maximum):
            raise ValueError(label + " must be a nonempty bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise ValueError(label + " is oversized")
    finally:
        os.close(descriptor)
    return path, raw, sha256(raw).hexdigest()


def _read_json_exact(
    value: str | Path, label: str, *, expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    path, raw, actual = _read_once(value, label)
    if expected_sha256 is not None and actual != _sha(
            expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    parsed = _strict_json_bytes(raw, label)
    if file_hash(path) != actual:
        raise RuntimeError(label + " changed during read")
    return path, parsed, actual


def _authenticate_claim(value: str | Path) -> tuple[Path, dict[str, Any]]:
    """Authenticate the permanent controller anchor before any other input."""
    path, anchor, _ = _read_json_exact(value, "permanent v9 final claim")
    inputs = anchor.get("attempt_key_inputs")
    bindings = anchor.get("bindings")
    sources = producer_sources()
    if (path.name != "attempt_anchor.json"
            or set(anchor) != _ANCHOR_FIELDS
            or anchor.get("version") != ANCHOR_VERSION
            or anchor.get("status") != ANCHOR_STATUS
            or not _content_valid(anchor)
            or not isinstance(inputs, Mapping)
            or set(inputs) != _ATTEMPT_INPUT_FIELDS
            or inputs.get("scheme") != FINAL_ONCE_VERSION + ".candidate-and-outer.v1"
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for name, child in inputs.items() if name != "scheme")
            or anchor.get("attempt_key") != digest(dict(inputs))
            or path.parent.name != anchor.get("attempt_key")
            or not isinstance(bindings, Mapping)
            or set(bindings) != _ANCHOR_BINDING_FIELDS
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for child in bindings.values())
            or bindings.get("final_materializer_source_closure_sha256")
                != digest(sources)
            or anchor.get("candidate_and_outer_authenticated_before_claim") is not True
            or anchor.get("final_identity_or_rows_accessed_before_claim") is not False
            or anchor.get("retry_allowed") is not False
            or anchor.get("formal_ready") is not False):
        raise ValueError("Permanent v9 final claim semantics differ")
    return path, deepcopy(anchor)


def _load_config() -> tuple[Path, dict[str, Path]]:
    supplied = os.environ.get(CONFIG_ENV)
    if not supplied:
        raise ValueError(CONFIG_ENV + " is required after the permanent claim")
    config_path, value, _ = _read_json_exact(supplied, "v9 final materializer config")
    paths = value.get("paths")
    if (set(value) != {"version", "paths", "content_sha256"}
            or value.get("version") != CONFIG_VERSION
            or not _content_valid(value)
            or not isinstance(paths, Mapping)
            or set(paths) != _CONFIG_PATH_FIELDS
            or any(type(child) is not str or not child for child in paths.values())):
        raise ValueError("V9 final materializer configuration differs")
    resolved = {
        name: Path(child).expanduser().absolute() for name, child in paths.items()
    }
    return config_path, resolved


def _authenticate_public_inputs(
    *, claim_path: Path, anchor: Mapping[str, Any], paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Authenticate every non-secret selection input after the claim."""
    permanent = _directory(paths["permanent_final_registry"],
                           "permanent v9 final registry")
    if claim_path.parent.parent != permanent:
        raise ValueError("Final claim is outside the configured permanent registry")
    inputs = anchor["attempt_key_inputs"]
    lock_path, lock, lock_bindings = outer_api._candidate_lock(
        paths["candidate_lock"],
        expected_sha256=inputs["candidate_lock_sha256"])
    if (lock.get("content_sha256") != inputs["candidate_lock_content_sha256"]
            or lock_bindings.get("actor_sha256") != inputs["actor_sha256"]
            or lock_bindings.get("protocol_sha256") != inputs["protocol_sha256"]
            or lock_bindings.get("runtime_manifest_sha256")
                != inputs["runtime_manifest_sha256"]
            or lock_bindings.get("designation_sha256")
                != inputs["designation_sha256"]
            or lock_bindings.get("program_sha256") != inputs["program_sha256"]
            or lock_bindings.get("public_feature_contract_sha256")
                != inputs["public_feature_contract_sha256"]
            or lock_bindings.get("source_closure_sha256")
                != inputs["candidate_source_closure_sha256"]):
        raise ValueError("Final claim and candidate lock differ")

    expected_paths = {
        "actor": "actor_sha256", "protocol": "protocol_sha256",
        "runtime_manifest": "runtime_manifest_sha256",
        "designation": "designation_sha256",
        "development_rows": "development_rows_sha256",
        "fresh_outer_registry": "fresh_outer_registry_sha256",
        "fresh_outer_registry_report": "fresh_outer_registry_report_sha256",
        "fresh_outer_hash_projection": "outer_hash_projection_sha256",
        "fresh_outer_hash_projection_receipt": (
            "outer_hash_projection_receipt_sha256"),
    }
    authenticated: dict[str, Path] = {}
    for name, binding in expected_paths.items():
        maximum = MAX_NPZ_BYTES if name in {"actor", "development_rows"} \
            else MAX_JSON_BYTES
        authenticated[name] = outer_api._authenticate_file(
            paths[name], name.replace("_", " "),
            expected_sha256=lock_bindings[binding], maximum=maximum)

    designation = designation_api.read_bound_designation(
        authenticated["designation"],
        expected_sha256=lock_bindings["designation_sha256"])
    if (designation.get("bindings", {}).get("actor_sha256")
            != lock_bindings["actor_sha256"]
            or designation.get("bindings", {}).get("protocol_file_sha256")
                != lock_bindings["protocol_sha256"]
            or designation.get("evidence", {}).get("action_authority_exact") is not True):
        raise ValueError("Final materializer Actor designation differs")

    registry, registry_report, selected_identity_sha256 = (
        outer_api._registry_bundle(
            authenticated["fresh_outer_registry"],
            authenticated["fresh_outer_registry_report"],
            bindings=lock_bindings))
    projection, projection_receipt = projection_api.read_saved_projection(
        projection_path=authenticated["fresh_outer_hash_projection"],
        receipt_path=authenticated["fresh_outer_hash_projection_receipt"],
        expected_projection_sha256=lock_bindings[
            "outer_hash_projection_sha256"],
        expected_receipt_sha256=lock_bindings[
            "outer_hash_projection_receipt_sha256"],
    )
    identity = projection.get("identity", {})
    if (identity.get("selected_identity_sha256") != selected_identity_sha256
            or identity.get("registry_file_sha256")
                != lock_bindings["fresh_outer_registry_sha256"]
            or identity.get("registry_content_sha256")
                != registry.get("content_sha256")):
        raise ValueError("Fresh outer identity and observation projection differ")

    development = outer_api._safe_row_projection(
        authenticated["development_rows"],
        expected_sha256=lock_bindings["development_rows_sha256"],
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="locked v9 development rows")
    development_hashes = set(outer_api._decode(
        development["observation_hashes"], "development observation hashes"))
    development_scenes = set(outer_api._decode(
        development["scene_fingerprints"], "development scene fingerprints"))
    fresh_outer_hashes = set(projection["projection"][
        "ordered_observation_hashes"])
    fresh_outer_scenes = {
        str(scene["fingerprint"]) for scene in registry["development_outer"]
    }
    if (not development_hashes or not development_scenes
            or not fresh_outer_hashes or not fresh_outer_scenes
            or development_hashes & fresh_outer_hashes
            or development_scenes & fresh_outer_scenes):
        raise ValueError("Development and fresh outer isolation differs")

    return {
        "lock_path": lock_path, "lock": lock, "bindings": lock_bindings,
        "paths": authenticated, "designation": designation,
        "registry": registry, "registry_report": registry_report,
        "projection": projection, "projection_receipt": projection_receipt,
        "development_hashes": development_hashes,
        "development_scenes": development_scenes,
        "fresh_outer_hashes": fresh_outer_hashes,
        "fresh_outer_scenes": fresh_outer_scenes,
    }


def _prior_outer_hashes(
    path: Path, *, expected_file_sha256: str,
    expected_content_sha256: str,
) -> set[str]:
    _, value, _ = _read_json_exact(
        path, "prior failed-outer hash projection",
        expected_sha256=expected_file_sha256)
    values = value.get("outer_observation_hashes")
    if (set(value) != _PRIOR_OUTER_PROJECTION_FIELDS
            or value.get("version")
                != registry_api.VERSION + ".validation-wins-exclusion.v1"
            or not _content_valid(value)
            or value.get("content_sha256") != expected_content_sha256
            or not isinstance(values, list) or not values
            or values != sorted(set(values))
            or value.get("unique_outer_observation_hash_count") != len(values)
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for child in values)
            or any(value.get(name) is not False for name in (
                "raw_observations_included", "actions_included",
                "probabilities_included", "labels_included", "formal_ready"))):
        raise ValueError("Prior failed-outer hash projection differs")
    return set(values)


def _read_frozen_retired_projection(
    path: Path, *, expected_projection_sha256: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    """Authenticate immutable retired evidence across later source evolution."""
    projection_path, projection_value, projection_sha256 = _read_json_exact(
        path, "retired identity projection",
        expected_sha256=expected_projection_sha256)
    projection = retired_api._validate_projection(projection_value)
    report_path, report, report_sha256 = _read_json_exact(
        projection_path.parent / "report.json", "retired projection report",
        expected_sha256=expected_report_sha256)
    frozen_sources = report.get("producer_sources")
    if (not isinstance(frozen_sources, Mapping) or not frozen_sources
            or any(type(name) is not str or not name
                   or type(child) is not str or _HEX.fullmatch(child) is None
                   for name, child in frozen_sources.items())
            or report.get("producer_sources_sha256")
                != digest(dict(frozen_sources))):
        raise ValueError("Retired projection frozen source receipt differs")
    retired_api._validate_report(
        report, projection=projection,
        projection_file_sha256=projection_sha256,
        report_file_sha256=report_sha256,
        expected_report_sha256=expected_report_sha256,
        sources=dict(frozen_sources))
    if (file_hash(projection_path) != projection_sha256
            or file_hash(report_path) != report_sha256):
        raise RuntimeError("Retired identity evidence changed during read")
    return projection


def _exposure_closure(
    *, paths: Mapping[str, Path], registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[int], set[str], set[str]]:
    """Rebuild the identity-only v9 exclusion closure from frozen sources."""
    bindings = registry.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Fresh outer registry bindings are missing")
    _, row_fingerprints = registry_api._scene_fingerprints_only(
        paths["failed_rows"],
        expected_sha256=bindings["failed_rows_file_sha256"])
    _, original = registry_api._strict_json(
        paths["original_expansion"], "original development expansion",
        expected_sha256=bindings["original_expansion_file_sha256"])
    trace_seeds, trace_fingerprints, old_outer_seeds, old_outer_fingerprints = (
        registry_api._source_expansion(original))
    current_file, current = registry_api._strict_json(
        paths["consumed_outer_registry"], "consumed v8 outer registry",
        expected_sha256=bindings["consumed_outer_registry_file_sha256"])
    _, current_report = registry_api._strict_json(
        paths["consumed_outer_report"], "consumed v8 outer report",
        expected_sha256=bindings["consumed_outer_report_file_sha256"])
    current_identities, current_seeds, current_fingerprints, registered_old = (
        registry_api._current_outer(
            current, current_report, registry_sha256=file_hash(current_file)))
    if registered_old != old_outer_fingerprints:
        raise ValueError("Prior outer exposure closure differs")
    _, formal = registry_api._strict_json(
        paths["formal_selection"], "formal X/Y selection",
        expected_sha256=bindings["formal_selection_file_sha256"])
    formal_seeds, formal_fingerprints = registry_api._formal_identities(formal)
    _, previous = registry_api._strict_json(
        paths["previous_development"], "previous development supplement",
        expected_sha256=bindings["previous_development_file_sha256"])
    previous_seeds, previous_fingerprints = registry_api._previous_identities(
        previous)
    retired = _read_frozen_retired_projection(
        paths["retired_identity_projection"],
        expected_projection_sha256=bindings["retired_projection_file_sha256"],
        expected_report_sha256=bindings[
            "retired_projection_report_file_sha256"])
    retired_seeds, retired_fingerprints = registry_api._unique_identities(
        retired.get("exposed_identities"),
        expected_count=retired_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        label="retired exposed projection")

    candidates = registry_api._candidate_identity_population()
    by_fingerprint = {row["fingerprint"]: row for row in candidates}
    if len(by_fingerprint) != registry_api.FIXED_CANDIDATE_SCENE_COUNT:
        raise ValueError("Fixed candidate identity population differs")
    row_candidate_fingerprints = row_fingerprints & set(by_fingerprint)
    row_candidate_seeds = {
        int(by_fingerprint[fingerprint]["seed"])
        for fingerprint in row_candidate_fingerprints
    }
    excluded_seeds = (
        trace_seeds | old_outer_seeds | current_seeds | formal_seeds
        | previous_seeds | retired_seeds | row_candidate_seeds
        | {int(row["seed"]) for row in current_identities}
    )
    excluded_fingerprints = (
        trace_fingerprints | old_outer_fingerprints | current_fingerprints
        | registered_old | formal_fingerprints | previous_fingerprints
        | retired_fingerprints | row_fingerprints
        | {str(row["fingerprint"]) for row in current_identities}
    )
    remaining, selected, remaining_counts = registry_api._remaining_and_selected(
        candidates, excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    selected_public = [{key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")}
        for row in selected]
    actual_outer = registry.get("selected_outer_identities")
    report_selection = registry.get("statistics", {})
    exclusion_counts = registry.get("exclusion_counts")
    exclusion_digests = registry.get("exclusion_digests")
    if (actual_outer != selected_public
            or report_selection.get("remaining_candidate_scene_count")
                != len(remaining)
            or report_selection.get("remaining_family_counts")
                != remaining_counts
            or exclusion_counts != {
                "source_row_scene_fingerprints": len(row_fingerprints),
                "source_rows_in_fixed_candidate_population": len(
                    row_candidate_fingerprints),
                "original_expansion_trace_identities": len(trace_fingerprints),
                "consumed_outer_identities": len(current_identities),
                "formal_xy_identities": len(formal_fingerprints),
                "previous_development_identities": len(previous_fingerprints),
                "retired_exposed_identities": len(retired_fingerprints),
                "union_candidate_identities_excluded": (
                    len(candidates) - len(remaining)),
            }
            or exclusion_digests != {
                "excluded_seeds_sha256": digest(sorted(excluded_seeds)),
                "excluded_scene_fingerprints_sha256": digest(
                    sorted(excluded_fingerprints)),
                "source_row_scene_fingerprints_sha256": digest(
                    sorted(row_fingerprints)),
                "original_expansion_trace_fingerprints_sha256": digest(
                    sorted(trace_fingerprints)),
                "consumed_outer_identities_sha256": digest(current_identities),
                "retired_exposed_fingerprints_sha256": digest(
                    sorted(retired_fingerprints)),
            }):
        raise ValueError("Fresh outer exposed-identity closure does not replay")

    fresh_seeds, fresh_fingerprints = registry_api._unique_identities(
        actual_outer, expected_count=registry_api.FRESH_OUTER_SCENE_COUNT,
        label="fresh v9 outer")
    excluded_seeds.update(fresh_seeds)
    excluded_fingerprints.update(fresh_fingerprints)
    return candidates, excluded_seeds, excluded_fingerprints, row_fingerprints


def _candidate_scene_population(
    *, actor_path: Path, manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = manifest_api.read_saved_manifest(
        manifest_path, expected_sha256=manifest_api.EXPECTED_MANIFEST_SHA256,
        actor_path=actor_path, replay_scope="none")
    authentication = manifest.get("authentication")
    batches, reports = manifest_api.regenerate_development_candidate_batches()
    if (not isinstance(authentication, Mapping)
            or authentication.get(
                "candidate_scenes_disjoint_from_all_base_splits_authenticated")
                is not True
            or authentication.get("protected_final_identity_sha256")
                != manifest_api.EXPECTED_FINAL_IDENTITY_SHA256
            or authentication.get("full_manifest_json_parsed") is not False
            or len(batches) != 3
            or digest(batches)
                != manifest_api.EXPECTED_CANDIDATE_BATCHES_SHA256
            or digest(reports)
                != manifest_api.EXPECTED_CANDIDATE_REPORTS_SHA256):
        raise ValueError("Authenticated public candidate projection required")
    rows = [deepcopy(dict(row)) for batch in batches for row in batch]
    mapping = {str(row.get("fingerprint")): row for row in rows}
    identities = registry_api._candidate_identity_population()
    expected = {(row["seed"], row["fingerprint"], row["family_id"],
                 row["batch_index"]) for row in identities}
    actual = {(row.get("seed"), row.get("fingerprint"), row.get("family_id"),
               row.get("batch_index")) for row in rows}
    if (len(rows) != registry_api.FIXED_CANDIDATE_SCENE_COUNT
            or len(mapping) != len(rows) or actual != expected):
        raise ValueError("Candidate scene and identity projections differ")
    return manifest, mapping


def _ordered_candidates(
    identities: Sequence[Mapping[str, Any]], *, family: str, salt: bytes,
) -> list[dict[str, Any]]:
    rows = [deepcopy(dict(row)) for row in identities
            if row.get("family_id") == family]
    rows.sort(key=lambda row: (digest({
        "salt": salt.hex(), "family_id": family,
        "fingerprint": row["fingerprint"],
    }), row["fingerprint"]))
    return rows


def _select_and_replay(
    *, runtime: Any, identities: Sequence[Mapping[str, Any]],
    scene_by_fingerprint: Mapping[str, Mapping[str, Any]], salt: bytes,
    excluded_seeds: set[int], excluded_fingerprints: set[str],
    forbidden_observation_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use v4 quotas/replay while keeping raw Actor outputs encapsulated."""
    accepted: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    evaluated = 0
    for family in FAMILY_IDS:
        count = 0
        for identity in _ordered_candidates(identities, family=family, salt=salt):
            if count >= FAMILY_QUOTAS[family]:
                break
            seed = int(identity["seed"])
            fingerprint = str(identity["fingerprint"])
            if seed in excluded_seeds or fingerprint in excluded_fingerprints:
                continue
            evaluated += 1
            source = scene_by_fingerprint.get(fingerprint)
            if source is None or source.get("seed") != seed:
                raise ValueError("Ranked final identity lacks its public scene")
            scene_index = len(accepted)
            # The full receipt remains local to this stack frame.  Only its
            # aggregate pass bit is consumed; no action/logit/probability field
            # is a ranking input or enters the material artifact.
            receipt = screen_scene(
                source, split="final_test", scene_index=scene_index,
                actor=runtime.actor)
            if receipt.get("passed") is not True:
                rejection_counts["fixed_workload_failed"] += 1
                continue
            hashes = v4._exact_final_workload_observations(
                runtime, source, scene_index)
            if not hashes:
                raise RuntimeError("Final public workload produced no observations")
            if hashes & forbidden_observation_hashes:
                rejection_counts["prior_public_observation_overlap"] += 1
                continue
            if hashes & accepted_hashes:
                rejection_counts["within_final_public_observation_overlap"] += 1
                continue
            frozen = deepcopy(dict(source))
            frozen.pop("workload_screen", None)
            frozen["id"] = f"diagnostic_v9_fresh_final_{scene_index:04d}"
            frozen["split"] = "fresh_final_test"
            # No replay receipt is embedded because historical receipts may
            # contain Actor-derived action summaries.  Final-once performs an
            # independent physical replay and publishes its own audit.
            accepted.append(frozen)
            accepted_hashes.update(hashes)
            forbidden_observation_hashes.update(hashes)
            excluded_fingerprints.add(fingerprint)
            excluded_seeds.add(seed)
            count += 1
        if count != FAMILY_QUOTAS[family]:
            raise RuntimeError("Fresh-final family quota unavailable: " + family)
    statistics = {
        "accepted": len(accepted), "evaluated": evaluated,
        "families": dict(sorted(Counter(
            row["family_id"] for row in accepted).items())),
        "rejected": dict(sorted(rejection_counts.items())),
        "public_observation_count": len(accepted_hashes),
        "public_observation_overlap": 0,
        "unique_fingerprints": len({row["fingerprint"] for row in accepted}),
        "unique_seeds": len({row["seed"] for row in accepted}),
        "raw_actor_outputs_exposed_to_selector": False,
        "raw_actor_outputs_written": False,
        "runtime_action_override": False,
    }
    if (len(accepted) != FINAL_SCENE_COUNT
            or statistics["families"] != dict(sorted(FAMILY_QUOTAS.items()))
            or statistics["unique_fingerprints"] != FINAL_SCENE_COUNT
            or statistics["unique_seeds"] != FINAL_SCENE_COUNT):
        raise RuntimeError("Fresh-final selection accounting differs")
    return accepted, statistics


def _read_committed_salt(path: Path) -> bytes:
    """Read the existing v4 private salt; caller must already hold a claim."""
    return v4._read_committed_salt(path)


def _prepare_selection(
    *, paths: Mapping[str, Path], authenticated: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and reconstruct all public inputs before reading salt."""
    registry = authenticated["registry"]
    identities, excluded_seeds, excluded_fingerprints, source_row_scenes = (
        _exposure_closure(paths=paths, registry=registry))
    excluded_fingerprints.update(authenticated["development_scenes"])
    candidate_manifest, scene_by_fingerprint = _candidate_scene_population(
        actor_path=authenticated["paths"]["actor"],
        manifest_path=authenticated["paths"]["runtime_manifest"])
    if candidate_manifest.get("frozen_actor", {}).get("sha256") \
            != authenticated["bindings"]["actor_sha256"]:
        raise ValueError("Candidate projection and locked Actor differ")
    by_fingerprint = {row["fingerprint"]: row for row in identities}
    excluded_seeds.update(
        int(by_fingerprint[fingerprint]["seed"])
        for fingerprint in authenticated["development_scenes"]
        if fingerprint in by_fingerprint)

    registry_bindings = registry["bindings"]
    prior_hashes = _prior_outer_hashes(
        paths["prior_outer_hash_projection"],
        expected_file_sha256=registry_bindings[
            "outer_observation_hash_projection_file_sha256"],
        expected_content_sha256=registry_bindings[
            "outer_observation_hash_projection_content_sha256"])
    forbidden = (
        set(authenticated["development_hashes"])
        | set(authenticated["fresh_outer_hashes"])
        | prior_hashes
    )
    runtime = manifest_api.build_runtime(
        actor_path=authenticated["paths"]["actor"],
        protocol_path=authenticated["paths"]["protocol"],
        manifest_path=authenticated["paths"]["runtime_manifest"])
    return {
        "identities": identities,
        "excluded_seeds": excluded_seeds,
        "excluded_fingerprints": excluded_fingerprints,
        "source_row_scenes": source_row_scenes,
        "scene_by_fingerprint": scene_by_fingerprint,
        "forbidden_observation_hashes": forbidden,
        "runtime": runtime,
    }


def _build_material(
    *, anchor: Mapping[str, Any], authenticated: Mapping[str, Any],
    prepared: Mapping[str, Any], salt: bytes,
) -> dict[str, Any]:
    scenes, statistics = _select_and_replay(
        runtime=prepared["runtime"], identities=prepared["identities"],
        scene_by_fingerprint=prepared["scene_by_fingerprint"], salt=salt,
        excluded_seeds=set(prepared["excluded_seeds"]),
        excluded_fingerprints=set(prepared["excluded_fingerprints"]),
        forbidden_observation_hashes=set(
            prepared["forbidden_observation_hashes"]))
    if ({scene["fingerprint"] for scene in scenes}
            & (authenticated["development_scenes"]
               | authenticated["fresh_outer_scenes"]
               | prepared["source_row_scenes"])):
        raise RuntimeError("Final scene identity overlaps prior exposure")
    sources = producer_sources()
    material: dict[str, Any] = {
        "version": MATERIAL_VERSION, "status": MATERIAL_STATUS,
        "claim": {
            "attempt_key": anchor["attempt_key"],
            "attempt_anchor_content_sha256": anchor["content_sha256"],
        },
        "scenes": scenes,
        "selection": {
            "whole_scene_selection": True,
            "scene_count": FINAL_SCENE_COUNT,
            "program_access": False,
            "program_predictions_access": False,
            "actor_outputs_access": False,
            "action_labels_access": False,
            "salt_access_after_permanent_claim": True,
            "candidate_adaptation": False,
            "runtime_action_override": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    material["content_sha256"] = digest(material)
    # Keep local accounting out of the public material schema while requiring
    # it to be internally sound before publication.
    if (statistics["accepted"] != FINAL_SCENE_COUNT
            or statistics["public_observation_overlap"] != 0):
        raise RuntimeError("Final replay accounting differs")
    return material


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def materialize(*, claim: str | Path, output: str | Path) -> dict[str, Any]:
    """Create one material artifact. The claim is always the first read."""
    claim_path, anchor = _authenticate_claim(claim)
    # No environment/configuration, salt, manifest, or identity source is read
    # above this line.  Tests enforce this ordering with hostile callbacks.
    _, paths = _load_config()
    authenticated = _authenticate_public_inputs(
        claim_path=claim_path, anchor=anchor, paths=paths)
    prepared = _prepare_selection(paths=paths, authenticated=authenticated)
    # The secret is deliberately the final selection input opened by this
    # process.  Every mutable public path has already been authenticated and
    # reduced to an immutable in-memory snapshot above.
    salt = _read_committed_salt(paths["private_salt"])
    try:
        material = _build_material(
            anchor=anchor, authenticated=authenticated,
            prepared=prepared, salt=salt)
    finally:
        salt = b""
    destination = Path(output).expanduser().absolute()
    if (destination.parent != claim_path.parent
            or destination.name != "materializer_output.json"
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Materializer output must be the new claim-local result")
    _write_exclusive(destination, material)
    if producer_sources() != material["producer_sources"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Final materializer source closure changed")
    return deepcopy(material)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", required=True,
                        help="Permanent v9 final-once attempt_anchor.json")
    parser.add_argument("--output", required=True,
                        help="Claim-local materializer_output.json")
    args = parser.parse_args(argv)
    result = materialize(claim=args.claim, output=args.output)
    print(canonical({
        "status": result["status"], "scene_count": len(result["scenes"]),
        "output": str(Path(args.output).expanduser().absolute()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "MATERIAL_VERSION", "MATERIAL_STATUS", "CONFIG_VERSION",
    "CONFIG_ENV", "FINAL_SCENE_COUNT", "FAMILY_IDS", "FAMILY_QUOTAS",
    "PRIVATE_SALT_DOMAIN", "PRIVATE_SALT_COMMITMENT",
    "contract", "producer_sources", "materialize", "main",
]
