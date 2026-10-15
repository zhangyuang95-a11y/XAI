"""Materialise the one protected v13 warehouse final split after its claim.

This program is intentionally launched only by
``warehouse_r41_diagnostic_final_once_v13``.  Its first filesystem read is the
already-created permanent attempt anchor.  Only after that anchor has been
strictly authenticated does it read the deployment configuration and the
committed private salt.

The final identities come from the frozen public candidate population.  Every
identity and every public observation already used through the fresh v13
outer is excluded.  Selection uses an independent v13 salt, fixed family
quotas, and the exact observation-only projection of the later final
collector.  The projection advances the frozen Actor inside environment
steps, but never reads or returns actions, logits, probabilities, program
outputs, or labels.  Its hash-only summary is retained so the controller can
prove parity against the rows it independently collects after materialisation.
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
from backend.training import warehouse_r41_diagnostic_outer_collection_v13 as collection_api
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v13 as projection_api
from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as promoted_closeout_api
from backend.training import warehouse_r41_diagnostic_final_observation_projection_v13 as final_projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_once as outer_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_split as registry_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_fit_selector as selector_api
from backend.training import warehouse_r41_diagnostic_retired_identity_projection_v8 as retired_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-materializer.v13"
MATERIAL_VERSION = "warehouse-r41-diagnostic-final-material.v13"
MATERIAL_STATUS = "materialized_after_irrevocable_final_claim"
CONFIG_VERSION = VERSION + ".config.v3"
CONFIG_ENV = "WAREHOUSE_R41_V13_FINAL_MATERIALIZER_CONFIG"
FINAL_ONCE_VERSION = "warehouse-r41-diagnostic-final-once.v13"
ANCHOR_VERSION = FINAL_ONCE_VERSION + ".attempt-anchor.v1"
ANCHOR_STATUS = "final_attempt_irrevocably_claimed"
FINAL_SCENE_COUNT = 64
FINAL_SCENE_OFFSET = 900_000
FAMILY_IDS = tuple(registry_api.FAMILY_IDS)
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
PRIVATE_SALT_DOMAIN = b"warehouse-r41-v13-final-holdout-salt\0"
PRIVATE_SALT_COMMITMENT = (
    "2df27d590f169a927812e6b46816f522a75677be2bb4ea521a576536e9c758b6"
)
PRIVATE_SALT_PATH = Path(
    "/Users/zhangyuang/.config/policylens/"
    "warehouse_r41_diagnostic_v13_holdout_salt.bin")
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
    "final_projection_source_closure_sha256",
    "final_projection_contract_sha256", "private_salt_commitment",
    "private_salt_domain_sha256",
))
_ANCHOR_BINDING_FIELDS = frozenset((
    "outer_result_content_sha256",
    "candidate_runtime_source_closure_sha256",
    "final_controller_source_closure_sha256",
    "final_materializer_source_closure_sha256",
    "actor_feature_names_sha256", "final_projection_source_closure_sha256",
    "final_projection_contract_sha256", "private_salt_commitment",
    "private_salt_domain_sha256",
))
_CONFIG_PATH_FIELDS = frozenset((
    "permanent_final_registry", "candidate_lock", "actor", "protocol",
    "runtime_manifest", "designation", "development_rows",
    "combined_promoted_rows", "promotion_closeout",
    "permanent_promotion_closeout_registry",
    "selector_report",
    "fresh_outer_registry", "fresh_outer_registry_report",
    "fresh_outer_hash_projection", "fresh_outer_hash_projection_receipt",
    "failed_rows", "failed_outer_closeout",
    "permanent_failure_closeout_registry", "original_expansion",
    "consumed_outer_registry",
    "consumed_outer_report", "formal_selection", "previous_development",
    "retired_identity_projection", "consumed_v9_attempt_closeout",
    "permanent_v9_attempt_registry", "burned_v10_final_closeout",
    "permanent_v10_final_closeout_registry", "consumed_v11_outer_closeout",
    "permanent_v11_outer_registry", "burned_v12_final_closeout",
    "permanent_v12_final_closeout_registry",
    "prior_outer_hash_projection", "private_salt",
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
            "exposure membership",
            "public observation-hash isolation",
        ],
        "raw_actor_actions_used_as_ranking_input": False,
        "actor_logits_or_probabilities_used_as_ranking_input": False,
        "actor_executes_only_inside_fixed_public_workload_replay": True,
        "actor_outputs_written_to_material_artifact": False,
        "exact_final_collector_projection": final_projection_api.contract(),
        "full_trajectory_hash_screen_before_acceptance": True,
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
    path, anchor, _ = _read_json_exact(value, "permanent v13 final claim")
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
            or inputs.get("private_salt_commitment")
                != PRIVATE_SALT_COMMITMENT
            or inputs.get("private_salt_domain_sha256")
                != sha256(PRIVATE_SALT_DOMAIN).hexdigest()
            or inputs.get("final_projection_source_closure_sha256")
                != digest(final_projection_api.producer_sources())
            or inputs.get("final_projection_contract_sha256")
                != digest(final_projection_api.contract())
            or anchor.get("attempt_key") != digest(dict(inputs))
            or path.parent.name != anchor.get("attempt_key")
            or not isinstance(bindings, Mapping)
            or set(bindings) != _ANCHOR_BINDING_FIELDS
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for child in bindings.values())
            or bindings.get("final_materializer_source_closure_sha256")
                != digest(sources)
            or bindings.get("final_projection_source_closure_sha256")
                != inputs.get("final_projection_source_closure_sha256")
            or bindings.get("final_projection_contract_sha256")
                != inputs.get("final_projection_contract_sha256")
            or bindings.get("private_salt_commitment")
                != PRIVATE_SALT_COMMITMENT
            or bindings.get("private_salt_domain_sha256")
                != sha256(PRIVATE_SALT_DOMAIN).hexdigest()
            or anchor.get("candidate_and_outer_authenticated_before_claim") is not True
            or anchor.get("final_identity_or_rows_accessed_before_claim") is not False
            or anchor.get("retry_allowed") is not False
            or anchor.get("formal_ready") is not False):
        raise ValueError("Permanent v13 final claim semantics differ")
    return path, deepcopy(anchor)


def _load_config() -> tuple[Path, dict[str, Path]]:
    supplied = os.environ.get(CONFIG_ENV)
    if not supplied:
        raise ValueError(CONFIG_ENV + " is required after the permanent claim")
    config_path, value, _ = _read_json_exact(supplied, "v13 final materializer config")
    paths = value.get("paths")
    if (set(value) != {"version", "paths", "content_sha256"}
            or value.get("version") != CONFIG_VERSION
            or not _content_valid(value)
            or not isinstance(paths, Mapping)
            or set(paths) != _CONFIG_PATH_FIELDS
            or any(type(child) is not str or not child for child in paths.values())):
        raise ValueError("V13 final materializer configuration differs")
    resolved = {
        name: Path(child).expanduser().absolute() for name, child in paths.items()
    }
    return config_path, resolved


def _authenticate_public_inputs(
    *, claim_path: Path, anchor: Mapping[str, Any], paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Authenticate every non-secret selection input after the claim."""
    permanent = _directory(
        paths["permanent_final_registry"], "permanent v13 final registry")
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
        "failed_outer_closeout": "failed_outer_closeout_sha256",
        "promotion_closeout": "promotion_closeout_sha256",
        "fresh_outer_registry": "fresh_outer_registry_sha256",
        "fresh_outer_registry_report": "fresh_outer_registry_report_sha256",
        "prior_outer_hash_projection": "prior_outer_hash_projection_sha256",
        "fresh_outer_hash_projection": "outer_hash_projection_sha256",
        "fresh_outer_hash_projection_receipt": (
            "outer_hash_projection_receipt_sha256"),
        "development_rows": "development_rows_sha256",
        "combined_promoted_rows": "combined_promoted_rows_sha256",
        "selector_report": "selector_report_sha256",
    }
    authenticated: dict[str, Path] = {}
    for name, binding in expected_paths.items():
        maximum = MAX_NPZ_BYTES if name in {
            "actor", "development_rows", "combined_promoted_rows"} \
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
            or designation.get("evidence", {}).get(
                "action_authority_exact") is not True):
        raise ValueError("Final materializer Actor designation differs")

    selector_report = collection_api.authenticate_locked_candidate_selector(
        lock=lock, selector_report_path=authenticated["selector_report"],
        expected_selector_report_sha256=lock_bindings[
            "selector_report_sha256"],
        expected_program_sha256=lock_bindings["program_sha256"])

    promoted_closeout = promoted_closeout_api.read_saved_closeout_public(
        authenticated["promotion_closeout"],
        expected_closeout_sha256=lock_bindings["promotion_closeout_sha256"],
        permanent_closeout_registry=_directory(
            paths["permanent_promotion_closeout_registry"],
            "permanent v13 promotion closeout registry"))
    promoted = promoted_closeout.get("combined_promoted_development")
    promoted_projection = promoted.get("observation_hash_projection") \
        if isinstance(promoted, Mapping) else None
    if (not isinstance(promoted, Mapping)
            or not isinstance(promoted_projection, Mapping)
            or promoted.get("rows_sha256")
                != lock_bindings["combined_promoted_rows_sha256"]
            or file_hash(authenticated["combined_promoted_rows"])
                != promoted.get("rows_sha256")):
        raise ValueError("Combined promoted rows differ from v13 closeout")

    registry, registry_report = registry_api.read_saved_registry(
        authenticated["fresh_outer_registry"],
        authenticated["fresh_outer_registry_report"],
        expected_registry_sha256=lock_bindings[
            "fresh_outer_registry_sha256"],
        expected_report_sha256=lock_bindings[
            "fresh_outer_registry_report_sha256"],
        burned_v12_final_closeout_path=authenticated["promotion_closeout"],
        expected_burned_v12_final_closeout_sha256=lock_bindings[
            "promotion_closeout_sha256"],
        permanent_v12_final_closeout_registry=_directory(
            paths["permanent_promotion_closeout_registry"],
            "permanent v13 promotion closeout registry"))
    selected_identity_sha256 = registry_report["selection"][
        "selected_identity_sha256"]

    prior_projection = selector_api.read_prior_outer_hash_projection(
        authenticated["prior_outer_hash_projection"],
        expected_sha256=lock_bindings["prior_outer_hash_projection_sha256"],
        expected_content_sha256=lock_bindings[
            "prior_outer_hash_projection_content_sha256"])
    projection, projection_receipt = projection_api.read_saved_projection(
        projection_path=authenticated["fresh_outer_hash_projection"],
        receipt_path=authenticated["fresh_outer_hash_projection_receipt"],
        expected_projection_sha256=lock_bindings[
            "outer_hash_projection_sha256"],
        expected_receipt_sha256=lock_bindings[
            "outer_hash_projection_receipt_sha256"])
    identity = projection.get("identity", {})
    receipt_bindings = projection_receipt.get("bindings", {})
    if (identity.get("selected_identity_sha256") != selected_identity_sha256
            or identity.get("registry_file_sha256")
                != lock_bindings["fresh_outer_registry_sha256"]
            or identity.get("registry_content_sha256")
                != registry.get("content_sha256")
            or not isinstance(receipt_bindings, Mapping)
            or receipt_bindings.get("actor_sha256")
                != lock_bindings["actor_sha256"]
            or receipt_bindings.get("fresh_outer_registry_sha256")
                != lock_bindings["fresh_outer_registry_sha256"]
            or receipt_bindings.get("outer_hash_projection_sha256")
                != lock_bindings["outer_hash_projection_sha256"]
            or receipt_bindings.get("outer_hash_projection_content_sha256")
                != projection.get("content_sha256")):
        raise ValueError("Fresh v13 outer identity and projection differ")

    development = outer_api._safe_row_projection(
        authenticated["development_rows"],
        expected_sha256=lock_bindings["development_rows_sha256"],
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="locked base development rows")
    base_ordered = outer_api._decode(
        development["observation_hashes"], "development observation hashes")
    base_scenes = outer_api._decode(
        development["scene_fingerprints"], "development scene fingerprints")
    promoted_rows = outer_api._safe_row_projection(
        authenticated["combined_promoted_rows"],
        expected_sha256=lock_bindings["combined_promoted_rows_sha256"],
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="locked combined promoted development rows")
    promoted_ordered = outer_api._decode(
        promoted_rows["observation_hashes"],
        "combined promoted observation hashes")
    promoted_scenes = outer_api._decode(
        promoted_rows["scene_fingerprints"],
        "combined promoted scene fingerprints")
    development_hashes, development_scenes = (
        _retained_combined_development_hashes(
            base_ordered=base_ordered,
            base_scene_ordered=base_scenes,
            promoted_ordered=promoted_ordered,
            promoted_scene_ordered=promoted_scenes,
            prior_unique_hashes=prior_projection["outer_observation_hashes"],
            promoted_unique_hashes=promoted_projection[
                "outer_observation_hashes"],
            fresh_unique_hashes=projection["projection"][
                "unique_observation_hashes"],
            validation_wins=selector_report["development"][
                "validation_wins"]))
    fresh_outer_hashes = set(projection["projection"][
        "unique_observation_hashes"])
    fresh_outer_scenes = {
        str(scene["fingerprint"]) for scene in registry["development_outer"]
    }
    outer_hashes = _v13_outer_audit_hashes(
        prior_unique_hashes=prior_projection["outer_observation_hashes"],
        promoted_unique_hashes=promoted_projection[
            "outer_observation_hashes"],
        fresh_unique_hashes=projection["projection"][
            "unique_observation_hashes"])
    if (not development_hashes or not development_scenes
            or not fresh_outer_hashes or not fresh_outer_scenes
            or development_hashes & outer_hashes
            or development_scenes & fresh_outer_scenes):
        raise ValueError("Development and v13 outer isolation differs")
    return {
        "lock_path": lock_path, "lock": lock, "bindings": lock_bindings,
        "paths": authenticated, "designation": designation,
        "registry": registry, "registry_report": registry_report,
        "projection": projection, "projection_receipt": projection_receipt,
        "prior_projection": prior_projection,
        "promotion_closeout": promoted_closeout,
        "promoted_projection": promoted_projection,
        "selector_report": selector_report,
        "development_hashes": development_hashes,
        "development_scenes": development_scenes,
        "fresh_outer_hashes": fresh_outer_hashes,
        "fresh_outer_scenes": fresh_outer_scenes,
        "outer_hashes": outer_hashes,
    }

def _retained_development_hashes(
    *, development_ordered: Sequence[str], outer_unique_hashes: Sequence[str],
    validation_wins: Mapping[str, Any],
) -> set[str]:
    """Authenticate the validation-wins mask frozen before candidate fitting."""
    keep = ~np.isin(
        np.asarray(development_ordered, dtype="U64"),
        np.asarray(outer_unique_hashes, dtype="U64"))
    retained = [value for value, selected in zip(development_ordered, keep)
                if bool(selected)]
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    recomputed = {
        "source_rows": len(development_ordered),
        "retained_rows": len(retained),
        "removed_rows": int(np.sum(~keep)),
        "source_unique_observations": len(set(development_ordered)),
        "retained_unique_observations": len(set(retained)),
        "fresh_outer_unique_observations": len(outer_unique_hashes),
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(retained),
    }
    if (not isinstance(validation_wins, Mapping)
            or any(validation_wins.get(name) != value
                   for name, value in recomputed.items())):
        raise ValueError("Locked v11 validation-wins projection differs")
    result = set(retained)
    if result & set(outer_unique_hashes):
        raise ValueError("Development and fresh outer isolation differs")
    return result


def _mask_projection_audit(
    *, ordered_hashes: Sequence[str], exclusions: Sequence[str],
    saved: Mapping[str, Any], label: str,
) -> tuple[list[str], np.ndarray]:
    """Replay one label-blind validation-wins mask from ordered hashes."""
    hashes = list(ordered_hashes)
    outer = list(exclusions)
    if (not hashes or any(type(value) is not str or _HEX.fullmatch(value) is None
                          for value in hashes)
            or outer != sorted(set(outer))
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in outer)):
        raise ValueError(label + " observation hash projection differs")
    keep = ~np.isin(
        np.asarray(hashes, dtype="U64"), np.asarray(outer, dtype="U64"))
    retained = [value for value, selected in zip(hashes, keep) if bool(selected)]
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    expected = {
        "source_rows": len(hashes),
        "retained_rows": len(retained),
        "removed_rows": int(np.sum(~keep)),
        "source_unique_observations": len(set(hashes)),
        "retained_unique_observations": len(set(retained)),
        "fresh_outer_unique_observations": len(outer),
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(retained),
    }
    if (not isinstance(saved, Mapping) or not _content_valid(saved)
            or any(saved.get(name) != value for name, value in expected.items())
            or saved.get("private_development_members_read_before_mask_frozen")
                is not False
            or saved.get("private_members_read_only_after_mask_frozen") is not True
            or saved.get("source_archive_reauthenticated_after_private_read")
                is not True
            or saved.get("all_retained_rows_marked_development") is not True
            or type(saved.get("retained_rows_semantic_sha256")) is not str
            or _HEX.fullmatch(saved["retained_rows_semantic_sha256"]) is None):
        raise ValueError("Locked " + label + " validation-wins projection differs")
    return retained, keep.astype(np.bool_)


def _retained_combined_development_hashes(
    *, base_ordered: Sequence[str], base_scene_ordered: Sequence[str],
    promoted_ordered: Sequence[str], promoted_scene_ordered: Sequence[str],
    prior_unique_hashes: Sequence[str], promoted_unique_hashes: Sequence[str],
    fresh_unique_hashes: Sequence[str], validation_wins: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Replay both frozen v13 label-blind masks without opening targets."""
    prior = selector_api.validation_hash_union(prior_unique_hashes, ())
    promoted = selector_api.validation_hash_union(promoted_unique_hashes, ())
    fresh = selector_api.validation_hash_union(fresh_unique_hashes, ())
    prior_set, promoted_set = set(prior), set(promoted)
    if (not promoted_set or not promoted_set <= prior_set
            or set(promoted_ordered) != promoted_set):
        raise ValueError("Combined promoted rows and prior projection differ")
    if (len(base_ordered) != len(base_scene_ordered)
            or len(promoted_ordered) != len(promoted_scene_ordered)):
        raise ValueError("Development row and scene projections differ")
    historical = sorted(prior_set - promoted_set)
    base_exclusions = selector_api.validation_hash_union(prior, fresh)
    promoted_exclusions = selector_api.validation_hash_union(historical, fresh)
    base_saved = validation_wins.get("base") \
        if isinstance(validation_wins, Mapping) else None
    promoted_saved = validation_wins.get("combined_promoted") \
        if isinstance(validation_wins, Mapping) else None
    base_retained, base_keep = _mask_projection_audit(
        ordered_hashes=base_ordered, exclusions=base_exclusions,
        saved=base_saved, label="base development")
    promoted_retained, promoted_keep = _mask_projection_audit(
        ordered_hashes=promoted_ordered, exclusions=promoted_exclusions,
        saved=promoted_saved, label="combined promoted development")
    base_set, promoted_retained_set = set(base_retained), set(promoted_retained)
    retained = base_set | promoted_retained_set
    retained_scenes = {
        str(scene) for scene, selected in zip(base_scene_ordered, base_keep)
        if bool(selected)
    } | {
        str(scene) for scene, selected in zip(
            promoted_scene_ordered, promoted_keep) if bool(selected)
    }
    expected = {
        "precedence": (
            "fresh-v13-validation > combined-promoted-development > "
            "older-development"),
        "prior_unique_observations": len(prior),
        "historical_unique_observations": len(historical),
        "combined_promoted_unique_observations": len(promoted),
        "fresh_v13_unique_observations": len(fresh),
        "combined_promoted_projection_sha256": digest(promoted),
        "combined_source_rows": len(base_ordered) + len(promoted_ordered),
        "combined_retained_rows": len(base_retained) + len(promoted_retained),
        "combined_retained_scene_count": len(retained_scenes),
        "cross_component_observation_overlap": 0,
        "retained_fresh_outer_observation_overlap": 0,
        "all_retained_rows_marked_development": True,
        "private_members_read_only_after_both_masks_frozen": True,
        "both_source_archives_reauthenticated_after_private_read": True,
    }
    if (not isinstance(validation_wins, Mapping)
            or not _content_valid(validation_wins)
            or any(validation_wins.get(name) != value
                   for name, value in expected.items())
            or type(validation_wins.get("retained_rows_semantic_sha256"))
                is not str
            or _HEX.fullmatch(
                validation_wins["retained_rows_semantic_sha256"]) is None
            or base_set & promoted_retained_set
            or retained & set(fresh)):
        raise ValueError("Locked combined v13 validation-wins projection differs")
    exposed_scenes = set(map(str, base_scene_ordered)) | set(map(
        str, promoted_scene_ordered))
    return retained, exposed_scenes


def _v13_outer_audit_hashes(
    *, prior_unique_hashes: Sequence[str],
    promoted_unique_hashes: Sequence[str],
    fresh_unique_hashes: Sequence[str],
) -> set[str]:
    """Separate promoted development evidence from true outer evidence."""
    prior = set(selector_api.validation_hash_union(prior_unique_hashes, ()))
    promoted = set(selector_api.validation_hash_union(
        promoted_unique_hashes, ()))
    fresh = set(selector_api.validation_hash_union(fresh_unique_hashes, ()))
    if not promoted or not promoted <= prior:
        raise ValueError("Combined promoted and prior projections differ")
    return (prior - promoted) | fresh

def _exposure_closure(
    *, paths: Mapping[str, Path], registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[int], set[str], set[str], dict[str, Any]]:
    """Replay the complete identity closure through the fresh v13 outer."""
    bindings = registry.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Fresh v13 outer registry bindings are missing")
    closure = registry_api.replay_exclusion_closure(
        actor_path=paths["actor"], protocol_path=paths["protocol"],
        designation_path=paths["designation"],
        manifest_path=paths["runtime_manifest"],
        failed_rows_path=paths["failed_rows"],
        original_expansion_path=paths["original_expansion"],
        current_outer_registry_path=paths["consumed_outer_registry"],
        current_outer_report_path=paths["consumed_outer_report"],
        failure_closeout_path=paths["failed_outer_closeout"],
        expected_failure_closeout_sha256=bindings[
            "failure_closeout_file_sha256"],
        permanent_closeout_registry=paths[
            "permanent_failure_closeout_registry"],
        formal_selection_path=paths["formal_selection"],
        previous_development_path=paths["previous_development"],
        retired_projection_path=paths["retired_identity_projection"],
        consumed_v9_attempt_closeout_path=paths[
            "consumed_v9_attempt_closeout"],
        expected_consumed_v9_attempt_closeout_sha256=bindings[
            "consumed_v9_attempt_closeout_file_sha256"],
        permanent_v9_attempt_registry=paths["permanent_v9_attempt_registry"],
        burned_v10_final_closeout_path=paths["burned_v10_final_closeout"],
        expected_burned_v10_final_closeout_sha256=bindings[
            "burned_v10_final_closeout_file_sha256"],
        permanent_v10_final_closeout_registry=paths[
            "permanent_v10_final_closeout_registry"],
        consumed_v11_outer_closeout_path=paths[
            "consumed_v11_outer_closeout"],
        expected_consumed_v11_outer_closeout_sha256=bindings[
            "consumed_v11_outer_closeout_file_sha256"],
        permanent_v11_outer_attempt_registry=paths[
            "permanent_v11_outer_registry"],
        burned_v12_final_closeout_path=paths["burned_v12_final_closeout"],
        expected_burned_v12_final_closeout_sha256=bindings[
            "burned_v12_final_closeout_file_sha256"],
        permanent_v12_final_closeout_registry=paths[
            "permanent_v12_final_closeout_registry"],
    )
    candidates = [deepcopy(dict(row)) for row in closure["candidates"]]
    excluded_seeds = set(map(int, closure["excluded_seeds"]))
    excluded_fingerprints = set(map(str, closure["excluded_fingerprints"]))
    row_fingerprints = set(map(str, closure["row_fingerprints"]))
    current = registry.get("selected_outer_identities")
    if not isinstance(current, list) or len(current) != registry_api.FRESH_OUTER_SCENE_COUNT:
        raise ValueError("Fresh v13 outer selected identities are missing")
    for raw in current:
        if not isinstance(raw, Mapping):
            raise ValueError("Fresh v13 outer identity differs")
        excluded_seeds.add(int(raw["seed"]))
        excluded_fingerprints.add(str(raw["fingerprint"]))
    return (candidates, excluded_seeds, excluded_fingerprints,
            row_fingerprints, closure)

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
    identities = registry_api.v12.v11.v9._candidate_identity_population()
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
    """Select only from identities and exact, action-blind observation hashes."""
    accepted: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    accepted_ordered_hashes: list[str] = []
    projection_summaries: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    evaluated = 0
    environment_steps = 0
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
            projected = final_projection_api.validate_projection(
                final_projection_api.project_observation_hashes(
                    runtime, [source],
                    scene_offset=FINAL_SCENE_OFFSET + scene_index,
                    dense_critical=False))
            hashes = set(projected["unique_observation_hashes"])
            if not hashes:
                raise RuntimeError("Final observation-only replay produced no rows")
            if hashes & forbidden_observation_hashes:
                rejection_counts["prior_public_observation_overlap"] += 1
                continue
            if hashes & accepted_hashes:
                rejection_counts[
                    "within_final_public_observation_overlap"] += 1
                continue
            frozen = deepcopy(dict(source))
            frozen.pop("workload_screen", None)
            frozen["id"] = f"diagnostic_v13_fresh_final_{scene_index:04d}"
            frozen["split"] = "fresh_final_test"
            accepted.append(frozen)
            accepted_hashes.update(hashes)
            accepted_ordered_hashes.extend(projected[
                "ordered_observation_hashes"])
            forbidden_observation_hashes.update(hashes)
            excluded_fingerprints.add(fingerprint)
            excluded_seeds.add(seed)
            environment_steps += int(projected["environment_steps"])
            summary = deepcopy(projected["scenes"][0])
            summary["accepted_scene_index"] = scene_index
            projection_summaries.append(summary)
            count += 1
        if count != FAMILY_QUOTAS[family]:
            raise RuntimeError("Fresh-final family quota unavailable: " + family)
    projection_sources = final_projection_api.producer_sources()
    projection_summary = {
        "version": final_projection_api.VERSION,
        "contract_sha256": digest(final_projection_api.contract()),
        "producer_sources_sha256": digest(projection_sources),
        "scene_offset": FINAL_SCENE_OFFSET,
        "scene_count": len(accepted),
        "partners": list(final_projection_api.contract()["partners"]),
        "critical_anchor_period": 5,
        "dense_critical": False,
        "row_count": len(accepted_ordered_hashes),
        "ordered_observation_hashes_sha256": digest(
            accepted_ordered_hashes),
        "unique_observation_count": len(accepted_hashes),
        "unique_observation_hashes_sha256": digest(sorted(accepted_hashes)),
        "scene_summaries": projection_summaries,
        "scene_summaries_sha256": digest(projection_summaries),
        "environment_steps": environment_steps,
        "prior_observation_overlap": 0,
        "within_final_observation_overlap": 0,
        "actions_read": False,
        "probabilities_read": False,
        "program_access": False,
        "labels_read": False,
    }
    projection_summary["content_sha256"] = digest(projection_summary)
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
        "projection": projection_summary,
    }
    if (len(accepted) != FINAL_SCENE_COUNT
            or statistics["families"] != dict(sorted(FAMILY_QUOTAS.items()))
            or statistics["unique_fingerprints"] != FINAL_SCENE_COUNT
            or statistics["unique_seeds"] != FINAL_SCENE_COUNT
            or projection_summary["scene_count"] != FINAL_SCENE_COUNT):
        raise RuntimeError("Fresh-final selection accounting differs")
    return accepted, statistics


def _read_committed_salt(path: Path) -> bytes:
    """Open and authenticate the independent v13 salt after the claim."""
    if Path(path).expanduser().absolute() != PRIVATE_SALT_PATH:
        raise ValueError("Only the frozen private v13 salt path is allowed")
    candidate, raw, _actual = _read_once(
        path, "private v13 holdout salt", maximum=32)
    info = candidate.stat(follow_symlinks=False)
    if (len(raw) != 32 or stat.S_IMODE(info.st_mode) != 0o600
            or sha256(PRIVATE_SALT_DOMAIN + raw).hexdigest()
                != PRIVATE_SALT_COMMITMENT):
        raise ValueError("Private v13 salt does not match its frozen commitment")
    return raw


def _prepare_selection(
    *, paths: Mapping[str, Path], authenticated: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct all public exclusions before the private salt is opened."""
    if (paths["burned_v12_final_closeout"]
            != paths["promotion_closeout"]
            or paths["permanent_v12_final_closeout_registry"]
                != paths["permanent_promotion_closeout_registry"]):
        raise ValueError("V12 promotion closeout paths must be identical")
    registry = authenticated["registry"]
    (identities, excluded_seeds, excluded_fingerprints, source_row_scenes,
     exposure_closure) = _exposure_closure(paths=paths, registry=registry)
    if (exposure_closure["burned_v12_closeout"]
            != authenticated["promotion_closeout"]):
        raise ValueError("V12 promotion closeout changed across authentication")
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

    expected_prior = registry_api._prior_validation_wins_projection(
        old_closeout=exposure_closure["old_closeout"],
        attempt_closeout=exposure_closure["attempt_closeout"],
        final_closeout=exposure_closure["burned_final_closeout"],
        consumed_v11_closeout=exposure_closure["consumed_v11_closeout"],
        consumed_v12_outer=exposure_closure["consumed_v12_outer"],
        burned_v12_final=exposure_closure["burned_v12_final"],
        burned_v12_closeout=exposure_closure["burned_v12_closeout"])
    if expected_prior != authenticated["prior_projection"]:
        raise ValueError("Complete v8-v12 prior observation union differs")
    prior_hashes = set(authenticated["prior_projection"][
        "outer_observation_hashes"])
    forbidden = (
        set(authenticated["development_hashes"])
        | set(authenticated["fresh_outer_hashes"])
        | prior_hashes)
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
            "exact_full_trajectory_observation_screening": True,
        },
        "observation_projection": statistics["projection"],
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    material["content_sha256"] = digest(material)
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
                        help="Permanent v13 final-once attempt_anchor.json")
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
    "CONFIG_ENV", "FINAL_SCENE_COUNT", "FINAL_SCENE_OFFSET", "FAMILY_IDS", "FAMILY_QUOTAS",
    "PRIVATE_SALT_DOMAIN", "PRIVATE_SALT_COMMITMENT", "PRIVATE_SALT_PATH",
    "contract", "producer_sources", "materialize", "main",
]
