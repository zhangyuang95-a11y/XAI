"""Irrevocable v13 final-test controller for the warehouse explanation program.

The controller authenticates the locked candidate and its passed one-shot
fresh outer before it creates a permanent ``O_EXCL`` claim. Final identities
are materialised only after it imports a source-bound producer after that
claim. A failure burns the attempt and is never retryable.

This module intentionally contains no final salt and importing or testing it
cannot reveal a final identity.  The final materializer is a separate, source-
bound component that may read its secret only after this controller calls it.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_api
from backend.training import warehouse_r41_diagnostic_explanation_audit_v9 as audit_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_api
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v13 as projection_api
from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as promoted_closeout_api
from backend.training import warehouse_r41_diagnostic_final_observation_projection_v13 as final_projection_api
from backend.training import warehouse_r41_diagnostic_outer_collection_v13 as collection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_fit_selector as selector_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_once as outer_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_split as registry_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-once.v13"
ROOT = Path(__file__).resolve().parents[2]
MATERIAL_VERSION = "warehouse-r41-diagnostic-final-material.v13"
MATERIAL_STATUS = "materialized_after_irrevocable_final_claim"
STATUS_PASSED = "completed_passed"
STATUS_FAILED = "burned_failed"
ANCHOR_NAME = "attempt_anchor.json"
COMPLETION_NAME = "attempt_completed.json"
MATERIAL_NAME = "final_material.json"
ROWS_NAME = "final_rows.npz"
AUDIT_NAME = "explanation_audit.json"
PARITY_NAME = "observation_projection_parity.json"
FINAL_SCENE_OFFSET = 900_000
MAX_JSON_BYTES = 512 * 1024 * 1024
OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH = (
    "backend/training/warehouse_r41_diagnostic_final_materializer_v13.py"
)
# Frozen before the protected-final attempt.  This is the digest of the
# materializer's complete transitive local source closure, not merely the
# entry-point file.  It may be re-frozen only before the first protected-final
# claim; any later producer change requires a new protocol version rather than
# silently changing a consumed final evaluator.
OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256 = (
    "fa2c1fcc84b12830b83accfa9462b54a6387ff9670c6bffb775c0840eeb1bdd5"
)
PRIVATE_SALT_DOMAIN = b"warehouse-r41-v13-final-holdout-salt\0"
PRIVATE_SALT_COMMITMENT = (
    "2df27d590f169a927812e6b46816f522a75677be2bb4ea521a576536e9c758b6"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MATERIAL_FIELDS = frozenset((
    "version", "status", "claim", "scenes", "selection",
    "observation_projection",
    "producer_sources", "producer_sources_sha256", "formal_ready",
    "content_sha256",
))
_MATERIAL_SELECTION_FIELDS = frozenset((
    "whole_scene_selection", "scene_count", "program_access",
    "program_predictions_access", "actor_outputs_access",
    "action_labels_access", "salt_access_after_permanent_claim",
    "candidate_adaptation", "runtime_action_override",
    "exact_full_trajectory_observation_screening",
))
_SUCCESS_COMPLETION_FIELDS = frozenset((
    "version", "status", "attempt_key", "attempt_anchor_content_sha256",
    "artifacts", "final_material_content_sha256", "audit_content_sha256",
    "projection_parity_content_sha256",
    "final_nine_gates_passed", "physical_counterfactual_replay_passed",
    "whole_scene_and_observation_isolation_passed", "program_fits",
    "materializer_collector_projection_parity_passed",
    "actor_updates", "runtime_action_override", "retry_allowed",
    "producer_sources_sha256", "formal_ready", "content_sha256",
))
_FAILURE_COMPLETION_FIELDS = frozenset((
    "version", "status", "attempt_key", "attempt_anchor_content_sha256",
    "reason", "final_consumed", "retry_allowed", "program_fits",
    "actor_updates", "runtime_action_override", "producer_sources_sha256",
    "formal_ready", "content_sha256",
))

def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "required_predecessor": outer_api.STATUS_PASSED,
        "permanent_o_excl_claim_before_final_identity_or_rows": True,
        "attempt_key_independent_of_final_identity_output_and_materializer": True,
        "whole_scene_isolation": True,
        "zero_cross_split_public_observation_overlap": True,
        "exact_materializer_collector_projection_parity": True,
        "projection_source_authenticated_before_claim": True,
        "private_salt_commitment": PRIVATE_SALT_COMMITMENT,
        "private_salt_domain_sha256": sha256(PRIVATE_SALT_DOMAIN).hexdigest(),
        "independent_physical_counterfactual_replay": True,
        "final_hard_gate_count": 9,
        "program_controls_runtime_actions": False,
        "runtime_action_override": False,
        "retry_allowed": False,
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


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _materializer_identity(source_path: str | Path) -> dict[str, str]:
    """Bind materializer source without importing or executing it."""
    source = Path(source_path).expanduser().absolute()
    if (not source.is_file() or source.is_symlink() or source.resolve() != source
            or source.suffix != ".py"):
        raise ValueError("Final materializer must be one canonical Python source")
    return dict(sorted(local_source_hashes((source,)).items()))


def _authenticate_official_materializer_source(
    source_path: str | Path,
) -> tuple[Path, dict[str, str]]:
    """Authenticate the one repository materializer allowed in production.

    ``source_path`` is an internal unit seam.  Production reaches this helper
    only through :func:`_official_materializer_binding`, which supplies the
    fixed repository path and exposes no caller-controlled override.
    """

    source = Path(source_path).expanduser().absolute()
    official = (ROOT / OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH).absolute()
    if (source != official or source.resolve() != official
            or official.relative_to(ROOT).as_posix()
                != OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH):
        raise ValueError("Only the frozen repository final materializer is allowed")
    sources = _materializer_identity(source)
    if (sources.get(OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH)
            != file_hash(official)
            or digest(sources)
                != OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256):
        raise RuntimeError("Frozen final materializer source closure differs")
    return official, sources


def _official_materializer_binding() -> tuple[Path, dict[str, str]]:
    """Return the fixed production materializer and its frozen source map."""

    return _authenticate_official_materializer_source(
        ROOT / OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH)


def _run_materializer(
    source_path: str | Path, anchor: Mapping[str, Any], campaign: Path,
) -> Mapping[str, Any]:
    """Execute the salt-bearing producer in an isolated post-claim process."""
    source = Path(source_path).expanduser().absolute()
    output = campaign / "materializer_output.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else "")
    subprocess.run(
        [sys.executable, str(source), "--claim", str(campaign / ANCHOR_NAME),
         "--output", str(output)], check=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600,
        cwd=ROOT, env=environment)
    if (not output.is_file() or output.is_symlink() or output.resolve() != output
            or output.stat(follow_symlinks=False).st_size <= 0
            or output.stat(follow_symlinks=False).st_size > MAX_JSON_BYTES):
        raise ValueError("Final materializer did not publish one bounded result")
    value = outer_api._strict_json_bytes(
        output.read_bytes(), "post-claim final materializer output")
    if not isinstance(value, Mapping):
        raise ValueError("Final materializer must return one mapping")
    return value


def _strict_outer_pass(value: Mapping[str, Any], *, bindings: Mapping[str, str],
                       candidate_lock_sha256: str) -> None:
    metrics = value.get("metrics")
    gate = value.get("gate")
    try:
        recomputed = metrics_api._gate(metrics)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V13 outer result metrics cannot be gated") from error
    if (value.get("status") != outer_api.STATUS_PASSED
            or value.get("candidate_lock_sha256") != candidate_lock_sha256
            or value.get("program_sha256") != bindings["program_sha256"]
            or value.get("actor_feature_names_sha256")
                != bindings["actor_feature_names_sha256"]
            or value.get("explanation_eligible") is not True
            or gate != recomputed or recomputed.get("passed") is not True
            or value.get("row_accounting", {}).get(
                "all_submitted_actions_equal_policy_actions") is not True
            or value.get("row_accounting", {}).get("runtime_action_overrides") != 0
            or value.get("execution", {}).get("candidate_refit") is not False
            or value.get("execution", {}).get("program_mutated") is not False):
        raise ValueError("A passed immutable v12 fresh outer is required")


def _authenticate_preclaim(
    *, candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    runtime_manifest_path: str | Path, designation_path: str | Path,
    failed_outer_closeout_path: str | Path,
    promotion_closeout_path: str | Path,
    expected_promotion_closeout_sha256: str,
    permanent_promotion_closeout_registry: str | Path,
    fresh_outer_registry_path: str | Path,
    fresh_outer_registry_report_path: str | Path,
    prior_outer_hash_projection_path: str | Path,
    outer_hash_projection_path: str | Path,
    outer_hash_projection_receipt_path: str | Path,
    development_rows_path: str | Path,
    combined_promoted_rows_path: str | Path,
    program_path: str | Path,
    selector_report_path: str | Path, outer_result_path: str | Path,
    expected_outer_result_sha256: str, outer_permanent_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate public/development evidence without any salt or final input."""
    lock_path, lock, bindings = outer_api._candidate_lock(
        candidate_lock_path, expected_sha256=expected_candidate_lock_sha256)
    if (_sha(expected_promotion_closeout_sha256, "promotion closeout")
            != bindings["promotion_closeout_sha256"]):
        raise ValueError("Expected promotion closeout differs from candidate lock")
    paths = outer_api._verify_lock_files(
        bindings, actor=actor_path, protocol=protocol_path,
        runtime_manifest=runtime_manifest_path, designation=designation_path,
        failed_outer_closeout=failed_outer_closeout_path,
        promotion_closeout=promotion_closeout_path,
        fresh_outer_registry=fresh_outer_registry_path,
        fresh_outer_registry_report=fresh_outer_registry_report_path,
        prior_outer_hash_projection=prior_outer_hash_projection_path,
        outer_hash_projection=outer_hash_projection_path,
        outer_hash_projection_receipt=outer_hash_projection_receipt_path,
        development_rows=development_rows_path,
        combined_promoted_rows=combined_promoted_rows_path,
        program=program_path, selector_report=selector_report_path)
    promoted_closeout = promoted_closeout_api.read_saved_closeout_public(
        paths["promotion_closeout"],
        expected_closeout_sha256=expected_promotion_closeout_sha256,
        permanent_closeout_registry=permanent_promotion_closeout_registry)
    promoted = promoted_closeout.get("combined_promoted_development")
    promoted_projection = promoted.get("observation_hash_projection") \
        if isinstance(promoted, Mapping) else None
    if (not isinstance(promoted, Mapping)
            or not isinstance(promoted_projection, Mapping)
            or promoted.get("rows_sha256")
                != bindings["combined_promoted_rows_sha256"]):
        raise ValueError("Combined promoted rows differ from closeout")
    selector_report = collection_api.authenticate_locked_candidate_selector(
        lock=lock, selector_report_path=paths["selector_report"],
        expected_selector_report_sha256=bindings["selector_report_sha256"],
        expected_program_sha256=bindings["program_sha256"])
    candidate_sources, runtime_sources = outer_api._validate_candidate_source_closure(
        lock, paths["selector_report"], bindings)
    registry, registry_report, selected_identity_sha256 = outer_api._registry_bundle(
        paths["fresh_outer_registry"], paths["fresh_outer_registry_report"],
        bindings=bindings)
    saved_registry, saved_registry_report = registry_api.read_saved_registry(
        paths["fresh_outer_registry"], paths["fresh_outer_registry_report"],
        expected_registry_sha256=bindings["fresh_outer_registry_sha256"],
        expected_report_sha256=bindings["fresh_outer_registry_report_sha256"],
        burned_v12_final_closeout_path=paths["promotion_closeout"],
        expected_burned_v12_final_closeout_sha256=(
            expected_promotion_closeout_sha256),
        permanent_v12_final_closeout_registry=(
            permanent_promotion_closeout_registry))
    if saved_registry != registry or saved_registry_report != registry_report:
        raise RuntimeError("V13 registry readers disagree")
    prior_projection = selector_api.read_prior_outer_hash_projection(
        paths["prior_outer_hash_projection"],
        expected_sha256=bindings["prior_outer_hash_projection_sha256"],
        expected_content_sha256=bindings[
            "prior_outer_hash_projection_content_sha256"])
    program_payload = outer_api._program_payload(paths["program"])
    actor_feature_names, feature_sha256 = outer_api._authenticate_program_actor_features(
        program_payload=program_payload, actor_path=paths["actor"],
        bindings=bindings)
    projection, projection_receipt = projection_api.read_saved_projection(
        projection_path=paths["outer_hash_projection"],
        receipt_path=paths["outer_hash_projection_receipt"],
        expected_projection_sha256=bindings["outer_hash_projection_sha256"],
        expected_receipt_sha256=bindings[
            "outer_hash_projection_receipt_sha256"])
    receipt_bindings = projection_receipt.get("bindings", {})
    if (projection["identity"].get("registry_file_sha256")
            != bindings["fresh_outer_registry_sha256"]
            or projection["identity"].get("registry_content_sha256")
                != registry["content_sha256"]
            or projection["identity"].get("selected_identity_sha256")
                != selected_identity_sha256
            or not isinstance(receipt_bindings, Mapping)
            or receipt_bindings.get("actor_sha256")
                != bindings["actor_sha256"]
            or receipt_bindings.get("fresh_outer_registry_sha256")
                != bindings["fresh_outer_registry_sha256"]
            or receipt_bindings.get("outer_hash_projection_sha256")
                != bindings["outer_hash_projection_sha256"]):
        raise ValueError("V13 outer identity/projection binding differs")

    manifest_api.read_saved_manifest(
        paths["runtime_manifest"],
        expected_sha256=bindings["runtime_manifest_sha256"],
        actor_path=paths["actor"], replay_scope="none")
    designation = designation_api.read_bound_designation(
        paths["designation"], expected_sha256=bindings["designation_sha256"])
    if (designation.get("runtime_action_override") is not False
            or designation.get("bindings", {}).get("actor_sha256")
                != bindings["actor_sha256"]):
        raise ValueError("Frozen designation does not preserve Actor authority")

    outer_result = outer_api.read_saved_result(
        outer_result_path, expected_result_sha256=expected_outer_result_sha256,
        permanent_registry=outer_permanent_registry)
    _strict_outer_pass(
        outer_result, bindings=bindings,
        candidate_lock_sha256=file_hash(lock_path))
    development = outer_api._safe_row_projection(
        paths["development_rows"],
        expected_sha256=bindings["development_rows_sha256"],
        fields=frozenset(("observation_hashes", "scene_fingerprints")),
        label="locked base development rows")
    base_ordered = outer_api._decode(
        development["observation_hashes"], "development observation hashes")
    base_scenes = outer_api._decode(
        development["scene_fingerprints"], "development scene fingerprints")
    promoted_rows = outer_api._safe_row_projection(
        paths["combined_promoted_rows"],
        expected_sha256=bindings["combined_promoted_rows_sha256"],
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
            base_ordered=base_ordered, base_scene_ordered=base_scenes,
            promoted_ordered=promoted_ordered,
            promoted_scene_ordered=promoted_scenes,
            prior_unique_hashes=prior_projection["outer_observation_hashes"],
            promoted_unique_hashes=promoted_projection[
                "outer_observation_hashes"],
            fresh_unique_hashes=projection["projection"][
                "unique_observation_hashes"],
            validation_wins=selector_report["development"][
                "validation_wins"]))
    outer_hashes = _v13_outer_audit_hashes(
        prior_unique_hashes=prior_projection["outer_observation_hashes"],
        promoted_unique_hashes=promoted_projection[
            "outer_observation_hashes"],
        fresh_unique_hashes=projection["projection"][
            "unique_observation_hashes"])
    outer_scenes = {str(scene["fingerprint"])
                    for scene in registry["development_outer"]}
    if development_hashes & outer_hashes or development_scenes & outer_scenes:
        raise ValueError("Locked development and passed outer splits overlap")
    projection_sources = final_projection_api.producer_sources()
    projection_contract = final_projection_api.contract()
    if (projection_contract.get("runtime_decision_output_read") is not False
            or projection_contract.get("raw_observations_returned") is not False
            or projection_contract.get("actor_actions_returned") is not False
            or projection_contract.get("actor_probabilities_returned") is not False
            or projection_contract.get("program_access") is not False
            or projection_contract.get("labels_returned") is not False):
        raise ValueError("Final observation projector is not target blind")
    return {
        "lock_path": lock_path, "lock": lock, "bindings": bindings,
        "paths": paths, "program_payload": program_payload,
        "actor_feature_names": actor_feature_names,
        "actor_feature_names_sha256": feature_sha256,
        "candidate_sources": candidate_sources,
        "runtime_sources": runtime_sources,
        "outer_result": outer_result,
        "outer_result_sha256": expected_outer_result_sha256,
        "outer_registry": registry,
        "outer_registry_report": registry_report,
        "development_hashes": development_hashes,
        "development_scenes": development_scenes,
        "outer_hashes": outer_hashes, "outer_scenes": outer_scenes,
        "promotion_closeout": promoted_closeout,
        "projection_sources": projection_sources,
        "projection_contract": projection_contract,
    }

def _retained_development_hashes(
    *, development_ordered: Sequence[str], outer_unique_hashes: Sequence[str],
    validation_wins: Mapping[str, Any],
) -> set[str]:
    """Reproduce the locked v11 validation-wins projection before final use."""
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
        raise ValueError("Locked development and passed outer splits overlap")
    return result


def _mask_projection_audit(
    *, ordered_hashes: Sequence[str], exclusions: Sequence[str],
    saved: Mapping[str, Any], label: str,
) -> tuple[list[str], np.ndarray]:
    """Replay one v12 label-blind keep mask from its public hashes."""
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
    retained = [value for value, selected in zip(hashes, keep)
                if bool(selected)]
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


def _validate_material(
    value: Mapping[str, Any], *, anchor: Mapping[str, Any],
    materializer_sources: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = value.get("selection")
    claim = value.get("claim")
    sources = value.get("producer_sources")
    scenes = value.get("scenes")
    projection = value.get("observation_projection")
    if (set(value) != _MATERIAL_FIELDS or value.get("version") != MATERIAL_VERSION
            or value.get("status") != MATERIAL_STATUS
            or value.get("formal_ready") is not False or not _content_valid(value)
            or claim != {
                "attempt_key": anchor["attempt_key"],
                "attempt_anchor_content_sha256": anchor["content_sha256"],
            }
            or not isinstance(selection, Mapping)
            or set(selection) != _MATERIAL_SELECTION_FIELDS
            or selection.get("whole_scene_selection") is not True
            or selection.get("scene_count") != audit_api.FINAL_SCENE_COUNT
            or any(selection.get(name) is not False for name in (
                "program_access", "program_predictions_access",
                "actor_outputs_access", "action_labels_access",
                "candidate_adaptation", "runtime_action_override"))
            or selection.get("salt_access_after_permanent_claim") is not True
            or selection.get(
                "exact_full_trajectory_observation_screening") is not True
            or not isinstance(sources, Mapping)
            or dict(sources) != dict(materializer_sources)
            or value.get("producer_sources_sha256") != digest(dict(sources))
            or not isinstance(scenes, list)
            or len(scenes) != audit_api.FINAL_SCENE_COUNT
            or not isinstance(projection, Mapping)
            or projection.get("version") != final_projection_api.VERSION
            or projection.get("contract_sha256")
                != digest(final_projection_api.contract())
            or projection.get("producer_sources_sha256")
                != digest(final_projection_api.producer_sources())
            or projection.get("scene_offset") != FINAL_SCENE_OFFSET
            or projection.get("scene_count") != audit_api.FINAL_SCENE_COUNT
            or projection.get("partners")
                != list(final_projection_api.contract()["partners"])
            or projection.get("critical_anchor_period") != 5
            or projection.get("dense_critical") is not False
            or any(projection.get(name) is not False for name in (
                "actions_read", "probabilities_read", "program_access",
                "labels_read"))
            or projection.get("prior_observation_overlap") != 0
            or projection.get("within_final_observation_overlap") != 0
            or projection.get("content_sha256") != digest({
                name: child for name, child in projection.items()
                if name != "content_sha256"})):
        raise ValueError("Final materializer output contract differs")
    fingerprints, seeds = [], []
    for index, scene in enumerate(scenes):
        if (not isinstance(scene, Mapping)
                or type(scene.get("fingerprint")) is not str
                or _HEX.fullmatch(scene["fingerprint"]) is None
                or type(scene.get("seed")) is not int
                or isinstance(scene.get("seed"), bool)
                or scene.get("id")
                    != f"diagnostic_v13_fresh_final_{index:04d}"
                or scene.get("split") != "fresh_final_test"
                or type(scene.get("family_id")) is not str
                or not scene["family_id"]):
            raise ValueError("Final scene public identity differs")
        fingerprints.append(scene["fingerprint"])
        seeds.append(scene["seed"])
    summaries = projection.get("scene_summaries")
    if (len(set(fingerprints)) != len(scenes)
            or len(set(seeds)) != len(scenes)
            or not isinstance(summaries, list)
            or len(summaries) != len(scenes)
            or projection.get("scene_summaries_sha256") != digest(summaries)):
        raise ValueError("Final material projection identity differs")
    for index, (scene, summary) in enumerate(zip(scenes, summaries)):
        if (not isinstance(summary, Mapping)
                or summary.get("accepted_scene_index") != index
                or summary.get("local_scene_index") != 0
                or summary.get("scene_index") != FINAL_SCENE_OFFSET + index
                or summary.get("fingerprint") != scene["fingerprint"]
                or type(summary.get("row_count")) is not int
                or summary["row_count"] <= 0
                or type(summary.get("unique_observation_count")) is not int
                or summary["unique_observation_count"] <= 0
                or any(type(summary.get(name)) is not str
                       or _HEX.fullmatch(summary[name]) is None
                       for name in (
                           "ordered_observation_hashes_sha256",
                           "unique_observation_hashes_sha256"))):
            raise ValueError("Final material scene projection differs")
    return deepcopy(scenes), deepcopy(dict(projection))


def _validate_material_projection_parity(
    projection: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    replay_arrays: Mapping[str, np.ndarray], scenes: Sequence[Mapping[str, Any]],
    *, environment_steps: int, replay_environment_steps: int,
) -> dict[str, Any]:
    """Match the selection projection to both independent final collections."""
    ordered = outer_api._decode(
        np.asarray(arrays["observation_hashes"]),
        "final observation hashes")
    replay_ordered = outer_api._decode(
        np.asarray(replay_arrays["observation_hashes"]),
        "replayed final observation hashes")
    scene_values = outer_api._decode(
        np.asarray(arrays["scene_fingerprints"]),
        "final scene fingerprints")
    replay_scenes = outer_api._decode(
        np.asarray(replay_arrays["scene_fingerprints"]),
        "replayed final scene fingerprints")
    if (ordered != replay_ordered or scene_values != replay_scenes
            or environment_steps != replay_environment_steps
            or projection.get("row_count") != len(ordered)
            or projection.get("ordered_observation_hashes_sha256")
                != digest(ordered)
            or projection.get("unique_observation_count") != len(set(ordered))
            or projection.get("unique_observation_hashes_sha256")
                != digest(sorted(set(ordered)))
            or projection.get("environment_steps") != environment_steps):
        raise ValueError("Materializer and final collector projection differ")
    summaries = projection["scene_summaries"]
    normalized = []
    for index, scene in enumerate(scenes):
        fingerprint = str(scene["fingerprint"])
        hashes = [value for value, observed_scene in zip(
            ordered, scene_values) if observed_scene == fingerprint]
        normalized.append({
            "accepted_scene_index": index,
            "scene_index": FINAL_SCENE_OFFSET + index,
            "fingerprint": fingerprint,
            "row_count": len(hashes),
            "ordered_observation_hashes_sha256": digest(hashes),
            "unique_observation_count": len(set(hashes)),
            "unique_observation_hashes_sha256": digest(sorted(set(hashes))),
        })
    for expected, actual in zip(summaries, normalized):
        # Per-scene projector calls have local_scene_index=0; normalize away
        # that implementation-local field before comparing collector evidence.
        projected = {name: child for name, child in expected.items()
                     if name != "local_scene_index"}
        if projected != actual:
            raise ValueError("Materializer and collector scene projection differ")
    value: dict[str, Any] = {
        "version": VERSION + ".materializer-collector-parity.v1",
        "passed": True, "scene_count": len(scenes),
        "row_count": len(ordered),
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_hashes_sha256": digest(sorted(set(ordered))),
        "scene_summaries_sha256": digest(normalized),
        "environment_steps": environment_steps,
        "independent_replay_equal": True,
        "raw_observations_published": False,
        "actions_or_probabilities_used_for_selection": False,
    }
    value["content_sha256"] = digest(value)
    return value

def _collect_final_rows(
    *, actor_path: Path, protocol_path: Path, manifest_path: Path,
    scenes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], int]:
    runtime = manifest_api.build_runtime(
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path)
    rows, steps = rows_api._collect(
        runtime, scenes, scene_offset=FINAL_SCENE_OFFSET,
        dense_critical=False, progress_label=None)
    arrays, _ = projection_api._projection_rows_to_arrays(rows)
    return arrays, steps


def _failure_completion(anchor: Mapping[str, Any], sources: Mapping[str, str]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": VERSION, "status": STATUS_FAILED,
        "attempt_key": anchor["attempt_key"],
        "attempt_anchor_content_sha256": anchor["content_sha256"],
        "reason": "protected_final_phase_failed",
        "final_consumed": True, "retry_allowed": False,
        "program_fits": 0, "actor_updates": 0,
        "runtime_action_override": False,
        "producer_sources_sha256": digest(dict(sources)),
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def run_final_once(
    *, candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    runtime_manifest_path: str | Path, designation_path: str | Path,
    failed_outer_closeout_path: str | Path,
    promotion_closeout_path: str | Path,
    expected_promotion_closeout_sha256: str,
    permanent_promotion_closeout_registry: str | Path,
    fresh_outer_registry_path: str | Path,
    fresh_outer_registry_report_path: str | Path,
    prior_outer_hash_projection_path: str | Path,
    outer_hash_projection_path: str | Path,
    outer_hash_projection_receipt_path: str | Path,
    development_rows_path: str | Path,
    combined_promoted_rows_path: str | Path,
    program_path: str | Path,
    selector_report_path: str | Path, outer_result_path: str | Path,
    expected_outer_result_sha256: str, outer_permanent_registry: str | Path,
    permanent_final_registry: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Run one v13 final attempt; neither salt nor final identity is an input."""
    sources = producer_sources()
    materializer_source, materializer_sources = _official_materializer_binding()
    preclaim = _authenticate_preclaim(
        candidate_lock_path=candidate_lock_path,
        expected_candidate_lock_sha256=expected_candidate_lock_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        runtime_manifest_path=runtime_manifest_path,
        designation_path=designation_path,
        failed_outer_closeout_path=failed_outer_closeout_path,
        promotion_closeout_path=promotion_closeout_path,
        expected_promotion_closeout_sha256=(
            expected_promotion_closeout_sha256),
        permanent_promotion_closeout_registry=(
            permanent_promotion_closeout_registry),
        fresh_outer_registry_path=fresh_outer_registry_path,
        fresh_outer_registry_report_path=fresh_outer_registry_report_path,
        prior_outer_hash_projection_path=prior_outer_hash_projection_path,
        outer_hash_projection_path=outer_hash_projection_path,
        outer_hash_projection_receipt_path=outer_hash_projection_receipt_path,
        development_rows_path=development_rows_path,
        combined_promoted_rows_path=combined_promoted_rows_path,
        program_path=program_path, selector_report_path=selector_report_path,
        outer_result_path=outer_result_path,
        expected_outer_result_sha256=expected_outer_result_sha256,
        outer_permanent_registry=outer_permanent_registry)
    bindings = preclaim["bindings"]
    projection_sources_sha256 = digest(preclaim["projection_sources"])
    projection_contract_sha256 = digest(preclaim["projection_contract"])
    attempt_inputs = {
        "scheme": VERSION + ".candidate-and-outer.v1",
        "candidate_lock_sha256": file_hash(preclaim["lock_path"]),
        "candidate_lock_content_sha256": preclaim["lock"]["content_sha256"],
        "actor_sha256": bindings["actor_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "runtime_manifest_sha256": bindings["runtime_manifest_sha256"],
        "designation_sha256": bindings["designation_sha256"],
        "program_sha256": bindings["program_sha256"],
        "public_feature_contract_sha256": bindings[
            "public_feature_contract_sha256"],
        "candidate_source_closure_sha256": bindings["source_closure_sha256"],
        "outer_result_sha256": preclaim["outer_result_sha256"],
        "outer_attempt_key": preclaim["outer_result"]["attempt_key"],
        "final_projection_source_closure_sha256": projection_sources_sha256,
        "final_projection_contract_sha256": projection_contract_sha256,
        "private_salt_commitment": PRIVATE_SALT_COMMITMENT,
        "private_salt_domain_sha256": sha256(PRIVATE_SALT_DOMAIN).hexdigest(),
    }
    attempt_key = digest(attempt_inputs)
    permanent = _directory(permanent_final_registry, "permanent final registry")
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "final output parent")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Final output must be a new path")
    anchor: dict[str, Any] = {
        "version": VERSION + ".attempt-anchor.v1",
        "status": "final_attempt_irrevocably_claimed",
        "attempt_key": attempt_key,
        "attempt_key_inputs": attempt_inputs,
        "bindings": {
            "outer_result_content_sha256": preclaim["outer_result"][
                "content_sha256"],
            "candidate_runtime_source_closure_sha256": digest(
                preclaim["runtime_sources"]),
            "final_controller_source_closure_sha256": digest(sources),
            "final_materializer_source_closure_sha256": digest(
                materializer_sources),
            "actor_feature_names_sha256": preclaim[
                "actor_feature_names_sha256"],
            "final_projection_source_closure_sha256": (
                projection_sources_sha256),
            "final_projection_contract_sha256": projection_contract_sha256,
            "private_salt_commitment": PRIVATE_SALT_COMMITMENT,
            "private_salt_domain_sha256": sha256(
                PRIVATE_SALT_DOMAIN).hexdigest(),
        },
        "candidate_and_outer_authenticated_before_claim": True,
        "final_identity_or_rows_accessed_before_claim": False,
        "retry_allowed": False, "formal_ready": False,
    }
    anchor["content_sha256"] = digest(anchor)
    campaign = permanent / attempt_key
    try:
        os.mkdir(campaign, 0o700)
    except FileExistsError:
        raise FileExistsError(
            "This candidate/outer final attempt is already consumed") from None
    _write_exclusive(campaign / ANCHOR_NAME, _json_bytes(anchor))
    descriptor = os.open(permanent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    completion: dict[str, Any]
    temporary: Path | None = None
    try:
        material = dict(_run_materializer(
            materializer_source, anchor, campaign))
        if (_official_materializer_binding()
                != (materializer_source, materializer_sources)
                or producer_sources() != sources
                or final_projection_api.producer_sources()
                    != preclaim["projection_sources"]
                or final_projection_api.contract()
                    != preclaim["projection_contract"]):
            raise RuntimeError("Final producer/controller source changed")
        scenes, material_projection = _validate_material(
            material, anchor=anchor, materializer_sources=materializer_sources)
        final_fingerprints = {str(scene["fingerprint"]) for scene in scenes}
        if (final_fingerprints & preclaim["development_scenes"]
                or final_fingerprints & preclaim["outer_scenes"]):
            raise ValueError("Final whole-scene identity overlaps a prior split")

        temporary = Path(tempfile.mkdtemp(
            prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
        _write_exclusive(temporary / ANCHOR_NAME, _json_bytes(anchor))
        _write_exclusive(temporary / MATERIAL_NAME, _json_bytes(material))
        arrays, environment_steps = _collect_final_rows(
            actor_path=preclaim["paths"]["actor"],
            protocol_path=preclaim["paths"]["protocol"],
            manifest_path=preclaim["paths"]["runtime_manifest"], scenes=scenes)
        _write_npz_exclusive(temporary / ROWS_NAME, arrays)
        replay_arrays, replay_environment_steps = _collect_final_rows(
            actor_path=preclaim["paths"]["actor"],
            protocol_path=preclaim["paths"]["protocol"],
            manifest_path=preclaim["paths"]["runtime_manifest"], scenes=scenes)
        parity = _validate_material_projection_parity(
            material_projection, arrays, replay_arrays, scenes,
            environment_steps=environment_steps,
            replay_environment_steps=replay_environment_steps)
        _write_exclusive(temporary / PARITY_NAME, _json_bytes(parity))
        audit_bindings = {
            "candidate_lock_sha256": file_hash(preclaim["lock_path"]),
            "outer_result_sha256": preclaim["outer_result_sha256"],
            "attempt_anchor_content_sha256": anchor["content_sha256"],
            "actor_sha256": bindings["actor_sha256"],
            "program_sha256": bindings["program_sha256"],
            "public_feature_contract_sha256": bindings[
                "public_feature_contract_sha256"],
            "final_material_file_sha256": file_hash(temporary / MATERIAL_NAME),
            "final_material_content_sha256": material["content_sha256"],
            "final_rows_sha256": file_hash(temporary / ROWS_NAME),
            "projection_parity_file_sha256": file_hash(
                temporary / PARITY_NAME),
            "projection_parity_content_sha256": parity["content_sha256"],
        }
        audit = audit_api.audit_rows(
            actor_path=preclaim["paths"]["actor"],
            program_path=preclaim["paths"]["program"],
            program_payload=preclaim["program_payload"],
            program_sha256=bindings["program_sha256"], arrays=arrays,
            replay_arrays=replay_arrays, scenes=scenes,
            development_observation_hashes=preclaim["development_hashes"],
            outer_observation_hashes=preclaim["outer_hashes"],
            development_scene_fingerprints=preclaim["development_scenes"],
            outer_scene_fingerprints=preclaim["outer_scenes"],
            bindings=audit_bindings, environment_steps=environment_steps,
            replay_environment_steps=replay_environment_steps)
        audit_api.validate_report(
            audit, expected_bindings=audit_bindings, require_passed=True)
        _write_exclusive(temporary / AUDIT_NAME, _json_bytes(audit))
        if (producer_sources() != sources
                or _official_materializer_binding()
                    != (materializer_source, materializer_sources)
                or file_hash(preclaim["lock_path"])
                    != expected_candidate_lock_sha256
                or file_hash(preclaim["paths"]["program"])
                    != bindings["program_sha256"]):
            raise RuntimeError("Frozen final inputs changed during audit")
        artifacts = {
            name: file_hash(temporary / name)
            for name in (ANCHOR_NAME, MATERIAL_NAME, ROWS_NAME,
                         PARITY_NAME, AUDIT_NAME)
        }
        completion = {
            "version": VERSION, "status": STATUS_PASSED,
            "attempt_key": attempt_key,
            "attempt_anchor_content_sha256": anchor["content_sha256"],
            "artifacts": artifacts,
            "final_material_content_sha256": material["content_sha256"],
            "audit_content_sha256": audit["content_sha256"],
            "projection_parity_content_sha256": parity["content_sha256"],
            "final_nine_gates_passed": True,
            "physical_counterfactual_replay_passed": True,
            "whole_scene_and_observation_isolation_passed": True,
            "materializer_collector_projection_parity_passed": True,
            "program_fits": 0, "actor_updates": 0,
            "runtime_action_override": False,
            "retry_allowed": False,
            "producer_sources_sha256": digest(sources),
            "formal_ready": False,
        }
        completion["content_sha256"] = digest(completion)
        _write_exclusive(temporary / COMPLETION_NAME, _json_bytes(completion))
        os.rename(temporary, destination)
        temporary = None
        _write_exclusive(campaign / COMPLETION_NAME, _json_bytes(completion))
    except BaseException:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        completion = _failure_completion(anchor, sources)
        try:
            _write_exclusive(campaign / COMPLETION_NAME, _json_bytes(completion))
        finally:
            raise RuntimeError("protected_final_phase_failed") from None
    return deepcopy(completion)

def read_completion(
    path: str | Path, *, expected_completion_sha256: str,
    permanent_final_registry: str | Path,
) -> dict[str, Any]:
    result_path = Path(path).expanduser().absolute()
    if (not result_path.is_file() or result_path.is_symlink()
            or result_path.resolve() != result_path
            or file_hash(result_path) != _sha(
                expected_completion_sha256, "v13 final completion")):
        raise ValueError("Exact v13 final completion bytes required")
    value = outer_api._strict_json_bytes(
        result_path.read_bytes(), "v13 final completion")
    if (not isinstance(value, Mapping) or not _content_valid(value)
            or value.get("version") != VERSION
            or value.get("status") not in {STATUS_PASSED, STATUS_FAILED}
            or value.get("retry_allowed") is not False
            or value.get("runtime_action_override") is not False
            or value.get("formal_ready") is not False
            or type(value.get("attempt_key")) is not str
            or _HEX.fullmatch(value["attempt_key"]) is None
            or value.get("producer_sources_sha256") != digest(producer_sources())
            or value.get("program_fits") != 0
            or value.get("actor_updates") != 0):
        raise ValueError("V13 final completion semantics differ")
    if value["status"] == STATUS_PASSED:
        artifacts = value.get("artifacts")
        if (set(value) != _SUCCESS_COMPLETION_FIELDS
                or value.get("final_nine_gates_passed") is not True
                or value.get("physical_counterfactual_replay_passed") is not True
                or value.get(
                    "whole_scene_and_observation_isolation_passed") is not True
                or value.get(
                    "materializer_collector_projection_parity_passed") is not True
                or not isinstance(artifacts, Mapping)
                or set(artifacts) != {
                    ANCHOR_NAME, MATERIAL_NAME, ROWS_NAME, PARITY_NAME,
                    AUDIT_NAME}
                or any(type(child) is not str or _HEX.fullmatch(child) is None
                       for child in artifacts.values())
                or any(not (result_path.parent / name).is_file()
                       or file_hash(result_path.parent / name) != child
                       for name, child in artifacts.items())):
            raise ValueError("Successful v13 final completion differs")
    elif (set(value) != _FAILURE_COMPLETION_FIELDS
            or value.get("reason") != "protected_final_phase_failed"
            or value.get("final_consumed") is not True):
        raise ValueError("Failed v13 final completion differs")
    permanent = _directory(permanent_final_registry, "permanent final registry")
    campaign = _directory(
        permanent / value["attempt_key"], "permanent final campaign")
    anchor_path = campaign / ANCHOR_NAME
    completion_path = campaign / COMPLETION_NAME
    if (not anchor_path.is_file() or anchor_path.is_symlink()
            or not completion_path.is_file() or completion_path.is_symlink()
            or completion_path.read_bytes() != result_path.read_bytes()
            or file_hash(completion_path) != expected_completion_sha256):
        raise ValueError("Permanent v13 final claim/completion differs")
    anchor = outer_api._strict_json_bytes(
        anchor_path.read_bytes(), "permanent v13 final anchor")
    _materializer_source, official_materializer_sources = (
        _official_materializer_binding())
    if (not isinstance(anchor, Mapping) or not _content_valid(anchor)
            or anchor.get("version") != VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "final_attempt_irrevocably_claimed"
            or anchor.get("attempt_key") != value["attempt_key"]
            or anchor.get("retry_allowed") is not False
            or anchor.get("candidate_and_outer_authenticated_before_claim") is not True
            or anchor.get("final_identity_or_rows_accessed_before_claim") is not False
            or anchor.get("bindings", {}).get(
                "final_controller_source_closure_sha256")
                != digest(producer_sources())
            or anchor.get("bindings", {}).get(
                "final_materializer_source_closure_sha256")
                != digest(official_materializer_sources)
            or anchor.get("bindings", {}).get(
                "final_projection_source_closure_sha256")
                != digest(final_projection_api.producer_sources())
            or anchor.get("bindings", {}).get(
                "final_projection_contract_sha256")
                != digest(final_projection_api.contract())
            or anchor.get("bindings", {}).get("private_salt_commitment")
                != PRIVATE_SALT_COMMITMENT
            or anchor.get("bindings", {}).get("private_salt_domain_sha256")
                != sha256(PRIVATE_SALT_DOMAIN).hexdigest()
            or value.get("attempt_anchor_content_sha256")
                != anchor.get("content_sha256")):
        raise ValueError("Permanent v13 final anchor differs")
    return deepcopy(dict(value))


__all__ = [
    "VERSION", "MATERIAL_VERSION", "MATERIAL_STATUS", "STATUS_PASSED",
    "STATUS_FAILED", "ANCHOR_NAME", "COMPLETION_NAME", "MATERIAL_NAME",
    "ROWS_NAME", "AUDIT_NAME", "PARITY_NAME", "PRIVATE_SALT_DOMAIN",
    "PRIVATE_SALT_COMMITMENT", "OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH",
    "OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256", "contract",
    "producer_sources", "run_final_once", "read_completion",
]
